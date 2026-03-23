from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import numpy as np

from transformers import AutoModel, AutoTokenizer, CLIPProcessor, CLIPModel
try:
    from transformers import AutoModelForImageClassification, AutoImageProcessor
except Exception:  # optional, only for aesthetic predictor fallback
    AutoModelForImageClassification = None  # type: ignore
    AutoImageProcessor = None  # type: ignore

try:
    from torchvision.transforms import Compose, Resize, Normalize, InterpolationMode
except Exception:  # optional, for tensor preprocessing
    Compose = None  # type: ignore
    Resize = None  # type: ignore
    Normalize = None  # type: ignore
    InterpolationMode = None  # type: ignore

from ..base import TextPromptRewardFunction, RewardConfig, RewardFunctionRegistry




class CLIPRewardHF(TextPromptRewardFunction):
    """CLIP cosine similarity via HuggingFace models.

    config.model_name defaults to openai/clip-vit-base-patch32
    """

    def __init__(self, config: RewardConfig):
        differentiable = getattr(config, "differentiable", False)
        super().__init__("clip", config.device, expects_pil=False, differentiable=differentiable)
        model_name = (getattr(config, "model_name", None) or "openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained(model_name, cache_dir=config.cache_dir)
        self.model = CLIPModel.from_pretrained(
            model_name,
            cache_dir=config.cache_dir,
            torch_dtype=getattr(config, "dtype", torch.float32),
            use_safetensors=True,
        ).to(self.device)
        if not differentiable:
            self.model.eval()

    def _evaluate_with_prompt(self, images: torch.Tensor, prompt: str) -> torch.Tensor:
        inputs = self.processor(text=[prompt], images=images, return_tensors="pt", padding=True).to(self.device)
        # Only use no_grad if not in differentiable mode
        with torch.set_grad_enabled(self.differentiable):
            out = self.model(**inputs)
            img_emb = out.image_embeds
            txt_emb = out.text_embeds
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
            sim = (img_emb * txt_emb).sum(dim=-1)
        return sim.to(torch.float32)

    def get_output_range(self) -> tuple[float, float]:
        return (-1.0, 1.0)


class AestheticRewardHF(TextPromptRewardFunction):
    """Aesthetic score using CLIP image features + a small MLP head (requires weights).

    Expects a weight file under config.metadata['aesthetic_weights'] if provided.
    """

    def __init__(self, config: RewardConfig):
        differentiable = getattr(config, "differentiable", False)
        super().__init__("aesthetic", config.device, expects_pil=False, differentiable=differentiable)
        self._use_predictor = False
        predictor_id = (getattr(config, "model_name", None) or "shunk031/aesthetics-predictor-v1-vit-large-patch14")
        # Try dedicated aesthetic predictor
        try:
            print(f"Using aesthetic predictor: {predictor_id}")
            from aesthetics_predictor import AestheticsPredictorV1  # type: ignore
            self.aesthetic_model = AestheticsPredictorV1.from_pretrained(predictor_id).to(self.device)
            if not differentiable:
                self.aesthetic_model.eval()
            self.aesthetic_processor = CLIPProcessor.from_pretrained(predictor_id, cache_dir=config.cache_dir)
            self._use_predictor = True
        except Exception as e:
            print(f"AestheticRewardHF failed: {e}")
            # Fallback: try transformers image-classification interface
            try:
                if AutoModelForImageClassification is not None and AutoImageProcessor is not None:
                    print(f"Using transformers image-classification interface: {predictor_id}")
                    self.aesthetic_model = AutoModelForImageClassification.from_pretrained(predictor_id, use_safetensors=True).to(self.device)  # type: ignore[arg-type]
                    if not differentiable:
                        self.aesthetic_model.eval()
                    self.aesthetic_processor = AutoImageProcessor.from_pretrained(predictor_id)  # type: ignore[assignment]
                    self._use_predictor = True
            except Exception as e:
                print(f"AestheticRewardHF failed: {e}")
                self._use_predictor = False

        if not self._use_predictor:
            # Fallback to CLIP image features + small MLP head
            base = (getattr(config, "model_name", None) or "openai/clip-vit-large-patch14")
            self.processor = CLIPProcessor.from_pretrained(base, cache_dir=config.cache_dir)
            self.clip = CLIPModel.from_pretrained(
                base,
                cache_dir=config.cache_dir,
                torch_dtype=getattr(config, "dtype", torch.float32),
                use_safetensors=True,
            ).to(self.device)
            if not differentiable:
                self.clip.eval()
            hidden_dim = getattr(self.clip.visual_projection, "out_features", 768)
            head_dtype = next(self.clip.parameters()).dtype
            self.head = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, 1024, dtype=head_dtype),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, 128, dtype=head_dtype),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 1, dtype=head_dtype),
            ).to(self.device)
            weights_path = (getattr(config, "metadata", {}) or {}).get("aesthetic_weights")
            if weights_path:
                try:
                    self.head.load_state_dict(torch.load(weights_path, map_location=self.device))
                except Exception:
                    pass
            if not differentiable:
                self.head.eval()

    def _evaluate_with_prompt(self, images: torch.Tensor, prompt: str) -> torch.Tensor:
        if self._use_predictor:
            # In differentiable mode, aesthetic predictor with PIL input is not supported
            # (it requires PIL conversion which breaks gradients)
            if self.differentiable:
                raise ValueError(
                    "AestheticRewardHF with predictor model cannot be used in differentiable mode "
                    "because it requires PIL image conversion. Use the CLIP+MLP fallback or set differentiable=False."
                )
            # Convert to PIL list if tensor
            imgs = images
            if isinstance(imgs, torch.Tensor):
                imgs = imgs.detach().cpu()
                if imgs.dim() == 3:
                    imgs = imgs.unsqueeze(0)
                # Use manual tensor to PIL conversion to avoid torchvision compatibility issues
                pil_list = []
                for img in imgs:
                    # Ensure tensor is in [0, 1] range and has correct shape
                    img = img.clamp(0, 1)
                    if img.shape[0] == 3:  # CHW format
                        img = img.permute(1, 2, 0)  # Convert to HWC
                    # Convert to numpy and then to PIL
                    img_np = img.numpy()
                    if img_np.dtype != np.uint8:
                        img_np = (img_np * 255).astype(np.uint8)
                    from PIL import Image
                    pil_list.append(Image.fromarray(img_np))
            else:
                pil_list = images if isinstance(images, list) else [images]
            proc = self.aesthetic_processor(images=pil_list, return_tensors="pt")
            proc = {k: v.to(self.device) for k, v in proc.items()}
            with torch.set_grad_enabled(self.differentiable):
                outputs = self.aesthetic_model(**proc)
                logits = outputs.logits.squeeze()
                if not isinstance(logits, torch.Tensor):
                    logits = torch.tensor(logits, device=self.device)
            return logits.to(self.device, dtype=torch.float32).view(-1)

        # Fallback path: CLIP + MLP head (fully differentiable)
        inputs = self.processor(images=images, return_tensors="pt", padding=True, do_rescale=False).to(self.device)
        with torch.set_grad_enabled(self.differentiable):
            img_feat = self.clip.get_image_features(**inputs)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            img_feat = img_feat.to(self.head[0].weight.dtype)
            score = self.head(img_feat).squeeze(-1)
            score = torch.relu(score)
        return score.to(torch.float32)

    def get_output_range(self) -> tuple[float, float]:
        return (0.0, 10.0)


class ImageRewardHF(TextPromptRewardFunction):
    """ImageReward-v1.0 via the ImageReward package, if installed.
    
    Optimized to batch evaluate all candidates for efficiency.
    Supports both PIL-based (non-differentiable) and tensor-based (differentiable) evaluation.
    """

    def __init__(self, config: RewardConfig):
        differentiable = getattr(config, "differentiable", False)
        # When differentiable=True, we use tensors instead of PIL
        super().__init__("imagereward", config.device, expects_pil=not differentiable, differentiable=differentiable)
        # Initialize attributes that may be set up later for differentiable mode
        self.imagereward_xform = None
        self._tokenizer = None
        # Get dtype from config (defaults to float32 for ImageReward)
        self._dtype = getattr(config, "dtype", torch.float32)
        
        try:
            import ImageReward
            self.model = ImageReward.load("ImageReward-v1.0", device=self.device)
            # Set to training mode if differentiable (to allow gradients)
            if differentiable:
                self.model.train()
                # For differentiable mode, we need the tensor transform and text tokenization
                self._setup_differentiable_mode()
            else:
                self.model.eval()
        except Exception as e:
            self.model = None
            print(f"[WARNING] ImageRewardHF: Failed to load ImageReward model: {e}")
    
    def _setup_differentiable_mode(self):
        """Set up tensor transform and tokenizer for differentiable mode."""
        if self.model is None:
            return
        try:
            from torchvision.transforms import (
                Compose,
                Resize,
                Normalize,
                InterpolationMode,
            )
            # ImageNet normalization constants
            OPENAI_DATASET_MEAN = [0.48145466, 0.4578275, 0.40821073]
            OPENAI_DATASET_STD = [0.26862954, 0.26130258, 0.27577711]
            
            self.imagereward_xform = Compose([
                lambda x: ((x / 2) + 0.5).clamp(0, 1),  # Convert from [-1, 1] to [0, 1]
                Resize(224, interpolation=InterpolationMode.BICUBIC),
                Normalize(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD),
            ])
            # Pre-tokenize the prompt for efficiency
            if hasattr(self.model, 'blip') and hasattr(self.model.blip, 'tokenizer'):
                self._tokenizer = self.model.blip.tokenizer
            else:
                # Fallback: try to find tokenizer in the model
                self._tokenizer = None
        except Exception as e:
            print(f"[WARNING] ImageRewardHF: Failed to set up differentiable mode: {e}")

    def _evaluate_with_prompt(self, images, prompt: str) -> torch.Tensor:
        if self.model is None:
            return torch.zeros(1, device=self.device)
        
        # Differentiable mode: use tensor-based evaluation with score_gard
        if self.differentiable:
            # Ensure differentiable mode is set up (may have been enabled after initialization)
            if self.imagereward_xform is None:
                self._setup_differentiable_mode()
                if self.imagereward_xform is None:
                    raise RuntimeError(
                        "ImageRewardHF differentiable mode requires imagereward_xform to be initialized. "
                        "Failed to set up differentiable mode. Model may not be available."
                    )
            
            # Ensure we have tensor input
            if not isinstance(images, torch.Tensor):
                # Try to convert from other formats if needed
                if isinstance(images, (list, tuple)) and len(images) > 0:
                    if isinstance(images[0], torch.Tensor):
                        images = torch.stack([img for img in images if isinstance(img, torch.Tensor)])
                    else:
                        raise TypeError(
                            f"ImageRewardHF differentiable mode requires torch.Tensor input, "
                            f"got {type(images[0])}"
                        )
                else:
                    raise TypeError(
                        f"ImageRewardHF differentiable mode requires torch.Tensor input, "
                        f"got {type(images)}"
                    )
            
            # Ensure batch dimension
            if images.dim() == 3:
                images = images.unsqueeze(0)
            
            # Transform images (clamp, resize, normalize)
            img_tensor = self.imagereward_xform(images).to(self.device)
            
            # Convert to the model's dtype (ImageReward models typically use float32)
            # Get the model's parameter dtype, or use the configured dtype
            try:
                model_dtype = next(self.model.parameters()).dtype
            except (StopIteration, AttributeError):
                model_dtype = self._dtype
            img_tensor = img_tensor.to(dtype=model_dtype)
            
            # Tokenize prompt if tokenizer is available
            if self._tokenizer is not None:
                try:
                    text_input = self._tokenizer(
                        prompt,
                        padding="max_length",
                        truncation=True,
                        max_length=35,
                        return_tensors="pt",
                    ).to(self.device)
                    
                    # Ensure input_ids and attention_mask match model dtype if needed
                    # (though tokenizer outputs are typically long/int64, which is fine)
                    
                    # Use score_gard for gradient-enabled scoring
                    if hasattr(self.model, 'score_gard'):
                        rewards = self.model.score_gard(
                            prompt_attention_mask=text_input.attention_mask,
                            prompt_ids=text_input.input_ids,
                            image=img_tensor,
                        )
                    elif hasattr(self.model, 'score_grad'):  # Handle typo variant
                        rewards = self.model.score_grad(
                            prompt_attention_mask=text_input.attention_mask,
                            prompt_ids=text_input.input_ids,
                            image=img_tensor,
                        )
                    else:
                        raise AttributeError(
                            "ImageReward model does not have score_gard or score_grad method. "
                            "Please ensure you have a version of ImageReward that supports gradient flow."
                        )
                    
                    # Ensure output is properly shaped
                    if rewards.dim() == 0:
                        rewards = rewards.unsqueeze(0)
                    elif rewards.shape[0] != img_tensor.shape[0]:
                        # If we got a single score for a batch, expand it
                        if rewards.numel() == 1:
                            rewards = rewards.expand(img_tensor.shape[0])
                    
                    return rewards.to(torch.float32)
                except Exception as e:
                    raise RuntimeError(
                        f"ImageRewardHF differentiable evaluation failed: {e}. "
                        f"Please ensure ImageReward package supports gradient flow (score_gard method)."
                    ) from e
            else:
                raise RuntimeError(
                    "ImageRewardHF differentiable mode requires access to model tokenizer, "
                    "but tokenizer was not found during initialization."
                )
        
        # Non-differentiable mode: use PIL-based evaluation (original implementation)
        # Robustly unwrap common pipeline outputs (diffusers PipelineOutput, dict, etc.)
        try:
            if hasattr(images, "images"):
                images = getattr(images, "images")
            elif isinstance(images, dict) and ("images" in images):
                images = images["images"]
        except Exception:
            pass

        # Ensure PIL for ImageReward (it only supports PIL.Image or file path)
        from ..base import ImageRewardFunction as IRF
        images = IRF._convert_to_expected_format(self, images)
        imgs = images if isinstance(images, list) else [images]
        # Normalize to RGB (ImageReward expects standard 3-channel images)
        try:
            from PIL import Image as _PILImage
            imgs = [img.convert("RGB") if isinstance(img, _PILImage.Image) and getattr(img, "mode", "RGB") != "RGB" else img for img in imgs]
        except Exception:
            pass
        vals: list[float] = []
        
        # Try batch evaluation first (more efficient)
        try:
            # Attempt to score all images at once
            batch_scores = self.model.score(prompt, imgs)
            
            # Handle different return types
            if isinstance(batch_scores, (list, tuple)):
                vals = [float(s) for s in batch_scores]
            elif hasattr(batch_scores, "__len__") and len(batch_scores) == len(imgs):
                # Tensor or numpy array with batch scores
                vals = [float(s) for s in batch_scores]
            else:
                # Single value returned, convert to list
                single_score = float(batch_scores)
                vals = [single_score] * len(imgs)
        except Exception:
            # Fallback to individual evaluation if batch fails
            for img in imgs:
                vals.append(float(self.model.score(prompt, img)))
        
        out = torch.tensor(vals, device=self.device, dtype=torch.float32)
        # One-time debug: if reward returns non-finite, emit a hint (the caller will sanitize).
        try:
            if not torch.isfinite(out).all():
                bad = int((~torch.isfinite(out)).sum().item())
                total = int(out.numel())
                print(f"[ImageRewardHF] Warning: produced {bad}/{total} non-finite scores for prompt={prompt!r}.")
        except Exception:
            pass
        return out

    def get_output_range(self) -> tuple[float, float]:
        return (-2.5, 2.5)


class HPSv2RewardHF(TextPromptRewardFunction):
    """HPSv2 reward if dependency available (uses hpsv2.score API).

    Optimized to batch evaluate all candidates for efficiency.
    Supports both PIL-based (non-differentiable) and tensor-based (differentiable) evaluation.
    """

    def __init__(self, config: RewardConfig):
        differentiable = getattr(config, "differentiable", False)
        # When differentiable=True, we use tensors instead of PIL
        super().__init__("hpsv2", config.device, expects_pil=not differentiable, differentiable=differentiable)
        self._hps_version = (getattr(config, "metadata", {}) or {}).get("hps_version", "v2.0")

        # Initialize attributes that may be set up later for differentiable mode
        self._hps_model = None
        self._hps_preprocess = None
        self._hps_tokenizer = None
        self._available = False  # Initialize early

        # Try to initialize once; keep a flag if available
        try:
            if differentiable:
                # For differentiable mode, we need direct access to the model
                self._setup_differentiable_mode()
            else:
                # For non-differentiable mode, use the high-level API
                from hpsv2.img_score import initialize_model as hps_init  # type: ignore
                hps_init()
            self._available = True
        except Exception as e:
            print(f"HPSv2RewardHF initialization failed: {e}")
            self._available = False

    def _setup_differentiable_mode(self):
        """Set up the HPSv2 model for differentiable evaluation."""
        if self._available and self._hps_model is None:
            try:
                from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
                import huggingface_hub
                from hpsv2.utils import hps_version_map
                from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, InterpolationMode

                # Create the model (without PIL preprocessing)
                model, preprocess_train, preprocess_val = create_model_and_transforms(
                    'ViT-H-14',
                    'laion2B-s32B-b79K',
                    precision='amp',
                    device=self.device,
                    jit=False,
                    force_quick_gelu=False,
                    force_custom_text=False,
                    force_patch_dropout=False,
                    force_image_size=None,
                    pretrained_image=False,
                    image_mean=None,
                    image_std=None,
                    light_augmentation=True,
                    aug_cfg={},
                    output_dict=True,
                    with_score_predictor=False,
                    with_region_predictor=False
                )

                # Load the checkpoint
                cp = huggingface_hub.hf_hub_download("xswu/HPSv2", hps_version_map[self._hps_version])
                checkpoint = torch.load(cp, map_location=self.device, weights_only=True)
                model.load_state_dict(checkpoint['state_dict'])

                # Create tensor-compatible preprocessing
                if Compose is None or Resize is None or Normalize is None or InterpolationMode is None:
                    raise ImportError("torchvision is required for HPSv2 differentiable mode")

                # HPSv2 uses OpenAI dataset normalization constants
                OPENAI_DATASET_MEAN = [0.48145466, 0.4578275, 0.40821073]
                OPENAI_DATASET_STD = [0.26862954, 0.26130258, 0.27577711]

                # For tensors: resize shortest side to 224, center crop to 224x224, and normalize
                # Note: Range conversion from [-1, 1] to [0, 1] is done in evaluate method
                self._hps_preprocess = Compose([
                    # Resize shortest side to 224 using bicubic interpolation
                    Resize(224, interpolation=InterpolationMode.BICUBIC),
                    # Center crop to 224x224
                    CenterCrop(224),
                    # Normalize with OpenAI dataset stats
                    Normalize(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD),
                ])

                # Store the components
                self._hps_model = model.to(self.device)
                self._hps_tokenizer = get_tokenizer('ViT-H-14')

                # Set to training mode to allow gradients
                if self.differentiable:
                    self._hps_model.train()
                else:
                    self._hps_model.eval()

            except Exception as e:
                print(f"HPSv2RewardHF: Failed to set up differentiable mode: {e}")
                self._available = False

    def _evaluate_with_prompt(self, images, prompt: str) -> torch.Tensor:
        if not self._available:
            return torch.zeros(1, device=self.device)

        # Differentiable mode: use tensor-based evaluation with the underlying model
        if self.differentiable:
            # Ensure differentiable mode is set up
            if self._hps_model is None:
                self._setup_differentiable_mode()
                if self._hps_model is None:
                    raise RuntimeError(
                        "HPSv2RewardHF differentiable mode requires model to be initialized. "
                        "Failed to set up differentiable mode."
                    )

            # Ensure we have tensor input
            if not isinstance(images, torch.Tensor):
                raise TypeError(
                    f"HPSv2RewardHF differentiable mode requires torch.Tensor input, "
                    f"got {type(images)}"
                )

            # Ensure batch dimension
            if images.dim() == 3:
                images = images.unsqueeze(0)

            # Convert from [-1, 1] to [0, 1] range and apply preprocessing
            # HPSv2 expects images in [0, 1] range, then applies its own preprocessing
            img_tensor = images
            if img_tensor.min() < 0:  # Convert from [-1, 1] to [0, 1]
                img_tensor = (img_tensor / 2 + 0.5).clamp(0, 1)

            # Apply HPSv2 preprocessing (resize, normalize)
            img_tensor = self._hps_preprocess(img_tensor).to(self.device)

            # Tokenize prompt
            text_tensor = self._hps_tokenizer([prompt]).to(self.device)

            # Forward pass with gradient preservation
            with torch.set_grad_enabled(self.differentiable):
                with torch.cuda.amp.autocast():
                    outputs = self._hps_model(img_tensor, text_tensor)
                    image_features, text_features = outputs["image_features"], outputs["text_features"]
                    logits_per_image = image_features @ text_features.T
                    # One score per image: diagonal when (B, B), else column when (B, 1) for single prompt
                    if logits_per_image.size(-1) == 1:
                        hps_scores = logits_per_image.squeeze(-1)
                    else:
                        hps_scores = torch.diagonal(logits_per_image).squeeze()

            # Ensure proper shape (single sample -> (1,))
            if hps_scores.dim() == 0:
                hps_scores = hps_scores.unsqueeze(0)

            return hps_scores.to(torch.float32)

        # Non-differentiable mode: use PIL-based evaluation (original implementation)
        imgs = images if isinstance(images, list) else [images]
        vals: list[float] = []
        try:
            import hpsv2  # type: ignore

            # Try batch evaluation first (more efficient)
            try:
                # Attempt to score all images at once
                batch_scores = hpsv2.score(imgs, prompt, hps_version=self._hps_version)

                # Handle different return types
                if isinstance(batch_scores, (list, tuple)):
                    for score in batch_scores:
                        try:
                            vals.append(float(score))
                        except Exception:
                            if hasattr(score, "item"):
                                vals.append(float(score.item()))
                            elif isinstance(score, (list, tuple)) and len(score) > 0:
                                vals.append(float(score[0]))
                            else:
                                vals.append(0.0)
                elif hasattr(batch_scores, "__len__") and len(batch_scores) == len(imgs):
                    # Tensor or numpy array with batch scores
                    for score in batch_scores:
                        try:
                            vals.append(float(score))
                        except Exception:
                            vals.append(0.0)
                else:
                    # Single value returned, convert to list
                    try:
                        single_score = float(batch_scores)
                        vals = [single_score] * len(imgs)
                    except Exception:
                        if hasattr(batch_scores, "item"):
                            single_score = float(batch_scores.item())
                            vals = [single_score] * len(imgs)
                        else:
                            # Fall back to individual evaluation
                            for img in imgs:
                                score = hpsv2.score(img, prompt, hps_version=self._hps_version)
                                try:
                                    vals.append(float(score))
                                except Exception:
                                    if hasattr(score, "item"):
                                        vals.append(float(score.item()))
                                    elif isinstance(score, (list, tuple)) and len(score) > 0:
                                        vals.append(float(score[0]))
                                    else:
                                        vals.append(0.0)
            except Exception:
                # Fallback to individual evaluation if batch fails
                for img in imgs:
                    score = hpsv2.score(img, prompt, hps_version=self._hps_version)
                    # score may be list/np/tensor; coerce to float
                    try:
                        vals.append(float(score))
                    except Exception:
                        if hasattr(score, "item"):
                            vals.append(float(score.item()))
                        elif isinstance(score, (list, tuple)) and len(score) > 0:
                            vals.append(float(score[0]))
                        else:
                            vals.append(0.0)
        except Exception as e:
            print(f"HPSv2RewardHF failed: {e}")
            return torch.zeros(len(imgs), device=self.device)
        return torch.tensor(vals, device=self.device, dtype=torch.float32)

    def get_output_range(self) -> tuple[float, float]:
        return (0.0, 100.0)


class CombinedRewardHF(TextPromptRewardFunction):
    """Combined reward function that balances CLIP and Aesthetic scores."""

    def __init__(self, config: RewardConfig):
        super().__init__("combined", config.device, expects_pil=False)
        
        # Attribute safety for weights
        weights = getattr(config, "weights", None)
        if weights is None and hasattr(config, "metadata"):
            weights = config.metadata.get("weights")
        if weights is None:
            weights = [0.5, 0.5]
            
        # FIX: Explicitly move to self.device (cuda:0)
        self.weights = torch.tensor(weights, device=self.device, dtype=torch.float32)
        
        # Instantiate sub-rewards
        self.clip_fn = CLIPRewardHF(config)
        self.aesthetic_fn = AestheticRewardHF(config)
        
        # FIX: Explicitly move scales to self.device (cuda:0)
        # CLIP scale (20x) and Aesthetic scale (1x)
        self.scales = torch.tensor([20.0, 1.0], device=self.device, dtype=torch.float32)

    def _evaluate_with_prompt(self, images: torch.Tensor, prompt: str) -> torch.Tensor:
        # Ensure images are on the right device and dtype
        images = images.to(self.device, dtype=torch.float32)
        
        context = {"prompt": prompt}
        
        # Get scores (these will be on cuda:0)
        clip_score = self.clip_fn.evaluate(images, context).view(-1)
        aesthetic_score = self.aesthetic_fn.evaluate(images, context).view(-1)
        
        # Stack into [2, batch_size]
        scores = torch.stack([clip_score, aesthetic_score])
        
        # Ensure our constants are on the same device as the scores
        # (Using .to(scores.device) is a foolproof way to avoid the RuntimeError)
        w = self.weights.to(scores.device).unsqueeze(1)
        s = self.scales.to(scores.device).unsqueeze(1)
        
        # Final weighted sum logic: (scores * scales * weights)
        weighted_sum = torch.sum(scores * s * w, dim=0)
        
        return weighted_sum

    def get_output_range(self) -> tuple[float, float]:
        return (-100.0, 100.0)


def register_t2i_rewards() -> None:
    RewardFunctionRegistry.register("clip", CLIPRewardHF)
    RewardFunctionRegistry.register("clip_hf", CLIPRewardHF)
    RewardFunctionRegistry.register("aesthetic", AestheticRewardHF)
    RewardFunctionRegistry.register("imagereward", ImageRewardHF)
    RewardFunctionRegistry.register("image_reward", ImageRewardHF)
    RewardFunctionRegistry.register("hps", HPSv2RewardHF)
    RewardFunctionRegistry.register("hps_v2", HPSv2RewardHF)
    RewardFunctionRegistry.register("combined", CombinedRewardHF)



