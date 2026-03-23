from __future__ import annotations

from typing import Any, Dict

import torch
from omegaconf import DictConfig, OmegaConf

from .image import ImageGenerationProblem
from ..rewards.base import RewardFunction


class ProblemBuilder:
    """Factory to build Problem instances from benchmark tasks.

    Each task is a dict with at least a 'context' field. Expected keys:
    - modality: 'image' | 'image_noise' | 'molecule' | custom
    - context: dict (e.g., {'prompt': 'a prompt'} for T2I)
    Additional keys may guide selection of noise injection or latent shapes.
    """

    def __init__(self, cfg: Any, generative_model: Any, reward: RewardFunction | Any, device: str):
        self.cfg = cfg
        self.generative_model = generative_model
        self.reward = reward
        self.device = device

    def build(self, task: Dict[str, Any]) -> Any:
        modality = task.get("modality", "image")
        context = task.get("context", {})

        if modality in ("t2i", "image", "t2i_noise", "image_noise"):
            if hasattr(self.generative_model, "get_noise_latent_shape"):
                latent_shape = tuple(self.generative_model.get_noise_latent_shape())  # type: ignore[attr-defined]
            elif hasattr(self.generative_model, "sample_latents"):
                sample = self.generative_model.sample_latents(batch_size=1)  # type: ignore[attr-defined]
                latent_shape = tuple(sample.shape[1:])
            else:
                latent_shape = (4, 64, 64)
            model_cfg_src = self.cfg.get("model", {}) or self.cfg.get("generative_model", {})
            model_config: Dict[str, Any] = {}
            for key in ("guidance_scale", "num_inference_steps", "height", "width", "num_images_per_prompt", "eta"):
                if key in model_cfg_src:
                    model_config[key] = model_cfg_src[key]
            reward_expects_pil = False
            if hasattr(self.reward, "get_input_format") and callable(getattr(self.reward, "get_input_format")):
                reward_expects_pil = (self.reward.get_input_format() == "pil")  # type: ignore[attr-defined]
            return ImageGenerationProblem(
                prompt=context.get("prompt", ""),
                generative_model=self.generative_model,
                reward_model=self.reward,
                device=self.device,
                latent_shape=latent_shape,
                model_config=model_config,
                reward_expects_pil=reward_expects_pil,
                context=context,
            )

        # Handle protein modality separately
        if modality in ("protein", "proteina"):
            noise_dims = None
            
            # Extract n_residues from context if available
            if "n_residues" in context:
                noise_dims = (int(context["n_residues"]), 3)
            
            if noise_dims is None:
                if hasattr(self.generative_model, "get_noise_latent_shape"):
                    try:
                        noise_dims = tuple(self.generative_model.get_noise_latent_shape())  # type: ignore[attr-defined]
                    except Exception:
                        noise_dims = None
                elif hasattr(self.generative_model, "sample_latents"):
                    try:
                        sample = self.generative_model.sample_latents(batch_size=1)  # type: ignore[attr-defined]
                        noise_dims = tuple(sample.shape[1:])
                    except Exception:
                        noise_dims = None

            model_config: Dict[str, Any] = {}
            cfg_candidates = [self.cfg.get("proteina"), self.cfg.get("model"), self.cfg.get("generative_model")]
            for cfg_src in cfg_candidates:
                model_config.update(self._to_container(cfg_src))
            
            # Merge context params relevant for model
            if "n_residues" in context:
                model_config["n_residues"] = context["n_residues"]
            # Pass fold code (CATH code) for fold-conditioned generation
            if "target_fold" in context:
                model_config["cath_code"] = context["target_fold"]
            elif "fold_code" in context:
                model_config["cath_code"] = context["fold_code"]
            
            # Handle motif scaffolding (inpainting)
            if "motif_pdb_path" in context and "contig" in context:
                # Parse motif information
                try:
                    # Lazy import to avoid dependency when not using motifs
                    import sys
                    import os
                    # Add proteina path if needed
                    proteina_path = os.path.join(os.path.dirname(__file__), "../../proteina")
                    if os.path.exists(proteina_path) and proteina_path not in sys.path:
                        sys.path.insert(0, proteina_path)
                    
                    from proteinfoundation.nn.motif_factory import parse_motif
                    
                    motif_pdb_path = context["motif_pdb_path"]
                    contig = context["contig"]
                    motif_only = context.get("motif_only", False)
                    min_length = context.get("min_length")
                    max_length = context.get("max_length")
                    
                    # Parse motif: returns (mask, x_motif_full, out_str)
                    # When nsamples=1: returns ([n] bool, [n, 3] float, str)
                    # When nsamples>1 and make_tensor=True: returns ([batch, n] bool, [batch, n, 3] float, list)
                    mask, x_motif_full, _ = parse_motif(
                        motif_pdb_path,
                        contig,
                        nsamples=1,
                        make_tensor=False,  # We'll handle tensor conversion ourselves
                        motif_only=motif_only,
                        min_length=min_length,
                        max_length=max_length,
                    )
                    
                    # parse_motif with nsamples=1 returns:
                    # - mask: [n] boolean tensor
                    # - x_motif_full: [n, 3] tensor
                    # Add batch dimension [1, n] and [1, n, 3]
                    if isinstance(mask, torch.Tensor) and isinstance(x_motif_full, torch.Tensor):
                        # Add batch dimension
                        mask = mask.unsqueeze(0)  # [n] -> [1, n]
                        x_motif_full = x_motif_full.unsqueeze(0)  # [n, 3] -> [1, n, 3]
                        
                        # Store motif information in model_config
                        model_config["motif_seq_mask"] = mask  # [1, n] boolean
                        model_config["motif_structure"] = x_motif_full  # [1, n, 3] coordinates
                        # Update n_residues from parsed motif length
                        if mask.shape[1] > 0:
                            model_config["n_residues"] = int(mask.shape[1])
                            noise_dims = (int(mask.shape[1]), 3)
                    else:
                        # Fallback: convert lists to tensors
                        import torch
                        if isinstance(mask, list) and len(mask) > 0:
                            # Handle list of masks/structures
                            mask_tensor = torch.nn.utils.rnn.pad_sequence(
                                [torch.tensor(m, dtype=torch.bool) for m in mask],
                                batch_first=True,
                                padding_value=False
                            )
                            x_motif_tensor = torch.nn.utils.rnn.pad_sequence(
                                [torch.tensor(x, dtype=torch.float32) for x in x_motif_full],
                                batch_first=True,
                                padding_value=0.0
                            )
                            model_config["motif_seq_mask"] = mask_tensor
                            model_config["motif_structure"] = x_motif_tensor
                            if mask_tensor.shape[1] > 0:
                                model_config["n_residues"] = int(mask_tensor.shape[1])
                                noise_dims = (int(mask_tensor.shape[1]), 3)
                except Exception as e:
                    print(f"[WARNING] Failed to parse motif: {e}. Continuing without motif conditioning.")
                    # Continue without motif if parsing fails

            # Drop helper blocks that should not be forwarded to wrappers
            model_config.pop("lengths", None)

            # Pass proteina config in context for diversity tracking option
            proteina_cfg = self.cfg.get("proteina", {})
            if proteina_cfg:
                context = dict(context) if context else {}
                context["proteina_config"] = self._to_container(proteina_cfg)

            # Lazy import to avoid pulling in dependencies when not needed
            from .protein import ProteinOptimizationProblem

            problem = ProteinOptimizationProblem(
                generative_model=self.generative_model,
                reward_model=self.reward,
                device=self.device,
                noise_dims=noise_dims,
                model_config=model_config,
                context=context,
            )
            return problem

        # Handle molecule modalities (qm9/molecule)
        if modality in ("molecule", "qm9"):
            model_config: Dict[str, Any] = {}
            cfg_candidates = [
                self.cfg.get("molecule"),
                self.cfg.get("model"),
                self.cfg.get("generative_model"),
            ]
            for cfg_src in cfg_candidates:
                model_config.update(self._to_container(cfg_src))

            # Update wrapper settings BEFORE discovering noise_dims
            if hasattr(self.generative_model, "update_settings_from_config") and model_config:
                try:
                    self.generative_model.update_settings_from_config(model_config)  # type: ignore[attr-defined]
                except Exception:
                    pass

            # Discover noise_dims
            noise_dims = None
            if hasattr(self.generative_model, "get_noise_latent_shape"):
                try:
                    noise_dims = tuple(self.generative_model.get_noise_latent_shape())  # type: ignore[attr-defined]
                except Exception:
                    noise_dims = None
            elif hasattr(self.generative_model, "sample_latents"):
                try:
                    sample = self.generative_model.sample_latents(batch_size=1)  # type: ignore[attr-defined]
                    noise_dims = tuple(sample.shape[1:])
                except Exception:
                    noise_dims = None

            from .molecule import QM9OptimizationProblem

            problem = QM9OptimizationProblem(
                generative_model=self.generative_model,
                reward_model=self.reward,
                device=self.device,
                noise_dims=noise_dims,
                model_config=model_config,
                context=context,
                track_stability=True,
            )
            return problem

        raise ValueError(f"Unsupported modality in task: {modality}")

    @staticmethod
    def _to_container(cfg_fragment: Any) -> Dict[str, Any]:
        if cfg_fragment is None:
            return {}
        if isinstance(cfg_fragment, DictConfig):
            return dict(OmegaConf.to_container(cfg_fragment, resolve=True))
        if isinstance(cfg_fragment, dict):
            return dict(cfg_fragment)
        return {}
