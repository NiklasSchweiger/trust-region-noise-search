from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import torch


def _debug_sdxl_batch_latents(latents: torch.Tensor, tag: str = "SDXLSamplingPipeline") -> None:
    """If NOISE_OPT_DEBUG_BATCH_LATENTS=1, log shape and uniqueness via logging (DEBUG level)."""
    if os.environ.get("NOISE_OPT_DEBUG_BATCH_LATENTS", "0") != "1":
        return
    import logging
    log = logging.getLogger(__name__)
    if not log.isEnabledFor(logging.DEBUG):
        return
    b = latents.shape[0]
    if b <= 1:
        log.debug("%s latents shape=%s batch_size=%d", tag, tuple(latents.shape), b)
        return
    diff_from_first = (latents - latents[0:1]).abs().sum(dim=tuple(range(1, latents.dim())))
    n_different = (diff_from_first > 1e-5).sum().item()
    all_unique = n_different >= (b - 1)
    min_d = float(diff_from_first[1:].min()) if b > 1 else 0.0
    max_d = float(diff_from_first.max())
    log.debug(
        "%s latents shape=%s batch_size=%d unique_rows=%s n_different=%d min_diff=%.6f max_diff=%.6f",
        tag, tuple(latents.shape), b, all_unique, n_different, min_d, max_d,
    )
    if not all_unique:
        log.debug("%s not all latents unique; same noise may be passed to the model", tag)

from diffusers.pipelines import DiffusionPipeline
from .full_pipelines.stable_diffusion import StableDiffusionPipeline
from .full_pipelines.stable_diffusion_xl import StableDiffusionXLPipeline

try:
    from diffusers import SanaSprintPipeline as DiffusersSanaSprintPipeline
    SANA_AVAILABLE = True
except ImportError:
    DiffusersSanaSprintPipeline = None
    SANA_AVAILABLE = False


class SamplingPipeline(ABC):
    """Abstract sampling wrapper compatible with the core GenerativeModel protocol.

    Provides optional helpers for latent shape discovery and latent sampling.
    """

    def __init__(
        self,
        pipeline: DiffusionPipeline,
        prompt: str,
        num_inference_steps: int,
        classifier_free_guidance: bool = True,
        guidance_scale: float = 7.5,
        height: int = 512,
        width: int = 512,
        generator: torch.Generator = torch.Generator(),
        output_type: str = "pil",
    ):
        super().__init__()
        self.device = pipeline.device if hasattr(pipeline, "device") else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.pipeline = pipeline
        self.prompt = prompt
        self.num_inference_steps = num_inference_steps
        self.height = height
        self.width = width
        self.generator = generator
        self.guidance_scale = guidance_scale
        self.classifier_free_guidance = classifier_free_guidance or guidance_scale > 0.0
        self.output_type = output_type

    # ---- Interface expected by core.problem.GenerativeModel protocol ----
    def get_noise_latent_shape(self) -> Tuple[int, ...]:  # pragma: no cover - implemented in subclasses
        sample = self.sample_latents(batch_size=1)
        return tuple(sample.shape[1:])

    def sample_latents(self, batch_size: int = 1) -> torch.Tensor:  # pragma: no cover - implemented in subclasses
        raise NotImplementedError

    def forward(
        self,
        prompt: str,
        *,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        num_images_per_prompt: Optional[int] = None,
        initial_latent_noise: Optional[torch.Tensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        output_type: str = "pt",
        generator: Optional[torch.Generator] = None,
        save_trajectory: bool = False,
        differentiable: bool = False,
        noise_list: Optional[List[torch.Tensor]] = None,
        eta: Optional[float] = None,
        max_sequence_length: Optional[int] = None,
        **kwargs: Any,
    ) -> Any:
        # Default impl delegates to __call__ in subclasses; kept for compatibility
        return self.__call__(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            initial_latent_noise=initial_latent_noise,
            height=height,
            width=width,
            output_type=output_type,
            generator=generator,
            grad_enabled=differentiable,
            noise_list=noise_list,
            eta=eta,
            max_sequence_length=max_sequence_length,
            **kwargs,
        )

    # ---- Subclass responsibilities ----
    @abstractmethod
    def embed_text(self, prompt: str):
        raise NotImplementedError

    @abstractmethod
    def generate_latents(self) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def __call__(
        self,
        *,
        prompt: str,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        num_images_per_prompt: Optional[int] = None,
        initial_latent_noise: Optional[torch.Tensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        output_type: str = "pt",
        generator: Optional[torch.Generator] = None,
        grad_enabled: bool = False,
        noise_list: Optional[List[torch.Tensor]] = None,
        eta: Optional[float] = None,
        max_sequence_length: Optional[int] = None,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def decode_latents(self, latents: torch.Tensor, output_type: str = "pt", grad_enabled: bool = False) -> Any:
        """Decode latent tensors into pixel space using the underlying VAE.
        
        Each pipeline has different VAE handling (scaling, denormalization, etc.),
        so this must be implemented by each subclass.
        """
        raise NotImplementedError




class SanaSprintSamplingPipeline(SamplingPipeline):
    """Wrapper for SanaSprintPipeline (SANA-Sprint) for noise optimization.

    SANA-Sprint is an ultra-fast 1-4 step distilled text-to-image model.
    Uses transformer-based DiT architecture, accepts latents for optimization.
    """

    def __init__(
        self,
        pipeline: "DiffusersSanaSprintPipeline",
        prompt: str,
        num_inference_steps: int,
        classifier_free_guidance: bool = True,
        guidance_scale: float = 4.5,
        height: int = 1024,
        width: int = 1024,
        generator: torch.Generator = torch.Generator(),
        add_noise: bool = True,
        output_type: str = "pt",
    ):
        super().__init__(
            pipeline,
            prompt,
            num_inference_steps,
            classifier_free_guidance,
            guidance_scale,
            height,
            width,
            generator,
            output_type,
        )
        self.add_noise = add_noise
        embeds, mask = self.embed_text(prompt)
        self.prompt_embeds = embeds
        self.prompt_attention_mask = mask
        self.latents = self.generate_latents()

        # Disable NSFW filtering
        if hasattr(self.pipeline, "safety_checker"):
            self.pipeline.safety_checker = None
        if hasattr(self.pipeline, "feature_extractor"):
            self.pipeline.feature_extractor = None

    @torch.inference_mode()
    def embed_text(self, prompt: str):
        result = self.pipeline.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
        )
        if isinstance(result, (list, tuple)):
            prompt_embeds = result[0]
            prompt_attention_mask = result[1] if len(result) > 1 else None
        else:
            prompt_embeds = result
            prompt_attention_mask = None
        return prompt_embeds, prompt_attention_mask

    @torch.inference_mode()
    def generate_latents(self):
        # Sana uses a 32x compression factor (DC-AE)
        vae_scale_factor = getattr(self.pipeline, "vae_scale_factor", 32)
        
        transformer = getattr(self.pipeline, "transformer", None)
        if transformer is not None and hasattr(transformer.config, "in_channels"):
            num_channel_latents = transformer.config.in_channels
        else:
            # Sana default is typically 32 channels for f32c32 models
            num_channel_latents = 32 
            
        height = int(self.height) // vae_scale_factor
        width = int(self.width) // vae_scale_factor
        
        latents = torch.randn(
            (1, num_channel_latents, height, width),
            device=self.pipeline.device,
            dtype=self.pipeline.dtype,
            generator=self.generator,
        )
        return latents

    def sample_latents(self, batch_size: int = 1) -> torch.Tensor:
        shape = (batch_size,) + tuple(self.generate_latents().shape[1:])
        return torch.randn(shape, device=self.pipeline.device, dtype=self.pipeline.dtype)

    def regenerate_latents(self):
        self.latents = self.generate_latents()

    def rembed_text(self, prompt: str):
        embeds, mask = self.embed_text(prompt)
        self.prompt_embeds = embeds
        self.prompt_attention_mask = mask

    @torch.inference_mode()
    def __call__(
        self,
        *,
        prompt: str,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        num_images_per_prompt: Optional[int] = None,
        initial_latent_noise: Optional[torch.Tensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        output_type: str = "pt",
        generator: Optional[torch.Generator] = None,
        noise_list: Optional[List[torch.Tensor]] = None,
        eta: Optional[float] = None,
        max_sequence_length: Optional[int] = None,
        **kwargs: Any,
    ):
        if prompt != self.prompt:
            self.rembed_text(prompt)
            self.prompt = prompt
        latents = self.latents
        if initial_latent_noise is not None:
            latents = latents + initial_latent_noise if self.add_noise else initial_latent_noise
        latents = latents.to(self.device, dtype=self.pipeline.dtype)

        batch_size = latents.shape[0]
        actual_num_images = num_images_per_prompt if num_images_per_prompt is not None else batch_size
        prompt_embeds = self.prompt_embeds
        prompt_attention_mask = getattr(self, "prompt_attention_mask", None)
        if actual_num_images > 1 and prompt_embeds.shape[0] < actual_num_images:
            prompt_embeds = prompt_embeds.repeat(actual_num_images, 1, 1)
            if prompt_attention_mask is not None:
                prompt_attention_mask = prompt_attention_mask.repeat(actual_num_images, 1)

        num_images_for_pipeline = 1 if actual_num_images > 1 else (num_images_per_prompt or 1)

        call_kwargs = dict(
            height=height or self.height,
            width=width or self.width,
            num_inference_steps=num_inference_steps or self.num_inference_steps,
            guidance_scale=guidance_scale if guidance_scale is not None else self.guidance_scale,
            prompt_embeds=prompt_embeds,
            generator=generator or self.generator,
            num_images_per_prompt=num_images_for_pipeline,
            latents=latents,
            output_type=output_type,
        )
        if prompt_attention_mask is not None:
            call_kwargs["prompt_attention_mask"] = prompt_attention_mask
        if eta is not None:
            call_kwargs["eta"] = float(eta)
        if max_sequence_length is not None:
            call_kwargs["max_sequence_length"] = max_sequence_length

        images = self.pipeline(**call_kwargs)
        result = images.images if hasattr(images, "images") else images

        # --- FIX: Support for Reward Models (CLIP) ---
        # Ensure the final output is Float32, otherwise CLIP/NumPy crashes on BFloat16
        if isinstance(result, torch.Tensor):
            result = result.to(torch.float32)
        elif isinstance(result, list) and len(result) > 0 and isinstance(result[0], torch.Tensor):
            result = [img.to(torch.float32) for img in result]
            if output_type == "pt":
                result = torch.stack(result)

        if output_type == "pt" and isinstance(result, list) and len(result) > 0:
            from PIL import Image
            import torchvision.transforms as transforms
            if isinstance(result[0], Image.Image):
                to_tensor = transforms.ToTensor()
                result = torch.stack([to_tensor(img) for img in result])

        return result

    def decode_latents(self, latents: torch.Tensor, output_type: str = "pt", grad_enabled: bool = False) -> Any:
        vae = self.pipeline.vae
        scale = getattr(vae.config, "scaling_factor", 1.0)
        
        was_unbatched = False
        if latents.dim() == 3:
            latents = latents.unsqueeze(0)
            was_unbatched = True

        with torch.set_grad_enabled(grad_enabled):
            # Convert to float32 for VAE decoding to avoid NaNs/Black images
            latents = latents.to(self.device, dtype=torch.float32)
            vae.to(dtype=torch.float32)
            
            latents = latents / scale
            decoded = vae.decode(latents, return_dict=False)[0]
            
        if output_type in ("pt", "tensor", None):
            normalized = (decoded * 0.5 + 0.5).clamp(0, 1)
            # Ensure output is float32 for CLIP reward processor compatibility
            normalized = normalized.to(torch.float32)
            
            if was_unbatched and normalized.shape[0] == 1:
                return normalized.squeeze(0)
            return normalized

        # Fallback processor logic
        processor = getattr(self.pipeline, "image_processor", None)
        if processor is not None:
            batch_size = decoded.shape[0]
            do_denormalize = [True] * batch_size
            # Result usually comes back as PIL or Tensor based on output_type
            result = processor.postprocess(decoded.to(torch.float32), output_type=output_type, do_denormalize=do_denormalize)
            return result[0] if was_unbatched and isinstance(result, list) else result
            
        return (decoded * 0.5 + 0.5).clamp(0, 1).to(torch.float32)


class SDXLSamplingPipeline(SamplingPipeline):
    def __init__(
        self,
        pipeline: StableDiffusionXLPipeline,
        prompt: str,
        num_inference_steps: int,
        classifier_free_guidance: bool = True,
        guidance_scale: float = 7.5,
        height: int = 512,
        width: int = 512,
        generator: torch.Generator = torch.Generator(),
        add_noise: bool = True,
        output_type: str = "pt",
    ):
        super().__init__(
            pipeline,
            prompt,
            num_inference_steps,
            classifier_free_guidance,
            guidance_scale,
            height,
            width,
            generator,
            output_type,
        )
        self.add_noise = add_noise
        (
            self.prompt_embeds,
            self.negative_prompt_embeds,
            self.pooled_prompt_embeds,
            self.negative_pooled_prompt_embeds,
        ) = self.embed_text(prompt)
        self.latents = self.generate_latents()

    @torch.inference_mode()
    def embed_text(self, prompt: str):
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipeline.encode_prompt(
            prompt=prompt,
            device=self.device,
            do_classifier_free_guidance=self.classifier_free_guidance,
            num_images_per_prompt=1,
        )
        return (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

    @torch.inference_mode()
    def generate_latents(self):
        num_channel_latents = self.pipeline.unet.config.in_channels
        height = int(self.height) // self.pipeline.vae_scale_factor
        width = int(self.width) // self.pipeline.vae_scale_factor
        latents = torch.randn(
            (1, num_channel_latents, height, width),
            device=self.pipeline.device,
            dtype=self.pipeline.dtype,
            generator=self.generator,
        )
        return latents

    def sample_latents(self, batch_size: int = 1) -> torch.Tensor:
        shape = (batch_size,) + tuple(self.generate_latents().shape[1:])
        # Use a fresh random tensor for each sample to ensure diversity
        return torch.randn(shape, device=self.pipeline.device, dtype=self.pipeline.dtype)

    def regenerate_latents(self):
        self.latents = self.generate_latents()

    def rembed_text(self, prompt: str):
        (
            self.prompt_embeds,
            self.negative_prompt_embeds,
            self.pooled_prompt_embeds,
            self.negative_pooled_prompt_embeds,
        ) = self.embed_text(prompt)

    def __call__(
        self,
        *,
        prompt: str,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        num_images_per_prompt: Optional[int] = None,
        initial_latent_noise: Optional[torch.Tensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        output_type: str = "pt",
        generator: Optional[torch.Generator] = None,
        grad_enabled: bool = False,
        noise_list: Optional[List[torch.Tensor]] = None,
        eta: Optional[float] = None,
        max_sequence_length: Optional[int] = None,
        **kwargs: Any,
    ):
        if prompt != self.prompt:
            self.rembed_text(prompt)
            self.prompt = prompt
        latents = self.latents
        if initial_latent_noise is not None:
            # Note: We do NOT scale initial_latent_noise here.
            # The diffusers pipeline handles the scaling by scheduler.init_noise_sigma internally
            # if we pass latents. Manually scaling here causes "blown out" or solid color images.
            latents = latents + initial_latent_noise if self.add_noise else initial_latent_noise
        latents = latents.to(self.device, dtype=self.pipeline.dtype)
        _debug_sdxl_batch_latents(latents, "SDXLSamplingPipeline.__call__")

        # Determine the actual batch size from latents
        batch_size = latents.shape[0]
        actual_num_images = num_images_per_prompt if num_images_per_prompt is not None else batch_size
        
        # CRITICAL FIX: Repeat prompt embeddings to match batch size
        # The embeddings were computed for num_images_per_prompt=1, so we need to repeat them
        # for batch generation to work correctly with pre-computed embeddings
        prompt_embeds = self.prompt_embeds
        negative_prompt_embeds = self.negative_prompt_embeds
        pooled_prompt_embeds = self.pooled_prompt_embeds
        negative_pooled_prompt_embeds = self.negative_pooled_prompt_embeds
        
        if actual_num_images > 1:
            if prompt_embeds is not None and prompt_embeds.shape[0] < actual_num_images * 2:
                prompt_embeds = prompt_embeds.repeat(actual_num_images, 1, 1)
            if negative_prompt_embeds is not None and negative_prompt_embeds.shape[0] < actual_num_images:
                negative_prompt_embeds = negative_prompt_embeds.repeat(actual_num_images, 1, 1)
            if pooled_prompt_embeds is not None and pooled_prompt_embeds.shape[0] < actual_num_images:
                pooled_prompt_embeds = pooled_prompt_embeds.repeat(actual_num_images, 1)
            if negative_pooled_prompt_embeds is not None and negative_pooled_prompt_embeds.shape[0] < actual_num_images:
                negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(actual_num_images, 1)

        # When we pass pre-expanded embeddings (actual_num_images > 1), we must pass
        # num_images_per_prompt=1 so the pipeline's encode_prompt does not expand again
        # (it would do repeat(1, num_images_per_prompt, 1), causing batch_size*num_images_per_prompt mismatch).
        num_images_for_pipeline = 1 if actual_num_images > 1 else (num_images_per_prompt or 1)

        pipeline_kwargs = dict(
            height=height or self.height,
            width=width or self.width,
            num_inference_steps=num_inference_steps or self.num_inference_steps,
            guidance_scale=guidance_scale if guidance_scale is not None else self.guidance_scale,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            generator=generator or self.generator,
            num_images_per_prompt=num_images_for_pipeline,
            latents=latents,
            output_type=output_type,
            noise_list=noise_list,
            differentiable=grad_enabled,
        )
        if eta is not None:
            pipeline_kwargs["eta"] = float(eta)
        if max_sequence_length is not None:
            pipeline_kwargs["max_sequence_length"] = max_sequence_length

        with torch.set_grad_enabled(grad_enabled):
            images = self.pipeline(**pipeline_kwargs)
        result = images.images if hasattr(images, "images") else images

        # Convert PIL to tensor if output_type="pt" was requested
        if output_type == "pt" and isinstance(result, list) and len(result) > 0:
            from PIL import Image
            import torchvision.transforms as transforms
            if isinstance(result[0], Image.Image):
                to_tensor = transforms.ToTensor()
                result = torch.stack([to_tensor(img) for img in result])

        return result

    def decode_latents(self, latents: torch.Tensor, output_type: str = "pt", grad_enabled: bool = False) -> Any:
        """Decode latent tensors into pixel space using SDXL's VAE.
        
        SDXL VAE requires special denormalization with latents_mean and latents_std
        if available in the VAE config. It also often requires upcasting to float32
        to avoid NaNs in float16.
        """
        if not hasattr(self.pipeline, "vae"):
            raise RuntimeError("SDXL pipeline does not expose a VAE for latent decoding.")
        
        vae = self.pipeline.vae
        
        # SDXL VAE often overflows in float16, so we upcast to float32 for decoding
        # and then move it back to its original dtype.
        original_vae_dtype = vae.dtype
        needs_upcasting = original_vae_dtype == torch.float16 and getattr(vae.config, "force_upcast", True)
        
        with torch.set_grad_enabled(grad_enabled):
            if needs_upcasting:
                vae.to(dtype=torch.float32)
                latents = latents.to(device=self.device, dtype=torch.float32)
            else:
                latents = latents.to(self.device, dtype=original_vae_dtype)
            
            # SDXL VAE has special denormalization
            has_latents_mean = hasattr(vae.config, "latents_mean") and vae.config.latents_mean is not None
            has_latents_std = hasattr(vae.config, "latents_std") and vae.config.latents_std is not None
            
            if has_latents_mean and has_latents_std:
                latents_mean = torch.tensor(vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                latents_std = torch.tensor(vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                latents = latents * latents_std / vae.config.scaling_factor + latents_mean
            else:
                latents = latents / vae.config.scaling_factor
            
            # Decode
            decoded = vae.decode(latents, return_dict=False)[0]
        
        # Move VAE back to original dtype if we upcasted
        if needs_upcasting:
            vae.to(dtype=original_vae_dtype)
        
        # Convert to output format
        if output_type in ("pt", "tensor", None):
            # Return normalized tensor [0, 1]
            return (decoded / 2 + 0.5).clamp(0, 1)
        
        # Use image processor if available
        processor = getattr(self.pipeline, "image_processor", None)
        if processor is not None:
            return processor.postprocess(decoded, output_type=output_type)
        
        # Fallback: return normalized tensor
        return (decoded / 2 + 0.5).clamp(0, 1)
    


class SDSamplingPipeline(SamplingPipeline):
    def __init__(
        self,
        pipeline: StableDiffusionPipeline,
        prompt: str,
        num_inference_steps: int,
        classifier_free_guidance: bool = True,
        guidance_scale: float = 7.5,
        height: int = 512,
        width: int = 512,
        generator: torch.Generator = torch.Generator(),
        add_noise: bool = True,
        output_type: str = "pt",
    ):
        super().__init__(
            pipeline,
            prompt,
            num_inference_steps,
            classifier_free_guidance,
            guidance_scale,
            height,
            width,
            generator,
            output_type,
        )
        self.add_noise = add_noise
        (
            self.prompt_embeds,
            self.negative_prompt_embeds,
        ) = self.embed_text(prompt)
        self.latents = self.generate_latents()

    @torch.inference_mode()
    def embed_text(self, prompt: str):
        (
            prompt_embeds,
            negative_prompt_embeds,
        ) = self.pipeline.encode_prompt(
            prompt=prompt,
            device=self.device,
            do_classifier_free_guidance=self.classifier_free_guidance,
            num_images_per_prompt=1,
        )
        return (
            prompt_embeds,
            negative_prompt_embeds,
        )

    @torch.inference_mode()
    def generate_latents(self):
        num_channel_latents = self.pipeline.unet.config.in_channels
        height = int(self.height) // self.pipeline.vae_scale_factor
        width = int(self.width) // self.pipeline.vae_scale_factor
        latents = torch.randn(
            (1, num_channel_latents, height, width),
            device=self.pipeline.device,
            dtype=self.pipeline.dtype,
            generator=self.generator,
        )
        return latents

    def sample_latents(self, batch_size: int = 1) -> torch.Tensor:
        shape = (batch_size,) + tuple(self.generate_latents().shape[1:])
        # Use a fresh random tensor for each sample to ensure diversity
        return torch.randn(shape, device=self.pipeline.device, dtype=self.pipeline.dtype)

    def regenerate_latents(self):
        self.latents = self.generate_latents()

    def rembed_text(self, prompt: str):
        (
            self.prompt_embeds,
            self.negative_prompt_embeds,
        ) = self.embed_text(prompt)

    def __call__(
        self,
        *,
        prompt: str,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        num_images_per_prompt: Optional[int] = None,
        initial_latent_noise: Optional[torch.Tensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        output_type: str = "pt",
        generator: Optional[torch.Generator] = None,
        grad_enabled: bool = False,
        noise_list: Optional[List[torch.Tensor]] = None,
        eta: Optional[float] = None,
        max_sequence_length: Optional[int] = None,
        **kwargs: Any,
    ):
        if prompt != self.prompt:
            self.rembed_text(prompt)
            self.prompt = prompt
        latents = self.latents
        if initial_latent_noise is not None:
            latents = latents + initial_latent_noise if self.add_noise else initial_latent_noise
        
        # Use provided dtype if available (e.g. from ImageProblem.evaluate), 
        # otherwise fallback to pipeline's native dtype.
        target_dtype = kwargs.get("dtype", self.pipeline.dtype)
        latents = latents.to(self.device, dtype=target_dtype)
        
        # Determine the actual batch size from latents
        batch_size = latents.shape[0]
        actual_num_images = num_images_per_prompt if num_images_per_prompt is not None else batch_size
        
        # CRITICAL FIX: Repeat prompt embeddings to match batch size
        # The embeddings were computed for num_images_per_prompt=1, so we need to repeat them
        prompt_embeds = self.prompt_embeds
        negative_prompt_embeds = self.negative_prompt_embeds
        
        if actual_num_images > 1:
            if prompt_embeds is not None and prompt_embeds.shape[0] < actual_num_images * 2:
                prompt_embeds = prompt_embeds.repeat(actual_num_images, 1, 1)
            if negative_prompt_embeds is not None and negative_prompt_embeds.shape[0] < actual_num_images:
                negative_prompt_embeds = negative_prompt_embeds.repeat(actual_num_images, 1, 1)
        
        num_images_for_pipeline = 1 if actual_num_images > 1 else (num_images_per_prompt or 1)
        
        call_kwargs = dict(
            height=height or self.height,
            width=width or self.width,
            num_inference_steps=num_inference_steps or self.num_inference_steps,
            guidance_scale=guidance_scale if guidance_scale is not None else self.guidance_scale,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            generator=generator or self.generator,
            num_images_per_prompt=num_images_for_pipeline,
            latents=latents,
            output_type=output_type,
            noise_list=noise_list,
            differentiable=grad_enabled,
        )
        if eta is not None:
            call_kwargs["eta"] = float(eta)
        if max_sequence_length is not None:
            call_kwargs["max_sequence_length"] = max_sequence_length

        with torch.set_grad_enabled(grad_enabled):
            images = self.pipeline(**call_kwargs)
        return images.images if hasattr(images, "images") else images

    def decode_latents(self, latents: torch.Tensor, output_type: str = "pt", grad_enabled: bool = False) -> Any:
        """Decode latent tensors into pixel space using SD1.5's VAE.

        Handles both batched [B, C, H, W] and unbatched [C, H, W] inputs.
        Returns normalized [0, 1] tensors for output_type="pt", or uses the
        image processor for other output types (e.g. "pil").
        """
        if not hasattr(self.pipeline, "vae"):
            raise RuntimeError("SD1.5 pipeline does not expose a VAE for latent decoding.")
        
        vae = self.pipeline.vae
        scale = getattr(vae.config, "scaling_factor", 1.0)
        
        # Handle both batched [B, C, H, W] and unbatched [C, H, W] latents
        # This ensures solvers can decode single latents without batch dimension issues
        was_unbatched = False
        if latents.dim() == 3:
            # Unbatched: [C, H, W] -> add batch dimension [1, C, H, W]
            latents = latents.unsqueeze(0)
            was_unbatched = True
        
        if latents.dtype != vae.dtype:
            if torch.backends.mps.is_available():
                # Some platforms (eg. apple mps) misbehave due to a pytorch bug
                vae = vae.to(latents.dtype)
        
        with torch.set_grad_enabled(grad_enabled):
            decode_dtype = vae.dtype
            latents = latents.to(self.device, dtype=decode_dtype)
            latents = latents / scale
            decoded = vae.decode(latents, return_dict=False)[0]
            # Convert decoded output back to pipeline dtype if needed
            if decoded.dtype != getattr(self.pipeline, "dtype", torch.float32):
                decoded = decoded.to(dtype=getattr(self.pipeline, "dtype", torch.float32))
        
        if output_type == "latent":
            if was_unbatched and latents.shape[0] == 1:
                return latents.squeeze(0)
            return latents
        
        if output_type in ("pt", "tensor", None):
            normalized = (decoded * 0.5 + 0.5).clamp(0, 1)
            if was_unbatched and normalized.shape[0] == 1:
                return normalized.squeeze(0)
            return normalized
        
        processor = getattr(self.pipeline, "image_processor", None)
        if processor is not None:
            # do_denormalize must be a list with one bool per image in the batch
            batch_size = decoded.shape[0] if isinstance(decoded, torch.Tensor) else len(decoded)
            do_denormalize = [True] * batch_size
            result = processor.postprocess(decoded, output_type=output_type, do_denormalize=do_denormalize)
            # For unbatched input, return single item (not list)
            if was_unbatched and isinstance(result, list) and len(result) == 1:
                return result[0]
            return result
        
        # Fallback: manual normalization
        normalized = (decoded * 0.5 + 0.5).clamp(0, 1)
        if was_unbatched and normalized.shape[0] == 1:
            return normalized.squeeze(0)
        return normalized
    


# ============================================================================
# Factory functions for T2I pipelines
# ============================================================================

def _to_torch_dtype(dtype: str) -> torch.dtype:
    name = str(dtype).lower()
    if hasattr(torch, name):
        return getattr(torch, name)
    return torch.float16


def make_sd15(
    model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
    height: int = 512,
    width: int = 512,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    add_noise: bool = False,
    output_type: str = "pil",
    device: str = "cuda",
    dtype: str = "float16",
    seed: Optional[int] = None,
    scheduler: Optional[str] = None,
):
    from .full_pipelines.stable_diffusion import StableDiffusionPipeline as HF

    torch_dtype = _to_torch_dtype(dtype)
    variant = "fp16" if torch_dtype == torch.float16 else None
    pipe = HF.from_pretrained(model_id, torch_dtype=torch_dtype, safety_checker=None, variant=variant, use_safetensors=True)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    if scheduler is not None:
        name = str(scheduler).lower().strip()
        try:
            from diffusers import DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler, EulerDiscreteScheduler

            sched_map = {
                "ddim": DDIMScheduler,
                "pndm": PNDMScheduler,
                "lms": LMSDiscreteScheduler,
                "lmsdiscrete": LMSDiscreteScheduler,
                "euler": EulerDiscreteScheduler,
                "eulerdiscrete": EulerDiscreteScheduler,
            }
            cls = sched_map.get(name)
            if cls is None:
                raise ValueError(f"unknown scheduler '{scheduler}'")
            old_scheduler_name = type(pipe.scheduler).__name__
            pipe.scheduler = cls.from_config(pipe.scheduler.config)
            new_scheduler_name = type(pipe.scheduler).__name__
            print(f"[INFO] Changed scheduler from {old_scheduler_name} to {new_scheduler_name}")
        except Exception as e:
            print(f"[WARNING] make_sd15: failed to set scheduler={scheduler!r}: {e}")

    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(int(seed))

    return SDSamplingPipeline(
        pipeline=pipe,
        prompt="",
        num_inference_steps=num_inference_steps,
        classifier_free_guidance=True,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        generator=gen,
        add_noise=add_noise,
        output_type=output_type,
    )




def make_sdxl_lightning(
    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 8,
    guidance_scale: float = 0.0,
    add_noise: bool = False,
    output_type: str = "pil",
    device: str = "cuda",
    dtype: str = "float16",
    seed: Optional[int] = None,
    scheduler: Optional[str] = None,
):
    """Create SDXL Lightning pipeline using the 8-step distilled version.
    
    This loads the base SDXL model and replaces the UNet with the Lightning
    distilled weights from ByteDance/SDXL-Lightning.
    """
    from .full_pipelines.stable_diffusion_xl import StableDiffusionXLPipeline as HF
    from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    torch_dtype = _to_torch_dtype(dtype)
    variant = "fp16" if torch_dtype == torch.float16 else None
    
    # Load base SDXL UNet
    unet = UNet2DConditionModel.from_pretrained(
        model_id, 
        subfolder="unet", 
        torch_dtype=torch_dtype, 
        use_safetensors=True, 
        variant=variant
    )
    
    # Load Lightning weights
    lightning_weights_path = hf_hub_download("ByteDance/SDXL-Lightning", "sdxl_lightning_8step_unet.safetensors")
    unet.load_state_dict(load_file(lightning_weights_path))
    
    # Load VAE (using the fp16-fix version for better stability)
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", 
        torch_dtype=torch_dtype
    )
    
    # Load scheduler with trailing timestep spacing (required for Lightning)
    scheduler_obj = DDIMScheduler.from_pretrained(
        model_id, 
        subfolder="scheduler",
        timestep_spacing="trailing"
    )
    
    # Create pipeline with custom components
    pipe = HF.from_pretrained(
        model_id, 
        unet=unet, 
        vae=vae, 
        scheduler=scheduler_obj,
        torch_dtype=torch_dtype, 
        use_safetensors=True, 
        variant=variant
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    
    # Enable VAE slicing for memory efficiency
    if hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()
    # Disable VAE encoder to save memory (only decoder is needed for generation)
    if hasattr(pipe.vae, "encoder"):
        pipe.vae.encoder = None

    # Override scheduler if specified (though Lightning typically uses DDIM with trailing)
    if scheduler is not None:
        name = str(scheduler).lower().strip()
        try:
            from diffusers import DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler, EulerDiscreteScheduler

            sched_map = {
                "ddim": DDIMScheduler,
                "pndm": PNDMScheduler,
                "lms": LMSDiscreteScheduler,
                "lmsdiscrete": LMSDiscreteScheduler,
                "euler": EulerDiscreteScheduler,
                "eulerdiscrete": EulerDiscreteScheduler,
            }
            cls = sched_map.get(name)
            if cls is not None:
                old_scheduler_name = type(pipe.scheduler).__name__
                # For DDIM, preserve trailing timestep spacing
                if cls == DDIMScheduler:
                    pipe.scheduler = cls.from_config(pipe.scheduler.config, timestep_spacing="trailing")
                else:
                    pipe.scheduler = cls.from_config(pipe.scheduler.config)
                new_scheduler_name = type(pipe.scheduler).__name__
                print(f"[INFO] Changed scheduler from {old_scheduler_name} to {new_scheduler_name}")
        except Exception as e:
            print(f"[WARNING] make_sdxl_lightning: failed to set scheduler={scheduler!r}: {e}")

    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(int(seed))

    return SDXLSamplingPipeline(
        pipeline=pipe,
        prompt="",
        num_inference_steps=num_inference_steps,
        classifier_free_guidance=True,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        generator=gen,
        add_noise=add_noise,
        output_type=output_type,
    )



def make_sana_sprint(
    model_id: str = "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers",
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 2,
    guidance_scale: float = 4.5,
    add_noise: bool = False,
    output_type: str = "pil",
    device: str = "cuda",
    dtype: str = "bfloat16",
    seed: Optional[int] = None,
    scheduler: Optional[str] = None,  # ignored for API compatibility with make_t2i_from_cfg
    **kwargs: Any,
):
    """Create SANA-Sprint pipeline for ultra-fast 1-4 step text-to-image generation.

    SANA-Sprint is an efficient distilled model from NVIDIA/MIT HAN Lab.
    Uses bfloat16 by default as recommended by the model.
    """
    if not SANA_AVAILABLE:
        raise ImportError("SanaSprintPipeline not available. Install diffusers with: pip install diffusers")
    torch_dtype = _to_torch_dtype(dtype)
    pipe = DiffusersSanaSprintPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    # Note: Do NOT enable VAE tiling/slicing for Sana - its VAE returns tensors directly
    # while diffusers' slicing path expects ModelOutput with .sample, causing AttributeError.

    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(int(seed))

    return SanaSprintSamplingPipeline(
        pipeline=pipe,
        prompt="",
        num_inference_steps=num_inference_steps,
        classifier_free_guidance=True,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        generator=gen,
        add_noise=add_noise,
        output_type=output_type,
    )


def make_t2i_from_cfg(pipeline: dict | None = None, device: str = "cuda"):
    """Factory that builds a T2I sampling pipeline from a generic pipeline dict.

    Expected keys in `pipeline`:
      - type: one of ["sd", "sdxl_lightning", "sana_sprint"]
      - model_id, height, width, num_inference_steps, guidance_scale, add_noise, output_type, dtype, seed
    """
    if pipeline is None:
        pipeline = {}
    ptype = str(pipeline.get("type", "sd")).lower()
    common = dict(
        height=int(pipeline.get("height", 512)),
        width=int(pipeline.get("width", 512)),
        num_inference_steps=int(pipeline.get("num_inference_steps", 50)),
        guidance_scale=float(pipeline.get("guidance_scale", 7.5)),
        add_noise=bool(pipeline.get("classifier_free_guidance", True) and pipeline.get("add_noise", False)),
        output_type=str(pipeline.get("output_type", "pil")),
        device=device,
        dtype=str(pipeline.get("dtype", "float16")),
        seed=pipeline.get("seed"),
        scheduler=pipeline.get("scheduler"),
    )
    if pipeline.get("model_id") is not None:
        common["model_id"] = pipeline.get("model_id")
    if ptype in {"sd", "stable-diffusion", "stable_diffusion"}:
        return make_sd15(**common)
    if ptype in {"sdxl_lightning", "sdxl-lightning", "sdxl_lightning_8step"}:
        # Adjust SDXL Lightning sensible defaults if width/height missing
        if "height" not in pipeline:
            common["height"] = 1024
        if "width" not in pipeline:
            common["width"] = 1024
        # Override defaults for Lightning (8 steps, guidance_scale=0.0)
        if "num_inference_steps" not in pipeline:
            common["num_inference_steps"] = 8
        if "guidance_scale" not in pipeline:
            common["guidance_scale"] = 0.0
        return make_sdxl_lightning(**common)
    if ptype in {"sana_sprint", "sana-sprint", "sana"}:
        if "height" not in pipeline:
            common["height"] = 1024
        if "width" not in pipeline:
            common["width"] = 1024
        if "num_inference_steps" not in pipeline:
            common["num_inference_steps"] = 2
        if "guidance_scale" not in pipeline:
            common["guidance_scale"] = 4.5
        if "dtype" not in pipeline:
            common["dtype"] = "bfloat16"
        common["model_id"] = pipeline.get("model_id", "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers")
        return make_sana_sprint(**common)
    raise ValueError(f"Unsupported pipeline.type: {ptype}")

