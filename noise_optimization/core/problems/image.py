from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import torch

from .base import Problem
from ..rewards.base import RewardFunction


def _sanitize_decoded_for_reward(decoded: Any) -> Any:
    """Sanitize decoded images before reward evaluation.
    Replaces NaN/inf and clamps to [0, 1] so CLIP/other processors can convert to uint8.
    """
    if isinstance(decoded, torch.Tensor):
        out = decoded.clone()
        out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
        out = out.clamp(0.0, 1.0)
        return out
    if isinstance(decoded, (list, tuple)):
        sanitized = []
        for item in decoded:
            if isinstance(item, torch.Tensor):
                x = item.clone()
                x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
                x = x.clamp(0.0, 1.0)
                sanitized.append(x)
            else:
                sanitized.append(item)
        return type(decoded)(sanitized)
    return decoded


def _debug_batch_latents_unique(t: torch.Tensor, tag: str) -> None:
    """If NOISE_OPT_DEBUG_BATCH_LATENTS=1, log shape and uniqueness via logging (DEBUG level)."""
    if os.environ.get("NOISE_OPT_DEBUG_BATCH_LATENTS", "0") != "1":
        return
    import logging
    log = logging.getLogger(__name__)
    if not log.isEnabledFor(logging.DEBUG):
        return
    b = t.shape[0]
    if b <= 1:
        log.debug("%s shape=%s batch_size=%d (no uniqueness check)", tag, tuple(t.shape), b)
        return
    diff_from_first = (t - t[0:1]).abs().sum(dim=tuple(range(1, t.dim())))
    n_different = (diff_from_first > 1e-5).sum().item()
    all_unique = n_different >= (b - 1)
    min_diff = diff_from_first[1:].min().item() if b > 1 else 0.0
    max_diff = diff_from_first.max().item()
    log.debug(
        "%s shape=%s batch_size=%d unique_rows=%s n_different=%d min_diff=%.6f max_diff=%.6f",
        tag, tuple(t.shape), b, all_unique, n_different, min_diff, max_diff,
    )
    if not all_unique:
        log.debug("%s not all batch rows unique; same latent may be evaluated multiple times", tag)


class ImageGenerationProblem(Problem):
    """Flexible problem for text-to-image optimization using a reward model."""

    def __init__(
        self,
        prompt: str,
        generative_model: Any,
        reward_model: RewardFunction,
        device: Optional[str] = None,
        latent_shape: Optional[tuple] = (4, 64, 64),
        model_config: Optional[Dict[str, Any]] = None,
        reward_expects_pil: bool = False,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self._prompt = prompt
        self.generative_model = generative_model
        self.reward_model = reward_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.latent_shape = latent_shape
        self.model_config = model_config or {}
        self.reward_expects_pil = reward_expects_pil
        self._context = context or {}
        self._current_iteration = 0
        self._printed_nonfinite_reward_warning = False


    def set_iteration(self, iteration: int) -> None:
        self._current_iteration = iteration

    def evaluate(self, candidates: torch.Tensor, noise_list: Optional[List[torch.Tensor]] = None, grad_enabled: bool = False, **kwargs: Any) -> torch.Tensor:
        if not isinstance(candidates, torch.Tensor):
            raise TypeError("candidates must be a torch.Tensor of latents")
        
        # When gradients are enabled, we MUST use float32. 
        # float16 often causes detaching in the VAE and numerical instability in backprop.
        target_dtype = torch.float32 if grad_enabled else (getattr(self.generative_model, "dtype", torch.float32))
        candidates = candidates.to(self.device, dtype=target_dtype)
        _debug_batch_latents_unique(candidates, "ImageProblem.evaluate(candidates)")
        
        # CRITICAL: When gradients are enabled, ensure candidates are not detached
        if grad_enabled:
            if not candidates.requires_grad:
                candidates = candidates.requires_grad_(True)

        # Track the number of samples evaluated
        self._total_samples_evaluated += candidates.shape[0]

        model_params = self.model_config.copy()
        model_params.update(kwargs)
        model_params['initial_latent_noise'] = candidates
        model_params['noise_list'] = noise_list
        model_params['differentiable'] = grad_enabled
        model_params['dtype'] = target_dtype  # Force the model to use the target dtype

        # If gradients are enabled, we MUST use tensor output to preserve gradients.
        # PIL conversion requires detaching tensors, which breaks the gradient graph.
        # Override reward_expects_pil when grad_enabled=True.
        if grad_enabled:
            # Force tensor output when gradients are needed
            model_params["output_type"] = "pt"
        elif self.reward_expects_pil and "output_type" not in model_params:
            # If the configured reward expects PIL images (e.g., ImageReward/HPSv2),
            # force the generative model to return PIL images. This avoids brittle
            # tensor->PIL conversion downstream.
            model_params["output_type"] = "pil"

        if grad_enabled:
            # Ensure we're in a gradient-enabled context
            with torch.set_grad_enabled(True):
                decoded = self.generative_model.forward(self._prompt, **model_params)
        else:
            decoded = self.generative_model.forward(self._prompt, **model_params)
        
        return self.evaluate_decoded(decoded)

    def decode_latents(self, latents: torch.Tensor, output_type: Optional[str] = None, grad_enabled: bool = False, noise_list: Optional[List[torch.Tensor]] = None, **kwargs: Any) -> Any:
        fmt = output_type or ("pil" if self.reward_expects_pil else "pt")
        if hasattr(self.generative_model, "decode_latents"):
            return self.generative_model.decode_latents(latents, output_type=fmt, grad_enabled=grad_enabled)  # type: ignore[attr-defined]
        # Fallback: treat latents as initial noise and run the full model
        # Extract debug parameter (used for logging, not passed to model)
        kwargs.pop("debug", None)
        fallback_kwargs = dict(self.model_config)
        fallback_kwargs.update(kwargs)
        fallback_kwargs["initial_latent_noise"] = latents
        fallback_kwargs["noise_list"] = noise_list
        fallback_kwargs["differentiable"] = grad_enabled
        return self.generative_model.forward(self._prompt, **fallback_kwargs)

    def evaluate_decoded(self, decoded: Any, extra_context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        context = {"prompt": self._prompt}
        if extra_context:
            context.update(extra_context)

        decoded = _sanitize_decoded_for_reward(decoded)
        
        # Count number of samples being evaluated in this call
        if isinstance(decoded, torch.Tensor):
            n_samples = decoded.shape[0] if decoded.dim() >= 4 else (1 if decoded.dim() >= 3 else decoded.shape[0] if decoded.dim() >= 1 else 1)
        elif isinstance(decoded, (list, tuple)):
            n_samples = len(decoded)
        else:
            n_samples = 1
        
        rewards = self.reward_model.evaluate(decoded, context)
        
        # Track reward evaluation count for reporting in metadata
        if not hasattr(self, '_reward_eval_count'):
            self._reward_eval_count = 0
        self._reward_eval_count += n_samples

        if not isinstance(rewards, torch.Tensor):
            rewards = torch.as_tensor(rewards, dtype=torch.float32)
        rewards = rewards.view(-1).to(torch.float32)
        # One-time diagnostic: if rewards are non-finite, report it (helps debug solver-specific issues).
        try:
            finite = torch.isfinite(rewards)
            if (not self._printed_nonfinite_reward_warning) and (not bool(finite.all().item())):
                bad = int((~finite).sum().item())
                total = int(rewards.numel())
                rname = getattr(self.reward_model, "name", type(self.reward_model).__name__)
                print(
                    f"[ImageProblem] Warning: reward '{rname}' produced {bad}/{total} non-finite values "
                    f"for prompt={self._prompt!r} (decoded type={type(decoded)}). Replacing with -1e9."
                )
                self._printed_nonfinite_reward_warning = True
        except Exception:
            pass
        # Sanitize non-finite values to prevent -inf/NaN from breaking optimization
        rewards = torch.nan_to_num(rewards, nan=-1e9, neginf=-1e9, posinf=1e9)
        return rewards

    def evaluate_images(self, images: Any, extra_context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """Evaluate already decoded images with the configured reward model."""
        context = {"prompt": self._prompt}
        if extra_context:
            context.update(extra_context)
        rewards = self.reward_model.evaluate(images, context)
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        return rewards.view(-1)

    def decode_images(self, latents: torch.Tensor, output_type: Optional[str] = None) -> Any:
        return self.decode_latents(latents, output_type=output_type)

    def get_initial_latents(self) -> Optional[torch.Tensor]:
        """Return the pipeline's baseline latents, if available."""
        gm = getattr(self, "generative_model", None)
        latents = None
        if gm is not None:
            latents = getattr(gm, "latents", None)
        if latents is None:
            return None
        return latents.detach().clone()

    def supports_denoising_callbacks(self) -> bool:
        return hasattr(self.generative_model, "run_with_callback")

    def sample(self, batch_size: int, latent_shape: Optional[tuple] = None) -> torch.Tensor:
        shape = latent_shape or self.latent_shape
        if shape is None:
            raise ValueError("latent_shape must be provided for ImageGenerationProblem")
        return torch.randn(batch_size, *shape, device=self.device)

    @property
    def context(self) -> Dict[str, Any]:
        base_context = {
            "prompt": self._prompt,
            "domain": "image",
            "latent_shape": self.latent_shape,
            "model_config": self.model_config,
            "device": self.device,
        }
        return {**base_context, **self._context}


