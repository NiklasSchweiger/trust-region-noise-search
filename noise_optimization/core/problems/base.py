from __future__ import annotations

from typing import Any, Dict, Optional, Protocol
from abc import ABC, abstractmethod

import torch


class GenerativeModel(Protocol):
    def forward(self, prompt: str, **kwargs: Any) -> Any:
        """Generate samples from the model."""

    def get_noise_latent_shape(self) -> tuple: ...
    def sample_latents(self, batch_size: int = 1) -> torch.Tensor: ...


class Problem(ABC):
    """Defines the optimization problem: space, reward, generator, and prompt/context."""

    def __init__(self):
        self._total_samples_evaluated = 0

    @property
    def total_samples_evaluated(self) -> int:
        """Total number of individual samples evaluated by this problem."""
        return self._total_samples_evaluated

    def reset_eval_count(self) -> None:
        """Reset the evaluation counter."""
        self._total_samples_evaluated = 0

    @abstractmethod
    def evaluate(self, candidates: Any) -> torch.Tensor:
        """Return rewards for given candidates as a 1D tensor (higher is better)."""
        raise NotImplementedError

    @abstractmethod
    def sample(self, batch_size: int, latent_shape: Optional[tuple] = None) -> Any:
        """Sample initial candidates from the search space (e.g., latents)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def context(self) -> Dict[str, Any]:
        """Problem context (e.g., prompt, data domain, constraints)."""
        raise NotImplementedError

    # ---- Optional helper hooks for advanced solvers ----
    def decode_latents(self, latents: Any, **kwargs: Any) -> Any:
        """Optional hook that maps latent tensors to decoded data."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement decode_latents; "
            "solvers that require decoded states cannot be used with this problem."
        )

    def evaluate_decoded(self, decoded: Any, extra_context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """Optional hook to score already-decoded samples without re-running the generator."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement evaluate_decoded; "
            "solvers that reuse decoded states cannot be used with this problem."
        )

    def supports_denoising_callbacks(self) -> bool:
        """Return True if the underlying generator exposes callback-driven sampling."""
        return False


