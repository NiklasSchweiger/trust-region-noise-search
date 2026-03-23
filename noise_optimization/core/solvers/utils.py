from __future__ import annotations

from typing import Tuple

import torch


def flatten_latents(latents: torch.Tensor) -> torch.Tensor:
    return latents.view(latents.shape[0], -1)


def unflatten_vector(vec: torch.Tensor, latent_shape: Tuple[int, ...]) -> torch.Tensor:
    return vec.view(vec.shape[0], *latent_shape)


def device_of(x: torch.Tensor) -> torch.device:
    return x.device if isinstance(x, torch.Tensor) else torch.device("cuda" if torch.cuda.is_available() else "cpu")


