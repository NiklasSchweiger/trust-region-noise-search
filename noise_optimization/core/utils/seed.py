from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """Set global RNG seeds across Python, NumPy, and PyTorch.

    Args:
        seed: The seed value to set.
        deterministic: If True, enable deterministic/cuDNN behaviors (may reduce performance).
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # cuDNN determinism knobs
    try:
        torch.backends.cudnn.deterministic = bool(deterministic)
        # Benchmark can introduce non-determinism; keep it off for determinism
        torch.backends.cudnn.benchmark = not bool(deterministic)
    except Exception:
        pass


def make_generator(device: str, seed: Optional[int]) -> Optional[torch.Generator]:
    """Create a torch.Generator for a device if a seed is provided.

    Returns None if seed is None.
    """
    if seed is None:
        return None
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    return gen


