"""Trust region archive management utilities.

This module provides pure functions for managing optimization archives
(history of evaluated points and their rewards) in trust region solvers.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch


def update_archive(
    Z_archive: torch.Tensor,
    R_archive: torch.Tensor,
    Z_new: torch.Tensor,
    R_new: torch.Tensor,
    max_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Append new points and rewards to existing archives.
    
    For center selection (global_topk) and restart we only need top-k by reward.
    When max_size is set, we keep only the best max_size points to bound memory.
    
    Args:
        Z_archive: Existing archive of z-space points, shape (N, d)
        R_archive: Existing archive of rewards, shape (N,)
        Z_new: New z-space points to append, shape (M, d)
        R_new: New rewards to append, shape (M,)
        max_size: If set, keep only the top max_size points by reward (None = unbounded)
        
    Returns:
        Tuple of (updated_Z_archive, updated_R_archive)
    """
    Z_updated = torch.cat([Z_archive, Z_new], dim=0)
    R_updated = torch.cat([R_archive, R_new], dim=0)
    
    if max_size is not None and max_size > 0 and R_updated.numel() > max_size:
        topk = torch.topk(R_updated.view(-1), k=max_size, largest=True)
        Z_updated = Z_updated[topk.indices].detach().clone()
        R_updated = R_updated[topk.indices].detach().clone()
    
    return Z_updated, R_updated

