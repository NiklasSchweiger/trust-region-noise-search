from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from .tr_utils import select_centers_diverse, select_centers_clustering


@dataclass
class CenterSelectionStrategy:
    """Encapsulate trust-region center selection logic for `TrustRegionSolver`.

    This class centralizes different center-selection modes so we can:
      - keep `trs.py` clean (no duplicated if/else center logic),
      - plug in new strategies (e.g. annealed exploration→exploitation),
      - preserve the original global top-k behaviour as a simple default.

    The strategy itself is stateless w.r.t. archives; any annealing is driven by
    the `(iteration, total_iterations)` arguments.
    """

    mode: str = "global_topk"
    min_center_dist: float = 0.1
    clustering_percentile: float = 0.2
    explore_frac: Optional[float] = None
    device: str = "cuda"

    @staticmethod
    def _normalize_mode(raw_mode: Optional[str]) -> str:
        """Map historical names to a small canonical set of modes."""
        if not raw_mode:
            return "global_topk"
        mode = raw_mode.lower()
        if mode in {"topk", "global", "global_best", "batch"}:
            return "global_topk"
        if mode in {"per_region", "local"}:
            return "per_region"
        if mode in {"diverse", "diverse_global"}:
            return "diverse_global"
        if mode in {"clustering", "cluster_global"}:
            return "cluster_global"
        if mode in {"annealed_diverse", "annealed_diverse_topk"}:
            return "annealed_diverse_topk"
        # Fallback: treat unknown values as the original behaviour
        return "global_topk"

    @classmethod
    def from_config(cls, tr_config: Dict[str, Any], device: str = "cuda") -> "CenterSelectionStrategy":
        """Build strategy from a `tr` config block (Hydra dict-style).

        Expected keys (all optional, with safe defaults):
          - center_selection: str   (mode, e.g. 'global_best', 'per_region', 'diverse')
          - min_center_dist: float  (for diverse selection)
          - clustering_percentile: float (for clustering-based selection)
          - center_anneal_frac: float (0–1, for annealed modes)

        The default corresponds to the existing behaviour: global top-k over the archive.
        """
        center_mode_raw = tr_config.get("center_selection", "global_topk")
        mode = cls._normalize_mode(center_mode_raw)
        min_center_dist = float(tr_config.get("min_center_dist", 0.1))
        clustering_percentile = float(tr_config.get("clustering_percentile", 0.2))
        explore_frac = tr_config.get("center_anneal_frac", None)
        explore_frac_f = float(explore_frac) if explore_frac is not None else None
        return cls(
            mode=mode,
            min_center_dist=min_center_dist,
            clustering_percentile=clustering_percentile,
            explore_frac=explore_frac_f,
            device=device,
        )

    def _resolve_effective_mode(self, iteration: Optional[int], total_iterations: Optional[int]) -> str:
        """Resolve the *current* effective mode, handling any annealing."""
        if self.mode != "annealed_diverse_topk":
            return self.mode

        if (
            self.explore_frac is None
            or iteration is None
            or total_iterations is None
            or total_iterations <= 0
        ):
            # Fall back to global_topk if annealing parameters are missing
            return "global_topk"

        frac = max(0.0, min(1.0, float(iteration) / float(total_iterations)))
        # Early phase: emphasize diverse centers; later: converge to global top-k
        return "diverse_global" if frac < self.explore_frac else "global_topk"

    def select(
        self,
        Z_archive: torch.Tensor,
        R_archive: torch.Tensor,
        centers_z: torch.Tensor,
        centers_val: torch.Tensor,
        tr_states: List[Any],
        per_region_best: Optional[List[Tuple[int, torch.Tensor, float]]] = None,
        iteration: Optional[int] = None,
        total_iterations: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select updated centers and their values.

        Args:
            Z_archive: Full archive of latent points (N, d)
            R_archive: Corresponding rewards (N,)
            centers_z: Current region centers (R, d)
            centers_val: Current center values (R,)
            tr_states: List of per-region trust region states (for future use)
            per_region_best: Optional list of (region_idx, best_z, best_val) from
                             the *current* iteration; required for 'per_region' mode.
            iteration: Current iteration index (0-based), for annealed modes.
            total_iterations: Total planned iterations, for annealed modes.
        """
        if Z_archive.numel() == 0 or R_archive.numel() == 0:
            # Nothing to update from; return current centers unchanged
            return centers_z, centers_val

        mode = self._resolve_effective_mode(iteration, total_iterations)
        kcent = min(centers_z.shape[0], int(R_archive.shape[0]))

        if mode == "per_region":
            # TuRBO-style per-region update: each region tracks its own local best.
            if per_region_best is None:
                # Fallback to global behaviour if we don't have per-region info
                mode = "global_topk"
            else:
                new_centers = centers_z.clone()
                new_vals = centers_val.clone()
                for ridx, best_z, best_val in per_region_best:
                    if 0 <= ridx < new_centers.shape[0]:
                        # Only move center if the new local best actually improves it
                        if best_val > float(new_vals[ridx].item()) + 1e-12:
                            new_centers[ridx] = best_z.to(new_centers.device)
                            new_vals[ridx] = torch.as_tensor(best_val, device=new_vals.device)
                return new_centers, new_vals

        if mode == "diverse_global":
            new_centers, new_vals = select_centers_diverse(
                Z_archive, R_archive, kcent, min_dist=self.min_center_dist, device=self.device
            )
            return new_centers.detach().clone(), new_vals.detach().clone()

        if mode == "cluster_global":
            new_centers, new_vals = select_centers_clustering(
                Z_archive,
                R_archive,
                kcent,
                top_percentile=self.clustering_percentile,
                device=self.device,
            )
            return new_centers.detach().clone(), new_vals.detach().clone()

        # Default: global top-k over the archive (original behaviour)
        vals = R_archive.view(-1)
        g_top = torch.topk(vals, k=kcent)
        new_centers = Z_archive[g_top.indices].detach().clone()
        new_vals = g_top.values.detach().clone()
        return new_centers, new_vals

