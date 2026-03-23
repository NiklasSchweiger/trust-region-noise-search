from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from .base import Problem
from ..rewards.base import RewardFunction


class ProteinOptimizationProblem(Problem):
    """Problem definition for protein generation/optimization.

    This class is specifically designed for protein design tasks, handling
    protein-specific data formats (atom37, coordinates) and parameters (n_residues).

    Expected generative model interface:
      - forward(batch_size: int, initial_noise: torch.Tensor, **kwargs) -> Any
      - Optional helpers:
          get_noise_latent_shape() -> Tuple[int, ...]
          sample_latents(batch_size: int, n_residues: int) -> torch.Tensor
    The reward can be any core RewardFunction (e.g., ProteinRewardFunction)
    or a legacy adapter exposing `predict_reward(candidates, ...)`.
    """

    def __init__(
        self,
        generative_model: Any,
        reward_model: RewardFunction | Any,
        *,
        device: Optional[str] = None,
        noise_dims: Optional[Tuple[int, ...]] = None,
        model_config: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.generative_model = generative_model
        self.reward_model = reward_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Discover latent shape if not provided
        if noise_dims is None:
            if hasattr(self.generative_model, "get_noise_latent_shape"):
                try:
                    noise_dims = tuple[int, ...](self.generative_model.get_noise_latent_shape())  # type: ignore[attr-defined]
                except Exception:
                    noise_dims = None
        self.noise_dims = noise_dims
        self.model_config = model_config or {}
        self._context = {
            "domain": "protein",
            "modality": "protein",
            "noise_dims": self.noise_dims,
            "model_config": dict[str, Any](self.model_config),
            **(context or {}),
        }
        self._current_iteration = 0
        
        # Track diversity and novelty metrics (optional, controlled by config)
        # Check config: proteina.track_diversity or model_config.track_diversity
        proteina_cfg = context.get("proteina_config") if context else {}
        if isinstance(proteina_cfg, dict):
            track_diversity = proteina_cfg.get("track_diversity", False)
        else:
            track_diversity = False
        
        # Also check model_config as fallback
        if not track_diversity and isinstance(self.model_config, dict):
            track_diversity = self.model_config.get("track_diversity", False)
        
        self._track_diversity = bool(track_diversity)
        
        self.last_diversity: Optional[float] = None
        self.last_novelty: Optional[float] = None
        self._seen_structures: List[torch.Tensor] = []  # Store recent structures for novelty
        self._max_seen_history: int = 20  # Limit history size (reduced for speed)
        self._max_novelty_checks: int = 10  # Only check against last N structures for novelty
        self._diversity_sample_size: int = 10  # Max pairwise comparisons for diversity
        
        # Cache for decoded structures to avoid re-decoding
        # Maps (latent_shape, latent_hash) to decoded structure for quick lookup
        self._decoded_cache: Dict[Tuple[Tuple[int, ...], int], Any] = {}
        self._last_decoded_batch: Optional[Tuple[Any, Any]] = None  # (latents, decoded) from last evaluate() call

    def evaluate(self, candidates: Any, noise_list: Optional[List[torch.Tensor]] = None, grad_enabled: bool = False, **kwargs: Any) -> torch.Tensor:
        """Decode candidates with model and compute rewards.
        
        Accepts candidate noise tensors (preferred) or other model-specific
        representations that the provided `generative_model` can handle.
        """
        import time
        debug = kwargs.get("debug", False)
        
        if isinstance(candidates, torch.Tensor):
            # Detach and clone to break computation graph if not grad enabled
            if not grad_enabled:
                candidates = candidates.detach().clone().to(self.device)
            else:
                candidates = candidates.to(self.device)
            batch_size = int(candidates.shape[0])
            self._total_samples_evaluated += batch_size
        else:
            batch_size = 1
            self._total_samples_evaluated += 1

        decode_start = time.perf_counter()
        with torch.set_grad_enabled(grad_enabled):
            decoded = self.decode_latents(candidates, noise_list=noise_list, grad_enabled=grad_enabled, **kwargs)
            decode_time = time.perf_counter() - decode_start
            
            # Cache decoded structures for this batch to avoid re-decoding
            if isinstance(candidates, torch.Tensor) and isinstance(decoded, torch.Tensor):
                self._last_decoded_batch = (candidates.detach().clone(), decoded.detach().clone())

            eval_start = time.perf_counter()
            rewards = self.evaluate_decoded(decoded)
            eval_time = time.perf_counter() - eval_start

        return rewards

    def decode_latents(self, latents: Any, noise_list: Optional[List[torch.Tensor]] = None, grad_enabled: bool = False, **kwargs: Any) -> Any:
        """Decode protein latents (coordinates) to atom37 format.
        
        For Proteina models, if latents are already coordinates [B, N, 3],
        they can be converted directly to atom37 without running full generation.
        
        If the latents were recently decoded in a batch evaluation, returns the cached decoded structure
        to avoid redundant forward passes.
        """
        # Check if we have a cached decoded structure from the last batch evaluation
        # This avoids redundant forward passes when the benchmark re-decodes the best candidate
        if isinstance(latents, torch.Tensor) and self._last_decoded_batch is not None:
            cached_latents, cached_decoded = self._last_decoded_batch
            # Check if this is a single latent that might be in the cached batch
            if latents.dim() == 2 or (latents.dim() == 3 and latents.shape[0] == 1):
                # Extract the actual latent (remove batch dim if present)
                single_latent = latents[0] if latents.dim() == 3 else latents
                
                # Try to find matching latent in cached batch using approximate matching
                # (tolerate small numerical differences from detach/clone operations)
                for i in range(cached_latents.shape[0]):
                    cached_single = cached_latents[i]
                    # Check shape and approximate equality
                    if single_latent.shape == cached_single.shape:
                        # Use a more lenient tolerance for matching (1e-5 for float32)
                        if torch.allclose(single_latent.cpu(), cached_single.cpu(), atol=1e-5, rtol=1e-5):
                            # Return the corresponding decoded structure
                            if isinstance(cached_decoded, torch.Tensor):
                                if cached_decoded.dim() >= 3:
                                    return cached_decoded[i:i+1]
                                else:
                                    return cached_decoded[i]
                            elif isinstance(cached_decoded, (list, tuple)):
                                return cached_decoded[i] if i < len(cached_decoded) else None
                            else:
                                return cached_decoded
        
        params = dict(self.model_config)
        params.update(kwargs)
        params['noise_list'] = noise_list
        params['differentiable'] = grad_enabled
        
        if isinstance(latents, torch.Tensor):
            batch_size = int(latents.shape[0])
            # Ensure latents are detached (no computation graph) to prevent memory leaks if not grad enabled
            if not grad_enabled and latents.requires_grad:
                latents = latents.detach()
        else:
            batch_size = int(params.pop("batch_size", 1))

        # Special handling for Proteina: if latents are PRE-COMPUTED coordinates [B, N, 3], convert directly to atom37
        # NOTE: This should ONLY be used when we have pre-computed coordinates (e.g., from look-ahead predictions),
        # NOT during normal generation where noise is also [B, N, 3] shape. We check for a flag to distinguish.
        # For normal generation, we MUST call forward() to apply sampling parameters (sampling_mode, sc_scale_noise, etc.)
        if isinstance(latents, torch.Tensor) and latents.dim() == 3 and latents.shape[-1] == 3:
            # Only use early return if explicitly flagged as pre-computed coordinates
            # This prevents skipping forward() during normal generation where noise is also [B, N, 3]
            is_precomputed_coords = kwargs.get("is_precomputed_coordinates", False) or params.get("is_precomputed_coordinates", False)
            if is_precomputed_coords:
                # Check if this is Proteina (coordinates, not noise)
                model_type = type(self.generative_model).__name__
                # import sys
                # sys.stdout.write(f"[PROTEIN PROBLEM] Early return: Pre-computed coordinates detected, converting directly to atom37\n")
                # sys.stdout.flush()
                if "Proteina" in model_type or hasattr(self.generative_model, "underlying_model"):
                    # Check if underlying model has samples_to_atom37 method (Proteina signature)
                    underlying = getattr(self.generative_model, "underlying_model", None)
                    if underlying is not None and hasattr(underlying, "samples_to_atom37"):
                        # These are coordinates, convert directly to atom37
                        # Ensure coordinates are detached to prevent computation graph accumulation
                        return underlying.samples_to_atom37(latents.detach() if latents.requires_grad else latents)
            else:
                # This is noise [B, N, 3] for normal generation - MUST call forward() to apply sampling parameters
                # import sys
                # sys.stdout.write(f"[PROTEIN PROBLEM] Latents are [B, N, 3] but NOT flagged as pre-computed - treating as noise, will call forward()\n")
                # sys.stdout.flush()
                pass

        # DEBUG: Print before try block
        # import sys
        # sys.stdout.write(f"[PROTEIN PROBLEM] DEBUG: Before try block, latents type={type(latents)}, shape={latents.shape if isinstance(latents, torch.Tensor) else 'N/A'}\n")
        # sys.stdout.flush()
        
        try:
            # sys.stdout.write(f"[PROTEIN PROBLEM] >>>>> About to call forward() with batch_size={batch_size}, latents shape={latents.shape if isinstance(latents, torch.Tensor) else 'not tensor'}\n")
            # sys.stdout.flush()
            result = self.generative_model.forward(
                batch_size=batch_size,
                initial_noise=latents,
                **params,
            )
            # sys.stdout.write(f"[PROTEIN PROBLEM] <<<<< forward() returned successfully\n")
            # sys.stdout.flush()
            return result
        except TypeError as e:
            # import sys
            # sys.stdout.write(f"[PROTEIN PROBLEM] TypeError caught: {e}, trying positional args\n")
            # sys.stdout.flush()
            result = self.generative_model.forward(batch_size, latents, **params)
            # sys.stdout.write(f"[PROTEIN PROBLEM] <<<<< forward() (positional) returned successfully\n")
            # sys.stdout.flush()
            return result
        except Exception as e:
            # import sys
            # sys.stdout.write(f"[PROTEIN PROBLEM] Exception in forward(): {type(e).__name__}: {e}\n")
            # sys.stdout.flush()
            raise

    def evaluate_decoded(self, decoded: Any, extra_context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """Evaluate decoded protein structures (atom37 format) with reward model."""
        context = dict(self._context)
        if extra_context:
            context.update(extra_context)

        # Compute diversity and novelty metrics from decoded structures (only if enabled)
        if self._track_diversity:
            self._compute_diversity_metrics(decoded)
            
            # Store in context for logging
            if self.last_diversity is not None:
                self._context["diversity"] = self.last_diversity
            if self.last_novelty is not None:
                self._context["novelty"] = self.last_novelty
        else:
            # Clear metrics if tracking is disabled
            self.last_diversity = None
            self.last_novelty = None

        if isinstance(self.reward_model, RewardFunction):
            rewards = self.reward_model.evaluate(decoded, context=context)
        else:
            if hasattr(self.reward_model, "evaluate") and callable(getattr(self.reward_model, "evaluate")):
                rewards = self.reward_model.evaluate(decoded, context=context)
            elif hasattr(self.reward_model, "predict_reward") and callable(getattr(self.reward_model, "predict_reward")):
                rewards = self.reward_model.predict_reward(decoded)
            else:
                raise TypeError("Reward model must implement evaluate(...) or predict_reward(...)")

        if isinstance(rewards, tuple) and len(rewards) > 0:
            rewards = rewards[0]

        if not isinstance(rewards, torch.Tensor):
            rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        if rewards.dim() > 1:
            rewards = rewards.view(-1)
        
        return rewards
    
    def _compute_diversity_metrics(self, decoded: Any) -> None:
        """Compute diversity and novelty metrics from protein structures.
        
        Uses Proteina's rmsd_metric function for consistency with their codebase.
        
        Diversity: Mean pairwise RMSD between structures in the current batch
        Novelty: Average minimum RMSD to previously seen structures
        
        These metrics are computed cheaply from atom37 coordinates.
        """
        try:
            from ..utils.path_utils import setup_proteina_path
            setup_proteina_path()
            from proteinfoundation.metrics.designability import rmsd_metric
            
            # Extract structures in atom37 format
            if isinstance(decoded, torch.Tensor):
                structures = decoded.detach().cpu()
            else:
                structures = torch.tensor(decoded) if not isinstance(decoded, list) else torch.stack([torch.tensor(s) for s in decoded])
            
            # Ensure atom37 format [B, N, 37, 3]
            if structures.dim() == 3:
                # Assume [B, N, 3] - need to convert to atom37
                # Create minimal atom37 with CA at index 1
                B, N, _ = structures.shape
                atom37 = torch.zeros(B, N, 37, 3)
                atom37[:, :, 1, :] = structures  # CA at index 1
                structures = atom37
            elif structures.dim() == 4:
                # Already atom37 format [B, N, 37, 3]
                pass
            else:
                # Can't compute diversity
                self.last_diversity = None
                self.last_novelty = None
                return
            
            batch_size = structures.shape[0]
            
            # Diversity: fast unaligned CA distance (much cheaper than RMSD with alignment)
            # Sample pairwise comparisons to limit cost
            if batch_size > 1:
                import random
                # Extract CA coordinates
                ca_coords = structures[:, :, 1, :]  # [B, N, 3]
                
                # Sample pairwise comparisons (limit to max_diversity_sample_size)
                total_pairs = batch_size * (batch_size - 1) // 2
                num_samples = min(self._diversity_sample_size, total_pairs)
                
                if num_samples > 0:
                    if num_samples >= total_pairs:
                        # Use all pairs
                        pairs = [(i, j) for i in range(batch_size) for j in range(i + 1, batch_size)]
                    else:
                        # Sample random pairs
                        pairs = random.sample(
                            [(i, j) for i in range(batch_size) for j in range(i + 1, batch_size)],
                            num_samples
                        )
                    
                    distances = []
                    for i, j in pairs:
                        try:
                            # Fast unaligned distance (no Kabsch alignment)
                            ca_i = ca_coords[i]  # [N, 3]
                            ca_j = ca_coords[j]  # [N, 3]
                            
                            # Handle length mismatch
                            min_len = min(ca_i.shape[0], ca_j.shape[0])
                            ca_i_trunc = ca_i[:min_len]
                            ca_j_trunc = ca_j[:min_len]
                            
                            # Center both structures
                            center_i = ca_i_trunc.mean(dim=0)
                            center_j = ca_j_trunc.mean(dim=0)
                            ca_i_centered = ca_i_trunc - center_i
                            ca_j_centered = ca_j_trunc - center_j
                            
                            # Mean distance (much faster than RMSD with alignment)
                            dist = torch.norm(ca_i_centered - ca_j_centered, dim=-1).mean().item()
                            distances.append(dist)
                        except Exception:
                            continue
                    
                    self.last_diversity = float(sum(distances) / len(distances)) if distances else 0.0
                else:
                    self.last_diversity = 0.0
            else:
                self.last_diversity = 0.0
            
            # Novelty: fast distance to a small subset of recently seen structures
            if len(self._seen_structures) > 0:
                # Only check against last N structures (much cheaper)
                recent_structures = self._seen_structures[-self._max_novelty_checks:]
                
                ca_coords = structures[:, :, 1, :]  # [B, N, 3]
                novelties = []
                
                for i in range(batch_size):
                    min_dist = float('inf')
                    ca_i = ca_coords[i]  # [N, 3]
                    
                    for seen_struct in recent_structures:
                        try:
                            if isinstance(seen_struct, torch.Tensor):
                                seen_ca = seen_struct[:, 1, :]  # Extract CA [N, 3]
                            else:
                                continue
                            
                            # Handle length mismatch
                            min_len = min(ca_i.shape[0], seen_ca.shape[0])
                            ca_i_trunc = ca_i[:min_len]
                            seen_ca_trunc = seen_ca[:min_len]
                            
                            # Fast unaligned distance
                            center_i = ca_i_trunc.mean(dim=0)
                            center_seen = seen_ca_trunc.mean(dim=0)
                            ca_i_centered = ca_i_trunc - center_i
                            seen_centered = seen_ca_trunc - center_seen
                            
                            dist = torch.norm(ca_i_centered - seen_centered, dim=-1).mean().item()
                            min_dist = min(min_dist, dist)
                        except Exception:
                            continue
                    
                    if min_dist < float('inf'):
                        novelties.append(min_dist)
                
                self.last_novelty = float(sum(novelties) / len(novelties)) if novelties else 0.0
            else:
                # No history yet - set novelty to a high value (very novel)
                self.last_novelty = 100.0 if batch_size > 0 else 0.0
            
            # Store current structures in history (keep only recent ones)
            for i in range(batch_size):
                self._seen_structures.append(structures[i].clone())
            # Limit history size
            if len(self._seen_structures) > self._max_seen_history:
                self._seen_structures = self._seen_structures[-self._max_seen_history:]
                
        except ImportError:
            # Proteina not available - fall back to simple computation
            self._compute_diversity_metrics_simple(decoded)
        except Exception as e:
            # Silently fail - don't break optimization if diversity computation fails
            self.last_diversity = None
            self.last_novelty = None
    
    def _compute_diversity_metrics_simple(self, decoded: Any) -> None:
        """Fallback simple diversity computation when Proteina is not available."""
        try:
            import numpy as np
            
            # Extract CA coordinates from decoded structures
            if isinstance(decoded, torch.Tensor):
                structures = decoded.detach().cpu()
            else:
                structures = torch.tensor(decoded) if not isinstance(decoded, list) else torch.stack([torch.tensor(s) for s in decoded])
            
            # Handle atom37 format [B, N, 37, 3] or [B, N, 3]
            if structures.dim() == 4 and structures.shape[2] >= 2:
                # atom37 format - extract CA (index 1)
                ca_coords = structures[:, :, 1, :].numpy()  # [B, N, 3]
            elif structures.dim() == 3:
                # Already CA coordinates [B, N, 3]
                ca_coords = structures.numpy()
            else:
                self.last_diversity = None
                self.last_novelty = None
                return
            
            batch_size = ca_coords.shape[0]
            
            # Simple diversity: mean pairwise distance (no alignment)
            if batch_size > 1:
                distances = []
                for i in range(batch_size):
                    for j in range(i + 1, batch_size):
                        # Simple mean distance without alignment
                        min_len = min(ca_coords[i].shape[0], ca_coords[j].shape[0])
                        diff = ca_coords[i][:min_len] - ca_coords[j][:min_len]
                        dist = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))
                        distances.append(dist)
                self.last_diversity = float(np.mean(distances)) if distances else 0.0
            else:
                self.last_diversity = 0.0
            
            # Simple novelty
            if len(self._seen_structures) > 0:
                novelties = []
                for i in range(batch_size):
                    min_dist = float('inf')
                    for seen_ca in self._seen_structures:
                        if isinstance(seen_ca, torch.Tensor):
                            seen = seen_ca.numpy()
                        else:
                            seen = seen_ca
                        min_len = min(ca_coords[i].shape[0], seen.shape[0])
                        diff = ca_coords[i][:min_len] - seen[:min_len]
                        dist = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))
                        min_dist = min(min_dist, dist)
                    if min_dist < float('inf'):
                        novelties.append(min_dist)
                self.last_novelty = float(np.mean(novelties)) if novelties else 0.0
            else:
                self.last_novelty = 100.0 if batch_size > 0 else 0.0
            
            # Store in history
            for i in range(batch_size):
                self._seen_structures.append(torch.tensor(ca_coords[i]))
            if len(self._seen_structures) > self._max_seen_history:
                self._seen_structures = self._seen_structures[-self._max_seen_history:]
        except Exception:
            self.last_diversity = None
            self.last_novelty = None

    def sample(self, batch_size: int, latent_shape: Optional[Tuple[int, ...]] = None) -> torch.Tensor:
        """Sample initial noise vectors for the protein model.
        
        For proteins, this should use n_residues from model_config to generate
        the correct latent shape.
        """
        # If wrapper offers sampling, use it
        if hasattr(self.generative_model, "sample_latents") and callable(getattr(self.generative_model, "sample_latents")):
            try:
                # Extract n_residues from model_config or latent_shape
                n_residues = None
                
                if isinstance(self.model_config, dict) and "n_residues" in self.model_config:
                    n_residues = int(self.model_config["n_residues"])
                elif latent_shape is not None and len(latent_shape) == 2 and latent_shape[1] == 3:
                    # Extract n_residues from latent_shape if it's (n_residues, 3)
                    n_residues = int(latent_shape[0])
                
                if n_residues is not None:
                    # Pass n_residues to sample_latents for proteins
                    lat = self.generative_model.sample_latents(batch_size=batch_size, n_residues=n_residues)  # type: ignore[attr-defined]
                else:
                    # No n_residues available, use default
                    lat = self.generative_model.sample_latents(batch_size=batch_size)  # type: ignore[attr-defined]
                if isinstance(lat, torch.Tensor):
                    return lat.to(self.device)
            except Exception:
                pass

        shape = latent_shape or self.noise_dims
        if shape is None:
            raise ValueError("noise_dims must be provided or discoverable for ProteinOptimizationProblem")
        return torch.randn(batch_size, *shape, device=self.device)

    @property
    def context(self) -> Dict[str, Any]:
        return dict(self._context, device=self.device)

    def supports_denoising_callbacks(self) -> bool:
        return hasattr(self.generative_model, "run_with_callback")

    @property
    def latent_shape(self) -> Optional[Tuple[int, ...]]:
        """Alias for noise_dims to maintain compatibility with image problems."""
        return self.noise_dims

    def set_iteration(self, iteration: int) -> None:
        """Set the current iteration number (for compatibility with solvers that track iteration).
        
        This method exists for API compatibility with ImageGenerationProblem.
        For protein problems, iteration tracking is optional and doesn't
        affect generation behavior unless the generative model uses it.
        """
        self._current_iteration = iteration

