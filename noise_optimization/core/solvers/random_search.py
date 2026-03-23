from __future__ import annotations

import math
import os
from typing import Any, Dict, Optional

import torch

from .base import Solver, SolveResult
from ..loggers.base import ExperimentLogger


class RandomSearchSolver(Solver):
    def __init__(self, device: Optional[str] = None, logger: Optional[ExperimentLogger] = None):
        super().__init__(device, logger)
        self._latent_shape = None
        self._batch_size = 16
        self._best_protein = None  # For protein modality
        self._sampling_mode = "auto"
        self._seed: Optional[int] = None
        self._rng: Optional[torch.Generator] = None
        self._sobol_engine: Optional[torch.quasirandom.SobolEngine] = None
        self._use_combined_gaussian: bool = False  # For testing: use single scale vs separate distributions

    @staticmethod
    def _extract_seed(problem: Any, kwargs: Dict[str, Any]) -> Optional[int]:
        """Extract seed from multiple sources with fallback.
        
        Priority: seed > kwargs['random_seed'/'sampling_seed'] > problem.context > problem.cfg > default 42
        
        IMPORTANT: If task_index is available in problem.context, it will be added to the base seed
        to ensure different tasks get different starting points (critical for VF determinism).
        """
        base_seed = None
        
        seed = kwargs.get("seed", None)
        if seed is not None:
            try:
                base_seed = int(seed)
            except Exception:
                pass
        
        # Try seed-related kwargs
        if base_seed is None:
            for k in ("seed", "random_seed", "sampling_seed"):
                if k in kwargs and kwargs.get(k) is not None:
                    try:
                        base_seed = int(kwargs.get(k))
                        break
                    except Exception:
                        pass
        
        # Try problem.context (matches trs.py)
        if base_seed is None:
            if hasattr(problem, "context") and "seed" in problem.context:
                try:
                    base_seed = int(problem.context["seed"])
                except Exception:
                    pass
        
        # Try problem.cfg (matches trs.py)
        if base_seed is None:
            if hasattr(problem, "cfg") and hasattr(problem.cfg, "get"):
                try:
                    cfg_seed = problem.cfg.get("seed")
                    if cfg_seed is not None:
                        base_seed = int(cfg_seed)
                except Exception:
                    pass
        
        # Default fallback for reproducibility (matches trs.py)
        if base_seed is None:
            base_seed = 42
        
        # CRITICAL FIX: Incorporate task_index into seed to ensure different tasks get different starting points
        # This is especially important for VF (ODE) sampling which is deterministic
        task_index = None
        if hasattr(problem, "context") and problem.context:
            # Try task_index first (set by benchmark)
            task_index = problem.context.get("task_index")
            # Fallback to length_index or local_index if task_index not available
            if task_index is None:
                task_index = problem.context.get("length_index")
            if task_index is None:
                task_index = problem.context.get("local_index")
        
        if task_index is not None:
            # Combine base seed with task_index to create unique seed per task
            # Use large multiplier to ensure seeds don't overlap
            return base_seed + int(task_index) * 10000
        else:
            return base_seed

    @staticmethod
    def _normalize_sampling_mode(val: Any) -> str:
        s = str(val or "auto").strip().lower()
        aliases = {
            "auto": "auto",
            "default": "auto",
            "problem": "problem",
            "problem_sample": "problem",
            "sample": "problem",
            "gaussian": "gaussian",
            "normal": "gaussian",
            "randn": "gaussian",
            "sobol": "sobol_erfinv",
            "sobol_erfinv": "sobol_erfinv",
            "sobol+erfinv": "sobol_erfinv",
            "sobol_erf_inv": "sobol_erfinv",
            "sobol_erf-inv": "sobol_erfinv",
            "sobol_ervinv": "sobol_erfinv",  # common typo
        }
        return aliases.get(s, s)

    def _ensure_rng(self, seed: Optional[int]) -> None:
        self._seed = seed
        if seed is None:
            self._rng = None
            return
        g = torch.Generator(device=self.device)
        g.manual_seed(int(seed))
        self._rng = g

    def _ensure_sobol(self, dim: int, *, seed: Optional[int], scramble: Optional[bool]) -> torch.quasirandom.SobolEngine:
        # If the dimension changes, we must re-create the engine.
        if self._sobol_engine is not None:
            try:
                if getattr(self._sobol_engine, "dimension", None) == dim:
                    return self._sobol_engine
            except Exception:
                pass

        # Default: scrambled sequence (better coverage) but deterministic via seed.
        if scramble is None:
            scramble = True
        sobol_seed = int(seed) if seed is not None else 0
        self._sobol_engine = torch.quasirandom.SobolEngine(dimension=int(dim), scramble=bool(scramble), seed=sobol_seed)
        return self._sobol_engine

    @staticmethod
    def _sobol_to_standard_normal(u01: torch.Tensor, *, eps: float = 1e-7) -> torch.Tensor:
        # Map U(0,1) -> N(0,1) via inverse CDF using erf^{-1}.
        # z = sqrt(2) * erfinv(2u - 1)
        u = torch.clamp(u01, eps, 1.0 - eps)
        return math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)

    @staticmethod
    def _numel_from_shape(shape: tuple) -> int:
        d = 1
        for s in shape:
            d *= int(s)
        return int(d)

    def _sample_latents(
        self,
        problem: Any,
        *,
        batch_size: int,
        latent_shape: tuple,
        sampling_mode: str,
        sobol_scramble: Optional[bool],
        use_combined_gaussian: Optional[bool] = None,
    ) -> torch.Tensor:
        if batch_size <= 0:
            return torch.empty((0, *latent_shape), device=self.device)

        # Sampling in full latent space.
        # Check if we should use combined Gaussian (single scale) vs separate distributions
        # This is for testing: combined_gaussian=True uses single scale for all components
        # combined_gaussian=False (default) uses problem.sample() which respects SDE scales
        use_combined = use_combined_gaussian if use_combined_gaussian is not None else self._use_combined_gaussian
        
        if use_combined:
            # Combined Gaussian: use single scale (1.0) for all components
            # This tests the "combined Gaussian" approach vs "separate distributions"
            # which use coord_noise_std=0.1, cat_noise_level=1.0
            if sampling_mode == "sobol_erfinv":
                D = self._numel_from_shape(latent_shape)
                engine = self._ensure_sobol(D, seed=self._seed, scramble=sobol_scramble)
                u = engine.draw(int(batch_size)).to(self.device)
                x = self._sobol_to_standard_normal(u)
                return x.view(int(batch_size), *latent_shape) * 1.0
            else:
                # Single scale Gaussian for all components
                return torch.randn(int(batch_size), *latent_shape, device=self.device, generator=self._rng) * 1.0
        
        # Default: use problem.sample() which respects SDE scales (separate distributions)
        if sampling_mode == "problem" and hasattr(problem, "sample"):
            return problem.sample(batch_size=int(batch_size), latent_shape=latent_shape)

        if sampling_mode == "sobol_erfinv":
            D = self._numel_from_shape(latent_shape)
            engine = self._ensure_sobol(D, seed=self._seed, scramble=sobol_scramble)
            u = engine.draw(int(batch_size)).to(self.device)
            x = self._sobol_to_standard_normal(u)
            return x.view(int(batch_size), *latent_shape)

        # Default Gaussian (matches most Problem.sample() implementations).
        return torch.randn(int(batch_size), *latent_shape, device=self.device, generator=self._rng)

    def _solve_impl(self, problem: Any, **kwargs: Any) -> SolveResult:
        num_iterations = int(kwargs.get("num_iterations", 100))
        batch_size = int(kwargs.get("batch_size", 16))
        latent_shape = getattr(problem, "latent_shape", None)
        logger = kwargs.get("logger", self._default_logger)
        oracle_budget = int(kwargs.get("oracle_budget", 0))
        store_iteration_images = kwargs.get("store_iteration_images", False)

        if latent_shape is None:
            raise ValueError("RandomSearchSolver requires problem.latent_shape to be set (got None).")

        # Sampling configuration
        seed = self._extract_seed(problem, kwargs)
        sampling_mode = self._normalize_sampling_mode(
            kwargs.get(
                "sampling_mode",
                kwargs.get(
                    "latent_sampling",
                    "sobol_erfinv" if bool(kwargs.get("use_sobol_sampling", False)) else "auto",
                ),
            )
        )
        if sampling_mode == "auto":
            # Default behavior: use problem.sample() if available. But if a seed is provided,
            # prefer a fully deterministic internal sampler unless user explicitly requests "problem".
            sampling_mode = "gaussian" if seed is not None else ("problem" if hasattr(problem, "sample") else "gaussian")

        sobol_scramble = kwargs.get("sobol_scramble", None)
        if sobol_scramble is not None:
            sobol_scramble = bool(sobol_scramble)

        # Option for testing: use combined Gaussian (single scale) vs separate distributions
        self._use_combined_gaussian = bool(kwargs.get("use_combined_gaussian", False))

        self._ensure_rng(seed)

        best_reward = float("-inf")
        best_candidate = None
        history: list[dict[str, Any]] = []
        n_eval = 0
        nfe = 0

        # No warmup for RandomSearch; it is used as the warmup for TrustRegion

        iter_log = self.make_iteration_logger(problem, logger)

        # Determine iteration schedule based on oracle budget if provided
        total_iters = num_iterations
        if oracle_budget > 0:
            remaining = max(0, oracle_budget)
            total_iters = 0 if batch_size <= 0 else (remaining + batch_size - 1) // batch_size

        # Track which scaling checkpoints have been recorded (so checkpoint_{k} reflects
        # the best-so-far when k was first reached, not the final best after the run).
        logged_checkpoints: set[int] = set()

        for it in range(total_iters):
            if oracle_budget > 0 and n_eval >= oracle_budget:
                break
            
            # Update iteration for progressive steps tracking
            if hasattr(problem, "set_iteration"):
                problem.set_iteration(it)
            
            # Respect budget per-iteration
            if oracle_budget > 0:
                remaining = oracle_budget - n_eval
                batch_this_iter = max(0, min(batch_size, remaining))
            else:
                batch_this_iter = batch_size
            
            # Sample candidate latents
            latents = self._sample_latents(
                problem,
                batch_size=batch_this_iter,
                latent_shape=latent_shape,
                sampling_mode=sampling_mode,
                sobol_scramble=sobol_scramble,
                use_combined_gaussian=self._use_combined_gaussian,
            )

            rewards = problem.evaluate(latents)
            n_eval += latents.shape[0]
            nfe += latents.shape[0]

            max_reward, max_idx = torch.max(rewards, dim=0)
            max_reward_val = float(max_reward.item())
            improved = max_reward_val > best_reward
            if improved:
                best_reward = max_reward_val
                best_candidate = latents[max_idx].detach().clone()

            iter_metrics = {
                "iteration": it,
                "batch_mean": float(rewards.mean().item()),
                "batch_max": max_reward_val,
                "num_evaluations": n_eval,
                "nfe": nfe,
            }
            
            # For scaling logger: track cumulative best reward at each evaluation count
            if hasattr(problem, "context") and problem.context.get("scaling_mode", False):
                iter_metrics["best"] = best_reward
                
                # Track checkpoint-specific rewards if checkpoints are provided
                if hasattr(problem, "context") and "scaling_checkpoints" in problem.context:
                    checkpoints = problem.context["scaling_checkpoints"]
                    for checkpoint in checkpoints:
                        ckpt_i = int(checkpoint)
                        # Only record the first time the checkpoint is crossed
                        if (ckpt_i not in logged_checkpoints) and (n_eval >= ckpt_i):
                            iter_metrics[f"checkpoint_{checkpoint}"] = best_reward
                            logged_checkpoints.add(ckpt_i)
            
            # Store best image at this iteration if requested
            if store_iteration_images and improved:
                try:
                    # For protein modality, skip - proteins use decode_latents() which uses cache
                    if hasattr(problem, "context") and problem.context.get("modality") in ("protein", "proteina"):
                        # Skip - proteins are handled via decode_latents() which uses cache
                        best_img = None
                    else:
                        # Render the best image from this iteration (for image modality)
                        best_latent = latents[max_idx].detach().clone()
                        model_cfg = {}
                        try:
                            model_cfg = problem.context.get("model_config", {}) if hasattr(problem, "context") else {}
                        except Exception:
                            model_cfg = {}
                        model_cfg = dict(model_cfg)
                        model_cfg["initial_latent_noise"] = best_latent.unsqueeze(0) if best_latent.dim() == 3 else best_latent
                        
                        # Extract prompt from problem context
                        prompt = problem.context.get("prompt", "") if hasattr(problem, "context") else ""
                        
                        with torch.no_grad():
                            imgs = problem.generative_model.forward(prompt, **model_cfg)  # type: ignore[attr-defined]
                        
                        if isinstance(imgs, torch.Tensor):
                            best_img = imgs[0] if imgs.dim() >= 4 else imgs
                        elif isinstance(imgs, list):
                            best_img = imgs[0] if len(imgs) > 0 else None
                        else:
                            best_img = imgs
                    
                    if best_img is not None:
                        iter_metrics["best_image"] = best_img
                        print(f"[RandomSearch] Stored best image at iteration {it} with reward {max_reward_val:.4f}")
                except Exception as e:
                    print(f"[RandomSearch] Warning: Could not store iteration image: {e}")
            
            # Always track best image for scaling logger (even if not storing iteration images)
            if improved:
                try:
                    # For protein modality, skip - proteins use decode_latents() which uses cache
                    if hasattr(problem, "context") and problem.context.get("modality") in ("protein", "proteina"):
                        # Skip - proteins are handled via decode_latents() which uses cache
                        best_img = None
                    else:
                        # Render the best image from this iteration (for image modality)
                        best_latent = latents[max_idx].detach().clone()
                        model_cfg = {}
                        try:
                            model_cfg = problem.context.get("model_config", {}) if hasattr(problem, "context") else {}
                        except Exception:
                            model_cfg = {}
                        model_cfg = dict(model_cfg)
                        model_cfg["initial_latent_noise"] = best_latent.unsqueeze(0) if best_latent.dim() == 3 else best_latent
                        
                        # Extract prompt from problem context
                        prompt = problem.context.get("prompt", "") if hasattr(problem, "context") else ""
                        
                        with torch.no_grad():
                            imgs = problem.generative_model.forward(prompt, **model_cfg)  # type: ignore[attr-defined]
                    
                        if isinstance(imgs, torch.Tensor):
                            best_img = imgs[0] if imgs.dim() >= 4 else imgs
                        elif isinstance(imgs, list):
                            best_img = imgs[0] if len(imgs) > 0 else None
                        else:
                            best_img = imgs
                    
                    if best_img is not None:
                        iter_metrics["best_image"] = best_img
                except Exception as e:
                    # Silently fail for scaling logger - don't spam warnings
                    pass
            
            history.append(iter_metrics)
            iter_log(it, rewards, console_prefix="RandomSearch")

        # Decode best protein/molecule if needed
        best_decoded = None
        if hasattr(problem, "context"):
            modality = problem.context.get("modality", "")
            if modality in ("protein", "proteina"):
                try:
                    if hasattr(problem, "decode_latents"):
                        latent_to_decode = best_candidate.unsqueeze(0) if best_candidate.dim() == 2 else best_candidate
                        decoded = problem.decode_latents(latent_to_decode)
                        if decoded is not None:
                            if isinstance(decoded, torch.Tensor):
                                best_decoded = decoded[0].detach().clone() if decoded.dim() >= 3 else decoded.detach().clone()
                            elif isinstance(decoded, (list, tuple)) and len(decoded) > 0:
                                best_decoded = decoded[0]
                            else:
                                best_decoded = decoded
                except Exception:
                    pass
            elif modality in ("molecule", "qm9", "qm9_target"):
                try:
                    if hasattr(problem, "decode_latents"):
                        latent_to_decode = best_candidate.unsqueeze(0) if best_candidate.dim() == 2 else best_candidate
                        decoded = problem.decode_latents(latent_to_decode)
                        if isinstance(decoded, (list, tuple)):
                            best_decoded = decoded[0]
                        elif isinstance(decoded, torch.Tensor):
                            best_decoded = decoded[0]
                        else:
                            best_decoded = decoded
                except Exception:
                    pass

        metadata = {
            "solver": "random_search",
            "sampling_mode": sampling_mode,
            "seed": seed,
            "sobol_scramble": sobol_scramble,
            "best_protein": best_decoded if hasattr(problem, "context") and problem.context.get("modality") in ("protein", "proteina") else None,
            "best_molecule": best_decoded if hasattr(problem, "context") and problem.context.get("modality") in ("molecule", "qm9", "qm9_target") else None,
        }
        return SolveResult(
            best_reward=best_reward,
            best_candidate=best_candidate,
            history=history,
            num_evaluations=n_eval,
            num_function_evals=nfe,
            metadata=metadata,
        )

    def _initialize_impl(self, problem: Any, **kwargs: Any) -> None:
        """Initialize the random search solver."""
        self._latent_shape = getattr(problem, "latent_shape", None)
        self._batch_size = int(kwargs.get("batch_size", 16))
        oracle_budget = int(kwargs.get("oracle_budget", 0))

        if self._latent_shape is None:
            raise ValueError("RandomSearchSolver requires problem.latent_shape to be set (got None).")

        # Sampling configuration for step-based execution
        self._seed = self._extract_seed(problem, kwargs)
        self._sampling_mode = self._normalize_sampling_mode(
            kwargs.get(
                "sampling_mode",
                kwargs.get(
                    "latent_sampling",
                    "sobol_erfinv" if bool(kwargs.get("use_sobol_sampling", False)) else "auto",
                ),
            )
        )
        if self._sampling_mode == "auto":
            self._sampling_mode = "gaussian" if self._seed is not None else ("problem" if hasattr(problem, "sample") else "gaussian")
        self._ensure_rng(self._seed)
        self._sobol_engine = None
        
        # Option for testing: use combined Gaussian (single scale) vs separate distributions
        self._use_combined_gaussian = bool(kwargs.get("use_combined_gaussian", False))

    def _execute_step_impl(self) -> Dict[str, Any]:
        """Execute one step of random search."""
        if self._problem is None:
            raise RuntimeError("Solver not initialized")
        
        # Determine batch size for this iteration
        oracle_budget = self._kwargs.get("oracle_budget", 0)
        if oracle_budget > 0:
            remaining = oracle_budget - self._n_eval
            batch_this_iter = max(0, min(self._batch_size, remaining))
        else:
            batch_this_iter = self._batch_size
        
        if batch_this_iter <= 0:
            return {"best_reward": self._best_reward, "best_candidate": self._best_candidate, "n_eval": self._n_eval, "nfe": self._nfe}
        
        # Update iteration for progressive steps tracking
        if hasattr(self._problem, "set_iteration"):
            self._problem.set_iteration(self._current_iteration - 1)
        
        # Sample candidate latents
        sobol_scramble = self._kwargs.get("sobol_scramble", None)
        if sobol_scramble is not None:
            sobol_scramble = bool(sobol_scramble)
        latents = self._sample_latents(
            self._problem,
            batch_size=batch_this_iter,
            latent_shape=self._latent_shape,
            sampling_mode=self._sampling_mode,
            sobol_scramble=sobol_scramble,
            use_combined_gaussian=self._use_combined_gaussian,
        )
        if os.environ.get("NOISE_OPT_DEBUG_BATCH_LATENTS", "0") == "1" and latents.shape[0] > 1:
            diff = (latents - latents[0:1]).abs().sum(dim=tuple(range(1, latents.dim())))
            n_different = (diff > 1e-5).sum().item()
            print(
                f"[DEBUG RandomSearch._execute_step_impl] sampled latents shape={tuple(latents.shape)} "
                f"unique_rows={n_different >= latents.shape[0] - 1} n_different_from_first={n_different}"
            )

        rewards = self._problem.evaluate(latents)
        self._n_eval += latents.shape[0]
        self._nfe += latents.shape[0]

        max_reward, max_idx = torch.max(rewards, dim=0)
        max_reward_val = float(max_reward.item())
        
        # Best candidate of THIS iteration
        iter_best_candidate = latents[max_idx].detach().clone()
        
        best_candidate = None
        best_image = None
        best_protein = None
        
        if max_reward_val > self._best_reward:
            self._best_reward = max_reward_val
            best_candidate = iter_best_candidate.clone()
            
            # For protein modality, get decoded structure using problem.decode_latents() which uses cache
            if hasattr(self._problem, "context") and self._problem.context.get("modality") in ("protein", "proteina"):
                try:
                    # Use problem.decode_latents() which checks the cache first to avoid redundant regeneration
                    if hasattr(self._problem, "decode_latents"):
                        latent_to_decode = best_candidate.unsqueeze(0) if best_candidate.dim() == 2 else best_candidate
                        decoded = self._problem.decode_latents(latent_to_decode)
                        if decoded is not None:
                            # Extract single structure from batch
                            if isinstance(decoded, torch.Tensor):
                                if decoded.dim() >= 3:
                                    best_protein = decoded[0].detach().clone()
                                else:
                                    best_protein = decoded.detach().clone()
                            elif isinstance(decoded, (list, tuple)) and len(decoded) > 0:
                                best_protein = decoded[0]
                            else:
                                best_protein = decoded
                            self._best_protein = best_protein
                except Exception:
                    # Silently continue - structure decoding is optional
                    pass
            else:
                # Try to render best image (for image modality)
                try:
                    model_cfg = {}
                    try:
                        model_cfg = self._problem.context.get("model_config", {}) if hasattr(self._problem, "context") else {}
                    except Exception:
                        model_cfg = {}
                    model_cfg = dict(model_cfg)
                    model_cfg["initial_latent_noise"] = best_candidate.unsqueeze(0) if best_candidate.dim() == 3 else best_candidate
                    
                    # Extract prompt from problem context
                    prompt = self._problem.context.get("prompt", "") if hasattr(self._problem, "context") else ""
                    
                    with torch.no_grad():
                        imgs = self._problem.generative_model.forward(prompt, **model_cfg)  # type: ignore[attr-defined]
                    
                    if isinstance(imgs, torch.Tensor):
                        best_image = imgs[0] if imgs.dim() >= 4 else imgs
                    elif isinstance(imgs, list):
                        best_image = imgs[0] if len(imgs) > 0 else None
                    else:
                        best_image = imgs
                except Exception:
                    # Silently fail for scaling logger
                    pass

        return {
            "best_reward": self._best_reward,
            "best_candidate": best_candidate,
            "iter_best_candidate": iter_best_candidate,
            "best_image": best_image,
            "best_protein": best_protein if best_protein is not None else getattr(self, "_best_protein", None),
            "n_eval": self._n_eval,
            "nfe": self._nfe,
            "batch_mean": float(rewards.mean().item()),
            "batch_max": max_reward_val,
            "batch_rewards": rewards,  # Include rewards for iteration logging
            "batch_size": int(latents.shape[0]),
        }

    def _is_done_impl(self) -> bool:
        """Check if random search is done."""
        if self._problem is None:
            return True
        
        max_iterations = self._kwargs.get("num_iterations", 100)
        oracle_budget = self._kwargs.get("oracle_budget", 0)
        
        # Check iteration limit
        if self._current_iteration >= max_iterations:
            return True
        
        # Check oracle budget
        if oracle_budget > 0 and self._n_eval >= oracle_budget:
            return True
        
        return False

    def _get_metadata_impl(self) -> Dict[str, Any]:
        """Get random search metadata."""
        return {
            "sampling_mode": self._sampling_mode,
            "seed": self._seed,
            "sobol_scramble": self._kwargs.get("sobol_scramble", None),
        }


