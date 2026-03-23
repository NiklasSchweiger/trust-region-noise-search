from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from .base import Solver, SolveResult
from ..loggers.base import ExperimentLogger


class ZeroOrderSolver(Solver):
    """Simple zero-order hill-climbing.

    Notes:
        - This solver first samples an initial pool of latent noises, evaluates them,
          and uses the best one as the starting center.
        - Then it repeatedly samples a batch of perturbations around the current
          best (the "center") and moves to the best candidate if it improves.
        - Historically this implementation used *unit-sphere* perturbations, but
          the docstring/config commonly assume *Gaussian* perturbations. This is
          now configurable via `perturbation`.
    """

    def __init__(self, device: Optional[str] = None, logger: Optional[ExperimentLogger] = None):
        super().__init__(device, logger)
        self._latent_shape = None
        self._batch_size = 16
        self._step_size = 0.1
        self._perturbation = "gaussian"
        self._center = None
        self._oracle_budget = 0
        self._seed: Optional[int] = None
        self._generator: Optional[torch.Generator] = None

    @staticmethod
    def _compute_warmup_size(batch_size: int, warmup_batches: int, warmup_samples: int) -> int:
        """Compute the number of initial samples used to pick the starting center.

        Precedence:
          - If `warmup_samples > 0`, use that exactly.
          - Else if `warmup_batches > 0`, use `warmup_batches * batch_size`.
          - Else fall back to `batch_size` (one batch).
        Always returns at least 1.
        """
        if warmup_samples > 0:
            return max(1, int(warmup_samples))
        if warmup_batches > 0:
            return max(1, int(warmup_batches) * int(batch_size))
        return max(1, int(batch_size))

    @staticmethod
    def _normalize_perturbation_mode(val: Any) -> str:
        s = str(val or "gaussian").strip().lower()
        aliases = {
            "gaussian": "gaussian",
            "normal": "gaussian",
            "sphere": "sphere",
            "unit_sphere": "sphere",
            "unit-sphere": "sphere",
        }
        return aliases.get(s, s)

    @staticmethod
    def _extract_seed(problem: Any, kwargs: Dict[str, Any]) -> int:
        """Extract seed from multiple sources with fallback.
        
        Priority: seed > kwargs['random_seed'/'sampling_seed'] > problem.context > problem.cfg > default 42.
        If task_index is in problem.context, it is combined with the base seed so each task gets a different
        starting point (important for diversity: each task then converges to a different local optimum).
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
        if base_seed is None and hasattr(problem, "context") and "seed" in problem.context:
            try:
                base_seed = int(problem.context["seed"])
            except Exception:
                pass
        
        # Try problem.cfg (matches trs.py)
        if base_seed is None and hasattr(problem, "cfg") and hasattr(problem.cfg, "get"):
            try:
                cfg_seed = problem.cfg.get("seed")
                if cfg_seed is not None:
                    base_seed = int(cfg_seed)
            except Exception:
                pass
        
        if base_seed is None:
            base_seed = 42
        
        # Incorporate task_index so each task gets a different starting point (matches random_search).
        # This is critical for diversity: different starts → different local optima per task.
        task_index = None
        if hasattr(problem, "context") and problem.context:
            task_index = problem.context.get("task_index")
            if task_index is None:
                task_index = problem.context.get("length_index")
            if task_index is None:
                task_index = problem.context.get("local_index")
        
        if task_index is not None:
            return base_seed + int(task_index) * 10000
        return base_seed

    def _ensure_generator(self, seed: int) -> None:
        """Ensure generator is created and seeded."""
        self._seed = seed
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(int(seed))

    def _solve_impl(self, problem: Any, **kwargs: Any) -> SolveResult:
        num_iterations = int(kwargs.get("num_iterations", 100))
        step_size = float(kwargs.get("step_size", 0.1))
        batch_size = int(kwargs.get("batch_size", 16))
        oracle_budget = int(kwargs.get("oracle_budget", 0))
        latent_shape = getattr(problem, "latent_shape", None)
        logger = kwargs.get("logger", self._default_logger)
        warmup_batches = int(kwargs.get("warmup_batches", 0))
        warmup_samples = int(kwargs.get("warmup_samples", 0))
        perturbation = self._normalize_perturbation_mode(kwargs.get("perturbation", kwargs.get("perturbation_mode", "gaussian")))

        if latent_shape is None:
            raise ValueError("ZeroOrderSolver requires problem.latent_shape to be set (got None).")

        seed = self._extract_seed(problem, kwargs)
        self._ensure_generator(seed)

        # Initialize center: warmup sample and evaluate in batches, pick best as starting center
        warmup_size = self._compute_warmup_size(batch_size, warmup_batches, warmup_samples)
        if oracle_budget > 0:
            warmup_size = min(warmup_size, oracle_budget)

        # Generate warmup samples in full latent space
        if hasattr(problem, "sample") and callable(problem.sample):
            try:
                center_batch_all = problem.sample(batch_size=warmup_size, latent_shape=latent_shape, seed=seed)
                z_batch_all = None
            except TypeError:
                center_batch_all = problem.sample(batch_size=warmup_size, latent_shape=latent_shape)
                z_batch_all = None
        else:
            D = int(torch.tensor(latent_shape).prod().item())
            z_batch_all = torch.randn(warmup_size, D, device=self.device, generator=self._generator)
            center_batch_all = None

        rewards_list = []
        best_reward = float("-inf")
        best_idx_global = None
        center = None

        for i in range(0, warmup_size, batch_size):
            end_idx = min(i + batch_size, warmup_size)
            this_batch_size = end_idx - i

            if z_batch_all is not None:
                z_batch = z_batch_all[i:end_idx]
                center_batch = z_batch.view(this_batch_size, *latent_shape)
            else:
                center_batch = center_batch_all[i:end_idx]

            rewards_batch = problem.evaluate(center_batch)
            rewards_list.append(rewards_batch)

            batch_best_val, batch_best_idx = torch.max(rewards_batch, dim=0)
            batch_best_val_float = float(batch_best_val.item())
            if batch_best_val_float > best_reward:
                best_reward = batch_best_val_float
                best_idx_global = i + int(batch_best_idx.item())
                center = center_batch[int(batch_best_idx.item())].detach().clone()
            elif center is None:
                center = center_batch[0].detach().clone()

            del center_batch
        
        # Concatenate all rewards for tracking
        rewards_w = torch.cat(rewards_list, dim=0)
        
        history = []
        n_eval = warmup_size
        nfe = warmup_size

        iter_log = self.make_iteration_logger(problem, logger)

        # Determine iteration schedule based on evaluations consumed by warmup.
        # We treat each batch-sized chunk as one iteration.
        warmup_iters = (n_eval + batch_size - 1) // batch_size if batch_size > 0 else 0
        if oracle_budget > 0:
            remaining = max(0, oracle_budget - n_eval)
            total_iters = 0 if batch_size <= 0 else (remaining + batch_size - 1) // batch_size
        else:
            total_iters = max(0, num_iterations - warmup_iters)

        for it in range(int(total_iters)):
            # Update iteration for progressive steps tracking
            if hasattr(problem, "set_iteration"):
                problem.set_iteration(it)
            
            if oracle_budget > 0:
                remaining = oracle_budget - n_eval
                if remaining <= 0:
                    break
                this_bs = min(batch_size, remaining)
            else:
                this_bs = batch_size

            noise = torch.randn(this_bs, *latent_shape, device=self.device, generator=self._generator)
            if perturbation == "sphere":
                noise_norm = torch.norm(noise.view(this_bs, -1), dim=1, keepdim=True)
                noise = noise / noise_norm.view(this_bs, *([1] * len(latent_shape)))
            candidates = center.unsqueeze(0) + step_size * noise

            rewards = problem.evaluate(candidates)
            n_eval += int(this_bs)
            nfe += int(this_bs)
            max_reward, max_idx = torch.max(rewards, dim=0)
            max_reward_val = float(max_reward.item())
            if max_reward_val > best_reward:
                best_reward = max_reward_val
                center = candidates[max_idx].detach().clone()
            iter_metrics = {
                "iteration": it,
                "best": best_reward,
                "step_size": step_size,
                "num_evaluations": n_eval,
                "nfe": nfe,
            }
            history.append(iter_metrics)
            iter_log(it, rewards, console_prefix="ZeroOrder")

        return SolveResult(
            best_reward=best_reward,
            best_candidate=center,
            history=history,
            num_evaluations=n_eval,
            num_function_evals=nfe,
            metadata={"solver": "zero_order", "seed": seed},
        )

    def _initialize_impl(self, problem: Any, **kwargs: Any) -> None:
        """Initialize the zero-order solver for step-based execution."""
        self._latent_shape = getattr(problem, "latent_shape", None)
        self._batch_size = int(kwargs.get("batch_size", 16))
        self._step_size = float(kwargs.get("step_size", 0.1))
        self._oracle_budget = int(kwargs.get("oracle_budget", 0))
        warmup_batches = int(kwargs.get("warmup_batches", 0))
        warmup_samples = int(kwargs.get("warmup_samples", 0))
        self._perturbation = self._normalize_perturbation_mode(kwargs.get("perturbation", kwargs.get("perturbation_mode", "gaussian")))

        if self._latent_shape is None:
            raise ValueError("ZeroOrderSolver requires problem.latent_shape to be set (got None).")

        self._seed = self._extract_seed(problem, kwargs)
        self._ensure_generator(self._seed)

        warmup_size = self._compute_warmup_size(self._batch_size, warmup_batches, warmup_samples)
        if self._oracle_budget > 0:
            warmup_size = min(warmup_size, self._oracle_budget)

        if hasattr(problem, "sample") and callable(problem.sample):
            try:
                center_batch_all = problem.sample(batch_size=warmup_size, latent_shape=self._latent_shape, seed=self._seed)
                z_batch_all = None
            except TypeError:
                center_batch_all = problem.sample(batch_size=warmup_size, latent_shape=self._latent_shape)
                z_batch_all = None
        else:
            D = int(torch.tensor(self._latent_shape).prod().item())
            z_batch_all = torch.randn(warmup_size, D, device=self.device, generator=self._generator)
            center_batch_all = None

        rewards_list = []
        best_reward = float("-inf")
        best_idx_global = None
        center = None

        for i in range(0, warmup_size, self._batch_size):
            end_idx = min(i + self._batch_size, warmup_size)
            this_batch_size = end_idx - i

            if z_batch_all is not None:
                z_batch = z_batch_all[i:end_idx]
                center_batch = z_batch.view(this_batch_size, *self._latent_shape)
            else:
                center_batch = center_batch_all[i:end_idx]

            rewards_batch = problem.evaluate(center_batch)
            rewards_list.append(rewards_batch)

            batch_best_val, batch_best_idx = torch.max(rewards_batch, dim=0)
            batch_best_val_float = float(batch_best_val.item())
            if batch_best_val_float > best_reward:
                best_reward = batch_best_val_float
                best_idx_global = i + int(batch_best_idx.item())
                center = center_batch[int(batch_best_idx.item())].detach().clone()
            elif center is None:
                center = center_batch[0].detach().clone()

            del center_batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self._center = center
        self._best_candidate = center.clone() if center is not None else None
        self._best_reward = best_reward
        self._n_eval = warmup_size
        self._nfe = warmup_size

    def _execute_step_impl(self) -> Dict[str, Any]:
        """Execute one step of zero-order search."""
        if self._problem is None:
            raise RuntimeError("Solver not initialized")

        # Check budget exhaustion
        if self._oracle_budget > 0 and self._n_eval >= self._oracle_budget:
            return {
                "best_reward": self._best_reward,
                "best_candidate": self._best_candidate,
                "best_image": None,
                "batch_rewards": None,
                "n_eval": self._n_eval,
                "nfe": self._nfe,
                "step_size": self._step_size,
                "batch_mean": self._best_reward,
                "batch_max": self._best_reward,
            }
        
        # Update iteration for progressive steps tracking
        if hasattr(self._problem, "set_iteration"):
            self._problem.set_iteration(self._current_iteration - 1)
        
        # Determine how many evaluations we can still afford this step
        if self._oracle_budget > 0:
            remaining = self._oracle_budget - self._n_eval
            if remaining <= 0:
                return {
                    "best_reward": self._best_reward,
                    "best_candidate": self._best_candidate,
                    "best_image": None,
                    "batch_rewards": None,
                    "n_eval": self._n_eval,
                    "nfe": self._nfe,
                    "step_size": self._step_size,
                    "batch_mean": self._best_reward,
                    "batch_max": self._best_reward,
                }
            this_bs = min(self._batch_size, remaining)
        else:
            this_bs = self._batch_size

        # Generate candidates in full latent space
        noise = torch.randn(this_bs, *self._latent_shape, device=self.device, generator=self._generator)
        if self._perturbation == "sphere":
            noise_norm = torch.norm(noise.view(this_bs, -1), dim=1, keepdim=True)
            noise = noise / noise_norm.view(this_bs, *([1] * len(self._latent_shape)))
        candidates = self._center.unsqueeze(0) + self._step_size * noise
        
        # Evaluate candidates
        rewards = self._problem.evaluate(candidates)
        self._n_eval += int(this_bs)
        self._nfe += int(this_bs)
        
        max_reward, max_idx = torch.max(rewards, dim=0)
        max_reward_val = float(max_reward.item())
        
        best_candidate = None
        best_image = None
        
        # Update center if improvement found
        if max_reward_val > self._best_reward:
            self._best_reward = max_reward_val
            self._center = candidates[max_idx].detach().clone()
            self._best_candidate = self._center.clone()
            best_candidate = self._center.clone()

            # For protein modality, get decoded structure using problem.decode_latents() which uses cache
            # For image modality, render the best image
            try:
                if hasattr(self._problem, "context") and self._problem.context.get("modality") in ("protein", "proteina"):
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
                            # Store for later use (though zero_order doesn't currently return best_protein)
                            # This prevents unnecessary regeneration if the benchmark needs it
                else:
                    # Render the best image (for image modality)
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
            "best_image": best_image,
            "batch_rewards": rewards.detach(),
            "n_eval": self._n_eval,
            "nfe": self._nfe,
            "step_size": self._step_size,
            "batch_mean": float(rewards.mean().item()),
            "batch_max": max_reward_val,
        }

    def _is_done_impl(self) -> bool:
        """Check if zero-order search is done."""
        return False

    def _get_metadata_impl(self) -> Dict[str, Any]:
        """Get zero-order search metadata."""
        return {"seed": self._seed}
