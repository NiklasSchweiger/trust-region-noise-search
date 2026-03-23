from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from torch.quasirandom import SobolEngine

from .base import Solver, SolveResult
from ..loggers.base import ExperimentLogger
from ..utils import (
    select_centers_diverse,
    select_centers_clustering,
    calculate_1_5_rule_length,
    calculate_variance_based_length,
    select_with_diversity,
    compute_region_improvement_rates,
    adaptive_region_allocation,
    calculate_cosine_adaptation,
    get_scoring_function,
    generate_warmup_samples,
    generate_and_select_candidates,
    update_trust_region_states,
    update_trust_region_centers,
    allocate_batch_across_regions,
    generate_candidates_around_center,
)
from ..utils.tr_archive import update_archive
from ..utils.tr_state import TRState


# ===========================
# Module-level Utilities
# ===========================
# Pure functions with no state dependency, reusable across solvers

# NOTE: Design decision for production code organization:
# - TRState: Moved to utils/tr_state.py because it's reused by both TrustRegionSolver
#   and BayesianOptimizationSolver. This reduces duplication and file length.
#
# - Candidate generation: Moved to tr_utils.py as generate_candidates_around_center()
#   because it's a large, reusable function that could benefit other solvers.
#
# - Archive updates: Moved to tr_archive.py because it's a simple, pure operation
#   that's duplicated and could be reused by other solvers.
#
# - Trust region utilities: Moved to tr_utils.py because they're reusable operations
#   (scoring, center selection, etc.) that could benefit other trust-region-based solvers.
#
# Principle: Extract utilities that are pure, reusable, and represent distinct concerns.
# Keep solver-specific logic in the solver file for cohesion.




# ===========================
# Trust Region Solver
# ===========================

def _reshape_to_latent(z: torch.Tensor, latent_shape: Tuple[int, ...]) -> torch.Tensor:
    """Reshape flat batch (B, D) to (B, *latent_shape)."""
    return z.view(z.size(0), *latent_shape)


class TrustRegionSolver(Solver):
    """Trust region search in the full noise space (surrogate-free).

    Maintains multiple trust regions (TuRBO-style) in the original latent space.
    Supports Sobol and Gaussian sampling for warmup and search. Candidate
    selection uses random scoring. For Gaussian sampling uses length/2 as sigma
    and clips candidates to trust region bounds.
    """

    def __init__(self, device: Optional[str] = None, logger: Optional[ExperimentLogger] = None):
        """Initialize trust region solver.
        
        Args:
            device: Device for tensor operations (defaults to CUDA if available)
            logger: Optional experiment logger for tracking progress
        """
        super().__init__(device, logger)
        
        # Problem-specific state (set during initialization)
        self._latent_shape = None
        
        # Optimization parameters
        self._batch_size = 16
        self._num_iterations = 50
        self._num_regions = 4
        self._oracle_budget = 0
        self._pool_per_region = 128
        
        self._seed = None  # Random seed (set during initialization)
        self._generator = None  # PyTorch generator for reproducibility
        
        # Sampling configuration
        self._init_sampling_mode = "sobol"  # "sobol" | "gaussian"
        self._search_sampling_mode = "sobol"  # "sobol" | "gaussian"
        
        # Trust region parameters
        self._init_length = 0.8
        self._min_length = 0.05
        self._max_length = 1.6
        self._success_tol = 3
        self._failure_tol = 3
        
        # Perturbation parameters
        self._prob_perturb_kw = None
        self._perturb_min_frac = 0.1
        self._perturb_max_frac = 0.9
        
        # Step-based execution state (initialized in _initialize_impl)
        self._Z_archive = None  # Archive of latent points
        self._R_archive = None  # Archive of rewards
        self._centers_z = None  # Trust region centers in z-space
        self._centers_val = None  # Trust region center values
        self._tr_states = None  # List of TRState objects (one per region)
        self._reward_curve = []

    def _generate_candidates_around_center(
        self,
        z_center: torch.Tensor,
        length: float,
        n_candidates: int,
        prob_perturb: Optional[float] = None,
        seed: Optional[int] = None,
        sampling_mode: str = "sobol",
    ) -> torch.Tensor:
        """Generate candidates around a trust region center using Sobol or Gaussian sampling.
        
        Wrapper around generate_candidates_around_center() that uses instance state.
        
        Args:
            z_center: Center point in reduced space, shape (d,)
            length: Trust region side length
            n_candidates: Number of candidate points to generate
            prob_perturb: Probability of perturbing each dimension (None to disable masking)
            seed: Random seed for Sobol sequence generation (uses self._seed if None)
            sampling_mode: "sobol" or "gaussian" for candidate generation
            
        Returns:
            Candidate points, shape (n_candidates, d)
        """
        return generate_candidates_around_center(
            z_center=z_center,
            length=length,
            n_candidates=n_candidates,
            device=self.device,
            prob_perturb=prob_perturb,
            seed=seed if seed is not None else self._seed,
            sampling_mode=sampling_mode,
            generator=self._generator,
            tr_shape=getattr(self, '_tr_shape', 'hypercube'),
        )
    
    def _choose_candidates_randomly(self, candidates: torch.Tensor) -> torch.Tensor:
        """Score candidates using random values (surrogate-free).
        
        Args:
            candidates: Candidate points in reduced space, shape (n, d)
            
        Returns:
            Random scores for each candidate, shape (n,)
        """
        scoring_func = get_scoring_function("random")
        # Dummy values for unused parameters
        dummy_center = torch.zeros(candidates.shape[1], device=candidates.device)
        dummy_archive = torch.zeros(0, candidates.shape[1], device=candidates.device)
        dummy_rewards = torch.zeros(0, device=candidates.device)
        return scoring_func(candidates, dummy_center, dummy_archive, dummy_rewards, None, self._generator, {})

    def _solve_impl(self, problem: Any, **kwargs: Any) -> SolveResult:
        """Run trust region search optimization.
        
        Implements the full trust region search algorithm:
        1. Initialize warmup samples and select top-k as trust region centers
        2. For each iteration:
           a. Generate candidate pools around each trust region center
           b. Select top candidates using random scoring (surrogate-free)
           c. Batch evaluate all candidates
           d. Update trust region states (expand/shrink based on success/failure)
           e. Update trust region centers to global top-k
        3. Return best solution and optimization history
        
        Args:
            problem: Optimization problem with evaluate() method and latent_shape attribute
            **kwargs: Solver configuration parameters:
                - num_iterations: Maximum number of iterations
                - batch_size: Number of evaluations per iteration
                - num_regions: Number of parallel trust regions
                - init_sampling_mode: "sobol" or "gaussian" for warmup
                - search_sampling_mode: "sobol" or "gaussian" for search
                - See config/main.yaml for full parameter list
            
        Returns:
            SolveResult containing best solution and optimization history
        """
        device = self.device
        latent_shape: Tuple[int, ...] = problem.latent_shape  # type: ignore[attr-defined]
        D = int(torch.tensor(latent_shape).prod().item())

        print(f"[TRS] Latent shape: {latent_shape}")
        print(f"[TRS] D: {D}")

        # Extract hyperparameters with sensible defaults
        num_iterations = int(kwargs.get("num_iterations", 50))
        batch_size = int(kwargs.get("batch_size", 16))
        num_regions = int(kwargs.get("num_regions", kwargs.get("trust_regions", 4)))
        oracle_budget = int(kwargs.get("oracle_budget", 0))
        
        # Trust region parameters
        init_length = float(kwargs.get("init_length", 0.8))
        min_length = float(kwargs.get("min_length", 0.05))
        max_length = float(kwargs.get("max_length", 1.6))
        success_tol = int(kwargs.get("success_tolerance", 3))
        failure_tol = int(kwargs.get("failure_tolerance", 3))
        update_factor = float(kwargs.get("update_factor", 2.0))
        
        # Perturbation parameters
        prob_perturb_kw = kwargs.get("prob_perturb", None)
        perturb_min_frac = float(kwargs.get("perturb_min_frac", 0.1))
        perturb_max_frac = float(kwargs.get("perturb_max_frac", 0.9))
        
        warmup_batches = int(kwargs.get("warmup_batches", 0))
        warmup_samples = (warmup_batches * batch_size) if warmup_batches > 0 else max(num_regions, batch_size)
        # Ensure warmup is a multiple of batch_size so every warmup batch uses the same batch size as solver iterations
        if batch_size > 0:
            warmup_samples = ((warmup_samples + batch_size - 1) // batch_size) * batch_size
            
        logger = kwargs.get("logger", self._default_logger)

        # Seed handling: seed > problem.context > problem.cfg > default 42
        seed_val = kwargs.get("seed", None)
        if seed_val is None:
            if hasattr(problem, "context") and "seed" in problem.context:
                seed_val = problem.context["seed"]
            elif hasattr(problem, "cfg") and hasattr(problem.cfg, "get"):
                seed_val = problem.cfg.get("seed")
            else:
                seed_val = 42
        base_seed = int(seed_val) if seed_val is not None else 42
        task_index = None
        if hasattr(problem, "context") and problem.context:
            task_index = problem.context.get("task_index")
            if task_index is None:
                task_index = problem.context.get("length_index")
            if task_index is None:
                task_index = problem.context.get("local_index")
        if task_index is not None:
            seed_val = base_seed + int(task_index) * 10000
        else:
            seed_val = base_seed

        # Working dimension: always full latent dimension D (no projection).
        sobol_dim = D

        # Determine total iterations (search iterations; warmup handled separately below)
        warmup_iters = (warmup_samples + batch_size - 1) // batch_size if batch_size > 0 else 0
        
        if oracle_budget > 0:
            # Oracle budget is a hard limit on total evaluations
            total_budget_iters = (oracle_budget + batch_size - 1) // batch_size
            # Respect num_iterations as the target if it's smaller than the oracle cap
            total_iters = min(num_iterations, total_budget_iters)
        else:
            total_iters = num_iterations
            
        # Search iterations = Total target - iterations consumed by warmup
        search_iters = max(0, total_iters - warmup_iters)

        # Ensure sufficient warmup samples (already set from warmup_batches or default)
        if warmup_samples <= 0:
            warmup_samples = max(num_regions, batch_size)

        # Initialization sampling strategy
        # Fallback to search_sampling_mode if not explicitly provided
        init_sampling_mode = str(kwargs.get("init_sampling_mode", kwargs.get("search_sampling_mode", "sobol"))).lower()
        if init_sampling_mode not in ("sobol", "gaussian", "gaussian_box"):
            init_sampling_mode = "sobol"
        
        # Search sampling strategy
        search_sampling_mode = str(kwargs.get("search_sampling_mode", "sobol")).lower()
        if search_sampling_mode not in ("sobol", "gaussian", "gaussian_box"):
            search_sampling_mode = "sobol"

        self._seed = seed_val
        self._init_sampling_mode = init_sampling_mode
        self._search_sampling_mode = search_sampling_mode
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed_val))
        self._generator = torch.Generator(device=device)
        self._generator.manual_seed(int(seed_val))

        z_init = generate_warmup_samples(
            n_samples=warmup_samples,
            dim=sobol_dim,
            sampling_mode=init_sampling_mode,
            device=device,
            seed=seed_val,
            generator=gen,
        )

        # Evaluate warmup samples in batches to avoid GPU memory spikes
        y_init_list = []
        best_warmup_val = float("-inf")
        best_warmup_latent = None

        warmup_batch_idx = 0
        for i in range(0, warmup_samples, batch_size):
            z_batch = z_init[i:i + batch_size]
            x0_batch = _reshape_to_latent(z_batch, latent_shape)
            y_batch = problem.evaluate(x0_batch).view(-1).detach()
            y_init_list.append(y_batch)

            # Track best candidate and clear the rest of the batch latents
            batch_best_val, batch_best_idx = torch.max(y_batch, dim=0)
            batch_mean = float(y_batch.mean().item())
            batch_max = float(batch_best_val.item())
            if batch_max > best_warmup_val:
                best_warmup_val = batch_max
                best_warmup_latent = x0_batch[batch_best_idx].detach().clone()
            
            # Log warmup iteration statistics
            print(f"[TRS] warmup_batch={warmup_batch_idx} batch_mean={batch_mean:.4f} batch_max={batch_max:.4f} global_best={best_warmup_val:.4f}")
            warmup_batch_idx += 1
            
            # Explicitly clear batch latents from GPU memory
            del x0_batch
        
        y_init = torch.cat(y_init_list, dim=0)

        # Initialize archives in z-space
        Z_archive = z_init.detach().clone()
        R_archive = y_init.detach().clone()

        # Select top-k points as trust region centers
        topk_vals, topk_idx = torch.topk(R_archive.view(-1), k=min(num_regions, R_archive.numel()))
        centers_z = Z_archive[topk_idx].detach().clone()
        centers_val = topk_vals.detach().clone()
        prev_centers_z = centers_z.detach().clone()
        
        # Update num_regions to match actual number of centers (important when warmup samples < num_regions)
        actual_num_regions = centers_z.shape[0]
        if actual_num_regions < num_regions:
            print(f"[TRS] Warning: Only {actual_num_regions} centers available from warmup (< num_regions={num_regions}). Reducing num_regions to {actual_num_regions}.")
            num_regions = actual_num_regions

        # Track global best
        best_reward = float(topk_vals.max().item()) if topk_vals.numel() > 0 else float("-inf")
        best_candidate_latent = best_warmup_latent
        
        # Initialize optimization history
        history = []
        logged_checkpoints: set[int] = set()
        
        # Initialize region rewards history for adaptive allocation
        region_rewards_history: List[List[float]] = [[] for _ in range(num_regions)]

        # Initialize trust region states
        states: List[TRState] = []
        for i in range(centers_z.shape[0]):
            states.append(TRState(
                update_factor=update_factor,
                update_mode=str(kwargs.get("length_update_mode", "standard")).lower(),
                length=init_length,
                min_length=min_length,
                max_length=max_length,
                success_tolerance=success_tol,
                failure_tolerance=failure_tol,
                best_value=float(centers_val[i].item()),
            ))

        # Initialize evaluation counters
        n_eval = int(warmup_samples)
        nfe = int(warmup_samples)
        
        iter_log = self.make_iteration_logger(problem, logger)
        reward_curve: List[float] = [best_reward]

        # Candidate generation: with fast-forward we use max_sobol_index as effective pool size
        use_fast_forward = bool(kwargs.get("use_fast_forward", False))
        max_sobol_index = int(kwargs.get("max_sobol_index", 128))
        pool_per_region = max_sobol_index if use_fast_forward else int(kwargs.get("cand_pool_per_region", 128))
        
        # Scoring: random only (fast-forward bypasses scoring)
        scoring_method = str(kwargs.get("scoring_method", "random")).lower()
        heuristic_kwargs = dict(kwargs.get("heuristic_kwargs", {}))
        
        # Normalize center selection mode (for _solve_impl which doesn't call _initialize_impl)
        # Check center_selection at top level, or fall back to tr.center_selection for backward compatibility
        center_selection_mode_raw = kwargs.get("center_selection", None)
        if center_selection_mode_raw is None:
            tr_config = kwargs.get("tr", {})
            center_selection_mode_raw = tr_config.get("center_selection", "topk")
        center_selection_mode_raw = str(center_selection_mode_raw).lower()
        if center_selection_mode_raw in ("topk", "global_topk", "global", "global_best"):
            center_selection_mode = "topk"
        elif center_selection_mode_raw in ("strict_local", "per_region_strict", "local_strict", "per_region"):
            center_selection_mode = "strict_local"
        elif center_selection_mode_raw in ("last_iter_local", "per_region_last", "local_last"):
            center_selection_mode = "last_iter_local"
        elif center_selection_mode_raw in ("last_iter_topk", "batch_topk", "iter_topk"):
            center_selection_mode = "last_iter_topk"
        elif center_selection_mode_raw in ("diverse", "diverse_global", "annealed_diverse_topk"):
            center_selection_mode = "diverse"
        elif center_selection_mode_raw in ("clustering", "cluster_global"):
            center_selection_mode = "clustering"
        else:
            center_selection_mode = "topk"
        # Store in instance for consistency with step-based execution
        self._center_selection_mode = center_selection_mode
        self._tr_config = kwargs  # Store config for use in update_trust_region_centers
        
        # Experimental features
        use_adaptive_allocation = bool(kwargs.get("use_adaptive_allocation", False))
        use_region_annealing = bool(kwargs.get("use_region_annealing", False))
        anneal_start_frac = float(kwargs.get("anneal_start_frac", 0.3))
        anneal_interval = int(kwargs.get("anneal_interval", max(1, search_iters // (num_regions if num_regions > 1 else 1))))
        
        allocation_temperature = float(kwargs.get("allocation_temperature", 1.0))
        use_two_phase = bool(kwargs.get("use_two_phase", False))
        phase_switch_frac = float(kwargs.get("phase_switch_frac", 0.3))
        use_diversity_selection = bool(kwargs.get("use_diversity_selection", False))
        diversity_weight = float(kwargs.get("diversity_weight", 0.3))
        raw_arch = kwargs.get("archive_max_size", kwargs.get("max_archive_size", None))
        archive_max_size = int(raw_arch) if raw_arch is not None and int(raw_arch) > 0 else None
        
        use_fast_forward = bool(kwargs.get("use_fast_forward", False))
        max_sobol_index = int(kwargs.get("max_sobol_index", 128))

        remaining_search_iters = search_iters

        # ============================================================
        # Main Trust Region Search Loop
        # ============================================================
        for it in range(remaining_search_iters):
            if oracle_budget > 0 and n_eval >= oracle_budget:
                break

            # Update iteration for progressive steps tracking
            if hasattr(problem, "set_iteration"):
                problem.set_iteration(it)

            # Allocate total batch across regions (at least 1 per region)
            batch_this_iter = batch_size if oracle_budget == 0 else min(batch_size, oracle_budget - n_eval)
            if batch_this_iter <= 0:
                break
            
            alloc = allocate_batch_across_regions(
                batch_size=batch_this_iter,
                num_regions=num_regions,
                use_adaptive_allocation=False,
                improvement_rates=None,
                allocation_temperature=1.0,
            )

            # ============================================================
            # Step 1: Generate candidates from all regions
            # ============================================================
            regions_data = []
            for ridx in range(num_regions):
                k_evals = alloc[ridx]
                if k_evals <= 0:
                    continue

                zc = centers_z[ridx].detach()
                st = states[ridx]
                prev_center = prev_centers_z[ridx] if ridx < prev_centers_z.shape[0] else zc
                region_seed = seed_val + (it * num_regions + ridx) if seed_val is not None else None

                chosen = generate_and_select_candidates(
                    z_center=zc,
                    tr_state=st,
                    generate_candidates_fn=self._generate_candidates_around_center,
                    Z_archive=Z_archive,
                    R_archive=R_archive,
                    prev_center=prev_center,
                    k_evals=k_evals,
                    pool_per_region=pool_per_region,
                    prob_perturb_kw=prob_perturb_kw,
                    perturb_min_frac=perturb_min_frac,
                    perturb_max_frac=perturb_max_frac,
                    search_sampling_mode=search_sampling_mode,
                    region_seed=region_seed,
                    generator=self._generator,
                    scoring_method=scoring_method,
                    heuristic_kwargs=heuristic_kwargs,
                    use_two_phase=False,
                    phase_switch_frac=0.3,
                    oracle_budget=oracle_budget,
                    num_iterations=search_iters,
                    batch_size=batch_size,
                    n_eval=n_eval,
                    use_diversity_selection=False,
                    diversity_weight=0.3,
                    use_fast_forward=use_fast_forward,
                    max_sobol_index=max_sobol_index,
                )
                
                regions_data.append((ridx, chosen, k_evals))

            if len(regions_data) == 0:
                break
            
            # ============================================================
            # Step 2: Batch evaluate all candidates (GPU efficiency)
            # ============================================================
            all_chosen_z = torch.cat([chosen for _, chosen, _ in regions_data], dim=0)
            all_x0 = _reshape_to_latent(all_chosen_z, latent_shape)
            all_y_true = problem.evaluate(all_x0).view(-1).detach()
            
            # ============================================================
            # Step 3: Update trust region states and archives
            # ============================================================
            new_z_all, new_y_all, _ = update_trust_region_states(
                regions_data=regions_data,
                all_y_true=all_y_true,
                all_x0=all_x0,
                states=states,
                centers_z=centers_z,
                Z_archive=Z_archive,
                R_archive=R_archive,
                init_length=init_length,
                use_adaptive_allocation=False,
                region_rewards_history=region_rewards_history,
                return_latents=False,
            )

            if len(new_y_all) == 0:
                break

            # Update archives (bounded when archive_max_size is set)
            Z_new = torch.cat(new_z_all, dim=0)
            Y_new = torch.cat(new_y_all, dim=0)
            Z_archive, R_archive = update_archive(Z_archive, R_archive, Z_new, Y_new, max_size=archive_max_size)
            n_eval += int(Y_new.shape[0])
            nfe += int(Y_new.shape[0])

            # Update trust region centers
            # Store previous centers for momentum/cosine calculations
            prev_centers_z = centers_z.detach().clone()
            
            centers_z, centers_val = update_trust_region_centers(
                Z_archive=Z_archive,
                R_archive=R_archive,
                num_regions=num_regions,
                center_selection_mode=getattr(self, '_center_selection_mode', 'topk'),
                tr_config=getattr(self, '_tr_config', {}),
                device=self.device,
                Z_last=Z_new,
                R_last=Y_new,
                new_z_all=new_z_all,
                new_y_all=new_y_all,
                current_centers_z=prev_centers_z,
                current_centers_val=centers_val,
            )
                
            for i in range(min(len(states), centers_val.shape[0])):
                states[i].best_value = float(centers_val[i].item())

            # Update global best from the entire batch evaluated this iteration
            iter_best_val, iter_best_idx = torch.max(all_y_true, dim=0)
            batch_mean = float(all_y_true.mean().item())
            batch_max = float(iter_best_val.item())
            if batch_max > best_reward:
                best_reward = batch_max
                best_candidate_latent = all_x0[int(iter_best_idx.item())].detach().clone()

            # Log iteration statistics
            print(f"[TRS] iter={it} batch_mean={batch_mean:.4f} batch_max={batch_max:.4f} global_best={best_reward:.4f}")

            # Clear temporary latent batch to free GPU memory
            del all_x0
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            reward_curve.append(best_reward)
            
            # Record history
            history_entry = {
                "iteration": it,
                "best": best_reward,
                "num_evaluations": n_eval,
                "nfe": nfe,
            }
            
            # Track checkpoint-specific rewards for scaling experiments
            if hasattr(problem, "context") and problem.context.get("scaling_mode", False):
                if "scaling_checkpoints" in problem.context:
                    checkpoints = problem.context["scaling_checkpoints"]
                    for checkpoint in checkpoints:
                        ckpt_i = int(checkpoint)
                        if (ckpt_i not in logged_checkpoints) and (n_eval >= ckpt_i):
                            history_entry[f"checkpoint_{checkpoint}"] = best_reward
                            logged_checkpoints.add(ckpt_i)
            
            history.append(history_entry)
            iter_log(it, Y_new, console_prefix="TrustRegion")

            # Region Annealing: drop least performing regions
            if use_region_annealing and it >= (search_iters * anneal_start_frac) and (it % anneal_interval == 0) and num_regions > 1:
                num_regions -= 1
                # centers_z and centers_val are already sorted by value from update_trust_region_centers
                centers_z = centers_z[:num_regions]
                centers_val = centers_val[:num_regions]
                states = states[:num_regions]
                if use_adaptive_allocation:
                    region_rewards_history = region_rewards_history[:num_regions]
                print(f"[TRS] Annealing: Reduced num_regions to {num_regions} at iteration {it}")

        # Decode best protein/molecule if needed
        best_decoded = None
        if hasattr(problem, "context"):
            modality = problem.context.get("modality", "")
            if modality in ("protein", "proteina"):
                try:
                    if hasattr(problem, "decode_latents"):
                        latent_to_decode = best_candidate_latent.unsqueeze(0) if best_candidate_latent.dim() == 2 else best_candidate_latent
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
                        latent_to_decode = best_candidate_latent.unsqueeze(0) if best_candidate_latent.dim() == 2 else best_candidate_latent
                        decoded = problem.decode_latents(latent_to_decode)
                        if isinstance(decoded, (list, tuple)):
                            best_decoded = decoded[0]
                        elif isinstance(decoded, torch.Tensor):
                            best_decoded = decoded[0]
                        else:
                            best_decoded = decoded
                except Exception:
                    pass

        metadata: Dict[str, Any] = {
            "solver": "trust_region",
            "reward_curve": reward_curve,
            "best_protein": best_decoded if hasattr(problem, "context") and problem.context.get("modality") in ("protein", "proteina") else None,
            "best_molecule": best_decoded if hasattr(problem, "context") and problem.context.get("modality") in ("molecule", "qm9", "qm9_target") else None,
        }
        return SolveResult(
            best_reward=best_reward,
            best_candidate=best_candidate_latent,
            history=history,
            num_evaluations=n_eval,
            num_function_evals=nfe,
            metadata=metadata,
        )

    def _initialize_impl(self, problem: Any, **kwargs: Any) -> None:
        """Initialize the trust region solver for step-based execution.
        
        Args:
            problem: Optimization problem with evaluate() method and latent_shape
            **kwargs: Configuration parameters (batch_size, num_iterations, etc.)
        """
        self._latent_shape = problem.latent_shape
        D = int(torch.tensor(self._latent_shape).prod().item())
        
        # Extract hyperparameters
        self._batch_size = int(kwargs.get("batch_size", 16))
        self._num_iterations = int(kwargs.get("num_iterations", 50))
        self._num_regions = int(kwargs.get("num_regions", kwargs.get("trust_regions", 4)))
        self._oracle_budget = int(kwargs.get("oracle_budget", 0))
        
        # Trust region parameters
        self._init_length = float(kwargs.get("init_length", 0.8))
        #print(f"[TRS] Init Region with init_length: {self._init_length}")
        self._min_length = float(kwargs.get("min_length", 0.05))
        self._max_length = float(kwargs.get("max_length", 1.6))
        self._success_tol = int(kwargs.get("success_tolerance", 3))
        self._failure_tol = int(kwargs.get("failure_tolerance", 3))
        self._update_factor = float(kwargs.get("update_factor", 2.0))
        
        # Perturbation parameters
        self._prob_perturb_kw = kwargs.get("prob_perturb", None)
        self._perturb_min_frac = float(kwargs.get("perturb_min_frac", 0.1))
        self._perturb_max_frac = float(kwargs.get("perturb_max_frac", 0.9))
        
        warmup_batches = int(kwargs.get("warmup_batches", 0))
        warmup_samples = (warmup_batches * self._batch_size) if warmup_batches > 0 else max(self._num_regions, self._batch_size)
        # Ensure warmup is a multiple of batch_size so every warmup batch uses the same batch size as solver iterations
        if self._batch_size > 0:
            warmup_samples = ((warmup_samples + self._batch_size - 1) // self._batch_size) * self._batch_size
        
        # Trust region shape: hypercube, hyperrectangle, or hypersphere
        # Note: Sobol sampling generates points in [0,1]^d (a hypercube), then we scale to the trust region.
        # - "hypercube": Same length in all dimensions (original behavior, axis-aligned cube)
        # - "hyperrectangle": Different lengths per dimension (axis-aligned box with varying side lengths)
        # - "hypersphere": Uniform sampling from sphere/disk (uses Gaussian normalization method)
        tr_shape_kw = kwargs.get("tr_shape", kwargs.get("trust_region_shape", None))
        if tr_shape_kw is None:
            # Try to get from tr config
            tr_config = kwargs.get("tr", {})
            tr_shape_kw = tr_config.get("shape", "hypercube")
        self._tr_shape = str(tr_shape_kw).lower()
        if self._tr_shape not in ("hypercube", "hyperrectangle", "hypersphere"):
            self._tr_shape = "hypercube"  # Default fallback
        
        # Seed handling: try multiple sources for reproducibility
        # Seed: seed > problem.context > problem.cfg > default 42
        seed_val = kwargs.get("seed", None)
        if seed_val is None:
            if hasattr(problem, "context") and "seed" in problem.context:
                seed_val = problem.context["seed"]
            elif hasattr(problem, "cfg") and hasattr(problem.cfg, "get"):
                seed_val = problem.cfg.get("seed")
            else:
                seed_val = 42
        base_seed = int(seed_val) if seed_val is not None else 42
        task_index = None
        if hasattr(problem, "context") and problem.context:
            task_index = problem.context.get("task_index")
            if task_index is None:
                task_index = problem.context.get("length_index")
            if task_index is None:
                task_index = problem.context.get("local_index")
        if task_index is not None:
            seed_val = base_seed + int(task_index) * 10000
        else:
            seed_val = base_seed

        self._seed = seed_val
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(int(seed_val))
        
        # Scoring configuration
        self._scoring_method = str(kwargs.get("scoring_method", "random")).lower()
        self._heuristic_kwargs = dict(kwargs.get("heuristic_kwargs", {}))
        
        # NEW: Experimental features
        self._use_diversity_selection = bool(kwargs.get("use_diversity_selection", False))
        self._diversity_weight = float(kwargs.get("diversity_weight", 0.3))
        self._use_adaptive_allocation = bool(kwargs.get("use_adaptive_allocation", False))
        self._use_region_annealing = bool(kwargs.get("use_region_annealing", False))
        self._anneal_start_frac = float(kwargs.get("anneal_start_frac", 0.3))
        self._anneal_interval = int(kwargs.get("anneal_interval", 10))
        self._allocation_temperature = float(kwargs.get("allocation_temperature", 1.0))
        self._use_two_phase = bool(kwargs.get("use_two_phase", False))
        self._phase_switch_frac = float(kwargs.get("phase_switch_frac", 0.3))
        
        # Fast forward: use max_sobol_index as effective pool size when enabled
        self._use_fast_forward = bool(kwargs.get("use_fast_forward", False))
        self._max_sobol_index = int(kwargs.get("max_sobol_index", 128))
        self._pool_per_region = self._max_sobol_index if self._use_fast_forward else int(kwargs.get("cand_pool_per_region", 128))
        
        # Bounded archive: keep only top N points by reward (None/0 = unbounded).
        # For global_topk center selection we only need top-k; full archive is not required.
        raw = kwargs.get("archive_max_size", kwargs.get("max_archive_size", None))
        self._archive_max_size = int(raw) if raw is not None and int(raw) > 0 else None
        
        # Initialize region rewards history for adaptive allocation
        self._region_rewards_history = [[] for _ in range(self._num_regions)]
        
        # Advanced TR strategies
        # Check center_selection at top level, or fall back to tr.center_selection for backward compatibility
        center_selection_mode_raw = kwargs.get("center_selection", None)
        if center_selection_mode_raw is None:
            tr_config = kwargs.get("tr", {})
            center_selection_mode_raw = tr_config.get("center_selection", "topk")
        # Normalize mode names: handle aliases and case-insensitive matching
        center_selection_mode_raw = str(center_selection_mode_raw).lower()
        if center_selection_mode_raw in ("topk", "global_topk", "global", "global_best"):
            center_selection_mode = "topk"
        elif center_selection_mode_raw in ("strict_local", "per_region_strict", "local_strict", "per_region"):
            center_selection_mode = "strict_local"
        elif center_selection_mode_raw in ("last_iter_local", "per_region_last", "local_last"):
            center_selection_mode = "last_iter_local"
        elif center_selection_mode_raw in ("last_iter_topk", "batch_topk", "iter_topk"):
            center_selection_mode = "last_iter_topk"
        elif center_selection_mode_raw in ("diverse", "diverse_global", "annealed_diverse_topk"):
            center_selection_mode = "diverse"
        elif center_selection_mode_raw in ("clustering", "cluster_global"):
            center_selection_mode = "clustering"
        else:
            # Default to topk for unknown modes
            center_selection_mode = "topk"
        
        length_update_mode = kwargs.get("length_update", "standard")
        
        # Store for use in methods
        self._center_selection_mode = center_selection_mode
        self._length_update_mode = length_update_mode
        self._tr_config = kwargs # Use all kwargs as config fallback
        
        # Override tr_shape if specified in tr config (takes precedence)
        if "shape" in tr_config:
            self._tr_shape = str(tr_config["shape"]).lower()
            if self._tr_shape not in ("hypercube", "hyperrectangle", "hypersphere"):
                self._tr_shape = "hypercube"  # Default fallback
        
        # Ensure sufficient warmup samples
        if warmup_samples <= 0:
            warmup_samples = max(self._num_regions, self._batch_size)
        
        # Initialization sampling strategy
        # Fallback to search_sampling_mode if not explicitly provided
        init_sampling_mode = str(kwargs.get("init_sampling_mode", kwargs.get("search_sampling_mode", "sobol"))).lower()
        if init_sampling_mode not in ("sobol", "gaussian", "gaussian_box"):
            init_sampling_mode = "sobol"
        
        # Search sampling strategy
        search_sampling_mode = str(kwargs.get("search_sampling_mode", "sobol")).lower()
        if search_sampling_mode not in ("sobol", "gaussian", "gaussian_box"):
            search_sampling_mode = "sobol"
        
        # Store sampling modes
        self._init_sampling_mode = init_sampling_mode
        self._search_sampling_mode = search_sampling_mode

        # Generate warmup samples (full dimension D)
        sobol_dim = D
        # Warmup prints controlled by iteration terminal settings
        logger = kwargs.get("logger", self._default_logger)
        should_print_iteration = logger.should_print_iteration() if logger and hasattr(logger, "should_print_iteration") else False

        Z = generate_warmup_samples(
            n_samples=warmup_samples,
            dim=sobol_dim,
            sampling_mode=init_sampling_mode,
            device=self.device,
            seed=self._seed,
            generator=self._generator,
        )


        # Finalize initialization
        self._warmup_z_queue = Z
        self._Z_archive = torch.empty(0, sobol_dim, device=self.device)
        self._R_archive = torch.empty(0, device=self.device)
        
        # Initialize counters and history
        self._n_eval = 0
        self._nfe = 0
        self._best_reward = float("-inf")
        self._best_candidate = None
        self._best_molecule = None
        self._best_protein = None
        self._reward_curve = []
        
        # Initialize placeholders for TR states
        self._tr_states: List[TRState] = []
        self._centers_z = torch.empty(0, sobol_dim, device=self.device)
        self._centers_val = torch.empty(0, device=self.device)
        self._prev_centers_z = torch.empty(0, sobol_dim, device=self.device)

        # CRITICAL: We are now handling warmup batches sequentially in step().
        # To prevent the base Solver class from reducing our iteration budget,
        # we set these to 0 in our local kwargs copy.
        self._kwargs["warmup_batches"] = 0
        self._kwargs["warmup_samples"] = 0

    def _initialize_tr_states(self) -> None:
        """Pick initial centers from archive and set up TR states."""
        # Select top-k points as trust region centers
        num_to_pick = min(self._num_regions, self._R_archive.numel())
        if num_to_pick > 0:
            topk_vals, topk_idx = torch.topk(self._R_archive.view(-1), k=num_to_pick)
            self._centers_z = self._Z_archive[topk_idx].detach().clone()
            self._centers_val = topk_vals.detach().clone()
        else:
            # Fallback if archive is empty (shouldn't happen after warmup)
            self._centers_z = torch.empty(0, self._Z_archive.shape[1], device=self.device)
            self._centers_val = torch.empty(0, device=self.device)

        self._prev_centers_z = self._centers_z.detach().clone()
        
        # Update num_regions to match actual number of centers (important when warmup samples < num_regions)
        actual_num_regions = self._centers_z.shape[0]
        if actual_num_regions < self._num_regions:
            print(f"[TRS] Warning: Only {actual_num_regions} centers available (warmup samples < num_regions={self._num_regions}). Reducing num_regions to {actual_num_regions}.")
            self._num_regions = actual_num_regions
            # Also adjust region_rewards_history size
            self._region_rewards_history = self._region_rewards_history[:self._num_regions]
        
        # Initialize trust region states
        self._tr_states = []
        for i in range(self._centers_z.shape[0]):
            self._tr_states.append(TRState(
                update_factor=self._update_factor,
                update_mode=str(self._length_update_mode).lower(),
                length=self._init_length,
                min_length=self._min_length,
                max_length=self._max_length,
                success_tolerance=self._success_tol,
                failure_tolerance=self._failure_tol,
                min_prob_perturb=self._perturb_min_frac,
                max_prob_perturb=self._perturb_max_frac,
                best_value=float(self._centers_val[i].item()),
            ))

    def _execute_step_impl(self) -> Dict[str, Any]:
        """Execute one iteration of trust region search.
        
        Returns:
            Dict with keys: best_reward, best_candidate, best_image, n_eval, nfe, 
                           batch_mean, batch_max
        """
        if self._problem is None:
            raise RuntimeError("Solver not initialized")
        
        # ============================================================
        # Warmup Phase: Evaluate queued batches
        # ============================================================
        if self._warmup_z_queue.numel() > 0:
            # Determine how many samples to evaluate this iteration
            # If we have a lot of samples, evaluate them in chunks of batch_size
            # If we have fewer than batch_size, evaluate all remaining
            num_to_eval = min(self._batch_size, self._warmup_z_queue.shape[0])
            
            z_batch = self._warmup_z_queue[:num_to_eval]
            self._warmup_z_queue = self._warmup_z_queue[num_to_eval:]
            
            x0_batch = _reshape_to_latent(z_batch, self._latent_shape)
            y_batch = self._problem.evaluate(x0_batch).view(-1).detach()
            
            # Update archive (bounded when archive_max_size is set)
            self._Z_archive, self._R_archive = update_archive(
                self._Z_archive, self._R_archive, z_batch, y_batch,
                max_size=getattr(self, "_archive_max_size", None),
            )
            self._n_eval += int(y_batch.shape[0])
            self._nfe += int(y_batch.shape[0])
            
            # Update global best
            batch_best_val, batch_best_idx = torch.max(y_batch, dim=0)
            if float(batch_best_val.item()) > self._best_reward:
                self._best_reward = float(batch_best_val.item())
                self._best_candidate = x0_batch[batch_best_idx].detach().clone()
            
            iter_best_candidate = x0_batch[batch_best_idx].detach().clone()
            
            # Finalize warmup if queue empty
            if self._warmup_z_queue.numel() == 0:
                self._initialize_tr_states()
            
            self._reward_curve.append(self._best_reward)
            
            return {
                "best_reward": self._best_reward,
                "best_candidate": self._best_candidate if float(batch_best_val.item()) > self._best_reward else None,
                "iter_best_candidate": iter_best_candidate,
                "n_eval": self._n_eval,
                "nfe": self._nfe,
                "batch_mean": float(y_batch.mean().item()),
                "batch_max": float(y_batch.max().item()),
                "batch_best_tr_length": 0.0,  # Warmup phase doesn't have TR yet
            }

        # ============================================================
        # Normal Phase: Trust Region Steps
        # ============================================================
        
        # Check budget exhaustion
        if self._oracle_budget > 0 and self._n_eval >= self._oracle_budget:
            return {
                "best_reward": self._best_reward,
                "best_candidate": self._best_candidate,
                "n_eval": self._n_eval,
                "nfe": self._nfe,
                "batch_mean": 0.0,
                "batch_max": 0.0,
            }
        
        # Update problem iteration counter
        if hasattr(self._problem, "set_iteration"):
            self._problem.set_iteration(self._current_iteration - 1)
        
        # Allocate batch across regions
        batch_this_iter = self._batch_size if self._oracle_budget == 0 else min(self._batch_size, self._oracle_budget - self._n_eval)
        if batch_this_iter <= 0:
            return {
                "best_reward": self._best_reward,
                "best_candidate": self._best_candidate,
                "n_eval": self._n_eval,
                "nfe": self._nfe,
                "batch_mean": 0.0,
                "batch_max": 0.0,
            }
        
        # Adaptive allocation based on improvement rates
        improvement_rates = None
        if getattr(self, '_use_adaptive_allocation', False) and self._current_iteration > 2:
            improvement_rates = compute_region_improvement_rates(
                self._region_rewards_history, window=3
            )
        alloc = allocate_batch_across_regions(
            batch_size=batch_this_iter,
            num_regions=self._num_regions,
            use_adaptive_allocation=getattr(self, '_use_adaptive_allocation', False) and self._current_iteration > 2,
            improvement_rates=improvement_rates,
            allocation_temperature=getattr(self, '_allocation_temperature', 1.0),
        )
        
        # ============================================================
        # Step 1: Generate candidates from all regions
        # ============================================================
        regions_data = []
        region_indices = []  # Track which region each candidate belongs to
        for ridx in range(self._num_regions):
            k_evals = alloc[ridx]
            if k_evals <= 0:
                continue
            
            zc = self._centers_z[ridx].detach()
            st = self._tr_states[ridx]
            prev_center = self._prev_centers_z[ridx] if ridx < self._prev_centers_z.shape[0] else zc
            region_seed = (
                self._seed + (self._current_iteration * self._num_regions + ridx) 
                if self._seed is not None else None
            )
            
            chosen = generate_and_select_candidates(
                z_center=zc,
                tr_state=st,
                generate_candidates_fn=self._generate_candidates_around_center,
                Z_archive=self._Z_archive,
                R_archive=self._R_archive,
                prev_center=prev_center,
                k_evals=k_evals,
                pool_per_region=self._pool_per_region,
                prob_perturb_kw=self._prob_perturb_kw,
                perturb_min_frac=self._perturb_min_frac,
                perturb_max_frac=self._perturb_max_frac,
                search_sampling_mode=self._search_sampling_mode,
                region_seed=region_seed,
                generator=self._generator,
                scoring_method=self._scoring_method,
                heuristic_kwargs=self._heuristic_kwargs,
                use_two_phase=getattr(self, '_use_two_phase', False),
                phase_switch_frac=getattr(self, '_phase_switch_frac', 0.3),
                oracle_budget=self._oracle_budget,
                num_iterations=self._num_iterations,
                batch_size=self._batch_size,
                n_eval=self._n_eval,
                use_diversity_selection=getattr(self, '_use_diversity_selection', False),
                diversity_weight=getattr(self, '_diversity_weight', 0.3),
                use_fast_forward=getattr(self, '_use_fast_forward', False),
                max_sobol_index=getattr(self, '_max_sobol_index', self._pool_per_region),
            )
            
            # Ensure chosen has the correct batch size (k_evals)
            if chosen.shape[0] > k_evals:
                chosen = chosen[:k_evals]
            elif chosen.shape[0] < k_evals and chosen.shape[0] > 0:
                # If we somehow got fewer, update k_evals to match actual count
                k_evals = chosen.shape[0]
            elif chosen.shape[0] == 0 and k_evals > 0:
                # CRITICAL: If candidate generation returned nothing, fallback to simple Gaussian
                # This can happen if Sobol/fast-forward paths fail or return empty tensors
                print(f"[TRS] WARNING: Region {ridx} generated 0 candidates (requested {k_evals}). Falling back to Gaussian.")
                chosen = self._generate_candidates_around_center(
                    z_center=zc,
                    length=st.length,
                    n_candidates=k_evals,
                    prob_perturb=self._prob_perturb_kw,
                    seed=region_seed,
                    sampling_mode="gaussian"
                )
                k_evals = chosen.shape[0]
            
            if chosen.shape[0] > 0:
                regions_data.append((ridx, chosen, k_evals))
                # Track region index for each candidate
                region_indices.extend([ridx] * k_evals)
        
        if len(regions_data) == 0:
            return {
                "best_reward": self._best_reward,
                "best_candidate": self._best_candidate,
                "n_eval": self._n_eval,
                "nfe": self._nfe,
                "batch_mean": 0.0,
                "batch_max": 0.0,
                "batch_best_tr_length": 0.0,
            }
        
        # ============================================================
        # Step 2: Batch evaluate all candidates
        # ============================================================
        all_chosen_z = torch.cat([chosen for _, chosen, _ in regions_data], dim=0)
        all_x0 = _reshape_to_latent(all_chosen_z, self._latent_shape)
        all_y_true = self._problem.evaluate(all_x0).view(-1).detach()
        
        # ============================================================
        # Step 3: Update trust region states and archives
        # ============================================================
        new_z_all, new_y_all, _ = update_trust_region_states(
            regions_data=regions_data,
            all_y_true=all_y_true,
            all_x0=all_x0,
            states=self._tr_states,
            centers_z=self._centers_z,
            Z_archive=self._Z_archive,
            R_archive=self._R_archive,
            init_length=self._init_length,
            use_adaptive_allocation=False,
            region_rewards_history=self._region_rewards_history,
            return_latents=False,
        )
        
        if len(new_y_all) == 0:
            return {
                "best_reward": self._best_reward,
                "best_candidate": self._best_candidate,
                "n_eval": self._n_eval,
                "nfe": self._nfe,
                "batch_mean": 0.0,
                "batch_max": 0.0,
            }
        
        Z_new = torch.cat(new_z_all, dim=0)
        Y_new = torch.cat(new_y_all, dim=0)
        
        # Update archives (bounded when archive_max_size is set)
        self._Z_archive, self._R_archive = update_archive(
            self._Z_archive, self._R_archive, Z_new, Y_new,
            max_size=getattr(self, "_archive_max_size", None),
        )
        self._n_eval += int(Y_new.shape[0])
        self._nfe += int(Y_new.shape[0])
        # Store previous centers for momentum/cosine calculations
        self._prev_centers_z = self._centers_z.detach().clone()
        
        self._centers_z, self._centers_val = update_trust_region_centers(
            Z_archive=self._Z_archive,
            R_archive=self._R_archive,
            num_regions=self._num_regions,
            center_selection_mode=getattr(self, '_center_selection_mode', 'topk'),
            tr_config=getattr(self, '_tr_config', {}),
            device=self.device,
            Z_last=Z_new,
            R_last=Y_new,
            new_z_all=new_z_all,
            new_y_all=new_y_all,
            current_centers_z=self._prev_centers_z,
            current_centers_val=self._centers_val,
        )
            
        for i in range(min(len(self._tr_states), self._centers_val.shape[0])):
            self._tr_states[i].best_value = float(self._centers_val[i].item())

        # Update global best from the entire batch evaluated this iteration
        iter_best_val, iter_best_idx = torch.max(all_y_true, dim=0)
        best_candidate = None
        iter_best_candidate = None  # Best candidate from this iteration (not necessarily global best)
        
        # Save iteration-best candidate before deleting all_x0
        iter_best_candidate = all_x0[int(iter_best_idx.item())].detach().clone()
        
        # Get trust region length for the best candidate
        best_tr_length = 0.0
        if len(region_indices) > 0 and int(iter_best_idx.item()) < len(region_indices):
            best_region_idx = region_indices[int(iter_best_idx.item())]
            if best_region_idx < len(self._tr_states):
                best_tr_length = float(self._tr_states[best_region_idx].length)
        
        if float(iter_best_val.item()) > self._best_reward:
            self._best_reward = float(iter_best_val.item())
            self._best_candidate = iter_best_candidate.clone()
            best_candidate = self._best_candidate.clone()
            
            # Decode best protein/molecule if needed
            if hasattr(self._problem, "context"):
                modality = self._problem.context.get("modality", "")
                if modality in ("protein", "proteina"):
                    try:
                        if hasattr(self._problem, "decode_latents"):
                            latent_to_decode = best_candidate.unsqueeze(0) if best_candidate.dim() == 2 else best_candidate
                            decoded = self._problem.decode_latents(latent_to_decode)
                            if decoded is not None:
                                if isinstance(decoded, torch.Tensor):
                                    self._best_protein = decoded[0].detach().clone() if decoded.dim() >= 3 else decoded.detach().clone()
                                elif isinstance(decoded, (list, tuple)) and len(decoded) > 0:
                                    self._best_protein = decoded[0]
                                else:
                                    self._best_protein = decoded
                    except Exception:
                        pass
                elif modality in ("molecule", "qm9", "qm9_target"):
                    try:
                        if hasattr(self._problem, "decode_latents"):
                            latent_to_decode = best_candidate.unsqueeze(0) if best_candidate.dim() == 2 else best_candidate
                            decoded = self._problem.decode_latents(latent_to_decode)
                            if isinstance(decoded, (list, tuple)):
                                self._best_molecule = decoded[0]
                            elif isinstance(decoded, torch.Tensor):
                                self._best_molecule = decoded[0]
                            else:
                                self._best_molecule = decoded
                    except Exception:
                        pass
        
        # Clear temporary latent batch to free GPU memory
        del all_x0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self._reward_curve.append(self._best_reward)
        
        # Region Annealing
        if self._use_region_annealing and self._current_iteration >= (self._num_iterations * self._anneal_start_frac) and (self._current_iteration % self._anneal_interval == 0) and self._num_regions > 1:
            self._num_regions -= 1
            self._centers_z = self._centers_z[:self._num_regions]
            self._centers_val = self._centers_val[:self._num_regions]
            self._tr_states = self._tr_states[:self._num_regions]
            if self._use_adaptive_allocation:
                self._region_rewards_history = self._region_rewards_history[:self._num_regions]
            print(f"[TRS] Annealing: Reduced num_regions to {self._num_regions} at iteration {self._current_iteration}")

        return {
            "best_reward": self._best_reward,
            "best_candidate": best_candidate,  # Global best (only if new global best found)
            "iter_best_candidate": iter_best_candidate,  # Best from this iteration
            "best_image": None,
            "best_molecule": getattr(self, "_best_molecule", None),
            "best_protein": getattr(self, "_best_protein", None),
            "n_eval": self._n_eval,
            "nfe": self._nfe,
            "batch_mean": float(Y_new.mean().item()),
            "batch_max": float(Y_new.max().item()),
            "batch_best_tr_length": best_tr_length,  # Trust region length for batch best
        }
    
    def _is_done_impl(self) -> bool:
        """Check if the solver should terminate early.
        
        Returns:
            False (trust region search doesn't have early termination criteria)
        """
        return False

    def _get_metadata_impl(self) -> Dict[str, Any]:
        """Get solver metadata for result tracking.
        
        Returns:
            Dict with reward_curve
        """
        metadata = {
            "reward_curve": self._reward_curve,
        }
        return metadata