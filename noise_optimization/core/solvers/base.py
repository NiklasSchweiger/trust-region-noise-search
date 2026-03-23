from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from ..loggers.base import ExperimentLogger
from ..loggers.t2i import T2IWandbLogger

@dataclass
class SolveResult:
    best_reward: float
    best_candidate: Any
    history: List[Dict[str, Any]] = field(default_factory=list)
    num_evaluations: int = 0
    num_function_evals: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class Solver(ABC):
    def __init__(self, device: Optional[str] = None, logger: Optional[ExperimentLogger] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._default_logger = logger
        self._problem = None
        self._kwargs = {}
        self._current_iteration = 0
        self._best_reward = float("-inf")
        self._best_candidate = None
        self._best_protein = None  # For protein modality
        self._history = []
        self._n_eval = 0
        self._nfe = 0
        # Track which scaling checkpoints have been recorded to avoid overwriting earlier values
        # (otherwise checkpoint_{k} would reflect the FINAL best for all k).
        self._logged_scaling_checkpoints: set[int] = set()

    def solve(self, problem: Any, **kwargs: Any) -> SolveResult:
        """Solve the problem."""
        return self._solve_impl(problem, **kwargs)

    def initialize(self, problem: Any, **kwargs: Any) -> None:
        """Initialize the solver for step-based execution."""
        self._problem = problem
        self._kwargs = kwargs
        self._current_iteration = 0
        self._best_reward = float("-inf")
        self._best_candidate = None
        self._best_protein = None  # For protein modality
        self._history = []
        
        # Reset problem evaluation count if possible
        if hasattr(problem, "reset_eval_count"):
            problem.reset_eval_count()
            
        self._initialize_impl(problem, **kwargs)
        
        # Sync evaluation counters from problem after initialization
        if hasattr(problem, "total_samples_evaluated"):
            self._n_eval = problem.total_samples_evaluated
            self._nfe = self._n_eval
        else:
            self._n_eval = 0
            self._nfe = 0
            
        self._logged_scaling_checkpoints = set()
    
    def step(self) -> Dict[str, Any]:
        """Execute one step of the solver and return step results."""
        if self._problem is None:
            raise RuntimeError("Solver not initialized. Call initialize() first.")
        
        # Clear CUDA cache at the start of each iteration to prevent memory accumulation
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        self._current_iteration += 1
        step_result = self._execute_step_impl()
        
        # Update tracking variables
        if step_result.get("best_reward", float("-inf")) > self._best_reward:
            self._best_reward = step_result.get("best_reward", float("-inf"))
            self._best_candidate = step_result.get("best_candidate")
            # Store best protein if available (for protein modality)
            if step_result.get("best_protein") is not None:
                self._best_protein = step_result.get("best_protein")
        
        self._n_eval = step_result.get("n_eval", self._n_eval)
        self._nfe = step_result.get("nfe", self._nfe)
        
        # Add to history
        history_entry = {
            "iteration": self._current_iteration - 1,
            "best": self._best_reward,
            "num_evaluations": self._n_eval,
            "nfe": self._nfe,
        }
        
        # Add checkpoint-specific data for scaling logger
        if hasattr(self._problem, "context") and self._problem.context.get("scaling_mode", False):
            if "scaling_checkpoints" in self._problem.context:
                checkpoints = self._problem.context["scaling_checkpoints"]
                for checkpoint in checkpoints:
                    ckpt_i = int(checkpoint)
                    # Only record the first time the checkpoint is crossed
                    if (ckpt_i not in self._logged_scaling_checkpoints) and (self._n_eval >= ckpt_i):
                        history_entry[f"checkpoint_{checkpoint}"] = self._best_reward
                        self._logged_scaling_checkpoints.add(ckpt_i)
        
        # Add best image if available
        if step_result.get("best_image") is not None:
            history_entry["best_image"] = step_result["best_image"]
        
        # Add best protein if available (for protein modality)
        if step_result.get("best_protein") is not None:
            history_entry["best_protein"] = step_result["best_protein"]
        
        self._history.append(history_entry)
        
        # Log iteration for protein/molecule loggers (step-based path)
        if self._default_logger is not None:
            # Check if logger supports protein iteration logging
            if hasattr(self._default_logger, "log_iteration_console"):
                # Extract rewards from step_result if available
                rewards = None
                if "batch_rewards" in step_result:
                    rewards = step_result["batch_rewards"]
                elif "rewards" in step_result:
                    rewards = step_result["rewards"]
                
                if rewards is not None:
                    context = getattr(self._problem, "context", None)
                    extra = {
                        "n_eval": self._n_eval,
                        "nfe": self._nfe,
                        "batch_size": step_result.get("batch_size", len(rewards) if hasattr(rewards, "__len__") else 1),
                    }
                    self._default_logger.log_iteration_console(
                        iteration=self._current_iteration - 1,
                        rewards=rewards,
                        context=context,
                        extra=extra,
                    )
        
        return step_result
    
    def is_done(self) -> bool:
        """Check if the solver is done (reached max iterations or budget)."""
        if self._problem is None:
            return True
        
        max_iterations = self._kwargs.get("num_iterations", 100)
        oracle_budget = self._kwargs.get("oracle_budget", 0)
        batch_size = self._kwargs.get("batch_size", 16)
        
        # Calculate how many batches were consumed during warm-up.
        # Priority: warmup_batches > warmup_samples
        warmup_batches = self._kwargs.get("warmup_batches", 0)
        warmup_samples = self._kwargs.get("warmup_samples", 0)
        if warmup_batches == 0 and warmup_samples > 0:
            warmup_batches = (warmup_samples + batch_size - 1) // batch_size
            
        # self._current_iteration tracks how many times step() has been called (including warmup).
        # num_iterations is the TOTAL number of iterations (warmup + search).
        # We are done when we've reached the total number of iterations.
        if self._current_iteration >= max_iterations:
            return True
        
        # Fallback hard limit: total evaluations vs oracle budget
        if oracle_budget > 0 and self._n_eval >= oracle_budget:
            return True
        
        return self._is_done_impl()
    
    def get_result(self) -> SolveResult:
        """Get the final result of the solver."""
        if self._problem is None:
            raise RuntimeError("Solver not initialized. Call initialize() first.")
        
        metadata = self._get_metadata_impl()
        metadata["solver"] = self.__class__.__name__.lower().replace("solver", "")
        
        # Add best_protein to metadata if available (for protein modality)
        if hasattr(self, "_best_protein") and self._best_protein is not None:
            metadata["best_protein"] = self._best_protein
        
        return SolveResult(
            best_reward=self._best_reward,
            best_candidate=self._best_candidate,
            history=self._history,
            num_evaluations=self._n_eval,
            num_function_evals=self._nfe,
            metadata=metadata,
        )
    
    @abstractmethod
    def _solve_impl(self, problem: Any, **kwargs: Any) -> SolveResult:
        """Implementation of the solve method (to be overridden by subclasses)."""
        raise NotImplementedError
    
    @abstractmethod
    def _initialize_impl(self, problem: Any, **kwargs: Any) -> None:
        """Initialize the solver implementation."""
        raise NotImplementedError
    
    @abstractmethod
    def _execute_step_impl(self) -> Dict[str, Any]:
        """Execute one step of the solver implementation."""
        raise NotImplementedError
    
    def _is_done_impl(self) -> bool:
        """Check if solver is done (implementation-specific)."""
        return False
    
    def _get_metadata_impl(self) -> Dict[str, Any]:
        """Get solver-specific metadata."""
        return {}
    
    def _print_solver_summary(self, kwargs: Dict[str, Any]) -> None:
        """Print a summary of the solver configuration.
        
        This method can be overridden by specific solvers to provide detailed
        configuration summaries. By default, prints basic solver information.
        
        Args:
            kwargs: Solver configuration parameters
        """
        solver_name = self.__class__.__name__
        # Rich styling (paper color palette)
        try:
            from rich.console import Console
            from rich.rule import Rule
            from ..utils.terminal_colors import CLI_SECTION
            console = Console()
            console.print()
            console.print(Rule(f"[bold {CLI_SECTION}]Solver Summary[/bold {CLI_SECTION}] [{CLI_SECTION}]{solver_name}[/{CLI_SECTION}]", style=CLI_SECTION, characters="─"))
            console.print()
        except ImportError:
            print(f"\n{'='*80}")
            print(f"[Solver Summary] {solver_name}")
            print(f"{'='*80}")
        
        # Basic configuration
        batch_size = kwargs.get("batch_size", "N/A")
        num_iterations = kwargs.get("num_iterations", "N/A")
        oracle_budget = kwargs.get("oracle_budget", kwargs.get("budget", "N/A"))
        
        print(f"Batch size: {batch_size}")
        print(f"Max iterations: {num_iterations}")
        if oracle_budget != "N/A":
            print(f"Oracle budget: {oracle_budget}")
        
        # Detailed configuration if enabled
        if kwargs.get("print_solver_config", False):
            # Styled separator
            try:
                from rich.console import Console
                from rich.rule import Rule
                from ..utils.terminal_colors import CLI_APP
                console = Console()
                console.print(Rule(style=CLI_APP, characters="─"))
            except ImportError:
                print("-" * 40)
            print("Full Configuration:")
            # Sort keys for readability, skip large binary/tensor data if any
            for k, v in sorted(kwargs.items()):
                if k in ("print_benchmark_summary", "print_solver_summary", "print_solver_config"):
                    continue
                # Format nested dicts nicely
                if isinstance(v, dict):
                    print(f"  {k}:")
                    for sub_k, sub_v in sorted(v.items()):
                        print(f"    {sub_k}: {sub_v}")
                else:
                    print(f"  {k}: {v}")
        
        print(f"{'='*80}\n")

    # --- Optional logging helpers ---
    def _log_iteration(self, logger: ExperimentLogger, problem: Any, iteration: int, rewards: torch.Tensor, extra: Optional[Dict[str, Any]] = None) -> None:
        if isinstance(logger, T2IWandbLogger):
            prompt = problem.context.get("prompt", "") if hasattr(problem, "context") else ""
            logger.log_iteration_metrics(
                iteration=iteration,
                prompt=prompt,
                rewards=rewards.detach().cpu(),
                task_index=extra.get("task_index") if extra else None,
                evals_cum=extra.get("evals_cum") if extra else None,
                time_s=extra.get("time_s") if extra else None,
                operator_stats=extra.get("operator_stats") if extra else None,
                topk_images=extra.get("topk_images") if extra else None,
                topk_rewards=extra.get("topk_rewards") if extra else None,
            )
        else:
            # generic scalar logging
            metrics = {
                "iteration_step": iteration,
                "reward_mean": float(rewards.mean().item()),
                "reward_max": float(rewards.max().item()),
                "reward_min": float(rewards.min().item()),
            }
            
            # Add protein diversity and novelty metrics if available
            if hasattr(problem, "context") and problem.context:
                if "diversity" in problem.context:
                    metrics["iteration/diversity"] = float(problem.context["diversity"])
                if "novelty" in problem.context:
                    metrics["iteration/novelty"] = float(problem.context["novelty"])
            
            if logger is not None:
                logger.log(metrics)

    def _log_benchmark(self, logger: ExperimentLogger, problem: Any, solver_name: str, best_reward: float, total_iterations: int, total_evals: int, extra: Optional[Dict[str, Any]] = None) -> None:
        if isinstance(logger, T2IWandbLogger):
            prompt = problem.context.get("prompt", "") if hasattr(problem, "context") else ""
            task_index = extra.get("task_index", 0) if extra else 0
            logger.log_benchmark_summary(
                prompt=prompt,
                solver_name=solver_name,
                task_index=task_index,
                best_reward=best_reward,
                total_iterations=total_iterations,
                total_evals=total_evals,
                best_image=extra.get("best_image") if extra else None,
                reward_curve=extra.get("reward_curve") if extra else None,
                extra=extra,
            )
        else:
            if logger is not None:
                task_index = extra.get("task_index", 0) if extra else 0
                logger.log({
                    "task_step": task_index,
                    "bench/best_reward": best_reward,
                    "bench/iterations": total_iterations,
                    "bench/total_evals": total_evals,
                    "bench/solver": solver_name,
                })

    # Helper to avoid repeating logging boilerplate inside solvers
    def make_iteration_logger(self, problem: Any, logger: Optional[ExperimentLogger]):
        if logger is None:
            def _noop(*args, **kwargs):
                return
            return _noop

        def _iter_log(iteration: int, rewards: torch.Tensor, *, extra: Optional[Dict[str, Any]] = None, console_prefix: Optional[str] = None) -> None:
            self._log_iteration(logger, problem, iteration, rewards, extra)
            # Optional concise console line (no images)
            if console_prefix is not None:
                # Build context string using logger's format_context method
                context_str = ""
                if logger is not None and hasattr(logger, "format_context"):
                    try:
                        context = problem.context if hasattr(problem, "context") else None
                        context_str = logger.format_context(context)
                    except Exception:
                        pass
                
                mean_val = float(rewards.mean().item()) if rewards.numel() > 0 else 0.0
                max_val = float(rewards.max().item()) if rewards.numel() > 0 else 0.0
                
                # Add stability info for molecule problems
                stability_info = ""
                if hasattr(problem, 'last_stability') and problem.last_stability is not None:
                    stability_info = f" stability={problem.last_stability:.3f}"
                
                print(f"[{console_prefix}] iter={iteration} {context_str} mean={mean_val:.4f} max={max_val:.4f}{stability_info}")

        return _iter_log


