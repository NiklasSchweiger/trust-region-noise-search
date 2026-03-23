from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Optional, Union

import torch

from .base import Benchmark
from ..loggers.base import ExperimentLogger
from ..loggers.molecule_logger import MoleculeLogger


class MoleculeBenchmarkBase(Benchmark):
    """Base class for QM9-style molecule benchmarks with step-based run loop."""

    def prompts(self) -> List[str]:
        # Molecule benchmarks drive tasks() directly; no textual prompts needed.
        return []

    def run(
        self,
        algorithm: Any,
        problem_builder: Any,
        *,
        algorithm_kwargs: Optional[Dict[str, Any]] = None,
        logger: Optional[ExperimentLogger] = None,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        algorithm_kwargs = algorithm_kwargs or {}
        running_sum_best = 0.0
        running_sum_stability = 0.0
        running_count = 0
        all_tasks = list(self.iter_tasks())

        # Print summaries
        if algorithm_kwargs.get("print_benchmark_summary", True):
            self._print_benchmark_summary(algorithm, problem_builder, all_tasks, algorithm_kwargs)
        if algorithm_kwargs.get("print_solver_summary", True) and hasattr(algorithm, "_print_solver_summary"):
            algorithm._print_solver_summary(algorithm_kwargs)

        for task_idx, task in enumerate(all_tasks):
            # Task-level prints controlled by benchmark terminal settings
            should_print_benchmark = logger.should_print_benchmark() if logger and hasattr(logger, "should_print_benchmark") else True
            if should_print_benchmark:
                print(f"[Task {task_idx + 1}/{len(all_tasks)}]")
            # Inject scaling checkpoints into task context so solvers can record checkpoint rewards
            context = task.get("context", {}) or {}
            if isinstance(logger, MoleculeLogger) and hasattr(logger, "should_log_scaling_metrics") and logger.should_log_scaling_metrics():
                try:
                    ctx = dict(context)
                    ctx["scaling_mode"] = True
                    if hasattr(logger, "get_scaling_checkpoints"):
                        ctx["scaling_checkpoints"] = list(logger.get_scaling_checkpoints())
                    task["context"] = ctx
                    context = ctx
                except Exception:
                    pass
            # Check for both n_atoms and n_nodes (QM9 uses n_nodes)
            n_atoms = context.get("n_atoms") or context.get("n_nodes", "N/A")
            if should_print_benchmark:
                print(f"  Starting task with n_atoms={n_atoms}...")

            problem = problem_builder.build(task)
            
            # Reset uniqueness tracking for fair comparison between tasks
            if hasattr(problem, "reset_uniqueness_tracking"):
                problem.reset_uniqueness_tracking()

            if torch.cuda.is_available():
                try:
                    torch.cuda.reset_peak_memory_stats()
                except Exception:
                    pass
            t0 = time.perf_counter()

            if all(
                hasattr(algorithm, x) for x in ("initialize", "step", "is_done", "get_result")
            ):
                algorithm.initialize(problem, **algorithm_kwargs, logger=logger)
                step_count = 0
                task_best_reward = float("-inf")
                max_iterations = algorithm_kwargs.get("num_iterations", 15)
                warmup_batches = algorithm_kwargs.get("warmup_batches", 0)
                # Iteration-level prints controlled by iteration terminal settings
                should_print_iteration = logger.should_print_iteration() if logger and hasattr(logger, "should_print_iteration") else False
                if should_print_iteration:
                    print(
                        f"  Solver initialized. Starting optimization iterations... (iters={max_iterations}, batch_size={algorithm_kwargs.get('batch_size', 24)}, warmup_batches={warmup_batches})"
                    )
                while not algorithm.is_done():
                    step_result = algorithm.step()
                    step_count += 1
                    
                    # Simple iteration-level logging: print batch mean and best
                    batch_mean = float(step_result.get("batch_mean", 0.0))
                    batch_max = float(step_result.get("batch_max", 0.0))
                    batch_best_tr_length = step_result.get("batch_best_tr_length", None)
                    if batch_best_tr_length is not None and float(batch_best_tr_length) > 0:
                        print(f"  [Iter {step_count}] batch_mean={batch_mean:.4f}, batch_best={batch_max:.4f}, length={batch_best_tr_length:.4f}")
                    else:
                        print(f"  [Iter {step_count}] batch_mean={batch_mean:.4f}, batch_best={batch_max:.4f}")
                    
                    # iteration_best is the best reward in this specific batch
                    iteration_best = float(step_result.get("batch_max", 0.0))
                    # current_best is the global best for this task
                    current_best = float(step_result.get("best_reward", iteration_best))
                    task_best_reward = max(task_best_reward, current_best)

                    if isinstance(logger, MoleculeLogger) and hasattr(logger, "log_iteration_metrics"):
                        try:
                            if logger.should_log_iteration_metrics():
                                batch_rewards = step_result.get("batch_rewards")
                                if batch_rewards is None:
                                    batch_rewards = torch.tensor(
                                        [
                                            step_result.get("batch_mean", current_best),
                                            step_result.get("batch_max", current_best),
                                        ]
                                    )
                                elif isinstance(batch_rewards, list):
                                    batch_rewards = torch.tensor(batch_rewards)

                                topk_molecules = None
                                topk_rewards_list = None
                                if logger.should_log_iteration_samples():
                                    best_molecule = step_result.get("best_molecule")
                                    if best_molecule is not None:
                                        topk_molecules = [best_molecule]
                                        topk_rewards_list = [current_best]
                                    
                                    # If iteration-specific best is available, use it for samples
                                    iter_best = step_result.get("iter_best_candidate")
                                    if iter_best is not None:
                                        topk_molecules = [iter_best]
                                        topk_rewards_list = [float(step_result.get("batch_max", current_best))]

                                properties = step_result.get("properties")
                                targets = problem.context.get("targets") if hasattr(problem, "context") else None

                                logger.log_iteration_metrics(
                                    iteration=step_count,
                                    rewards=batch_rewards,
                                    task_index=running_count,
                                    evals_cum=step_result.get("n_eval", step_result.get("num_evaluations")),
                                    time_s=step_result.get("time_s"),
                                    operator_stats=step_result.get("operator_stats"),
                                    topk_molecules=topk_molecules,
                                    topk_rewards=topk_rewards_list,
                                    properties=properties,
                                    targets=targets,
                                    stability=step_result.get("stability", getattr(problem, "last_stability_dict", None)),
                                )
                        except Exception as e:
                            print(f"Error logging iteration metrics: {e}")

                    # Iteration-level structure saving (requested for visualization)
                    if logger is not None and getattr(logger, "should_save_outputs", lambda: False)():
                        try:
                            # Use iter_best_candidate from step_result (if available) or best_molecule
                            best_molecule = step_result.get("iter_best_candidate", step_result.get("best_molecule"))
                            
                            # If we have a latent tensor but not a decoded molecule, decode it
                            if best_molecule is not None and isinstance(best_molecule, torch.Tensor):
                                # Check if it's already decoded (molecule problems usually return tensors as coordinates)
                                # Molecule latents are usually flat or (N, D). Decoded are (N, 3) or similar.
                                # If it's a latent tensor from a solver, we should decode it.
                                if hasattr(problem, "decode_latents"):
                                    # Add batch dim if missing
                                    lats = best_molecule.detach().clone()
                                    latent_shape = getattr(problem, "latent_shape", None)
                                    if latent_shape is not None and lats.dim() == len(latent_shape):
                                        lats = lats.unsqueeze(0)
                                    elif lats.dim() == 2:  # Fallback for molecules [N, 3] or [N, D]
                                        lats = lats.unsqueeze(0)
                                    
                                    decoded = problem.decode_latents(lats)
                                    if isinstance(decoded, (list, tuple)):
                                        best_molecule = decoded[0]
                                    elif isinstance(decoded, torch.Tensor):
                                        best_molecule = decoded[0]
                                    else:
                                        best_molecule = decoded

                            if best_molecule is not None and hasattr(logger, "save_best_molecule_pdb"):
                                # Ensure the reward matches the structure being saved
                                # If we are saving the iteration best, we should use the iteration best reward (batch_max)
                                iter_reward = float(step_result.get("batch_max", current_best))
                                logger.save_best_molecule_pdb(
                                    task_index=task_idx, 
                                    best_molecule=best_molecule, 
                                    context={**(problem.context or {}), "iteration": step_count, "reward": iter_reward}
                                )
                        except Exception as e:
                            print(f"Warning: Could not save iteration PDB: {e}")
            search_result = algorithm.get_result()
            
            wall_time_s = float(time.perf_counter() - t0)
            total_evals = int(search_result.num_evaluations)

            running_count += 1
            running_sum_best += float(search_result.best_reward)
            running_avg_best = running_sum_best / max(1, running_count)
            
            task_stability = getattr(problem, "last_stability", 0.0)
            if task_stability is not None:
                running_sum_stability += float(task_stability)
            running_avg_stability = running_sum_stability / max(1, running_count)

            reward_curve = None
            if hasattr(search_result, "history"):
                reward_curve = []
                for h in search_result.history:
                    if isinstance(h, dict):
                        v = h.get("batch_max") if h.get("batch_max") is not None else h.get("best")
                        if v is not None:
                            reward_curve.append(float(v))

            best_molecule = None
            if hasattr(search_result, "metadata") and search_result.metadata:
                best_molecule = search_result.metadata.get("best_molecule")
            if best_molecule is None and hasattr(search_result, "history"):
                for h in reversed(search_result.history):
                    if isinstance(h, dict) and h.get("best_molecule") is not None:
                        best_molecule = h.get("best_molecule")
                        break

            if logger and hasattr(logger, "log_benchmark_summary"):
                try:
                    logger.log_benchmark_summary(
                        context=problem.context if hasattr(problem, "context") else {},
                        solver_name=algorithm.__class__.__name__,
                        task_index=running_count - 1,
                        best_reward=float(search_result.best_reward),
                        total_iterations=len(search_result.history) if hasattr(search_result, "history") else 0,
                        total_evals=total_evals,
                        wall_time_s=wall_time_s,
                        best_molecule=best_molecule,
                        running_best_molecule=getattr(self, "_running_best_molecule", None),
                        reward_curve=reward_curve,
                        extra={
                            "num_tasks": running_count,
                            "running_average_best_reward": running_avg_best,
                            "stability": task_stability,
                            "running_average_stability": running_avg_stability,
                        },
                    )
                except Exception as e:
                    print(f"Error logging benchmark summary: {e}")

            # Scaling logs (budget checkpoints) for this task
            if isinstance(logger, MoleculeLogger) and hasattr(logger, "log_scaling_for_task"):
                try:
                    logger.log_scaling_for_task(
                        task_index=running_count - 1,
                        task_oracles=total_evals,
                        task_history=list(search_result.history) if hasattr(search_result, "history") else None,
                        best_reward=float(search_result.best_reward),
                    )
                except Exception:
                    pass

            # Optional: export best molecule to PDB for later visualization
            if logger is not None and hasattr(logger, "should_save_outputs") and getattr(logger, "should_save_outputs")():
                try:
                    ctx = {**(problem.context or {}), "reward": float(search_result.best_reward)}
                    logger.save_best_molecule_pdb(task_index=running_count - 1, best_molecule=best_molecule, context=ctx)
                except Exception:
                    pass

            # Track running best molecule
            if not hasattr(self, "_running_best_reward"):
                self._running_best_reward = float("-inf")
            if not hasattr(self, "_running_best_molecule"):
                self._running_best_molecule = None
            if float(search_result.best_reward) > self._running_best_reward:
                self._running_best_reward = float(search_result.best_reward)
                if best_molecule is not None:
                    self._running_best_molecule = best_molecule
            elif self._running_best_molecule is None and best_molecule is not None:
                self._running_best_molecule = best_molecule

            results.append(
                {
                    "task": task,
                    "best_reward": search_result.best_reward,
                    "num_evaluations": total_evals,
                    "nfe": search_result.num_function_evals,
                    "wall_time_s": wall_time_s,
                }
            )

            # Print task completion summary with running average (like T2I benchmark)
            should_print_benchmark = logger.should_print_benchmark() if logger and hasattr(logger, "should_print_benchmark") else True
            if should_print_benchmark:
                context_str = ""
                problem_context = problem.context if hasattr(problem, "context") else None
                if logger is not None and hasattr(logger, "format_context"):
                    try:
                        context_str = logger.format_context(problem_context)
                    except Exception:
                        pass
                # Format context string for molecules (n_atoms info)
                if not context_str:
                    # Check for both n_atoms and n_nodes (QM9 uses n_nodes)
                    n_atoms = "N/A"
                    if problem_context is not None:
                        n_atoms = problem_context.get("n_atoms") or problem_context.get("n_nodes", "N/A")
                    elif hasattr(task, "get") and task.get("context"):
                        n_atoms = task["context"].get("n_atoms") or task["context"].get("n_nodes", "N/A")
                    context_str = f"n_atoms={n_atoms}"
                
                # Get GPU memory if available
                gpu_mem_mib = None
                if torch.cuda.is_available():
                    try:
                        gpu_mem_max_alloc = torch.cuda.max_memory_allocated()
                        gpu_mem_mib = gpu_mem_max_alloc / (1024 * 1024)
                    except Exception:
                        pass
                
                if gpu_mem_mib is not None:
                    print(f"[Molecule] {context_str} | solver={algorithm.__class__.__name__} | best={search_result.best_reward:.4f} | avg={running_avg_best:.4f} | evals={total_evals} | gpu_mem_MiB={gpu_mem_mib:.1f} | time_s={wall_time_s:.2f}")
                else:
                    print(f"[Molecule] {context_str} | solver={algorithm.__class__.__name__} | best={search_result.best_reward:.4f} | avg={running_avg_best:.4f} | evals={total_evals} | time_s={wall_time_s:.2f}")

            # Cleanup between tasks
            if hasattr(problem_builder, "reward") and hasattr(problem_builder.reward, "cleanup"):
                try:
                    problem_builder.reward.cleanup()
                except Exception:
                    pass
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                except Exception:
                    pass
            del problem

        if running_count > 0 and logger is not None:
            try:
                logger.log(
                    {
                        "benchmark/final_average_best_reward": float(running_sum_best / running_count),
                        "benchmark/final_average_stability": float(running_sum_stability / running_count),
                        "benchmark/num_tasks": int(running_count),
                    }
                )
            except Exception:
                pass

        return results

    def _print_benchmark_summary(
        self,
        algorithm: Any,
        problem_builder: Any,
        tasks: List[Dict[str, Any]],
        algorithm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        algorithm_kwargs = algorithm_kwargs or {}
        solver_name = algorithm.__class__.__name__
        num_tasks = len(tasks)

        reward_name = "N/A"
        try:
            if hasattr(problem_builder, "reward"):
                reward = problem_builder.reward
                if hasattr(reward, "__class__"):
                    reward_name = reward.__class__.__name__
        except Exception:
            pass

        print(f"\n{'='*80}")
        print(f"[Benchmark Summary] {self.__class__.__name__}")
        print(f"{'='*80}")
        print("Modality: molecule (QM9)")
        print(f"Solver: {solver_name}")
        print(f"Reward function: {reward_name}")
        print(f"Number of tasks: {num_tasks}")
        if algorithm_kwargs:
            oracle_budget = algorithm_kwargs.get("oracle_budget", algorithm_kwargs.get("budget", "N/A"))
            batch_size = algorithm_kwargs.get("batch_size", "N/A")
            num_iterations = algorithm_kwargs.get("num_iterations", "N/A")
            print(f"Oracle budget: {oracle_budget}")
            print(f"Batch size: {batch_size}")
            print(f"Max iterations: {num_iterations}")
        print(f"{'='*80}\n")


class QM9PropertyTargetsBenchmark(MoleculeBenchmarkBase):
    """Benchmark for QM9 property optimization.

    Creates tasks for optimizing QM9 molecular properties. The task sequence is
    deterministic based on the seed, ensuring fair comparison between algorithms.
    All algorithms will see the same sequence of target values and atom counts.
    """

    def __init__(
        self,
        properties: List[str],
        *,
        num_runs: int = 1000,
        start_index: int = 0,
        end_index: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        super().__init__(num_runs=num_runs, randomize=False)
        self.properties = list(properties)
        self.start_index = start_index
        self.end_index = end_index
        self.seed = int(seed)
        self.rng = random.Random(self.seed)

    def prompts(self) -> List[str]:
        return []

    def _sample_n_nodes(self, rng: random.Random) -> int:
        """Sample n_nodes from the QM9 training atom-count histogram."""
        from ..models.qm9.consts import qm9_with_h
        items = list(qm9_with_h['n_nodes'].items())
        nodes, counts = zip(*items)
        total = float(sum(counts))
        r = rng.random() * total
        acc = 0.0
        for n, c in items:
            acc += float(c)
            if r <= acc:
                return int(n)
        return int(nodes[-1])

    def tasks(self) -> List[Dict[str, Any]]:
        """Generate deterministic tasks using a seeded RNG."""
        from ..models.qm9.consts import qm9_stats
        tasks: List[Dict[str, Any]] = []
        actual_num_runs = self.num_runs
        if self.end_index is not None:
            actual_num_runs = min(actual_num_runs, self.end_index - self.start_index)
        task_rng = random.Random(self.seed)
        for _ in range(actual_num_runs):
            n_nodes = self._sample_n_nodes(task_rng)
            targets_dict = {}
            for p in self.properties:
                mu, mad = qm9_stats.get(p, (0.0, 1.0))
                val = task_rng.gauss(mu, mad)
                targets_dict[p] = float((val - mu) / (mad if mad != 0 else 1.0))
            tasks.append({
                "modality": "molecule",
                "context": {
                    "properties": list(self.properties),
                    "targets": targets_dict,
                    "n_nodes": int(n_nodes),
                },
            })
        return tasks


class QM9ScalarPropertyBenchmark(MoleculeBenchmarkBase):
    """Benchmark for QM9 scalar property optimization (QED, SA, molecule_combined).

    Creates tasks with n_nodes sampled from the QM9 distribution. No property targets
    are needed — used with reward functions like qed, sa, or molecule_combined.
    """

    def __init__(
        self,
        *,
        num_runs: int = 100,
        start_index: int = 0,
        end_index: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        super().__init__(num_runs=num_runs, randomize=False)
        self.start_index = start_index
        self.end_index = end_index
        self.seed = int(seed)
        self.rng = random.Random(self.seed)

    def prompts(self) -> List[str]:
        return []

    def _sample_n_nodes(self, rng: random.Random) -> int:
        from ..models.qm9.consts import qm9_with_h
        items = list(qm9_with_h['n_nodes'].items())
        nodes, counts = zip(*items)
        total = float(sum(counts))
        r = rng.random() * total
        acc = 0.0
        for n, c in items:
            acc += float(c)
            if r <= acc:
                return int(n)
        return int(nodes[-1])

    def tasks(self) -> List[Dict[str, Any]]:
        tasks: List[Dict[str, Any]] = []
        actual_num_runs = self.num_runs
        if self.end_index is not None:
            actual_num_runs = min(actual_num_runs, self.end_index - self.start_index)
        task_rng = random.Random(self.seed)
        for task_idx in range(actual_num_runs):
            n_nodes = self._sample_n_nodes(task_rng)
            tasks.append({
                "modality": "molecule",
                "context": {
                    "n_nodes": int(n_nodes),
                    "task_idx": task_idx,
                },
            })
        return tasks


class QM9CustomTargetBenchmark(MoleculeBenchmarkBase):
    """Benchmark for QM9 property optimization with custom target values.

    Allows specifying exact target values for one or more properties.
    Properties not in ``custom_targets`` but listed in ``properties`` will be
    sampled from the dataset distribution.
    """

    def __init__(
        self,
        custom_targets: Dict[str, float],
        properties: Optional[List[str]] = None,
        custom_n_nodes: Optional[int] = None,
        *,
        num_runs: int = 1,
        seed: int = 42,
    ) -> None:
        super().__init__(num_runs=num_runs, randomize=False)
        self.custom_targets = dict(custom_targets)
        if properties is None:
            self.properties = list(self.custom_targets.keys())
        else:
            self.properties = [p for p in properties if p in self.custom_targets]
            if not self.properties:
                self.properties = list(self.custom_targets.keys())
        self.custom_n_nodes = custom_n_nodes
        self.seed = int(seed)
        self.rng = random.Random(self.seed)

    def tasks(self) -> List[Dict[str, Any]]:
        from ..models.qm9.consts import qm9_stats, qm9_with_h
        tasks: List[Dict[str, Any]] = []
        task_rng = random.Random(self.seed)
        for task_idx in range(self.num_runs):
            if self.custom_n_nodes is not None:
                n_nodes = self.custom_n_nodes
            else:
                items = list(qm9_with_h['n_nodes'].items())
                nodes, counts = zip(*items)
                total = float(sum(counts))
                r = task_rng.random() * total
                acc = 0.0
                n_nodes = nodes[-1]
                for n, c in items:
                    acc += float(c)
                    if r <= acc:
                        n_nodes = n
                        break
            targets_dict = {}
            for p in self.properties:
                mu, mad = qm9_stats.get(p, (0.0, 1.0))
                if p in self.custom_targets:
                    val = float(self.custom_targets[p])
                else:
                    val = task_rng.gauss(mu, mad)
                targets_dict[p] = float((val - mu) / (mad if mad != 0 else 1.0))
            tasks.append({
                "modality": "molecule",
                "context": {
                    "properties": list(self.properties),
                    "targets": targets_dict,
                    "n_nodes": int(n_nodes),
                    "task_idx": task_idx,
                },
            })
        return tasks


class PropertyTargetsBenchmark(MoleculeBenchmarkBase):
    """Benchmark that emits molecule tasks with explicit property targets.

    Args:
        properties: List of property names (e.g., ``["qed"]``, ``["qed", "sa"]``).
        targets: Target values aligned to properties, or a dict ``{prop: target}``.
        num_runs: Number of repeated runs; identical targets are emitted each time.
        weights: Optional per-property weights for multi-property aggregation.
    """

    def __init__(
        self,
        properties: List[str],
        targets: Union[List[float], Dict[str, float]],
        num_runs: int = 1,
        weights: Optional[List[float]] = None,
    ):
        super().__init__(num_runs=num_runs, randomize=False)
        self.properties = list(properties)
        if isinstance(targets, dict):
            self.targets = [float(targets[p]) for p in self.properties]
        else:
            self.targets = [float(v) for v in targets]
        if len(self.targets) != len(self.properties):
            raise ValueError("targets must have the same length as properties")
        self.weights = [float(w) for w in weights] if weights is not None else None

    def prompts(self) -> List[str]:
        raise NotImplementedError

    def tasks(self) -> List[Dict[str, Any]]:
        tasks: List[Dict[str, Any]] = []
        for _ in range(self.num_runs):
            tasks.append({
                "modality": "molecule",
                "context": {
                    "properties": list(self.properties),
                    "targets": list(self.targets),
                    **({"weights": list(self.weights)} if self.weights is not None else {}),
                },
            })
        return tasks


__all__ = [
    "MoleculeBenchmarkBase",
    "QM9PropertyTargetsBenchmark",
    "QM9ScalarPropertyBenchmark",
    "QM9CustomTargetBenchmark",
    "PropertyTargetsBenchmark",
]
