"""Protein-specific benchmarks."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
import random
import os

import torch

from .molecule import MoleculeBenchmarkBase
from ..loggers.base import ExperimentLogger
from ..loggers.molecule_logger import MoleculeLogger


class ProteinBenchmarkBase(MoleculeBenchmarkBase):
    """Base class for protein-specific benchmarks.
    
    Extends MoleculeBenchmarkBase with protein-specific summary printing.
    """

    def run(
        self,
        algorithm: Any,
        problem_builder: Any,
        *,
        algorithm_kwargs: Optional[Dict[str, Any]] = None,
        logger: Optional[ExperimentLogger] = None,
    ) -> List[Dict[str, Any]]:
        """Override run() to use n_residues instead of n_atoms for protein tasks."""
        results: List[Dict[str, Any]] = []
        algorithm_kwargs = algorithm_kwargs or {}
        running_sum_best = 0.0
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
            # Add task_index to context so solvers can use it for task-specific seeding
            context["task_index"] = task_idx
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
            else:
                task["context"] = context
            # Use n_residues for proteins instead of n_atoms
            n_residues = context.get("n_residues", "N/A")
            if should_print_benchmark:
                print(f"  Starting task with n_residues={n_residues}...")

            problem = problem_builder.build(task)

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
                    if should_print_iteration:
                        print(f"  Running step {step_count + 1}...", end="", flush=True)
                    step_result = algorithm.step()
                    step_count += 1
                    
                    # iteration_best is the best reward in this specific batch
                    iteration_best = float(step_result.get("batch_max", 0.0))
                    # current_best is the global best for this task
                    current_best = float(step_result.get("best_reward", iteration_best))
                    task_best_reward = max(task_best_reward, current_best)

                    if should_print_iteration:
                        n_eval = step_result.get("n_eval", step_result.get("num_evaluations", "?"))
                        if step_count <= 3 or step_count % 5 == 0:
                            print(f"  Step {step_count}/{max_iterations} complete: iter_best={iteration_best:.4f}, global_best={current_best:.4f}, n_eval={n_eval}")
                        else:
                            print(f"  Step {step_count}/{max_iterations}: iter_best={iteration_best:.4f}")

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
                                )
                        except Exception as e:
                            print(f"Error logging iteration metrics: {e}")

                    if logger is not None and getattr(logger, "should_save_outputs", lambda: False)():
                        # Use iter_best_candidate from step_result (if available) or best_molecule/best_protein
                        best_molecule = step_result.get("iter_best_candidate")
                        if best_molecule is None:
                            best_molecule = step_result.get("best_protein")
                        if best_molecule is None:
                            best_molecule = step_result.get("best_molecule")
                        
                        # If we have a latent tensor but not a decoded protein, decode it
                        if best_molecule is not None and isinstance(best_molecule, torch.Tensor):
                            if hasattr(problem, "decode_latents"):
                                # Add batch dim if missing
                                lats = best_molecule.detach().clone()
                                latent_shape = getattr(problem, "latent_shape", None)
                                if latent_shape is not None and lats.dim() == len(latent_shape):
                                    lats = lats.unsqueeze(0)
                                elif lats.dim() == 2:  # Fallback for proteins [N, 3]
                                    lats = lats.unsqueeze(0)
                                
                                decoded = problem.decode_latents(lats)
                                if isinstance(decoded, (list, tuple)):
                                    best_molecule = decoded[0]
                                elif isinstance(decoded, torch.Tensor):
                                    best_molecule = decoded[0]
                                else:
                                    best_molecule = decoded

                        if best_molecule is not None and hasattr(logger, "save_best_protein_pdb"):
                            # Ensure the reward matches the structure being saved
                            iter_reward = float(step_result.get("batch_max", current_best))
                            logger.save_best_protein_pdb(
                                task_index=task_idx, 
                                best_protein=best_molecule, 
                                context={**(problem.context or {}), "iteration": step_count, "reward": iter_reward}
                            )
            search_result = algorithm.get_result()

            wall_time_s = float(time.perf_counter() - t0)
            total_evals = int(search_result.num_evaluations)
            total_reward_calls = search_result.metadata.get("num_reward_calls", total_evals)
            
            # Extract GPU memory stats
            gpu_mem_max_alloc = None
            gpu_mem_max_reserved = None
            if torch.cuda.is_available():
                try:
                    gpu_mem_max_alloc = int(torch.cuda.max_memory_allocated())
                    gpu_mem_max_reserved = int(torch.cuda.max_memory_reserved())
                except Exception:
                    pass

            running_count += 1
            running_sum_best += float(search_result.best_reward)
            running_avg_best = running_sum_best / max(1, running_count)

            reward_curve = None
            if hasattr(search_result, "history"):
                reward_curve = []
                for h in search_result.history:
                    if isinstance(h, dict):
                        v = h.get("batch_max") if h.get("batch_max") is not None else h.get("best")
                        if v is not None:
                            reward_curve.append(float(v))

            best_molecule = None
            is_already_decoded = False  # Track if we already have a decoded structure
            
            # First, check metadata for already-decoded structures (most efficient)
            if hasattr(search_result, "metadata") and search_result.metadata:
                # Prefer already-decoded structures from solvers
                best_molecule = search_result.metadata.get("best_protein")
                if best_molecule is not None:
                    is_already_decoded = True
                if best_molecule is None:
                    best_molecule = search_result.metadata.get("best_molecule")
                    if best_molecule is not None:
                        is_already_decoded = True
                # Only use best_candidate if no decoded structure is available
                if best_molecule is None:
                    best_molecule = search_result.metadata.get("best_candidate")
            
            # Fallback to direct attribute
            if best_molecule is None and hasattr(search_result, "best_protein"):
                best_molecule = search_result.best_protein
                is_already_decoded = True
            elif best_molecule is None and hasattr(search_result, "best_molecule"):
                best_molecule = search_result.best_molecule
                is_already_decoded = True
            elif best_molecule is None and hasattr(search_result, "best_candidate"):
                best_molecule = search_result.best_candidate

            if best_molecule is None and hasattr(search_result, "history"):
                for h in reversed(search_result.history):
                    if isinstance(h, dict):
                        best_molecule = h.get("best_molecule")
                        if best_molecule is None:
                            best_molecule = h.get("best_candidate")
                        if best_molecule is not None:
                            break

            # CRITICAL: Only decode if we have a latent tensor AND it's not already decoded
            # Solvers (random_search, trs, zero_order) already decode and store best_protein in metadata
            # We should NOT re-decode if best_protein is already available (avoids redundant computation)
            if best_molecule is not None and isinstance(best_molecule, torch.Tensor) and not is_already_decoded:
                # This is a latent tensor that needs decoding
                if hasattr(problem, "decode_latents"):
                    try:
                        lats = best_molecule.detach().clone()
                        latent_shape = getattr(problem, "latent_shape", None)
                        # Add batch dim if missing
                        if latent_shape is not None and lats.dim() == len(latent_shape):
                            lats = lats.unsqueeze(0)
                        elif lats.dim() == 2:  # Fallback for proteins [N, 3]
                            lats = lats.unsqueeze(0)
                        
                        decoded = problem.decode_latents(lats)
                        if isinstance(decoded, (list, tuple)):
                            best_molecule = decoded[0]
                        elif isinstance(decoded, torch.Tensor):
                            best_molecule = decoded[0] if decoded.dim() >= 3 else decoded
                        else:
                            best_molecule = decoded
                    except Exception as e:
                        print(f"Warning: Could not decode best_molecule latent: {e}")
                        best_molecule = None

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
                        running_average_best_reward=running_avg_best,
                        gpu_mem_max_alloc_bytes=gpu_mem_max_alloc,
                        gpu_mem_max_reserved_bytes=gpu_mem_max_reserved,
                        extra={
                            "num_tasks": running_count,
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

            # Optional: export best protein to PDB (ProteinLogger handles atom37 properly)
            if logger is not None and hasattr(logger, "should_save_outputs") and getattr(logger, "should_save_outputs")():
                try:
                    # ProteinLogger maps best_molecule -> best_protein, so passing best_molecule is OK
                    if hasattr(logger, "save_best_protein_pdb"):
                        ctx = {**(problem.context or {}), "reward": float(search_result.best_reward)}
                        logger.save_best_protein_pdb(task_index=running_count - 1, best_protein=best_molecule, context=ctx)
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
                    "num_reward_calls": total_reward_calls,
                    "nfe": search_result.num_function_evals,
                    "wall_time_s": wall_time_s,
                }
            )
            
            reward_calls_str = f" | reward_calls={total_reward_calls}" if total_reward_calls != total_evals else ""
            n_residues = problem.context.get("n_residues", "?") if hasattr(problem, "context") and problem.context else "?"
            mem_mib = (gpu_mem_max_alloc / (1024 * 1024)) if gpu_mem_max_alloc is not None else None
            if mem_mib is not None:
                print(f"[Protein] n_res={n_residues} | task={running_count - 1} | solver={algorithm.__class__.__name__} | best={search_result.best_reward:.4f} | avg={running_avg_best:.4f} | evals={total_evals}{reward_calls_str} | gpu_mem_MiB={mem_mib:.1f} | time_s={wall_time_s:.2f}")
            else:
                print(f"[Protein] n_res={n_residues} | task={running_count - 1} | solver={algorithm.__class__.__name__} | best={search_result.best_reward:.4f} | avg={running_avg_best:.4f} | evals={total_evals}{reward_calls_str} | time_s={wall_time_s:.2f}")

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
        """Print protein-specific benchmark summary.
        
        Args:
            algorithm: The solver algorithm
            problem_builder: Problem builder (to extract reward function info)
            tasks: List of protein tasks
            algorithm_kwargs: Algorithm configuration
        """
        algorithm_kwargs = algorithm_kwargs or {}
        solver_name = algorithm.__class__.__name__
        num_tasks = len(tasks)
        
        # Try to extract reward function name from problem builder
        reward_name = "N/A"
        try:
            if hasattr(problem_builder, "reward"):
                reward = problem_builder.reward
                if hasattr(reward, "__class__"):
                    reward_name = reward.__class__.__name__
        except Exception:
            pass
        
        print(f"\n{'='*80}")
        print(f"[Protein Benchmark Summary] {self.__class__.__name__}")
        print(f"{'='*80}")
        print(f"Solver: {solver_name}")
        print(f"Reward function: {reward_name}")
        print(f"Number of tasks: {num_tasks}")
        
        # Extract task details
        if num_tasks > 0:
            first_task = tasks[0]
            context = first_task.get("context", {})
            
            # Check if motif scaffolding
            if "motif_pdb_path" in context and "contig" in context:
                print(f"Mode: Motif Scaffolding (inpainting)")
                motif_tasks = []
                for task in tasks:
                    ctx = task.get("context", {})
                    motif_path = ctx.get("motif_pdb_path", "N/A")
                    contig = ctx.get("contig", "N/A")
                    n_res = ctx.get("n_residues", "N/A")
                    motif_tasks.append((motif_path, contig, n_res))
                
                # Show unique motif tasks
                unique_motifs = sorted(set((p, c) for p, c, _ in motif_tasks))
                print(f"Number of unique motif tasks: {len(unique_motifs)}")
                if len(unique_motifs) <= 5:
                    for p, c in unique_motifs:
                        pdb_name = p.split("/")[-1] if "/" in p else p
                        print(f"  - {pdb_name}: {c}")
                else:
                    print(f"  (Showing first 3 of {len(unique_motifs)} unique motifs)")
                    for p, c in unique_motifs[:3]:
                        pdb_name = p.split("/")[-1] if "/" in p else p
                        print(f"  - {pdb_name}: {c}")
                
                # Show residue range
                n_residues_list = [n for _, _, n in motif_tasks]
                if n_residues_list:
                    min_res = min(n for n in n_residues_list if isinstance(n, int))
                    max_res = max(n for n in n_residues_list if isinstance(n, int))
                    print(f"Residue counts: {min_res}-{max_res}")
            # Check if unconditional or conditional
            elif "target_fold" in context or "fold_code" in context:
                print(f"Mode: Conditional (fold-conditioned)")
                fold_codes = []
                n_residues_list = []
                for task in tasks:
                    ctx = task.get("context", {})
                    fold_code = ctx.get("target_fold") or ctx.get("fold_code", "N/A")
                    n_res = ctx.get("n_residues", "N/A")
                    fold_codes.append(fold_code)
                    n_residues_list.append(n_res)
                
                # Show unique fold codes
                unique_folds = sorted(set(fold_codes))
                print(f"Fold codes: {', '.join(unique_folds)}")
                if len(unique_folds) < num_tasks:
                    print(f"  (Total: {num_tasks} tasks across {len(unique_folds)} folds)")
                
                # Show residue range
                if n_residues_list:
                    min_res = min(n for n in n_residues_list if isinstance(n, int))
                    max_res = max(n for n in n_residues_list if isinstance(n, int))
                    print(f"Residue counts: {min_res}-{max_res}")
            else:
                print(f"Mode: Unconditional")
                n_residues_list = [task.get("context", {}).get("n_residues") for task in tasks]
                n_residues_list = [n for n in n_residues_list if n is not None]
                
                if n_residues_list:
                    unique_lengths = sorted(set(n_residues_list))
                    if len(unique_lengths) <= 10:
                        print(f"Residue counts: {', '.join(map(str, unique_lengths))}")
                    else:
                        min_res = min(unique_lengths)
                        max_res = max(unique_lengths)
                        print(f"Residue counts: {min_res}-{max_res} (range)")
                        print(f"  Unique lengths: {len(unique_lengths)}")
        
        # Extract noise dimension information
        noise_dim = None
        try:
            if num_tasks > 0:
                temp_problem = problem_builder.build(tasks[0])
                if hasattr(temp_problem, "latent_shape") and temp_problem.latent_shape:
                    import torch
                    noise_dim = int(torch.tensor(temp_problem.latent_shape).prod().item())
        except Exception:
            pass  # Silently fail if we can't get dimension info

        if algorithm_kwargs:
            oracle_budget = algorithm_kwargs.get("oracle_budget", algorithm_kwargs.get("budget"))
            batch_size = algorithm_kwargs.get("batch_size")
            num_iterations = algorithm_kwargs.get("num_iterations")

            if oracle_budget:
                print(f"Oracle budget: {oracle_budget} evaluations per task")
            if batch_size:
                print(f"Batch size: {batch_size}")
            if num_iterations:
                print(f"Max iterations: {num_iterations}")

        if noise_dim is not None:
            print(f"Noise Dimension D: {noise_dim}")
        
        print(f"{'='*80}\n")


class ProteinLengthBenchmark(ProteinBenchmarkBase):
    """Benchmark that iterates over a fixed list of residue counts.
    
    Supports start_index and end_index for deterministic resumption/parallelization.
    The task sequence is deterministic based on the seed, so you can safely split
    the benchmark across multiple runs by specifying different index ranges.
    
    Supports `repeat` parameter to run multiple tasks per length (e.g., repeat=3 
    means each length gets 3 tasks).
    
    By default, samples 200 lengths uniformly between 50 and 274 (inclusive) using
    a fixed seed for reproducibility.
    """

    DEFAULT_SAMPLE_SEED = 42
    DEFAULT_MIN_LENGTH = 50
    DEFAULT_MAX_LENGTH = 250
    DEFAULT_NUM_SAMPLES = 200

    @classmethod
    def _get_default_lengths(cls) -> List[int]:
        """Generate default lengths by sampling 200 values between 50 and 250."""
        rng = random.Random(cls.DEFAULT_SAMPLE_SEED)
        return [rng.randint(cls.DEFAULT_MIN_LENGTH, cls.DEFAULT_MAX_LENGTH) 
                for _ in range(cls.DEFAULT_NUM_SAMPLES)]

    DEFAULT_LENGTHS = None  # Will be generated on first access

    def __init__(
        self,
        *,
        lengths: Optional[List[int]] = None,
        repeat: int = 1,
        shuffle: bool = False,
        seed: int = 0,
        num_runs: Optional[int] = None,
        start_index: Optional[int] = None,
        end_index: Optional[int] = None,
        # Sampling parameters (used when lengths=None)
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        num_samples: Optional[int] = None,
        sample_seed: Optional[int] = None,
    ) -> None:
        # Generate default lengths if not provided
        if lengths is None:
            # Check if instance-specific sampling parameters are provided
            if min_length is not None or max_length is not None or num_samples is not None or sample_seed is not None:
                # Use instance-specific sampling
                min_len = int(min_length) if min_length is not None else self.DEFAULT_MIN_LENGTH
                max_len = int(max_length) if max_length is not None else self.DEFAULT_MAX_LENGTH
                num_samp = int(num_samples) if num_samples is not None else self.DEFAULT_NUM_SAMPLES
                samp_seed = int(sample_seed) if sample_seed is not None else self.DEFAULT_SAMPLE_SEED
                rng = random.Random(samp_seed)
                default_lengths = [rng.randint(min_len, max_len) for _ in range(num_samp)]
            else:
                # Use class-level default sampling
                if ProteinLengthBenchmark.DEFAULT_LENGTHS is None:
                    ProteinLengthBenchmark.DEFAULT_LENGTHS = ProteinLengthBenchmark._get_default_lengths()
                default_lengths = ProteinLengthBenchmark.DEFAULT_LENGTHS
        else:
            default_lengths = None
        resolved_lengths = [int(x) for x in (lengths or default_lengths)]
        self.repeat = max(1, int(repeat))
        
        # Apply start_index/end_index slicing if provided
        if start_index is not None or end_index is not None:
            start = int(start_index) if start_index is not None else 0
            end = int(end_index) if end_index is not None else len(resolved_lengths)
            resolved_lengths = resolved_lengths[start:end]
        
        # Expand lengths with repeats
        expanded_lengths = []
        for length in resolved_lengths:
            expanded_lengths.extend([length] * self.repeat)
        
        # Determine total runs: use num_runs if provided, otherwise use full expanded list
        total_runs = len(expanded_lengths) if num_runs is None else int(num_runs)
        super().__init__(num_runs=total_runs, randomize=False)
        self.lengths = expanded_lengths
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.start_index = start_index
        self.end_index = end_index

    def tasks(self) -> List[Dict[str, Any]]:
        seq = list(self.lengths)
        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(seq)
        
        # If lengths were not provided and we are using default sampling, 
        # ensure we have enough for num_runs by sampling more if needed
        if not seq and self.num_runs > 0:
            rng = random.Random(self.seed)
            seq = [rng.randint(self.DEFAULT_MIN_LENGTH, self.DEFAULT_MAX_LENGTH) for _ in range(self.num_runs)]
        
        seq = seq[: self.num_runs]

        tasks: List[Dict[str, Any]] = []
        # Use global index offset if start_index was provided (for tracking across parallel runs)
        global_offset = int(self.start_index) if self.start_index is not None else 0
        for local_idx, length in enumerate(seq):
            tasks.append({
                "modality": "protein",
                "context": {
                    "n_residues": int(length),
                    "length_index": global_offset + local_idx,  # Global index across all runs
                    "local_index": local_idx,  # Local index within this run
                },
            })
        return tasks


class ProteinUnconditionalBenchmark(ProteinBenchmarkBase):
    """Flexible unconditional benchmark that sweeps over residue counts."""

    def __init__(
        self,
        *,
        lengths: Optional[List[int]] = None,
        num_runs: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 0,
        start_index: Optional[int] = None,
        end_index: Optional[int] = None,
    ) -> None:
        # Generate default lengths if not provided
        if lengths is None:
            if ProteinLengthBenchmark.DEFAULT_LENGTHS is None:
                ProteinLengthBenchmark.DEFAULT_LENGTHS = ProteinLengthBenchmark._get_default_lengths()
            default_lengths = ProteinLengthBenchmark.DEFAULT_LENGTHS
        else:
            default_lengths = None
        resolved_lengths = [int(x) for x in (lengths or default_lengths)]
        if start_index is not None or end_index is not None:
            start = int(start_index) if start_index is not None else 0
            end = int(end_index) if end_index is not None else len(resolved_lengths)
            resolved_lengths = resolved_lengths[start:end]

        total_runs = int(num_runs) if num_runs is not None else len(resolved_lengths)
        super().__init__(num_runs=total_runs, randomize=False)
        self.lengths = resolved_lengths
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.start_index = start_index

    def tasks(self) -> List[Dict[str, Any]]:
        seq = list(self.lengths)
        if not seq:
            return []
        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(seq)
        if len(seq) < self.num_runs:
            repeats = (self.num_runs + len(seq) - 1) // len(seq)
            seq = (seq * repeats)[: self.num_runs]
        else:
            seq = seq[: self.num_runs]

        global_offset = int(self.start_index) if self.start_index is not None else 0
        tasks: List[Dict[str, Any]] = []
        for local_idx, length in enumerate(seq):
            tasks.append({
                "modality": "protein",
                "context": {
                    "n_residues": int(length),
                    "mode": "unconditional",
                    "length_index": global_offset + local_idx,
                },
            })
        return tasks


class ProteinFoldConditionalBenchmark(ProteinBenchmarkBase):
    """Benchmark for fold-conditioned generation using predefined fold codes.
    
    Supports loading empirical (length, CATH code) distributions from AlphaFold Database
    to sample lengths according to the natural distribution for each CATH code.
    """

    def __init__(
        self,
        *,
        conditions: Optional[List[Dict[str, Any]]] = None,
        repeat: int = 1,
        shuffle: bool = False,
        seed: int = 0,
        len_cath_code_path: Optional[str] = None,
        cath_code_level: str = "C",
        min_length: int = 50,
        max_length: int = 250,
        num_samples_per_cath: Optional[int] = None,
        num_runs: Optional[int] = None,
    ) -> None:
        """Initialize fold-conditional benchmark.
        
        Args:
            conditions: List of condition dicts with 'fold_code' and optionally 'n_residues'.
                       If None, defaults to C-level codes ["1.x.x.x", "2.x.x.x", "3.x.x.x"].
            repeat: Number of times to repeat each condition.
            shuffle: Whether to shuffle the task order.
            seed: Random seed for shuffling and sampling.
            len_cath_code_path: Path to .pth file containing empirical (length, CATH code) distribution.
                               If provided, lengths will be sampled from empirical distribution.
            cath_code_level: CATH hierarchy level ("C", "A", "T", "H"). Used when loading from file.
            min_length: Minimum protein length to sample (default: 50).
            max_length: Maximum protein length to sample (default: 250).
            num_samples_per_cath: Number of samples per CATH code. If None and len_cath_code_path
                                 is provided, will sample from empirical distribution. Otherwise
                                 uses repeat parameter.
            num_runs: Total number of runs (truncates the generated task list).
        """
        # Default to C-level codes if conditions not provided
        if conditions is None:
            conditions = [
                {"fold_code": "1.x.x.x"},  # Mainly alpha
                {"fold_code": "2.x.x.x"},  # Mainly beta
                {"fold_code": "3.x.x.x"},  # Mixed alpha/beta
            ]
        
        if not conditions:
            raise ValueError("ProteinFoldConditionalBenchmark requires at least one condition.")
        
        self.conditions = conditions
        self.repeat = int(repeat)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.len_cath_code_path = len_cath_code_path
        self.cath_code_level = cath_code_level
        self.min_length = int(min_length)
        self.max_length = int(max_length)
        self.num_samples_per_cath = num_samples_per_cath
        
        # Load empirical distribution if path provided
        self.len_cath_distribution = None
        if self.len_cath_code_path is not None and os.path.exists(self.len_cath_code_path):
            self.len_cath_distribution = self._load_cath_distribution()
            # Verify which codes are available vs requested
            requested_codes = [cond.get("fold_code") for cond in self.conditions if "fold_code" in cond]
            available_codes = set(self.len_cath_distribution.keys()) if self.len_cath_distribution else set()
            missing_codes = set(requested_codes) - available_codes
            if missing_codes:
                print(f"Warning: The following CATH codes from conditions are not in the distribution: {sorted(missing_codes)}")
                print(f"  Available codes in distribution: {sorted(list(available_codes))[:10]}{'...' if len(available_codes) > 10 else ''}")
                print(f"  Total codes loaded: {len(available_codes)}")
        
        # Determine total runs and defaults for repeats
        num_conds = len(self.conditions)
        if self.len_cath_distribution is not None:
            # When using empirical distribution, target ~200 samples total
            if self.num_samples_per_cath is None:
                self.num_samples_per_cath = (200 + num_conds - 1) // num_conds
            total_runs = num_conds * self.num_samples_per_cath
        else:
            # Use repeat parameter when not using empirical distribution
            # If repeat is 1 (default), adjust it to target ~200 tasks total
            if self.repeat == 1:
                self.repeat = (200 + num_conds - 1) // num_conds
            total_runs = num_conds * self.repeat
        
        # Use num_runs override if provided
        if num_runs is not None:
            total_runs = int(num_runs)
            
        super().__init__(num_runs=total_runs, randomize=False)

    def _mask_cath_code(self, cath_code: str, level: str) -> str:
        """Mask CATH code to specified level.
        
        Matches the sequential masking approach used in Proteina inference.py:
        - For level "C": mask H → mask T → mask A → "1.x.x.x"
        - For level "A": mask H → mask T → "1.10.x.x"
        - For level "T": mask H → "1.10.8.x"
        - For level "H": no masking → "1.10.8.10"
        
        Args:
            cath_code: CATH code string (e.g., "1.10.8.10")
            level: Level to mask to ("C", "A", "T", "H")
            
        Returns:
            Masked CATH code (e.g., "1.x.x.x" for level "C")
        """
        # Mapping matches Proteina's mask_cath_code_by_level: {"H": 3, "T": 2, "A": 1, "C": 0}
        mapping = {"H": 3, "T": 2, "A": 1, "C": 0}  # Position index to mask
        if level not in mapping:
            raise ValueError(f"Invalid CATH level: {level}. Must be one of C, A, T, H")
        
        parts = cath_code.split(".")
        if len(parts) != 4:
            return cath_code  # Already masked or invalid format
        
        # Sequential masking to match Proteina inference.py behavior
        # Always mask H level first (position 3)
        if level in ("C", "A", "T"):
            parts[3] = "x"  # Mask H level
        
        # Then mask T level (position 2) if needed
        if level in ("C", "A"):
            parts[2] = "x"  # Mask T level
        
        # Finally mask A level (position 1) if needed
        if level == "C":
            parts[1] = "x"  # Mask A level
        
        # Level "H" keeps all positions unchanged
        return ".".join(parts)

    def _load_cath_distribution(self) -> Dict[str, List[int]]:
        """Load and process CATH code distribution from file.
        
        Returns:
            Dictionary mapping CATH codes (at specified level) to lists of lengths.
        """
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required to load CATH code distribution files.")
        
        _len_cath_codes = torch.load(self.len_cath_code_path)
        
        # Group by CATH code (masked to specified level)
        cath_to_lengths: Dict[str, List[int]] = {}
        
        for _len, code in _len_cath_codes:
            # Handle both single codes and lists of codes
            if isinstance(code, list):
                codes = code
            else:
                codes = [code]
            
            # Mask each code to the specified level
            for cath_code in codes:
                if isinstance(cath_code, str):
                    masked_code = self._mask_cath_code(cath_code, self.cath_code_level)
                    
                    # Filter by length range
                    if self.min_length <= _len <= self.max_length:
                        if masked_code not in cath_to_lengths:
                            cath_to_lengths[masked_code] = []
                        cath_to_lengths[masked_code].append(int(_len))
        
        return cath_to_lengths

    def tasks(self) -> List[Dict[str, Any]]:
        """Generate tasks from conditions, optionally sampling lengths from empirical distribution."""
        tasks: List[Dict[str, Any]] = []
        rng = random.Random(self.seed)
        
        for cond_idx, cond in enumerate(self.conditions):
            if "fold_code" not in cond:
                raise ValueError("Each condition must include a 'fold_code'.")
            
            fold_code = cond["fold_code"]
            
            # Determine lengths for this CATH code
            if self.len_cath_distribution is not None:
                # Sample from empirical distribution
                num_samples = self.num_samples_per_cath
                
                if fold_code in self.len_cath_distribution:
                    available_lengths = self.len_cath_distribution[fold_code]
                    if not available_lengths:
                        print(f"Warning: No lengths found for CATH code {fold_code} in range [{self.min_length}, {self.max_length}]")
                        continue
                    
                    # Sample lengths with replacement from empirical distribution
                    sampled_lengths = rng.choices(available_lengths, k=num_samples)
                else:
                    # CATH code not in distribution, use default length or from condition
                    explicit_res = cond.get("n_residues", cond.get("length"))
                    if explicit_res is not None:
                        sampled_lengths = [int(explicit_res)] * num_samples
                    else:
                        # Sample random lengths if not specified
                        sampled_lengths = [rng.randint(self.min_length, self.max_length) for _ in range(num_samples)]
                    print(f"Warning: CATH code {fold_code} not found in distribution. Using {'specified' if explicit_res else 'sampled'} lengths.")
            else:
                # Use explicit length from condition or sample random lengths
                explicit_res = cond.get("n_residues", cond.get("length"))
                if explicit_res is not None:
                    # If explicit length is provided, use it for all repeats
                    sampled_lengths = [int(explicit_res)] * self.repeat
                else:
                    # Sample random lengths between min_length and max_length for each repeat
                    # This ensures each task gets a different random length, similar to unconditional benchmark
                    sampled_lengths = [rng.randint(self.min_length, self.max_length) for _ in range(self.repeat)]
            
            # Create tasks for each sampled length
            for length_idx, length in enumerate(sampled_lengths):
                tasks.append({
                    "modality": "protein",
                    "context": {
                        "n_residues": int(length),
                        "target_fold": fold_code,
                        "fold_metadata": {
                            **cond,
                            "cath_code_level": self.cath_code_level,
                            "sampled_from_distribution": self.len_cath_distribution is not None,
                        },
                        "condition_index": cond_idx,
                        "length_index": length_idx,
                    },
                })
        
        if self.shuffle:
            rng.shuffle(tasks)
        
        return tasks[: self.num_runs]


class ProteinMotifScaffoldBenchmark(ProteinBenchmarkBase):
    """Benchmark for motif-scaffold generation (inpainting).
    
    Generates scaffold structures around fixed motif regions. Each task specifies:
    - A PDB file containing the motif structure
    - A contig string defining which residues are fixed (motif) and scaffold lengths
    
    Contig format: "15/A45-65/20/A20-30"
    - Numbers are scaffold lengths (to be generated)
    - Letters (A, B, etc.) followed by numbers/ranges are motif positions from the PDB
    - Example: "15/A45-65/20" means: scaffold 15 residues, then motif from chain A residues 45-65, then scaffold 20 residues
    """

    def __init__(
        self,
        *,
        motif_tasks: Optional[List[Dict[str, Any]]] = None,
        repeat: int = 1,
        shuffle: bool = False,
        seed: int = 0,
        motif_only: bool = False,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
    ) -> None:
        """Initialize motif-scaffold benchmark.
        
        Args:
            motif_tasks: List of task dicts, each with:
                - "motif_pdb_path": Path to PDB file containing motif structure
                - "contig": Contig string defining motif and scaffold regions
                - Optionally: "motif_task_name", "segment_order", "min_length", "max_length"
            repeat: Number of times to repeat each motif task.
            shuffle: Whether to shuffle the task order.
            seed: Random seed for shuffling and scaffold length sampling.
            motif_only: If True, extract only motif atoms (CA only). If False, extract all atoms.
            min_length: Minimum total protein length (default: None, uses contig-specified lengths).
            max_length: Maximum total protein length (default: None, uses contig-specified lengths).
        """
        if motif_tasks is None or len(motif_tasks) == 0:
            raise ValueError("ProteinMotifScaffoldBenchmark requires at least one motif task.")
        
        # Validate motif tasks
        for task in motif_tasks:
            if "motif_pdb_path" not in task:
                raise ValueError("Each motif task must include 'motif_pdb_path'.")
            if "contig" not in task:
                raise ValueError("Each motif task must include 'contig'.")
            if not os.path.exists(task["motif_pdb_path"]):
                raise FileNotFoundError(f"Motif PDB file not found: {task['motif_pdb_path']}")
        
        self.motif_tasks = motif_tasks
        self.repeat = max(1, int(repeat))
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.motif_only = bool(motif_only)
        self.min_length = int(min_length) if min_length is not None else None
        self.max_length = int(max_length) if max_length is not None else None
        
        total_runs = len(self.motif_tasks) * self.repeat
        super().__init__(num_runs=total_runs, randomize=False)

    def tasks(self) -> List[Dict[str, Any]]:
        """Generate tasks from motif configurations."""
        tasks: List[Dict[str, Any]] = []
        rng = random.Random(self.seed)
        
        # Expand tasks with repeats
        expanded_tasks = []
        for _ in range(self.repeat):
            for task in self.motif_tasks:
                expanded_tasks.append(dict(task))
        
        if self.shuffle:
            rng.shuffle(expanded_tasks)
        
        for task_idx, motif_task in enumerate(expanded_tasks[: self.num_runs]):
            # Extract task-specific parameters
            motif_pdb_path = motif_task["motif_pdb_path"]
            contig = motif_task["contig"]
            motif_task_name = motif_task.get("motif_task_name", f"motif_task_{task_idx}")
            segment_order = motif_task.get("segment_order", "A")
            task_min_length = motif_task.get("min_length", self.min_length)
            task_max_length = motif_task.get("max_length", self.max_length)
            
            tasks.append({
                "modality": "protein",
                "context": {
                    "motif_pdb_path": motif_pdb_path,
                    "contig": contig,
                    "motif_task_name": motif_task_name,
                    "segment_order": segment_order,
                    "motif_only": self.motif_only,
                    "min_length": task_min_length,
                    "max_length": task_max_length,
                    "task_index": task_idx,
                },
            })
        
        return tasks

