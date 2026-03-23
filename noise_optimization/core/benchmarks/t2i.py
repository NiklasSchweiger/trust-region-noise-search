from __future__ import annotations

import os
import json
import random
from typing import List, Dict, Any, Optional

import time
import torch
from .base import Benchmark
from ..loggers.base import ExperimentLogger
from ..loggers.t2i import T2IWandbLogger
from ..loggers.scaling_logger import ScalingLogger


class T2IBenchmarkBase(Benchmark):
    """Base class for T2I benchmarks with a clean step-based run loop."""

    @staticmethod
    def _render_best_image(problem: Any, candidate: Any) -> Optional[Any]:
        """Decode the best latent candidate into an image for logging."""
        if candidate is None or not hasattr(problem, "generative_model"):
            return None
        if not isinstance(candidate, torch.Tensor):
            return None

        try:
            latent_shape = getattr(problem, "latent_shape", None)
            lat = candidate.detach().clone()
            if latent_shape is not None and lat.dim() == len(latent_shape):
                lat = lat.unsqueeze(0)
            elif lat.dim() == 3:
                lat = lat.unsqueeze(0)
            lat = lat.to(getattr(problem, "device", lat.device))

            model_cfg = {}
            try:
                model_cfg = dict(problem.context.get("model_config", {}))
            except Exception:
                model_cfg = {}
            model_cfg["initial_latent_noise"] = lat
            prompt = problem.context.get("prompt", "") if hasattr(problem, "context") else ""

            with torch.no_grad():
                imgs = problem.generative_model.forward(prompt, **model_cfg)  # type: ignore[attr-defined]

            if isinstance(imgs, torch.Tensor):
                return imgs[0] if imgs.dim() >= 4 else imgs
            if isinstance(imgs, list):
                return imgs[0] if len(imgs) > 0 else None
            return imgs
        except Exception as exc:
            print(f"[T2I] Failed to render best image: {exc}")
            return None

    def _print_benchmark_summary(
        self,
        algorithm: Any,
        problem_builder: Any,
        tasks: List[Dict[str, Any]],
        algorithm_kwargs: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
    ) -> None:
        """Print T2I benchmark summary.
        
        Args:
            algorithm: The solver algorithm
            problem_builder: Problem builder (to extract reward function info)
            tasks: List of tasks to run
            algorithm_kwargs: Algorithm configuration
            enabled: Whether to print the summary (controlled by config)
        """
        if not enabled:
            return
            
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
        
        # Extract prompt information
        prompts = []
        for task in tasks[:10]:  # Show first 10 prompts as examples
            prompt = task.get("prompt", task.get("context", {}).get("prompt", ""))
            if prompt:
                prompts.append(prompt)
        
        # Rich styling (paper color palette)
        try:
            from rich.console import Console
            from rich.rule import Rule
            from ..utils.terminal_colors import CLI_SECTION
            console = Console()
            console.print()
            console.print(Rule(f"[bold {CLI_SECTION}]T2I Benchmark Summary[/bold {CLI_SECTION}] [{CLI_SECTION}]{self.__class__.__name__}[/{CLI_SECTION}]", style=CLI_SECTION, characters="─"))
            console.print()
        except ImportError:
            print(f"\n{'='*80}")
            print(f"[T2I Benchmark Summary] {self.__class__.__name__}")
            print(f"{'='*80}")
        print(f"Modality: image")
        print(f"Solver: {solver_name}")
        print(f"Reward function: {reward_name}")
        print(f"Number of tasks: {num_tasks}")
        
        # Show example prompts
        if prompts:
            print(f"Example prompts ({min(len(prompts), 3)} of {num_tasks}):")
            for i, prompt in enumerate(prompts[:3]):
                prompt_preview = prompt[:60] + "..." if len(prompt) > 60 else prompt
                print(f"  {i+1}. {prompt_preview}")
        
        # Algorithm configuration
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

        if noise_dim is not None:
            print(f"Noise dimension: {noise_dim}")
        
        # Styled footer
        try:
            from rich.console import Console
            from rich.rule import Rule
            from ..utils.terminal_colors import CLI_SECTION
            console = Console()
            console.print(Rule(style=CLI_SECTION, characters="─"))
            console.print()
        except ImportError:
            print(f"{'='*80}\n")

    def run(self, algorithm: Any, problem_builder: Any, *, algorithm_kwargs: Optional[Dict[str, Any]] = None, logger: Optional[ExperimentLogger] = None) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        algorithm_kwargs = algorithm_kwargs or {}
        running_sum_best = 0.0
        running_count = 0
        all_tasks = list(self.iter_tasks())
        total_tasks = len(all_tasks)

        # Check if benchmark summary is enabled (from config)
        print_benchmark_summary = algorithm_kwargs.get("print_benchmark_summary", True)
        print_solver_summary = algorithm_kwargs.get("print_solver_summary", True)
        
        # Print benchmark summary
        self._print_benchmark_summary(algorithm, problem_builder, all_tasks, algorithm_kwargs, enabled=print_benchmark_summary)
        
        # Print solver summary
        if print_solver_summary and hasattr(algorithm, "_print_solver_summary"):
            algorithm._print_solver_summary(algorithm_kwargs)

        # Track cumulative oracles across all tasks for scaling logger
        cumulative_oracles = 0
        best_reward_so_far = -float('inf')
        best_image_so_far = None

        # Determine logging preferences (defaults: benchmark-only)
        is_t2i_logger = isinstance(logger, T2IWandbLogger)
        iteration_logging_enabled = bool(is_t2i_logger and logger.should_log_iteration_metrics())
        iteration_samples_enabled = bool(is_t2i_logger and logger.should_log_iteration_samples())
        iteration_sample_limit = logger.get_iteration_sample_limit() if iteration_samples_enabled and is_t2i_logger else 0
        scaling_logger = logger if isinstance(logger, ScalingLogger) else None

        # Per-iteration data-sample logging flag (images for T2I, molecules otherwise)
        log_iteration_images = getattr(self, 'log_iteration_images', False)
        if iteration_samples_enabled:
            log_iteration_images = True

        for task_idx, task in enumerate(all_tasks):
            # Print task start progress
            print(f"[Task {task_idx + 1}/{total_tasks}]")
            
            # Enable scaling mode
            if hasattr(logger, "budget_checkpoints"):
                if "context" not in task or not isinstance(task.get("context"), dict):
                    task["context"] = {}
                task["context"]["scaling_mode"] = True
                task["context"]["scaling_checkpoints"] = logger.budget_checkpoints

            problem = problem_builder.build(task)
            task_history: Dict[int, float] = {}
            task_image_history: Dict[int, Any] = {}

            if torch.cuda.is_available():
                try:
                    torch.cuda.reset_peak_memory_stats()
                except Exception as e:
                    print(f"Error resetting peak memory stats: {e}")
            t0 = time.perf_counter()

            # Allow per-iteration images
            solver_kwargs = dict(algorithm_kwargs)
            if log_iteration_images:
                solver_kwargs['log_iteration_images'] = True

            # Baseline image for visual comparisons
            if log_iteration_images and isinstance(logger, T2IWandbLogger):
                try:
                    prompt = problem.context.get("prompt", "") if hasattr(problem, "context") else ""
                    lat = problem.sample(batch_size=1, latent_shape=getattr(problem, "latent_shape", None))
                    model_cfg = dict(problem.context.get("model_config", {})) if hasattr(problem, "context") else {}
                    model_cfg["initial_latent_noise"] = lat
                    with torch.no_grad():
                        base_imgs = problem.generative_model.forward(prompt, **model_cfg)  # type: ignore[attr-defined]
                        # Evaluate reward for baseline image
                        if hasattr(problem, "evaluate_decoded"):
                            base_reward = problem.evaluate_decoded(base_imgs)
                            if isinstance(base_reward, torch.Tensor):
                                base_reward = base_reward.item()
                        else:
                            base_reward = None
                    
                    img0 = base_imgs[0] if isinstance(base_imgs, torch.Tensor) and base_imgs.dim() >= 4 else base_imgs
                    caption = f"Baseline | Reward: {base_reward:.4f} | {prompt}" if base_reward is not None else f"Baseline | {prompt}"
                    logger.wandb.log({
                        "task_step": running_count,
                        "benchmark/baseline_image": logger.wandb.Image(img0, caption=caption),
                    })
                    
                    # Local saving
                    if hasattr(logger, "save_image_locally"):
                        # save_image_locally handles the extension based on config
                        logger.save_image_locally(img0, "baseline", task_index=running_count, prompt=prompt)
                except Exception as e:
                    print(f"Error logging baseline image: {e}")

            if all(hasattr(algorithm, x) for x in ("initialize", "step", "is_done", "get_result")):
                algorithm.initialize(problem, **solver_kwargs, logger=logger)
                step_count = 0
                task_best_reward = float("-inf")
                task_best_image = None
                task_history = {}
                task_image_history = {}
                
                # Track task-specific oracle count for checkpoint purposes
                # Warmup batches are treated exactly like regular steps for checkpoint tracking
                # If warmup_batches=3 and batch_size=24:
                #   After warmup batch 1: 24 → checkpoint 24
                #   After warmup batch 2: 48 → checkpoint 48
                #   After warmup batch 3: 72 → checkpoint 72
                #   After step 1: 96 → checkpoint 96
                #   After step 2: 120 → checkpoint 120
                #   etc.
                batch_size = solver_kwargs.get("batch_size", 1)
                warmup_batches = solver_kwargs.get("warmup_batches", 0)
                warmup_samples = solver_kwargs.get("warmup_samples", 0)
                
                # Start oracle count at 0
                task_oracle_count = 0
                
                # If we have warmup, we need to create checkpoint entries for each warmup batch
                # Since warmup happens during initialize(), we'll capture it from the first step result
                # We'll backfill warmup checkpoints using the best reward from warmup
                warmup_count = int(warmup_samples) if warmup_samples > 0 else int(warmup_batches * batch_size)
                warmup_captured = False

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
                    
                    # Get the best reward from this step (includes warmup if this is the first step)
                    current_best = float(step_result.get("best_reward", step_result.get("batch_max", 0.0)))
                    if current_best > task_best_reward:
                        task_best_reward = current_best
                        maybe_img = step_result.get("best_image")
                        if maybe_img is not None:
                            task_best_image = maybe_img
                    
                    # If this is the first step and we have warmup, backfill warmup checkpoints
                    # Treat each warmup batch as a step (24, 48, 72, etc. for warmup_batches=3, batch_size=24)
                    if step_count == 1 and warmup_count > 0 and not warmup_captured:
                        # Get warmup-specific reward if available, otherwise use current best
                        warmup_best = float(step_result.get("warmup_best_reward", current_best))
                        if warmup_best > task_best_reward:
                            task_best_reward = warmup_best
                        
                        # Create checkpoint entries for each warmup batch
                        # Each warmup batch is treated as a step, so we have checkpoints at:
                        # batch_size, 2*batch_size, 3*batch_size, ... up to warmup_count
                        if warmup_batches > 0:
                            # Use warmup_batches: create checkpoints at each batch
                            for warmup_step in range(1, warmup_batches + 1):
                                warmup_checkpoint = warmup_step * batch_size
                                task_history[warmup_checkpoint] = task_best_reward
                                if task_best_image is not None:
                                    task_image_history[warmup_checkpoint] = task_best_image
                        else:
                            # Use warmup_samples: create checkpoint at the exact warmup_count
                            # Also create intermediate checkpoints at batch_size intervals
                            for checkpoint in range(batch_size, warmup_count + 1, batch_size):
                                task_history[checkpoint] = task_best_reward
                                if task_best_image is not None:
                                    task_image_history[checkpoint] = task_best_image
                            # Ensure exact warmup_count is included
                            if warmup_count % batch_size != 0:
                                task_history[warmup_count] = task_best_reward
                                if task_best_image is not None:
                                    task_image_history[warmup_count] = task_best_image
                        
                        task_oracle_count = warmup_count
                        warmup_captured = True
                    
                    # Each step evaluates one batch (batch_size evaluations)
                    # Increment our task-specific oracle count for checkpoint tracking
                    task_oracle_count += int(batch_size)
                    step_oracles = task_oracle_count

                    # Store best reward seen so far at this oracle count
                    # This ensures we have entries at the correct checkpoints
                    task_history[step_oracles] = task_best_reward
                    if task_best_image is not None:
                        task_image_history[step_oracles] = task_best_image

                    if is_t2i_logger and iteration_logging_enabled:
                        try:
                            prompt = problem.context.get("prompt", "") if hasattr(problem, "context") else ""
                            topk_images = None
                            topk_rewards = None
                            if log_iteration_images:
                                warmup_best_img = step_result.get("warmup_best_image")
                                warmup_best_reward = step_result.get("warmup_best_reward")
                                warmup_imgs = step_result.get("warmup_images")
                                warmup_rewards = step_result.get("warmup_rewards")
                                iter_best_img = step_result.get("best_image")
                                # If best_image is not provided, try to render from iter_best_candidate (best from this iteration)
                                # Fall back to best_candidate (global best) if iter_best_candidate is not available
                                if iter_best_img is None:
                                    iter_best_candidate = step_result.get("iter_best_candidate")
                                    if iter_best_candidate is not None:
                                        try:
                                            iter_best_img = self._render_best_image(problem, iter_best_candidate)
                                        except Exception as e:
                                            print(f"Warning: Could not render best image from iteration candidate: {e}")
                                            iter_best_img = None
                                    # Fallback to global best if iteration best not available
                                    if iter_best_img is None:
                                        best_candidate = step_result.get("best_candidate")
                                        if best_candidate is not None:
                                            try:
                                                iter_best_img = self._render_best_image(problem, best_candidate)
                                            except Exception as e:
                                                print(f"Warning: Could not render best image from candidate: {e}")
                                                iter_best_img = None
                                if warmup_imgs:
                                    topk_images = list(warmup_imgs)
                                    topk_rewards = [float(x) for x in (warmup_rewards or [])]
                                elif warmup_best_img is not None:
                                    topk_images = [warmup_best_img]
                                    topk_rewards = [float(warmup_best_reward) if warmup_best_reward is not None else 0.0]
                                if iter_best_img is not None:
                                    if topk_images is None:
                                        topk_images = [iter_best_img]
                                        topk_rewards = [float(step_result.get("batch_max", 0.0))]
                                    else:
                                        topk_images.append(iter_best_img)
                                        topk_rewards.append(float(step_result.get("batch_max", 0.0)))
                                if topk_images is not None and iteration_sample_limit > 0:
                                    topk_images = topk_images[:iteration_sample_limit]
                                    if topk_rewards is not None:
                                        topk_rewards = topk_rewards[:iteration_sample_limit]
                            logger.log_iteration_metrics(
                                iteration=step_count,
                                prompt=prompt,
                                rewards=torch.tensor([step_result.get("batch_mean", 0.0), step_result.get("batch_max", 0.0)]),
                                task_index=running_count - 1,
                                evals_cum=step_oracles,
                                topk_images=topk_images,
                                topk_rewards=topk_rewards,
                            )
                        except Exception as e:
                            print(f"Error logging iteration metrics: {e}")

                search_result = algorithm.get_result()

            wall_time_s = float(time.perf_counter() - t0)
            gpu_mem_max_alloc = None
            gpu_mem_max_reserved = None
            if torch.cuda.is_available():
                try:
                    gpu_mem_max_alloc = int(torch.cuda.max_memory_allocated())
                    gpu_mem_max_reserved = int(torch.cuda.max_memory_reserved())
                except Exception:
                    pass

            total_evals = int(search_result.num_evaluations)
            # Get total reward calls from metadata if available (for OC-Flow and other solvers that track it)
            total_reward_calls = search_result.metadata.get("num_reward_calls", None)
            cumulative_oracles += total_evals
            running_count += 1
            running_sum_best += float(search_result.best_reward)
            running_avg_best = running_sum_best / max(1, running_count)

            # Console summary
            context_str = ""
            if logger is not None and hasattr(logger, "format_context"):
                try:
                    context = problem.context if hasattr(problem, "context") else None
                    context_str = logger.format_context(context)
                except Exception as e:
                    print(f"Error formatting context: {e}")
                    pass
            mem_mib = (gpu_mem_max_alloc / (1024 * 1024)) if gpu_mem_max_alloc is not None else None
            # Show reward_calls if available in metadata
            reward_calls_str = f" | reward_calls={total_reward_calls}" if total_reward_calls is not None else ""
            if mem_mib is not None:
                print(f"[T2I] {context_str} | solver={algorithm.__class__.__name__} | best={search_result.best_reward:.4f} | avg={running_avg_best:.4f} | evals={total_evals}{reward_calls_str} | gpu_mem_MiB={mem_mib:.1f} | time_s={wall_time_s:.2f}")
            else:
                print(f"[T2I] {context_str} | solver={algorithm.__class__.__name__} | best={search_result.best_reward:.4f} | avg={running_avg_best:.4f} | evals={total_evals}{reward_calls_str} | time_s={wall_time_s:.2f}")

            # Derive best image and reward curve for logging/scaling
            metadata = getattr(search_result, "metadata", {}) or {}
            # Check for best_decoded (generic term used by SMC solvers) or best_image (legacy)
            best_image = metadata.get("best_decoded") or metadata.get("best_image")
            reward_curve = None
            try:
                if best_image is None and hasattr(search_result, "history"):
                    for h in reversed(search_result.history):
                        if isinstance(h, dict):
                            # Check both best_decoded and best_image for backward compatibility
                            best_image = h.get("best_decoded") or h.get("best_image")
                            if best_image is not None:
                                break
                if best_image is None:
                    best_image = self._render_best_image(problem, getattr(search_result, "best_candidate", None))
                if hasattr(search_result, "history"):
                    reward_curve = []
                    for h in search_result.history:
                        if isinstance(h, dict):
                            v = h.get("batch_max") if h.get("batch_max") is not None else h.get("best")
                            if v is not None:
                                reward_curve.append(v)
            except Exception as e:
                print(f"Error preparing benchmark summary artifacts: {e}")

            # Track cross-task best image for scaling visuals
            # Convert search_result.best_reward to float safely
            current_best_reward = search_result.best_reward
            if isinstance(current_best_reward, torch.Tensor):
                current_best_reward = float(current_best_reward.item() if current_best_reward.numel() == 1 else current_best_reward.max().item())
            else:
                current_best_reward = float(current_best_reward)
            
            if current_best_reward > best_reward_so_far:
                best_reward_so_far = current_best_reward
                if best_image is not None:
                    best_image_so_far = best_image
            elif best_image_so_far is None and best_image is not None:
                best_image_so_far = best_image

            # Fill in task history for scaling checkpoints if solver didn't expose per-iteration data
            if not task_history and hasattr(search_result, "history"):
                for entry in search_result.history:
                    if not isinstance(entry, dict):
                        continue
                    oc = entry.get("num_evaluations")
                    best_val = entry.get("best")
                    if oc is None or best_val is None:
                        continue
                    # Convert tensor to float safely
                    if isinstance(best_val, torch.Tensor):
                        best_val = float(best_val.item() if best_val.numel() == 1 else best_val.max().item())
                    else:
                        best_val = float(best_val)
                    task_history[int(oc)] = best_val
                    # Check both best_decoded (generic) and best_image (legacy) for backward compatibility
                    best_img = entry.get("best_decoded") or entry.get("best_image")
                    if best_img is not None:
                        task_image_history[int(oc)] = best_img

            # Ensure final oracle count is represented for checkpoint interpolation
            # Convert search_result.best_reward to float safely
            best_reward_val = search_result.best_reward
            if isinstance(best_reward_val, torch.Tensor):
                best_reward_val = float(best_reward_val.item() if best_reward_val.numel() == 1 else best_reward_val.max().item())
            else:
                best_reward_val = float(best_reward_val)
            task_history[int(total_evals)] = best_reward_val
            if best_image is not None:
                task_image_history[int(total_evals)] = best_image

            # Optional summary logging for T2I
            if is_t2i_logger and logger.should_log_benchmark_metrics():
                try:
                    logger.log_benchmark_summary(
                        prompt=task.get("context", {}).get("prompt", ""),
                        solver_name=algorithm.__class__.__name__,
                        task_index=running_count - 1,
                        best_reward=search_result.best_reward,
                        total_iterations=len(search_result.history) if hasattr(search_result, "history") else 0,
                        total_evals=total_evals,
                        wall_time_s=wall_time_s,
                        best_image=best_image,
                        running_best_image=best_image_so_far,
                        reward_curve=reward_curve,
                        extra={
                            "gpu_mem_max_alloc_bytes": gpu_mem_max_alloc,
                            "gpu_mem_max_reserved_bytes": gpu_mem_max_reserved,
                            "running_average_best_reward": running_avg_best,
                            "num_tasks": running_count,
                        },
                    )
                    
                    # Local saving of best image
                    if best_image is not None and hasattr(logger, "save_image_locally"):
                        prompt = task.get("context", {}).get("prompt", "")
                        # save_image_locally handles the extension based on config
                        logger.save_image_locally(best_image, "best", task_index=running_count - 1, prompt=prompt)
                except Exception as e:
                    print(f"Error logging benchmark summary: {e}")

            if scaling_logger is not None:
                try:
                    scaling_logger.log_with_checkpoints(
                        iteration=running_count,
                        cumulative_oracles=cumulative_oracles,
                        best_reward=best_reward_so_far,
                        current_task_best_image=best_image,
                        current_task_best_reward=float(search_result.best_reward),
                        prompt=task.get("context", {}).get("prompt", ""),
                        task_history=task_history,
                        task_oracles=total_evals,
                        task_image_history=task_image_history,
                        problem=problem,
                    )
                except Exception as e:
                    import traceback
                    print(f"[ScalingLogger] Failed to log checkpoints: {e}")
                    traceback.print_exc()

            results.append({
                "task": task,
                "best_reward": search_result.best_reward,
                "num_evaluations": total_evals,
                "nfe": search_result.num_function_evals,
                "wall_time_s": wall_time_s,
                "gpu_mem_max_alloc_bytes": gpu_mem_max_alloc,
                "gpu_mem_max_reserved_bytes": gpu_mem_max_reserved,
            })

        # Final aggregate
        if running_count > 0 and logger is not None:
            try:
                logger.log({
                    "benchmark/final_average_best_reward": float(running_sum_best / running_count),
                    "benchmark/num_tasks": int(running_count),
                })
            except Exception as e:
                print(f"Error logging final aggregate: {e}")

        # Final scaling summary for ScalingLogger
        if scaling_logger is not None:
            try:
                scaling_logger.log_final_scaling_summary(
                    num_tasks=int(running_count),
                    task_index=None,  # No specific task index for final summary
                )
            except Exception as e:
                print(f"Error logging final scaling summary: {e}")

        return results


class PromptFileBenchmark(T2IBenchmarkBase):
    """Benchmark that reads one prompt per line from a text file.

    Options:
    - path: path to the prompt file (one prompt per line)
    - randomize: shuffle prompts (default: False)
    - max_prompts: limit number of prompts loaded
    - strip_comments: lines starting with '#' are ignored
    - delimiter: if not None, split on delimiter and take first field
    """

    def __init__(
        self,
        num_runs: int = 1,
        *,
        path: str,
        randomize: bool = False,
        max_prompts: Optional[int] = None,
        strip_comments: bool = True,
        delimiter: Optional[str] = None,
        start_index: Optional[int] = None,
        end_index: Optional[int] = None,
    ) -> None:
        super().__init__(num_runs=num_runs, randomize=randomize)
        self.path = path
        self.max_prompts = max_prompts
        self.strip_comments = strip_comments
        self.delimiter = delimiter
        self.start_index = start_index
        self.end_index = end_index

    def _load_lines(self) -> List[str]:
        with open(self.path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        prompts = []
        for line in lines:
            s = line.strip()
            if not s:
                continue
            if self.strip_comments and s.startswith("#"):
                continue
            if self.delimiter is not None:
                s = s.split(self.delimiter)[0]
            prompts.append(s)
        # Optional slicing before randomization (mimics older Runner semantics)
        if self.start_index is not None or self.end_index is not None:
            s = int(self.start_index or 0)
            e = int(self.end_index) if self.end_index is not None else len(prompts)
            s = max(0, s)
            e = max(s, min(e, len(prompts)))
            prompts = prompts[s:e]
        if self.randomize:
            random.shuffle(prompts)
        if self.max_prompts:
            prompts = prompts[: self.max_prompts]
        return prompts

    def prompts(self) -> List[str]:
        return self._load_lines()


class DrawBenchBenchmark(PromptFileBenchmark):
    """DrawBench benchmark (Imagen). Paper: https://arxiv.org/abs/2205.11487

    By default reads ``my_image/benchmark/draw_bench.txt``. You can select different
    subsets by passing ``version`` (e.g., "10", "50", "100"), which will resolve to
    ``draw_bench_<version>.txt`` in the same directory. If ``path`` is provided, it
    takes precedence over ``version``.
    """

    def __init__(
        self,
        num_runs: int = 1,
        randomize: bool = False,
        path: Optional[str] = None,
        version: Optional[str] = None,
        max_prompts: Optional[int] = None,
        start_index: Optional[int] = None,
        end_index: Optional[int] = None,
    ):
        if path is None:
            # Resolve default path relative to core package
            core_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # core/
            bench_dir = os.path.join(core_dir, "benchmarks", "data")

            # Determine filename from version
            file_name = "draw_bench.txt"
            if version is not None:
                v = str(version).strip().lower()
                if v in {"full", "all", "default", "", "none"}:
                    file_name = "draw_bench.txt"
                else:
                    # Accept numeric-like or any token to allow future variants
                    file_name = f"draw_bench_{v}.txt"

            path = os.path.join(bench_dir, file_name)

        # Debug: show which prompts file/version is being used
        try:
            exists = os.path.exists(path)
            print(f"[DrawBenchBenchmark] version={version!r} -> file='{path}' (exists={exists})")
        except Exception as e:
            print(f"Error checking if prompts file exists: {e}")

        super().__init__(
            num_runs=num_runs,
            path=path,
            randomize=randomize,
            delimiter="\t",
            max_prompts=max_prompts,
            start_index=start_index,
            end_index=end_index,
        )


class SimpleAnimalsBenchmark(PromptFileBenchmark):
    """A small T2I prompt list for quick smoke tests; reads simple_animals.txt."""

    def __init__(self, num_runs: int = 1, randomize: bool = False, path: Optional[str] = None, max_prompts: Optional[int] = None, start_index: Optional[int] = None, end_index: Optional[int] = None):
        if path is None:
            core_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            default_path = os.path.join(core_dir, "benchmarks", "data", "simple_animals.txt")
            path = default_path
        super().__init__(num_runs=num_runs, path=path, randomize=randomize, max_prompts=max_prompts, start_index=start_index, end_index=end_index)


class COCOCaptionsBenchmark(Benchmark):
    """COCO Captions-based T2I prompts. Requires a captions JSON.

    Args:
        captions_json: path to COCO annotations (captions_train2017.json or similar)
        num_runs: number of tasks to emit
        randomize: shuffle captions (default: False)
        caption_key: JSON key holding the list of annotations (default: 'annotations')
        text_key: field name for the caption text (default: 'caption')
        max_prompts: optional cap on number of prompts
    """

    def __init__(
        self,
        captions_json: str,
        num_runs: int = 1,
        randomize: bool = False,
        caption_key: str = "annotations",
        text_key: str = "caption",
        max_prompts: Optional[int] = None,
    ) -> None:
        super().__init__(num_runs=num_runs, randomize=randomize)
        self.captions_json = captions_json
        self.caption_key = caption_key
        self.text_key = text_key
        self.max_prompts = max_prompts

    def _load_captions(self) -> List[str]:
        with open(self.captions_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        annotations = data.get(self.caption_key, [])
        captions = [ann.get(self.text_key, "").strip() for ann in annotations]
        captions = [c for c in captions if c]
        if self.randomize:
            random.shuffle(captions)
        if self.max_prompts:
            captions = captions[: self.max_prompts]
        return captions

    def prompts(self) -> List[str]:
        return self._load_captions()

 
class Gen2EvalBenchmark(T2IBenchmarkBase):
    """Benchmark that loads prompts from a Gen2Eval JSONL file.

    Each line is a JSON object with at least "prompt". May also contain
    vqa_list, atom_count, skills for VQA-based evaluation.

    Options:
    - path: path to the .jsonl file
    - randomize: shuffle prompts (default: False)
    - max_prompts: limit number of prompts loaded
    - start_index, end_index: slice range (before randomization)
    """

    def __init__(
        self,
        num_runs: int = 1,
        *,
        path: Optional[str] = None,
        randomize: bool = False,
        max_prompts: Optional[int] = None,
        start_index: Optional[int] = None,
        end_index: Optional[int] = None,
    ) -> None:
        super().__init__(num_runs=num_runs, randomize=randomize)
        if path is None:
            core_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = os.path.join(core_dir, "benchmarks", "data", "gen_eval2.jsonl")
        self.path = path
        self.max_prompts = max_prompts
        self.start_index = start_index
        self.end_index = end_index

    def _load_records(self) -> List[Dict[str, Any]]:
        p = os.path.abspath(os.path.expanduser(self.path))
        if not os.path.isfile(p):
            _dir = os.path.dirname(os.path.abspath(__file__))
            p_alt = os.path.join(_dir, "data", os.path.basename(self.path))
            if os.path.isfile(p_alt):
                p = p_alt
            else:
                raise FileNotFoundError(f"Gen2Eval JSONL not found: {self.path} (also tried {p_alt})")
        records = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        if self.start_index is not None or self.end_index is not None:
            s = int(self.start_index or 0)
            e = int(self.end_index) if self.end_index is not None else len(records)
            s = max(0, s)
            e = max(s, min(e, len(records)))
            records = records[s:e]
        if self.randomize:
            random.shuffle(records)
        if self.max_prompts:
            records = records[: self.max_prompts]
        return records

    def prompts(self) -> List[str]:
        return [r.get("prompt", "").strip() for r in self._load_records() if r.get("prompt")]

    def tasks(self) -> List[Dict[str, Any]]:
        """Return task dicts with full context (prompt, vqa_list, skills, etc.)."""
        records = self._load_records()
        tasks = []
        for r in records:
            prompt = (r.get("prompt") or "").strip()
            if not prompt:
                continue
            context = {"prompt": prompt}
            if "vqa_list" in r:
                context["vqa_list"] = r["vqa_list"]
            if "skills" in r:
                context["skills"] = r["skills"]
            if "atom_count" in r:
                context["atom_count"] = r["atom_count"]
            tasks.append({"modality": "image", "context": context})
        return tasks


class ListPromptsBenchmark(Benchmark):
    """Benchmark from an in-memory list of prompts (for configs)."""

    def __init__(self, prompts: List[str], num_runs: int = 1, randomize: bool = False) -> None:
        super().__init__(num_runs=num_runs, randomize=randomize)
        self._prompts = list(prompts)

    def prompts(self) -> List[str]:
        prompts = list(self._prompts)
        if self.randomize:
            random.shuffle(prompts)
        return prompts


def _load_prompts_from_file(path: str) -> List[str]:
    """Load prompts from a text file, one per line. Skips empty lines."""
    p = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(p):
        # Try relative to this module's directory (benchmarks/data/ is sibling)
        _dir = os.path.dirname(os.path.abspath(__file__))
        p_alt = os.path.join(_dir, "data", os.path.basename(path))
        if os.path.isfile(p_alt):
            p = p_alt
        else:
            raise FileNotFoundError(f"Prompts file not found: {path} (also tried {p_alt})")
    with open(p, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


class CustomPromptBenchmark(T2IBenchmarkBase):
    """Custom prompt benchmark for quick experiments with custom prompts.
    
    This benchmark logs the best image per iteration (instead of only at the end)
    for better visualization during paper experiments.
    
    Args:
        prompts: List of custom prompts to optimize (ignored if prompts_path is set)
        prompts_path: Path to text file with prompts, one per line (overrides prompts)
        num_runs: Number of times to run each prompt
        randomize: Whether to shuffle prompts
        log_iteration_images: If True, logs best image per iteration (default: True)
        start_index: Optional start index for slicing prompts (for compatibility)
        end_index: Optional end index for slicing prompts (for compatibility)
    """

    def __init__(
        self, 
        prompts: Optional[List[str]] = None, 
        prompts_path: Optional[str] = None,
        num_runs: Optional[int] = None, 
        randomize: bool = False,
        log_iteration_images: bool = True,
        start_index: Optional[int] = None,
        end_index: Optional[int] = None,
    ) -> None:
        if prompts_path:
            raw_prompts = _load_prompts_from_file(prompts_path)
        else:
            raw_prompts = list(prompts or [])
        
        if not raw_prompts:
            raise ValueError("CustomPromptBenchmark requires either prompts or prompts_path with at least one prompt")
        
        # Fix prompts that were incorrectly split by Hydra (only when from config/CLI)
        if not prompts_path:
            raw_prompts = self._fix_split_prompts(raw_prompts)
        
        self._prompts = raw_prompts
        if num_runs is None:
            num_runs = len(self._prompts)
        super().__init__(num_runs=num_runs, randomize=randomize)
        self.log_iteration_images = log_iteration_images
        self.start_index = start_index
        self.end_index = end_index
    
    def _fix_split_prompts(self, prompts: List[str]) -> List[str]:
        """Fix prompts that were incorrectly split by Hydra's comma parsing.
        
        When Hydra parses a list from command line like:
            benchmark.prompts=["Prompt1, with commas", "Prompt2, with commas"]
        
        It splits on ALL commas, treating each comma-separated part as a separate item:
            ["Prompt1", " with commas", "Prompt2", " with commas"]
        
        Solution: Join everything back together and split at quotes to get actual prompts.
        """
        if not prompts:
            return prompts
        
        # Join all items with commas (reconstructing the original string)
        joined = ", ".join(prompts)
        
        # Split by quotes to get individual prompts.
        # Try double quotes first, then single quotes; use whichever produces more splits.
        parts_double = joined.split('"')
        parts_single = joined.split("'")
        
        if len(parts_double) > len(parts_single):
            parts = parts_double
        else:
            parts = parts_single
        
        # Extract prompts (odd-indexed tokens are the quoted content)
        fixed = []
        for i in range(1, len(parts), 2):
            if i < len(parts):
                prompt = parts[i].strip()
                if prompt:
                    fixed.append(prompt)
        
        # If no quoted prompts found, return original (e.g. prompts came from a config file)
        return fixed if fixed else prompts

    def prompts(self) -> List[str]:
        prompts = list(self._prompts)
        
        # Apply slicing if indices are provided
        if self.start_index is not None or self.end_index is not None:
            s = int(self.start_index or 0)
            e = int(self.end_index) if self.end_index is not None else len(prompts)
            s = max(0, s)
            e = max(s, min(e, len(prompts)))
            prompts = prompts[s:e]
        
        if self.randomize:
            random.shuffle(prompts)
        return prompts


