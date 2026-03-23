"""Scaling experiment logger for tracking performance at multiple compute budgets."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import torch
import numpy as np

from .t2i import T2IWandbLogger


class ScalingLogger(T2IWandbLogger):
    """Logger for scaling experiments that tracks performance at multiple compute budgets.
    
    This logger allows running a single experiment with the highest compute budget
    and extracting performance metrics at intermediate budget checkpoints.
    
    Example:
        logger = ScalingLogger(
            project="ScaleAnimals0",
            name="trust_region_scaling",
            config=cfg,
            budget_checkpoints=[120, 240, 480, 960]  # oracle calls
        )
        
        # During optimization, track cumulative oracle calls
        logger.log_with_checkpoints(
            iteration=10,
            cumulative_oracles=150,  # We've used 150 oracles so far
            best_reward=0.85,
            best_image=best_img,
            prompt="A red car"
        )
        
        # This will log metrics at checkpoint 120 (since 150 > 120)
        # When cumulative_oracles reaches 240, it will log at that checkpoint too
    """

    def configure_logging(self, logging_config: Optional[Dict[str, Any]]) -> None:
        super().configure_logging(logging_config)
        self._define_scaling_metrics()

    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        enable: bool = True,
        budget_checkpoints: Optional[List[int]] = None,
        wandb_project: Optional[str] = None,  # For backward compatibility
        logging_config: Optional[Dict[str, Any]] = None,
        wandb_dir: Optional[str] = None,
    ):
        """Initialize scaling logger.
        
        Args:
            project: Wandb project name
            name: Wandb run name
            config: Experiment configuration
            enable: Whether to enable wandb logging
            budget_checkpoints: List of oracle budget checkpoints to track (e.g., [120, 240, 480, 960])
                               If None, will use default checkpoints [120, 240, 480, 960]
            wandb_project: For backward compatibility (maps to project)
            wandb_dir: Optional directory for wandb run files (overrides WANDB_DIR env).
        """
        if wandb_project is not None and project is None:
            project = wandb_project
        super().__init__(project=project, name=name, config=config, enable=enable, logging_config=logging_config, wandb_dir=wandb_dir)
        
        # Set budget checkpoints
        if budget_checkpoints is None:
            budget_checkpoints = [120, 240, 480, 960]
        self.budget_checkpoints = sorted(budget_checkpoints)
        
        # Get verbosity setting from terminal config
        terminal_scaling = self._terminal_section("scaling")
        verbosity = str(terminal_scaling.get("verbosity", "summary")).lower()
        self.verbose = verbosity in ("detailed", "debug")
        self.debug = verbosity == "debug"
        
        # Track which checkpoints have been logged
        self.logged_checkpoints: set[int] = set()
        
        # Track best values at each checkpoint for final summary
        self.checkpoint_best_rewards: Dict[int, float] = {}
        self.checkpoint_best_images: Dict[int, Any] = {}
        self.checkpoint_iterations: Dict[int, int] = {}
        
        # Track running average of best rewards at each checkpoint across tasks
        self.checkpoint_running_sums: Dict[int, float] = {}
        self.checkpoint_running_counts: Dict[int, int] = {}
        
        # Track cumulative best across all tasks at each checkpoint
        self.cumulative_best_rewards: Dict[int, float] = {}
        self.cumulative_best_images: Dict[int, Any] = {}
        
        # Efficient checkpoint tracking (your algorithm)
        self.checkpoint_rewards_list = []  # List of best rewards at each checkpoint
        self.checkpoint_images_list = []   # List of best images at each checkpoint
        self.current_task_best_reward = -float('inf')
        self.current_task_best_image = None
        
        # Define scaling-specific metrics
        self._define_scaling_metrics()

        # Always show initialization confirmation if terminal output is enabled
        if self.should_print_scaling():
            verbosity_str = "debug" if self.debug else ("detailed" if self.verbose else "summary")
            print(f"[ScalingLogger] Active | checkpoints={self.budget_checkpoints} | verbosity={verbosity_str}")

    def _define_scaling_metrics(self) -> None:
        """Define scaling-specific metrics for wandb tracking."""
        if not self.enable or not self.should_log_scaling_metrics():
            return
        
        # Define step metrics for both task and scaling steps
        self.wandb.define_metric("task_step")
        self.wandb.define_metric("scaling_step")
        
        # Define metrics for each checkpoint with both task and scaling steps
        for checkpoint in self.budget_checkpoints:
            self.wandb.define_metric(f"scaling/best_reward_at_{checkpoint}", step_metric="task_step")
            self.wandb.define_metric(f"scaling/iteration_at_{checkpoint}", step_metric="task_step")
            self.wandb.define_metric(f"scaling/oracles_at_{checkpoint}", step_metric="task_step")
            if self.should_log_scaling_running_average():
                self.wandb.define_metric(f"scaling/running_avg_at_{checkpoint}", step_metric="task_step")
            self.wandb.define_metric(f"scaling/task_reward_at_{checkpoint}", step_metric="task_step")
        
        # Define final summary metrics (no step)
        self.wandb.define_metric("scaling/final_average_best_reward")
        self.wandb.define_metric("scaling/num_checkpoints")
        
        # Define checkpoint images (no step metric for images)
        if self.should_log_scaling_samples():
            for checkpoint in self.budget_checkpoints:
                self.wandb.define_metric(f"scaling/best_image_at_{checkpoint}")

        # Debug output (only if terminal output enabled and in debug mode)
        if self.should_print_scaling() and self.debug:
            print(f"[ScalingLogger] Defined scaling metrics for checkpoints: {self.budget_checkpoints}")

    def log_with_checkpoints(
        self,
        iteration: int,
        cumulative_oracles: int,
        best_reward: float,
        current_task_best_image: Optional[Any] = None,
        current_task_best_reward: Optional[float] = None,
        prompt: Optional[str] = None,
        task_history: Optional[Dict[int, float]] = None,
        task_oracles: Optional[int] = None,
        task_step_history: Optional[Dict[int, int]] = None,
        task_image_history: Optional[Dict[int, Any]] = None,
        problem: Optional[Any] = None,
    ) -> None:
        """Log metrics and check if we've hit any budget checkpoints.
        
        This method tracks the best reward and image at each checkpoint across all tasks.
        """
        if not self.should_log_scaling_metrics():
            return
        
        # Helper to convert reward to float (handles tensors)
        def _to_float(val):
            if val is None:
                return None
            if isinstance(val, torch.Tensor):
                return float(val.item() if val.numel() == 1 else val.max().item())
            return float(val)
        
        # Terminal output: input summary (if enabled and verbosity allows)
        if self.should_print_scaling():
            if self.debug or self.verbose:
                try:
                    th_len = len(task_history) if task_history is not None else 0
                    task_reward_str = f"{_to_float(current_task_best_reward):.4f}" if current_task_best_reward is not None else "None"
                    cum_reward_str = f"{cumulative_best_reward:.4f}" if best_reward is not None else "None"
                    print(f"[ScalingLogger] Processing task {iteration} | cum_oracles={cumulative_oracles} | task_oracles={task_oracles} | history_len={th_len}")
                    print(f"[ScalingLogger]   current_task_best_reward={task_reward_str} | cumulative_best_reward={cum_reward_str}")
                    if task_history and len(task_history) > 0:
                        sample_keys = sorted(list(task_history.keys()))[:5]  # First 5 keys
                        print(f"[ScalingLogger]   task_history sample: {[(k, f'{_to_float(v):.4f}') for k, v in [(k, task_history[k]) for k in sample_keys]]}")
                except Exception as e:
                    print(f"[ScalingLogger] Debug output error: {e}")
        # Update current task best reward and image
        if current_task_best_reward is not None:
            # Convert to float if tensor
            task_reward_float = _to_float(current_task_best_reward)
            if task_reward_float > self.current_task_best_reward:
                self.current_task_best_reward = task_reward_float
                if current_task_best_image is not None:
                    self.current_task_best_image = current_task_best_image
        
        # Use the cumulative best reward across all tasks (passed as best_reward parameter)
        # Convert to float to ensure it's not a tensor
        cumulative_best_reward = _to_float(best_reward)
        def _first_non_none(*values):
            for val in values:
                if val is not None:
                    return val
            return None
        cumulative_best_image = _first_non_none(current_task_best_image, self.current_task_best_image)
        
        # Defensive: ensure all task_history values are floats (not tensors)
        if task_history is not None:
            task_history = {k: _to_float(v) for k, v in task_history.items()}
        
        # Track which checkpoints were crossed in this task for running average calculation
        checkpoints_crossed_this_task = []
        # Track this task's per-checkpoint rewards for a concise summary printout
        task_ckpt_rewards: Dict[int, float] = {}
        
        # Track the best reward seen so far as we iterate through checkpoints (for monotonicity)
        best_reward_so_far_this_task = float("-inf")
        best_image_so_far_this_task = None
        
        # Compute task-specific best reward once (same for all checkpoints in this task)
        # CRITICAL: Use current_task_best_reward as fallback (task-specific), not cumulative_best_reward (across all tasks)
        task_specific_best = _to_float(current_task_best_reward) if current_task_best_reward is not None else None
        
        # Check each checkpoint using per-task oracle counts only
        for checkpoint in self.budget_checkpoints:
            # Only include this task for the checkpoint if the task reached it
            if self.should_print_scaling() and (self.debug or self.verbose):
                print(f"[ScalingLogger] -> ckpt={checkpoint} | task_oracles={task_oracles} | cum_oracles={cumulative_oracles}")
            if task_oracles is not None and task_oracles < checkpoint:
                if self.should_print_scaling() and (self.debug or self.verbose):
                    print(f"[ScalingLogger]    skip ckpt={checkpoint} (task did not reach)")
                continue
            
            # Calculate the reward/image at this specific checkpoint
            # CRITICAL: We need the best reward/image that was available AT OR BEFORE this checkpoint's oracle count (within this task)
            # CRITICAL: The reward must be monotonically non-decreasing (later checkpoints >= earlier checkpoints)
            checkpoint_reward = task_specific_best if task_specific_best is not None else cumulative_best_reward  # Fallback to task-specific, then cumulative
            checkpoint_image = None
            
            # Use checkpoint-specific data from solver history if available
            if task_history:
                # Check if we have checkpoint-specific data from the solver
                checkpoint_key = f"checkpoint_{checkpoint}"
                if checkpoint_key in task_history:
                    checkpoint_reward = _to_float(task_history[checkpoint_key])
                    if self.should_print_scaling() and self.debug:
                        print(f"[ScalingLogger] Using solver checkpoint data for {checkpoint}: {checkpoint_reward:.4f}")
                else:
                    # Interpolate from task history: find the MAXIMUM reward at or before this checkpoint within the task
                    # This ensures we get the best reward, not just the last one
                    checkpoint_in_task = checkpoint
                    best_reward_at_checkpoint = None
                    best_image_at_checkpoint = None
                    best_oracle_count = -1
                    
                    # Find the maximum reward at or before this checkpoint
                    for oracle_count in sorted(task_history.keys()):
                        if oracle_count <= checkpoint_in_task:
                            reward_val = _to_float(task_history[oracle_count])
                            # Track the maximum reward and its corresponding oracle count
                            if best_reward_at_checkpoint is None or reward_val > best_reward_at_checkpoint:
                                best_reward_at_checkpoint = reward_val
                                best_oracle_count = oracle_count
                                # If image history is provided, track best image at this oracle
                                if task_image_history is not None and oracle_count in task_image_history:
                                    best_image_at_checkpoint = task_image_history[oracle_count]
                        else:
                            break
                    
                    if best_reward_at_checkpoint is not None:
                        checkpoint_reward = best_reward_at_checkpoint
                        checkpoint_image = best_image_at_checkpoint
                        if self.should_print_scaling() and self.debug:
                            print(f"[ScalingLogger] Using task history for checkpoint {checkpoint}: {checkpoint_reward:.4f} (max at or before checkpoint, found at oracle {best_oracle_count})")
                    else:
                        # No history up to this checkpoint (should be rare if task_oracles >= checkpoint)
                        # Use task-specific best reward as fallback
                        checkpoint_reward = task_specific_best if task_specific_best is not None else _to_float(cumulative_best_reward)
                        checkpoint_image = _first_non_none(current_task_best_image, self.current_task_best_image)
                        if self.should_print_scaling() and self.debug:
                            print(f"[ScalingLogger] No per-task history for checkpoint {checkpoint}, falling back to task-specific best: {checkpoint_reward:.4f}")
            else:
                # No task history available, use task-specific best reward
                checkpoint_reward = task_specific_best if task_specific_best is not None else _to_float(cumulative_best_reward)
                checkpoint_image = _first_non_none(current_task_best_image, self.current_task_best_image)
                if self.should_print_scaling() and self.debug:
                    print(f"[ScalingLogger] No task history, using task-specific best for checkpoint {checkpoint}: {checkpoint_reward:.4f}")
            
            # CRITICAL: Enforce monotonicity - checkpoint_reward must be >= best_reward_so_far_this_task
            checkpoint_reward = _to_float(checkpoint_reward)
            if checkpoint_reward < best_reward_so_far_this_task:
                if self.should_print_scaling() and self.debug:
                    print(f"[ScalingLogger] Enforcing monotonicity: checkpoint {checkpoint} reward {checkpoint_reward:.4f} < previous best {best_reward_so_far_this_task:.4f}, using {best_reward_so_far_this_task:.4f}")
                checkpoint_reward = best_reward_so_far_this_task
                checkpoint_image = best_image_so_far_this_task
            else:
                # Update best seen so far
                best_reward_so_far_this_task = checkpoint_reward
                if checkpoint_image is not None:
                    best_image_so_far_this_task = checkpoint_image

            if self.should_print_scaling() and self.debug:
                print(f"[ScalingLogger] ckpt={checkpoint} | chosen_reward={checkpoint_reward:.4f}")
            
            # CRITICAL: The checkpoint_reward now represents the best reward that was available at or before this checkpoint
            # This is the correct data to use for this checkpoint
            
            # CRITICAL: Each checkpoint should only be updated with data that was available AT OR BEFORE that checkpoint
            # The key insight: checkpoint_reward represents the best reward found up to this checkpoint's oracle count
            should_update_checkpoint = False
            
            # Ensure checkpoint_reward is a float
            checkpoint_reward = _to_float(checkpoint_reward)
            
            if checkpoint not in self.cumulative_best_rewards:
                # First time seeing this checkpoint - initialize it
                self.cumulative_best_rewards[checkpoint] = checkpoint_reward
                if checkpoint_image is not None:
                    self.cumulative_best_images[checkpoint] = checkpoint_image
                should_update_checkpoint = True
                if self.should_print_scaling() and self.debug:
                    print(f"[ScalingLogger] Initialized checkpoint {checkpoint} with reward: {checkpoint_reward:.4f}")
            else:
                # Only update if we found a better reward that was discovered AT OR BEFORE this checkpoint
                # This is the key fix: we only update if the new reward is better AND was found at or before this checkpoint
                # Defensive: ensure existing value is also a float
                existing_reward = _to_float(self.cumulative_best_rewards[checkpoint])
                if checkpoint_reward > existing_reward:
                    self.cumulative_best_rewards[checkpoint] = checkpoint_reward
                    if checkpoint_image is not None:
                        self.cumulative_best_images[checkpoint] = checkpoint_image
                    should_update_checkpoint = True
                    if self.should_print_scaling() and self.debug:
                        print(f"[ScalingLogger] Updated checkpoint {checkpoint} with better reward: {checkpoint_reward:.4f}")
                else:
                    # No improvement - checkpoint keeps previous best
                    if self.should_print_scaling() and self.debug:
                        print(f"[ScalingLogger] Checkpoint {checkpoint} no improvement, keeping best: {existing_reward:.4f}")
            
            current_final_reward = _to_float(self.cumulative_best_rewards.get(checkpoint, checkpoint_reward))
            max_prev_reward = float("-inf")
            max_prev_image = None
            
            # Find the maximum reward from all previous checkpoints
            for prev_checkpoint in sorted(self.cumulative_best_rewards.keys()):
                if prev_checkpoint < checkpoint:
                    prev_reward = _to_float(self.cumulative_best_rewards[prev_checkpoint])
                    if prev_reward > max_prev_reward:
                        max_prev_reward = prev_reward
                        prev_image = self.cumulative_best_images.get(prev_checkpoint)
                        if prev_image is not None:
                            max_prev_image = prev_image
            
            # Ensure monotonicity: current checkpoint reward must be >= max of previous checkpoints
            if max_prev_reward > current_final_reward:
                if self.should_print_scaling() and self.debug:
                    print(f"[ScalingLogger] Enforcing monotonicity: checkpoint {checkpoint} reward {current_final_reward:.4f} < max_prev {max_prev_reward:.4f}, using {max_prev_reward:.4f}")
                current_final_reward = max_prev_reward
                if max_prev_image is not None:
                    checkpoint_image = max_prev_image
            
            # Update the checkpoint with the final (monotonic) reward
            if current_final_reward != _to_float(self.cumulative_best_rewards.get(checkpoint, float("-inf"))):
                self.cumulative_best_rewards[checkpoint] = current_final_reward
                if checkpoint_image is not None:
                    self.cumulative_best_images[checkpoint] = checkpoint_image
                should_update_checkpoint = True
            
            # Track this checkpoint for running average calculation
            # CRITICAL: Use checkpoint_reward (task-specific, monotonic within task) for task_ckpt_rewards,
            # NOT current_final_reward (which is cumulative best across all tasks, forced to be monotonic)
            # This ensures each task's contribution reflects its actual performance, not the best across all tasks
            checkpoints_crossed_this_task.append(checkpoint)
            task_ckpt_rewards[checkpoint] = float(checkpoint_reward)
            
            # Update running averages for this checkpoint
            if checkpoint not in self.checkpoint_running_sums:
                self.checkpoint_running_sums[checkpoint] = 0.0
                self.checkpoint_running_counts[checkpoint] = 0
            
            # Add current task's contribution to running average
            # CRITICAL: Use checkpoint_reward (the reward available at this specific checkpoint)
            # not current_task_best_reward (the final reward from the task)
            task_contribution = checkpoint_reward
            prev_sum = self.checkpoint_running_sums[checkpoint]
            prev_cnt = self.checkpoint_running_counts[checkpoint]
            self.checkpoint_running_sums[checkpoint] = prev_sum + float(task_contribution)
            self.checkpoint_running_counts[checkpoint] = prev_cnt + 1
            if self.should_print_scaling() and self.debug:
                dbg_avg = self.checkpoint_running_sums[checkpoint] / max(1, self.checkpoint_running_counts[checkpoint])
                print(f"[ScalingLogger] Accumulate ckpt {checkpoint}: +{float(task_contribution):.4f} -> sum={self.checkpoint_running_sums[checkpoint]:.4f}, cnt={self.checkpoint_running_counts[checkpoint]}, avg={dbg_avg:.4f}")
            
            # Store for final summary (always use cumulative best across tasks)
            # Ensure we store a float value (defensive conversion)
            self.checkpoint_best_rewards[checkpoint] = _to_float(self.cumulative_best_rewards[checkpoint])
            if checkpoint in self.cumulative_best_images:
                self.checkpoint_best_images[checkpoint] = self.cumulative_best_images[checkpoint]
            
            # Log once per task step per checkpoint (no global gating)
            running_avg_at_checkpoint = self.checkpoint_running_sums[checkpoint] / self.checkpoint_running_counts[checkpoint]

            # Prefer the per-task checkpoint image for logging; fall back to best-across-tasks if missing
            image_for_log = checkpoint_image if checkpoint_image is not None else self.cumulative_best_images.get(checkpoint)

            self._log_checkpoint(
                checkpoint=checkpoint,
                iteration=iteration,
                cumulative_oracles=cumulative_oracles,
                best_reward=checkpoint_reward,
                running_avg=running_avg_at_checkpoint,
                current_task_image=image_for_log,
                prompt=prompt,
                task_step=iteration,
            )
            self.checkpoint_iterations[checkpoint] = iteration
            if self.should_print_scaling() and (self.debug or self.verbose):
                print(f"[ScalingLogger] Logged task {iteration} at checkpoint {checkpoint}: reward={checkpoint_reward:.4f}, running_avg={running_avg_at_checkpoint:.4f}")

        # After processing all checkpoints for this task, print a compact summary of running averages
        if checkpoints_crossed_this_task and self.should_print_scaling():
            print(f"[ScalingLogger] Running averages after task {iteration}:")
            for ckpt in sorted(self.budget_checkpoints):
                if ckpt in self.checkpoint_running_sums and self.checkpoint_running_counts.get(ckpt, 0) > 0:
                    avg = self.checkpoint_running_sums[ckpt] / self.checkpoint_running_counts[ckpt]
                    last = task_ckpt_rewards.get(ckpt, float('nan'))
                    cnt = self.checkpoint_running_counts[ckpt]
                    print(f"  - checkpoint {ckpt}: avg={avg:.4f} (count={cnt}) | task_reward={last if not np.isnan(last) else 'n/a'}")

    def _log_checkpoint(
        self,
        checkpoint: int,
        iteration: int,
        cumulative_oracles: int,
        best_reward: float,
        running_avg: float,
        current_task_image: Optional[Any] = None,
        prompt: Optional[str] = None,
        task_step: Optional[int] = None,
    ) -> None:
        """Log metrics for a specific budget checkpoint.
        
        Args:
            checkpoint: Budget checkpoint (e.g., 120, 240, 480, 960)
            iteration: Iteration number when checkpoint was crossed (cumulative task number)
            cumulative_oracles: Cumulative oracle calls at checkpoint
            best_reward: Best reward at checkpoint for this task
            running_avg: Running average of best rewards at checkpoint across all tasks
            current_task_image: Best image from current task
            prompt: Text prompt
            task_step: Task step (iteration within the current task) when checkpoint was crossed
        """
        if not self.enable:
            return
        
        # Helper to convert to float safely
        def _to_float_safe(val):
            if val is None:
                return None
            if isinstance(val, torch.Tensor):
                return float(val.item() if val.numel() == 1 else val.max().item())
            return float(val)
        
        # Build payload with task_step for proper tracking
        best_reward_at_ckpt = self.checkpoint_best_rewards.get(checkpoint, best_reward)
        payload: Dict[str, Any] = {
            "task_step": iteration,  # Use task_step for proper tracking
            f"scaling/task_reward_at_{checkpoint}": _to_float_safe(best_reward),  # Reward for this task at this checkpoint
            f"scaling/best_reward_at_{checkpoint}": _to_float_safe(best_reward_at_ckpt),  # Best across all tasks
            f"scaling/iteration_at_{checkpoint}": int(iteration),
            f"scaling/oracles_at_{checkpoint}": int(cumulative_oracles),
        }

        if self.should_log_scaling_running_average():
            payload[f"scaling/running_avg_at_{checkpoint}"] = _to_float_safe(running_avg)
        
        # Log best image at checkpoint (use the best image across all tasks at this checkpoint)
        best_img_at_checkpoint = self.checkpoint_best_images.get(checkpoint, current_task_image)
        if best_img_at_checkpoint is not None and self.should_log_scaling_samples():
            try:
                img_obj = best_img_at_checkpoint
                
                # Support torch.Tensor inputs in [C,H,W] or [1,C,H,W] with values in [0,1]
                if isinstance(img_obj, torch.Tensor):
                    x = img_obj.detach().cpu()
                    
                    if x.dim() == 4:
                        x = x[0]
                    if x.dim() == 3:
                        # Check if values are in [0,1] or [0,255]
                        max_val = float(x.max().item() if x.numel() > 0 else 1.0)
                        if max_val <= 1.0:
                            x = (x.clamp(0, 1) * 255).to(torch.uint8)
                        else:
                            x = x.clamp(0, 255).to(torch.uint8)
                        img_obj = x.permute(1, 2, 0).numpy()
                    elif x.dim() == 2:
                        # Check if values are in [0,1] or [0,255]
                        max_val = float(x.max().item() if x.numel() > 0 else 1.0)
                        if max_val <= 1.0:
                            x = (x.clamp(0, 1) * 255).to(torch.uint8)
                        else:
                            x = x.clamp(0, 255).to(torch.uint8)
                        img_obj = x.numpy()
                
                # If it's a list with a single image, unwrap
                if isinstance(img_obj, list) and len(img_obj) == 1:
                    img_obj = img_obj[0]
                
                # Create wandb image
                best_reward_overall = self.checkpoint_best_rewards.get(checkpoint, best_reward)
                caption = f"Checkpoint {checkpoint} oracles | task={iteration} | best_reward={best_reward_overall:.4f} | running_avg={running_avg:.4f}"
                if prompt:
                    caption += f" | {prompt}"
                payload[f"scaling/best_image_at_{checkpoint}"] = self.wandb.Image(img_obj, caption=caption)
                
                # Local saving: use save_checkpoint_structures (not save_outputs) so only benchmark images
                # are saved when save_outputs=True; scaling checkpoint images require save_checkpoint_structures=True
                if self.should_save_checkpoint_outputs():
                    save_format = self.get_save_format()
                    self.save_image_locally(
                        img_obj, 
                        f"checkpoint_{checkpoint}.{save_format}", 
                        task_index=iteration, 
                        prompt=prompt, 
                        subfolder="checkpoints"
                    )
            except Exception as e:
                print(f"[WARNING] Failed to log checkpoint image: {e}")
        
        # Debug payload preview (only in debug mode)
        if self.should_print_scaling() and self.debug:
            try:
                dbg_keys = [k for k in payload.keys() if k.startswith("scaling/")] + ["task_step"]
                dbg = {k: payload[k] for k in dbg_keys if k in payload and not k.endswith("best_image_at_" + str(checkpoint))}
                print(f"[ScalingLogger] wandb.log payload (ckpt={checkpoint}): {dbg}")
            except Exception:
                pass
        self.wandb.log(payload)
        if self.should_print_scaling() and (self.debug or self.verbose):
            print(f"[ScalingLogger] Checkpoint {checkpoint}: reward={best_reward:.4f}, running_avg={running_avg:.4f}, task={iteration}")

    def log_final_scaling_summary(
        self,
        num_tasks: int,
        task_index: Optional[int] = None,
    ) -> None:
        """Log final summary metrics across all checkpoints.

        Args:
            num_tasks: Total number of tasks in the benchmark
            task_index: Optional task index for tracking
        """
        if not self.enable or not self.should_log_scaling_metrics():
            return

        # Calculate average best reward across all checkpoints
        if self.checkpoint_best_rewards:
            avg_best_reward = np.mean(list(self.checkpoint_best_rewards.values()))

            payload: Dict[str, Any] = {
                "scaling/final_average_best_reward": float(avg_best_reward),
                "scaling/num_checkpoints": len(self.checkpoint_best_rewards),
            }

            # Add task index if provided
            if task_index is not None:
                payload["task_step"] = task_index

            # Log individual checkpoint rewards for visibility
            for checkpoint, reward in self.checkpoint_best_rewards.items():
                payload[f"scaling/final_reward_at_{checkpoint}"] = float(reward)

            self.wandb.log(payload)
            print(f"Final scaling summary: avg_best_reward={avg_best_reward:.4f} across {len(self.checkpoint_best_rewards)} checkpoints")

        # Save CSV file with final running averages
        csv_path = self.save_checkpoint_csv()
        if csv_path:
            print(f"Scaling checkpoint data saved to CSV: {csv_path}")

    def get_checkpoint_summary(self) -> Dict[str, Any]:
        """Get summary of all logged checkpoints.

        Returns:
            Dictionary with checkpoint summaries
        """
        summary = {
            "checkpoints": {},
            "average_best_reward": None,
            "num_checkpoints": len(self.checkpoint_best_rewards),
        }

        if self.checkpoint_best_rewards:
            summary["average_best_reward"] = float(np.mean(list(self.checkpoint_best_rewards.values())))

            for checkpoint in self.budget_checkpoints:
                if checkpoint in self.checkpoint_best_rewards:
                    summary["checkpoints"][checkpoint] = {
                        "best_reward": float(self.checkpoint_best_rewards[checkpoint]),
                        "iteration": int(self.checkpoint_iterations[checkpoint]),
                    }

        return summary

    def save_checkpoint_csv(self) -> Optional[str]:
        """Save a CSV file with final running averages for each scale checkpoint.

        Returns:
            Path to the saved CSV file, or None if saving failed
        """
        import os
        import csv
        import datetime

        try:
            # Get final running averages for each checkpoint
            checkpoint_data = []
            for checkpoint in self.budget_checkpoints:
                if checkpoint in self.checkpoint_running_sums and self.checkpoint_running_counts.get(checkpoint, 0) > 0:
                    running_avg = self.checkpoint_running_sums[checkpoint] / self.checkpoint_running_counts[checkpoint]
                    checkpoint_data.append({
                        'checkpoint': checkpoint,
                        'running_average': running_avg,
                        'count': self.checkpoint_running_counts[checkpoint]
                    })

            if not checkpoint_data:
                return None

            # Extract metadata from config
            modality = str(self.config.get("modality", "unknown")).lower() if self.config else "unknown"
            model = str(self.config.get("model", "unknown")).lower() if self.config else "unknown"
            reward_function = str(self.config.get("reward", "unknown")).lower() if self.config else "unknown"

            # Create unique run name with timestamp
            base_run_name = self.name or "run"
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            unique_run_name = f"{base_run_name}_{timestamp}"

            # Create directory structure
            output_dir = os.path.join("outputs", modality, model, reward_function, unique_run_name)
            os.makedirs(output_dir, exist_ok=True)

            # Create CSV file path
            csv_path = os.path.join(output_dir, "scaling_checkpoints.csv")

            # Try to get terminal command (don't fail if not available)
            terminal_command = None
            try:
                import sys
                # Check if we have command line arguments
                if len(sys.argv) > 1:
                    terminal_command = "python " + " ".join(sys.argv)
                else:
                    # Try to get from environment
                    import os
                    if "SLURM_JOB_NAME" in os.environ:
                        terminal_command = f"sbatch job: {os.environ.get('SLURM_JOB_NAME', 'unknown')}"
            except Exception:
                # Silently skip if we can't get the command
                pass

            # Write CSV file
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)

                # Write terminal command at the beginning if available
                if terminal_command:
                    writer.writerow(["Terminal Command", terminal_command])
                    writer.writerow([])  # Empty row for separation

                # Write header
                writer.writerow(["Checkpoint", "Running Average", "Task Count"])

                # Write data
                for data in checkpoint_data:
                    writer.writerow([
                        data['checkpoint'],
                        f"{data['running_average']:.6f}",
                        data['count']
                    ])

            print(f"Saved scaling checkpoint CSV to: {csv_path}")
            return csv_path

        except Exception as e:
            print(f"[WARNING] Failed to save scaling checkpoint CSV: {e}")
            return None

    def log_step_checkpoint(self, step_result: Dict[str, Any], cumulative_oracles: int, task_index: int) -> None:
        """Log checkpoint data from a solver step.
        
        Args:
            step_result: Result from solver.step()
            cumulative_oracles: Cumulative oracle count across all tasks
            task_index: Current task index
        """
        if not self.enable or not self.should_log_scaling_metrics():
            return
        
        # Helper to convert reward to float (handles tensors)
        def _to_float(val):
            if val is None:
                return None
            if isinstance(val, torch.Tensor):
                return float(val.item() if val.numel() == 1 else val.max().item())
            return float(val)
        
        # Check which checkpoints we've crossed
        for checkpoint in self.budget_checkpoints:
            if cumulative_oracles >= checkpoint:
                # Get checkpoint-specific reward if available
                checkpoint_key = f"checkpoint_{checkpoint}"
                checkpoint_reward_raw = step_result.get(checkpoint_key, step_result.get("best_reward", 0.0))
                checkpoint_reward = _to_float(checkpoint_reward_raw)
                checkpoint_image = step_result.get("best_image")
                
                # Update cumulative best at this checkpoint
                if checkpoint not in self.cumulative_best_rewards:
                    self.cumulative_best_rewards[checkpoint] = checkpoint_reward
                    self.cumulative_best_images[checkpoint] = checkpoint_image
                else:
                    # Defensive: ensure existing value is also a float
                    existing_reward = _to_float(self.cumulative_best_rewards[checkpoint])
                    if checkpoint_reward > existing_reward:
                        self.cumulative_best_rewards[checkpoint] = checkpoint_reward
                        self.cumulative_best_images[checkpoint] = checkpoint_image
                
                # Update running averages
                if checkpoint not in self.checkpoint_running_sums:
                    self.checkpoint_running_sums[checkpoint] = 0.0
                    self.checkpoint_running_counts[checkpoint] = 0
                
                self.checkpoint_running_sums[checkpoint] += checkpoint_reward
                self.checkpoint_running_counts[checkpoint] += 1
                
                # Log to wandb if this is a new checkpoint or if we have better data
                if checkpoint not in self.logged_checkpoints:
                    running_avg = self.checkpoint_running_sums[checkpoint] / self.checkpoint_running_counts[checkpoint]
                    
                    self._log_checkpoint(
                        checkpoint=checkpoint,
                        iteration=task_index,
                        cumulative_oracles=cumulative_oracles,
                        best_reward=self.cumulative_best_rewards[checkpoint],
                        running_avg=running_avg,
                        current_task_image=self.cumulative_best_images[checkpoint],
                        prompt=f"task_{task_index}",
                        task_step=task_index,
                    )
                    
                    self.logged_checkpoints.add(checkpoint)
                    self.checkpoint_iterations[checkpoint] = task_index
                    if self.should_print_scaling() and (self.debug or self.verbose):
                        print(f"[ScalingLogger] New checkpoint at {checkpoint} oracles: reward={self.cumulative_best_rewards[checkpoint]:.4f}, running_avg={running_avg:.4f}")

    def reset_checkpoints(self) -> None:
        """Reset checkpoint tracking for a new run."""
        self.logged_checkpoints.clear()
        self.checkpoint_best_rewards.clear()
        self.checkpoint_best_images.clear()
        self.checkpoint_iterations.clear()
        self.checkpoint_running_sums.clear()
        self.checkpoint_running_counts.clear()
        self.cumulative_best_rewards.clear()
        self.cumulative_best_images.clear()

    def close(self) -> None:
        """Close logger and save final CSV if scaling was used."""
        # Save scaling checkpoint CSV if we have checkpoint data
        if hasattr(self, 'checkpoint_running_sums') and self.checkpoint_running_sums:
            csv_path = self.save_checkpoint_csv()
            if csv_path:
                print(f"Scaling checkpoint data saved to CSV: {csv_path}")

        # Call parent close method
        super().close()

