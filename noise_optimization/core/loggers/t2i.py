from __future__ import annotations

from typing import Any, Dict, List, Optional
from copy import deepcopy

import torch
from omegaconf import DictConfig, OmegaConf

from .base import WandbLogger, _as_plain_dict


class T2IWandbLogger(WandbLogger):
    """Wandb logger with helpers for text-to-image experiments."""

    DEFAULT_LOGGING_CONFIG: Dict[str, Any] = {
        "save_format": "jpg",  # Default format for local image saving
        "wandb": {
            "iteration": {
                "enabled": False,
                "log_metrics": True,
                "log_data_samples": False,
                "data_sample_limit": 4,
                "log_operator_stats": True,
            },
            "benchmark": {
                "enabled": True,
                "log_best_data_sample": True,
                "log_running_best_data_sample": True,
                "log_reward_curve": True,
            },
            "scaling": {
                "enabled": False,
                "log_running_average": True,
                "log_checkpoint_data_samples": False,  # Off by default; set true to log/save scaling checkpoint images (can fill disk in t2i)
            },
        },
        "terminal": {
            "iteration": {
                "enabled": False,
                "verbosity": "minimal",
            },
            "benchmark": {
                "enabled": True,
                "verbosity": "summary",
            },
            "scaling": {
                "enabled": True,
                "verbosity": "summary",
            },
        },
    }

    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        enable: bool = True,
        logging_config: Optional[Dict[str, Any]] = None,
        wandb_dir: Optional[str] = None,
    ):
        merged_cfg = deepcopy(self.DEFAULT_LOGGING_CONFIG)
        if logging_config:
            merged_cfg = self._merge_logging_config(merged_cfg, logging_config)
        self.logging_config: Dict[str, Any] = self._normalize_logging_config(merged_cfg)
        super().__init__(project=project, name=name, config=config, enable=enable, wandb_dir=wandb_dir)

    def _merge_logging_config(self, base: Dict[str, Any], logging_config: Dict[str, Any]) -> Dict[str, Any]:
        """Deep-merge user provided logging config with defaults."""
        merged = deepcopy(base)
        cfg_plain = _as_plain_dict(logging_config)

        def _merge(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
            for key, value in src.items():
                if isinstance(value, dict) and isinstance(dst.get(key), dict):
                    dst[key] = _merge(dict(dst.get(key, {})), value)
                else:
                    dst[key] = value
            return dst
        return _merge(dict(merged), cfg_plain or {})

    def _normalize_logging_config(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize logging config to new structure (wandb/terminal) with backward compatibility."""
        cfg = deepcopy(cfg)
        
        # Check if already in new format (has wandb/terminal keys)
        has_new_format = "wandb" in cfg or "terminal" in cfg
        
        if not has_new_format:
            # Old format: migrate top-level keys to wandb section
            wandb_cfg = {}
            terminal_cfg = {}
            
            # Migrate iteration settings
            if "iteration" in cfg:
                iter_old = cfg.pop("iteration")
                wandb_cfg["iteration"] = {
                    "enabled": iter_old.get("enabled", False),
                    "log_metrics": True,
                    "log_data_samples": iter_old.get("log_data_samples", iter_old.get("log_images", False)),
                    "data_sample_limit": iter_old.get("data_sample_limit", iter_old.get("topk_images", 4)),
                    "log_operator_stats": iter_old.get("log_operator_stats", True),
                }
                terminal_cfg["iteration"] = {
                    "enabled": iter_old.get("enabled", False),
                    "verbosity": "minimal",
                }
            
            # Migrate benchmark settings
            if "benchmark" in cfg:
                bench_old = cfg.pop("benchmark")
                wandb_cfg["benchmark"] = {
                    "enabled": bench_old.get("enabled", True),
                    "log_best_data_sample": bench_old.get("log_best_data_sample", bench_old.get("log_best_image", True)),
                    "log_running_best_data_sample": bench_old.get("log_running_best_data_sample", bench_old.get("log_running_best_image", True)),
                    "log_reward_curve": bench_old.get("log_reward_curve", True),
                }
                terminal_cfg["benchmark"] = {
                    "enabled": bench_old.get("enabled", True),
                    "verbosity": "summary",
                }
            
            # Migrate scaling settings
            if "scaling" in cfg:
                scaling_old = cfg.pop("scaling")
                wandb_cfg["scaling"] = {
                    "enabled": scaling_old.get("enabled", False),
                    "budget_checkpoints": scaling_old.get("budget_checkpoints", [120, 240, 480, 960]),
                    "log_running_average": scaling_old.get("log_running_average", True),
                    "log_checkpoint_data_samples": scaling_old.get("log_checkpoint_data_samples", scaling_old.get("log_checkpoint_images", False)),
                }
                terminal_cfg["scaling"] = {
                    "enabled": scaling_old.get("enabled", True),
                    "verbosity": scaling_old.get("verbosity", "summary"),
                }
            
            cfg["wandb"] = wandb_cfg
            cfg["terminal"] = terminal_cfg
        
        # Ensure both sections exist with defaults
        wandb_section = cfg.setdefault("wandb", {})
        terminal_section = cfg.setdefault("terminal", {})
        
        # Normalize wandb section
        for section_name in ["iteration", "benchmark", "scaling"]:
            if section_name not in wandb_section:
                wandb_section[section_name] = self.DEFAULT_LOGGING_CONFIG["wandb"].get(section_name, {})
            if section_name not in terminal_section:
                terminal_section[section_name] = self.DEFAULT_LOGGING_CONFIG["terminal"].get(section_name, {})
        
        return cfg

    # ------------------------------------------------------------------ #
    # Logging preference helpers
    # ------------------------------------------------------------------ #
    def configure_logging(self, logging_config: Optional[Dict[str, Any]]) -> None:
        if logging_config:
            merged = self._merge_logging_config(self.logging_config, logging_config)
            self.logging_config = self._normalize_logging_config(merged)
        # Re-declare metrics so wandb tracks only enabled streams
        self._define_base_metrics()

    def _wandb_section(self, name: str) -> Dict[str, Any]:
        """Get wandb logging section."""
        return self.logging_config.get("wandb", {}).get(name, {})
    
    def _terminal_section(self, name: str) -> Dict[str, Any]:
        """Get terminal output section."""
        return self.logging_config.get("terminal", {}).get(name, {})
    
    def _section(self, name: str) -> Dict[str, Any]:
        """Backward compatibility: returns wandb section."""
        return self._wandb_section(name)

    # Wandb logging controls
    def should_log_iteration_metrics(self) -> bool:
        return bool(self._wandb_section("iteration").get("enabled", False))

    def should_log_iteration_samples(self) -> bool:
        iteration_cfg = self._wandb_section("iteration")
        return bool(iteration_cfg.get("enabled", False) and iteration_cfg.get("log_data_samples", False))

    def get_iteration_sample_limit(self) -> int:
        return int(self._wandb_section("iteration").get("data_sample_limit", 0) or 0)

    def should_log_iteration_operator_stats(self) -> bool:
        iteration_cfg = self._wandb_section("iteration")
        return bool(iteration_cfg.get("log_operator_stats", True))

    def should_log_benchmark_metrics(self) -> bool:
        return bool(self._wandb_section("benchmark").get("enabled", True))

    def should_log_benchmark_best_sample(self) -> bool:
        bench_cfg = self._wandb_section("benchmark")
        return bool(bench_cfg.get("log_best_data_sample", True) and bench_cfg.get("enabled", True))

    def should_log_running_best_sample(self) -> bool:
        bench_cfg = self._wandb_section("benchmark")
        return bool(bench_cfg.get("log_running_best_data_sample", True) and bench_cfg.get("enabled", True))

    def should_log_reward_curve(self) -> bool:
        bench_cfg = self._wandb_section("benchmark")
        return bool(bench_cfg.get("log_reward_curve", True) and bench_cfg.get("enabled", True))

    def should_log_scaling_metrics(self) -> bool:
        return bool(self._wandb_section("scaling").get("enabled", False))

    def should_log_scaling_running_average(self) -> bool:
        scaling_cfg = self._wandb_section("scaling")
        return bool(scaling_cfg.get("log_running_average", True) and scaling_cfg.get("enabled", False))

    def should_log_scaling_samples(self) -> bool:
        scaling_cfg = self._wandb_section("scaling")
        return bool(scaling_cfg.get("log_checkpoint_data_samples", False) and scaling_cfg.get("enabled", False))
    
    # Terminal output controls
    def should_print_iteration(self) -> bool:
        return bool(self._terminal_section("iteration").get("enabled", False))
    
    def get_iteration_verbosity(self) -> str:
        return str(self._terminal_section("iteration").get("verbosity", "minimal"))
    
    def should_print_benchmark(self) -> bool:
        return bool(self._terminal_section("benchmark").get("enabled", True))
    
    def get_benchmark_verbosity(self) -> str:
        return str(self._terminal_section("benchmark").get("verbosity", "summary"))
    
    def should_print_scaling(self) -> bool:
        return bool(self._terminal_section("scaling").get("enabled", True))
    
    def get_scaling_verbosity(self) -> str:
        return str(self._terminal_section("scaling").get("verbosity", "summary"))

    # Backward-compat helper names (image-specific)
    def should_log_iteration_images(self) -> bool:
        return self.should_log_iteration_samples()

    def get_iteration_topk_images(self) -> int:
        return self.get_iteration_sample_limit()

    def should_log_benchmark_best_image(self) -> bool:
        return self.should_log_benchmark_best_sample()

    def should_log_running_best_image(self) -> bool:
        return self.should_log_running_best_sample()

    def should_log_scaling_images(self) -> bool:
        return self.should_log_scaling_samples()

    def should_save_outputs(self) -> bool:
        """Check if local output saving is enabled (benchmark best images)."""
        if not self.config:
            return False
        return bool(self.config.get("save_outputs", False))

    def should_save_checkpoint_outputs(self) -> bool:
        """Check if scaling checkpoint images should be saved locally.
        Uses save_checkpoint_structures so benchmark images (save_outputs) and
        scaling checkpoint images are controlled separately.
        """
        if not self.config:
            return False
        return bool(self.config.get("save_checkpoint_structures", False))

    def get_save_format(self) -> str:
        """Get the image format for local saving (default: jpg)."""
        # 1. Check main config first (user override from top-level config)
        fmt = None
        if self.config:
            fmt = self.config.get("save_format")
        
        # 2. Check logging_config (where defaults from DEFAULT_LOGGING_CONFIG are)
        if fmt is None:
            fmt = self.logging_config.get("save_format")
            
        # 3. Fallback to default
        if fmt is None:
            fmt = "jpg"
            
        fmt = str(fmt).lower()
        if fmt == "jpeg":
            return "jpg"
        if fmt not in ("jpg", "png"):
            return "jpg"
        return fmt

    def save_image_locally(
        self,
        image: Any,
        filename: str,
        task_index: Optional[int] = None,
        prompt: Optional[str] = None,
        subfolder: Optional[str] = None,
    ) -> Optional[str]:
        """Save an image to the local filesystem.
        
        Args:
            image: Image data (Tensor, PIL Image, or numpy array)
            filename: Name of the file (e.g., 'best.png', 'baseline.png')
            task_index: Optional task index for folder structure
            prompt: Optional prompt for folder naming
            subfolder: Optional subfolder within the task folder (e.g., 'iterations')
            
        Returns:
            Path to the saved image, or None if saving failed
        """
        # Benchmark/baseline images: save_outputs. Scaling checkpoint images: save_checkpoint_structures
        if subfolder == "checkpoints":
            if not self.should_save_checkpoint_outputs():
                return None
        else:
            if not self.should_save_outputs():
                return None

        try:
            import os
            from PIL import Image
            import numpy as np

            # Determine base directory from config
            base_dir = self.config.get("save_outputs_dir", "outputs")
            
            # Incorporate project and run name for organization
            project = self.project or "default_project"
            run_name = self.name or "run"
            
            # Sanitize run name and project for filesystem
            def sanitize(s: str) -> str:
                return "".join([c if c.isalnum() or c in ("-", "_") else "_" for c in s])

            path_parts = [base_dir, "images", sanitize(project), sanitize(run_name)]
            
            if task_index is not None:
                task_folder = f"task_{task_index:04d}"
                if prompt:
                    # Sanitize and truncate prompt for folder name
                    prompt_slug = sanitize(prompt.replace(" ", "_"))[:50]
                    task_folder += f"_{prompt_slug}"
                path_parts.append(task_folder)
            
            if subfolder:
                path_parts.append(sanitize(subfolder))
            
            save_dir = os.path.join(*path_parts)
            os.makedirs(save_dir, exist_ok=True)
            
            save_format = self.get_save_format()
            
            # Replace extension in filename if it exists, or append it
            if "." in filename:
                base_filename = os.path.splitext(filename)[0]
                filename = f"{base_filename}.{save_format}"
            else:
                filename = f"{filename}.{save_format}"
                
            save_path = os.path.join(save_dir, filename)

            # Convert to PIL Image
            img_obj = image
            if isinstance(img_obj, torch.Tensor):
                x = img_obj.detach().cpu()
                if x.dim() == 4:
                    x = x[0]
                if x.dim() == 3:
                    if x.max() <= 1.0:
                        x = (x.clamp(0, 1) * 255).to(torch.uint8)
                    else:
                        x = x.clamp(0, 255).to(torch.uint8)
                    img_obj = Image.fromarray(x.permute(1, 2, 0).numpy())
                elif x.dim() == 2:
                    if x.max() <= 1.0:
                        x = (x.clamp(0, 1) * 255).to(torch.uint8)
                    else:
                        x = x.clamp(0, 255).to(torch.uint8)
                    img_obj = Image.fromarray(x.numpy())
            elif isinstance(img_obj, np.ndarray):
                if img_obj.dtype != np.uint8:
                    if img_obj.max() <= 1.0:
                        img_obj = (img_obj * 255).astype(np.uint8)
                    else:
                        img_obj = img_obj.astype(np.uint8)
                img_obj = Image.fromarray(img_obj)
            
            if isinstance(img_obj, Image.Image):
                # Ensure compatibility for JPG (must be RGB, no alpha)
                if save_format == "jpg" and img_obj.mode in ("RGBA", "P"):
                    img_obj = img_obj.convert("RGB")
                
                # Save with appropriate quality for JPG
                if save_format == "jpg":
                    img_obj.save(save_path, "JPEG", quality=95)
                else:
                    img_obj.save(save_path)
                return save_path
            
            return None
        except Exception as e:
            print(f"[ERROR] Saving image locally: {e}")
            return None

    def _define_base_metrics(self) -> None:
        """Define step metrics and iteration/benchmark metrics for wandb tracking."""
        if not self.enable:
            return
        # Call parent to define base step metrics
        super()._define_base_metrics()

        if self.should_log_iteration_metrics():
            # Define iteration-level metrics (step by iteration_step)
            self.wandb.define_metric("iteration/*", step_metric="iteration_step")
            self.wandb.define_metric("iteration/reward_mean", step_metric="iteration_step")
            self.wandb.define_metric("iteration/reward_max", step_metric="iteration_step")
            self.wandb.define_metric("iteration/reward_min", step_metric="iteration_step")
            self.wandb.define_metric("iteration/reward_std", step_metric="iteration_step")
            self.wandb.define_metric("iteration/reward_median", step_metric="iteration_step")
            self.wandb.define_metric("iteration/evals_cum", step_metric="iteration_step")
            self.wandb.define_metric("iteration/time_s", step_metric="iteration_step")
            if self.should_log_iteration_operator_stats():
                self.wandb.define_metric("iteration/ops/*", step_metric="iteration_step")

        if self.should_log_benchmark_metrics():
            # Define benchmark-level metrics (step by task_step)
            self.wandb.define_metric("benchmark/*", step_metric="task_step")
            self.wandb.define_metric("benchmark/best_reward", step_metric="task_step")
            self.wandb.define_metric("benchmark/total_evals", step_metric="task_step")
            self.wandb.define_metric("benchmark/iterations", step_metric="task_step")
            self.wandb.define_metric("benchmark/wall_time_s", step_metric="task_step")
            self.wandb.define_metric("benchmark/running_average_best_reward", step_metric="task_step")
            if self.should_log_reward_curve():
                self.wandb.define_metric("benchmark/curve_last", step_metric="task_step")
            if self.should_log_benchmark_best_sample():
                # Define image metrics (no step_metric for images)
                self.wandb.define_metric("benchmark/best_image")
            if self.should_log_running_best_sample():
                self.wandb.define_metric("benchmark/running_best_image")

            # Define final summary metrics (no step)
            self.wandb.define_metric("benchmark/final_average_best_reward")
            self.wandb.define_metric("benchmark/num_tasks")

    @staticmethod
    def _to_serializable(obj: Any) -> Any:
        """Convert OmegaConf DictConfig/ListConfig to regular Python objects for JSON serialization."""
        if isinstance(obj, DictConfig):
            return OmegaConf.to_container(obj, resolve=True)
        elif isinstance(obj, dict):
            return {k: T2IWandbLogger._to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [T2IWandbLogger._to_serializable(item) for item in obj]
        return obj

    def format_context(self, context: Optional[Dict[str, Any]]) -> str:
        """Format context for console logging (image modality).
        
        Args:
            context: Problem context dictionary
            
        Returns:
            Formatted string with prompt
        """
        if context is None:
            return ""
        if "prompt" in context:
            return f"prompt='{context['prompt']}'"
        return ""

    def log_generation(
        self,
        prompt: str,
        images: Any,
        rewards: Optional[List[float]] = None,
        step: Optional[int] = None,
    ) -> None:
        """Log generated images to wandb.
        
        Args:
            prompt: Text prompt used for generation
            images: Tensor or list of images to log
            rewards: Optional list of reward values for each image
            step: Optional step number for tracking
        """
        if not self.enable:
            return
        # Convert tensor batches to list of images compatible with wandb.Image
        imgs = images
        if isinstance(imgs, torch.Tensor):
            x = imgs.detach().cpu()
            if x.dim() == 3:
                x = x.unsqueeze(0)
            # assume [B, C, H, W] in [0,1]
            x = (x.clamp(0, 1) * 255).to(torch.uint8)
            imgs = [x[i].permute(1, 2, 0).numpy() for i in range(x.shape[0])]
        elif isinstance(imgs, list):
            imgs = imgs
        else:
            imgs = [imgs]

        wandb_images = []
        for i, img in enumerate(imgs):
            cap = f"prompt: {prompt}"
            if rewards is not None and i < len(rewards):
                cap += f" | reward: {rewards[i]:.4f}"
            wandb_images.append(self.wandb.Image(img, caption=cap))

        payload: Dict[str, Any] = {"iteration/generated_images": wandb_images}
        if step is not None:
            payload["iteration_step"] = step
        self.wandb.log(payload)

    def log_iteration_metrics(
        self,
        iteration: int,
        prompt: str,
        rewards: torch.Tensor | List[float],
        *,
        task_index: Optional[int] = None,
        evals_cum: Optional[int] = None,
        time_s: Optional[float] = None,
        operator_stats: Optional[Dict[str, Any]] = None,
        topk_images: Optional[List[Any]] = None,
        topk_rewards: Optional[List[float]] = None,
    ) -> None:
        """Log iteration-level metrics with proper step tracking.
        
        Args:
            iteration: Current iteration number
            prompt: Text prompt for this task
            rewards: Tensor or list of reward values for this iteration
            task_index: Optional task index for tracking which task this iteration belongs to
            evals_cum: Cumulative number of evaluations
            time_s: Wall time for this iteration
            operator_stats: Optional operator statistics (mutation, combination, etc.)
            topk_images: Optional top-k images to log
            topk_rewards: Optional rewards for top-k images
        """
        if not self.enable:
            return
        if not self.should_log_iteration_metrics():
            return

        if isinstance(rewards, list):
            rewards_t = torch.tensor(rewards)
        else:
            rewards_t = rewards.detach().cpu().view(-1)

        # Build metrics with proper step key for iteration-level tracking
        metrics: Dict[str, Any] = {
            "iteration_step": iteration,
            "iteration/reward_mean": float(rewards_t.mean().item()),
            "iteration/reward_median": float(rewards_t.median().item()),
            "iteration/reward_std": float(rewards_t.std(unbiased=False).item()) if rewards_t.numel() > 1 else 0.0,
            "iteration/reward_min": float(rewards_t.min().item()),
            "iteration/reward_max": float(rewards_t.max().item()),
        }
        
        # Add task index if provided for better tracking
        if task_index is not None:
            metrics["iteration/task_index"] = int(task_index)
        
        # Optional extras
        if evals_cum is not None:
            metrics["iteration/evals_cum"] = int(evals_cum)
        if time_s is not None:
            metrics["iteration/time_s"] = float(time_s)
        if operator_stats and self.should_log_iteration_operator_stats():
            # Convert any DictConfig objects to regular dicts for JSON serialization
            operator_stats = self._to_serializable(operator_stats)
            # e.g., {"mutation": {"count": 8, "best": 0.8}, "combination": {...}}
            for k, v in operator_stats.items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        metrics[f"iteration/ops/{k}.{kk}"] = vv
                else:
                    metrics[f"iteration/ops/{k}"] = v

        # Reward histograms removed per request

        self.wandb.log(metrics)

        # Optionally log top-k images with captions
        if topk_images is not None and self.should_log_iteration_samples():
            limit = self.get_iteration_sample_limit()
            if limit > 0:
                topk_images = topk_images[:limit]
                if topk_rewards is not None:
                    topk_rewards = topk_rewards[:limit]
            caps = None
            if topk_rewards is not None:
                caps = [f"iter={iteration} | R={float(r):.4f} | {prompt}" for r in topk_rewards]
            self.log_images("iteration/topk_images", topk_images, captions=caps, step_key="iteration_step", step=iteration)
            
            # Local saving of top-k images
            if self.should_save_outputs():
                save_format = self.get_save_format()
                for i, img in enumerate(topk_images):
                    reward_suffix = f"_R{topk_rewards[i]:.4f}" if topk_rewards is not None and i < len(topk_rewards) else ""
                    filename = f"iter{iteration:03d}_top{i}{reward_suffix}.{save_format}"
                    self.save_image_locally(
                        img, 
                        filename, 
                        task_index=task_index, 
                        prompt=prompt, 
                        subfolder="iterations"
                    )

    def log_benchmark_summary(
        self,
        prompt: str,
        solver_name: str,
        *,
        task_index: int,
        best_reward: float,
        total_iterations: int,
        total_evals: int,
        wall_time_s: Optional[float] = None,
        best_image: Optional[Any] = None,
        running_best_image: Optional[Any] = None,
        reward_curve: Optional[List[float]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log benchmark-level summary metrics with proper step tracking.
        
        Args:
            prompt: Text prompt for this task
            solver_name: Name of the solver used
            task_index: Task index for tracking (used as step)
            best_reward: Best reward achieved for this task
            total_iterations: Total number of iterations run
            total_evals: Total number of evaluations performed
            wall_time_s: Wall clock time taken
            best_image: Best image generated
            reward_curve: List of reward values over iterations
            extra: Additional metrics to log
        """
        if not self.enable or not self.should_log_benchmark_metrics():
            return
        # Build payload with task_step for benchmark-level tracking
        payload: Dict[str, Any] = {
            "task_step": task_index,
            "benchmark/solver": solver_name,
            "benchmark/best_reward": float(best_reward),
            "benchmark/iterations": int(total_iterations),
            "benchmark/total_evals": int(total_evals),
        }
        if wall_time_s is not None:
            payload["benchmark/wall_time_s"] = float(wall_time_s)
        if reward_curve is not None and self.should_log_reward_curve():
            # Log last only; histogram removed per request
            payload["benchmark/curve_last"] = float(reward_curve[-1]) if len(reward_curve) > 0 else 0.0
        if extra:
            # Convert any DictConfig objects to regular dicts for JSON serialization
            extra = self._to_serializable(extra)
            for k, v in extra.items():
                # Skip config values that shouldn't be plotted
                if k in ("solver_kwargs", "algorithm_kwargs", "algo_kwargs") and isinstance(v, dict):
                    continue
                elif k in ("prompt_index", "num_tasks", "running_average_best_reward"):
                    # These are important metrics to log
                    payload[f"benchmark/{k}"] = v
                elif k not in ("gpu_mem_max_alloc_bytes", "gpu_mem_max_reserved_bytes", "gpu_mem_max_alloc_mib", "gpu_mem_max_reserved_mib"):
                    # Skip detailed GPU memory metrics to reduce clutter
                    continue

        def _to_wandb_image(img: Any, caption: str):
            try:
                img_obj = img
                if isinstance(img_obj, torch.Tensor):
                    x = img_obj.detach().cpu()
                    if x.dim() == 4:
                        x = x[0]
                    if x.dim() == 3:
                        if x.max() <= 1.0:
                            x = (x.clamp(0, 1) * 255).to(torch.uint8)
                        else:
                            x = x.clamp(0, 255).to(torch.uint8)
                        img_obj = x.permute(1, 2, 0).numpy()
                    elif x.dim() == 2:
                        if x.max() <= 1.0:
                            x = (x.clamp(0, 1) * 255).to(torch.uint8)
                        else:
                            x = x.clamp(0, 255).to(torch.uint8)
                        img_obj = x.numpy()
                if isinstance(img_obj, list) and len(img_obj) == 1:
                    img_obj = img_obj[0]
                return self.wandb.Image(img_obj, caption=caption)
            except Exception as exc:
                print(f"[ERROR] Converting image to wandb image: {exc}")
                return None

        # Single best image (per prompt)
        if best_image is not None and self.should_log_benchmark_best_sample():
            wb_img = _to_wandb_image(best_image, f"Task {task_index} | {prompt} | best={best_reward:.4f}")
            if wb_img is not None:
                payload["benchmark/best_image"] = wb_img
                if "task_step" not in payload:
                    payload["task_step"] = task_index

        # Running best image across prompts
        # if running_best_image is not None and self.should_log_running_best_sample():
        #     pass
        #     wb_img = _to_wandb_image(running_best_image, f"Running best up to task {task_index} | {prompt}")
        #     if wb_img is not None:
        #         payload["benchmark/running_best_image"] = wb_img
        #         if "task_step" not in payload:
        #             payload["task_step"] = task_index

        self.wandb.log(payload)


