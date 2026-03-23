"""Protein-specific experiment logger."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from copy import deepcopy
import os

from .molecule_logger import MoleculeLogger


class ProteinLogger(MoleculeLogger):
    """Logger for protein optimization experiments.
    
    Extends MoleculeLogger with protein-specific context formatting,
    metric tracking (n_residues, fold codes, etc.), and benchmark
    summary logging similar to T2I logger.
    """

    DEFAULT_LOGGING_CONFIG: Dict[str, Any] = {
        "wandb": {
            "iteration": {
                "enabled": True,
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
        },
        "terminal": {
            "iteration": {
                "enabled": True,
                "verbosity": "minimal",
            },
            "benchmark": {
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
        if logging_config is None and config is not None:
            logging_config = config.get("logging")
        super().__init__(
            project=project,
            name=name,
            config=config,
            enable=enable,
            logging_config=logging_config,
            wandb_dir=wandb_dir,
        )
        
        # Override with protein-specific defaults if needed
        merged_cfg = deepcopy(self.DEFAULT_LOGGING_CONFIG)
        if logging_config:
            merged_cfg = self._merge_logging_config(merged_cfg, logging_config)
        self.logging_config: Dict[str, Any] = self._normalize_logging_config(merged_cfg)
        
        # Extract terminal output control
        terminal_cfg = self.logging_config.get("terminal", {})
        self.iteration_logging_enabled = terminal_cfg.get("iteration", {}).get("enabled", True)
        self.iteration_verbosity = terminal_cfg.get("iteration", {}).get("verbosity", "minimal")
        self.benchmark_logging_enabled = terminal_cfg.get("benchmark", {}).get("enabled", True)
        self.benchmark_verbosity = terminal_cfg.get("benchmark", {}).get("verbosity", "summary")
        
        # Define wandb metrics for proteins
        self._define_protein_metrics()

    def _merge_logging_config(self, base: Dict[str, Any], logging_config: Dict[str, Any]) -> Dict[str, Any]:
        """Deep-merge user provided logging config with defaults."""
        merged = deepcopy(base)
        
        def _merge(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
            for key, value in src.items():
                if isinstance(value, dict) and isinstance(dst.get(key), dict):
                    dst[key] = _merge(dict(dst.get(key, {})), value)
                else:
                    dst[key] = value
            return dst
        return _merge(dict(merged), logging_config or {})

    def _normalize_logging_config(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize logging config structure."""
        cfg = deepcopy(cfg)
        # Ensure both sections exist with defaults
        wandb_section = cfg.setdefault("wandb", {})
        terminal_section = cfg.setdefault("terminal", {})
        
        for section_name in ["iteration", "benchmark"]:
            if section_name not in wandb_section:
                wandb_section[section_name] = self.DEFAULT_LOGGING_CONFIG["wandb"].get(section_name, {})
            if section_name not in terminal_section:
                terminal_section[section_name] = self.DEFAULT_LOGGING_CONFIG["terminal"].get(section_name, {})
        
        return cfg

    def _wandb_section(self, name: str) -> Dict[str, Any]:
        """Get wandb logging section."""
        return self.logging_config.get("wandb", {}).get(name, {})
    
    def _terminal_section(self, name: str) -> Dict[str, Any]:
        """Get terminal output section."""
        return self.logging_config.get("terminal", {}).get(name, {})

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

    def should_print_benchmark(self) -> bool:
        return bool(self._terminal_section("benchmark").get("enabled", True))
    
    def get_benchmark_verbosity(self) -> str:
        return str(self._terminal_section("benchmark").get("verbosity", "summary"))

    def _define_protein_metrics(self) -> None:
        """Define protein-specific wandb metrics."""
        if not self.enable or self.wandb is None:
            return
        try:
            # Define iteration-level metrics (step by iteration)
            wandb_section = self._wandb_section("iteration")
            if wandb_section.get("enabled", True) and wandb_section.get("log_metrics", True):
                self.wandb.define_metric("iteration_step")
                self.wandb.define_metric("iteration/*", step_metric="iteration_step")
                self.wandb.define_metric("iteration/diversity", step_metric="iteration_step")
                self.wandb.define_metric("iteration/novelty", step_metric="iteration_step")
            
            # Define benchmark-level metrics (step by task_step)
            if self.should_log_benchmark_metrics():
                self.wandb.define_metric("task_step")
                self.wandb.define_metric("benchmark/*", step_metric="task_step")
                self.wandb.define_metric("benchmark/best_reward", step_metric="task_step")
                self.wandb.define_metric("benchmark/total_evals", step_metric="task_step")
                self.wandb.define_metric("benchmark/iterations", step_metric="task_step")
                self.wandb.define_metric("benchmark/wall_time_s", step_metric="task_step")
                self.wandb.define_metric("benchmark/gpu_mem_MiB", step_metric="task_step")
                self.wandb.define_metric("benchmark/running_average_best_reward", step_metric="task_step")
                if self.should_log_reward_curve():
                    self.wandb.define_metric("benchmark/curve_last", step_metric="task_step")
                # Final summary metrics (no step)
                self.wandb.define_metric("benchmark/final_average_best_reward")
                self.wandb.define_metric("benchmark/num_tasks")
        except Exception as e:
            print(f"[WARNING] Could not define protein wandb metrics: {e}")

    def format_context(self, context: Optional[Dict[str, Any]]) -> str:
        """Format context for console logging (protein modality).
        
        Args:
            context: Problem context dictionary
            
        Returns:
            Formatted string with protein-specific info (n_residues, fold_code, etc.)
        """
        if context is None:
            return ""
        
        parts = []
        
        # Show n_residues (protein length)
        if "n_residues" in context:
            parts.append(f"n_res={context['n_residues']}")
        
        # Show fold code if conditional
        if "target_fold" in context:
            parts.append(f"fold={context['target_fold']}")
        elif "fold_code" in context:
            parts.append(f"fold={context['fold_code']}")
        
        # Show mode if present
        if "mode" in context:
            parts.append(f"mode={context['mode']}")
        
        # Show length index for tracking
        if "length_index" in context:
            parts.append(f"task={context['length_index']}")
        
        # Fallback to molecule-style targets if present
        if not parts and "targets" in context:
            targets = context["targets"]
            if isinstance(targets, dict):
                target_strs = [f"{k}={v:.3f}" for k, v in targets.items()]
                return f"targets=[{', '.join(target_strs)}]"
        
        return " | ".join(parts) if parts else ""

    def log_iteration_console(
        self,
        iteration: int,
        rewards: Any,
        context: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log iteration to console with protein-specific formatting.
        
        Args:
            iteration: Current iteration number
            rewards: Reward tensor or list
            context: Problem context (for formatting)
            extra: Additional info (batch stats, etc.)
        """
        if not self.iteration_logging_enabled:
            return
        
        import torch
        
        # Extract reward stats
        if isinstance(rewards, torch.Tensor):
            mean_val = float(rewards.mean().item()) if rewards.numel() > 0 else 0.0
            max_val = float(rewards.max().item()) if rewards.numel() > 0 else 0.0
            min_val = float(rewards.min().item()) if rewards.numel() > 0 else 0.0
        else:
            import numpy as np
            rewards_arr = np.array(rewards)
            mean_val = float(np.mean(rewards_arr)) if len(rewards_arr) > 0 else 0.0
            max_val = float(np.max(rewards_arr)) if len(rewards_arr) > 0 else 0.0
            min_val = float(np.min(rewards_arr)) if len(rewards_arr) > 0 else 0.0
        
        # Extract diversity and novelty from context
        diversity_str = ""
        novelty_str = ""
        if context:
            # Check for diversity and novelty in context (stored by problem)
            if "diversity" in context:
                diversity_str = f" div={context['diversity']:.2f}"
            if "novelty" in context:
                novelty_str = f" nov={context['novelty']:.2f}"
        
        # Format context string
        context_str = self.format_context(context)
        if context_str:
            context_str = f" {context_str}"
        
        # Build output based on verbosity
        if self.iteration_verbosity == "minimal":
            print(f"[Protein] iter={iteration}{context_str} best={max_val:.4f}{diversity_str}{novelty_str}")
        elif self.iteration_verbosity == "detailed":
            extra_info = ""
            if extra:
                if "n_eval" in extra:
                    extra_info += f" evals={extra['n_eval']}"
                if "batch_size" in extra:
                    extra_info += f" batch={extra['batch_size']}"
            print(f"[Protein] iter={iteration}{context_str} mean={mean_val:.4f} max={max_val:.4f} min={min_val:.4f}{diversity_str}{novelty_str}{extra_info}")
        else:  # debug
            # Calculate std
            if isinstance(rewards, torch.Tensor):
                std_val = float(rewards.std().item()) if rewards.numel() > 1 else 0.0
            else:
                import numpy as np
                rewards_arr = np.array(rewards)
                std_val = float(np.std(rewards_arr)) if len(rewards_arr) > 1 else 0.0
            
            extra_info = ""
            if extra:
                for k, v in extra.items():
                    if isinstance(v, (int, float)):
                        extra_info += f" {k}={v}"
            print(f"[Protein] iter={iteration}{context_str} mean={mean_val:.4f} max={max_val:.4f} min={min_val:.4f} std={std_val:.4f}{diversity_str}{novelty_str}{extra_info}")

    def _create_protein_visualization(self, atom37: Any, caption: str) -> Optional[Any]:
        """Create a visualization of a protein structure.
        
        Args:
            atom37: Protein structure as atom37 tensor [N, 37, 3] or numpy array
            caption: Caption for the visualization
            
        Returns:
            PIL Image for wandb logging, or None if visualization fails
        """
        try:
            from .protein_visualizer import ProteinVisualizer
            
            visualizer = ProteinVisualizer(image_size=(800, 600), dpi=100)
            img = visualizer.atom37_to_image(atom37, title=caption)
            
            return img
        except ImportError:
            # Fallback: try to create PDB string if visualizer not available
            try:
                import torch
                import numpy as np
                
                if isinstance(atom37, torch.Tensor):
                    coords = atom37.detach().cpu().numpy()
                else:
                    coords = np.array(atom37)
                
                if coords.ndim == 3 and coords.shape[1] >= 2:
                    ca_coords = coords[:, 1, :]
                else:
                    ca_coords = coords[:, 0, :] if coords.ndim == 3 else coords
                
                pdb_lines = [f"REMARK {caption}"]
                for i, ca in enumerate(ca_coords):
                    pdb_lines.append(
                        f"ATOM  {i+1:5d}  CA  ALA A{i+1:4d}    "
                        f"{ca[0]:8.3f}{ca[1]:8.3f}{ca[2]:8.3f}  1.00 20.00           C  "
                    )
                pdb_lines.append("END")
                return "\n".join(pdb_lines)
            except Exception as e:
                print(f"[WARNING] Failed to create protein visualization: {e}")
                return None
        except Exception as e:
            print(f"[WARNING] Failed to create protein visualization: {e}")
            return None

    def log_benchmark_summary(
        self,
        context: Dict[str, Any],
        solver_name: str,
        *,
        task_index: int,
        best_reward: float,
        total_iterations: int,
        total_evals: int,
        wall_time_s: Optional[float] = None,
        best_protein: Optional[Any] = None,
        running_best_protein: Optional[Any] = None,
        best_molecule: Optional[Any] = None,  # Accept molecule name for compatibility
        running_best_molecule: Optional[Any] = None,  # Accept molecule name for compatibility
        reward_curve: Optional[List[float]] = None,
        running_average_best_reward: Optional[float] = None,
        gpu_mem_max_alloc_bytes: Optional[int] = None,
        gpu_mem_max_reserved_bytes: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log benchmark-level summary metrics with proper step tracking.
        
        Args:
            context: Problem context dictionary (for formatting)
            solver_name: Name of the solver used
            task_index: Task index for tracking (used as step)
            best_reward: Best reward achieved for this task
            total_iterations: Total number of iterations run
            total_evals: Total number of evaluations performed
            wall_time_s: Wall clock time taken
            best_protein: Best protein structure (atom37 tensor)
            running_best_protein: Best protein structure across all tasks so far
            best_molecule: Best molecule/protein (for compatibility with molecule benchmarks)
            running_best_molecule: Running best molecule/protein (for compatibility)
            reward_curve: List of reward values over iterations
            running_average_best_reward: Running average of best rewards across tasks
            gpu_mem_max_alloc_bytes: Maximum GPU memory allocated
            gpu_mem_max_reserved_bytes: Maximum GPU memory reserved
            extra: Additional metrics to log
        """
        # Map molecule names to protein names for compatibility
        if best_protein is None and best_molecule is not None:
            best_protein = best_molecule
        if running_best_protein is None and running_best_molecule is not None:
            running_best_protein = running_best_molecule
        
        if not self.enable or not self.should_log_benchmark_metrics():
            pass  # Still log to terminal even if wandb is disabled
        
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
        
        if gpu_mem_max_alloc_bytes is not None:
            gpu_mem_mib = gpu_mem_max_alloc_bytes / (1024 * 1024)
            payload["benchmark/gpu_mem_MiB"] = float(gpu_mem_mib)
        
        if running_average_best_reward is not None:
            payload["benchmark/running_average_best_reward"] = float(running_average_best_reward)
        
        if reward_curve is not None and self.should_log_reward_curve():
            payload["benchmark/curve_last"] = float(reward_curve[-1]) if len(reward_curve) > 0 else 0.0
        
        if extra:
            for k, v in extra.items():
                if isinstance(v, (int, float, str, bool)):
                    payload[f"benchmark/{k}"] = v
        
        # Log protein visualizations as images
        context_str = self.format_context(context)
        if best_protein is not None and self.should_log_benchmark_best_sample():
            vis = self._create_protein_visualization(
                best_protein,
                f"Task {task_index} | {context_str} | best={best_reward:.4f}"
            )
            if vis is not None:
                # Check if it's a PIL Image (new visualization) or string (fallback PDB)
                try:
                    from PIL import Image
                    if isinstance(vis, Image.Image):
                        # Log as wandb.Image for interactive visualization
                        import wandb
                        payload["benchmark/best_protein"] = wandb.Image(
                            vis, 
                            caption=f"Task {task_index} | {context_str} | best={best_reward:.4f}"
                        )
                    else:
                        # Fallback: log as PDB string
                        payload["benchmark/best_protein_pdb"] = vis
                except Exception as e:
                    # Fallback: log as string
                    payload["benchmark/best_protein_pdb"] = str(vis) if vis is not None else None
        
        if running_best_protein is not None and self.should_log_running_best_sample():
            vis = self._create_protein_visualization(
                running_best_protein,
                f"Running best up to task {task_index} | {context_str}"
            )
            if vis is not None:
                try:
                    from PIL import Image
                    if isinstance(vis, Image.Image):
                        import wandb
                        payload["benchmark/running_best_protein"] = wandb.Image(
                            vis,
                            caption=f"Running best up to task {task_index} | {context_str}"
                        )
                    else:
                        payload["benchmark/running_best_protein_pdb"] = vis
                except Exception as e:
                    payload["benchmark/running_best_protein_pdb"] = str(vis) if vis is not None else None
        
        # Log to wandb
        if self.enable and self.wandb is not None:
            try:
                self.wandb.log(payload)
            except Exception as e:
                print(f"[WARNING] Failed to log benchmark summary to wandb: {e}")
        
        # Log to terminal
        if self.should_print_benchmark():
            verbosity = self.get_benchmark_verbosity()
            if verbosity == "summary":
                mem_str = ""
                if gpu_mem_max_alloc_bytes is not None:
                    mem_mib = gpu_mem_max_alloc_bytes / (1024 * 1024)
                    mem_str = f" | gpu_mem_MiB={mem_mib:.1f}"
                avg_str = ""
                if running_average_best_reward is not None:
                    avg_str = f" | avg={running_average_best_reward:.4f}"
                print(f"[Protein] {context_str} | solver={solver_name} | best={best_reward:.4f}{avg_str} | evals={total_evals}{mem_str} | time_s={wall_time_s:.2f}")
            elif verbosity == "detailed":
                mem_str = ""
                if gpu_mem_max_alloc_bytes is not None:
                    mem_mib = gpu_mem_max_alloc_bytes / (1024 * 1024)
                    mem_reserved_mib = gpu_mem_max_reserved_bytes / (1024 * 1024) if gpu_mem_max_reserved_bytes else None
                    mem_str = f" | gpu_mem_alloc_MiB={mem_mib:.1f}"
                    if mem_reserved_mib:
                        mem_str += f" gpu_mem_reserved_MiB={mem_reserved_mib:.1f}"
                avg_str = ""
                if running_average_best_reward is not None:
                    avg_str = f" | avg={running_average_best_reward:.4f}"
                curve_str = ""
                if reward_curve and len(reward_curve) > 0:
                    curve_str = f" | curve_last={reward_curve[-1]:.4f}"
                print(f"[Protein] {context_str} | solver={solver_name} | best={best_reward:.4f}{avg_str} | evals={total_evals} | iterations={total_iterations}{curve_str}{mem_str} | time_s={wall_time_s:.2f}")

            if self._last_saved_dir:
                print(f"  Task complete. Results saved to: {self._last_saved_dir}")

        # Optional: export best protein to PDB for later py3Dmol visualization
        if self.should_save_outputs() and best_protein is not None:
            try:
                # Add best_reward to context for filename tracking
                ctx = {**(context or {}), "reward": float(best_reward)}
                self.save_best_protein_pdb(task_index=task_index, best_protein=best_protein, context=ctx)
            except Exception:
                pass

    def save_best_protein_pdb(self, *, task_index: int, best_protein: Any, context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Save a protein to a PDB file, reconstructing backbone if necessary for valid NGLView display."""
        if not self.should_save_outputs():
            return None
        
        # Explicitly check for None or empty tensors to avoid ambiguity errors
        if best_protein is None:
            return None
        
        import torch
        if isinstance(best_protein, torch.Tensor) and best_protein.numel() == 0:
            return None

        # Determine output directory and path
        run_name = str(self._run_id)
        try:
            from hydra.utils import get_original_cwd
            base_dir = get_original_cwd()
        except (ImportError, ValueError):
            base_dir = os.getcwd()

        outputs_base = self.config.get("save_outputs_dir", "outputs")
        # Organize in <outputs_base>/proteins/<project>/<run_name>/<task_prefix>
        project = self.project or "default_project"
        def sanitize(s: str) -> str:
            return "".join([c if c.isalnum() or c in ("-", "_") else "_" for c in s])

        task_prefix = f"task_{int(task_index):04d}"
        out_dir = os.path.join(base_dir, outputs_base, "proteins", sanitize(project), sanitize(run_name), task_prefix)
        os.makedirs(out_dir, exist_ok=True)
        self._saved_dirs.add(os.path.dirname(out_dir)) # Add parent run directory
        
        # Construct a descriptive filename
        # Order: iteration -> size -> fold
        filename_parts = []
        
        if context:
            # 1. Iteration if present (first for sorting)
            iteration = context.get("iteration")
            if iteration is not None:
                filename_parts.append(f"iter_{int(iteration):03d}")
            
            # 2. Size (n_res for proteins, check model_config too)
            n_res = context.get("n_residues")
            if not n_res and "model_config" in context:
                n_res = context["model_config"].get("n_residues")
            if n_res:
                filename_parts.append(f"{n_res}res")
            
            # 3. Fold/Mode info
            fold = context.get("target_fold") or context.get("fold_code")
            if fold:
                filename_parts.append(str(fold).replace('.', '_'))
            elif context.get("mode"):
                filename_parts.append(context['mode'])

        # Determine suffix based on whether this is an iteration-level save
        is_iter = context and context.get("iteration") is not None
        suffix = "iter" if is_iter else "best"
        
        # Add reward to filename if present
        reward_suffix = ""
        if context and context.get("reward") is not None:
            try:
                reward_val = float(context["reward"])
                reward_suffix = f"_rew{reward_val:.4f}"
            except (ValueError, TypeError):
                pass

        if not filename_parts:
            filename = f"{task_prefix}_{suffix}{reward_suffix}.pdb"
        else:
            filename = "_".join(filename_parts) + f"_{suffix}{reward_suffix}.pdb"
            
        out_path = os.path.join(out_dir, filename)

        # Prepare metadata for PDB header
        header_lines = []
        header_lines.append(f"HEADER    PROTEIN DESIGN OPTIMIZATION             {run_name[:10].upper():<10}")
        header_lines.append(f"TITLE     Protein Optimization: {run_name}")
        if context:
            it = context.get("iteration", "final")
            rew = context.get("reward")
            header_lines.append(f"REMARK 001 TASK INDEX: {task_index}")
            header_lines.append(f"REMARK 001 ITERATION: {it}")
            if rew is not None:
                header_lines.append(f"REMARK 001 REWARD: {float(rew):.4f}")

        # 1. Try to save using Proteina's native writer as well (requested by user)
        try:
            from ..utils.path_utils import setup_proteina_path
            setup_proteina_path()
            
            from proteinfoundation.utils.ff_utils.pdb_utils import write_prot_to_pdb
            import torch
            
            filename_proteina = filename.replace(".pdb", "_proteina.pdb")
            out_path_proteina = os.path.join(out_dir, filename_proteina)
            coords_raw = best_protein
            if isinstance(coords_raw, torch.Tensor):
                coords_raw = coords_raw.detach().cpu().numpy()
            
            # Proteina writer handles [N, 37, 3] or [N, 3]
            write_prot_to_pdb(coords_raw, out_path_proteina, no_indexing=True, overwrite=True)
            
            # Note: We don't prepend header to Proteina output as it's a binary-like writer
        except Exception as e:
            # Print warning but continue to manual reconstruction
            print(f"[WARNING] Proteina native PDB writer failed: {e}")

        # 2. Proceed with manual reconstruction for the main "fixed" PDB (to ensure NGLView compatibility)
        try:
            import torch
            import numpy as np

            # Ensure best_protein is a numpy array for the reconstruction logic
            if isinstance(best_protein, torch.Tensor):
                coords_t = best_protein.detach().cpu()
            else:
                coords_t = torch.tensor(best_protein).detach().cpu()
            
            if coords_t.ndim == 3:
                # If it's [N, 37, 3], atom index 1 is CA
                ca_coords = coords_t[:, 1, :].numpy() if coords_t.shape[1] > 1 else coords_t[:, 0, :].numpy()
            else:
                ca_coords = coords_t.numpy()
            
            n_residues = ca_coords.shape[0]
            if n_residues == 0:
                print("[WARNING] Skipping PDB save, zero residues found.")
                return None

            # ... rest of logic ...


            # 2. Reconstruct backbone atoms to ensure physical validity and NGLView compatibility
            # We follow the path of CA atoms but ensure 3.8A spacing and place N, C, O, CB
            
            # Step A: Normalize CA-CA distances to ~3.8A to fix "unphysical distance" issues
            fixed_ca = [ca_coords[0]]
            for i in range(1, n_residues):
                prev = fixed_ca[-1]
                curr = ca_coords[i]
                diff = curr - prev
                dist = np.linalg.norm(diff)
                direction = diff / dist if dist > 1e-6 else np.array([1.0, 0.0, 0.0])
                fixed_ca.append(prev + direction * 3.8)
            fixed_ca = np.array(fixed_ca)

            # Step B: Place backbone atoms using idealized geometry
            lines = list(header_lines)
            atom_idx = 1
            for i in range(n_residues):
                res_idx = i + 1
                ca = fixed_ca[i]
                
                # Determine local orientation based on segment direction
                if i < n_residues - 1:
                    z_axis = fixed_ca[i+1] - ca
                else:
                    z_axis = ca - fixed_ca[i-1]
                
                z_norm = np.linalg.norm(z_axis)
                z_axis = z_axis / z_norm if z_norm > 1e-10 else np.array([0, 0, 1.0])
                
                # Define a local coordinate system (Gram-Schmidt-ish)
                perp = np.array([0, 1.0, 0])
                if abs(z_axis[1]) > 0.9:
                    perp = np.array([1.0, 0, 0])
                
                x_axis = np.cross(perp, z_axis)
                x_axis /= np.linalg.norm(x_axis)
                y_axis = np.cross(z_axis, x_axis)
                
                # Rotation matrix from local to global
                R = np.stack([x_axis, y_axis, z_axis], axis=1)
                
                # Idealized local coordinates for ALA (CA at origin)
                # These are roughly oriented such that the CA-CA axis is Z
                # Coordinates are chosen to satisfy bond lengths:
                # CA-C ~ 1.53, CA-N ~ 1.46, C-N ~ 1.33, CA-CB ~ 1.5
                local_atoms = {
                    'N':  np.array([ 0.94,  0.00, -1.12]),
                    'CA': np.array([ 0.00,  0.00,  0.00]),
                    'C':  np.array([ 0.70,  0.00,  1.35]),
                    'O':  np.array([ 1.62,  0.00,  2.15]),
                    'CB': np.array([-1.20,  0.90,  0.00])
                }
                
                for atom_name in ['N', 'CA', 'C', 'O', 'CB']:
                    pos = ca + R @ local_atoms[atom_name]
                    # Write ATOM record in strict PDB format (exactly 80 chars)
                    # 1-6: "ATOM  ", 7-11: serial, 13-16: name, 17: altLoc, 18-20: resName, 21: chainID, 22: chainID, 23-26: resSeq, 27: iCode, 31-38: x, 39-46: y, 47-54: z, 55-60: occupancy, 61-66: tempFactor, 77-78: element
                    name_field = f" {atom_name:<3}" if len(atom_name) < 4 else atom_name
                    line = (f"ATOM  {atom_idx:>5} {name_field:4s} ALA A{res_idx:>4}    "
                            f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}  1.00  0.00           {atom_name[0]:>2}")
                    # Ensure the line is exactly 80 characters long
                    lines.append(line.ljust(80))
                    atom_idx += 1
            
            lines.append(f"TER   {atom_idx:>5}      ALA A{n_residues:>4}")
            lines.append("END")
            
            with open(out_path, "w") as f:
                f.write("\n".join(lines) + "\n")
            
            return out_path

        except Exception as e:
            print(f"[WARNING] Failed to save fixed PDB: {e}")
            return None


        return None

