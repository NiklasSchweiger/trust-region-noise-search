"""Molecule-specific experiment logger."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from copy import deepcopy
import os

import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf

from .base import WandbLogger, _as_plain_dict
from .molecule_visualizer import MoleculeVisualizer, visualize_best_molecules


class MoleculeLogger(WandbLogger):
    """Wandb logger with helpers for molecule optimization experiments.
    
    Tracks molecule-specific metrics like:
    - Property predictions (QED, SA, gap, homo, lumo, etc.)
    - Stability metrics (atom stability, molecule stability)
    - Target distances
    - Molecular structures (if available)
    """

    DEFAULT_LOGGING_CONFIG: Dict[str, Any] = {
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
                "log_checkpoint_data_samples": True,
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
        
        # Initialize molecular visualizer
        self.visualizer = MoleculeVisualizer()
        
        # Initialize run name for unique local logging if no name provided
        run_base = self.name or self.config.get("wandb_name") or self.config.get("hydra_job_name") or "run"
        import datetime
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # If it's a wandb run or specific name, it's already unique enough
        # Otherwise, add timestamp to ensure 'run' folders don't collide
        if run_base in ("run", "molecule", "protein", "interactive"):
            self._run_id = f"{run_base}_{timestamp}"
        else:
            self._run_id = run_base
        
        # Initialize molecule-specific metric tracking
        self.property_history: Dict[str, List[float]] = {}
        self.stability_history: List[float] = []
        self.target_distance_history: List[float] = []
        # Scaling tracking across tasks
        self._scaling_running_sums: Dict[int, float] = {}
        self._scaling_running_counts: Dict[int, int] = {}
        
        # Track saved directories for summary print
        self._saved_dirs: set[str] = set()
        self._last_saved_dir: Optional[str] = None

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
                    "log_data_samples": iter_old.get("log_data_samples", iter_old.get("log_molecules", False)),
                    "data_sample_limit": iter_old.get("data_sample_limit", iter_old.get("topk_molecules", 4)),
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
                    "log_best_data_sample": bench_old.get("log_best_data_sample", bench_old.get("log_best_molecule", True)),
                    "log_running_best_data_sample": bench_old.get("log_running_best_data_sample", bench_old.get("log_running_best_molecule", True)),
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
                    "log_checkpoint_data_samples": scaling_old.get("log_checkpoint_data_samples", scaling_old.get("log_checkpoint_molecules", True)),
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

    def configure_logging(self, logging_config: Optional[Dict[str, Any]]) -> None:
        """Update logging configuration."""
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
        return bool(scaling_cfg.get("log_checkpoint_data_samples", True) and scaling_cfg.get("enabled", False))
    
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

    # Backward-compat helper names (molecule-specific)
    def should_log_iteration_molecules(self) -> bool:
        return self.should_log_iteration_samples()

    def get_iteration_topk_molecules(self) -> int:
        return self.get_iteration_sample_limit()

    def should_log_benchmark_best_molecule(self) -> bool:
        return self.should_log_benchmark_best_sample()

    def should_log_running_best_molecule(self) -> bool:
        return self.should_log_running_best_sample()

    def should_log_scaling_molecules(self) -> bool:
        return self.should_log_scaling_samples()

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
            self.wandb.define_metric("iteration/stability", step_metric="iteration_step")
            self.wandb.define_metric("iteration/atom_stability", step_metric="iteration_step")
            self.wandb.define_metric("iteration/mol_stability", step_metric="iteration_step")
            self.wandb.define_metric("iteration/asp", step_metric="iteration_step")  # Atom Stability Percentage
            self.wandb.define_metric("iteration/msp", step_metric="iteration_step")  # Molecule Stability Percentage
            self.wandb.define_metric("iteration/vup", step_metric="iteration_step")  # Valid & Unique Percentage
            self.wandb.define_metric("iteration/target_distance", step_metric="iteration_step")
            self.wandb.define_metric("iteration/property/*", step_metric="iteration_step")
            if self.should_log_iteration_operator_stats():
                self.wandb.define_metric("iteration/ops/*", step_metric="iteration_step")

        if self.should_log_benchmark_metrics():
            # Define benchmark-level metrics (step by task_step)
            self.wandb.define_metric("benchmark/*", step_metric="task_step")
            self.wandb.define_metric("benchmark/solver", step_metric="task_step")
            self.wandb.define_metric("benchmark/best_reward", step_metric="task_step")
            self.wandb.define_metric("benchmark/total_evals", step_metric="task_step")
            self.wandb.define_metric("benchmark/iterations", step_metric="task_step")
            self.wandb.define_metric("benchmark/wall_time_s", step_metric="task_step")
            self.wandb.define_metric("benchmark/running_average_best_reward", step_metric="task_step")
            self.wandb.define_metric("benchmark/gpu_mem_MiB", step_metric="task_step")
            if self.should_log_reward_curve():
                self.wandb.define_metric("benchmark/curve_last", step_metric="task_step")
            if self.should_log_benchmark_best_sample():
                self.wandb.define_metric("benchmark/best_molecule")
            if self.should_log_running_best_sample():
                self.wandb.define_metric("benchmark/running_best_molecule")

            # Define final summary metrics (no step)
            self.wandb.define_metric("benchmark/final_average_best_reward")
            self.wandb.define_metric("benchmark/num_tasks")

        # Define scaling metrics (checkpointed budgets) for non-image modalities too
        if self.should_log_scaling_metrics():
            checkpoints = self.get_scaling_checkpoints()
            # step_metric is task_step (one step per task)
            self.wandb.define_metric("task_step")
            for ckpt in checkpoints:
                self.wandb.define_metric(f"scaling/task_reward_at_{ckpt}", step_metric="task_step")
                self.wandb.define_metric(f"scaling/oracles_at_{ckpt}", step_metric="task_step")
                if self.should_log_scaling_running_average():
                    self.wandb.define_metric(f"scaling/running_avg_at_{ckpt}", step_metric="task_step")
            self.wandb.define_metric("scaling/num_checkpoints")

    @staticmethod
    def _to_serializable(obj: Any) -> Any:
        """Convert OmegaConf DictConfig/ListConfig to regular Python objects for JSON serialization."""
        if isinstance(obj, DictConfig):
            return OmegaConf.to_container(obj, resolve=True)
        elif isinstance(obj, dict):
            return {k: MoleculeLogger._to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [MoleculeLogger._to_serializable(item) for item in obj]
        return obj

    def format_context(self, context: Optional[Dict[str, Any]]) -> str:
        """Format context for console logging (molecule modality).
        
        Args:
            context: Problem context dictionary
            
        Returns:
            Formatted string with property targets
        """
        if context is None:
            return ""
        if "targets" in context:
            targets = context["targets"]
            if isinstance(targets, dict):
                target_strs = [f"{k}={v:.3f}" for k, v in targets.items()]
                return f"targets=[{', '.join(target_strs)}]"
        return ""

    # ---------------------------------------------------------------------
    # Scaling helpers (budget checkpoints)
    # ---------------------------------------------------------------------

    def get_scaling_checkpoints(self) -> List[int]:
        cfg = self._wandb_section("scaling") or {}
        cps = cfg.get("budget_checkpoints") or []
        try:
            return sorted([int(x) for x in list(cps)])
        except Exception:
            return []

    def log_scaling_for_task(
        self,
        *,
        task_index: int,
        task_oracles: int,
        task_history: Optional[List[Dict[str, Any]]] = None,
        best_reward: Optional[float] = None,
    ) -> None:
        """Log scaling metrics for a single task at configured oracle checkpoints.

        This is intentionally lightweight vs `ScalingLogger` and works for molecules + proteins.
        """
        if not self.enable or not self.should_log_scaling_metrics():
            return

        checkpoints = self.get_scaling_checkpoints()
        if not checkpoints:
            return

        # Helper: determine reward at (or before) checkpoint from solver history
        def _reward_at_checkpoint(ckpt: int) -> Optional[float]:
            if not task_history:
                return float(best_reward) if best_reward is not None else None

            # Prefer explicit checkpoint_{ckpt} values if present
            ck_key = f"checkpoint_{ckpt}"
            # IMPORTANT: some solvers write checkpoint keys at *every* later iteration once crossed.
            # In that case, taking max(vals) incorrectly returns the FINAL best reward for all checkpoints.
            # We want the best-so-far AT THE TIME the checkpoint was first crossed, so we take the first occurrence.
            for h in task_history:
                if not isinstance(h, dict):
                    continue
                v = h.get(ck_key)
                if v is not None:
                    return float(v)

            # Fallback: use the closest history record to the checkpoint based on num_evaluations/n_eval.
            # Prefer the first record with n_eval >= ckpt (closest AFTER crossing), otherwise use the last record.
            best_before: Optional[float] = None
            best_after: Optional[float] = None
            best_after_n: Optional[int] = None
            last_best: Optional[float] = None
            last_n: Optional[int] = None

            for h in task_history:
                if not isinstance(h, dict):
                    continue
                n_eval = h.get("num_evaluations", h.get("n_eval"))
                if n_eval is None:
                    continue
                try:
                    n_eval_i = int(n_eval)
                except Exception:
                    continue
                v = h.get("best", h.get("best_reward", h.get("batch_max")))
                if v is None:
                    continue
                fv = float(v)
                last_best = fv
                last_n = n_eval_i
                if n_eval_i <= ckpt:
                    best_before = fv
                if n_eval_i >= ckpt and best_after is None:
                    best_after = fv
                    best_after_n = n_eval_i

            if best_after is not None:
                return best_after
            if best_before is not None:
                return best_before
            if last_best is not None:
                return last_best
            return float(best_reward) if best_reward is not None else None

        payload: Dict[str, Any] = {"task_step": int(task_index), "scaling/num_checkpoints": int(len(checkpoints))}
        for ckpt in checkpoints:
            if task_oracles < ckpt:
                continue
            r_ckpt = _reward_at_checkpoint(ckpt)
            if r_ckpt is None:
                continue
            payload[f"scaling/task_reward_at_{ckpt}"] = float(r_ckpt)
            payload[f"scaling/oracles_at_{ckpt}"] = int(ckpt)
            if self.should_log_scaling_running_average():
                self._scaling_running_sums[ckpt] = self._scaling_running_sums.get(ckpt, 0.0) + float(r_ckpt)
                self._scaling_running_counts[ckpt] = self._scaling_running_counts.get(ckpt, 0) + 1
                payload[f"scaling/running_avg_at_{ckpt}"] = float(
                    self._scaling_running_sums[ckpt] / max(1, self._scaling_running_counts[ckpt])
                )

        # If no checkpoint reached, don't log
        if len(payload.keys()) <= 2:
            return
        self.wandb.log(payload)

    # ---------------------------------------------------------------------
    # Export best sample to PDB (molecules)
    # ---------------------------------------------------------------------

    def should_save_outputs(self) -> bool:
        if not self.config:
            return False
        return bool(self.config.get("save_outputs", False))

    def should_save_checkpoint_structures(self) -> bool:
        """Whether to export best-so-far structures at scaling checkpoints.

        Off by default because it can generate many PDB files.
        """
        if not self.config:
            return False
        return bool(self.config.get("save_checkpoint_structures", False))

    def get_best_structures_dir(self) -> str:
        if not self.config:
            return "outputs"
        return str(self.config.get("save_outputs_dir", "outputs"))

    def save_best_molecule_pdb(self, *, task_index: int, best_molecule: Any, context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Save best molecule for a task to a .pdb file (best-effort).

        Supports RDKit Mol or SMILES-like objects; falls back to a minimal PDB if needed.
        """
        if not self.should_save_outputs():
            return None
        if best_molecule is None:
            return None

        # Determine run name for folder
        run_name = str(self._run_id)
        
        # Determine base directory (project root vs current hydra output dir)
        try:
            from hydra.utils import get_original_cwd
            base_dir = get_original_cwd()
        except (ImportError, ValueError):
            base_dir = os.getcwd()

        # Get base dir from config
        outputs_base = self.config.get("save_outputs_dir", "outputs")

        category = "molecules"

        # Organize in <outputs_base>/<category>/<project>/<run_name>/<task_prefix>
        project = self.project or "default_project"
        def sanitize(s: str) -> str:
            return "".join([c if c.isalnum() or c in ("-", "_") else "_" for c in s])

        task_prefix = f"task_{int(task_index):04d}"
        out_dir = os.path.join(base_dir, outputs_base, category, sanitize(project), sanitize(run_name), task_prefix)
        os.makedirs(out_dir, exist_ok=True)
        self._saved_dirs.add(os.path.dirname(out_dir)) # Add parent run directory
        self._last_saved_dir = os.path.dirname(out_dir)
        
        # Construct a descriptive filename
        # Order: iteration -> size -> targets
        filename_parts = []
        
        if context:
            # 1. Iteration if present (first for sorting)
            iteration = context.get("iteration")
            if iteration is not None:
                filename_parts.append(f"iter_{int(iteration):03d}")
            
            # 2. Size (n_atoms for molecules, check model_config too)
            n_atoms = context.get("n_atoms")
            if not n_atoms and "model_config" in context:
                n_atoms = context["model_config"].get("n_atoms")
            if n_atoms:
                filename_parts.append(f"{n_atoms}at")
            
            # 3. Target info if available
            targets = context.get("targets")
            if targets and isinstance(targets, dict):
                # Format floating point targets to 4 decimal places for cleaner filenames
                target_str = "_".join([f"{k}{float(v):.4f}" for k, v in targets.items()])
                filename_parts.append(target_str)

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
            # If it's the final best, we might not have an iteration, so ensure it still looks good
            filename = "_".join(filename_parts) + f"_{suffix}{reward_suffix}.pdb"
            
        out_path = os.path.join(out_dir, filename)

        # Prepare metadata for PDB header
        header_lines = []
        header_lines.append(f"HEADER    MOLECULE DESIGN OPTIMIZATION            {run_name[:10].upper():<10}")
        header_lines.append(f"TITLE     Molecule Optimization: {run_name}")
        if context:
            it = context.get("iteration", "final")
            rew = context.get("reward")
            header_lines.append(f"REMARK 001 TASK INDEX: {task_index}")
            header_lines.append(f"REMARK 001 ITERATION: {it}")
            if rew is not None:
                header_lines.append(f"REMARK 001 REWARD: {float(rew):.4f}")

        # Pre-process tensor molecules into a dictionary for easier handling
        if isinstance(best_molecule, torch.Tensor):
            # QM9 decoded format: [n_nodes, 9] where first 3 are x,y,z and next 5 are one-hot, last is charge
            mol_t = best_molecule.detach().cpu()
            if mol_t.ndim == 2 and mol_t.shape[1] >= 8:
                # Filter out padding atoms (one-hot sum > 0)
                one_hot = mol_t[:, 3:8]
                mask = (one_hot.sum(dim=-1) > 1e-6)
                real_atoms = mol_t[mask]
                
                best_molecule = {
                    "positions": real_atoms[:, :3].numpy(),
                    "one_hot": real_atoms[:, 3:8].numpy(),
                }
                # Add atomic numbers
                atom_types = torch.argmax(real_atoms[:, 3:8], dim=-1).numpy()
                # Map 0..4 to H, C, N, O, F (QM9 standard)
                atomic_numbers = [1, 6, 7, 8, 9]
                best_molecule["elements"] = [atomic_numbers[i] for i in atom_types]

        # Try RDKit export first
        try:
            from rdkit import Chem  # type: ignore
            from rdkit.Chem import AllChem  # type: ignore

            mol = None
            if hasattr(best_molecule, "GetConformer"):
                mol = best_molecule
            elif isinstance(best_molecule, str):
                mol = Chem.MolFromSmiles(best_molecule)
                if mol is not None:
                    mol = Chem.AddHs(mol)
                    AllChem.EmbedMolecule(mol, randomSeed=0)
                    AllChem.MMFFOptimizeMolecule(mol)
            elif hasattr(best_molecule, "smiles"):
                smi = getattr(best_molecule, "smiles")
                if isinstance(smi, str):
                    mol = Chem.MolFromSmiles(smi)
                    if mol is not None:
                        mol = Chem.AddHs(mol)
                        AllChem.EmbedMolecule(mol, randomSeed=0)
                        AllChem.MMFFOptimizeMolecule(mol)
            elif isinstance(best_molecule, dict) and "positions" in best_molecule and "elements" in best_molecule:
                # Create Mol from coordinates and atomic numbers
                mol = Chem.RWMol()
                for z in best_molecule["elements"]:
                    mol.AddAtom(Chem.Atom(int(z)))
                
                conf = Chem.Conformer(len(best_molecule["elements"]))
                for i, pos in enumerate(best_molecule["positions"]):
                    conf.SetAtomPosition(i, (float(pos[0]), float(pos[1]), float(pos[2])))
                mol.AddConformer(conf)
                
                # Attempt to infer bonds based on distances
                # This is a bit heuristic but better than no bonds
                try:
                    # RDKit's ConnectTheDots/DetermineBonds needs some help or just use distance-based
                    # For now, we'll just save the atoms; PDB viewers usually handle bond inference
                    pass
                except Exception:
                    pass

            if mol is not None:
                # Ensure conformer exists
                if mol.GetNumConformers() == 0:
                    mol_h = Chem.AddHs(mol)
                    AllChem.EmbedMolecule(mol_h, randomSeed=0)
                    AllChem.MMFFOptimizeMolecule(mol_h)
                    mol = mol_h
                
                # Write with our custom header
                pdb_block = Chem.MolToPDBBlock(mol)
                with open(out_path, "w") as f:
                    f.write("\n".join(header_lines) + "\n")
                    f.write(pdb_block)
                return out_path
        except Exception:
            pass

        # Fallback: standard PDB representation if molecule is array-like coords
        try:
            coords = None
            elements = None
            if isinstance(best_molecule, dict):
                if "positions" in best_molecule:
                    coords = best_molecule["positions"]
                if "elements" in best_molecule:
                    elements = best_molecule["elements"]
                elif "atoms" in best_molecule:
                    elements = best_molecule["atoms"]
            
            if coords is not None:
                coords_np = np.asarray(coords)
                lines = list(header_lines)
                
                lines.append("MODEL     1")
                for i, xyz in enumerate(coords_np):
                    elem = "C"
                    if elements is not None and i < len(elements):
                        e = elements[i]
                        if isinstance(e, str):
                            elem = e
                        elif isinstance(e, int):
                            # Map atomic number to symbol (common ones)
                            ELEM_MAP = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 15: "P", 16: "S", 17: "CL"}
                            elem = ELEM_MAP.get(e, "C")
                    
                    # Align atom name (13-16)
                    if len(elem) == 1:
                        name_str = f" {elem}  "
                    else:
                        name_str = f"{elem:^4}"
                        
                    lines.append(
                        f"HETATM{i+1:>5} {name_str} UNK A{i+1:>4}    "
                        f"{xyz[0]:>8.3f}{xyz[1]:>8.3f}{xyz[2]:>8.3f}  1.00 20.00           {elem[0]}"
                    )
                lines.append("TER")
                lines.append("ENDMDL")
                lines.append("END")
                
                # Pad all lines to 80 chars
                padded_lines = [line.ljust(80) for line in lines]
                
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(padded_lines) + "\n")
                return out_path
        except Exception:
            return None

        return None

    def log_iteration_metrics(
        self,
        iteration: int,
        rewards: torch.Tensor | List[float],
        *,
        task_index: Optional[int] = None,
        evals_cum: Optional[int] = None,
        time_s: Optional[float] = None,
        operator_stats: Optional[Dict[str, Any]] = None,
        topk_molecules: Optional[List[Any]] = None,
        topk_rewards: Optional[List[float]] = None,
        properties: Optional[Dict[str, torch.Tensor | List[float]]] = None,
        stability: Optional[Dict[str, Any]] = None,
        targets: Optional[Dict[str, float]] = None,
    ) -> None:
        """Log iteration-level metrics with proper step tracking.
        
        Args:
            iteration: Current iteration number
            rewards: Tensor or list of reward values for this iteration
            task_index: Optional task index for tracking which task this iteration belongs to
            evals_cum: Cumulative number of evaluations
            time_s: Wall time for this iteration
            operator_stats: Optional operator statistics (mutation, combination, etc.)
            topk_molecules: Optional top-k molecules to log
            topk_rewards: Optional rewards for top-k molecules
            properties: Optional dictionary of predicted properties
            stability: Optional stability analysis results
            targets: Optional target property values
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
        
            # Log properties if available
            if properties is not None:
                for prop_name, prop_values in properties.items():
                    if isinstance(prop_values, torch.Tensor):
                        prop_t = prop_values.detach().cpu().view(-1)
                    elif isinstance(prop_values, list):
                        prop_t = torch.tensor(prop_values)
                    else:
                        prop_t = torch.tensor(prop_values)
                metrics[f"iteration/property/{prop_name}_mean"] = float(prop_t.mean().item())
                metrics[f"iteration/property/{prop_name}_std"] = float(prop_t.std(unbiased=False).item()) if prop_t.numel() > 1 else 0.0
                
                # Track in history
                if prop_name not in self.property_history:
                    self.property_history[prop_name] = []
                self.property_history[prop_name].append(float(prop_t.mean().item()))
            
            # Log stability if available
            if stability is not None:
                # Log ASP (Atom Stability Percentage)
                if "asp" in stability:
                    metrics["iteration/asp"] = float(stability["asp"])
                    metrics["iteration/atom_stability"] = float(stability["asp"]) / 100.0
                elif "atom_stable" in stability:
                    atom_stable = stability["atom_stable"]
                    if isinstance(atom_stable, (list, np.ndarray)):
                        val = float(np.mean(atom_stable))
                    else:
                        val = float(atom_stable)
                    metrics["iteration/atom_stability"] = val
                    metrics["iteration/asp"] = val * 100.0
                
                # Log MSP (Molecule Stability Percentage)
                if "msp" in stability:
                    metrics["iteration/msp"] = float(stability["msp"])
                    metrics["iteration/mol_stability"] = float(stability["msp"]) / 100.0
                elif "mol_stable" in stability:
                    mol_stable = stability["mol_stable"]
                    if isinstance(mol_stable, (list, np.ndarray)):
                        val = float(np.mean(mol_stable))
                    else:
                        val = float(mol_stable)
                    metrics["iteration/mol_stability"] = val
                    metrics["iteration/msp"] = val * 100.0
                    self.stability_history.append(val)
                
                # Log VUP (Valid & Unique Percentage)
                if "vup" in stability:
                    metrics["iteration/vup"] = float(stability["vup"])
                
                if "stability" in stability:
                    metrics["iteration/stability"] = float(stability["stability"])
            
            # Log target distances if targets provided
            if targets is not None and properties is not None:
                distances = []
                for prop_name, target_value in targets.items():
                    if prop_name in properties:
                        prop_values = properties[prop_name]
                        if isinstance(prop_values, torch.Tensor):
                            prop_np = prop_values.detach().cpu().numpy()
                        else:
                            prop_np = np.array(prop_values)
                        dist = np.abs(prop_np - target_value)
                    mean_dist = float(np.mean(dist))
                    metrics[f"iteration/target_distance/{prop_name}"] = mean_dist
                    distances.append(mean_dist)
                if distances:
                    total_dist = float(np.mean(distances))
                    metrics["iteration/target_distance"] = total_dist
                    self.target_distance_history.append(total_dist)
        
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

        self.wandb.log(metrics)

        # Optionally log top-k molecules with visualizations
        if topk_molecules is not None and self.should_log_iteration_samples():
            limit = self.get_iteration_sample_limit()
            if limit > 0:
                topk_molecules = topk_molecules[:limit]
                if topk_rewards is not None:
                    topk_rewards = topk_rewards[:limit]
            
            # Create molecular visualization
            try:
                # Prepare properties dict for visualization (needs to be List[float] per property)
                vis_properties = None
                if properties is not None:
                    vis_properties = {}
                    for k, v in properties.items():
                        if isinstance(v, torch.Tensor):
                            v_list = v.detach().cpu().tolist()
                        elif isinstance(v, list):
                            v_list = v
                        else:
                            v_list = [float(v)] * len(topk_molecules)
                        # Ensure we have enough values
                        if len(v_list) < len(topk_molecules):
                            v_list = v_list + [v_list[-1]] * (len(topk_molecules) - len(v_list))
                        vis_properties[k] = v_list[:len(topk_molecules)]
                
                rewards_list = topk_rewards if topk_rewards is not None else [0.0] * len(topk_molecules)
                img = visualize_best_molecules(
                    topk_molecules,
                    rewards_list,
                    properties=vis_properties,
                    top_k=min(limit, len(topk_molecules)),
                )
                if img is not None:
                    caption = f"iter={iteration} | top-{min(limit, len(topk_molecules))} molecules"
                    if topk_rewards is not None and len(topk_rewards) > 0:
                        caption += f" | best R={float(max(topk_rewards)):.4f}"
                    self.log_image("iteration/topk_molecules", img, caption=caption, step_key="iteration_step", step=iteration)
            except Exception as e:
                print(f"[WARNING] Failed to visualize top-k molecules: {e}")
                import traceback
                traceback.print_exc()

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
        best_molecule: Optional[Any] = None,
        running_best_molecule: Optional[Any] = None,
        reward_curve: Optional[List[float]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log benchmark-level summary metrics with proper step tracking.
        
        Args:
            context: Problem context dictionary
            solver_name: Name of the solver used
            task_index: Task index for tracking (used as step)
            best_reward: Best reward achieved for this task
            total_iterations: Total number of iterations run
            total_evals: Total number of evaluations performed
            wall_time_s: Wall clock time taken
            best_molecule: Best molecule generated
            running_best_molecule: Running best molecule across tasks
            reward_curve: List of reward values over iterations
            extra: Additional metrics to log
        """
        if not self.enable or not self.should_log_benchmark_metrics():
            return
        
        # Format context for caption
        context_str = self.format_context(context)
        
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
            payload["benchmark/curve_last"] = float(reward_curve[-1]) if len(reward_curve) > 0 else 0.0
        
        # Extract running_average_best_reward from extra if provided
        running_avg = None
        if extra:
            running_avg = extra.get("running_average_best_reward")
        
        # Log running_average_best_reward as a direct metric (important for tracking over tasks)
        if running_avg is not None:
            payload["benchmark/running_average_best_reward"] = float(running_avg)
        
        if extra:
            # Convert any DictConfig objects to regular dicts for JSON serialization
            extra = self._to_serializable(extra)
            for k, v in extra.items():
                # Skip config values that shouldn't be plotted
                if k in ("solver_kwargs", "algorithm_kwargs", "algo_kwargs") and isinstance(v, dict):
                    continue
                elif k == "running_average_best_reward":
                    # Already logged above as direct metric
                    continue
                elif k == "num_tasks":
                    # These are important metrics to log
                    payload[f"benchmark/{k}"] = v
                elif k in ("gpu_mem_max_alloc_bytes", "gpu_mem_max_reserved_bytes"):
                    # Log GPU memory in MB for readability
                    if v is not None:
                        payload[f"benchmark/gpu_mem_MiB"] = float(v) / (1024 * 1024)
                elif k not in ("gpu_mem_max_alloc_mib", "gpu_mem_max_reserved_mib"):
                    # Log other metrics that might be useful
                    payload[f"benchmark/{k}"] = v

        def _to_wandb_molecule(mol: Any, caption: str):
            """Convert molecule to wandb image."""
            try:
                img = self.visualizer.visualize_molecule_3d(mol, title=caption, show_bonds=True)
                if img is not None:
                    return self.wandb.Image(img, caption=caption)
            except Exception as exc:
                print(f"[ERROR] Converting molecule to wandb image: {exc}")
            return None

        # Single best molecule (per task)
        if best_molecule is not None and self.should_log_benchmark_best_sample():
            wb_mol = _to_wandb_molecule(best_molecule, f"Task {task_index} | {context_str} | best={best_reward:.4f}")
            if wb_mol is not None:
                payload["benchmark/best_molecule"] = wb_mol
                if "task_step" not in payload:
                    payload["task_step"] = task_index

        # Running best molecule across tasks
        if running_best_molecule is not None and self.should_log_running_best_sample():
            wb_mol = _to_wandb_molecule(running_best_molecule, f"Running best up to task {task_index} | {context_str}")
            if wb_mol is not None:
                payload["benchmark/running_best_molecule"] = wb_mol
                if "task_step" not in payload:
                    payload["task_step"] = task_index

        self.wandb.log(payload)

        if self.should_print_benchmark() and self._last_saved_dir:
            print(f"  Task complete. Results saved to: {self._last_saved_dir}")

        # Optional: export best molecule to PDB
        if self.should_save_outputs() and best_molecule is not None:
            try:
                # Add best_reward to context for filename tracking
                ctx = {**(context or {}), "reward": float(best_reward)}
                self.save_best_molecule_pdb(task_index=task_index, best_molecule=best_molecule, context=ctx)
            except Exception:
                pass

    def log_iteration(
        self,
        iteration: int,
        metrics: Dict[str, Any],
        step_type: str = "molecule_step",
    ) -> None:
        """Log metrics for a single iteration (backward compatibility).
        
        Args:
            iteration: Current iteration number
            metrics: Dictionary of metrics to log
            step_type: Type of step (molecule_step, bo_iteration, gp_step)
        """
        # Convert to new format
        rewards = metrics.get("reward")
        if rewards is None:
            # Try to extract from metrics
            if "rewards" in metrics:
                rewards = metrics["rewards"]
            elif "batch_rewards" in metrics:
                rewards = metrics["batch_rewards"]
        
        self.log_iteration_metrics(
            iteration=iteration,
            rewards=rewards if rewards is not None else [0.0],
            properties=metrics.get("properties"),
            stability=metrics.get("stability"),
            targets=metrics.get("targets"),
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the experiment."""
        summary = {
            "total_iterations": len(self.property_history.get(list(self.property_history.keys())[0], [])) if self.property_history else 0,
            "wandb_enabled": self.enable,
        }
        
        # Add molecule-specific summary stats
        if self.stability_history:
            summary["stability"] = {
                "mean": float(np.mean(self.stability_history)),
                "std": float(np.std(self.stability_history)),
                "final": float(self.stability_history[-1]) if self.stability_history else 0.0,
            }
        
        if self.target_distance_history:
            summary["target_distance"] = {
                "mean": float(np.mean(self.target_distance_history)),
                "std": float(np.std(self.target_distance_history)),
                "final": float(self.target_distance_history[-1]) if self.target_distance_history else 0.0,
            }
        
        # Add property-specific summaries
        if self.property_history:
            summary["properties"] = {}
            for prop_name, values in self.property_history.items():
                summary["properties"][prop_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "final": float(values[-1]) if values else 0.0,
                }
        
        return summary

    def save_scaling_checkpoint_csv(self) -> Optional[str]:
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
            checkpoints = self.get_scaling_checkpoints()

            for checkpoint in checkpoints:
                if checkpoint in self._scaling_running_sums and self._scaling_running_counts.get(checkpoint, 0) > 0:
                    running_avg = self._scaling_running_sums[checkpoint] / self._scaling_running_counts[checkpoint]
                    checkpoint_data.append({
                        'checkpoint': checkpoint,
                        'running_average': running_avg,
                        'count': self._scaling_running_counts[checkpoint]
                    })

            if not checkpoint_data:
                return None

            # Extract metadata from config
            modality = str(self.config.get("modality", "unknown")).lower() if self.config else "unknown"
            model = str(self.config.get("model", "unknown")).lower() if self.config else "unknown"
            reward_function = str(self.config.get("reward", "unknown")).lower() if self.config else "unknown"

            # Create unique run name with timestamp
            base_run_name = self._run_id or "run"
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

    def close(self) -> None:
        """Close logger and print summary of saved outputs."""
        # Save scaling checkpoint CSV if scaling was enabled
        if hasattr(self, '_scaling_running_sums') and self._scaling_running_sums:
            csv_path = self.save_scaling_checkpoint_csv()
            if csv_path:
                print(f"Scaling checkpoint data saved to CSV: {csv_path}")

        if self._saved_dirs:
            print("\n" + "="*80)
            print("RUN COMPLETE: Local outputs saved to:")
            for d in sorted(list(self._saved_dirs)):
                print(f"  - {d}")
            print("="*80 + "\n")

        super().close()
