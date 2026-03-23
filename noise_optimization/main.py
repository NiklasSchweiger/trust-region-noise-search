from __future__ import annotations

from typing import Any, Dict, Optional

import os
import sys

import hydra
import torch
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

# Rich for terminal output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.rule import Rule
    from rich import box
    from .core.utils.terminal_colors import CLI_APP, CLI_SUCCESS, CLI_WARNING, CLI_ERROR, CLI_DIM
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None
    box = None
    CLI_APP = CLI_SUCCESS = CLI_WARNING = CLI_ERROR = CLI_DIM = "white"

# Allow running this file directly via `python noise_optimization/main.py`
if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    pkg_dir = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.dirname(pkg_dir)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    __package__ = os.path.basename(pkg_dir)


# --- Compatibility helpers (public API used by other scripts) -----------------

def _instantiate_generative_model(cfg: DictConfig, device: Optional[str] = None) -> torch.nn.Module:
    from .core.models.factories import instantiate_generative_model

    return instantiate_generative_model(cfg, device=device)


def _instantiate_reward(cfg: DictConfig) -> Any:
    from .core.rewards.factory import instantiate_reward

    return instantiate_reward(cfg)


# --- Pipeline selection -------------------------------------------------------

PIPELINE_BY_MODALITY = {
    "image": "noise_optimization.core.pipelines.t2i",
    "image_noise": "noise_optimization.core.pipelines.t2i",
    "t2i": "noise_optimization.core.pipelines.t2i",
    "molecule": "noise_optimization.core.pipelines.molecule",
    "qm9": "noise_optimization.core.pipelines.molecule",
    "qm9_target": "noise_optimization.core.pipelines.molecule",
    "protein": "noise_optimization.core.pipelines.protein",
    "proteina": "noise_optimization.core.pipelines.protein",
}


def run(cfg: DictConfig) -> Dict[str, Any]:
    """Dispatch to the appropriate pipeline based on cfg.modality."""
    console = Console() if RICH_AVAILABLE else None
    
    modality = (cfg.get("modality") or "image").lower()
    target = PIPELINE_BY_MODALITY.get(modality)
    if target is None:
        error_msg = f"Unsupported modality '{modality}'. Add a pipeline mapping in main.py."
        if console:
            console.print(f"[red]Error:[/red] {error_msg}")
        raise ValueError(error_msg)

    # Display pipeline info with colors
    if console:
        pipeline_name = target.split(".")[-1]
        modality_display = modality.upper()
        
        # Import helper functions to extract solver and reward names
        from .core.pipelines.common import get_solver_name, _get_reward_name
        
        # Extract solver name
        solver_name = get_solver_name(cfg) or "unknown"
        # Capitalize first letter for display
        if solver_name and solver_name != "unknown":
            solver_name = solver_name[0].upper() + solver_name[1:] if len(solver_name) > 1 else solver_name.upper()
        
        # Extract reward function name
        reward_name = _get_reward_name(cfg) or "unknown"
        # Capitalize for display
        if reward_name and reward_name != "unknown":
            reward_name = reward_name.replace("_", " ").title()
        
        # Pipeline banner: blue accent for app identity
        info_lines = [
            f"Pipeline:  [bold]{pipeline_name}[/bold]",
            f"Modality:  [bold]{modality_display}[/bold]",
            f"Solver:    [bold]{solver_name}[/bold]",
            f"Reward:    [bold]{reward_name}[/bold]",
        ]
        console.print()
        console.print(Rule(style=CLI_APP, characters="─"))
        console.print(Panel(
            "\n".join(info_lines),
            title=f"[bold {CLI_APP}]Noise Optimization[/bold {CLI_APP}]",
            border_style=CLI_APP,
            box=box.ROUNDED if box else None,
            padding=(0, 2),
        ))
        console.print(Rule(style=CLI_APP, characters="─"))
        console.print()
    
    pipeline = hydra.utils.get_method(f"{target}.run")
    return pipeline(cfg)


def _print_run_summary(result: Optional[Dict[str, Any]], console: Any) -> None:
    """Print a one-line run summary when pipeline returns results (tasks, mean best, total evals)."""
    if not console or not result or not isinstance(result, dict):
        return
    results = result.get("results")
    if not results or not isinstance(results, list):
        return
    n = len(results)
    total_evals = sum(r.get("num_evaluations", 0) for r in results if isinstance(r, dict))
    best_rewards = [r.get("best_reward") for r in results if isinstance(r, dict) and r.get("best_reward") is not None]
    mean_best = sum(best_rewards) / len(best_rewards) if best_rewards else None
    parts = [f"Tasks: {n}"]
    if total_evals:
        parts.append(f"Total evals: {total_evals}")
    if mean_best is not None:
        parts.append(f"Mean best reward: {mean_best:.4f}")
    console.print(f"[{CLI_DIM}]" + " | ".join(parts) + f"[/{CLI_DIM}]")


def _write_results_csv(result: Optional[Dict[str, Any]], output_dir: str) -> None:
    """Write a run-level results CSV (task_id, best_reward, num_evaluations) to output_dir."""
    if not result or not isinstance(result, dict):
        return
    results = result.get("results")
    if not results or not isinstance(results, list):
        return
    import csv
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "results.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task_id", "best_reward", "num_evaluations"])
        for i, r in enumerate(results):
            if isinstance(r, dict):
                w.writerow([
                    i,
                    r.get("best_reward", ""),
                    r.get("num_evaluations", ""),
                ])


@hydra.main(config_path="config", config_name="main.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    console = Console() if RICH_AVAILABLE else None
    
    # Note: Colored printing is available via core.utils.colored_logging
    # but we don't enable the global print override by default as it can
    # interfere with existing code. Individual modules can use colored_print
    # directly if desired.
    
    # Capture Hydra overrides for debugging/recording
    if GlobalHydra().is_initialized():
        hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        if hydra_cfg is not None:
            overrides = hydra_cfg.overrides.task
            OmegaConf.set_struct(cfg, False)
            cfg["hydra_overrides"] = overrides
            OmegaConf.set_struct(cfg, True)
            
            # Display Hydra overrides in a nice format
            if console and overrides:
                override_tags = [f"[dim]{o}[/dim]" for o in overrides]
                console.print(f"[dim]Overrides:[/dim] {' '.join(override_tags)}")
                console.print()

    try:
        result = run(cfg)
        if console:
            console.print()
            console.print(Rule("Pipeline completed", style=CLI_SUCCESS, characters="─"))
            _print_run_summary(result, console)
            console.print()
        if result and OmegaConf.select(cfg, "save_results_csv", default=False):
            out_dir = OmegaConf.select(cfg, "save_outputs_dir", default="outputs")
            _write_results_csv(result, str(out_dir))
    except KeyboardInterrupt:
        if console:
            console.print()
            console.print(Rule("Interrupted by user", style=CLI_WARNING, characters="─"))
            console.print()
        raise
    except Exception as e:
        if console:
            console.print()
            console.print(Rule(f"Pipeline failed: {type(e).__name__}", style=CLI_ERROR, characters="─"))
            console.print(f"[{CLI_ERROR}]{str(e)}[/{CLI_ERROR}]")
            console.print()
        raise


if __name__ == "__main__":
    main()
