"""Utility functions for printing model summaries."""

import math
from typing import Any, Dict, Optional

import torch
from omegaconf import DictConfig, OmegaConf

# Rich for beautiful terminal output
try:
    from rich.console import Console
    from rich.rule import Rule
    RICH_AVAILABLE = True
    _console = Console()
except ImportError:
    RICH_AVAILABLE = False
    _console = None


def count_parameters(model: torch.nn.Module) -> int:
    """Count the total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_parameter_count(num_params: int) -> str:
    """Format parameter count in human-readable format (M for millions, B for billions)."""
    if num_params >= 1_000_000_000:
        return f"{num_params / 1_000_000_000:.2f}B"
    elif num_params >= 1_000_000:
        return f"{num_params / 1_000_000:.2f}M"
    elif num_params >= 1_000:
        return f"{num_params / 1_000:.2f}K"
    else:
        return str(num_params)


def get_model_name(gen_model: Any) -> str:
    """Extract model name from generative model."""
    # Try to get model_id from pipeline
    if hasattr(gen_model, "pipeline"):
        pipeline = gen_model.pipeline
        if hasattr(pipeline, "model_id"):
            return str(pipeline.model_id)
        # Try to infer from class name
        model_class = type(pipeline).__name__
        if "StableDiffusion" in model_class:
            if "XL" in model_class:
                return "Stable Diffusion XL"
            elif "3" in model_class:
                return "Stable Diffusion 3"
            else:
                return "Stable Diffusion"
        elif "PixArt" in model_class:
            if "Sigma" in model_class:
                return "PixArt Sigma"
            else:
                return "PixArt Alpha"
        elif "Flux" in model_class:
            return "Flux"
        elif "LCM" in model_class or "LatentConsistency" in model_class:
            return "Latent Consistency Model"
        return model_class
    
    # Try to get from gen_model directly
    if hasattr(gen_model, "model_id"):
        return str(gen_model.model_id)
    
    model_class = type(gen_model).__name__
    if "Proteina" in model_class:
        return "Proteina"
    
    # Fallback to class name
    return model_class


def get_scheduler_name(gen_model: Any) -> Optional[str]:
    """Extract scheduler name from generative model."""
    if hasattr(gen_model, "pipeline"):
        pipeline = gen_model.pipeline
        if hasattr(pipeline, "scheduler"):
            scheduler = pipeline.scheduler
            if scheduler is not None:
                return type(scheduler).__name__.replace("Scheduler", "")
    # Proteina doesn't use schedulers (uses Flow Matching)
    return None


def get_model_parameters(gen_model: Any) -> Optional[int]:
    """Count total parameters in the generative model."""
    try:
        if hasattr(gen_model, "pipeline"):
            pipeline = gen_model.pipeline
            # Count parameters in UNet (main model component)
            if hasattr(pipeline, "unet"):
                return count_parameters(pipeline.unet)
            # Fallback: count all parameters in pipeline
            return count_parameters(pipeline)
        # Try to count parameters in gen_model directly
        if isinstance(gen_model, torch.nn.Module):
            return count_parameters(gen_model)
        # Try to get model from wrapper (for protein)
        if hasattr(gen_model, "model"):
            model = gen_model.model
            if isinstance(model, torch.nn.Module):
                return count_parameters(model)
    except Exception:
        pass
    return None


def print_model_summary(
    gen_model: Any,
    cfg: Dict[str, Any],
    modality: str = "image",
    problem_builder: Optional[Any] = None,
) -> None:
    """Print a summary of the generative model configuration.
    
    Args:
        gen_model: The generative model instance
        cfg: Configuration dictionary
        modality: Modality type ("image", "protein", "qm9")
        problem_builder: Optional problem builder for output size information
    """
    # Print styled header (paper color palette)
    if RICH_AVAILABLE and _console:
        from ..utils.terminal_colors import CLI_SECTION
        _console.print()
        _console.print(Rule(f"[bold {CLI_SECTION}]Model Summary[/bold {CLI_SECTION}]", style=CLI_SECTION, characters="─"))
        _console.print()
    else:
        print(f"\n{'='*80}")
        print(f"[Model Summary]")
        print(f"{'='*80}")
    
    # Model name
    model_name = get_model_name(gen_model)
    print(f"Model: {model_name}")
    
    # Number of parameters
    num_params = get_model_parameters(gen_model)
    if num_params is not None:
        param_str = format_parameter_count(num_params)
        print(f"Parameters: {param_str} ({num_params:,} total)")
    else:
        print(f"Parameters: N/A")
    
    # Scheduler
    scheduler_name = get_scheduler_name(gen_model)
    if scheduler_name:
        print(f"Scheduler: {scheduler_name}")
    else:
        print(f"Scheduler: N/A")
    
    # ODE vs SDE (based on eta for diffusion models, sampling_mode for Flow Matching)
    pipeline_cfg = cfg.get("model") or {}
    model_cfg = cfg.get("model") or {}
    
    # Convert DictConfig to dict if needed for easier access
    if isinstance(pipeline_cfg, DictConfig):
        pipeline_cfg = OmegaConf.to_container(pipeline_cfg, resolve=True) or {}
    if isinstance(model_cfg, DictConfig):
        model_cfg = OmegaConf.to_container(model_cfg, resolve=True) or {}
    
    # Check if this is a Flow Matching model (Proteina)
    is_flow_matching = "Proteina" in model_name

    if is_flow_matching:
        proteina_cfg = cfg.get("proteina") or {}

        # Convert DictConfig to dict if needed
        if isinstance(proteina_cfg, DictConfig):
            proteina_cfg = OmegaConf.to_container(proteina_cfg, resolve=True) or {}

        # Get sampling_mode from proteina config
        sampling_mode = None
        if isinstance(proteina_cfg, dict):
            sampling_mode = proteina_cfg.get("sampling_mode")

        if sampling_mode is None:
            sampling_mode = "vf"

        # Proteina uses sc_scale_noise to determine SDE vs ODE
        sc_scale_noise = None
        if isinstance(proteina_cfg, dict):
            sc_scale_noise = proteina_cfg.get("sc_scale_noise")
        if sc_scale_noise is None:
            try:
                sc_scale_noise = OmegaConf.select(cfg, "proteina.sc_scale_noise", default=None)
            except:
                pass
        if sc_scale_noise is None:
            sc_scale_noise = 0.4
        sc_scale_noise_val = float(sc_scale_noise)

        if sampling_mode == "sc" or sampling_mode == "self_conditioning":
            if sc_scale_noise_val > 0.0:
                variant = f"SDE (stochastic, self-conditioning mode, sc_scale_noise={sc_scale_noise_val})"
                noise_factor = str(sc_scale_noise_val)
            else:
                variant = f"SDE (stochastic, self-conditioning mode, sc_scale_noise={sc_scale_noise_val}) [WARNING: sc_scale_noise=0 in SC mode]"
                noise_factor = str(sc_scale_noise_val)
        elif sampling_mode == "vf" or sampling_mode == "velocity_field" or sampling_mode is None:
            if sc_scale_noise_val > 0.0:
                variant = f"ODE (deterministic, velocity field mode) [WARNING: sc_scale_noise={sc_scale_noise_val} ignored in VF mode]"
                noise_factor = "0.0 (sc_scale_noise ignored in VF mode)"
            else:
                variant = "ODE (deterministic, velocity field mode)"
                noise_factor = "0.0"
        else:
            if sc_scale_noise_val > 0.0:
                variant = f"SDE (stochastic, sc_scale_noise={sc_scale_noise_val}, sampling_mode={sampling_mode})"
                noise_factor = str(sc_scale_noise_val)
            else:
                variant = f"ODE (deterministic, sampling_mode={sampling_mode})"
                noise_factor = "0.0"

        print(f"Variant: {variant}")
        if sampling_mode == "sc" or sampling_mode == "self_conditioning":
            sampling_mode_desc = "sc (self-conditioning, SDE)"
        elif sampling_mode == "vf" or sampling_mode == "velocity_field" or sampling_mode is None:
            sampling_mode_desc = "vf (velocity field, ODE)"
        else:
            sampling_mode_desc = f"{sampling_mode} (unknown)"
        print(f"Sampling mode: {sampling_mode_desc}")
        print(f"SDE noise factor: {noise_factor}")
        
        # Also show dt (time step) for Flow Matching
        dt = proteina_cfg.get("dt") if isinstance(proteina_cfg, dict) else None
        if dt is not None:
            print(f"Time step (dt): {dt}")
    else:
        # For diffusion models, use eta
        # Try multiple sources: model config, pipeline config, and direct cfg access
        eta = None
        
        # First try from converted dicts
        if isinstance(model_cfg, dict):
            eta = model_cfg.get("eta")
        if eta is None and isinstance(pipeline_cfg, dict):
            eta = pipeline_cfg.get("eta")
        
        # If not found, try accessing directly from cfg using OmegaConf.select (handles DictConfig)
        if eta is None:
            try:
                # Try cfg.model.eta using OmegaConf.select
                eta = OmegaConf.select(cfg, "model.eta", default=None)
            except:
                pass
        
        if eta is None:
            try:
                # Try cfg.model.eta using OmegaConf.select
                eta = OmegaConf.select(cfg, "pipeline.eta", default=None)
            except:
                pass
        
        if eta is not None:
            eta_val = float(eta)
            if eta_val == 0.0:
                variant = "ODE (deterministic)"
                noise_factor = "0.0"
            else:
                variant = f"SDE (stochastic, eta={eta_val})"
                noise_factor = str(eta_val)
            print(f"Variant: {variant}")
            print(f"SDE noise factor: {noise_factor}")
        else:
            # If eta is not found in config, try pipeline scheduler then infer from solver name
            eta_from_scheduler = None
            if hasattr(gen_model, "pipeline") and getattr(gen_model.pipeline, "scheduler", None) is not None:
                sched = gen_model.pipeline.scheduler
                if hasattr(sched, "config") and hasattr(sched.config, "get"):
                    eta_from_scheduler = getattr(sched.config, "eta", None)
                if eta_from_scheduler is None and hasattr(sched, "config"):
                    eta_from_scheduler = getattr(sched.config, "eta", None)
            if eta_from_scheduler is not None:
                eta_val = float(eta_from_scheduler)
                if eta_val == 0.0:
                    variant = "ODE (deterministic, from scheduler)"
                    noise_factor = "0.0"
                else:
                    variant = f"SDE (stochastic, eta={eta_val}, from scheduler)"
                    noise_factor = str(eta_val)
                print(f"Variant: {variant}")
                print(f"SDE noise factor: {noise_factor}")
            else:
                # Infer from solver name (same logic as pipelines/common.get_solver_name)
                from .common import get_solver_name
                solver_name = get_solver_name(cfg)
                if solver_name:
                    solver_name_lower = str(solver_name).lower()
                    # Extract just the class name if it's a full path
                    if "." in solver_name_lower:
                        solver_name_lower = solver_name_lower.split(".")[-1]
                    known_solvers = {"trs", "trust_region", "random_search", "zero_order"}
                    if any(name in solver_name_lower for name in known_solvers):
                        variant = "ODE (deterministic, inferred from solver)"
                        noise_factor = "0.0"
                        print(f"Variant: {variant}")
                        print(f"SDE noise factor: {noise_factor}")
                    else:
                        print(f"Variant: N/A")
                        print(f"SDE noise factor: N/A")
                else:
                    print(f"Variant: N/A")
                    print(f"SDE noise factor: N/A")
    
    # Inference steps
    if modality == "image":
        # Try to get num_inference_steps from pipeline config or model config
        num_inference_steps = None
        if isinstance(pipeline_cfg, dict):
            num_inference_steps = pipeline_cfg.get("num_inference_steps")
        if num_inference_steps is None and isinstance(model_cfg, dict):
            num_inference_steps = model_cfg.get("num_inference_steps")
        
        # If not found, try accessing directly from cfg using OmegaConf
        if num_inference_steps is None:
            try:
                num_inference_steps = OmegaConf.select(cfg, "pipeline.num_inference_steps", default=None)
            except:
                pass
        if num_inference_steps is None:
            try:
                num_inference_steps = OmegaConf.select(cfg, "model.num_inference_steps", default=None)
            except:
                pass
        
        if num_inference_steps is not None:
            print(f"Inference steps: {num_inference_steps}")
        else:
            print(f"Inference steps: N/A")
    elif modality == "protein":
        # Proteina uses dt (time step) and computes nsteps = ceil(1.0 / dt)
        proteina_cfg = cfg.get("proteina") or {}
        if isinstance(proteina_cfg, DictConfig):
            proteina_cfg = OmegaConf.to_container(proteina_cfg, resolve=True) or {}
        
        dt = None
        if isinstance(proteina_cfg, dict):
            dt = proteina_cfg.get("dt")
        
        # If not found, try accessing directly from cfg using OmegaConf
        if dt is None:
            try:
                dt = OmegaConf.select(cfg, "proteina.dt", default=None)
            except:
                pass
        
        if dt is not None:
            nsteps = math.ceil(1.0 / float(dt))
            print(f"Inference steps: {nsteps} (dt={dt})")
        else:
            print(f"Inference steps: N/A")
    
    # Output size
    if modality == "image":
        pipeline_cfg = cfg.get("model") or {}
        height = pipeline_cfg.get("height") or model_cfg.get("height") or 512
        width = pipeline_cfg.get("width") or model_cfg.get("width") or 512
        print(f"Output size: {height}x{width} (resolution)")
    elif modality == "protein":
        # Try to get from benchmark config, proteina config, or cfg
        benchmark_cfg = cfg.get("benchmark") or {}
        proteina_cfg = cfg.get("proteina") or {}
        n_residues = None
        
        # Check benchmark config (may have range like "50-248")
        if isinstance(benchmark_cfg, dict):
            n_residues = benchmark_cfg.get("n_residues")
            # Also check for residue_counts or length_range
            if n_residues is None:
                residue_counts = benchmark_cfg.get("residue_counts")
                if residue_counts:
                    if isinstance(residue_counts, (list, tuple)) and len(residue_counts) > 0:
                        if len(residue_counts) == 2:
                            n_residues = f"{residue_counts[0]}-{residue_counts[1]}"
                        else:
                            n_residues = str(residue_counts)
        
        # Check proteina config
        if n_residues is None and isinstance(proteina_cfg, dict):
            n_residues = proteina_cfg.get("n_residues")
        
        # Check top-level cfg
        if n_residues is None and "n_residues" in cfg:
            n_residues = cfg.get("n_residues")
        
        # Check if gen_model has default_n_residues
        if n_residues is None and hasattr(gen_model, "default_n_residues"):
            n_residues = gen_model.default_n_residues
        
        if n_residues is not None:
            print(f"Output size: {n_residues} residues")
        else:
            print(f"Output size: Variable (depends on protein length)")
    else:
        print(f"Output size: N/A")
    
    print(f"{'='*80}\n")

