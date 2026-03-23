from __future__ import annotations

from typing import Any, Dict

import hydra
from omegaconf import DictConfig, OmegaConf

from ..benchmarks.protein import (
    ProteinLengthBenchmark,
    ProteinFoldConditionalBenchmark,
    ProteinMotifScaffoldBenchmark,
)
from ..models.factories import instantiate_generative_model
from ..rewards.factory import instantiate_reward
from ..loggers.base import ExperimentLogger
from ..loggers.protein_logger import ProteinLogger
from .common import (
    build_problem,
    collect_algorithm_kwargs,
    extend_pythonpath,
    get_device,
    get_solver_name,
    get_wandb_run_name,
    make_solver,
    set_seed,
    validate_and_set_defaults,
)


def make_benchmark(cfg: DictConfig) -> Any:
    """Instantiate protein benchmark."""
    bc = cfg.get("benchmark") or {}

    if "_target_" in bc:
        target = bc.get("_target_", "")
        is_protein = "benchmarks.protein" in str(target) or "Protein" in str(target)
        if not is_protein:
            # Default to length benchmark if a non-protein target is given
            return _default_length_benchmark(cfg)
        
        # For ProteinFoldConditionalBenchmark, merge proteina config if needed
        if "ProteinFoldConditionalBenchmark" in str(target):
            # Allow len_cath_code_path to be set from proteina config if not in benchmark config
            pcfg = cfg.get("proteina") or {}
            if isinstance(pcfg, DictConfig):
                pcfg = OmegaConf.to_container(pcfg, resolve=True)
            
            # Merge proteina config into benchmark config for fold conditional
            # Convert bc to dict if it's a DictConfig to make it mutable
            if isinstance(bc, DictConfig):
                bc_dict = OmegaConf.to_container(bc, resolve=True)
            else:
                bc_dict = dict(bc)
            
            # Add proteina config values if not already in benchmark config
            if "len_cath_code_path" not in bc_dict or bc_dict.get("len_cath_code_path") is None:
                if pcfg.get("len_cath_code_path"):
                    bc_dict["len_cath_code_path"] = pcfg.get("len_cath_code_path")
            if "cath_code_level" not in bc_dict or bc_dict.get("cath_code_level") is None:
                if pcfg.get("cath_code_level"):
                    bc_dict["cath_code_level"] = pcfg.get("cath_code_level")
            
            # Remove keys that are not benchmark parameters (like 'proteina' used for model config)
            bc_dict.pop("proteina", None)
            
            # Convert back to DictConfig for hydra instantiation
            bc = OmegaConf.create(bc_dict)
        
        return hydra.utils.instantiate(bc, _recursive_=False)

    return _default_length_benchmark(cfg)


def _default_length_benchmark(cfg: DictConfig) -> ProteinLengthBenchmark:
    pcfg = cfg.get("proteina") or {}
    if isinstance(pcfg, DictConfig):
        pcfg = OmegaConf.to_container(pcfg, resolve=True)
    
    # Check for explicit lengths override
    lengths = pcfg.get("lengths") if hasattr(pcfg, "get") else None
    
    # Get benchmark configuration (sampling parameters)
    benchmark_cfg = pcfg.get("benchmark", {}) if hasattr(pcfg, "get") else {}
    if isinstance(benchmark_cfg, DictConfig):
        benchmark_cfg = OmegaConf.to_container(benchmark_cfg, resolve=True)
    
    # Get benchmark parameters with defaults
    shuffle = benchmark_cfg.get("shuffle", False) if hasattr(benchmark_cfg, "get") else False
    seed = benchmark_cfg.get("seed", 0) if hasattr(benchmark_cfg, "get") else 0
    repeat = benchmark_cfg.get("repeat", 1) if hasattr(benchmark_cfg, "get") else 1
    start_index = benchmark_cfg.get("start_index") if hasattr(benchmark_cfg, "get") else None
    end_index = benchmark_cfg.get("end_index") if hasattr(benchmark_cfg, "get") else None
    num_runs = benchmark_cfg.get("num_runs") if hasattr(benchmark_cfg, "get") else None
    
    # Get sampling parameters (used when lengths=None)
    min_length = benchmark_cfg.get("min_length") if hasattr(benchmark_cfg, "get") else None
    max_length = benchmark_cfg.get("max_length") if hasattr(benchmark_cfg, "get") else None
    num_samples = benchmark_cfg.get("num_samples") if hasattr(benchmark_cfg, "get") else None
    sample_seed = benchmark_cfg.get("sample_seed") if hasattr(benchmark_cfg, "get") else None
    
    # Build kwargs for ProteinLengthBenchmark
    kwargs = {
        "shuffle": shuffle,
        "seed": seed,
        "repeat": repeat,
    }
    if lengths is not None:
        kwargs["lengths"] = lengths
    if start_index is not None:
        kwargs["start_index"] = start_index
    if end_index is not None:
        kwargs["end_index"] = end_index
    if num_runs is not None:
        kwargs["num_runs"] = num_runs
    # Add sampling parameters (only used when lengths=None)
    if min_length is not None:
        kwargs["min_length"] = min_length
    if max_length is not None:
        kwargs["max_length"] = max_length
    if num_samples is not None:
        kwargs["num_samples"] = num_samples
    if sample_seed is not None:
        kwargs["sample_seed"] = sample_seed
    
    return ProteinLengthBenchmark(**kwargs)


def _make_logger(cfg: DictConfig) -> ExperimentLogger:
    logging_cfg = cfg.get("logging") or {}
    enable = bool(cfg.get("wandb", True))
    # Get wandb_project from config, falling back to main.yaml default if not set
    project = cfg.get("wandb_project") or cfg.get("project") or "noise_optimization"
    name = get_wandb_run_name(cfg)
    return ProteinLogger(
        project=project,
        name=name,
        config=cfg,
        enable=enable,
        logging_config=logging_cfg,
    )


def run(cfg: DictConfig) -> Dict[str, Any]:
    """Run protein/proteina experiment."""
    validate_and_set_defaults(cfg)
    set_seed(cfg)
    extend_pythonpath(cfg)
    device = get_device(cfg)
    
    # Auto-configure Proteina sampling parameters based on noise settings
    solver_name = get_solver_name(cfg) or ""

    # Check if precomputed noise or SDE noise is being used
    proteina_cfg = cfg.get("proteina") or {}
    has_precomputed_noise = (
        isinstance(proteina_cfg, dict) and
        ("precomputed_noise_path" in proteina_cfg or "precomputed_noise" in proteina_cfg)
    )
    has_precomputed_sde_noise = (
        isinstance(proteina_cfg, dict) and
        ("precomputed_sde_noise_path" in proteina_cfg or "precomputed_sde_noise" in proteina_cfg)
    )

    # Set sampling parameters for deterministic behavior
    # Only set if not already specified in config
    if isinstance(proteina_cfg, dict):
        OmegaConf.set_struct(cfg, False)
        if "proteina" not in cfg:
            cfg["proteina"] = {}

        # Use SDE sampling for better performance, but make it deterministic with precomputed noise
        if "sampling_mode" not in proteina_cfg:
            cfg["proteina"]["sampling_mode"] = "sc"  # Score-based SDE sampling (better performance)

        if has_precomputed_sde_noise:
            # With precomputed SDE noise, keep stochastic scale but use deterministic noise
            if "sc_scale_noise" not in proteina_cfg:
                cfg["proteina"]["sc_scale_noise"] = 0.4  # Keep stochastic scale for SDE strength
            sampling_desc = "SDE (deterministic noise)"
        elif has_precomputed_noise:
            # For precomputed initial noise, auto-generate reproducible SDE noise sequence
            if "sc_scale_noise" not in proteina_cfg:
                cfg["proteina"]["sc_scale_noise"] = 0.4  # Keep stochastic scale for SDE strength
            if "sde_noise_seed" not in proteina_cfg:
                cfg["proteina"]["sde_noise_seed"] = 42  # Fixed seed for reproducibility
            if "sde_noise_seq_length" not in proteina_cfg:
                cfg["proteina"]["sde_noise_seq_length"] = 1000  # Sufficient for most runs
            sampling_desc = "SDE (reproducible)"
        else:
            # Use deterministic SDE
            if "sc_scale_noise" not in proteina_cfg:
                cfg["proteina"]["sc_scale_noise"] = 0.0  # No stochastic noise
            sampling_desc = "SDE (deterministic)"

        OmegaConf.set_struct(cfg, True)

        if has_precomputed_sde_noise:
            print(f"[INFO] Using precomputed SDE noise sequence with {sampling_desc} sampling")
        elif has_precomputed_noise:
            print(f"[INFO] Using precomputed initial noise with auto-generated SDE sequence (seed=42) for full reproducibility")
        else:
            print(f"[INFO] Auto-configured Proteina sampling: {sampling_desc} for {solver_name}")

    logger = _make_logger(cfg)
    gen_model = instantiate_generative_model(cfg, device=device)
    reward = instantiate_reward(cfg)
    benchmark = make_benchmark(cfg)
    solver = make_solver(cfg, logger)
    problem_builder = build_problem(cfg, gen_model, reward)
    algo_kwargs = collect_algorithm_kwargs(cfg, solver_name)
    
    # Print model summary
    from .model_summary import print_model_summary
    print_model_summary(gen_model, cfg, modality="protein", problem_builder=problem_builder)

    results = benchmark.run(
        algorithm=solver,
        problem_builder=problem_builder,
        algorithm_kwargs=algo_kwargs,
        logger=logger,
    )
    logger.close()
    return {"results": results}

