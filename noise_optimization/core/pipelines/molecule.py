from __future__ import annotations

from typing import Any, Dict

import hydra
from omegaconf import DictConfig, OmegaConf

from ..models.factories import instantiate_generative_model
from ..rewards.factory import instantiate_reward
from ..loggers.molecule_logger import MoleculeLogger
from ..loggers.base import ExperimentLogger
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


def _make_logger(cfg: DictConfig) -> ExperimentLogger:
    logging_cfg = cfg.get("logging") or {}
    enable = bool(cfg.get("wandb", True))
    project = cfg.get("wandb_project") or "molecule target mean"
    name = get_wandb_run_name(cfg)
    
    return MoleculeLogger(
        project=project,
        name=name,
        config=OmegaConf.to_container(cfg, resolve=True),
        enable=enable,
        logging_config=logging_cfg,
    )


def _make_benchmark(cfg: DictConfig) -> Any:
    """QM9 / molecule benchmark selection.

    Resolution order:
    1. If cfg.benchmark is already a DictConfig/dict with ``_target_``, instantiate directly.
    2. If cfg.benchmark is a string config-group name, load the YAML file and apply any
       ``+benchmark.properties`` Hydra overrides found in the active config.
    3. Fall back to ``cfg.molecule.benchmark`` if present.
    """
    # Resolve benchmark config through multiple access paths (Hydra may compose differently)
    bc = None
    try:
        bc_select = OmegaConf.select(cfg, "benchmark")
        if bc_select is not None:
            bc = bc_select
    except Exception:
        pass
    if bc is None:
        try:
            bc = cfg.benchmark if hasattr(cfg, "benchmark") else cfg.get("benchmark")
        except Exception:
            pass
    if bc is None:
        bc = {}

    # If Hydra has already composed the benchmark (DictConfig/dict with _target_ or properties),
    # use it directly so that command-line overrides (e.g. +benchmark.properties) are preserved.
    if isinstance(bc, DictConfig) and ("_target_" in bc or "properties" in bc):
        OmegaConf.set_struct(bc, False)
        bc["num_runs"] = cfg.get("num_runs", 1)
        OmegaConf.set_struct(bc, True)
    elif isinstance(bc, dict) and "_target_" in bc:
        if "num_runs" not in bc:
            bc["num_runs"] = cfg.get("num_runs", 1)
    else:
        # If benchmark is the 'simple' string but reward is multi-property, switch to qm9_properties
        if isinstance(bc, str) and bc == "simple":
            reward_cfg = cfg.get("reward_function") or {}
            reward_str = str(reward_cfg).lower() if reward_cfg else ""
            reward_target = str(reward_cfg.get("_target_", "")).lower() if isinstance(reward_cfg, dict) else ""
            if "multipropertytargetreward" in reward_target or "multi_property_target" in reward_str or "multipropertytarget" in reward_str:
                bc = "qm9_properties"

        # Load benchmark YAML when specified as a config-group string
        if isinstance(bc, str):
            try:
                import os
                current_dir = os.path.dirname(os.path.abspath(__file__))
                config_dir = os.path.join(current_dir, "..", "..", "config", "benchmark")
                benchmark_config_path = os.path.join(config_dir, f"{bc}.yaml")
                if os.path.exists(benchmark_config_path):
                    bc_raw = OmegaConf.load(benchmark_config_path)

                    # Apply +benchmark.properties override from Hydra task overrides if present
                    properties_override = None
                    try:
                        from hydra.core.global_hydra import GlobalHydra
                        if GlobalHydra().is_initialized():
                            import hydra.core.hydra_config
                            hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
                            if hydra_cfg is not None:
                                for override in hydra_cfg.overrides.task:
                                    if override.startswith("+benchmark.properties=") or override.startswith("benchmark.properties="):
                                        value_str = override.split("=", 1)[1]
                                        import ast, json
                                        try:
                                            properties_override = ast.literal_eval(value_str)
                                        except Exception:
                                            try:
                                                properties_override = json.loads(value_str)
                                            except Exception:
                                                if value_str.strip().startswith('[') and value_str.strip().endswith(']'):
                                                    content = value_str.strip()[1:-1]
                                                    if content:
                                                        properties_override = [p.strip() for p in content.split(',') if p.strip()]
                    except Exception:
                        pass

                    # Also accept properties pre-merged into cfg.benchmark by Hydra
                    if properties_override is None:
                        try:
                            benchmark_from_cfg = cfg.get("benchmark")
                            if isinstance(benchmark_from_cfg, DictConfig) and "properties" in benchmark_from_cfg:
                                properties_override = list(benchmark_from_cfg.properties)
                        except Exception:
                            pass

                    OmegaConf.set_struct(bc_raw, False)
                    if properties_override is not None:
                        bc_raw["properties"] = properties_override
                    bc_raw["num_runs"] = cfg.get("num_runs", 1)
                    OmegaConf.set_struct(bc_raw, True)
                    bc = bc_raw
            except Exception as e:
                import traceback
                traceback.print_exc()

    if isinstance(bc, (dict, DictConfig)) and "_target_" in bc:
        if isinstance(bc, DictConfig):
            OmegaConf.set_struct(bc, False)
            bc["num_runs"] = cfg.get("num_runs", 1)
            OmegaConf.set_struct(bc, True)
            bc_dict = OmegaConf.to_container(bc, resolve=True)
        else:
            bc_dict = dict(bc)
            if "num_runs" not in bc_dict:
                bc_dict["num_runs"] = cfg.get("num_runs", 1)
        return hydra.utils.instantiate(bc_dict, _recursive_=False)

    # Check molecule.benchmark as a secondary source
    mcfg = cfg.get("molecule") or {}
    if isinstance(mcfg, DictConfig):
        mcfg = OmegaConf.to_container(mcfg, resolve=True)
    mb = (mcfg.get("benchmark") or {}) if hasattr(mcfg, "get") else {}
    if isinstance(mb, DictConfig):
        mb = OmegaConf.to_container(mb, resolve=True)
    if hasattr(mb, "get") and mb.get("_target_"):
        if "num_runs" not in mb:
            mb["num_runs"] = cfg.get("num_runs", 1)
        return hydra.utils.instantiate(mb, _recursive_=False)

    raise ValueError(
        "No benchmark configured. Use modality=qm9_target or specify a benchmark config with _target_."
    )


def run(cfg: DictConfig) -> Dict[str, Any]:
    """Run QM9/equiFM molecule generation experiment."""
    validate_and_set_defaults(cfg)
    set_seed(cfg)
    extend_pythonpath(cfg)
    device = get_device(cfg)

    logger = _make_logger(cfg)
    gen_model = instantiate_generative_model(cfg, device=device)
    reward = instantiate_reward(cfg)
    benchmark = _make_benchmark(cfg)
    solver = make_solver(cfg, logger)
    problem_builder = build_problem(cfg, gen_model, reward)
    solver_name = get_solver_name(cfg) or ""
    algo_kwargs = collect_algorithm_kwargs(cfg, solver_name)

    results = benchmark.run(
        algorithm=solver,
        problem_builder=problem_builder,
        algorithm_kwargs=algo_kwargs,
        logger=logger,
    )
    logger.close()
    return {"results": results}

