from __future__ import annotations

from typing import Any, Dict

import hydra
from omegaconf import DictConfig, OmegaConf

from ..benchmarks.t2i import (
    COCOCaptionsBenchmark,
    CustomPromptBenchmark,
    DrawBenchBenchmark,
    Gen2EvalBenchmark,
    ListPromptsBenchmark,
    SimpleAnimalsBenchmark,
)
from ..models.factories import instantiate_generative_model
from ..rewards.factory import instantiate_reward
from ..loggers.base import ExperimentLogger
from ..loggers.t2i import T2IWandbLogger
from ..loggers.scaling_logger import ScalingLogger
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
    """Instantiate image/text benchmark (T2I)."""
    bc = cfg.get("benchmark") or {}

    # Handle case where benchmark is passed as a string (e.g., benchmark=draw_bench)
    if isinstance(bc, str):
        name = bc.lower()
        # Convert to dict-like structure for compatibility with rest of function
        bc = {"name": name}

    # Hydra-targeted benchmark
    if "_target_" in bc:
        return hydra.utils.instantiate(bc, _recursive_=False)

    # Normalize name by removing underscores for flexible matching
    name = (bc.get("name") or "drawbench").lower().replace("_", "")
    if name == "drawbench":
        start_val = cfg.get("start_index") or cfg.get("start_prompt")
        start_index = int(start_val) if start_val is not None else 0
        end_val = cfg.get("end_index") or cfg.get("end_prompt")
        end_index = int(end_val) if end_val is not None else cfg.get("num_runs", 1)
        return DrawBenchBenchmark(
            num_runs=cfg.get("num_runs", 1),
            randomize=bc.get("randomize", False),
            path=bc.get("path"),
            max_prompts=bc.get("max_prompts"),
            start_index=start_index,
            end_index=end_index,
        )
    if name == "simpleanimals":
        return SimpleAnimalsBenchmark(
            num_runs=cfg.get("num_runs", 1),
            randomize=bc.get("randomize", False),
            path=bc.get("path"),
            max_prompts=bc.get("max_prompts"),
        )
    if name == "gen2eval":
        return Gen2EvalBenchmark(
            path=bc.get("path"),
            num_runs=cfg.get("num_runs", 1),
            randomize=bc.get("randomize", False),
            max_prompts=bc.get("max_prompts"),
            start_index=bc.get("start_index") or cfg.get("start_index"),
            end_index=bc.get("end_index") or cfg.get("end_index"),
        )
    if name == "cococaptions":
        return COCOCaptionsBenchmark(
            captions_json=bc.get("captions_json"),
            num_runs=cfg.get("num_runs", 1),
            randomize=bc.get("randomize", False),
            caption_key=bc.get("caption_key", "annotations"),
            text_key=bc.get("text_key", "caption"),
            max_prompts=bc.get("max_prompts"),
        )
    if name == "promptlist":
        return ListPromptsBenchmark(
            prompts=bc.get("prompts", []),
            num_runs=cfg.get("num_runs", 1),
            randomize=bc.get("randomize", False),
        )
    if name == "customprompt":
        return CustomPromptBenchmark(
            prompts=bc.get("prompts", []),
            num_runs=cfg.get("num_runs", None),
            randomize=bc.get("randomize", False),
            log_iteration_images=bc.get("log_iteration_images", True),
            start_index=bc.get("start_index"),
            end_index=bc.get("end_index"),
        )
    raise ValueError(f"Unknown T2I benchmark name: {name}")


def _make_logger(cfg: DictConfig) -> ExperimentLogger:
    logging_cfg = cfg.get("logging") or {}
    enable = bool(cfg.get("wandb", True))
    project = cfg.get("wandb_project", cfg.get("project", "draw_bench"))
    name = get_wandb_run_name(cfg)

    wandb_section = logging_cfg.get("wandb", {})
    scaling_cfg = logging_cfg.get("scaling", {})
    scaling_enabled = bool(
        scaling_cfg.get("enabled", False) or wandb_section.get("scaling", {}).get("enabled", False)
    )
    if scaling_enabled:
        budget_checkpoints = (
            scaling_cfg.get("budget_checkpoints")
            or wandb_section.get("scaling", {}).get("budget_checkpoints")
            or None
        )
        return ScalingLogger(
            project=project,
            name=name,
            config=cfg,
            enable=enable,
            budget_checkpoints=budget_checkpoints,
            logging_config=logging_cfg,
        )

    return T2IWandbLogger(
        project=project,
        name=name,
        config=cfg,
        enable=enable,
        logging_config=logging_cfg,
    )


def run(cfg: DictConfig) -> Dict[str, Any]:
    """Run a text-to-image experiment."""
    validate_and_set_defaults(cfg)
    set_seed(cfg)
    extend_pythonpath(cfg)
    device = get_device(cfg)

    solver_name = get_solver_name(cfg) or ""

    # Set DDIM scheduler and eta=0 (deterministic) if not already specified in config
    OmegaConf.set_struct(cfg, False)
    if "model" not in cfg:
        cfg["model"] = {}
    pipeline_cfg = cfg.get("model") or {}
    if not pipeline_cfg.get("scheduler") or str(pipeline_cfg.get("scheduler")).lower() in ("none", "null", ""):
        cfg["model"]["scheduler"] = "ddim"
    if OmegaConf.select(cfg, "model.eta", default=None) is None:
        cfg["model"]["eta"] = 0.0
    OmegaConf.set_struct(cfg, True)

    logger = _make_logger(cfg)
    gen_model = instantiate_generative_model(cfg, device=device)
    reward = instantiate_reward(cfg)
    benchmark = make_benchmark(cfg)
    solver = make_solver(cfg, logger)
    problem_builder = build_problem(cfg, gen_model, reward)
    algo_kwargs = collect_algorithm_kwargs(cfg, solver_name)
    
    # Print model summary
    from .model_summary import print_model_summary
    print_model_summary(gen_model, cfg, modality="image", problem_builder=problem_builder)

    results = benchmark.run(
        algorithm=solver,
        problem_builder=problem_builder,
        algorithm_kwargs=algo_kwargs,
        logger=logger,
    )
    logger.close()
    return {"results": results}

