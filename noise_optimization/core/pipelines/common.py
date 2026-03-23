from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..loggers.base import ExperimentLogger, WandbLogger
    from ..loggers.t2i import T2IWandbLogger
    from ..loggers.scaling_logger import ScalingLogger

import os
import sys

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from ..problems.builder import ProblemBuilder


# --- shared utilities ---------------------------------------------------------

def extend_pythonpath(cfg: DictConfig) -> None:
    """Optionally extend sys.path from cfg.pythonpath."""
    pyp = cfg.get("pythonpath")
    if pyp is None:
        return
    paths = list(pyp) if not isinstance(pyp, (list, tuple)) else pyp
    for p in paths:
        if isinstance(p, str):
            ap = os.path.expanduser(p)
            if ap and (ap not in sys.path):
                sys.path.append(ap)


def set_seed(cfg: DictConfig) -> None:
    """Seed global RNGs if cfg.seed (or model.seed) is provided."""
    from ..utils import set_global_seed

    seed_val = cfg.get("seed")
    if seed_val is None:
        pip = cfg.get("model") or {}
        if hasattr(pip, "get"):
            seed_val = pip.get("seed")
    if seed_val is not None:
        set_global_seed(int(seed_val), deterministic=bool(cfg.get("deterministic", False)))


def get_device(cfg: DictConfig) -> str:
    """Choose device with SLURM awareness, warn if GPU allocated but not visible."""
    cuda_available = torch.cuda.is_available()
    requested_device = cfg.get("device", "cuda")
    gpu_allocated = any(
        [
            os.environ.get("SLURM_GPUS_ON_NODE", "0") != "0",
            os.environ.get("SLURM_JOB_GPUS", "0") != "0",
            os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        ]
    )
    if requested_device == "cuda" and not cuda_available and gpu_allocated:
        print(
            f"[WARNING] GPU allocated (SLURM_GPUS_ON_NODE={os.environ.get('SLURM_GPUS_ON_NODE')}) "
            f"but torch.cuda.is_available()=False"
        )
        print("[WARNING] Proceeding on CPU; check CUDA setup or SLURM launch if unexpected.")
    return "cuda" if cuda_available and requested_device == "cuda" else "cpu"


def build_problem(cfg: DictConfig, gen_model: Any, reward: Any) -> ProblemBuilder:
    device = "cuda" if torch.cuda.is_available() and cfg.get("device", "cuda") == "cuda" else "cpu"
    return ProblemBuilder(cfg, gen_model, reward, device)


def get_solver_name(cfg: DictConfig) -> Optional[str]:
    """Extract solver name from config.
    
    Supports both new structure (solver config group) and legacy (solver string or solvers dict).
    """
    sol_cfg = cfg.get("solver") or cfg.get("algorithm")
    name: Optional[str] = None
    
    if isinstance(sol_cfg, (dict, DictConfig)):
        # New structure: solver is loaded from config group, check for common params
        # to infer the solver type, or look for explicit name
        name = sol_cfg.get("name") or sol_cfg.get("Name")
        if name is None:
            # Infer from config structure
            if "tr" in sol_cfg:
                if "surrogate" in sol_cfg:
                    name = "bo"
                else:
                    name = "trs"
            elif "num_particles" in sol_cfg:
                if "potential_type" in sol_cfg:
                    name = "smc_fks"
                elif "duplicate_size" in sol_cfg:
                    name = "smc_dsearch"
                elif "epsilon" in sol_cfg and "extreme_threshold" in sol_cfg:
                    name = "smc_nts"
                elif "repeats" in sol_cfg:
                    name = "smc_svdd"
            elif "popsize" in sol_cfg:
                if "tournament_size" in sol_cfg:
                    name = "cosyne"
                elif "lr_mean" in sol_cfg:
                    name = "snes"
            elif "elite_frac" in sol_cfg:
                name = "cem"
            elif "step_size" in sol_cfg:
                name = "zero_order"
            elif "cand_pool_size" in sol_cfg:  # MSR uses global pool
                name = "msr"
            elif "use_pca_scoring" in sol_cfg:  # FuRBO specific
                name = "furbo"
    elif isinstance(sol_cfg, str):
        name = sol_cfg
    
    return name or None


# --- modality validation -----------------------------------------------------

# Valid rewards per modality
VALID_REWARDS_BY_MODALITY: Dict[str, list[str]] = {
    "image": ["clip", "image_reward", "imagereward", "aesthetic", "hps", "hps_v2", "brightness", "contrast", "sharpness", "saturation", "redness", "jpeg_compressibility", "radial_brightness", "combined"],
    "image_noise": ["clip", "image_reward", "imagereward", "aesthetic", "hps", "hps_v2", "brightness", "contrast", "sharpness", "saturation", "redness", "jpeg_compressibility", "radial_brightness", "combined"],
    "t2i": ["clip", "image_reward", "imagereward", "aesthetic", "hps", "hps_v2", "brightness", "contrast", "sharpness", "saturation", "redness", "jpeg_compressibility", "radial_brightness", "combined"],
    "molecule": ["multi_property_target"],
    "qm9": ["multi_property_target"],
    "qm9_target": ["multi_property_target"],
    "protein": ["designability", "scrmsd", "backbone_quality", "fold_confidence", "fold_match", "plddt", "end_to_end_distance", "mirror", "aa_composition_fraction_str"],
    "proteina": ["designability", "scrmsd", "backbone_quality", "fold_confidence", "fold_match", "plddt", "end_to_end_distance", "mirror", "aa_composition_fraction_str"],
}

# Default rewards per modality
DEFAULT_REWARDS_BY_MODALITY: Dict[str, str] = {
    "image": "clip",
    "image_noise": "clip",
    "t2i": "clip",
    "molecule": "multi_property_target",
    "qm9": "multi_property_target",
    "protein": "plddt",
    "proteina": "plddt",
}

# Valid benchmarks per modality
VALID_BENCHMARKS_BY_MODALITY: Dict[str, list[str]] = {
    "image": ["draw_bench", "draw_bench_50", "draw_bench_100", "draw_bench_core", "draw_bench_10", "simple_animals", "paper_prompts", "custom_prompt", "coco_captions", "gen2eval"],
    "image_noise": ["draw_bench", "draw_bench_50", "draw_bench_100", "draw_bench_core", "draw_bench_10", "simple_animals", "paper_prompts", "custom_prompt", "coco_captions", "gen2eval"],
    "t2i": ["draw_bench", "draw_bench_50", "draw_bench_100", "draw_bench_core", "draw_bench_10", "simple_animals", "paper_prompts", "custom_prompt", "coco_captions", "gen2eval"],
    "molecule": ["simple", "qm9_properties", "qm9_scalar"],
    "qm9": ["simple", "qm9_properties", "qm9_scalar"],
    "qm9_target": ["simple", "qm9_properties", "qm9_scalar"],
    "protein": ["protein_lengths", "protein_unconditional", "protein_fold_conditional", "protein_motif_scaffold"],
    "proteina": ["protein_lengths", "protein_unconditional", "protein_fold_conditional", "protein_motif_scaffold"],
}

# Default benchmarks per modality
DEFAULT_BENCHMARKS_BY_MODALITY: Dict[str, str] = {
    "image": "draw_bench",
    "image_noise": "draw_bench",
    "t2i": "draw_bench",
    "molecule": "simple",
    "qm9": "simple",
    "qm9_target": "qm9_properties",
    "protein": "protein_lengths",
    "proteina": "protein_lengths",
}


def _get_reward_name(cfg: DictConfig) -> Optional[str]:
    """Extract reward name from config."""
    # Check objective first
    objective = cfg.get("objective")
    if objective:
        return str(objective).lower()
    
    # Check reward_function (Hydra config group name)
    rf_cfg = cfg.get("reward_function") or cfg.get("reward_model")
    if rf_cfg is None:
        return None
    
    # If it's a string, it's likely a config group name (e.g., "clip", "qed", "plddt")
    if isinstance(rf_cfg, str):
        return rf_cfg.lower()
    
    # If it's a dict or DictConfig, check for _target_ (Hydra instantiation target)
    # Handle both dict and DictConfig (OmegaConf)
    if isinstance(rf_cfg, (dict, DictConfig)):
        target = rf_cfg.get("_target_", "")
        if target:
            # Extract name from target path
            target_lower = target.lower()
            if "imagereward" in target_lower or "image_reward" in target_lower:
                return "image_reward"
            if "clip" in target_lower and "image" in target_lower:
                return "clip"
            if "combined" in target_lower:
                return "combined"
            if "aesthetic" in target_lower:
                return "aesthetic"
            if "hps" in target_lower and "image" in target_lower:
                return "hps"
            if "pickscore" in target_lower or "pick_score" in target_lower:
                return "pick_score"
            if "qed" in target_lower and "molecule" in target_lower:
                return "qed"
            if ("sa" in target_lower or "syntheticaccessibility" in target_lower) and "molecule" in target_lower:
                return "sa"
            if "logp" in target_lower and "molecule" in target_lower:
                return "logp"
            if ("molecularweight" in target_lower or "molecular_weight" in target_lower) and "molecule" in target_lower:
                return "molecular_weight"
            if ("gfn2xtb" in target_lower or "gfn2_xtb" in target_lower or "xtbenergy" in target_lower) and "molecule" in target_lower:
                return "gfn2xtb_energy"
            if ("valencystability" in target_lower or "valency_stability" in target_lower) and "molecule" in target_lower:
                return "valency_stability"
            if ("valencycompliance" in target_lower or "valency_compliance" in target_lower) and "molecule" in target_lower:
                return "valency_compliance"
            if ("geometricconsistency" in target_lower or "geometric_consistency" in target_lower) and "molecule" in target_lower:
                return "geometric_consistency"
            if "combined" in target_lower and "molecule" in target_lower:
                return "molecule_combined"
            if "multipropertytarget" in target_lower or "multi_property_target" in target_lower:
                return "multi_property_target"
            # Check for plddt first (more specific check - look for class name pattern)
            # PLDDTReward class name should appear in target: "plddtreward" or ends with ".plddt"
            if "protein" in target_lower and ("plddtreward" in target_lower or target_lower.endswith(".plddt") or ".plddt" in target_lower):
                return "plddt"
            if ("designability" in target_lower or "scrmsd" in target_lower) and "protein" in target_lower:
                return "designability"
            if "backbonequality" in target_lower and "protein" in target_lower:
                return "backbone_quality"
            if "foldconfidence" in target_lower and "protein" in target_lower:
                return "fold_confidence"
            if "foldmatch" in target_lower and "protein" in target_lower:
                return "fold_match"
        # Check for name field (fallback)
        name = rf_cfg.get("name")
        if name:
            return str(name).lower()
        # Last resort: try to infer from _target_ class name
        if target:
            # Extract class name from full path (e.g., "noise_optimization.core.rewards.protein.PLDDTReward" -> "plddtreward")
            parts = target_lower.split(".")
            if parts:
                class_name = parts[-1]
                # Map common class name patterns to reward names
                if class_name == "plddtreward":
                    return "plddt"
                if class_name == "designabilityreward" or class_name == "scrmsdreward":
                    return "designability"
                if class_name == "backbonequalityreward":
                    return "backbone_quality"
                if class_name == "foldconfidencereward":
                    return "fold_confidence"
                if class_name == "foldmatchreward":
                    return "fold_match"
                if class_name == "gfn2xtbenergyreward":
                    return "gfn2xtb_energy"
                if class_name == "valencystabilityreward":
                    return "valency_stability"
                if class_name == "valencycompliancereward":
                    return "valency_compliance"
                if class_name == "geometricconsistencyreward":
                    return "geometric_consistency"
                if class_name == "multipropertytargetreward":
                    return "multi_property_target"
    return None


def _get_benchmark_name(cfg: DictConfig) -> Optional[str]:
    """Extract benchmark name from config."""
    bc = cfg.get("benchmark")
    if bc is None:
        return None
    
    if isinstance(bc, str):
        return bc.lower()
    
    # Handle both dict and OmegaConf DictConfig
    if isinstance(bc, (dict, DictConfig)):
        # Check for _target_ (Hydra target)
        target = bc.get("_target_", "")
        if target:
            # Extract name from target (e.g., "DrawBenchBenchmark" -> "draw_bench")
            if "CustomPrompt" in target:
                return "custom_prompt"
            if "DrawBench" in target:
                return "draw_bench"
            if "SimpleAnimals" in target:
                return "simple_animals"
            if "PaperPrompts" in target:
                return "paper_prompts"
            if "COCOCaptions" in target:
                return "coco_captions"
            if "Gen2Eval" in target:
                return "gen2eval"
            if "SimpleMolecule" in target:
                return "simple"
            if "QM9Properties" in target:
                return "qm9_properties"
            if "ProteinLength" in target:
                return "protein_lengths"
            if "ProteinUnconditional" in target:
                return "protein_unconditional"
            if "ProteinFoldConditional" in target:
                return "protein_fold_conditional"
            if "ProteinMotifScaffold" in target:
                return "protein_motif_scaffold"
        # Check for name field
        name = bc.get("name")
        if name:
            return str(name).lower()
    
    return None


def _get_available_options() -> Dict[str, list]:
    """Get available options for each category."""
    return {
        "benchmarks": {
            "image": ["draw_bench", "custom_prompt", "simple_animals", "paper_prompts", "gen2eval"],
            "protein": ["protein_lengths", "protein_fold_conditional", "protein_motif_scaffold", "protein_unconditional"],
            "molecule": ["qm9_properties", "simple"],
        },
        "reward_functions": {
            "image": ["aesthetic", "hps", "clip", "image_reward", "brightness", "contrast", "sharpness", "saturation", "redness", "jpeg_compressibility", "combined"],
            "protein": ["plddt", "designability", "fold_confidence", "fold_match", "backbone_quality", "scrmsd"],
            "molecule": ["qed", "sa", "logp", "molecular_weight", "valency_stability", "valency_compliance", "gfn2xtb_energy", "geometric_consistency", "molecule_combined"],
        },
        "solvers": ["trs", "random_search", "zero_order"],
        "modalities": ["image", "protein", "molecule"],
    }


def _print_helpful_error(cfg: DictConfig) -> None:
    """Print helpful error message when no benchmark/prompt is specified."""
    options = _get_available_options()
    modality = (cfg.get("modality") or "image").lower()
    
    print("\n" + "="*80)
    print("❌ Please specify a prompt or choose one of the following benchmarks:")
    print("="*80)
    
    if modality == "image":
        print("\n📝 Quick Start - Use a custom prompt:")
        print("   python main.py prompt=\"Your prompt here\"")
        print("\n📊 Available Image Benchmarks:")
        for bench in options["benchmarks"]["image"]:
            print(f"   - benchmark={bench}")
    else:
        print(f"\n📊 Available {modality.capitalize()} Benchmarks:")
        for bench in options["benchmarks"].get(modality, []):
            print(f"   - benchmark={bench}")
    
    print(f"\n🎯 Current Modality: {modality}")
    if modality != "image":
        print("   To optimize images, use: modality=image")
    if modality != "protein":
        print("   To optimize proteins, use: modality=protein")
    if modality != "molecule":
        print("   To optimize small molecules (QM9), use: modality=molecule")
    
    print(f"\n🎨 Available Reward Functions for {modality}:")
    for reward in options["reward_functions"].get(modality, []):
        print(f"   - reward_function={reward}")
    
    print(f"\n⚙️  Available Solvers:")
    for solver in options["solvers"]:
        print(f"   - solver={solver}")
    
    print("\n💡 Example Commands:")
    if modality == "image":
        print('   python main.py prompt="A beautiful sunset" reward_function=aesthetic solver=trs')
    else:
        print(f'   python main.py modality={modality} benchmark={options["benchmarks"][modality][0] if options["benchmarks"].get(modality) else "N/A"} reward_function={options["reward_functions"][modality][0] if options["reward_functions"].get(modality) else "N/A"} solver=trs')
    
    print("\n📖 For further details, refer to the README.md")
    print("="*80 + "\n")


def validate_and_set_defaults(cfg: DictConfig) -> None:
    """Validate reward and benchmark for the given modality, set defaults if invalid."""
    modality = (cfg.get("modality") or "image").lower()
    
    # Normalize modality aliases
    if modality in ("image_noise", "t2i"):
        modality = "image"
    elif modality == "qm9":
        modality = "molecule"
    
    # Handle legacy keywords for backward compatibility
    # Renamed save_best_structures / save_images -> save_outputs
    if cfg.get("save_best_structures") is not None:
        OmegaConf.set_struct(cfg, False)
        cfg["save_outputs"] = cfg.get("save_best_structures")
        OmegaConf.set_struct(cfg, True)
    if cfg.get("save_images") is not None:
        OmegaConf.set_struct(cfg, False)
        cfg["save_outputs"] = cfg.get("save_images")
        OmegaConf.set_struct(cfg, True)
    if cfg.get("save_best_structures_dir") is not None:
        OmegaConf.set_struct(cfg, False)
        cfg["save_outputs_dir"] = cfg.get("save_best_structures_dir")
        OmegaConf.set_struct(cfg, True)
    if cfg.get("save_images_dir") is not None:
        OmegaConf.set_struct(cfg, False)
        cfg["save_outputs_dir"] = cfg.get("save_images_dir")
        OmegaConf.set_struct(cfg, True)
    
    # Validate and set reward
    reward_name = _get_reward_name(cfg)
    valid_rewards = VALID_REWARDS_BY_MODALITY.get(modality, [])
    default_reward = DEFAULT_REWARDS_BY_MODALITY.get(modality, "clip")
    
    rf_cfg = cfg.get("reward_function") or cfg.get("reward_model")
    # Also check if reward_function is a dict/DictConfig with _target_ pointing to wrong modality
    if isinstance(rf_cfg, (dict, DictConfig)):
        target = rf_cfg.get("_target_", "").lower()
        # Check if target points to wrong modality
        if modality == "molecule" and "image" in target and "reward" in target:
            reward_name = None  # Force replacement
        elif modality == "protein" and ("image" in target or "molecule" in target) and "reward" in target:
            reward_name = None  # Force replacement
        elif modality == "image" and ("molecule" in target or "protein" in target) and "reward" in target:
            reward_name = None  # Force replacement
    
    if reward_name is None or reward_name not in valid_rewards:
        if reward_name is not None:
            print(f"[WARNING] Reward '{reward_name}' is not valid for modality '{modality}'. Using default '{default_reward}'.")
        OmegaConf.set_struct(cfg, False)
        cfg["reward_function"] = default_reward
        OmegaConf.set_struct(cfg, True)
    
    # Handle prompt= shortcut for image modality (convert to custom benchmark)
    # Override the DEFAULT image benchmark if prompt is provided
    if modality == "image":
        prompt = cfg.get("prompt")
        if prompt and prompt not in (None, "null", "none", ""):
            benchmark = cfg.get("benchmark")
            # If benchmark is DrawBench (default), override it with the prompt.
            bench_target = ""
            bench_name = ""
            if isinstance(benchmark, (dict, DictConfig)):
                bench_target = str(benchmark.get("_target_", "") or "")
                bench_name = str(benchmark.get("name", "") or "")
            elif isinstance(benchmark, str):
                bench_name = benchmark

            is_default_benchmark = (
                str(bench_name).strip().lower().replace("_", "") in ("drawbench", "draw_bench")
                or ("DrawBenchBenchmark" in bench_target)
                or (bench_target.lower().endswith("drawbenchbenchmark"))
            )
            
            # Convert prompt to custom benchmark if it's the default or no benchmark was set
            if is_default_benchmark or not benchmark or (isinstance(benchmark, (dict, DictConfig)) and not benchmark.get("_target_") and not benchmark.get("prompts")):
                if isinstance(prompt, str):
                    prompts_list = [prompt]
                elif isinstance(prompt, list):
                    prompts_list = prompt
                else:
                    prompts_list = [str(prompt)]
                
                # Set up custom benchmark
                OmegaConf.set_struct(cfg, False)
                cfg["benchmark"] = {
                    "_target_": "noise_optimization.core.benchmarks.t2i.CustomPromptBenchmark",
                    "prompts": prompts_list,
                    "log_iteration_images": True
                }
                
                # Force iteration image logging in the global config
                if "logging" not in cfg: cfg["logging"] = {}
                if "wandb" not in cfg["logging"]: cfg["logging"]["wandb"] = {}
                if "iteration" not in cfg["logging"]["wandb"]: cfg["logging"]["wandb"]["iteration"] = {}
                
                cfg["logging"]["wandb"]["iteration"]["enabled"] = True
                cfg["logging"]["wandb"]["iteration"]["log_data_samples"] = True
                
                OmegaConf.set_struct(cfg, True)

    # Handle n_residues= shortcut for protein modality
    if modality in ("protein", "proteina"):
        n_res = cfg.get("n_residues")
        if n_res and str(n_res).lower() not in ("null", "none", ""):
            # Set up ProteinLengthBenchmark
            OmegaConf.set_struct(cfg, False)
            cfg["benchmark"] = {
                "_target_": "noise_optimization.core.benchmarks.protein.ProteinLengthBenchmark",
                "lengths": [int(n_res)],
                "num_runs": 1
            }
            # Automatically enable structure saving
            cfg["save_outputs"] = True

            # Force iteration logging for visualization
            if "logging" not in cfg: cfg["logging"] = {}
            if "wandb" not in cfg["logging"]: cfg["logging"]["wandb"] = {}
            if "iteration" not in cfg["logging"]["wandb"]: cfg["logging"]["wandb"]["iteration"] = {}
            cfg["logging"]["wandb"]["iteration"]["enabled"] = True
            cfg["logging"]["wandb"]["iteration"]["log_data_samples"] = True

            OmegaConf.set_struct(cfg, True)

    # Handle n_atoms= shortcut for qm9 modality
    if modality in ("molecule", "qm9"):
        n_atoms = cfg.get("n_atoms")
        if n_atoms and str(n_atoms).lower() not in ("null", "none", ""):
            OmegaConf.set_struct(cfg, False)
            cfg["benchmark"] = "simple"
            cfg["save_outputs"] = True

            if "logging" not in cfg: cfg["logging"] = {}
            if "wandb" not in cfg["logging"]: cfg["logging"]["wandb"] = {}
            if "iteration" not in cfg["logging"]["wandb"]: cfg["logging"]["wandb"]["iteration"] = {}
            cfg["logging"]["wandb"]["iteration"]["enabled"] = True
            cfg["logging"]["wandb"]["iteration"]["log_data_samples"] = True

            OmegaConf.set_struct(cfg, True)

    # Validate and set benchmark
    benchmark_name = _get_benchmark_name(cfg)
    valid_benchmarks = VALID_BENCHMARKS_BY_MODALITY.get(modality, [])
    default_benchmark = DEFAULT_BENCHMARKS_BY_MODALITY.get(modality, "draw_bench")
    
    # Also check if benchmark is a dict with _target_ pointing to wrong modality
    bc = cfg.get("benchmark")
    if isinstance(bc, (dict, DictConfig)):
        target = bc.get("_target_", "").lower()
        # Check if target points to wrong modality
        if modality == "molecule" and ("image" in target or "protein" in target) and "benchmark" in target:
            benchmark_name = None  # Force replacement
        elif modality == "protein" and ("image" in target or "molecule" in target) and "benchmark" in target:
            benchmark_name = None  # Force replacement
        elif modality == "image" and ("molecule" in target or "protein" in target) and "benchmark" in target:
            benchmark_name = None  # Force replacement
    
    # For image modality, check if benchmark or prompt is specified
    if modality == "image":
        has_prompt = bool(cfg.get("prompt"))
        benchmark = cfg.get("benchmark")
        
        has_valid_benchmark = (
            benchmark and (
                (isinstance(benchmark, (dict, DictConfig)) and (benchmark.get("_target_") or benchmark.get("name") or benchmark.get("prompts"))) or
                (isinstance(benchmark, str) and benchmark.strip() and benchmark.lower() not in ("null", "none", ""))
            )
        )
        
        # Backward compatibility/robustness: if benchmark is a string that matches a known benchmark, 
        # it will be handled later by setting the default, but we should consider it "valid" for this check
        if not has_valid_benchmark and isinstance(benchmark, str) and benchmark.lower() in VALID_BENCHMARKS_BY_MODALITY.get(modality, []):
            has_valid_benchmark = True
            
        # Check if this is a "no arguments" run (only task override is the config name)
        is_no_args_run = False
        try:
            from hydra.core.global_hydra import GlobalHydra
            if GlobalHydra().is_initialized():
                import hydra.core.hydra_config
                hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
                if hydra_cfg is not None:
                    overrides = hydra_cfg.overrides.task
                    # A truly no-args run will have no task overrides
                    is_no_args_run = len(overrides) == 0
        except Exception:
            pass
        
        # Show error ONLY if we have neither a valid benchmark nor a prompt
        # AND we are in a no-args run (to prevent accidental default runs)
        if not has_valid_benchmark and not has_prompt:
            _print_helpful_error(cfg)
            raise ValueError("No benchmark or prompt specified. Please specify a prompt or choose a benchmark.")
        
        # If it's a no-args run, we still might want to caution the user, 
        # but let's allow it if there is a default benchmark.
        if is_no_args_run and not has_prompt:
            print("[INFO] No prompt or benchmark specified in command line. Using default benchmark.")

    # Only set default benchmark if we don't have a valid one AND we didn't just create one from prompt
    if benchmark_name is None or benchmark_name not in valid_benchmarks:
        # Check if we just created a custom benchmark from prompt (it won't be in valid_benchmarks list)
        bc_after = cfg.get("benchmark")
        has_custom_from_prompt = (
            isinstance(bc_after, (dict, DictConfig)) and 
            bc_after.get("_target_", "").endswith("CustomPromptBenchmark")
        )
        
        if not has_custom_from_prompt:
            if benchmark_name is not None:
                print(f"[WARNING] Benchmark '{benchmark_name}' is not valid for modality '{modality}'. Using default '{default_benchmark}'.")
            OmegaConf.set_struct(cfg, False)
            cfg["benchmark"] = default_benchmark
            OmegaConf.set_struct(cfg, True)
    


# --- solver factories ---------------------------------------------------------

def make_solver(cfg: DictConfig, logger: Any) -> Any:
    sol_cfg = cfg.get("solver") or cfg.get("algorithm")
    if sol_cfg is None:
        raise ValueError("solver config is required (set cfg.solver or legacy cfg.algorithm)")

    # Handle both dict and DictConfig
    if isinstance(sol_cfg, (dict, DictConfig)) and ("_target_" in sol_cfg):
        return hydra.utils.instantiate(sol_cfg, _recursive_=False, logger=logger)

    name: Optional[str] = None
    if isinstance(sol_cfg, (dict, DictConfig)):
        name = sol_cfg.get("name")
    elif isinstance(sol_cfg, str):
        name = sol_cfg

    if isinstance(name, str):
        n = name.lower()
        from ..solvers.trs import TrustRegionSolver
        from ..solvers.random_search import RandomSearchSolver
        from ..solvers.zero_order import ZeroOrderSolver

        if n in ("trust_region", "trustregion", "turbo", "trs"):
            return TrustRegionSolver(logger=logger)
        if n in ("random_search", "randomsearch", "rs"):
            return RandomSearchSolver(logger=logger)
        if n in ("zero_order", "zero", "zoo"):
            return ZeroOrderSolver(logger=logger)

        # Unknown solver name - raise error instead of silently returning DictConfig
        raise ValueError(f"Unknown solver name: '{name}'. Available: trs, random_search, zero_order")
    
    # If we have a _target_ field, use Hydra instantiation
    if isinstance(sol_cfg, (dict, DictConfig)) and "_target_" in sol_cfg:
        return hydra.utils.instantiate(sol_cfg, _recursive_=False, logger=logger)
    
    # No name and no _target_ - this is an error
    raise ValueError(f"Solver config must have either 'name' field or '_target_' field. Got keys: {list(sol_cfg.keys()) if isinstance(sol_cfg, (dict, DictConfig)) else 'N/A'}")


def collect_algorithm_kwargs(cfg: DictConfig, solver_name: str) -> Dict[str, Any]:
    """Collect algorithm kwargs from config.
    
    Supports both:
    - New structure: solver config loaded from config/solver/*.yaml into cfg.solver
    - Legacy structure: solver config in cfg.solvers.{solver_name}
    """
    name = (solver_name or "").lower()
    
    # New structure: solver config is directly in cfg.solver (loaded by Hydra config group)
    # Legacy structure: solver config is in cfg.solvers.{name}
    solver_cfg = cfg.get("solver") or {}
    if isinstance(solver_cfg, str):
        # solver is just a string name, look in legacy solvers block
        solver_cfg = {}
    
    # Fall back to legacy solvers.{name} if solver_cfg is empty or doesn't have params
    solvers_cfg = cfg.get("solvers") or {}
    legacy_scfg = solvers_cfg.get(name) or {}

    # Merge: new structure takes precedence, fall back to legacy
    def _get(key: str, default: Any = None) -> Any:
        """Get from solver_cfg, then legacy_scfg, then cfg, then default."""
        value = None
        if (isinstance(solver_cfg, (dict, DictConfig))) and key in solver_cfg:
            value = solver_cfg.get(key)
        elif (isinstance(legacy_scfg, (dict, DictConfig))) and key in legacy_scfg:
            value = legacy_scfg.get(key)
        elif key in cfg:
            value = cfg.get(key)
        else:
            return default
        
        # Treat empty strings as missing values (return default)
        if value == '':
            return default
        return value
    
    # Use merged config (prefer new structure, but allow legacy overrides)
    # Merge solver_cfg and legacy_scfg so that command-line overrides in solvers.{name} work
    if (isinstance(solver_cfg, (dict, DictConfig))) and len(solver_cfg) > 1:
        # Merge with legacy_scfg so command-line overrides take precedence
        # OmegaConf.merge(a, b) means b overrides a, so legacy_scfg (override) should be second
        # This allows overrides like solvers.smc_fks.num_particles=64 to work
        if legacy_scfg:
            scfg = OmegaConf.merge(solver_cfg, legacy_scfg)
        else:
            scfg = solver_cfg
    else:
        scfg = legacy_scfg

    num_iterations = int(_get("num_iterations", cfg.get("optim_iters", 100)))
    batch_size = int(_get("batch_size", 16))
    warmup_batches = int(_get("warmup_batches", 0))
    num_regions = int(_get("num_regions", cfg.get("trust_regions", 1)))
    oracle_budget = int(_get("oracle_budget", cfg.get("oracle_calls", 0)))
    warmup_samples = int(_get("warmup_samples", 0))

    print_benchmark_summary = cfg.get("print_benchmark_summary", True)
    print_solver_summary = cfg.get("print_solver_summary", True)
    print_solver_config = cfg.get("print_solver_config", False)
    if scfg:
        print_benchmark_summary = scfg.get("print_benchmark_summary", print_benchmark_summary)
        print_solver_summary = scfg.get("print_solver_summary", print_solver_summary)
        print_solver_config = scfg.get("print_solver_config", print_solver_config)

    kwargs: Dict[str, Any] = {
        "num_iterations": num_iterations,
        "batch_size": batch_size,
        "warmup_batches": warmup_batches,
        "store_iteration_images": False,
        "num_regions": num_regions,
        "oracle_budget": oracle_budget,
        "warmup_samples": warmup_samples,
        "print_benchmark_summary": bool(print_benchmark_summary),
        "print_solver_summary": bool(print_solver_summary),
        "print_solver_config": bool(print_solver_config),
    }

    if name in ("trust_region", "trustregion", "turbo", "trs"):
        # Get sub-configs
        tr = scfg.get("tr") or {}

        # Flatten TR settings into kwargs
        if tr:
            tr_dict = OmegaConf.to_container(tr, resolve=True) if isinstance(tr, DictConfig) else tr
            if isinstance(tr_dict, (dict, DictConfig)):
                kwargs.update(tr_dict)

        # Ensure core keys exist at top level. Prefer: kwargs (from tr_dict) -> tr -> scfg (solver top-level) -> cfg -> default.
        # TR params (init_length, etc.) live under solver.tr in YAML; perturb_* live at solver top-level.
        def _tr_val(key: str, default: float) -> float:
            v = kwargs.get(key)
            if v is not None:
                return float(v)
            if tr and tr.get(key) is not None:
                return float(tr.get(key))
            if scfg and scfg.get(key) is not None:
                return float(scfg.get(key))
            return float(cfg.get(f"tr_{key}", default))

        kwargs["init_length"] = _tr_val("init_length", 0.8)
        kwargs["min_length"] = _tr_val("min_length", 0.05)
        kwargs["max_length"] = _tr_val("max_length", 1.6)
        kwargs["update_factor"] = _tr_val("update_factor", 2.0)
        _pmin = kwargs.get("perturb_min_frac")
        if _pmin is None and scfg and scfg.get("perturb_min_frac") is not None:
            _pmin = scfg.get("perturb_min_frac")
        if _pmin is None:
            _pmin = cfg.get("tr_perturb_min_frac", 0.1)
        kwargs["perturb_min_frac"] = float(_pmin or 0.1)

        _pmax = kwargs.get("perturb_max_frac")
        if _pmax is None and scfg and scfg.get("perturb_max_frac") is not None:
            _pmax = scfg.get("perturb_max_frac")
        if _pmax is None:
            _pmax = cfg.get("tr_perturb_max_frac", 0.9)
        kwargs["perturb_max_frac"] = float(_pmax or 0.9)

        # Sampling modes
        kwargs["init_sampling_mode"] = str(scfg.get("init_sampling_mode", "sobol"))
        kwargs["search_sampling_mode"] = str(scfg.get("search_sampling_mode", "sobol"))
        
        # Scoring (default random; fast-forward bypasses scoring)
        kwargs["scoring_method"] = str(scfg.get("scoring_method", "random"))
        heuristic_kwargs = scfg.get("heuristic_kwargs", {})
        kwargs["heuristic_kwargs"] = OmegaConf.to_container(heuristic_kwargs, resolve=True) if isinstance(heuristic_kwargs, DictConfig) else heuristic_kwargs
        scoring_method_val = scfg.get("scoring_method", "random") if scfg else "random"
        heuristic_kwargs_val = scfg.get("heuristic_kwargs", {}) if scfg else {}
        # Sobol fast-forward (TRS uses max_sobol_index as pool size when enabled)
        use_fast_forward = False
        max_sobol_index = 1024
        if scfg:
            use_fast_forward = bool(scfg.get("use_fast_forward", False))
            max_sobol_index = int(scfg.get("max_sobol_index", 1024))
        if solver_cfg:
            use_fast_forward = bool(solver_cfg.get("use_fast_forward", use_fast_forward))
            max_sobol_index = int(solver_cfg.get("max_sobol_index", max_sobol_index))
        # Center selection and annealing parameters (top-level solver config)
        # Check top-level center_selection first, then fall back to tr.center_selection for backward compatibility
        center_selection_val = None
        if scfg and "center_selection" in scfg:
            center_selection_val = scfg.get("center_selection")
        elif tr and "center_selection" in tr:
            center_selection_val = tr.get("center_selection")
        if center_selection_val is not None:
            kwargs["center_selection"] = str(center_selection_val)
        
        # Region annealing parameters
        if scfg:
            # Always add use_region_annealing (solver has default False)
            kwargs["use_region_annealing"] = bool(scfg.get("use_region_annealing", False))
            if "anneal_start_frac" in scfg:
                kwargs["anneal_start_frac"] = float(scfg.get("anneal_start_frac", 0.3))
            if "anneal_interval" in scfg and scfg.get("anneal_interval") is not None:
                kwargs["anneal_interval"] = int(scfg.get("anneal_interval", 10))
        
        # Candidate pool (optional; TRS uses max_sobol_index when use_fast_forward)
        if scfg:
            if "cand_pool_per_region" in scfg and scfg.get("cand_pool_per_region") is not None:
                kwargs["cand_pool_per_region"] = int(scfg.get("cand_pool_per_region", 128))
            if "use_adaptive_allocation" in scfg:
                kwargs["use_adaptive_allocation"] = bool(scfg.get("use_adaptive_allocation", False))
            if "allocation_temperature" in scfg:
                kwargs["allocation_temperature"] = float(scfg.get("allocation_temperature", 1.0))
            if "use_two_phase" in scfg:
                kwargs["use_two_phase"] = bool(scfg.get("use_two_phase", False))
            if "phase_switch_frac" in scfg:
                kwargs["phase_switch_frac"] = float(scfg.get("phase_switch_frac", 0.3))
            if "use_diversity_selection" in scfg:
                kwargs["use_diversity_selection"] = bool(scfg.get("use_diversity_selection", False))
            if "diversity_weight" in scfg:
                kwargs["diversity_weight"] = float(scfg.get("diversity_weight", 0.3))
            if "shape" in scfg:
                kwargs["shape"] = str(scfg.get("shape", "hypercube"))
            if "archive_max_size" in scfg and scfg.get("archive_max_size") is not None:
                kwargs["archive_max_size"] = int(scfg.get("archive_max_size"))
        
        _heuristic = heuristic_kwargs_val
        if _heuristic is not None and isinstance(_heuristic, DictConfig):
            _heuristic = OmegaConf.to_container(_heuristic, resolve=True) or {}
        elif not isinstance(_heuristic, dict):
            _heuristic = {}
        kwargs.update(
            {
                "scoring_method": str(scoring_method_val),
                "heuristic_kwargs": _heuristic,
                "use_fast_forward": use_fast_forward,
                "max_sobol_index": max_sobol_index,
            }
        )

    if name in ("bo", "bayes", "bayesopt", "bayesian_optimization", "bayes_opt"):
        tr = scfg.get("tr") or {}
        acq_kwargs_val = scfg.get("acq_kwargs", {})
        acq_kwargs_dict = OmegaConf.to_container(acq_kwargs_val, resolve=True) or {} if acq_kwargs_val else {}

        kwargs.update(
            {
                "surrogate": scfg.get("surrogate", cfg.get("surrogate", None)),
                "acquisition": scfg.get("acquisition", scfg.get("acquisition", "surrogate_topk")),
                "acq_kwargs": acq_kwargs_dict,
                "cand_pool_per_region": int(scfg.get("cand_pool_per_region", cfg.get("cand_pool_per_region", 128))),
                "acq_strategy": str(scfg.get("acq_strategy", "default")).lower(),
                "elite_percentile": float(scfg.get("elite_percentile", 0.2)),
                "use_uncertainty_tr": bool(scfg.get("use_uncertainty_tr", False)),
                "use_diverse_training": bool(scfg.get("use_diverse_training", False)),
                "disable_gp_guidance": bool(scfg.get("disable_gp_guidance", False)),
                "fallback_scoring": str(scfg.get("fallback_scoring", "random")).lower(),
                "crossover_fraction": float(scfg.get("crossover_fraction", 0.0)),
                "gradient_fraction": float(scfg.get("gradient_fraction", 0.0)),
                "use_msr_gp_init": bool(scfg.get("use_msr_gp_init", True)),
                "gp_lengthscale_base_scale": float(scfg.get("gp_lengthscale_base_scale", 1.0)),
                "enable_gp_diagnostics": bool(scfg.get("enable_gp_diagnostics", False)),
                "init_length": float(tr.get("init_length", cfg.get("tr_init_length", 0.8))),
                "min_length": float(tr.get("min_length", cfg.get("tr_min_length", 0.05))),
                "max_length": float(tr.get("max_length", cfg.get("tr_max_length", 1.6))),
                "success_tolerance": int(tr.get("success_tolerance", cfg.get("tr_success_tolerance", 3))),
                "failure_tolerance": int(tr.get("failure_tolerance", cfg.get("tr_failure_tolerance", 3))),
                "prob_perturb": tr.get("prob_perturb", cfg.get("tr_prob_perturb", None)),
                "perturb_min_frac": float(tr.get("perturb_min_frac", cfg.get("tr_perturb_min_frac", 0.1))),
                "perturb_max_frac": float(tr.get("perturb_max_frac", cfg.get("tr_perturb_max_frac", 0.9))),
                "init_sampling_mode": str(scfg.get("init_sampling_mode", "sobol")),
            }
        )

    if name in ("furbo", "furbo_solver"):
        tr = scfg.get("tr") or {}
        acq_kwargs_val = scfg.get("acq_kwargs", {})
        acq_kwargs_dict = OmegaConf.to_container(acq_kwargs_val, resolve=True) or {} if acq_kwargs_val else {}

        kwargs.update(
            {
                "surrogate": scfg.get("surrogate", cfg.get("surrogate", "gp")),
                "acquisition": scfg.get("acquisition", scfg.get("acquisition", "qei")),
                "acq_kwargs": acq_kwargs_dict,
                "init_length": float(tr.get("init_length", cfg.get("tr_init_length", 0.8))),
                "min_length": float(tr.get("min_length", cfg.get("tr_min_length", 0.05))),
                "max_length": float(tr.get("max_length", cfg.get("tr_max_length", 1.6))),
                "success_tolerance": int(tr.get("success_tolerance", cfg.get("tr_success_tolerance", 3))),
                "failure_tolerance": int(tr.get("failure_tolerance", cfg.get("tr_failure_tolerance", 3))),
                "cand_pool_per_region": int(scfg.get("cand_pool_per_region", cfg.get("cand_pool_per_region", 128))),
                "gp_lengthscale_base_scale": float(
                    scfg.get("gp_lengthscale_base_scale", cfg.get("gp_lengthscale_base_scale", 1.0))
                ),
                "gp_training_size": int(scfg.get("gp_training_size", cfg.get("gp_training_size", 200))),
                "enable_enhanced_gp_diagnostics": bool(
                    scfg.get("enable_enhanced_gp_diagnostics", cfg.get("enable_enhanced_gp_diagnostics", False))
                ),
                "use_pca_scoring": bool(scfg.get("use_pca_scoring", cfg.get("use_pca_scoring", False))),
                "pca_scoring_mode": str(scfg.get("pca_scoring_mode", cfg.get("pca_scoring_mode", "fallback"))),
                "pca_k": int(scfg.get("pca_k", cfg.get("pca_k", 3))),
                "pca_alpha": float(scfg.get("pca_alpha", cfg.get("pca_alpha", 0.8))),
                "pca_diversity_weight": float(scfg.get("pca_diversity_weight", cfg.get("pca_diversity_weight", 2.0))),
                "pca_min_points": int(scfg.get("pca_min_points", cfg.get("pca_min_points", 10))),
                "pca_use_signed": bool(scfg.get("pca_use_signed", cfg.get("pca_use_signed", False))),
                "disable_gp": bool(scfg.get("disable_gp", cfg.get("disable_gp", False)) if scfg else cfg.get("disable_gp", False)),
                "constraint_functions": scfg.get("constraint_functions", None) if scfg else None,
            }
        )
        # Sobol fast-forward (check both scfg and solver_cfg)
        use_fast_forward = False
        max_sobol_index = 1024
        if scfg:
            use_fast_forward = bool(scfg.get("use_fast_forward", False))
            max_sobol_index = int(scfg.get("max_sobol_index", 1024))
        if solver_cfg:
            use_fast_forward = bool(solver_cfg.get("use_fast_forward", use_fast_forward))
            max_sobol_index = int(solver_cfg.get("max_sobol_index", max_sobol_index))
        kwargs.update({
            "use_fast_forward": use_fast_forward,
            "max_sobol_index": max_sobol_index,
        })

    if name in ("zero_order", "zero", "zoo"):
        zo = scfg or {}
        kwargs.update({"step_size": float(zo.get("step_size", cfg.get("zo_step_size", 0.1)))})

    if name in ("random_search", "randomsearch", "rs"):
        rs = scfg or {}
        # Sampling mode: auto (default), problem, gaussian, sobol_erfinv
        sampling_mode = rs.get("sampling_mode", "auto")
        if sampling_mode:
            kwargs["sampling_mode"] = str(sampling_mode)
        # Legacy alias: use_sobol_sampling -> sampling_mode="sobol_erfinv"
        if rs.get("use_sobol_sampling", False):
            kwargs["sampling_mode"] = "sobol_erfinv"
        # Sobol scrambling (default: True for better coverage)
        if "sobol_scramble" in rs:
            kwargs["sobol_scramble"] = bool(rs.get("sobol_scramble", True))
        # Pass seed from config to ensure reproducibility
        seed_val = cfg.get("seed")
        if seed_val is not None and "seed" not in kwargs:
            kwargs["seed"] = int(seed_val)
        # Also check solver config for seed (allows override)
        if "seed" in rs:
            kwargs["seed"] = int(rs.get("seed"))

    if name in ("cem", "cross_entropy", "crossentropy"):
        c = scfg or {}
        kwargs.update({"elite_frac": float(c.get("elite_frac", 0.2)), "init_std": float(c.get("init_std", 0.5))})

    if name in ("snes",):
        s = scfg or {}
        kwargs.update(
            {
                "popsize": int(s.get("popsize", 64)),
                "lr_mean": float(s.get("lr_mean", 0.6)),
                "lr_sigma": float(s.get("lr_sigma", 0.2)),
                "init_sigma": float(s.get("init_sigma", 0.5)),
            }
        )


    # Pass through any remaining keys from scfg that aren't already in kwargs
    # This ensures experimental parameters (like use_fast_forward, max_sobol_index, etc.)
    # added via Hydra overrides with + prefix are included
    if scfg:
        scfg_dict = OmegaConf.to_container(scfg, resolve=True)
        if isinstance(scfg_dict, (dict, DictConfig)):
            # Keys to skip (already handled explicitly above)
            skip_keys = {
                "name", "num_iterations", "batch_size", "warmup_batches", "num_regions",
                "oracle_budget", "warmup_samples", "print_benchmark_summary", "print_solver_summary",
                "tr", "candidate_mode", "cand_pool_per_region", "init_sampling_mode",
                "search_sampling_mode", "scoring_method", "heuristic_kwargs", "surrogate",
                "acquisition", "acq_kwargs", "acq_strategy", "elite_percentile", "constraint_functions",
                "use_pca_scoring", "pca_scoring_mode", "pca_k", "pca_alpha", "pca_diversity_weight",
                "pca_min_points", "pca_use_signed", "disable_gp", "gp_training_size",
                "gp_lengthscale_base_scale", "enable_enhanced_gp_diagnostics"
            }
            for key, value in scfg_dict.items():
                if key not in skip_keys and key not in kwargs:
                    # Convert value to appropriate type
                    if isinstance(value, (int, float, str, bool, type(None))):
                        kwargs[key] = value
                    elif isinstance(value, (dict, DictConfig)):
                        kwargs[key] = value
                    elif isinstance(value, list):
                        kwargs[key] = value
    
    # Also check solver_cfg directly for experimental parameters (in case they're not in merged scfg)
    if solver_cfg:
        solver_cfg_dict = OmegaConf.to_container(solver_cfg, resolve=True)
        if isinstance(solver_cfg_dict, (dict, DictConfig)):
            for key in ["use_fast_forward", "max_sobol_index", "archive_max_size"]:
                if key in solver_cfg_dict and key not in kwargs:
                    value = solver_cfg_dict[key]
                    if isinstance(value, (int, float, str, bool, type(None))):
                        kwargs[key] = value
    
    # Add proteina config for protein modality (needed for DTS and other solvers)
    modality = cfg.get("modality", "").lower()
    if modality in ("protein", "proteina"):
        proteina_cfg = cfg.get("proteina")
        if proteina_cfg is not None:
            # Convert to dict if it's a DictConfig
            if isinstance(proteina_cfg, DictConfig):
                proteina_dict = OmegaConf.to_container(proteina_cfg, resolve=True)
            else:
                proteina_dict = proteina_cfg
            
            if isinstance(proteina_dict, dict):
                # Add proteina config to kwargs (for solvers like DTS that need sampling_mode, sc_scale_noise, etc.)
                kwargs["proteina"] = proteina_dict
                # Also add flattened keys for convenience (e.g., sampling_mode, sc_scale_noise)
                if "sampling_mode" in proteina_dict and "sampling_mode" not in kwargs:
                    kwargs["sampling_mode"] = proteina_dict["sampling_mode"]
                if "sc_scale_noise" in proteina_dict and "sc_scale_noise" not in kwargs:
                    kwargs["sc_scale_noise"] = proteina_dict["sc_scale_noise"]
                if "sc_scale_score" in proteina_dict and "sc_scale_score" not in kwargs:
                    kwargs["sc_scale_score"] = proteina_dict["sc_scale_score"]
    
    return kwargs


# --- logging ------------------------------------------------------------------

def get_wandb_run_name(cfg: DictConfig) -> str:
    """Generate wandb run name in format: modality_solver_reward_function.
    
    Priority:
    1. Explicit wandb_name config
    2. SLURM_JOB_NAME environment variable
    3. Constructed as: modality_solver_reward_function
    """
    base_name = str(cfg.get("wandb_name", None))
    slurm_job_name = os.environ.get("SLURM_JOB_NAME")
    
    # Priority 1: explicit wandb_name config
    if base_name and base_name != "None" and base_name.strip():
        return base_name
    
    # Priority 2: SLURM job name
    if slurm_job_name:
        return slurm_job_name
    
    # Priority 3: construct from parts: modality_solver_reward_function
    modality = (cfg.get("modality") or "image").lower()
    
    # Normalize modality names
    if modality in ("image", "image_noise", "t2i"):
        modality = "t2i"
    elif modality in ("protein", "proteina"):
        modality = "protein"
    elif modality in ("molecule", "qm9"):
        modality = "molecule"
    # Get solver name
    solver_name = get_solver_name(cfg)
    if not solver_name:
        solver_name = "unknown"
    
    # Normalize solver name (use common names)
    solver_name_lower = solver_name.lower()
    if solver_name_lower in ("trust_region", "trustregion", "turbo", "trs"):
        solver_name = "trs"
    elif solver_name_lower in ("random_search", "randomsearch", "rs"):
        solver_name = "random"
    else:
        # Keep original name but normalize to lowercase with underscores
        solver_name = solver_name.lower().replace("-", "_")
    
    # Get reward name
    reward_name = _get_reward_name(cfg)
    if not reward_name:
        reward_name = "unknown"
    
    # Construct name: modality_solver_reward
    name = f"{modality}_{solver_name}_{reward_name}"
    return name


def make_logger(cfg: DictConfig) -> ExperimentLogger:
    """Instantiate and configure the experiment logger (W&B / terminal)."""
    lg = cfg.get("logger") or {}
    logging_cfg_raw = cfg.get("logging") or {}

    def _to_plain(data: Any) -> Any:
        if isinstance(data, DictConfig):
            return OmegaConf.to_container(data, resolve=True)
        return data
    
    # Convert logging config to plain dict so logger can properly merge it
    logging_cfg = _to_plain(logging_cfg_raw) or {}

    if "_target_" in lg:
        return hydra.utils.instantiate(lg, _recursive_=False)

    enable = bool(cfg.get("wandb", True))
    base_project = str(cfg.get("wandb_project", "image-bo"))
    base_name = str(cfg.get("wandb_name", None))
    wandb_dir = cfg.get("wandb_dir") or os.environ.get("WANDB_DIR") or None
    if isinstance(wandb_dir, str) and wandb_dir.strip():
        wandb_dir = os.path.abspath(wandb_dir)
    else:
        wandb_dir = None
    modality = (cfg.get("modality") or "image").lower()

    benchmark_cfg = cfg.get("benchmark") or {}
    # Prioritize wandb_project if explicitly set
    project = cfg.get("wandb_project")
    if project:
        pass # Already set
    elif isinstance(benchmark_cfg, (dict, DictConfig)):
        benchmark_name = benchmark_cfg.get("name") or benchmark_cfg.get("_target_", "").split(".")[-1]
        if modality in ("molecule", "qm9"):
            project = "molecule target mean"
        elif benchmark_name and "DrawBenchBenchmark" in str(benchmark_cfg.get("_target_", "")):
            version = benchmark_cfg.get("version")
            if version and str(version).strip().lower() not in {"full", "all", "default", "", "none", "null"}:
                project = f"DrawBench{version}"
            else:
                project = "DrawBench"
        elif benchmark_name:
            project = benchmark_name
    
    if not project:
        project = base_project

    # Get wandb run name using helper function
    name = get_wandb_run_name(cfg)
    if os.environ.get("SLURM_JOB_NAME"):
        print(f"[INFO] Using SLURM job name as wandb run name: {name}")
    elif base_name and base_name != "None" and base_name.strip():
        print(f"[INFO] Using explicit wandb_name: {name}")
    else:
        print(f"[INFO] Using constructed wandb run name: {name}")

    if modality == "image":
        scaling_cfg: Dict[str, Any] = {}
        if isinstance(logging_cfg, (dict, DictConfig)):
            wandb_cfg = logging_cfg.get("wandb", {})
            scaling_cfg = wandb_cfg.get("scaling", {}) if isinstance(wandb_cfg, (dict, DictConfig)) else {}
        if not scaling_cfg:
            scaling_cfg = logging_cfg.get("scaling", {}) if isinstance(logging_cfg, (dict, DictConfig)) else {}
        if not scaling_cfg:
            scaling_cfg = cfg.get("scaling", {})
        scaling_cfg = _to_plain(scaling_cfg) or {}
        scaling_enabled = bool(scaling_cfg.get("enabled", False))
        if scaling_enabled:
            budget_checkpoints = scaling_cfg.get("budget_checkpoints", [120, 240, 480, 960])
            return ScalingLogger(
                project=project,
                name=name,
                config=OmegaConf.to_container(cfg, resolve=True),
                enable=enable,
                budget_checkpoints=budget_checkpoints,
                logging_config=logging_cfg,
                wandb_dir=wandb_dir,
            )
        return T2IWandbLogger(
            project=project,
            name=name,
            config=OmegaConf.to_container(cfg, resolve=True),
            enable=enable,
            logging_config=logging_cfg,
            wandb_dir=wandb_dir,
        )

    if modality in ("protein", "proteina"):
        from ..loggers.protein_logger import ProteinLogger

        return ProteinLogger(
            project=project,
            name=name,
            config=OmegaConf.to_container(cfg, resolve=True),
            enable=enable,
            logging_config=logging_cfg,
            wandb_dir=wandb_dir,
        )

    if modality in ("molecule", "qm9"):
        from ..loggers.molecule_logger import MoleculeLogger

        return MoleculeLogger(
            project=project,
            name=name,
            config=OmegaConf.to_container(cfg, resolve=True),
            enable=enable,
            logging_config=logging_cfg,
            wandb_dir=wandb_dir,
        )

    return WandbLogger(
        project=project,
        name=name,
        config=OmegaConf.to_container(cfg, resolve=True),
        enable=enable,
        wandb_dir=wandb_dir,
    )

