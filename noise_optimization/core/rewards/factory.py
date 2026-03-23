from __future__ import annotations

from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from .base import RewardConfig, RewardFunction
from . import get_reward_function


def _objective_to_reward_name(objective: str, modality: str) -> str:
    """Map a high-level objective string to a concrete reward name."""
    o = (objective or "").strip().lower()
    if modality in ("molecule", "qm9"):
        if o in ("multi_property_target", "multi_property", "r3", "r6"):
            return "multi_property_target"
    if modality in ("protein", "proteina"):
        if o in ("designability", "scrmsd"):
            return "designability"
    return o


def instantiate_reward(cfg: DictConfig) -> Any:
    """Instantiate the reward function based on the experiment config."""
    modality = (cfg.get("modality") or "image").lower()
    
    # Normalize modality aliases
    if modality in ("image_noise", "t2i"):
        modality = "image"
    elif modality == "qm9":
        modality = "molecule"

    # 1) Preferred: top-level objective for convenience
    objective = cfg.get("objective")
    if objective:
        name = _objective_to_reward_name(str(objective), modality)
        rcfg = dict(cfg.get("reward") or {})
        params = {k: v for k, v in rcfg.items() if k != "name"}
        ow = cfg.get("objective_weights")
        if ow is not None:
            md = dict(params.get("metadata") or {})
            md["weights"] = list(ow)
            params["metadata"] = md
        return get_reward_function(name, **params)

    # 2) Configured reward via core registry
    reward_cfg = cfg.get("reward")
    if reward_cfg and reward_cfg.get("name"):
        params = {k: v for k, v in reward_cfg.items() if k != "name"}
        return get_reward_function(reward_cfg["name"], **params)

    # 3) Hydra-targeted reward_function / legacy reward_model
    rf_cfg = cfg.get("reward_function") or cfg.get("reward_model")
    if rf_cfg is None:
        raise ValueError("reward, reward_function, or reward_model config is required")
    
    # Check for nested reward config
    if isinstance(rf_cfg, (dict, DictConfig)) and "reward" in rf_cfg:
        nested_reward = rf_cfg.get("reward")
        if isinstance(nested_reward, (dict, DictConfig)) and nested_reward.get("name"):
            reward_name_from_nested = str(nested_reward.get("name")).lower()
            params = {k: v for k, v in nested_reward.items() if k != "name"}
            return get_reward_function(reward_name_from_nested, **params)
    
    # Validate reward is appropriate for modality
    from ..pipelines.common import VALID_REWARDS_BY_MODALITY, DEFAULT_REWARDS_BY_MODALITY
    
    valid_rewards = VALID_REWARDS_BY_MODALITY.get(modality, [])
    default_reward = DEFAULT_REWARDS_BY_MODALITY.get(modality, "image_reward")
    
    # Extract reward name
    reward_name = None
    
    # Try Hydra choices
    try:
        from hydra.core.global_hydra import GlobalHydra
        if GlobalHydra().is_initialized():
            hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
            if hydra_cfg and hasattr(hydra_cfg, "runtime"):
                runtime = hydra_cfg.runtime
                if hasattr(runtime, "choices") and "reward_function" in runtime.choices:
                    reward_name = str(runtime.choices["reward_function"]).lower()
    except Exception:
        pass
    
    if reward_name is None:
        if isinstance(rf_cfg, str):
            reward_name = rf_cfg.lower()
    if reward_name is None and isinstance(rf_cfg, (dict, DictConfig)):
        target = rf_cfg.get("_target_", "").lower()
        if target:
            parts = target.split(".")
            class_name = parts[-1].lower() if parts else ""
            if class_name == "designabilityreward" or class_name == "scrmsdreward":
                reward_name = "designability"
            elif class_name == "multipropertytargetreward":
                reward_name = "multi_property_target"
            elif class_name in ("imagerewardhf", "imagereward"):
                reward_name = "image_reward"
            elif class_name in ("hpsv2rewardhf", "hpsreward"):
                reward_name = "hps"
            elif class_name == "combinedrewardhf":
                reward_name = "combined"
            elif class_name in ("cliprewardhf", "clipreward"):
                reward_name = "clip"
            elif class_name in ("aestheticrewardhf", "aestheticreward"):
                reward_name = "aesthetic"
        if reward_name is None:
            name = rf_cfg.get("name")
            if name:
                reward_name = str(name).lower()
    
    if reward_name is not None and reward_name in valid_rewards:
        params = {}
        if isinstance(rf_cfg, (dict, DictConfig)):
            if "reward" in rf_cfg:
                nested_reward = rf_cfg.get("reward")
                if isinstance(nested_reward, (dict, DictConfig)):
                    params = {k: v for k, v in nested_reward.items() if k != "name"}
            else:
                allowed_params = {"device", "dtype", "cache_dir", "metadata", "differentiable", "model_name", "weights"}
                params = {k: v for k, v in rf_cfg.items() if k in allowed_params}
                if "weights" in params and params["weights"] is not None:
                    w = params["weights"]
                    params["weights"] = list(w) if hasattr(w, "__iter__") and not isinstance(w, str) else w
        return get_reward_function(reward_name, **params)
    
    if isinstance(rf_cfg, (dict, DictConfig)) and rf_cfg.get("_target_") and reward_name is None:
        pass  # proceed to instantiation below
    elif reward_name is None or reward_name not in valid_rewards:
        if reward_name is not None:
            print(f"[WARNING] Reward '{reward_name}' is not valid for modality '{modality}'. Using default '{default_reward}'.")
        OmegaConf.set_struct(cfg, False)
        cfg["reward_function"] = default_reward
        OmegaConf.set_struct(cfg, True)
        rf_cfg = cfg.get("reward_function") or cfg.get("reward_model")
        if isinstance(rf_cfg, str):
            return get_reward_function(rf_cfg)

    if isinstance(rf_cfg, (dict, DictConfig)) and "reward" in rf_cfg:
        nested_reward = rf_cfg.get("reward")
        if isinstance(nested_reward, (dict, DictConfig)) and nested_reward.get("name"):
            params = {k: v for k, v in nested_reward.items() if k != "name"}
            return get_reward_function(nested_reward["name"], **params)

    if isinstance(rf_cfg, str):
        return get_reward_function(rf_cfg)
    
    target = rf_cfg.get("_target_", "")
    if target:
        try:
            cls = hydra.utils.get_class(target)
            if issubclass(cls, RewardFunction):
                import inspect
                sig = inspect.signature(cls.__init__)
                params = list(sig.parameters.keys())
                if len(params) > 1 and params[1] == "config":
                    config = RewardConfig()
                    requested_device = rf_cfg.get("device", "cuda")
                    config.device = "cuda" if requested_device == "cuda" and torch.cuda.is_available() else "cpu"
                    config.model_name = rf_cfg.get("model_name")
                    config.cache_dir = rf_cfg.get("cache_dir")
                    config.dtype = rf_cfg.get("dtype", torch.float32)
                    config.metadata = dict(rf_cfg.get("metadata", {}))
                    if "weights" in rf_cfg and rf_cfg.get("weights") is not None:
                        w = rf_cfg.get("weights")
                        config.weights = list(w) if hasattr(w, "__iter__") and not isinstance(w, str) else w
                    return cls(config)
        except Exception:
            pass

    reward = hydra.utils.instantiate(rf_cfg, _recursive_=False)
    if callable(reward):
        reward = reward()
    if hasattr(reward, "to"):
        device = "cuda" if torch.cuda.is_available() and cfg.get("device", "cuda") == "cuda" else "cpu"
        reward = reward.to(device)
    return reward
