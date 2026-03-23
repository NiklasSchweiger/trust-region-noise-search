from importlib import import_module
from typing import Dict, Tuple

from .base import (
    RewardFunction,
    RewardConfig,
    RewardFunctionRegistry,
    ImageRewardFunction,
    TextPromptRewardFunction,
    CompositeReward,
    MoleculeRewardFunction,
    get_reward_function as _base_get_reward_function,
    list_reward_functions,
)

__all__ = [
    "RewardFunction",
    "RewardConfig",
    "RewardFunctionRegistry",
    "ImageRewardFunction",
    "TextPromptRewardFunction",
    "CompositeReward",
    "MoleculeRewardFunction",
    "get_reward_function",
    "list_reward_functions",
]

_LAZY_REGISTRY: Dict[str, Tuple[str, str]] = {
    # Image/text rewards (T2I)
    "clip": ("noise_optimization.core.rewards.image.t2i_rewards", "register_t2i_rewards"),
    "clip_hf": ("noise_optimization.core.rewards.image.t2i_rewards", "register_t2i_rewards"),
    "aesthetic": ("noise_optimization.core.rewards.image.t2i_rewards", "register_t2i_rewards"),
    "aesthetic_hf": ("noise_optimization.core.rewards.image.t2i_rewards", "register_t2i_rewards"),
    "imagereward": ("noise_optimization.core.rewards.image.t2i_rewards", "register_t2i_rewards"),
    "image_reward": ("noise_optimization.core.rewards.image.t2i_rewards", "register_t2i_rewards"),
    "imagereward_hf": ("noise_optimization.core.rewards.image.t2i_rewards", "register_t2i_rewards"),
    "hps": ("noise_optimization.core.rewards.image.t2i_rewards", "register_t2i_rewards"),
    "hps_v2": ("noise_optimization.core.rewards.image.t2i_rewards", "register_t2i_rewards"),
    "combined": ("noise_optimization.core.rewards.image.t2i_rewards", "register_t2i_rewards"),
    # Molecule rewards
    "multi_property_target": ("noise_optimization.core.rewards.molecule", "register_target_property_rewards"),
    # Protein rewards
    "designability": ("noise_optimization.core.rewards.protein", "register_protein_rewards"),
}

_LOADED_MODULES: set[str] = set()


def _ensure_registered(name: str) -> None:
    key = name.lower()
    module_entry = _LAZY_REGISTRY.get(key)
    if module_entry is None:
        return
    module_path, register_fn = module_entry
    if module_path in _LOADED_MODULES:
        return
    module = import_module(module_path)
    getattr(module, register_fn)()
    _LOADED_MODULES.add(module_path)


def get_reward_function(name: str, **kwargs):
    try:
        return _base_get_reward_function(name, **kwargs)
    except ValueError:
        _ensure_registered(name)
        return _base_get_reward_function(name, **kwargs)
