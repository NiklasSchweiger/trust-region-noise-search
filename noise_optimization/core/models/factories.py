from __future__ import annotations

from typing import Optional

import hydra
import torch
from omegaconf import DictConfig, OmegaConf


def instantiate_generative_model(cfg: DictConfig, device: Optional[str] = None) -> torch.nn.Module:
    """Instantiate a generative model based on modality and model.name.

    This is the extracted version of `_instantiate_generative_model` from `main.py`.
    It is kept API-compatible so existing entrypoints continue to work.
    """
    # Use provided device or determine from config
    if device is None:
        cuda_available = torch.cuda.is_available()
        requested_device = cfg.get("device", "cuda")
        device = "cuda" if cuda_available and requested_device == "cuda" else "cpu"

    model_cfg = cfg.get("model")
    modality = (cfg.get("modality") or "image").lower()
    if model_cfg and modality == "image" and ("_target_" in model_cfg):
        model = hydra.utils.instantiate(model_cfg, _recursive_=False)
        if hasattr(model, "to"):
            model = model.to(device)
        return model

    if modality == "image" and cfg.get("model") is not None:
        from .t2i import make_t2i_from_cfg
        return make_t2i_from_cfg(cfg.get("model"), device=device)

    # Molecule generator: qm9_flow
    if modality in ("molecule", "qm9", "qm9_target"):
        from .molecule import make_qm9_flow_from_cfg
        # Convert DictConfig to dict if needed
        cfg_dict = OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else dict(cfg)
        return make_qm9_flow_from_cfg(cfg_dict, device=device)

    # Protein generator: proteina
    if modality in ("protein", "proteina"):
        from .proteina import make_proteina_from_cfg
        # Convert DictConfig to dict if needed
        cfg_dict = OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else dict(cfg)
        return make_proteina_from_cfg(cfg_dict, device=device)

    gen_cfg = cfg.get("generative_model")
    if gen_cfg is None:
        raise ValueError("Provide either model.name or generative_model")
    model = hydra.utils.instantiate(gen_cfg, _recursive_=False)
    if hasattr(model, "to"):
        model = model.to(device)
    return model


