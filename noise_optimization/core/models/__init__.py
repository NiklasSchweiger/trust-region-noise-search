"""Generative models for each modality.

Entry points: t2i (text-to-image), molecule (QM9), proteina (protein).
Instantiation: use factories.instantiate_generative_model(cfg, device) which
dispatches by modality and config (e.g. model=sd, model=sdxl_lightning).
"""
from __future__ import annotations
