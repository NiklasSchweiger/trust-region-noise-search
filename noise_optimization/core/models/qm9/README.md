# QM9 EquiFM Model

This directory contains the EquiFM generative model and EGNN property classifiers used for
QM9 molecule generation experiments in the Trust-Region Noise Search paper.

## Attribution

The model architecture (`egnn.py`, `cnf_models.py`) and pretrained weights
(`generative_model_ema_0.npy`, `args.pickle`) are taken from:

> **Guided Flow Matching with Optimal Control** (OC-Flow)
> Wang et al., 2024
> GitHub: https://github.com/WangLuran/Guided-Flow-Matching-with-Optimal-Control
> arXiv: https://arxiv.org/abs/2410.18070

The EGNN property classifiers in `property_prediction/` are also from the same repository.

We bundle these files directly so that QM9 experiments are fully reproducible without
requiring an external installation of OC-Flow.

## Contents

| File / Directory | Description |
|---|---|
| `egnn.py` | Equivariant Graph Neural Network architecture |
| `cnf_models.py` | Continuous normalizing flow model (Cnflows, EGNN_dynamics_QM9) |
| `utils.py` | Model loading utilities; `get_flow_model()` loads the bundled weights |
| `consts.py` | QM9 dataset constants and property statistics |
| `generative_model_ema_0.npy` | Pretrained EquiFM weights (EMA checkpoint) |
| `args.pickle` | Training hyperparameters used to build the model architecture |
| `property_prediction/` | Pre-trained EGNN classifiers for 6 QM9 properties (alpha, gap, homo, lumo, mu, Cv) |
