"""QM9 EquiFM generative model and property prediction.

Architecture (EGNN, Cnflows, EGNN_dynamics_QM9) and pretrained weights
(generative_model_ema_0.npy, args.pickle) are taken from Guided Flow Matching
with Optimal Control (OC-Flow / EquiFM).

Attribution:
  GitHub  https://github.com/WangLuran/Guided-Flow-Matching-with-Optimal-Control
  arXiv   https://arxiv.org/abs/2410.18070
"""
from .egnn import EGNN
from .cnf_models import Cnflows, EGNN_dynamics_QM9, DistributionNodes
from .utils import UniformDequantizer, setup_generation, get_flow_model
from .distribution_property import DistributionProperty, compute_mean_mad_from_dataloader

__all__ = [
    "EGNN",
    "Cnflows",
    "EGNN_dynamics_QM9",
    "DistributionNodes",
    "UniformDequantizer",
    "setup_generation",
    "get_flow_model",
    "DistributionProperty",
    "compute_mean_mad_from_dataloader",
]


