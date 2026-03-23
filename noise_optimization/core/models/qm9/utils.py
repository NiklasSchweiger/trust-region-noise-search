"""QM9 flow model construction utilities.

Bundles the pretrained EquiFM weights and loads the model using the architecture
from Guided Flow Matching with Optimal Control (OC-Flow).
Attribution:
  GitHub  https://github.com/WangLuran/Guided-Flow-Matching-with-Optimal-Control
  arXiv   https://arxiv.org/abs/2410.18070
"""
import os
import torch

from .cnf_models import Cnflows, EGNN_dynamics_QM9, DistributionNodes
from .consts import qm9_with_h

class UniformDequantizer(torch.nn.Module):
    def __init__(self):
        super(UniformDequantizer, self).__init__()

    def forward(self, tensor, node_mask, edge_mask, context):
        category, integer = tensor['categorical'], tensor['integer']
        zeros = torch.zeros(integer.size(0), device=integer.device)
        out_category = category + torch.rand_like(category) - 0.5
        out_integer = integer + torch.rand_like(integer) - 0.5
        if node_mask is not None:
            out_category = out_category * node_mask
            out_integer = out_integer * node_mask
        out = {'categorical': out_category, 'integer': out_integer}
        return out, zeros

    def reverse(self, tensor):
        categorical, integer = tensor['categorical'], tensor['integer']
        integer = torch.round(integer)
        categorical = torch.round(categorical)
        return {'categorical': categorical, 'integer': integer}


def setup_generation(batch_size, nodes_dist, max_n_nodes, device):
    nodesxsample = nodes_dist.sample(batch_size)
    assert int(torch.max(nodesxsample)) <= max_n_nodes
    node_mask = torch.zeros(batch_size, max_n_nodes)
    for i in range(batch_size):
        node_mask[i, 0: nodesxsample[i]] = 1
    edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask
    edge_mask = edge_mask.view(batch_size * max_n_nodes * max_n_nodes, 1).to(device)
    node_mask = node_mask.unsqueeze(2).to(device)
    return node_mask, edge_mask, nodesxsample


_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_WEIGHTS = os.path.join(_PACKAGE_DIR, "generative_model_ema_0.npy")
_DEFAULT_ARGS = os.path.join(_PACKAGE_DIR, "args.pickle")


def get_flow_model(device, weights_path: str = None):
    """Build QM9 EquiFM flow model from bundled pretrained weights.

    Loads architecture config from the bundled args.pickle and weights from
    generative_model_ema_0.npy (both shipped inside this package).

    Args:
        device: torch device string (e.g. "cuda", "cpu")
        weights_path: override path to checkpoint; defaults to bundled weights.

    Returns:
        (cnf, nodes_dist, dequantizer, args)
    """
    import pickle

    args_path = _DEFAULT_ARGS
    if not os.path.isfile(args_path):
        raise FileNotFoundError(
            f"EquiFM args.pickle not found at {args_path}. "
            "Ensure the bundled model files are present in the qm9/ package directory."
        )
    with open(args_path, "rb") as f:
        args = pickle.load(f)

    # Ensure attributes added after the original pickle was created are present
    if not hasattr(args, "normalization_factor"):
        args.normalization_factor = 1
    if not hasattr(args, "aggregation_method"):
        args.aggregation_method = "sum"

    histogram = qm9_with_h["n_nodes"]
    in_node_nf = len(qm9_with_h["atom_decoder"]) + int(args.include_charges)
    nodes_dist = DistributionNodes(histogram)

    dynamics_in_node_nf = in_node_nf + 1 if args.condition_time else in_node_nf
    net_dynamics = EGNN_dynamics_QM9(
        in_node_nf=dynamics_in_node_nf,
        context_node_nf=args.context_node_nf,
        n_dims=3,
        device=device,
        hidden_nf=args.nf,
        act_fn=torch.nn.SiLU(),
        n_layers=args.n_layers,
        attention=args.attention,
        tanh=args.tanh,
        mode=args.model,
        norm_constant=args.norm_constant,
        inv_sublayers=args.inv_sublayers,
        sin_embedding=args.sin_embedding,
        normalization_factor=args.normalization_factor,
        aggregation_method=args.aggregation_method,
    )
    dequantizer = UniformDequantizer()
    cnf = Cnflows(
        dynamics=net_dynamics,
        in_node_nf=in_node_nf,
        n_dims=3,
        timesteps=args.diffusion_steps,
        noise_schedule=args.diffusion_noise_schedule,
        noise_precision=args.diffusion_noise_precision,
        loss_type=args.diffusion_loss_type,
        norm_values=args.normalize_factors,
        include_charges=args.include_charges,
        discrete_path=args.discrete_path,
        cat_loss=args.cat_loss,
        cat_loss_step=args.cat_loss_step,
        on_hold_batch=args.on_hold_batch,
        sampling_method=args.sampling_method,
        weighted_methods=args.weighted_methods,
        ode_method=args.ode_method,
        without_cat_loss=args.without_cat_loss,
        angle_penalty=args.angle_penalty,
    )
    cnf = cnf.to(device)
    dequantizer = dequantizer.to(device)

    ckpt = weights_path if weights_path is not None else _DEFAULT_WEIGHTS
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(
            f"EquiFM checkpoint not found at {ckpt}. "
            "Ensure the bundled model files are present in the qm9/ package directory."
        )
    cnf.load_state_dict(torch.load(ckpt, map_location=device))

    return cnf, nodes_dist, dequantizer, args

