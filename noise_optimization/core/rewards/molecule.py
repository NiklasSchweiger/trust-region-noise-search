"""Molecule reward functions for QM9 property target matching.

Multi-property QM9 reward uses EGNN classifiers from the in-repo property_prediction
module (noise_optimization/core/models/qm9/property_prediction/). That module is
taken from Guided Flow Matching with Optimal Control. Attribution:
GitHub https://github.com/WangLuran/Guided-Flow-Matching-with-Optimal-Control
arXiv https://arxiv.org/abs/2410.18070
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from .base import MoleculeRewardFunction, RewardFunctionRegistry, RewardConfig


_adj_matrix_cache: Dict[int, Dict[int, Any]] = {}


def _get_adj_matrix(n_nodes: int, batch_size: int, device: str) -> Any:
    """Build a full adjacency matrix (all node pairs) for a batched graph.

    Inlined from OC-Flow qm9/utils.py to avoid an external package dependency.
    """
    if n_nodes not in _adj_matrix_cache:
        _adj_matrix_cache[n_nodes] = {}
    if batch_size in _adj_matrix_cache[n_nodes]:
        return _adj_matrix_cache[n_nodes][batch_size]
    rows, cols = [], []
    for batch_idx in range(batch_size):
        for i in range(n_nodes):
            for j in range(n_nodes):
                rows.append(i + batch_idx * n_nodes)
                cols.append(j + batch_idx * n_nodes)
    edges = [torch.LongTensor(rows).to(device), torch.LongTensor(cols).to(device)]
    _adj_matrix_cache[n_nodes][batch_size] = edges
    return edges


def _get_property_prediction_base() -> str:
    """Return the base directory for QM9 property prediction (in-repo or from env)."""
    env = os.environ.get("OC_FLOW_MOLECULE_PATH", "").strip()
    if env:
        return os.path.join(env, "qm9", "property_prediction")
    base = Path(__file__).resolve().parent.parent / "models" / "qm9" / "property_prediction"
    return str(base) if base.is_dir() else ""


class MultiPropertyTargetReward(MoleculeRewardFunction):
    """Multi-property QM9 reward using OC-Flow / EquiFM EGNN classifiers.

    Reads ``properties`` and ``targets`` from the evaluation context (provided by
    ``QM9PropertyTargetsBenchmark``) and returns the mean negative absolute deviation
    across all properties.

    Set the ``OC_FLOW_MOLECULE_PATH`` environment variable to the OC-Flow molecule
    directory to enable real classifier predictions.
    """

    def __init__(
        self,
        config: Optional["RewardConfig"] = None,
        device: Optional[str] = None,
        weights: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        if config is not None:
            device = device or config.device
            metadata = config.metadata or {}
            weights = weights or metadata.get("weights", {})
        if device is None:
            device = kwargs.get("device", None)
        if weights is None:
            weights = kwargs.get("weights", {})
        super().__init__("multi_property_target", device)
        self.weights: Dict[str, float] = weights or {}
        from ..models.qm9.consts import qm9_stats
        self.property_stats = {p: {"mean": v[0], "mad": v[1]} for p, v in qm9_stats.items()}
        self.classifiers: Dict[str, Any] = {}
        self._tried_loading_classifiers = False
        self._property_prediction_base: str = ""

    def _load_classifiers(self, properties: List[str]) -> None:
        """Load pre-trained EGNN classifiers from the in-repo property_prediction module."""
        if self._tried_loading_classifiers:
            return
        import pickle
        base = _get_property_prediction_base()
        if not base or not os.path.isdir(base):
            self._tried_loading_classifiers = True
            return
        self._property_prediction_base = base
        try:
            # Import EGNN via the package path — avoids executing main_qm9_prop.py
            # which has module-level imports requiring the external OC-Flow qm9 package.
            from noise_optimization.core.models.qm9.property_prediction.models_property import EGNN

            for prop in properties:
                if prop not in self.property_stats:
                    continue
                ckpt = os.path.join(base, "outputs", f"exp_class_{prop}", "best_checkpoint.npy")
                args_file = os.path.join(base, "outputs", f"exp_class_{prop}", "args.pickle")
                if os.path.exists(ckpt) and os.path.exists(args_file):
                    with open(args_file, "rb") as f:
                        cargs = pickle.load(f)
                    clf = EGNN(
                        in_node_nf=5, in_edge_nf=0,
                        hidden_nf=cargs.nf, device=self.device,
                        n_layers=cargs.n_layers, coords_weight=1.0,
                        attention=cargs.attention, node_attr=cargs.node_attr,
                    )
                    clf.load_state_dict(torch.load(ckpt, map_location=self.device))
                    clf.to(self.device).eval()
                    self.classifiers[prop] = clf
        except Exception as e:
            print(f"[MultiPropertyTargetReward] Failed to load classifiers from {base}: {e}")
        self._tried_loading_classifiers = True

    def evaluate(self, candidates: Any, context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        if not isinstance(candidates, torch.Tensor):
            candidates = torch.as_tensor(candidates, device=self.device, dtype=torch.float32)
        else:
            candidates = candidates.to(self.device)
        batch_size = candidates.shape[0]
        if context is None:
            return torch.zeros(batch_size, device=self.device)
        properties = context.get("properties", [])
        targets = context.get("targets", {})
        if not properties or not targets:
            return torch.zeros(batch_size, device=self.device)
        if not self._tried_loading_classifiers:
            self._load_classifiers(properties)
        total_reward = torch.zeros(batch_size, device=self.device)
        for prop_name in properties:
            if prop_name not in targets:
                continue
            target_value = float(targets[prop_name])
            predicted = self._predict_property(candidates, prop_name, context)
            # Sanitise NaN/Inf before reward computation
            predicted = torch.nan_to_num(predicted, nan=0.0, posinf=0.0, neginf=0.0)
            prop_reward = -torch.abs(predicted - target_value)
            prop_reward = torch.nan_to_num(prop_reward, nan=-1e5, posinf=-1e5, neginf=-1e5)
            total_reward = total_reward + prop_reward * self.weights.get(prop_name, 1.0)
        num_props = len([p for p in properties if p in targets])
        if num_props > 0:
            total_reward = total_reward / num_props
        return torch.nan_to_num(total_reward, nan=-1e5, posinf=-1e5, neginf=-1e5)

    def _predict_property(
        self,
        molecules: torch.Tensor,
        property_name: str,
        context: Dict[str, Any],
    ) -> torch.Tensor:
        """Predict property using OC-Flow classifier, or fall back to heuristic."""
        batch_size = molecules.shape[0]
        if property_name in self.classifiers:
            try:
                clf = self.classifiers[property_name]
                x = molecules[:, :, :3]
                one_hot = molecules[:, :, 3:8]
                n_nodes = molecules.shape[1]
                node_mask = (one_hot.abs().sum(dim=-1, keepdim=True) > 1e-6).float()
                atom_pos = x.view(batch_size * n_nodes, -1)
                node_mask_flat = node_mask.view(batch_size * n_nodes, -1)
                nodes = one_hot.view(batch_size * n_nodes, -1)
                edges = _get_adj_matrix(n_nodes, batch_size, self.device)
                node_mask_2d = node_mask.squeeze(-1)
                edge_mask_2d = node_mask_2d.unsqueeze(1) * node_mask_2d.unsqueeze(2)
                diag = ~torch.eye(n_nodes, dtype=torch.bool, device=self.device).unsqueeze(0)
                edge_mask = (edge_mask_2d * diag).view(batch_size * n_nodes * n_nodes, 1)
                with torch.set_grad_enabled(molecules.requires_grad):
                    pred_raw = clf(
                        h0=nodes, x=atom_pos, edges=edges, edge_attr=None,
                        node_mask=node_mask_flat, edge_mask=edge_mask, n_nodes=n_nodes,
                    )
                return pred_raw.view(-1)
            except Exception as e:
                if not hasattr(self, f"_warned_{property_name}"):
                    print(f"[MultiPropertyTargetReward] Classifier error for {property_name}: {e}, using heuristic")
                    setattr(self, f"_warned_{property_name}", True)
        # Heuristic fallback (normalized scale)
        pos = molecules[:, :, :3]
        one_hot = molecules[:, :, 3:8]
        node_mask = (one_hot.abs().sum(dim=-1, keepdim=True) > 1e-6).float()
        n_atoms = node_mask.sum(dim=1).clamp(min=1)
        if property_name == "mu":
            sq_dist = (pos ** 2).sum(dim=-1)
            avg_dist = (sq_dist * node_mask.squeeze(-1)).sum(dim=1) / n_atoms.view(-1)
            return (avg_dist - 2.0) * 0.5
        elif property_name == "Cv":
            return (n_atoms.view(-1).float() - 18.0) / 5.0
        else:
            atomic_weights = torch.tensor([1.0, 12.0, 14.0, 16.0, 19.0], device=self.device)
            type_sum = (one_hot * atomic_weights).sum(dim=-1).sum(dim=1)
            return (type_sum / n_atoms.view(-1) - 7.0) / 2.0

    def _evaluate_molecule(self, molecule: Any) -> float:
        return 0.0

    def get_input_format(self) -> str:
        return "tensor"


def register_target_property_rewards() -> None:
    RewardFunctionRegistry.register("multi_property_target", MultiPropertyTargetReward)
