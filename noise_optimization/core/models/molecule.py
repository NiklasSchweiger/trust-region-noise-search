from __future__ import annotations

from typing import Any, Callable, Optional, Tuple, List

import torch


class MoleculeSamplingWrapper(torch.nn.Module):
    """Generic wrapper to adapt molecule generators to the core interface.

    This wrapper expects an underlying model that can advance samples from
    initial noise through a dynamics function or decode process. It exposes
    a `forward(batch_size, initial_noise, **kwargs)` that returns generated
    molecular representations (format is model-specific), along with helpers
    to discover noise shapes and sample latents when possible.

    Optionally, a `graph_to_molecule` callback can be provided to convert model
    outputs (e.g., positions/one_hot/masks) into RDKit molecules or SMILES.
    """

    def __init__(
        self,
        underlying_model: Any,
        *,
        get_noise_shape_fn: Optional[Callable[[], Tuple[int, ...]]] = None,
        sample_latents_fn: Optional[Callable[[int], torch.Tensor]] = None,
        graph_to_molecule: Optional[Callable[[Any], Any]] = None,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.underlying_model = underlying_model
        self._get_noise_shape_fn = get_noise_shape_fn
        self._sample_latents_fn = sample_latents_fn
        self._graph_to_molecule = graph_to_molecule
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ----- Optional helpers used by Problem builders -----
    def get_noise_latent_shape(self) -> Tuple[int, ...]:
        if self._get_noise_shape_fn is not None:
            return tuple(self._get_noise_shape_fn())
        if hasattr(self.underlying_model, "get_noise_latent_shape"):
            return tuple(self.underlying_model.get_noise_latent_shape())  # type: ignore[attr-defined]
        if hasattr(self.underlying_model, "sample_latents"):
            sample = self.underlying_model.sample_latents(batch_size=1)  # type: ignore[attr-defined]
            return tuple(sample.shape[1:])
        raise RuntimeError("Noise latent shape is not available; provide get_noise_shape_fn")

    def sample_latents(self, batch_size: int = 1) -> torch.Tensor:
        batch_size = int(batch_size)
        if self._sample_latents_fn is not None:
            return self._sample_latents_fn(batch_size)
        if hasattr(self.underlying_model, "sample_latents"):
            return self.underlying_model.sample_latents(batch_size=batch_size)  # type: ignore[attr-defined]
        # Fall back to discovered shape
        shape = self.get_noise_latent_shape()
        return torch.randn(batch_size, *shape, device=self.device)

    # ----- Core generation API -----
    def forward(self, batch_size: int = 1, initial_noise: Optional[torch.Tensor] = None, **kwargs: Any) -> Any:
        # Use underlying model to produce raw graph/object representation
        batch_size = int(batch_size)
        grad_enabled = kwargs.get("differentiable", False)
        
        with torch.set_grad_enabled(grad_enabled):
            if initial_noise is None:
                initial_noise = self.sample_latents(batch_size=batch_size)
            
            if not grad_enabled and initial_noise.requires_grad:
                initial_noise = initial_noise.detach()

            # Two common patterns supported:
            # 1) underlying_model.forward(batch_size=..., initial_noise=..., **kwargs)
            # 2) underlying_model.forward(batch_size, initial_noise, **kwargs)
            try:
                raw = self.underlying_model.forward(batch_size=batch_size, initial_noise=initial_noise, **kwargs)
            except TypeError:
                raw = self.underlying_model.forward(batch_size, initial_noise, **kwargs)

            # Optionally convert to RDKit molecules / SMILES if a converter is provided
            if self._graph_to_molecule is not None:
                try:
                    return self._graph_to_molecule(raw)
                except Exception:
                    # Fall back to raw if conversion fails
                    return raw
            return raw


class QM9FlowWrapper(MoleculeSamplingWrapper):
    """Lightweight convenience wrapper around a QM9 flow model.

    The underlying model is expected to implement a callable drift function
    `f(t, x)` used by an explicit Euler sampler (as in typical CNF/GDSS flows).
    Provide a pre-constructed object exposing `__call__(t, x)` or `forward(t, x)`.

    The wrapper integrates from t=1 to t=0 over `n_step` steps to produce `x1`.
    If `graph_to_molecule` is provided, it will be applied to the output.
    """

    def __init__(
        self,
        flow_drift_model: Any,
        *,
        node_mask: Optional[torch.Tensor] = None,
        edge_mask: Optional[torch.Tensor] = None,
        n_step: int = 50,
        max_n_nodes: int = 29,
        nodes_dist: Optional[Any] = None,
        get_noise_shape_fn: Optional[Callable[[], Tuple[int, ...]]] = None,
        sample_latents_fn: Optional[Callable[[int], torch.Tensor]] = None,
        graph_to_molecule: Optional[Callable[[Any], Any]] = None,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(
            flow_drift_model,
            get_noise_shape_fn=get_noise_shape_fn,
            sample_latents_fn=sample_latents_fn,
            graph_to_molecule=graph_to_molecule,
            device=device,
        )
        self.n_step = int(n_step)
        self.max_n_nodes = max_n_nodes
        self.nodes_dist = nodes_dist  # For sampling n_nodes if not specified
        # Store template masks (batch_size=1) - used as fallback
        self._template_node_mask = node_mask
        self._template_edge_mask = edge_mask

    def _create_masks_for_n_nodes(self, batch_size: int, n_nodes: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create node_mask and edge_mask for a specific n_nodes.
        
        This matches OC-Flow's setup_generation but for a fixed n_nodes.
        """
        max_n = self.max_n_nodes
        # node_mask: [batch, max_n_nodes, 1] with first n_nodes positions set to 1
        node_mask = torch.zeros(batch_size, max_n, device=device)
        node_mask[:, :n_nodes] = 1.0
        
        # edge_mask: [batch * max_n * max_n, 1]
        # Only edges between valid nodes (first n_nodes) are valid
        edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)  # [batch, max_n, max_n]
        diag_mask = ~torch.eye(max_n, dtype=torch.bool, device=device).unsqueeze(0)  # No self-loops
        edge_mask = edge_mask * diag_mask
        edge_mask = edge_mask.view(batch_size * max_n * max_n, 1)
        
        node_mask = node_mask.unsqueeze(2)  # [batch, max_n, 1]
        return node_mask, edge_mask

    def sample_latents(self, batch_size: int = 1, n_nodes: Optional[int] = None) -> torch.Tensor:
        """Sample initial latents using the flow model's special sampling."""
        batch_size = int(batch_size)
        
        # Determine n_nodes to use
        if n_nodes is None:
            if self.nodes_dist is not None:
                # Sample from distribution
                nodesxsample = self.nodes_dist.sample(batch_size)
                n_nodes = int(nodesxsample[0])  # Use first sample for uniform batch
            else:
                # Use template's n_nodes
                n_nodes = int(self._template_node_mask.sum(dim=1).max().item()) if self._template_node_mask is not None else self.max_n_nodes
        
        if hasattr(self.underlying_model, 'sample_combined_position_feature_noise'):
            # Create proper masks for this n_nodes
            node_mask, _ = self._create_masks_for_n_nodes(batch_size, n_nodes, 
                                                          torch.device(self.device))
            return self.underlying_model.sample_combined_position_feature_noise(batch_size, self.max_n_nodes, node_mask)
        
        # Fallback to parent implementation
        return super().sample_latents(batch_size)

    def forward(self, batch_size: int = 1, initial_noise: Optional[torch.Tensor] = None, reverse_t: bool = True, n_step: Optional[int] = None, **kwargs: Any) -> Any:
        batch_size = int(batch_size)
        grad_enabled = kwargs.get("differentiable", False)
        
        # Get n_nodes from kwargs (passed from task context) or use default
        n_nodes = kwargs.get("n_nodes", None)
        
        with torch.set_grad_enabled(grad_enabled):
            x = initial_noise if initial_noise is not None else self.sample_latents(batch_size=batch_size, n_nodes=n_nodes)
            if not grad_enabled:
                x = x.detach().clone().to(self.device)
            else:
                x = x.to(self.device)
            
            actual_batch_size = x.shape[0]

        # Determine n_nodes for mask creation
        if n_nodes is None:
            # Try to infer from template or use max
            if self._template_node_mask is not None:
                n_nodes = int(self._template_node_mask.sum(dim=1).max().item())
            else:
                n_nodes = self.max_n_nodes
        
        # Create FRESH masks for this specific batch and n_nodes
        # This matches OC-Flow's pattern of creating masks per-batch
        node_mask_actual, edge_mask_actual = self._create_masks_for_n_nodes(
            actual_batch_size, n_nodes, x.device
        )
        
        # CRITICAL: Preprocess the noise to satisfy OC-Flow's requirements
        # OC-Flow expects:
        # 1. Masked positions (invalid nodes) to be zero
        # 2. Positions (first 3 features) to be mean-centered (zero center of gravity)
        
        # Zero out masked (invalid) positions
        x = x * node_mask_actual
        
        # Mean-center positions (first 3 dims) per molecule
        # This ensures zero center of gravity, which OC-Flow expects
        positions = x[:, :, :3]  # [batch, n_nodes, 3]
        n_valid_nodes = node_mask_actual.sum(dim=1, keepdim=True).clamp(min=1)  # [batch, 1, 1]
        # Compute center of mass for each molecule
        center_of_mass = (positions * node_mask_actual).sum(dim=1, keepdim=True) / n_valid_nodes  # [batch, 1, 3]
        # Subtract center of mass (only from valid nodes)
        positions_centered = positions - center_of_mass * node_mask_actual
        # Update x with centered positions
        x = torch.cat([positions_centered, x[:, :, 3:]], dim=2)
        
        # Update model's conditional params with fresh masks
        if hasattr(self.underlying_model, 'set_conditional_param'):
            self.underlying_model.set_conditional_param(node_mask=node_mask_actual, edge_mask=edge_mask_actual, context=None)

        # IMPORTANT: OC-Flow's decode() uses odeint which stores ALL 100 intermediate states
        # This causes severe memory accumulation over iterations (34+ GB after a few iterations)
        # Always use manual Euler integration which is much more memory-efficient
        # The quality difference is negligible for optimization purposes
        use_decode = False  # Disabled to prevent OOM
        
        # Manual Euler integration for models without decode method or if decode failed
        if not use_decode:
            step_count = int(n_step) if n_step is not None else self.n_step
            tlist = [1, 0] if reverse_t else [0, 1]
            ts = torch.linspace(*tlist, step_count + 1, device=x.device)
            dt = -1.0 / step_count if reverse_t else 1.0 / step_count

            # Mask out invalid nodes if mask is provided
            if self._template_node_mask is not None:
                x = x.masked_fill(~node_mask_actual.bool().to(x.device), 0)

            model = self.underlying_model
            f = model if callable(model) else getattr(model, "forward")
            for t in ts[:-1]:
                x = x + f(t, x) * dt
                if self._template_node_mask is not None:
                    x = x.masked_fill(~node_mask_actual.bool().to(x.device), 0)

        # Cleanup: clear intermediate tensors and CUDA cache to prevent memory accumulation
        del node_mask_actual, edge_mask_actual
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Optional conversion to RDKit/SMILES via callback
        if self._graph_to_molecule is not None:
            try:
                return self._graph_to_molecule(x)
            except Exception:
                return x
        return x


# ============================================================================
# Factory function for QM9 molecules
# ============================================================================

def make_qm9_flow_from_cfg(cfg: dict, device: str = "cuda"):
    """Factory that builds a QM9 flow model from config.
    
    Expected keys in cfg:
      - molecule.generator.type: "qm9_flow" or "qm9-flow" or "qm9"
      - molecule.generator.weights_path: optional path to weights
      - molecule.generator.n_step: number of integration steps (default: 50)
    """
    from .qm9.utils import get_flow_model, setup_generation
    from .qm9.consts import qm9_with_h
    from omegaconf import DictConfig, OmegaConf
    
    mol_cfg = cfg.get("molecule") or {}
    # Normalize DictConfig -> dict for robust access
    if isinstance(mol_cfg, DictConfig):
        mol_cfg = OmegaConf.to_container(mol_cfg, resolve=True)
    gen = (mol_cfg.get("generator") or {}) if hasattr(mol_cfg, "get") else {}
    if isinstance(gen, DictConfig):
        gen = OmegaConf.to_container(gen, resolve=True)
    gtype = str((gen or {}).get("type", "")).lower()
    
    if gtype not in ("qm9_flow", "qm9-flow", "qm9"):
        raise ValueError(f"Unsupported generator type: {gtype}")
    
    weights_path = (gen or {}).get("weights_path")
    flow, nodes_dist, deq, margs = get_flow_model(device, weights_path=weights_path)
    max_n = qm9_with_h["max_n_nodes"]
    # Create initial template masks (will be recreated properly in forward based on task's n_nodes)
    node_mask, edge_mask, nodesxsample = setup_generation(1, nodes_dist, max_n, device)
    
    n_step = int((gen or {}).get("n_step", 50))
    # Infer noise shape from node_mask: [B, N, 1] + in_node_nf features
    in_node_nf = len(qm9_with_h["atom_decoder"]) + int(margs.include_charges)
    noise_shape = (max_n, 3 + in_node_nf)  # positions (3) + features (in_node_nf)
    wrapper = QM9FlowWrapper(
        flow,
        node_mask=node_mask,
        edge_mask=edge_mask,
        n_step=n_step,
        max_n_nodes=max_n,
        nodes_dist=nodes_dist,  # Pass for sampling n_nodes if not specified
        get_noise_shape_fn=lambda: noise_shape,
        device=device,
    )
    return wrapper


