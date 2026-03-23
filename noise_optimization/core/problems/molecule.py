from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from .base import Problem
from ..rewards.base import RewardFunction


class QM9OptimizationProblem(Problem):
    """Problem definition for QM9/small-molecule generation/optimization."""

    def __init__(
        self,
        generative_model: Any,
        reward_model: RewardFunction | Any,
        *,
        device: Optional[str] = None,
        noise_dims: Optional[Tuple[int, ...]] = None,
        model_config: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        track_stability: bool = True,
    ) -> None:
        super().__init__()
        self.generative_model = generative_model
        self.reward_model = reward_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_config = model_config or {}
        self.noise_dims = noise_dims
        self._context = {
            "domain": "molecule",
            "noise_dims": self.noise_dims,
            "model_config": dict(self.model_config),
            **(context or {}),
        }
        self.track_stability = track_stability
        self.last_stability: Optional[float] = None
        self.last_stability_dict: Optional[Dict[str, Any]] = None
        self._current_iteration = 0
        # Track unique molecules for VUP calculation
        self._seen_molecules: set = set()

    def evaluate(self, candidates: Any, noise_list: Optional[List[torch.Tensor]] = None, grad_enabled: bool = False, **kwargs: Any) -> torch.Tensor:
        if isinstance(candidates, torch.Tensor):
            if not grad_enabled:
                candidates = candidates.detach().clone().to(self.device)
            else:
                candidates = candidates.to(self.device)
            # Track the number of samples evaluated
            self._total_samples_evaluated += int(candidates.shape[0])
        elif isinstance(candidates, list):
            # Track the number of samples evaluated
            self._total_samples_evaluated += len(candidates)
            
        with torch.set_grad_enabled(grad_enabled):
            decoded = self.decode_latents(candidates, noise_list=noise_list, grad_enabled=grad_enabled, **kwargs)
            return self.evaluate_decoded(decoded)

    def decode_latents(self, latents: Any, noise_list: Optional[List[torch.Tensor]] = None, grad_enabled: bool = False, **kwargs: Any) -> Any:
        params = dict(self.model_config)
        params.update(kwargs)
        params['noise_list'] = noise_list
        params['differentiable'] = grad_enabled
        
        # CRITICAL: Pass n_nodes from task context to the model for proper mask creation
        if 'n_nodes' not in params and 'n_nodes' in self._context:
            params['n_nodes'] = self._context['n_nodes']

        if isinstance(latents, torch.Tensor):
            batch_size = int(latents.shape[0])
            if not grad_enabled and latents.requires_grad:
                latents = latents.detach()
        else:
            batch_size = int(params.pop("batch_size", 1))

        try:
            return self.generative_model.forward(
                batch_size=batch_size,
                initial_noise=latents,
                **params,
            )
        except TypeError:
            return self.generative_model.forward(batch_size, latents, **params)

    def evaluate_decoded(self, decoded: Any, extra_context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        context = dict(self._context)
        if extra_context:
            context.update(extra_context)

        # RDKit-based rewards (QED, SA, logp, etc.) expect SMILES/Mol, not tensors.
        # Convert QM9 decoded tensor to list of molecules when needed.
        reward_input = decoded
        if isinstance(decoded, torch.Tensor) and isinstance(self.reward_model, RewardFunction):
            if getattr(self.reward_model, "get_input_format", lambda: "tensor")() == "smiles":
                from ..models.qm9.tensor_to_mol import qm9_tensor_to_molecules
                mols = qm9_tensor_to_molecules(decoded)
                reward_input = mols  # List of Mol or None

        if isinstance(self.reward_model, RewardFunction):
            rewards = self.reward_model.evaluate(reward_input, context=context)
        else:
            if hasattr(self.reward_model, "evaluate") and callable(getattr(self.reward_model, "evaluate")):
                rewards = self.reward_model.evaluate(reward_input, context=context)
            elif hasattr(self.reward_model, "predict_reward") and callable(getattr(self.reward_model, "predict_reward")):
                rewards = self.reward_model.predict_reward(decoded)
            else:
                raise TypeError("Reward model must implement evaluate(...) or predict_reward(...)")

        if isinstance(rewards, tuple) and len(rewards) > 0:
            rewards = rewards[0]

        if not isinstance(rewards, torch.Tensor):
            rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        if rewards.dim() > 1:
            rewards = rewards.view(-1)

        if self.track_stability and isinstance(decoded, torch.Tensor):
            try:
                from ..models.qm9.stability import analyze_stability_for_molecules
                import hashlib

                # Handle different decoded tensor shapes
                # Expected: [batch, n_nodes, features] where features = [x, y, z, one_hot...]
                # odeint returns: [time_steps, batch, n_nodes, features] - TIME FIRST
                if decoded.ndim == 4:
                    # Check if first dim is time_steps (usually 100 for OC-Flow)
                    # vs batch which is typically smaller
                    if decoded.shape[0] >= 50 and decoded.shape[1] < decoded.shape[0]:
                        # [time_steps, batch, n_nodes, features] -> take last time step
                        decoded = decoded[-1]  # [batch, n_nodes, features]
                    elif decoded.shape[1] < decoded.shape[2]:
                        # [batch, time_steps, n_nodes, features] -> take last time step
                        decoded = decoded[:, -1, :, :]
                    else:
                        # Likely [batch, n_nodes, features, extra] - squeeze last dim
                        decoded = decoded.squeeze(-1) if decoded.shape[-1] == 1 else decoded
                
                batch_size = decoded.shape[0]
                # Decoded tensor format: [batch, n_nodes, features] where features = [x, y, z, one_hot...]
                x = decoded[:, :, :3]
                one_hot = decoded[:, :, 3:8]

                # Atoms with non-zero one_hot encoding are real (threshold handles float precision).
                node_mask = (one_hot.abs().sum(dim=-1) > 1e-6).float().unsqueeze(-1)  # [batch, n_nodes, 1]
                
                # Format matches OC-Flow / EquiFM: list of per-molecule tensors
                molecules_dict = {
                    "one_hot": [one_hot[i] for i in range(batch_size)],
                    "x": [x[i] for i in range(batch_size)],
                    "node_mask": [node_mask[i] for i in range(batch_size)],
                }

                stability_dict = analyze_stability_for_molecules(molecules_dict)
                
                # Compute ASP (Atom Stability Percentage)
                atom_stable_list = stability_dict.get("atom_stable", [])
                if atom_stable_list:
                    # Filter out empty lists (molecules with no atoms)
                    valid_molecules = [atoms for atoms in atom_stable_list if len(atoms) > 0]
                    if valid_molecules:
                        total_atoms = sum(len(atoms) for atoms in valid_molecules)
                        stable_atoms = sum(sum(atoms) for atoms in valid_molecules)
                        asp = (stable_atoms / total_atoms * 100.0) if total_atoms > 0 else 0.0
                    else:
                        asp = 0.0
                        total_atoms = 0
                        stable_atoms = 0
                else:
                    asp = 0.0
                    total_atoms = 0
                    stable_atoms = 0
                stability_dict["asp"] = asp
                
                # Compute MSP (Molecule Stability Percentage)
                mol_stable = stability_dict.get("mol_stable", [])
                if mol_stable:
                    msp = (sum(mol_stable) / len(mol_stable) * 100.0) if mol_stable else 0.0
                    self.last_stability = float(sum(mol_stable)) / len(mol_stable)
                else:
                    msp = 0.0
                    self.last_stability = 0.0
                stability_dict["msp"] = msp
                
                # Compute VUP (Valid & Unique Percentage)
                # For uniqueness, create a hash based on atom types and approximate positions
                # Round positions to reduce noise from small variations
                valid_and_unique = []
                for i in range(batch_size):
                    is_valid = mol_stable[i] if i < len(mol_stable) else False
                    if is_valid:
                        # Create a hash from atom types and rounded positions
                        atom_types = torch.argmax(one_hot[i], dim=-1).cpu().numpy()
                        pos_rounded = (x[i].cpu().numpy() * 10).round() / 10  # Round to 0.1 Angstrom
                        mol_hash = hashlib.md5(
                            (str(atom_types.tolist()) + str(pos_rounded.tolist())).encode()
                        ).hexdigest()
                        
                        is_unique = mol_hash not in self._seen_molecules
                        if is_unique:
                            self._seen_molecules.add(mol_hash)
                        valid_and_unique.append(is_unique)
                    else:
                        valid_and_unique.append(False)
                
                vup = (sum(valid_and_unique) / len(valid_and_unique) * 100.0) if valid_and_unique else 0.0
                stability_dict["vup"] = vup
                stability_dict["valid_and_unique"] = valid_and_unique
                
                # Store extra metrics in stability_dict for logger
                stability_dict["stability"] = self.last_stability
                stability_dict["atom_stability"] = asp / 100.0
                stability_dict["mol_stability"] = msp / 100.0
                
                self.last_stability_dict = stability_dict
            except Exception as e:
                self.last_stability = None
                self.last_stability_dict = None
                if not hasattr(self, "_warned_stability_error"):
                    print(f"[QM9Problem] Error computing stability metrics: {e}")
                    self._warned_stability_error = True

        return rewards

    def sample(self, batch_size: int, latent_shape: Optional[Tuple[int, ...]] = None) -> torch.Tensor:
        if hasattr(self.generative_model, "sample_latents") and callable(getattr(self.generative_model, "sample_latents")):
            try:
                lat = self.generative_model.sample_latents(batch_size=batch_size)  # type: ignore[attr-defined]
                if isinstance(lat, torch.Tensor):
                    return lat.to(self.device)
            except Exception:
                pass

        shape = latent_shape or self.noise_dims
        if shape is None:
            raise ValueError("noise_dims must be provided or discoverable for QM9OptimizationProblem")
        return torch.randn(batch_size, *shape, device=self.device)

    @property
    def context(self) -> Dict[str, Any]:
        return dict(self._context, device=self.device)

    def supports_denoising_callbacks(self) -> bool:
        return hasattr(self.generative_model, "run_with_callback")

    @property
    def latent_shape(self) -> Optional[Tuple[int, ...]]:
        return self.noise_dims

    def set_iteration(self, iteration: int) -> None:
        self._current_iteration = iteration
    
    def reset_uniqueness_tracking(self) -> None:
        """Reset the uniqueness tracking set. Call this at the start of each task for fair comparison."""
        self._seen_molecules.clear()


__all__ = ["QM9OptimizationProblem"]
