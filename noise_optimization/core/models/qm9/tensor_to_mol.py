"""Convert QM9 decoded tensors to RDKit molecules/SMILES for reward evaluation.

QM9 model outputs [batch, n_nodes, 9] where 9 = x,y,z (3) + one_hot (5) + charge (1).
RDKit-based rewards (QED, SA, logp, etc.) need SMILES or Mol - this module converts.
"""
from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
import torch

from .consts import qm9_with_h
from .stability import get_bond_order

# RDKit bond type mapping: 0=None, 1=SINGLE, 2=DOUBLE, 3=TRIPLE
try:
    from rdkit import Chem
    BOND_DICT = [
        None,
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
    ]
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    BOND_DICT = []


def _build_molecule_from_atoms(
    positions: np.ndarray,
    atom_types: np.ndarray,
    dataset_info: dict,
) -> Optional[Any]:
    """Build RDKit Mol from positions and atom types. Returns None if conversion fails."""
    if not RDKIT_AVAILABLE:
        return None
    atom_decoder = dataset_info["atom_decoder"]
    n = len(positions)
    if n == 0:
        return None

    # Build adjacency and bond type from distances
    A = np.zeros((n, n), dtype=bool)
    E = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i):
            atom1 = atom_decoder[atom_types[i]]
            atom2 = atom_decoder[atom_types[j]]
            dist = float(np.linalg.norm(positions[i] - positions[j]))
            order = get_bond_order(atom1, atom2, dist)
            if order > 0:
                A[i, j] = True
                E[i, j] = order

    mol = Chem.RWMol()
    for i in range(n):
        a = Chem.Atom(atom_decoder[atom_types[i]])
        mol.AddAtom(a)

    for i in range(n):
        for j in range(i):
            if A[i, j] and E[i, j] < len(BOND_DICT) and BOND_DICT[E[i, j]] is not None:
                mol.AddBond(int(i), int(j), BOND_DICT[E[i, j]])

    # Add 3D conformer from actual positions (required for energy rewards like gfn2xtb)
    try:
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i in range(n):
            conf.SetAtomPosition(i, (float(positions[i, 0]), float(positions[i, 1]), float(positions[i, 2])))
        mol.AddConformer(conf, assignId=True)
    except Exception:
        pass  # Mol still usable for 2D rewards without conformer
    return mol


def _mol_to_smiles(mol: Any) -> Optional[str]:
    """Convert RDKit Mol to SMILES. Returns None if sanitization fails."""
    if not RDKIT_AVAILABLE or mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def qm9_tensor_to_molecules(
    decoded: torch.Tensor,
    dataset_info: Optional[dict] = None,
) -> List[Any]:
    """Convert QM9 decoded tensor to list of RDKit Mol objects.

    Args:
        decoded: [batch, n_nodes, 9] - x,y,z, one_hot(5), charge
        dataset_info: QM9 dataset info dict with atom_decoder (default: qm9_with_h)

    Returns:
        List of RDKit Mol or None for each molecule (None = conversion failed)
    """
    if not RDKIT_AVAILABLE:
        return [None] * decoded.shape[0]
    dataset_info = dataset_info or qm9_with_h

    # Handle 4D (time, batch, n_nodes, feat) - take last timestep
    if decoded.ndim == 4:
        if decoded.shape[0] >= 50 and decoded.shape[1] < decoded.shape[0]:
            decoded = decoded[-1]
        else:
            decoded = decoded[:, -1, :, :]

    batch_size = decoded.shape[0]
    x = decoded[:, :, :3]  # positions
    one_hot = decoded[:, :, 3:8]  # atom type one-hot
    node_mask = (one_hot.abs().sum(dim=-1) > 1e-6)  # valid atoms

    results = []
    for i in range(batch_size):
        mask = node_mask[i].cpu().numpy()
        n_real = int(mask.sum())
        if n_real == 0:
            results.append(None)
            continue
        positions = x[i][mask].detach().cpu().numpy()
        atom_types = one_hot[i][mask].argmax(dim=-1).cpu().numpy()
        mol = _build_molecule_from_atoms(positions, atom_types, dataset_info)
        results.append(mol)
    return results


def qm9_tensor_to_smiles_list(
    decoded: torch.Tensor,
    dataset_info: Optional[dict] = None,
) -> List[Optional[str]]:
    """Convert QM9 decoded tensor to list of SMILES strings.

    Returns:
        List of SMILES or None for each molecule (None = conversion failed)
    """
    mols = qm9_tensor_to_molecules(decoded, dataset_info)
    return [_mol_to_smiles(m) if m is not None else None for m in mols]
