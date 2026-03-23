"""Molecule stability analysis for QM9.

Bond-order inference and valency-checking following the EquiFM / OC-Flow approach.
Attribution: GitHub https://github.com/AI4Science-WestlakeU/CIF
arXiv https://arxiv.org/abs/2312.07168
"""
import torch
from typing import Dict, List, Any, Optional
import numpy as np

from .consts import qm9_with_h


# Bond-length thresholds in picometers (distance in Angstrom x 100).
# Source: OC-Flow / EquiFM codebase.
bonds1 = {
    'H': {'H': 74, 'C': 109, 'N': 101, 'O': 96,  'F': 92},
    'C': {'H': 109, 'C': 154, 'N': 147, 'O': 143, 'F': 135},
    'N': {'H': 101, 'C': 147, 'N': 145, 'O': 140, 'F': 136},
    'O': {'H': 96,  'C': 143, 'N': 140, 'O': 148, 'F': 142},
    'F': {'H': 92,  'C': 135, 'N': 136, 'O': 142, 'F': 142},
}

bonds2 = {
    'C': {'C': 134, 'N': 129, 'O': 120},
    'N': {'C': 129, 'N': 125, 'O': 121},
    'O': {'C': 120, 'N': 121, 'O': 121},
}

bonds3 = {
    'C': {'C': 120, 'N': 116, 'O': 113},
    'N': {'C': 116, 'N': 110},
    'O': {'C': 113},
}

# Margins tuned to maximise stability on QM9 true samples (from OC-Flow).
margin1, margin2, margin3 = 10, 5, 3

# Maximum allowed bonds per atom type.
allowed_bonds = {'H': 1, 'C': 4, 'N': 3, 'O': 2, 'F': 1}


def get_bond_order(atom1: str, atom2: str, distance: float) -> int:
    """Determine bond order from interatomic distance (in Angstroms)."""
    distance_pm = distance * 100  # Angstrom to pm

    if atom1 in bonds1 and atom2 in bonds1[atom1]:
        if distance_pm < bonds1[atom1][atom2] + margin1:
            if atom1 in bonds2 and atom2 in bonds2[atom1]:
                if distance_pm < bonds2[atom1][atom2] + margin2:
                    if atom1 in bonds3 and atom2 in bonds3[atom1]:
                        if distance_pm < bonds3[atom1][atom2] + margin3:
                            return 3
                    return 2
            return 1
    return 0


def check_stability(positions: np.ndarray, atom_types: np.ndarray, dataset_info: Dict) -> tuple:
    """Check stability of a single molecule.

    Returns:
        (molecule_stable, nr_stable_bonds, n_atoms)
    """
    assert len(positions.shape) == 2 and positions.shape[1] == 3

    atom_decoder = dataset_info['atom_decoder']
    n_atoms = len(positions)
    nr_bonds = np.zeros(n_atoms, dtype='int')

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            dist = np.linalg.norm(positions[i] - positions[j])
            order = get_bond_order(atom_decoder[atom_types[i]], atom_decoder[atom_types[j]], dist)
            nr_bonds[i] += order
            nr_bonds[j] += order

    nr_stable_bonds = 0
    for atom_type_idx, nr_bonds_i in zip(atom_types, nr_bonds):
        atom_symbol = atom_decoder[atom_type_idx]
        possible_bonds = allowed_bonds[atom_symbol]
        is_stable = (possible_bonds == nr_bonds_i) if isinstance(possible_bonds, int) else (nr_bonds_i in possible_bonds)
        nr_stable_bonds += int(is_stable)

    molecule_stable = (nr_stable_bonds == n_atoms)
    return molecule_stable, nr_stable_bonds, n_atoms


def analyze_stability_for_molecules(
    molecules_dict: Dict[str, List],
    node_mask: Optional[torch.Tensor] = None,
    edge_mask: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Analyse stability of a batch of molecules.

    Args:
        molecules_dict: dict with keys:
            'one_hot'   -- list of [n_nodes, n_atom_types] tensors
            'x'         -- list of [n_nodes, 3] position tensors
            'node_mask' -- optional list of [n_nodes, 1] mask tensors
        node_mask, edge_mask: legacy positional args (ignored).

    Returns:
        dict with 'atom_stable', 'mol_stable', 'valid_bonds'.
    """
    one_hot_list = molecules_dict.get('one_hot', [])
    x_list = molecules_dict.get('x', [])
    node_mask_list = molecules_dict.get('node_mask', [])

    if not one_hot_list or not x_list:
        n = len(one_hot_list) if one_hot_list else 0
        return {'atom_stable': [False] * n, 'mol_stable': [False] * n, 'valid_bonds': [False] * n}

    # Handle batched-tensor input (OC-Flow / EquiFM convention).
    if isinstance(one_hot_list, torch.Tensor):
        if node_mask_list and isinstance(node_mask_list, torch.Tensor):
            atomsxmol = torch.sum(node_mask_list, dim=1)
            one_hot_list = [one_hot_list[i][:int(atomsxmol[i])] for i in range(len(one_hot_list))]
            x_list = [x_list[i][:int(atomsxmol[i])] for i in range(len(x_list))]
            node_mask_list = [node_mask_list[i][:int(atomsxmol[i])] for i in range(len(node_mask_list))]
        else:
            one_hot_list = list(one_hot_list)
            x_list = list(x_list)

    dataset_info = qm9_with_h
    atom_stable, mol_stable, valid_bonds = [], [], []

    for i, (one_hot, x) in enumerate(zip(one_hot_list, x_list)):
        mask = node_mask_list[i] if i < len(node_mask_list) else None

        if isinstance(one_hot, torch.Tensor):
            one_hot = one_hot.detach().cpu().numpy()
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if mask is not None and isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()

        if mask is not None:
            valid_mask = mask.squeeze(-1) > 0 if mask.ndim > 1 else mask > 0
            x = x[valid_mask]
            one_hot = one_hot[valid_mask]

        atom_types = np.argmax(one_hot, axis=-1)
        molecule_stable_bool, _, n_atoms = check_stability(x, atom_types, dataset_info)

        if molecule_stable_bool:
            atoms_stable_list = [True] * n_atoms
        else:
            atom_decoder = dataset_info['atom_decoder']
            nr_bonds = np.zeros(n_atoms, dtype='int')
            for j in range(n_atoms):
                for k in range(j + 1, n_atoms):
                    dist = np.linalg.norm(x[j] - x[k])
                    order = get_bond_order(atom_decoder[atom_types[j]], atom_decoder[atom_types[k]], dist)
                    nr_bonds[j] += order
                    nr_bonds[k] += order
            atoms_stable_list = []
            for atom_type_idx, nr_bonds_i in zip(atom_types, nr_bonds):
                atom_symbol = atom_decoder[atom_type_idx]
                possible_bonds = allowed_bonds[atom_symbol]
                is_stable = (possible_bonds == nr_bonds_i) if isinstance(possible_bonds, int) else (nr_bonds_i in possible_bonds)
                atoms_stable_list.append(is_stable)

        atom_stable.append(atoms_stable_list)
        mol_stable.append(molecule_stable_bool)
        valid_bonds.append(molecule_stable_bool)

    return {'atom_stable': atom_stable, 'mol_stable': mol_stable, 'valid_bonds': valid_bonds}


def compute_stability_score(
    molecules_dict: Dict[str, List],
    node_mask: Optional[torch.Tensor] = None,
    edge_mask: Optional[torch.Tensor] = None,
) -> float:
    """Return the fraction of stable molecules in a batch."""
    mol_stable = analyze_stability_for_molecules(molecules_dict, node_mask, edge_mask)['mol_stable']
    return float(sum(mol_stable)) / len(mol_stable) if mol_stable else 0.0
