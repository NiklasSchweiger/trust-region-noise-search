# Atom colors (RGB tuples) for visualization
qm9_colors = [
    [1.0, 1.0, 1.0],  # H - White
    [0.565, 0.565, 0.565],  # C - Gray
    [0.188, 0.314, 0.973],  # N - Blue
    [1.0, 0.051, 0.051],  # O - Red
    [0.565, 0.878, 0.314],  # F - Green
]

# Atom radii (in Angstroms) for visualization
qm9_radii = [0.31, 0.76, 0.71, 0.66, 0.57]

qm9_with_h = {
    'name': 'qm9',
    'atom_encoder': {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4},
    'atom_decoder': ['H', 'C', 'N', 'O', 'F'],
    'colors_dic': qm9_colors,
    'radius_dic': qm9_radii,
    'n_nodes': {
        22: 3393, 17: 13025, 23: 4848, 21: 9970, 19: 13832, 20: 9482, 16: 10644,
        13: 3060, 15: 7796, 25: 1506, 18: 13364, 12: 1689, 11: 807, 24: 539,
        14: 5136, 26: 48, 7: 16, 10: 362, 8: 49, 9: 124, 27: 266, 4: 4, 29: 25,
        6: 9, 5: 5, 3: 1
    },
    'max_n_nodes': 29,
    'with_h': True,
}

# Approximate dataset property statistics (mean, std). Used to sample targets
# and to normalize sampled targets without external dataset dependencies.
qm9_stats = {
    'alpha': [75.37342834472656, 6.272772312164307],
    'gap': [6.863183975219727, 1.061903715133667],
    'homo': [-6.538593769073486, 0.43957480788230896],
    'lumo': [0.3245810568332672, 1.0337715148925781],
    'mu': [2.6750874519348145, 1.1757327318191528],
    'Cv': [-22.166629791259766, 4.888548851013184],
}


