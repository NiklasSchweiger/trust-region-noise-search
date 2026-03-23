"""Molecular visualization utilities for the core framework."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING
import torch
import numpy as np
import io

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    Image = None  # type: ignore
    PIL_AVAILABLE = False

if TYPE_CHECKING:
    from PIL.Image import Image as PILImageType

try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("[WARNING] RDKit not available. Molecular visualization will be limited.")

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    Axes3D = None  # type: ignore
    print("[WARNING] Matplotlib not available. Some visualization features will be limited.")


class MoleculeVisualizer:
    """Utility class for visualizing molecular structures and properties."""
    
    def __init__(self, image_size: tuple = (400, 400), dpi: int = 200):
        self.image_size = image_size
        self.dpi = dpi
    
    def smiles_to_image(self, smiles: str, title: Optional[str] = None) -> Optional["PILImageType"]:
        """Convert SMILES string to molecular structure image."""
        if not RDKIT_AVAILABLE or not PIL_AVAILABLE:
            return None
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Generate 2D coordinates
            AllChem.Compute2DCoords(mol)
            
            # Create image
            img = Draw.MolToImage(
                mol, 
                size=self.image_size,
                kekulize=True,
                wedgeBonds=True,
                imageType=None
            )
            
            # Add title if provided
            if title and MATPLOTLIB_AVAILABLE:
                img = self._add_title_to_image(img, title)
            
            return img
            
        except Exception as e:
            print(f"[ERROR] Creating molecular image: {e}")
            return None
    
    def molecules_to_grid(
        self, 
        molecules: List[Union[str, Any]], 
        rewards: Optional[List[float]] = None,
        properties: Optional[Dict[str, List[float]]] = None,
        max_molecules: int = 16
    ) -> Optional["PILImageType"]:
        """Create a grid of molecular structures."""
        if not RDKIT_AVAILABLE or not MATPLOTLIB_AVAILABLE or not PIL_AVAILABLE:
            return None
        
        # Limit number of molecules
        molecules = molecules[:max_molecules]
        if rewards is not None:
            rewards = rewards[:max_molecules]
        if properties is not None:
            properties = {k: v[:max_molecules] for k, v in properties.items()}
        
        # Convert molecules to images
        mol_images = []
        for i, mol in enumerate(molecules):
            if isinstance(mol, str):  # SMILES
                img = self.smiles_to_image(mol)
            else:
                # Try to convert to SMILES if possible
                try:
                    if hasattr(mol, 'smiles'):
                        img = self.smiles_to_image(mol.smiles)
                    else:
                        continue
                except:
                    continue
            
            if img is not None:
                mol_images.append(img)
        
        if not mol_images:
            return None
        
        # Create grid
        n_mols = len(mol_images)
        cols = min(4, n_mols)
        rows = (n_mols + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for i, (img, ax) in enumerate(zip(mol_images, axes)):
            ax.imshow(img)
            ax.axis('off')
            
            # Add title with reward/properties
            title_parts = []
            if rewards is not None and i < len(rewards):
                title_parts.append(f"R={rewards[i]:.3f}")
            if properties is not None:
                for prop_name, prop_values in properties.items():
                    if i < len(prop_values):
                        title_parts.append(f"{prop_name}={prop_values[i]:.3f}")
            
            if title_parts:
                ax.set_title(" | ".join(title_parts), fontsize=8)
        
        # Hide unused subplots
        for i in range(len(mol_images), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        # Convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        plt.close()
        
        return img
    
    def _add_title_to_image(self, img: "PILImageType", title: str) -> "PILImageType":
        """Add title to molecular image."""
        if not MATPLOTLIB_AVAILABLE or not PIL_AVAILABLE:
            return img
        
        fig, ax = plt.subplots(figsize=(self.image_size[0]/self.dpi, self.image_size[1]/self.dpi))
        ax.imshow(img)
        ax.set_title(title, fontsize=12, pad=10)
        ax.axis('off')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight')
        buf.seek(0)
        new_img = Image.open(buf)
        plt.close()
        
        return new_img
    
    def create_property_plot(
        self, 
        properties: Dict[str, List[float]], 
        rewards: Optional[List[float]] = None,
        targets: Optional[Dict[str, float]] = None
    ) -> Optional["PILImageType"]:
        """Create property vs reward scatter plots."""
        if not MATPLOTLIB_AVAILABLE or not PIL_AVAILABLE:
            return None
        
        n_props = len(properties)
        if n_props == 0:
            return None
        
        fig, axes = plt.subplots(1, n_props, figsize=(n_props * 4, 4))
        if n_props == 1:
            axes = [axes]
        
        for i, (prop_name, prop_values) in enumerate(properties.items()):
            ax = axes[i]
            
            if rewards is not None and len(rewards) == len(prop_values):
                scatter = ax.scatter(prop_values, rewards, alpha=0.6, s=20)
                ax.set_xlabel(prop_name)
                ax.set_ylabel('Reward')
            else:
                ax.hist(prop_values, bins=20, alpha=0.7)
                ax.set_xlabel(prop_name)
                ax.set_ylabel('Count')
            
            # Add target line if available
            if targets and prop_name in targets:
                target_val = targets[prop_name]
                ax.axvline(target_val, color='red', linestyle='--', alpha=0.7, label=f'Target: {target_val:.3f}')
                ax.legend()
        
        plt.tight_layout()
        
        # Convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        plt.close()
        
        return img
    
    def visualize_3d(
        self,
        smiles: str,
        title: Optional[str] = "3D Molecular Structure",
        spheres_3d: bool = True,
        bg: str = 'white',
        camera_elev: int = 0,
        camera_azim: int = 0,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> Optional[Any]:
        """Create advanced 3D visualization of a molecule with sphere rendering.
        
        Args:
            smiles: SMILES string of the molecule
            title: Plot title
            spheres_3d: If True, use 3D spheres; if False, use scatter points
            bg: Background color ('black' or 'white')
            camera_elev: Camera elevation angle
            camera_azim: Camera azimuth angle
            save_path: Optional path to save the figure
            show: Whether to display the plot
            
        Returns:
            Figure and axes if show=True, or save_path if save_path is provided
        """
        if not RDKIT_AVAILABLE or not MATPLOTLIB_AVAILABLE:
            return None
        
        try:
            # Convert SMILES to molecule
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"Invalid SMILES: {smiles}")
                return None
            
            # Add hydrogens
            mol = Chem.AddHs(mol)
            
            # Generate 3D coordinates
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
            
            # Get atom coordinates and symbols
            conf = mol.GetConformer()
            positions = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
            atom_types = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(mol.GetNumAtoms())]
            
            # Center the molecule
            positions = positions - positions.mean(axis=0, keepdims=True)
            
            # Create visualization
            return _plot_molecule_3d_advanced(
                positions, atom_types, title=title, spheres_3d=spheres_3d,
                bg=bg, camera_elev=camera_elev, camera_azim=camera_azim,
                save_path=save_path, show=show, dpi=self.dpi
            )
            
        except Exception as e:
            print(f"[ERROR] Visualizing molecule: {e}")
            import traceback
            traceback.print_exc()
            return None


# Atom colors and radii (for QM9 dataset: H, C, N, O, F)
ATOM_COLORS = {
    'H': '#FFFFFF',  # White
    'C': '#909090',  # Gray
    'N': '#3050F8',  # Blue
    'O': '#FF0D0D',  # Red
    'F': '#90E050',  # Green
}

ATOM_RADII = {
    'H': 0.31,
    'C': 0.76,
    'N': 0.71,
    'O': 0.66,
    'F': 0.57,
}


def _hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def _draw_sphere(ax: Any, x: float, y: float, z: float, size: float, color: tuple, alpha: float) -> None:
    """Draw a 3D sphere at the given position (matching reference implementation)."""
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    
    xs = size * np.outer(np.cos(u), np.sin(v))
    ys = size * np.outer(np.sin(u), np.sin(v)) * 0.8  # Correct for matplotlib
    zs = size * np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax.plot_surface(x + xs, y + ys, z + zs, rstride=2, cstride=2, 
                   color=color, linewidth=0, alpha=alpha)


def _plot_molecule_3d_advanced(
    positions: np.ndarray,
    atom_types: List[str],
    title: str = "3D Molecular Structure",
    spheres_3d: bool = True,
    bg: str = 'white',
    camera_elev: int = 0,
    camera_azim: int = 0,
    save_path: Optional[str] = None,
    show: bool = True,
    dpi: int = 120
) -> Optional[Any]:
    """Plot molecule with sphere rendering and bonds.
    
    Args:
        positions: numpy array of shape [N, 3] with atom positions
        atom_types: list of atom symbols (e.g., ['C', 'C', 'O', 'H', ...])
        title: Plot title
        spheres_3d: if True, use 3D spheres; if False, use scatter points
        bg: background color ('black' or 'white')
        camera_elev: Camera elevation angle
        camera_azim: Camera azimuth angle
        save_path: Optional path to save the figure
        show: Whether to display the plot
        dpi: DPI for saved images
    """
    if not MATPLOTLIB_AVAILABLE:
        return None
    
    hex_bg_color = '#FFFFFF' if bg == 'black' else '#666666'
    
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]
    
    # Get colors and radii for each atom
    colors = [_hex_to_rgb(ATOM_COLORS.get(atom, '#909090')) for atom in atom_types]
    radii = [ATOM_RADII.get(atom, 0.5) for atom in atom_types]
    areas = [1500 * r ** 2 for r in radii]
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('auto')
    ax.view_init(elev=camera_elev, azim=camera_azim)
    
    # Set background
    if bg == 'black':
        ax.set_facecolor((0, 0, 0))
    else:
        ax.set_facecolor((1, 1, 1))
    
    # Hide axes panes
    ax.xaxis.pane.set_alpha(0)
    ax.yaxis.pane.set_alpha(0)
    ax.zaxis.pane.set_alpha(0)
    ax._axis3don = False
    
    # Set axis line colors (compatible with different matplotlib versions)
    try:
        if bg == 'black':
            ax.w_xaxis.line.set_color("black")
        else:
            ax.w_xaxis.line.set_color("white")
    except AttributeError:
        # Newer matplotlib versions don't have w_xaxis
        # Use set_visible(False) instead since we already set _axis3don = False
        pass
    
    # Draw atoms
    if spheres_3d:
        for i, (xi, yi, zi, r, c) in enumerate(zip(x, y, z, radii, colors)):
            _draw_sphere(ax, float(xi), float(yi), float(zi), 0.7 * r, c, 1.0)
    else:
        ax.scatter(x, y, z, s=areas, alpha=0.9, c=colors, edgecolors='black', linewidths=0.5)
    
    # Draw bonds based on distance
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            p1 = np.array([x[i], y[i], z[i]])
            p2 = np.array([x[j], y[j], z[j]])
            dist = np.sqrt(np.sum((p1 - p2) ** 2))
            
            # Simple bond detection: draw if distance is reasonable for a bond
            max_bond_length = 2.0  # Maximum bond length in Angstroms
            if dist < max_bond_length:
                # Determine bond order based on distance
                if dist < 1.2:
                    linewidth = 3  # Triple or very short bond
                elif dist < 1.4:
                    linewidth = 2.5  # Double bond
                else:
                    linewidth = 2  # Single bond
                
                ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]],
                       linewidth=linewidth, c=hex_bg_color, alpha=1.0)
    
    # Set axis limits
    max_value = np.abs(positions).max()
    axis_lim = min(40, max(max_value / 1.5 + 0.3, 3.2))
    ax.set_xlim(-axis_lim, axis_lim)
    ax.set_ylim(-axis_lim, axis_lim)
    ax.set_zlim(-axis_lim, axis_lim)
    
    ax.set_xlabel('X (Å)', fontsize=12)
    ax.set_ylabel('Y (Å)', fontsize=12)
    ax.set_zlabel('Z (Å)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', 
                color='black' if bg == 'white' else 'white')
    
    if save_path is not None:
        dpi_used = 120 if spheres_3d else 50
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0, dpi=dpi_used, 
                   facecolor=bg, edgecolor='none')
        plt.close()
        return save_path
    elif show:
        plt.tight_layout()
        plt.show()
        return fig, ax
    else:
        return fig, ax


def visualize_molecule_3d(
    smiles: str,
    title: str = "3D Molecular Structure",
    spheres_3d: bool = True,
    bg: str = 'white',
    camera_elev: int = 0,
    camera_azim: int = 0,
    save_path: Optional[str] = None,
    show: bool = True
) -> Optional[Any]:
    """Create advanced 3D visualization of a molecule with sphere rendering.
    
    Args:
        smiles: SMILES string of the molecule
        title: Plot title
        spheres_3d: If True, use 3D spheres; if False, use scatter points
        bg: Background color ('black' or 'white')
        camera_elev: Camera elevation angle
        camera_azim: Camera azimuth angle
        save_path: Optional path to save the figure
        show: Whether to display the plot
        
    Returns:
        Figure and axes if show=True, or save_path if save_path is provided
    """
    if not RDKIT_AVAILABLE:
        print("[WARNING] RDKit required for 3D molecular visualization")
        return None
    
    try:
        # Convert SMILES to molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Invalid SMILES: {smiles}")
            return None
        
        # Add hydrogens
        mol = Chem.AddHs(mol)
        
        # Generate 3D coordinates
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        
        # Get atom coordinates and symbols
        conf = mol.GetConformer()
        positions = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
        atom_types = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(mol.GetNumAtoms())]
        
        # Center the molecule
        positions = positions - positions.mean(axis=0, keepdims=True)
        
        # Create visualization
        return _plot_molecule_3d_advanced(
            positions, atom_types, title=title, spheres_3d=spheres_3d,
            bg=bg, camera_elev=camera_elev, camera_azim=camera_azim,
            save_path=save_path, show=show
        )
        
    except Exception as e:
        print(f"[ERROR] Visualizing molecule: {e}")
        import traceback
        traceback.print_exc()
        return None


def _plot_molecule_qm9_style(
    ax: Any,
    positions: np.ndarray,
    atom_type: np.ndarray,
    alpha: float,
    spheres_3d: bool,
    hex_bg_color: str,
    dataset_info: Dict[str, Any]
) -> None:
    """Plot molecule on given axes (QM9/GEOM style, matching the reference implementation).
    
    This is the internal plotting function that matches the structure of the reference code.
    """
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]
    
    # Get colors and radii from dataset_info
    colors_dic = np.array(dataset_info['colors_dic'])
    radius_dic = np.array(dataset_info['radius_dic'])
    area_dic = 1500 * radius_dic ** 2
    
    areas = area_dic[atom_type]
    radii = radius_dic[atom_type]
    colors = colors_dic[atom_type]
    
    # Draw atoms
    if spheres_3d:
        for i, j, k, s, c in zip(x, y, z, radii, colors):
            # Ensure color is a tuple (RGB)
            if isinstance(c, (list, np.ndarray)):
                c_tuple = tuple(c) if len(c) == 3 else tuple(c[:3])
            else:
                c_tuple = c
            _draw_sphere(ax, float(i), float(j), float(k), 0.7 * s, c_tuple, alpha)
    else:
        # For scatter, convert colors to proper format for matplotlib
        # colors_dic is array of RGB tuples, so colors[atom_type] gives RGB arrays
        if isinstance(colors, np.ndarray) and colors.ndim == 2 and colors.shape[1] == 3:
            # Colors are RGB arrays - convert to list of tuples for matplotlib
            colors_list = [tuple(c) for c in colors]
            ax.scatter(x, y, z, s=areas, alpha=0.9 * alpha, c=colors_list, 
                      edgecolors='black', linewidths=0.5)
        else:
            ax.scatter(x, y, z, s=areas, alpha=0.9 * alpha, c=colors, 
                      edgecolors='black', linewidths=0.5)
    
    # Draw bonds using bond_analyze if available, otherwise use distance-based fallback
    try:
        from qm9 import bond_analyze
        use_bond_analyze = True
    except ImportError:
        use_bond_analyze = False
    
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            p1 = np.array([x[i], y[i], z[i]])
            p2 = np.array([x[j], y[j], z[j]])
            dist = np.sqrt(np.sum((p1 - p2) ** 2))
            
            if use_bond_analyze:
                # Use bond_analyze for proper bond detection
                atom1 = dataset_info['atom_decoder'][atom_type[i]]
                atom2 = dataset_info['atom_decoder'][atom_type[j]]
                s = sorted((atom_type[i], atom_type[j]))
                pair = (dataset_info['atom_decoder'][s[0]], dataset_info['atom_decoder'][s[1]])
                
                dataset_name = dataset_info.get('name', 'qm9')
                if 'qm9' in dataset_name:
                    draw_edge_int = bond_analyze.get_bond_order(atom1, atom2, dist)
                    line_width = (3 - 2) * 2 * 2
                elif dataset_name == 'geom':
                    draw_edge_int = bond_analyze.geom_predictor(pair, dist)
                    line_width = 2
                else:
                    draw_edge_int = 0
                    line_width = 2
                
                draw_edge = draw_edge_int > 0
                if draw_edge:
                    if draw_edge_int == 4:
                        linewidth_factor = 1.5
                    else:
                        linewidth_factor = 1
                    ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]],
                           linewidth=line_width * linewidth_factor,
                           c=hex_bg_color, alpha=alpha)
            else:
                # Fallback: simple distance-based bond detection
                max_bond_length = 2.0
                if dist < max_bond_length:
                    if dist < 1.2:
                        linewidth = 3
                    elif dist < 1.4:
                        linewidth = 2.5
                    else:
                        linewidth = 2
                    ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]],
                           linewidth=linewidth, c=hex_bg_color, alpha=alpha)


def plot_molecule_3d_from_coords(
    positions: Union[torch.Tensor, np.ndarray],
    atom_type: Union[torch.Tensor, np.ndarray],
    dataset_info: Dict[str, Any],
    camera_elev: int = 0,
    camera_azim: int = 0,
    save_path: Optional[str] = None,
    spheres_3d: bool = False,
    bg: str = 'black',
    alpha: float = 1.0,
    node_mask: Optional[Union[torch.Tensor, np.ndarray]] = None
) -> Optional[Any]:
    """Plot molecule from 3D coordinates and atom type indices (QM9/GEOM style).
    
    This function matches the structure and behavior of the reference QM9 visualization code.
    
    Args:
        positions: Tensor/array of shape [N, 3] with atom positions
        atom_type: Tensor/array of shape [N] with atom type indices (not one-hot)
        dataset_info: Dictionary with 'atom_decoder', 'colors_dic', 'radius_dic', 'name'
        camera_elev: Camera elevation angle
        camera_azim: Camera azimuth angle
        save_path: Optional path to save the figure
        spheres_3d: If True, use 3D spheres; if False, use scatter points
        bg: Background color ('black' or 'white')
        alpha: Transparency
        node_mask: Optional mask to filter atoms (shape [N] or [N, 1])
        
    Returns:
        save_path if save_path is provided, None otherwise
    """
    if not MATPLOTLIB_AVAILABLE:
        return None
    
    # Convert to torch tensor first (to match reference code style)
    if isinstance(positions, np.ndarray):
        positions = torch.from_numpy(positions).float()
    if isinstance(atom_type, np.ndarray):
        atom_type = torch.from_numpy(atom_type).long()
    
    # Handle node mask
    if node_mask is not None:
        if isinstance(node_mask, np.ndarray):
            node_mask = torch.from_numpy(node_mask).bool()
        if node_mask.ndim > 1:
            node_mask = node_mask.squeeze()
        positions = positions[node_mask]
        atom_type = atom_type[node_mask]
    
    # Center the molecule (matching reference: positions_centered = positions - positions.mean(dim=0, keepdim=True))
    positions = positions - positions.mean(dim=0, keepdim=True)
    
    # Convert to numpy for plotting
    positions_np = positions.detach().cpu().numpy()
    atom_type_np = atom_type.detach().cpu().numpy()
    
    # Create 3D plot (matching reference structure)
    black = (0, 0, 0)
    white = (1, 1, 1)
    hex_bg_color = '#FFFFFF' if bg == 'black' else '#666666'
    
    # Use larger figure size for better quality
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    ax.set_aspect('auto')
    ax.view_init(elev=camera_elev, azim=camera_azim)
    
    if bg == 'black':
        ax.set_facecolor(black)
    else:
        ax.set_facecolor(white)
    
    ax.xaxis.pane.set_alpha(0)
    ax.yaxis.pane.set_alpha(0)
    ax.zaxis.pane.set_alpha(0)
    ax._axis3don = False
    
    # Set axis line colors (compatible with different matplotlib versions)
    try:
        if bg == 'black':
            ax.w_xaxis.line.set_color("black")
        else:
            ax.w_xaxis.line.set_color("white")
    except AttributeError:
        # Newer matplotlib versions don't have w_xaxis
        pass
    
    # Plot molecule (matching reference: plot_molecule(ax, positions, atom_type, alpha, spheres_3d, hex_bg_color, dataset_info))
    _plot_molecule_qm9_style(ax, positions_np, atom_type_np, alpha, spheres_3d, hex_bg_color, dataset_info)
    
    # Set axis limits (matching reference: using positions.abs().max().item())
    dataset_name = dataset_info.get('name', 'qm9')
    if 'qm9' in dataset_name or dataset_name == 'geom':
        max_value = positions.abs().max().item()
        axis_lim = min(40, max(max_value / 1.5 + 0.3, 3.2))
        ax.set_xlim(-axis_lim, axis_lim)
        ax.set_ylim(-axis_lim, axis_lim)
        ax.set_zlim(-axis_lim, axis_lim)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    
    dpi = 120 if spheres_3d else 50
    
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0, dpi=dpi, 
                   facecolor=bg, edgecolor='none')
        if spheres_3d:
            try:
                import imageio
                img = imageio.imread(save_path)
                img_brighter = np.clip(img * 1.4, 0, 255).astype('uint8')
                imageio.imsave(save_path, img_brighter)
            except ImportError:
                pass
        plt.close()
        return save_path
    else:
        plt.tight_layout()
        plt.show()
        plt.close()
        return None


def visualize_best_molecules(
    molecules: List[Any],
    rewards: List[float],
    properties: Optional[Dict[str, List[float]]] = None,
    top_k: int = 5,
    save_path: Optional[str] = None
) -> Optional["PILImageType"]:
    """Visualize the best molecules from optimization results."""
    visualizer = MoleculeVisualizer()
    
    # Get top-k molecules
    top_indices = np.argsort(rewards)[-top_k:][::-1]
    top_molecules = [molecules[i] for i in top_indices]
    top_rewards = [rewards[i] for i in top_indices]
    top_properties = None
    if properties is not None:
        top_properties = {k: [v[i] for i in top_indices] for k, v in properties.items()}
    
    # Create visualization
    img = visualizer.molecules_to_grid(
        top_molecules, 
        top_rewards, 
        top_properties,
        max_molecules=top_k
    )
    
    if img is not None and save_path is not None:
        img.save(save_path)
    
    return img
