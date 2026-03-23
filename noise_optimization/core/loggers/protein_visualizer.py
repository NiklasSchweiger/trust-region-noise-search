"""Protein structure visualization utilities for the core framework.

This module provides visualization of protein structures similar to the style
used in the Proteina paper - ribbon diagrams showing secondary structure elements
(alpha-helices, beta-sheets, loops).
"""

from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING
import os
import tempfile
import torch
import numpy as np
import io

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    Image = None  # type: ignore
    PIL_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

if TYPE_CHECKING:
    from PIL.Image import Image as PILImageType


class ProteinVisualizer:
    """Utility class for visualizing protein 3D structures.
    
    Provides visualization similar to Proteina paper style - ribbon diagrams
    showing secondary structure (alpha-helices, beta-sheets, loops).
    
    Currently uses matplotlib 3D plotting. In the future, could be enhanced
    to use PyMOL headless rendering for publication-quality ribbon diagrams.
    """
    
    def __init__(self, image_size: tuple = (800, 600), dpi: int = 100, style: str = "backbone"):
        """Initialize visualizer.
        
        Args:
            image_size: Output image size (width, height)
            dpi: Resolution (dots per inch)
            style: Visualization style - "backbone" (simple CA trace) or "ribbon" (future: secondary structure)
        """
        self.image_size = image_size
        self.dpi = dpi
        self.style = style
        
        # Check if PyMOL is available for better ribbon diagrams (like Proteina paper)
        self._pymol_available = self._check_pymol()
    
    def _check_pymol(self) -> bool:
        """Check if PyMOL is available for high-quality ribbon rendering."""
        try:
            import pymol
            pymol.finish_launching(['pymol', '-cq'])  # Launch headless
            return True
        except:
            return False
    
    def atom37_to_image(
        self, 
        atom37: Any, 
        title: Optional[str] = None,
        show_backbone: bool = True,
        show_atoms: bool = False,
        color_by_residue: bool = True
    ) -> Optional["PILImageType"]:
        """Convert atom37 protein structure to 3D visualization image.
        
        Args:
            atom37: Protein structure as atom37 tensor/array [N, 37, 3] or [N, 3] (CA-only)
            title: Title for the visualization
            show_backbone: Whether to show backbone trace (CA atoms connected)
            show_atoms: Whether to show individual atoms (can be slow for large proteins)
            color_by_residue: Whether to color backbone by residue index
        
        Returns:
            PIL Image of the 3D protein structure, or None if visualization fails
        """
        if not MATPLOTLIB_AVAILABLE or not PIL_AVAILABLE:
            return None
        
        try:
            # Convert to numpy
            if isinstance(atom37, torch.Tensor):
                coords = atom37.detach().cpu().numpy()
            else:
                coords = np.array(atom37)
            
            # Handle different input shapes
            if coords.ndim == 3:
                # [N, 37, 3] - extract CA atoms (index 1)
                if coords.shape[1] >= 2:
                    ca_coords = coords[:, 1, :]  # CA atoms
                else:
                    ca_coords = coords[:, 0, :]  # Fallback to first atom
            elif coords.ndim == 2 and coords.shape[1] == 3:
                # [N, 3] - already CA coordinates
                ca_coords = coords
            else:
                print(f"[WARNING] Unexpected atom37 shape {coords.shape}")
                return None
            
            # Remove any NaN or Inf coordinates
            valid_mask = np.isfinite(ca_coords).all(axis=1)
            if not valid_mask.any():
                print("[WARNING] No valid coordinates in protein structure")
                return None
            ca_coords = ca_coords[valid_mask]
            
            if len(ca_coords) == 0:
                return None
            
            # Create 3D plot
            fig = plt.figure(figsize=(self.image_size[0]/self.dpi, self.image_size[1]/self.dpi), dpi=self.dpi)
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot backbone trace (CA atoms connected)
            if show_backbone and len(ca_coords) > 1:
                if color_by_residue:
                    # Color by residue index (rainbow gradient)
                    n_res = len(ca_coords)
                    colors = plt.cm.rainbow(np.linspace(0, 1, n_res))
                    for i in range(len(ca_coords) - 1):
                        ax.plot(
                            [ca_coords[i, 0], ca_coords[i+1, 0]],
                            [ca_coords[i, 1], ca_coords[i+1, 1]],
                            [ca_coords[i, 2], ca_coords[i+1, 2]],
                            color=colors[i], linewidth=1.5, alpha=0.8
                        )
                else:
                    # Single color
                    ax.plot(
                        ca_coords[:, 0], ca_coords[:, 1], ca_coords[:, 2],
                        'b-', linewidth=1.5, alpha=0.8, label='Backbone'
                    )
                
                # Plot CA atoms as points
                if color_by_residue:
                    n_res = len(ca_coords)
                    colors = plt.cm.rainbow(np.linspace(0, 1, n_res))
                    ax.scatter(
                        ca_coords[:, 0], ca_coords[:, 1], ca_coords[:, 2],
                        c=colors, s=20, alpha=0.9
                    )
                else:
                    ax.scatter(
                        ca_coords[:, 0], ca_coords[:, 1], ca_coords[:, 2],
                        c='blue', s=20, alpha=0.9
                    )
            
            # Set labels and title
            ax.set_xlabel('X (Å)', fontsize=8)
            ax.set_ylabel('Y (Å)', fontsize=8)
            ax.set_zlabel('Z (Å)', fontsize=8)
            
            if title:
                ax.set_title(title, fontsize=10, pad=10)
            
            # Set equal aspect ratio for better 3D visualization
            max_range = np.array([
                ca_coords[:, 0].max() - ca_coords[:, 0].min(),
                ca_coords[:, 1].max() - ca_coords[:, 1].min(),
                ca_coords[:, 2].max() - ca_coords[:, 2].min()
            ]).max() / 2.0
            
            mid_x = (ca_coords[:, 0].max() + ca_coords[:, 0].min()) * 0.5
            mid_y = (ca_coords[:, 1].max() + ca_coords[:, 1].min()) * 0.5
            mid_z = (ca_coords[:, 2].max() + ca_coords[:, 2].min()) * 0.5
            
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            
            # Adjust viewing angle for better visibility
            ax.view_init(elev=20, azim=45)
            
            plt.tight_layout()
            
            # Convert to PIL Image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight')
            buf.seek(0)
            img = Image.open(buf)
            plt.close(fig)
            
            return img
            
        except Exception as e:
            print(f"[WARNING] Failed to create protein visualization: {e}")
            if MATPLOTLIB_AVAILABLE:
                plt.close('all')
            return None


def create_protein_image(
    atom37: Any,
    title: Optional[str] = None,
    image_size: tuple = (800, 600),
    dpi: int = 100
) -> Optional["PILImageType"]:
    """Convenience function to create a protein structure image.
    
    Args:
        atom37: Protein structure as atom37 tensor/array
        title: Title for the visualization
        image_size: Size of the output image (width, height)
        dpi: Resolution (dots per inch)
    
    Returns:
        PIL Image or None if visualization fails
    """
    visualizer = ProteinVisualizer(image_size=image_size, dpi=dpi)
    return visualizer.atom37_to_image(atom37, title=title)
