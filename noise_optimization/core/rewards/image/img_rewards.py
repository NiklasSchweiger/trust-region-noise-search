from __future__ import annotations

import io
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

try:
    from PIL.Image import Image
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "Pillow is required for image metric rewards. Install Pillow to enable "
        "brightness/contrast/etc rewards."
    ) from exc

from ..base import ImageRewardFunction, RewardFunctionRegistry


def _handle_input(img: Union[torch.Tensor, np.ndarray, Image]):
    if isinstance(img, torch.Tensor):
        if img.dim() == 3:
            return [img]
        if img.dim() == 4:
            return [img[i] for i in range(img.shape[0])]
    elif isinstance(img, list):
        return img
    else:
        return [img]


class BrightnessReward(ImageRewardFunction):
    def __init__(self, device: Optional[str] = None, config: Optional[Any] = None):
        # Support both direct device parameter and RewardConfig object
        if config is not None:
            device = config.device if hasattr(config, 'device') else device
        super().__init__("brightness", device, expects_pil=False)

    def evaluate(self, candidates: Any, context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        imgs = _handle_input(candidates)
        vals = []
        for t in imgs:
            vals.append(float(torch.clamp((t / 2 + 0.5), 0, 1).mean().item()))
        return torch.tensor(vals, device=self.device, dtype=torch.float32)

    def get_output_range(self) -> tuple[float, float]:
        return (0.0, 1.0)


class ContrastReward(ImageRewardFunction):
    def __init__(self, device: Optional[str] = None, config: Optional[Any] = None):
        # Support both direct device parameter and RewardConfig object
        if config is not None:
            device = config.device if hasattr(config, 'device') else device
        super().__init__("contrast", device, expects_pil=False)

    def evaluate(self, candidates: Any, context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        imgs = _handle_input(candidates)
        vals = []
        for t in imgs:
            x = torch.clamp((t / 2 + 0.5), 0, 1)
            vals.append(float(x.view(x.shape[0], -1).std(dim=1).mean().item()))
        return torch.tensor(vals, device=self.device, dtype=torch.float32)

    def get_output_range(self) -> tuple[float, float]:
        return (0.0, 1.0)


class JPEGCompressibilityReward(ImageRewardFunction):
    def __init__(self, device: Optional[str] = None, config: Optional[Any] = None):
        # Support both direct device parameter and RewardConfig object
        if config is not None:
            device = config.device if hasattr(config, 'device') else device
        super().__init__("jpeg_compressibility", device, expects_pil=True)

    def evaluate(self, candidates: Any, context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        imgs = _handle_input(candidates)
        vals = []
        for t in imgs:
            # Convert tensors to PIL if needed
            if isinstance(t, torch.Tensor):
                t = self._convert_to_expected_format(t)
            if isinstance(t, list):
                t = t[0]
            if not isinstance(t, Image):
                # As a last resort try numpy conversion
                try:
                    from diffusers.utils import numpy_to_pil
                    if hasattr(t, "numpy"):
                        t = numpy_to_pil(t.numpy())[0]
                except Exception:
                    pass
            if not isinstance(t, Image):
                continue
            buffer = io.BytesIO()
            t.save(buffer, format="JPEG", quality=95)
            size_kb = buffer.tell() / 1000.0
            buffer.close()
            # Negate to minimize file size (smaller files = higher reward)
            vals.append(-size_kb)
        if not vals:
            vals = [0.0]
        return torch.tensor(vals, device=self.device, dtype=torch.float32)

    def get_output_range(self) -> tuple[float, float]:
        # Negated range: smaller files give higher (less negative) rewards
        return (-1e6, 0.0)


class RednessReward(ImageRewardFunction):
    def __init__(self, device: Optional[str] = None, config: Optional[Any] = None):
        # Support both direct device parameter and RewardConfig object
        if config is not None:
            device = config.device if hasattr(config, 'device') else device
        super().__init__("redness", device, expects_pil=False)

    def evaluate(self, candidates: Any, context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        imgs = _handle_input(candidates)
        vals = []
        for t in imgs:
            x = torch.clamp((t / 2 + 0.5), 0, 1)
            # Calculate redness as the ratio of red channel to total intensity
            red_channel = x[0] if x.shape[0] >= 3 else x[0]
            total_intensity = x.sum(dim=0) if x.shape[0] >= 3 else x[0]
            redness = (red_channel / (total_intensity + 1e-8)).mean().item()
            vals.append(redness)
        return torch.tensor(vals, device=self.device, dtype=torch.float32)

    def get_output_range(self) -> tuple[float, float]:
        return (0.0, 1.0)


class SharpnessReward(ImageRewardFunction):
    def __init__(self, device: Optional[str] = None, config: Optional[Any] = None):
        # Support both direct device parameter and RewardConfig object
        if config is not None:
            device = config.device if hasattr(config, 'device') else device
        super().__init__("sharpness", device, expects_pil=False)

    def evaluate(self, candidates: Any, context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        imgs = _handle_input(candidates)
        vals = []
        for t in imgs:
            x = torch.clamp((t / 2 + 0.5), 0, 1)
            # Convert to grayscale if needed
            if x.shape[0] == 3:
                gray = 0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]
            else:
                gray = x[0]
            
            # Apply Laplacian filter for sharpness detection
            # Simple 3x3 Laplacian kernel
            kernel = torch.tensor([
                [0.0, -1.0, 0.0],
                [-1.0, 4.0, -1.0],
                [0.0, -1.0, 0.0]
            ], device=x.device, dtype=x.dtype)
            
            # Apply convolution
            gray = gray.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
            kernel = kernel.unsqueeze(0).unsqueeze(0)
            
            # Pad to maintain size
            gray_padded = torch.nn.functional.pad(gray, (1, 1, 1, 1), mode='reflect')
            laplacian = torch.nn.functional.conv2d(gray_padded, kernel)
            
            # Sharpness is the variance of the Laplacian
            sharpness = laplacian.var().item()
            vals.append(sharpness)
        return torch.tensor(vals, device=self.device, dtype=torch.float32)

    def get_output_range(self) -> tuple[float, float]:
        return (0.0, 1.0)


class SaturationReward(ImageRewardFunction):
    def __init__(self, device: Optional[str] = None, config: Optional[Any] = None):
        # Support both direct device parameter and RewardConfig object
        if config is not None:
            device = config.device if hasattr(config, 'device') else device
        super().__init__("saturation", device, expects_pil=False)

    def evaluate(self, candidates: Any, context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        imgs = _handle_input(candidates)
        vals = []
        for t in imgs:
            x = torch.clamp((t / 2 + 0.5), 0, 1)
            # Convert RGB to HSV
            if x.shape[0] >= 3:
                r, g, b = x[0], x[1], x[2]
                
                # Calculate saturation
                max_val = torch.maximum(torch.maximum(r, g), b)
                min_val = torch.minimum(torch.minimum(r, g), b)
                delta = max_val - min_val
                
                # Saturation formula: S = delta / (max + epsilon)
                saturation = delta / (max_val + 1e-8)
                avg_saturation = saturation.mean().item()
            else:
                avg_saturation = 0.0
            vals.append(avg_saturation)
        return torch.tensor(vals, device=self.device, dtype=torch.float32)

    def get_output_range(self) -> tuple[float, float]:
        return (0.0, 1.0)


class RadialBrightnessReward(ImageRewardFunction):
    def __init__(self, device: Optional[str] = None, config: Optional[Any] = None):
        # Support both direct device parameter and RewardConfig object
        if config is not None:
            device = config.device if hasattr(config, 'device') else device
        super().__init__("radial_brightness", device, expects_pil=False)

    def evaluate(self, candidates: Any, context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        imgs = _handle_input(candidates)
        vals = []
        for t in imgs:
            x = torch.clamp((t / 2 + 0.5), 0, 1)
            # Convert to grayscale if needed
            if x.shape[0] == 3:
                gray = 0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]
            else:
                gray = x[0]
            
            # Create radial mask (brighter in center)
            h, w = gray.shape
            y_coords, x_coords = torch.meshgrid(
                torch.arange(h, device=gray.device, dtype=gray.dtype),
                torch.arange(w, device=gray.device, dtype=gray.dtype),
                indexing='ij'
            )
            
            # Distance from center
            center_y, center_x = h / 2, w / 2
            dist_from_center = torch.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
            max_dist = torch.sqrt(torch.tensor(center_x**2 + center_y**2, device=gray.device))
            
            # Create radial weight (1 at center, 0 at edges)
            radial_weight = 1.0 - (dist_from_center / (max_dist + 1e-8))
            radial_weight = torch.clamp(radial_weight, 0, 1)
            
            # Weighted brightness
            weighted_brightness = (gray * radial_weight).sum() / (radial_weight.sum() + 1e-8)
            vals.append(weighted_brightness.item())
        return torch.tensor(vals, device=self.device, dtype=torch.float32)

    def get_output_range(self) -> tuple[float, float]:
        return (0.0, 1.0)


def register_image_metric_rewards() -> None:
    RewardFunctionRegistry.register("brightness", BrightnessReward)
    RewardFunctionRegistry.register("contrast", ContrastReward)
    RewardFunctionRegistry.register("jpeg_compressibility", JPEGCompressibilityReward)
    RewardFunctionRegistry.register("redness", RednessReward)
    RewardFunctionRegistry.register("sharpness", SharpnessReward)
    RewardFunctionRegistry.register("saturation", SaturationReward)
    RewardFunctionRegistry.register("radial_brightness", RadialBrightnessReward)


