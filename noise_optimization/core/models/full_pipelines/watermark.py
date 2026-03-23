"""Watermark module for Stable Diffusion XL pipeline.

This module provides watermarking functionality for SDXL images.
It uses the invisible-watermark library when available, otherwise provides a no-op implementation.
"""

from diffusers.utils import is_invisible_watermark_available

if is_invisible_watermark_available():
    from diffusers.pipelines.stable_diffusion_xl.watermark import StableDiffusionXLWatermarker
else:
    # Fallback: no-op watermarker if library is not available
    class StableDiffusionXLWatermarker:
        """No-op watermarker when invisible-watermark library is not available."""
        
        def apply_watermark(self, image):
            """Apply watermark to image (no-op when library is not available)."""
            return image

