"""
Cloud, shadow, and snow masking utilities for optical satellite data.
Handles Sentinel-2 Scene Classification Layer (SCL) filtering and fallback threshold masking.
"""

import numpy as np
from typing import Tuple, Optional
from scipy.ndimage import binary_dilation

def create_cloud_shadow_mask_scl(scl_band: np.ndarray, dilate_pixels: int = 3) -> np.ndarray:
    """
    Generate boolean mask (True = Clear/Valid, False = Cloud/Shadow/Invalid)
    from Sentinel-2 Scene Classification Layer (SCL).
    
    SCL Class mapping:
      3: Cloud shadows
      8: Cloud medium probability
      9: Cloud high probability
      10: Thin cirrus
      11: Snow / ice
    """
    invalid_mask = np.isin(scl_band, [3, 8, 9, 10])
    if dilate_pixels > 0:
        struct = np.ones((dilate_pixels * 2 + 1, dilate_pixels * 2 + 1), dtype=bool)
        invalid_mask = binary_dilation(invalid_mask, structure=struct)
    
    valid_mask = ~invalid_mask
    return valid_mask

def create_cloud_mask_threshold(
    blue_band: np.ndarray,
    swir_band: Optional[np.ndarray] = None,
    blue_thresh: float = 0.28,
    swir_thresh: float = 0.16
) -> np.ndarray:
    """
    Threshold-based cloud detector when SCL band is unavailable.
    Clouds typically have high reflectance in both Blue and SWIR bands.
    """
    # Normalize if values are 0-10000 (reflectance scale)
    b = blue_band / 10000.0 if np.nanmax(blue_band) > 10.0 else blue_band
    is_cloud = b > blue_thresh
    
    if swir_band is not None:
        s = swir_band / 10000.0 if np.nanmax(swir_band) > 10.0 else swir_band
        is_cloud = is_cloud & (s > swir_thresh)
        
    return ~is_cloud  # Returns True for valid/clear pixels

def apply_mask_to_multiband(
    data: np.ndarray,
    valid_mask: np.ndarray,
    fill_value: float = np.nan
) -> np.ndarray:
    """
    Applies 2D valid_mask to multi-band array (Channels, Height, Width).
    """
    masked_data = data.copy().astype(np.float32)
    if masked_data.ndim == 3:
        for c in range(masked_data.shape[0]):
            masked_data[c, ~valid_mask] = fill_value
    elif masked_data.ndim == 2:
        masked_data[~valid_mask] = fill_value
    return masked_data
