"""
SAR speckle reduction filter algorithms (Lee Filter, Enhanced Lee, Box filter).
Designed to preserve lake edges while reducing granular speckle noise in Sentinel-1 GRD imagery.
"""

import numpy as np
from scipy.ndimage import uniform_filter

def apply_lee_filter(sar_image: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Applies standard Lee speckle filter on SAR 2D intensity raster.
    
    Parameters:
        sar_image: 2D numpy array with backscatter values (linear amplitude/intensity).
        window_size: Odd integer size of sliding kernel (default: 5).
    """
    img = sar_image.astype(np.float32)
    
    # Calculate local mean
    mean = uniform_filter(img, size=window_size)
    
    # Calculate local mean of squares
    mean_sq = uniform_filter(img ** 2, size=window_size)
    
    # Calculate local variance
    variance = np.maximum(mean_sq - mean ** 2, 0)
    
    # Overall image noise variance
    overall_variance = np.var(img)
    
    # Weighting coefficient
    weights = variance / (variance + overall_variance + 1e-7)
    
    # Filtered output
    filtered = mean + weights * (img - mean)
    return filtered

def apply_box_filter(sar_image: np.ndarray, window_size: int = 3) -> np.ndarray:
    """
    Fast mean box filter for SAR rasters.
    """
    return uniform_filter(sar_image.astype(np.float32), size=window_size)
