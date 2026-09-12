"""
Spectral water indices computation module for optical & SAR remote sensing data.
Implements Normalized Difference Water Index (NDWI), Modified NDWI (MNDWI),
Automated Water Extraction Index (AWEI), and SAR backscatter ratios.
"""

import numpy as np
from typing import Union

def compute_ndwi(green_band: np.ndarray, nir_band: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """
    McFeeters Normalized Difference Water Index (NDWI).
    NDWI = (Green - NIR) / (Green + NIR)
    
    Water bodies generally show positive values (NDWI > 0).
    """
    green = green_band.astype(np.float32)
    nir = nir_band.astype(np.float32)
    numerator = green - nir
    denominator = green + nir + eps
    ndwi = numerator / denominator
    return np.clip(ndwi, -1.0, 1.0)

def compute_mndwi(green_band: np.ndarray, swir_band: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """
    Xu Modified Normalized Difference Water Index (MNDWI).
    MNDWI = (Green - SWIR1) / (Green + SWIR1)
    
    Superior at distinguishing open water from built-up surfaces and bare mountain rock.
    """
    green = green_band.astype(np.float32)
    swir = swir_band.astype(np.float32)
    numerator = green - swir
    denominator = green + swir + eps
    mndwi = numerator / denominator
    return np.clip(mndwi, -1.0, 1.0)

def compute_ndsi(green_band: np.ndarray, swir_band: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """
    Normalized Difference Snow Index (NDSI).
    NDSI = (Green - SWIR1) / (Green + SWIR1)
    """
    return compute_mndwi(green_band, swir_band, eps)

def compute_awei_sh(
    blue_band: np.ndarray,
    green_band: np.ndarray,
    nir_band: np.ndarray,
    swir1_band: np.ndarray,
    swir2_band: np.ndarray
) -> np.ndarray:
    """
    Automated Water Extraction Index with Shadow handling (AWEI_sh).
    AWEI_sh = Blue + 2.5 * Green - 1.5 * (NIR + SWIR1) - 0.25 * SWIR2
    
    Designed to eliminate dark mountain and cloud shadows in high-relief terrain.
    """
    b = blue_band.astype(np.float32)
    g = green_band.astype(np.float32)
    nir = nir_band.astype(np.float32)
    swir1 = swir1_band.astype(np.float32)
    swir2 = swir2_band.astype(np.float32)
    
    awei = b + 2.5 * g - 1.5 * (nir + swir1) - 0.25 * swir2
    return awei

def compute_sar_ratio(vv_band: np.ndarray, vh_band: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """
    Calculates SAR polarization ratio (VH / VV) in linear scale or dB difference (VH_dB - VV_dB).
    Smooth water surface reflects specularly, producing very low backscatter in both VV and VH.
    """
    vv = np.maximum(vv_band.astype(np.float32), eps)
    vh = np.maximum(vh_band.astype(np.float32), eps)
    return vh / vv

def approximate_ndwi_from_rgb(rgb_image: np.ndarray) -> np.ndarray:
    """
    Approximates a normalized water contrast index from standard 3-channel RGB image (H, W, 3).
    Formula: (Blue + Green - 2 * Red) / (Blue + Green + 2 * Red + eps)
    """
    r = rgb_image[..., 0].astype(np.float32)
    g = rgb_image[..., 1].astype(np.float32)
    b = rgb_image[..., 2].astype(np.float32)
    
    water_score = (b + g - 2.0 * r) / (b + g + 2.0 * r + 1e-7)
    return np.clip(water_score, -1.0, 1.0)
