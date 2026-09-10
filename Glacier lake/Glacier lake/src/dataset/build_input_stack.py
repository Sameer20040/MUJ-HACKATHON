"""
Multi-band input stack builder for glacial lake segmentation.
Stacks Optical (B2, B3, B4, B8, B11), Spectral Water Indices (NDWI/MNDWI),
and DEM Topography into a unified (7, H, W) float32 tensor.
Also includes synthetic benchmark scene generator for end-to-end demonstrations.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
import rasterio
from rasterio.transform import from_bounds
import cv2

from ..preprocessing.compute_indices import compute_ndwi, compute_mndwi
from ..utils.config import config

def build_multichannel_stack(
    blue_band: np.ndarray,
    green_band: np.ndarray,
    red_band: np.ndarray,
    nir_band: np.ndarray,
    swir1_band: np.ndarray,
    dem_band: np.ndarray,
    calculate_indices: bool = True
) -> np.ndarray:
    """
    Assembles a 7-channel input stack:
    [0: Blue, 1: Green, 2: Red, 3: NIR, 4: SWIR1, 5: NDWI, 6: DEM]
    """
    h, w = blue_band.shape[-2:]
    
    # Rescale integer digital numbers (0-10000) to float reflectance (0.0 - 1.0)
    def to_float(arr):
        a = arr.astype(np.float32)
        if np.nanmax(a) > 10.0:
            a = a / 10000.0
        return np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0)

    b2 = to_float(blue_band)
    b3 = to_float(green_band)
    b4 = to_float(red_band)
    b8 = to_float(nir_band)
    b11 = to_float(swir1_band)
    
    # Normalize DEM (typical Himalayan range: 2500m to 6500m)
    dem = dem_band.astype(np.float32)
    dem_norm = (dem - 3000.0) / 2000.0
    
    if calculate_indices:
        ndwi = compute_ndwi(b3, b8)
    else:
        ndwi = np.zeros((h, w), dtype=np.float32)

    stack = np.stack([b2, b3, b4, b8, b11, ndwi, dem_norm], axis=0).astype(np.float32)
    return stack

def create_synthetic_glacial_scene(
    height: int = 1024,
    width: int = 1024,
    seed: int = 42,
    epoch: int = 2016
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Generates a realistic multi-spectral Himalayan glacial valley scene
    with glacier tongues, moraines, bedrock, snow peaks, and lakes.
    
    Returns:
        stack: (7, height, width) multi-band array
        lake_mask: (height, width) binary ground truth mask (1 = lake, 0 = background)
        profile: Rasterio metadata profile with geotransform
    """
    np.random.seed(seed)
    
    # 1. Base Elevation Grid (DEM) - valley terrain with ridgelines
    x = np.linspace(-3, 3, width)
    y = np.linspace(-3, 3, height)
    xx, yy = np.meshgrid(x, y)
    
    valley = np.sin(xx * 1.5) * np.cos(yy * 1.2) * 800 + (xx**2 + yy**2) * 200 + 4200
    noise = np.random.normal(0, 30, (height, width))
    dem = valley + noise
    
    # 2. Base terrain surface reflectance (Rock & Moraine)
    rock_red = 0.22 + 0.05 * np.sin(xx * 4) + np.random.normal(0, 0.02, (height, width))
    rock_green = 0.20 + 0.04 * np.sin(xx * 4) + np.random.normal(0, 0.02, (height, width))
    rock_blue = 0.18 + 0.04 * np.sin(xx * 4) + np.random.normal(0, 0.02, (height, width))
    rock_nir = 0.25 + 0.05 * np.cos(yy * 4) + np.random.normal(0, 0.02, (height, width))
    rock_swir = 0.28 + 0.06 * np.cos(yy * 4) + np.random.normal(0, 0.02, (height, width))
    
    # 3. Snow and Ice on high peaks (DEM > 4800)
    snow_mask = dem > 4700
    rock_red[snow_mask] = 0.85
    rock_green[snow_mask] = 0.90
    rock_blue[snow_mask] = 0.95
    rock_nir[snow_mask] = 0.70
    rock_swir[snow_mask] = 0.15 # Snow absorbs SWIR
    
    # 4. Glacial Lake Locations & Temporal Dynamics
    lake_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Define lake centers and base radiuses
    # (x_center, y_center, radius, type)
    lakes_def = [
        # Survived Lake 1 (Terminus lake - grows in 2022)
        (350, 400, 45 if epoch == 2016 else 65),
        # Survived Lake 2 (Moraine dammed - stable)
        (720, 650, 35 if epoch == 2016 else 38),
        # Survived Lake 3 (Supraglacial)
        (520, 320, 25 if epoch == 2016 else 30),
    ]
    
    if epoch == 2016:
        # Drained Lake (Present in 2016, drains before 2022)
        lakes_def.append((260, 780, 40))
        lakes_def.append((600, 850, 28))
    else: # 2022
        # Newly Formed Lakes (Formed due to retreat by 2022)
        lakes_def.append((480, 220, 32))
        lakes_def.append((800, 420, 35))
        lakes_def.append((380, 600, 24))

    for (cx, cy, rad) in lakes_def:
        cv2.circle(lake_mask, (cx, cy), rad, 1, -1)
        # Add irregular morphology
        dist = np.sqrt((xx * (width/6) - (cx - width/2))**2 + (yy * (height/6) - (cy - height/2))**2)
    
    # Apply lake spectral signatures (Water has high Blue/Green, low NIR/SWIR, high NDWI)
    is_lake = lake_mask == 1
    rock_blue[is_lake] = 0.42 + np.random.normal(0, 0.01, np.sum(is_lake))
    rock_green[is_lake] = 0.38 + np.random.normal(0, 0.01, np.sum(is_lake))
    rock_red[is_lake] = 0.15 + np.random.normal(0, 0.01, np.sum(is_lake))
    rock_nir[is_lake] = 0.04 + np.random.normal(0, 0.005, np.sum(is_lake))
    rock_swir[is_lake] = 0.02 + np.random.normal(0, 0.005, np.sum(is_lake))

    # Calculate NDWI
    ndwi = compute_ndwi(rock_green, rock_nir)
    
    # Normalize DEM
    dem_norm = (dem - 3000.0) / 2000.0
    
    stack = np.stack([
        np.clip(rock_blue, 0, 1),
        np.clip(rock_green, 0, 1),
        np.clip(rock_red, 0, 1),
        np.clip(rock_nir, 0, 1),
        np.clip(rock_swir, 0, 1),
        ndwi,
        dem_norm
    ], axis=0).astype(np.float32)

    # Geospatial profile in Karakoram/Himalayan region (Hunza / Gilgit coords around 74.5°E, 36.5°N)
    transform = from_bounds(74.45, 36.40, 74.85, 36.75, width, height)
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "nodata": None,
        "width": width,
        "height": height,
        "count": 7,
        "crs": "EPSG:4326",
        "transform": transform
    }
    
    return stack, lake_mask, profile
