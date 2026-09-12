"""
Scene-level inference engine.
Performs sliding-window tiled predictions with Gaussian edge weighting
to eliminate boundary stitching artifacts on arbitrarily large satellite scenes.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union, Any
import numpy as np
import torch
import cv2

from ..segmentation.unet_model import LakeUNet, UNet
from ..dataset.normalize import ChannelNormalizer
from ..postprocessing.vectorize import mask_to_geodataframe
from ..postprocessing.smooth_boundaries import clean_binary_mask
from ..preprocessing.cloud_mask import create_cloud_mask_threshold
from .load_upload import load_input_image, prepare_tensor_from_upload
from .overlay import create_boundary_overlay
from ..utils.config import config

def create_gaussian_weight_window(size: int = 256, sigma: float = 64.0) -> np.ndarray:
    """Creates a 2D Gaussian weight window to smoothly blend overlapping tiles."""
    x = np.linspace(-size / 2, size / 2, size)
    gauss_1d = np.exp(-0.5 * (x / sigma) ** 2)
    gauss_2d = np.outer(gauss_1d, gauss_1d)
    return (gauss_2d / np.max(gauss_2d)).astype(np.float32)

def predict_large_scene_tiled(
    input_tensor: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    tile_size: int = 256,
    stride: int = 192,
    batch_size: int = 8
) -> np.ndarray:
    """
    Sliding-window tiled inference with Gaussian blending across entire scene.
    
    Parameters:
        input_tensor: (C, H, W) normalized numpy array
        model: PyTorch segmentation model
        device: torch device
        tile_size: Patch size (256)
        stride: Stride between tiles (< tile_size ensures overlap)
    """
    c, h, w = input_tensor.shape
    model.eval()

    prob_map = np.zeros((h, w), dtype=np.float32)
    weight_map = np.zeros((h, w), dtype=np.float32)
    gauss_win = create_gaussian_weight_window(tile_size)

    # Collect all tile coordinates
    tile_coords = []
    patches = []

    for r in range(0, h, stride):
        for col in range(0, w, stride):
            r_start = min(r, h - tile_size) if h >= tile_size else 0
            c_start = min(col, w - tile_size) if w >= tile_size else 0
            r_end = r_start + tile_size
            c_end = c_start + tile_size

            patch = input_tensor[:, r_start:r_end, c_start:c_end]
            
            # Handle pad if scene smaller than tile_size
            if patch.shape[1] != tile_size or patch.shape[2] != tile_size:
                pad_h = tile_size - patch.shape[1]
                pad_w = tile_size - patch.shape[2]
                patch = np.pad(patch, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')

            tile_coords.append((r_start, c_start))
            patches.append(patch)

    # Run batched inference
    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            batch_patches = patches[i:i + batch_size]
            batch_coords = tile_coords[i:i + batch_size]
            
            batch_tensor = torch.from_numpy(np.stack(batch_patches)).float().to(device)
            logits = model(batch_tensor)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()

            if probs.ndim == 2:
                probs = probs[np.newaxis, ...]

            for b_idx, (r_s, c_s) in enumerate(batch_coords):
                p_tile = probs[b_idx]
                r_e = min(r_s + tile_size, h)
                c_e = min(c_s + tile_size, w)
                
                h_eff = r_e - r_s
                w_eff = c_e - c_s

                prob_map[r_s:r_e, c_s:c_e] += p_tile[:h_eff, :w_eff] * gauss_win[:h_eff, :w_eff]
                weight_map[r_s:r_e, c_s:c_e] += gauss_win[:h_eff, :w_eff]

    # Normalize accumulated probabilities by accumulated weights
    valid = weight_map > 1e-6
    prob_map[valid] /= weight_map[valid]
    return prob_map

def run_scene_inference(
    file_source: Union[str, Path, bytes, Any],
    model_path: Optional[Union[str, Path]] = None,
    threshold: float = 0.50,
    min_area_pixels: int = 20,
    apply_cloud_screening: bool = True,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    High-level end-to-end inference function for a single satellite scene / user upload.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load image and determine type
    raw_array, profile, is_geotiff = load_input_image(file_source)

    # 2. Normalize and prepare input tensor
    normalizer = ChannelNormalizer(config.BAND_STATS_PATH) if config.BAND_STATS_PATH.exists() else None
    input_stack, display_rgb, num_channels = prepare_tensor_from_upload(raw_array, is_geotiff, normalizer)

    # 3. Load appropriate model
    if model_path is None:
        model_path = config.MULTIBAND_MODEL_PATH if num_channels == 7 else config.RGB_MODEL_PATH

    model = LakeUNet(n_channels=num_channels, n_classes=1, base_filters=32).to(device)
    
    if Path(model_path).exists():
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    else:
        print(f"Warning: Model weights not found at {model_path}. Running with initialized weights.")

    # 4. Predict probability map & threshold
    prob_map = predict_large_scene_tiled(input_stack, model, device)

    # GeoTIFF inputs with spectral bands can be screened for likely cloud pixels.
    # The model still receives the original stack; screening is applied only to its
    # final probability map so terrain detail is not replaced with NaN values.
    cloud_coverage_pct = 0.0
    if apply_cloud_screening and is_geotiff and raw_array.ndim == 3 and raw_array.shape[0] >= 5:
        valid_mask = create_cloud_mask_threshold(raw_array[0], raw_array[4])
        cloud_coverage_pct = round(float((~valid_mask).mean() * 100.0), 2)
        prob_map = np.where(valid_mask, prob_map, 0.0)

    binary_mask = (prob_map >= threshold).astype(np.uint8)
    cleaned_mask = clean_binary_mask(binary_mask, min_size=int(min_area_pixels))

    # 5. Extract GeoPandas vector polygons
    transform = profile.get("transform") if profile else None
    crs = str(profile.get("crs", "EPSG:4326")) if profile else "EPSG:4326"
    gdf = mask_to_geodataframe(
        cleaned_mask,
        transform=transform,
        crs=crs,
        min_area_pixels=int(min_area_pixels),
    )

    # 6. Generate boundary overlay
    overlay_img = create_boundary_overlay(display_rgb, cleaned_mask, color_hex="#2ECC71", line_thickness=2)

    return {
        "display_rgb": display_rgb,
        "prob_map": prob_map,
        "binary_mask": cleaned_mask,
        "overlay_rgb": overlay_img,
        "geodataframe": gdf,
        "profile": profile,
        "lake_count": len(gdf),
        "total_lake_area_sqkm": round(float(gdf["area_sqkm"].sum()), 4) if len(gdf) > 0 else 0.0,
        "cloud_coverage_pct": cloud_coverage_pct,
    }
