"""
Raster tiling and patch extraction module.
Splits large multi-band remote sensing scenes into 256x256 tiles with stride/overlap.
Includes intelligent class-balancing to subsample background-only tiles.
"""

from pathlib import Path
from typing import List, Tuple, Union, Optional
import numpy as np
import rasterio

from ..utils.config import config

def extract_patches_from_array(
    image_stack: np.ndarray,
    mask: Optional[np.ndarray] = None,
    tile_size: int = 256,
    stride: int = 224,
    min_lake_pixels: int = 20,
    background_keep_ratio: float = 0.10,
    seed: int = 42
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Tuple[int, int]]]:
    """
    Extracts 2D/3D patches from an image array and optional mask.
    
    Parameters:
        image_stack: (C, H, W) or (H, W, C) array
        mask: (H, W) binary mask array
        tile_size: Size of square tile (default 256)
        stride: Stride between tiles (default 224 for overlap)
        min_lake_pixels: Minimum positive pixels to consider a tile 'lake-containing'
        background_keep_ratio: Fraction of pure background tiles to keep (controls class balance)
        
    Returns:
        image_tiles: List of (C, tile_size, tile_size) patches
        mask_tiles: List of (1, tile_size, tile_size) patches
        coordinates: List of (row_start, col_start) coordinates
    """
    np.random.seed(seed)
    
    # Standardize to (C, H, W)
    if image_stack.ndim == 3 and image_stack.shape[-1] <= 12:
        img = np.transpose(image_stack, (2, 0, 1))
    else:
        img = image_stack

    channels, height, width = img.shape
    image_tiles = []
    mask_tiles = []
    coordinates = []

    # Iterate over rows and columns
    for r in range(0, height, stride):
        for c in range(0, width, stride):
            # Handle border edge cases by shifting backwards
            r_start = min(r, height - tile_size) if height >= tile_size else 0
            c_start = min(c, width - tile_size) if width >= tile_size else 0
            r_end = r_start + tile_size
            c_end = c_start + tile_size

            # Extract image patch
            img_patch = img[:, r_start:r_end, c_start:c_end]
            
            # Handle padding if smaller than tile_size
            if img_patch.shape[1] != tile_size or img_patch.shape[2] != tile_size:
                pad_h = tile_size - img_patch.shape[1]
                pad_w = tile_size - img_patch.shape[2]
                img_patch = np.pad(img_patch, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')

            if mask is not None:
                mask_patch = mask[r_start:r_end, c_start:c_end]
                if mask_patch.shape[0] != tile_size or mask_patch.shape[1] != tile_size:
                    pad_h = tile_size - mask_patch.shape[0]
                    pad_w = tile_size - mask_patch.shape[1]
                    mask_patch = np.pad(mask_patch, ((0, pad_h), (0, pad_w)), mode='reflect')
                
                lake_pixels = np.sum(mask_patch > 0)
                is_lake_tile = lake_pixels >= min_lake_pixels
                
                # Intelligent class balancing: Keep all lake tiles, subsample background tiles
                if is_lake_tile or (np.random.rand() < background_keep_ratio):
                    image_tiles.append(img_patch)
                    mask_tiles.append(mask_patch[np.newaxis, ...])
                    coordinates.append((r_start, c_start))
            else:
                image_tiles.append(img_patch)
                coordinates.append((r_start, c_start))

    return image_tiles, mask_tiles, coordinates

def tile_scene_to_patches(
    scene_path: Union[str, Path],
    mask_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    tile_size: int = 256,
    stride: int = 224
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Tiles a GeoTIFF scene and saves chips as individual numpy/GeoTIFF files.
    """
    with rasterio.open(scene_path) as src:
        image_data = src.read()
        
    mask_data = None
    if mask_path and Path(mask_path).exists():
        with rasterio.open(mask_path) as m_src:
            mask_data = m_src.read(1)

    img_tiles, m_tiles, _ = extract_patches_from_array(
        image_data, mask=mask_data, tile_size=tile_size, stride=stride
    )

    if output_dir:
        out_path = Path(output_dir)
        img_out = out_path / "images"
        mask_out = out_path / "masks"
        img_out.mkdir(parents=True, exist_ok=True)
        mask_out.mkdir(parents=True, exist_ok=True)
        
        for idx, (img_t, m_t) in enumerate(zip(img_tiles, m_tiles)):
            np.save(img_out / f"patch_{idx:05d}.npy", img_t)
            if m_t is not None:
                np.save(mask_out / f"patch_{idx:05d}.npy", m_t)

    return img_tiles, m_tiles
