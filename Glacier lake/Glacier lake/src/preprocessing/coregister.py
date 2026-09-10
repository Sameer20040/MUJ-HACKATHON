"""
Multi-temporal and multi-sensor spatial co-registration and raster grid matching module.
"""

from pathlib import Path
from typing import Union, Tuple
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

def coregister_and_match_rasters(
    source_path: Union[str, Path],
    reference_path: Union[str, Path],
    output_path: Union[str, Path],
    resampling_method: Resampling = Resampling.bilinear
) -> Tuple[np.ndarray, dict]:
    """
    Reprojects and resamples `source_path` raster so it exactly aligns
    with the bounding box, CRS, resolution, and dimensions of `reference_path`.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with rasterio.open(reference_path) as ref:
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_width = ref.width
        ref_height = ref.height
        ref_profile = ref.profile.copy()

    with rasterio.open(source_path) as src:
        src_count = src.count
        aligned_data = np.zeros((src_count, ref_height, ref_width), dtype=np.float32)
        
        ref_profile.update({
            "count": src_count,
            "dtype": "float32",
            "nodata": src.nodata if src.nodata is not None else 0.0
        })

        for band_idx in range(1, src_count + 1):
            reproject(
                source=rasterio.band(src, band_idx),
                destination=aligned_data[band_idx - 1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=resampling_method
            )

    with rasterio.open(output_path, "w", **ref_profile) as dst:
        dst.write(aligned_data)
        
    return aligned_data, ref_profile
