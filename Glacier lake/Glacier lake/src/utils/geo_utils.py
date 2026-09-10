"""
Geospatial utility functions for reading/writing rasters, CRS reprojection,
and polygon geometry calculations.
"""

import os
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, Union
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import geopandas as gpd
from shapely.geometry import shape, Polygon, MultiPolygon
from shapely.ops import transform
import pyproj

def read_geotiff(file_path: Union[str, Path]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Read GeoTIFF raster and return array (bands, H, W) and metadata profile.
    """
    file_path = str(file_path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"GeoTIFF file not found: {file_path}")
    
    with rasterio.open(file_path) as src:
        data = src.read()
        profile = src.profile.copy()
    return data, profile

def write_geotiff(
    data: np.ndarray,
    output_path: Union[str, Path],
    profile: Dict[str, Any],
    dtype: Optional[str] = None
) -> None:
    """
    Write multi-band or single-band numpy array to a GeoTIFF.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if data.ndim == 2:
        data = data[np.newaxis, ...]
        
    num_bands, height, width = data.shape
    out_profile = profile.copy()
    out_profile.update({
        "count": num_bands,
        "height": height,
        "width": width,
        "dtype": dtype or str(data.dtype)
    })
    
    with rasterio.open(output_path, "w", **out_profile) as dst:
        dst.write(data.astype(out_profile["dtype"]))

def reproject_raster(
    src_path: Union[str, Path],
    dst_path: Union[str, Path],
    target_crs: str = "EPSG:4326",
    resampling_method: Resampling = Resampling.bilinear
) -> None:
    """
    Reproject a raster file to a target coordinate reference system (CRS).
    """
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({
            "crs": target_crs,
            "transform": transform,
            "width": width,
            "height": height
        })

        with rasterio.open(dst_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=resampling_method
                )

def polygon_to_geodataframe(
    polygons: list,
    properties: Optional[list] = None,
    crs: str = "EPSG:4326"
) -> gpd.GeoDataFrame:
    """
    Converts a list of Shapely Polygons and optional property dicts into a GeoDataFrame.
    """
    if properties is None:
        properties = [{} for _ in polygons]
        
    data = properties if len(properties) > 0 else [{}] * len(polygons)
    gdf = gpd.GeoDataFrame(data, geometry=polygons, crs=crs)
    return gdf

def calculate_polygon_area_sqkm(geom: Union[Polygon, MultiPolygon], source_crs: str = "EPSG:4326") -> float:
    """
    Calculates geodetic area of polygon in square kilometers using equal-area projection.
    Gracefully handles pixel space and geodetic coordinate geometries.
    """
    if geom is None or geom.is_empty:
        return 0.0
    
    try:
        # Check if coordinates are in degree range (-180 to 180, -90 to 90)
        bounds = geom.bounds # minx, miny, maxx, maxy
        is_degree = (-180.0 <= bounds[0] <= 180.0) and (-90.0 <= bounds[1] <= 90.0) and \
                    (-180.0 <= bounds[2] <= 180.0) and (-90.0 <= bounds[3] <= 90.0)
        
        if is_degree and str(source_crs).upper().endswith("4326"):
            transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True).transform
            projected_geom = transform(transformer, geom)
            calc_area = float(projected_geom.area / 1e6)
            if not np.isnan(calc_area) and calc_area > 0:
                return round(calc_area, 6)
            return round(float(geom.area * 111.32 * 111.32), 6)
        else:
            # Assume local pixel / projected coordinates (e.g. 1 pixel = 10m x 10m = 100 sq meters = 0.0001 sq km)
            return round(float(geom.area * 0.0001), 6)
    except Exception:
        # Robust fallback
        area_val = float(geom.area)
        return round(area_val * 0.0001, 6) if not np.isnan(area_val) else 0.0

