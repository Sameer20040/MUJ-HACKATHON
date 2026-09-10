"""
Raster mask to GeoPandas vector polygon conversion module.
Extracts vector boundaries from segmented binary raster masks using Rasterio shapes / OpenCV,
computes geodetic area and perimeter, and exports Shapefiles / GeoJSONs.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape, Polygon, MultiPolygon
from shapely.validation import make_valid

from .smooth_boundaries import clean_binary_mask, simplify_polygon_geometry
from ..utils.geo_utils import calculate_polygon_area_sqkm

def mask_to_polygons(
    binary_mask: np.ndarray,
    transform: Optional[rasterio.Affine] = None,
    crs: str = "EPSG:4326",
    min_area_pixels: int = 15,
    simplify_tol: float = 0.0001
) -> List[Tuple[Polygon, dict]]:
    """
    Converts binary 2D mask into list of Shapely Polygons and property dictionaries.
    """
    cleaned = clean_binary_mask(binary_mask, min_size=min_area_pixels)
    if cleaned.sum() == 0:
        return []

    # If no geospatial affine transform is provided, use pixel coordinates
    if transform is None:
        transform = rasterio.Affine.identity()

    extracted_shapes = shapes(cleaned.astype(np.int16), mask=(cleaned > 0), transform=transform)
    results = []

    for geom_dict, val in extracted_shapes:
        if val == 1:
            geom = shape(geom_dict)
            if not geom.is_valid:
                geom = make_valid(geom)
            
            if simplify_tol > 0:
                geom = simplify_polygon_geometry(geom, tolerance=simplify_tol)
                
            if geom.is_empty:
                continue

            area_sqkm = calculate_polygon_area_sqkm(geom, source_crs=crs)
            centroid = geom.centroid

            props = {
                "area_sqkm": round(area_sqkm, 4),
                "perimeter_km": round(float(geom.length * 111.32), 4),
                "centroid_lon": round(float(centroid.x), 6),
                "centroid_lat": round(float(centroid.y), 6)
            }
            results.append((geom, props))

    return results

def mask_to_geodataframe(
    binary_mask: np.ndarray,
    transform: Optional[rasterio.Affine] = None,
    crs: str = "EPSG:4326",
    min_area_pixels: int = 15
) -> gpd.GeoDataFrame:
    """
    Converts raster mask into a formatted GeoDataFrame.
    """
    poly_props = mask_to_polygons(binary_mask, transform=transform, crs=crs, min_area_pixels=min_area_pixels)
    
    if len(poly_props) == 0:
        return gpd.GeoDataFrame(
            columns=["lake_id", "area_sqkm", "perimeter_km", "centroid_lon", "centroid_lat", "geometry"],
            geometry="geometry",
            crs=crs
        )

    geoms = [p[0] for p in poly_props]
    records = []
    for idx, (_, prop) in enumerate(poly_props):
        r = {"lake_id": f"GL_{idx+1:04d}", **prop}
        records.append(r)

    gdf = gpd.GeoDataFrame(records, geometry=geoms, crs=crs)
    return gdf

def export_mask_to_shapefile(
    binary_mask: np.ndarray,
    output_path: Union[str, Path],
    transform: Optional[rasterio.Affine] = None,
    crs: str = "EPSG:4326"
) -> gpd.GeoDataFrame:
    """
    Vectorizes mask and exports to ESRI Shapefile or GeoJSON.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf = mask_to_geodataframe(binary_mask, transform=transform, crs=crs)
    
    if str(output_path).endswith(".geojson"):
        gdf.to_file(output_path, driver="GeoJSON")
    else:
        gdf.to_file(output_path, driver="ESRI Shapefile")
    return gdf
