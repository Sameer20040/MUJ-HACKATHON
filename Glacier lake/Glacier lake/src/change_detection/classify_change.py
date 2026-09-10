"""
Temporal change classification and summary reporting engine.
Applies domain logic:
  - If lake exists in Year 2 only -> 'Newly Formed' (Yellow)
  - If lake exists in Year 1 only -> 'Drained' (Red)
  - If lake exists in both years -> 'Survived' (Green) + Area Delta %
Generates tabular metrics (change_summary.csv) and classified Shapefile.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

from .polygon_matcher import match_lake_polygons_across_epochs
from ..utils.config import config
from .risk_scoring import add_risk_scores, compute_terrain_features

def classify_temporal_change(
    gdf_y1: gpd.GeoDataFrame,
    gdf_y2: gpd.GeoDataFrame,
    year1_label: str = "2016",
    year2_label: str = "2022",
    dem_y1: Optional[np.ndarray] = None,
    dem_y2: Optional[np.ndarray] = None,
    transform_y1: Optional[rasterio.Affine] = None,
    transform_y2: Optional[rasterio.Affine] = None,
) -> Tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """
    Classifies all lake polygons and computes area changes.

    Returns:
        classified_gdf: GeoDataFrame with geometry, status, color, and area metrics.
        summary_df: Tabular dataframe summarizing changes for every lake instance.
    """
    matches, drained_indices, new_indices = match_lake_polygons_across_epochs(gdf_y1, gdf_y2)

    records = []
    geometries = []

    # 1. Survived Lakes (Matched in both years)
    for idx1, idx2, iou_score in matches:
        row1 = gdf_y1.iloc[idx1]
        row2 = gdf_y2.iloc[idx2]

        area1 = float(row1.get("area_sqkm", 0.0))
        area2 = float(row2.get("area_sqkm", 0.0))
        delta_area = round(area2 - area1, 4)
        pct_change = round(((area2 - area1) / (area1 + 1e-6)) * 100.0, 2)

        c = row2.geometry.centroid
        rec = {
            "lake_id": f"SURV_{idx2+1:03d}",
            "status": "Survived",
            "status_code": 2,
            "color": config.COLOR_SURVIVED,
            f"area_{year1_label}_sqkm": area1,
            f"area_{year2_label}_sqkm": area2,
            "area_change_sqkm": delta_area,
            "area_change_pct": pct_change,
            "match_iou": round(iou_score, 3),
            "centroid_lon": round(float(c.x), 6),
            "centroid_lat": round(float(c.y), 6),
        }
        records.append(rec)
        geometries.append(row2.geometry)

    # 2. Newly Formed Lakes (Present in Y2 only)
    for idx2 in new_indices:
        row2 = gdf_y2.iloc[idx2]
        area2 = float(row2.get("area_sqkm", 0.0))
        c = row2.geometry.centroid

        rec = {
            "lake_id": f"NEW_{idx2+1:03d}",
            "status": "Newly Formed",
            "status_code": 1,
            "color": config.COLOR_NEW,
            f"area_{year1_label}_sqkm": 0.0,
            f"area_{year2_label}_sqkm": area2,
            "area_change_sqkm": area2,
            "area_change_pct": 100.0,
            "match_iou": 0.0,
            "centroid_lon": round(float(c.x), 6),
            "centroid_lat": round(float(c.y), 6),
        }
        records.append(rec)
        geometries.append(row2.geometry)

    # 3. Drained / Disappeared Lakes (Present in Y1 only)
    for idx1 in drained_indices:
        row1 = gdf_y1.iloc[idx1]
        area1 = float(row1.get("area_sqkm", 0.0))
        c = row1.geometry.centroid

        rec = {
            "lake_id": f"DRAIN_{idx1+1:03d}",
            "status": "Drained",
            "status_code": 0,
            "color": config.COLOR_DRAINED,
            f"area_{year1_label}_sqkm": area1,
            f"area_{year2_label}_sqkm": 0.0,
            "area_change_sqkm": -area1,
            "area_change_pct": -100.0,
            "match_iou": 0.0,
            "centroid_lon": round(float(c.x), 6),
            "centroid_lat": round(float(c.y), 6),
        }
        records.append(rec)
        geometries.append(row1.geometry)

    crs = gdf_y2.crs if len(gdf_y2) > 0 else gdf_y1.crs if len(gdf_y1) > 0 else "EPSG:4326"

    # Build summary DataFrame
    summary_df = pd.DataFrame(records)

    # Compute terrain features if DEM is available (use Year 2 DEM as primary)
    if dem_y2 is not None and len(summary_df) > 0:
        # Extract centroids for terrain computation
        centroid_lons = summary_df["centroid_lon"].to_numpy()
        centroid_lats = summary_df["centroid_lat"].to_numpy()
        terrain_slopes = compute_terrain_features(dem_y2, centroid_lons, centroid_lats, transform_y2)
        summary_df["mean_slope"] = terrain_slopes

    # Apply GLOF risk scoring (will use mean_slope if present)
    summary_df = add_risk_scores(summary_df)

    # Build classified GeoDataFrame by joining the risk-enriched summary with geometries
    # This guarantees both outputs have identical risk columns and lake_id ordering
    if not summary_df.empty and len(geometries) == len(summary_df):
        classified_gdf = gpd.GeoDataFrame(
            summary_df.assign(geometry=geometries),
            geometry="geometry",
            crs=crs,
        )
    else:
        # Fallback (should not happen with valid inputs)
        classified_gdf = gpd.GeoDataFrame(records, geometry=geometries, crs=crs)

    return classified_gdf, summary_df

def generate_change_summary_report(summary_df: pd.DataFrame) -> Dict[str, Union[int, float]]:
    """Computes high-level aggregated metrics across all classes."""
    if len(summary_df) == 0:
        return {
            "total_lakes_detected": 0,
            "newly_formed_count": 0,
            "survived_count": 0,
            "drained_count": 0,
            "net_area_change_sqkm": 0.0,
        }

    counts = summary_df["status"].value_counts().to_dict()
    net_change = float(summary_df["area_change_sqkm"].sum())
    
    return {
        "total_lakes_detected": len(summary_df),
        "newly_formed_count": counts.get("Newly Formed", 0),
        "survived_count": counts.get("Survived", 0),
        "drained_count": counts.get("Drained", 0),
        "net_area_change_sqkm": round(net_change, 4),
    }

def run_change_detection_pipeline(
    gdf_y1: gpd.GeoDataFrame,
    gdf_y2: gpd.GeoDataFrame,
    output_shapefile_path: Optional[Union[str, Path]] = None,
    output_csv_path: Optional[Union[str, Path]] = None
) -> Tuple[gpd.GeoDataFrame, pd.DataFrame, dict]:
    """Complete change detection workflow execution."""
    classified_gdf, summary_df = classify_temporal_change(gdf_y1, gdf_y2)
    stats_dict = generate_change_summary_report(summary_df)

    if output_shapefile_path:
        out_shp = Path(output_shapefile_path)
        out_shp.parent.mkdir(parents=True, exist_ok=True)
        if str(out_shp).endswith(".geojson"):
            classified_gdf.to_file(out_shp, driver="GeoJSON")
        else:
            classified_gdf.to_file(out_shp, driver="ESRI Shapefile")

    if output_csv_path:
        out_csv = Path(output_csv_path)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(out_csv, index=False)

    return classified_gdf, summary_df, stats_dict
