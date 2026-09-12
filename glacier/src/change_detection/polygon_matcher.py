"""
Spatial polygon matching module.
Associates lake instances across Epoch 1 (e.g. 2016) and Epoch 2 (e.g. 2022)
using Intersection over Union (IoU) overlap and centroid distance thresholds.
"""

from typing import List, Dict, Tuple, Optional
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
import numpy as np

def compute_polygon_iou(geom1: Polygon, geom2: Polygon) -> float:
    """Computes spatial IoU between two shapely geometries."""
    if not geom1.intersects(geom2):
        return 0.0
    
    intersection_area = geom1.intersection(geom2).area
    union_area = geom1.union(geom2).area
    return float(intersection_area / (union_area + 1e-9))

def match_lake_polygons_across_epochs(
    gdf_y1: gpd.GeoDataFrame,
    gdf_y2: gpd.GeoDataFrame,
    iou_thresh: float = 0.10,
    max_centroid_dist_deg: float = 0.005 # ~500m
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """
    Matches lake polygons between Year 1 (Y1) and Year 2 (Y2).
    
    Returns:
        matches: List of tuples (index_y1, index_y2, iou_score)
        unmatched_y1: List of Y1 indices (Drained candidates)
        unmatched_y2: List of Y2 indices (Newly Formed candidates)
    """
    if len(gdf_y1) == 0:
        return [], [], list(range(len(gdf_y2)))
    if len(gdf_y2) == 0:
        return [], list(range(len(gdf_y1))), []

    matches = []
    matched_y1 = set()
    matched_y2 = set()

    # Step 1: Spatial Intersection & IoU Matching
    for idx1, row1 in gdf_y1.iterrows():
        best_match_idx2 = None
        best_iou = 0.0

        for idx2, row2 in gdf_y2.iterrows():
            if idx2 in matched_y2:
                continue
            
            iou = compute_polygon_iou(row1.geometry, row2.geometry)
            if iou > best_iou and iou >= iou_thresh:
                best_iou = iou
                best_match_idx2 = idx2

        if best_match_idx2 is not None:
            matches.append((idx1, best_match_idx2, best_iou))
            matched_y1.add(idx1)
            matched_y2.add(best_match_idx2)

    # Step 2: Fallback Centroid Proximity Matching for shifted / receding lakes
    for idx1, row1 in gdf_y1.iterrows():
        if idx1 in matched_y1:
            continue
        
        c1 = row1.geometry.centroid
        closest_idx2 = None
        min_dist = float("inf")

        for idx2, row2 in gdf_y2.iterrows():
            if idx2 in matched_y2:
                continue
            
            c2 = row2.geometry.centroid
            dist = c1.distance(c2)
            if dist < min_dist and dist <= max_centroid_dist_deg:
                min_dist = dist
                closest_idx2 = idx2

        if closest_idx2 is not None:
            matches.append((idx1, closest_idx2, 0.0))
            matched_y1.add(idx1)
            matched_y2.add(closest_idx2)

    unmatched_y1 = [i for i in range(len(gdf_y1)) if i not in matched_y1]
    unmatched_y2 = [i for i in range(len(gdf_y2)) if i not in matched_y2]

    return matches, unmatched_y1, unmatched_y2
