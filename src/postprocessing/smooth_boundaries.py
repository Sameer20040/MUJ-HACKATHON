"""
Boundary smoothing and morphological cleaning module.
Removes isolated single-pixel noise, fills small internal holes,
and applies Douglas-Peucker polygon geometric simplification.
"""

from typing import Union
import cv2
import numpy as np
from shapely.geometry import Polygon, MultiPolygon

def clean_binary_mask(
    mask: np.ndarray,
    min_size: int = 25,
    morph_kernel_size: int = 3
) -> np.ndarray:
    """
    Applies morphological opening & closing, and removes connected components
    smaller than min_size pixels.
    """
    binary = (mask > 0).astype(np.uint8)
    
    if morph_kernel_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
        # Close small holes
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        # Open to remove single pixel noise spikes
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Filter out tiny connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            cleaned[labels == i] = 1

    return cleaned

def simplify_polygon_geometry(
    geom: Union[Polygon, MultiPolygon],
    tolerance: float = 0.0001,
    preserve_topology: bool = True
) -> Union[Polygon, MultiPolygon]:
    """
    Applies Douglas-Peucker algorithm to simplify vertex density of lake contours.
    """
    if geom is None or geom.is_empty:
        return geom
    return geom.simplify(tolerance=tolerance, preserve_topology=preserve_topology)
