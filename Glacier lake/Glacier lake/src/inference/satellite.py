"""
Satellite imagery fetch module with Yandex primary and Esri fallback.
Provides a reliable way to obtain real RGB satellite imagery for a given
(lat, lon, zoom, size) without requiring API keys for the fallback path.
"""

from __future__ import annotations

from typing import Tuple, Optional
import io

import numpy as np
import requests
from PIL import Image

from ..utils.yandex_satellite import fetch_yandex_satellite_image, get_yandex_api_key
from ..utils.config import config


def fetch_esri_world_imagery(
    lat: float,
    lon: float,
    zoom: int = 12,
    size: Tuple[int, int] = (1024, 1024),
) -> np.ndarray:
    """
    Fetches a real satellite RGB image from the **public, key-free**
    Esri World Imagery map export service.

    Esri World Imagery provides high-resolution satellite and aerial imagery
    (max zoom ~19) without an API key for non-commercial / fair usage.
    Endpoint: https://server.arcgisonline.com/ArcGIS/rest/services/
              World_Imagery/MapServer/export

    Parameters
    ----------
    lat, lon : float
        Center coordinates in decimal degrees (WGS84).
    zoom : int, default 12
        Map zoom level (0-23). Higher = closer.
    size : tuple(int, int), default (1024, 1024)
        Requested image size in pixels (width, height).

    Returns
    -------
    rgb : np.ndarray
        (H, W, 3) uint8 numpy array of the satellite image.

    Raises
    ------
    ValueError
        If the request fails or the image cannot be decoded.
    """
    # Convert lat/lon/zoom/size to bounding box in EPSG:4326 (WGS84)
    # Approximate degrees per pixel at equator: 360 / (256 * 2^zoom)
    # We'll compute a simple square bbox centered on (lon, lat)
    # Source: https://wiki.openstreetmap.org/wiki/Zoom_levels
    DEG_PER_PIXEL_LAT = 360.0 / (256.0 * (2 ** zoom))
    half_lat = size[1] * DEG_PER_PIXEL_LAT / 2.0
    DEG_PER_PIXEL_LON = DEG_PER_PIXEL_LAT / max(np.cos(np.radians(lat)), 1e-6)  # avoid div0 at poles
    half_lon = size[0] * DEG_PER_PIXEL_LON / 2.0

    min_lon = lon - half_lon
    max_lon = lon + half_lon
    min_lat = lat - half_lat
    max_lat = lat + half_lat

    bbox = f"{min_lon},{min_lat},{max_lon},{max_lat}"

    params = {
        "bbox": bbox,
        "bboxSR": "4326",
        "imageSR": "4326",
        "size": f"{size[0]},{size[1]}",
        "format": "png",
        "f": "image",
    }

    url = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"
    response = requests.get(url, params=params, timeout=30)

    if response.status_code != 200:
        raise ValueError(
            f"Esri World Imagery export failed (HTTP {response.status_code}). "
            f"Response: {response.text[:200]}"
        )

    try:
        with Image.open(io.BytesIO(response.content)) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            rgb = np.array(img)
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"Could not decode Esri imagery: {exc}") from exc

    return rgb


import rasterio


def compute_satellite_transform(
    lat: float,
    lon: float,
    zoom: int = 12,
    size: Tuple[int, int] = (1024, 1024),
) -> rasterio.Affine:
    """Computes exact WGS84 Affine transform for imagery centered at (lat, lon)."""
    DEG_PER_PIXEL_LAT = 360.0 / (256.0 * (2 ** zoom))
    DEG_PER_PIXEL_LON = DEG_PER_PIXEL_LAT / max(np.cos(np.radians(lat)), 1e-6)
    min_lon = lon - (size[0] * DEG_PER_PIXEL_LON / 2.0)
    max_lat = lat + (size[1] * DEG_PER_PIXEL_LAT / 2.0)
    return rasterio.Affine(DEG_PER_PIXEL_LON, 0.0, min_lon, 0.0, -DEG_PER_PIXEL_LAT, max_lat)


def fetch_satellite_scene(
    lat: float,
    lon: float,
    zoom: int = 12,
    size: Tuple[int, int] = (1024, 1024),
    api_key: Optional[str] = None,
) -> Tuple[np.ndarray, str, rasterio.Affine]:
    """
    Fetch a real satellite RGB image, trying Yandex Static Maps first and
    falling back to the public Esri World Imagery service on any failure.

    Returns a tuple (rgb_array, source_name, transform) so the caller can disclose
    which provider served the image and map boundaries to real WGS84 coordinates.
    """
    transform = compute_satellite_transform(lat, lon, zoom, size)
    # Try Yandex first (requires key)
    try:
        key = api_key or get_yandex_api_key()
        if key:  # only attempt if a key is configured
            rgb = fetch_yandex_satellite_image(lat, lon, zoom, size, api_key=key)
            return rgb, "Yandex Static Maps", transform
    except Exception:  # any error triggers fallback
        pass  # fall through to Esri

    # Fallback: Esri World Imagery (always works, no key needed)
    rgb = fetch_esri_world_imagery(lat, lon, zoom, size)
    return rgb, "Esri World Imagery", transform