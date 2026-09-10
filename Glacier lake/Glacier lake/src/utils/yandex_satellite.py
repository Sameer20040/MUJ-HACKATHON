"""
Yandex Static Maps API client for real satellite imagery.
Fetches a real Earth satellite RGB image for a given lat/lon/zoom using the Yandex
Static Maps API (`l=sat`), so the pipeline can segment actual terrain instead of the
pre-generated synthetic scenes.

API reference: https://yandex.ru/dev/maps/staticmaps/
URL: https://static-maps.yandex.ru/v1?ll={lon},{lat}&z={zoom}&size={W}x{H}&l=sat&apikey={KEY}
"""

import io
from typing import Optional, Tuple

import numpy as np
import requests
from PIL import Image

from ..utils.config import config

YANDEX_STATIC_MAPS_URL = "https://static-maps.yandex.ru/v1"


def get_yandex_api_key() -> str:
    """Returns the configured Yandex Static Maps API key (or empty string)."""
    return config.YANDEX_API_KEY or ""


def fetch_yandex_satellite_image(
    lat: float,
    lon: float,
    zoom: int = 12,
    size: Tuple[int, int] = (1024, 1024),
    layer: str = "sat",
    api_key: Optional[str] = None,
) -> np.ndarray:
    """
    Fetches a real satellite RGB image of the given location from Yandex Static Maps.

    Parameters:
        lat: Center latitude (degrees N).
        lon: Center longitude (degrees E).
        zoom: Map zoom level (0-17). Higher = closer.
        size: Requested image size in pixels (width, height).
        layer: Map layer. "sat" = satellite imagery only.
        api_key: Optional key override; defaults to config.YANDEX_API_KEY.

    Returns:
        rgb: (H, W, 3) uint8 numpy array of the satellite image.

    Raises:
        ValueError: If no API key is configured or the request fails.
    """
    key = api_key or get_yandex_api_key()
    if not key:
        raise ValueError(
            "Yandex API key is not configured. Add YANDEX_API_KEY=... to the .env file "
            "or set the YANDEX_API_KEY environment variable."
        )

    params = {
        "ll": f"{lon},{lat}",
        "z": zoom,
        "size": f"{int(size[0])},{int(size[1])}",
        "l": layer,
        "apikey": key,
    }

    response = requests.get(YANDEX_STATIC_MAPS_URL, params=params, timeout=30)
    if response.status_code != 200:
        raise ValueError(
            f"Yandex Static Maps API request failed (HTTP {response.status_code}). "
            f"Response: {response.text[:200]}"
        )

    try:
        with Image.open(io.BytesIO(response.content)) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            rgb = np.array(img)
    except Exception as exc:  # pragma: no cover - defensive decoding guard
        raise ValueError(f"Could not decode satellite image from API: {exc}") from exc

    return rgb
