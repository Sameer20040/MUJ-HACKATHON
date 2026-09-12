"""
Flexible input image loader and channel adapter module.
Handles both geospatial GeoTIFFs (with multi-spectral bands & CRS) and plain image uploads (JPG/PNG).
Includes robust memory-safe stream decoding and automatic dimension scaling.
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Optional, Union
import io
import numpy as np
import rasterio
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # Prevent DecompressionBombError on large satellite captures
import cv2

from ..preprocessing.compute_indices import compute_ndwi, approximate_ndwi_from_rgb
from ..dataset.normalize import ChannelNormalizer

def load_input_image(
    file_source: Union[str, Path, bytes, Any],
    max_dimension: int = 3072
) -> Tuple[np.ndarray, Optional[Dict[str, Any]], bool]:
    """
    Safely loads uploaded file or path and determines whether it is a GeoTIFF or standard RGB image.
    Uses memory-safe OpenCV buffer decoding for standard images and Rasterio for GeoTIFFs.
    
    Returns:
        image_array: Numpy array (C, H, W) or (H, W, 3)
        profile: Rasterio metadata profile if GeoTIFF, else None
        is_geotiff: Boolean flag
    """
    # 1. Direct numpy array or PIL Image support
    if isinstance(file_source, np.ndarray):
        arr = file_source
        if arr.ndim == 3 and arr.shape[-1] in [3, 4]:
            h, w = arr.shape[:2]
            if max(h, w) > max_dimension:
                scale = max_dimension / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                arr = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return arr, None, False

    if isinstance(file_source, Image.Image):
        try:
            file_source.load()  # Force decode pixels (safe no-op if already loaded)
        except Exception:
            pass
        arr = np.array(file_source.convert("RGB"))
        return arr, None, False

    # 2. Extract bytes safely
    bytes_data = None
    file_path = None

    if isinstance(file_source, (str, Path)):
        p = Path(file_source)
        if p.suffix.lower() in [".tif", ".tiff"]:
            try:
                with rasterio.open(p) as src:
                    arr = src.read()
                    profile = src.profile.copy()
                return arr, profile, True
            except Exception:
                pass
        file_path = str(p)
    else:
        # Streamlit UploadedFile or BytesIO
        if hasattr(file_source, "seek"):
            try:
                file_source.seek(0)
            except Exception:
                pass

        if hasattr(file_source, "getvalue"):
            bytes_data = file_source.getvalue()
        elif hasattr(file_source, "read"):
            bytes_data = file_source.read()
        elif isinstance(file_source, bytes):
            bytes_data = file_source

        # Discard empty reads
        if bytes_data is not None and len(bytes_data) == 0:
            bytes_data = None

    # 2. Check if GeoTIFF
    if bytes_data is not None:
        if bytes_data.startswith(b"II*\x00") or bytes_data.startswith(b"MM\x00*"):
            try:
                from rasterio.io import MemoryFile
                with MemoryFile(bytes_data) as memfile:
                    with memfile.open() as src:
                        arr = src.read()
                        profile = src.profile.copy()
                return arr, profile, True
            except Exception:
                pass  # Fallback to standard RGB decoding

    # 3. Standard RGB Decoding (Fast, memory-safe OpenCV buffer decode)
    if bytes_data is not None:
        try:
            np_buf = np.frombuffer(bytes_data, dtype=np.uint8)
            cv_img = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
            if cv_img is not None:
                # Convert BGR to RGB
                rgb_arr = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                
                # Check if dimensions exceed max_dimension (e.g. 3072px) to prevent OOM
                h, w = rgb_arr.shape[:2]
                if max(h, w) > max_dimension:
                    scale = max_dimension / max(h, w)
                    new_w, new_h = int(w * scale), int(h * scale)
                    rgb_arr = cv2.resize(rgb_arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    
                return rgb_arr, None, False
        except Exception:
            pass

        # Fallback to PIL with memory safety
        try:
            bio = io.BytesIO(bytes_data)
            bio.seek(0)
            with Image.open(bio) as pil_img:
                if pil_img.mode != "RGB":
                    pil_img = pil_img.convert("RGB")
                
                # Downsample if overly large
                if max(pil_img.size) > max_dimension:
                    pil_img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
                    
                arr = np.array(pil_img)
                return arr, None, False
        except Exception as e:
            raise ValueError(f"Unable to decode uploaded image: {e}")

    elif file_path is not None:
        try:
            cv_img = cv2.imread(file_path, cv2.IMREAD_COLOR)
            if cv_img is not None:
                rgb_arr = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                h, w = rgb_arr.shape[:2]
                if max(h, w) > max_dimension:
                    scale = max_dimension / max(h, w)
                    new_w, new_h = int(w * scale), int(h * scale)
                    rgb_arr = cv2.resize(rgb_arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
                return rgb_arr, None, False
        except Exception:
            pass

        with Image.open(file_path) as pil_img:
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            if max(pil_img.size) > max_dimension:
                pil_img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
            arr = np.array(pil_img)
            return arr, None, False

    raise ValueError("No valid image data could be loaded from source.")

def prepare_tensor_from_upload(
    raw_array: np.ndarray,
    is_geotiff: bool,
    normalizer: Optional[ChannelNormalizer] = None
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Converts raw image data into model-ready (C, H, W) normalized tensor and RGB display image.
    
    Returns:
        input_stack: (C, H, W) float32 array
        display_rgb: (H, W, 3) uint8 image for UI visualization
        num_channels: 7 (multi-band) or 3 (RGB)
    """
    if is_geotiff and raw_array.ndim == 3 and raw_array.shape[0] >= 5:
        # Multi-band satellite imagery (B2, B3, B4, B8, B11, etc.)
        c, h, w = raw_array.shape
        b2 = raw_array[0].astype(np.float32)
        b3 = raw_array[1].astype(np.float32)
        b4 = raw_array[2].astype(np.float32)
        b8 = raw_array[3].astype(np.float32)
        b11 = raw_array[4].astype(np.float32)

        # Scale reflectance if needed
        if np.nanmax(b4) > 10.0:
            b2, b3, b4, b8, b11 = b2/10000.0, b3/10000.0, b4/10000.0, b8/10000.0, b11/10000.0

        ndwi = compute_ndwi(b3, b8)
        dem = raw_array[5].astype(np.float32) if c > 5 else np.zeros((h, w), dtype=np.float32)
        dem_norm = (dem - 3000.0) / 2000.0

        stack = np.stack([b2, b3, b4, b8, b11, ndwi, dem_norm], axis=0).astype(np.float32)
        
        # Display RGB
        rgb_disp = np.stack([b4, b3, b2], axis=-1)
        rgb_disp = np.clip(rgb_disp * 255.0, 0, 255).astype(np.uint8)
        num_channels = 7
    else:
        # Standard 3-channel RGB image (H, W, 3)
        if raw_array.ndim == 2:
            rgb_arr = cv2.cvtColor(raw_array, cv2.COLOR_GRAY2RGB)
        elif raw_array.ndim == 3 and raw_array.shape[0] in [3, 4]:
            rgb_arr = np.transpose(raw_array[:3], (1, 2, 0))
        else:
            rgb_arr = raw_array[..., :3]

        display_rgb = rgb_arr.copy()
        rgb_disp = display_rgb
        
        # Build 3-channel normalized RGB stack [Red, Green, Blue]
        r = rgb_arr[..., 0].astype(np.float32) / 255.0
        g = rgb_arr[..., 1].astype(np.float32) / 255.0
        b = rgb_arr[..., 2].astype(np.float32) / 255.0
        stack = np.stack([r, g, b], axis=0).astype(np.float32)
        num_channels = 3

    if normalizer is not None and len(normalizer.means) >= 7:
        if num_channels == 7:
            stack = normalizer.transform(stack)
        elif num_channels == 3:
            # Map Red (idx 2), Green (idx 1), Blue (idx 0)
            stack[0] = (stack[0] - normalizer.means[2]) / (normalizer.stds[2] + 1e-7)
            stack[1] = (stack[1] - normalizer.means[1]) / (normalizer.stds[1] + 1e-7)
            stack[2] = (stack[2] - normalizer.means[0]) / (normalizer.stds[0] + 1e-7)
    elif num_channels == 3:
        # Standard normalization fallback
        for c_i in range(3):
            m_val = float(np.mean(stack[c_i]))
            s_val = float(np.std(stack[c_i])) + 1e-7
            stack[c_i] = (stack[c_i] - m_val) / s_val

    return stack, rgb_disp, num_channels
