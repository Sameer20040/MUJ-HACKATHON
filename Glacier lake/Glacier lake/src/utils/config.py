"""
Configuration file for Glacial Lake Detection & Temporal Change Analysis.
Contains global hyperparameters, path references, band index definitions, and classification thresholds.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent.parent


def _load_dotenv() -> None:
    """Minimal .env loader (no external dependency). Reads BASE_DIR/.env into os.environ."""
    env_path = BASE_DIR / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()

@dataclass
class Config:
    # Base Directories
    ROOT_DIR: Path = BASE_DIR
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = BASE_DIR / "data" / "raw"
    PROCESSED_DATA_DIR: Path = BASE_DIR / "data" / "processed"
    REFERENCE_DATA_DIR: Path = BASE_DIR / "data" / "reference"
    MODELS_DIR: Path = BASE_DIR / "models"
    OUTPUTS_DIR: Path = BASE_DIR / "outputs"
    SHAPEFILES_DIR: Path = BASE_DIR / "outputs" / "shapefiles"
    STATS_DIR: Path = BASE_DIR / "outputs" / "stats"
    MAPS_DIR: Path = BASE_DIR / "outputs" / "maps"
    USER_UPLOADS_DIR: Path = BASE_DIR / "outputs" / "user_uploads"

    # Imagery Bands (Sentinel-2 L2A)
    # Order in 7-channel stack: [Blue, Green, Red, NIR, SWIR1, NDWI, DEM]
    MULTISPECTRAL_BANDS: List[str] = field(
        default_factory=lambda: ["B02", "B03", "B04", "B08", "B11", "NDWI", "DEM"]
    )
    RGB_BANDS: List[str] = field(default_factory=lambda: ["Red", "Green", "Blue"])
    NUM_INPUT_CHANNELS: int = 7
    NUM_RGB_CHANNELS: int = 3
    NUM_CLASSES: int = 1  # Binary segmentation: Water (1) vs Non-water (0)

    # Patch / Tiling Parameters
    TILE_SIZE: int = 256
    TILE_STRIDE: int = 224  # 32px overlap for inference stitching
    MIN_LAKE_PIXELS_FOR_TRAIN_TILE: int = 25

    # Training Hyperparameters
    BATCH_SIZE: int = 8
    LEARNING_RATE: float = 1e-3
    WEIGHT_DECAY: float = 1e-4
    NUM_EPOCHS: int = 50
    EARLY_STOPPING_PATIENCE: int = 10
    DICE_LOSS_WEIGHT: float = 0.5
    BCE_LOSS_WEIGHT: float = 0.5
    SEGMENTATION_THRESHOLD: float = 0.5

    # Change Detection Thresholds
    IOU_MATCH_THRESHOLD: float = 0.15
    CENTROID_DISTANCE_THRESHOLD_METERS: float = 150.0
    MIN_LAKE_AREA_SQ_METERS: float = 900.0  # ~9 pixels (30m Sentinel/Landsat resolution)

    # Visualization Colors (Hex & RGB)
    COLOR_YEAR_1: str = "#2A82E4"      # Blue (Lake in Year 1)
    COLOR_YEAR_2: str = "#E74C3C"      # Red (Lake in Year 2)
    COLOR_NEW: str = "#F1C40F"         # Yellow (Newly Formed Lake)
    COLOR_SURVIVED: str = "#2ECC71"    # Green (Survived Lake)
    COLOR_DRAINED: str = "#E74C3C"     # Red (Drained Lake)

    # Model Checkpoint Filenames
    MULTIBAND_MODEL_PATH: Path = BASE_DIR / "models" / "unet_lake_seg.pth"
    RGB_MODEL_PATH: Path = BASE_DIR / "models" / "unet_rgb_fallback.pth"
    BAND_STATS_PATH: Path = BASE_DIR / "models" / "band_stats.json"

    # Yandex Static Maps API (real satellite imagery)
    YANDEX_API_KEY: str = os.environ.get("YANDEX_API_KEY", "")
    YANDEX_CENTER_LAT: float = 36.58     # Hunza / Karakoram region
    YANDEX_CENTER_LON: float = 74.65
    YANDEX_ZOOM: int = 12
    YANDEX_IMAGE_SIZE: Tuple[int, int] = (1024, 1024)

    def ensure_directories(self):
        """Ensure all required project directories exist."""
        for path in [
            self.DATA_DIR,
            self.RAW_DATA_DIR / "sentinel2" / "2016",
            self.RAW_DATA_DIR / "sentinel2" / "2022",
            self.RAW_DATA_DIR / "sentinel1_sar" / "2016",
            self.RAW_DATA_DIR / "sentinel1_sar" / "2022",
            self.RAW_DATA_DIR / "dem",
            self.REFERENCE_DATA_DIR / "glims_icimod_shapefiles",
            self.PROCESSED_DATA_DIR / "indices",
            self.PROCESSED_DATA_DIR / "masks",
            self.PROCESSED_DATA_DIR / "aligned",
            self.PROCESSED_DATA_DIR / "tiles" / "images",
            self.PROCESSED_DATA_DIR / "tiles" / "masks",
            self.MODELS_DIR,
            self.OUTPUTS_DIR,
            self.SHAPEFILES_DIR,
            self.STATS_DIR,
            self.MAPS_DIR,
            self.USER_UPLOADS_DIR / "input",
            self.USER_UPLOADS_DIR / "predicted",
        ]:
            path.mkdir(parents=True, exist_ok=True)

config = Config()
config.ensure_directories()
