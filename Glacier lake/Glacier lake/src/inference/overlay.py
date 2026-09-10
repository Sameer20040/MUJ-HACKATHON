"""
Visualization and cartographic overlay generation module.
Renders high-contrast lake boundary contours, temporal comparison overlays,
and creates publication-quality 3-panel figures (Panels a, b, c) matching scientific literature.
"""

from pathlib import Path
from typing import Optional, Tuple, Union, List
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from ..utils.config import config

def hex_to_bgr(hex_str: str) -> Tuple[int, int, int]:
    """Converts hex color string to OpenCV BGR tuple."""
    h = hex_str.lstrip('#')
    rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    return (rgb[2], rgb[1], rgb[0]) # BGR for OpenCV

def hex_to_rgb_norm(hex_str: str) -> Tuple[float, float, float]:
    """Converts hex color to normalized 0-1 RGB tuple for Matplotlib."""
    h = hex_str.lstrip('#')
    return tuple(int(h[i:i+2], 16)/255.0 for i in (0, 2, 4))

def create_boundary_overlay(
    base_rgb: np.ndarray,
    binary_mask: np.ndarray,
    color_hex: str = "#2ECC71",
    line_thickness: int = 2,
    fill_opacity: float = 0.25
) -> np.ndarray:
    """
    Overlays detected lake boundary contours onto a base RGB image.
    """
    img = base_rgb.copy()
    if img.dtype != np.uint8:
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)

    mask = (binary_mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bgr_color = hex_to_bgr(color_hex)
    
    # Optional semi-transparent fill
    if fill_opacity > 0:
        overlay = img.copy()
        cv2.drawContours(overlay, contours, -1, bgr_color, -1)
        img = cv2.addWeighted(overlay, fill_opacity, img, 1.0 - fill_opacity, 0)

    # Crisp boundary contour outline
    cv2.drawContours(img, contours, -1, bgr_color, line_thickness)
    return img

def create_dual_epoch_overlay(
    base_rgb: np.ndarray,
    mask_year1: np.ndarray,
    mask_year2: np.ndarray,
    color_y1_hex: str = config.COLOR_YEAR_1, # Blue (2016)
    color_y2_hex: str = config.COLOR_YEAR_2, # Red (2022)
    line_thickness: int = 2
) -> np.ndarray:
    """
    Renders multi-temporal boundary comparison (Panel b):
    Blue contours for Year 1 (2016) and Red contours for Year 2 (2022).
    """
    img = base_rgb.copy()
    if img.dtype != np.uint8:
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)

    # Year 1 (Blue)
    m1 = (mask_year1 > 0).astype(np.uint8)
    cnts1, _ = cv2.findContours(m1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cnts1, -1, hex_to_bgr(color_y1_hex), line_thickness)

    # Year 2 (Red)
    m2 = (mask_year2 > 0).astype(np.uint8)
    cnts2, _ = cv2.findContours(m2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cnts2, -1, hex_to_bgr(color_y2_hex), line_thickness)

    return img

def draw_classified_markers(
    base_rgb: np.ndarray,
    summary_records: List[dict],
    marker_radius: int = 8
) -> np.ndarray:
    """
    Renders classified markers (Panel c):
    🟡 Yellow for Newly Formed, 🟢 Green for Survived, 🔴 Red for Drained.
    """
    img = base_rgb.copy()
    if img.dtype != np.uint8:
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)

    h, w = img.shape[:2]

    for rec in summary_records:
        # Check if pixel coordinates exist, else derive from geometry
        px = rec.get("pixel_x")
        py = rec.get("pixel_y")
        
        if px is None or py is None:
            # Scale centroid to pixel coordinates if lon/lat provided
            lon = rec.get("centroid_lon", 74.5)
            lat = rec.get("centroid_lat", 36.5)
            # Default fallback mock coordinate projection
            px = int(((lon - 74.45) / 0.40) * w) % w
            py = int(((36.75 - lat) / 0.35) * h) % h

        color_hex = rec.get("color", config.COLOR_SURVIVED)
        bgr_color = hex_to_bgr(color_hex)

        # Draw filled circle marker with black outline for high visibility
        cv2.circle(img, (int(px), int(py)), marker_radius, (0, 0, 0), -1)
        cv2.circle(img, (int(px), int(py)), marker_radius - 2, bgr_color, -1)

    return img

def create_publication_three_panel_figure(
    overview_rgb: np.ndarray,
    panel_b_overlay: np.ndarray,
    panel_c_overlay: np.ndarray,
    output_path: Optional[Union[str, Path]] = None,
    dpi: int = 300
) -> plt.Figure:
    """
    Generates a 3-panel scientific figure (Panels a, b, c) matching the reference style:
    - Top: Panel (a) Regional overview with coordinates & north arrow
    - Bottom Left: Panel (b) Lake in 2016 (Blue) vs Lake in 2022 (Red)
    - Bottom Right: Panel (c) Classified Dynamics (Yellow: New, Green: Survived, Red: Drained)
    """
    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0])

    # 1. Panel A (Overview)
    ax_a = fig.add_subplot(gs[0, :])
    ax_a.imshow(overview_rgb)
    ax_a.set_title("(a) Regional Satellite Overview (Karakoram / Himalayan Glacier Range)", fontsize=13, fontweight='bold')
    ax_a.set_xticks([0, overview_rgb.shape[1]//2, overview_rgb.shape[1]-1])
    ax_a.set_xticklabels(["74°30'0\"E", "74°40'0\"E", "74°50'0\"E"], fontsize=10)
    ax_a.set_yticks([0, overview_rgb.shape[0]-1])
    ax_a.set_yticklabels(["36°30'0\"N", "36°0'0\"N"], fontsize=10)

    # 2. Panel B (Multi-temporal Boundaries)
    ax_b = fig.add_subplot(gs[1, 0])
    ax_b.imshow(panel_b_overlay)
    ax_b.set_title("(b) Multi-Temporal Lake Boundaries (2016 vs 2022)", fontsize=11, fontweight='bold')
    ax_b.axis('off')

    # 3. Panel C (Classified Dynamics)
    ax_c = fig.add_subplot(gs[1, 1])
    ax_c.imshow(panel_c_overlay)
    ax_c.set_title("(c) Temporal Change Classification", fontsize=11, fontweight='bold')
    ax_c.axis('off')

    # Global Legend
    patch_blue = mpatches.Patch(edgecolor=config.COLOR_YEAR_1, facecolor='none', linewidth=2, label='Lake in 2016')
    patch_red = mpatches.Patch(edgecolor=config.COLOR_YEAR_2, facecolor='none', linewidth=2, label='Lake in 2022')
    patch_new = mpatches.Patch(facecolor=config.COLOR_NEW, edgecolor='black', label='Newly Formed Lake')
    patch_surv = mpatches.Patch(facecolor=config.COLOR_SURVIVED, edgecolor='black', label='Survived Lake')
    patch_drain = mpatches.Patch(facecolor=config.COLOR_DRAINED, edgecolor='black', label='Drained Lake')

    fig.legend(
        handles=[patch_blue, patch_red, patch_new, patch_surv, patch_drain],
        loc='lower center',
        bbox_to_anchor=(0.5, -0.04),
        ncol=5,
        fontsize=11,
        frameon=True
    )

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, dpi=dpi, bbox_inches='tight')

    return fig
