"""
Sample dataset and baseline product generator.
Creates sample GeoTIFF satellite scenes, exports benchmark Shapefiles,
and renders publication maps.
"""

from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_bounds
import geopandas as gpd
from PIL import Image

from src.utils.config import config
from src.utils.geo_utils import write_geotiff
from src.dataset.build_input_stack import create_synthetic_glacial_scene
from src.postprocessing.vectorize import mask_to_geodataframe, export_mask_to_shapefile
from src.change_detection.classify_change import classify_temporal_change, generate_change_summary_report
from src.inference.overlay import (
    create_dual_epoch_overlay,
    draw_classified_markers,
    create_publication_three_panel_figure,
)

def generate_benchmark_products():
    config.ensure_directories()
    print("Generating Benchmark Products (2016 vs 2022)...")

    # 1. Generate 2016 and 2022 multi-band scenes
    stack_16, mask_16, prof_16 = create_synthetic_glacial_scene(height=1024, width=1024, seed=42, epoch=2016)
    stack_22, mask_22, prof_22 = create_synthetic_glacial_scene(height=1024, width=1024, seed=100, epoch=2022)

    # 2. Save GeoTIFFs
    tif_16_path = config.RAW_DATA_DIR / "sentinel2" / "2016" / "sentinel2_2016.tif"
    tif_22_path = config.RAW_DATA_DIR / "sentinel2" / "2022" / "sentinel2_2022.tif"
    write_geotiff(stack_16, tif_16_path, prof_16)
    write_geotiff(stack_22, tif_22_path, prof_22)
    print(f"Saved: {tif_16_path}")
    print(f"Saved: {tif_22_path}")

    # 3. Vectorize and export individual shapefiles
    shp_16_path = config.SHAPEFILES_DIR / "lakes_2016.shp"
    shp_22_path = config.SHAPEFILES_DIR / "lakes_2022.shp"
    gdf_16 = export_mask_to_shapefile(mask_16, shp_16_path, transform=prof_16["transform"], crs=prof_16["crs"])
    gdf_22 = export_mask_to_shapefile(mask_22, shp_22_path, transform=prof_22["transform"], crs=prof_22["crs"])
    print(f"Exported Shapefiles: {shp_16_path}, {shp_22_path}")

    # 4. Multi-temporal change classification
    classified_gdf, summary_df = classify_temporal_change(gdf_16, gdf_22, year1_label="2016", year2_label="2022")
    
    classified_shp_path = config.SHAPEFILES_DIR / "lake_change_classified.shp"
    classified_geojson_path = config.SHAPEFILES_DIR / "lake_change_classified.geojson"
    classified_gdf.to_file(classified_shp_path, driver="ESRI Shapefile")
    classified_gdf.to_file(classified_geojson_path, driver="GeoJSON")

    summary_csv_path = config.STATS_DIR / "change_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Saved Change Stats: {summary_csv_path}")

    # 5. Generate 3-Panel Scientific Visualization
    rgb_22 = np.stack([stack_22[2], stack_22[1], stack_22[0]], axis=-1)
    rgb_22 = np.clip(rgb_22 * 255.0, 0, 255).astype(np.uint8)

    panel_b = create_dual_epoch_overlay(rgb_22, mask_16, mask_22)
    panel_c = draw_classified_markers(rgb_22, summary_df.to_dict(orient="records"))
    
    map_png_path = config.MAPS_DIR / "overlay_visualization.png"
    create_publication_three_panel_figure(rgb_22, panel_b, panel_c, output_path=map_png_path, dpi=200)
    print(f"Generated 3-Panel Scientific Map: {map_png_path}")

    # Save sample JPG images for quick upload testing in dashboard
    sample_jpg_path = config.USER_UPLOADS_DIR / "input" / "sample_satellite_rgb.jpg"
    Image.fromarray(rgb_22).save(sample_jpg_path, quality=95)
    print(f"Saved Test Sample Image: {sample_jpg_path}")

    stats = generate_change_summary_report(summary_df)
    print("Baseline Generation Complete!")
    print("Summary:", stats)

if __name__ == "__main__":
    generate_benchmark_products()
