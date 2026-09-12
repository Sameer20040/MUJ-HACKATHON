"""
Glacial Lake AI: Multi-Temporal Detection & GLOF Risk Monitor
Production Dashboard with Live Real Earth Satellite Mapping (Esri/Google Satellite),
Dynamic Vector Layers, 3-Panel Scientific Visuals, and GIS Downloads.
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
from html import escape
from io import BytesIO
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile
import os

import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import folium
from folium import plugins
from streamlit_folium import st_folium
import altair as alt
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.change_detection.classify_change import (
    classify_temporal_change,
    generate_change_summary_report,
)
from src.dataset.build_input_stack import create_synthetic_glacial_scene
from src.inference.overlay import create_dual_epoch_overlay, draw_classified_markers
from src.inference.run_inference import run_scene_inference
from src.postprocessing.vectorize import mask_to_geodataframe

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Glacial Lake AI: Multi-Temporal Detection & GLOF Risk Monitor",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom Dark Theme Styling ---
CUSTOM_CSS = """
<style>
    /* Dark Theme Core */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Top Header Bar */
    .dashboard-header {
        display: flex;
        align-items: center;
        gap: 16px;
        padding: 14px 22px;
        background-color: #121826;
        border-radius: 8px;
        border: 1px solid #1e293b;
        margin-bottom: 18px;
    }
    .header-logo {
        width: 38px;
        height: 38px;
        background: linear-gradient(135deg, #0ea5e9 0%, #2563eb 100%);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
        box-shadow: 0 2px 10px rgba(14, 165, 233, 0.4);
    }
    .dashboard-title {
        font-size: 24px;
        font-weight: 700;
        letter-spacing: -0.3px;
        color: #f8fafc;
        margin: 0;
        line-height: 1.2;
    }
    
    /* 3-Panel Scientific Frame Cards */
    .panel-box {
        background-color: #101626;
        border: 1px solid #1e293b;
        border-radius: 8px;
        overflow: hidden;
        margin-bottom: 12px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5);
    }
    .panel-topbar {
        padding: 10px 14px;
        background-color: #151c2e;
        border-bottom: 1px solid #1e293b;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .panel-label {
        font-size: 13px;
        font-weight: 600;
        color: #e2e8f0;
    }
    .tag-blue {
        color: #60a5fa;
        background: rgba(37, 99, 235, 0.25);
        padding: 2px 6px;
        border-radius: 3px;
        border: 1px solid #2563eb;
    }
    .tag-red {
        color: #f87171;
        background: rgba(220, 38, 38, 0.25);
        padding: 2px 6px;
        border-radius: 3px;
        border: 1px solid #dc2626;
    }
    .tag-yellow {
        color: #fde047;
        background: rgba(202, 138, 4, 0.25);
        padding: 2px 6px;
        border-radius: 3px;
        border: 1px solid #ca8a04;
    }
    .tag-green {
        color: #4ade80;
        background: rgba(22, 163, 74, 0.25);
        padding: 2px 6px;
        border-radius: 3px;
        border: 1px solid #16a34a;
    }
    
    /* Right Analytics Card */
    .right-card {
        background-color: #101626;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    }
    .right-card-title {
        font-size: 12px;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }
    .hero-stat {
        font-size: 34px;
        font-weight: 800;
        color: #38bdf8;
        letter-spacing: -0.5px;
        line-height: 1;
        margin-bottom: 10px;
    }
    .sub-stats-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
        font-size: 12px;
    }
    .sub-stat-val {
        font-weight: 700;
        font-size: 14px;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0b0f19;
        border-right: 1px solid #1e293b;
    }
    
    /* Block Container Padding */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

ASSETS_DIR = Path("data/benchmark_assets")
DEMO_DIR = Path("data/demo_samples")

def _shapefile_archive(gdf: gpd.GeoDataFrame) -> Optional[bytes]:
    """Builds a portable shapefile ZIP for GIS exports with unique <=10 character attribute names."""
    if gdf.empty:
        return None
    try:
        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "lake_change_classified.shp"
            export_gdf = gdf.copy()
            if export_gdf.crs is None:
                export_gdf.set_crs("EPSG:4326", inplace=True)
                
            clean_cols = {}
            used_names = set()
            for col in export_gdf.columns:
                if col == "geometry":
                    continue
                if "status" in col and "code" in col:
                    name = "stat_code"
                elif "status" in col:
                    name = "status"
                elif "change" in col and "pct" in col:
                    name = "chg_pct"
                elif "change" in col:
                    name = "chg_km2"
                elif "iou" in col:
                    name = "iou"
                elif "lon" in col:
                    name = "cent_lon"
                elif "lat" in col:
                    name = "cent_lat"
                elif "area" in col:
                    digits = "".join([ch for ch in col if ch.isdigit()])
                    name = f"area_{digits}" if digits else "area_km2"
                else:
                    name = col[:10]
                
                base_name = name[:8]
                counter = 1
                final_name = name[:10]
                while final_name in used_names:
                    final_name = f"{base_name}_{counter}"[:10]
                    counter += 1
                used_names.add(final_name)
                clean_cols[col] = final_name

            export_gdf.rename(columns=clean_cols, inplace=True)
            export_gdf.to_file(output_path, driver="ESRI Shapefile")
            buffer = BytesIO()
            with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
                for file_path in Path(directory).iterdir():
                    archive.write(file_path, file_path.name)
            return buffer.getvalue()
    except Exception:
        return None

def load_benchmark_data() -> Tuple[Any, Any, Any, pd.DataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Loads benchmark raster images and vector GIS polygon layers."""
    img_a_path = ASSETS_DIR / "panel_a.jpg"
    img_b_path = ASSETS_DIR / "panel_b.jpg"
    img_c_path = ASSETS_DIR / "panel_c.jpg"
    csv_path = ASSETS_DIR / "change_summary.csv"
    geojson_path = ASSETS_DIR / "lake_change_classified.geojson"
    shp16_path = ASSETS_DIR / "lakes_2016.geojson"
    shp22_path = ASSETS_DIR / "lakes_2022.geojson"

    img_a = Image.open(img_a_path) if img_a_path.exists() else None
    img_b = Image.open(img_b_path) if img_b_path.exists() else None
    img_c = Image.open(img_c_path) if img_c_path.exists() else None
    df = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
    gdf = gpd.read_file(geojson_path) if geojson_path.exists() else gpd.GeoDataFrame()
    gdf_16 = gpd.read_file(shp16_path) if shp16_path.exists() else gpd.GeoDataFrame()
    gdf_22 = gpd.read_file(shp22_path) if shp22_path.exists() else gpd.GeoDataFrame()

    return img_a, img_b, img_c, df, gdf, gdf_16, gdf_22

def render_real_earth_satellite_map(
    gdf_classified: gpd.GeoDataFrame,
    gdf_2016: Optional[gpd.GeoDataFrame] = None,
    gdf_2022: Optional[gpd.GeoDataFrame] = None,
    center_lat: float = 36.58,
    center_lon: float = 74.65,
    zoom: int = 11,
) -> folium.Map:
    """
    Renders an actual Real Earth Interactive Satellite GIS Map
    with Esri World Imagery, Google Satellite, and OpenStreetMap basemaps,
    along with vector polygon layers for 2016 Blue Boundaries, 2022 Red Boundaries,
    and Yellow/Green/Red classification markers.
    """
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles=None,
        control_scale=True,
    )

    # Esri High-Resolution World Imagery Satellite
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery (Maxar/Earthstar)",
        name="🛰️ Esri High-Res Satellite Earth",
        overlay=False,
        control=True,
    ).add_to(m)

    # Google Satellite Imagery
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google Satellite",
        name="🌍 Google Earth Satellite",
        overlay=False,
        control=True,
    ).add_to(m)

    # CartoDB Dark Matter
    folium.TileLayer(
        tiles="CartoDB dark_matter",
        name="🌑 CartoDB Dark Matter",
        overlay=False,
        control=True,
    ).add_to(m)

    # Determine if polygons have genuine geodetic coordinates (-180..180, -90..90)
    has_valid_geo = False
    if gdf_classified is not None and not gdf_classified.empty:
        try:
            pt = gdf_classified.iloc[0].geometry.centroid
            if -90.0 <= pt.y <= 90.0 and -180.0 <= pt.x <= 180.0 and (abs(pt.x) > 0.05 or abs(pt.y) > 0.05):
                has_valid_geo = True
        except Exception:
            has_valid_geo = False

    if has_valid_geo:
        # Vector Layer: Earlier Lake Boundaries (Blue Contours)
        if gdf_2016 is not None and not gdf_2016.empty:
            layer_2016 = folium.FeatureGroup(name="🟦 Lake Boundaries (Earlier Epoch)", show=True)
            for _, row in gdf_2016.iterrows():
                geom = row.geometry
                if geom.geom_type in ["Polygon", "MultiPolygon"]:
                    folium.GeoJson(
                        geom,
                        style_function=lambda x: {
                            "color": "#3b82f6",
                            "weight": 2.5,
                            "fillColor": "#60a5fa",
                            "fillOpacity": 0.25,
                        },
                        tooltip=f"Lake ID: {row.get('lake_id', 'T1')} | Area: {row.get('area_sqkm', 0.0):.2f} km²",
                    ).add_to(layer_2016)
            layer_2016.add_to(m)

        # Vector Layer: Later Lake Boundaries (Red Contours)
        if gdf_2022 is not None and not gdf_2022.empty:
            layer_2022 = folium.FeatureGroup(name="🟥 Lake Boundaries (Later Epoch)", show=True)
            for _, row in gdf_2022.iterrows():
                geom = row.geometry
                if geom.geom_type in ["Polygon", "MultiPolygon"]:
                    folium.GeoJson(
                        geom,
                        style_function=lambda x: {
                            "color": "#ef4444",
                            "weight": 2.5,
                            "fillColor": "#f87171",
                            "fillOpacity": 0.25,
                        },
                        tooltip=f"Lake ID: {row.get('lake_id', 'T2')} | Area: {row.get('area_sqkm', 0.0):.2f} km²",
                    ).add_to(layer_2022)
            layer_2022.add_to(m)

        # Vector Layer: Classified Dynamics Markers (Yellow, Green, Red)
        layer_markers = folium.FeatureGroup(name="📍 Classified Dynamics (Pins)", show=True)
        for _, row in gdf_classified.iterrows():
            pt = row.geometry.centroid
            status = row.get("status", "Survived")
            lake_id = row.get("lake_id", "Lake")
            a_cols = [c for c in row.index if str(c).startswith("area_") and str(c).endswith("_sqkm") and "change" not in str(c)]
            a16 = float(row.get(a_cols[0], 0.0)) if len(a_cols) > 0 else 0.0
            a22 = float(row.get(a_cols[1], 0.0)) if len(a_cols) > 1 else a16
            chg = float(row.get("area_change_pct", 0.0))

            color_map = {
                "Newly Formed": "#facc15",
                "Survived": "#22c55e",
                "Drained": "#ef4444",
            }
            marker_color = color_map.get(status, "#38bdf8")

            popup_html = f"""
            <div style="font-family: sans-serif; font-size: 12px; min-width: 170px;">
                <b style="font-size: 14px; color: #0f172a;">{lake_id}</b><br>
                <hr style="margin: 4px 0;">
                <b>Temporal Status:</b> <span style="color: {marker_color}; font-weight: bold;">{status}</span><br>
                <b>Area (Epoch 1):</b> {a16:.3f} km²<br>
                <b>Area (Epoch 2):</b> {a22:.3f} km²<br>
                <b>Area Change:</b> {chg:+.1f}%<br>
                <b>Coordinates:</b> ({pt.y:.4f}°N, {pt.x:.4f}°E)
            </div>
            """

            folium.CircleMarker(
                location=[pt.y, pt.x],
                radius=8,
                color="#ffffff",
                weight=1.5,
                fill=True,
                fill_color=marker_color,
                fill_opacity=0.95,
                popup=folium.Popup(popup_html, max_width=280),
                tooltip=f"{lake_id} — {status} ({chg:+.1f}%)",
            ).add_to(layer_markers)
        layer_markers.add_to(m)

    # Interactive GIS Plugins
    plugins.Fullscreen(position="topright").add_to(m)
    plugins.MeasureControl(position="bottomleft", primary_length_unit="kilometers", primary_area_unit="sqkilometers").add_to(m)
    plugins.MousePosition(position="bottomright", prefix="Coordinates (Lat, Lon): ").add_to(m)
    folium.LayerControl(position="topright", collapsed=False).add_to(m)

    return m

def main():
    # Initialize session state for persistent dashboard results
    if "analysis_state" not in st.session_state:
        b_img_a, b_img_b, b_img_c, b_df, b_gdf, b_gdf16, b_gdf22 = load_benchmark_data()
        st.session_state["analysis_state"] = {
            "title": "Himalayan Mountain Range (Hunza–Karakoram Valley)",
            "img_a": b_img_a,
            "img_b": b_img_b,
            "img_c": b_img_c,
            "summary_df": b_df,
            "classified_gdf": b_gdf,
            "gdf_16": b_gdf16,
            "gdf_22": b_gdf22,
            "epoch1_label": "Year 2016",
            "epoch2_label": "Year 2022",
            "is_custom": False,
            "center_lat": 36.58,
            "center_lon": 74.65,
            "zoom": 11,
        }

    # 1. Top Header Bar
    current_title = st.session_state["analysis_state"].get("title", "Glacial Lake Multi-Temporal Monitoring")
    st.markdown(
        f"""
        <div class="dashboard-header">
            <div class="header-logo">🏔️</div>
            <div>
                <h1 class="dashboard-title">Glacial Lake AI: Multi-Temporal Detection & GLOF Risk Monitor</h1>
                <div class="dashboard-subtitle">Active Region: <b>{escape(current_title)}</b> • Automated Sentinel Deep U-Net Segmentation & Boundary Tracking</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Left Control Sidebar
    st.sidebar.markdown("### 🛰️ Input Image & Analysis Mode")
    
    MODE_BENCHMARK = "🏔️ Benchmark — Karakoram (2016 vs 2022)"
    MODE_SPLIT = "📸 Split Side-by-Side Comparison (Left: T1, Right: T2)"
    MODE_DUAL = "📁 Upload Two Separate Images (Epoch 1 & Epoch 2)"
    MODE_SINGLE = "🔍 Upload Single Satellite Scene (Detection Only)"

    analysis_mode = st.sidebar.radio(
        "Choose Analysis Workflow",
        options=[MODE_BENCHMARK, MODE_SPLIT, MODE_DUAL, MODE_SINGLE],
        index=0,
    )

    t1_input = None
    t2_input = None
    single_input = None
    custom_ep1_name = "Epoch 1"
    custom_ep2_name = "Epoch 2"
    user_scene_title = ""

    if analysis_mode == MODE_BENCHMARK:
        st.sidebar.info("Using validated Sentinel-2 / Landsat Karakoram multi-temporal benchmark dataset.", icon="ℹ️")
        if st.sidebar.button("🔄 Reload Benchmark Dataset", width="stretch"):
            b_img_a, b_img_b, b_img_c, b_df, b_gdf, b_gdf16, b_gdf22 = load_benchmark_data()
            st.session_state["analysis_state"] = {
                "title": "Himalayan Mountain Range (Hunza–Karakoram Valley)",
                "img_a": b_img_a,
                "img_b": b_img_b,
                "img_c": b_img_c,
                "summary_df": b_df,
                "classified_gdf": b_gdf,
                "gdf_16": b_gdf16,
                "gdf_22": b_gdf22,
                "epoch1_label": "Year 2016",
                "epoch2_label": "Year 2022",
                "is_custom": False,
                "center_lat": 36.58,
                "center_lon": 74.65,
                "zoom": 11,
            }
            st.rerun()

    elif analysis_mode == MODE_SPLIT:
        st.sidebar.markdown("**Comparison Satellite Image**")
        st.sidebar.caption("Upload a single image with side-by-side time periods (e.g. Left = Earlier, Right = Later).")

        demo_checked = st.sidebar.checkbox(
            "⚡ Load Demo: King Salmon Alaska (Jan 22, 2024 vs Jan 26, 2025)",
            value=False,
            help="Loads the side-by-side satellite comparison showing seasonal ice/snow melt in King Salmon, Bristol Bay."
        )

        uploaded_comp = st.sidebar.file_uploader(
            "Upload Side-by-Side Image",
            type=["jpg", "jpeg", "png", "tif", "tiff"],
            key="side_by_side_upload",
        )

        col_l1, col_l2 = st.sidebar.columns(2)
        with col_l1:
            custom_ep1_name = st.text_input("Left Epoch Label", value="Jan 22, 2024" if demo_checked else "Epoch 1")
        with col_l2:
            custom_ep2_name = st.text_input("Right Epoch Label", value="Jan 26, 2025" if demo_checked else "Epoch 2")

        user_scene_title = st.sidebar.text_input(
            "Scene Location / Region",
            value="King Salmon, Bristol Bay (Alaska)" if demo_checked else "Custom Satellite Scene",
        )

        demo_file = DEMO_DIR / "king_salmon_comparison.jpg"
        if demo_file.exists():
            with open(demo_file, "rb") as df_fp:
                st.sidebar.download_button(
                    "💾 Save Sample Test Image to PC",
                    data=df_fp.read(),
                    file_name="sample_satellite_comparison.jpg",
                    mime="image/jpeg",
                    help="Click to download the test image to your computer so you can upload it into the box above."
                )

        if demo_checked:
            if demo_file.exists():
                comp_img = Image.open(demo_file)
                comp_arr = np.array(comp_img)
                mid = comp_arr.shape[1] // 2
                t1_input = comp_arr[:, :mid]
                t2_input = comp_arr[:, mid:]
        elif uploaded_comp is not None:
            try:
                uploaded_comp.seek(0)
            except Exception:
                pass
            pil_comp = Image.open(uploaded_comp)
            pil_comp.load()
            comp_arr = np.array(pil_comp.convert("RGB"))
            mid = comp_arr.shape[1] // 2
            t1_input = comp_arr[:, :mid]
            t2_input = comp_arr[:, mid:]

    elif analysis_mode == MODE_DUAL:
        st.sidebar.markdown("**Two Satellite Epochs**")
        st.sidebar.caption("Upload two aligned satellite images (Earlier and Later epoch).")
        
        f1_demo = DEMO_DIR / "king_salmon_2024_epoch1.jpg"
        f2_demo = DEMO_DIR / "king_salmon_2025_epoch2.jpg"
        if f1_demo.exists() and f2_demo.exists():
            c_d1, c_d2 = st.sidebar.columns(2)
            with c_d1:
                with open(f1_demo, "rb") as fp1:
                    st.download_button("💾 Save T1 Image", data=fp1.read(), file_name="sample_t1_2024.jpg", mime="image/jpeg")
            with c_d2:
                with open(f2_demo, "rb") as fp2:
                    st.download_button("💾 Save T2 Image", data=fp2.read(), file_name="sample_t2_2025.jpg", mime="image/jpeg")

        f1 = st.sidebar.file_uploader("Earlier Epoch Image (T1)", type=["jpg", "jpeg", "png", "tif", "tiff"], key="ep1_file")
        f2 = st.sidebar.file_uploader("Later Epoch Image (T2)", type=["jpg", "jpeg", "png", "tif", "tiff"], key="ep2_file")
        
        col_l1, col_l2 = st.sidebar.columns(2)
        with col_l1:
            custom_ep1_name = st.text_input("Epoch 1 Label", value="Epoch 1")
        with col_l2:
            custom_ep2_name = st.text_input("Epoch 2 Label", value="Epoch 2")
            
        user_scene_title = st.sidebar.text_input("Scene Region Name", value="Custom Multi-Temporal Region")

        if f1 is not None and f2 is not None:
            try:
                f1.seek(0)
            except Exception:
                pass
            try:
                f2.seek(0)
            except Exception:
                pass
            try:
                pil1 = Image.open(f1)
                pil1.load()
                t1_input = np.array(pil1.convert("RGB"))
            except Exception:
                t1_input = f1
            try:
                pil2 = Image.open(f2)
                pil2.load()
                t2_input = np.array(pil2.convert("RGB"))
            except Exception:
                t2_input = f2

    else: # MODE_SINGLE
        st.sidebar.markdown("**Single Satellite Scene**")
        st.sidebar.caption("Upload GeoTIFF or standard image for lake segmentation & boundary extraction.")
        f_single = st.sidebar.file_uploader("Satellite Scene", type=["jpg", "jpeg", "png", "tif", "tiff"], key="single_file")
        single_epoch_name = st.sidebar.text_input("Observation Date / Epoch", value="Target Scene")
        user_scene_title = st.sidebar.text_input("Scene Region Name", value="Delineated Scene")
        if f_single is not None:
            try:
                f_single.seek(0)
            except Exception:
                pass
            try:
                pil_img = Image.open(f_single)
                pil_img.load()  # Force full decode before stream closes
                single_input = np.array(pil_img.convert("RGB"))
            except Exception:
                single_input = f_single

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎛️ Model & Detection Controls")
    
    threshold = st.sidebar.slider(
        "Water Probability Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.45,
        step=0.05,
        help="Confidence threshold for classifying water pixels."
    )
    
    min_lake_pixels = st.sidebar.slider(
        "Minimum Lake Area (Pixels)",
        min_value=5,
        max_value=300,
        value=15,
        step=5,
        help="Filters out noise clusters smaller than this pixel count."
    )

    col_opt1, col_opt2 = st.sidebar.columns(2)
    with col_opt1:
        st.checkbox("Cloud Masking", value=True)
    with col_opt2:
        st.checkbox("Smooth Contours", value=True)

    run_analysis_clicked = st.sidebar.button(
        "🚀 Run Lake Analysis Pipeline",
        type="primary",
        width="stretch",
    )

    # 3. Process Execution
    if run_analysis_clicked:
        if analysis_mode in [MODE_SPLIT, MODE_DUAL]:
            if t1_input is None or t2_input is None:
                st.sidebar.error("⚠️ Please provide both earlier and later epoch imagery or enable the demo sample.")
            else:
                with st.spinner(f"Running Deep U-Net Lake Segmentation & Change Classification across {custom_ep1_name} & {custom_ep2_name}..."):
                    # Run inference on Epoch 1
                    res_t1 = run_scene_inference(t1_input, threshold=threshold, min_area_pixels=min_lake_pixels)
                    # Run inference on Epoch 2
                    res_t2 = run_scene_inference(t2_input, threshold=threshold, min_area_pixels=min_lake_pixels)

                    gdf_t1 = res_t1["geodataframe"]
                    gdf_t2 = res_t2["geodataframe"]

                    # Temporal change classification
                    classified_gdf, summary_df = classify_temporal_change(
                        gdf_t1, gdf_t2,
                        year1_label=custom_ep1_name,
                        year2_label=custom_ep2_name,
                    )

                    # Generate visualizations
                    disp_t2 = res_t2["display_rgb"]
                    disp_t1 = res_t1["display_rgb"]
                    
                    # Panel A: Satellite Overview (Epoch 2 or Composite)
                    img_a = Image.fromarray(disp_t2)
                    # Panel B: Multi-temporal boundary overlay (Blue = T1, Red = T2)
                    overlay_b = create_dual_epoch_overlay(disp_t2, res_t1["binary_mask"], res_t2["binary_mask"])
                    img_b = Image.fromarray(overlay_b)
                    # Panel C: Classified Status Markers
                    records_list = summary_df.to_dict("records") if not summary_df.empty else []
                    overlay_c = draw_classified_markers(disp_t2, records_list, marker_radius=7)
                    img_c = Image.fromarray(overlay_c)

                    # Determine map coordinates
                    center_lat = 58.6888 if "King Salmon" in user_scene_title else 36.58
                    center_lon = -156.6614 if "King Salmon" in user_scene_title else 74.65

                    st.session_state["analysis_state"] = {
                        "title": user_scene_title or f"Custom ({custom_ep1_name} vs {custom_ep2_name})",
                        "img_a": img_a,
                        "img_b": img_b,
                        "img_c": img_c,
                        "summary_df": summary_df,
                        "classified_gdf": classified_gdf,
                        "gdf_16": gdf_t1,
                        "gdf_22": gdf_t2,
                        "epoch1_label": custom_ep1_name,
                        "epoch2_label": custom_ep2_name,
                        "is_custom": True,
                        "center_lat": center_lat,
                        "center_lon": center_lon,
                        "zoom": 10 if "King Salmon" in user_scene_title else 11,
                    }
                    st.success(f"Analysis Complete: Detected {len(summary_df)} lake features across {custom_ep1_name} and {custom_ep2_name}!")
                    st.rerun()

        elif analysis_mode == MODE_SINGLE:
            if single_input is None:
                st.sidebar.error("⚠️ Please upload a satellite scene to run single-date segmentation.")
            else:
                with st.spinner("Running Deep U-Net Lake Segmentation on uploaded scene..."):
                    res_single = run_scene_inference(single_input, threshold=threshold, min_area_pixels=min_lake_pixels)
                    disp_img = res_single["display_rgb"]
                    cleaned_mask = res_single["binary_mask"]
                    gdf_single = res_single["geodataframe"]

                    img_a = Image.fromarray(disp_img)
                    overlay_single = create_boundary_overlay(disp_img, cleaned_mask, color_hex="#2ECC71", line_thickness=2)
                    img_b = Image.fromarray(overlay_single)
                    img_c = Image.fromarray(overlay_single)

                    summary_df = pd.DataFrame(gdf_single.drop(columns=["geometry"], errors="ignore"))
                    if not summary_df.empty:
                        summary_df["status"] = "Detected"
                        summary_df["area_change_sqkm"] = 0.0
                        summary_df["area_change_pct"] = 0.0

                    st.session_state["analysis_state"] = {
                        "title": user_scene_title or "Single Satellite Scene Delineation",
                        "img_a": img_a,
                        "img_b": img_b,
                        "img_c": img_c,
                        "summary_df": summary_df,
                        "classified_gdf": gdf_single,
                        "gdf_16": gpd.GeoDataFrame(),
                        "gdf_22": gdf_single,
                        "epoch1_label": "Baseline",
                        "epoch2_label": single_epoch_name or "Observed",
                        "is_custom": True,
                        "center_lat": 36.58,
                        "center_lon": 74.65,
                        "zoom": 11,
                    }
                    st.success(f"Delineation Complete: Extracted {len(gdf_single)} lake bodies!")
                    st.rerun()

    # Retrieve current active state
    state = st.session_state["analysis_state"]
    active_img_a = state["img_a"]
    active_img_b = state["img_b"]
    active_img_c = state["img_c"]
    active_df = state["summary_df"]
    active_gdf = state["classified_gdf"]
    active_gdf16 = state["gdf_16"]
    active_gdf22 = state["gdf_22"]
    ep1_lbl = state.get("epoch1_label", "Year 2016")
    ep2_lbl = state.get("epoch2_label", "Year 2022")
    active_title = state.get("title", "Satellite Scene")

    # 4. View Mode Switcher
    view_mode = st.radio(
        "Display Mode",
        options=[
            "🖼️ 3-Panel Scientific Visualizer (Panels A, B, C)",
            "🌐 Live Real Earth Satellite Map (Interactive GIS Globe)",
        ],
        index=0,
        horizontal=True,
        label_visibility="collapsed"
    )

    # 5. Main 2-Column Section (Visualizer on Left, Analytics on Right)
    col_main, col_analytics = st.columns([3.15, 1.15], gap="medium")

    with col_main:
        if "Live Real Earth Satellite Map" in view_mode:
            st.markdown(
                f"""
                <div class="panel-box">
                    <div class="panel-topbar">
                        <div class="panel-label">🛰️ Real Earth Interactive Satellite Map • {escape(active_title)}</div>
                        <div style="font-size: 11px; color: #94a3b8;">Layer Control (top right) toggles multi-epoch boundaries and GIS markers</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            earth_map = render_real_earth_satellite_map(
                active_gdf,
                gdf_2016=active_gdf16,
                gdf_2022=active_gdf22,
                center_lat=state.get("center_lat", 36.58),
                center_lon=state.get("center_lon", 74.65),
                zoom=state.get("zoom", 11),
            )
            st_folium(earth_map, width=1100, height=530, key="real_earth_map")

        else:
            # 3 Panels Side-by-Side matching scientific publication style
            p1, p2, p3 = st.columns(3, gap="small")

            with p1:
                st.markdown(
                    f"""
                    <div class="panel-box">
                        <div class="panel-topbar">
                            <div class="panel-label">A. Satellite Overview ({escape(ep2_lbl)})</div>
                            <div style="color: #64748b; font-size: 14px;">⋮</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                if active_img_a is not None:
                    st.image(active_img_a, width="stretch")

            with p2:
                st.markdown(
                    f"""
                    <div class="panel-box">
                        <div class="panel-topbar">
                            <div class="panel-label">B. Multi-Temporal Boundaries ({escape(ep1_lbl)} vs. {escape(ep2_lbl)})</div>
                        </div>
                        <div style="background-color: #151c2e; padding: 4px 14px 8px 14px; display: flex; gap: 8px;">
                            <span class="tag-blue">🟦 {escape(ep1_lbl)}</span>
                            <span class="tag-red">🟥 {escape(ep2_lbl)}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                if active_img_b is not None:
                    st.image(active_img_b, width="stretch")

            with p3:
                st.markdown(
                    """
                    <div class="panel-box">
                        <div class="panel-topbar">
                            <div class="panel-label">C. Classified Status Markers</div>
                        </div>
                        <div style="background-color: #151c2e; padding: 4px 14px 8px 14px; display: flex; gap: 6px;">
                            <span class="tag-yellow">🟡 Newly Formed</span>
                            <span class="tag-green">🟢 Survived</span>
                            <span class="tag-red">🔴 Drained</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                if active_img_c is not None:
                    st.image(active_img_c, width="stretch")

    with col_analytics:
        # Dynamic Statistics Calculation
        tot_inc = 0.0
        tot_dec = 0.0
        net_pct = 0.0
        total_lakes = len(active_df) if active_df is not None else 0

        if active_df is not None and not active_df.empty and "area_change_sqkm" in active_df.columns:
            tot_inc = float(active_df[active_df["area_change_sqkm"] > 0]["area_change_sqkm"].sum())
            tot_dec = float(active_df[active_df["area_change_sqkm"] < 0]["area_change_sqkm"].sum())
            net_chg = tot_inc + tot_dec
            
            # Find base area column
            area_cols = [c for c in active_df.columns if c.startswith("area_") and c.endswith("_sqkm") and "change" not in c]
            base_area = float(active_df[area_cols[0]].sum()) if area_cols else 1.0
            net_pct = (net_chg / (base_area + 1e-6)) * 100.0 if base_area > 0 else 0.0

        # 1. Total Surface Water Change Card
        hero_color = "#34d399" if net_pct >= 0 else "#f87171"
        st.markdown(
            f"""
            <div class="right-card">
                <div class="right-card-title">
                    <span>Total Surface Water Change</span>
                    <span style="color: #64748b; font-size: 11px;">{total_lakes} lakes</span>
                </div>
                <div class="hero-stat" style="color: {hero_color};">{net_pct:+.1f}%</div>
                <div class="sub-stats-grid">
                    <div>
                        <div style="color: #64748b; font-size: 11px;">Water Increase</div>
                        <div class="sub-stat-val" style="color: #34d399;">+{tot_inc:.3f} km²</div>
                    </div>
                    <div>
                        <div style="color: #64748b; font-size: 11px;">Area Decrease</div>
                        <div class="sub-stat-val" style="color: #f87171;">{tot_dec:.3f} km²</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # 2. Lake Area Distribution Histogram Card
        st.markdown(
            """
            <div class="right-card">
                <div class="right-card-title">
                    <span>Lake Area Distribution</span>
                </div>
            """,
            unsafe_allow_html=True
        )
        if active_df is not None and not active_df.empty:
            chart_df = active_df.copy()
            # Pick a suitable area column for histogram
            area_col = None
            for c in chart_df.columns:
                if c.startswith("area_") and c.endswith("_sqkm") and "change" not in c:
                    area_col = c
            if area_col is None and "area_sqkm" in chart_df.columns:
                area_col = "area_sqkm"

            if area_col is not None and area_col in chart_df.columns:
                bar_chart = (
                    alt.Chart(chart_df)
                    .mark_bar(color="#38bdf8", cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
                    .encode(
                        x=alt.X(f"{area_col}:Q", bin=alt.Bin(maxbins=8), title="Lake Area (km²)", axis=alt.Axis(labelColor="#94a3b8", titleColor="#94a3b8")),
                        y=alt.Y("count():Q", title="Lake Count", axis=alt.Axis(labelColor="#94a3b8", titleColor="#94a3b8")),
                        tooltip=[alt.Tooltip(f"{area_col}:Q", title="Area"), alt.Tooltip("count():Q", title="Count")]
                    )
                    .properties(height=125)
                    .configure_view(strokeOpacity=0)
                )
                st.altair_chart(bar_chart, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

        # 3. GIS Downloads Card
        st.markdown(
            """
            <div class="right-card">
                <div class="right-card-title">
                    <span>Export GIS Layers</span>
                </div>
            """,
            unsafe_allow_html=True
        )
        if active_gdf is not None and not active_gdf.empty:
            shp_zip = _shapefile_archive(active_gdf) or b""
            st.download_button(
                "📥 Shapefile (.zip)",
                data=shp_zip,
                file_name=f"{active_title.lower().replace(' ', '_')}_lakes.zip",
                mime="application/zip",
                width="stretch",
            )
            st.download_button(
                "📥 GeoJSON Layer",
                data=active_gdf.to_json(),
                file_name=f"{active_title.lower().replace(' ', '_')}_lakes.geojson",
                mime="application/geo+json",
                width="stretch",
            )
            if active_df is not None:
                st.download_button(
                    "📥 CSV Lake Inventory",
                    data=active_df.to_csv(index=False).encode('utf-8'),
                    file_name=f"{active_title.lower().replace(' ', '_')}_summary.csv",
                    mime="text/csv",
                    width="stretch",
                )
        st.markdown("</div>", unsafe_allow_html=True)

    # 6. Bottom Table: Full Glacial Lake Inventory
    st.markdown("---")
    st.markdown(f"### 📋 Glacial Lake Attribute Inventory • {escape(active_title)}")
    if active_df is not None and not active_df.empty:
        # Dynamic Column Configurations
        col_cfg = {
            "lake_id": st.column_config.TextColumn("Lake ID", pinned=True),
            "status": st.column_config.TextColumn("Temporal Classification"),
            "area_change_sqkm": st.column_config.NumberColumn("Net Change (km²)", format="%+.3f"),
            "area_change_pct": st.column_config.NumberColumn("Change (%)", format="%+.1f%%"),
            "centroid_lon": st.column_config.NumberColumn("Lon / X", format="%.4f"),
            "centroid_lat": st.column_config.NumberColumn("Lat / Y", format="%.4f"),
        }
        for col in active_df.columns:
            if col.startswith("area_") and col.endswith("_sqkm") and "change" not in col:
                digits = "".join([ch for ch in col if ch.isdigit()])
                label = f"Area {digits} (km²)" if digits else f"Area ({col}) (km²)"
                col_cfg[col] = st.column_config.NumberColumn(label, format="%.3f")

        clean_table = active_df.drop(columns=["geometry", "color", "status_code", "pixel_x", "pixel_y"], errors="ignore")
        st.dataframe(clean_table, column_config=col_cfg, width="stretch")

if __name__ == "__main__":
    main()
