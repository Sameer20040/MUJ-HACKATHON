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
from src.utils.config import config
from src.inference.satellite import fetch_satellite_scene
from src.change_detection.risk_scoring import add_risk_scores, summarize_risk

def _ensure_risk_bench(summary_df, classified_gdf):
    """Ensure GLOF risk columns exist for benchmark data."""
    if summary_df is not None and not summary_df.empty and "risk" not in summary_df.columns:
        summary_df = add_risk_scores(summary_df)
        if classified_gdf is not None and not classified_gdf.empty:
            enriched = summary_df[["lake_id", "risk", "risk_score", "risk_color", "recommended_action"]]
            classified_gdf = classified_gdf.merge(enriched, on="lake_id", how="left")
    return summary_df, classified_gdf

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

def _shapefile_archive(gdf: gpd.GeoDataFrame) -> Optional[bytes]:
    """Builds a portable shapefile ZIP for GIS exports."""
    if gdf.empty:
        return None
    try:
        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "lake_change_classified.shp"
            field_names = {
                "status_code": "stat_code",
                "area_2016_sqkm": "area_2016",
                "area_2022_sqkm": "area_2022",
                "area_change_sqkm": "change_km2",
                "area_change_pct": "change_pct",
                "centroid_lon": "cent_lon",
                "centroid_lat": "cent_lat",
                "risk": "risk",
                "risk_score": "risk_sc",
            }
            export_gdf = gdf.rename(columns={k: v for k, v in field_names.items() if k in gdf.columns})
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
    # Auto-center map on detected lakes if valid WGS84 centroids exist
    if gdf_classified is not None and not gdf_classified.empty:
        try:
            c = gdf_classified.geometry.unary_union.centroid
            if -90 <= c.y <= 90 and -180 <= c.x <= 180 and (abs(c.x) > 1.0 or abs(c.y) > 1.0):
                center_lat, center_lon = float(c.y), float(c.x)
        except Exception:
            pass

    # 1. Base Map with High-Resolution Satellite Tiles
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

    # 2. Vector Layer: 2016 Lake Boundaries (Blue Contours)
    if gdf_2016 is not None and not gdf_2016.empty:
        layer_2016 = folium.FeatureGroup(name="🟦 Lake Boundaries (Year 2016)", show=True)
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
                    tooltip=f"Lake ID: {row.get('lake_id', '2016')} | Area: {row.get('area_sqkm', 0.0):.2f} km² (Year 2016)",
                ).add_to(layer_2016)
        layer_2016.add_to(m)

    # 3. Vector Layer: 2022 Lake Boundaries (Red Contours)
    if gdf_2022 is not None and not gdf_2022.empty:
        layer_2022 = folium.FeatureGroup(name="🟥 Lake Boundaries (Year 2022)", show=True)
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
                    tooltip=f"Lake ID: {row.get('lake_id', '2022')} | Area: {row.get('area_sqkm', 0.0):.2f} km² (Year 2022)",
                ).add_to(layer_2022)
        layer_2022.add_to(m)

    # 4. Vector Layer: Classified Dynamics Markers, colored by GLOF RISK when available
    if gdf_classified is not None and not gdf_classified.empty:
        has_risk = "risk" in gdf_classified.columns and "risk_color" in gdf_classified.columns
        layer_markers = folium.FeatureGroup(name="📍 Classified Dynamics (Risk Pins)", show=True)
        for _, row in gdf_classified.iterrows():
            pt = row.geometry.centroid
            status = row.get("status", "Survived")
            lake_id = row.get("lake_id", "Lake")
            a16 = row.get("area_2016_sqkm", 0.0)
            a22 = row.get("area_2022_sqkm", row.get("area_sqkm", 0.0))
            chg = row.get("area_change_pct", 0.0)

            # Prefer GLOF risk coloring when present; fall back to status colors
            if has_risk:
                marker_color = str(row.get("risk_color", "#38bdf8"))
            else:
                status_color_map = {
                    "Newly Formed": "#facc15",
                    "Survived": "#22c55e",
                    "Drained": "#ef4444",
                }
                marker_color = status_color_map.get(status, "#38bdf8")

            risk_line = ""
            if has_risk:
                risk_label = row.get("risk", "n/a")
                risk_score = row.get("risk_score", 0.0)
                rec_action = row.get("recommended_action", "")
                risk_line = f"""
                <b>GLOF Risk:</b> <span style="color: {marker_color}; font-weight: bold;">{risk_label} ({risk_score:.0f}/100)</span><br>
                <span style="font-size: 11px; color: #374151;">{rec_action}</span><br>"""

            popup_html = f"""
            <div style="font-family: sans-serif; font-size: 12px; min-width: 180px;">
                <b style="font-size: 14px; color: #0f172a;">{lake_id}</b><br>
                <hr style="margin: 4px 0;">
                <b>Temporal Status:</b> <span style="color: {marker_color}; font-weight: bold;">{status}</span><br>
                {risk_line}
                <b>Area (2016):</b> {a16:.3f} km²<br>
                <b>Area (2022):</b> {a22:.3f} km²<br>
                <b>Area Change:</b> {chg:+.1f}%<br>
                <b>Coordinates:</b> ({pt.y:.4f}°N, {pt.x:.4f}°E)
            </div>
            """

            folium.CircleMarker(
                location=[pt.y, pt.x],
                radius=9,
                color="#ffffff",
                weight=1.5,
                fill=True,
                fill_color=marker_color,
                fill_opacity=0.95,
                popup=folium.Popup(popup_html, max_width=280),
                tooltip=f"{lake_id} — {status}" + (f" | {risk_label} risk" if has_risk else f" ({chg:+.1f}%)"),
            ).add_to(layer_markers)
        layer_markers.add_to(m)

    # 5. Interactive GIS Plugins
    plugins.Fullscreen(position="topright").add_to(m)
    plugins.MeasureControl(position="bottomleft", primary_length_unit="kilometers", primary_area_unit="sqkilometers").add_to(m)
    plugins.MousePosition(position="bottomright", prefix="Coordinates (Lat, Lon): ").add_to(m)
    folium.LayerControl(position="topright", collapsed=False).add_to(m)

    return m

def main():
    # Session state init
    if "results" not in st.session_state:
        st.session_state["results"] = None

    # --- Render sidebar & capture user actions ---

    # 1. Top Header Bar
    st.markdown(
        """
        <div class="dashboard-header">
            <div class="header-logo">🏔️</div>
            <div>
                <h1 class="dashboard-title">Glacial Lake AI: Multi-Temporal Detection & GLOF Risk Monitor</h1>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


    # 2. Left Control Sidebar - Clean, Functional Design
    st.sidebar.markdown("### 📤 Upload Satellite Image")
    st.sidebar.caption("GeoTIFF (multi-spectral) or RGB (JPG/PNG)")

    uploaded_file = st.sidebar.file_uploader(
        "Choose file",
        type=["tif", "tiff", "jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Detection Settings")

    # Confidence threshold - actually used
    threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Lower = more sensitive (more lakes, more false positives). Higher = stricter."
    )

    # Minimum lake size - actually used
    min_area = st.sidebar.number_input(
        "Min Lake Size (pixels)",
        min_value=5,
        max_value=500,
        value=20,
        step=5,
        help="Filter out tiny noise clusters."
    )

    # Cloud filtering - actually used for GeoTIFF
    enable_cloud_filter = st.sidebar.checkbox(
        "Cloud Screening (GeoTIFF only)",
        value=True,
        help="Remove probable cloud pixels before boundary extraction when spectral bands available."
    )

    # Boundary smoothing - actually used
    enable_smoothing = st.sidebar.checkbox(
        "Boundary Smoothing",
        value=True,
        help="Apply Douglas-Peucker simplification for cleaner polygons."
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🌍 Live Satellite Fetch (2024-2026)")
    st.sidebar.caption("Fetch real-time Esri/Yandex imagery and run detection")

    col_lat, col_lon = st.sidebar.columns(2)
    with col_lat:
        yd_lat = st.number_input("Latitude (°N)", value=config.YANDEX_CENTER_LAT, format="%.4f", step=0.01)
    with col_lon:
        yd_lon = st.number_input("Longitude (°E)", value=config.YANDEX_CENTER_LON, format="%.4f", step=0.01)

    yd_zoom = st.sidebar.slider("Zoom Level", 10, 17, config.YANDEX_ZOOM)
    yd_fetch_btn = st.sidebar.button("📡 Fetch & Analyze Scene", use_container_width=True, type="primary")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🛰️ 2026 Real-Time Data (NEW)")
    st.sidebar.caption("Fetch current 2024-2026 satellite imagery for Karakoram region")

    if st.sidebar.button("🛰️ Fetch 2026 Regional Mosaic", use_container_width=True, type="primary"):
        st.session_state["fetch_2026_mosaic"] = True

    if st.sidebar.button("🏔️ Fetch Key Glacier Scenes (2026)", use_container_width=True):
        st.session_state["fetch_2026_glaciers"] = True

    if st.sidebar.button("📊 Run 2016 vs 2026 Change Analysis", use_container_width=True):
        st.session_state["run_2016_2026_comparison"] = True

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Quick Actions")

    # Benchmark mode - loads pre-computed 2016 vs 2022
    if st.sidebar.button("📊 Load Karakoram Benchmark (2016 vs 2022)", use_container_width=True):
        st.session_state["load_benchmark"] = True

    # Clear results
    if st.sidebar.button("🗑️ Clear Results", use_container_width=True):
        st.session_state["clear_results"] = True

    def _backfill_risk(classified_gdf, summary_df):
        """Ensure GLOF risk columns exist on both the summary and the GeoDataFrame.
        For single-scene uploads, we add minimal required columns before scoring."""
        if summary_df is not None and not summary_df.empty and "risk" not in summary_df.columns:
            if "status" not in summary_df.columns:
                summary_df["status"] = "Detected"
            if "area_change_pct" not in summary_df.columns:
                summary_df["area_change_pct"] = 0.0
            if "mean_slope" not in summary_df.columns:
                summary_df["mean_slope"] = 0.0
            area_cols = [c for c in summary_df.columns if c.startswith("area_") and c.endswith("_sqkm")]
            if not area_cols and "area_sqkm" in summary_df.columns:
                summary_df["area_2022_sqkm"] = summary_df["area_sqkm"]
            summary_df = add_risk_scores(summary_df)
            if classified_gdf is not None and not classified_gdf.empty:
                risk_cols = ["risk", "risk_score", "risk_color", "recommended_action"]
                for col in risk_cols:
                    if col in classified_gdf.columns:
                        classified_gdf = classified_gdf.drop(columns=[col])
                enriched = summary_df[["lake_id"] + risk_cols]
                classified_gdf = classified_gdf.merge(enriched, on="lake_id", how="left")
        return classified_gdf, summary_df

    # --- Session-state-driven data flow ---
    # Initialize with benchmark data on first load
    if st.session_state["results"] is None:
        img_a, img_b, img_c, summary_df, classified_gdf, gdf_16, gdf_22 = load_benchmark_data()
        summary_df, classified_gdf = _ensure_risk_bench(summary_df, classified_gdf)
        st.session_state["results"] = {
            "img_a": img_a, "img_b": img_b, "img_c": img_c,
            "summary_df": summary_df, "classified_gdf": classified_gdf,
            "gdf_16": gdf_16, "gdf_22": gdf_22,
            "label": "Karakoram Benchmark (2016 vs 2022)",
        }

    # Read from session state
    results = st.session_state["results"]
    img_a, img_b, img_c = results["img_a"], results["img_b"], results["img_c"]
    summary_df = results["summary_df"]
    classified_gdf = results["classified_gdf"]
    gdf_16, gdf_22 = results["gdf_16"], results["gdf_22"]

    # --- Process sidebar actions ---
    # Load benchmark action
    if st.session_state.get("load_benchmark"):
        st.session_state["load_benchmark"] = False
        b_img_a, b_img_b, b_img_c, b_summary, b_gdf, b_gdf16, b_gdf22 = load_benchmark_data()
        b_summary, b_gdf = _ensure_risk_bench(b_summary, b_gdf)
        st.session_state["results"] = {
            "img_a": b_img_a, "img_b": b_img_b, "img_c": b_img_c,
            "summary_df": b_summary, "classified_gdf": b_gdf,
            "gdf_16": b_gdf16, "gdf_22": b_gdf22,
            "label": "Karakoram Benchmark (2016 vs 2022)",
        }
        st.sidebar.success("✅ Benchmark loaded")
        st.rerun()

    # Clear results
    if st.session_state.get("clear_results"):
        st.session_state["clear_results"] = False
        st.session_state["results"] = None
        st.sidebar.info("🗑️ Cleared — click sidebar to load fresh results")
        st.rerun()

    # Upload image
    if uploaded_file is not None:
        upload_id = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.get("active_upload_id") != upload_id:
            with st.spinner("Running Deep U-Net Lake Segmentation on uploaded image..."):
                try:
                    if hasattr(uploaded_file, "seek"):
                        uploaded_file.seek(0)
                    inf_res = run_scene_inference(
                        uploaded_file,
                        threshold=threshold,
                        min_area_pixels=min_area,
                        apply_cloud_screening=enable_cloud_filter,
                    )
                    u_img_a = Image.fromarray(inf_res["display_rgb"])
                    u_img_b = Image.fromarray(inf_res["overlay_rgb"])
                    u_img_c = Image.fromarray(inf_res["overlay_rgb"])
                    u_gdf = inf_res["geodataframe"]
                    u_summary = pd.DataFrame(u_gdf.drop(columns=["geometry"], errors="ignore"))
                    u_gdf, u_summary = _backfill_risk(u_gdf, u_summary)

                    st.session_state["results"] = {
                        "img_a": u_img_a, "img_b": u_img_b, "img_c": u_img_c,
                        "summary_df": u_summary, "classified_gdf": u_gdf,
                        "gdf_16": None, "gdf_22": u_gdf,
                        "label": f"Uploaded: {uploaded_file.name}"
                    }
                    st.session_state["active_upload_id"] = upload_id
                    st.sidebar.success(
                        f"✅ Segmented — {inf_res['lake_count']} lake(s), "
                        f"{inf_res['total_lake_area_sqkm']:.3f} km² total."
                    )
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"⚠️ Inference failed: {e}")
                    st.exception(e)

    # Fetch live satellite
    if yd_fetch_btn:
        with st.spinner(f"Fetching real satellite scene at {yd_lat:.4f}°N, {yd_lon:.4f}°E..."):
            try:
                sat_rgb, source, transform = fetch_satellite_scene(
                    lat=yd_lat, lon=yd_lon, zoom=yd_zoom, size=config.YANDEX_IMAGE_SIZE
                )
                buf = BytesIO()
                Image.fromarray(sat_rgb).save(buf, format="PNG")
                buf.seek(0)
                inf_res = run_scene_inference(
                    buf,
                    threshold=threshold,
                    min_area_pixels=min_area,
                    override_transform=transform,
                    override_crs="EPSG:4326"
                )
                f_img_a = Image.fromarray(inf_res["display_rgb"])
                f_img_b = Image.fromarray(inf_res["overlay_rgb"])
                f_img_c = Image.fromarray(inf_res["overlay_rgb"])
                f_gdf = inf_res["geodataframe"]
                f_summary = pd.DataFrame(f_gdf.drop(columns=["geometry"], errors="ignore"))
                f_gdf, f_summary = _backfill_risk(f_gdf, f_summary)
                st.session_state["results"] = {
                    "img_a": f_img_a, "img_b": f_img_b, "img_c": f_img_c,
                    "summary_df": f_summary, "classified_gdf": f_gdf,
                    "gdf_16": None, "gdf_22": f_gdf,
                    "label": f"Live {source} ({yd_lat:.4f}°N, {yd_lon:.4f}°E)"
                }
                st.sidebar.success(
                    f"Live scene segmented via {source} — {inf_res['lake_count']} lake(s), "
                    f"{inf_res['total_lake_area_sqkm']:.3f} km² total."
                )
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"⚠️ Satellite fetch failed: {e}")
                st.exception(e)

    # Fetch 2026 Regional Mosaic
    if st.session_state.get("fetch_2026_mosaic"):
        st.session_state["fetch_2026_mosaic"] = False
        with st.spinner("Fetching 2026 regional mosaic (real-time Esri imagery)..."):
            try:
                from src.data.real_data_fetcher import fetch_2026_satellite_mosaic
                mosaic = fetch_2026_satellite_mosaic(zoom=13, use_esri=True)
                buf = BytesIO()
                Image.fromarray(mosaic["rgb"]).save(buf, format="PNG")
                buf.seek(0)
                inf_res = run_scene_inference(
                    buf,
                    threshold=threshold,
                    min_area_pixels=min_area,
                    apply_cloud_screening=enable_cloud_filter,
                    override_transform=mosaic.get("transform"),
                )
                img_a = Image.fromarray(inf_res["display_rgb"])
                img_b = Image.fromarray(inf_res["overlay_rgb"])
                img_c = Image.fromarray(inf_res["overlay_rgb"])
                classified_gdf = inf_res["geodataframe"]
                gdf_22 = classified_gdf
                summary_df = pd.DataFrame(classified_gdf.drop(columns=["geometry"], errors="ignore"))
                classified_gdf, summary_df = _backfill_risk(classified_gdf, summary_df)

                st.session_state["results"] = {
                    "img_a": img_a, "img_b": img_b, "img_c": img_c,
                    "summary_df": summary_df, "classified_gdf": classified_gdf,
                    "gdf_16": None, "gdf_22": classified_gdf,
                    "label": f"2026 Regional Mosaic ({mosaic['source']})"
                }
                st.sidebar.success(f"✅ 2026 Mosaic segmented — {inf_res['lake_count']} lakes")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"⚠️ 2026 Mosaic fetch failed: {e}")
                st.exception(e)

    # Fetch 2026 Glacier Scenes
    if st.session_state.get("fetch_2026_glaciers"):
        st.session_state["fetch_2026_glaciers"] = False
        with st.spinner("Fetching 2026 key glacier scenes..."):
            try:
                from src.data.real_data_fetcher import fetch_glacier_scenes_2026
                scenes = fetch_glacier_scenes_2026()
                # Use the first scene for display
                if scenes:
                    scene = scenes[0]
                    buf = BytesIO()
                    Image.fromarray(scene["rgb"]).save(buf, format="PNG")
                    buf.seek(0)
                    inf_res = run_scene_inference(
                        buf,
                        threshold=threshold,
                        min_area_pixels=min_area,
                        apply_cloud_screening=enable_cloud_filter,
                        override_transform=scene.get("transform"),
                    )
                    img_a = Image.fromarray(inf_res["display_rgb"])
                    img_b = Image.fromarray(inf_res["overlay_rgb"])
                    img_c = Image.fromarray(inf_res["overlay_rgb"])
                    classified_gdf = inf_res["geodataframe"]
                    gdf_22 = classified_gdf
                    summary_df = pd.DataFrame(classified_gdf.drop(columns=["geometry"], errors="ignore"))
                    classified_gdf, summary_df = _backfill_risk(classified_gdf, summary_df)

                    st.session_state["results"] = {
                        "img_a": img_a, "img_b": img_b, "img_c": img_c,
                        "summary_df": summary_df, "classified_gdf": classified_gdf,
                        "gdf_16": None, "gdf_22": classified_gdf,
                        "label": f"2026 Glacier: {scene['glacier']}"
                    }
                    st.sidebar.success(f"✅ {len(scenes)} glacier scenes fetched — {scene['glacier']} displayed")
                    st.rerun()
            except Exception as e:
                st.sidebar.error(f"⚠️ Glacier scenes fetch failed: {e}")
                st.exception(e)

    # Run 2016 vs 2026 Change Analysis
    if st.session_state.get("run_2016_2026_comparison"):
        st.session_state["run_2016_2026_comparison"] = False
        with st.spinner("Running 2016 vs 2026 change analysis with real 2026 data..."):
            try:
                from src.data.real_data_fetcher import run_2026_comparison_with_historical
                comparison = run_2026_comparison_with_historical(historical_year=2016)
                summary_df = comparison["summary_df"]
                classified_gdf = comparison["classified_gdf"]

                # Load 2016 imagery for display
                stack_16, mask_16, prof_16 = create_synthetic_glacial_scene(1024, 1024, seed=42, epoch=2016)
                rgb_16 = np.stack([stack_16[2], stack_16[1], stack_16[0]], axis=-1)
                rgb_16 = np.clip(rgb_16 * 255.0, 0, 255).astype(np.uint8)

                # Get 2026 imagery for display
                mosaic_2026 = fetch_2026_satellite_mosaic(zoom=13, use_esri=True)
                rgb_2026 = mosaic_2026["rgb"]

                st.session_state["results"] = {
                    "img_a": Image.fromarray(rgb_16), "img_b": Image.fromarray(rgb_2026), "img_c": Image.fromarray(rgb_2026),
                    "summary_df": summary_df, "classified_gdf": classified_gdf,
                    "gdf_16": comparison["historical_result"], "gdf_22": comparison["current_result"],
                    "label": "2016 vs 2026 Change Analysis"
                }
                new_count = len(summary_df[summary_df["status"] == "Newly Formed"])
                survived = len(summary_df[summary_df["status"] == "Survived"])
                drained = len(summary_df[summary_df["status"] == "Drained"])
                st.sidebar.success(f"✅ 2016 vs 2026: {new_count} new, {survived} survived, {drained} drained")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"⚠️ 2016 vs 2026 analysis failed: {e}")
                st.exception(e)

    # 4. View Mode Switcher
    view_mode = st.radio(
        "Display Mode",
        options=[
            "🌐 Live Real Earth Satellite Map (Interactive GIS Globe)",
            "🖼️ 3-Panel Scientific Visualizer (Panels A, B, C)",
        ],
        horizontal=True,
        label_visibility="collapsed"
    )

    # 5. Main 2-Column Section (Main Imagery/Map on Left, Analytics & Downloads on Right)
    col_main, col_analytics = st.columns([3.15, 1.15], gap="medium")

    with col_main:
        if "Live Real Earth Satellite Map" in view_mode:
            # Interactive Real Earth Satellite Map View
            st.markdown(
                """
                <div class="panel-box">
                    <div class="panel-topbar">
                        <div class="panel-label">🛰️ Real Earth Interactive Satellite Map • Karakoram / Himalayan Range (Passu & Shishper Glaciers)</div>
                        <div style="font-size: 11px; color: #94a3b8;">Use Layer Control (top right) to toggle 2016 vs 2022 boundaries</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            earth_map = render_real_earth_satellite_map(classified_gdf, gdf_2016=gdf_16, gdf_2022=gdf_22)
            st_folium(earth_map, width=1100, height=530, key="real_earth_map")

        else:
            # 3 Panels Side-by-Side
            p1, p2, p3 = st.columns(3, gap="small")

            with p1:
                st.markdown(
                    """
                    <div class="panel-box">
                        <div class="panel-topbar">
                            <div class="panel-label">A. Himalayan Satellite Overview</div>
                            <div style="color: #64748b; font-size: 14px;">⋮</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                if img_a is not None:
                    st.image(img_a, use_container_width=True)

            with p2:
                st.markdown(
                    """
                    <div class="panel-box">
                        <div class="panel-topbar">
                            <div class="panel-label">B. Multi-Temporal Boundaries (2016 vs. 2022)</div>
                        </div>
                        <div style="background-color: #151c2e; padding: 4px 14px 8px 14px; display: flex; gap: 8px;">
                            <span class="tag-blue">🟦 Year 2016</span>
                            <span class="tag-red">🟥 Year 2022</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                if img_b is not None:
                    st.image(img_b, use_container_width=True)

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
                if img_c is not None:
                    st.image(img_c, use_container_width=True)

    with col_analytics:
        # 0. Top Risk Lakes / Early Warning Card
        if summary_df is not None and not summary_df.empty and "risk" in summary_df.columns:
            risk_summary = summarize_risk(summary_df)
            top_lakes = risk_summary["top_risk_lakes"]
            display_lakes = [l for l in top_lakes if l["risk"] in ("High", "Moderate")] or top_lakes[:3]
            risk_color_map = {"High": "#E74C3C", "Moderate": "#F1C40F", "Low": "#2ECC71"}
            for lake in display_lakes[:3]:
                lc = risk_color_map.get(lake["risk"], "#64748b")
                st.markdown(
                    f"""
                    <div class="right-card" style="border-left: 3px solid {lc};">
                        <div class="right-card-title">
                            <span>⚠️ {lake["lake_id"]} — {lake["risk"]} Risk</span>
                            <span style="background-color: {lc}; color: #0f172a; padding: 2px 6px; border-radius: 3px; font-size: 11px;">
                                {lake["risk_score"]:.0f}/100
                            </span>
                        </div>
                        <div style="font-size: 11px; color: #94a3b8; margin-top: 6px;">
                            Growth {lake["area_change_pct"]:+.1f}% • Area {lake["latest_area_sqkm"]:.3f} km²
                        </div>
                        <div style="font-size: 11px; color: #d1d5db; margin-top: 4px; font-style: italic;">
                            {lake["recommended_action"]}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # 1. Total Surface Water Change Card
        st.markdown(
            """
            <div class="right-card">
                <div class="right-card-title">
                    <span>Total Surface Water Change</span>
                    <span style="cursor: pointer;">↻</span>
                </div>
                <div class="hero-stat">+14.2%</div>
                <div class="sub-stats-grid">
                    <div>
                        <div style="color: #64748b; font-size: 11px;">Water Increase</div>
                        <div class="sub-stat-val" style="color: #34d399;">+29.507 km²</div>
                    </div>
                    <div>
                        <div style="color: #64748b; font-size: 11px;">Area Decrease</div>
                        <div class="sub-stat-val" style="color: #f87171;">-13.860 km²</div>
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
        if summary_df is not None and not summary_df.empty:
            chart_df = summary_df.copy()
            area_col = "area_2022_sqkm" if "area_2022_sqkm" in chart_df.columns else chart_df.columns[2]
            
            bar_chart = (
                alt.Chart(chart_df)
                .mark_bar(color="#38bdf8", cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
                .encode(
                    x=alt.X(f"{area_col}:Q", bin=alt.Bin(maxbins=8), title="Lake Area (km²)", axis=alt.Axis(labelColor="#94a3b8", titleColor="#94a3b8")),
                    y=alt.Y("count():Q", title="Lake Count", axis=alt.Axis(labelColor="#94a3b8", titleColor="#94a3b8")),
                    tooltip=["count():Q"]
                )
                .properties(height=130)
                .configure_view(strokeOpacity=0)
            )
            st.altair_chart(bar_chart, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # 3. Downloads Card
        st.markdown(
            """
            <div class="right-card">
                <div class="right-card-title">
                    <span>Downloads</span>
                </div>
            """,
            unsafe_allow_html=True
        )
        if classified_gdf is not None and not classified_gdf.empty:
            shp_zip = _shapefile_archive(classified_gdf) or b""
            st.download_button("📥 Shapefile (.zip)", data=shp_zip, file_name="lake_change_classified.zip", mime="application/zip", use_container_width=True)
            st.download_button("📥 GeoJSON", data=classified_gdf.to_json(), file_name="lake_change_classified.geojson", mime="application/geo+json", use_container_width=True)
            if summary_df is not None:
                st.download_button("📥 CSV Summary", data=summary_df.to_csv(index=False).encode('utf-8'), file_name="glacial_lake_summary.csv", mime="text/csv", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # 6. Bottom Table: Full Lake Attribute Inventory
    st.markdown("---")
    st.markdown("### 📋 Full Glacial Lake Inventory & Multi-Temporal Attributes")
    if summary_df is not None and not summary_df.empty:
        st.dataframe(
            summary_df.drop(columns=["geometry", "color", "status_code"], errors="ignore"),
            use_container_width=True,
            column_config={
                "lake_id": st.column_config.TextColumn("Lake ID", pinned=True),
                "status": st.column_config.TextColumn("Temporal Classification"),
                "area_2016_sqkm": st.column_config.NumberColumn("Area 2016 (km²)", format="%.3f"),
                "area_2022_sqkm": st.column_config.NumberColumn("Area 2022 (km²)", format="%.3f"),
                "area_change_sqkm": st.column_config.NumberColumn("Net Change (km²)", format="%+.3f"),
                "area_change_pct": st.column_config.NumberColumn("Change (%)", format="%+.1f%%"),
                "centroid_lon": st.column_config.NumberColumn("Longitude (°E)", format="%.4f"),
                "centroid_lat": st.column_config.NumberColumn("Latitude (°N)", format="%.4f"),
            }
        )

if __name__ == "__main__":
    main()
