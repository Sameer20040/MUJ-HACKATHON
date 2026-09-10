"""
Streamlit UI Upload and Configuration Sidebar Component.
"""

from typing import Dict, Any
import streamlit as st


MODE_BENCHMARK = "Benchmark — Karakoram (2016 vs 2022)"
MODE_CUSTOM = "Custom multi-epoch comparison"
MODE_SINGLE = "Single-scene delineation"

def render_upload_sidebar() -> Dict[str, Any]:
    """Render the analyst workflow and return the selected controls."""
    st.sidebar.title(":material/landscape: Glacial lake AI")
    st.sidebar.caption("Multi-temporal detection workspace")
    st.sidebar.markdown("**1. Choose analysis**")
    analysis_mode = st.sidebar.radio(
        "Analysis mode",
        options=[
            MODE_BENCHMARK,
            MODE_CUSTOM,
            MODE_SINGLE,
        ],
        index=0,
        label_visibility="collapsed",
    )
    
    uploaded_files: Dict[str, Any] = {}
    if analysis_mode == MODE_BENCHMARK:
        st.sidebar.info(
            "Prepared Sentinel-2 benchmark over the Hunza–Karakoram glacial valley.",
            icon=":material/satellite_alt:",
        )
    elif analysis_mode == MODE_CUSTOM:
        st.sidebar.markdown("**2. Add comparison imagery**")
        st.sidebar.caption("Upload two aligned GeoTIFFs or satellite images.")
        f1 = st.sidebar.file_uploader(
            "Earlier epoch image",
            type=["tif", "tiff", "jpg", "jpeg", "png"],
            key="file_ep1",
        )
        f2 = st.sidebar.file_uploader(
            "Later epoch image",
            type=["tif", "tiff", "jpg", "jpeg", "png"],
            key="file_ep2",
        )
        uploaded_files = {"epoch1": f1, "epoch2": f2}
    else:
        st.sidebar.markdown("**2. Add satellite scene**")
        st.sidebar.caption("GeoTIFF preserves location; JPG/PNG enables a quick visual delineation.")
        f = st.sidebar.file_uploader(
            "Satellite scene",
            type=["tif", "tiff", "jpg", "jpeg", "png"],
            key="file_single",
        )
        uploaded_files = {"single": f}

    st.sidebar.markdown("**3. Tune detection**")
    threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Probability threshold for classifying water pixels."
    )
    
    min_area = st.sidebar.number_input(
        "Min Lake Size (pixels)",
        min_value=5,
        max_value=500,
        value=20,
        step=5,
        help="Filters out tiny noise clusters and speckle."
    )

    enable_cloud_filter = st.sidebar.toggle(
        "Cloud screening for GeoTIFF",
        value=True,
        help="Removes probable cloud pixels before boundaries are extracted when spectral bands are available.",
    )

    st.sidebar.markdown("**4. Run analysis**")
    run_analysis = st.sidebar.button(
        "Run lake analysis",
        type="primary",
        icon=":material/play_arrow:",
        width="stretch",
    )
    clear_results = st.sidebar.button(
        "Clear current results",
        icon=":material/delete_sweep:",
        width="stretch",
    )

    with st.sidebar.expander("Classification legend", icon=":material/info:"):
        st.markdown("🟡 **Newly formed** — appears only in the later epoch")
        st.markdown("🟢 **Survived** — present in both epochs")
        st.markdown("🔴 **Drained** — absent from the later epoch")
        st.caption("Screening priorities are decision-support indicators, not GLOF forecasts.")

    return {
        "analysis_mode": analysis_mode,
        "threshold": threshold,
        "min_area": min_area,
        "enable_cloud_filter": enable_cloud_filter,
        "uploaded_files": uploaded_files,
        "run_analysis": run_analysis,
        "clear_results": clear_results,
    }
