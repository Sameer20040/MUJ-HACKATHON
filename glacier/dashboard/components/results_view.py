"""Analysis workspace, geospatial map, lake catalogue, and export tools."""

from html import escape
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Optional
from zipfile import ZIP_DEFLATED, ZipFile

import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import geopandas as gpd


STATUS_COLORS = {
    "Newly Formed": "#f4d03f",
    "Survived": "#58d68d",
    "Drained": "#ec7063",
    "Detected": "#5dade2",
}


def _latest_area_column(summary_df: pd.DataFrame) -> Optional[str]:
    if "area_2022_sqkm" in summary_df.columns:
        return "area_2022_sqkm"
    epoch_columns = [
        column for column in summary_df.columns
        if column.startswith("area_") and column.endswith("_sqkm")
    ]
    if epoch_columns:
        return epoch_columns[-1]
    return "area_sqkm" if "area_sqkm" in summary_df.columns else None


def _to_wgs84(gdf: Optional[gpd.GeoDataFrame]) -> Optional[gpd.GeoDataFrame]:
    if gdf is None or len(gdf) == 0:
        return gdf
    try:
        if gdf.crs is not None and str(gdf.crs).upper() != "EPSG:4326":
            return gdf.to_crs(epsg=4326)
    except Exception:
        return gdf
    return gdf


def _empty_workspace() -> None:
    st.subheader("Analysis workspace", divider="blue")
    st.caption("Select an analysis mode in the sidebar, then run the workflow to populate the map and decision-support results.")
    stages = st.columns(3)
    content = [
        (":material/satellite_alt:", "1. Select imagery", "Use the Karakoram benchmark or add your own single or multi-epoch satellite scenes."),
        (":material/tune:", "2. Tune detection", "Set the confidence threshold, noise filter, and optional cloud screening for GeoTIFF inputs."),
        (":material/insights:", "3. Inspect change", "Review boundaries, change classes, screening priorities, and GIS-ready downloads."),
    ]
    for column, (icon, title, body) in zip(stages, content):
        with column.container(border=True):
            st.markdown(icon)
            st.markdown(f"**{title}**")
            st.caption(body)


def _render_kpis(stats: Dict[str, Any], results_y2: Dict[str, Any]) -> None:
    net_change = float(stats.get("net_area_change_sqkm", 0.0))
    metrics = st.columns(4, vertical_alignment="center")
    with metrics[0]:
        st.metric(
            "Lakes tracked",
            int(stats.get("total_lakes_detected", results_y2.get("lake_count", 0))),
            border=True,
        )
    with metrics[1]:
        st.metric(
            "Newly formed",
            int(stats.get("newly_formed_count", 0)),
            border=True,
        )
    with metrics[2]:
        st.metric(
            "Drained",
            int(stats.get("drained_count", 0)),
            border=True,
        )
    with metrics[3]:
        st.metric(
            "Net water change",
            f"{net_change:+.3f} km²",
            delta=f"{net_change:+.3f} km²",
            delta_color="normal",
            border=True,
        )


def render_results_workspace(analysis: Optional[Dict[str, Any]]) -> None:
    """Render the visual workspace and its selected exploration view."""
    if analysis is None:
        _empty_workspace()
        return

    results_y1 = analysis.get("results_y1")
    results_y2 = analysis.get("results_y2")
    if results_y2 is None:
        _empty_workspace()
        return

    summary_df = analysis.get("summary_df")
    classified_gdf = analysis.get("classified_gdf")
    stats = analysis.get("stats", {})
    st.subheader("Analysis workspace", divider="blue")
    st.caption(analysis.get("label", "Lake detection analysis") + " • Results remain available while you explore the dashboard.")
    _render_kpis(stats, results_y2)

    panels = st.columns(3)
    with panels[0].container(border=True):
        st.markdown("**A. Satellite overview**")
        st.image(results_y2["display_rgb"], width="stretch")
        st.caption("Latest epoch RGB or uploaded scene")
    with panels[1].container(border=True):
        st.markdown("**B. Multi-temporal boundaries**")
        st.image(results_y2["overlay_rgb"], width="stretch")
        st.caption("Earlier epoch: blue • Later epoch: red")
    with panels[2].container(border=True):
        st.markdown("**C. Classified change markers**")
        classified_panel = analysis.get("panel_c")
        if classified_panel is None:
            classified_panel = results_y2["overlay_rgb"]
        st.image(classified_panel, width="stretch")
        st.caption("New: yellow • Survived: green • Drained: red")

    if results_y1 is None:
        st.caption("Single-scene mode reports detected boundaries only; temporal change classes require two epochs.")
    if results_y2.get("cloud_coverage_pct", 0.0) > 0:
        st.caption(
            f"Cloud screening excluded {results_y2['cloud_coverage_pct']:.1f}% of the GeoTIFF scene from boundary extraction."
        )

    exploration = st.segmented_control(
        "Explore results",
        options=["Interactive map", "Lake catalogue", "Download centre"],
        default="Interactive map",
        key="analysis_exploration_view",
        label_visibility="collapsed",
    )
    if exploration == "Interactive map":
        render_folium_map(classified_gdf)
    elif exploration == "Lake catalogue":
        _render_catalogue(summary_df)
    else:
        _render_exports(summary_df, classified_gdf)


def _render_catalogue(summary_df: Optional[pd.DataFrame]) -> None:
    with st.container(border=True):
        st.markdown("**Lake catalogue**")
        if summary_df is None or summary_df.empty:
            st.info("No lake records were produced for this analysis.", icon=":material/info:")
            return
        display = summary_df.drop(columns=["geometry", "color", "status_code"], errors="ignore").copy()
        if "status" not in display.columns:
            display["status"] = "Detected"
        area_col = _latest_area_column(display)
        column_config: Dict[str, Any] = {
            "lake_id": st.column_config.TextColumn("Lake ID", pinned=True),
            "status": st.column_config.TextColumn("Status"),
            "area_change_pct": st.column_config.NumberColumn("Area change", format="%.1f%%"),
        }
        if area_col:
            column_config[area_col] = st.column_config.NumberColumn("Latest area (km²)", format="%.4f")
        st.dataframe(display, column_config=column_config, hide_index=True, key="lake_catalogue")


def _shapefile_archive(gdf: gpd.GeoDataFrame) -> Optional[bytes]:
    """Build a portable shapefile ZIP only when the user opens the export view."""
    if gdf.empty:
        return None
    try:
        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "lake_change_classified.shp"
            # ESRI Shapefile limits field names to ten characters. Preserve a
            # stable, GIS-friendly schema instead of silently laundering names.
            field_names = {
                "status_code": "stat_code",
                "area_2016_sqkm": "area_2016",
                "area_2022_sqkm": "area_2022",
                "area_Epoch1_sqkm": "area_ep1",
                "area_Epoch2_sqkm": "area_ep2",
                "area_change_sqkm": "change_km2",
                "area_change_pct": "change_pct",
                "centroid_lon": "cent_lon",
                "centroid_lat": "cent_lat",
            }
            export_gdf = gdf.rename(columns={key: value for key, value in field_names.items() if key in gdf.columns})
            export_gdf.to_file(output_path, driver="ESRI Shapefile")
            buffer = BytesIO()
            with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
                for file_path in Path(directory).iterdir():
                    archive.write(file_path, file_path.name)
                archive.writestr(
                    "README.txt",
                    "Field aliases: stat_code=status_code; area_2016/area_2022 or area_ep1/area_ep2=lake area in km²; "
                    "change_km2=area change in km²; change_pct=area change percent; cent_lon/cent_lat=centroid coordinates.\n",
                )
            return buffer.getvalue()
    except Exception:
        return None


def _render_exports(
    summary_df: Optional[pd.DataFrame],
    classified_gdf: Optional[gpd.GeoDataFrame],
) -> None:
    with st.container(border=True):
        st.markdown("**Download centre**")
        st.caption("Exports are generated from the current analysis only; source imagery is never included.")
        if summary_df is None or summary_df.empty:
            st.info("Run an analysis with detected lakes to enable exports.", icon=":material/download:")
            return
        csv_data = summary_df.drop(columns=["geometry"], errors="ignore").to_csv(index=False).encode("utf-8")
        actions = st.columns(3)
        with actions[0]:
            st.download_button(
                "Summary CSV",
                data=csv_data,
                file_name="glacial_lake_change_summary.csv",
                mime="text/csv",
                icon=":material/table_chart:",
                width="stretch",
            )
        if classified_gdf is not None and not classified_gdf.empty:
            export_gdf = _to_wgs84(classified_gdf)
            with actions[1]:
                st.download_button(
                    "Classified GeoJSON",
                    data=export_gdf.to_json(),
                    file_name="lake_change_classified.geojson",
                    mime="application/geo+json",
                    icon=":material/public:",
                    width="stretch",
                )
            with actions[2]:
                archive = _shapefile_archive(export_gdf)
                if archive:
                    st.download_button(
                        "Shapefile ZIP",
                        data=archive,
                        file_name="lake_change_classified.zip",
                        mime="application/zip",
                        icon=":material/folder_zip:",
                        width="stretch",
                    )
                else:
                    st.caption("Shapefile export is unavailable for this result.")


def render_folium_map(classified_gdf: Optional[gpd.GeoDataFrame]) -> None:
    """Render grouped, CRS-safe classification markers on a dark basemap."""
    with st.container(border=True):
        st.markdown("**Interactive map explorer**")
        st.caption("Use the layer control to focus on a specific change class. Click a marker for lake details.")
        map_gdf = _to_wgs84(classified_gdf)
        center = [36.55, 74.65]
        if map_gdf is not None and not map_gdf.empty:
            try:
                center_point = map_gdf.geometry.unary_union.centroid
                if -90 <= center_point.y <= 90 and -180 <= center_point.x <= 180:
                    center = [center_point.y, center_point.x]
            except Exception:
                pass

        lake_map = folium.Map(location=center, zoom_start=11, tiles="CartoDB dark_matter", control_scale=True)
        if map_gdf is None or map_gdf.empty:
            folium.Marker(
                center,
                tooltip="No classified lake geometry is available for this run.",
                icon=folium.Icon(color="blue", icon="info-sign"),
            ).add_to(lake_map)
        else:
            for status, group in map_gdf.groupby(map_gdf.get("status", pd.Series("Detected", index=map_gdf.index))):
                feature_group = folium.FeatureGroup(name=str(status), show=True)
                marker_color = STATUS_COLORS.get(str(status), STATUS_COLORS["Detected"])
                for _, row in group.iterrows():
                    point = row.geometry.centroid
                    if not (-90 <= point.y <= 90 and -180 <= point.x <= 180):
                        continue
                    lake_id = escape(str(row.get("lake_id", "Lake")))
                    row_status = escape(str(row.get("status", "Detected")))
                    change = float(row.get("area_change_pct", 0.0) or 0.0)
                    area = float(row.get("area_2022_sqkm", row.get("area_sqkm", 0.0)) or 0.0)
                    popup = (
                        f"<b>{lake_id}</b><br>Class: {row_status}<br>"
                        f"Latest area: {area:.4f} km²<br>Area change: {change:+.1f}%"
                    )
                    folium.CircleMarker(
                        location=[point.y, point.x],
                        radius=7,
                        popup=folium.Popup(popup, max_width=260),
                        tooltip=f"{lake_id} — {row_status}",
                        color="#ffffff",
                        weight=1,
                        fill=True,
                        fill_color=marker_color,
                        fill_opacity=0.9,
                    ).add_to(feature_group)
                feature_group.add_to(lake_map)
            folium.LayerControl(collapsed=False).add_to(lake_map)
        st_folium(lake_map, width=1100, height=540, key="glacial_lake_map")
