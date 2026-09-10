"""Analytics view for change dynamics and transparent screening priorities."""

from typing import Any, Dict, Optional
import altair as alt
import streamlit as st
import pandas as pd


STATUS_COLORS = {
    "Newly Formed": "#f4d03f",
    "Survived": "#58d68d",
    "Drained": "#ec7063",
    "Detected": "#5dade2",
}


def _latest_area_column(summary_df: pd.DataFrame) -> Optional[str]:
    if "area_2022_sqkm" in summary_df.columns:
        return "area_2022_sqkm"
    columns = [column for column in summary_df if column.startswith("area_") and column.endswith("_sqkm")]
    return columns[-1] if columns else ("area_sqkm" if "area_sqkm" in summary_df else None)


def build_screening_priorities(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Create a transparent prioritisation score; it is not a GLOF forecast."""
    if summary_df.empty:
        return pd.DataFrame()
    working = summary_df.copy()
    if "status" not in working.columns:
        working["status"] = "Detected"
    area_column = _latest_area_column(working)
    working["latest_area_sqkm"] = working[area_column].fillna(0.0) if area_column else 0.0
    working["area_change_pct"] = working.get("area_change_pct", pd.Series(0.0, index=working.index)).fillna(0.0)
    newly_formed = working["status"].eq("Newly Formed").astype(int)
    growth = working["area_change_pct"].clip(lower=0, upper=100)
    working["screening_score"] = (
        20
        + newly_formed * 25
        + growth * 0.35
        + working["latest_area_sqkm"].clip(lower=0, upper=2.0) * 10
    ).clip(upper=100).round().astype(int)
    working["screening_priority"] = pd.cut(
        working["screening_score"],
        bins=[-1, 44, 69, 100],
        labels=["Routine", "Watch", "High"],
    ).astype(str)
    working["recommended_action"] = working["screening_priority"].map(
        {
            "High": "Review imagery and field context",
            "Watch": "Track at next acquisition",
            "Routine": "Retain in inventory",
        }
    )
    columns = [
        column for column in ["lake_id", "status", "latest_area_sqkm", "area_change_pct", "screening_score", "screening_priority", "recommended_action"]
        if column in working.columns
    ]
    return working[columns].sort_values(["screening_score", "latest_area_sqkm"], ascending=False)

def render_analytics_dashboard(summary_df: Optional[pd.DataFrame], stats_dict: Optional[Dict[str, Any]]):
    """Render metrics, charts, and a caveated screening-priority queue."""
    st.subheader("Temporal dynamics and screening priorities", divider="blue")

    if stats_dict is None or summary_df is None or len(summary_df) == 0:
        st.info("Run an analysis to view temporal metrics and screening priorities.", icon=":material/insights:")
        return

    working = summary_df.copy()
    if "status" not in working.columns:
        working["status"] = "Detected"
    area_column = _latest_area_column(working)
    status_counts = working["status"].value_counts().rename_axis("Status").reset_index(name="Lakes")
    status_domain = status_counts["Status"].tolist()
    status_range = [STATUS_COLORS.get(status, STATUS_COLORS["Detected"]) for status in status_domain]

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1.container(border=True):
        st.markdown("**Change-class distribution**")
        donut = (
            alt.Chart(status_counts)
            .mark_arc(innerRadius=52)
            .encode(
                theta=alt.Theta("Lakes:Q"),
                color=alt.Color("Status:N", scale=alt.Scale(domain=status_domain, range=status_range)),
                tooltip=[alt.Tooltip("Status:N"), alt.Tooltip("Lakes:Q")],
            )
        )
        st.altair_chart(donut, width="stretch")
    with chart_col2.container(border=True):
        st.markdown("**Surface area distribution**")
        if area_column:
            area_plot = working.rename(columns={area_column: "Lake area (km²)"})
            histogram = (
                alt.Chart(area_plot)
                .mark_bar()
                .encode(
                    x=alt.X("Lake area (km²):Q", bin=alt.Bin(maxbins=14)),
                    y=alt.Y("count():Q", title="Lakes"),
                    color=alt.Color("status:N", scale=alt.Scale(domain=status_domain, range=status_range)),
                    tooltip=[alt.Tooltip("count():Q", title="Lakes")],
                )
            )
            st.altair_chart(histogram, width="stretch")
        else:
            st.caption("Area information is unavailable for this result.")

    priorities = build_screening_priorities(working)
    with st.container(border=True):
        st.markdown("**Screening priority queue**")
        st.caption("A transparent triage aid based on area, observed growth, and newly formed status. It is not a GLOF hazard forecast.")
        st.dataframe(
            priorities,
            column_config={
                "lake_id": st.column_config.TextColumn("Lake ID", pinned=True),
                "latest_area_sqkm": st.column_config.NumberColumn("Latest area (km²)", format="%.4f"),
                "area_change_pct": st.column_config.NumberColumn("Area change", format="%.1f%%"),
                "screening_score": st.column_config.ProgressColumn("Screening score", min_value=0, max_value=100),
                "screening_priority": st.column_config.TextColumn("Priority"),
                "recommended_action": st.column_config.TextColumn("Recommended action"),
            },
            hide_index=True,
            key="screening_priority_queue",
        )
