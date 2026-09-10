"""
GLOF (Glacial Lake Outburst Flood) risk scoring module.
Assigns a transparent, explainable risk score (0-100) and risk tier
(Low/Moderate/High) to each detected lake based on its temporal change,
physical characteristics, and terrain context, suitable for early-warning
prioritization.

The score combines:
- Base awareness (20)
- Newly formed lake bonus (+25)
- Growth rate contribution (0-24) from area_change_pct (capped at 60%)
- Size contribution (0-40) from latest area (capped at 5 km²)
- Terrain slope bonus (0-25) from DEM-derived slope proxy
- Drained lake penalty (-60) — no longer holds water

Final score = clip(base + newly + growth + size + terrain - drained, 0, 100)
Risk tiers: Low [0,40), Moderate [40,70), High [70,100]
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..utils.config import config


def compute_terrain_features(
    dem: np.ndarray,
    centroid_lons: np.ndarray,
    centroid_lats: np.ndarray,
    transform=None,
    window_size: int = 5,
) -> np.ndarray:
    """
    Compute a crude slope proxy (std-dev of DEM in a local window) at each
    lake centroid. Higher std-dev = steeper terrain = higher GLOF risk.

    Parameters
    ----------
    dem : np.ndarray
        2D DEM array (H, W) in meters.
    centroid_lons, centroid_lats : np.ndarray
        1D arrays of centroid coordinates (EPSG:4326).
    transform : rasterio.Affine or None
        Affine transform from pixel coords to geographic coords.
        If None, uses a heuristic mapping from lon/lat to pixel indices.
    window_size : int
        Neighborhood size for slope computation (odd number).

    Returns
    -------
    mean_slope : np.ndarray
        1D array of slope proxy values (DEM std-dev in window) at each centroid.
        Returns zeros if DEM is unavailable.
    """
    if dem is None or dem.ndim != 2:
        return np.zeros(len(centroid_lons))

    h, w = dem.shape
    half_w = window_size // 2
    result = np.zeros(len(centroid_lons), dtype=np.float64)

    for i, (lon, lat) in enumerate(zip(centroid_lons, centroid_lats)):
        # Convert geographic coords to pixel indices
        if transform is not None:
            # Use the affine transform
            try:
                from rasterio.transform import rowcol
                row, col = rowcol(transform, lon, lat)
            except Exception:
                # Fallback: heuristic pixel mapping (works for Hunza region bounds)
                col = int(((lon - 74.45) / 0.40) * (w - 1))
                row = int(((36.75 - lat) / 0.35) * (h - 1))
        else:
            col = int(((lon - 74.45) / 0.40) * (w - 1))
            row = int(((36.75 - lat) / 0.35) * (h - 1))

        # Clamp to valid range
        row = max(half_w, min(row, h - half_w - 1))
        col = max(half_w, min(col, w - half_w - 1))

        # Extract local window and compute std-dev as slope proxy
        window = dem[row - half_w : row + half_w + 1, col - half_w : col + half_w + 1]
        result[i] = float(np.std(window))

    return result


def _latest_area_column(df: pd.DataFrame) -> str:
    """Return the column name holding the latest epoch area (km²)."""
    if "area_2022_sqkm" in df.columns:
        return "area_2022_sqkm"
    # Generic fallback: pick the last area_*_sqkm column
    area_cols = [c for c in df.columns if c.startswith("area_") and c.endswith("_sqkm")]
    if area_cols:
        return area_cols[-1]
    # Last resort: a generic area_sqkm column (from single-scene mode)
    return "area_sqkm" if "area_sqkm" in df.columns else "area_2022_sqkm"


def add_risk_scores(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Append GLOF risk scores and tiers to a change-classification summary DataFrame.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Output from classify_temporal_change containing at least:
        - lake_id, status, area_change_pct, and an area column
          (area_2022_sqkm, area_sqkm, or similar).

    Returns
    -------
    pd.DataFrame
        Copy of summary_df with four new columns:
        - risk_score (float 0-100): composite risk score
        - risk (str): 'Low', 'Moderate', or 'High'
        - risk_color (str): hex color for UI rendering
        - recommended_action (str): short actionable advice
    """
    if summary_df.empty:
        # Return early with expected columns to avoid KeyErrors downstream
        df = summary_df.copy()
        df["risk_score"] = pd.Series(dtype="float64")
        df["risk"] = pd.Series(dtype="object")
        df["risk_color"] = pd.Series(dtype="object")
        df["recommended_action"] = pd.Series(dtype="object")
        return df

    df = summary_df.copy()

    # Identify the column holding the latest epoch area
    area_col = _latest_area_column(df)
    if area_col not in df.columns:
        raise KeyError(
            f"Could not find a latest-area column in summary_df. "
            f"Expected one of ['area_2022_sqkm', 'area_sqkm'] or similar; "
            f"got columns {list(df.columns)}"
        )

    # Work with cleaned numeric series - use a Series if column missing
    if "area_change_pct" in df.columns:
        area_change_pct = pd.to_numeric(df["area_change_pct"], errors="coerce").fillna(0.0)
    else:
        area_change_pct = pd.Series(0.0, index=df.index)
    latest_area = pd.to_numeric(df[area_col], errors="coerce").fillna(0.0)
    if "status" in df.columns:
        status = df["status"].astype(str).fillna("Detected")
    else:
        status = pd.Series("Detected", index=df.index)

    # ----- Component 1: Base awareness -----
    base_score = 20.0

    # ----- Component 2: Newly formed bonus -----
    newly_formed = (status == "Newly Formed").astype(float) * 25.0

    # ----- Component 3: Growth contribution (0-24) -----
    # Only positive growth (expansion) increases risk; shrinkage/drained does not via this term
    growth_positive = np.clip(area_change_pct, 0, 60)  # cap at 60% for scoring
    growth_bonus = growth_positive * 0.4  # 60 * 0.4 = 24 max

    # ----- Component 4: Size contribution (0-40) -----
    # Larger lakes pose greater potential hazard; cap at 5 km² for scoring
    size_bonus = np.clip(latest_area, 0, 5) * 8.0  # 5 * 8 = 40 max

    # ----- Component 5: Terrain slope bonus (0-25) -----
    # Steeper terrain => higher outburst potential. Uses a DEM-derived slope
    # proxy passed in via terrain_df (optional). Defaults to 0 when absent.
    mean_slope = None
    if "mean_slope" in df.columns:
        mean_slope = pd.to_numeric(df["mean_slope"], errors="coerce").fillna(0.0)
    elif "terrain_slope" in df.columns:
        mean_slope = pd.to_numeric(df["terrain_slope"], errors="coerce").fillna(0.0)
    if mean_slope is not None:
        terrain_bonus = np.where(mean_slope > 400, 25.0,
                       np.where(mean_slope > 200, 15.0, 0.0))
    else:
        terrain_bonus = np.zeros(len(df))

    # ----- Component 6: Drained lake penalty -----
    # A drained lake no longer holds water -> greatly reduced immediate GLOF concern
    drained_penalty = (status == "Drained").astype(float) * 60.0

    # ----- Composite score -----
    raw_score = (
        base_score
        + newly_formed
        + growth_bonus
        + size_bonus
        + terrain_bonus
        - drained_penalty
    )
    risk_score = np.clip(raw_score, 0, 100).round(2)

    # ----- Risk tiers -----
    def _risk_tier(score: float) -> str:
        if score < 40:
            return "Low"
        if score < 70:
            return "Moderate"
        return "High"

    risk = risk_score.apply(_risk_tier)

    # ----- Risk colors (reuse existing config palette) -----
    def _risk_color(tier: str) -> str:
        return {
            "Low": config.COLOR_SURVIVED,      # Green
            "Moderate": config.COLOR_NEW,      # Yellow
            "High": config.COLOR_DRAINED,      # Red
        }.get(tier, config.COLOR_SURVIVED)

    risk_color = risk.apply(_risk_color)

    # ----- Recommended الشمول -----
    def _recommended_action(tier: str, score: float) -> str:
        if tier == "High":
            return "High priority: validate with field survey & consider early warning"
        if tier == "Moderate":
            return "Monitor: prioritize in next acquisition cycle"
        return "Low: retain in inventory; routine revisit"

    recommended_action = [ _recommended_action(t, s) for t, s in zip(risk, risk_score) ]

    # Assign new columns
    df["risk_score"] = risk_score
    df["risk"] = risk
    df["risk_color"] = risk_color
    df["recommended_action"] = recommended_action

    return df


def summarize_risk(summary_df: pd.DataFrame) -> Dict[str, Union[int, float, List[Dict]]]:
    """
    Compute aggregate risk statistics for dashboard KPIs and early-warning panel.

    Returns a dict with:
    - total_lakes: int
    - risk_counts: dict {Low: int, Moderate: int, High: int}
    - top_risk_lakes: list of dicts (sorted descending by risk_score),
      each dict containing lake_id, risk, risk_score, area_change_pct, latest_area_sqkm,
      and recommended_action for the top N (default 5) lakes.
    """
    if summary_df.empty:
        return {
            "total_lakes": 0,
            "risk_counts": {"Low": 0, "Moderate": 0, "High": 0},
            "top_risk_lakes": [],
        }

    df = add_risk_scores(summary_df)  # ensure columns exist

    total = len(df)
    risk_counts = df["risk"].value_counts().reindex(["Low", "Moderate", "High"], fill_value=0).to_dict()

    # Top N riskiest lakes (by score descending)
    TOP_N = 5
    top_df = (
        df[["lake_id", "risk", "risk_score", area_col := _latest_area_column(df), "area_change_pct", "recommended_action"]]
        .sort_values("risk_score", ascending=False)
        .head(TOP_N)
    )

    top_lakes: List[Dict] = []
    for _, row in top_df.iterrows():
        # Handle None/NaN area_change_pct (single-scene mode)
        area_change_pct = row.get("area_change_pct")
        if area_change_pct is None or (isinstance(area_change_pct, float) and np.isnan(area_change_pct)):
            area_change_pct = 0.0

        lake_dict = {
            "lake_id": str(row["lake_id"]),
            "risk": str(row["risk"]),
            "risk_score": float(row["risk_score"]),
            "latest_area_sqkm": float(row[area_col]),
            "area_change_pct": float(area_change_pct),
            "recommended_action": str(row["recommended_action"]),
        }
        # Add terrain info if available
        if "mean_slope" in row:
            lake_dict["mean_slope"] = float(row["mean_slope"])
        elif "terrain_slope" in row:
            lake_dict["terrain_slope"] = float(row["terrain_slope"])
        top_lakes.append(lake_dict)

    return {
        "total_lakes": int(total),
        "risk_counts": {k: int(v) for k, v in risk_counts.items()},
        "top_risk_lakes": top_lakes,
    }