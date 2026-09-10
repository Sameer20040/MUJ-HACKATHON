# GLOF Risk Scoring Model — Technical Documentation

**Project:** Glacial Lake AI — Multi-Temporal Detection & GLOF Risk Monitor  
**Version:** 1.0  
**Date:** 2025

---

## 1. Purpose

This document describes the **GLOF (Glacial Lake Outburst Flood) Risk Scoring Model** used in the Glacial Lake AI pipeline. The model assigns a **0–100 risk score** and a **risk tier** (Low / Moderate / High) to each detected glacial lake, based on observable satellite-derived characteristics.

**This is a screening tool, not a GLOF forecast.** It prioritizes lakes for further monitoring and field validation. It does **not** predict when or if a specific lake will burst.

---

## 2. Input Features

The model uses five input features, all derived automatically from satellite imagery and the change-detection pipeline:

| Feature | Source | Description |
|---|---|---|
| **Temporal Status** | Change detection | `Newly Formed` / `Survived` / `Drained` — whether the lake appeared, persisted, or vanished between two epochs |
| **Area Change %** | Change detection | Percentage change in surface area: `(A_t2 − A_t1) / A_t1 × 100` |
| **Latest Area (km²)** | Vectorization | Surface area in the most recent epoch |
| **Terrain Slope Proxy** | DEM (if available) | Standard deviation of elevation in a 5×5 pixel window around the lake centroid. Higher = steeper terrain = greater outburst potential. |
| **Drained Flag** | Change detection | Lake present in epoch 1 but absent in epoch 2 → penalty |

---

## 3. Scoring Formula

```
Risk Score = clip(
    Base (20)
  + Newly_Formed_Bonus (0 or 25)
  + Growth_Bonus (0–24)
  + Size_Bonus (0–40)
  + Terrain_Bonus (0–25)
  − Drained_Penalty (0 or 60),
  0, 100
)
```

### Component Details

| Component | Formula | Range | Rationale |
|---|---|---|---|
| **Base** | 20 | fixed | Minimum awareness level for any detected water body |
| **Newly Formed Bonus** | `+25 if status == "Newly Formed" else 0` | 0 / 25 | New lakes lack historical stability data and may be impounded by unstable moraine dams |
| **Growth Bonus** | `clip(max(area_change_pct, 0), 60) × 0.4` | 0 – 24 | Rapid expansion signals increasing hydrostatic pressure on the dam. Capped at 60% growth to avoid outliers |
| **Size Bonus** | `clip(latest_area_km2, 5) × 8` | 0 – 40 | Larger lakes store more potential energy. Capped at 5 km² |
| **Terrain Bonus** | `0 if mean_slope ≤ 200`<br>`+15 if 200 < mean_slope ≤ 400`<br>`+25 if mean_slope > 400` | 0 / 15 / 25 | Steep terrain increases dam failure probability and downstream flow velocity. Derived from DEM std-dev in local window. If DEM unavailable, defaults to 0 |
| **Drained Penalty** | `−60 if status == "Drained" else 0` | 0 / −60 | A drained lake no longer impounds water → immediate GLOF concern is negligible |

---

## 4. Risk Tiers

| Tier | Score Range | Color | Recommended Action |
|---|---|---|---|
| **Low** | 0 – 39 | Green (`#2ECC71`) | Low: retain in inventory; routine revisit |
| **Moderate** | 40 – 69 | Yellow (`#F1C40F`) | Monitor: prioritize in next acquisition cycle |
| **High** | 70 – 100 | Red (`#E74C3C`) | High priority: validate with field survey & consider early warning |

---

## 5. What This Model IS

- **Data-driven**: All inputs come from the automated satellite pipeline (no manual tuning per lake)
- **Explainable**: Every point in the score traces to a named component with a clear physical rationale
- **Reproducible**: Deterministic given the same inputs; same scene → same score
- **Action-oriented**: Outputs a recommended action per tier

---

## 6. What This Model IS NOT

- **Not a GLOF forecast**: Does not model dam breach physics, seepage, ice avalanches, seismic triggers, or downstream flow routing
- **Not a probability**: A score of 80 does not mean "80% chance of outburst"
- **Not a substitute for field surveys**: High scores flag lakes that *deserve* expert attention
- **Not calibrated to historical outbursts**: No historical GLOF database was used to fit weights

---

## 7. Limitations & Future Work

| Limitation | Mitigation / Next Step |
|---|---|
| No moraine dam width/height | Integrate high-res DEM (e.g., ICIMOD) or SAR interferometry |
| No downstream exposure | Add population/infrastructure buffers (WorldPop, OSM) |
| Slope proxy is crude (DEM std-dev) | Replace with proper slope/aspect from reprojected DEM |
| No seasonal / climate drivers | Add precipitation, temperature anomaly layers |
| Weights are expert-chosen, not learned | Calibrate against historical GLOF events when data available |

---

## 8. Implementation Reference

**Module:** `src/change_detection/risk_scoring.py`  
**Key functions:**
- `add_risk_scores(summary_df)` — enriches a change-detection summary DataFrame
- `compute_terrain_features(dem, lons, lats, transform)` — computes the slope proxy
- `summarize_risk(summary_df)` — aggregate statistics for dashboard

---

## 9. Example Output

```
lake_id      status         area_change_pct  risk_score  risk        recommended_action
GLK_012      Newly Formed   100.0            100.0       High        High priority: validate with field survey & consider early warning
GLK_007      Newly Formed   100.0            78.2        High        High priority: validate with field survey & consider early warning
SURV_004     Survived       108.7            84.0        High        High priority: validate with field survey & consider early warning
DRAIN_004    Drained        -100.0           0.0         Low         Low: retain in inventory; routine revisit
```

---

## 9. Citation

If you use this risk model in research, please cite:

```
Glacial Lake AI Team. (2025). GLOF Risk Scoring Model v1.0.
Manipal HackX 4.0 — Disaster Tech Track.
```