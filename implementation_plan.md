# Deep Research & Comprehensive Audit Implementation Plan

A comprehensive analysis of the entire Glacial Lake AI codebase was conducted across all 46 Python files, including data loaders, U-Net models, postprocessing pipelines, risk scoring engines, FastAPI endpoints, Streamlit dashboard, and unit test suites.

## Identified Issues & Required Fixes

### 1. GLOF Risk Scoring Logic Bug
> [!IMPORTANT]
> **Location:** [src/change_detection/risk_scoring.py](file:///d:/MUJ/Glacier%20lake/Glacier%20lake/src/change_detection/risk_scoring.py#L150-L154)
> **Issue:** Line 151 attempts `pd.to_numeric(df.get("status", ...), errors="coerce").fillna("Detected")`. Because `status` contains string labels (`"Newly Formed"`, `"Survived"`, `"Drained"`), `pd.to_numeric` converts ALL text statuses into `NaN`, which `.fillna("Detected")` overwrites with `"Detected"`.
> **Impact:** `(status == "Newly Formed")` (+25 points) and `(status == "Drained")` (-60 points) evaluate to `False` for every lake, producing inaccurate risk scores for newly formed and drained lakes.
> **Fix:** Keep text status labels intact as strings (`df["status"].astype(str)`).

---

### 2. GeoDataFrame Column & Type Preservation
> [!NOTE]
> **Location:** [src/change_detection/classify_change.py](file:///d:/MUJ/Glacier%20lake/Glacier%20lake/src/change_detection/classify_change.py#L132-L137)
> **Issue:** Line 133 constructs `gpd.GeoDataFrame(summary_df.to_dict(orient="records"), geometry=geometries, crs=crs)`. Converting to a list of dicts can strip pandas numeric/category types.
> **Fix:** Use `gpd.GeoDataFrame(summary_df.assign(geometry=geometries), geometry="geometry", crs=crs)` to preserve exact data types.

---

### 3. Risk Attribute Merging Safety in Dashboard
> [!NOTE]
> **Location:** [dashboard/app.py](file:///d:/MUJ/Glacier%20lake/Glacier%20lake/dashboard/app.py#L608-L625)
> **Issue:** `_backfill_risk()` merges risk columns (`risk`, `risk_score`, `risk_color`, `recommended_action`) back into `classified_gdf`. If any risk columns already exist, `merge()` appends `_risk` suffixes, creating redundant columns.
> **Fix:** Drop pre-existing risk columns from `classified_gdf` before merging new scores.

---

### 4. Package Import Warnings & Module Hygiene
> [!NOTE]
> **Locations:** `src/__init__.py`, `src/dataset/__init__.py`, `src/segmentation/__init__.py`
> **Issue:** `python -m src.dataset.build_input_stack` and `python -m src.segmentation.train` trigger Python `RuntimeWarning: ... found in sys.modules after import of package`.
> **Fix:** Explicitly define package exports in `__init__.py` files to prevent module namespace collisions.

---

## User Review Required

> [!TIP]
> All core deep learning segmentation architectures (U-Net 7-channel multi-spectral & 3-channel RGB fallback), sliding-window inference with Gaussian blending, vectorization, and Streamlit dashboard layout are fully intact. The proposed changes fix logic bugs in risk scoring and clean up data type handling.

---

## Proposed Changes

### Change Detection & Risk Scoring

#### [MODIFY] [risk_scoring.py](file:///d:/MUJ/Glacier%20lake/Glacier%20lake/src/change_detection/risk_scoring.py)
- Fix line 150-153 so text status strings are preserved and properly evaluated for Newly Formed (+25) and Drained (-60) risk adjustments.

#### [MODIFY] [classify_change.py](file:///d:/MUJ/Glacier%20lake/Glacier%20lake/src/change_detection/classify_change.py)
- Refactor GeoDataFrame assembly to use `summary_df.assign(geometry=geometries)`.

---

### Dashboard

#### [MODIFY] [app.py](file:///d:/MUJ/Glacier%20lake/Glacier%20lake/dashboard/app.py)
- Update `_backfill_risk()` to safely update risk attributes without creating duplicate column suffixes.

---

### Unit Tests

#### [MODIFY] [test_change_detection.py](file:///d:/MUJ/Glacier%20lake/Glacier%20lake/tests/test_change_detection.py)
- Add specific test assertion verifying that `"Newly Formed"` lakes receive the +25 risk bonus and `"Drained"` lakes receive the -60 risk penalty.

---

## Verification Plan

### Automated Tests
- Run complete test suite:
  `python -m unittest discover -s tests`
- Run pipeline CLI commands:
  - `python generate_sample_data.py`
  - `python -m src.dataset.build_input_stack`
  - `python -m src.change_detection.classify_change`
  - `python -m src.inference.run_inference`

### Manual Verification
- Dry-run `dashboard/app.py` import and launch Streamlit server.
- Verify FastAPI `/api/health` and `/api/segment` endpoints.
