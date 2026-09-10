"""
FastAPI inference service for Glacial Lake Detection.

Provides a REST API for running the U-Net segmentation pipeline on uploaded
satellite images, returning detected lakes with GLOF risk scores.

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /api/health           - Health check
    POST /api/segment          - Segment uploaded image, return lakes + risk
"""

from __future__ import annotations

import io
import json
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image
import numpy as np

from src.inference.run_inference import run_scene_inference
from src.change_detection.risk_scoring import summarize_risk
from src.utils.config import config

app = FastAPI(
    title="Glacial Lake AI API",
    description="Multi-temporal glacial lake detection and GLOF risk scoring",
    version="1.0.0",
)

# CORS for local development / demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LakeInfo(BaseModel):
    """Single detected lake with risk assessment."""
    lake_id: str
    status: str
    risk: str
    risk_score: float
    area_sqkm: float = Field(..., alias="area_sqkm")
    area_change_pct: Optional[float] = None
    centroid_lon: float
    centroid_lat: float
    recommended_action: str


class SegmentationResponse(BaseModel):
    """Response from /api/segment endpoint."""
    success: bool
    source: str = "upload"
    image_shape: list[int]
    lake_count: int
    total_lake_area_sqkm: float
    cloud_coverage_pct: float
    lakes: list[LakeInfo]
    risk_summary: dict


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model": "unet_rgb_fallback",
        "multiband_model_available": config.MULTIBAND_MODEL_PATH.exists(),
        "rgb_model_available": config.RGB_MODEL_PATH.exists(),
    }


@app.post("/api/segment", response_model=SegmentationResponse)
async def segment_image(
    file: UploadFile = File(..., description="Satellite image (JPG, PNG, TIF, TIFF)"),
    threshold: float = Form(0.50, ge=0.1, le=0.9, description="Water probability threshold"),
    min_area_pixels: int = Form(20, ge=5, le=500, description="Minimum lake size in pixels"),
):
    """
    Run glacial lake segmentation on an uploaded satellite image.

    Accepts JPG, PNG, TIF, or TIFF files. GeoTIFF files with multi-spectral
    bands will use the 7-channel model; RGB images use the 3-channel fallback model.

    Returns:
    - lake_count: number of detected lakes
    - total_lake_area_sqkm: sum of lake areas
    - lakes: list of lakes with geometry, status, and GLOF risk scores
    - risk_summary: aggregated risk statistics
    """
    # Read file
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    # Run inference
    try:
        # Wrap bytes in BytesIO for the inference pipeline
        bio = io.BytesIO(contents)
        result = run_scene_inference(
            bio,
            threshold=threshold,
            min_area_pixels=min_area_pixels,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    # Extract GeoDataFrame
    gdf = result["geodataframe"]

    # Build lakes list
    lakes = []
    if gdf is not None and not gdf.empty:
        for _, row in gdf.iterrows():
            pt = row.geometry.centroid
            lakes.append(LakeInfo(
                lake_id=str(row.get("lake_id", "unknown")),
                status=str(row.get("status", "Detected")),
                risk=str(row.get("risk", "Low")),
                risk_score=float(row.get("risk_score", 0.0)),
                area_sqkm=float(row.get("area_sqkm", row.get("area_2022_sqkm", 0.0))),
                area_change_pct=float(row.get("area_change_pct", 0.0)) if row.get("area_change_pct") is not None else None,
                centroid_lon=float(pt.x),
                centroid_lat=float(pt.y),
                recommended_action=str(row.get("recommended_action", "Monitor")),
            ))

    # Risk summary
    risk_summary = {}
    if len(lakes) > 0:
        # Create a temp DataFrame for summarize_risk
        import pandas as pd
        summary_df = pd.DataFrame([lake.model_dump() for lake in lakes])
        risk_summary = summarize_risk(summary_df)

    return SegmentationResponse(
        success=True,
        source="upload",
        image_shape=list(result["display_rgb"].shape),
        lake_count=result["lake_count"],
        total_lake_area_sqkm=result["total_lake_area_sqkm"],
        cloud_coverage_pct=result.get("cloud_coverage_pct", 0.0),
        lakes=lakes,
        risk_summary=risk_summary,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)