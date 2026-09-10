"""
Evaluation harness for the Glacial Lake Detection pipeline.

Runs the full pipeline (segment → vectorize → classify → risk) on synthetic
test scenes with known ground truth, and computes standard segmentation metrics
(IoU, Dice, Precision, Recall) plus temporal change classification accuracy.

Usage:
    python -m src.evaluation.validate
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, Any

from ..segmentation.unet_model import LakeUNet
from ..dataset.build_input_stack import create_synthetic_glacial_scene
from ..dataset.normalize import ChannelNormalizer
from ..dataset.tile_raster import extract_patches_from_array
from ..dataset.lake_dataset import get_train_val_dataloaders
from ..postprocessing.vectorize import mask_to_geodataframe
from ..postprocessing.smooth_boundaries import clean_binary_mask
from ..change_detection.classify_change import classify_temporal_change, generate_change_summary_report
from ..change_detection.risk_scoring import add_risk_scores
from ..inference.run_inference import run_scene_inference, predict_large_scene_tiled
from ..inference.overlay import create_boundary_overlay
from ..utils.config import config
from ..utils.geo_utils import write_geotiff
from ..segmentation.losses import BCEDiceLoss
from ..segmentation.metrics import compute_metrics_from_tensors


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_full_pipeline_evaluation(
    seed_16: int = 42,
    seed_22: int = 100,
    tile_size: int = 256,
    stride: int = 128,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Run end-to-end pipeline on synthetic scenes and compute metrics.

    Returns a dict with:
    - segmentation: IoU, Dice, Precision, Recall, Accuracy
    - change_detection: per-class counts, match accuracy
    - risk_scoring: risk tier distribution
    """
    config.ensure_directories()

    # 1. Generate synthetic test scenes (using different seeds than training)
    print("Generating synthetic test scenes...")
    stack_16, mask_16, prof_16 = create_synthetic_glacial_scene(
        height=1024, width=1024, seed=seed_16, epoch=2016
    )
    stack_22, mask_22, prof_22 = create_synthetic_glacial_scene(
        height=1024, width=1024, seed=seed_22, epoch=2022
    )

    # 2. Normalize
    normalizer = ChannelNormalizer(config.BAND_STATS_PATH) if config.BAND_STATS_PATH.exists() else None
    if normalizer:
        norm_stack_16 = normalizer.transform(stack_16)
        norm_stack_22 = normalizer.transform(stack_22)
    else:
        norm_stack_16 = stack_16
        norm_stack_22 = stack_22

    # 3. Load models
    print("Loading models...")
    model_7ch = LakeUNet(n_channels=7, n_classes=1, base_filters=32).to(DEVICE)
    model_3ch = LakeUNet(n_channels=3, n_classes=1, base_filters=32).to(DEVICE)

    if config.MULTIBAND_MODEL_PATH.exists():
        state_dict = torch.load(config.MULTIBAND_MODEL_PATH, map_location=DEVICE, weights_only=True)
        model_7ch.load_state_dict(state_dict)
        print(f"Loaded multiband model from {config.MULTIBAND_MODEL_PATH}")
    else:
        print("WARNING: No multiband model weights found")

    if config.RGB_MODEL_PATH.exists():
        state_dict = torch.load(config.RGB_MODEL_PATH, map_location=DEVICE, weights_only=True)
        model_3ch.load_state_dict(state_dict)
        print(f"Loaded RGB model from {config.RGB_MODEL_PATH}")
    else:
        print("WARNING: No RGB model weights found")

    model_7ch.eval()
    model_3ch.eval()

    # 4. Run tiled inference on both scenes
    print("Running inference on test scenes...")
    prob_16 = predict_large_scene_tiled(norm_stack_16, model_7ch, DEVICE, tile_size=tile_size, stride=stride)
    prob_22 = predict_large_scene_tiled(norm_stack_22, model_7ch, DEVICE, tile_size=tile_size, stride=stride)

    binary_16 = (prob_16 >= threshold).astype(np.uint8)
    binary_22 = (prob_22 >= threshold).astype(np.uint8)

    cleaned_16 = clean_binary_mask(binary_16)
    cleaned_22 = clean_binary_mask(binary_22)

    # 5. Segmentation metrics (against ground truth masks)
    print("Computing segmentation metrics...")
    seg_metrics_16 = compute_metrics_from_tensors(
        torch.from_numpy(prob_16[None, None]).float().to(DEVICE),
        torch.from_numpy(mask_16[None, None]).float().to(DEVICE)
    )
    seg_metrics_22 = compute_metrics_from_tensors(
        torch.from_numpy(prob_22[None, None]).float().to(DEVICE),
        torch.from_numpy(mask_22[None, None]).float().to(DEVICE)
    )

    # 6. Vectorize
    print("Vectorizing...")
    gdf_16 = mask_to_geodataframe(
        cleaned_16, transform=prof_16["transform"], crs=prof_16["crs"]
    )
    gdf_22 = mask_to_geodataframe(
        cleaned_22, transform=prof_22["transform"], crs=prof_22["crs"]
    )

    # 7. Change detection
    print("Running change detection...")
    classified_gdf, summary_df = classify_temporal_change(gdf_16, gdf_22)

    # 8. Risk scoring
    summary_df = add_risk_scores(summary_df)

    # 9. Ground truth change classification (from synthetic scene generator)
    # We can derive GT change from the original synthetic masks
    gt_gdf_16 = mask_to_geodataframe(mask_16, transform=prof_16["transform"], crs=prof_16["crs"])
    gt_gdf_22 = mask_to_geodataframe(mask_22, transform=prof_22["transform"], crs=prof_22["crs"])
    gt_classified_gdf, gt_summary_df = classify_temporal_change(gt_gdf_16, gt_gdf_22)

    # 10. Change detection accuracy (match predicted vs GT status)
    change_accuracy = {}
    if len(summary_df) > 0 and len(gt_summary_df) > 0:
        # Compare by lake_id if possible, otherwise by count per status
        pred_counts = summary_df["status"].value_counts().to_dict()
        gt_counts = gt_summary_df["status"].value_counts().to_dict()
        all_statuses = ["Survived", "Newly Formed", "Drained"]
        for s in all_statuses:
            change_accuracy[s] = {
                "predicted": int(pred_counts.get(s, 0)),
                "ground_truth": int(gt_counts.get(s, 0)),
            }

    # 11. Risk tier distribution
    risk_counts = summary_df["risk"].value_counts().to_dict() if "risk" in summary_df.columns else {}

    # Compile results
    results = {
        "segmentation": {
            "epoch_2016": {k: float(v) for k, v in seg_metrics_16.items()},
            "epoch_2022": {k: float(v) for k, v in seg_metrics_22.items()},
            "mean": {
                "iou": float((seg_metrics_16["iou"] + seg_metrics_22["iou"]) / 2),
                "dice": float((seg_metrics_16["dice"] + seg_metrics_22["dice"]) / 2),
                "precision": float((seg_metrics_16["precision"] + seg_metrics_22["precision"]) / 2),
                "recall": float((seg_metrics_16["recall"] + seg_metrics_22["recall"]) / 2),
                "accuracy": float((seg_metrics_16["accuracy"] + seg_metrics_22["accuracy"]) / 2),
            }
        },
        "change_detection": change_accuracy,
        "risk_scoring": {
            "distribution": {k: int(v) for k, v in risk_counts.items()},
            "total_lakes": int(len(summary_df)),
        },
        "config": {
            "tile_size": tile_size,
            "stride": stride,
            "threshold": threshold,
            "device": str(DEVICE),
            "model_7ch_loaded": config.MULTIBAND_MODEL_PATH.exists(),
            "model_3ch_loaded": config.RGB_MODEL_PATH.exists(),
        }
    }

    return results


def run_real_data_validation(
    lat: float = 36.58,
    lon: float = 74.65,
    zoom: int = 13,
) -> Dict[str, Any]:
    """
    Fetch a real satellite scene via Esri fallback and run inference.
    Since we don't have ground truth for real data, we report raw predictions.
    """
    from ..inference.satellite import fetch_satellite_scene
    from io import BytesIO
    from PIL import Image

    print(f"Fetching real satellite scene at {lat:.4f}°N, {lon:.4f}°E...")
    rgb, source, transform = fetch_satellite_scene(lat, lon, zoom=zoom, size=config.YANDEX_IMAGE_SIZE)

    buf = BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    buf.seek(0)

    result = run_scene_inference(buf, threshold=0.5, override_transform=transform)

    return {
        "source": source,
        "image_shape": list(rgb.shape),
        "lake_count": result["lake_count"],
        "total_lake_area_sqkm": result["total_lake_area_sqkm"],
        "cloud_coverage_pct": result.get("cloud_coverage_pct", 0.0),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("GLACIAL LAKE DETECTION PIPELINE EVALUATION")
    print("=" * 60)

    # Run synthetic evaluation
    results = run_full_pipeline_evaluation()

    # Run real data validation
    print("\n" + "=" * 60)
    real_results = run_real_data_validation()
    results["real_data_validation"] = real_results

    # Save report
    output_dir = config.OUTPUTS_DIR / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"\nSegmentation (mean):")
    for k, v in results["segmentation"]["mean"].items():
        print(f"  {k.capitalize()}: {v:.4f}")
    print(f"\nChange Detection:")
    for status, counts in results["change_detection"].items():
        print(f"  {status}: pred={counts['predicted']}, gt={counts['ground_truth']}")
    print(f"\nRisk Distribution: {results['risk_scoring']['distribution']}")
    print(f"\nReal Data Validation ({results['real_data_validation']['source']}):")
    print(f"  Lakes detected: {results['real_data_validation']['lake_count']}")
    print(f"  Total area: {results['real_data_validation']['total_lake_area_sqkm']:.3f} km²")
    print(f"\nReport saved to: {report_path}")