# src/change_detection/__init__.py
from .polygon_matcher import match_lake_polygons_across_epochs
from .classify_change import (
    classify_temporal_change,
    generate_change_summary_report,
    run_change_detection_pipeline,
)

__all__ = [
    "match_lake_polygons_across_epochs",
    "classify_temporal_change",
    "generate_change_summary_report",
    "run_change_detection_pipeline",
]
