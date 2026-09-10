"""
Unit tests for multi-temporal matching and change classification.
"""

import unittest
from shapely.geometry import Point, box
import geopandas as gpd
from src.change_detection.polygon_matcher import match_lake_polygons_across_epochs
from src.change_detection.classify_change import classify_temporal_change, generate_change_summary_report

class TestChangeDetection(unittest.TestCase):
    def setUp(self):
        # Create test geometries
        # Lake 1: In both Epoch 1 and Epoch 2 (Survived, grew in area)
        geom1_y1 = box(0.0, 0.0, 0.02, 0.02)
        geom1_y2 = box(0.0, 0.0, 0.03, 0.03)

        # Lake 2: In Epoch 1 only (Drained)
        geom2_y1 = box(0.10, 0.10, 0.12, 0.12)

        # Lake 3: In Epoch 2 only (Newly Formed)
        geom3_y2 = box(0.30, 0.30, 0.32, 0.32)

        self.gdf_y1 = gpd.GeoDataFrame(
            [{"lake_id": "L1_16", "area_sqkm": 0.5}, {"lake_id": "L2_16", "area_sqkm": 0.2}],
            geometry=[geom1_y1, geom2_y1],
            crs="EPSG:4326"
        )
        self.gdf_y2 = gpd.GeoDataFrame(
            [{"lake_id": "L1_22", "area_sqkm": 0.9}, {"lake_id": "L3_22", "area_sqkm": 0.3}],
            geometry=[geom1_y2, geom3_y2],
            crs="EPSG:4326"
        )

    def test_matching_logic(self):
        matches, drained, new = match_lake_polygons_across_epochs(self.gdf_y1, self.gdf_y2)
        self.assertEqual(len(matches), 1)
        self.assertEqual(len(drained), 1)
        self.assertEqual(len(new), 1)

    def test_classification_logic(self):
        classified_gdf, summary_df = classify_temporal_change(self.gdf_y1, self.gdf_y2)
        
        statuses = summary_df["status"].tolist()
        self.assertIn("Survived", statuses)
        self.assertIn("Newly Formed", statuses)
        self.assertIn("Drained", statuses)
        
        report = generate_change_summary_report(summary_df)
        self.assertEqual(report["newly_formed_count"], 1)
        self.assertEqual(report["survived_count"], 1)
        self.assertEqual(report["drained_count"], 1)

        # Assert GLOF risk score adjustments for Newly Formed (+25) vs Drained (-60)
        new_lake = summary_df[summary_df["status"] == "Newly Formed"].iloc[0]
        drained_lake = summary_df[summary_df["status"] == "Drained"].iloc[0]
        self.assertGreater(new_lake["risk_score"], 40.0)
        self.assertLess(drained_lake["risk_score"], 10.0)

if __name__ == "__main__":
    unittest.main()
