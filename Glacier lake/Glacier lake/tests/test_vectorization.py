"""
Unit tests for postprocessing and raster-to-vector polygon conversion.
"""

import unittest
import numpy as np
import cv2
from src.postprocessing.smooth_boundaries import clean_binary_mask
from src.postprocessing.vectorize import mask_to_geodataframe, mask_to_polygons

class TestVectorization(unittest.TestCase):
    def test_clean_binary_mask(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        # Small isolated noise (4 pixels)
        mask[10:12, 10:12] = 1
        # Real lake (400 pixels)
        mask[40:60, 40:60] = 1
        
        cleaned = clean_binary_mask(mask, min_size=25)
        self.assertEqual(cleaned[10:12, 10:12].sum(), 0) # Noise removed
        self.assertGreater(cleaned[40:60, 40:60].sum(), 300) # Real lake preserved

    def test_mask_to_geodataframe(self):
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(mask, (100, 100), 20, 1, -1)
        
        gdf = mask_to_geodataframe(mask, min_area_pixels=10)
        self.assertEqual(len(gdf), 1)
        self.assertIn("lake_id", gdf.columns)
        self.assertIn("area_sqkm", gdf.columns)
        self.assertGreater(gdf.iloc[0]["area_sqkm"], 0.0)

if __name__ == "__main__":
    unittest.main()
