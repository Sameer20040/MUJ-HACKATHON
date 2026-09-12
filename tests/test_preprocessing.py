"""
Unit tests for data preprocessing and spectral index computations.
"""

import unittest
import numpy as np
from src.preprocessing.compute_indices import compute_ndwi, compute_mndwi, approximate_ndwi_from_rgb
from src.preprocessing.cloud_mask import create_cloud_mask_threshold, apply_mask_to_multiband
from src.preprocessing.sar_speckle import apply_lee_filter

class TestPreprocessing(unittest.TestCase):
    def test_compute_ndwi(self):
        green = np.array([[0.5, 0.2], [0.8, 0.1]], dtype=np.float32)
        nir = np.array([[0.1, 0.6], [0.2, 0.9]], dtype=np.float32)
        ndwi = compute_ndwi(green, nir)
        
        self.assertEqual(ndwi.shape, (2, 2))
        self.assertGreater(ndwi[0, 0], 0.0) # Water pixel
        self.assertLess(ndwi[0, 1], 0.0)    # Non-water pixel

    def test_approximate_ndwi_from_rgb(self):
        # Create a water-like RGB pixel (high blue/green, low red)
        rgb = np.zeros((10, 10, 3), dtype=np.uint8)
        rgb[:, :, 0] = 30   # Red
        rgb[:, :, 1] = 120  # Green
        rgb[:, :, 2] = 180  # Blue
        
        score = approximate_ndwi_from_rgb(rgb)
        self.assertEqual(score.shape, (10, 10))
        self.assertTrue(np.all(score > 0.0))

    def test_apply_lee_filter(self):
        sar = np.random.uniform(0.01, 1.0, (64, 64)).astype(np.float32)
        filtered = apply_lee_filter(sar, window_size=5)
        self.assertEqual(filtered.shape, sar.shape)
        self.assertFalse(np.isnan(filtered).any())

if __name__ == "__main__":
    unittest.main()
