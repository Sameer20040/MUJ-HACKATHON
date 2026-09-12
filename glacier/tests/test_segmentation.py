"""
Unit tests for U-Net model forward pass, loss functions, and metrics.
"""

import unittest
import torch
import numpy as np
from src.segmentation.unet_model import LakeUNet, UNet
from src.segmentation.losses import BCEDiceLoss, DiceLoss
from src.segmentation.metrics import compute_metrics_from_tensors, compute_iou

class TestSegmentation(unittest.TestCase):
    def test_multiband_unet_forward(self):
        model = LakeUNet.create_multiband_model(n_channels=7)
        x = torch.randn(2, 7, 128, 128)
        out = model(x)
        self.assertEqual(out.shape, (2, 1, 128, 128))

    def test_rgb_fallback_unet_forward(self):
        model = LakeUNet.create_rgb_fallback_model()
        x = torch.randn(2, 3, 128, 128)
        out = model(x)
        self.assertEqual(out.shape, (2, 1, 128, 128))

    def test_bce_dice_loss(self):
        criterion = BCEDiceLoss(dice_weight=0.5, bce_weight=0.5)
        logits = torch.randn(2, 1, 64, 64)
        targets = torch.randint(0, 2, (2, 1, 64, 64)).float()
        loss = criterion(logits, targets)
        self.assertTrue(torch.is_tensor(loss))
        self.assertGreater(loss.item(), 0.0)

    def test_iou_metric(self):
        m1 = np.zeros((50, 50), dtype=bool)
        m2 = np.zeros((50, 50), dtype=bool)
        m1[10:30, 10:30] = True
        m2[10:30, 10:30] = True
        iou = compute_iou(m1, m2)
        self.assertAlmostEqual(iou, 1.0, places=3)

if __name__ == "__main__":
    unittest.main()
