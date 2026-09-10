"""
Loss functions for glacial lake semantic segmentation.
Implements Soft Dice Loss, Binary Cross Entropy (BCE), and Combined BCE+Dice Loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0, eps: float = 1e-7):
        super().__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            logits: (B, 1, H, W) raw un-normalized model outputs
            targets: (B, 1, H, W) binary ground truth (0 or 1)
        """
        probs = torch.sigmoid(logits)
        
        # Flatten batch and spatial dimensions
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (probs_flat * targets_flat).sum()
        cardinality = probs_flat.sum() + targets_flat.sum()
        
        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth + self.eps)
        return 1.0 - dice_score

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1.0 - p_t) ** self.gamma
        alpha_factor = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        loss = alpha_factor * focal_weight * bce_loss
        return loss.mean()

class BCEDiceLoss(nn.Module):
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5, pos_weight: float = 1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_fn = DiceLoss()
        self.bce_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]) if pos_weight != 1.0 else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce_fn(logits, targets)
        dice = self.dice_fn(logits, targets)
        return self.bce_weight * bce + self.dice_weight * dice
