"""
Evaluation metrics for glacial lake semantic segmentation.
Computes Intersection over Union (IoU / Jaccard Index), Dice Coefficient (F1-score),
Precision, Recall, and Accuracy.
"""

from typing import Dict
import torch
import numpy as np

def compute_metrics_from_tensors(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7
) -> Dict[str, float]:
    """
    Computes segmentation metrics from PyTorch tensors.
    """
    if preds.ndim == 4 and preds.shape[1] == 1:
        probs = torch.sigmoid(preds) if preds.min() < 0 or preds.max() > 1 else preds
        binary_preds = (probs >= threshold).float()
    else:
        binary_preds = (preds >= threshold).float()

    binary_targets = (targets >= threshold).float()

    p_flat = binary_preds.view(-1)
    t_flat = binary_targets.view(-1)

    true_pos = (p_flat * t_flat).sum().item()
    false_pos = (p_flat * (1.0 - t_flat)).sum().item()
    false_neg = ((1.0 - p_flat) * t_flat).sum().item()
    true_neg = ((1.0 - p_flat) * (1.0 - t_flat)).sum().item()

    iou = (true_pos + eps) / (true_pos + false_pos + false_neg + eps)
    dice = (2.0 * true_pos + eps) / (2.0 * true_pos + false_pos + false_neg + eps)
    precision = (true_pos + eps) / (true_pos + false_pos + eps)
    recall = (true_pos + eps) / (true_pos + false_neg + eps)
    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg + eps)

    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
    }

def compute_iou(mask1: np.ndarray, mask2: np.ndarray, eps: float = 1e-7) -> float:
    """Computes IoU between two numpy boolean masks."""
    b1 = mask1 > 0
    b2 = mask2 > 0
    intersection = np.logical_and(b1, b2).sum()
    union = np.logical_or(b1, b2).sum()
    return float((intersection + eps) / (union + eps))

def compute_dice(mask1: np.ndarray, mask2: np.ndarray, eps: float = 1e-7) -> float:
    """Computes Dice (F1) between two numpy boolean masks."""
    b1 = mask1 > 0
    b2 = mask2 > 0
    intersection = np.logical_and(b1, b2).sum()
    cardinality = b1.sum() + b2.sum()
    return float((2.0 * intersection + eps) / (cardinality + eps))

def compute_precision_recall(pred_mask: np.ndarray, target_mask: np.ndarray, eps: float = 1e-7) -> Dict[str, float]:
    """Computes Precision and Recall on numpy arrays."""
    p = pred_mask > 0
    t = target_mask > 0
    tp = np.logical_and(p, t).sum()
    fp = np.logical_and(p, ~t).sum()
    fn = np.logical_and(~p, t).sum()
    
    precision = float((tp + eps) / (tp + fp + eps))
    recall = float((tp + eps) / (tp + fn + eps))
    return {"precision": precision, "recall": recall}
