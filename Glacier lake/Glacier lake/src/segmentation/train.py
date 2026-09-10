"""
Training script and pipeline for Glacial Lake Segmentation models.
Trains multi-channel U-Net and RGB fallback models with early stopping and checkpointing.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np

from .unet_model import LakeUNet, UNet
from .losses import BCEDiceLoss
from .metrics import compute_metrics_from_tensors
from ..dataset.build_input_stack import create_synthetic_glacial_scene
from ..dataset.tile_raster import extract_patches_from_array
from ..dataset.lake_dataset import get_train_val_dataloaders
from ..dataset.normalize import ChannelNormalizer
from ..utils.config import config

def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device
) -> Tuple[float, Dict[str, float]]:
    model.train()
    total_loss = 0.0
    total_metrics = {"iou": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0}

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batch_metrics = compute_metrics_from_tensors(logits.detach(), targets)
        for k in total_metrics:
            total_metrics[k] += batch_metrics[k]

    num_batches = max(len(loader), 1)
    avg_loss = total_loss / num_batches
    avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
    return avg_loss, avg_metrics

def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total_metrics = {"iou": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0}

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            loss = criterion(logits, targets)

            total_loss += loss.item()
            batch_metrics = compute_metrics_from_tensors(logits, targets)
            for k in total_metrics:
                total_metrics[k] += batch_metrics[k]

    num_batches = max(len(loader), 1)
    avg_loss = total_loss / num_batches
    avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
    return avg_loss, avg_metrics

def train_segmentation_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_channels: int = 7,
    num_epochs: int = 30,
    learning_rate: float = 1e-3,
    save_path: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None
) -> Tuple[torch.nn.Module, Dict[str, list]]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LakeUNet(n_channels=num_channels, n_classes=1, base_filters=32).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)
    criterion = BCEDiceLoss(dice_weight=0.5, bce_weight=0.5)

    history = {"train_loss": [], "val_loss": [], "val_iou": [], "val_dice": []}
    best_iou = 0.0
    best_weights = None

    print(f"--- Starting Lake Segmentation Training ({num_channels} channels) on {device} ---")
    
    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = validate(model, val_loader, criterion, device)

        scheduler.step(val_metrics["iou"])

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_metrics["iou"])
        history["val_dice"].append(val_metrics["dice"])

        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            best_weights = model.state_dict().copy()
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_weights, save_path)

        if epoch % 5 == 0 or epoch == num_epochs:
            print(
                f"Epoch [{epoch:02d}/{num_epochs:02d}] "
                f"Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val IoU: {val_metrics['iou']:.4f} | Val Dice: {val_metrics['dice']:.4f}"
            )

    if best_weights is not None:
        model.load_state_dict(best_weights)
        
    print(f"Training Complete! Best Val IoU: {best_iou:.4f}")
    return model, history

def train_and_save_benchmark_models():
    """Generates synthetic multi-temporal training scenes and trains both multi-band and RGB models."""
    config.ensure_directories()
    
    print("1. Generating Synthetic Himalayan Multi-temporal Scenes...")
    stack_2016, mask_2016, _ = create_synthetic_glacial_scene(height=1024, width=1024, seed=42, epoch=2016)
    stack_2022, mask_2022, _ = create_synthetic_glacial_scene(height=1024, width=1024, seed=100, epoch=2022)

    # Normalize channels
    normalizer = ChannelNormalizer()
    normalizer.fit(stack_2016)
    normalizer.save_stats(config.BAND_STATS_PATH)
    
    norm_stack_2016 = normalizer.transform(stack_2016)
    norm_stack_2022 = normalizer.transform(stack_2022)

    # Tile both scenes
    img_tiles_16, mask_tiles_16, _ = extract_patches_from_array(norm_stack_2016, mask_2016, tile_size=256, stride=128)
    img_tiles_22, mask_tiles_22, _ = extract_patches_from_array(norm_stack_2022, mask_2022, tile_size=256, stride=128)

    all_images = img_tiles_16 + img_tiles_22
    all_masks = mask_tiles_16 + mask_tiles_22

    # A. Train Multi-Band Model (7 Channels)
    tr_loader, val_loader = get_train_val_dataloaders(all_images, all_masks, batch_size=8, is_rgb_only=False)
    train_segmentation_model(
        tr_loader, val_loader,
        num_channels=7,
        num_epochs=15,
        save_path=config.MULTIBAND_MODEL_PATH
    )

    # B. Train RGB Fallback Model (3 Channels)
    tr_loader_rgb, val_loader_rgb = get_train_val_dataloaders(all_images, all_masks, batch_size=8, is_rgb_only=True)
    train_segmentation_model(
        tr_loader_rgb, val_loader_rgb,
        num_channels=3,
        num_epochs=15,
        save_path=config.RGB_MODEL_PATH
    )

if __name__ == "__main__":
    train_and_save_benchmark_models()
