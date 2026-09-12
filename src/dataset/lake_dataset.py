"""
PyTorch Dataset and DataLoader constructors for Glacial Lake Segmentation.
Supports spatial augmentations (random flips, 90-degree rotations) and normalization.
"""

from typing import List, Tuple, Optional, Union
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class GlacialLakeDataset(Dataset):
    def __init__(
        self,
        images: List[np.ndarray],
        masks: Optional[List[np.ndarray]] = None,
        is_rgb_only: bool = False,
        augment: bool = True
    ):
        """
        Parameters:
            images: List of (C, H, W) numpy arrays
            masks: Optional list of (1, H, W) or (H, W) binary arrays
            is_rgb_only: If True, only extracts bands [2, 1, 0] (RGB)
            augment: If True, applies random flips and rotations
        """
        self.images = images
        self.masks = masks
        self.is_rgb_only = is_rgb_only
        self.augment = augment

    def __len__(self) -> int:
        return len(self.images)

    def _apply_augmentations(self, img: np.ndarray, mask: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        # Random horizontal flip
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=2).copy()
            if mask is not None:
                mask = np.flip(mask, axis=2).copy()
                
        # Random vertical flip
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=1).copy()
            if mask is not None:
                mask = np.flip(mask, axis=1).copy()
                
        # Random 90 deg rotation
        k = np.random.choice([0, 1, 2, 3])
        if k > 0:
            img = np.rot90(img, k, axes=(1, 2)).copy()
            if mask is not None:
                mask = np.rot90(mask, k, axes=(1, 2)).copy()

        return img, mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Union[torch.Tensor, torch.Tensor]]:
        img = self.images[idx].astype(np.float32)
        
        # If RGB model requested, select Red (index 2), Green (index 1), Blue (index 0)
        if self.is_rgb_only:
            if img.shape[0] >= 3:
                img = img[[2, 1, 0], :, :]
            elif img.shape[0] == 1:
                img = np.repeat(img, 3, axis=0)

        mask = None
        if self.masks is not None:
            m = self.masks[idx].astype(np.float32)
            if m.ndim == 2:
                m = m[np.newaxis, ...]
            mask = m

        if self.augment:
            img, mask = self._apply_augmentations(img, mask)

        img_tensor = torch.from_numpy(img).float()
        
        if mask is not None:
            mask_tensor = torch.from_numpy((mask > 0.5).astype(np.float32)).float()
            return img_tensor, mask_tensor
        else:
            return img_tensor, torch.zeros((1, img.shape[1], img.shape[2]), dtype=torch.float32)

def get_train_val_dataloaders(
    images: List[np.ndarray],
    masks: List[np.ndarray],
    batch_size: int = 8,
    val_split: float = 0.20,
    is_rgb_only: bool = False,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates stratified training and validation PyTorch DataLoaders.
    """
    np.random.seed(seed)
    total_samples = len(images)
    indices = np.random.permutation(total_samples)
    
    val_count = int(total_samples * val_split)
    val_idx = indices[:val_count]
    train_idx = indices[val_count:]

    train_imgs = [images[i] for i in train_idx]
    train_masks = [masks[i] for i in train_idx]
    val_imgs = [images[i] for i in val_idx]
    val_masks = [masks[i] for i in val_idx]

    train_dataset = GlacialLakeDataset(train_imgs, train_masks, is_rgb_only=is_rgb_only, augment=True)
    val_dataset = GlacialLakeDataset(val_imgs, val_masks, is_rgb_only=is_rgb_only, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=len(train_dataset) > batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
