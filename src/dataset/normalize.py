"""
Channel normalization module for multi-spectral and topographic raster stacks.
Computes and serializes per-channel mean and standard deviation statistics.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import torch

class ChannelNormalizer:
    def __init__(self, stats_path: Optional[Union[str, Path]] = None):
        self.stats_path = Path(stats_path) if stats_path else None
        self.means: List[float] = []
        self.stds: List[float] = []
        
        if self.stats_path and self.stats_path.exists():
            self.load_stats(self.stats_path)

    def fit(self, data_stack: np.ndarray, band_names: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        """
        Compute mean and std across spatial dimensions for each channel.
        Parameters:
            data_stack: numpy array of shape (C, H, W) or (N, C, H, W)
        """
        if data_stack.ndim == 3:
            num_channels = data_stack.shape[0]
            axis = (1, 2)
        elif data_stack.ndim == 4:
            num_channels = data_stack.shape[1]
            axis = (0, 2, 3)
        else:
            raise ValueError("Expected data_stack with 3 or 4 dimensions.")

        self.means = []
        self.stds = []
        stats_dict = {}

        for c in range(num_channels):
            channel_data = data_stack[c] if data_stack.ndim == 3 else data_stack[:, c]
            valid_pixels = channel_data[~np.isnan(channel_data) & ~np.isinf(channel_data)]
            
            if len(valid_pixels) == 0:
                mean_val, std_val = 0.0, 1.0
            else:
                mean_val = float(np.mean(valid_pixels))
                std_val = float(np.std(valid_pixels))
                if std_val < 1e-6:
                    std_val = 1.0
                    
            self.means.append(mean_val)
            self.stds.append(std_val)
            
            b_name = band_names[c] if band_names and c < len(band_names) else f"channel_{c}"
            stats_dict[b_name] = {"mean": mean_val, "std": std_val}

        return stats_dict

    def transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Applies z-score standardization per channel: (x - mean) / std.
        """
        is_torch = isinstance(data, torch.Tensor)
        arr = data.clone() if is_torch else data.copy()

        num_channels = arr.shape[0] if arr.ndim == 3 else arr.shape[1]
        for c in range(min(num_channels, len(self.means))):
            m = self.means[c]
            s = self.stds[c] if self.stds[c] > 1e-6 else 1.0
            if arr.ndim == 3:
                arr[c] = (arr[c] - m) / s
            else:
                arr[:, c] = (arr[:, c] - m) / s
                
        return arr

    def save_stats(self, output_path: Union[str, Path]) -> None:
        """Serialize normalization parameters to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        stats = {
            "means": self.means,
            "stds": self.stds,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

    def load_stats(self, input_path: Union[str, Path]) -> None:
        """Load normalization parameters from JSON."""
        with open(input_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        self.means = stats["means"]
        self.stds = stats["stds"]
