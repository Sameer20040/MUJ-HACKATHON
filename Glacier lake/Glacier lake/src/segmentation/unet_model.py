"""
Deep Learning U-Net Architecture for Glacial Lake Segmentation.
Supports arbitrary input channels (e.g., 7-channel multi-spectral+indices+DEM or 3-channel RGB).
Includes batch normalization, residual features, and sigmoid/logit output.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class DoubleConv(nn.Module):
    """(Convolution => [BatchNorm] => ReLU) * 2"""
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels: int = 7, n_classes: int = 1, bilinear: bool = False, base_filters: int = 32):
        """
        Parameters:
            n_channels: Number of input channels (7 for Multi-band, 3 for RGB)
            n_classes: Number of output channels (1 for binary lake mask)
            bilinear: If True uses bilinear interpolation for upsampling
            base_filters: Base number of convolution filters (default 32)
        """
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        f = base_filters
        self.inc = DoubleConv(n_channels, f)
        self.down1 = Down(f, f * 2)
        self.down2 = Down(f * 2, f * 4)
        self.down3 = Down(f * 4, f * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(f * 8, f * 16 // factor)
        
        self.up1 = Up(f * 16, f * 8 // factor, bilinear)
        self.up2 = Up(f * 8, f * 4 // factor, bilinear)
        self.up3 = Up(f * 4, f * 2 // factor, bilinear)
        self.up4 = Up(f * 2, f, bilinear)
        self.outc = OutConv(f, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class LakeUNet(UNet):
    """Convenience alias and factory for Lake segmentation models."""
    @classmethod
    def create_multiband_model(cls, n_channels: int = 7) -> 'LakeUNet':
        return cls(n_channels=n_channels, n_classes=1, base_filters=32)

    @classmethod
    def create_rgb_fallback_model(cls) -> 'LakeUNet':
        return cls(n_channels=3, n_classes=1, base_filters=32)
