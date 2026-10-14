"""
3D Squeeze-and-Excitation ResNet (SE-ResNet3D) Architecture.

Core tumor presence classifier for weakly supervised 3D localization.
Extracts 3D volumetric representations with channel-wise self-attention
to suppress non-lesion soft-tissue background noise while enhancing focal tumor signals.
Matches the pre-trained weights in checkpoints/se_resnet3d_best.pt.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock3D(nn.Module):
    """
    Squeeze-and-Excitation 3D Channel Attention Block.
    Recalibrates channel feature maps by:
    1. Global Average Pooling (GAP) across spatial depth, height, width.
    2. Two-layer MLP bottleneck with ReLU and Sigmoid gating.
    3. Channel-wise scaling to suppress scanner background and amplify lesion signal.
    """
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.fc1 = nn.Linear(channels, max(channels // reduction, 4))
        self.fc2 = nn.Linear(max(channels // reduction, 4), channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _, _ = x.size()
        y = F.adaptive_avg_pool3d(x, 1).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1, 1)
        return x * y


class SEResNetBlock3D(nn.Module):
    """Residual block with GroupNorm and 3D Squeeze-and-Excitation attention."""
    def __init__(self, in_c: int, out_c: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(8, out_c)
        self.conv2 = nn.Conv3d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(8, out_c)
        self.se = SEBlock3D(out_c)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(8, out_c)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out = self.se(out)
        out = F.relu(out + res)
        return out


class SEResNet3D(nn.Module):
    """
    SOTA 3D Residual Network with Channel Attention.
    Pre-trained with weak binary labels on multi-center MRI cohorts.
    """
    def __init__(self, in_channels: int = 1, base_c: int = 24, base_channels: int | None = None) -> None:
        super().__init__()
        if base_channels is not None:
            base_c = base_channels
        self.base_c = base_c
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, base_c),
            nn.ReLU(inplace=True)
        )
        self.layer1 = SEResNetBlock3D(base_c, base_c, stride=1)
        self.layer2 = SEResNetBlock3D(base_c, base_c * 2, stride=2)
        self.layer3 = SEResNetBlock3D(base_c * 2, base_c * 4, stride=2)
        self.layer4 = SEResNetBlock3D(base_c * 4, base_c * 8, stride=2)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(base_c * 8, 1)
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.pool(x).view(x.size(0), -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.extract_features(x)
        return self.fc(feat)
