"""
Cross-Modality Neural Translation Bridge: 3D/2.5D CycleGAN Training.

NOTE: This module is currently under active development/extension for full multi-center
unpaired MRI <-> CT volumetric translation. It provides the standalone training pipeline
for learning bidirectional mapping G: MRI -> CT and F: CT -> MRI without paired scans.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim


# -----------------------------------------------------------------------------
# 1. 3D CycleGAN Generator & PatchGAN Discriminator Architectures
# -----------------------------------------------------------------------------

class ResNetBlock3D(nn.Module):
    """Residual block with InstanceNorm for cycle-consistent translation."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x)


class CycleGANGenerator3D(nn.Module):
    """Encoder-ResNet-Decoder 3D Generator for cross-domain translation."""
    def __init__(self, in_c: int = 1, out_c: int = 1, base_c: int = 32, num_blocks: int = 4):
        super().__init__()
        # Initial convolution
        layers = [
            nn.Conv3d(in_c, base_c, kernel_size=7, padding=3, bias=False),
            nn.InstanceNorm3d(base_c),
            nn.ReLU(inplace=True)
        ]
        # Downsampling
        curr_c = base_c
        for _ in range(2):
            layers += [
                nn.Conv3d(curr_c, curr_c * 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm3d(curr_c * 2),
                nn.ReLU(inplace=True)
            ]
            curr_c *= 2

        # Residual blocks
        for _ in range(num_blocks):
            layers.append(ResNetBlock3D(curr_c))

        # Upsampling
        for _ in range(2):
            layers += [
                nn.ConvTranspose3d(curr_c, curr_c // 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                nn.InstanceNorm3d(curr_c // 2),
                nn.ReLU(inplace=True)
            ]
            curr_c //= 2

        # Output projection
        layers += [
            nn.Conv3d(base_c, out_c, kernel_size=7, padding=3),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class PatchGANDiscriminator3D(nn.Module):
    """3D PatchGAN Discriminator distinguishing real vs. synthesized patches."""
    def __init__(self, in_c: int = 1, base_c: int = 32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv3d(in_c, base_c, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(base_c, base_c * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(base_c * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(base_c * 2, base_c * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(base_c * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(base_c * 4, 1, kernel_size=4, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# -----------------------------------------------------------------------------
# 2. CycleGAN Training Loop
# -----------------------------------------------------------------------------

def train_cyclegan(args: argparse.Namespace):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"=== 3D CycleGAN Domain Translation Training on {device} ===")
    print("[Status] Module under process: initializing architectures and loss functions...")

    # Generators: G: MRI -> CT, F: CT -> MRI
    G_mri2ct = CycleGANGenerator3D().to(device)
    F_ct2mri = CycleGANGenerator3D().to(device)

    # Discriminators: D_ct, D_mri
    D_ct = PatchGANDiscriminator3D().to(device)
    D_mri = PatchGANDiscriminator3D().to(device)

    # Losses
    criterion_gan = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    optimizer_G = optim.Adam(
        list(G_mri2ct.parameters()) + list(F_ct2mri.parameters()),
        lr=args.lr, betas=(0.5, 0.999)
    )
    optimizer_D = optim.Adam(
        list(D_ct.parameters()) + list(D_mri.parameters()),
        lr=args.lr, betas=(0.5, 0.999)
    )

    print(f"[CycleGAN] Initialized generators and discriminators successfully.")
    print(f"[CycleGAN] Ready for unpaired batch loading. Running demonstration step...")

    # Mock batch for verification
    real_mri = torch.randn(1, 1, 32, 32, 16, device=device)
    real_ct = torch.randn(1, 1, 32, 32, 16, device=device)

    # Forward translation
    fake_ct = G_mri2ct(real_mri)
    rec_mri = F_ct2mri(fake_ct)

    fake_mri = F_ct2mri(real_ct)
    rec_ct = G_mri2ct(fake_mri)

    # Compute cycle loss
    loss_cycle = (criterion_cycle(rec_mri, real_mri) + criterion_cycle(rec_ct, real_ct)) * args.lambda_cycle
    print(f"[CycleGAN] Demo Cycle Loss: {loss_cycle.item():.4f}")

    out_dir = Path(args.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "G_mri2ct": G_mri2ct.state_dict(),
        "F_ct2mri": F_ct2mri.state_dict(),
    }, out_dir / "cyclegan_init.pt")
    print(f"[CycleGAN] Checkpoint framework saved to: {out_dir / 'cyclegan_init.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 3D CycleGAN Domain Translation")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lambda_cycle", type=float, default=10.0, help="Cycle consistency loss weight")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cuda or cpu)")
    args = parser.parse_args()

    train_cyclegan(args)
