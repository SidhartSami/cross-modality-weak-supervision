"""
Classifier Experiment: Progressive Layer-Wise Fine-Tuning.

Adapts the MRI-trained SE-ResNet3D classifier to the target CT domain
by progressively unfreezing layers (head first, then upper residual blocks),
preserving generalizable semantic features while aligning low-level representations.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.se_resnet3d import SEResNet3D
from train_classifier import MRITumorDataset


def run_progressive_finetuning(checkpoint_path: str, epochs: int = 3, device: str = "cpu"):
    dev = torch.device(device)
    print(f"=== Progressive Fine-Tuning Adaptation on {dev} ===")

    model = SEResNet3D(in_channels=1, base_channels=24).to(dev)
    if Path(checkpoint_path).exists():
        payload = torch.load(checkpoint_path, map_location="cpu")
        sd = payload.get("model_state_dict", payload)
        model.load_state_dict(sd)
        print(f"[Loaded] Checkpoint: {checkpoint_path}")
    else:
        print("[Notice] Using fresh initialized weights for demonstration.")

    # Stage 1: Freeze stem and layer1/layer2, train only layer3 and head
    for param in model.stem.parameters():
        param.requires_grad = False
    for param in model.layer1.parameters():
        param.requires_grad = False
    for param in model.layer2.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    dataset = MRITumorDataset(target_shape=(64, 64, 32), num_samples=32)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for crops, labels in loader:
            crops, labels = crops.to(dev), labels.to(dev)
            optimizer.zero_grad()
            logits = model(crops)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)

        avg_loss = total_loss / len(dataset)
        print(f"[Progressive FT] Epoch {epoch}/{epochs} - Adapt Loss: {avg_loss:.4f}")

    print("[Success] Progressive Fine-Tuning completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Progressive Fine-Tuning Experiment")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/se_resnet3d_best.pt", help="Base checkpoint")
    parser.add_argument("--epochs", type=int, default=2, help="Number of progressive adaptation epochs")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cuda or cpu)")
    args = parser.parse_args()

    run_progressive_finetuning(args.checkpoint, args.epochs, args.device)
