"""
Cross-Modality Weak Supervision: MRI Tumor Presence Classifier Training.

Trains a 3D Squeeze-and-Excitation ResNet (SE-ResNet3D) using binary image-level
tumor presence labels derived from MRI scans and pathology reports.
Once trained, this classifier provides the reward landscape for the zero-shot
RL tumor localization agent on unseen CT volumes.
"""

from __future__ import annotations
import argparse
import os
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from models.se_resnet3d import SEBlock3D, SEResNetBlock3D, SEResNet3D


# -----------------------------------------------------------------------------
# 2. Dataset & Data Loader
# -----------------------------------------------------------------------------

class MRITumorDataset(Dataset):
    """
    Dataset for 3D MRI crops with weakly supervised binary tumor presence labels.
    If the specified data path is empty or unavailable, synthetic volumetric crops
    are dynamically generated to enable end-to-end testing and review out-of-the-box.
    """
    def __init__(self, data_dir: str | Path | None = None, target_shape=(64, 64, 32), num_samples: int = 64):
        self.target_shape = target_shape
        self.data_files = []
        self.synthetic = False

        if data_dir and Path(data_dir).exists():
            for ext in ("*.pt", "*.npz", "*.npy"):
                self.data_files.extend(list(Path(data_dir).glob(ext)))

        if len(self.data_files) == 0:
            print(f"[Dataset] No pre-extracted crops found in '{data_dir}'. Enabling synthetic review mode ({num_samples} samples).")
            self.synthetic = True
            self.num_samples = num_samples
        else:
            print(f"[Dataset] Loaded {len(self.data_files)} real volumetric MRI crops from {data_dir}.")

    def __len__(self) -> int:
        return self.num_samples if self.synthetic else len(self.data_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.synthetic:
            # Deterministic pseudo-random generation for testing
            rng = np.random.RandomState(idx)
            label = float(idx % 2 == 1)
            crop = rng.randn(1, *self.target_shape).astype(np.float32) * 0.1
            if label > 0.5:
                # Add central high-intensity hyperdense lesion signature
                c = [s // 2 for s in self.target_shape]
                crop[0, c[0]-4:c[0]+4, c[1]-4:c[1]+4, c[2]-2:c[2]+2] += 1.5
            return torch.tensor(crop, dtype=torch.float32), torch.tensor([label], dtype=torch.float32)

        file_path = self.data_files[idx]
        if file_path.suffix == ".pt":
            payload = torch.load(file_path, map_location="cpu")
            crop = payload["crop"]
            label = float(payload["label"])
        else:
            arr = np.load(file_path)
            crop = torch.from_numpy(arr["crop"]).float()
            label = float(arr["label"])

        if crop.ndim == 3:
            crop = crop.unsqueeze(0)
        return crop, torch.tensor([label], dtype=torch.float32)


# -----------------------------------------------------------------------------
# 3. Main Training Execution
# -----------------------------------------------------------------------------

def train_classifier(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"=== Training 3D SE-ResNet Tumor Classifier on {device} ===")

    dataset = MRITumorDataset(data_dir=args.data_dir, target_shape=(64, 64, 32))
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    model = SEResNet3D(in_channels=1, base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_loss = float("inf")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    save_path = Path(args.checkpoint_dir) / "se_resnet3d_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for crops, labels in train_loader:
            crops = crops.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(crops)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += len(labels)

        avg_loss = total_loss / max(total, 1)
        accuracy = (correct / max(total, 1)) * 100.0
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] - Loss: {avg_loss:.4f} | Accuracy: {accuracy:.1f}%")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "loss": best_loss,
                "accuracy": accuracy
            }, save_path)

    print(f"\n[Success] Best classifier weights saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 3D SE-ResNet MRI Tumor Presence Classifier")
    parser.add_argument("--data_dir", type=str, default="./data/mri_crops", help="Path to preprocessed 3D MRI crops")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save model weights")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--base_channels", type=int, default=24, help="Base channel width for SE-ResNet3D")
    parser.add_argument("--device", type=str, default="cuda", help="Execution device (cuda or cpu)")
    args = parser.parse_args()

    train_classifier(args)
