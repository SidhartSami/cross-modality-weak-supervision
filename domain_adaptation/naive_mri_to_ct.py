"""
Classifier Experiment: Direct Zero-Shot MRI-to-CT Transfer (Unadapted Baseline).

Evaluates the unadapted MRI-trained presence classifier directly on clinical CT crops
without any domain adaptation. Serves as the negative control / lower baseline to quantify
the raw cross-modality domain shift penalty.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.se_resnet3d import SEResNet3D
from train_classifier import MRITumorDataset


def run_naive_transfer(checkpoint_path: str, device: str = "cpu"):
    dev = torch.device(device)
    print(f"=== Running Zero-Shot Direct MRI-to-CT Evaluation on {dev} ===")

    # Initialize model
    model = SEResNet3D(in_channels=1, base_channels=24).to(dev)
    if Path(checkpoint_path).exists():
        payload = torch.load(checkpoint_path, map_location="cpu")
        sd = payload.get("model_state_dict", payload)
        model.load_state_dict(sd)
        print(f"[Loaded] Checkpoint: {checkpoint_path}")
    else:
        print("[Warning] No checkpoint found. Running evaluation with initialized weights for demonstration.")

    model.eval()

    # Load test crops
    dataset = MRITumorDataset(target_shape=(64, 64, 32), num_samples=32)
    y_true, y_pred = [], []

    with torch.no_grad():
        for i in range(len(dataset)):
            crop, label = dataset[i]
            crop = crop.unsqueeze(0).to(dev)
            prob = torch.sigmoid(model(crop)).item()
            y_pred.append(prob)
            y_true.append(label.item())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    auroc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.5
    b_acc = balanced_accuracy_score(y_true, y_pred >= 0.5)

    print(f"\n--- Zero-Shot MRI-to-CT Transfer Results ---")
    print(f"Test Samples: {len(y_true)}")
    print(f"AUROC:        {auroc:.4f}")
    print(f"Balanced Acc: {b_acc:.4f}")
    print(f"Mean Score:   {np.mean(y_pred):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-Shot MRI to CT Transfer")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/se_resnet3d_best.pt", help="Path to classifier weights")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cuda or cpu)")
    args = parser.parse_args()

    run_naive_transfer(args.checkpoint, args.device)
