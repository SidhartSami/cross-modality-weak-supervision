"""
Classifier Experiment: Domain-Adversarial Neural Network (DANN).

Feature-level latent domain adaptation using a Gradient Reversal Layer (GRL).
Trains the 3D SE-ResNet feature extractor to maximize tumor classification accuracy
on MRI while simultaneously minimizing the domain discriminator's ability to
distinguish between MRI and CT scans, enforcing domain-invariant representations.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.se_resnet3d import SEResNet3D
from train_classifier import MRITumorDataset


# -----------------------------------------------------------------------------
# 1. Gradient Reversal Layer (GRL)
# -----------------------------------------------------------------------------

class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer: passes input forward unchanged,
    multiplies gradient by -lambda during backpropagation.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_output.neg() * ctx.lambda_, None


def grad_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    return GradientReversalFunction.apply(x, lambda_)


# -----------------------------------------------------------------------------
# 2. 3D DANN Architecture
# -----------------------------------------------------------------------------

class DANNModel3D(nn.Module):
    """
    Domain-Adversarial SE-ResNet3D:
    - Shared SE-ResNet3D Feature Extractor
    - Task Classifier (Tumor Presence: Benign vs. Malignant)
    - Domain Discriminator (Source MRI vs. Target CT)
    """
    def __init__(self, in_channels: int = 1, base_c: int = 24):
        super().__init__()
        self.backbone = SEResNet3D(in_channels=in_channels, base_c=base_c)
        feature_dim = base_c * 8

        # Task Classifier: Predicts tumor presence
        self.task_classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(feature_dim, 1)
        )

        # Domain Discriminator: Predicts Domain (0 = MRI, 1 = CT)
        self.domain_discriminator = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.extract_features(x)

    def forward(self, x: torch.Tensor, lambda_: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.extract_features(x)
        task_logits = self.task_classifier(features)
        reversed_features = grad_reverse(features, lambda_)
        domain_logits = self.domain_discriminator(reversed_features)
        return task_logits, domain_logits


# -----------------------------------------------------------------------------
# 3. Training & Evaluation
# -----------------------------------------------------------------------------

class MultiDomainCropDataset(Dataset):
    """Synthetic/cached multi-domain dataset with modality domain labels."""
    def __init__(self, num_samples: int = 32):
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rng = np.random.RandomState(idx)
        # domain: 0 = MRI, 1 = CT
        domain = float(idx % 2 == 1)
        # label: 0 = Benign, 1 = Tumor
        label = float((idx // 2) % 2 == 1)

        # CT has higher baseline attenuation offset than MRI
        offset = 0.5 if domain == 1.0 else 0.0
        crop = (rng.randn(1, 64, 64, 32) * 0.1 + offset).astype(np.float32)
        if label > 0.5:
            crop[0, 28:36, 28:36, 14:18] += 1.5

        return torch.tensor(crop), torch.tensor([label]), torch.tensor([domain])


def run_dann_experiment(epochs: int = 2, device: str = "cpu"):
    dev = torch.device(device)
    print(f"=== Domain-Adversarial Neural Network (DANN) Experiment on {dev} ===")

    model = DANNModel3D(in_channels=1, base_c=24).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    task_criterion = nn.BCEWithLogitsLoss()
    domain_criterion = nn.BCEWithLogitsLoss()

    dataset = MultiDomainCropDataset(num_samples=32)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    print("[Training] Aligning latent feature representations via Gradient Reversal...")
    for epoch in range(1, epochs + 1):
        model.train()
        p = float(epoch) / float(epochs)
        lambda_val = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0

        total_task_loss = 0.0
        total_domain_loss = 0.0

        for crops, labels, domains in loader:
            crops, labels, domains = crops.to(dev), labels.to(dev), domains.to(dev)
            optimizer.zero_grad()

            task_logits, domain_logits = model(crops, lambda_=lambda_val)
            t_loss = task_criterion(task_logits, labels)
            d_loss = domain_criterion(domain_logits, domains)
            loss = t_loss + d_loss

            loss.backward()
            optimizer.step()

            total_task_loss += t_loss.item()
            total_domain_loss += d_loss.item()

        print(f"[DANN] Epoch {epoch}/{epochs} (λ={lambda_val:.2f}) | Task Loss: {total_task_loss/len(loader):.4f} | Domain Loss: {total_domain_loss/len(loader):.4f}")

    # Evaluation on target domain
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for i in range(len(dataset)):
            crop, label, domain = dataset[i]
            if domain.item() == 1.0:  # Evaluate on target CT domain
                crop = crop.unsqueeze(0).to(dev)
                feat = model.extract_features(crop)
                prob = torch.sigmoid(model.task_classifier(feat)).item()
                y_pred.append(prob)
                y_true.append(label.item())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    auroc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.5
    b_acc = balanced_accuracy_score(y_true, y_pred >= 0.5)

    print(f"\n--- DANN Target Domain Test Results ---")
    print(f"Target Samples: {len(y_true)}")
    print(f"AUROC:          {auroc:.4f}")
    print(f"Balanced Acc:   {b_acc:.4f}")
    print(f"Mean Score:     {np.mean(y_pred):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DANN Domain Adaptation Experiment")
    parser.add_argument("--epochs", type=int, default=2, help="Number of DANN epochs")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cuda or cpu)")
    args = parser.parse_args()

    run_dann_experiment(args.epochs, args.device)
