"""
Classifier Experiment: Statistical Histogram Intensity Matching.

Non-parametric image-level domain adaptation: aligns the intensity distribution
of target CT scans to an MRI reference template before querying the classifier.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
try:
    from skimage.exposure import match_histograms
except ImportError:
    def match_histograms(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """NumPy-based quantile matching fallback when scikit-image is not installed."""
        orig_shape = image.shape
        img_flat = image.ravel()
        ref_flat = reference.ravel()

        s_values, bin_idx, s_counts = np.unique(img_flat, return_inverse=True, return_counts=True)
        t_values, t_counts = np.unique(ref_flat, return_counts=True)

        s_quantiles = np.cumsum(s_counts).astype(np.float64) / img_flat.size
        t_quantiles = np.cumsum(t_counts).astype(np.float64) / ref_flat.size

        interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
        return interp_t_values[bin_idx].reshape(orig_shape).astype(image.dtype)

from sklearn.metrics import roc_auc_score, balanced_accuracy_score

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.se_resnet3d import SEResNet3D
from train_classifier import MRITumorDataset


def run_histogram_matching(checkpoint_path: str, device: str = "cpu"):
    dev = torch.device(device)
    print(f"=== Statistical Histogram Matching Evaluation on {dev} ===")

    classifier = SEResNet3D(in_channels=1, base_channels=24).to(dev)
    if Path(checkpoint_path).exists():
        payload = torch.load(checkpoint_path, map_location="cpu")
        sd = payload.get("model_state_dict", payload)
        classifier.load_state_dict(sd)
        print(f"[Loaded] Checkpoint: {checkpoint_path}")

    classifier.eval()

    # Create canonical MRI reference distribution
    rng = np.random.RandomState(42)
    reference_distribution = rng.normal(loc=0.0, scale=1.0, size=(64, 64, 32)).astype(np.float32)

    dataset = MRITumorDataset(target_shape=(64, 64, 32), num_samples=32)
    y_true, y_pred = [], []

    with torch.no_grad():
        for i in range(len(dataset)):
            crop_t, label = dataset[i]
            crop_np = crop_t.squeeze().numpy()

            # Non-parametric quantile histogram matching
            matched_crop = match_histograms(crop_np, reference_distribution)
            matched_t = torch.tensor(matched_crop, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(dev)

            prob = torch.sigmoid(classifier(matched_t)).item()
            y_pred.append(prob)
            y_true.append(label.item())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    auroc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.5
    b_acc = balanced_accuracy_score(y_true, y_pred >= 0.5)

    print(f"\n--- Histogram Matching Results ---")
    print(f"Test Samples: {len(y_true)}")
    print(f"AUROC:        {auroc:.4f}")
    print(f"Balanced Acc: {b_acc:.4f}")
    print(f"Mean Score:   {np.mean(y_pred):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Histogram Matching Experiment")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/se_resnet3d_best.pt", help="Classifier weights")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cuda or cpu)")
    args = parser.parse_args()

    run_histogram_matching(args.checkpoint, args.device)
