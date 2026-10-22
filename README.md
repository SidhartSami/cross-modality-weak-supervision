# Reinforcing intra-operative tumour localisation with weak supervision from pre-operative appearances

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/license-Research-lightgrey)

## Abstract
Limited tumour visibility on intra-operative images complicates surgical navigation, including liver hepatocellular carcinoma (HCC) targeting on computed tomography (CT) imaging. This makes labelled datasets difficult to curate for deep learning, and transferring pre-operative delineations through registration remains challenging due to intra-operative deformation and lack of paired pre- and intra-operative data. We propose to transfer tumour appearances learned from pre-operative magnetic resonance (MR) images to guide localisation on intra-operative CT through weak supervision. A classifier is trained to classify tumour presence on pre-operative images. This is applied to windows on intra-operative images, and despite the domain gap, it assigns higher tumour-presence probabilities on average to tumour-containing windows. These weak cross-domain signals guide a reinforcement learning agent to localise tumours on intra-operative images. We compare direct transfer, few-shot adaptation and synthetic transfer (trained using unpaired MR and CT), which translates CT into synthetic MR before localisation. On an external HCC cohort, synthetic transfer outperformed other tested methods and CT-only baselines. Our framework enables cross-modality tumour localisation without paired MR-CT training images.

## Hyperparameters
### Object Presence Classifier Parameters (SE-ResNet3D)

| **Parameters**            | **Details**                           |
|---------------------------|---------------------------------------|
| Architecture              | 3D Squeeze-and-Excitation ResNet      |
| Base channels             | 24                                    |
| Convolution stages        | 4 (Stem + 3 Residual SE Stages)       |
| Input crop size           | 64 × 64 × 32 voxels                   |
| Batch size                | 8                                     |
| Epochs                    | 32                                    |
| Optimizer                 | AdamW (weight decay 1e-4)             |
| Criterion                 | Binary Cross Entropy with Logits      |
| Learning rate             | 0.001                                 |

### Reinforcement Learning Parameters (PPO)

| **Parameters**            | **Details**                           |
|---------------------------|---------------------------------------|
| Full volume shape         | 256 × 256 × 180 (or 128 × 128 × 64)   |
| Observation window size   | 32 × 32 × 16 voxels                   |
| Action space              | 6 discrete directional shifts         |
| Navigation stride         | 8 voxels                              |
| Agent algorithm           | Proximal Policy Optimization (PPO)    |
| Discount factor (γ)       | 0.99                                  |
| Policy learning rate      | 0.0003                                |
| Operating threshold ($\tau^*$)| 0.90 – 0.92                       |
| Max steps per episode     | 50 steps                              |

## Repository Overview

This repository provides a clean, minimal, and fully runnable implementation of our cross-modality weakly supervised framework. The repository explicitly distinguishes between the **primary proposed method** and the **comparative domain adaptation baselines/ablations** benchmarked in the paper.

### Proposed Method
- `models/se_resnet3d.py`: Standalone architecture of the 3D Squeeze-and-Excitation ResNet presence classifier.
- `train_classifier.py`: Trains the primary 3D SE-ResNet tumor presence classifier on MRI scans using weak binary labels.
- `train_rl.py`: Trains the RL agent (PPO) to navigate 3D CT volumes using the MRI classifier.

### Synthetic Transfer Pipeline
- `cyclegan/train_cyclegan.py`: The underlying 3D/2.5D CycleGAN unpaired image translation training pipeline used for the neural domain bridge experiments.

## Setting Up a Virtual Environment

1. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   ```

2. **Activate the virtual environment**:
   - On macOS and Linux:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```powershell
     venv\Scripts\activate
     ```

## Installing the Project

Install the project in editable mode:
```bash
pip install -e .
```

Install dependencies listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

## Running Experiments

### Primary Method: Train Classifier & Zero-Shot RL Agent
Run the main 3D SE-ResNet classifier training on MRI:
```bash
python train_classifier.py --data_dir ./data/mri_crops --epochs 32 --device cuda
```

Deploy the autonomous RL agent on CT volumes using zero-shot reward guidance:
```bash
python train_rl.py --classifier_checkpoint ./checkpoints/se_resnet3d_best.pt --threshold 0.90 --device cuda
```

### CycleGAN Training
```bash
python cyclegan/train_cyclegan.py --epochs 100 --device cuda
```

## Reproducibility Notes

- **Reviewer Demo Mode**: If no local medical datasets are provided, all scripts automatically generate synthetic volumetric tensors, enabling reviewers to verify end-to-end execution immediately out-of-the-box.
- **Data Paths**: Update `--data_dir` in `train_classifier.py` and `--ct_dir` in `train_rl.py` to point to your local NIfTI cohorts.
- **Hardware Acceleration**: Pass `--device cuda` or `--device cpu`. Specific GPUs can be selected via `CUDA_VISIBLE_DEVICES`.

## Citation

If you use this codebase or framework in academic work, please cite:

```bibtex
to-be-added
```
