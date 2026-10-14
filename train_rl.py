"""
Cross-Modality Weak Supervision: Autonomous 3D RL Navigation Agent Training.

Trains a Reinforcement Learning agent (PPO) to autonomously navigate 3D CT volumes
and localize liver tumors (HCC) guided by the reward landscape of a pre-trained
MRI tumor presence classifier (Zero-Shot Cross-Modality Localization).
"""

from __future__ import annotations
import argparse
import os
import glob
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv

# Import classifier architecture
from models.se_resnet3d import SEResNet3D


# -----------------------------------------------------------------------------
# 1. CT Preprocessing & Volume Utilities
# -----------------------------------------------------------------------------

def normalize_ct(volume: np.ndarray) -> np.ndarray:
    """Standard soft-tissue window clipping (-100 to 400 HU) and Z-score normalization."""
    volume = np.clip(volume, -100.0, 400.0).astype(np.float32)
    mu, sigma = volume.mean(), volume.std()
    if sigma < 1e-6:
        sigma = 1.0
    return (volume - mu) / sigma


class ClassifierRewardWrapper(nn.Module):
    """Wraps pre-trained 3D SE-ResNet to output tumor probability scores in [0, 1]."""
    def __init__(self, checkpoint_path: str | Path | None, device: torch.device):
        super().__init__()
        self.device = device
        self.model = SEResNet3D(in_channels=1, base_channels=24).to(device)

        if checkpoint_path and Path(checkpoint_path).exists():
            payload = torch.load(checkpoint_path, map_location="cpu")
            state_dict = payload.get("model_state_dict", payload)
            self.model.load_state_dict(state_dict)
            print(f"[Reward Model] Loaded trained classifier checkpoint from {checkpoint_path}")
        else:
            print("[Reward Model] Initialized default SE-ResNet3D backbone for reward guidance.")

        self.model.eval()

    def score(self, crop_np: np.ndarray) -> float:
        """Evaluates a 3D crop and returns scalar tumor probability."""
        t = torch.tensor(crop_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
        t = F.interpolate(t, size=(64, 64, 32), mode="trilinear", align_corners=False)
        t = t.to(self.device)
        with torch.no_grad():
            logit = self.model(t).squeeze()
            prob = torch.sigmoid(logit).item()
        return float(prob)


# -----------------------------------------------------------------------------
# 2. 3D Volumetric Navigation Environment (Gymnasium)
# -----------------------------------------------------------------------------

class VolumetricNavigationEnv(gym.Env):
    """
    3D Gymnasium Environment for weak-supervision tumor localization.
    The agent controls a 3D sliding window over the abdominal CT volume.
    Actions move the window along 3 spatial axes (x, y, z).
    Reward is proportional to increase in classifier tumor confidence.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        reward_wrapper: ClassifierRewardWrapper,
        ct_dirs: list[str] | None = None,
        volume_shape: tuple[int, int, int] = (128, 128, 64),
        window_size: tuple[int, int, int] = (32, 32, 16),
        stride: int = 8,
        max_steps: int = 50,
        threshold: float = 0.90
    ):
        super().__init__()
        self.reward_wrapper = reward_wrapper
        self.volume_shape = volume_shape
        self.window_size = window_size
        self.stride = stride
        self.max_steps = max_steps
        self.threshold = threshold

        # 6 Discrete movement directions: [-x, +x, -y, +y, -z, +z]
        self.action_space = spaces.Discrete(6)
        
        # Observation is the normalized 3D crop inside current window
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=window_size, dtype=np.float32
        )

        self.ct_paths = []
        if ct_dirs:
            for d in ct_dirs:
                self.ct_paths.extend(glob.glob(os.path.join(d, "**", "*.nii*"), recursive=True))

        self.volume = None
        self.current_pos = [0, 0, 0]
        self.current_step = 0
        self.last_score = 0.0

    def _get_crop(self) -> np.ndarray:
        x, y, z = self.current_pos
        wx, wy, wz = self.window_size
        return self.volume[x:x+wx, y:y+wy, z:z+wz]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # Load real CT volume if available, otherwise construct synthetic volume
        if len(self.ct_paths) > 0:
            import nibabel as nib
            chosen_path = np.random.choice(self.ct_paths)
            raw = nib.load(chosen_path).get_fdata()
            # Rescale to target shape
            t = torch.tensor(raw, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            t = F.interpolate(t, size=self.volume_shape, mode="trilinear", align_corners=False)
            self.volume = normalize_ct(t.squeeze().numpy())
        else:
            # Deterministic synthetic abdominal tissue volume
            vol = np.random.randn(*self.volume_shape).astype(np.float32) * 0.1
            # Inject focal high-confidence tumor lesion target
            tx, ty, tz = self.volume_shape[0]//2, self.volume_shape[1]//2, self.volume_shape[2]//2
            vol[tx-10:tx+10, ty-10:ty+10, tz-5:tz+5] += 1.8
            self.volume = vol

        # Initialize cursor at random coordinate allowing full window placement
        max_x = self.volume_shape[0] - self.window_size[0]
        max_y = self.volume_shape[1] - self.window_size[1]
        max_z = self.volume_shape[2] - self.window_size[2]
        self.current_pos = [
            np.random.randint(0, max(max_x, 1)),
            np.random.randint(0, max(max_y, 1)),
            np.random.randint(0, max(max_z, 1)),
        ]

        crop = self._get_crop()
        self.last_score = self.reward_wrapper.score(crop)
        return crop, {}

    def step(self, action: int):
        self.current_step += 1
        x, y, z = self.current_pos

        # Update position based on action
        if action == 0: x = max(0, x - self.stride)
        elif action == 1: x = min(self.volume_shape[0] - self.window_size[0], x + self.stride)
        elif action == 2: y = max(0, y - self.stride)
        elif action == 3: y = min(self.volume_shape[1] - self.window_size[1], y + self.stride)
        elif action == 4: z = max(0, z - self.stride)
        elif action == 5: z = min(self.volume_shape[2] - self.window_size[2], z + self.stride)
        self.current_pos = [x, y, z]

        crop = self._get_crop()
        current_score = self.reward_wrapper.score(crop)

        # Reward is the gain in tumor confidence plus bonus for surpassing optimal operating threshold
        reward = (current_score - self.last_score) * 10.0
        if current_score >= self.threshold:
            reward += 2.0  # Threshold hit reward
        self.last_score = current_score

        terminated = bool(current_score >= self.threshold)
        truncated = bool(self.current_step >= self.max_steps)

        return crop, reward, terminated, truncated, {"tumor_score": current_score}


# -----------------------------------------------------------------------------
# 3. Main RL Policy Training
# -----------------------------------------------------------------------------

def train_rl(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"=== Training Autonomous 3D Tumor Navigation Agent (PPO) on {device} ===")

    reward_wrapper = ClassifierRewardWrapper(checkpoint_path=args.classifier_checkpoint, device=device)

    def make_env():
        return VolumetricNavigationEnv(
            reward_wrapper=reward_wrapper,
            ct_dirs=[args.ct_dir] if args.ct_dir else None,
            window_size=(32, 32, 16),
            threshold=args.threshold
        )

    env = DummyVecEnv([make_env])

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.lr,
        n_steps=64,
        batch_size=16,
        n_epochs=4,
        gamma=0.99,
        verbose=1
    )

    print(f"[Training] Running PPO for {args.total_timesteps} environment timesteps...")
    model.learn(total_timesteps=args.total_timesteps)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    save_path = Path(args.checkpoint_dir) / "ppo_navigation_agent.zip"
    model.save(save_path)
    print(f"\n[Success] Trained RL agent checkpoint saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Autonomous 3D Tumor Navigation RL Agent")
    parser.add_argument("--ct_dir", type=str, default="", help="Path to clinical CT volumes (.nii.gz)")
    parser.add_argument("--classifier_checkpoint", type=str, default="./checkpoints/se_resnet3d_best.pt", help="Path to pre-trained MRI classifier")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save RL weights")
    parser.add_argument("--total_timesteps", type=int, default=500, help="Total PPO interaction timesteps")
    parser.add_argument("--threshold", type=float, default=0.90, help="Confidence threshold tau for localization")
    parser.add_argument("--lr", type=float, default=3e-4, help="PPO policy learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Execution device (cuda or cpu)")
    args = parser.parse_args()

    train_rl(args)
