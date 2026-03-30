"""
Step 2A — PyTorch Dataset & DataLoader for Triplet Training
============================================================
Loads synced LiDAR range images and camera frames as triplets
(Anchor, Positive, Negative) for metric learning.

Includes adverse-weather augmentation and modality dropout.

Usage:
    # Imported by step6_train.py — not run directly.
    from src.step4_dataset import TripletDataset, build_dataloaders
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import pandas as pd
import yaml


# ── Augmentation pipelines ────────────────────────────────────────────────────

def get_camera_transform(train: bool, input_size: tuple[int, int] = (224, 224)):
    """Camera augmentation pipeline."""
    if train:
        return T.Compose([
            T.ToPILImage(),
            T.Resize(input_size),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
            T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.RandomErasing(p=0.3, scale=(0.02, 0.15)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return T.Compose([
            T.ToPILImage(),
            T.Resize(input_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def augment_range_image(img: np.ndarray, train: bool) -> np.ndarray:
    """
    LiDAR range image augmentation.
    - Normalize to [0, 1] using the configured max_range
    - During training: add Gaussian noise + random patch dropout + rotation
    """
    img = img.copy()

    if train:
        # Additive Gaussian noise (simulates rain scatter / sensor noise)
        noise = np.random.normal(0, 0.02, img.shape).astype(np.float32)
        img = img + noise

        # Random rectangular patch dropout (simulates partial occlusion)
        if np.random.random() < 0.3:
            h, w = img.shape
            rh, rw = np.random.randint(5, 20), np.random.randint(20, 100)
            ry, rx = np.random.randint(0, h - rh), np.random.randint(0, w - rw)
            img[ry:ry + rh, rx:rx + rw] = 0.0

        # Random horizontal flip
        if np.random.random() < 0.5:
            img = np.flip(img, axis=1).copy()

        # Random small rotation ±5° (simulates sensor mounting variation)
        if np.random.random() < 0.5:
            angle = np.random.uniform(-5.0, 5.0)
            h, w = img.shape
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    img = np.clip(img, 0.0, 1.0)
    return img


# ── Dataset class ─────────────────────────────────────────────────────────────

class TripletDataset(Dataset):
    """
    Loads triplets of (LiDAR range image, Camera image) for metric learning.

    Each __getitem__ returns:
        anchor:   (lidar_tensor, camera_tensor)
        positive: (lidar_tensor, camera_tensor)
        negative: (lidar_tensor, camera_tensor)
    """

    def __init__(
        self,
        triplets_csv: str | Path,
        range_dir: str | Path,
        camera_dir: str | Path,
        max_range: float = 100.0,
        camera_size: tuple[int, int] = (224, 224),
        train: bool = True,
    ):
        self.triplets = pd.read_csv(triplets_csv)
        self.range_dir = Path(range_dir)
        self.camera_dir = Path(camera_dir)
        self.max_range = max_range
        self.train = train
        self.cam_transform = get_camera_transform(train, camera_size)

    def __len__(self) -> int:
        return len(self.triplets)

    def _load_pair(self, frame_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load and preprocess a single (range_image, camera_image) pair."""
        # Range image: (128, 1024) float32
        ri_path = self.range_dir / f"range_{frame_idx:06d}.npy"
        ri = np.load(ri_path).astype(np.float32)
        ri = ri / self.max_range  # normalize to ~[0, 1]
        ri = augment_range_image(ri, self.train)
        ri_tensor = torch.from_numpy(ri).unsqueeze(0)  # (1, 128, 1024)

        # Camera image: (H, W, 3) uint8 → (3, 224, 224) float32
        cam_path = self.camera_dir / f"cam_{frame_idx:06d}.jpg"
        cam = cv2.imread(str(cam_path))
        cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
        cam_tensor = self.cam_transform(cam)  # (3, 224, 224)

        return ri_tensor, cam_tensor

    def __getitem__(self, idx: int):
        row = self.triplets.iloc[idx]
        anchor = self._load_pair(int(row["anchor_idx"]))
        positive = self._load_pair(int(row["positive_idx"]))
        negative = self._load_pair(int(row["negative_idx"]))
        return anchor, positive, negative


# ── DataLoader builder ────────────────────────────────────────────────────────

def build_dataloaders(cfg: dict) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, val, and test DataLoaders from config."""
    processed = Path(cfg["paths"]["processed_root"])
    range_dir = processed / "range_images"
    camera_dir = processed / "camera_synced"

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    cam_size = tuple(model_cfg["camera_input_size"])
    max_range = cfg["projection"]["max_range"]

    train_ds = TripletDataset(
        triplets_csv=processed / "triplets_train.csv",
        range_dir=range_dir,
        camera_dir=camera_dir,
        max_range=max_range,
        camera_size=cam_size,
        train=True,
    )
    val_ds = TripletDataset(
        triplets_csv=processed / "triplets_val.csv",
        range_dir=range_dir,
        camera_dir=camera_dir,
        max_range=max_range,
        camera_size=cam_size,
        train=False,
    )
    test_ds = TripletDataset(
        triplets_csv=processed / "triplets_test.csv",
        range_dir=range_dir,
        camera_dir=camera_dir,
        max_range=max_range,
        camera_size=cam_size,
        train=False,
    )

    num_workers = 8

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    print(f"Train: {len(train_loader.dataset)} triplets, {len(train_loader)} batches")
    print(f"Val:   {len(val_loader.dataset)} triplets, {len(val_loader)} batches")
    print(f"Test:  {len(test_loader.dataset)} triplets, {len(test_loader)} batches")

    # Load one batch and print shapes
    (a_li, a_cam), (p_li, p_cam), (n_li, n_cam) = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  Anchor LiDAR:  {a_li.shape}  (B, 1, 128, 1024)")
    print(f"  Anchor Camera: {a_cam.shape}  (B, 3, 224, 224)")
    print(f"  Positive LiDAR:  {p_li.shape}")
    print(f"  Negative Camera: {n_cam.shape}")
    print(f"\n  LiDAR value range: [{a_li.min():.3f}, {a_li.max():.3f}]")
    print(f"  Camera value range: [{a_cam.min():.3f}, {a_cam.max():.3f}]")
