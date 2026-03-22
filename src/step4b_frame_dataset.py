"""
Step 2A-v2 — Frame-Level Dataset with PK Sampling for Batch-Hard Mining
========================================================================
Instead of pre-mined triplets, this dataset loads individual frames
grouped by "place" (GPS clusters).  A custom PK batch sampler yields
batches of P places × K frames per place, enabling online hard-negative
mining inside the training loop.

Usage:
    from src.step4b_frame_dataset import build_hard_dataloaders
"""

from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
import torchvision.transforms as T
import yaml

from src.step4_dataset import get_camera_transform, augment_range_image


# ── Place clustering ──────────────────────────────────────────────────────────

def cluster_places(
    poses_xy: np.ndarray,
    timestamps: np.ndarray,
    d_pos: float = 5.0,
    d_time: float = 60.0,
) -> dict[int, list[int]]:
    """
    Group frames into discrete "places" using spatial grid quantization.

    Each cell in a d_pos × d_pos grid defines one place. We then filter
    to only keep places that have at least two frames with dt > d_time
    (i.e., genuine revisits).

    Returns:
        place_to_frames: dict mapping place_id -> list of frame indices
    """
    # Quantize GPS coordinates into grid cells
    cell_x = np.floor(poses_xy[:, 0] / d_pos).astype(int)
    cell_y = np.floor(poses_xy[:, 1] / d_pos).astype(int)

    # Group frames by grid cell
    cell_to_frames: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx in range(len(poses_xy)):
        cell_to_frames[(cell_x[idx], cell_y[idx])].append(idx)

    # Filter: keep only cells with at least 2 frames that are >d_time apart
    place_to_frames: dict[int, list[int]] = {}
    place_id = 0
    for cell_key, frames in cell_to_frames.items():
        if len(frames) < 2:
            continue
        ts = timestamps[frames]
        # Check for at least one revisit pair
        has_revisit = (ts.max() - ts.min()) > d_time
        if has_revisit:
            place_to_frames[place_id] = frames
            place_id += 1

    return place_to_frames


# ── PK Batch Sampler ──────────────────────────────────────────────────────────

class PKSampler(Sampler):
    """
    Yields batches of P places × K frames each.

    Each batch contains exactly P*K frame indices.  For places with more
    than K frames we randomly subsample; for places with fewer than K we
    oversample with replacement.
    """

    def __init__(
        self,
        place_to_frames: dict[int, list[int]],
        P: int,
        K: int,
        epochs_worth: int = 1,
        seed: int = 42,
    ):
        self.place_to_frames = place_to_frames
        self.place_ids = list(place_to_frames.keys())
        self.P = P
        self.K = K
        self.rng = np.random.RandomState(seed)
        # Number of batches per "epoch"
        self.n_batches = max(len(self.place_ids) // P, 1) * epochs_worth

    def __iter__(self):
        # Re-seed each epoch so batches vary across epochs
        self.rng = np.random.RandomState(self.rng.randint(2**31))
        for _ in range(self.n_batches):
            # Sample P places without replacement (if enough places)
            if len(self.place_ids) >= self.P:
                chosen_places = self.rng.choice(
                    self.place_ids, size=self.P, replace=False
                )
            else:
                chosen_places = self.rng.choice(
                    self.place_ids, size=self.P, replace=True
                )

            batch = []
            for pid in chosen_places:
                frames = self.place_to_frames[pid]
                if len(frames) >= self.K:
                    selected = self.rng.choice(frames, size=self.K, replace=False)
                else:
                    selected = self.rng.choice(frames, size=self.K, replace=True)
                batch.extend(selected.tolist())
            yield batch

    def __len__(self):
        return self.n_batches


# ── Frame-level Dataset ──────────────────────────────────────────────────────

class PlaceFrameDataset(Dataset):
    """
    Loads individual (LiDAR, Camera) frame pairs by frame index.
    Each __getitem__ returns (lidar_tensor, camera_tensor, place_label).
    """

    def __init__(
        self,
        frame_to_place: dict[int, int],
        range_dir: str | Path,
        camera_dir: str | Path,
        max_range: float = 100.0,
        camera_size: tuple[int, int] = (224, 224),
        train: bool = True,
    ):
        self.frame_indices = sorted(frame_to_place.keys())
        self.frame_to_place = frame_to_place
        self.idx_to_frame = {i: f for i, f in enumerate(self.frame_indices)}
        self.frame_to_idx = {f: i for i, f in enumerate(self.frame_indices)}
        self.range_dir = Path(range_dir)
        self.camera_dir = Path(camera_dir)
        self.max_range = max_range
        self.train = train
        self.cam_transform = get_camera_transform(train, camera_size)

    def __len__(self) -> int:
        return len(self.frame_indices)

    def __getitem__(self, idx: int):
        # idx here is the actual frame index (from PKSampler), not sequential
        frame_idx = idx
        place_label = self.frame_to_place.get(frame_idx, -1)

        # Range image
        ri_path = self.range_dir / f"range_{frame_idx:06d}.npy"
        ri = np.load(ri_path).astype(np.float32)
        ri = ri / self.max_range
        ri = augment_range_image(ri, self.train)
        ri_tensor = torch.from_numpy(ri).unsqueeze(0)  # (1, 128, 1024)

        # Camera image
        cam_path = self.camera_dir / f"cam_{frame_idx:06d}.jpg"
        cam = cv2.imread(str(cam_path))
        cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
        cam_tensor = self.cam_transform(cam)  # (3, 224, 224)

        return ri_tensor, cam_tensor, place_label


# ── DataLoader builder ────────────────────────────────────────────────────────

def build_hard_dataloaders(cfg: dict):
    """
    Build a PK-sampled train DataLoader and standard val/test DataLoaders.

    Returns: (train_loader, val_loader, test_loader, place_to_frames)
    """
    processed = Path(cfg["paths"]["processed_root"])
    range_dir = processed / "range_images"
    camera_dir = processed / "camera_synced"

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    mining_cfg = cfg["mining"]
    split_cfg = cfg["split"]
    cam_size = tuple(model_cfg["camera_input_size"])
    max_range = cfg["projection"]["max_range"]

    # Load poses
    poses = np.load(processed / "frame_poses.npy")  # (N, 4): x, y, z, timestamp
    N = len(poses)
    poses_xy = poses[:, :2]
    timestamps = poses[:, 3]

    # Determine train frame range
    train_end = int(N * split_cfg["train_ratio"])

    # Cluster places using only training frames
    train_poses_xy = poses_xy[:train_end]
    train_timestamps = timestamps[:train_end]
    d_pos = mining_cfg["d_pos_m"]
    d_time = mining_cfg["d_time_s"]

    print("Clustering training frames into places...")
    place_to_frames = cluster_places(train_poses_xy, train_timestamps, d_pos, d_time)
    print(f"  Found {len(place_to_frames)} places with revisits")

    total_frames = sum(len(f) for f in place_to_frames.values())
    print(f"  Total frames in places: {total_frames}")

    # Build frame_to_place mapping
    frame_to_place: dict[int, int] = {}
    for place_id, frames in place_to_frames.items():
        for f in frames:
            frame_to_place[f] = place_id

    # Create dataset
    train_ds = PlaceFrameDataset(
        frame_to_place=frame_to_place,
        range_dir=range_dir,
        camera_dir=camera_dir,
        max_range=max_range,
        camera_size=cam_size,
        train=True,
    )

    # PK sampler
    P = train_cfg.get("P", 8)
    K = train_cfg.get("K", 4)
    pk_sampler = PKSampler(place_to_frames, P=P, K=K)

    train_loader = DataLoader(
        train_ds,
        batch_sampler=pk_sampler,
        num_workers=4,
        pin_memory=True,
    )

    # Val / Test loaders use the original triplet-based approach
    from src.step4_dataset import build_dataloaders as build_triplet_dataloaders
    _, val_loader, test_loader = build_triplet_dataloaders(cfg)

    return train_loader, val_loader, test_loader


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    train_loader, val_loader, test_loader = build_hard_dataloaders(cfg)
    print(f"\nTrain PK loader: {len(train_loader)} batches")
    print(f"Val:   {len(val_loader.dataset)} triplets, {len(val_loader)} batches")

    # Load one batch and print shapes
    batch = next(iter(train_loader))
    lidar, camera, labels = batch
    print(f"\nBatch shapes:")
    print(f"  LiDAR:  {lidar.shape}")
    print(f"  Camera: {camera.shape}")
    print(f"  Labels: {labels.shape}  unique places: {labels.unique().tolist()}")
