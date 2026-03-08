"""
Step 1A — Time Synchronization & Range Image Projection
========================================================
Pairs each LiDAR scan with its nearest camera frame (by timestamp),
projects unstructured 3D point clouds into dense 128×1024 range images,
and writes the synced dataset to disk.

Usage:
    python -m src.step1_sync_and_project                     # uses defaults from config
    python -m src.step1_sync_and_project --config configs/config.yaml
"""

import argparse
import os
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm


# ── Timestamp parsing ─────────────────────────────────────────────────────────

LIDAR_RE = re.compile(r"cloud_(\d+)_(\d+)_(\d+)\.bin$")
CAMERA_RE = re.compile(r"frame_(\d+)_(\d+)_(\d+)\.jpg$")


def parse_timestamp(filename: str, pattern: re.Pattern) -> tuple[int, float] | None:
    """Return (sequence_index, timestamp_seconds) from a filename, or None."""
    m = pattern.search(filename)
    if m is None:
        return None
    seq = int(m.group(1))
    ts = int(m.group(2)) + int(m.group(3)) / 1e9
    return seq, ts


def build_file_index(directory: Path, pattern: re.Pattern) -> pd.DataFrame:
    """Scan a directory and return a DataFrame of (seq, timestamp, filename)."""
    records = []
    for fname in sorted(os.listdir(directory)):
        parsed = parse_timestamp(fname, pattern)
        if parsed is not None:
            seq, ts = parsed
            records.append({"seq": seq, "timestamp": ts, "filename": fname})
    df = pd.DataFrame(records)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ── Nearest-neighbor time sync ────────────────────────────────────────────────

def sync_lidar_to_camera(
    lidar_df: pd.DataFrame,
    camera_df: pd.DataFrame,
    max_error_ms: float,
) -> pd.DataFrame:
    """For each LiDAR scan, find the closest camera frame by timestamp."""
    cam_ts = camera_df["timestamp"].values
    pairs = []

    for _, row in tqdm(lidar_df.iterrows(), total=len(lidar_df), desc="Syncing"):
        lidar_ts = row["timestamp"]
        idx = np.searchsorted(cam_ts, lidar_ts)

        # Compare the two nearest candidates (idx-1 and idx)
        candidates = []
        if idx > 0:
            candidates.append(idx - 1)
        if idx < len(cam_ts):
            candidates.append(idx)

        best_idx = min(candidates, key=lambda i: abs(cam_ts[i] - lidar_ts))
        error_ms = abs(cam_ts[best_idx] - lidar_ts) * 1000.0

        if error_ms <= max_error_ms:
            pairs.append({
                "frame_idx": len(pairs),
                "lidar_file": row["filename"],
                "camera_file": camera_df.iloc[best_idx]["filename"],
                "lidar_ts": row["timestamp"],
                "camera_ts": cam_ts[best_idx],
                "sync_error_ms": round(error_ms, 3),
            })

    return pd.DataFrame(pairs)


# ── Spherical range image projection ──────────────────────────────────────────

def load_pointcloud(filepath: Path, fields: int = 5) -> np.ndarray:
    """Load a binary point cloud as (N, fields) float32 array."""
    data = np.fromfile(str(filepath), dtype=np.float32)
    return data.reshape(-1, fields)


def project_to_range_image(
    points: np.ndarray,
    num_beams: int = 128,
    num_cols: int = 1024,
    max_range: float = 100.0,
) -> np.ndarray:
    """
    Project an unstructured (N, 5) point cloud into a (num_beams, num_cols)
    range image using the ring index and azimuth angle.

    Fields: x, y, z, intensity, ring
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    rings = points[:, 4].astype(np.int32)

    ranges = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    azimuth = np.arctan2(y, x)  # [-pi, pi]

    # Quantize azimuth into column indices: -pi → col 0, +pi → col 1023
    col = np.floor((azimuth / np.pi + 1.0) * 0.5 * num_cols).astype(np.int32)
    col = np.clip(col, 0, num_cols - 1)

    row = np.clip(rings, 0, num_beams - 1)

    # Build the image — when multiple points map to the same pixel, keep closest
    img = np.zeros((num_beams, num_cols), dtype=np.float32)
    for i in range(len(ranges)):
        r, c, rng = row[i], col[i], ranges[i]
        if rng > max_range:
            continue
        if img[r, c] == 0.0 or rng < img[r, c]:
            img[r, c] = rng

    return img


def project_to_range_image_vectorized(
    points: np.ndarray,
    num_beams: int = 128,
    num_cols: int = 1024,
    max_range: float = 100.0,
) -> np.ndarray:
    """
    Vectorized range image projection.  Much faster than the loop version
    for large point clouds (~145K points).
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    rings = points[:, 4].astype(np.int32)
    ranges = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    azimuth = np.arctan2(y, x)

    col = np.floor((azimuth / np.pi + 1.0) * 0.5 * num_cols).astype(np.int32)
    col = np.clip(col, 0, num_cols - 1)
    row = np.clip(rings, 0, num_beams - 1)

    valid = ranges <= max_range
    row, col, ranges = row[valid], col[valid], ranges[valid]

    # Flatten 2D index to 1D for efficient min-scatter
    flat_idx = row * num_cols + col
    img_flat = np.full(num_beams * num_cols, fill_value=np.inf, dtype=np.float32)

    # np.minimum.at is the vectorized way to keep the closest range per pixel
    np.minimum.at(img_flat, flat_idx, ranges)

    img_flat[img_flat == np.inf] = 0.0
    return img_flat.reshape(num_beams, num_cols)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(cfg: dict) -> None:
    paths = cfg["paths"]
    dataset_root = Path(paths["dataset_root"])
    processed_root = Path(paths["processed_root"])

    lidar_dir = dataset_root / paths["lidar_subdir"]
    camera_dir = dataset_root / paths["camera_subdir"]

    range_out = processed_root / "range_images"
    cam_out = processed_root / "camera_synced"
    range_out.mkdir(parents=True, exist_ok=True)
    cam_out.mkdir(parents=True, exist_ok=True)

    proj_cfg = cfg["projection"]
    num_beams = proj_cfg["num_beams"]
    num_cols = proj_cfg["num_columns"]
    max_range = proj_cfg["max_range"]
    fields = proj_cfg["fields_per_point"]

    # ── 1. Index files ────────────────────────────────────────────────────
    print("Indexing LiDAR scans...")
    lidar_df = build_file_index(lidar_dir, LIDAR_RE)
    print(f"  Found {len(lidar_df)} LiDAR scans")

    print("Indexing camera frames...")
    camera_df = build_file_index(camera_dir, CAMERA_RE)
    print(f"  Found {len(camera_df)} camera frames")

    # ── 2. Time-sync ─────────────────────────────────────────────────────
    max_error = cfg["sync"]["max_sync_error_ms"]
    print(f"\nTime-syncing (max error = {max_error} ms)...")
    sync_df = sync_lidar_to_camera(lidar_df, camera_df, max_error)
    print(f"  Synced pairs: {len(sync_df)}")
    print(f"  Mean sync error: {sync_df['sync_error_ms'].mean():.2f} ms")
    print(f"  Max sync error:  {sync_df['sync_error_ms'].max():.2f} ms")

    sync_csv = processed_root / "sync_pairs.csv"
    sync_df.to_csv(sync_csv, index=False)
    print(f"  Saved → {sync_csv}")

    # ── 3. Project range images & copy camera frames ─────────────────────
    print(f"\nProjecting {len(sync_df)} range images ({num_beams}×{num_cols})...")
    for _, row in tqdm(sync_df.iterrows(), total=len(sync_df), desc="Projecting"):
        idx = row["frame_idx"]

        # Range image
        pc = load_pointcloud(lidar_dir / row["lidar_file"], fields=fields)
        ri = project_to_range_image_vectorized(pc, num_beams, num_cols, max_range)
        np.save(range_out / f"range_{idx:06d}.npy", ri)

        # Symlink (or copy) the synced camera frame
        dst = cam_out / f"cam_{idx:06d}.jpg"
        if not dst.exists():
            src = camera_dir / row["camera_file"]
            shutil.copy2(str(src), str(dst))

    # ── 4. Summary ───────────────────────────────────────────────────────
    print("\n✓ Step 1A complete.")
    print(f"  Range images : {range_out}  ({len(list(range_out.glob('*.npy')))} files)")
    print(f"  Synced camera: {cam_out}  ({len(list(cam_out.glob('*.jpg')))} files)")
    print(f"  Manifest     : {sync_csv}")


# ── Sanity-check visualization ────────────────────────────────────────────────

def visualize_samples(processed_root: str, num_samples: int = 5) -> None:
    """Quick visual check: show a few range images as heatmaps."""
    import matplotlib.pyplot as plt

    ri_dir = Path(processed_root) / "range_images"
    cam_dir = Path(processed_root) / "camera_synced"
    files = sorted(ri_dir.glob("*.npy"))

    step = max(1, len(files) // num_samples)
    samples = files[::step][:num_samples]

    fig, axes = plt.subplots(num_samples, 2, figsize=(18, 3 * num_samples))
    if num_samples == 1:
        axes = axes[np.newaxis, :]

    for i, ri_path in enumerate(samples):
        idx_str = ri_path.stem.split("_")[1]

        ri = np.load(ri_path)
        axes[i, 0].imshow(ri, cmap="turbo", aspect="auto")
        axes[i, 0].set_title(f"Range Image — {ri_path.name}")
        axes[i, 0].set_ylabel(f"Beam (0–127)")

        cam_path = cam_dir / f"cam_{idx_str}.jpg"
        if cam_path.exists():
            cam = cv2.cvtColor(cv2.imread(str(cam_path)), cv2.COLOR_BGR2RGB)
            axes[i, 1].imshow(cam)
            axes[i, 1].set_title(f"Camera — cam_{idx_str}.jpg")
        axes[i, 1].axis("off")

    plt.tight_layout()
    out_path = Path(processed_root) / "sample_pairs.png"
    plt.savefig(out_path, dpi=120)
    print(f"Saved visualization → {out_path}")
    plt.close()


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Step 1A: Sync + Range Image Projection")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--visualize", action="store_true", help="Generate sample visualizations after processing")
    parser.add_argument("--vis-only", action="store_true", help="Skip processing, only visualize existing outputs")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if not args.vis_only:
        run(cfg)

    if args.visualize or args.vis_only:
        visualize_samples(cfg["paths"]["processed_root"])


if __name__ == "__main__":
    main()
