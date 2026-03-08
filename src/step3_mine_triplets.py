"""
Step 1C — Triplet Mining
========================
Uses GPS poses to find Anchor/Positive/Negative triplets for metric learning.

Positive: same place (< d_pos meters), different time (> d_time seconds)
Negative: different place (> d_neg meters)

Usage:
    python -m src.step3_mine_triplets
    python -m src.step3_mine_triplets --config configs/config.yaml
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.spatial import KDTree
from tqdm import tqdm


def mine_triplets(
    poses_xy: np.ndarray,
    timestamps: np.ndarray,
    d_pos: float,
    d_neg: float,
    d_time: float,
    max_positives_per_anchor: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Mine triplets from frame poses.

    For each anchor, find positive candidates (close in space, far in time)
    and negative candidates (far in space). Sample up to
    max_positives_per_anchor triplets per anchor.
    """
    rng = random.Random(seed)
    N = len(poses_xy)

    print(f"  Building KD-tree over {N} poses...")
    tree = KDTree(poses_xy)

    # Pre-compute positive candidates for every frame
    print(f"  Querying positive candidates (d < {d_pos}m, dt > {d_time}s)...")
    pos_neighbors = tree.query_ball_tree(tree, r=d_pos)

    # Pre-compute negative candidates: frames farther than d_neg
    # For efficiency, we sample negatives on-the-fly per anchor
    print(f"  Mining triplets...")
    triplets = []

    for anchor in tqdm(range(N), desc="Mining"):
        # Filter positives: must be > d_time apart in time
        positives = [
            j for j in pos_neighbors[anchor]
            if j != anchor and abs(timestamps[j] - timestamps[anchor]) > d_time
        ]
        if not positives:
            continue

        # Sample a limited number of positives per anchor
        if len(positives) > max_positives_per_anchor:
            positives = rng.sample(positives, max_positives_per_anchor)

        for pos in positives:
            # Find a hard-ish negative: random frame that is far away
            # Try up to 10 times to find one
            for _ in range(10):
                neg = rng.randint(0, N - 1)
                dist = np.linalg.norm(poses_xy[anchor] - poses_xy[neg])
                if dist > d_neg:
                    triplets.append((anchor, pos, neg))
                    break

    df = pd.DataFrame(triplets, columns=["anchor_idx", "positive_idx", "negative_idx"])
    return df


def split_triplets(
    triplets_df: pd.DataFrame,
    n_frames: int,
    train_ratio: float,
    val_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split triplets by anchor frame index (time-based) to prevent data leakage.

    Only the anchor determines the split. Positives and negatives are allowed
    to come from any time — this matches real-world inference where you query
    the current frame against all historical frames.
    """
    train_end = int(n_frames * train_ratio)
    val_end = int(n_frames * (train_ratio + val_ratio))

    train = triplets_df[triplets_df["anchor_idx"] < train_end].copy()
    val = triplets_df[
        (triplets_df["anchor_idx"] >= train_end) &
        (triplets_df["anchor_idx"] < val_end)
    ].copy()
    test = triplets_df[triplets_df["anchor_idx"] >= val_end].copy()

    return train, val, test


def run(cfg: dict) -> None:
    paths = cfg["paths"]
    processed_root = Path(paths["processed_root"])
    mining_cfg = cfg["mining"]
    split_cfg = cfg["split"]

    # ── Load poses ────────────────────────────────────────────────────────
    poses_path = processed_root / "frame_poses.npy"
    if not poses_path.exists():
        print("ERROR: frame_poses.npy not found. Run step2_visualize_trajectory first.")
        return

    poses = np.load(poses_path)  # (N, 4): x, y, z, timestamp
    poses_xy = poses[:, :2]
    timestamps = poses[:, 3]
    N = len(poses)

    print(f"Loaded {N} frame poses")
    print(f"Mining parameters:")
    print(f"  d_pos  = {mining_cfg['d_pos_m']} m")
    print(f"  d_neg  = {mining_cfg['d_neg_m']} m")
    print(f"  d_time = {mining_cfg['d_time_s']} s")

    # ── Mine triplets ─────────────────────────────────────────────────────
    triplets_df = mine_triplets(
        poses_xy, timestamps,
        d_pos=mining_cfg["d_pos_m"],
        d_neg=mining_cfg["d_neg_m"],
        d_time=mining_cfg["d_time_s"],
    )
    print(f"\nTotal triplets mined: {len(triplets_df)}")

    if len(triplets_df) == 0:
        print("WARNING: No triplets found! Try adjusting d_pos, d_neg, or d_time.")
        return

    # ── Split by time ─────────────────────────────────────────────────────
    train_df, val_df, test_df = split_triplets(
        triplets_df, N,
        train_ratio=split_cfg["train_ratio"],
        val_ratio=split_cfg["val_ratio"],
    )

    print(f"\n── Split Summary (by time) ──")
    print(f"  Train: {len(train_df):>7} triplets  (anchors in frames 0–{int(N * split_cfg['train_ratio'])-1})")
    print(f"  Val:   {len(val_df):>7} triplets  (anchors in frames {int(N * split_cfg['train_ratio'])}–{int(N * (split_cfg['train_ratio'] + split_cfg['val_ratio']))-1})")
    print(f"  Test:  {len(test_df):>7} triplets  (anchors in frames {int(N * (split_cfg['train_ratio'] + split_cfg['val_ratio']))}–{N-1})")

    # ── Save ──────────────────────────────────────────────────────────────
    triplets_df.to_csv(processed_root / "triplets_all.csv", index=False)
    train_df.to_csv(processed_root / "triplets_train.csv", index=False)
    val_df.to_csv(processed_root / "triplets_val.csv", index=False)
    test_df.to_csv(processed_root / "triplets_test.csv", index=False)

    print(f"\nSaved:")
    print(f"  {processed_root / 'triplets_all.csv'}")
    print(f"  {processed_root / 'triplets_train.csv'}")
    print(f"  {processed_root / 'triplets_val.csv'}")
    print(f"  {processed_root / 'triplets_test.csv'}")

    # ── Triplet quality stats ─────────────────────────────────────────────
    sample = triplets_df.head(min(5000, len(triplets_df)))
    pos_dists = np.linalg.norm(
        poses_xy[sample["anchor_idx"].values] - poses_xy[sample["positive_idx"].values],
        axis=1,
    )
    neg_dists = np.linalg.norm(
        poses_xy[sample["anchor_idx"].values] - poses_xy[sample["negative_idx"].values],
        axis=1,
    )
    time_gaps = np.abs(
        timestamps[sample["anchor_idx"].values] - timestamps[sample["positive_idx"].values]
    )

    print(f"\n── Triplet Quality (sampled {len(sample)}) ──")
    print(f"  Anchor↔Positive distance: {pos_dists.mean():.2f}m ± {pos_dists.std():.2f}m  (max {pos_dists.max():.2f}m)")
    print(f"  Anchor↔Negative distance: {neg_dists.mean():.1f}m ± {neg_dists.std():.1f}m  (min {neg_dists.min():.1f}m)")
    print(f"  Anchor↔Positive time gap: {time_gaps.mean():.0f}s ± {time_gaps.std():.0f}s  (min {time_gaps.min():.0f}s)")


def main():
    parser = argparse.ArgumentParser(description="Step 1C: Triplet Mining")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run(cfg)


if __name__ == "__main__":
    main()
