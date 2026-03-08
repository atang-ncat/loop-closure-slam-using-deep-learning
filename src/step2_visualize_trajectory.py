"""
Step 1B — GPS Trajectory Visualization
=======================================
Plots the vehicle's UTM trajectory color-coded by time, identifies
loop-closure regions, and assigns a UTM pose to every synced frame.

Usage:
    python -m src.step2_visualize_trajectory
    python -m src.step2_visualize_trajectory --config configs/config.yaml
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.spatial import KDTree


def load_odometry(odom_path: Path) -> pd.DataFrame:
    """Load the NovAtel odometry CSV (UTM coordinates)."""
    df = pd.read_csv(odom_path)
    df["timestamp"] = df["timestamp_sec"] + df["timestamp_nsec"] / 1e9
    return df


def assign_poses_to_frames(sync_df: pd.DataFrame, odom_df: pd.DataFrame) -> np.ndarray:
    """
    For each synced frame, find the closest odometry pose by timestamp.
    Returns (N, 4): [x, y, z, timestamp] per frame.
    """
    odom_ts = odom_df["timestamp"].values
    odom_x = odom_df["position_x"].values
    odom_y = odom_df["position_y"].values
    odom_z = odom_df["position_z"].values

    poses = np.zeros((len(sync_df), 4))

    for i, row in sync_df.iterrows():
        lidar_ts = row["lidar_ts"]
        idx = np.searchsorted(odom_ts, lidar_ts)
        candidates = []
        if idx > 0:
            candidates.append(idx - 1)
        if idx < len(odom_ts):
            candidates.append(idx)
        best = min(candidates, key=lambda j: abs(odom_ts[j] - lidar_ts))
        poses[i] = [odom_x[best], odom_y[best], odom_z[best], lidar_ts]

    return poses


def find_loop_regions(
    poses: np.ndarray,
    d_pos: float,
    d_time: float,
    timestamps: np.ndarray,
) -> list[tuple[int, int, float]]:
    """
    Find pairs (i, j) where spatial distance < d_pos and temporal gap > d_time.
    Returns a list of (frame_i, frame_j, distance).
    """
    xy = poses[:, :2]
    tree = KDTree(xy)

    loops = []
    seen = set()
    pairs = tree.query_pairs(r=d_pos)
    for i, j in pairs:
        if abs(timestamps[i] - timestamps[j]) > d_time:
            key = (min(i, j), max(i, j))
            if key not in seen:
                seen.add(key)
                dist = np.linalg.norm(xy[i] - xy[j])
                loops.append((i, j, dist))

    return loops


def run(cfg: dict) -> None:
    paths = cfg["paths"]
    dataset_root = Path(paths["dataset_root"])
    processed_root = Path(paths["processed_root"])

    mining_cfg = cfg["mining"]
    d_pos = mining_cfg["d_pos_m"]
    d_time = mining_cfg["d_time_s"]

    # ── Load data ─────────────────────────────────────────────────────────
    sync_df = pd.read_csv(processed_root / "sync_pairs.csv")
    odom_df = load_odometry(dataset_root / paths["odometry_csv"])

    print(f"Synced frames: {len(sync_df)}")
    print(f"Odometry rows: {len(odom_df)}")

    # ── Assign UTM pose to every frame ────────────────────────────────────
    print("Assigning UTM poses to synced frames...")
    poses = assign_poses_to_frames(sync_df, odom_df)

    # Save for downstream use (Step 1C)
    poses_path = processed_root / "frame_poses.npy"
    np.save(poses_path, poses)
    print(f"  Saved {len(poses)} poses → {poses_path}")

    xy = poses[:, :2]
    timestamps = poses[:, 3]

    # Center for plotting
    xy_centered = xy - xy.mean(axis=0)

    # ── Find loop regions ─────────────────────────────────────────────────
    print(f"\nFinding loops (d_pos={d_pos}m, d_time={d_time}s)...")
    loops = find_loop_regions(poses, d_pos, d_time, timestamps)
    print(f"  Found {len(loops)} loop-closure pairs")

    if loops:
        loop_frames = set()
        for i, j, _ in loops:
            loop_frames.add(i)
            loop_frames.add(j)
        print(f"  Involving {len(loop_frames)} unique frames")

        dists = [d for _, _, d in loops]
        print(f"  Distance range: {min(dists):.2f}m – {max(dists):.2f}m")

    # ── Trajectory plot ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    # Left: trajectory colored by time
    time_norm = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min())
    sc = axes[0].scatter(
        xy_centered[:, 0], xy_centered[:, 1],
        c=time_norm, cmap="viridis", s=1, alpha=0.7,
    )
    axes[0].set_xlabel("East (m)")
    axes[0].set_ylabel("North (m)")
    axes[0].set_title("Vehicle Trajectory (colored by time)")
    axes[0].set_aspect("equal")
    cbar = plt.colorbar(sc, ax=axes[0], shrink=0.7)
    cbar.set_label("Normalized Time (0=start, 1=end)")

    # Mark start / end
    axes[0].plot(xy_centered[0, 0], xy_centered[0, 1], "g^", markersize=14, label="Start")
    axes[0].plot(xy_centered[-1, 0], xy_centered[-1, 1], "rs", markersize=14, label="End")
    axes[0].legend(fontsize=11)

    # Right: trajectory with loop regions highlighted
    axes[1].plot(xy_centered[:, 0], xy_centered[:, 1], "b-", alpha=0.3, linewidth=0.5)
    if loops:
        loop_i = np.array([l[0] for l in loops])
        loop_j = np.array([l[1] for l in loops])
        all_loop = np.unique(np.concatenate([loop_i, loop_j]))

        # Subsample for visibility
        step = max(1, len(all_loop) // 2000)
        sampled = all_loop[::step]

        axes[1].scatter(
            xy_centered[sampled, 0], xy_centered[sampled, 1],
            c="red", s=4, alpha=0.5, label=f"Loop regions ({len(loops)} pairs)",
        )

    axes[1].set_xlabel("East (m)")
    axes[1].set_ylabel("North (m)")
    axes[1].set_title("Loop Closure Regions (red)")
    axes[1].set_aspect("equal")
    axes[1].legend(fontsize=11)

    plt.tight_layout()
    out_path = processed_root / "trajectory.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved trajectory plot → {out_path}")
    plt.close()

    # ── Stats ─────────────────────────────────────────────────────────────
    total_dist = np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1))
    duration = timestamps[-1] - timestamps[0]
    print(f"\n── Route Summary ──")
    print(f"  Total distance: {total_dist:.0f} m ({total_dist/1000:.2f} km)")
    print(f"  Duration:       {duration:.0f} s ({duration/60:.1f} min)")
    print(f"  Avg speed:      {total_dist/duration:.1f} m/s ({total_dist/duration*3.6:.1f} km/h)")


def main():
    parser = argparse.ArgumentParser(description="Step 1B: Trajectory Visualization")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run(cfg)


if __name__ == "__main__":
    main()
