"""
Step 4 — Evaluation: PR Curves & Ablation
==========================================
Generates descriptors for all test frames, computes pairwise similarity,
and produces Precision-Recall curves comparing:
  (a) Fused model (LiDAR + Camera)
  (b) LiDAR-only  (camera zeroed out)
  (c) Camera-only  (LiDAR zeroed out)

Usage:
    python -m src.step7_evaluate
    python -m src.step7_evaluate --config configs/config.yaml --checkpoint checkpoints/best.pt
"""

import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import auc, precision_recall_curve
from tqdm import tqdm

from src.step5_model import SiameseNetwork
from src.step4_dataset import get_camera_transform


# ── Descriptor extraction ─────────────────────────────────────────────────────

@torch.no_grad()
def extract_descriptors(
    model: SiameseNetwork,
    frame_indices: np.ndarray,
    range_dir: Path,
    camera_dir: Path,
    max_range: float,
    camera_size: tuple[int, int],
    device: torch.device,
    mode: str = "fused",
    batch_size: int = 64,
) -> np.ndarray:
    """
    Extract descriptors for a set of frames.

    mode: "fused" | "lidar_only" | "camera_only"
        - fused:       both sensors active
        - lidar_only:  camera tensor zeroed out
        - camera_only: LiDAR tensor zeroed out
    """
    model.eval()
    cam_transform = get_camera_transform(train=False, input_size=camera_size)

    all_descs = []

    for start in tqdm(
        range(0, len(frame_indices), batch_size),
        desc=f"  Extracting ({mode})",
        leave=False,
    ):
        batch_idx = frame_indices[start : start + batch_size]
        lidar_batch = []
        cam_batch = []

        for idx in batch_idx:
            # Range image
            ri = np.load(range_dir / f"range_{idx:06d}.npy").astype(np.float32)
            ri = ri / max_range
            ri = np.clip(ri, 0.0, 1.0)
            lidar_batch.append(torch.from_numpy(ri).unsqueeze(0))

            # Camera
            cam = cv2.imread(str(camera_dir / f"cam_{idx:06d}.jpg"))
            cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
            cam_batch.append(cam_transform(cam))

        lidar_t = torch.stack(lidar_batch).to(device)   # (B, 1, 128, 1024)
        cam_t = torch.stack(cam_batch).to(device)        # (B, 3, 224, 224)

        descs = model.get_descriptor(lidar_t, cam_t, mode=mode)
        all_descs.append(descs.cpu().numpy())

    return np.concatenate(all_descs, axis=0)


# ── Precision-Recall computation ──────────────────────────────────────────────

def compute_pr_curve(
    descriptors: np.ndarray,
    poses_xy: np.ndarray,
    d_pos: float,
    d_time: float,
    timestamps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, dict]:
    """
    Compute Precision-Recall curve from pairwise descriptor similarity.

    Ground truth: two frames are a true match if GPS distance < d_pos
    AND time gap > d_time (to exclude trivially adjacent frames).

    Returns: (precision, recall, auc_pr, metrics_dict)
    """
    N = len(descriptors)

    # Pairwise cosine similarity (descriptors are L2-normalized → dot product)
    sim_matrix = descriptors @ descriptors.T

    # Ground truth: binary match matrix
    gt_match = np.zeros((N, N), dtype=bool)
    for i in range(N):
        for j in range(i + 1, N):
            spatial_close = np.linalg.norm(poses_xy[i] - poses_xy[j]) < d_pos
            temporal_far = abs(timestamps[i] - timestamps[j]) > d_time
            if spatial_close and temporal_far:
                gt_match[i, j] = True
                gt_match[j, i] = True

    # Extract upper triangle (avoid self-comparisons and duplicates)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    scores = sim_matrix[mask]
    labels = gt_match[mask].astype(int)

    if labels.sum() == 0:
        print("  WARNING: No positive pairs in evaluation set!")
        return np.array([1.0]), np.array([0.0]), 0.0, {}

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    auc_pr = auc(recall, precision)

    # F1-max
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    f1_max_idx = np.argmax(f1)
    f1_max = f1[f1_max_idx]
    best_threshold = thresholds[f1_max_idx] if f1_max_idx < len(thresholds) else 0.0

    # Recall@1: for each query with at least one true match,
    # is the top-1 retrieval correct?
    recall_at_1_hits = 0
    recall_at_1_total = 0
    for i in range(N):
        if gt_match[i].sum() == 0:
            continue
        recall_at_1_total += 1
        # Top-1 retrieval: highest similarity excluding self
        sims = sim_matrix[i].copy()
        sims[i] = -np.inf
        top1 = np.argmax(sims)
        if gt_match[i, top1]:
            recall_at_1_hits += 1

    recall_at_1 = recall_at_1_hits / max(recall_at_1_total, 1)

    metrics = {
        "auc_pr": auc_pr,
        "f1_max": f1_max,
        "best_threshold": best_threshold,
        "recall_at_1": recall_at_1,
        "n_queries": recall_at_1_total,
        "n_positive_pairs": int(labels.sum()),
        "n_total_pairs": int(len(labels)),
    }

    return precision, recall, auc_pr, metrics


# ── Main evaluation ───────────────────────────────────────────────────────────

def run(cfg: dict, checkpoint_path: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    processed = Path(cfg["paths"]["processed_root"])
    range_dir = processed / "range_images"
    camera_dir = processed / "camera_synced"

    model_cfg = cfg["model"]
    mining_cfg = cfg["mining"]
    cam_size = tuple(model_cfg["camera_input_size"])
    max_range = cfg["projection"]["max_range"]

    # ── Load model ────────────────────────────────────────────────────────
    model = SiameseNetwork(
        embedding_dim=model_cfg["embedding_dim"],
        backbone=model_cfg.get("backbone", "resnet50"),
        pretrained=False,
        gem_p=model_cfg.get("gem_p", 3.0),
    )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")

    # ── Load test frame indices ───────────────────────────────────────────
    test_triplets = pd.read_csv(processed / "triplets_test.csv")
    all_test_idx = np.unique(np.concatenate([
        test_triplets["anchor_idx"].values,
        test_triplets["positive_idx"].values,
        test_triplets["negative_idx"].values,
    ]))
    all_test_idx.sort()

    # Subsample if too many frames (for tractable N×N similarity matrix)
    max_eval_frames = 2000
    if len(all_test_idx) > max_eval_frames:
        step = len(all_test_idx) // max_eval_frames
        all_test_idx = all_test_idx[::step][:max_eval_frames]

    print(f"Evaluating on {len(all_test_idx)} test frames")

    # ── Load poses ────────────────────────────────────────────────────────
    poses = np.load(processed / "frame_poses.npy")
    eval_poses = poses[all_test_idx, :2]
    eval_timestamps = poses[all_test_idx, 3]

    # ── Extract descriptors for all three modes ───────────────────────────
    print("\nExtracting descriptors...")
    descs_fused = extract_descriptors(
        model, all_test_idx, range_dir, camera_dir,
        max_range, cam_size, device, mode="fused",
    )
    descs_lidar = extract_descriptors(
        model, all_test_idx, range_dir, camera_dir,
        max_range, cam_size, device, mode="lidar_only",
    )
    descs_camera = extract_descriptors(
        model, all_test_idx, range_dir, camera_dir,
        max_range, cam_size, device, mode="camera_only",
    )

    # ── Compute PR curves ─────────────────────────────────────────────────
    d_pos = mining_cfg["d_pos_m"]
    d_time = mining_cfg["d_time_s"]

    print(f"\nComputing PR curves (d_pos={d_pos}m, d_time={d_time}s)...")
    print("  Fused model...")
    pr_fused, re_fused, auc_fused, m_fused = compute_pr_curve(
        descs_fused, eval_poses, d_pos, d_time, eval_timestamps,
    )
    print("  LiDAR-only...")
    pr_lidar, re_lidar, auc_lidar, m_lidar = compute_pr_curve(
        descs_lidar, eval_poses, d_pos, d_time, eval_timestamps,
    )
    print("  Camera-only...")
    pr_camera, re_camera, auc_camera, m_camera = compute_pr_curve(
        descs_camera, eval_poses, d_pos, d_time, eval_timestamps,
    )

    # ── Print results ─────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"{'Metric':<25} {'Fused':>10} {'LiDAR-only':>12} {'Camera-only':>13}")
    print(f"{'='*65}")
    print(f"{'AUC-PR':<25} {m_fused['auc_pr']:>10.4f} {m_lidar['auc_pr']:>12.4f} {m_camera['auc_pr']:>13.4f}")
    print(f"{'F1-max':<25} {m_fused['f1_max']:>10.4f} {m_lidar['f1_max']:>12.4f} {m_camera['f1_max']:>13.4f}")
    print(f"{'Recall@1':<25} {m_fused['recall_at_1']:>10.4f} {m_lidar['recall_at_1']:>12.4f} {m_camera['recall_at_1']:>13.4f}")
    print(f"{'Best threshold':<25} {m_fused['best_threshold']:>10.4f} {m_lidar['best_threshold']:>12.4f} {m_camera['best_threshold']:>13.4f}")
    print(f"{'='*65}")
    print(f"  Positive pairs: {m_fused['n_positive_pairs']:,}  |  Total pairs: {m_fused['n_total_pairs']:,}  |  Queries with matches: {m_fused['n_queries']}")

    # ── Plot PR curves ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(re_fused, pr_fused, "b-", linewidth=2,
            label=f"Fused (AUC={m_fused['auc_pr']:.4f}, R@1={m_fused['recall_at_1']:.3f})")
    ax.plot(re_lidar, pr_lidar, "g--", linewidth=2,
            label=f"LiDAR-only (AUC={m_lidar['auc_pr']:.4f}, R@1={m_lidar['recall_at_1']:.3f})")
    ax.plot(re_camera, pr_camera, "r:", linewidth=2,
            label=f"Camera-only (AUC={m_camera['auc_pr']:.4f}, R@1={m_camera['recall_at_1']:.3f})")

    ax.set_xlabel("Recall", fontsize=13)
    ax.set_ylabel("Precision", fontsize=13)
    ax.set_title("Precision-Recall: Loop Closure Detection", fontsize=14)
    ax.legend(fontsize=11, loc="lower left")
    ax.set_xlim([0, 1.02])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)

    out_path = processed / "pr_curves.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved PR curves → {out_path}")

    # ── Save metrics to CSV ───────────────────────────────────────────────
    metrics_df = pd.DataFrame([
        {"mode": "fused", **m_fused},
        {"mode": "lidar_only", **m_lidar},
        {"mode": "camera_only", **m_camera},
    ])
    metrics_path = processed / "eval_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics  → {metrics_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Step 4: Evaluation & PR Curves")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run(cfg, args.checkpoint)


if __name__ == "__main__":
    main()
