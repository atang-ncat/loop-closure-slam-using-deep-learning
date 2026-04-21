"""
Step 4 — Evaluation: PR Curves, Recall@K, mAP, Ablations
=========================================================
Generates descriptors for all test frames, computes pairwise similarity,
and produces Precision-Recall curves comparing:
  (a) Fused model (LiDAR + Camera)
  (b) LiDAR-only  (camera zeroed out)
  (c) Camera-only  (LiDAR zeroed out)

Adds on top of the original:
  - Recall@K  for K in {1, 5, 10}
  - mean Average Precision (mAP)
  - PCA whitening (fit on train descriptors)
  - Horizontal-flip TTA at inference time
  - Sequence averaging (temporal smoothing window)

Usage:
    python -m src.step7_evaluate
    python -m src.step7_evaluate --config configs/config.yaml --checkpoint checkpoints/best.pt
    python -m src.step7_evaluate --no-whiten --no-tta --seq-len 1    # raw baseline
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
    tta: bool = False,
) -> np.ndarray:
    """
    Extract descriptors for a set of frames.

    mode: "fused" | "lidar_only" | "camera_only"
    tta:  if True, average descriptor over {x, hflip(x)} and re-normalize.
    """
    model.eval()
    cam_transform = get_camera_transform(train=False, input_size=camera_size)

    all_descs = []
    iterator = range(0, len(frame_indices), batch_size)
    pbar = tqdm(iterator, desc=f"  Extract ({mode}{', TTA' if tta else ''})", leave=False)
    for start in pbar:
        batch_idx = frame_indices[start : start + batch_size]
        lidar_batch = []
        cam_batch = []

        for idx in batch_idx:
            ri = np.load(range_dir / f"range_{idx:06d}.npy").astype(np.float32)
            ri = np.clip(ri / max_range, 0.0, 1.0)
            lidar_batch.append(torch.from_numpy(ri).unsqueeze(0))

            cam = cv2.imread(str(camera_dir / f"cam_{idx:06d}.jpg"))
            cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
            cam_batch.append(cam_transform(cam))

        lidar_t = torch.stack(lidar_batch).to(device)
        cam_t = torch.stack(cam_batch).to(device)

        descs = model.get_descriptor(lidar_t, cam_t, mode=mode)

        if tta:
            # Horizontal flip both modalities; already-normalized descriptors
            # are averaged then re-normalized.
            lidar_flip = torch.flip(lidar_t, dims=[-1])
            cam_flip = torch.flip(cam_t, dims=[-1])
            descs_flip = model.get_descriptor(lidar_flip, cam_flip, mode=mode)
            descs = descs + descs_flip
            descs = torch.nn.functional.normalize(descs, p=2, dim=1)

        all_descs.append(descs.cpu().numpy())

    return np.concatenate(all_descs, axis=0)


# ── PCA whitening ─────────────────────────────────────────────────────────────

def fit_pca_whitening(
    train_descs: np.ndarray, out_dim: int, eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit PCA whitening on a bank of training descriptors.

    Returns (mean, projection) where:
        whitened = L2_normalize((x - mean) @ projection)
    with projection of shape (D, out_dim).
    """
    mean = train_descs.mean(axis=0, keepdims=True)
    X = train_descs - mean
    # Covariance and eigendecomp
    cov = (X.T @ X) / max(X.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order][:out_dim]
    eigvecs = eigvecs[:, order][:, :out_dim]
    projection = eigvecs / np.sqrt(eigvals + eps)
    return mean.astype(np.float32), projection.astype(np.float32)


def apply_whitening(
    descs: np.ndarray, mean: np.ndarray, projection: np.ndarray,
) -> np.ndarray:
    out = (descs - mean) @ projection
    # L2-normalize so cosine similarity remains well-defined
    norms = np.linalg.norm(out, axis=1, keepdims=True) + 1e-8
    return (out / norms).astype(np.float32)


# ── Sequence averaging ────────────────────────────────────────────────────────

def sequence_smooth(
    descs: np.ndarray, frame_indices: np.ndarray, window: int,
) -> np.ndarray:
    """
    Average descriptors over a centered window of consecutive (in time) frames.

    `frame_indices` must be sorted and correspond row-wise to `descs`.
    For boundary frames we use the largest valid symmetric window.
    """
    if window <= 1:
        return descs
    N, D = descs.shape
    half = window // 2
    out = np.zeros_like(descs)
    for i in range(N):
        lo = max(0, i - half)
        hi = min(N, i + half + 1)
        out[i] = descs[lo:hi].mean(axis=0)
    norms = np.linalg.norm(out, axis=1, keepdims=True) + 1e-8
    return (out / norms).astype(np.float32)


# ── Re-ranking: α-QE and dual-softmax ─────────────────────────────────────────

def alpha_query_expansion(
    descs: np.ndarray, top_k: int = 5, alpha: float = 3.0, d_time: float = 60.0,
    timestamps: np.ndarray | None = None,
) -> np.ndarray:
    """
    α-weighted query expansion (Radenović et al. 2018).

    For each descriptor d_i, build an expanded descriptor:
        d_i' = L2_normalize( Σ_{j ∈ topK ∪ {i}}  s_ij^α · d_j )
    where s_ij is the cosine similarity between d_i and d_j.

    If `timestamps` is provided we exclude temporally-close neighbors (|Δt| ≤
    d_time) from the top-K pool; that prevents QE from just averaging over
    temporally-adjacent near-duplicates, which would reinforce the current
    frame rather than pull in diverse visits to the same place.
    """
    N = descs.shape[0]
    sim = descs @ descs.T
    if timestamps is not None:
        dt = np.abs(timestamps[:, None] - timestamps[None, :])
        temporal_mask = dt > d_time
        np.fill_diagonal(temporal_mask, False)
    else:
        temporal_mask = ~np.eye(N, dtype=bool)

    sim_masked = np.where(temporal_mask, sim, -np.inf)

    # top-K indices per row (excluding self / temporally-close neighbors)
    top_idx = np.argpartition(-sim_masked, kth=min(top_k, N - 1), axis=1)[:, :top_k]
    rows = np.arange(N)[:, None]
    top_sim = sim_masked[rows, top_idx]

    weights = np.clip(top_sim, 0.0, None) ** alpha
    weights[~np.isfinite(weights)] = 0.0

    out = descs.copy()
    for i in range(N):
        idx = top_idx[i]
        w = weights[i]
        w_sum = w.sum()
        if w_sum <= 0:
            continue
        out[i] = descs[i] + (w[:, None] * descs[idx]).sum(axis=0) / w_sum

    norms = np.linalg.norm(out, axis=1, keepdims=True) + 1e-8
    return (out / norms).astype(np.float32)


def dual_softmax_similarity(sim: np.ndarray, temp: float = 0.1) -> np.ndarray:
    """
    Dual-softmax similarity (symmetrizes mutual top-1 behavior).
        s'_ij = softmax_j( sim_i / T )  *  softmax_i( sim_j / T )
    Strengthens i→j pairs only when both directions agree.
    """
    s = sim / max(temp, 1e-6)
    s_row = s - s.max(axis=1, keepdims=True)
    row = np.exp(s_row); row /= row.sum(axis=1, keepdims=True) + 1e-12
    s_col = s - s.max(axis=0, keepdims=True)
    col = np.exp(s_col); col /= col.sum(axis=0, keepdims=True) + 1e-12
    return (row * col).astype(np.float32)


# ── Metrics: PR, Recall@K, mAP ─────────────────────────────────────────────────

def compute_retrieval_metrics(
    descriptors: np.ndarray,
    poses_xy: np.ndarray,
    timestamps: np.ndarray,
    d_pos: float,
    d_time: float,
    ranks: list[int],
    sim_matrix: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Compute PR curve + Recall@K + mAP from pairwise descriptor similarity.

    Ground truth: pair (i, j) is a match iff spatial distance < d_pos AND
    time gap > d_time. Excludes self-comparisons and ignores pairs with time
    gap ≤ d_time (trivially adjacent).

    If `sim_matrix` is provided, it is used instead of `descriptors @ descriptors.T`
    (lets callers pass dual-softmax / re-ranked similarities directly).
    """
    N = len(descriptors)
    if sim_matrix is None:
        sim_matrix = descriptors @ descriptors.T

    gt_match = np.zeros((N, N), dtype=bool)
    for i in range(N):
        dx = poses_xy - poses_xy[i]
        spatial_close = np.einsum("ij,ij->i", dx, dx) < d_pos * d_pos
        temporal_far = np.abs(timestamps - timestamps[i]) > d_time
        gt_match[i] = spatial_close & temporal_far
        gt_match[i, i] = False

    # PR curve over upper triangle
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    scores = sim_matrix[mask]
    labels = gt_match[mask].astype(int)

    metrics = {
        "n_total_pairs": int(len(labels)),
        "n_positive_pairs": int(labels.sum()),
    }

    if labels.sum() == 0:
        print("  WARNING: No positive pairs in evaluation set!")
        return np.array([1.0]), np.array([0.0]), {**metrics, "auc_pr": 0.0}

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    auc_pr = auc(recall, precision)

    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    f1_max_idx = int(np.argmax(f1))
    f1_max = float(f1[f1_max_idx])
    best_threshold = float(thresholds[f1_max_idx]) if f1_max_idx < len(thresholds) else 0.0

    # Recall@K and mAP per query
    recall_at_k = {k: 0 for k in ranks}
    ap_list = []
    queries_with_match = 0

    for i in range(N):
        if not gt_match[i].any():
            continue
        queries_with_match += 1
        sims = sim_matrix[i].copy()
        sims[i] = -np.inf
        order = np.argsort(-sims)  # descending
        gt_row = gt_match[i][order]

        # Recall@K: is any of the top-K a true positive?
        for k in ranks:
            if gt_row[:k].any():
                recall_at_k[k] += 1

        # Average Precision for this query
        hits = np.cumsum(gt_row).astype(np.float32)
        precisions_at_hits = hits / np.arange(1, len(gt_row) + 1, dtype=np.float32)
        relevant_count = int(gt_row.sum())
        if relevant_count > 0:
            ap = float((precisions_at_hits * gt_row).sum() / relevant_count)
            ap_list.append(ap)

    metrics.update({
        "auc_pr": float(auc_pr),
        "f1_max": f1_max,
        "best_threshold": best_threshold,
        "n_queries": queries_with_match,
        "mAP": float(np.mean(ap_list)) if ap_list else 0.0,
    })
    for k in ranks:
        metrics[f"recall_at_{k}"] = recall_at_k[k] / max(queries_with_match, 1)

    return precision, recall, metrics


# ── Main evaluation ───────────────────────────────────────────────────────────

def run(cfg: dict, checkpoint_path: str, overrides: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    processed = Path(cfg["paths"]["processed_root"])
    range_dir = processed / "range_images"
    camera_dir = processed / "camera_synced"

    model_cfg = cfg["model"]
    mining_cfg = cfg["mining"]
    split_cfg = cfg["split"]
    eval_cfg = cfg.get("eval", {})
    cam_size = tuple(model_cfg["camera_input_size"])
    max_range = cfg["projection"]["max_range"]

    # Eval switches (CLI overrides config)
    use_whiten = overrides.get("whiten", eval_cfg.get("whiten", True))
    whiten_dim = int(overrides.get("whiten_dim", eval_cfg.get("whiten_dim", 256)))
    use_tta = overrides.get("tta", eval_cfg.get("tta", True))
    seq_len = int(overrides.get("seq_len", eval_cfg.get("seq_len", 1)))
    max_eval_frames = int(overrides.get(
        "max_eval_frames", eval_cfg.get("max_eval_frames", 2000),
    ))
    ranks = eval_cfg.get("ranks", [1, 5, 10])

    # Re-ranking switches (0 / False = disabled)
    alpha_qe_k = int(overrides.get("alpha_qe_k", eval_cfg.get("alpha_qe_k", 0)))
    alpha_qe_pow = float(overrides.get("alpha_qe_pow", eval_cfg.get("alpha_qe_pow", 3.0)))
    use_dual_softmax = bool(overrides.get(
        "dual_softmax", eval_cfg.get("dual_softmax", False),
    ))
    dual_softmax_temp = float(overrides.get(
        "dual_softmax_temp", eval_cfg.get("dual_softmax_temp", 0.1),
    ))
    tag = str(overrides.get("tag", ""))

    print(
        f"Eval config: whiten={use_whiten} (dim={whiten_dim}), "
        f"tta={use_tta}, seq_len={seq_len}, max_frames={max_eval_frames}"
    )
    if alpha_qe_k > 0:
        print(f"Re-ranking: α-QE (top_k={alpha_qe_k}, α={alpha_qe_pow})")
    if use_dual_softmax:
        print(f"Re-ranking: dual-softmax (T={dual_softmax_temp})")

    # ── Load model ────────────────────────────────────────────────────────
    model = SiameseNetwork(
        embedding_dim=model_cfg["embedding_dim"],
        backbone=model_cfg.get("backbone", "resnet50"),
        pretrained=False,
        gem_p=model_cfg.get("gem_p", 3.0),
        lidar_backbone=model_cfg.get("lidar_backbone", "resnet50"),
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

    if len(all_test_idx) > max_eval_frames:
        step = len(all_test_idx) // max_eval_frames
        all_test_idx = all_test_idx[::step][:max_eval_frames]
    print(f"Evaluating on {len(all_test_idx)} test frames")

    poses = np.load(processed / "frame_poses.npy")
    eval_poses = poses[all_test_idx, :2]
    eval_timestamps = poses[all_test_idx, 3]

    # ── Fit PCA whitening on training descriptors (optional) ──────────────
    whiten_mean = None
    whiten_proj = None
    if use_whiten:
        print("\nFitting PCA whitening on training descriptors...")
        N_total = len(poses)
        train_end = int(N_total * split_cfg["train_ratio"])
        train_idx = np.arange(train_end)
        max_train_whiten = int(eval_cfg.get("whiten_fit_frames", 3000))
        if len(train_idx) > max_train_whiten:
            step = len(train_idx) // max_train_whiten
            train_idx = train_idx[::step][:max_train_whiten]
        print(f"  Fitting on {len(train_idx)} train frames (fused, TTA={use_tta})")
        train_descs = extract_descriptors(
            model, train_idx, range_dir, camera_dir,
            max_range, cam_size, device, mode="fused", tta=use_tta,
        )
        whiten_mean, whiten_proj = fit_pca_whitening(train_descs, whiten_dim)
        print(f"  Whitening fitted: {train_descs.shape[1]}-d → {whiten_dim}-d")

    # ── Extract descriptors for all three modes ───────────────────────────
    print("\nExtracting test descriptors...")
    descs_by_mode = {}
    for mode in ["fused", "lidar_only", "camera_only"]:
        raw = extract_descriptors(
            model, all_test_idx, range_dir, camera_dir,
            max_range, cam_size, device, mode=mode, tta=use_tta,
        )
        if use_whiten:
            raw = apply_whitening(raw, whiten_mean, whiten_proj)
        raw = sequence_smooth(raw, all_test_idx, window=seq_len)
        descs_by_mode[mode] = raw

    # ── Optional re-ranking (α-QE on descriptors) ─────────────────────────
    if alpha_qe_k > 0:
        print(f"\nApplying α-QE re-ranking (top_k={alpha_qe_k}, α={alpha_qe_pow})...")
        for mode in list(descs_by_mode.keys()):
            descs_by_mode[mode] = alpha_query_expansion(
                descs_by_mode[mode],
                top_k=alpha_qe_k,
                alpha=alpha_qe_pow,
                d_time=mining_cfg["d_time_s"],
                timestamps=eval_timestamps,
            )

    # ── Compute metrics for each mode ─────────────────────────────────────
    d_pos = mining_cfg["d_pos_m"]
    d_time = mining_cfg["d_time_s"]
    print(f"\nComputing metrics (d_pos={d_pos}m, d_time={d_time}s, ranks={ranks})...")

    results = {}
    for mode, descs in descs_by_mode.items():
        print(f"  {mode}...")
        sim_for_metrics = None
        if use_dual_softmax:
            sim_for_metrics = dual_softmax_similarity(descs @ descs.T, temp=dual_softmax_temp)
        pr, re_, m = compute_retrieval_metrics(
            descs, eval_poses, eval_timestamps, d_pos, d_time, ranks,
            sim_matrix=sim_for_metrics,
        )
        results[mode] = {"precision": pr, "recall": re_, "metrics": m}

    # ── Print results ─────────────────────────────────────────────────────
    modes = ["fused", "lidar_only", "camera_only"]
    metric_keys = ["auc_pr", "f1_max", "mAP"] + [f"recall_at_{k}" for k in ranks] + ["best_threshold"]
    header = f"{'Metric':<20}" + "".join(f"{m:>14}" for m in modes)
    print(f"\n{'=' * len(header)}")
    print(header)
    print(f"{'=' * len(header)}")
    for key in metric_keys:
        row = f"{key:<20}" + "".join(
            f"{results[m]['metrics'].get(key, 0.0):>14.4f}" for m in modes
        )
        print(row)
    print(f"{'=' * len(header)}")
    m_f = results["fused"]["metrics"]
    print(
        f"  Positive pairs: {m_f['n_positive_pairs']:,}  |  "
        f"Total pairs: {m_f['n_total_pairs']:,}  |  "
        f"Queries with matches: {m_f['n_queries']}"
    )

    # ── Plot PR curves ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"fused": ("b-", "Fused"), "lidar_only": ("g--", "LiDAR-only"), "camera_only": ("r:", "Camera-only")}
    for mode in modes:
        style, label = colors[mode]
        m = results[mode]["metrics"]
        ax.plot(
            results[mode]["recall"], results[mode]["precision"], style, linewidth=2,
            label=f"{label} (AUC={m['auc_pr']:.3f}, R@1={m['recall_at_1']:.3f}, R@5={m['recall_at_5']:.3f})",
        )

    title_parts = ["Precision-Recall: Loop Closure Detection"]
    tags = []
    if use_whiten:
        tags.append(f"whiten→{whiten_dim}d")
    if use_tta:
        tags.append("TTA")
    if seq_len > 1:
        tags.append(f"seq={seq_len}")
    if tags:
        title_parts.append(f"[{', '.join(tags)}]")
    ax.set_xlabel("Recall", fontsize=13)
    ax.set_ylabel("Precision", fontsize=13)
    ax.set_title(" ".join(title_parts), fontsize=13)
    ax.legend(fontsize=10, loc="lower left")
    ax.set_xlim([0, 1.02])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)

    suffix_bits = []
    if use_whiten: suffix_bits.append("whiten")
    if use_tta: suffix_bits.append("tta")
    if seq_len > 1: suffix_bits.append(f"seq{seq_len}")
    if alpha_qe_k > 0: suffix_bits.append(f"aqe{alpha_qe_k}")
    if use_dual_softmax: suffix_bits.append("dsm")
    if tag: suffix_bits.insert(0, tag)
    suffix = ("_" + "_".join(suffix_bits)) if suffix_bits else ""
    out_path = processed / f"pr_curves{suffix}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved PR curves → {out_path}")

    # ── Save metrics to CSV ───────────────────────────────────────────────
    rows = []
    for mode in modes:
        row = {"mode": mode, **results[mode]["metrics"]}
        row["whiten"] = use_whiten
        row["whiten_dim"] = whiten_dim if use_whiten else 0
        row["tta"] = use_tta
        row["seq_len"] = seq_len
        row["alpha_qe_k"] = alpha_qe_k
        row["alpha_qe_pow"] = alpha_qe_pow if alpha_qe_k > 0 else 0.0
        row["dual_softmax"] = use_dual_softmax
        row["tag"] = tag
        rows.append(row)
    metrics_df = pd.DataFrame(rows)
    metrics_path = processed / f"eval_metrics{suffix}.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics  → {metrics_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Step 4: Evaluation & PR Curves")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--no-whiten", action="store_true", help="Disable PCA whitening")
    parser.add_argument("--no-tta", action="store_true", help="Disable horizontal-flip TTA")
    parser.add_argument("--whiten-dim", type=int, default=None, help="Whitened descriptor dim (default from config)")
    parser.add_argument("--seq-len", type=int, default=None, help="Sequence-averaging window (1 = disabled)")
    parser.add_argument("--max-frames", type=int, default=None, help="Cap on test frames for N×N")
    parser.add_argument("--alpha-qe", type=int, default=None, help="α-QE top_k (0 = disabled)")
    parser.add_argument("--alpha-qe-pow", type=float, default=None, help="α-QE exponent")
    parser.add_argument("--dual-softmax", action="store_true", help="Apply dual-softmax re-ranking")
    parser.add_argument("--dual-softmax-temp", type=float, default=None, help="Dual-softmax temperature")
    parser.add_argument("--tag", type=str, default=None, help="Custom prefix for output filenames (keeps runs separated)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    overrides = {}
    if args.no_whiten: overrides["whiten"] = False
    if args.no_tta: overrides["tta"] = False
    if args.whiten_dim is not None: overrides["whiten_dim"] = args.whiten_dim
    if args.seq_len is not None: overrides["seq_len"] = args.seq_len
    if args.max_frames is not None: overrides["max_eval_frames"] = args.max_frames
    if args.alpha_qe is not None: overrides["alpha_qe_k"] = args.alpha_qe
    if args.alpha_qe_pow is not None: overrides["alpha_qe_pow"] = args.alpha_qe_pow
    if args.dual_softmax: overrides["dual_softmax"] = True
    if args.dual_softmax_temp is not None: overrides["dual_softmax_temp"] = args.dual_softmax_temp
    if args.tag is not None: overrides["tag"] = args.tag

    run(cfg, args.checkpoint, overrides)


if __name__ == "__main__":
    main()
