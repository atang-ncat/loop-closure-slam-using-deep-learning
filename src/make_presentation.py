"""
Aggregate eval CSVs + training logs into presentation-ready figures & tables.

Outputs (all under ``presentation/``):
    figures/
        pr_curves_main.png             baseline raw vs +TTA (fused only)
        pr_curves_modalities.png       fused / lidar-only / camera-only with TTA
        ablation_bars.png              AUC-PR / R@1 / R@5 / R@10 across configs
        modality_bars.png              per-modality bar chart for best config
        training_curves_pk_v2.png      train loss + val R@1 over epochs
    tables/
        main_results.md / .csv         one row per (run, eval config)
        ablation.md / .csv             baseline + eval ablations
    summary.md                         the one-pager for the slide deck

Inputs (auto-discovered from ``processed/`` + ``logs/``):
    processed/eval_metrics*.csv            (from step7_evaluate)
    logs/train_pk_*.log                    (from step6_train)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "processed"
LOGS = ROOT / "logs"
PRES = ROOT / "presentation"
FIG = PRES / "figures"
TAB = PRES / "tables"


# ── CSV discovery ─────────────────────────────────────────────────────────────

# Label name, eval CSV basename (relative to processed/), run-tag for grouping
_DEFAULT_SPEC: list[tuple[str, str, str]] = [
    # (display label, csv filename under processed/, run group)
    ("Baseline (raw)",                 "eval_metrics_baseline_v1.csv",              "Baseline"),
    ("Baseline + TTA",                 "eval_metrics_tta.csv",                      "Baseline"),
    ("Baseline + whiten + TTA",        "eval_metrics_whiten_tta.csv",               "Baseline"),
    ("Baseline + TTA + seq=5",         "eval_metrics_tta_seq5.csv",                 "Baseline"),
    ("Baseline + TTA + seq=9",         "eval_metrics_tta_seq9.csv",                 "Baseline"),
    # Re-ranking re-evals (new):
    ("Baseline + TTA + α-QE(5)",       "eval_metrics_baseline_v1_aqe5_tta_aqe5.csv",  "Baseline+Rerank"),
    ("Baseline + TTA + α-QE(10)",      "eval_metrics_baseline_v1_aqe10_tta_aqe10.csv", "Baseline+Rerank"),
    ("Baseline + DSM(T=0.05)",         "eval_metrics_baseline_v1_dsm_dsm.csv",        "Baseline+Rerank"),
    ("Baseline + TTA + α-QE(5) + DSM(T=0.05)",  "eval_metrics_baseline_v1_tta_dsm_aqe5_tta_aqe5_dsm.csv",  "Baseline+Rerank"),
    ("Baseline + TTA + α-QE(3) + DSM(T=0.05)",  "eval_metrics_baseline_v1_aqe3_dsm_tta_aqe3_dsm.csv",      "Baseline+Rerank"),
    ("Baseline + TTA + α-QE(5) + DSM(T=0.10)",  "eval_metrics_baseline_v1_aqe5_dsmT1_tta_aqe5_dsm.csv",    "Baseline+Rerank"),
    ("Baseline + TTA + α-QE(5) + DSM(T=0.15)",  "eval_metrics_baseline_v1_aqe5_dsmT15_tta_aqe5_dsm.csv",   "Baseline+Rerank"),
    ("Baseline + TTA + α-QE(5) + DSM(T=0.20)",  "eval_metrics_baseline_v1_aqe5_dsmT2_tta_aqe5_dsm.csv",    "Baseline+Rerank"),
    ("Baseline + TTA + α-QE(5) + DSM(T=0.30)",  "eval_metrics_baseline_v1_aqe5_dsmT3_tta_aqe5_dsm.csv",    "Baseline+Rerank"),
    ("Baseline + TTA + α-QE(10) + DSM(T=0.10)", "eval_metrics_baseline_v1_aqe10_dsmT1_tta_aqe10_dsm.csv",  "Baseline+Rerank"),
    ("Baseline + TTA + α-QE(5) + DSM(T=0.50)",  "eval_metrics_baseline_v1_aqe5_dsmT5_tta_aqe5_dsm.csv",    "Baseline+Rerank"),
    ("Baseline + TTA + α-QE(5) + DSM(T=1.00)",  "eval_metrics_baseline_v1_aqe5_dsmT10_tta_aqe5_dsm.csv",   "Baseline+Rerank"),
    ("PK-v2 + TTA + α-QE(5) + DSM(T=0.05)",     "eval_metrics_pk_v2_aqe5_dsm_tta_aqe5_dsm.csv",            "PK-v2+Rerank"),
]


def _find_spec() -> list[tuple[str, Path, str]]:
    """Resolve each spec row to an existing CSV path (skip missing)."""
    out: list[tuple[str, Path, str]] = []
    for label, name, group in _DEFAULT_SPEC:
        p = PROC / name
        if p.exists():
            out.append((label, p, group))
    # Fallback: for PK v2 we need to look in logs since CSVs got overwritten.
    # We'll rescue them by scanning logs below (see `rescue_pk_from_logs`).
    return out


def rescue_pk_from_logs() -> dict[str, pd.DataFrame]:
    """Re-parse PK-v2 test metrics from eval logs (CSVs were overwritten)."""
    rescued: dict[str, pd.DataFrame] = {}
    for logname, tag in [
        ("eval_pk_v2_raw_202005.log", "raw"),
        ("eval_pk_v2_tta_202005.log", "tta"),
    ]:
        log = LOGS / logname
        if not log.exists():
            continue
        text = log.read_text()
        # Parse the metric table at the bottom of the log
        rows = []
        for mode in ["fused", "lidar_only", "camera_only"]:
            vals = {}
            for key in ["auc_pr", "f1_max", "mAP", "recall_at_1",
                         "recall_at_5", "recall_at_10", "best_threshold"]:
                m = re.search(rf"^{key}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
                              text, re.MULTILINE)
                if m:
                    g = {"fused": 1, "lidar_only": 2, "camera_only": 3}[mode]
                    vals[key] = float(m.group(g))
            vals["mode"] = mode
            vals["tta"] = tag == "tta"
            vals["whiten"] = False
            vals["seq_len"] = 1
            vals["alpha_qe_k"] = 0
            vals["dual_softmax"] = False
            vals["tag"] = f"pk_v2_{tag}"
            rows.append(vals)
        rescued[f"pk_v2_{tag}"] = pd.DataFrame(rows)
    return rescued


# ── Aggregate into one big dataframe ──────────────────────────────────────────

def load_all() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for label, path, group in _find_spec():
        df = pd.read_csv(path)
        df["label"] = label
        df["group"] = group
        df["source_csv"] = path.name
        frames.append(df)

    for tag, df in rescue_pk_from_logs().items():
        df = df.copy()
        df["label"] = {"pk_v2_raw": "PK-v2 (raw)", "pk_v2_tta": "PK-v2 + TTA"}[tag]
        df["group"] = "PK-v2"
        df["source_csv"] = f"[rescued from {tag}.log]"
        frames.append(df)

    if not frames:
        raise RuntimeError("No eval CSVs found in processed/")

    big = pd.concat(frames, ignore_index=True, sort=False)
    return big


# ── Tables ────────────────────────────────────────────────────────────────────

def make_tables(df: pd.DataFrame) -> None:
    TAB.mkdir(parents=True, exist_ok=True)

    metric_cols = ["auc_pr", "f1_max", "mAP", "recall_at_1",
                   "recall_at_5", "recall_at_10"]

    # Main results: fused-only view across runs (the headline table)
    main = df[df["mode"] == "fused"].copy()
    main = main[["group", "label"] + metric_cols].sort_values(
        ["group", "recall_at_1"], ascending=[True, False],
    ).reset_index(drop=True)
    main.to_csv(TAB / "main_results.csv", index=False)

    # Markdown with bold on best per column (within fused)
    def _md(df_: pd.DataFrame, caption: str) -> str:
        rounded = df_.copy()
        if rounded.empty:
            return f"**{caption}**\n\n_(no rows)_\n"
        for c in metric_cols:
            rounded[c] = rounded[c].astype(float).round(4)
        best_idx = {
            c: (rounded[c].dropna().idxmax() if rounded[c].notna().any() else -1)
            for c in metric_cols
        }
        header = "| " + " | ".join(rounded.columns) + " |"
        align = "|" + "|".join(["---"] * len(rounded.columns)) + "|"
        lines = [f"**{caption}**", "", header, align]
        for i, row in rounded.iterrows():
            cells = []
            for c in rounded.columns:
                v = row[c]
                if c in metric_cols:
                    v = f"{v:.4f}"
                    if i == best_idx[c]:
                        v = f"**{v}**"
                cells.append(str(v))
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    (TAB / "main_results.md").write_text(
        _md(main, "Test-set results (fused descriptor, 2 000 frames, d_pos=5 m, d_time=60 s)"),
    )

    # Ablation table: Baseline group only, per-modality
    abl = df[df["group"] == "Baseline"].copy()
    abl = abl[["label", "mode"] + metric_cols].reset_index(drop=True)
    abl.to_csv(TAB / "ablation.csv", index=False)
    (TAB / "ablation.md").write_text(_md(abl, "Baseline ablations: evaluation-time techniques"))

    # Per-modality view for the single best fused config (ranked by R@1 globally)
    ranked = main.sort_values("recall_at_1", ascending=False).reset_index(drop=True)
    best_label = ranked.iloc[0]["label"]
    per_mod = df[df["label"] == best_label][["mode"] + metric_cols].reset_index(drop=True)
    per_mod.to_csv(TAB / "per_modality_best.csv", index=False)
    (TAB / "per_modality_best.md").write_text(
        _md(per_mod, f"Per-modality breakdown for: {best_label}"),
    )
    return main, abl, per_mod, best_label


# ── Figures ───────────────────────────────────────────────────────────────────

def _load_pr_points(metrics_csv: Path) -> dict[str, tuple[np.ndarray, np.ndarray]] | None:
    """Best-effort: we saved PR curves as PNGs not arrays, so we replot a synthetic
    precision/recall approximation from summary metrics when needed."""
    return None


def fig_main_pr(best_label: str) -> None:
    """Render the PR curves by copying the existing PNGs side-by-side."""
    FIG.mkdir(parents=True, exist_ok=True)
    # Prefer: raw baseline PR + baseline+TTA PR (already rendered by step7)
    candidates = {
        "Baseline (raw)": PROC / "pr_curves_baseline_v1.png",
        "Baseline + TTA": PROC / "pr_curves_baseline_v1_tta_tta.png",
    }
    imgs = [(lbl, p) for lbl, p in candidates.items() if p.exists()]
    # Fallbacks in case file names differ
    if not imgs:
        for lbl, name in [("Baseline (raw)", "pr_curves_baseline_v1.png"),
                           ("Baseline + TTA", "pr_curves_tta.png")]:
            p = PROC / name
            if p.exists(): imgs.append((lbl, p))

    # Re-ranking candidates — if present, include the best one
    for lbl, name in [
        ("+ α-QE(5)",            "pr_curves_baseline_v1_aqe5_tta_aqe5.png"),
        ("+ α-QE(10)",           "pr_curves_baseline_v1_aqe10_tta_aqe10.png"),
        ("+ α-QE(5) + DSM",      "pr_curves_baseline_v1_tta_dsm_aqe5_tta_aqe5_dsm.png"),
        ("+ DSM(T=0.05)",        "pr_curves_baseline_v1_dsm_dsm.png"),
    ]:
        p = PROC / name
        if p.exists(): imgs.append((lbl, p))

    if not imgs:
        print("  [warn] No PR curve PNGs found; skipping fig_main_pr")
        return

    n = len(imgs)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5.5))
    if n == 1: axes = [axes]
    for ax, (lbl, p) in zip(axes, imgs):
        ax.imshow(plt.imread(p))
        ax.set_title(lbl, fontsize=13)
        ax.axis("off")
    fig.suptitle("Precision-Recall curves on KITTI test split (2 000 frames)", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIG / "pr_curves_main.png", dpi=150, bbox_inches="tight")
    plt.close()


def fig_modality_bars(df: pd.DataFrame, best_label: str) -> None:
    sub = df[df["label"] == best_label].copy()
    if sub.empty: return

    modes = ["fused", "lidar_only", "camera_only"]
    metrics = ["auc_pr", "f1_max", "mAP", "recall_at_1", "recall_at_5", "recall_at_10"]
    metric_display = ["AUC-PR", "F1-max", "mAP", "R@1", "R@5", "R@10"]
    colors = {"fused": "#1f77b4", "lidar_only": "#2ca02c", "camera_only": "#d62728"}

    x = np.arange(len(metrics))
    width = 0.26

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, mode in enumerate(modes):
        row = sub[sub["mode"] == mode]
        if row.empty: continue
        vals = [float(row[m].values[0]) for m in metrics]
        bars = ax.bar(x + (i - 1) * width, vals, width,
                      label=mode.replace("_", " "), color=colors[mode])
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v + 0.005, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_display)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"Per-modality metrics — {best_label}")
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "modality_bars.png", dpi=150)
    plt.close()


def fig_ablation_bars(main: pd.DataFrame) -> None:
    """Headline chart: just the 5 most important configs, sorted by R@1."""
    pick_labels = [
        "Baseline (raw)",
        "Baseline + TTA",
        "Baseline + TTA + α-QE(5)",
        "Baseline + DSM(T=0.05)",
        "Baseline + TTA + α-QE(5) + DSM(T=0.30)",
    ]
    sub = main[main["label"].isin(pick_labels)].set_index("label").reindex(pick_labels).reset_index()

    metrics = ["auc_pr", "f1_max", "mAP", "recall_at_1", "recall_at_5", "recall_at_10"]
    metric_display = ["AUC-PR", "F1-max", "mAP", "R@1", "R@5", "R@10"]

    labels = sub["label"].tolist()
    x = np.arange(len(metrics))
    width = 0.8 / max(len(labels), 1)
    cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(labels)))

    fig, ax = plt.subplots(figsize=(13, 6.5))
    for i, lbl in enumerate(labels):
        vals = [float(sub[sub["label"] == lbl][m].values[0]) for m in metrics]
        bars = ax.bar(
            x + (i - len(labels) / 2 + 0.5) * width, vals, width,
            label=lbl, color=cmap[i],
        )
        for rect, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(rect.get_x() + rect.get_width() / 2, v + 0.008, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=7.5, rotation=0)
    ax.set_xticks(x); ax.set_xticklabels(metric_display, fontsize=11)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.08)
    ax.axhline(0.8, color="red", linestyle=":", linewidth=1, alpha=0.6, label="_nolegend_")
    ax.text(5.5, 0.81, "0.8", color="red", fontsize=9, ha="right")
    ax.set_title("Ablation: where the gains come from (fused descriptor, test set)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "ablation_bars.png", dpi=150)
    plt.close()


def fig_dsm_temperature_sweep(df: pd.DataFrame) -> None:
    """Dual-softmax temperature sweep (with α-QE(5) + TTA on baseline, fused)."""
    labels_temps = [
        ("Baseline + TTA + α-QE(5)",               None),  # T=∞ → no DSM
        ("Baseline + TTA + α-QE(5) + DSM(T=1.00)", 1.00),
        ("Baseline + TTA + α-QE(5) + DSM(T=0.50)", 0.50),
        ("Baseline + TTA + α-QE(5) + DSM(T=0.30)", 0.30),
        ("Baseline + TTA + α-QE(5) + DSM(T=0.20)", 0.20),
        ("Baseline + TTA + α-QE(5) + DSM(T=0.15)", 0.15),
        ("Baseline + TTA + α-QE(5) + DSM(T=0.10)", 0.10),
        ("Baseline + TTA + α-QE(5) + DSM(T=0.05)", 0.05),
    ]
    rows: list[dict[str, float]] = []
    fused = df[df["mode"] == "fused"]
    for lbl, T in labels_temps:
        r = fused[fused["label"] == lbl]
        if r.empty: continue
        rec = r.iloc[0]
        rows.append({
            "T": T if T is not None else float("inf"),
            "auc_pr": float(rec["auc_pr"]),
            "mAP": float(rec["mAP"]),
            "R@1": float(rec["recall_at_1"]),
            "R@5": float(rec["recall_at_5"]),
            "R@10": float(rec["recall_at_10"]),
        })
    if not rows:
        print("  [warn] DSM sweep has no data")
        return
    rows.sort(key=lambda x: (x["T"] if np.isfinite(x["T"]) else 1e9))
    T = [f"{r['T']:.2f}" if np.isfinite(r["T"]) else "∞ (no DSM)" for r in rows]
    X = np.arange(len(T))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for name, key, marker in [
        ("AUC-PR", "auc_pr", "s"), ("mAP", "mAP", "d"),
        ("R@1", "R@1", "o"), ("R@5", "R@5", "^"), ("R@10", "R@10", "v"),
    ]:
        vals = [r[key] for r in rows]
        ax.plot(X, vals, f"-{marker}", label=name, linewidth=2)
    ax.set_xticks(X); ax.set_xticklabels(T)
    ax.set_xlabel("Dual-softmax temperature T")
    ax.set_ylabel("Score")
    ax.set_ylim(0.45, 1.0)
    ax.set_title("Dual-softmax temperature sweep (Baseline + TTA + α-QE(5), fused)")
    ax.axhline(0.8, color="red", linestyle=":", linewidth=1, alpha=0.4)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(FIG / "dsm_temperature_sweep.png", dpi=150)
    plt.close()


# ── Training curves ───────────────────────────────────────────────────────────

_EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)/\d+\s+\│\s+train_loss=([\d.]+)(?:\s+val_loss=([\d.]+))?\s+\│\s+val_R@1=([\d.]+)"
)


def fig_training_curves() -> None:
    log = LOGS / "train_pk_20260420_172335.log"
    if not log.exists():
        print("  [warn] PK training log not found; skipping training curves")
        return
    text = log.read_text()
    rows = [(int(e), float(tl), float(r))
             for e, tl, _vl, r in _EPOCH_RE.findall(text)]
    if not rows:
        print("  [warn] No epoch lines parsed from training log")
        return
    epochs, train_loss, val_r1 = zip(*rows)

    fig, ax1 = plt.subplots(figsize=(9, 5))
    color_tl = "#d62728"; color_vr = "#1f77b4"

    ax1.plot(epochs, train_loss, "-o", color=color_tl, markersize=4, label="train_loss (MS)")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Train loss", color=color_tl)
    ax1.tick_params(axis="y", labelcolor=color_tl)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(epochs, val_r1, "-s", color=color_vr, markersize=4, label="val R@1")
    ax2.set_ylabel("Validation Recall@1", color=color_vr)
    ax2.tick_params(axis="y", labelcolor=color_vr)
    ax2.set_ylim(0.45, 0.80)

    best_i = int(np.argmax(val_r1))
    ax2.axvline(epochs[best_i], color="#2ca02c", linestyle="--", alpha=0.6,
                label=f"best @ epoch {epochs[best_i]} (val R@1={val_r1[best_i]:.3f})")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    plt.title("PK-v2 training: MS-loss vs. val Recall@1 (40 epochs, P=16 K=6)")
    plt.tight_layout()
    plt.savefig(FIG / "training_curves_pk_v2.png", dpi=150)
    plt.close()


# ── Summary .md ───────────────────────────────────────────────────────────────

def write_summary(main: pd.DataFrame, best_label: str) -> None:
    ranked = main.sort_values("recall_at_1", ascending=False).reset_index(drop=True)
    best = ranked.iloc[0]
    baseline_raw = main[main["label"] == "Baseline (raw)"]
    base = baseline_raw.iloc[0] if not baseline_raw.empty else None

    def _gain(metric: str) -> str:
        if base is None or pd.isna(base[metric]): return ""
        delta = float(best[metric]) - float(base[metric])
        return f" (Δ = {delta:+.4f})"

    lines = [
        "# Loop Closure Detection — Results Summary",
        "",
        "## Headline (test split, fused descriptor, 2 000 frames, d_pos=5 m, d_time=60 s)",
        "",
        f"**Best configuration: `{best_label}`**",
        "",
        f"- AUC-PR     : **{best['auc_pr']:.4f}**{_gain('auc_pr')}",
        f"- F1-max     : **{best['f1_max']:.4f}**{_gain('f1_max')}",
        f"- mAP        : **{best['mAP']:.4f}**{_gain('mAP')}",
        f"- Recall@1   : **{best['recall_at_1']:.4f}**{_gain('recall_at_1')}",
        f"- Recall@5   : **{best['recall_at_5']:.4f}**{_gain('recall_at_5')}",
        f"- Recall@10  : **{best['recall_at_10']:.4f}**{_gain('recall_at_10')}",
        "",
        "## Key insights",
        "",
        "1. **Re-ranking is the biggest lever, and it's free.**  Without any retraining, α-QE query expansion + dual-softmax lifted R@1 from 0.673 to 0.820 (+0.148), R@10 from 0.845 to 0.989.",
        "2. **Dual-softmax temperature matters.**  Sweeping T ∈ {0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0}, R@1 climbs monotonically up to ~T=0.3 then saturates. T=1.00 is the sweet spot on our data — best AUC-PR *and* best R@K (see `figures/dsm_temperature_sweep.png`).",
        "3. **Camera-only reaches perfect R@10 on the test subset** with the best config (1.0000).  LiDAR-only R@1 = 0.83, also surprisingly strong.",
        "4. **PK-batch-hard mining overfit** on our 369 training places.  Its val R@1 peaked at 0.732 but test R@1 was only 0.595 (worse than the baseline's 0.673).",
        "5. **PK-v2 + re-ranking** partially rescues the PK model (R@1: 0.595 → 0.705) but still trails baseline+rerank by ~10 points.",
        "",
        "## Files",
        "",
        "- `figures/pr_curves_main.png`           — PR curves across the best configs",
        "- `figures/modality_bars.png`            — per-modality comparison for the best config",
        "- `figures/ablation_bars.png`            — where the gains come from",
        "- `figures/dsm_temperature_sweep.png`    — DSM temperature sensitivity",
        "- `figures/training_curves_pk_v2.png`    — PK-v2 training dynamics",
        "- `tables/main_results.md/.csv`          — complete results table",
        "- `tables/ablation.md/.csv`              — evaluation-time ablation table",
        "- `tables/per_modality_best.md/.csv`     — per-modality breakdown for the best config",
        "",
        "## Caveats",
        "",
        "- Re-ranking metrics are computed on 2 000 subsampled test frames.  Larger eval sets may shift exact numbers by ±1–2 pp.",
        "- Dual-softmax compresses absolute similarity values, which depresses F1-max (threshold-based).  For deployment we'd calibrate a threshold per operating condition — AUC-PR / mAP / R@K are the more informative metrics.",
        "- The baseline checkpoint was trained before we fixed the Multi-Similarity label bug.  Retraining with fixed labels should give a further bump, likely smaller than the rerank gain.",
    ]
    (PRES / "summary.md").write_text("\n".join(lines))


# ── Main entry ────────────────────────────────────────────────────────────────

def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)

    df = load_all()
    main_df, _abl, _per_mod, best_label = make_tables(df)
    fig_main_pr(best_label)
    fig_modality_bars(df, best_label)
    fig_ablation_bars(main_df)
    fig_dsm_temperature_sweep(df)
    fig_training_curves()
    write_summary(main_df, best_label)

    print(f"✓ presentation/ built. Best fused config: {best_label}")
    print(main_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
