# Figure inventory

All numbers below are on the test split (2 000 frames, `d_pos = 5 m`,
`d_time = 60 s`). See `presentation/summary.md` and
`presentation/tables/main_results.md` for the full numeric story.

## Current results (re-ranking era — April 2026)

| File | Description |
|---|---|
| `pr_curves_main.png`         | 5-panel PR-curve grid: `Baseline (raw)` → `+α-QE(5)` → `+α-QE(10)` → `+α-QE(5)+DSM` → `+DSM(T=0.05)`. |
| `ablation_bars.png`          | Headline ablation chart — shows where the gains come from. Five canonical configs across AUC-PR / F1 / mAP / R@1 / R@5 / R@10 with a 0.8 reference line. |
| `dsm_temperature_sweep.png`  | Dual-softmax temperature sensitivity curve (T ∈ {0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, ∞}). Shows R@1 / R@5 / R@10 saturate around T ≈ 0.3 while AUC-PR keeps rising to T = 1.0. |
| `modality_bars.png`          | Per-modality breakdown (fused / lidar_only / camera_only) for the best config (`Baseline + TTA + α-QE(5) + DSM(T=1.00)`). Camera-only hits **R@10 = 1.0000** on this test subset. |
| `training_curves_pk_v2.png`  | PK-v2 training dynamics (40 epochs, P=16 K=6). Shows the MS-loss curve and val Recall@1, highlighting the best epoch (28). |

## Methodology / dataset figures (unchanged)

| File | Description |
|---|---|
| `trajectory.png`   | GPS trajectory for the raw KITTI-like route used by the dataset. Loop regions are highlighted. |
| `sample_pairs.png` | Example range-image + camera pairs used for training. Illustrates modality alignment. |

## Legacy — kept for historical reference only

| File | Description |
|---|---|
| `legacy/pr_curves_v0_baseline_raw.png`     | The PR-curve figure that accompanied the very first "Round 1" results (AUC-PR ≈ 0.12, R@1 ≈ 0.40). Superseded by every figure in the top-level directory. |
| `legacy/training_curves_v0_baseline.png`   | Older two-phase training curve (triplet → batch-hard). Superseded by `training_curves_pk_v2.png`. |

These are referenced by `revised_methodology.tex` for the legacy results section only; do **not** cite them as current results.
