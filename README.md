# Loop Closure Detection Using Multi-Modal Deep Learning

A deep-learning pipeline for loop closure detection in autonomous navigation.
Fuses LiDAR range images and RGB camera data into compact location descriptors
via metric learning, and applies eval-time re-ranking
(α-weighted query expansion + dual-softmax) to push retrieval metrics well
above naive cosine-similarity retrieval.

## Headline results

Test split, fused descriptor, 2 000 frames, `d_pos=5 m`, `d_time=60 s`:

| Metric | Baseline (raw) | **Best** (`Baseline + TTA + α-QE(5) + DSM(T=1.00)`) | Δ |
|---|---|---|---|
| AUC-PR | 0.5367 | **0.6417** | +0.105 |
| F1-max | 0.7351 | 0.7010 | −0.034¹ |
| mAP    | — | **0.7568** | new |
| **R@1**  | 0.6727 | **0.8252** | **+0.153** |
| **R@5**  | 0.8181 | **0.9792** | **+0.161** |
| **R@10** | 0.8450 | **0.9919** | **+0.147** |

¹ Dual-softmax compresses absolute similarity values → threshold-based F1-max
drops even as ranking improves. AUC-PR / mAP / R@K are more informative.

Per-modality on the best config:

| modality | AUC-PR | F1 | mAP | R@1 | R@5 | **R@10** |
|---|---|---|---|---|---|---|
| fused        | 0.642 | 0.701 | 0.757 | 0.825 | 0.979 | 0.992 |
| lidar only   | 0.481 | 0.622 | 0.661 | **0.833** | 0.959 | 0.964 |
| camera only  | 0.597 | 0.690 | 0.764 | **0.839** | 0.990 | **1.0000** |

See `presentation/summary.md` and `presentation/tables/main_results.md` for
the full 21-config comparison and caveats.

## Overview

Two-Stream Siamese network produces L2-normalized location fingerprints for
visual place recognition. The robot detects revisits by comparing descriptors
in embedding space, optionally with inference-time re-ranking.

### Architecture

- **LiDAR branch:** `OverlapTransformer` (azimuth-wise transformer + NetVLAD
  aggregation) on `128 × 1024` range images from an Ouster OS1-128.
- **Camera branch:** ResNet-50 + Generalized Mean (GeM) pooling on
  `224 × 224` RGB images.
- **Fusion head:** gated fusion MLP → 512-d L2-normalized descriptor.

### Training

- Multi-Similarity loss with **place-level labels** (per-place clusters built
  from GPS).
- **PK batch sampling** (P places × K frames per place) for batch-hard mining.
- Differential learning rates: low on pre-trained backbones, high on the
  randomly-initialized fusion head. Linear warmup → Cosine Annealing Warm
  Restarts.
- Modality dropout (0.3) so the fused descriptor degrades gracefully when a
  sensor fails.
- AMP mixed-precision training.
- **Retrieval-based validation** (`val_R@1`) for early stopping — a much
  stronger signal than the old triplet val loss.

### Evaluation & re-ranking

- PR curves, AUC-PR, F1-max, mAP, **R@1 / R@5 / R@10** across three modes
  (fused, LiDAR-only, camera-only).
- **Flip-TTA** (horizontal-flip test-time augmentation on both modalities).
- **α-QE** (α-weighted query expansion) — reweights query with its top-k
  neighbours before recomputing similarity.
- **Dual-softmax** — symmetric softmax re-ranking that strengthens mutually
  consistent matches.
- Optional PCA whitening and temporal sequence averaging (both disabled by
  default — they hurt on this dataset).

## Dataset

Proprietary 500+ GB dataset collected on the NC A&T State University campus:

- Ouster OS1-128 LiDAR (10 Hz)
- Forward-facing USB RGB camera (30 Hz)
- NovAtel SPAN GNSS/INS ground truth (centimeter-level GPS)

Frames are time-synchronized using an Approximate Time Synchronizer with a
50 ms tolerance. Only 369 geographic "places" end up with a genuine revisit
(Δt > 60 s) in the training split — small for a retrieval problem, which is
why augmentation and re-ranking matter so much here.

## Project structure

```
loop-closure-slam/
├── configs/
│   └── config.yaml                # All hyperparameters + rerank defaults
├── src/
│   ├── step1_sync_and_project.py  # Sensor sync + range-image projection
│   ├── step2_visualize_trajectory.py
│   ├── step3_mine_triplets.py     # GPS-supervised triplet mining
│   ├── step4_dataset.py           # Triplet dataset + augmentation
│   ├── step4b_frame_dataset.py    # Frame-level place dataset + PK sampler
│   ├── step5_model.py             # Two-Stream Siamese network
│   ├── overlap_transformer.py     # LiDAR global descriptor backbone
│   ├── step6_train.py             # PK / triplet training loop
│   ├── step7_evaluate.py          # PR curves + re-ranking eval
│   └── make_presentation.py       # Aggregate CSVs + logs → figures/tables
├── presentation/
│   ├── summary.md                 # One-page results writeup
│   ├── figures/                   # Current figures (+ legacy/ subdir)
│   ├── tables/                    # main_results, ablation, per_modality
│   └── data/                      # Frozen baseline raw/TTA snapshots
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Setup

### Local

```bash
pip install -r requirements.txt
```

### Docker (recommended for GPU clusters)

Requires Docker and `nvidia-container-toolkit`.

```bash
docker compose build
```

## Usage

### 1. Pre-process data

```bash
python3 -m src.step1_sync_and_project --config configs/config.yaml
python3 -m src.step2_visualize_trajectory --config configs/config.yaml
```

### 2. Mine triplets (used for the triplet-mode path and for test splits)

```bash
python3 -m src.step3_mine_triplets --config configs/config.yaml
```

### 3. Train

PK-sampler mode (default — requires `training.use_pk: true`):

```bash
python3 -m src.step6_train --config configs/config.yaml
```

Resume from a checkpoint:

```bash
python3 -m src.step6_train --config configs/config.yaml --resume checkpoints/last.pt
```

### 4. Evaluate

Vanilla evaluation (no re-ranking):

```bash
python3 -m src.step7_evaluate \
    --checkpoint checkpoints/best.pt --no-whiten --no-tta
```

Best configuration (re-ranking enabled, matches headline numbers):

```bash
python3 -m src.step7_evaluate \
    --checkpoint checkpoints/best.pt \
    --alpha-qe 5 --alpha-qe-pow 3.0 \
    --dual-softmax --dual-softmax-temp 1.00 \
    --tag best
```

Useful flags:

| Flag | Purpose |
|---|---|
| `--tag NAME`            | Prefix for output CSV / PNG so parallel runs don't clobber. |
| `--alpha-qe K`          | α-QE top-k neighbours (0 disables). |
| `--alpha-qe-pow A`      | α-QE exponent on neighbour similarities. |
| `--dual-softmax`        | Apply dual-softmax rerank on top of descriptors. |
| `--dual-softmax-temp T` | DSM temperature (higher ≈ softer; `T=1.0` best on this data). |
| `--no-tta`              | Disable flip-TTA. |
| `--no-whiten`           | Disable PCA whitening (default — hurts on this dataset). |
| `--seq-len N`           | Temporal smoothing window (1 = disabled). |

### 5. Build presentation figures & tables

Aggregates every `processed/eval_metrics*.csv` + the PK training log into
figures and markdown tables under `presentation/`:

```bash
python3 -m src.make_presentation
```

### Docker commands

```bash
docker compose run mine
docker compose run train
docker compose run evaluate
```

## Configuration

All hyperparameters live in `configs/config.yaml`. Key knobs:

| Parameter | Default | Description |
|---|---|---|
| `model.backbone`                | `resnet50`             | Camera backbone |
| `model.lidar_backbone`          | `overlap_transformer`  | LiDAR backbone |
| `model.embedding_dim`           | `512`                  | Descriptor dimensionality |
| `model.gem_p`                   | `3.0`                  | GeM pooling initial power |
| `training.loss`                 | `multi_similarity`     | `triplet` or `multi_similarity` |
| `training.use_pk`               | `true`                 | PK batch sampler + place labels |
| `training.P` / `training.K`     | `16` / `6`             | Places per batch × frames per place |
| `training.epochs_worth`         | `12`                   | Batches per epoch multiplier |
| `training.backbone_lr`          | `1e-5`                 | LR for pre-trained backbones |
| `training.fusion_lr`            | `1e-3`                 | LR for fusion head |
| `training.weight_decay`         | `1e-4`                 |  |
| `training.modality_dropout_prob`| `0.3`                  | Per-batch probability of zeroing a modality |
| `training.num_epochs`           | `40`                   |  |
| `training.early_stopping_patience` | `15`                |  |
| `training.val_retrieval_frames` | `800`                  | Frames used for `val_R@1` early-stopping |
| `mining.d_pos_m`                | `5.0`                  | Max GPS distance for positives |
| `mining.d_neg_m`                | `25.0`                 | Min GPS distance for negatives |
| `mining.d_time_s`               | `60.0`                 | Min temporal gap for positives |
| `eval.tta`                      | `true`                 | Flip-TTA |
| `eval.alpha_qe_k`               | `5`                    | α-QE top-k (0 disables) |
| `eval.alpha_qe_pow`             | `3.0`                  | α-QE exponent |
| `eval.dual_softmax`             | `true`                 | Dual-softmax rerank |
| `eval.dual_softmax_temp`        | `1.00`                 | DSM temperature |
| `eval.whiten`                   | `false`                | PCA whitening (disabled — hurts here) |
| `eval.seq_len`                  | `1`                    | Sequence-averaging window |
| `eval.ranks`                    | `[1, 5, 10]`           | Recall@K values to report |

## Evaluation metrics

- **AUC-PR** — Area under the Precision-Recall curve over all test pairs.
- **F1-max** — Peak F1 across similarity thresholds (threshold-based).
- **mAP** — Mean Average Precision (averaged over queries).
- **Recall@K** — Fraction of queries whose top-K retrievals include at least
  one true loop closure. K ∈ {1, 5, 10}.

Ground truth: pair `(i, j)` is a match iff the GPS distance is `< d_pos_m`
**and** the time gap is `> d_time_s` (so we don't count near-adjacent frames
as loop closures).

## Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA
- See `requirements.txt` for the full dependency list.

## Presentation assets

`presentation/summary.md` is the one-pager. Key figures:

- `presentation/figures/pr_curves_main.png`         — PR-curve progression across configs
- `presentation/figures/ablation_bars.png`          — headline gain-attribution chart
- `presentation/figures/dsm_temperature_sweep.png`  — DSM hyperparameter sensitivity
- `presentation/figures/modality_bars.png`          — per-modality breakdown for best config
- `presentation/figures/training_curves_pk_v2.png`  — PK-v2 training dynamics
- `presentation/figures/legacy/…`                   — historical Round-1 figures only

## Authors

NC A&T State University — COMP 841
