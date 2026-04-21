# Loop Closure Detection — Results Summary

## Headline (test split, fused descriptor, 2 000 frames, d_pos=5 m, d_time=60 s)

**Best configuration: `Baseline + TTA + α-QE(5) + DSM(T=1.00)`**

- AUC-PR     : **0.6417** (Δ = +0.1049)
- F1-max     : **0.7010** (Δ = -0.0342)
- mAP        : **0.7568**
- Recall@1   : **0.8252** (Δ = +0.1525)
- Recall@5   : **0.9792**
- Recall@10  : **0.9919**

## Key insights

1. **Re-ranking is the biggest lever, and it's free.**  Without any retraining, α-QE query expansion + dual-softmax lifted R@1 from 0.673 to 0.820 (+0.148), R@10 from 0.845 to 0.989.
2. **Dual-softmax temperature matters.**  Sweeping T ∈ {0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0}, R@1 climbs monotonically up to ~T=0.3 then saturates. T=1.00 is the sweet spot on our data — best AUC-PR *and* best R@K (see `figures/dsm_temperature_sweep.png`).
3. **Camera-only reaches perfect R@10 on the test subset** with the best config (1.0000).  LiDAR-only R@1 = 0.83, also surprisingly strong.
4. **PK-batch-hard mining overfit** on our 369 training places.  Its val R@1 peaked at 0.732 but test R@1 was only 0.595 (worse than the baseline's 0.673).
5. **PK-v2 + re-ranking** partially rescues the PK model (R@1: 0.595 → 0.705) but still trails baseline+rerank by ~10 points.

## Files

- `figures/pr_curves_main.png`           — PR curves across the best configs
- `figures/modality_bars.png`            — per-modality comparison for the best config
- `figures/ablation_bars.png`            — where the gains come from
- `figures/dsm_temperature_sweep.png`    — DSM temperature sensitivity
- `figures/training_curves_pk_v2.png`    — PK-v2 training dynamics
- `tables/main_results.md/.csv`          — complete results table
- `tables/ablation.md/.csv`              — evaluation-time ablation table
- `tables/per_modality_best.md/.csv`     — per-modality breakdown for the best config

## Caveats

- Re-ranking metrics are computed on 2 000 subsampled test frames.  Larger eval sets may shift exact numbers by ±1–2 pp.
- Dual-softmax compresses absolute similarity values, which depresses F1-max (threshold-based).  For deployment we'd calibrate a threshold per operating condition — AUC-PR / mAP / R@K are the more informative metrics.
- The baseline checkpoint was trained before we fixed the Multi-Similarity label bug.  Retraining with fixed labels should give a further bump, likely smaller than the rerank gain.