**Test-set results (fused descriptor, 2 000 frames, d_pos=5 m, d_time=60 s)**

| group | label | auc_pr | f1_max | mAP | recall_at_1 | recall_at_5 | recall_at_10 |
|---|---|---|---|---|---|---|---|
| Baseline | Baseline + TTA | 0.5348 | **0.7370** | 0.7356 | 0.6874 | 0.8166 | 0.8526 |
| Baseline | Baseline (raw) | 0.5367 | 0.7351 | nan | 0.6727 | nan | nan |
| Baseline | Baseline + TTA + seq=5 | 0.5178 | 0.7293 | 0.6984 | 0.6510 | 0.7903 | 0.8328 |
| Baseline | Baseline + whiten + TTA | 0.5023 | 0.6864 | 0.7029 | 0.6312 | 0.7989 | 0.8430 |
| Baseline | Baseline + TTA + seq=9 | 0.4898 | 0.6751 | 0.6107 | 0.5061 | 0.7639 | 0.8202 |
| Baseline+Rerank | Baseline + TTA + α-QE(5) + DSM(T=1.00) | **0.6417** | 0.7010 | **0.7568** | **0.8252** | **0.9792** | **0.9919** |
| Baseline+Rerank | Baseline + TTA + α-QE(5) + DSM(T=0.50) | 0.6372 | 0.6965 | 0.7561 | 0.8232 | 0.9777 | 0.9899 |
| Baseline+Rerank | Baseline + TTA + α-QE(5) + DSM(T=0.30) | 0.6293 | 0.6894 | 0.7552 | 0.8202 | 0.9742 | 0.9889 |
| Baseline+Rerank | Baseline + TTA + α-QE(5) + DSM(T=0.20) | 0.6175 | 0.6795 | 0.7537 | 0.8181 | 0.9681 | 0.9878 |
| Baseline+Rerank | Baseline + TTA + α-QE(10) + DSM(T=0.10) | 0.5860 | 0.6571 | 0.7515 | 0.8166 | 0.9519 | 0.9772 |
| Baseline+Rerank | Baseline + TTA + α-QE(5) + DSM(T=0.15) | 0.6052 | 0.6696 | 0.7519 | 0.8156 | 0.9625 | 0.9863 |
| Baseline+Rerank | Baseline + TTA + α-QE(5) + DSM(T=0.10) | 0.5894 | 0.6590 | 0.7490 | 0.8024 | 0.9412 | 0.9661 |
| Baseline+Rerank | Baseline + TTA + α-QE(3) + DSM(T=0.05) | 0.6200 | 0.6920 | 0.7432 | 0.7801 | 0.9058 | 0.9362 |
| Baseline+Rerank | Baseline + TTA + α-QE(5) + DSM(T=0.05) | 0.6181 | 0.6880 | 0.7464 | 0.7751 | 0.9068 | 0.9362 |
| Baseline+Rerank | Baseline + TTA + α-QE(5) | 0.5480 | 0.7289 | 0.7477 | 0.7579 | 0.8815 | 0.9144 |
| Baseline+Rerank | Baseline + TTA + α-QE(10) | 0.5525 | 0.7292 | 0.7483 | 0.7538 | 0.8749 | 0.9189 |
| Baseline+Rerank | Baseline + DSM(T=0.05) | 0.6204 | 0.6993 | 0.7301 | 0.6677 | 0.8278 | 0.8637 |
| PK-v2 | PK-v2 + TTA | 0.3520 | 0.5139 | 0.6144 | 0.5947 | 0.7730 | 0.8116 |
| PK-v2 | PK-v2 (raw) | 0.3481 | 0.4957 | 0.6084 | 0.5861 | 0.7766 | 0.8151 |
| PK-v2+Rerank | PK-v2 + TTA + α-QE(5) + DSM(T=0.05) | 0.4617 | 0.5020 | 0.6347 | 0.7052 | 0.8425 | 0.8708 |