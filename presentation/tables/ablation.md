**Baseline ablations: evaluation-time techniques**

| label | mode | auc_pr | f1_max | mAP | recall_at_1 | recall_at_5 | recall_at_10 |
|---|---|---|---|---|---|---|---|
| Baseline (raw) | fused | **0.5367** | 0.7351 | nan | 0.6727 | nan | nan |
| Baseline (raw) | lidar_only | 0.3877 | 0.6296 | nan | 0.6261 | nan | nan |
| Baseline (raw) | camera_only | 0.5316 | 0.7274 | nan | 0.6575 | nan | nan |
| Baseline + TTA | fused | 0.5348 | **0.7370** | **0.7356** | **0.6874** | 0.8166 | 0.8526 |
| Baseline + TTA | lidar_only | 0.3877 | 0.6296 | 0.6303 | 0.6261 | 0.7609 | 0.7913 |
| Baseline + TTA | camera_only | 0.5312 | 0.7298 | 0.7308 | 0.6641 | **0.8197** | **0.8551** |
| Baseline + whiten + TTA | fused | 0.5023 | 0.6864 | 0.7029 | 0.6312 | 0.7989 | 0.8430 |
| Baseline + whiten + TTA | lidar_only | 0.3699 | 0.6129 | 0.6034 | 0.6018 | 0.7503 | 0.7862 |
| Baseline + whiten + TTA | camera_only | 0.4998 | 0.6991 | 0.6985 | 0.6185 | 0.7893 | 0.8404 |
| Baseline + TTA + seq=5 | fused | 0.5178 | 0.7293 | 0.6984 | 0.6510 | 0.7903 | 0.8328 |
| Baseline + TTA + seq=5 | lidar_only | 0.4746 | 0.6903 | 0.6298 | 0.5481 | 0.7766 | 0.8323 |
| Baseline + TTA + seq=5 | camera_only | 0.5176 | 0.7241 | 0.6935 | 0.6393 | 0.7898 | 0.8354 |
| Baseline + TTA + seq=9 | fused | 0.4898 | 0.6751 | 0.6107 | 0.5061 | 0.7639 | 0.8202 |
| Baseline + TTA + seq=9 | lidar_only | 0.4487 | 0.6545 | 0.5622 | 0.4068 | 0.7229 | 0.7994 |
| Baseline + TTA + seq=9 | camera_only | 0.4899 | 0.6831 | 0.6057 | 0.4873 | 0.7573 | 0.8207 |