# Implementation Plan: Multi-Modal Loop Closure Detection

**Project**: Deep Learning-Based Loop Closure Detection for Robust 3D LiDAR SLAM  
**Team**: Andrews Tang, Abhinav Pendem, Elijah Love  
**Last Updated**: March 8, 2026

---

## Overview

This document describes the step-by-step engineering plan to build, train, and evaluate a **Two-Stream Siamese CNN** for multi-modal Loop Closure Detection (LCD). The system fuses LiDAR range images with synchronized RGB camera frames to produce robust location fingerprints.

**Golden Rule**: We do NOT train with live ROS data. All data has been pre-extracted to disk. We build, train, and evaluate in pure **PyTorch**. ROS is only revisited for final vehicle deployment.

### Current Dataset (Trimmed for Experimentation)

| Asset               | Count       | Format / Details                                      |
|---------------------|-------------|-------------------------------------------------------|
| LiDAR scans         | 15,001      | `.bin` — float32 × 5 fields: `x, y, z, intensity, ring` |
| Camera frames       | 45,004      | `.jpg` — 1920×1280 RGB                                |
| GPS ground truth    | 75,009 rows | `inspva_precision.csv` — lat/lon/alt + roll/pitch/azimuth |
| Odometry            | 69,009 rows | `novatel_odom.csv` — UTM position (meters) + quaternion  |
| Drive duration      | ~25 min     | 3–4 revisits of the same campus route                 |

> **Note**: This is a trimmed subset (~25 min) of the full 50+ minute dataset.  
> Once the pipeline is validated, we swap in the full dataset with zero code changes.

### Directory Layout (Current → Target)

```
loop-closure-slam/
├── dataset/                          # Raw extracted data (already exists)
│   ├── lidar/pointclouds/            # 15,001 .bin files
│   ├── camera/images/                # 45,004 .jpg files
│   ├── gps/                          # GPS CSVs (inspva, bestpos, fixes)
│   ├── odometry/                     # novatel_odom.csv (UTM coords)
│   ├── imu/                          # IMU data
│   ├── calibration/                  # TF transforms, camera intrinsics
│   └── timestamps.csv               # 129K rows, all topic timestamps
│
├── processed/                        # OUTPUT of Phase 1 (generated)
│   ├── range_images/                 # 15,001 .npy files (128×1024)
│   ├── camera_synced/                # 15,001 .jpg files (matched to LiDAR)
│   ├── sync_pairs.csv               # LiDAR↔Camera pairing manifest
│   ├── trajectory.png               # GPS trajectory visualization
│   └── triplets.csv                  # Anchor/Positive/Negative index sets
│
├── src/                              # All source code
│   ├── step1_sync_and_project.py     # Phase 1A: Time sync + range images
│   ├── step2_visualize_trajectory.py # Phase 1B: Plot GPS + find loops
│   ├── step3_mine_triplets.py        # Phase 1C: Build triplets.csv
│   ├── step4_dataset.py              # Phase 2A: PyTorch Dataset class
│   ├── step5_model.py                # Phase 2B: Two-Stream Siamese CNN
│   ├── step6_train.py                # Phase 3:  Training loop
│   └── step7_evaluate.py            # Phase 4:  PR curves + ablation
│
├── configs/
│   └── config.yaml                   # All hyperparameters in one place
│
├── IMPLEMENTATION_PLAN.md            # This file
└── requirements.txt                  # Python dependencies
```

---

## Phase 1 — Data Engineering & Synchronization

Everything downstream depends on this phase. The goal is to turn the raw extracted data into a clean, indexed, GPU-ready dataset.

---

### Step 1A: Time Synchronization & Range Image Projection

**Script**: `src/step1_sync_and_project.py`  
**Input**: `dataset/lidar/pointclouds/*.bin`, `dataset/camera/images/*.jpg`  
**Output**: `processed/range_images/*.npy`, `processed/camera_synced/*.jpg`, `processed/sync_pairs.csv`

#### 1A.1 — Parse Timestamps from Filenames

Both the LiDAR and camera files embed their ROS timestamps in the filename:

```
cloud_000134_1771266607_329643680.bin   →  timestamp = 1771266607.329643680
frame_000000_1771266593_867975609.jpg   →  timestamp = 1771266593.867975609
```

Extract all timestamps into two sorted arrays: `lidar_times[15001]` and `camera_times[45004]`.

#### 1A.2 — Nearest-Neighbor Time Sync

For each LiDAR timestamp, find the camera frame with the closest timestamp. Since the camera runs at 3× the LiDAR rate, the maximum sync error is ~16 ms (half of 1/30 Hz).

```
For each lidar_ts in lidar_times:
    idx = argmin(|camera_times - lidar_ts|)
    sync_error = |camera_times[idx] - lidar_ts|
    if sync_error > MAX_SYNC_TOLERANCE (e.g. 50ms):
        discard this pair
    else:
        record (lidar_file, camera_file, sync_error)
```

Save the pairing manifest as `processed/sync_pairs.csv`:
```
frame_idx, lidar_file, camera_file, lidar_ts, camera_ts, sync_error_ms
0, cloud_000000_....bin, frame_000002_....jpg, ..., ..., 3.2
1, cloud_000001_....bin, frame_000005_....jpg, ..., ..., 1.8
...
```

#### 1A.3 — Spherical Range Image Projection

Each `.bin` file contains ~145K unstructured 3D points with fields `(x, y, z, intensity, ring)`. We project these into a dense **128 × 1024** range image.

**Algorithm per point** `(x, y, z, intensity, ring)`:

```
range     = sqrt(x² + y² + z²)
azimuth   = atan2(y, x)                          # horizontal angle, [-π, π]
row       = ring                                  # 0–127 (beam index)
col       = floor((0.5 * (azimuth / π) + 0.5) * 1024) % 1024
image[row, col] = range
```

- The `ring` field (0–127) maps directly to the row, eliminating the need to compute elevation angles.
- The azimuth wraps around 360°, quantized into 1024 columns.
- Save each range image as `processed/range_images/range_XXXXXX.npy` (float32, 128×1024).

#### 1A.4 — Copy Synced Camera Frames

Symlink or copy the matched camera frame into `processed/camera_synced/` with matching index names:
```
processed/camera_synced/cam_000000.jpg  (matched to range_000000.npy)
processed/camera_synced/cam_000001.jpg  (matched to range_000001.npy)
...
```

#### Sanity Checks

- Visualize 5–10 range images as heatmaps. They should look like a panoramic depth map of the campus.
- Confirm sync_pairs.csv has ~15,000 valid pairs with sync errors < 50 ms.
- Spot-check a few camera/range pairs to make sure they're from the same moment.

---

### Step 1B: GPS Trajectory Visualization

**Script**: `src/step2_visualize_trajectory.py`  
**Input**: `dataset/odometry/novatel_odom.csv`  
**Output**: `processed/trajectory.png`, printed loop-crossing statistics

#### What It Does

1. Load the UTM `position_x, position_y` columns from `novatel_odom.csv`.
2. Subsample to match LiDAR frame rate (one pose per LiDAR scan) using the same nearest-neighbor timestamp matching.
3. Plot the 2D trajectory, color-coded by time.
4. Identify loop regions: locations where the vehicle was within **D_pos** meters of a previous location at least **D_time** seconds earlier.

#### Why It Matters

This visualization tells us:
- How many usable loop closures exist in the trimmed dataset.
- Where the loops physically occur on campus.
- Whether the route geometry supports enough positive pairs for training.

---

### Step 1C: Triplet Mining

**Script**: `src/step3_mine_triplets.py`  
**Input**: `processed/sync_pairs.csv`, `dataset/odometry/novatel_odom.csv`  
**Output**: `processed/triplets.csv`

#### 1C.1 — Assign a UTM Pose to Every Synced Frame

For each synced frame index, find the closest odometry pose (by timestamp). Result: an array `poses[N, 2]` of (x, y) UTM positions, one per synced frame.

#### 1C.2 — Build a Distance Matrix

Compute the pairwise Euclidean distance between all poses. For 15K frames, the full matrix is ~900M entries — too large for RAM. Instead, use a KD-tree:

```python
from scipy.spatial import KDTree
tree = KDTree(poses)
```

#### 1C.3 — Mine Triplets

For each anchor frame `i`:

- **Positive candidates**: frames `j` where `distance(i, j) < D_pos` (e.g., 5m) AND `|time(i) - time(j)| > D_time` (e.g., 60s). The time gap ensures the positive is from a *different traversal*, not the next frame.
- **Negative candidates**: frames `k` where `distance(i, k) > D_neg` (e.g., 25m).
- Randomly sample one positive and one negative per anchor (or use hard-negative mining later).

#### 1C.4 — Output Format

Save `processed/triplets.csv`:

```
anchor_idx, positive_idx, negative_idx
0, 8742, 3201
1, 8743, 3450
...
```

#### Key Hyperparameters (Configurable)

| Parameter   | Symbol   | Default | Meaning                                       |
|-------------|----------|---------|-----------------------------------------------|
| Pos radius  | `D_pos`  | 5 m     | Max distance to count as "same place"         |
| Neg radius  | `D_neg`  | 25 m    | Min distance to count as "different place"    |
| Time gap    | `D_time` | 60 s    | Min temporal separation for positive pairs    |

> **Expect**: With 3–4 revisits of the same route, we should get several thousand valid triplets from the 15K frames.

---

## Phase 2 — Network Architecture

Build the model in pure PyTorch. Start with single-frame inference (no LSTM). The temporal extension comes later as an ablation.

---

### Step 2A: PyTorch Dataset & DataLoader

**Script**: `src/step4_dataset.py`

#### The `TripletDataset` Class

```
__init__(triplets_csv, range_dir, camera_dir, transform):
    Load triplets.csv into memory.

__getitem__(idx):
    row = triplets[idx]  →  (anchor_idx, positive_idx, negative_idx)

    For each of [anchor, positive, negative]:
        Load range_XXXXXX.npy  →  normalize to [0, 1]  →  tensor (1, 128, 1024)
        Load cam_XXXXXX.jpg    →  resize to (224, 224)  →  tensor (3, 224, 224)

    Return: (anchor_lidar, anchor_cam), (pos_lidar, pos_cam), (neg_lidar, neg_cam)

__len__():
    return len(triplets)
```

#### Data Augmentation (Applied During Training)

| Augmentation         | Applied To | Purpose                                      |
|----------------------|------------|----------------------------------------------|
| Random horizontal flip | Both    | Simulate reverse traversal                   |
| Random noise injection | LiDAR   | Simulate rain / degraded returns             |
| Color jitter + blur  | Camera     | Simulate fog, glare, lighting changes        |
| Random dropout (zero-out) | Either | Force cross-modal reliance (key for fusion) |

The **random modality dropout** is critical: during training, occasionally zero out the entire camera input or the entire LiDAR input. This forces the fusion layer to learn to rely on whichever sensor survives.

#### Train/Val/Test Split Strategy

Split **by time**, not randomly, to prevent data leakage (adjacent frames are nearly identical):

```
|---- Train (first 70% of frames) ----|-- Val (15%) --|-- Test (15%) --|
```

Triplets that cross split boundaries are discarded.

---

### Step 2B: Two-Stream Siamese CNN

**Script**: `src/step5_model.py`

#### Architecture Diagram

```
                    ┌─────────────────────────┐
                    │      Input Frame i       │
                    └────────┬────────┬────────┘
                             │        │
                    ┌────────▼──┐  ┌──▼────────┐
                    │  LiDAR    │  │  Camera    │
                    │  Range Img│  │  RGB Image │
                    │ (1,128,1024) │ (3,224,224)│
                    └────────┬──┘  └──┬────────┘
                             │        │
                    ┌────────▼──┐  ┌──▼────────┐
                    │ ResNet-18 │  │ ResNet-18  │
                    │ (modified │  │ (pretrained│
                    │  1-ch in) │  │  3-ch in)  │
                    └────────┬──┘  └──┬────────┘
                             │        │
                         [512-d]  [512-d]
                             │        │
                    ┌────────▼────────▼────────┐
                    │     Concatenate (1024)    │
                    └────────────┬──────────────┘
                                │
                    ┌───────────▼───────────┐
                    │   Fusion MLP          │
                    │   1024 → 512 → 256    │
                    │   (ReLU + BatchNorm)  │
                    └───────────┬───────────┘
                                │
                          [256-d descriptor]
                       "Location Fingerprint"
```

#### Key Design Decisions

1. **LiDAR backbone**: Take a pretrained ResNet-18, replace the first `conv1` layer from `in_channels=3` to `in_channels=1`. Initialize the new conv layer with the mean of the pretrained RGB weights. Freeze early layers initially.

2. **Camera backbone**: Standard pretrained ResNet-18 on ImageNet. Remove the final classification head, use the 512-d feature vector from the average pooling layer.

3. **Fusion MLP**: `Linear(1024→512) → BN → ReLU → Linear(512→256) → L2-Normalize`. The output is a unit-length 256-dimensional vector.

4. **Weight sharing**: The three Siamese branches (Anchor, Positive, Negative) share **identical** weights. In code, this means we define ONE encoder and call it three times.

```python
class TwoStreamEncoder(nn.Module):
    def __init__(self):
        self.lidar_backbone = modified_resnet18(in_channels=1)
        self.camera_backbone = resnet18(pretrained=True)
        self.fusion = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 256)
        )

    def forward(self, lidar_img, camera_img):
        l_feat = self.lidar_backbone(lidar_img)   # (B, 512)
        c_feat = self.camera_backbone(camera_img)  # (B, 512)
        fused  = torch.cat([l_feat, c_feat], dim=1) # (B, 1024)
        desc   = self.fusion(fused)                  # (B, 256)
        return F.normalize(desc, p=2, dim=1)         # unit vector
```

---

## Phase 3 — Training

**Script**: `src/step6_train.py`

### Loss Function: Triplet Margin Loss

```
L = max( d(A, P) − d(A, N) + α,  0 )
```

Where:
- `d(A, P)` = Euclidean distance between anchor and positive descriptors
- `d(A, N)` = Euclidean distance between anchor and negative descriptors  
- `α` = margin (default: 0.2)

PyTorch provides this directly: `nn.TripletMarginLoss(margin=0.2, p=2)`.

### Training Hyperparameters

| Parameter       | Value      | Rationale                                      |
|-----------------|------------|------------------------------------------------|
| Batch size      | 32         | Each sample is a triplet (3× forward pass)     |
| Learning rate   | 1e-4       | Adam optimizer with weight decay               |
| Epochs          | 50         | With early stopping on validation loss          |
| Margin (α)      | 0.2        | Standard for metric learning                   |
| Embedding dim   | 256        | Balance between expressiveness and speed        |
| LR scheduler    | ReduceOnPlateau | Halve LR when val loss plateaus            |

### Training Loop (Pseudocode)

```
for epoch in range(num_epochs):
    for (anchor, positive, negative) in train_loader:
        a_lidar, a_cam = anchor
        p_lidar, p_cam = positive
        n_lidar, n_cam = negative

        desc_a = encoder(a_lidar, a_cam)
        desc_p = encoder(p_lidar, p_cam)
        desc_n = encoder(n_lidar, n_cam)

        loss = triplet_loss(desc_a, desc_p, desc_n)
        loss.backward()
        optimizer.step()

    # Validate
    val_loss = evaluate(val_loader)
    scheduler.step(val_loss)
    if val_loss < best_val_loss:
        save_checkpoint("best_model.pt")
```

### Adverse Weather Augmentation (Phase 3 Enhancement)

During training, apply stochastic sensor degradation:

- **Camera dropout** (p=0.15): Replace the entire camera tensor with zeros (simulates complete camera failure / heavy fog).
- **LiDAR noise** (p=0.15): Add Gaussian noise to the range image and randomly zero-out patches (simulates rain scatter).
- **Both active** (p=0.70): Normal operation, both modalities clean.

This forces the fusion MLP to learn a fallback strategy rather than over-relying on one modality.

---

## Phase 4 — Evaluation & Visualization

**Script**: `src/step7_evaluate.py`

### Step 4A: Precision-Recall Curves

1. Run the trained encoder on every frame in the **test split**.
2. For each test frame (query), compute cosine similarity against all other test frames.
3. Ground truth: two frames are a "true match" if their GPS distance < `D_pos` meters.
4. Sweep the similarity threshold from 0→1, computing precision and recall at each point.

**Generate three curves on one plot**:
- **(a) Fused model** (LiDAR + Camera) — our full system
- **(b) LiDAR-only** — zero out the camera input at inference
- **(c) Camera-only** — zero out the LiDAR input at inference

This directly demonstrates the value of sensor fusion.

### Step 4B: Ablation — Sequence vs. Single Frame

*(Deferred until the LSTM extension is implemented)*

After adding the temporal LSTM layer, repeat the PR evaluation and compare:
- Single frame (T=1) vs. sequence (T=5)
- Report the reduction in false positive rate.

### Key Metrics to Report

| Metric                     | What It Measures                            |
|----------------------------|---------------------------------------------|
| **Recall@1**               | % of queries where the top match is correct |
| **Recall@1%**              | Recall within the top 1% of candidates      |
| **F1-max**                 | Peak F1 score on the PR curve               |
| **AUC-PR**                 | Area under the Precision-Recall curve       |

---

## Phase 5 — Temporal Extension (Deferred)

Once Phase 1–4 produces a working single-frame baseline, we upgrade to the spatio-temporal architecture.

### Changes Required

1. **Dataset**: Instead of loading a single `(lidar, camera)` pair, load a window of `T=5` consecutive synced pairs.
2. **Model**: After the fusion MLP produces a 256-d vector **per frame**, stack T=5 vectors into a sequence and pass through a single-layer LSTM. The final hidden state becomes the new 256-d descriptor.
3. **Triplets**: Anchors, positives, and negatives each become length-5 sequences. Triplet mining must ensure all 5 frames in a positive sequence remain within `D_pos` of the anchor sequence.

---

## Phase 6 — Live ROS 2 Deployment (Stretch Goal)

Write a ROS 2 node (`loop_closure_node.py`) that:
1. Subscribes to `/ouster/points` and `/camera_2/.../compressed`.
2. Buffers incoming frames into a sliding window.
3. Projects LiDAR to range image on-the-fly.
4. Runs the trained `.pt` model to generate a descriptor.
5. Compares against a database of past descriptors.
6. Publishes a `PoseGraphConstraint` message if similarity exceeds the threshold.

---

## Execution Order & Dependencies

```
Step 1A ──→ Step 1B ──→ Step 1C ──→ Step 2A ──→ Step 2B ──→ Step 3 ──→ Step 4
(sync +      (plot       (mine       (dataset     (model)     (train)    (eval)
 project)    trajectory)  triplets)   class)

              Step 1B can run in parallel with 1A
              (it only needs odometry, not range images)
```

## Dependencies

```
torch >= 2.0
torchvision
numpy
scipy
pandas
matplotlib
opencv-python
PyYAML
tqdm
scikit-learn
```

---

## How to Swap in the Full Dataset

When ready to train on the full 50+ minute drive:

1. Extract the full bag using the same extraction tool into a new `dataset_full/` directory.
2. Update the paths in `configs/config.yaml` to point to `dataset_full/`.
3. Re-run Steps 1A → 1C. The scripts are parameterized — no code changes needed.
4. Re-run Step 3 (training). Expect more triplets and better generalization.
