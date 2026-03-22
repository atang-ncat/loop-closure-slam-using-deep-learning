# Deep Learning-Based Loop Closure Detection for Robust 3D LiDAR SLAM
## Implementation Progress Presentation
### Andrews Tang, Abhinav Pendem, Elijah Love
### COMP 841 — North Carolina A&T State University

---

# Slide 1: The Problem — Odometry Drift

**Why does SLAM fail?**

Every sensor measurement contains noise. When a robot chains thousands of
relative pose estimates together, these tiny errors **compound unboundedly**.

- A fraction of a degree in heading → meters of error over distance
- After 5 km of driving, the estimated path **completely diverges** from reality
- The only fix: **Loop Closure Detection (LCD)** — recognizing a previously visited place to "snap" the trajectory back

**Our goal**: Train a deep neural network to generate robust **location fingerprints** that identify the same place across different traversals, using fused LiDAR + Camera data.

> *Figure reference: Use drift diagrams from the paper (drifta.png, driftb.jpg)*

---

# Slide 2: System Architecture Overview

**Two-Stream Siamese CNN with Triplet Margin Loss**

```
  LiDAR Range Image (1×128×1024)          RGB Camera (3×224×224)
          │                                        │
    ┌─────▼─────┐                          ┌───────▼──────┐
    │ ResNet-18  │                          │  ResNet-18   │
    │ (1-ch mod) │                          │ (pretrained) │
    └─────┬─────┘                          └───────┬──────┘
          │ 512-d                                  │ 512-d
          └──────────────┬─────────────────────────┘
                    Concatenate (1024-d)
                         │
                  ┌──────▼──────┐
                  │  Fusion MLP │
                  │ 1024→512→256│
                  └──────┬──────┘
                         │
                  256-d L2-normalized
                  "Location Fingerprint"
```

- **23M parameters** total (two ResNet-18 backbones + fusion MLP)
- Trained with **Triplet Margin Loss**: pull matching locations together, push different locations apart
- **Modality dropout** (15%): randomly zero out one sensor during training to force cross-modal resilience

> *Figure reference: Use triplet-architecture.png from the paper*

---

# Slide 3: Dataset — NCAT Campus Collection

**Proprietary dataset collected on the NC A&T campus**

| Sensor             | Spec                           | Data Collected        |
|--------------------|--------------------------------|-----------------------|
| Ouster OS1-128     | 128-beam LiDAR @ 10 Hz        | 15,001 point clouds   |
| USB RGB Camera     | 1920×1280 @ 30 Hz             | 45,004 images         |
| NovAtel SPAN       | RTK GPS/INS (cm-level)        | 69,009 odometry poses |

**Route statistics (trimmed subset for experimentation):**
- Distance: **5.14 km** over **25 minutes** at ~12 km/h
- **3–4 revisits** of the same campus loop
- 177,573 loop-closure pair candidates detected by GPS

> *Figure: Show `processed/trajectory.png` — left panel (trajectory colored by time) and right panel (loop regions in red)*

---

# Slide 4: Phase 1A — Time Synchronization & Range Image Projection

**Challenge**: Camera runs at 30 Hz, LiDAR at 10 Hz — cannot naively pair them.

**Solution** (`step1_sync_and_project.py`):

1. **Timestamp extraction**: Both file types embed ROS timestamps in filenames
   ```
   cloud_000134_1771266607_329643680.bin  →  t = 1771266607.330s
   frame_000200_1771266607_335000000.jpg  →  t = 1771266607.335s
   ```
2. **Nearest-neighbor sync**: For each LiDAR scan, find the closest camera frame
   - Result: **15,001 synced pairs**, mean sync error = **8.28 ms**

3. **Spherical projection**: Convert unstructured 3D point clouds (~145K points) into dense **128×1024 range images** using the `ring` field (beam index) and azimuth angle

**Output**: Perfectly aligned (range_image, camera_frame) pairs on disk

> *Figure: Show `processed/sample_pairs.png` — 5 examples of range image heatmaps alongside their matched camera photos*

---

# Slide 5: Phase 1B–C — Trajectory Analysis & Triplet Mining

**Step 1B** (`step2_visualize_trajectory.py`):
- Assign a UTM (x,y) pose to every synced frame from NovAtel odometry
- Identify loop closure regions using spatial + temporal thresholds

**Step 1C** (`step3_mine_triplets.py`) — the "Teacher" (GPS) creates training labels:

For each frame (Anchor), find:
- **Positive**: Another frame < **5m** away but > **60s** later (different traversal, same place)
- **Negative**: A frame > **25m** away (different place entirely)

| Parameter      | Value  | Purpose                              |
|----------------|--------|--------------------------------------|
| d_pos          | 5 m    | "Same place" radius                  |
| d_neg          | 25 m   | "Different place" minimum distance   |
| d_time         | 60 s   | Ensures positive is a re-visit       |

**Result**: **38,971 triplets** split 70/15/15 by time:
- Train: 22,807 | Val: 9,912 | Test: 6,252

---

# Slide 6: Phase 2 — Network Architecture Details

**LiDAR Backbone** (modified ResNet-18):
- Replace `conv1` from 3→1 input channel
- Initialize with mean of pretrained RGB weights (preserves learned edge features)
- Input: `(B, 1, 128, 1024)` → Output: `(B, 512)`

**Camera Backbone** (standard ResNet-18):
- ImageNet-pretrained, remove classification head
- Input: `(B, 3, 224, 224)` → Output: `(B, 512)`

**Fusion MLP**:
- `Linear(1024→512) → BatchNorm → ReLU → Dropout(0.1) → Linear(512→256) → L2-Normalize`
- Output: **256-d unit vector** — the location fingerprint

**Key design: Weight sharing**
- Anchor, Positive, and Negative pass through the **same** encoder (Siamese)
- One model, called three times — ensures consistent representation

---

# Slide 7: Phase 2A — Data Pipeline & Augmentation

**PyTorch Dataset** (`step4_dataset.py`) loads triplets and applies augmentation:

| Augmentation             | Sensor  | Simulates                    |
|--------------------------|---------|------------------------------|
| Gaussian noise + patches | LiDAR   | Rain scatter, sensor noise   |
| Color jitter + blur      | Camera  | Fog, glare, lighting changes |
| Horizontal flip          | Both    | Reverse traversal direction  |
| **Modality dropout (15%)** | Either | **Complete sensor failure**  |

**Modality dropout is critical**: During training, 15% of the time the camera is zeroed out, and 15% the LiDAR is zeroed out. This forces the fusion MLP to learn a **fallback strategy** — relying on whichever sensor survives.

**Train/Val/Test split by time** (not random) to prevent data leakage from adjacent frames.

---

# Slide 8: Phase 3 — Training with Triplet Margin Loss

**Loss function:**
```
L = max( d(Anchor, Positive) − d(Anchor, Negative) + margin,  0 )
```
Minimizes distance between same-place pairs while maximizing distance between different-place pairs.

**Round 1 (Random Negatives, margin=0.2):**
- Loss collapsed to **0.0** by epoch 3 — negatives were trivially different (avg 218m away)
- Model learned "parking lot ≠ road" but not fine-grained place discrimination

**Round 2 (Hard Negatives, margin=1.0):**
- Mined negatives using the Round 1 model's own embeddings
- Loss started at **0.14** and converged meaningfully over 15 epochs
- Embedding separation improved: d_pos=0.14, d_neg=1.45

| Hyperparameter       | Value      |
|----------------------|------------|
| Optimizer            | Adam       |
| Learning rate        | 1e-4 (→ halved on plateau) |
| Batch size           | 32         |
| Margin               | 1.0        |
| Early stopping       | 10 epochs  |

> *Figure: Show `checkpoints/training_curves_round2.png` — loss convergence + embedding distance plot*

---

# Slide 9: Preliminary Results — Precision-Recall Evaluation

**Evaluation method**: Extract 256-d descriptors for 2,000 test frames, compute pairwise cosine similarity, sweep threshold to generate PR curves.

**Three configurations tested** (same trained model):
1. **Fused** — both sensors active
2. **LiDAR-only** — camera zeroed at inference
3. **Camera-only** — LiDAR zeroed at inference

| Metric     | Fused     | LiDAR-only | Camera-only |
|------------|-----------|------------|-------------|
| **AUC-PR** | **0.119** | 0.101      | 0.114       |
| **F1-max** | **0.257** | 0.198      | 0.248       |
| **R@1**    | **0.406** | 0.402      | 0.361       |

**Key takeaway**: Fused model outperforms both single-modality baselines on AUC-PR and F1, confirming that **sensor fusion adds value**. Recall@1 of 40.6% means the correct loop closure is the top retrieval 4 out of 10 times — far above random chance (0.26% positive rate).

> *Figure: Show `processed/pr_curves.png` — three-curve comparison plot*

---

# Slide 10: Current & Next Steps

**In progress — Round 3: Top-K Hard Negative Mining**
- Previous hard mining used too-wide a selection window (sim 0.0–0.98)
- New strategy: directly pick the **20 most confusing** negatives per anchor
- Expected: neg similarity jumps from 0.19 → 0.7+, forcing true feature learning

**Upcoming milestones:**

| Phase | Task                                       | Status        |
|-------|--------------------------------------------|---------------|
| 3b    | Top-K hard negative mining + retrain       | **In progress** |
| 4.2   | LSTM temporal extension (T=5 frame window) | Planned       |
| 4.2   | Sequence vs. single-frame ablation study   | Planned       |
| 5     | Full 50-minute dataset training            | Planned       |
| 6     | Live ROS 2 node deployment on NCAT vehicle | Stretch goal  |

**The pipeline is fully modular** — swapping in the full dataset or adding the LSTM layer requires config changes and minimal code modifications, not a rewrite.

**Code**: [github.com/atang-ncat/loop-closure-slam-using-deep-learning](https://github.com/atang-ncat/loop-closure-slam-using-deep-learning)
