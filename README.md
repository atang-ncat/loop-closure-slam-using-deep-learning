# Loop Closure Detection Using Multi-Modal Deep Learning

A deep learning pipeline for loop closure detection in autonomous navigation, fusing LiDAR range images and RGB camera data into compact location descriptors using metric learning.

## Overview

This project trains a Two-Stream Siamese CNN to produce location fingerprints for visual place recognition. The system detects when a robot revisits a previously seen location by comparing learned descriptors in embedding space.

**Architecture:**
- Two-Stream encoder with ResNet-50 backbones and Generalized Mean (GeM) pooling
- LiDAR stream processes 128x1024 range images (Ouster OS1-128)
- Camera stream processes 224x224 RGB images
- Fusion MLP compresses to 512-d L2-normalized descriptors
- 51.7M trainable parameters

**Training:**
- Multi-Similarity Loss with automatic hard pair mining
- GPS-supervised triplet mining with semi-hard negatives (10-30m band)
- Modality dropout for single-sensor robustness
- CosineAnnealingWarmRestarts scheduler, AdamW optimizer
- Mixed precision training (AMP) with gradient accumulation

## Dataset

Proprietary 500+ GB dataset collected on the NC A&T State University campus:
- Ouster OS1-128 LiDAR (10 Hz)
- Forward-facing USB RGB camera (30 Hz)
- NovAtel SPAN GNSS/INS ground truth (centimeter-level GPS)

Frames are time-synchronized using an Approximate Time Synchronizer with a 50ms tolerance.

## Project Structure

```
loop-closure-slam/
├── configs/
│   └── config.yaml            # All hyperparameters and paths
├── src/
│   ├── step1_sync_sensors.py   # Time synchronization
│   ├── step1a_project_lidar.py # Point cloud to range image projection
│   ├── step2_visualize_trajectory.py
│   ├── step3_mine_triplets.py  # GPS-supervised triplet mining
│   ├── step4_dataset.py        # Triplet dataset and dataloaders
│   ├── step4b_frame_dataset.py # Frame-level dataset for hard mining
│   ├── step5_model.py          # Two-Stream Siamese Network
│   ├── step6_train.py          # Training loop with MS-loss
│   └── step7_evaluate.py       # PR curves and ablation study
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

Requires Docker and nvidia-container-toolkit.

```bash
docker compose build
```

## Usage

### 1. Pre-process data

```bash
python3 -m src.step1_sync_sensors --config configs/config.yaml
python3 -m src.step1a_project_lidar --config configs/config.yaml
python3 -m src.step2_visualize_trajectory --config configs/config.yaml
```

### 2. Mine triplets

```bash
python3 -m src.step3_mine_triplets --config configs/config.yaml
```

### 3. Train

```bash
python3 -m src.step6_train --config configs/config.yaml

# Resume from checkpoint
python3 -m src.step6_train --config configs/config.yaml --resume checkpoints/last.pt
```

### 4. Evaluate

```bash
python3 -m src.step7_evaluate --config configs/config.yaml --checkpoint checkpoints/best.pt
```

### Docker commands

```bash
docker compose run mine
docker compose run train
docker compose run evaluate
```

## Configuration

All hyperparameters are in `configs/config.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.backbone` | resnet50 | ResNet-18 or ResNet-50 |
| `model.embedding_dim` | 512 | Descriptor dimensionality |
| `model.gem_p` | 3.0 | GeM pooling initial power |
| `training.loss` | multi_similarity | Loss function (triplet or multi_similarity) |
| `training.batch_size` | 12 | Samples per batch |
| `training.learning_rate` | 1e-4 | Initial learning rate |
| `mining.d_pos_m` | 5.0 | Max distance for same place (m) |
| `mining.d_neg_m` | 10.0 | Min distance for different place (m) |

## Evaluation Metrics

The model is evaluated on held-out test frames using pairwise descriptor similarity:
- **AUC-PR** (Area Under Precision-Recall Curve)
- **F1-max** (Maximum F1 score across thresholds)
- **Recall@1** (Top-1 nearest neighbor retrieval accuracy)

Three ablation modes are evaluated: Fused (both sensors), LiDAR-only, and Camera-only.

## Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA
- See `requirements.txt` for full dependencies

## Authors

NC A&T State University -- COMP 841
