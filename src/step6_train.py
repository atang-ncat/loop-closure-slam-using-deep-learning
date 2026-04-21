"""
Step 3 — Training Loop with Multi-Similarity Loss
===================================================
Trains the Two-Stream Siamese CNN to produce location fingerprints
using metric learning with Multi-Similarity Loss.

Usage:
    python -m src.step6_train
    python -m src.step6_train --config configs/config.yaml
    python -m src.step6_train --config configs/config.yaml --resume checkpoints/last.pt
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.profiler
import yaml
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from tqdm import tqdm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from src.step4_dataset import build_dataloaders
from src.step4b_frame_dataset import build_hard_dataloaders
from src.step5_model import SiameseNetwork


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ── Multi-Similarity Loss ────────────────────────────────────────────────────

def _safe_logsumexp_with_one(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Per-row: log(1 + sum_{k in mask} exp(scores[k])). Returns (B,).

    Rows with empty mask return 0 (because log(1) = 0), contributing nothing
    to the loss.  Implemented by prepending a zero-column and using the
    numerically stable logsumexp.
    """
    scores = scores.masked_fill(~mask, float("-inf"))
    zeros = torch.zeros(scores.size(0), 1, device=scores.device, dtype=scores.dtype)
    scores_with_zero = torch.cat([zeros, scores], dim=1)
    return torch.logsumexp(scores_with_zero, dim=1)


class MultiSimilarityLoss(nn.Module):
    """
    Multi-Similarity Loss (Wang et al., CVPR 2019) — fully vectorized.

    Supports arbitrary integer place labels (no assumption that each label
    appears exactly twice, so this works with PK batches where each place
    contributes K samples).
    """

    def __init__(self, alpha: float = 2.0, beta: float = 50.0, base: float = 0.5, margin: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.base = base
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        embeddings: (B, D) L2-normalized
        labels: (B,) integer place labels (any integers; same int = same place)
        """
        sim_mat = embeddings @ embeddings.T  # (B, B), cosine since L2-normed

        B = embeddings.size(0)
        labels = labels.view(-1, 1)
        same = labels == labels.T
        eye = torch.eye(B, dtype=torch.bool, device=embeddings.device)
        pos_mask = same & ~eye
        neg_mask = ~same

        # Per-row easiest positive / hardest negative for hard-pair mining
        pos_sim_for_min = sim_mat.masked_fill(~pos_mask, float("inf"))
        pos_min, _ = pos_sim_for_min.min(dim=1, keepdim=True)  # (B, 1)
        neg_sim_for_max = sim_mat.masked_fill(~neg_mask, float("-inf"))
        neg_max, _ = neg_sim_for_max.max(dim=1, keepdim=True)  # (B, 1)

        # Hard-pair masks per MS-loss definition
        hard_pos_mask = pos_mask & (sim_mat - self.margin < neg_max)
        hard_neg_mask = neg_mask & (sim_mat + self.margin > pos_min)

        pos_arg = -self.alpha * (sim_mat - self.base)
        neg_arg = self.beta * (sim_mat - self.base)

        pos_term = _safe_logsumexp_with_one(pos_arg, hard_pos_mask) / self.alpha
        neg_term = _safe_logsumexp_with_one(neg_arg, hard_neg_mask) / self.beta

        # Count rows that actually have both a hard positive and a hard negative
        valid = hard_pos_mask.any(dim=1) & hard_neg_mask.any(dim=1)
        n_valid = valid.float().sum().clamp(min=1.0)

        loss = ((pos_term + neg_term) * valid.float()).sum() / n_valid
        return loss


# ── Training / Validation steps ───────────────────────────────────────────────

def train_one_epoch(
    model: SiameseNetwork,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler,
    accumulate_grad_batches: int,
    loss_type: str = "triplet",
    profiler=None,
) -> dict:
    model.train()
    running_loss = 0.0
    n_batches = 0
    last_d_pos = 0.0
    last_d_neg = 0.0

    for i, batch in enumerate(tqdm(loader, desc="  Train", leave=False)):
        if loss_type == "multi_similarity":
            # Unpack triplet batch and create a combined batch with labels
            (a_li, a_cam), (p_li, p_cam), (n_li, n_cam) = batch
            # Stack anchor + positive as "same place" (label 0..B-1)
            # and negative as "different place" (label B..2B-1)
            B = a_li.size(0)

            all_lidar = torch.cat([a_li, p_li, n_li], dim=0).to(device)
            all_camera = torch.cat([a_cam, p_cam, n_cam], dim=0).to(device)
            # Labels: anchor_i and positive_i share label i, negative_i gets label B+i
            labels = torch.cat([
                torch.arange(B),
                torch.arange(B),        # positive shares label with anchor
                torch.arange(B, 2 * B), # negative gets unique label
            ]).to(device)

            with torch.autocast(device_type=device.type, enabled=scaler is not None):
                embeddings = model.encoder(all_lidar, all_camera)
                loss = criterion(embeddings, labels)
                loss = loss / accumulate_grad_batches

            # Compute metrics from the embeddings
            with torch.no_grad():
                desc_a = embeddings[:B]
                desc_p = embeddings[B:2*B]
                desc_n = embeddings[2*B:]
                last_d_pos = (desc_a - desc_p).pow(2).sum(dim=1).sqrt().mean().item()
                last_d_neg = (desc_a - desc_n).pow(2).sum(dim=1).sqrt().mean().item()
        else:
            # Standard triplet loss
            (a_li, a_cam), (p_li, p_cam), (n_li, n_cam) = batch
            a_li, a_cam = a_li.to(device), a_cam.to(device)
            p_li, p_cam = p_li.to(device), p_cam.to(device)
            n_li, n_cam = n_li.to(device), n_cam.to(device)

            with torch.autocast(device_type=device.type, enabled=scaler is not None):
                desc_a, desc_p, desc_n = model(
                    (a_li, a_cam), (p_li, p_cam), (n_li, n_cam),
                )
                loss = criterion(desc_a, desc_p, desc_n)
                loss = loss / accumulate_grad_batches

            with torch.no_grad():
                last_d_pos = (desc_a - desc_p).pow(2).sum(dim=1).sqrt().mean().item()
                last_d_neg = (desc_a - desc_n).pow(2).sum(dim=1).sqrt().mean().item()

        if scaler is not None:
            scaler.scale(loss).backward()
            if (i + 1) % accumulate_grad_batches == 0 or (i + 1) == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            loss.backward()
            if (i + 1) % accumulate_grad_batches == 0 or (i + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * accumulate_grad_batches
        n_batches += 1

        # Step the profiler (if active)
        if profiler is not None:
            profiler.step()

    avg_loss = running_loss / max(n_batches, 1)
    return {"loss": avg_loss, "d_pos": last_d_pos, "d_neg": last_d_neg}


def train_one_epoch_pk(
    model: SiameseNetwork,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler,
    accumulate_grad_batches: int,
) -> dict:
    """
    PK-batch training: each batch is (lidar, camera, labels) where `labels`
    are real place IDs.  Works with MultiSimilarityLoss (label-aware).
    """
    model.train()
    running_loss = 0.0
    n_batches = 0
    last_d_pos = 0.0
    last_d_neg = 0.0

    for i, (lidar, camera, labels) in enumerate(tqdm(loader, desc="  Train (PK)", leave=False)):
        lidar = lidar.to(device, non_blocking=True)
        camera = camera.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            embeddings = model.encoder(lidar, camera)
            loss = criterion(embeddings, labels)
            loss = loss / accumulate_grad_batches

        # Monitoring: mean positive/negative cosine distance within batch
        with torch.no_grad():
            sim = embeddings @ embeddings.T
            labs = labels.view(-1, 1)
            same = (labs == labs.T) & ~torch.eye(
                labels.size(0), dtype=torch.bool, device=device
            )
            diff = ~same & ~torch.eye(
                labels.size(0), dtype=torch.bool, device=device
            )
            if same.any():
                last_d_pos = (1.0 - sim[same]).mean().item()
            if diff.any():
                last_d_neg = (1.0 - sim[diff]).mean().item()

        if scaler is not None:
            scaler.scale(loss).backward()
            if (i + 1) % accumulate_grad_batches == 0 or (i + 1) == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            loss.backward()
            if (i + 1) % accumulate_grad_batches == 0 or (i + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * accumulate_grad_batches
        n_batches += 1

    avg_loss = running_loss / max(n_batches, 1)
    return {"loss": avg_loss, "d_pos": last_d_pos, "d_neg": last_d_neg}


@torch.no_grad()
def validate_retrieval(
    model: SiameseNetwork,
    cfg: dict,
    device: torch.device,
    max_frames: int = 800,
) -> dict:
    """
    Retrieval-based validation: extract descriptors over a subsample of val
    frames and compute Recall@1 using GPS ground truth.  Much more meaningful
    than triplet loss for monitoring place-recognition training.
    """
    import cv2
    import pandas as pd
    from src.step4_dataset import get_camera_transform

    model.eval()
    processed = Path(cfg["paths"]["processed_root"])
    range_dir = processed / "range_images"
    camera_dir = processed / "camera_synced"

    cam_size = tuple(cfg["model"]["camera_input_size"])
    max_range = cfg["projection"]["max_range"]
    d_pos = cfg["mining"]["d_pos_m"]
    d_time = cfg["mining"]["d_time_s"]

    val_triplets = pd.read_csv(processed / "triplets_val.csv")
    val_idx = np.unique(np.concatenate([
        val_triplets["anchor_idx"].values,
        val_triplets["positive_idx"].values,
    ]))
    val_idx.sort()
    if len(val_idx) > max_frames:
        step = len(val_idx) // max_frames
        val_idx = val_idx[::step][:max_frames]

    poses = np.load(processed / "frame_poses.npy")
    eval_xy = poses[val_idx, :2]
    eval_t = poses[val_idx, 3]

    cam_transform = get_camera_transform(train=False, input_size=cam_size)
    batch_size = 64
    descs = []
    for start in range(0, len(val_idx), batch_size):
        batch_idx = val_idx[start : start + batch_size]
        lidar_batch, cam_batch = [], []
        for idx in batch_idx:
            ri = np.load(range_dir / f"range_{idx:06d}.npy").astype(np.float32)
            ri = np.clip(ri / max_range, 0.0, 1.0)
            lidar_batch.append(torch.from_numpy(ri).unsqueeze(0))
            cam = cv2.imread(str(camera_dir / f"cam_{idx:06d}.jpg"))
            cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
            cam_batch.append(cam_transform(cam))
        lidar_t = torch.stack(lidar_batch).to(device)
        cam_t = torch.stack(cam_batch).to(device)
        d = model.get_descriptor(lidar_t, cam_t, mode="fused")
        descs.append(d.cpu().numpy())
    descs = np.concatenate(descs, axis=0)

    sim = descs @ descs.T
    N = len(descs)

    # Vectorized GT matrix: same-place AND temporally-far
    dx = eval_xy[:, None, :] - eval_xy[None, :, :]
    dist_sq = np.einsum("ijk,ijk->ij", dx, dx)
    spatial_close = dist_sq < (d_pos * d_pos)
    temporal_far = np.abs(eval_t[:, None] - eval_t[None, :]) > d_time
    gt = spatial_close & temporal_far
    np.fill_diagonal(gt, False)

    # Recall@1 (mask self on the similarity matrix)
    sim_masked = sim.copy()
    np.fill_diagonal(sim_masked, -np.inf)
    top1 = np.argmax(sim_masked, axis=1)
    has_match = gt.any(axis=1)
    total = int(has_match.sum())
    hits = int(gt[np.arange(N), top1][has_match].sum())
    return {
        "recall_at_1": hits / max(total, 1),
        "n_queries": total,
        "n_frames": N,
    }


@torch.no_grad()
def validate(
    model: SiameseNetwork,
    loader,
    criterion_triplet: nn.TripletMarginLoss,
    device: torch.device,
) -> dict:
    """Validate using triplet loss (consistent metric across loss types)."""
    model.eval()
    running_loss = 0.0
    total_d_pos = 0.0
    total_d_neg = 0.0
    n_batches = 0

    for (a_li, a_cam), (p_li, p_cam), (n_li, n_cam) in tqdm(loader, desc="  Val", leave=False):
        a_li, a_cam = a_li.to(device), a_cam.to(device)
        p_li, p_cam = p_li.to(device), p_cam.to(device)
        n_li, n_cam = n_li.to(device), n_cam.to(device)

        with torch.autocast(device_type=device.type):
            desc_a, desc_p, desc_n = model(
                (a_li, a_cam), (p_li, p_cam), (n_li, n_cam),
            )
            loss = criterion_triplet(desc_a, desc_p, desc_n)
        running_loss += loss.item()

        total_d_pos += (desc_a - desc_p).pow(2).sum(dim=1).sqrt().mean().item()
        total_d_neg += (desc_a - desc_n).pow(2).sum(dim=1).sqrt().mean().item()
        n_batches += 1

    n = max(n_batches, 1)
    return {
        "loss": running_loss / n,
        "d_pos": total_d_pos / n,
        "d_neg": total_d_neg / n,
    }


# ── Checkpoint I/O ────────────────────────────────────────────────────────────

def save_checkpoint(
    path: Path,
    model: SiameseNetwork,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_val_loss: float,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
    }, path)


def load_checkpoint(
    path: Path,
    model: SiameseNetwork,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
) -> tuple[int, float]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in ckpt:
        try:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        except Exception:
            pass  # Scheduler type may have changed
    return ckpt.get("epoch", 0), ckpt.get("best_val_loss", float("inf"))


def transfer_backbone_from(path: Path, model: SiameseNetwork) -> int:
    """
    Transfer learning: load only backbone weights from an old checkpoint.

    Useful when the fusion architecture has changed but we want to keep
    the trained LiDAR/Camera backbone features.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    old_state = ckpt["model_state_dict"]
    new_state = model.state_dict()

    loaded = 0
    skipped = 0
    for key, value in old_state.items():
        if key in new_state and new_state[key].shape == value.shape:
            new_state[key] = value
            loaded += 1
        else:
            skipped += 1

    model.load_state_dict(new_state)
    epoch = ckpt.get("epoch", 0)
    print(f"  Transfer: loaded {loaded} params, skipped {skipped} (shape mismatch or new)")
    print(f"  Source checkpoint was from epoch {epoch}")
    return epoch


# ── Main training loop ────────────────────────────────────────────────────────

def train(cfg: dict, resume_path: str | None = None, transfer_path: str | None = None,
         profile: bool = False) -> None:
    device = get_device()
    print(f"Device: {device}")

    train_cfg = cfg["training"]
    loss_type = train_cfg.get("loss", "triplet")
    use_pk = bool(train_cfg.get("use_pk", False))
    if use_pk and loss_type != "multi_similarity":
        print("WARNING: use_pk=true requires multi_similarity loss. Forcing it.")
        loss_type = "multi_similarity"

    # ── Wandb ─────────────────────────────────────────────────────────────
    use_wandb = HAS_WANDB and train_cfg.get("wandb", True)
    if use_wandb:
        tag = "pk" if use_pk else "triplet"
        wandb.init(
            project="loop-closure-slam",
            config=cfg,
            name=f"{cfg['model'].get('backbone', 'resnet50')}_{tag}_{loss_type}",
            tags=["loop-closure", cfg["model"].get("backbone", "resnet50"), loss_type, tag],
            resume="allow",
        )
        print(f"Wandb: logging to run {wandb.run.name}")
    else:
        print("Wandb: disabled (set training.wandb: true or install wandb)")

    # ── Data ──────────────────────────────────────────────────────────────
    if use_pk:
        train_loader, val_loader, _ = build_hard_dataloaders(cfg)
        P, K = train_cfg.get("P", 16), train_cfg.get("K", 6)
        print(f"Train (PK): P={P}, K={K} → batch={P*K} × {len(train_loader)} batches/epoch")
        print(f"Val: {len(val_loader.dataset)} triplets ({len(val_loader)} batches)")
    else:
        train_loader, val_loader, _ = build_dataloaders(cfg)
        print(f"Train: {len(train_loader)} batches")
        print(f"Val:   {len(val_loader.dataset)} triplets ({len(val_loader)} batches)")

    # ── Model ─────────────────────────────────────────────────────────────
    model_cfg = cfg["model"]
    emb_dim = model_cfg["embedding_dim"]
    backbone = model_cfg.get("backbone", "resnet50")
    lidar_bb = model_cfg.get("lidar_backbone", "resnet50")
    gem_p = model_cfg.get("gem_p", 3.0)

    model = SiameseNetwork(
        embedding_dim=emb_dim,
        backbone=backbone,
        pretrained=True,
        gem_p=gem_p,
        modality_dropout_prob=train_cfg["modality_dropout_prob"],
        lidar_backbone=lidar_bb,
    ).to(device)

    total, trainable = count_parameters(model)
    print(f"Model: LiDAR={lidar_bb}, Camera={backbone} + GeM, {trainable:,} trainable params ({total:,} total)")

    if use_wandb:
        wandb.watch(model, log="gradients", log_freq=100)

    # ── Loss ──────────────────────────────────────────────────────────────
    if loss_type == "multi_similarity":
        criterion = MultiSimilarityLoss(
            alpha=train_cfg.get("ms_alpha", 2.0),
            beta=train_cfg.get("ms_beta", 50.0),
            base=train_cfg.get("ms_base", 0.5),
        )
        print(f"Loss: Multi-Similarity (α={criterion.alpha}, β={criterion.beta})")
    else:
        criterion = nn.TripletMarginLoss(margin=train_cfg["triplet_margin"], p=2)
        print(f"Loss: Triplet Margin (margin={train_cfg['triplet_margin']})")

    # Triplet loss for validation (consistent metric)
    val_criterion = nn.TripletMarginLoss(margin=train_cfg.get("triplet_margin", 1.0), p=2)

    # ── Optimizer / Scheduler ─────────────────────────────────────────────
    backbone_lr = train_cfg.get("backbone_lr", train_cfg["learning_rate"])
    fusion_lr = train_cfg.get("fusion_lr", train_cfg["learning_rate"])
    warmup_epochs = train_cfg.get("warmup_epochs", 0)

    # Differential learning rates: slow for pretrained backbones, fast for fusion
    backbone_params = []
    fusion_params = []
    for name, param in model.named_parameters():
        if "backbone" in name:
            backbone_params.append(param)
        else:
            fusion_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": backbone_lr},
        {"params": fusion_params, "lr": fusion_lr},
    ], weight_decay=train_cfg["weight_decay"])

    num_epochs = train_cfg["num_epochs"]

    # Cosine annealing (applied after warmup)
    cosine_scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6,
    )

    # Linear warmup + cosine annealing
    if warmup_epochs > 0:
        warmup_scheduler = LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup_epochs,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = cosine_scheduler

    # ── Resume or Transfer from checkpoint ─────────────────────────────────
    start_epoch = 0
    best_val_loss = float("inf")
    if resume_path and Path(resume_path).exists():
        start_epoch, best_val_loss = load_checkpoint(
            Path(resume_path), model, optimizer, scheduler,
        )
        start_epoch += 1
        print(f"Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")
    elif transfer_path and Path(transfer_path).exists():
        transfer_backbone_from(Path(transfer_path), model)
        print(f"Transferred backbone weights from {transfer_path}")
        print(f"  Starting fresh from epoch 0 (new fusion head)")

    # ── AMP & Accumulation ────────────────────────────────────────────────
    use_amp = train_cfg.get("amp", False)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp and device.type == "cuda" else None
    acc_steps = train_cfg.get("accumulate_grad_batches", 1)

    # ── Training ──────────────────────────────────────────────────────────
    ckpt_dir = Path("checkpoints")
    patience = train_cfg["early_stopping_patience"]
    epochs_no_improve = 0

    print(f"\n{'='*60}")
    print(f"Training for up to {num_epochs} epochs (patience={patience})")
    print(f"  Backbone: {backbone} + GeM(p={gem_p})")
    print(f"  Embedding dim: {emb_dim}")
    print(f"  Backbone LR: {backbone_lr}, Fusion LR: {fusion_lr}")
    print(f"  Weight decay: {train_cfg['weight_decay']}")
    print(f"  Scheduler: LinearWarmup({warmup_epochs}ep) → CosineAnnealingWarmRestarts")
    print(f"  Modality dropout: {train_cfg['modality_dropout_prob']}")
    print(f"  AMP Enabled: {bool(scaler)}")
    print(f"  Accumulate Grad Batches: {acc_steps}")
    print(f"{'='*60}\n")

    history = {"train_loss": [], "val_loss": [], "d_pos": [], "d_neg": []}
    optimizer.zero_grad(set_to_none=True)

    # ── PyTorch Profiler setup ────────────────────────────────────────
    profile_epochs = 2  # profile only the first N epochs
    profiler = None
    if profile:
        profiler_log_dir = Path("profiler_logs")
        profiler_log_dir.mkdir(exist_ok=True)
        total_profile_steps = len(train_loader) * profile_epochs
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=1, warmup=3, active=5, repeat=0,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                str(profiler_log_dir)
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        profiler.__enter__()
        print(f"\n🔬 Profiler ENABLED — profiling first {profile_epochs} epochs")
        print(f"   Output → {profiler_log_dir}/")
        print(f"   View with: tensorboard --logdir={profiler_log_dir}\n")

    # Track best R@1 (primary metric in PK mode), val_loss as fallback
    best_val_r1 = -1.0
    val_retrieval_frames = train_cfg.get("val_retrieval_frames", 800)

    for epoch in range(start_epoch, num_epochs):
        t0 = time.time()

        # Train
        if use_pk:
            train_metrics = train_one_epoch_pk(
                model, train_loader, criterion, optimizer, device, scaler, acc_steps,
            )
        else:
            train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer, device, scaler, acc_steps,
                loss_type=loss_type,
                profiler=profiler if (profile and epoch < start_epoch + profile_epochs) else None,
            )

        # Stop profiler after profile_epochs
        if profile and profiler is not None and epoch == start_epoch + profile_epochs - 1:
            profiler.__exit__(None, None, None)
            profiler = None
            print(f"\n🔬 Profiler stopped after epoch {epoch+1}. Traces saved.\n")

        # Validate — in PK mode, retrieval R@1 is the primary signal.
        # Skip the slow triplet-loss pass to keep epochs fast.
        if use_pk:
            val_metrics = {"loss": 0.0, "d_pos": 0.0, "d_neg": 0.0}
        else:
            val_metrics = validate(model, val_loader, val_criterion, device)
        retrieval_metrics = validate_retrieval(
            model, cfg, device, max_frames=val_retrieval_frames,
        )

        # LR scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - t0

        # Log — in PK mode, prefer train-side d_pos/d_neg since val is retrieval-only
        d_pos_disp = train_metrics["d_pos"] if use_pk else val_metrics["d_pos"]
        d_neg_disp = train_metrics["d_neg"] if use_pk else val_metrics["d_neg"]
        print(
            f"Epoch {epoch+1:>3}/{num_epochs} │ "
            f"train_loss={train_metrics['loss']:.4f}  "
            + ("" if use_pk else f"val_loss={val_metrics['loss']:.4f}  ")
            + f"│ val_R@1={retrieval_metrics['recall_at_1']:.3f} "
            f"(n={retrieval_metrics['n_queries']})  │ "
            f"d_pos={d_pos_disp:.3f} d_neg={d_neg_disp:.3f}  │ "
            f"lr={current_lr:.1e}  │ "
            f"{elapsed:.0f}s"
        )

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["d_pos"].append(val_metrics["d_pos"])
        history["d_neg"].append(val_metrics["d_neg"])

        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": train_metrics["loss"],
                "train/d_pos": train_metrics["d_pos"],
                "train/d_neg": train_metrics["d_neg"],
                "train/margin": train_metrics["d_neg"] - train_metrics["d_pos"],
                "val/loss": val_metrics["loss"],
                "val/d_pos": val_metrics["d_pos"],
                "val/d_neg": val_metrics["d_neg"],
                "val/margin": val_metrics["d_neg"] - val_metrics["d_pos"],
                "val/recall_at_1": retrieval_metrics["recall_at_1"],
                "val/n_queries": retrieval_metrics["n_queries"],
                "lr": current_lr,
                "epoch_time_s": elapsed,
            })

        # Save last checkpoint every epoch
        save_checkpoint(ckpt_dir / "last.pt", model, optimizer, scheduler, epoch, best_val_loss)

        # Primary early-stopping signal: retrieval R@1 on val (PK mode) or val loss (legacy)
        primary_improved = (
            retrieval_metrics["recall_at_1"] > best_val_r1
            if use_pk else
            val_metrics["loss"] < best_val_loss
        )

        if primary_improved:
            best_val_r1 = max(best_val_r1, retrieval_metrics["recall_at_1"])
            best_val_loss = min(best_val_loss, val_metrics["loss"])
            epochs_no_improve = 0
            save_checkpoint(ckpt_dir / "best.pt", model, optimizer, scheduler, epoch, best_val_loss)
            if use_pk:
                print(f"  ✓ New best model saved (val_R@1={best_val_r1:.4f})")
            else:
                print(f"  ✓ New best model saved (val_loss={best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping: no improvement for {patience} epochs.")
                break

    # ── Save training curves ──────────────────────────────────────────────
    try:
        save_training_curves(history, ckpt_dir / "training_curves.png")
        if use_wandb:
            wandb.log({"training_curves": wandb.Image(str(ckpt_dir / "training_curves.png"))})
    except Exception as e:
        print(f"Warning: Could not save training curves: {e}")
    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
    print(f"Best model: {ckpt_dir / 'best.pt'}")

    if use_wandb:
        wandb.summary["best_val_loss"] = best_val_loss
        wandb.summary["best_val_recall_at_1"] = best_val_r1
        wandb.summary["total_epochs"] = len(history["train_loss"])
        wandb.finish()


# ── Training curves plot ──────────────────────────────────────────────────────

def save_training_curves(history: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], "b-", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], "r-", label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["d_pos"], "g-", label="d(Anchor, Positive)")
    axes[1].plot(epochs, history["d_neg"], "r-", label="d(Anchor, Negative)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Euclidean Distance")
    axes[1].set_title("Embedding Distances")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved training curves → {path}")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Step 3: Train the Siamese Network")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from (same architecture)")
    parser.add_argument("--transfer", default=None, help="Path to checkpoint to transfer backbone weights from (different architecture OK)")
    parser.add_argument("--profile", action="store_true", help="Enable PyTorch profiler for the first 2 training epochs")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg, resume_path=args.resume, transfer_path=args.transfer, profile=args.profile)


if __name__ == "__main__":
    main()
