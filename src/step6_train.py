"""
Step 3 — Training Loop with Triplet Margin Loss
=================================================
Trains the Two-Stream Siamese CNN to produce location fingerprints
using metric learning.

Usage:
    python -m src.step6_train
    python -m src.step6_train --config configs/config.yaml
    python -m src.step6_train --config configs/config.yaml --resume checkpoints/last.pt
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

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


# ── Training / Validation steps ───────────────────────────────────────────────

def train_one_epoch(
    model: SiameseNetwork,
    loader,
    criterion: nn.TripletMarginLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None,
    accumulate_grad_batches: int,
) -> dict:
    model.train()
    running_loss = 0.0
    n_batches = 0

    for i, ((a_li, a_cam), (p_li, p_cam), (n_li, n_cam)) in enumerate(tqdm(loader, desc="  Train", leave=False)):
        a_li, a_cam = a_li.to(device), a_cam.to(device)
        p_li, p_cam = p_li.to(device), p_cam.to(device)
        n_li, n_cam = n_li.to(device), n_cam.to(device)

        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            desc_a, desc_p, desc_n = model(
                (a_li, a_cam), (p_li, p_cam), (n_li, n_cam),
            )
            loss = criterion(desc_a, desc_p, desc_n)
            loss = loss / accumulate_grad_batches

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

        running_loss += loss.item() * accumulate_grad_batches  # undo scaling for logging
        n_batches += 1

    avg_loss = running_loss / max(n_batches, 1)

    # Compute some embedding quality metrics on the last batch
    with torch.no_grad():
        d_pos = (desc_a - desc_p).pow(2).sum(dim=1).sqrt().mean().item()
        d_neg = (desc_a - desc_n).pow(2).sum(dim=1).sqrt().mean().item()

    return {"loss": avg_loss, "d_pos": d_pos, "d_neg": d_neg}


def batch_hard_triplet_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float,
) -> tuple[torch.Tensor, float, float]:
    """
    Batch-hard triplet loss.

    For each anchor, find:
      - the hardest positive (same label, max distance)
      - the hardest negative (different label, min distance)

    Returns (loss, mean_d_pos, mean_d_neg).
    """
    # Pairwise distance matrix
    dist_mat = torch.cdist(embeddings, embeddings, p=2)  # (B, B)

    B = embeddings.size(0)
    labels = labels.unsqueeze(0)  # (1, B)
    same_place = (labels == labels.T)  # (B, B)
    diff_place = ~same_place

    # Mask out self-comparisons
    eye = torch.eye(B, dtype=torch.bool, device=embeddings.device)
    same_place = same_place & ~eye

    # Hardest positive: max dist among same-place pairs
    # Set non-positive pairs to -1 so they won't be selected as max
    pos_dists = dist_mat.clone()
    pos_dists[~same_place] = -1.0
    hardest_pos, _ = pos_dists.max(dim=1)  # (B,)

    # Hardest negative: min dist among different-place pairs
    # Set same-place pairs to large value so they won't be selected as min
    neg_dists = dist_mat.clone()
    neg_dists[~diff_place] = 1e6
    hardest_neg, _ = neg_dists.min(dim=1)  # (B,)

    # Only compute loss for anchors that have at least one positive
    valid = same_place.any(dim=1)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True), 0.0, 0.0

    hardest_pos = hardest_pos[valid]
    hardest_neg = hardest_neg[valid]

    loss = F.relu(hardest_pos - hardest_neg + margin).mean()

    return loss, hardest_pos.mean().item(), hardest_neg.mean().item()


def train_one_epoch_hard(
    model: SiameseNetwork,
    loader,
    margin: float,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None,
    accumulate_grad_batches: int,
) -> dict:
    """Training loop with online batch-hard mining."""
    model.train()
    running_loss = 0.0
    n_batches = 0

    for i, (lidar, camera, labels) in enumerate(tqdm(loader, desc="  Train (hard)", leave=False)):
        lidar = lidar.to(device)
        camera = camera.to(device)
        labels = labels.to(device)

        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            # Forward all P*K frames through the shared encoder
            embeddings = model.encoder(lidar, camera)  # (P*K, emb_dim)
            loss, d_pos, d_neg = batch_hard_triplet_loss(embeddings, labels, margin)
            loss = loss / accumulate_grad_batches

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

        running_loss += loss.item() * accumulate_grad_batches  # undo scaling for logging
        n_batches += 1

    avg_loss = running_loss / max(n_batches, 1)
    return {"loss": avg_loss, "d_pos": d_pos, "d_neg": d_neg}



@torch.no_grad()
def validate(
    model: SiameseNetwork,
    loader,
    criterion: nn.TripletMarginLoss,
    device: torch.device,
) -> dict:
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
            loss = criterion(desc_a, desc_p, desc_n)
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
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt.get("epoch", 0), ckpt.get("best_val_loss", float("inf"))


# ── Main training loop ────────────────────────────────────────────────────────

def train(cfg: dict, resume_path: str | None = None) -> None:
    device = get_device()
    print(f"Device: {device}")

    train_cfg = cfg["training"]
    use_hard_mining = train_cfg.get("hard_mining", False)

    # ── Data ──────────────────────────────────────────────────────────────
    triplet_train_loader, val_loader, _ = build_dataloaders(cfg)
    hard_train_loader = None
    if use_hard_mining:
        hard_train_loader, _, _ = build_hard_dataloaders(cfg)
    print(f"Train (triplet): {len(triplet_train_loader)} batches")
    if hard_train_loader:
        print(f"Train (hard):    {len(hard_train_loader)} batches")
    print(f"Val:   {len(val_loader.dataset)} triplets ({len(val_loader)} batches)")

    # ── Model ─────────────────────────────────────────────────────────────
    emb_dim = cfg["model"]["embedding_dim"]
    model = SiameseNetwork(embedding_dim=emb_dim, pretrained=True, modality_dropout_prob=train_cfg["modality_dropout_prob"]).to(device)

    total, trainable = count_parameters(model)
    print(f"Model: {trainable:,} trainable params ({total:,} total)")

    # ── Loss / Optimizer / Scheduler ──────────────────────────────────────
    criterion = nn.TripletMarginLoss(
        margin=train_cfg["triplet_margin"], p=2,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3,
    )
    warmup_epochs = 3
    base_lr = train_cfg["learning_rate"]

    # ── Resume from checkpoint ────────────────────────────────────────────
    start_epoch = 0
    best_val_loss = float("inf")
    if resume_path and Path(resume_path).exists():
        start_epoch, best_val_loss = load_checkpoint(
            Path(resume_path), model, optimizer, scheduler,
        )
        start_epoch += 1
        print(f"Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    # ── AMP & Accumulation ────────────────────────────────────────────────
    use_amp = train_cfg.get("amp", False)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp and device.type == "cuda" else None
    acc_steps = train_cfg.get("accumulate_grad_batches", 1)

    # ── Phase configuration ───────────────────────────────────────────────
    ckpt_dir = Path("checkpoints")
    patience = train_cfg["early_stopping_patience"]
    num_epochs = train_cfg["num_epochs"]
    hard_mining_epochs = train_cfg.get("hard_mining_epochs", 20)

    # Phase 1: Triplet pre-training
    # Phase 2: Hard mining fine-tuning (if enabled)
    phases = [{"name": "Phase 1 (triplet)", "loader": triplet_train_loader, "hard": False, "epochs": num_epochs}]
    if use_hard_mining and hard_train_loader:
        phases.append({"name": "Phase 2 (hard mining)", "loader": hard_train_loader, "hard": True, "epochs": hard_mining_epochs})

    history = {"train_loss": [], "val_loss": [], "d_pos": [], "d_neg": []}
    global_epoch = start_epoch

    for phase in phases:
        phase_name = phase["name"]
        train_loader = phase["loader"]
        is_hard = phase["hard"]
        phase_epochs = phase["epochs"]

        # Reset early stopping and LR for phase 2
        if is_hard:
            epochs_no_improve = 0
            best_val_loss = float("inf")
            # Reduce LR for fine-tuning
            for pg in optimizer.param_groups:
                pg["lr"] = train_cfg["learning_rate"] * 0.1
            scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
            print(f"\n{'─'*60}")
            print(f"Switching to {phase_name} (lr={optimizer.param_groups[0]['lr']:.1e})")
            print(f"{'─'*60}\n")
        else:
            epochs_no_improve = 0

        print(f"\n{'='*60}")
        print(f"{phase_name}: up to {phase_epochs} epochs (patience={patience})")
        print(f"  Triplet margin: {train_cfg['triplet_margin']}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.1e}, weight_decay: {train_cfg['weight_decay']}")
        print(f"  Modality dropout: {train_cfg['modality_dropout_prob']}")
        print(f"  AMP Enabled: {bool(scaler)}")
        print(f"  Accumulate Grad Batches: {acc_steps}")
        print(f"{'='*60}\n")

        optimizer.zero_grad(set_to_none=True)

        for ep in range(phase_epochs):
            t0 = time.time()

            # Train
            if is_hard:
                train_metrics = train_one_epoch_hard(
                    model, train_loader, train_cfg["triplet_margin"],
                    optimizer, device, scaler, acc_steps
                )
            else:
                train_metrics = train_one_epoch(
                    model, train_loader, criterion, optimizer, device, scaler, acc_steps
                )

            # Validate
            val_metrics = validate(model, val_loader, criterion, device)

            # LR scheduler: warmup for first N epochs, then ReduceLROnPlateau
            if global_epoch < warmup_epochs:
                # Linear warmup: start at 0.1*base_lr, ramp to base_lr
                warmup_factor = 0.1 + 0.9 * (global_epoch + 1) / warmup_epochs
                for pg in optimizer.param_groups:
                    pg["lr"] = base_lr * warmup_factor
            else:
                scheduler.step(val_metrics["loss"])
            current_lr = optimizer.param_groups[0]["lr"]

            elapsed = time.time() - t0

            global_epoch += 1

            # Log
            print(
                f"Epoch {ep+1:>3}/{phase_epochs} │ "
                f"train_loss={train_metrics['loss']:.4f}  "
                f"val_loss={val_metrics['loss']:.4f}  │ "
                f"d_pos={val_metrics['d_pos']:.3f}  "
                f"d_neg={val_metrics['d_neg']:.3f}  │ "
                f"lr={current_lr:.1e}  │ "
                f"{elapsed:.0f}s"
            )

            history["train_loss"].append(train_metrics["loss"])
            history["val_loss"].append(val_metrics["loss"])
            history["d_pos"].append(val_metrics["d_pos"])
            history["d_neg"].append(val_metrics["d_neg"])

            # Save last checkpoint
            save_checkpoint(ckpt_dir / "last.pt", model, optimizer, scheduler, global_epoch, best_val_loss)

            # Save best checkpoint
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                epochs_no_improve = 0
                save_checkpoint(ckpt_dir / "best.pt", model, optimizer, scheduler, global_epoch, best_val_loss)
                print(f"  ✓ New best model saved (val_loss={best_val_loss:.4f})")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"\nEarly stopping: no improvement for {patience} epochs.")
                    break

    # ── Save training curves ──────────────────────────────────────────────
    save_training_curves(history, ckpt_dir / "training_curves.png")
    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
    print(f"Best model: {ckpt_dir / 'best.pt'}")


# ── Training curves plot ──────────────────────────────────────────────────────

def save_training_curves(history: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss curves
    axes[0].plot(epochs, history["train_loss"], "b-", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], "r-", label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Triplet Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Distance curves
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
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg, resume_path=args.resume)


if __name__ == "__main__":
    main()
