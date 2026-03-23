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

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

from src.step4_dataset import build_dataloaders
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

class MultiSimilarityLoss(nn.Module):
    """
    Multi-Similarity Loss (Wang et al., CVPR 2019).

    Considers ALL positive and negative pairs in a batch, weighting them
    by their similarity. Much stronger gradient signal than triplet loss.
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
        labels: (B,) integer place labels
        """
        # Cosine similarity matrix (since embeddings are L2-normalized)
        sim_mat = embeddings @ embeddings.T  # (B, B)

        B = embeddings.size(0)
        labels = labels.unsqueeze(0)
        same = (labels == labels.T)
        eye = torch.eye(B, dtype=torch.bool, device=embeddings.device)
        pos_mask = same & ~eye
        neg_mask = ~same

        loss = torch.tensor(0.0, device=embeddings.device)
        n_valid = 0

        for i in range(B):
            pos_idx = pos_mask[i].nonzero(as_tuple=True)[0]
            neg_idx = neg_mask[i].nonzero(as_tuple=True)[0]

            if len(pos_idx) == 0 or len(neg_idx) == 0:
                continue

            pos_sim = sim_mat[i, pos_idx]
            neg_sim = sim_mat[i, neg_idx]

            # Mining: keep hard positives and hard negatives
            neg_max = neg_sim.max()
            pos_min = pos_sim.min()

            # Keep negatives harder than easiest positive
            hard_neg = neg_sim[neg_sim + self.margin > pos_min]
            # Keep positives harder than easiest negative
            hard_pos = pos_sim[pos_sim - self.margin < neg_max]

            if len(hard_neg) == 0 or len(hard_pos) == 0:
                continue

            # Positive term: pull together
            pos_loss = (1.0 / self.alpha) * torch.log(
                1 + torch.sum(torch.exp(-self.alpha * (hard_pos - self.base)))
            )
            # Negative term: push apart
            neg_loss = (1.0 / self.beta) * torch.log(
                1 + torch.sum(torch.exp(self.beta * (hard_neg - self.base)))
            )

            loss = loss + pos_loss + neg_loss
            n_valid += 1

        return loss / max(n_valid, 1)


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

    avg_loss = running_loss / max(n_batches, 1)
    return {"loss": avg_loss, "d_pos": last_d_pos, "d_neg": last_d_neg}


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


# ── Main training loop ────────────────────────────────────────────────────────

def train(cfg: dict, resume_path: str | None = None) -> None:
    device = get_device()
    print(f"Device: {device}")

    train_cfg = cfg["training"]
    loss_type = train_cfg.get("loss", "triplet")

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader, val_loader, _ = build_dataloaders(cfg)
    print(f"Train: {len(train_loader)} batches")
    print(f"Val:   {len(val_loader.dataset)} triplets ({len(val_loader)} batches)")

    # ── Model ─────────────────────────────────────────────────────────────
    model_cfg = cfg["model"]
    emb_dim = model_cfg["embedding_dim"]
    backbone = model_cfg.get("backbone", "resnet50")
    gem_p = model_cfg.get("gem_p", 3.0)

    model = SiameseNetwork(
        embedding_dim=emb_dim,
        backbone=backbone,
        pretrained=True,
        gem_p=gem_p,
        modality_dropout_prob=train_cfg["modality_dropout_prob"],
    ).to(device)

    total, trainable = count_parameters(model)
    print(f"Model: {backbone} + GeM, {trainable:,} trainable params ({total:,} total)")

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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    num_epochs = train_cfg["num_epochs"]
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6,
    )

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
    print(f"  LR: {train_cfg['learning_rate']}, weight_decay: {train_cfg['weight_decay']}")
    print(f"  Scheduler: CosineAnnealingWarmRestarts (T_0=10)")
    print(f"  Modality dropout: {train_cfg['modality_dropout_prob']}")
    print(f"  AMP Enabled: {bool(scaler)}")
    print(f"  Accumulate Grad Batches: {acc_steps}")
    print(f"{'='*60}\n")

    history = {"train_loss": [], "val_loss": [], "d_pos": [], "d_neg": []}
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, num_epochs):
        t0 = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, acc_steps,
            loss_type=loss_type,
        )

        # Validate (always use triplet loss for consistent comparison)
        val_metrics = validate(model, val_loader, val_criterion, device)

        # LR scheduler
        scheduler.step(epoch + 1)
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - t0

        # Log
        print(
            f"Epoch {epoch+1:>3}/{num_epochs} │ "
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
        save_checkpoint(ckpt_dir / "last.pt", model, optimizer, scheduler, epoch, best_val_loss)

        # Save best checkpoint
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            epochs_no_improve = 0
            save_checkpoint(ckpt_dir / "best.pt", model, optimizer, scheduler, epoch, best_val_loss)
            print(f"  ✓ New best model saved (val_loss={best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping: no improvement for {patience} epochs.")
                break

    # ── Save training curves ──────────────────────────────────────────────
    try:
        save_training_curves(history, ckpt_dir / "training_curves.png")
    except Exception as e:
        print(f"Warning: Could not save training curves: {e}")
    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
    print(f"Best model: {ckpt_dir / 'best.pt'}")


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
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg, resume_path=args.resume)


if __name__ == "__main__":
    main()
