"""
Step 2B — Two-Stream Siamese CNN for Multi-Modal Loop Closure
==============================================================
Architecture:
    LiDAR Range Image (1, 128, 1024) ──→ ResNet-50 + GeM ──→ 2048-d
                                                                  │
                                                            Concatenate (4096-d)
                                                                  │
    Camera RGB Image  (3, 224, 224)  ──→ ResNet-50 + GeM ──→ 2048-d
                                                                  │
                                                            Fusion MLP
                                                                  │
                                                            512-d L2-normalized
                                                           "Location Fingerprint"

Usage:
    from src.step5_model import TwoStreamEncoder, SiameseNetwork
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import yaml


# ── GeM Pooling ──────────────────────────────────────────────────────────────

class GeM(nn.Module):
    """
    Generalized Mean Pooling (Radenovic et al., 2018).

    Learns a power parameter `p` that interpolates between average pooling
    (p=1) and max pooling (p→∞). Focuses on discriminative spatial regions.
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) → (B, C, 1, 1)"""
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), 1
        ).pow(1.0 / self.p)


# ── Backbones ────────────────────────────────────────────────────────────────

class LiDARBackbone(nn.Module):
    """
    ResNet backbone modified for single-channel range image input.
    Supports ResNet-18 (512-d output) and ResNet-50 (2048-d output).
    Uses GeM pooling instead of average pooling.
    """

    def __init__(self, backbone: str = "resnet50", pretrained: bool = True, gem_p: float = 3.0):
        super().__init__()

        if backbone == "resnet50":
            base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            self.feat_dim = 2048
        else:
            base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            self.feat_dim = 512

        # Replace conv1: 3-channel → 1-channel
        old_conv = base.conv1
        new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
        base.conv1 = new_conv

        # Feature extraction layers (without avgpool and fc)
        self.features = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
        )
        self.pool = GeM(p=gem_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, 128, 1024) → (B, feat_dim)"""
        out = self.features(x)
        out = self.pool(out)
        return out.flatten(1)


class CameraBackbone(nn.Module):
    """
    Standard pretrained ResNet backbone for 3-channel RGB input.
    Uses GeM pooling instead of average pooling.
    """

    def __init__(self, backbone: str = "resnet50", pretrained: bool = True, gem_p: float = 3.0):
        super().__init__()

        if backbone == "resnet50":
            base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            self.feat_dim = 2048
        else:
            base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            self.feat_dim = 512

        self.features = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
        )
        self.pool = GeM(p=gem_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 224, 224) → (B, feat_dim)"""
        out = self.features(x)
        out = self.pool(out)
        return out.flatten(1)


# ── Encoder ──────────────────────────────────────────────────────────────────

class TwoStreamEncoder(nn.Module):
    """
    Fuses LiDAR geometry and camera texture into a single descriptor.

    Two parallel ResNet backbones with GeM pooling extract features.
    A fusion MLP compresses the concatenated vector into
    a compact, L2-normalized embedding.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        backbone: str = "resnet50",
        pretrained: bool = True,
        gem_p: float = 3.0,
        modality_dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.modality_dropout_prob = modality_dropout_prob
        self.lidar_backbone = LiDARBackbone(backbone, pretrained, gem_p)
        self.camera_backbone = CameraBackbone(backbone, pretrained, gem_p)

        feat_dim = self.lidar_backbone.feat_dim  # 2048 for resnet50, 512 for resnet18
        fused_dim = feat_dim * 2  # 4096 for resnet50

        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(1024, embedding_dim),
        )

    def forward(
        self, lidar: torch.Tensor, camera: torch.Tensor, mode: str = "fused"
    ) -> torch.Tensor:
        l_feat = self.lidar_backbone(lidar)
        c_feat = self.camera_backbone(camera)

        if self.training and self.modality_dropout_prob > 0:
            B = l_feat.size(0)
            p = self.modality_dropout_prob
            r = torch.rand(B, 1, device=l_feat.device)
            drop_lidar = (r < p).float()
            drop_camera = ((r >= p) & (r < 2 * p)).float()
            l_scale = (1 - drop_lidar) * (1 + drop_camera)
            c_scale = (1 - drop_camera) * (1 + drop_lidar)
            l_feat = l_feat * l_scale
            c_feat = c_feat * c_scale

        if mode == "lidar_only":
            c_feat = torch.zeros_like(c_feat)
        elif mode == "camera_only":
            l_feat = torch.zeros_like(l_feat)

        fused = torch.cat([l_feat, c_feat], dim=1)
        desc = self.fusion(fused)
        return F.normalize(desc, p=2, dim=1)


# ── Siamese Wrapper ──────────────────────────────────────────────────────────

class SiameseNetwork(nn.Module):
    """
    Wraps TwoStreamEncoder for triplet or metric learning training.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        backbone: str = "resnet50",
        pretrained: bool = True,
        gem_p: float = 3.0,
        modality_dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.encoder = TwoStreamEncoder(
            embedding_dim, backbone, pretrained, gem_p, modality_dropout_prob
        )

    def forward(
        self,
        anchor: tuple[torch.Tensor, torch.Tensor],
        positive: tuple[torch.Tensor, torch.Tensor],
        negative: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lidar_all = torch.cat([anchor[0], positive[0], negative[0]], dim=0)
        camera_all = torch.cat([anchor[1], positive[1], negative[1]], dim=0)

        desc_all = self.encoder(lidar_all, camera_all)

        B = anchor[0].size(0)
        return desc_all[:B], desc_all[B:2*B], desc_all[2*B:]

    def get_descriptor(
        self, lidar: torch.Tensor, camera: torch.Tensor, mode: str = "fused"
    ) -> torch.Tensor:
        """Single-pair inference for evaluation / deployment."""
        return self.encoder(lidar, camera, mode=mode)


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    emb_dim = cfg["model"]["embedding_dim"]
    backbone = cfg["model"].get("backbone", "resnet50")
    gem_p = cfg["model"].get("gem_p", 3.0)
    modality_dropout_prob = cfg["training"]["modality_dropout_prob"]
    model = SiameseNetwork(
        embedding_dim=emb_dim, backbone=backbone, pretrained=True,
        gem_p=gem_p, modality_dropout_prob=modality_dropout_prob,
    )

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: SiameseNetwork (backbone={backbone}, embedding_dim={emb_dim}, GeM p={gem_p})")
    print(f"  Total parameters:     {total:,}")
    print(f"  Trainable parameters: {trainable:,}")

    B = 4
    dummy_lidar = torch.randn(B, 1, 128, 1024)
    dummy_cam = torch.randn(B, 3, 224, 224)

    model.eval()
    with torch.no_grad():
        anchor = (dummy_lidar, dummy_cam)
        positive = (dummy_lidar, dummy_cam)
        negative = (dummy_lidar, dummy_cam)
        desc_a, desc_p, desc_n = model(anchor, positive, negative)

    print(f"\n  Input LiDAR:  {dummy_lidar.shape}")
    print(f"  Input Camera: {dummy_cam.shape}")
    print(f"  Output desc:  {desc_a.shape}")
    print(f"  L2 norm:      {desc_a.norm(dim=1)}")

    with torch.no_grad():
        single = model.get_descriptor(dummy_lidar, dummy_cam)
    print(f"\n  Single inference: {single.shape}")
    print(f"  Matches anchor:  {torch.allclose(desc_a, single)}")
