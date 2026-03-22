"""
Step 2B — Two-Stream Siamese CNN for Multi-Modal Loop Closure
==============================================================
Architecture:
    LiDAR Range Image (1, 128, 1024) ──→ ResNet-18 ──→ 512-d
                                                          │
                                                    Concatenate (1024-d)
                                                          │
    Camera RGB Image  (3, 224, 224)  ──→ ResNet-18 ──→ 512-d
                                                          │
                                                    Fusion MLP
                                                          │
                                                    256-d L2-normalized
                                                   "Location Fingerprint"

The three Siamese branches (Anchor, Positive, Negative) share identical
weights — we define ONE encoder and call it three times.

Usage:
    from src.step5_model import TwoStreamEncoder, SiameseNetwork
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import yaml


class LiDARBackbone(nn.Module):
    """
    ResNet-18 modified for single-channel range image input.

    The first conv layer is replaced from 3-channel to 1-channel.
    Its weights are initialized as the mean of the pretrained RGB filters,
    preserving the learned low-level edge/texture features.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)

        # Replace conv1: (3, 64, 7, 7) → (1, 64, 7, 7)
        old_conv = base.conv1
        new_conv = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False,
        )
        if pretrained:
            # Mean of RGB weights → grayscale initialization
            new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
        base.conv1 = new_conv

        # Remove the final classification head (fc layer)
        # Keep everything up to and including avgpool → output is (B, 512)
        self.features = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
            base.avgpool,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, 128, 1024) → (B, 512)"""
        out = self.features(x)
        return out.flatten(1)


class CameraBackbone(nn.Module):
    """
    Standard pretrained ResNet-18 for 3-channel RGB input.
    Output is the 512-d feature vector from global average pooling.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)

        self.features = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
            base.avgpool,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 224, 224) → (B, 512)"""
        out = self.features(x)
        return out.flatten(1)


class TwoStreamEncoder(nn.Module):
    """
    Fuses LiDAR geometry and camera texture into a single descriptor.

    Two parallel ResNet-18 backbones extract 512-d features each.
    A fusion MLP compresses the concatenated 1024-d vector into
    a compact, L2-normalized embedding.
    """

    def __init__(self, embedding_dim: int = 256, pretrained: bool = True, modality_dropout_prob: float = 0.0):
        super().__init__()
        self.modality_dropout_prob = modality_dropout_prob
        self.lidar_backbone = LiDARBackbone(pretrained=pretrained)
        self.camera_backbone = CameraBackbone(pretrained=pretrained)

        self.fusion = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, embedding_dim),
        )

    def forward(
        self, lidar: torch.Tensor, camera: torch.Tensor, mode: str = "fused"
    ) -> torch.Tensor:
        """
        lidar:  (B, 1, 128, 1024)
        camera: (B, 3, 224, 224)
        returns: (B, embedding_dim) L2-normalized descriptor
        """
        l_feat = self.lidar_backbone(lidar)     # (B, 512)
        c_feat = self.camera_backbone(camera)   # (B, 512)

        if self.training and self.modality_dropout_prob > 0:
            B = l_feat.size(0)
            p = self.modality_dropout_prob
            r = torch.rand(B, 1, device=l_feat.device)
            drop_lidar = (r < p).float()
            drop_camera = ((r >= p) & (r < 2 * p)).float()
            # Scale surviving modality by 2x to compensate for zeroed-out partner
            l_scale = (1 - drop_lidar) * (1 + drop_camera)
            c_scale = (1 - drop_camera) * (1 + drop_lidar)
            l_feat = l_feat * l_scale
            c_feat = c_feat * c_scale

        if mode == "lidar_only":
            c_feat = torch.zeros_like(c_feat)
        elif mode == "camera_only":
            l_feat = torch.zeros_like(l_feat)

        fused = torch.cat([l_feat, c_feat], dim=1)  # (B, 1024)
        desc = self.fusion(fused)               # (B, 256)
        return F.normalize(desc, p=2, dim=1)    # unit vector


class SiameseNetwork(nn.Module):
    """
    Wraps TwoStreamEncoder for triplet training.

    Takes anchor/positive/negative inputs, passes each through the
    shared encoder, and returns the three descriptors.
    """

    def __init__(self, embedding_dim: int = 256, pretrained: bool = True, modality_dropout_prob: float = 0.0):
        super().__init__()
        self.encoder = TwoStreamEncoder(embedding_dim, pretrained, modality_dropout_prob)

    def forward(
        self,
        anchor: tuple[torch.Tensor, torch.Tensor],
        positive: tuple[torch.Tensor, torch.Tensor],
        negative: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Each input is a (lidar, camera) tuple.
        Returns (desc_a, desc_p, desc_n), each (B, embedding_dim).
        """
        lidar_all = torch.cat([anchor[0], positive[0], negative[0]], dim=0)
        camera_all = torch.cat([anchor[1], positive[1], negative[1]], dim=0)
        
        desc_all = self.encoder(lidar_all, camera_all)
        
        B = anchor[0].size(0)
        desc_a = desc_all[:B]
        desc_p = desc_all[B:2*B]
        desc_n = desc_all[2*B:]
        
        return desc_a, desc_p, desc_n

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
    modality_dropout_prob = cfg["training"]["modality_dropout_prob"]
    model = SiameseNetwork(embedding_dim=emb_dim, pretrained=True, modality_dropout_prob=modality_dropout_prob)

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: SiameseNetwork (embedding_dim={emb_dim})")
    print(f"  Total parameters:     {total:,}")
    print(f"  Trainable parameters: {trainable:,}")

    # Forward pass with dummy data
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
    print(f"  L2 norm:      {desc_a.norm(dim=1)}")  # should be ~1.0

    # Verify single-pair inference
    with torch.no_grad():
        single = model.get_descriptor(dummy_lidar, dummy_cam)
    print(f"\n  Single inference: {single.shape}")
    print(f"  Matches anchor:  {torch.allclose(desc_a, single)}")
