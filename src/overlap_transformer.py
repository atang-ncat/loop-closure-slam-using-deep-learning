"""
OverlapTransformer — Purpose-built LiDAR Range Image Feature Extractor
======================================================================
Adapted from: Chen et al., "OverlapTransformer: An Efficient and
Rotation-Invariant Transformer Network for LiDAR-Based Place
Recognition" (IROS 2022).

Architecture:
    Range Image (1, 128, 1024)
        ↓ Vertical CNN (compress height 128 → 1)
        ↓ 1-layer Transformer (azimuth attention)
        ↓ NetVLAD aggregation
        → 256-d global descriptor

Original: 64×900 input, adapted here for 128×1024.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── NetVLAD Aggregation ─────────────────────────────────────────────────────

class GatingContext(nn.Module):
    """Context gating for NetVLAD output."""

    def __init__(self, dim: int, add_batch_norm: bool = True):
        super().__init__()
        self.gating_weights = nn.Parameter(
            torch.randn(dim, dim) * 1 / math.sqrt(dim)
        )
        self.sigmoid = nn.Sigmoid()

        if add_batch_norm:
            self.gating_biases = None
            self.bn1 = nn.BatchNorm1d(dim)
        else:
            self.gating_biases = nn.Parameter(
                torch.randn(dim) * 1 / math.sqrt(dim)
            )
            self.bn1 = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = torch.matmul(x, self.gating_weights)
        if self.bn1 is not None:
            gates = self.bn1(gates)
        else:
            gates = gates + self.gating_biases
        return x * self.sigmoid(gates)


class NetVLADLoupe(nn.Module):
    """
    NetVLAD with context gating for global descriptor aggregation.

    Takes spatial features (B, C, W, 1) and produces a compact
    global descriptor via soft-assignment to learned cluster centers.
    """

    def __init__(
        self,
        feature_size: int,
        max_samples: int,
        cluster_size: int = 64,
        output_dim: int = 256,
        gating: bool = True,
        add_batch_norm: bool = False,
    ):
        super().__init__()
        self.feature_size = feature_size
        self.max_samples = max_samples
        self.cluster_size = cluster_size
        self.output_dim = output_dim
        self.softmax = nn.Softmax(dim=-1)

        self.cluster_weights = nn.Parameter(
            torch.randn(feature_size, cluster_size) / math.sqrt(feature_size)
        )
        self.cluster_weights2 = nn.Parameter(
            torch.randn(1, feature_size, cluster_size) / math.sqrt(feature_size)
        )
        self.hidden1_weights = nn.Parameter(
            torch.randn(cluster_size * feature_size, output_dim) / math.sqrt(feature_size)
        )

        if add_batch_norm:
            self.cluster_biases = None
            self.bn1 = nn.BatchNorm1d(cluster_size)
        else:
            self.cluster_biases = nn.Parameter(
                torch.randn(cluster_size) / math.sqrt(feature_size)
            )
            self.bn1 = None

        self.bn2 = nn.BatchNorm1d(output_dim)
        self.gating = GatingContext(output_dim, add_batch_norm=add_batch_norm) if gating else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, W, 1) → (B, output_dim)"""
        x = x.transpose(1, 3).contiguous()  # (B, 1, W, C)
        x = x.view(-1, self.max_samples, self.feature_size)  # (B, W, C)

        activation = torch.matmul(x, self.cluster_weights)  # (B, W, K)
        if self.bn1 is not None:
            activation = self.bn1(activation.view(-1, self.cluster_size))
            activation = activation.view(-1, self.max_samples, self.cluster_size)
        else:
            activation = activation + self.cluster_biases
        activation = self.softmax(activation)

        a_sum = activation.sum(-2, keepdim=True)
        a = a_sum * self.cluster_weights2

        activation = activation.transpose(2, 1)  # (B, K, W)
        vlad = torch.matmul(activation, x)  # (B, K, C)
        vlad = vlad.transpose(2, 1) - a  # (B, C, K)

        vlad = F.normalize(vlad, dim=1, p=2)
        vlad = vlad.reshape(-1, self.cluster_size * self.feature_size)
        vlad = F.normalize(vlad, dim=1, p=2)

        vlad = torch.matmul(vlad, self.hidden1_weights)  # (B, output_dim)

        if self.gating is not None:
            vlad = self.gating(vlad)

        return vlad


# ── OverlapTransformer Backbone ──────────────────────────────────────────────

class OverlapTransformerBackbone(nn.Module):
    """
    OverlapTransformer feature extractor for LiDAR range images.

    Vertical CNN compresses height dimension while preserving azimuth.
    A transformer encoder captures long-range azimuth dependencies.
    NetVLAD aggregation produces a compact global descriptor.

    Adapted for 128×1024 range images (original paper uses 64×900).
    """

    def __init__(
        self,
        height: int = 128,
        width: int = 1024,
        channels: int = 1,
        output_dim: int = 256,
        use_transformer: bool = True,
    ):
        super().__init__()
        self.use_transformer = use_transformer
        self.feat_dim = output_dim

        BN = nn.BatchNorm2d

        # Vertical CNN: compress height 128 → 1
        # All kernels are (k, 1) — only compress vertically
        self.conv1 = nn.Conv2d(channels, 16, kernel_size=(5, 1), stride=(1, 1), bias=False)
        self.bn1 = BN(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 1), stride=(2, 1), bias=False)
        self.bn2 = BN(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 1), stride=(2, 1), bias=False)
        self.bn3 = BN(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(3, 1), stride=(2, 1), bias=False)
        self.bn4 = BN(64)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=(3, 1), stride=(2, 1), bias=False)
        self.bn5 = BN(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=(3, 1), stride=(2, 1), bias=False)
        self.bn6 = BN(128)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(2, 1), bias=False)
        self.bn7 = BN(128)

        self.relu = nn.ReLU(inplace=True)

        # Ensure height=1 regardless of exact CNN arithmetic
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, width))

        # Transformer for azimuth attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=4, dim_feedforward=1024,
            activation="relu", batch_first=False, dropout=0.0,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # Pre/post transformer projections
        self.conv_pre_tf = nn.Conv2d(128, 256, kernel_size=(1, 1), bias=False)
        self.bn_pre_tf = BN(256)
        self.conv_post_tf = nn.Conv2d(512, 1024, kernel_size=(1, 1), bias=False)
        self.bn_post_tf = BN(1024)

        # NetVLAD aggregation → output_dim
        self.net_vlad = NetVLADLoupe(
            feature_size=1024, max_samples=width,
            cluster_size=64, output_dim=output_dim,
            gating=True, add_batch_norm=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, 128, 1024) → (B, output_dim)"""
        # Vertical CNN compression
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))
        out = self.relu(self.bn4(self.conv4(out)))
        out = self.relu(self.bn5(self.conv5(out)))
        out = self.relu(self.bn6(self.conv6(out)))
        out = self.relu(self.bn7(self.conv7(out)))

        # Ensure height=1
        out = self.adaptive_pool(out)  # (B, 128, 1, W)

        # Transpose to (B, 128, W, 1) for azimuth processing
        out = out.permute(0, 1, 3, 2)  # (B, 128, W, 1)

        # Pre-transformer projection
        out_proj = self.relu(self.bn_pre_tf(self.conv_pre_tf(out)))  # (B, 256, W, 1)

        if self.use_transformer:
            # Transformer: (W, B, 256)
            tf_in = out_proj.squeeze(3).permute(2, 0, 1)  # (W, B, 256)
            tf_out = self.transformer_encoder(tf_in)  # (W, B, 256)
            tf_out = tf_out.permute(1, 2, 0).unsqueeze(3)  # (B, 256, W, 1)

            # Concat pre+post transformer features
            combined = torch.cat([out_proj, tf_out], dim=1)  # (B, 512, W, 1)
        else:
            combined = torch.cat([out_proj, out_proj], dim=1)

        # Post-transformer projection
        combined = self.relu(self.bn_post_tf(self.conv_post_tf(combined)))  # (B, 1024, W, 1)
        combined = F.normalize(combined, dim=1)

        # NetVLAD aggregation → (B, output_dim)
        desc = self.net_vlad(combined)
        desc = F.normalize(desc, dim=1)

        return desc


# ── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = OverlapTransformerBackbone(height=128, width=1024, channels=1)

    total = sum(p.numel() for p in model.parameters())
    print(f"OverlapTransformer: {total:,} parameters, output_dim={model.feat_dim}")

    x = torch.randn(2, 1, 128, 1024)
    model.eval()
    with torch.no_grad():
        out = model(x)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  L2 norm: {out.norm(dim=1)}")
