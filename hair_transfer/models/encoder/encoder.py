"""
pSp-style encoder: maps a real face image → W+ latent space of StyleGAN2.

Architecture:
  - ResNet50 feature pyramid backbone
  - 14 style heads (one per StyleGAN layer at 256px)
  - Each head outputs a 512-dim style vector
  - Together they form W+ ∈ R^{14×512}
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


class StyleHead(nn.Module):
    """Maps a feature map to one 512-dim style vector."""
    def __init__(self, in_channels: int, style_dim: int = 512):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, style_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(style_dim, style_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(x))


class HairEncoder(nn.Module):
    """
    Encodes a 256×256 face image into W+ latent space.

    Returns:
        w_plus: [B, n_styles, 512]  — W+ latent codes
        features: dict of intermediate feature maps for F-space injection
    """

    def __init__(self, n_styles: int = 14, style_dim: int = 512):
        super().__init__()
        self.n_styles  = n_styles
        self.style_dim = style_dim

        # ── Backbone (ResNet50, pretrained on ImageNet) ──────────────────────
        backbone = resnet50(weights="IMAGENET1K_V1")

        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1,
                                    backbone.relu, backbone.maxpool)  # /4
        self.layer1 = backbone.layer1   # /4,  256ch
        self.layer2 = backbone.layer2   # /8,  512ch
        self.layer3 = backbone.layer3   # /16, 1024ch
        self.layer4 = backbone.layer4   # /32, 2048ch

        # Enable gradient checkpointing to save VRAM (~30% less memory)
        self.use_checkpoint = True

        # ── Style heads — coarse to fine ─────────────────────────────────────
        # Coarse styles (4–8px) from layer4
        coarse_heads = n_styles // 3
        # Medium styles (16–32px) from layer3
        medium_heads = n_styles // 3
        # Fine styles (64–256px) from layer2
        fine_heads   = n_styles - coarse_heads - medium_heads

        self.coarse_heads = nn.ModuleList(
            [StyleHead(2048, style_dim) for _ in range(coarse_heads)]
        )
        self.medium_heads = nn.ModuleList(
            [StyleHead(1024, style_dim) for _ in range(medium_heads)]
        )
        self.fine_heads = nn.ModuleList(
            [StyleHead(512, style_dim) for _ in range(fine_heads)]
        )

        # ── F-space projection (32×32 feature map) ───────────────────────────
        self.f_proj = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, padding=1),
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, 3, 256, 256] normalized to [-1, 1]
        Returns:
            w_plus:   [B, n_styles, 512]
            f_latent: [B, 512, 32, 32]  — F-space feature map
        """
        from torch.utils.checkpoint import checkpoint as ckpt
        if self.use_checkpoint and self.training:
            f0 = ckpt(self.layer0, x,  use_reentrant=False)
            f1 = ckpt(self.layer1, f0, use_reentrant=False)
            f2 = ckpt(self.layer2, f1, use_reentrant=False)
            f3 = ckpt(self.layer3, f2, use_reentrant=False)
            f4 = ckpt(self.layer4, f3, use_reentrant=False)
        else:
            f0 = self.layer0(x)
            f1 = self.layer1(f0)
            f2 = self.layer2(f1)
            f3 = self.layer3(f2)
            f4 = self.layer4(f3)

        coarse = [h(f4) for h in self.coarse_heads]
        medium = [h(f3) for h in self.medium_heads]
        fine   = [h(f2) for h in self.fine_heads]

        w_plus   = torch.stack(coarse + medium + fine, dim=1)  # [B, n_styles, 512]
        f_latent = self.f_proj(f3)                              # [B, 512, 16, 16]
        f_latent = F.interpolate(f_latent, size=(32, 32),
                                 mode='bilinear', align_corners=False)

        return w_plus, f_latent
