"""
Refinement UNet.

Takes the blended output image and the original source face,
restores fine facial details (eyes, skin texture, identity features)
that may have been lost during the GAN synthesis.

Architecture: UNet with skip connections from source face encoder.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 3, padding=1)]
        if norm:
            layers.append(nn.InstanceNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        feat = self.conv(x)
        return self.pool(feat), feat   # (downsampled, skip)


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class RefinementUNet(nn.Module):
    """
    Refines the blended image using source face as a guide.

    Inputs:
        blended: [B, 3, 256, 256]  GAN output after hair transfer
        source:  [B, 3, 256, 256]  original source face

    Output:
        refined: [B, 3, 256, 256]  detail-restored result
    """

    def __init__(self, base_ch: int = 64):
        super().__init__()
        b = base_ch

        # Encoder — takes concatenated blended + source (6 channels)
        self.enc1 = DownBlock(6,      b)       # 256 → 128
        self.enc2 = DownBlock(b,      b * 2)   # 128 → 64
        self.enc3 = DownBlock(b * 2,  b * 4)   # 64  → 32
        self.enc4 = DownBlock(b * 4,  b * 8)   # 32  → 16

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock(b * 8, b * 16),
            ConvBlock(b * 16, b * 16),
        )

        # Decoder with skip connections
        self.dec4 = UpBlock(b * 16, b * 8,  b * 8)
        self.dec3 = UpBlock(b * 8,  b * 4,  b * 4)
        self.dec2 = UpBlock(b * 4,  b * 2,  b * 2)
        self.dec1 = UpBlock(b * 2,  b,      b)

        # Output head — residual correction
        self.out_conv = nn.Sequential(
            nn.Conv2d(b, 3, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, blended: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        x = torch.cat([blended, source], dim=1)   # [B, 6, 256, 256]

        x, s1 = self.enc1(x)
        x, s2 = self.enc2(x)
        x, s3 = self.enc3(x)
        x, s4 = self.enc4(x)

        x = self.bottleneck(x)

        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        # Residual: refine blended rather than generate from scratch
        delta = self.out_conv(x)
        # blended is in [-1,1], delta is in [-1,1], average them
        return ((blended + delta) / 2).clamp(-1, 1)
