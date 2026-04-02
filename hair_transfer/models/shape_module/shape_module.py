"""
Shape Transfer Module.

Operates in F-space (32×32×512 feature maps).
Takes the source F-latent and reference F-latent + hair masks,
and produces a new F-latent with the reference hair shape
transplanted onto the source face structure.

Architecture:
  - Dual-branch attention: source face features + reference hair features
  - Cross-attention to blend hair region from reference into source
  - Mask-guided blending to preserve non-hair regions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels),
        )
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.act(x + self.net(x))


class CrossAttention(nn.Module):
    """
    Spatial cross-attention: query from source, key/value from reference.
    Used to transfer hair structure from reference to source.
    """
    def __init__(self, channels: int, heads: int = 4):
        super().__init__()
        self.heads   = heads
        self.scale   = (channels // heads) ** -0.5
        self.to_q    = nn.Conv2d(channels, channels, 1)
        self.to_k    = nn.Conv2d(channels, channels, 1)
        self.to_v    = nn.Conv2d(channels, channels, 1)
        self.to_out  = nn.Conv2d(channels, channels, 1)

    def forward(self, src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        B, C, H, W = src.shape
        h = self.heads
        d = C // h

        q = self.to_q(src).view(B, h, d, H * W)
        k = self.to_k(ref).view(B, h, d, H * W)
        v = self.to_v(ref).view(B, h, d, H * W)

        attn = torch.einsum('bhdi,bhdj->bhij', q, k) * self.scale
        attn = attn.softmax(dim=-1)

        out = torch.einsum('bhij,bhdj->bhdi', attn, v)
        out = out.reshape(B, C, H, W)
        return self.to_out(out)


class ShapeModule(nn.Module):
    """
    Transfers hair shape from reference to source in F-space.

    Inputs:
        f_src:      [B, 512, 32, 32]  source F-latent
        f_ref:      [B, 512, 32, 32]  reference F-latent
        mask_src:   [B, 1,  32, 32]   source hair mask (0/1)
        mask_ref:   [B, 1,  32, 32]   reference hair mask (0/1)

    Output:
        f_out:      [B, 512, 32, 32]  modified F-latent with new hair shape
    """

    def __init__(self, channels: int = 512, n_res: int = 4):
        super().__init__()

        # Encode source and reference separately
        self.src_enc = nn.Sequential(
            nn.Conv2d(channels + 1, channels, 3, padding=1),
            nn.LeakyReLU(0.2),
            *[ResBlock(channels) for _ in range(n_res // 2)],
        )
        self.ref_enc = nn.Sequential(
            nn.Conv2d(channels + 1, channels, 3, padding=1),
            nn.LeakyReLU(0.2),
            *[ResBlock(channels) for _ in range(n_res // 2)],
        )

        # Cross-attention: pull hair features from reference
        self.cross_attn = CrossAttention(channels)

        # Decode merged features
        self.decoder = nn.Sequential(
            *[ResBlock(channels) for _ in range(n_res // 2)],
            nn.Conv2d(channels, channels, 3, padding=1),
        )

        # Mask-guided blending gate
        self.blend_gate = nn.Sequential(
            nn.Conv2d(channels * 2 + 2, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, f_src, f_ref, mask_src, mask_ref):
        # Resize masks to F-space resolution
        m_src = F.interpolate(mask_src.float(), size=f_src.shape[2:], mode='nearest')
        m_ref = F.interpolate(mask_ref.float(), size=f_ref.shape[2:], mode='nearest')

        # Encode with mask context
        e_src = self.src_enc(torch.cat([f_src, m_src], dim=1))
        e_ref = self.ref_enc(torch.cat([f_ref, m_ref], dim=1))

        # Cross-attend: source queries reference hair
        attended = self.cross_attn(e_src, e_ref)

        # Decode
        decoded = self.decoder(attended)

        # Mask-guided blend: only modify hair region
        gate = self.blend_gate(
            torch.cat([f_src, decoded, m_src, m_ref], dim=1)
        )
        f_out = f_src * (1 - gate) + decoded * gate

        return f_out
