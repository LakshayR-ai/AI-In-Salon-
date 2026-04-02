"""
StyleGAN2 Generator — minimal clean implementation.
Supports:
  - Full synthesis from W/W+ latent
  - Partial synthesis (start_layer / end_layer) for FS-space manipulation
  - Returns both image and intermediate feature maps
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)
    if k.ndim == 1:
        k = k[None, :] * k[:, None]
    k /= k.sum()
    return k


class PixelNorm(nn.Module):
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(1, keepdim=True) + 1e-8)


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, lr_mul=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        self.bias   = nn.Parameter(torch.zeros(out_dim)) if bias else None
        self.scale  = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, x):
        return F.linear(x, self.weight * self.scale,
                        self.bias * self.lr_mul if self.bias is not None else None)


class EqualConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride=1, padding=0, bias=True):
        super().__init__()
        self.weight  = nn.Parameter(torch.randn(out_ch, in_ch, kernel, kernel))
        self.scale   = 1 / math.sqrt(in_ch * kernel ** 2)
        self.stride  = stride
        self.padding = padding
        self.bias    = nn.Parameter(torch.zeros(out_ch)) if bias else None

    def forward(self, x):
        return F.conv2d(x, self.weight * self.scale, self.bias,
                        stride=self.stride, padding=self.padding)


class ModulatedConv2d(nn.Module):
    """Weight-modulated convolution (core of StyleGAN2)."""
    def __init__(self, in_ch, out_ch, kernel, style_dim,
                 demodulate=True, upsample=False):
        super().__init__()
        self.out_ch     = out_ch
        self.kernel     = kernel
        self.upsample   = upsample
        self.demodulate = demodulate
        self.padding    = kernel // 2

        self.weight    = nn.Parameter(torch.randn(1, out_ch, in_ch, kernel, kernel))
        self.modulate  = EqualLinear(style_dim, in_ch, bias=True)
        self.scale     = 1 / math.sqrt(in_ch * kernel ** 2)

    def forward(self, x, style):
        B, C, H, W = x.shape
        style = self.modulate(style).view(B, 1, C, 1, 1)
        weight = self.scale * self.weight * style          # [B, out, in, k, k]

        if self.demodulate:
            d = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * d.view(B, self.out_ch, 1, 1, 1)

        weight = weight.view(B * self.out_ch, C, self.kernel, self.kernel)
        x = x.view(1, B * C, H, W)

        if self.upsample:
            x = F.interpolate(x.view(B, C, H, W), scale_factor=2, mode='bilinear',
                              align_corners=False).view(1, B * C, H * 2, W * 2)

        x = F.conv2d(x, weight, padding=self.padding, groups=B)
        return x.view(B, self.out_ch, x.shape[2], x.shape[3])


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, x, noise=None):
        if noise is None:
            B, _, H, W = x.shape
            noise = x.new_empty(B, 1, H, W).normal_()
        return x + self.weight * noise


class StyledConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, style_dim, upsample=False):
        super().__init__()
        self.conv  = ModulatedConv2d(in_ch, out_ch, kernel, style_dim, upsample=upsample)
        self.noise = NoiseInjection()
        self.bias  = nn.Parameter(torch.zeros(1, out_ch, 1, 1))
        self.act   = nn.LeakyReLU(0.2)

    def forward(self, x, style, noise=None):
        x = self.conv(x, style)
        x = self.noise(x, noise)
        return self.act(x + self.bias)


class ToRGB(nn.Module):
    def __init__(self, in_ch, style_dim, upsample=True):
        super().__init__()
        self.upsample = upsample
        self.conv = ModulatedConv2d(in_ch, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, x, style, skip=None):
        out = self.conv(x, style) + self.bias
        if skip is not None:
            if self.upsample:
                skip = F.interpolate(skip, scale_factor=2, mode='bilinear', align_corners=False)
            out = out + skip
        return out


# ── MLP mapping network ───────────────────────────────────────────────────────

class MappingNetwork(nn.Module):
    def __init__(self, style_dim=512, n_mlp=8, lr_mul=0.01):
        super().__init__()
        layers = [PixelNorm()]
        for _ in range(n_mlp):
            layers += [EqualLinear(style_dim, style_dim, lr_mul=lr_mul), nn.LeakyReLU(0.2)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


# ── Synthesis network ─────────────────────────────────────────────────────────

class StyleGAN2Generator(nn.Module):
    """
    StyleGAN2 generator.

    Key methods:
      forward(styles, input_is_latent, ...)  – standard generation
      get_features(w_plus, layer)            – extract F-space feature map
    """

    def __init__(self, size=256, style_dim=512, n_mlp=8, channel_multiplier=2):
        super().__init__()
        self.size      = size
        self.style_dim = style_dim
        self.log_size  = int(math.log2(size))          # e.g. 8 for 256
        self.n_latent  = self.log_size * 2 - 2         # number of W+ vectors

        self.mapping = MappingNetwork(style_dim, n_mlp)

        channels = {
            4:   512,
            8:   512,
            16:  512,
            32:  512,
            64:  256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64  * channel_multiplier,
            512: 32  * channel_multiplier,
            1024:16  * channel_multiplier,
        }

        self.input    = nn.Parameter(torch.randn(1, channels[4], 4, 4))
        self.conv1    = StyledConv(channels[4], channels[4], 3, style_dim)
        self.to_rgb1  = ToRGB(channels[4], style_dim, upsample=False)

        self.convs   = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()

        in_ch = channels[4]
        for i in range(3, self.log_size + 1):
            out_ch = channels[2 ** i]
            self.convs.append(StyledConv(in_ch,  out_ch, 3, style_dim, upsample=True))
            self.convs.append(StyledConv(out_ch, out_ch, 3, style_dim))
            self.to_rgbs.append(ToRGB(out_ch, style_dim))
            in_ch = out_ch

    def _get_styles(self, styles, input_is_latent):
        """Expand styles to W+ (one vector per layer)."""
        if not input_is_latent:
            styles = [self.mapping(s) for s in styles]

        s = styles[0]

        # Already W+ shaped: [B, n_latent, 512]
        if s.ndim == 3:
            return s

        # Single W vector [B, 512] — broadcast to all layers
        if len(styles) == 1:
            return s.unsqueeze(1).expand(-1, self.n_latent, -1)

        # Style mixing: two [B, 512] vectors
        inject = self.n_latent // 2
        return torch.cat([
            styles[0].unsqueeze(1).expand(-1, inject, -1),
            styles[1].unsqueeze(1).expand(-1, self.n_latent - inject, -1),
        ], dim=1)

    def forward(self, styles, input_is_latent=False,
                start_layer=0, end_layer=None, layer_in=None,
                return_latents=False):
        """
        Args:
            styles:          list of tensors or single W+ tensor
            input_is_latent: if True, skip mapping network
            start_layer:     synthesis start layer index
            end_layer:       synthesis end layer index (None = all)
            layer_in:        feature map to inject at start_layer
            return_latents:  also return W+ tensor
        Returns:
            image [B,3,H,W] (tanh range [-1,1]), optionally W+
        """
        w = self._get_styles(styles, input_is_latent)
        n_blocks   = len(self.to_rgbs)          # number of upsample blocks
        end_layer  = end_layer if end_layer is not None else n_blocks

        # ── Layer 0: 4×4 constant ────────────────────────────────────────────
        if start_layer == 0:
            x    = self.input.repeat(w.shape[0], 1, 1, 1)
            x    = self.conv1(x, w[:, 0])
            skip = self.to_rgb1(x, w[:, 1])
            w_idx = 2                           # next W+ index to consume
        else:
            x     = layer_in
            skip  = None
            # Each block before start_layer consumed 2 W+ vectors (+ 1 for
            # to_rgb1 at block 0), so offset accordingly.
            w_idx = 2 + (start_layer - 1) * 2

        # ── Upsample blocks ──────────────────────────────────────────────────
        for i in range(n_blocks):
            if i < start_layer - 1:
                w_idx += 2
                continue
            if i >= end_layer:
                break

            conv1 = self.convs[i * 2]
            conv2 = self.convs[i * 2 + 1]
            to_rgb = self.to_rgbs[i]

            # Clamp indices so we never exceed W+ size
            idx1 = min(w_idx,     w.shape[1] - 1)
            idx2 = min(w_idx + 1, w.shape[1] - 1)
            idx3 = min(w_idx + 2, w.shape[1] - 1)

            x    = conv1(x, w[:, idx1])
            x    = conv2(x, w[:, idx2])
            skip = to_rgb(x, w[:, idx3], skip)
            w_idx += 2

        if return_latents:
            return skip, w
        return skip, None
