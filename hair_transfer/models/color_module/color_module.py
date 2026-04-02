"""
Color/Style Transfer Module.

Operates in S-space (W+ latent, layers 6–14).
Uses CLIP image embeddings to guide color transfer from reference hair
to source hair while preserving face identity in early layers.

Architecture:
  - CLIP ViT-B/32 encodes both source and reference images
  - Modulation network blends S-space style vectors
  - Only modifies layers 6–14 (fine detail / color layers)
  - Layers 0–5 (coarse structure) are preserved from source
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModulationBlock(nn.Module):
    """
    Modulates source style vector using CLIP embedding of reference.
    Inspired by HairFastGAN's ClipBlendingModel.
    """
    def __init__(self, style_dim: int = 512, clip_dim: int = 512,
                 n_layers: int = 8):
        super().__init__()
        self.n_layers  = n_layers
        self.style_dim = style_dim

        # Condition on: source style + CLIP(source face) + CLIP(reference hair)
        cond_dim = style_dim + clip_dim * 2

        self.norm   = nn.LayerNorm([n_layers, style_dim])
        self.gamma  = nn.Sequential(
            nn.Linear(cond_dim, 1024), nn.LayerNorm(1024),
            nn.LeakyReLU(0.2), nn.Linear(1024, style_dim),
        )
        self.beta   = nn.Sequential(
            nn.Linear(cond_dim, 1024), nn.LayerNorm(1024),
            nn.LeakyReLU(0.2), nn.Linear(1024, style_dim),
        )
        self.fc     = nn.Linear(style_dim, style_dim)
        self.act    = nn.LeakyReLU(0.2)

    def forward(self, s_src, clip_src, clip_ref):
        """
        Args:
            s_src:    [B, n_layers, 512]  source style vectors
            clip_src: [B, 512]            CLIP embedding of source face
            clip_ref: [B, 512]            CLIP embedding of reference hair
        Returns:
            s_out:    [B, n_layers, 512]  modified style vectors
        """
        # Expand CLIP embeddings to match layer dimension
        c_src = clip_src.unsqueeze(1).expand(-1, self.n_layers, -1)
        c_ref = clip_ref.unsqueeze(1).expand(-1, self.n_layers, -1)
        cond  = torch.cat([s_src, c_src, c_ref], dim=-1)  # [B, L, cond_dim]

        x = self.fc(s_src)
        x = self.norm(x)
        gamma = self.gamma(cond)
        beta  = self.beta(cond)
        return self.act(x * (1 + gamma) + beta)


class ColorModule(nn.Module):
    """
    Transfers hair color from reference to source via S-space modulation.

    Inputs:
        w_src:    [B, n_styles, 512]  source W+ latent
        img_src:  [B, 3, 256, 256]   source face image (for CLIP)
        img_ref:  [B, 3, 256, 256]   reference hair image (for CLIP)

    Output:
        w_out:    [B, n_styles, 512]  W+ with color transferred in layers 6–14
    """

    COLOR_LAYERS_START = 6   # only modify layers 6 onward

    def __init__(self, n_styles: int = 14, style_dim: int = 512,
                 clip_model: str = "ViT-B/32"):
        super().__init__()
        self.n_styles  = n_styles
        self.style_dim = style_dim
        self.n_color   = n_styles - self.COLOR_LAYERS_START

        # Load CLIP
        import clip
        self.clip, self.clip_preprocess = clip.load(clip_model, device="cpu")
        for p in self.clip.parameters():
            p.requires_grad_(False)

        # CLIP output dim for ViT-B/32 is 512
        clip_dim = 512

        self.modulation = ModulationBlock(
            style_dim=style_dim,
            clip_dim=clip_dim,
            n_layers=self.n_color,
        )

        # Normalisation for CLIP input
        self.register_buffer("clip_mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1,3,1,1))
        self.register_buffer("clip_std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1,3,1,1))

    def _clip_embed(self, img: torch.Tensor) -> torch.Tensor:
        """Encode image with CLIP. img in [-1,1]."""
        x = (img * 0.5 + 0.5).clamp(0, 1)
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        mean = self.clip_mean.to(x.device)
        std  = self.clip_std.to(x.device)
        x = (x - mean) / std
        with torch.no_grad():
            feat = self.clip.encode_image(x).float()
        return feat  # [B, 512]

    def forward(self, w_src, img_src, img_ref):
        clip_src = self._clip_embed(img_src)
        clip_ref = self._clip_embed(img_ref)

        s_color = w_src[:, self.COLOR_LAYERS_START:]   # [B, n_color, 512]
        s_mod   = self.modulation(s_color, clip_src, clip_ref)

        w_out = torch.cat([w_src[:, :self.COLOR_LAYERS_START], s_mod], dim=1)
        return w_out
