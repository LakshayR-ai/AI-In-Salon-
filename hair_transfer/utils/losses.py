"""
Loss functions for hair transfer training.

  - ReconstructionLoss  : L1 pixel loss
  - PerceptualLoss      : VGG16 feature matching
  - IdentityLoss        : ArcFace cosine similarity
  - CLIPLoss            : CLIP image-space similarity
  - AdversarialLoss     : LSGAN hinge loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ── L1 ────────────────────────────────────────────────────────────────────────

class ReconstructionLoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(pred, target)


# ── Perceptual (VGG16) ────────────────────────────────────────────────────────

class PerceptualLoss(nn.Module):
    """
    Multi-scale VGG16 feature matching loss.
    Extracts features from relu1_2, relu2_2, relu3_3, relu4_3.
    """
    LAYERS = {"3": 1.0, "8": 1.0, "15": 1.0, "22": 0.5}

    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.slices = nn.ModuleList()
        prev = 0
        for end in [3, 8, 15, 22]:
            self.slices.append(nn.Sequential(*list(vgg.children())[prev:end + 1]))
            prev = end + 1
        for p in self.parameters():
            p.requires_grad_(False)

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def _preprocess(self, x):
        x = x * 0.5 + 0.5          # [-1,1] → [0,1]
        return (x - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred   = self._preprocess(pred)
        target = self._preprocess(target)
        loss   = 0.0
        weights = [1.0, 1.0, 1.0, 0.5]
        for w, sl in zip(weights, self.slices):
            pred   = sl(pred)
            target = sl(target)
            loss  += w * F.l1_loss(pred, target)
        return loss


# ── Identity (ArcFace) ────────────────────────────────────────────────────────

class IdentityLoss(nn.Module):
    """
    Cosine similarity loss in ArcFace embedding space.
    Penalises identity change between source and output.

    Requires a pretrained ArcFace model (iResNet50).
    If checkpoint is None, falls back to a random network (for testing).
    """

    def __init__(self, checkpoint: str | None = None, device: str = "cuda"):
        super().__init__()
        self.net = self._build_arcface(checkpoint, device)
        for p in self.net.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _build_arcface(checkpoint, device):
        # Minimal iResNet50 — replace with full ArcFace if checkpoint available
        net = models.resnet50(weights=None)
        net.fc = nn.Linear(2048, 512)
        if checkpoint:
            state = torch.load(checkpoint, map_location=device)
            net.load_state_dict(state, strict=False)
        return net.to(device).eval()

    def _embed(self, x):
        x = F.interpolate(x, size=(112, 112), mode='bilinear', align_corners=False)
        x = x * 0.5 + 0.5
        feat = self.net(x)
        return F.normalize(feat, dim=1)

    def forward(self, pred: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        e_pred = self._embed(pred)
        e_src  = self._embed(source)
        return (1 - (e_pred * e_src).sum(dim=1)).mean()


# ── CLIP ──────────────────────────────────────────────────────────────────────

class CLIPLoss(nn.Module):
    """
    Directional CLIP loss: the edit direction in CLIP space should match
    the direction from source hair to reference hair.
    """

    def __init__(self, clip_model: str = "ViT-B/32", device: str = "cuda"):
        super().__init__()
        import clip
        self.model, _ = clip.load(clip_model, device=device)
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.register_buffer("mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1,3,1,1))
        self.register_buffer("std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1,3,1,1))

    def _encode(self, x):
        x = F.interpolate(x * 0.5 + 0.5, size=(224, 224),
                          mode='bilinear', align_corners=False)
        mean = self.mean.to(x.device)
        std  = self.std.to(x.device)
        x = (x - mean) / std
        return F.normalize(self.model.encode_image(x).float(), dim=1)

    def forward(self, pred, source, reference) -> torch.Tensor:
        """
        Directional loss: direction(source→pred) ≈ direction(source→reference)
        """
        e_pred = self._encode(pred)
        e_src  = self._encode(source)
        e_ref  = self._encode(reference)

        dir_edit = F.normalize(e_pred - e_src, dim=1)
        dir_ref  = F.normalize(e_ref  - e_src, dim=1)
        return (1 - (dir_edit * dir_ref).sum(dim=1)).mean()


# ── Adversarial (LSGAN) ───────────────────────────────────────────────────────

class AdversarialLoss(nn.Module):
    """Least-squares GAN loss (Mao et al. 2017)."""

    def discriminator_loss(self, real_pred, fake_pred) -> torch.Tensor:
        return (F.mse_loss(real_pred, torch.ones_like(real_pred)) +
                F.mse_loss(fake_pred, torch.zeros_like(fake_pred))) * 0.5

    def generator_loss(self, fake_pred) -> torch.Tensor:
        return F.mse_loss(fake_pred, torch.ones_like(fake_pred))

    def forward(self, fake_pred) -> torch.Tensor:
        return self.generator_loss(fake_pred)
