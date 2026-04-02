"""
Utility to load a pretrained StyleGAN2 checkpoint.
Supports the rosinality / NVIDIA checkpoint format.
"""
import torch
from .generator import StyleGAN2Generator


def load_stylegan2(checkpoint_path: str, size: int = 256,
                   latent_dim: int = 512, n_mlp: int = 8,
                   channel_multiplier: int = 2,
                   device: str = "cuda") -> StyleGAN2Generator:
    """Load StyleGAN2 generator from checkpoint and freeze weights."""
    generator = StyleGAN2Generator(
        size=size,
        style_dim=latent_dim,
        n_mlp=n_mlp,
        channel_multiplier=channel_multiplier,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    # Support both raw state_dict and {'g_ema': ...} format
    if "g_ema" in ckpt:
        state = ckpt["g_ema"]
    elif "generator" in ckpt:
        state = ckpt["generator"]
    else:
        state = ckpt

    generator.load_state_dict(state, strict=False)
    generator.eval()

    for p in generator.parameters():
        p.requires_grad_(False)

    return generator
