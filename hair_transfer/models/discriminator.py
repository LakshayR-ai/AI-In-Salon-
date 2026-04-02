"""
PatchGAN discriminator for adversarial training.
Classifies 70×70 image patches as real or fake.
"""
import torch
import torch.nn as nn


class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch: int = 3, base_ch: int = 64, n_layers: int = 3):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, base_ch, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        ]
        ch = base_ch
        for _ in range(n_layers - 1):
            layers += [
                nn.Conv2d(ch, ch * 2, 4, stride=2, padding=1),
                nn.InstanceNorm2d(ch * 2),
                nn.LeakyReLU(0.2),
            ]
            ch *= 2
        layers += [
            nn.Conv2d(ch, ch * 2, 4, stride=1, padding=1),
            nn.InstanceNorm2d(ch * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ch * 2, 1, 4, stride=1, padding=1),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
