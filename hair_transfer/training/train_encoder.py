"""
Stage 1 — Train the encoder to map real face images into StyleGAN2 W+ space.

Objective: encoder(img) → w_plus such that G(w_plus) ≈ img

Losses:
  - L1 reconstruction
  - Perceptual (VGG)
  - Identity (ArcFace)

Usage:
    python training/train_encoder.py --config configs/train_encoder.yaml
"""
import sys
import argparse
from pathlib import Path

import yaml
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.encoder import HairEncoder
from models.stylegan import load_stylegan2
from utils.dataset_loader import HairTransferDataset
from utils.losses import ReconstructionLoss, PerceptualLoss, IdentityLoss
from training.base_trainer import BaseTrainer


class EncoderTrainer(BaseTrainer):

    def __init__(self, cfg: dict):
        super().__init__(cfg, device=cfg["device"])

        # ── Models ────────────────────────────────────────────────────────────
        self.generator = load_stylegan2(
            cfg["stylegan"]["checkpoint"],
            size=cfg["stylegan"]["size"],
            latent_dim=cfg["stylegan"]["latent_dim"],
            n_mlp=cfg["stylegan"]["n_mlp"],
            channel_multiplier=cfg["stylegan"]["channel_multiplier"],
            device=self.device,
        )

        self.encoder = HairEncoder(
            n_styles=cfg["encoder"]["n_styles"],
            style_dim=cfg["stylegan"]["latent_dim"],
        ).to(self.device)

        # ── Losses ────────────────────────────────────────────────────────────
        lw = cfg["loss"]
        self.l1   = ReconstructionLoss()
        self.perc = PerceptualLoss().to(self.device)
        self.id_  = IdentityLoss(device=self.device)
        self.lw   = lw

        # ── Optimizer ─────────────────────────────────────────────────────────
        self.opt = self.make_optimizer(self.encoder.parameters())

        # ── Data ──────────────────────────────────────────────────────────────
        dataset = HairTransferDataset(
            root=cfg["data"]["root"],
            image_size=cfg["data"]["image_size"],
        )
        self.loader = DataLoader(
            dataset,
            batch_size=cfg["data"]["batch_size"],
            shuffle=True,
            num_workers=cfg["data"]["num_workers"],
            pin_memory=cfg["data"].get("pin_memory", False),
        )

    def train(self):
        max_epochs = self.cfg["train"]["max_epochs"]
        save_every = self.cfg["train"]["save_every"]

        self.load({"encoder": self.encoder}, {"opt": self.opt})

        for epoch in range(self.epoch, max_epochs):
            self.epoch = epoch
            for batch in self.loader:
                src_img = batch["src_img"].to(self.device)

                with autocast('cuda', enabled=self.cfg["train"]["mixed_precision"]):
                    # Encode → W+
                    w_plus, _ = self.encoder(src_img)

                    # Decode with frozen StyleGAN2
                    recon, _ = self.generator(
                        [w_plus], input_is_latent=True
                    )

                    # Losses
                    loss_l1   = self.l1(recon, src_img)
                    loss_perc = self.perc(recon, src_img)
                    loss_id   = self.id_(recon, src_img)

                    loss = (
                        self.lw["lambda_l1"]         * loss_l1 +
                        self.lw["lambda_perceptual"]  * loss_perc +
                        self.lw["lambda_identity"]    * loss_id
                    )

                self.step_optimizer(self.opt, loss)
                self.log_step({
                    "l1": loss_l1.item(),
                    "perc": loss_perc.item(),
                    "id": loss_id.item(),
                    "total": loss.item(),
                })
                self.step += 1

            if (epoch + 1) % save_every == 0:
                self.save({"encoder": self.encoder}, {"opt": self.opt},
                          tag=f"epoch_{epoch+1}")

        self.save({"encoder": self.encoder}, {"opt": self.opt}, tag="final")
        print("Stage 1 training complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_encoder.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    trainer = EncoderTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
