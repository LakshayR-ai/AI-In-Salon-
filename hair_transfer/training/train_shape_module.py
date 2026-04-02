"""
Stage 2 — Train the Shape Transfer Module (F-space).

Objective: given source and reference F-latents + hair masks,
produce a modified F-latent that has the reference hair shape
on the source face.

Losses:
  - L1 reconstruction (hair region)
  - Perceptual
  - Identity (non-hair region preserved)
  - Adversarial

Usage:
    python training/train_shape_module.py --config configs/train_shape.yaml
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
from models.shape_module import ShapeModule
from models.discriminator import PatchDiscriminator
from models.stylegan import load_stylegan2
from utils.dataset_loader import HairTransferDataset
from utils.losses import (ReconstructionLoss, PerceptualLoss,
                           IdentityLoss, AdversarialLoss)
from training.base_trainer import BaseTrainer


class ShapeTrainer(BaseTrainer):

    def __init__(self, cfg: dict):
        super().__init__(cfg, device=cfg["device"])

        self.generator = load_stylegan2(
            cfg["stylegan"]["checkpoint"],
            size=cfg["stylegan"]["size"],
            latent_dim=cfg["stylegan"]["latent_dim"],
            device=self.device,
        )

        self.encoder = HairEncoder(
            n_styles=cfg["encoder"]["n_styles"],
        ).to(self.device)
        # Load pretrained encoder from Stage 1
        enc_ckpt = Path(cfg["train"]["output_dir"]) / "encoder" / "ckpt_final.pt"
        if enc_ckpt.exists():
            state = torch.load(enc_ckpt, map_location=self.device)
            self.encoder.load_state_dict(state["models"]["encoder"])
            print(f"Loaded encoder from {enc_ckpt}")
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()

        self.shape_module = ShapeModule(channels=512).to(self.device)
        self.discriminator = PatchDiscriminator().to(self.device)

        lw = cfg["loss"]
        self.l1   = ReconstructionLoss()
        self.perc = PerceptualLoss().to(self.device)
        self.id_  = IdentityLoss(device=self.device)
        self.adv  = AdversarialLoss()
        self.lw   = lw

        self.opt_g = self.make_optimizer(self.shape_module.parameters())
        self.opt_d = self.make_optimizer(self.discriminator.parameters())

        dataset = HairTransferDataset(
            root=cfg["data"]["root"],
            image_size=cfg["data"]["image_size"],
        )
        self.loader = DataLoader(
            dataset,
            batch_size=cfg["data"]["batch_size"],
            shuffle=True,
            num_workers=cfg["data"]["num_workers"],
            pin_memory=True,
        )

    def train(self):
        max_epochs = self.cfg["train"]["max_epochs"]
        save_every = self.cfg["train"]["save_every"]

        self.load(
            {"shape": self.shape_module, "disc": self.discriminator},
            {"opt_g": self.opt_g, "opt_d": self.opt_d},
        )

        for epoch in range(self.epoch, max_epochs):
            self.epoch = epoch
            for batch in self.loader:
                src_img  = batch["src_img"].to(self.device)
                ref_img  = batch["ref_img"].to(self.device)
                src_mask = batch["src_mask"].to(self.device)
                ref_mask = batch["ref_mask"].to(self.device)

                with autocast('cuda', enabled=self.cfg["train"]["mixed_precision"]):
                    with torch.no_grad():
                        w_src, f_src = self.encoder(src_img)
                        w_ref, f_ref = self.encoder(ref_img)

                    # Shape transfer in F-space
                    f_out = self.shape_module(f_src, f_ref, src_mask, ref_mask)

                    # Decode: inject modified F at layer 3
                    out_img, _ = self.generator(
                        [w_src], input_is_latent=True,
                        start_layer=4, end_layer=None, layer_in=f_out,
                    )

                    # ── Discriminator step ────────────────────────────────────
                    real_pred = self.discriminator(src_img.detach())
                    fake_pred = self.discriminator(out_img.detach())
                    loss_d    = self.adv.discriminator_loss(real_pred, fake_pred)

                self.step_optimizer(self.opt_d, loss_d)

                with autocast('cuda', enabled=self.cfg["train"]["mixed_precision"]):
                    fake_pred_g = self.discriminator(out_img)

                    # Hair region L1
                    hair_out = out_img * ref_mask
                    hair_ref = ref_img * ref_mask
                    loss_l1  = self.l1(hair_out, hair_ref)

                    loss_perc = self.perc(out_img, src_img)
                    loss_id   = self.id_(out_img, src_img)
                    loss_adv  = self.adv(fake_pred_g)

                    loss_g = (
                        self.lw["lambda_l1"]        * loss_l1 +
                        self.lw["lambda_perceptual"] * loss_perc +
                        self.lw["lambda_identity"]   * loss_id +
                        self.lw["lambda_adv"]        * loss_adv
                    )

                self.step_optimizer(self.opt_g, loss_g)
                self.log_step({
                    "l1": loss_l1.item(), "perc": loss_perc.item(),
                    "id": loss_id.item(), "adv_g": loss_adv.item(),
                    "d": loss_d.item(),
                })
                self.step += 1

            if (epoch + 1) % save_every == 0:
                self.save(
                    {"shape": self.shape_module, "disc": self.discriminator},
                    {"opt_g": self.opt_g, "opt_d": self.opt_d},
                    tag=f"epoch_{epoch+1}",
                )

        self.save(
            {"shape": self.shape_module, "disc": self.discriminator},
            {"opt_g": self.opt_g, "opt_d": self.opt_d},
            tag="final",
        )
        print("Stage 2 training complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_shape.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    ShapeTrainer(cfg).train()


if __name__ == "__main__":
    main()
