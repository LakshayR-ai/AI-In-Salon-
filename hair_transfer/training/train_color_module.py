"""
Stage 3 — Train the Color/Style Transfer Module (S-space via CLIP).

Objective: modify W+ layers 6–14 so the output hair color matches
the reference, while preserving face identity.

Losses:
  - CLIP directional similarity
  - L1 (hair region)
  - Perceptual
  - Identity
  - Adversarial

Usage:
    python training/train_color_module.py --config configs/train_color.yaml
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
from models.color_module import ColorModule
from models.discriminator import PatchDiscriminator
from models.stylegan import load_stylegan2
from utils.dataset_loader import HairTransferDataset
from utils.losses import (ReconstructionLoss, PerceptualLoss,
                           IdentityLoss, CLIPLoss, AdversarialLoss)
from training.base_trainer import BaseTrainer


class ColorTrainer(BaseTrainer):

    def __init__(self, cfg: dict):
        super().__init__(cfg, device=cfg["device"])

        self.generator = load_stylegan2(
            cfg["stylegan"]["checkpoint"],
            size=cfg["stylegan"]["size"],
            latent_dim=cfg["stylegan"]["latent_dim"],
            device=self.device,
        )

        out_dir = Path(cfg["train"]["output_dir"])

        # Load frozen encoder
        self.encoder = HairEncoder(n_styles=cfg["encoder"]["n_styles"]).to(self.device)
        enc_ckpt = out_dir / "encoder" / "ckpt_final.pt"
        if enc_ckpt.exists():
            s = torch.load(enc_ckpt, map_location=self.device)
            self.encoder.load_state_dict(s["models"]["encoder"])
        self.encoder.eval()
        for p in self.encoder.parameters(): p.requires_grad_(False)

        # Load frozen shape module
        self.shape_module = ShapeModule(channels=512).to(self.device)
        shape_ckpt = out_dir / "shape" / "ckpt_final.pt"
        if shape_ckpt.exists():
            s = torch.load(shape_ckpt, map_location=self.device)
            self.shape_module.load_state_dict(s["models"]["shape"])
        self.shape_module.eval()
        for p in self.shape_module.parameters(): p.requires_grad_(False)

        self.color_module  = ColorModule(
            n_styles=cfg["encoder"]["n_styles"],
            clip_model=cfg["color"]["clip_model"],
        ).to(self.device)
        self.discriminator = PatchDiscriminator().to(self.device)

        lw = cfg["loss"]
        self.l1   = ReconstructionLoss()
        self.perc = PerceptualLoss().to(self.device)
        self.id_  = IdentityLoss(device=self.device)
        self.clip = CLIPLoss(device=self.device)
        self.adv  = AdversarialLoss()
        self.lw   = lw

        self.opt_g = self.make_optimizer(self.color_module.parameters())
        self.opt_d = self.make_optimizer(self.discriminator.parameters())

        dataset = HairTransferDataset(root=cfg["data"]["root"],
                                      image_size=cfg["data"]["image_size"])
        self.loader = DataLoader(dataset, batch_size=cfg["data"]["batch_size"],
                                 shuffle=True, num_workers=cfg["data"]["num_workers"],
                                 pin_memory=True)

    def train(self):
        max_epochs = self.cfg["train"]["max_epochs"]
        save_every = self.cfg["train"]["save_every"]
        self.load({"color": self.color_module, "disc": self.discriminator},
                  {"opt_g": self.opt_g, "opt_d": self.opt_d})

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
                        _, f_ref     = self.encoder(ref_img)
                        f_out        = self.shape_module(f_src, f_ref, src_mask, ref_mask)

                    # Color transfer in W+ space
                    w_out = self.color_module(w_src, src_img, ref_img)

                    out_img, _ = self.generator(
                        [w_out], input_is_latent=True,
                        start_layer=4, end_layer=None, layer_in=f_out,
                    )

                    # Discriminator
                    real_pred = self.discriminator(src_img.detach())
                    fake_pred = self.discriminator(out_img.detach())
                    loss_d    = self.adv.discriminator_loss(real_pred, fake_pred)

                self.step_optimizer(self.opt_d, loss_d)

                with autocast('cuda', enabled=self.cfg["train"]["mixed_precision"]):
                    fake_pred_g = self.discriminator(out_img)
                    loss_clip   = self.clip(out_img, src_img, ref_img)
                    loss_l1     = self.l1(out_img * ref_mask, ref_img * ref_mask)
                    loss_perc   = self.perc(out_img, src_img)
                    loss_id     = self.id_(out_img, src_img)
                    loss_adv    = self.adv(fake_pred_g)

                    loss_g = (
                        self.lw["lambda_clip"]       * loss_clip +
                        self.lw["lambda_l1"]         * loss_l1 +
                        self.lw["lambda_perceptual"]  * loss_perc +
                        self.lw["lambda_identity"]    * loss_id +
                        self.lw["lambda_adv"]         * loss_adv
                    )

                self.step_optimizer(self.opt_g, loss_g)
                self.log_step({
                    "clip": loss_clip.item(), "l1": loss_l1.item(),
                    "perc": loss_perc.item(), "id": loss_id.item(),
                    "adv_g": loss_adv.item(), "d": loss_d.item(),
                })
                self.step += 1

            if (epoch + 1) % save_every == 0:
                self.save({"color": self.color_module, "disc": self.discriminator},
                          {"opt_g": self.opt_g, "opt_d": self.opt_d},
                          tag=f"epoch_{epoch+1}")

        self.save({"color": self.color_module, "disc": self.discriminator},
                  {"opt_g": self.opt_g, "opt_d": self.opt_d}, tag="final")
        print("Stage 3 training complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_color.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    ColorTrainer(cfg).train()


if __name__ == "__main__":
    main()
