"""
Stage 4 — Train the Refinement UNet.

Objective: restore fine facial details lost during GAN synthesis.
Takes (blended_output, source_face) → refined_output.

Losses:
  - L1
  - Perceptual
  - Identity
  - Adversarial

Usage:
    python training/train_refinement.py --config configs/train_refinement.yaml
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
from models.refinement import RefinementUNet
from models.discriminator import PatchDiscriminator
from models.stylegan import load_stylegan2
from utils.dataset_loader import HairTransferDataset
from utils.losses import (ReconstructionLoss, PerceptualLoss,
                           IdentityLoss, AdversarialLoss)
from training.base_trainer import BaseTrainer


class RefinementTrainer(BaseTrainer):

    def __init__(self, cfg: dict):
        super().__init__(cfg, device=cfg["device"])

        out_dir = Path(cfg["train"]["output_dir"])

        self.generator = load_stylegan2(
            cfg["stylegan"]["checkpoint"],
            size=cfg["stylegan"]["size"],
            latent_dim=cfg["stylegan"]["latent_dim"],
            device=self.device,
        )

        def _load_frozen(model, ckpt_path, key):
            if Path(ckpt_path).exists():
                s = torch.load(ckpt_path, map_location=self.device)
                model.load_state_dict(s["models"][key])
            model.eval()
            for p in model.parameters(): p.requires_grad_(False)
            return model

        self.encoder = _load_frozen(
            HairEncoder(n_styles=cfg["encoder"]["n_styles"]).to(self.device),
            out_dir / "encoder" / "ckpt_final.pt", "encoder",
        )
        self.shape_module = _load_frozen(
            ShapeModule(channels=512).to(self.device),
            out_dir / "shape" / "ckpt_final.pt", "shape",
        )
        self.color_module = _load_frozen(
            ColorModule(n_styles=cfg["encoder"]["n_styles"]).to(self.device),
            out_dir / "color" / "ckpt_final.pt", "color",
        )

        self.refinement    = RefinementUNet(base_ch=cfg["refinement"].get("base_channels", 64)).to(self.device)
        self.discriminator = PatchDiscriminator().to(self.device)

        lw = cfg["loss"]
        self.l1   = ReconstructionLoss()
        self.perc = PerceptualLoss().to(self.device)
        self.id_  = IdentityLoss(device=self.device)
        self.adv  = AdversarialLoss()
        self.lw   = lw

        self.opt_g = self.make_optimizer(self.refinement.parameters())
        self.opt_d = self.make_optimizer(self.discriminator.parameters())

        dataset = HairTransferDataset(root=cfg["data"]["root"],
                                      image_size=cfg["data"]["image_size"])
        self.loader = DataLoader(dataset, batch_size=cfg["data"]["batch_size"],
                                 shuffle=True, num_workers=cfg["data"]["num_workers"],
                                 pin_memory=True)

    def train(self):
        max_epochs = self.cfg["train"]["max_epochs"]
        save_every = self.cfg["train"]["save_every"]
        self.load({"refine": self.refinement, "disc": self.discriminator},
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
                        w_out        = self.color_module(w_src, src_img, ref_img)
                        blended, _   = self.generator(
                            [w_out], input_is_latent=True,
                            start_layer=4, end_layer=None, layer_in=f_out,
                        )

                    refined = self.refinement(blended, src_img)

                    real_pred = self.discriminator(src_img.detach())
                    fake_pred = self.discriminator(refined.detach())
                    loss_d    = self.adv.discriminator_loss(real_pred, fake_pred)

                self.step_optimizer(self.opt_d, loss_d)

                with autocast('cuda', enabled=self.cfg["train"]["mixed_precision"]):
                    fake_pred_g = self.discriminator(refined)
                    loss_l1     = self.l1(refined, src_img)
                    loss_perc   = self.perc(refined, src_img)
                    loss_id     = self.id_(refined, src_img)
                    loss_adv    = self.adv(fake_pred_g)

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
                self.save({"refine": self.refinement, "disc": self.discriminator},
                          {"opt_g": self.opt_g, "opt_d": self.opt_d},
                          tag=f"epoch_{epoch+1}")

        self.save({"refine": self.refinement, "disc": self.discriminator},
                  {"opt_g": self.opt_g, "opt_d": self.opt_d}, tag="final")
        print("Stage 4 training complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_refinement.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    RefinementTrainer(cfg).train()


if __name__ == "__main__":
    main()
