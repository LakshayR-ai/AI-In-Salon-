"""
Base trainer with shared logic: checkpointing, logging, mixed precision.
All stage trainers inherit from this.
"""
import os
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

log = logging.getLogger("hair_transfer")


class BaseTrainer:
    def __init__(self, cfg: dict, device: str = "cuda"):
        self.cfg    = cfg
        self.device = device
        self.step   = 0
        self.epoch  = 0
        self.scaler = GradScaler("cuda", enabled=cfg["train"].get("mixed_precision", True))

        out = Path(cfg["train"]["output_dir"]) / cfg["train"]["stage"]
        out.mkdir(parents=True, exist_ok=True)
        self.out_dir = out

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )

    # ── Checkpoint helpers ────────────────────────────────────────────────────

    def save(self, models: dict, optimizers: dict, tag: str = "latest"):
        path = self.out_dir / f"ckpt_{tag}.pt"
        torch.save({
            "epoch":      self.epoch,
            "step":       self.step,
            "models":     {k: v.state_dict() for k, v in models.items()},
            "optimizers": {k: v.state_dict() for k, v in optimizers.items()},
        }, path)
        log.info(f"Saved checkpoint → {path}")

    def load(self, models: dict, optimizers: dict, tag: str = "latest"):
        path = self.out_dir / f"ckpt_{tag}.pt"
        if not path.exists():
            log.info("No checkpoint found, starting from scratch.")
            return
        ckpt = torch.load(path, map_location=self.device)
        self.epoch = ckpt["epoch"]
        self.step  = ckpt["step"]
        for k, v in models.items():
            if k in ckpt["models"]:
                v.load_state_dict(ckpt["models"][k])
        for k, v in optimizers.items():
            if k in ckpt["optimizers"]:
                v.load_state_dict(ckpt["optimizers"][k])
        log.info(f"Loaded checkpoint from {path} (epoch {self.epoch})")

    # ── Optimizer factory ─────────────────────────────────────────────────────

    def make_optimizer(self, params) -> torch.optim.Optimizer:
        oc = self.cfg["optimizer"]
        return torch.optim.Adam(
            params,
            lr=oc["lr"],
            betas=tuple(oc["betas"]),
            weight_decay=oc.get("weight_decay", 0.0),
        )

    # ── Training step wrapper ─────────────────────────────────────────────────

    def step_optimizer(self, optimizer, loss):
        optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(optimizer)
        self.scaler.update()

    def log_step(self, losses: dict):
        if self.step % self.cfg["train"]["log_every"] == 0:
            parts = "  ".join(f"{k}={v:.4f}" for k, v in losses.items())
            log.info(f"[epoch {self.epoch} step {self.step}]  {parts}")
