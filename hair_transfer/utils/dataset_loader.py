"""
Dataset loader for hair transfer training.

Expects the following directory structure:
    dataset/
        aligned_faces/   *.jpg / *.png
        hair_masks/      *.png  (binary, same stem as aligned_faces)

Each training sample is a pair:
    source face  — the face whose identity we preserve
    reference    — a different face whose hairstyle we transfer

Pairs are sampled randomly within the same batch.
"""
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class HairTransferDataset(Dataset):
    """
    Returns (source_img, source_mask, ref_img, ref_mask) tuples.

    Args:
        root:       path to dataset/ folder
        image_size: spatial resolution (default 256)
        augment:    apply random horizontal flip
    """

    def __init__(self, root: str, image_size: int = 256, augment: bool = True):
        self.root       = Path(root)
        self.image_size = image_size
        self.augment    = augment

        face_dir = self.root / "aligned_faces"
        mask_dir = self.root / "hair_masks"

        # Collect images that have a corresponding mask
        self.samples = sorted([
            p for p in face_dir.glob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            and (mask_dir / (p.stem + ".png")).exists()
        ])

        if len(self.samples) == 0:
            raise FileNotFoundError(
                f"No aligned face images found in {face_dir}. "
                "Run preprocessing first: python datasets/preprocess.py"
            )

        self.mask_dir = mask_dir

        self.img_tf = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.LANCZOS),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),   # → [-1, 1]
        ])
        self.mask_tf = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),   # → [0, 1]
        ])

    def __len__(self):
        return len(self.samples)

    def _load(self, idx: int):
        img_path  = self.samples[idx]
        mask_path = self.mask_dir / (img_path.stem + ".png")

        img  = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.augment and random.random() < 0.5:
            img  = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return self.img_tf(img), self.mask_tf(mask)

    def __getitem__(self, idx: int):
        src_img, src_mask = self._load(idx)

        # Sample a different image as reference
        ref_idx = random.randint(0, len(self.samples) - 1)
        while ref_idx == idx:
            ref_idx = random.randint(0, len(self.samples) - 1)
        ref_img, ref_mask = self._load(ref_idx)

        return {
            "src_img":  src_img,    # [3, H, W]
            "src_mask": src_mask,   # [1, H, W]
            "ref_img":  ref_img,
            "ref_mask": ref_mask,
        }
