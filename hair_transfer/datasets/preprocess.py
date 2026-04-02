"""
Dataset Preprocessing Pipeline
================================
Processes raw FFHQ / CelebA-HQ images into training-ready format:

    dataset/
        aligned_faces/   — 256×256 aligned face crops
        hair_masks/      — binary hair segmentation masks

Usage:
    python datasets/preprocess.py \
        --src_dir /path/to/raw/images \
        --out_dir dataset \
        --seg_ckpt pretrained/face_parsing.pth \
        --device cuda

Requirements:
    pip install face-alignment Pillow tqdm torch torchvision
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.face_alignment import FaceAligner
from utils.hair_segmentation import HairSegmenter


SUPPORTED = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src_dir",  required=True, help="Raw image directory")
    p.add_argument("--out_dir",  default="dataset", help="Output directory")
    p.add_argument("--seg_ckpt", required=True, help="BiSeNet checkpoint path")
    p.add_argument("--device",   default="cpu")
    p.add_argument("--size",     type=int, default=256)
    p.add_argument("--skip_existing", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    src_dir   = Path(args.src_dir)
    face_dir  = Path(args.out_dir) / "aligned_faces"
    mask_dir  = Path(args.out_dir) / "hair_masks"
    face_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in src_dir.rglob("*") if p.suffix.lower() in SUPPORTED])
    print(f"Found {len(images)} images in {src_dir}")

    aligner   = FaceAligner(output_size=args.size, device=args.device)
    segmenter = HairSegmenter(args.seg_ckpt, device=args.device)

    ok = skip = fail = 0

    for img_path in tqdm(images, desc="Preprocessing"):
        stem      = img_path.stem
        face_out  = face_dir / f"{stem}.png"
        mask_out  = mask_dir / f"{stem}.png"

        if args.skip_existing and face_out.exists() and mask_out.exists():
            skip += 1
            continue

        try:
            img = Image.open(img_path).convert("RGB")

            # 1. Align face
            aligned = aligner.align(img)
            if aligned is None:
                fail += 1
                continue

            # 2. Segment hair
            mask = segmenter.segment(aligned)   # [H, W] uint8

            # 3. Save
            aligned.save(face_out)
            Image.fromarray(mask).save(mask_out)
            ok += 1

        except Exception as e:
            print(f"\n  ✗ {img_path.name}: {e}")
            fail += 1

    print(f"\nDone — processed: {ok}  skipped: {skip}  failed: {fail}")
    print(f"Output: {args.out_dir}/")


if __name__ == "__main__":
    main()
