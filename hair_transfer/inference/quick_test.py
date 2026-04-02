"""
Quick pipeline test — bypasses trained modules.
Directly blends source + reference hair using segmentation masks only.
This verifies the preprocessing and segmentation are working correctly
before committing to full training.

Usage:
    python inference/quick_test.py --source raw_images/6.png --reference raw_images/7.png --output quick_result.png
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.face_alignment import FaceAligner
from utils.hair_segmentation import HairSegmenter


def blend_hair(source: Image.Image, reference: Image.Image,
               seg_ckpt: str, size: int = 256, device: str = "cpu") -> Image.Image:
    """
    Simple mask-based hair transplant (no GAN, no training needed).
    Copies the hair region from reference onto source using segmentation masks.
    Result shows whether segmentation and alignment are working.
    """
    aligner   = FaceAligner(output_size=size, device=device)
    segmenter = HairSegmenter(seg_ckpt, device=device)

    # Align both faces
    src_aligned = aligner.align(source) or source.resize((size, size))
    ref_aligned = aligner.align(reference) or reference.resize((size, size))

    src_aligned = src_aligned.resize((size, size))
    ref_aligned = ref_aligned.resize((size, size))

    # Get hair masks
    src_mask = segmenter.segment(src_aligned)   # [H,W] 0/255
    ref_mask = segmenter.segment(ref_aligned)

    # Normalise masks to [0,1] float
    src_m = (src_mask / 255.0)[..., None]   # [H,W,1]
    ref_m = (ref_mask / 255.0)[..., None]

    src_arr = np.array(src_aligned).astype(float)
    ref_arr = np.array(ref_aligned).astype(float)

    # Resize ref to match src dimensions
    ref_resized = np.array(
        Image.fromarray(ref_arr.astype(np.uint8)).resize(
            src_aligned.size, Image.LANCZOS)
    ).astype(float)

    # Smooth mask edges for natural blending
    from PIL import ImageFilter
    mask_pil   = Image.fromarray((ref_m[:,:,0]*255).astype(np.uint8))
    mask_blur  = mask_pil.filter(ImageFilter.GaussianBlur(radius=8))
    ref_m_soft = np.array(mask_blur).astype(float)[:,:,None] / 255.0

    # Also remove source hair region for clean replacement
    src_mask_pil  = Image.fromarray((src_m[:,:,0]*255).astype(np.uint8))
    src_mask_blur = src_mask_pil.filter(ImageFilter.GaussianBlur(radius=6))
    src_m_soft    = np.array(src_mask_blur).astype(float)[:,:,None] / 255.0

    # Blend: remove source hair, add reference hair
    combined_mask = np.clip(ref_m_soft + src_m_soft * 0.5, 0, 1)
    blended = src_arr * (1 - combined_mask) + ref_resized * combined_mask

    print(f"Source hair pixels:    {(src_mask > 0).sum()}")
    print(f"Reference hair pixels: {(ref_mask > 0).sum()}")

    return Image.fromarray(blended.astype(np.uint8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",    required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output",    default="quick_result.png")
    parser.add_argument("--seg_ckpt",  default="pretrained/face_parsing.pth")
    parser.add_argument("--size",      type=int, default=256)
    parser.add_argument("--device",    default="cpu")
    args = parser.parse_args()

    src = Image.open(args.source).convert("RGB")
    ref = Image.open(args.reference).convert("RGB")

    print("Running quick mask-based blend (no training required)...")
    result = blend_hair(src, ref, args.seg_ckpt, args.size, args.device)
    result.save(args.output)
    print(f"Result saved → {args.output}")
    print("\nThis is a simple mask blend, not GAN output.")
    print("For proper hair transfer, train the full pipeline with more data.")


if __name__ == "__main__":
    main()
