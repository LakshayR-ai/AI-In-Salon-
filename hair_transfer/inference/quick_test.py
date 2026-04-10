"""
Hair Blend — CPU fallback when HairFastGAN GPU is unavailable.

Uses:
  - BiSeNet hair segmentation
  - Color histogram matching
  - Multi-scale pyramid blending
  - Edge feathering

Note: This is a mask-based approach. For production quality,
use HairFastGAN with a GPU (automatic when CUDA is available).
"""
import sys
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.face_alignment import FaceAligner
from utils.hair_segmentation import HairSegmenter


def match_color(src: np.ndarray, ref: np.ndarray,
                mask: np.ndarray) -> np.ndarray:
    """Match color statistics of ref hair to src hair region."""
    result = ref.astype(float).copy()
    hair_px = mask > 0
    if hair_px.sum() == 0:
        return result.astype(np.uint8)

    for ch in range(3):
        s = src[:, :, ch].astype(float)[hair_px]
        r = ref[:, :, ch].astype(float)[hair_px]
        if r.std() < 1e-6:
            continue
        matched = (r - r.mean()) / r.std() * s.std() + s.mean()
        result[:, :, ch][hair_px] = np.clip(matched, 0, 255)

    return result.astype(np.uint8)


def build_gaussian_pyramid(img: np.ndarray, levels: int) -> list:
    pyramid = [img.astype(float)]
    for _ in range(levels):
        prev = pyramid[-1]
        sigma = [1, 1, 0] if prev.ndim == 3 else 1
        blurred = gaussian_filter(prev, sigma=sigma)
        pyramid.append(blurred[::2, ::2])
    return pyramid


def build_laplacian_pyramid(img: np.ndarray, levels: int) -> list:
    gp = build_gaussian_pyramid(img, levels)
    lp = []
    for i in range(levels):
        h, w = gp[i].shape[:2]
        up = np.repeat(np.repeat(gp[i+1], 2, axis=0), 2, axis=1)[:h, :w]
        lp.append(gp[i] - up)
    lp.append(gp[levels])
    return lp


def pyramid_blend(src: np.ndarray, ref: np.ndarray,
                  mask: np.ndarray, levels: int = 4) -> np.ndarray:
    mask_f = mask.astype(float) / 255.0

    lp_src  = build_laplacian_pyramid(src, levels)
    lp_ref  = build_laplacian_pyramid(ref, levels)
    gp_mask = build_gaussian_pyramid(mask_f, levels)

    blended = []
    for ls, lr, gm in zip(lp_src, lp_ref, gp_mask):
        if gm.ndim == 2:
            gm = gm[:, :, None]
        if gm.shape[:2] != ls.shape[:2]:
            gm_pil = Image.fromarray((gm[:,:,0]*255).astype(np.uint8))
            gm_pil = gm_pil.resize((ls.shape[1], ls.shape[0]), Image.LANCZOS)
            gm = np.array(gm_pil).astype(float)[:,:,None] / 255.0
        blended.append(ls * (1 - gm) + lr * gm)

    result = blended[-1]
    for i in range(len(blended) - 2, -1, -1):
        h, w = blended[i].shape[:2]
        up = np.repeat(np.repeat(result, 2, axis=0), 2, axis=1)[:h, :w]
        result = up + blended[i]

    return np.clip(result, 0, 255).astype(np.uint8)


def blend_hair(source: Image.Image, reference: Image.Image,
               seg_ckpt: str, size: int = 512,
               device: str = "cpu") -> Image.Image:
    """
    Transfer hairstyle using segmentation + color matching + pyramid blend.
    """
    aligner   = FaceAligner(output_size=size, device=device)
    segmenter = HairSegmenter(seg_ckpt, device=device)

    # Align faces
    src_aligned = aligner.align(source) or source.resize((size, size), Image.LANCZOS)
    ref_aligned = aligner.align(reference) or reference.resize((size, size), Image.LANCZOS)
    src_aligned = src_aligned.resize((size, size), Image.LANCZOS)
    ref_aligned = ref_aligned.resize((size, size), Image.LANCZOS)

    src_arr = np.array(src_aligned)
    ref_arr = np.array(ref_aligned)

    # Segment hair
    src_mask = segmenter.segment(src_aligned)   # [H,W] 0/255
    ref_mask = segmenter.segment(ref_aligned)

    # Dilate ref mask to cover full hair area
    ref_mask_pil     = Image.fromarray(ref_mask).filter(ImageFilter.MaxFilter(size=21))
    ref_mask_dilated = np.array(ref_mask_pil)

    # Dilate src mask to fully erase original hair
    src_mask_pil     = Image.fromarray(src_mask).filter(ImageFilter.MaxFilter(size=21))
    src_mask_dilated = np.array(src_mask_pil)

    # Color match: adapt ref hair color to match src hair tone
    ref_color = match_color(src_arr, ref_arr, ref_mask)

    # Build blend mask — feathered edges
    blend_mask = gaussian_filter(ref_mask_dilated.astype(float), sigma=12)
    if blend_mask.max() > 0:
        blend_mask = blend_mask / blend_mask.max()
    blend_mask = np.clip(blend_mask, 0, 1)

    # Erase source hair: fill with surrounding skin tone
    src_no_hair = src_arr.copy().astype(float)
    erase_mask  = gaussian_filter(src_mask_dilated.astype(float), sigma=4)
    if erase_mask.max() > 0:
        erase_mask = erase_mask / erase_mask.max()

    # Sample skin color from face center (nose/cheek area)
    h, w = src_arr.shape[:2]
    face_center = src_arr[h//3:2*h//3, w//4:3*w//4]
    # Exclude hair pixels from skin sample
    face_mask_region = src_mask[h//3:2*h//3, w//4:3*w//4]
    skin_pixels = face_center[face_mask_region == 0]
    skin_color  = skin_pixels.mean(axis=0) if len(skin_pixels) > 100 \
                  else face_center.mean(axis=(0, 1))

    src_no_hair = src_arr.astype(float) * (1 - erase_mask[:,:,None]) + \
                  skin_color * erase_mask[:,:,None]

    # Pyramid blend reference hair onto erased source
    result = pyramid_blend(
        src_no_hair.astype(np.uint8),
        ref_color,
        (blend_mask * 255).astype(np.uint8)
    )

    # Final composite with feathered edges
    alpha = blend_mask[:, :, None]
    final = result.astype(float) * alpha + src_no_hair * (1 - alpha)
    final = np.clip(final, 0, 255).astype(np.uint8)

    return Image.fromarray(final)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",    required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output",    default="result.png")
    parser.add_argument("--seg_ckpt",  default="pretrained/face_parsing.pth")
    parser.add_argument("--size",      type=int, default=512)
    parser.add_argument("--device",    default="cpu")
    args = parser.parse_args()

    src    = Image.open(args.source).convert("RGB")
    ref    = Image.open(args.reference).convert("RGB")
    result = blend_hair(src, ref, args.seg_ckpt, args.size, args.device)
    result.save(args.output)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
