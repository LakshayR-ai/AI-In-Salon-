"""
Advanced Hair Blend — no OpenCV dependency.

Techniques:
  1. Face alignment (dlib landmarks)
  2. BiSeNet hair segmentation
  3. LAB color histogram matching (scipy)
  4. Multi-scale Gaussian pyramid blending (pure numpy)
  5. Edge-aware feathering
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


# ── Color matching ────────────────────────────────────────────────────────────

def rgb_to_lab(img: np.ndarray) -> np.ndarray:
    """Simple RGB → LAB approximation using numpy."""
    img = img.astype(float) / 255.0
    # sRGB to linear
    mask = img > 0.04045
    img[mask]  = ((img[mask]  + 0.055) / 1.055) ** 2.4
    img[~mask] = img[~mask] / 12.92
    # Linear to XYZ
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = img @ M.T
    # XYZ to LAB
    xyz /= np.array([0.95047, 1.00000, 1.08883])
    mask2 = xyz > 0.008856
    xyz[mask2]  = xyz[mask2]  ** (1/3)
    xyz[~mask2] = 7.787 * xyz[~mask2] + 16/116
    L = 116 * xyz[..., 1] - 16
    a = 500 * (xyz[..., 0] - xyz[..., 1])
    b = 200 * (xyz[..., 1] - xyz[..., 2])
    return np.stack([L, a, b], axis=-1)


def match_color(src: np.ndarray, ref: np.ndarray,
                mask: np.ndarray) -> np.ndarray:
    """Match hair color statistics from reference to source."""
    result = src.astype(float).copy()
    hair_px = mask > 0

    for ch in range(3):
        src_ch = src[:, :, ch].astype(float)[hair_px]
        ref_ch = ref[:, :, ch].astype(float)[hair_px]
        if src_ch.std() < 1e-6 or ref_ch.std() < 1e-6:
            continue
        matched = (src_ch - src_ch.mean()) / src_ch.std() * ref_ch.std() + ref_ch.mean()
        result[:, :, ch][hair_px] = np.clip(matched, 0, 255)

    return result.astype(np.uint8)


# ── Pyramid blending ──────────────────────────────────────────────────────────

def build_gaussian_pyramid(img: np.ndarray, levels: int) -> list:
    pyramid = [img.astype(float)]
    for _ in range(levels):
        prev = pyramid[-1]
        if prev.ndim == 3:
            blurred = gaussian_filter(prev, sigma=[1, 1, 0])
        else:
            blurred = gaussian_filter(prev, sigma=1)
        pyramid.append(blurred[::2, ::2])
    return pyramid


def build_laplacian_pyramid(img: np.ndarray, levels: int) -> list:
    gp = build_gaussian_pyramid(img, levels)
    lp = []
    for i in range(levels):
        h, w = gp[i].shape[:2]
        up = np.repeat(np.repeat(gp[i+1], 2, axis=0), 2, axis=1)
        up = up[:h, :w]
        lp.append(gp[i] - up)
    lp.append(gp[levels])
    return lp


def pyramid_blend(src: np.ndarray, ref: np.ndarray,
                  mask: np.ndarray, levels: int = 4) -> np.ndarray:
    """Multi-scale pyramid blending for seamless transitions."""
    mask_f = mask.astype(float) / 255.0
    if mask_f.ndim == 2:
        mask_f = mask_f[:, :, None]

    lp_src  = build_laplacian_pyramid(src,  levels)
    lp_ref  = build_laplacian_pyramid(ref,  levels)
    gp_mask = build_gaussian_pyramid(mask_f[:, :, 0], levels)

    blended = []
    for ls, lr, gm in zip(lp_src, lp_ref, gp_mask):
        gm3 = gm[:, :, None] if gm.ndim == 2 else gm
        # Resize gm3 to match ls if needed
        if gm3.shape[:2] != ls.shape[:2]:
            from PIL import Image as PILImage
            gm_pil = PILImage.fromarray((gm3[:,:,0]*255).astype(np.uint8))
            gm_pil = gm_pil.resize((ls.shape[1], ls.shape[0]), PILImage.LANCZOS)
            gm3 = np.array(gm_pil).astype(float)[:,:,None] / 255.0
        blended.append(ls * (1 - gm3) + lr * gm3)

    result = blended[-1]
    for i in range(len(blended) - 2, -1, -1):
        h, w = blended[i].shape[:2]
        up = np.repeat(np.repeat(result, 2, axis=0), 2, axis=1)
        up = up[:h, :w]
        result = up + blended[i]

    return np.clip(result, 0, 255).astype(np.uint8)


# ── Main blend function ───────────────────────────────────────────────────────

def blend_hair(source: Image.Image, reference: Image.Image,
               seg_ckpt: str, size: int = 512,
               device: str = "cpu") -> Image.Image:
    """
    Advanced hair transfer:
    1. Face alignment
    2. Hair segmentation
    3. Color histogram matching
    4. Multi-scale pyramid blending
    5. Edge feathering
    """
    aligner   = FaceAligner(output_size=size, device=device)
    segmenter = HairSegmenter(seg_ckpt, device=device)

    # Align
    src_aligned = aligner.align(source) or source.resize((size, size), Image.LANCZOS)
    ref_aligned = aligner.align(reference) or reference.resize((size, size), Image.LANCZOS)
    src_aligned = src_aligned.resize((size, size), Image.LANCZOS)
    ref_aligned = ref_aligned.resize((size, size), Image.LANCZOS)

    src_arr = np.array(src_aligned)
    ref_arr = np.array(ref_aligned)

    # Segment
    src_mask = segmenter.segment(src_aligned)
    ref_mask = segmenter.segment(ref_aligned)

    # Dilate ref mask
    ref_mask_pil = Image.fromarray(ref_mask)
    ref_mask_pil = ref_mask_pil.filter(ImageFilter.MaxFilter(size=15))
    ref_mask_dilated = np.array(ref_mask_pil)

    # Color match
    ref_color = match_color(ref_arr, src_arr, src_mask)

    # Feather mask
    ref_mask_soft = gaussian_filter(ref_mask_dilated.astype(float), sigma=12)
    ref_mask_soft = np.clip(ref_mask_soft / ref_mask_soft.max(), 0, 1) \
                    if ref_mask_soft.max() > 0 else ref_mask_soft

    # Pyramid blend
    result = pyramid_blend(
        src_arr, ref_color,
        (ref_mask_soft * 255).astype(np.uint8)
    )

    # Final feathered composite
    alpha  = ref_mask_soft[:, :, None]
    final  = result.astype(float) * alpha + src_arr.astype(float) * (1 - alpha)
    final  = np.clip(final, 0, 255).astype(np.uint8)

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

    src = Image.open(args.source).convert("RGB")
    ref = Image.open(args.reference).convert("RGB")
    result = blend_hair(src, ref, args.seg_ckpt, args.size, args.device)
    result.save(args.output)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
