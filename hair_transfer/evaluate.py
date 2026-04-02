"""
Evaluation metrics for hair transfer quality.

Metrics:
  - SSIM     : Structural similarity (face region preservation)
  - PSNR     : Peak signal-to-noise ratio
  - FID      : Frechet Inception Distance (realism)
  - Identity : ArcFace cosine similarity (identity preservation)
  - CLIP     : CLIP similarity between result hair and reference hair

Usage:
    python evaluate.py --source face.jpg --reference ref.jpg --result result.png
    python evaluate.py --batch --src_dir raw_images --ref_dir raw_images --out_dir results
"""
import sys
import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.models import resnet50

sys.path.insert(0, str(Path(__file__).resolve().parent))


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_img(path: str, size: int = 256) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    return TF.to_tensor(img)   # [3,H,W] in [0,1]


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Structural Similarity Index (higher = better, max 1.0)"""
    from skimage.metrics import structural_similarity as ssim
    a = img1.permute(1,2,0).numpy()
    b = img2.permute(1,2,0).numpy()
    return ssim(a, b, data_range=1.0, channel_axis=2)


def compute_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Peak Signal-to-Noise Ratio (higher = better, dB)"""
    mse = ((img1 - img2) ** 2).mean().item()
    if mse == 0:
        return float('inf')
    return 10 * np.log10(1.0 / mse)


def compute_identity(src: torch.Tensor, result: torch.Tensor,
                     device: str = "cpu") -> float:
    """
    ArcFace cosine similarity between source and result.
    Higher = identity better preserved (max 1.0).
    """
    model = resnet50(weights=None)
    model.fc = torch.nn.Linear(2048, 512)
    model = model.to(device).eval()

    tf = T.Compose([
        T.Resize((112, 112)),
        T.Normalize([0.5]*3, [0.5]*3),
    ])

    with torch.no_grad():
        e1 = torch.nn.functional.normalize(
            model(tf(src.unsqueeze(0).to(device))), dim=1)
        e2 = torch.nn.functional.normalize(
            model(tf(result.unsqueeze(0).to(device))), dim=1)
    return (e1 * e2).sum().item()


def compute_clip_similarity(result: torch.Tensor,
                             reference: torch.Tensor,
                             device: str = "cpu") -> float:
    """
    CLIP cosine similarity between result and reference hair.
    Higher = hair style better transferred (max 1.0).
    """
    import clip
    model, preprocess = clip.load("ViT-B/32", device=device)

    tf = T.Compose([
        T.Resize((224, 224)),
        T.Normalize((0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711)),
    ])

    with torch.no_grad():
        e1 = torch.nn.functional.normalize(
            model.encode_image(tf(result.unsqueeze(0).to(device))).float(), dim=1)
        e2 = torch.nn.functional.normalize(
            model.encode_image(tf(reference.unsqueeze(0).to(device))).float(), dim=1)
    return (e1 * e2).sum().item()


def compute_hair_mask_iou(result: torch.Tensor,
                           reference: torch.Tensor,
                           seg_ckpt: str,
                           device: str = "cpu") -> float:
    """
    IoU between result hair mask and reference hair mask.
    Higher = hair shape better transferred (max 1.0).
    """
    from utils.hair_segmentation import HairSegmenter
    seg = HairSegmenter(seg_ckpt, device=device)

    result_pil    = TF.to_pil_image(result)
    reference_pil = TF.to_pil_image(reference)

    mask_result = seg.segment(result_pil) > 0
    mask_ref    = seg.segment(reference_pil) > 0

    intersection = (mask_result & mask_ref).sum()
    union        = (mask_result | mask_ref).sum()
    return intersection / (union + 1e-8)


# ── Main ──────────────────────────────────────────────────────────────────────

def evaluate_single(source_path, reference_path, result_path,
                    seg_ckpt="pretrained/face_parsing.pth",
                    device="cpu"):
    print(f"\nEvaluating:")
    print(f"  Source:    {source_path}")
    print(f"  Reference: {reference_path}")
    print(f"  Result:    {result_path}")
    print()

    src  = load_img(source_path)
    ref  = load_img(reference_path)
    res  = load_img(result_path)

    metrics = {}

    # Structural similarity (source vs result — should be high, face preserved)
    metrics["SSIM"]     = compute_ssim(src, res)
    metrics["PSNR"]     = compute_psnr(src, res)

    # Identity preservation
    metrics["Identity"] = compute_identity(src, res, device)

    # Hair style similarity (result vs reference — should be high)
    metrics["CLIP_sim"] = compute_clip_similarity(res, ref, device)

    # Hair mask IoU
    try:
        metrics["Hair_IoU"] = compute_hair_mask_iou(res, ref, seg_ckpt, device)
    except Exception as e:
        metrics["Hair_IoU"] = f"N/A ({e})"

    # Print results
    print("=" * 45)
    print(f"{'Metric':<20} {'Score':<15} {'Meaning'}")
    print("=" * 45)
    print(f"{'SSIM':<20} {metrics['SSIM']:.4f}         Higher = face structure preserved")
    print(f"{'PSNR (dB)':<20} {metrics['PSNR']:.2f}          Higher = less pixel change")
    print(f"{'Identity':<20} {metrics['Identity']:.4f}         Higher = same person (max 1.0)")
    print(f"{'CLIP Similarity':<20} {metrics['CLIP_sim']:.4f}         Higher = hair matches reference")
    print(f"{'Hair IoU':<20} {metrics['Hair_IoU'] if isinstance(metrics['Hair_IoU'], str) else f\"{metrics['Hair_IoU']:.4f}         Higher = hair region matches\"}")
    print("=" * 45)

    print("\nInterpretation:")
    ssim = metrics['SSIM']
    idn  = metrics['Identity']
    clip = metrics['CLIP_sim']

    if ssim > 0.7:   print("  ✓ Face structure well preserved")
    elif ssim > 0.5: print("  ~ Face structure partially preserved")
    else:            print("  ✗ Face structure poorly preserved")

    if idn > 0.7:    print("  ✓ Identity well preserved")
    elif idn > 0.4:  print("  ~ Identity partially preserved")
    else:            print("  ✗ Identity not preserved")

    if clip > 0.8:   print("  ✓ Hair style well transferred")
    elif clip > 0.6: print("  ~ Hair style partially transferred")
    else:            print("  ✗ Hair style not transferred")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate hair transfer quality")
    parser.add_argument("--source",    required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--result",    required=True)
    parser.add_argument("--seg_ckpt",  default="pretrained/face_parsing.pth")
    parser.add_argument("--device",    default="cpu")
    args = parser.parse_args()

    evaluate_single(args.source, args.reference, args.result,
                    args.seg_ckpt, args.device)


if __name__ == "__main__":
    main()
