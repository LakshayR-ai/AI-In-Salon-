"""
HairFastGAN Official Pretrained Pipeline.

Uses the official AIRI-Institute pretrained models from HuggingFace.
These are trained on full FFHQ 70k images — production quality.

Download weights:
    python app/hairfastgan_pipeline.py --download

Models needed:
    pretrained/HairFastGAN/Rotate/rotate_best.pth
    pretrained/HairFastGAN/Blending/checkpoint.pth
    pretrained/HairFastGAN/PostProcess/pp_model.pth
    pretrained/HairFastGAN/encoder4editing/e4e_ffhq_encode.pt
    pretrained/HairFastGAN/FeatureStyleEncoder/encoder.pt
    pretrained/HairFastGAN/StyleGAN/ffhq.pt
    pretrained/HairFastGAN/ShapeAdaptor/mask_generator.pth
    pretrained/HairFastGAN/ShapeAdaptor/shape_predictor_68_face_landmarks.dat
    pretrained/HairFastGAN/SEAN/sean.pth
    pretrained/HairFastGAN/ArcFace/backbone_ir50.pth
"""
import io
import os
import sys
import base64
import logging
from pathlib import Path
from PIL import Image

log = logging.getLogger("salon.hairfastgan")

HF_BASE = "https://huggingface.co/AIRI-Institute/HairFastGAN/resolve/main/pretrained_models"

MODELS = {
    "Rotate/rotate_best.pth":           f"{HF_BASE}/Rotate/rotate_best.pth",
    "Blending/checkpoint.pth":          f"{HF_BASE}/Blending/checkpoint.pth",
    "PostProcess/pp_model.pth":         f"{HF_BASE}/PostProcess/pp_model.pth",
    "encoder4editing/e4e_ffhq_encode.pt": f"{HF_BASE}/encoder4editing/e4e_ffhq_encode.pt",
    "FeatureStyleEncoder/encoder.pt":   f"{HF_BASE}/FeatureStyleEncoder/psp_ffhq_encode.pt",
    "StyleGAN/ffhq.pt":                 f"{HF_BASE}/StyleGAN/ffhq.pt",
    "ShapeAdaptor/mask_generator.pth":  f"{HF_BASE}/ShapeAdaptor/mask_generator.pth",
    "ShapeAdaptor/shape_predictor_68_face_landmarks.dat":
        f"{HF_BASE}/ShapeAdaptor/shape_predictor_68_face_landmarks.dat",
    "SEAN/sean.pth":                    f"{HF_BASE}/sean_checkpoints/CelebA-HQ_pretrained/latest_net_G.pth",
    "ArcFace/backbone_ir50.pth":        f"{HF_BASE}/ArcFace/backbone_ir50.pth",
    "BiSeNet/face_parsing.pth":         f"{HF_BASE}/BiSeNet/face_parsing_79999_iter.pth",
}


def download_models(pretrained_dir: Path):
    """Download all HairFastGAN pretrained models from HuggingFace."""
    import requests
    from tqdm import tqdm

    for rel_path, url in MODELS.items():
        dest = pretrained_dir / "HairFastGAN" / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)

        if dest.exists() and dest.stat().st_size > 1024 * 100:
            log.info(f"  ✓ already exists: {rel_path}")
            continue

        log.info(f"  ↓ downloading {rel_path}...")
        try:
            r = requests.get(url, stream=True, timeout=120)
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(dest, "wb") as f, tqdm(total=total, unit="B",
                                              unit_scale=True,
                                              desc=dest.name) as bar:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
                    bar.update(len(chunk))
            log.info(f"  ✓ saved {rel_path}")
        except Exception as e:
            log.error(f"  ✗ FAILED {rel_path}: {e}")


def models_available(pretrained_dir: Path) -> bool:
    """Check if all required HairFastGAN models are downloaded."""
    required = [
        "HairFastGAN/Rotate/rotate_best.pth",
        "HairFastGAN/Blending/checkpoint.pth",
        "HairFastGAN/StyleGAN/ffhq.pt",
        "HairFastGAN/encoder4editing/e4e_ffhq_encode.pt",
    ]
    return all((pretrained_dir / p).exists() for p in required)


def load_hairfastgan(pretrained_dir: Path, device: str = "cpu"):
    """
    Load the official HairFastGAN pipeline.
    Requires HairFastGAN-main to be in the Python path.
    """
    # Add HairFastGAN repo to path if available
    repo_candidates = [
        Path(__file__).parents[3] / "HairFastGAN-main",
        Path(__file__).parents[2] / "HairFastGAN-main",
        Path(__file__).parents[1] / "HairFastGAN-main",
        Path("HairFastGAN-main"),
        Path("../HairFastGAN-main"),
        Path("../../HairFastGAN-main"),
    ]
    for repo in repo_candidates:
        if repo.exists() and str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
            break

    try:
        import argparse
        from hair_swap import HairFast

        hfg_dir = pretrained_dir / "HairFastGAN"
        original_cwd = os.getcwd()
        os.chdir(hfg_dir.parent)

        args = argparse.Namespace(
            save_all_dir=Path("output"),
            size=1024,
            ckpt=str(hfg_dir / "StyleGAN/ffhq.pt"),
            channel_multiplier=2,
            latent=512,
            n_mlp=8,
            device=device,
            batch_size=1,
            save_all=False,
            mixing=0.95,
            smooth=5,
            rotate_checkpoint=str(hfg_dir / "Rotate/rotate_best.pth"),
            blending_checkpoint=str(hfg_dir / "Blending/checkpoint.pth"),
            pp_checkpoint=str(hfg_dir / "PostProcess/pp_model.pth"),
        )

        model = HairFast(args)
        os.chdir(original_cwd)
        return model

    except ImportError:
        log.warning("HairFastGAN-main not found — official pipeline unavailable")
        return None


def run_hairfastgan(model, face_pil: Image.Image,
                    reference_pil: Image.Image) -> Image.Image:
    """Run official HairFastGAN inference."""
    import torch
    import torchvision.transforms.functional as TF

    face_t = TF.to_tensor(face_pil.convert("RGB"))
    ref_t  = TF.to_tensor(reference_pil.convert("RGB"))

    result = model.swap(face_t, ref_t, ref_t, align=True)

    if isinstance(result, tuple):
        result = result[0]

    return TF.to_pil_image(result.clamp(0, 1).cpu())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true",
                        help="Download all pretrained models")
    parser.add_argument("--pretrained_dir", default="pretrained",
                        help="Directory to save models")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.download:
        print("Downloading HairFastGAN pretrained models (~3GB total)...")
        download_models(Path(args.pretrained_dir))
        print("\nDone! Models saved to:", args.pretrained_dir + "/HairFastGAN/")
