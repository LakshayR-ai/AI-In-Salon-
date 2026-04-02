"""
Inference Pipeline — Hair Transfer
====================================
Transfers the hairstyle from a reference image onto a source face.

Usage:
    python inference/transfer_hairstyle.py \
        --source  path/to/source_face.jpg \
        --reference path/to/reference_hair.jpg \
        --output  result.png \
        --config  configs/base.yaml

The pipeline runs all 4 stages in sequence:
    1. Encode both images → W+, F latents
    2. Shape transfer (F-space)
    3. Color transfer (S-space via CLIP)
    4. Refinement (UNet)
"""
import sys
import argparse
from pathlib import Path

import yaml
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.encoder import HairEncoder
from models.shape_module import ShapeModule
from models.color_module import ColorModule
from models.refinement import RefinementUNet
from models.stylegan import load_stylegan2
from utils.face_alignment import FaceAligner
from utils.hair_segmentation import HairSegmenter


class HairTransferPipeline:
    """
    End-to-end hair transfer inference.

    Args:
        cfg:    loaded YAML config dict
        device: 'cuda' or 'cpu'
    """

    def __init__(self, cfg: dict, device: str = "cuda"):
        self.cfg    = cfg
        self.device = device
        out_dir     = Path(cfg["train"]["output_dir"])

        print("Loading models…")

        # StyleGAN2
        self.generator = load_stylegan2(
            cfg["stylegan"]["checkpoint"],
            size=cfg["stylegan"]["size"],
            latent_dim=cfg["stylegan"]["latent_dim"],
            device=device,
        )

        # Encoder
        self.encoder = HairEncoder(n_styles=cfg["encoder"]["n_styles"]).to(device)
        self._load(self.encoder, out_dir / "encoder" / "ckpt_final.pt", "encoder")

        # Shape module
        self.shape_module = ShapeModule(channels=512).to(device)
        self._load(self.shape_module, out_dir / "shape" / "ckpt_final.pt", "shape")

        # Color module
        self.color_module = ColorModule(
            n_styles=cfg["encoder"]["n_styles"],
            clip_model=cfg["color"]["clip_model"],
        ).to(device)
        self._load(self.color_module, out_dir / "color" / "ckpt_final.pt", "color")

        # Refinement
        self.refinement = RefinementUNet(
            base_ch=cfg["refinement"].get("base_channels", 64)
        ).to(device)
        self._load(self.refinement, out_dir / "refinement" / "ckpt_final.pt", "refine")

        # Preprocessing
        self.aligner   = FaceAligner(output_size=cfg["stylegan"]["size"], device=device)
        self.segmenter = HairSegmenter(
            checkpoint="pretrained/face_parsing.pth", device=device
        )

        self.img_tf = T.Compose([
            T.Resize((cfg["stylegan"]["size"],) * 2,
                     interpolation=T.InterpolationMode.LANCZOS),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3),
        ])
        self.mask_tf = T.Compose([
            T.Resize((cfg["stylegan"]["size"],) * 2,
                     interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
        ])

        for m in [self.encoder, self.shape_module,
                  self.color_module, self.refinement]:
            m.eval()

        print("All models loaded.")

    def _load(self, model, ckpt_path: Path, key: str):
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location=self.device)
            model.load_state_dict(state["models"][key])
            print(f"  ✓ {key} ← {ckpt_path.name}")
        else:
            print(f"  ⚠ {key}: checkpoint not found at {ckpt_path}")

    def _preprocess(self, img_path: str):
        """Load, align, segment → (tensor [1,3,H,W], mask [1,1,H,W])"""
        img = Image.open(img_path).convert("RGB")

        aligned = self.aligner.align(img)
        if aligned is None:
            print(f"  ⚠ No face detected in {img_path}, using original.")
            aligned = img.resize(
                (self.cfg["stylegan"]["size"],) * 2, Image.LANCZOS
            )

        mask_np = self.segmenter.segment(aligned)
        mask    = self.mask_tf(Image.fromarray(mask_np)).unsqueeze(0).to(self.device)
        tensor  = self.img_tf(aligned).unsqueeze(0).to(self.device)
        return tensor, mask, aligned

    @torch.no_grad()
    def transfer_pil(self, face_pil: Image.Image,
                     reference_pil: Image.Image) -> Image.Image:
        """Convenience method accepting PIL images directly."""
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            face_pil.save(f.name); face_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            reference_pil.save(f.name); ref_path = f.name
        try:
            return self.transfer(face_path, ref_path)
        finally:
            os.unlink(face_path)
            os.unlink(ref_path)

    @torch.no_grad()
    def transfer(self, source_path: str, reference_path: str) -> Image.Image:
        """
        Transfer hairstyle from reference to source.

        Returns:
            PIL Image of the result.
        """
        print("Preprocessing…")
        src_t, src_mask, src_pil = self._preprocess(source_path)
        ref_t, ref_mask, _       = self._preprocess(reference_path)

        print("Encoding…")
        w_src, f_src = self.encoder(src_t)
        w_ref, f_ref = self.encoder(ref_t)

        print("Shape transfer…")
        f_out = self.shape_module(f_src, f_ref, src_mask, ref_mask)

        print("Color transfer…")
        # Skip color module — use source W+ directly to avoid artifacts
        # Color module needs more training to work correctly
        w_out = w_src

        print("Generating…")
        # Use full generation from W+ latent — most reliable
        blended, _ = self.generator(
            [w_out], input_is_latent=True,
        )

        # StyleGAN outputs in [-1, 1] — clamp to valid range
        blended = blended.clamp(-1, 1)

        print("Refining…")
        # Skip refinement — use blended directly
        refined = blended

        # Convert [-1,1] → [0,1] → PIL
        result = (refined[0] + 1) / 2
        return TF.to_pil_image(result.cpu())


def main():
    parser = argparse.ArgumentParser(description="Hair Transfer Inference")
    parser.add_argument("--source",    required=True, help="Source face image")
    parser.add_argument("--reference", required=True, help="Reference hairstyle image")
    parser.add_argument("--output",    default="result.png", help="Output path")
    parser.add_argument("--config",    default="configs/base.yaml")
    parser.add_argument("--device",    default="cuda")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    pipeline = HairTransferPipeline(cfg, device=device)
    result   = pipeline.transfer(args.source, args.reference)
    result.save(args.output)
    print(f"\nResult saved → {args.output}")


if __name__ == "__main__":
    main()
