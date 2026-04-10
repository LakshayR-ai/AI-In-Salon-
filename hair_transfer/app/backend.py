"""
AI Salon — FastAPI backend

Pipeline priority:
  1. Official HairFastGAN (GPU only — production quality)
  2. Quick mask blend (CPU fallback)
"""
import io
import os
import sys
import base64
import logging
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

log = logging.getLogger("salon")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

hairfast_model  = None
pipeline_mode   = "quick_blend"


def _load_cfg():
    cfg_path = ROOT / "configs" / "base.yaml"
    if cfg_path.exists():
        with open(cfg_path) as f:
            return yaml.safe_load(f)
    return {"device": "cpu", "stylegan": {"size": 256}}


def _try_load_hairfastgan():
    """Try to load official HairFastGAN model."""
    import torch
    if not torch.cuda.is_available():
        log.info("No GPU — skipping HairFastGAN")
        return None

    # Find HairFastGAN repo
    candidates = [
        Path("/content/HairFastGAN-main"),
        ROOT.parents[1] / "HairFastGAN-main",
        ROOT.parents[0] / "HairFastGAN-main",
    ]
    repo = next((p for p in candidates if p.exists()), None)
    if not repo:
        log.info("HairFastGAN repo not found")
        return None

    try:
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))

        original_cwd = os.getcwd()
        os.chdir(repo)

        from hair_swap import HairFast, get_parser
        args = get_parser().parse_args([])
        args.device = "cuda"
        model = HairFast(args)

        os.chdir(original_cwd)
        log.info("HairFastGAN loaded — production quality")
        return model
    except Exception as e:
        log.warning(f"HairFastGAN failed: {e}")
        try:
            os.chdir(original_cwd)
        except Exception:
            pass
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global hairfast_model, pipeline_mode

    hairfast_model = _try_load_hairfastgan()
    if hairfast_model:
        pipeline_mode = "hairfastgan"
    else:
        pipeline_mode = "quick_blend"
        log.info("Using quick blend fallback")

    log.info(f"Pipeline: {pipeline_mode}")
    yield


app = FastAPI(title="AI Salon", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


def read_pil(upload: UploadFile) -> Image.Image:
    try:
        return Image.open(io.BytesIO(upload.file.read())).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Cannot read image: {e}")


def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def run_hairfastgan(face_pil: Image.Image,
                    ref_pil: Image.Image) -> Image.Image:
    """Run official HairFastGAN inference."""
    import torch
    import torchvision.transforms.functional as TF

    # Find HairFastGAN repo for chdir
    candidates = [
        Path("/content/HairFastGAN-main"),
        ROOT.parents[1] / "HairFastGAN-main",
    ]
    repo = next((p for p in candidates if p.exists()), None)

    original_cwd = os.getcwd()
    if repo:
        os.chdir(repo)

    try:
        face_t = TF.to_tensor(face_pil)
        ref_t  = TF.to_tensor(ref_pil)

        result = hairfast_model.swap(face_t, ref_t, ref_t, align=True)

        if isinstance(result, tuple):
            result = result[0]

        return TF.to_pil_image(result.clamp(0, 1).cpu())
    finally:
        os.chdir(original_cwd)


@app.get("/health")
def health():
    import torch
    return {
        "status":   "ok",
        "pipeline": pipeline_mode,
        "cuda":     torch.cuda.is_available(),
    }


@app.post("/swap")
async def swap(
    face:      UploadFile = File(...),
    reference: UploadFile = File(...),
):
    face_pil = read_pil(face)
    ref_pil  = read_pil(reference)
    cfg      = _load_cfg()

    try:
        if hairfast_model:
            result = run_hairfastgan(face_pil, ref_pil)
        else:
            from inference.quick_test import blend_hair
            seg_ckpt = str(ROOT / "pretrained" / "face_parsing.pth")
            result   = blend_hair(face_pil, ref_pil, seg_ckpt,
                                  size=512,
                                  device=cfg.get("device", "cpu"))
    except Exception as e:
        log.exception("Swap failed")
        raise HTTPException(500, str(e))

    return JSONResponse({
        "result":   pil_to_b64(result),
        "pipeline": pipeline_mode,
    })


# ── Serve frontend ────────────────────────────────────────────────────────────
STATIC = Path(__file__).parent / "static"
if STATIC.exists():
    app.mount("/", StaticFiles(directory=str(STATIC), html=True), name="static")
