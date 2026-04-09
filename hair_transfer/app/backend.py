"""
AI Salon — FastAPI backend

Pipeline priority:
  1. Official HairFastGAN (best quality — download with --download flag)
  2. Custom trained pipeline (our trained checkpoints)
  3. Quick mask blend (fallback — always works, no training needed)
"""
import io
import os
import sys
import base64
import logging
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
sys.path.insert(0, str(ROOT.parent))  # workspace root

# Change working directory to hair_transfer so relative imports work
os.chdir(ROOT)

from inference.quick_test import blend_hair


def _try_huggingface_spaces(face_pil: Image.Image,
                             ref_pil: Image.Image) -> Image.Image | None:
    """
    Call the official HairFastGAN HuggingFace Spaces demo as an API.
    Returns result PIL image or None if unavailable.
    Requires: pip install gradio-client
    """
    try:
        import tempfile
        from gradio_client import Client, handle_file

        client = Client("AIRI-Institute/HairFastGAN")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f1, \
             tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f2:
            face_pil.save(f1.name)
            ref_pil.save(f2.name)

            result = client.predict(
                handle_file(f1.name),   # face
                handle_file(f2.name),   # shape
                handle_file(f2.name),   # color (same as shape)
                api_name="/swap"
            )

        import os
        os.unlink(f1.name)
        os.unlink(f2.name)

        if isinstance(result, str):
            return Image.open(result).convert("RGB")
        return None

    except Exception as e:
        log.debug(f"HuggingFace Spaces unavailable: {e}")
        return None

log = logging.getLogger("salon")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

official_model  = None
custom_pipeline = None
pipeline_mode   = "quick_blend"


def _load_cfg():
    with open(ROOT / "configs" / "base.yaml") as f:
        return yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global official_model, custom_pipeline, pipeline_mode
    cfg    = _load_cfg()
    device = cfg.get("device", "cpu")
    pretrained = ROOT / "pretrained"

    # ── Tier 1: Official HairFastGAN (GPU only) ──────────────────────────────
    import torch
    if torch.cuda.is_available():
        try:
            from app.hairfastgan_pipeline import models_available, load_hairfastgan
            if models_available(pretrained):
                log.info("Loading official HairFastGAN models (production quality)...")
                official_model = load_hairfastgan(pretrained, device)
                if official_model:
                    pipeline_mode = "official_hairfastgan"
                    log.info("Official HairFastGAN ready.")
        except Exception as e:
            log.info(f"Official pipeline not available: {e}")
    else:
        log.info("No GPU detected — skipping official HairFastGAN (requires CUDA)")

    # ── Tier 2: Custom trained pipeline ──────────────────────────────────────
    if not official_model:
        try:
            sys.path.insert(0, str(ROOT))
            from inference.transfer_hairstyle import HairTransferPipeline
            custom_pipeline = HairTransferPipeline(cfg, device=device)
            pipeline_mode   = "custom_trained"
            log.info("Custom trained pipeline loaded.")
        except Exception as e:
            log.info(f"Custom pipeline not available: {e}")

    # ── Tier 3: Quick blend fallback ─────────────────────────────────────────
    if not official_model and not custom_pipeline:
        pipeline_mode = "quick_blend"
        log.info("Using quick mask blend (no training required).")

    log.info(f"Active pipeline: {pipeline_mode}")
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
        if official_model:
            from app.hairfastgan_pipeline import run_hairfastgan
            result = run_hairfastgan(official_model, face_pil, ref_pil)

        elif _try_huggingface_spaces(face_pil, ref_pil) is not None:
            result = _try_huggingface_spaces(face_pil, ref_pil)

        else:
            seg_ckpt = str(ROOT / "pretrained" / "face_parsing.pth")
            result   = blend_hair(face_pil, ref_pil, seg_ckpt,
                                  size=512,
                                  device=cfg.get("device", "cpu"))
    except Exception as e:
        log.exception("Swap failed")
        raise HTTPException(500, str(e))

    return JSONResponse({"result": pil_to_b64(result),
                         "pipeline": pipeline_mode})


# ── Serve frontend ────────────────────────────────────────────────────────────
STATIC = Path(__file__).parent / "static"
if STATIC.exists():
    app.mount("/", StaticFiles(directory=str(STATIC), html=True), name="static")
