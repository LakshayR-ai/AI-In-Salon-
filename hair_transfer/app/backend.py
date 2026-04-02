"""
AI Salon — FastAPI backend
Serves the hair transfer inference pipeline via HTTP.
"""
import io
import sys
import base64
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image

# ── Make hair_transfer importable ────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference.transfer_hairstyle import HairTransferPipeline
from inference.quick_test import blend_hair

import yaml
log = logging.getLogger("salon")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

# ── Global pipeline ───────────────────────────────────────────────────────────
pipeline: HairTransferPipeline | None = None


def _load_cfg():
    with open(ROOT / "configs" / "base.yaml") as f:
        return yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    cfg = _load_cfg()
    # Try loading full trained pipeline
    try:
        pipeline = HairTransferPipeline(cfg, device=cfg.get("device", "cpu"))
        log.info("Full pipeline loaded.")
    except Exception as e:
        log.warning(f"Full pipeline failed ({e}) — using quick blend fallback.")
        pipeline = None
    yield


app = FastAPI(title="AI Salon", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def read_pil(upload: UploadFile) -> Image.Image:
    try:
        return Image.open(io.BytesIO(upload.file.read())).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Cannot read image: {e}")


def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    import torch
    return {
        "status": "ok",
        "pipeline": "full" if pipeline else "quick_blend",
        "cuda": torch.cuda.is_available(),
    }


@app.post("/swap")
async def swap(
    face:      UploadFile = File(...),
    reference: UploadFile = File(...),
):
    """Transfer hairstyle from reference onto face."""
    face_pil = read_pil(face)
    ref_pil  = read_pil(reference)

    try:
        cfg      = _load_cfg()
        seg_ckpt = str(ROOT / "pretrained" / "face_parsing.pth")
        result   = blend_hair(face_pil, ref_pil, seg_ckpt,
                              size=512,
                              device=cfg.get("device", "cpu"))
    except Exception as e:
        log.exception("Swap failed")
        raise HTTPException(500, str(e))

    return JSONResponse({"result": pil_to_b64(result)})


# ── Serve frontend ────────────────────────────────────────────────────────────
STATIC = Path(__file__).parent / "static"
if STATIC.exists():
    app.mount("/", StaticFiles(directory=str(STATIC), html=True), name="static")
