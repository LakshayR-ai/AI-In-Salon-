"""
AI Salon — FastAPI backend
Uses HairFastGAN directly for production quality results.
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

log = logging.getLogger("salon")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

ROOT    = Path(__file__).resolve().parents[1]
model   = None


def _find_hairfastgan():
    candidates = [
        Path("/content/HairFastGAN-main"),
        ROOT.parent.parent / "HairFastGAN-main",
        ROOT.parent / "HairFastGAN-main",
        Path.cwd().parent / "HairFastGAN-main",
    ]
    for p in candidates:
        if p.exists() and (p / "hair_swap.py").exists():
            log.info(f"Found HairFastGAN at: {p}")
            return p
    log.warning(f"HairFastGAN not found. Checked: {[str(p) for p in candidates]}")
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    repo = _find_hairfastgan()
    if repo:
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        try:
            import torch
            original_cwd = os.getcwd()
            os.chdir(repo)
            log.info(f"Changed to directory: {repo}")
            log.info(f"Importing HairFastGAN from: {repo / 'hair_swap.py'}")
            
            from hair_swap import HairFast, get_parser
            args = get_parser().parse_args([])
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            log.info(f"Loading HairFastGAN model on {args.device}...")
            model = HairFast(args)
            os.chdir(original_cwd)
            log.info(f"✅ HairFastGAN loaded successfully on {args.device}")
        except Exception as e:
            log.error(f"❌ HairFastGAN failed to load: {e}")
            import traceback
            log.error(traceback.format_exc())
            try: os.chdir(original_cwd)
            except: pass
    else:
        log.warning("❌ HairFastGAN repo not found in any candidate location")
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
        "pipeline": "hairfastgan" if model else "unavailable",
        "cuda":     torch.cuda.is_available(),
    }


@app.post("/swap")
async def swap(
    face:      UploadFile = File(...),
    reference: UploadFile = File(...),
):
    if not model:
        raise HTTPException(503, "HairFastGAN not loaded. Ensure GPU is available and repo is cloned.")

    face_pil = read_pil(face)
    ref_pil  = read_pil(reference)

    try:
        import torch
        import torchvision.transforms.functional as TF

        repo = _find_hairfastgan()
        original_cwd = os.getcwd()
        if repo: os.chdir(repo)

        face_t = TF.to_tensor(face_pil)
        ref_t  = TF.to_tensor(ref_pil)

        result = model.swap(face_t, ref_t, ref_t, align=True)
        os.chdir(original_cwd)

        if isinstance(result, tuple):
            result = result[0]

        result_pil = TF.to_pil_image(result.clamp(0, 1).cpu())

    except Exception as e:
        log.exception("Swap failed")
        try: os.chdir(original_cwd)
        except: pass
        raise HTTPException(500, str(e))

    return JSONResponse({"result": pil_to_b64(result_pil)})


# Serve frontend
STATIC = Path(__file__).parent / "static"
if STATIC.exists():
    app.mount("/", StaticFiles(directory=str(STATIC), html=True), name="static")
