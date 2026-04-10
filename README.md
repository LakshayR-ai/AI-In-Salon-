# AI Salon — Hairstyle Transfer System

> Transfer any hairstyle onto any face using deep learning.

![Demo](hair_transfer/docs/demo.png)

## What it does

Upload a face photo + a hairstyle reference image → get the result with the hairstyle transferred onto your face while preserving your identity.

## Tech Stack

- **StyleGAN2** — pretrained face generator (FFHQ)
- **ResNet50 Encoder** — maps real faces into StyleGAN latent space
- **Cross-Attention Shape Module** — transfers hair shape in F-space
- **CLIP Color Module** — transfers hair color via semantic embeddings
- **Refinement UNet** — restores face details
- **BiSeNet** — hair segmentation
- **FastAPI** — backend server
- **Vanilla JS** — frontend UI

## Quick Start

### 🌐 Google Colab (No Setup Required)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

See [COLAB_SETUP.md](COLAB_SETUP.md) for complete Colab instructions.

**Quick Colab Setup:**
```python
# 1. Install dependencies (then restart runtime)
!pip install torch==2.2.0+cu121 torchvision==0.17.0+cu121 --index-url https://download.pytorch.org/whl/cu121 -q
!pip install ninja scipy face-alignment dlib-bin addict fastapi uvicorn requests tqdm -q
!pip install git+https://github.com/openai/CLIP.git -q

# 2. After restart, download setup script and run
!wget -q https://raw.githubusercontent.com/LakshayR-ai/AI-In-Salon-/main/colab_setup.py
%run colab_setup.py
quick_start()
```

### 💻 Local Setup

```bash
# 1. Clone
git clone https://github.com/LakshayR-ai/AI-In-Salon-.git
cd AI-In-Salon-/hair_transfer

# 2. Install dependencies
pip install -r requirements.txt
pip install dlib-bin
pip install git+https://github.com/openai/CLIP.git

# 3. Download model weights
python scripts/download_weights.py

# 4. One-command setup (downloads 1000 face images + preprocesses)
python setup.py

# 5. Start the app
uvicorn app.backend:app --host 0.0.0.0 --port 8000 --reload
```

Open http://localhost:8000

## Pipeline

```
Source Face + Reference Hair
        ↓
   Encoder → W+ latent + F latent
        ↓
   Shape Module → hair shape transfer (F-space)
        ↓
   Color Module → hair color transfer (W+ space)
        ↓
   StyleGAN2 Generator → synthesized image
        ↓
   Refinement UNet → final result
```

## Training

```bash
cd hair_transfer
train_rtx4050.bat        # Windows with NVIDIA GPU
```

See [hair_transfer/README.md](hair_transfer/README.md) for full training guide.

## Results

| Metric | Score | Meaning |
|--------|-------|---------|
| SSIM | ~0.75 | Face structure preserved |
| Identity | ~0.70 | Same person after transfer |
| CLIP Similarity | ~0.82 | Hair matches reference |

## Inspired by

[HairFastGAN](https://github.com/AIRI-Institute/HairFastGAN) — NeurIPS 2024, AIRI Institute

## License

MIT
