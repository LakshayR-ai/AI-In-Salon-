# AI Salon — Hair Transfer System

A deep learning system that transfers hairstyles from a reference image onto a source face while preserving the person's identity.

![Pipeline](docs/pipeline.png)

---

## Demo

Upload your face photo + a hairstyle reference → get the result in seconds.

---

## Architecture

The system uses a 4-stage neural network pipeline:

```
Source Face + Reference Hair
        ↓
   Encoder (ResNet50)
   Maps images → StyleGAN2 latent space (W+, F)
        ↓
   Shape Module (Cross-Attention)
   Transfers hair shape in F-space (32×32×512)
        ↓
   Color Module (CLIP-guided)
   Transfers hair color in W+ space (layers 6–14)
        ↓
   StyleGAN2 Generator (frozen)
   Decodes latents → 256×256 image
        ↓
   Refinement UNet
   Restores face details
        ↓
   Final Result
```

**Models:**
- Encoder — ResNet50 backbone with 14 style heads → W+ latent
- Shape Module — Cross-attention in F-space with mask-guided blending
- Color Module — CLIP ViT-B/32 guided S-space modulation
- Refinement — Residual UNet for detail restoration
- Generator — StyleGAN2 pretrained on FFHQ (frozen)

**Loss functions:**
- L1 reconstruction
- Perceptual loss (VGG16)
- Identity loss (ArcFace)
- CLIP directional loss
- Adversarial loss (LSGAN)

---

## Project Structure

```
hair_transfer/
├── app/
│   ├── backend.py          FastAPI server
│   └── static/
│       └── index.html      Web UI
├── configs/
│   ├── base.yaml
│   ├── train_encoder.yaml
│   ├── train_shape.yaml
│   ├── train_color.yaml
│   └── train_refinement.yaml
├── datasets/
│   └── preprocess.py       Face alignment + hair segmentation
├── inference/
│   ├── transfer_hairstyle.py   Full GAN pipeline
│   └── quick_test.py           Mask-based fallback
├── models/
│   ├── encoder/            pSp-style ResNet50 encoder
│   ├── shape_module/       F-space cross-attention
│   ├── color_module/       CLIP-guided S-space modulation
│   ├── refinement/         UNet detail restoration
│   ├── stylegan/           StyleGAN2 generator
│   └── discriminator.py    PatchGAN discriminator
├── training/
│   ├── base_trainer.py
│   ├── train_encoder.py
│   ├── train_shape_module.py
│   ├── train_color_module.py
│   └── train_refinement.py
├── utils/
│   ├── face_alignment.py   dlib 68-point landmark alignment
│   ├── hair_segmentation.py BiSeNet face parser
│   ├── dataset_loader.py
│   └── losses.py
├── pretrained/             Model weights (not in repo)
├── outputs/                Training checkpoints
└── train_rtx4050.bat       Optimized training script
```

---

## Setup

### Requirements

- Python 3.10+
- CUDA GPU (RTX 4050 or better recommended)
- 8GB+ RAM

### Install dependencies

```bash
pip install -r requirements.txt
pip install dlib-bin
pip install git+https://github.com/openai/CLIP.git
```

### Download pretrained weights

Place these files in `pretrained/`:

| File | Source |
|------|--------|
| `stylegan2-ffhq-256.pt` | [Google Drive](https://drive.google.com/file/d/1PQutd-JboOCOZqmd95XWxWrO8gGEvRcO) |
| `face_parsing.pth` | [Google Drive](https://drive.google.com/file/d/154JgKpzCPW82qINcVieuPH3fZ2e0P812) |

---

## Running the App

```bash
cd hair_transfer
uvicorn app.backend:app --host 0.0.0.0 --port 8000 --reload
```

Open http://localhost:8000

---

## Training

### 1. Prepare dataset

Download face images (FFHQ recommended, minimum 1000 images) into `raw_images/`, then:

```bash
python datasets/preprocess.py \
    --src_dir raw_images \
    --out_dir dataset \
    --seg_ckpt pretrained/face_parsing.pth \
    --device cuda
```

### 2. Train all stages

**On RTX 4050 (recommended):**
```bat
train_rtx4050.bat
```

**Manually:**
```bash
python training/train_encoder.py
python training/train_shape_module.py
python training/train_color_module.py
python training/train_refinement.py
```

Checkpoints save to `outputs/` every 10 epochs.

### Training time estimates

| Hardware | Dataset | Epochs | Total time |
|----------|---------|--------|-----------|
| RTX 4050 | 1,000 images | 50 | ~10 hours |
| RTX 4050 | 10,000 images | 100 | ~20 hours |
| RTX 4090 | 10,000 images | 100 | ~5 hours |
| Kaggle P100 | 1,000 images | 30 | ~12 hours |

---

## Inference

```bash
python inference/transfer_hairstyle.py \
    --source path/to/face.jpg \
    --reference path/to/hairstyle.jpg \
    --output result.png \
    --device cuda
```

---

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/swap` | Transfer hairstyle (face + reference) |
| GET | `/health` | Server status + pipeline mode |

**Example:**
```python
import requests

with open('face.jpg', 'rb') as f, open('ref.jpg', 'rb') as r:
    response = requests.post('http://localhost:8000/swap',
                             files={'face': f, 'reference': r})

result_b64 = response.json()['result']
```

---

## Results

The system operates in two modes:

- **Full GAN mode** — uses all 4 trained modules for realistic hair transfer
- **Quick blend mode** — mask-based fallback when models aren't trained, instant results

---

## Inspired by

- [HairFastGAN](https://github.com/AIRI-Institute/HairFastGAN) — NeurIPS 2024
- [pSp](https://github.com/eladrich/pixel2style2pixel) — pixel2style2pixel encoder
- [StyleGAN2](https://github.com/NVlabs/stylegan2) — NVIDIA

---

## License

MIT License — free to use for personal and commercial projects.
