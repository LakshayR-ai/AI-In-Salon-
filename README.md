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

```bash
# Install dependencies
pip install -r hair_transfer/requirements.txt
pip install dlib-bin
pip install git+https://github.com/openai/CLIP.git

# Download pretrained weights
cd hair_transfer
python -c "
import requests, os
os.makedirs('pretrained', exist_ok=True)
files = {
    'pretrained/stylegan2-ffhq-256.pt': 'https://huggingface.co/AIRI-Institute/HairFastGAN/resolve/main/pretrained_models/StyleGAN/ffhq.pt',
    'pretrained/face_parsing.pth': 'https://huggingface.co/AI2lab/face-parsing.PyTorch/resolve/main/79999_iter.pth',
}
for dest, url in files.items():
    r = requests.get(url, stream=True)
    with open(dest, 'wb') as f:
        for chunk in r.iter_content(8192): f.write(chunk)
    print(f'Downloaded {dest}')
"

# Start the app
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
