# Google Colab Setup Guide

Complete guide to run HairFastGAN Hair Transfer in Google Colab without Google Drive.

## 🚀 Quick Start

### Option 1: One-Click Setup (Recommended)

Copy and paste this into a Colab cell:

```python
# Download and run setup script
!wget -q https://raw.githubusercontent.com/LakshayR-ai/AI-In-Salon-/main/colab_setup.py
%run colab_setup.py

# After runtime restart, run:
quick_start()
```

### Option 2: Manual Step-by-Step

#### Step 1: Install Dependencies

```python
import os
os.chdir('/content')

# Install PyTorch with CUDA
!pip install torch==2.2.0+cu121 torchvision==0.17.0+cu121 --index-url https://download.pytorch.org/whl/cu121 -q
!pip install ninja scipy face-alignment dlib-bin addict -q
!pip install git+https://github.com/openai/CLIP.git -q
!pip install fastapi uvicorn python-multipart pyngrok requests tqdm -q

print("✓ Dependencies installed!")
print("⚠️  RESTART RUNTIME NOW: Runtime → Restart runtime")
```

**⚠️ IMPORTANT: Restart runtime after this step!**

---

#### Step 2: Clone Repositories (After Restart)

```python
import os
os.chdir('/content')

# Clone HairFastGAN
!git clone --depth=1 https://github.com/AIRI-Institute/HairFastGAN /content/HairFastGAN-main 2>/dev/null || echo "exists"

# Clone AI-In-Salon
!git clone --depth=1 https://github.com/LakshayR-ai/AI-In-Salon-.git /content/salon 2>/dev/null || echo "exists"

print("✓ Repositories cloned!")
```

---

#### Step 3: Download Model Weights from HuggingFace

```python
import requests, os
from tqdm import tqdm

HF = "https://huggingface.co/AIRI-Institute/HairFastGAN/resolve/main/pretrained_models"
base = '/content/HairFastGAN-main/pretrained_models'

files = {
    f"{base}/PostProcess/latent_avg.pt": 
        f"{HF}/PostProcess/latent_avg.pt",
    f"{base}/sean_checkpoints/CelebA-HQ_pretrained/latest_net_G.pth": 
        f"{HF}/sean_checkpoints/CelebA-HQ_pretrained/latest_net_G.pth",
    f"{base}/sean_checkpoints/CelebA-HQ_pretrained/latest_net_D.pth": 
        f"{HF}/sean_checkpoints/CelebA-HQ_pretrained/latest_net_D.pth",
    f"{base}/BiSeNet/face_parsing_79999_iter.pth":
        f"{HF}/BiSeNet/face_parsing_79999_iter.pth",
}

print("📥 Downloading model weights...\n")

for dest, url in files.items():
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    
    if os.path.exists(dest) and os.path.getsize(dest) > 1024*100:
        print(f"✓ {os.path.basename(dest)} already exists")
        continue
    
    print(f"↓ Downloading {os.path.basename(dest)}...")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    total = int(r.headers.get('content-length', 0))
    
    with open(dest, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True, desc=os.path.basename(dest)) as bar:
        for chunk in r.iter_content(8192):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))
    
    print(f"✓ Downloaded\n")

print("✓ All weights downloaded!")
```

---

#### Step 4: Test Model Loading

```python
import sys, os
sys.path.insert(0, '/content/HairFastGAN-main')
os.chdir('/content/HairFastGAN-main')

from hair_swap import HairFast, get_parser

args = get_parser().parse_args([])
args.device = 'cuda'

print("🔄 Loading model...")
model = HairFast(args)
print("✓ Model loaded successfully!")
```

---

#### Step 5: Setup and Start Server

```python
import sys, os, threading, time, requests, shutil
from google.colab.output import eval_js

# Kill existing server
os.system("fuser -k 8000/tcp 2>/dev/null")

# Setup paths
sys.path.insert(0, '/content/HairFastGAN-main')
sys.path.insert(0, '/content/salon/hair_transfer')

# Copy face parsing model
os.makedirs('/content/salon/hair_transfer/pretrained', exist_ok=True)
shutil.copy(
    '/content/HairFastGAN-main/pretrained_models/BiSeNet/face_parsing_79999_iter.pth',
    '/content/salon/hair_transfer/pretrained/face_parsing.pth'
)

# Create config
os.makedirs('/content/salon/hair_transfer/configs', exist_ok=True)
with open('/content/salon/hair_transfer/configs/base.yaml', 'w') as f:
    f.write("device: cuda\nstylegan:\n  size: 256\n")

# Start server
def run():
    os.chdir('/content/salon/hair_transfer')
    os.system("python -m uvicorn app.backend:app --host 0.0.0.0 --port 8000")

threading.Thread(target=run, daemon=True).start()

print("🚀 Starting server...")
for i in range(40):
    time.sleep(5)
    try:
        r = requests.get('http://localhost:8000/health', timeout=2)
        print(f"\n✓ Server ready: {r.json()}")
        break
    except:
        print(f"  {(i+1)*5}s...", end='\r')

public_url = eval_js("google.colab.kernel.proxyPort(8000)")
print(f"\n🌐 Public URL: {public_url}")
print("\n✓ Server is running!")
```

---

## 📊 Model Files Downloaded

The script downloads these files from HuggingFace:

| File | Size | Purpose |
|------|------|---------|
| `latent_avg.pt` | ~1 MB | StyleGAN latent space average |
| `latest_net_G.pth` | ~200 MB | SEAN Generator |
| `latest_net_D.pth` | ~100 MB | SEAN Discriminator |
| `face_parsing_79999_iter.pth` | ~50 MB | BiSeNet face parsing |

**Total: ~350 MB**

## 🔧 Troubleshooting

### Runtime Disconnected
If Colab disconnects, just re-run from Step 2 (repos and weights will be cached).

### Out of Memory
- Use GPU runtime: Runtime → Change runtime type → GPU
- Reduce batch size in config if needed

### Download Fails
- Check internet connection
- Try downloading individual files manually from HuggingFace
- The script skips already downloaded files, so you can retry safely

### Model Loading Fails
- Ensure all 4 model files are downloaded completely
- Check file sizes match expected sizes
- Restart runtime and try again

## 📝 API Usage

Once the server is running, you can use it:

```python
import requests
from PIL import Image
import io

# Your public URL from step 5
url = "YOUR_PUBLIC_URL_HERE"

# Upload images
face_img = open('face.jpg', 'rb')
hair_img = open('hair.jpg', 'rb')

response = requests.post(
    f"{url}/transfer",
    files={
        'face_image': face_img,
        'hair_image': hair_img
    }
)

# Get result
result_img = Image.open(io.BytesIO(response.content))
result_img.show()
```

## 🔗 Resources

- **HairFastGAN**: https://github.com/AIRI-Institute/HairFastGAN
- **Model Weights**: https://huggingface.co/AIRI-Institute/HairFastGAN
- **Project Repo**: https://github.com/LakshayR-ai/AI-In-Salon-

## 💡 Tips

1. **Save Time**: Weights are cached in `/content/`, so they persist during the session
2. **Faster Setup**: After first run, you can skip to Step 4
3. **Public URL**: The Colab proxy URL changes each time you restart the server
4. **GPU Usage**: Monitor GPU usage in Colab's Resources tab
