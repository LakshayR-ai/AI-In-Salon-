# Simple Colab Setup - Hair Transfer with Gradio UI

## One-Time Setup (Run Once Per Session)

### Cell 1: Install Dependencies
```python
# Install required packages
!pip install torch==2.2.0+cu121 torchvision==0.17.0+cu121 --index-url https://download.pytorch.org/whl/cu121 -q
!pip install "numpy<2.0" -q
!pip install gradio face-alignment dlib-bin addict -q
!pip install git+https://github.com/openai/CLIP.git -q

print("✅ Dependencies installed")
print("⚠️  RESTART RUNTIME: Runtime → Restart runtime")
```

**⚠️ RESTART RUNTIME after this cell!**

---

### Cell 2: Complete Setup (After Restart)
```python
import os
os.chdir('/content')

# Clone HairFastGAN
!git clone https://github.com/AIRI-Institute/HairFastGAN.git /content/HairFastGAN-main 2>/dev/null || echo "exists"

# Clone AI-In-Salon (for Gradio app)
!git clone https://github.com/LakshayR-ai/AI-In-Salon-.git /content/salon 2>/dev/null || echo "exists"

# Download ALL pretrained models using git-lfs (easier than individual downloads)
print("📥 Downloading pretrained models...")
os.chdir('/content/HairFastGAN-main')

# Use HuggingFace CLI to download entire pretrained_models folder
!pip install huggingface_hub -q
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="AIRI-Institute/HairFastGAN",
    allow_patterns="pretrained_models/**",
    local_dir="/content/HairFastGAN-main",
    local_dir_use_symlinks=False
)

print("✅ All models downloaded!")
```

---

### Cell 3: Launch Gradio UI
```python
import os
os.environ['HAIRFASTGAN_PATH'] = '/content/HairFastGAN-main'

# Run Gradio app
!python /content/salon/gradio_app.py
```

This will:
- Load the HairFastGAN model
- Launch a professional Gradio UI
- Provide a public URL you can share

---

## Alternative: Quick Test (No UI)
```python
import sys
import os
sys.path.insert(0, '/content/HairFastGAN-main')
os.chdir('/content/HairFastGAN-main')

from hair_swap import HairFast, get_parser
import torch

# Load model
args = get_parser().parse_args([])
args.device = 'cuda'
model = HairFast(args)

print("✅ Model loaded! Upload images to test.")
```

---

## Features

✅ **Simple Setup** - Just 3 cells  
✅ **Professional UI** - Clean Gradio interface  
✅ **Public URL** - Share with anyone  
✅ **No Missing Files** - Downloads everything at once  
✅ **Fast** - Uses GPU acceleration  

---

## Troubleshooting

### NumPy Error
If you see NumPy errors, run:
```python
!pip uninstall numpy -y -q
!pip install "numpy<2.0" -q
```
Then restart runtime.

### Model Not Loading
Check if all files downloaded:
```python
!ls -lh /content/HairFastGAN-main/pretrained_models/
```

### Out of Memory
Use smaller images or restart runtime to clear memory.
