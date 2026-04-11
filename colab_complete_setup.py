"""
Complete One-Cell Colab Setup for AI-In-Salon Hair Transfer
Run this after installing dependencies and restarting runtime
"""

import os
import sys
import subprocess
import time
import requests
from tqdm import tqdm

print("="*60)
print("AI-IN-SALON HAIR TRANSFER - COMPLETE SETUP")
print("="*60 + "\n")

# ============================================================================
# STEP 1: Clone Repositories
# ============================================================================
print("📥 STEP 1: Cloning repositories...\n")

os.chdir('/content')

# Clone HairFastGAN
if not os.path.exists('/content/HairFastGAN-main'):
    os.system('git clone https://github.com/AIRI-Institute/HairFastGAN.git /content/HairFastGAN-main')
    print("✓ HairFastGAN cloned")
else:
    print("✓ HairFastGAN exists")

# Clone AI-In-Salon
if not os.path.exists('/content/salon'):
    os.system('git clone https://github.com/LakshayR-ai/AI-In-Salon-.git /content/salon')
    print("✓ AI-In-Salon cloned")
else:
    print("✓ AI-In-Salon exists")

# Verify
if not os.path.exists('/content/HairFastGAN-main/hair_swap.py'):
    raise Exception("❌ HairFastGAN clone failed!")
if not os.path.exists('/content/salon/hair_transfer'):
    raise Exception("❌ AI-In-Salon clone failed!")

print("\n✅ Repositories ready\n")

# ============================================================================
# STEP 2: Download ALL Pretrained Models (5.8 GB)
# ============================================================================
print("📥 STEP 2: Downloading pretrained models (~5.8 GB)...\n")

HF = "https://huggingface.co/AIRI-Institute/HairFastGAN/resolve/main/pretrained_models"
base = '/content/HairFastGAN-main/pretrained_models'

all_models = {
    f"{base}/StyleGAN/ffhq.pt": f"{HF}/StyleGAN/ffhq.pt",
    f"{base}/PostProcess/latent_avg.pt": f"{HF}/PostProcess/latent_avg.pt",
    f"{base}/PostProcess/pp_model.pth": f"{HF}/PostProcess/pp_model.pth",
    f"{base}/sean_checkpoints/CelebA-HQ_pretrained/latest_net_G.pth": f"{HF}/sean_checkpoints/CelebA-HQ_pretrained/latest_net_G.pth",
    f"{base}/sean_checkpoints/CelebA-HQ_pretrained/latest_net_D.pth": f"{HF}/sean_checkpoints/CelebA-HQ_pretrained/latest_net_D.pth",
    f"{base}/BiSeNet/face_parsing_79999_iter.pth": f"{HF}/BiSeNet/face_parsing_79999_iter.pth",
    f"{base}/Rotate/rotate_best.pth": f"{HF}/Rotate/rotate_best.pth",
    f"{base}/Blending/checkpoint.pth": f"{HF}/Blending/checkpoint.pth",
    f"{base}/FeatureStyleEncoder/backbone.pth": f"{HF}/FeatureStyleEncoder/backbone.pth",
    f"{base}/FeatureStyleEncoder/psp_ffhq_encode.pt": f"{HF}/FeatureStyleEncoder/psp_ffhq_encode.pt",
    f"{base}/FeatureStyleEncoder/79999_iter.pth": f"{HF}/FeatureStyleEncoder/79999_iter.pth",
    f"{base}/FeatureStyleEncoder/143_enc.pth": f"{HF}/FeatureStyleEncoder/143_enc.pth",
    f"{base}/encoder4editing/e4e_ffhq_encode.pt": f"{HF}/encoder4editing/e4e_ffhq_encode.pt",
    f"{base}/ShapeAdaptor/mask_generator.pth": f"{HF}/ShapeAdaptor/mask_generator.pth",
}

downloaded = 0
skipped = 0

for dest, url in all_models.items():
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    
    if os.path.exists(dest) and os.path.getsize(dest) > 1024*100:
        skipped += 1
        continue
    
    try:
        r = requests.get(url, stream=True)
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        
        with open(dest, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True, desc=os.path.basename(dest)) as bar:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        downloaded += 1
    except Exception as e:
        print(f"❌ Failed to download {os.path.basename(dest)}: {e}")

print(f"\n✅ Models ready (Downloaded: {downloaded}, Cached: {skipped})\n")

# ============================================================================
# STEP 3: Setup Configuration
# ============================================================================
print("📋 STEP 3: Setting up configuration...\n")

import shutil

# Copy face parsing
os.makedirs('/content/salon/hair_transfer/pretrained', exist_ok=True)
shutil.copy(f'{base}/BiSeNet/face_parsing_79999_iter.pth',
            '/content/salon/hair_transfer/pretrained/face_parsing.pth')

# Create config
os.makedirs('/content/salon/hair_transfer/configs', exist_ok=True)
with open('/content/salon/hair_transfer/configs/base.yaml', 'w') as f:
    f.write("device: cuda\nstylegan:\n  size: 256\n")

# Set environment variable for HairFastGAN path
os.environ['HAIRFASTGAN_PATH'] = '/content/HairFastGAN-main'

print("✅ Configuration complete\n")

# ============================================================================
# STEP 4: Start Server
# ============================================================================
print("🚀 STEP 4: Starting FastAPI server...\n")

# Kill existing
os.system("fuser -k 8000/tcp 2>/dev/null")
time.sleep(2)

# Add to path
sys.path.insert(0, '/content/HairFastGAN-main')
sys.path.insert(0, '/content/salon/hair_transfer')

# Start server
process = subprocess.Popen(
    ['python', '-m', 'uvicorn', 'app.backend:app', '--host', '0.0.0.0', '--port', '8000'],
    cwd='/content/salon/hair_transfer',
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
    env={**os.environ, 'HAIRFASTGAN_PATH': '/content/HairFastGAN-main'}
)

print("⏳ Loading HairFastGAN model (60-90 seconds)...\n")

success = False
for i in range(150):
    time.sleep(1)
    line = process.stdout.readline()
    if line:
        line = line.rstrip()
        print(line)
        
        if '✅ HairFastGAN loaded successfully' in line:
            success = True
            break
        elif 'Application startup complete' in line:
            time.sleep(3)
            try:
                r = requests.get('http://localhost:8000/health', timeout=5)
                health = r.json()
                print(f"\nHealth check: {health}")
                if health.get('pipeline') == 'hairfastgan':
                    success = True
            except:
                pass
            break

# ============================================================================
# STEP 5: Get Public URL
# ============================================================================
if success:
    from google.colab.output import eval_js
    try:
        public_url = eval_js('google.colab.kernel.proxyPort(8000)')
        print(f"\n🌐 Public URL: {public_url}")
    except:
        print("\n🌐 Server: http://localhost:8000")
    
    print("\n" + "="*60)
    print("🎉 SUCCESS! HAIR TRANSFER API IS READY!")
    print("="*60)
    print("\nTest with:")
    print("  import requests")
    print(f"  requests.get('{public_url if 'public_url' in locals() else 'YOUR_URL'}/health').json()")
else:
    print("\n⚠️  Server started but model may not be loaded")
    print("Check logs above for errors")

print("\n" + "="*60)
print("SETUP COMPLETE")
print("="*60)
