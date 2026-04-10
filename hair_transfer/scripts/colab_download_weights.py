"""
Colab-optimized script to download HairFastGAN pretrained models.
Run this in a Colab cell before running inference.
"""
import requests
import os
from tqdm import tqdm

# Configuration
HF = "https://huggingface.co/AIRI-Institute/HairFastGAN/resolve/main/pretrained_models"
base = '/content/HairFastGAN-main/pretrained_models'

# All required model files
files = {
    # PostProcess
    f"{base}/PostProcess/latent_avg.pt": 
        f"{HF}/PostProcess/latent_avg.pt",
    
    # SEAN Generator and Discriminator
    f"{base}/sean_checkpoints/CelebA-HQ_pretrained/latest_net_G.pth": 
        f"{HF}/sean_checkpoints/CelebA-HQ_pretrained/latest_net_G.pth",
    f"{base}/sean_checkpoints/CelebA-HQ_pretrained/latest_net_D.pth": 
        f"{HF}/sean_checkpoints/CelebA-HQ_pretrained/latest_net_D.pth",
}

print("Starting model download...\n")

for dest, url in files.items():
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    
    # Skip if already downloaded
    if os.path.exists(dest) and os.path.getsize(dest) > 1024 * 100:
        print(f"✓ {os.path.basename(dest)} already exists")
        continue
    
    print(f"↓ Downloading {os.path.basename(dest)}...")
    
    try:
        r = requests.get(url, stream=True)
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        
        with open(dest, 'wb') as f, tqdm(
            total=total,
            unit='B',
            unit_scale=True,
            desc=os.path.basename(dest)
        ) as bar:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        
        print(f"✓ Downloaded {os.path.basename(dest)}\n")
    
    except Exception as e:
        print(f"✗ Error: {e}\n")

print("✓ Download complete! You can now run the model.")
