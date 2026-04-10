"""
Download pretrained model weights from HuggingFace for HairFastGAN.
"""
import os
import requests
from tqdm import tqdm


def download_file(url, dest):
    """Download a file with progress bar."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    
    # Skip if file already exists and is reasonably sized
    if os.path.exists(dest) and os.path.getsize(dest) > 1024 * 100:
        print(f"✓ {os.path.basename(dest)} already exists")
        return
    
    print(f"↓ Downloading {os.path.basename(dest)}...")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    
    total = int(r.headers.get('content-length', 0))
    
    with open(dest, 'wb') as f, tqdm(
        total=total,
        unit='B',
        unit_scale=True,
        desc=os.path.basename(dest)
    ) as bar:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))
    
    print(f"✓ Downloaded {os.path.basename(dest)}")


def main():
    # Base URLs and paths
    HF_BASE = "https://huggingface.co/AIRI-Institute/HairFastGAN/resolve/main/pretrained_models"
    
    # Determine base path (works for both local and Colab)
    if os.path.exists('/content'):
        BASE_PATH = '/content/HairFastGAN-main/pretrained_models'
    else:
        # Local path - adjust to your project structure
        BASE_PATH = os.path.join(os.path.dirname(__file__), '../../HairFastGAN-main/pretrained_models')
        BASE_PATH = os.path.abspath(BASE_PATH)
    
    # Define all required model files
    files_to_download = {
        # PostProcess models
        f"{BASE_PATH}/PostProcess/latent_avg.pt": 
            f"{HF_BASE}/PostProcess/latent_avg.pt",
        
        # SEAN checkpoints
        f"{BASE_PATH}/sean_checkpoints/CelebA-HQ_pretrained/latest_net_G.pth": 
            f"{HF_BASE}/sean_checkpoints/CelebA-HQ_pretrained/latest_net_G.pth",
        f"{BASE_PATH}/sean_checkpoints/CelebA-HQ_pretrained/latest_net_D.pth": 
            f"{HF_BASE}/sean_checkpoints/CelebA-HQ_pretrained/latest_net_D.pth",
    }
    
    print(f"Downloading models to: {BASE_PATH}\n")
    
    # Download all files
    for dest, url in files_to_download.items():
        try:
            download_file(url, dest)
        except Exception as e:
            print(f"✗ Error downloading {os.path.basename(dest)}: {e}")
            continue
    
    print("\n✓ All downloads complete!")


if __name__ == "__main__":
    main()
