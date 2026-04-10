# Model Weight Download Scripts

These scripts download the required pretrained model weights from HuggingFace for HairFastGAN.

## Required Models

The following pretrained models are needed:

1. **PostProcess/latent_avg.pt** - Latent space average for StyleGAN
2. **sean_checkpoints/CelebA-HQ_pretrained/latest_net_G.pth** - SEAN Generator
3. **sean_checkpoints/CelebA-HQ_pretrained/latest_net_D.pth** - SEAN Discriminator

## Usage

### For Local Development

```bash
# Install dependencies first
pip install requests tqdm

# Run the download script
python hair_transfer/scripts/download_weights.py
```

### For Google Colab

In a Colab cell, run:

```python
# Install dependencies
!pip install requests tqdm

# Download weights
!python hair_transfer/scripts/colab_download_weights.py
```

Or copy-paste the content of `colab_download_weights.py` directly into a Colab cell.

### Alternative: Direct Cell Execution in Colab

You can also run this directly in a Colab cell:

```python
import requests, os
from tqdm import tqdm

HF = "https://huggingface.co/AIRI-Institute/HairFastGAN/resolve/main/pretrained_models"
base = '/content/HairFastGAN-main/pretrained_models'

files = {
    f"{base}/PostProcess/latent_avg.pt": f"{HF}/PostProcess/latent_avg.pt",
    f"{base}/sean_checkpoints/CelebA-HQ_pretrained/latest_net_G.pth": f"{HF}/sean_checkpoints/CelebA-HQ_pretrained/latest_net_G.pth",
    f"{base}/sean_checkpoints/CelebA-HQ_pretrained/latest_net_D.pth": f"{HF}/sean_checkpoints/CelebA-HQ_pretrained/latest_net_D.pth",
}

for dest, url in files.items():
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) > 1024*100:
        print(f"✓ {os.path.basename(dest)}"); continue
    print(f"↓ {os.path.basename(dest)}...")
    r = requests.get(url, stream=True)
    total = int(r.headers.get('content-length', 0))
    with open(dest, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True, desc=os.path.basename(dest)) as bar:
        for chunk in r.iter_content(8192):
            f.write(chunk); bar.update(len(chunk))

print("✓ All models downloaded!")
```

## Troubleshooting

### Connection Errors
If you get connection errors, try:
- Checking your internet connection
- Retrying the download (the script skips already downloaded files)
- Downloading manually from: https://huggingface.co/AIRI-Institute/HairFastGAN/tree/main/pretrained_models

### File Size Issues
The models are large files:
- `latest_net_G.pth`: ~200-300 MB
- `latest_net_D.pth`: ~100-200 MB
- `latent_avg.pt`: ~1-10 MB

Ensure you have sufficient disk space and a stable connection.

### Permission Errors
If you get permission errors on local systems:
- Ensure the target directory is writable
- Try running with appropriate permissions
- Check that the path exists or can be created

## Model Sources

All models are from the official HairFastGAN repository:
- HuggingFace: https://huggingface.co/AIRI-Institute/HairFastGAN
- GitHub: https://github.com/AIRI-Institute/HairFastGAN
