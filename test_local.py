"""
Test HairFastGAN locally with your downloaded weights
"""

import sys
import os
from pathlib import Path

# Setup paths
HAIRFASTGAN_PATH = Path(__file__).parent / "HairFastGAN-main"
sys.path.insert(0, str(HAIRFASTGAN_PATH))
os.chdir(HAIRFASTGAN_PATH)

print("="*60)
print("Testing HairFastGAN with Local Weights")
print("="*60 + "\n")

# Check if weights exist
print("📋 Checking pretrained models...")
weights_path = HAIRFASTGAN_PATH / "pretrained_models"

if not weights_path.exists():
    print("❌ pretrained_models folder not found!")
    exit(1)

# Count model files
model_files = list(weights_path.rglob("*.pt")) + list(weights_path.rglob("*.pth"))
total_size = sum(f.stat().st_size for f in model_files) / (1024**3)

print(f"✅ Found {len(model_files)} model files")
print(f"✅ Total size: {total_size:.2f} GB\n")

# Import and load model
print("🔄 Loading HairFastGAN model...")

try:
    from hair_swap import HairFast, get_parser
    import torch
    
    args = get_parser().parse_args([])
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Device: {args.device}")
    print("Loading model (this takes 30-60 seconds)...\n")
    
    model = HairFast(args)
    
    print("\n" + "="*60)
    print("✅ SUCCESS! HairFastGAN loaded successfully!")
    print("="*60)
    print("\nYou can now:")
    print("1. Run the Gradio UI: python gradio_app.py")
    print("2. Use the model programmatically")
    
except Exception as e:
    print(f"\n❌ Error loading model: {e}")
    import traceback
    traceback.print_exc()
