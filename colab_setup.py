"""
Complete Colab Setup Script for HairFastGAN Hair Transfer
Run this in Google Colab to set up the entire environment.
"""

# ============================================================================
# STEP 1: Install Dependencies
# ============================================================================
print("=" * 60)
print("STEP 1: Installing Dependencies")
print("=" * 60)

import os
os.chdir('/content')

# Install PyTorch with CUDA support
print("\n📦 Installing PyTorch...")
os.system('pip install torch==2.2.0+cu121 torchvision==0.17.0+cu121 --index-url https://download.pytorch.org/whl/cu121 -q')

# Install other dependencies
print("📦 Installing other dependencies...")
os.system('pip install ninja scipy face-alignment dlib-bin addict -q')
os.system('pip install git+https://github.com/openai/CLIP.git -q')
os.system('pip install fastapi uvicorn python-multipart pyngrok requests tqdm -q')

print("✓ Dependencies installed!")
print("\n⚠️  IMPORTANT: Restart runtime now (Runtime → Restart runtime)")
print("Then run STEP 2 after restart.\n")


# ============================================================================
# STEP 2: Clone Repositories (Run after runtime restart)
# ============================================================================
def step2_clone_repos():
    print("=" * 60)
    print("STEP 2: Cloning Repositories")
    print("=" * 60)
    
    import os
    os.chdir('/content')
    
    # Clone HairFastGAN
    print("\n📥 Cloning HairFastGAN...")
    result = os.system('git clone --depth=1 https://github.com/AIRI-Institute/HairFastGAN /content/HairFastGAN-main 2>/dev/null')
    if result != 0:
        print("  (Repository already exists)")
    
    # Clone your salon project
    print("📥 Cloning AI-In-Salon project...")
    result = os.system('git clone --depth=1 https://github.com/LakshayR-ai/AI-In-Salon-.git /content/salon 2>/dev/null')
    if result != 0:
        print("  (Repository already exists)")
    
    print("\n✓ Repositories cloned!")
    print("Next: Run step3_download_weights()")


# ============================================================================
# STEP 3: Download Model Weights from HuggingFace
# ============================================================================
def step3_download_weights():
    print("=" * 60)
    print("STEP 3: Downloading Model Weights")
    print("=" * 60)
    
    import requests
    import os
    from tqdm import tqdm
    
    HF = "https://huggingface.co/AIRI-Institute/HairFastGAN/resolve/main/pretrained_models"
    base = '/content/HairFastGAN-main/pretrained_models'
    
    files = {
        # PostProcess
        f"{base}/PostProcess/latent_avg.pt": 
            f"{HF}/PostProcess/latent_avg.pt",
        
        # SEAN checkpoints
        f"{base}/sean_checkpoints/CelebA-HQ_pretrained/latest_net_G.pth": 
            f"{HF}/sean_checkpoints/CelebA-HQ_pretrained/latest_net_G.pth",
        f"{base}/sean_checkpoints/CelebA-HQ_pretrained/latest_net_D.pth": 
            f"{HF}/sean_checkpoints/CelebA-HQ_pretrained/latest_net_D.pth",
        
        # BiSeNet face parsing
        f"{base}/BiSeNet/face_parsing_79999_iter.pth":
            f"{HF}/BiSeNet/face_parsing_79999_iter.pth",
    }
    
    print(f"\n📥 Downloading {len(files)} model files...\n")
    
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
            return False
    
    print("✓ All weights downloaded!")
    print("Next: Run step4_test_model()")
    return True


# ============================================================================
# STEP 4: Test Model Loading
# ============================================================================
def step4_test_model():
    print("=" * 60)
    print("STEP 4: Testing Model Load")
    print("=" * 60)
    
    import sys
    import os
    
    sys.path.insert(0, '/content/HairFastGAN-main')
    os.chdir('/content/HairFastGAN-main')
    
    print("\n🔄 Loading HairFastGAN model...")
    
    try:
        from hair_swap import HairFast, get_parser
        
        args = get_parser().parse_args([])
        args.device = 'cuda'
        
        model = HairFast(args)
        
        print("✓ Model loaded successfully!")
        print("Next: Run step5_setup_server()")
        return model
    
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# STEP 5: Setup and Start Server
# ============================================================================
def step5_setup_server():
    print("=" * 60)
    print("STEP 5: Setting Up Server")
    print("=" * 60)
    
    import sys
    import os
    import shutil
    import threading
    import time
    import requests
    from google.colab.output import eval_js
    
    # Kill any existing server on port 8000
    os.system("fuser -k 8000/tcp 2>/dev/null")
    
    # Add paths
    sys.path.insert(0, '/content/HairFastGAN-main')
    sys.path.insert(0, '/content/salon/hair_transfer')
    
    # Copy face parsing model
    print("\n📋 Copying face parsing model...")
    os.makedirs('/content/salon/hair_transfer/pretrained', exist_ok=True)
    shutil.copy(
        '/content/HairFastGAN-main/pretrained_models/BiSeNet/face_parsing_79999_iter.pth',
        '/content/salon/hair_transfer/pretrained/face_parsing.pth'
    )
    
    # Create config file
    print("📋 Creating config file...")
    os.makedirs('/content/salon/hair_transfer/configs', exist_ok=True)
    with open('/content/salon/hair_transfer/configs/base.yaml', 'w') as f:
        f.write("device: cuda\nstylegan:\n  size: 256\n")
    
    # Start server in background
    def run_server():
        os.chdir('/content/salon/hair_transfer')
        os.system("python -m uvicorn app.backend:app --host 0.0.0.0 --port 8000")
    
    print("\n🚀 Starting server...")
    threading.Thread(target=run_server, daemon=True).start()
    
    # Wait for server to be ready
    print("⏳ Waiting for server to start...")
    for i in range(40):
        time.sleep(5)
        try:
            r = requests.get('http://localhost:8000/health', timeout=2)
            print(f"\n✓ Server ready: {r.json()}")
            break
        except:
            print(f"  {(i+1)*5}s...", end='\r')
    
    # Get public URL
    public_url = eval_js("google.colab.kernel.proxyPort(8000)")
    print(f"\n🌐 Public URL: {public_url}")
    print("\n✓ Setup complete! Server is running.")
    
    return public_url


# ============================================================================
# QUICK START: Run All Steps (after runtime restart)
# ============================================================================
def quick_start():
    """Run all setup steps in sequence (after runtime restart)."""
    print("\n" + "=" * 60)
    print("QUICK START: Running Full Setup")
    print("=" * 60 + "\n")
    
    # Step 2: Clone repos
    step2_clone_repos()
    
    # Step 3: Download weights
    if not step3_download_weights():
        print("\n✗ Failed to download weights. Please check your connection.")
        return
    
    # Step 4: Test model
    model = step4_test_model()
    if model is None:
        print("\n✗ Failed to load model. Please check the logs.")
        return
    
    # Step 5: Start server
    url = step5_setup_server()
    
    print("\n" + "=" * 60)
    print("✓ ALL DONE! Your hair transfer API is ready.")
    print("=" * 60)
    print(f"\nAccess your API at: {url}")


# ============================================================================
# USAGE INSTRUCTIONS
# ============================================================================
if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║         HairFastGAN Hair Transfer - Colab Setup              ║
╚══════════════════════════════════════════════════════════════╝

FIRST TIME SETUP:
-----------------
1. Run the code at the top (STEP 1) to install dependencies
2. Restart runtime (Runtime → Restart runtime)
3. After restart, run: quick_start()

ALTERNATIVE (Step by Step):
---------------------------
After runtime restart:
1. step2_clone_repos()
2. step3_download_weights()
3. step4_test_model()
4. step5_setup_server()

SUBSEQUENT RUNS:
----------------
If repos and weights are already downloaded:
- Just run: quick_start()

Or start from step 4:
- step4_test_model()
- step5_setup_server()
    """)
