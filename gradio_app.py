"""
Simple Gradio UI for HairFastGAN Hair Transfer
Professional interface with minimal dependencies
"""

import gradio as gr
import torch
import sys
import os
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as TF

# Setup paths
HAIRFASTGAN_PATH = os.getenv('HAIRFASTGAN_PATH', '/content/HairFastGAN-main')
sys.path.insert(0, HAIRFASTGAN_PATH)
os.chdir(HAIRFASTGAN_PATH)

# Import HairFastGAN
from hair_swap import HairFast, get_parser

# Global model
model = None

def load_model():
    """Load HairFastGAN model once"""
    global model
    if model is None:
        print("Loading HairFastGAN model...")
        args = get_parser().parse_args([])
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = HairFast(args)
        print(f"✅ Model loaded on {args.device}")
    return model

def transfer_hair(face_image, hair_image):
    """
    Transfer hair from hair_image to face_image
    
    Args:
        face_image: PIL Image of the face
        hair_image: PIL Image with desired hairstyle
    
    Returns:
        PIL Image with transferred hair
    """
    try:
        # Load model
        hair_model = load_model()
        
        # Convert to tensors
        face_tensor = TF.to_tensor(face_image)
        hair_tensor = TF.to_tensor(hair_image)
        
        # Perform hair transfer
        with torch.no_grad():
            result = hair_model.swap(
                face_tensor, 
                hair_tensor, 
                hair_tensor,
                align=True,
                benchmark=False
            )
        
        # Handle tuple return (aligned images included)
        if isinstance(result, tuple):
            result = result[0]
        
        # Convert back to PIL
        result_pil = TF.to_pil_image(result.clamp(0, 1).cpu())
        
        return result_pil
    
    except Exception as e:
        print(f"Error during hair transfer: {e}")
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Hair transfer failed: {str(e)}")

# Create Gradio interface
def create_ui():
    """Create professional Gradio UI"""
    
    with gr.Blocks(
        title="AI Hair Salon - Hair Transfer",
        theme=gr.themes.Soft()
    ) as demo:
        
        gr.Markdown(
            """
            # 💇 AI Hair Salon - Hair Transfer
            
            Upload a face photo and a hairstyle reference to transfer the hair onto your face.
            
            **How to use:**
            1. Upload your face photo
            2. Upload a reference photo with the desired hairstyle
            3. Click "Transfer Hair"
            4. Download your result!
            """
        )
        
        with gr.Row():
            with gr.Column():
                face_input = gr.Image(
                    label="Your Face Photo",
                    type="pil",
                    sources=["upload", "webcam"]
                )
                
            with gr.Column():
                hair_input = gr.Image(
                    label="Hairstyle Reference",
                    type="pil",
                    sources=["upload"]
                )
        
        transfer_btn = gr.Button(
            "✨ Transfer Hair",
            variant="primary",
            size="lg"
        )
        
        with gr.Row():
            result_output = gr.Image(
                label="Result",
                type="pil"
            )
        
        # Examples
        gr.Markdown("### 📸 Example Images")
        gr.Markdown("Try with your own photos or use sample images")
        
        # Connect button
        transfer_btn.click(
            fn=transfer_hair,
            inputs=[face_input, hair_input],
            outputs=result_output
        )
        
        gr.Markdown(
            """
            ---
            **Tips for best results:**
            - Use clear, front-facing photos
            - Ensure good lighting
            - Photos should show the full face and hair
            - Higher resolution images work better
            
            **Powered by:** [HairFastGAN](https://github.com/AIRI-Institute/HairFastGAN) (NeurIPS 2024)
            """
        )
    
    return demo

if __name__ == "__main__":
    # Pre-load model
    print("Initializing AI Hair Salon...")
    load_model()
    
    # Launch UI
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,  # Creates public URL
        show_error=True
    )
