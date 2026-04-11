"""
Professional Gradio UI for HairFastGAN Hair Transfer
Features: Live camera, sample hairstyles, organized categories
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

# Sample hairstyles organized by category
SAMPLE_HAIRSTYLES = {
    "Women - Long Hair": [
        "https://images.unsplash.com/photo-1594744803329-e58b31de8bf5?w=400",  # Long blonde
        "https://images.unsplash.com/photo-1580489944761-15a19d654956?w=400",  # Long brunette
        "https://images.unsplash.com/photo-1529626455594-4ff0802cfb7e?w=400",  # Long wavy
    ],
    "Women - Short Hair": [
        "https://images.unsplash.com/photo-1487412720507-e7ab37603c6f?w=400",  # Short bob
        "https://images.unsplash.com/photo-1524504388940-b1c1722653e1?w=400",  # Pixie cut
        "https://images.unsplash.com/photo-1531746020798-e6953c6e8e04?w=400",  # Short wavy
    ],
    "Women - Colored Hair": [
        "https://images.unsplash.com/photo-1508214751196-bcfd4ca60f91?w=400",  # Red hair
        "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=400",  # Blonde highlights
        "https://images.unsplash.com/photo-1488426862026-3ee34a7d66df?w=400",  # Dark with highlights
    ],
    "Men - Short Hair": [
        "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=400",  # Classic short
        "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400",  # Fade cut
        "https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?w=400",  # Textured short
    ],
    "Men - Medium Hair": [
        "https://images.unsplash.com/photo-1492562080023-ab3db95bfbce?w=400",  # Medium length
        "https://images.unsplash.com/photo-1519085360753-af0119f7cbe7?w=400",  # Wavy medium
        "https://images.unsplash.com/photo-1504257432389-52343af06ae3?w=400",  # Styled medium
    ],
    "Men - Long Hair": [
        "https://images.unsplash.com/photo-1552374196-1ab2a1c593e8?w=400",  # Long flowing
        "https://images.unsplash.com/photo-1531891437562-4301cf35b7e4?w=400",  # Long tied back
        "https://images.unsplash.com/photo-1506277886164-e25aa3f4ef7f?w=400",  # Long wavy
    ],
}

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
    """Create professional Gradio UI with categories and live camera"""
    
    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        font-family: 'Inter', sans-serif;
    }
    .category-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
    }
    .sample-gallery {
        border: 2px solid #e5e7eb;
        border-radius: 10px;
        padding: 10px;
    }
    """
    
    with gr.Blocks(
        title="AI Hair Salon - Professional Hair Transfer",
        theme=gr.themes.Soft(
            primary_hue="purple",
            secondary_hue="pink",
        ),
        css=custom_css
    ) as demo:
        
        gr.HTML("""
            <div class="category-header">
                <h1>💇‍♀️ AI Hair Salon - Professional Hair Transfer 💇‍♂️</h1>
                <p>Transform your look with AI-powered hairstyle transfer</p>
            </div>
        """)
        
        with gr.Tabs() as tabs:
            
            # Tab 1: Quick Transfer
            with gr.Tab("✨ Quick Transfer"):
                gr.Markdown("""
                ### Upload or capture your photo, then choose a hairstyle
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 📸 Your Photo")
                        face_input = gr.Image(
                            label="Face Photo",
                            type="pil",
                            sources=["upload", "webcam"],
                            height=400
                        )
                        
                    with gr.Column(scale=1):
                        gr.Markdown("#### 💇 Hairstyle Reference")
                        hair_input = gr.Image(
                            label="Hairstyle Reference",
                            type="pil",
                            sources=["upload", "webcam"],
                            height=400
                        )
                
                transfer_btn = gr.Button(
                    "✨ Transfer Hair",
                    variant="primary",
                    size="lg",
                    scale=2
                )
                
                gr.Markdown("#### 🎨 Result")
                result_output = gr.Image(
                    label="Your New Look",
                    type="pil",
                    height=500
                )
            
            # Tab 2: Browse Hairstyles
            with gr.Tab("🎨 Browse Hairstyles"):
                gr.Markdown("""
                ### Choose from our curated hairstyle collection
                Click on any hairstyle to use it as reference
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 📸 Your Photo")
                        face_input_gallery = gr.Image(
                            label="Face Photo",
                            type="pil",
                            sources=["upload", "webcam"],
                            height=400
                        )
                
                # Women's Hairstyles
                gr.Markdown("### 👩 Women's Hairstyles")
                
                with gr.Accordion("Long Hair", open=True):
                    gallery_women_long = gr.Gallery(
                        value=SAMPLE_HAIRSTYLES["Women - Long Hair"],
                        label="Long Hairstyles",
                        columns=3,
                        height=200,
                        object_fit="cover"
                    )
                
                with gr.Accordion("Short Hair"):
                    gallery_women_short = gr.Gallery(
                        value=SAMPLE_HAIRSTYLES["Women - Short Hair"],
                        label="Short Hairstyles",
                        columns=3,
                        height=200,
                        object_fit="cover"
                    )
                
                with gr.Accordion("Colored Hair"):
                    gallery_women_colored = gr.Gallery(
                        value=SAMPLE_HAIRSTYLES["Women - Colored Hair"],
                        label="Colored Hairstyles",
                        columns=3,
                        height=200,
                        object_fit="cover"
                    )
                
                # Men's Hairstyles
                gr.Markdown("### 👨 Men's Hairstyles")
                
                with gr.Accordion("Short Hair", open=True):
                    gallery_men_short = gr.Gallery(
                        value=SAMPLE_HAIRSTYLES["Men - Short Hair"],
                        label="Short Hairstyles",
                        columns=3,
                        height=200,
                        object_fit="cover"
                    )
                
                with gr.Accordion("Medium Hair"):
                    gallery_men_medium = gr.Gallery(
                        value=SAMPLE_HAIRSTYLES["Men - Medium Hair"],
                        label="Medium Hairstyles",
                        columns=3,
                        height=200,
                        object_fit="cover"
                    )
                
                with gr.Accordion("Long Hair"):
                    gallery_men_long = gr.Gallery(
                        value=SAMPLE_HAIRSTYLES["Men - Long Hair"],
                        label="Long Hairstyles",
                        columns=3,
                        height=200,
                        object_fit="cover"
                    )
                
                selected_style = gr.Image(
                    label="Selected Hairstyle",
                    type="pil",
                    visible=True
                )
                
                transfer_btn_gallery = gr.Button(
                    "✨ Apply This Hairstyle",
                    variant="primary",
                    size="lg"
                )
                
                result_output_gallery = gr.Image(
                    label="Your New Look",
                    type="pil",
                    height=500
                )
            
            # Tab 3: Tips & Info
            with gr.Tab("ℹ️ Tips & Info"):
                gr.Markdown("""
                ## 📖 How to Get Best Results
                
                ### 📸 Photo Guidelines
                - **Use clear, well-lit photos**
                - **Face should be front-facing**
                - **Show full face and hair**
                - **Higher resolution = better results**
                - **Avoid heavy filters or editing**
                
                ### 💡 Tips for Success
                1. **Lighting**: Natural daylight works best
                2. **Background**: Plain backgrounds help
                3. **Expression**: Neutral expression recommended
                4. **Hair**: Make sure hair is visible in reference photo
                5. **Angle**: Both photos should have similar angles
                
                ### 🎨 Hairstyle Selection
                - **Match gender**: Use appropriate category
                - **Consider face shape**: Some styles suit certain face shapes better
                - **Hair texture**: Similar textures transfer better
                - **Color**: You can try different colors!
                
                ### ⚡ Processing Time
                - First transfer: ~10-15 seconds
                - Subsequent transfers: ~5-10 seconds
                - GPU acceleration enabled
                
                ### 🔒 Privacy
                - All processing happens in this session
                - Images are not stored permanently
                - Your photos are private and secure
                
                ---
                
                ## 🛠️ Technical Details
                
                **Powered by:** [HairFastGAN](https://github.com/AIRI-Institute/HairFastGAN) (NeurIPS 2024)
                
                **Model Features:**
                - StyleGAN2-based architecture
                - Cross-attention shape transfer
                - CLIP-guided color transfer
                - Face detail preservation
                
                **Credits:**
                - AIRI Institute for HairFastGAN
                - Gradio for the UI framework
                - Unsplash for sample images
                
                ---
                
                ### 🐛 Troubleshooting
                
                **Issue**: Blurry results
                - **Solution**: Use higher resolution input images
                
                **Issue**: Hair doesn't match reference
                - **Solution**: Try a different reference with clearer hair
                
                **Issue**: Face looks different
                - **Solution**: This is normal - the model preserves identity while transferring hair
                
                **Issue**: Processing takes too long
                - **Solution**: First load takes longer; subsequent ones are faster
                """)
        
        # Connect buttons for Quick Transfer tab
        transfer_btn.click(
            fn=transfer_hair,
            inputs=[face_input, hair_input],
            outputs=result_output
        )
        
        # Connect buttons for Gallery tab
        def select_from_gallery(evt: gr.SelectData):
            """Handle gallery selection - load image from URL"""
            try:
                import requests
                from io import BytesIO
                
                # evt.value is a dict with 'image' key containing 'url'
                if isinstance(evt.value, dict):
                    image_url = evt.value.get('image', {}).get('url') or evt.value.get('image', {}).get('path')
                else:
                    image_url = evt.value
                
                print(f"Loading image from: {image_url}")
                
                # Download and load the image
                response = requests.get(image_url)
                response.raise_for_status()
                img = Image.open(BytesIO(response.content))
                
                return img
            except Exception as e:
                print(f"Error loading image: {e}")
                # Return a placeholder or None
                return None
        
        # Connect gallery selections
        for gallery in [gallery_women_long, gallery_women_short, gallery_women_colored,
                       gallery_men_short, gallery_men_medium, gallery_men_long]:
            gallery.select(
                fn=select_from_gallery,
                outputs=selected_style
            )
        
        transfer_btn_gallery.click(
            fn=transfer_hair,
            inputs=[face_input_gallery, selected_style],
            outputs=result_output_gallery
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
