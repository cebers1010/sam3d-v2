import os
import sys
import subprocess
import uuid
import shutil
import numpy as np
from PIL import Image
import torch
import gradio as gr
import spaces
import rembg
from huggingface_hub import snapshot_download

# --- Constants & Configuration ---
REPO_URL = "https://github.com/facebookresearch/sam-3d-objects.git"
REPO_DIR = "sam-3d-objects"
CHECKPOINT_REPO = "facebook/sam-3d-objects"
CHECKPOINT_DIR = "checkpoints/hf"

# --- 1. Setup Environment Dynamically ---
print("--- Starting SAM3D-Objects Setup ---")

# Clone Repository
if not os.path.exists(REPO_DIR):
    print(f"Cloning {REPO_URL}...")
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)

# Add to sys.path
sys.path.append(os.path.abspath(REPO_DIR))
sys.path.append(os.path.abspath(os.path.join(REPO_DIR, "notebook")))

# Download Checkpoints
# We assume the user has set HF_TOKEN in Space Secrets if the repo is gated (it is).
print("Downloading checkpoints...")
try:
    # Ensure the target directory exists inside the repo structure because the config expects it there
    target_ckpt_path = os.path.join(REPO_DIR, CHECKPOINT_DIR)
    os.makedirs(target_ckpt_path, exist_ok=True)
    
    snapshot_download(
        repo_id=CHECKPOINT_REPO,
        local_dir=target_ckpt_path,
        repo_type="model"
    )
    print("Checkpoints downloaded successfully.")
except Exception as e:
    print(f"Error downloading checkpoints: {e}")
    print("Ensure you have set HF_TOKEN in your Space secrets and accepted the model license.")

# --- 2. Inference Logic ---
# We wrap imports inside a function or try-block to avoid crashing immediately if setup failed
try:
    from inference import Inference
except ImportError as e:
    print(f"Failed to import Inference: {e}")
    # Retrying with explicit path
    sys.path.append(os.path.join(os.getcwd(), REPO_DIR))
    sys.path.append(os.path.join(os.getcwd(), REPO_DIR, "notebook"))
    try:
        from inference import Inference
    except ImportError:
        print("CRITICAL: Still cannot import Inference.")
        Inference = None

# Global Model Loader
model = None

def load_model():
    global model
    if model is None:
        if Inference is None:
            raise ImportError("Could not import SAM3D Inference module.")
        
        print("Loading Model...")
        # Config path relative to the repo root where we are running? 
        # The inference class expects 'config_file'.
        # The README says: config_path = f"checkpoints/{tag}/pipeline.yaml"
        # Since we added REPO_DIR to sys.path, imports work, but file paths might need to be absolute or relative to CWD.
        # But 'Inference' instantiation uses: config.workspace_dir = os.path.dirname(config_file)
        # So providing the absolute path to the config file is best.
        
        config_path = os.path.abspath(os.path.join(REPO_DIR, CHECKPOINT_DIR, "pipeline.yaml"))
        print(f"Using config: {config_path}")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
            
        model = Inference(config_path, compile=False)
        print("Model loaded.")
    return model

# --- 3. Processing Function ---
@spaces.GPU
def process(input_image):
    if input_image is None:
        return None, None
    
    try:
        current_model = load_model()
        
        # unique run id
        run_id = str(uuid.uuid4())
        
        # Prepare Image and Mask
        # SAM3D expects uint8 numpy array for image, boolean/uint8 for mask.
        if input_image.mode != "RGB":
            input_image_rgb = input_image.convert("RGB")
        else:
            input_image_rgb = input_image
            
        image_np = np.array(input_image_rgb).astype(np.uint8)
        
        # Remove Background to get mask
        print("removing background...")
        # rembg returns an RGBA image
        rembg_output = rembg.remove(input_image_rgb) 
        mask_pil = rembg_output.split()[-1] # Alpha channel
        mask_np = np.array(mask_pil) > 0 # Boolean mask
        
        print("Running Inference...")
        # Inference call
        # output = inference(image, mask, seed=42)
        output = current_model(image_np, mask_np, seed=42)
        
        # Save output
        output_ply_name = f"output_{run_id}.ply"
        output_ply_path = os.path.abspath(output_ply_name)
        
        output["gs"].save_ply(output_ply_path)
        print(f"Saved to {output_ply_path}")
        
        return output_ply_path, output_ply_path
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Error processing image: {str(e)}")

# --- 4. Gradio Interface ---
css = """
#col-container {
    margin: 0 auto;
    max-width: 960px;
}
"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown("# SAM 3D Objects Generator")
        gr.Markdown("Upload an image to generate a 3D PLY model using [Meta's SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects).")
        
        with gr.Row():
            with gr.Column():
                input_img = gr.Image(type="pil", label="Input Image")
                btn = gr.Button("Generate 3D Model", variant="primary")
            
            with gr.Column():
                output_3d = gr.Model3D(label="3D Preview", clear_color=(1.0, 1.0, 1.0, 1.0))
                output_file = gr.File(label="Download .ply")
                
        btn.click(process, inputs=input_img, outputs=[output_3d, output_file])

if __name__ == "__main__":
    demo.launch()
