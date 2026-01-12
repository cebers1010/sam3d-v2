import os
import sys
import subprocess
import uuid
import base64
import io
import shutil
import numpy as np
from PIL import Image
import torch
import runpod
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
print("Downloading checkpoints...")
try:
    # Ensure the target directory exists inside the repo structure because the config expects it there
    target_ckpt_path = os.path.join(REPO_DIR, CHECKPOINT_DIR)
    os.makedirs(target_ckpt_path, exist_ok=True)
    
    # RunPod usually passes env vars. Ensure HF_TOKEN is in the RunPod template env.
    snapshot_download(
        repo_id=CHECKPOINT_REPO,
        local_dir=target_ckpt_path,
        repo_type="model"
    )
    print("Checkpoints downloaded successfully.")
except Exception as e:
    print(f"Error downloading checkpoints: {e}")
    print("Ensure you have set HF_TOKEN in your RunPod Template Env Vars.")

# --- 2. Inference Logic ---
# We wrap imports inside a function or try-block to avoid crashing immediately if setup failed
try:
    from inference import Inference
except ImportError as e:
    print(f"Failed to import Inference: {e}")
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
        config_path = os.path.abspath(os.path.join(REPO_DIR, CHECKPOINT_DIR, "pipeline.yaml"))
        print(f"Using config: {config_path}")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
            
        model = Inference(config_path, compile=False)
        print("Model loaded.")
    return model

# --- 3. RunPod Handler ---

def handler(job):
    """ Handler function that will be used to process jobs. """
    job_input = job['input']
    
    # Input validation
    if 'image' not in job_input:
        return {"error": "No image provided in input. Expecting json: {'image': 'base64str'}"}
    
    try:
        current_model = load_model()
        
        # Decode Image
        image_data = base64.b64decode(job_input['image'])
        input_image = Image.open(io.BytesIO(image_data))
        
        # unique run id
        run_id = str(uuid.uuid4())
        
        # Prepare Image and Mask
        if input_image.mode != "RGB":
            input_image_rgb = input_image.convert("RGB")
        else:
            input_image_rgb = input_image
            
        image_np = np.array(input_image_rgb).astype(np.uint8)
        
        # Remove Background to get mask
        print("removing background...")
        rembg_output = rembg.remove(input_image_rgb) 
        mask_pil = rembg_output.split()[-1] # Alpha channel
        mask_np = np.array(mask_pil) > 0 # Boolean mask
        
        print("Running Inference...")
        output = current_model(image_np, mask_np, seed=42)
        
        # Save output temporarily
        output_ply_name = f"output_{run_id}.ply"
        output_ply_path = os.path.abspath(output_ply_name)
        
        output["gs"].save_ply(output_ply_path)
        print(f"Saved to {output_ply_path}")
        
        # Read and Encode Output
        with open(output_ply_path, "rb") as f:
            ply_content = f.read()
            ply_base64 = base64.b64encode(ply_content).decode('utf-8')
            
        # Cleanup
        if os.path.exists(output_ply_path):
            os.remove(output_ply_path)
            
        return {
            "ply": ply_base64,
            "filename": output_ply_name
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

if __name__ == '__main__':
    runpod.serverless.start({"handler": handler})
