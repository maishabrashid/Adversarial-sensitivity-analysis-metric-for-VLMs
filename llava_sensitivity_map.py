import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from pycocotools.coco import COCO
import gc

# ----------------------------
# Config
# ----------------------------
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
NUM_SAMPLES = 10
IMAGE_RESIZE = 224 
COCO_ANN_FILE = "/data/rashidm/COCO/annotations/captions_val2017.json"
IMAGES_DIR = "/data/rashidm/COCO/val2017"
OUTPUT_DIR = "./sensitivity_maps_green"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define custom Colormap: White (low) to Green (high)
white_green_cmap = LinearSegmentedColormap.from_list("white_green", ["white", "green"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Load and Isolate
# ----------------------------
print(f"Loading model: {MODEL_ID}...")
model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, low_cpu_mem_usage=True, attn_implementation="sdpa"
)

vision_tower = model.model.vision_tower.to(device).eval()
model.language_model = model.language_model.to("cpu")
processor = AutoProcessor.from_pretrained(MODEL_ID)

for p in vision_tower.parameters():
    p.requires_grad = False

def get_sensitivity_data(pixel_values):
    x = pixel_values.detach().to(device, dtype=torch.float16).requires_grad_(True)
    outputs = vision_tower(x)
    hs = outputs[0] if isinstance(outputs, (tuple, list)) else outputs.last_hidden_state
    feat = hs.mean(dim=1) 

    u = torch.ones_like(feat) 
    grads = torch.autograd.grad(feat, x, grad_outputs=u)[0]
    
    pixel_contribution = grads[0].pow(2).sum(dim=0).detach().cpu().numpy()
    trace_est = pixel_contribution.mean()
    frob_norm = np.sqrt(max(trace_est, 0.0))
    
    return frob_norm, pixel_contribution

# ----------------------------
# Execution Loop
# ----------------------------
coco = COCO(COCO_ANN_FILE)
img_ids = list(coco.imgs.keys())
random.shuffle(img_ids)
selected_ids = img_ids[:NUM_SAMPLES]

print(f"Generating White-Green sensitivity maps for {NUM_SAMPLES} samples...")

for i, img_id in enumerate(selected_ids):
    info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(IMAGES_DIR, info["file_name"])
    
    try:
        raw_image = Image.open(img_path).convert("RGB").resize((IMAGE_RESIZE, IMAGE_RESIZE))
        inputs = processor(text="USER: <image>\nASSISTANT:", images=raw_image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        frob_score, contribution_map = get_sensitivity_data(pixel_values)
        
        # Log-scale normalization
        viz_map = np.log1p(contribution_map)
        viz_map = (viz_map - viz_map.min()) / (viz_map.max() - viz_map.min() + 1e-8)

        # Plotting Side-by-Side
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # 1. Original Image
        axes[0].imshow(raw_image)
        axes[0].set_title(f"Original Image (ID: {img_id})")
        axes[0].axis("off")
        
        # 2. Raw Sensitivity Map (Custom White-Green)
        # Using the custom cmap defined above
        im = axes[1].imshow(viz_map, cmap=white_green_cmap)
        axes[1].set_title(f"Sensitivity Map\nFrobenius Norm: {frob_score:.4f}")
        axes[1].axis("off")
        
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label='Sensitivity (Green = High)')
        
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/green_map_{img_id}.png")
        plt.close()
        
        print(f"[{i+1}/{NUM_SAMPLES}] ID {img_id} | Saved.")

        del inputs, pixel_values
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Failed on {img_id}: {e}")

print(f"\nCompleted! Check the '{OUTPUT_DIR}' folder.")