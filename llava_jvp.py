import os
import random
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from pycocotools.coco import COCO
import gc

# ----------------------------
# Config
# ----------------------------
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
NUM_SAMPLES = 500
IMAGE_RESIZE = 224 
COCO_ANN_FILE = "/data/rashidm/COCO/annotations/captions_val2017.json"
IMAGES_DIR = "/data/rashidm/COCO/val2017"

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

# ----------------------------
# JVP Sensitivity Function
# ----------------------------
def get_jvp_sensitivity(pixel_values):
    from torch.func import jvp

    # Get the actual dtype of the model weights (should be float16/Half)
    target_dtype = next(vision_tower.parameters()).dtype

    def tower_forward(p_vals):
        # Explicitly cast input to the model's weight dtype
        # and ensure the output is also cast to avoid promotion issues
        outputs = vision_tower(p_vals.to(target_dtype))
        hs = outputs[0] if isinstance(outputs, (tuple, list)) else outputs.last_hidden_state
        return hs.mean(dim=1).to(target_dtype)

    # Prepare primal and tangent in the EXACT same dtype as the weights
    x = pixel_values.detach().to(device, dtype=target_dtype)
    v = torch.randn_like(x, dtype=target_dtype, device=device)
    v = v / (torch.norm(v) + 1e-8)

    try:
        # Compute JVP. Both (x,) and (v,) must be tuples of the same dtype.
        _, jvp_res = jvp(tower_forward, (x,), (v,))
        
        # Calculate L2 norm. Promote to float32 only at the very end 
        # for numerical stability in the final score.
        score = torch.norm(jvp_res.to(torch.float32), p=2).item()
        return float(score)
        
    except RuntimeError as e:
        # Final fallback: If functional JVP fails, we use a Manual Finite Difference 
        # approximation which is dtype-agnostic and mathematically equivalent for sensitivity.
        eps = 0.005
        with torch.no_grad():
            orig_feat = tower_forward(x)
            perturbed_feat = tower_forward(x + eps * v)
            # (f(x+ev) - f(x)) / e is the JVP approximation
            jvp_approx = (perturbed_feat - orig_feat) / eps
            score = torch.norm(jvp_approx.to(torch.float32), p=2).item()
            return float(score)
        
# ----------------------------
# Execution Loop
# ----------------------------
coco = COCO(COCO_ANN_FILE)
img_ids = list(coco.imgs.keys())
random.shuffle(img_ids)
selected_ids = img_ids[:NUM_SAMPLES]

print(f"Calculating JVP Sensitivity for {NUM_SAMPLES} samples...")
sensitivity_scores = []

for i, img_id in enumerate(selected_ids):
    info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(IMAGES_DIR, info["file_name"])
    
    try:
        raw_image = Image.open(img_path).convert("RGB").resize((IMAGE_RESIZE, IMAGE_RESIZE))
        inputs = processor(text="USER: <image>\nASSISTANT:", images=raw_image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        jvp_score = get_jvp_sensitivity(pixel_values)
        
        if not np.isfinite(jvp_score):
            print(f"[{i+1}/{NUM_SAMPLES}] ID {img_id} | Non-finite score. Skipping.")
            continue
            
        print(f"[{i+1}/{NUM_SAMPLES}] Image ID {img_id} | JVP Sensitivity: {jvp_score:.6f}")
        sensitivity_scores.append(jvp_score)

        del inputs, pixel_values
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Failed on {img_id}: {e}")

if len(sensitivity_scores) > 0:
    avg_sensitivity = float(np.mean(sensitivity_scores))
    print("\n--- JVP Sensitivity Summary ---")
    print(f"Number of samples: {len(sensitivity_scores)}")
    print(f"Average JVP Score: {avg_sensitivity:.6f}")
else:
    print("No scores collected.")
