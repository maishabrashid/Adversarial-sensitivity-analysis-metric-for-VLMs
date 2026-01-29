import os
import random
import torch
import clip
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
import gc

# ----------------------------
# Config
# ----------------------------
DATA_PATH = '/data/rashidm/caltech/archive/256_ObjectCategories/'
NUM_SAMPLES_PCT = 0.1  # 5% of the dataset
IMAGE_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# Load Model & Dataset
# ----------------------------
print(f"Loading CLIP model (ViT-B/32)...")
# Load in float16 to match typical research settings, or stay in default
model, preprocess = clip.load("ViT-B/32", device=DEVICE)
model.eval()

# Identify the model precision to prevent "mat1 and mat2" errors
target_dtype = next(model.parameters()).dtype

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Lambda(lambda img: img.convert('RGB')), 
    transforms.ToTensor(),
])

full_dataset = datasets.ImageFolder(root=DATA_PATH, transform=transform)
total_size = len(full_dataset)
subset_size = int(NUM_SAMPLES_PCT * total_size)
indices = random.sample(range(total_size), subset_size)
subset = Subset(full_dataset, indices)
dataloader = DataLoader(subset, batch_size=1, shuffle=True)

# ----------------------------
# JVP Sensitivity Logic
# ----------------------------
def get_clip_jvp_sensitivity(image_tensor):
    from torch.func import jvp

    # 1. Define pure function for the image encoder
    def encode_forward(img_input):
        # Ensure input matches model dtype
        return model.encode_image(img_input.to(target_dtype))

    # 2. Prepare primal (x) and random unit tangent (v)
    x = image_tensor.detach().to(DEVICE, dtype=target_dtype)
    v = torch.randn_like(x, dtype=target_dtype, device=DEVICE)
    v = v / (torch.norm(v) + 1e-8) # Normalize v

    try:
        # 3. Compute Jacobian-Vector Product
        # Returns (output, jvp_result)
        _, jvp_res = jvp(encode_forward, (x,), (v,))
        
        # 4. Score is the L2 norm of the change in embedding space
        score = torch.norm(jvp_res.float(), p=2).item()
        return float(score)

    except RuntimeError as e:
        # Fallback to Finite Difference if jvp tracing fails
        eps = 0.05
        with torch.no_grad():
            f_x = encode_forward(x)
            f_x_eps = encode_forward(x + eps * v)
            jvp_approx = (f_x_eps - f_x) / eps
            return float(torch.norm(jvp_approx.float(), p=2).item())

# ----------------------------
# Execution Loop
# ----------------------------
print(f"Calculating JVP sensitivity for {subset_size} samples...")
sensitivity_scores = []

for i, (image, label) in enumerate(dataloader):
    try:
        score = get_clip_jvp_sensitivity(image)
        
        if np.isfinite(score):
            sensitivity_scores.append(score)
        
        if (i + 1) % 50 == 0:
            print(f"[{i+1}/{subset_size}] Current Mean JVP Score: {np.mean(sensitivity_scores):.6f}")

        # Memory management
        del image
        if i % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error on sample {i}: {e}")

# ----------------------------
# Summary
# ----------------------------
if sensitivity_scores:
    print("\n--- CLIP JVP Sensitivity Summary (Caltech-256) ---")
    print(f"Samples Evaluated: {len(sensitivity_scores)}")
    print(f"Average JVP Score: {np.mean(sensitivity_scores):.6f}")
    print(f"Standard Deviation: {np.std(sensitivity_scores):.6f}")
