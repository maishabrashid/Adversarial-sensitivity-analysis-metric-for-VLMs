import torch
from torch.func import jvp
from torchvision.datasets import CocoCaptions
import torchvision.transforms as T
from transformers import CLIPVisionModel
from tqdm import tqdm  # Recommended for progress tracking

# 1. Setup Data and Model
img_root = '/data/rashidm/COCO/train2017'
ann_file = '/data/rashidm/COCO/annotations/captions_train2017.json'

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "openai/clip-vit-base-patch32"

model = CLIPVisionModel.from_pretrained(
    model_id, 
    attn_implementation="eager"
).to(device).eval()

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize((0.481, 0.457, 0.408), (0.268, 0.261, 0.275))
])

dataset = CocoCaptions(root=img_root, annFile=ann_file, transform=transform)

# 2. Define Functional Forward Pass
def get_embedding(pixel_values):
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        return model(pixel_values=pixel_values).pooler_output

# 3. Iteration Loop
num_images = 100
total_random_sens = 0.0
total_adv_sens = 0.0
count = 0

print(f"Calculating sensitivity for {num_images} images...")

for i in tqdm(range(num_images)):
    try:
        # Load single image
        image_tensor, _ = dataset[i]
        image_tensor = image_tensor.unsqueeze(0).to(device)
        image_tensor.requires_grad = True

        # --- A. Random Sensitivity ---
        v_rand = torch.randn_like(image_tensor).to(device)
        v_rand = v_rand / torch.norm(v_rand)
        _, jvp_rand = jvp(get_embedding, (image_tensor.detach(),), (v_rand,))
        total_random_sens += torch.norm(jvp_rand).item()

        # --- B. Adversarial (FGSM) Sensitivity ---
        # 1. Generate FGSM Direction
        embedding = get_embedding(image_tensor)
        loss = embedding.norm()
        model.zero_grad()
        loss.backward()
        
        v_fgsm = image_tensor.grad.data.sign()
        v_fgsm_norm = v_fgsm / torch.norm(v_fgsm)
        
        # 2. Calculate JVP in that direction
        _, jvp_adv = jvp(get_embedding, (image_tensor.detach(),), (v_fgsm_norm,))
        total_adv_sens += torch.norm(jvp_adv).item()
        
        count += 1
        
    except Exception as e:
        print(f"Skipping image {i} due to error: {e}")

# 4. Final Averaging
avg_random = total_random_sens / count
avg_adv = total_adv_sens / count

print("\n" + "="*30)
print(f"Results over {count} images:")
print(f"Avg Random Sensitivity:      {avg_random:.6f}")
print(f"Avg Adversarial Sensitivity: {avg_adv:.6f}")
print(f"Sensitivity Ratio (Adv/Rand): {avg_adv/avg_random:.2f}x")
print("="*30)