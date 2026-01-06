import os
import random
import warnings
import numpy as np
import gc
from PIL import Image
import matplotlib.pyplot as plt

import torch
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    CLIPProcessor,
    CLIPModel,
)
from pycocotools.coco import COCO

warnings.filterwarnings("ignore")

# ----------------------------
# Config (edit as needed)
# ----------------------------
LLAVA_MODEL_ID = "llava-hf/llava-1.5-7b-hf"
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

COCO_ANN_FILE = "/data/rashidm/COCO/annotations/captions_val2017.json"
IMAGES_DIR = "/data/rashidm/COCO/val2017"

NUM_SAMPLES = 1200
IMAGE_SIZE = 224  # fixed size so all noises align

# Gaussian noise schedule (sigma in [0,1] pixel space)
SIGMA_START = 0.00
SIGMA_STEP = 0.02
SIGMA_MAX = 0.50

# Stopping criterion for CLIPScore drop:
# Use either ABSOLUTE drop OR RELATIVE drop.
USE_RELATIVE_DROP = True
REL_DROP_FACTOR = 0.75   # stop when noisy_score < clean_score * 0.75
ABS_DROP_DELTA = 0.20    # stop when noisy_score < clean_score - 0.20

# LLaVA generation settings (keep deterministic-ish)
MAX_NEW_TOKENS = 32
NUM_BEAMS = 3
DO_SAMPLE = False

SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Reduce fragmentation (best set before CUDA allocations; harmless here too)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# ----------------------------
# Utils: image conversions + noise
# ----------------------------
def pil_to_float01_rgb(pil_img: Image.Image, size: int) -> np.ndarray:
    pil_img = pil_img.convert("RGB").resize((size, size))
    arr = np.asarray(pil_img).astype(np.float32) / 255.0
    return arr  # (H,W,3), float32 in [0,1]


def float01_rgb_to_pil(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0.0, 1.0)
    uint8 = (arr * 255.0).round().astype(np.uint8)
    return Image.fromarray(uint8, mode="RGB")


def add_gaussian_noise(clean: np.ndarray, sigma: float) -> np.ndarray:
    noise = np.random.normal(0.0, sigma, size=clean.shape).astype(np.float32)
    noisy = np.clip(clean + noise, 0.0, 1.0)
    return noisy


# ----------------------------
# Load models
# ----------------------------
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load LLaVA ---
    if device.type == "cuda":
        llava = LlavaForConditionalGeneration.from_pretrained(
            LLAVA_MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="eager",
        )
    else:
        llava = LlavaForConditionalGeneration.from_pretrained(
            LLAVA_MODEL_ID,
            torch_dtype=torch.float32,
            attn_implementation="eager",
        ).to(device)

    llava_processor = AutoProcessor.from_pretrained(LLAVA_MODEL_ID)
    llava.eval()
    for p in llava.parameters():
        p.requires_grad = False

    # --- Load CLIP ---
    clip = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    clip.eval()
    for p in clip.parameters():
        p.requires_grad = False

    return llava, llava_processor, clip, clip_processor, device


# ----------------------------
# Caption generation (LLaVA)
# ----------------------------
@torch.no_grad()
def generate_caption(llava, llava_processor, device, pil_image: Image.Image) -> str:
    prompt = "USER: <image>\nDescribe the image in one sentence. ASSISTANT:"
    inputs = llava_processor(text=prompt, images=pil_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    gen_ids = llava.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        num_beams=NUM_BEAMS,
        do_sample=DO_SAMPLE,
    )
    text = llava_processor.batch_decode(gen_ids, skip_special_tokens=True)[0]

    # Extract assistant span if present
    if "ASSISTANT:" in text:
        text = text.split("ASSISTANT:", 1)[-1].strip()
    return text.strip()


# ----------------------------
# CLIPScore (image-text alignment)
# ----------------------------
@torch.no_grad()
def clipscore(clip, clip_processor, device, pil_image: Image.Image, text: str) -> float:
    """
    CLIPScore as cosine similarity between normalized image/text embeddings.
    Returns a value in roughly [-1, 1] (often ~[0.1, 0.4+] for reasonable matches).
    """
    inputs = clip_processor(text=[text], images=pil_image, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    img_feat = clip.get_image_features(pixel_values=inputs["pixel_values"])
    txt_feat = clip.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

    img_feat = img_feat / (img_feat.norm(dim=-1, keepdim=True) + 1e-12)
    txt_feat = txt_feat / (txt_feat.norm(dim=-1, keepdim=True) + 1e-12)

    score = (img_feat * txt_feat).sum(dim=-1).item()
    return float(score)


def should_stop(clean_score: float, noisy_score: float) -> bool:
    if USE_RELATIVE_DROP:
        return noisy_score < (clean_score * REL_DROP_FACTOR)
    return noisy_score < (clean_score - ABS_DROP_DELTA)


# ----------------------------
# Main experiment
# ----------------------------
def main():
    assert os.path.exists(COCO_ANN_FILE), f"Missing COCO annotations: {COCO_ANN_FILE}"
    assert os.path.isdir(IMAGES_DIR), f"Missing COCO images dir: {IMAGES_DIR}"

    llava, llava_processor, clip, clip_processor, device = load_models()

    coco = COCO(COCO_ANN_FILE)
    img_ids = list(coco.imgs.keys())
    random.shuffle(img_ids)
    selected = img_ids[:NUM_SAMPLES]
    print(f"Selected {len(selected)} random COCO images.")

    extracted_noises = []  # list of (H,W,3) signed float32 arrays

    for j, img_id in enumerate(selected, start=1):
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(IMAGES_DIR, info["file_name"])

        try:
            raw = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[{j}] Skipping (cannot open): {img_path} ({e})")
            continue

        # Load ONE GT caption just for logging/context (not used by CLIPScore threshold)
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        gt_caption = anns[0].get("caption", "a photo") if anns else "a photo"

        clean_arr = pil_to_float01_rgb(raw, IMAGE_SIZE)
        clean_pil = float01_rgb_to_pil(clean_arr)

        # --- clean caption + clean CLIPScore ---
        try:
            cap_clean = generate_caption(llava, llava_processor, device, clean_pil)
            score_clean = clipscore(clip, clip_processor, device, clean_pil, cap_clean)
        except torch.cuda.OutOfMemoryError:
            print(f"[{j}] OOM on clean generation/scoring; skipping this image.")
            continue

        print(f"\n[{j}/{NUM_SAMPLES}] img_id={img_id} file={info['file_name']}")
        print(f"GT (for reference): {gt_caption}")
        print(f"GEN(clean):         {cap_clean}")
        print(f"CLIPScore(clean):   {score_clean:.4f}")
        if USE_RELATIVE_DROP:
            print(f"Stop when CLIPScore < {score_clean * REL_DROP_FACTOR:.4f} (relative factor={REL_DROP_FACTOR})")
        else:
            print(f"Stop when CLIPScore < {score_clean - ABS_DROP_DELTA:.4f} (abs delta={ABS_DROP_DELTA})")

        sigma = SIGMA_START
        fail_noise = None
        found = False

        while sigma <= SIGMA_MAX:
            noisy_arr = add_gaussian_noise(clean_arr, sigma)
            noisy_pil = float01_rgb_to_pil(noisy_arr)

            try:
                cap_noisy = generate_caption(llava, llava_processor, device, noisy_pil)
                score_noisy = clipscore(clip, clip_processor, device, noisy_pil, cap_noisy)
            except torch.cuda.OutOfMemoryError:
                print(f"  sigma={sigma:.3f} -> OOM; treating as failure.")
                fail_noise = (noisy_arr - clean_arr).astype(np.float32)
                found = True
                break

            #print(f"  sigma={sigma:.3f} | CLIPScore={score_noisy:.4f} | GEN: {cap_noisy}")

            if should_stop(score_clean, score_noisy):
                fail_noise = (noisy_arr - clean_arr).astype(np.float32)
                found = True
                break

            sigma += SIGMA_STEP
        print(f"  sigma={sigma:.3f}")
        if not found:
            print(f"  Did not cross threshold up to sigma={SIGMA_MAX:.2f}. Using sigma_max noise.")
            noisy_arr = add_gaussian_noise(clean_arr, SIGMA_MAX)
            fail_noise = (noisy_arr - clean_arr).astype(np.float32)

        extracted_noises.append(fail_noise)

    if len(extracted_noises) == 0:
        print("No noises collected. Exiting.")
        return

    # --- Average the extracted noises across images ---
    avg_noise = np.mean(np.stack(extracted_noises, axis=0), axis=0).astype(np.float32)

    # --- Visualization: map signed noise to [0,1] for plotting ---
    # Robust scaling for visibility (percentile-based)
    viz = avg_noise.copy()
    lo, hi = np.percentile(viz, 2), np.percentile(viz, 98)
    if hi - lo < 1e-8:
        lo, hi = -0.05, 0.05
    viz = (viz - lo) / (hi - lo)
    viz = np.clip(viz, 0.0, 1.0)

    out_png = "avg_noise.png"
    plt.figure()
    plt.imshow(viz)
    plt.axis("off")
    plt.title("Average extracted noise (CLIPScore collapse)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print(f"\nSaved average noise plot to: {out_png}")

    out_npy = "avg_noise.npy"
    np.save(out_npy, avg_noise)
    print(f"Saved raw average noise array to: {out_npy}")


if __name__ == "__main__":
    main()
