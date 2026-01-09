import os
import random
import warnings
import numpy as np
import gc
from PIL import Image

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, CLIPProcessor, CLIPModel
from pycocotools.coco import COCO

warnings.filterwarnings("ignore")

# ----------------------------
# Config
# ----------------------------
LLAVA_MODEL_ID = "llava-hf/llava-1.5-7b-hf"
CLIP_MODEL_ID  = "openai/clip-vit-base-patch32"

COCO_ANN_FILE = "/data/rashidm/COCO/annotations/captions_val2017.json"
IMAGES_DIR    = "/data/rashidm/COCO/val2017"

NUM_SAMPLES = 1200          # set to 1200 if you really want, but start smaller to validate
IMAGE_SIZE  = 224

# Evaluate multiple eps values (FGSM in [0,1] RGB space)
EPS_LIST = [0.00, 0.02, 0.05, 0.10, 0.15, 0.20]

# LLaVA generation (make stable/cheap)
MAX_NEW_TOKENS = 32
NUM_BEAMS = 1
DO_SAMPLE = False

SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Optional: save some examples
SAVE_EXAMPLES = False
EXAMPLE_DIR = "./fgsm_examples"
MAX_EXAMPLES_TO_SAVE = 10

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# CLIP normalization constants (for CLIP ViT-B/32 preprocessing)
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


# ----------------------------
# Utils
# ----------------------------
def pil_to_float01_rgb(pil_img: Image.Image, size: int) -> np.ndarray:
    pil_img = pil_img.convert("RGB").resize((size, size))
    return (np.asarray(pil_img).astype(np.float32) / 255.0)

def float01_rgb_to_pil(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0.0, 1.0)
    return Image.fromarray((arr * 255.0).round().astype(np.uint8), mode="RGB")

def _clip_normalize(img01_bchw: torch.Tensor, device: torch.device) -> torch.Tensor:
    mean = CLIP_MEAN.to(device)
    std  = CLIP_STD.to(device)
    return (img01_bchw - mean) / std


# ----------------------------
# Load models
# ----------------------------
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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

    llava_proc = AutoProcessor.from_pretrained(LLAVA_MODEL_ID)
    llava.eval()
    for p in llava.parameters():
        p.requires_grad = False

    clip = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device)
    clip_proc = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    clip.eval()
    for p in clip.parameters():
        p.requires_grad = False

    return llava, llava_proc, clip, clip_proc, device


# ----------------------------
# LLaVA caption generation
# ----------------------------
@torch.no_grad()
def generate_caption(llava, llava_proc, device, pil_image: Image.Image) -> str:
    prompt = "USER: <image>\nDescribe the image in one sentence. ASSISTANT:"
    inputs = llava_proc(text=prompt, images=pil_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    gen_ids = llava.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        num_beams=NUM_BEAMS,
        do_sample=DO_SAMPLE,
    )
    text = llava_proc.batch_decode(gen_ids, skip_special_tokens=True)[0]
    if "ASSISTANT:" in text:
        text = text.split("ASSISTANT:", 1)[-1].strip()
    return text.strip()


# ----------------------------
# CLIPScore (cosine similarity)
# ----------------------------
@torch.no_grad()
def clipscore(clip, clip_proc, device, pil_image: Image.Image, text: str) -> float:
    inputs = clip_proc(text=[text], images=pil_image, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    img_feat = clip.get_image_features(pixel_values=inputs["pixel_values"])
    txt_feat = clip.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

    img_feat = img_feat / (img_feat.norm(dim=-1, keepdim=True) + 1e-12)
    txt_feat = txt_feat / (txt_feat.norm(dim=-1, keepdim=True) + 1e-12)

    return float((img_feat * txt_feat).sum(dim=-1).mean().item())


# ----------------------------
# FGSM attack (single-step) on image to reduce CLIP similarity to CLEAN caption
# ----------------------------
def fgsm_attack_clip_to_clean_caption(
    clip_model: CLIPModel,
    device: torch.device,
    clean_img01_hwc: np.ndarray,
    clean_caption: str,
    clip_proc: CLIPProcessor,
    eps: float
) -> np.ndarray:
    """
    Create adversarial image x_adv from clean x by descending on CLIP similarity:
      x_adv = clamp(x - eps * sign(∇_x sim(CLIP_I(x), CLIP_T(clean_caption))), 0, 1)
    Returns x_adv in numpy HWC [0,1].
    """
    # Tokenize text once (no grad)
    text_inputs = clip_proc(text=[clean_caption], return_tensors="pt", padding=True)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

    # x in [0,1], shape (1,3,H,W)
    x = torch.from_numpy(clean_img01_hwc).permute(2, 0, 1).unsqueeze(0).to(device).float()
    x.requires_grad_(True)

    x_norm = _clip_normalize(x, device)

    img_feat = clip_model.get_image_features(pixel_values=x_norm)
    txt_feat = clip_model.get_text_features(
        input_ids=text_inputs["input_ids"],
        attention_mask=text_inputs["attention_mask"]
    )

    img_feat = img_feat / (img_feat.norm(dim=-1, keepdim=True) + 1e-12)
    txt_feat = txt_feat / (txt_feat.norm(dim=-1, keepdim=True) + 1e-12)

    sim = (img_feat * txt_feat).sum(dim=-1).mean()  # scalar
    sim.backward()

    adv = (x - eps * x.grad.sign()).clamp(0.0, 1.0).detach()
    adv_img01 = adv.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)

    # cleanup
    del text_inputs, x, x_norm, img_feat, txt_feat, sim, adv
    return adv_img01


# ----------------------------
# Main evaluation
# ----------------------------
def main():
    assert os.path.exists(COCO_ANN_FILE), f"Missing COCO annotations: {COCO_ANN_FILE}"
    assert os.path.isdir(IMAGES_DIR), f"Missing COCO images dir: {IMAGES_DIR}"

    if SAVE_EXAMPLES:
        os.makedirs(EXAMPLE_DIR, exist_ok=True)

    llava, llava_proc, clip, clip_proc, device = load_models()

    coco = COCO(COCO_ANN_FILE)
    img_ids = list(coco.imgs.keys())
    random.shuffle(img_ids)
    selected = img_ids[:NUM_SAMPLES]
    print(f"Selected {len(selected)} random COCO images.")

    # Collect per-eps stats
    stats = {eps: {"clean": [], "adv": []} for eps in EPS_LIST}

    saved = 0

    for idx, img_id in enumerate(selected, start=1):
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(IMAGES_DIR, info["file_name"])

        try:
            raw = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        clean_arr = pil_to_float01_rgb(raw, IMAGE_SIZE)
        clean_pil = float01_rgb_to_pil(clean_arr)

        try:
            cap_clean = generate_caption(llava, llava_proc, device, clean_pil)
            score_clean = clipscore(clip, clip_proc, device, clean_pil, cap_clean)
        except torch.cuda.OutOfMemoryError:
            continue

        # Evaluate each eps
        for eps in EPS_LIST:
            if eps == 0.0:
                # No attack: "adv" equals clean
                stats[eps]["clean"].append(score_clean)
                stats[eps]["adv"].append(score_clean)
                continue

            try:
                adv_arr = fgsm_attack_clip_to_clean_caption(
                    clip_model=clip,
                    device=device,
                    clean_img01_hwc=clean_arr,
                    clean_caption=cap_clean,
                    clip_proc=clip_proc,
                    eps=eps
                )
                adv_pil = float01_rgb_to_pil(adv_arr)

                cap_adv = generate_caption(llava, llava_proc, device, adv_pil)
                score_adv = clipscore(clip, clip_proc, device, adv_pil, cap_adv)

            except torch.cuda.OutOfMemoryError:
                continue

            stats[eps]["clean"].append(score_clean)
            stats[eps]["adv"].append(score_adv)

            # Save a few examples for qualitative inspection
            if SAVE_EXAMPLES and saved < MAX_EXAMPLES_TO_SAVE:
                adv_save = os.path.join(EXAMPLE_DIR, f"img{img_id}_eps{eps:.2f}_adv.png")
                clean_save = os.path.join(EXAMPLE_DIR, f"img{img_id}_clean.png")
                clean_pil.save(clean_save)
                adv_pil.save(adv_save)
                with open(os.path.join(EXAMPLE_DIR, f"img{img_id}_eps{eps:.2f}_captions.txt"), "w") as f:
                    f.write(f"clean_caption: {cap_clean}\n")
                    f.write(f"adv_caption:   {cap_adv}\n")
                    f.write(f"clean_score:   {score_clean:.4f}\n")
                    f.write(f"adv_score:     {score_adv:.4f}\n")
                saved += 1

        if idx % 25 == 0:
            print(f"Processed {idx}/{NUM_SAMPLES}")

    # Print summary
    print("\n--- FGSM Robustness Summary (Variant A) ---")
    for eps in EPS_LIST:
        clean_scores = np.array(stats[eps]["clean"], dtype=np.float32)
        adv_scores   = np.array(stats[eps]["adv"], dtype=np.float32)

        if len(clean_scores) == 0:
            print(f"eps={eps:.2f} | no data")
            continue

        clean_mean = float(clean_scores.mean())
        adv_mean   = float(adv_scores.mean())
        drop_abs   = clean_mean - adv_mean
        drop_rel   = (drop_abs / max(clean_mean, 1e-8)) * 100.0

        print(
            f"eps={eps:.2f} | n={len(clean_scores)} | "
            f"clean={clean_mean:.4f}, {clean_scores.std():.4f} | "
            f"adv={adv_mean:.4f}, {adv_scores.std():.4f} | "
            f"drop={drop_abs:.4f} ({drop_rel:.2f}%)"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
