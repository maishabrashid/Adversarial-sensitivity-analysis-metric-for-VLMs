import json
import os
import re
from collections import Counter
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForQuestionAnswering


# =====================================================
# ================ USER CONFIG ========================
# =====================================================

QUESTIONS_JSON   = "/data/rashidm/COCO/v2_OpenEnded_mscoco_val2014_questions.json"
ANNOTATIONS_JSON = "/data/rashidm/COCO/v2_mscoco_val2014_annotations.json"
IMAGES_DIR       = "/data/rashidm/COCO/val2014"

MODEL_ID = "Salesforce/blip-vqa-base"

BATCH_SIZE     = 8
NUM_WORKERS    = 4
MAX_NEW_TOKENS = 10
MAX_SAMPLES    = 1000   # -1 for full val

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =====================================================
# ============ VQA NORMALIZATION ======================
# =====================================================

_ARTICLES = {"a", "an", "the"}
_PUNCT = re.compile(r"[^\w\s]")
_MULTI_SPACE = re.compile(r"\s+")

_NUMBERS = {
    "zero": "0", "one": "1", "two": "2", "three": "3",
    "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "ten": "10"
}

def normalize_vqa_answer(ans: str) -> str:
    ans = ans.lower().strip()
    ans = _PUNCT.sub(" ", ans)
    ans = _MULTI_SPACE.sub(" ", ans).strip()
    words = [w for w in ans.split() if w not in _ARTICLES]
    words = [_NUMBERS.get(w, w) for w in words]
    return " ".join(words)

def vqa_soft_accuracy(pred: str, gt_answers: List[str]) -> float:
    pred_n = normalize_vqa_answer(pred)
    gts_n = [normalize_vqa_answer(a) for a in gt_answers]
    match_count = sum(1 for a in gts_n if a == pred_n)
    return min(match_count / 3.0, 1.0)


# =====================================================
# ================= DATASET ===========================
# =====================================================

class VQAv2ValDataset(Dataset):
    def __init__(self, questions_json: str, annotations_json: str, images_dir: str, max_samples: int = -1):
        self.images_dir = images_dir

        with open(questions_json, "r") as f:
            q_data = json.load(f)
        with open(annotations_json, "r") as f:
            a_data = json.load(f)

        q_map: Dict[int, Dict[str, Any]] = {}
        for q in q_data["questions"]:
            q_map[q["question_id"]] = {"question": q["question"], "image_id": q["image_id"]}

        self.samples: List[Dict[str, Any]] = []
        for ann in a_data["annotations"]:
            qid = ann["question_id"]
            if qid not in q_map:
                continue

            self.samples.append({
                "question": q_map[qid]["question"],
                "image_id": q_map[qid]["image_id"],
                "answers": [a["answer"] for a in ann["answers"]],
                "answer_type": ann.get("answer_type", "unknown"),
                "question_id": qid
            })

        if max_samples is not None and max_samples > 0:
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        img_path = os.path.join(self.images_dir, f"COCO_val2014_{s['image_id']:012d}.jpg")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Missing image: {img_path}")

        image = Image.open(img_path).convert("RGB")

        return {
            "image": image,                  # keep as PIL
            "question": s["question"],
            "answers": s["answers"],
            "answer_type": s["answer_type"],
            "question_id": s["question_id"],
        }


# =====================================================
# =============== COLLATE FUNCTION ====================
# =====================================================

class BlipCollator:
    """
    Converts a list of dataset samples (with PIL images) into BLIP processor tensors.
    """
    def __init__(self, processor: BlipProcessor):
        self.processor = processor

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        images = [b["image"] for b in batch]
        questions = [b["question"] for b in batch]

        inputs = self.processor(
            images=images,
            text=questions,
            return_tensors="pt",
            padding=True
        )

        return {
            "inputs": inputs,
            "answers": [b["answers"] for b in batch],
            "answer_type": [b["answer_type"] for b in batch],
            "question_id": [b["question_id"] for b in batch],
        }


# =====================================================
# ================= EVALUATION ========================
# =====================================================

@torch.no_grad()
def evaluate(model, processor, dataloader):
    model.eval()

    total = 0
    total_acc = 0.0
    acc_by_type = Counter()
    cnt_by_type = Counter()

    for batch in tqdm(dataloader, desc="Evaluating"):
        inputs = batch["inputs"].to(DEVICE)
        gt_answers = batch["answers"]
        answer_types = batch["answer_type"]

        generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
        preds = processor.batch_decode(generated_ids, skip_special_tokens=True)

        for pred, gts, atype in zip(preds, gt_answers, answer_types):
            acc = vqa_soft_accuracy(pred, gts)

            total_acc += acc
            total += 1
            acc_by_type[atype] += acc
            cnt_by_type[atype] += 1

    overall = total_acc / max(total, 1)

    print("\n================ RESULTS ================")
    print(f"Samples: {total}")
    print(f"Overall VQA Soft Accuracy: {overall:.4f}")
    print("\nBreakdown by Answer Type:")
    for k in sorted(cnt_by_type.keys()):
        print(f"{k:10s} : {(acc_by_type[k] / cnt_by_type[k]):.4f}")


# =====================================================
# ================= MAIN ==============================
# =====================================================

def main():
    print("Device:", DEVICE)
    print("Loading BLIP model:", MODEL_ID)

    processor = BlipProcessor.from_pretrained(MODEL_ID)
    model = BlipForQuestionAnswering.from_pretrained(MODEL_ID).to(DEVICE)

    dataset = VQAv2ValDataset(
        QUESTIONS_JSON,
        ANNOTATIONS_JSON,
        IMAGES_DIR,
        max_samples=MAX_SAMPLES
    )

    collate_fn = BlipCollator(processor)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
    )

    evaluate(model, processor, loader)


if __name__ == "__main__":
    main()
