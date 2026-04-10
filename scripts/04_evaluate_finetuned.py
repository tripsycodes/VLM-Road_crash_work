#!/usr/bin/env python3

import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
import torch
import gc

import nltk
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import get_config
from PIL import Image


# -------------------------------
def load_frames(video_path: str, max_frames: int = 30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % 5 == 0:
            frames.append(frame)
            if len(frames) >= max_frames:
                break
        count += 1

    cap.release()
    return frames


# -------------------------------
def load_finetuned_model(checkpoint_path, base_model_name, device):
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
    from peft import LoraConfig, get_peft_model, TaskType

    print(f"Loading base model: {base_model_name}")

    processor = LlavaNextProcessor.from_pretrained(base_model_name)

    model = LlavaNextForConditionalGeneration.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    state_dict = checkpoint["model_state_dict"]

    # ✅ Correct LoRA config
    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    # Load only LoRA weights
    lora_state_dict = {k: v for k, v in state_dict.items() if "lora" in k.lower()}
    model.load_state_dict(lora_state_dict, strict=False)

    model = model.half()
    model = model.to("cpu") 
    model.eval()

    # 🔥 Free memory
    gc.collect()
    torch.cuda.empty_cache()

    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    train_loss = checkpoint.get("train_loss", None)
    print(f"Training loss: {train_loss:.4f}" if train_loss else "Training loss: N/A")

    val_loss = checkpoint.get("val_loss", None)
    print(f"Validation loss: {val_loss:.4f}" if val_loss else "Validation loss: N/A")

    # -------------------------------
    class Wrapper:
        def __init__(self, model, processor):
            self.model = model
            self.processor = processor

        def generate_summary(self, frames):
            if not frames:
                return {"text_summary": ""}

            rep = frames[len(frames) // 2]
            img = Image.fromarray(rep).convert("RGB")

            prompt = "USER: <image>\nDescribe the accident.\nASSISTANT:"

            inputs = self.processor(text=prompt, images=img, return_tensors="pt")
            
            inputs = {k: v.to("cpu") for k, v in inputs.items()}

            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=False
                )

            text = self.processor.decode(out[0], skip_special_tokens=True)
            return {"text_summary": text.split("ASSISTANT:")[-1].strip()}

    return Wrapper(model, processor)


# -------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = get_config(args.config)

    root_dir = Path(config["dataset"]["root_dir"])
    processed_dir = root_dir / config["dataset"]["processed_dir"]

    with open(processed_dir / "split_info.json") as f:
        split = json.load(f)

    with open(processed_dir / "annotations_test.json") as f:
        annotations = json.load(f)

    model = load_finetuned_model(
        args.checkpoint,
        config["model"]["vision_model"],
        device="cpu"
    )

    predictions, references = [], []

    print("\nRunning inference...")

    for video_path in tqdm(split["splits"]["test"]):
        vid = Path(video_path).stem
        if vid not in annotations:
            continue

        frames = load_frames(video_path)
        if not frames:
            continue

        pred = model.generate_summary(frames)["text_summary"]
        gt = annotations[vid]["text_summary"]

        predictions.append(pred)
        references.append(gt)

    print("\nDone. Samples:", len(predictions))


if __name__ == "__main__":
    main()
