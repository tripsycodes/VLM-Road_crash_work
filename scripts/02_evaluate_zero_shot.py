# #!/usr/bin/env python3
# """Zero-shot evaluation on pretrained model with full metric suite."""

# import sys
# import os
# import json
# from pathlib import Path
# from tqdm import tqdm
# import cv2
# import numpy as np
# import torch

# import nltk
# nltk.download("wordnet")
# nltk.download("omw-1.4")

# # ------------------------------------------------------------------
# # Add project root to path
# # ------------------------------------------------------------------
# project_root = Path(__file__).parent.parent
# sys.path.insert(0, str(project_root))

# from src.models import LLaVANeXTWrapper
# from src.evaluation import BLEUEvaluator, NLIEvaluator
# from src.utils import get_config

# from rouge_score import rouge_scorer
# from nltk.translate.meteor_score import meteor_score
# from bert_score import score as bert_score

# # ------------------------------------------------------------------
# def load_frames(video_path: str, max_frames: int = 30):
#     cap = cv2.VideoCapture(video_path)
#     frames = []
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frames.append(frame)
#         if len(frames) >= max_frames:
#             break
#     cap.release()
#     return frames

# # ------------------------------------------------------------------
# def main():
#     config = get_config()

#     processed_dir = Path(config.get("dataset.root_dir")) / config.get("dataset.processed_dir")
#     split_info_file = processed_dir / "split_info.json"
#     annotations_file = processed_dir / "annotations_test.json"

#     if not split_info_file.exists() or not annotations_file.exists():
#         print("ERROR: Run scripts/01_process_data.py first.")
#         sys.exit(1)

#     with open(split_info_file) as f:
#         split_info = json.load(f)

#     with open(annotations_file) as f:
#         test_annotations = json.load(f)

#     test_videos = split_info["splits"]["test"]

#     print("=" * 60)
#     print("ZERO-SHOT EVALUATION (FULL TEST SET)")
#     print("=" * 60)
#     print(f"Test videos: {len(test_videos)}")

#     # ------------------------------------------------------------------
#     # Device
#     # ------------------------------------------------------------------
#     device = config.get("model.device", "cuda")
#     if device == "cuda" and not torch.cuda.is_available():
#         device = "cpu"

#     # ------------------------------------------------------------------
#     # Model
#     # ------------------------------------------------------------------
#     print("\nLoading model...")
#     model = LLaVANeXTWrapper(
#         model_name=config.get("model.vision_model"),
#         device=device
#     )

#     # ------------------------------------------------------------------
#     # Evaluators
#     # ------------------------------------------------------------------
#     bleu_evaluator = BLEUEvaluator(
#         max_order=config.get("evaluation.bleu.max_order", 4),
#         smooth=config.get("evaluation.bleu.smooth", True)
#     )

#     nli_evaluator = NLIEvaluator(
#         model_name=config.get("evaluation.nli.model_name", "roberta-large-mnli"),
#         device=config.get("evaluation.nli.device", "cuda"),
#         batch_size=config.get("evaluation.nli.batch_size", 16)
#     )

#     predictions, references, results = [], [], []

#     # ------------------------------------------------------------------
#     # Inference
#     # ------------------------------------------------------------------
#     print("\nRunning zero-shot inference...")
#     for video_path in tqdm(test_videos, desc="Evaluating"):
#         video_path = Path(video_path)
#         video_id = video_path.stem

#         if video_id not in test_annotations:
#             continue

#         gt = test_annotations[video_id]["text_summary"]
#         if not gt.strip():
#             continue

#         frames = load_frames(str(video_path), max_frames=config.get("model.max_frames", 30))
#         if not frames:
#             continue

#         try:
#             out = model.generate_summary(frames, task_type="v2t")
#             pred = out.get("text_summary", "").strip()
#         except Exception as e:
#             print(f"Error on {video_id}: {e}")
#             continue

#         if pred:
#             predictions.append(pred)
#             references.append(gt)
#             results.append({
#                 "video_id": video_id,
#                 "prediction": pred,
#                 "reference": gt
#             })

#     if len(predictions) == 0:
#         print("No valid predictions generated.")
#         return

#     # ------------------------------------------------------------------
#     # Metrics
#     # ------------------------------------------------------------------
#     print("\nComputing BLEU...")
#     bleu_scores = bleu_evaluator.compute_bleu_batch(predictions, references)

#     print("Computing NLI...")
#     nli_scores = nli_evaluator.evaluate(predictions, references)

#     print("Computing METEOR, ROUGE, BERTScore, CIDEr...")

#     # METEOR
#     mean_meteor = float(np.mean([
#         meteor_score([r], p) for p, r in zip(predictions, references)
#     ]))

#     # ROUGE
#     scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
#     r1, r2, rl = [], [], []
#     for p, r in zip(predictions, references):
#         s = scorer.score(r, p)
#         r1.append(s["rouge1"].fmeasure)
#         r2.append(s["rouge2"].fmeasure)
#         rl.append(s["rougeL"].fmeasure)

#     mean_rouge_1 = float(np.mean(r1))
#     mean_rouge_2 = float(np.mean(r2))
#     mean_rouge_l = float(np.mean(rl))

#     # BERTScore
#     _, _, F1 = bert_score(predictions, references, lang="en", verbose=False)
#     mean_bertscore = float(F1.mean())

#     # CIDEr
#     try:
#         from pycocoevalcap.cider.cider import Cider
#         gts = {i: [references[i]] for i in range(len(references))}
#         res = {i: [predictions[i]] for i in range(len(predictions))}
#         cider = Cider()
#         mean_cider, _ = cider.compute_score(gts, res)
#         mean_cider = float(mean_cider)
#     except Exception:
#         mean_cider = None

#     # ------------------------------------------------------------------
#     # Save results
#     # ------------------------------------------------------------------
#     results_dir = Path(config.get("paths.results")) / "zero_shot"
#     results_dir.mkdir(parents=True, exist_ok=True)

#     with open(results_dir / "detailed_results.json", "w") as f:
#         json.dump(results, f, indent=2, ensure_ascii=False)

#     metrics = {
#         "num_samples": len(predictions),
#         "bleu_scores": bleu_scores,
#         "mean_meteor": mean_meteor,
#         "mean_rouge_1": mean_rouge_1,
#         "mean_rouge_2": mean_rouge_2,
#         "mean_rouge_l": mean_rouge_l,
#         "mean_bertscore": mean_bertscore,
#         "mean_cider": mean_cider,
#         "nli_scores": nli_scores
#     }

#     with open(results_dir / "metrics.json", "w") as f:
#         json.dump(metrics, f, indent=2)

#     # ------------------------------------------------------------------
#     # Print summary
#     # ------------------------------------------------------------------
#     print("\n" + "=" * 60)
#     print("FINAL RESULTS")
#     print("=" * 60)
#     print(f"Samples: {len(predictions)}")
#     print(f"BLEU-1: {bleu_scores['bleu_1']:.4f}")
#     print(f"METEOR: {mean_meteor:.4f}")
#     print(f"ROUGE-1: {mean_rouge_1:.4f}")
#     print(f"BERTScore: {mean_bertscore:.4f}")
#     print(f"CIDEr: {mean_cider}")
#     print("Saved to:", results_dir)
#     print("=" * 60)

# # ------------------------------------------------------------------
# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3

#!/usr/bin/env python3
"""
Zero-shot evaluation on pretrained model with full metric suite.
"""

import sys
import json
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
import torch

import nltk
nltk.download("wordnet")
nltk.download("omw-1.4")

# ------------------------------------------------------------------
# Add project root
# ------------------------------------------------------------------
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models import LLaVANeXTWrapper
from src.evaluation import BLEUEvaluator, NLIEvaluator
from src.utils import get_config

from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score


# ------------------------------------------------------------------
def load_frames(video_path: str, max_frames: int = 30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if len(frames) >= max_frames:
            break
    cap.release()
    return frames


# ------------------------------------------------------------------
def main():
    config = get_config()

    processed_dir = Path(config["dataset"]["root_dir"]) / config["dataset"]["processed_dir"]
    split_info_file = processed_dir / "split_info.json"
    annotations_file = processed_dir / "annotations_test.json"

    if not split_info_file.exists() or not annotations_file.exists():
        print("ERROR: Run scripts/01_process_data.py first.")
        sys.exit(1)

    with open(split_info_file) as f:
        split_info = json.load(f)

    with open(annotations_file) as f:
        test_annotations = json.load(f)

    test_videos = split_info["splits"]["test"]

    print("=" * 60)
    print("ZERO-SHOT EVALUATION (FULL TEST SET)")
    print("=" * 60)
    print(f"Test videos: {len(test_videos)}")

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    device = config["model"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    print("\nLoading model...")
    model = LLaVANeXTWrapper(
        model_name=config["model"]["vision_model"],
        device=device
    )

    # ------------------------------------------------------------------
    # Evaluators
    # ------------------------------------------------------------------
    bleu_eval = BLEUEvaluator(
        max_order=config["evaluation"]["bleu"]["max_order"],
        smooth=config["evaluation"]["bleu"]["smooth"]
    )

    nli_eval = NLIEvaluator(
        model_name=config["evaluation"]["nli"]["model_name"],
        device=config["evaluation"]["nli"]["device"],
        batch_size=config["evaluation"]["nli"]["batch_size"]
    )

    predictions, references, results = [], [], []

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    print("\nRunning zero-shot inference...")
    for video_path in tqdm(test_videos, desc="Evaluating"):
        video_path = Path(video_path)
        video_id = video_path.stem

        if video_id not in test_annotations:
            continue

        gt = test_annotations[video_id]["text_summary"]
        if not gt.strip():
            continue

        frames = load_frames(str(video_path), max_frames=config["model"]["max_frames"])
        if not frames:
            continue

        try:
            out = model.generate_summary(frames, task_type="v2t")
            pred = out.get("text_summary", "").strip()
        except Exception as e:
            print(f"Error on {video_id}: {e}")
            continue

        if pred:
            predictions.append(pred)
            references.append(gt)
            results.append({
                "video_id": video_id,
                "prediction": pred,
                "reference": gt
            })

    if len(predictions) == 0:
        print("No valid predictions generated.")
        return

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    print("\nComputing BLEU...")
    bleu_scores = bleu_eval.compute_bleu_batch(predictions, references)

    print("Computing NLI...")
    nli_scores = nli_eval.evaluate(predictions, references)

    print("Computing METEOR, ROUGE, BERTScore, CIDEr...")

    # METEOR (FIXED)
    meteor_vals = [
        meteor_score([r.split()], p.split())
        for p, r in zip(predictions, references)
    ]
    mean_meteor = float(np.mean(meteor_vals))

    # ROUGE
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )

    r1, r2, rl = [], [], []
    for p, r in zip(predictions, references):
        s = scorer.score(r, p)
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rl.append(s["rougeL"].fmeasure)

    mean_rouge_1 = float(np.mean(r1))
    mean_rouge_2 = float(np.mean(r2))
    mean_rouge_l = float(np.mean(rl))

    # BERTScore
    _, _, F1 = bert_score(predictions, references, lang="en", verbose=False)
    mean_bertscore = float(F1.mean())

    # CIDEr
    from pycocoevalcap.cider.cider import Cider
    cider = Cider()
    cider_vals = []
    for p, r in zip(predictions, references):
        s, _ = cider.compute_score({0: [r]}, {0: [p]})
        cider_vals.append(s)
    mean_cider = float(np.mean(cider_vals))

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results_dir = Path(config["paths"]["results"]) / "zero_shot"
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "detailed_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    metrics = {
        "num_samples": len(predictions),
        "bleu_scores": bleu_scores,
        "meteor": mean_meteor,
        "rouge_1": mean_rouge_1,
        "rouge_2": mean_rouge_2,
        "rouge_l": mean_rouge_l,
        "bertscore": mean_bertscore,
        "cider": mean_cider,
        "nli_scores": nli_scores
    }

    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Samples: {len(predictions)}")
    print(f"BLEU-1: {bleu_scores['bleu_1']:.4f}")
    print(f"METEOR: {mean_meteor:.4f}")
    print(f"ROUGE-1: {mean_rouge_1:.4f}")
    print(f"BERTScore: {mean_bertscore:.4f}")
    print(f"CIDEr: {mean_cider:.4f}")
    print("Saved to:", results_dir)
    print("=" * 60)


# ------------------------------------------------------------------
if __name__ == "__main__":
    main()
