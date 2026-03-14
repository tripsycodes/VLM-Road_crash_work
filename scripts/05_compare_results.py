#!/usr/bin/env python3
"""
Compare zero-shot vs fine-tuned model evaluation results.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import get_config


def load_metrics(file_path: Path) -> Dict[str, Any]:
    """Load metrics from JSON file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {file_path}")
    
    with open(file_path) as f:
        return json.load(f)


def calculate_improvement(old_val: float, new_val: float) -> Dict[str, Any]:
    """Calculate improvement metrics."""
    if old_val == 0:
        return {
            "absolute": new_val - old_val,
            "relative": float('inf') if new_val > 0 else 0.0,
            "percentage": float('inf') if new_val > 0 else 0.0
        }
    
    absolute = new_val - old_val
    relative = (new_val - old_val) / old_val
    percentage = relative * 100
    
    return {
        "absolute": absolute,
        "relative": relative,
        "percentage": percentage
    }


def format_metric_value(val: float, decimals: int = 4) -> str:
    """Format metric value for display."""
    return f"{val:.{decimals}f}"


def print_comparison_table(zero_shot: Dict, finetuned: Dict, metric_name: str):
    """Print a comparison table for a metric."""
    print(f"\n{'='*70}")
    print(f"{metric_name.upper()} COMPARISON")
    print(f"{'='*70}")
    print(f"{'Metric':<30} {'Zero-Shot':<15} {'Fine-Tuned':<15} {'Improvement':<15}")
    print(f"{'-'*70}")
    
    if metric_name == "bleu":
        for i in range(1, 5):
            key = f"bleu_{i}"
            if key in zero_shot and key in finetuned:
                zs_val = zero_shot[key]
                ft_val = finetuned[key]
                imp = calculate_improvement(zs_val, ft_val)
                imp_str = f"{imp['absolute']:+.4f} ({imp['percentage']:+.2f}%)"
                print(f"{f'BLEU-{i}':<30} {format_metric_value(zs_val):<15} {format_metric_value(ft_val):<15} {imp_str:<15}")
        
        if "bleu_corpus" in zero_shot and "bleu_corpus" in finetuned:
            zs_val = zero_shot["bleu_corpus"]
            ft_val = finetuned["bleu_corpus"]
            imp = calculate_improvement(zs_val, ft_val)
            imp_str = f"{imp['absolute']:+.4f} ({imp['percentage']:+.2f}%)"
            print(f"{'BLEU-Corpus':<30} {format_metric_value(zs_val):<15} {format_metric_value(ft_val):<15} {imp_str:<15}")
    
    elif metric_name == "rouge":
        for key in ["rouge_1", "rouge_2", "rouge_l"]:
            if key in zero_shot and key in finetuned:
                zs_val = zero_shot[key]
                ft_val = finetuned[key]
                imp = calculate_improvement(zs_val, ft_val)
                imp_str = f"{imp['absolute']:+.4f} ({imp['percentage']:+.2f}%)"
                display_name = key.replace("_", "-").upper()
                print(f"{display_name:<30} {format_metric_value(zs_val):<15} {format_metric_value(ft_val):<15} {imp_str:<15}")
    
    elif metric_name == "nli":
        nli_zs = zero_shot.get("nli_scores", {})
        nli_ft = finetuned.get("nli_scores", {})
        
        for key in ["entailment_accuracy", "avg_entailment_prob"]:
            if key in nli_zs and key in nli_ft:
                zs_val = nli_zs[key]
                ft_val = nli_ft[key]
                imp = calculate_improvement(zs_val, ft_val)
                imp_str = f"{imp['absolute']:+.4f} ({imp['percentage']:+.2f}%)"
                display_name = key.replace("_", " ").title()
                print(f"{display_name:<30} {format_metric_value(zs_val):<15} {format_metric_value(ft_val):<15} {imp_str:<15}")
        
        for key in ["contradiction_rate", "neutral_rate"]:
            if key in nli_zs and key in nli_ft:
                zs_val = nli_zs[key]
                ft_val = nli_ft[key]
                imp = calculate_improvement(zs_val, ft_val)
                imp_str = f"{imp['absolute']:+.4f} ({imp['percentage']:+.2f}%)"
                display_name = key.replace("_", " ").title()
                print(f"{display_name:<30} {format_metric_value(zs_val):<15} {format_metric_value(ft_val):<15} {imp_str:<15}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare zero-shot vs fine-tuned evaluation results"
    )
    parser.add_argument(
        "--zero_shot_metrics",
        type=str,
        default="results/zero_shot/metrics.json",
        help="Path to zero-shot metrics JSON file"
    )
    parser.add_argument(
        "--finetuned_metrics",
        type=str,
        required=True,
        help="Path to fine-tuned metrics JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save comparison report (JSON format)"
    )
    args = parser.parse_args()
    
    # Load metrics
    print("Loading metrics...")
    zero_shot_path = Path(args.zero_shot_metrics)
    finetuned_path = Path(args.finetuned_metrics)
    
    zero_shot = load_metrics(zero_shot_path)
    finetuned = load_metrics(finetuned_path)
    
    print(f"Zero-shot metrics loaded from: {zero_shot_path}")
    print(f"Fine-tuned metrics loaded from: {finetuned_path}")
    
    # Print header
    print("\n" + "="*70)
    print("ZERO-SHOT vs FINE-TUNED MODEL COMPARISON")
    print("="*70)
    print(f"Zero-shot samples: {zero_shot.get('num_samples', 'N/A')}")
    print(f"Fine-tuned samples: {finetuned.get('num_samples', 'N/A')}")
    
    # Compare BLEU scores
    if "bleu_scores" in zero_shot and "bleu_scores" in finetuned:
        print_comparison_table(zero_shot["bleu_scores"], finetuned["bleu_scores"], "bleu")
    
    # Compare METEOR
    if "meteor" in zero_shot and "meteor" in finetuned:
        print(f"\n{'='*70}")
        print("METEOR COMPARISON")
        print(f"{'='*70}")
        zs_meteor = zero_shot["meteor"]
        ft_meteor = finetuned["meteor"]
        imp = calculate_improvement(zs_meteor, ft_meteor)
        print(f"Zero-shot:  {format_metric_value(zs_meteor)}")
        print(f"Fine-tuned: {format_metric_value(ft_meteor)}")
        print(f"Improvement: {imp['absolute']:+.4f} ({imp['percentage']:+.2f}%)")
    
    # Compare ROUGE
    if any(k.startswith("rouge") for k in zero_shot.keys()) and any(k.startswith("rouge") for k in finetuned.keys()):
        rouge_zs = {k: zero_shot[k] for k in zero_shot.keys() if k.startswith("rouge")}
        rouge_ft = {k: finetuned[k] for k in finetuned.keys() if k.startswith("rouge")}
        print_comparison_table(rouge_zs, rouge_ft, "rouge")
    
    # Compare BERTScore
    if "bertscore" in zero_shot and "bertscore" in finetuned:
        print(f"\n{'='*70}")
        print("BERTSCORE COMPARISON")
        print(f"{'='*70}")
        zs_bert = zero_shot["bertscore"]
        ft_bert = finetuned["bertscore"]
        imp = calculate_improvement(zs_bert, ft_bert)
        print(f"Zero-shot:  {format_metric_value(zs_bert)}")
        print(f"Fine-tuned: {format_metric_value(ft_bert)}")
        print(f"Improvement: {imp['absolute']:+.4f} ({imp['percentage']:+.2f}%)")
    
    # Compare CIDEr
    if "cider" in zero_shot and "cider" in finetuned:
        print(f"\n{'='*70}")
        print("CIDEr COMPARISON")
        print(f"{'='*70}")
        zs_cider = zero_shot["cider"]
        ft_cider = finetuned["cider"]
        imp = calculate_improvement(zs_cider, ft_cider)
        print(f"Zero-shot:  {format_metric_value(zs_cider)}")
        print(f"Fine-tuned: {format_metric_value(ft_cider)}")
        print(f"Improvement: {imp['absolute']:+.4f} ({imp['percentage']:+.2f}%)")
    
    # Compare NLI scores
    if "nli_scores" in zero_shot and "nli_scores" in finetuned:
        print_comparison_table(zero_shot, finetuned, "nli")
    
    # Create comparison summary
    comparison = {
        "zero_shot_metrics": zero_shot_path.as_posix(),
        "finetuned_metrics": finetuned_path.as_posix(),
        "comparison": {}
    }
    
    # Calculate improvements for key metrics
    key_metrics = ["bleu_1", "bleu_4", "meteor", "rouge_1", "rouge_l", "bertscore"]
    
    for metric in key_metrics:
        if metric in zero_shot and metric in finetuned:
            zs_val = zero_shot[metric]
            ft_val = finetuned[metric]
            imp = calculate_improvement(zs_val, ft_val)
            comparison["comparison"][metric] = {
                "zero_shot": zs_val,
                "finetuned": ft_val,
                "improvement": imp
            }
    
    # Handle BLEU scores separately
    if "bleu_scores" in zero_shot and "bleu_scores" in finetuned:
        comparison["comparison"]["bleu_scores"] = {}
        for i in range(1, 5):
            key = f"bleu_{i}"
            if key in zero_shot["bleu_scores"] and key in finetuned["bleu_scores"]:
                zs_val = zero_shot["bleu_scores"][key]
                ft_val = finetuned["bleu_scores"][key]
                imp = calculate_improvement(zs_val, ft_val)
                comparison["comparison"]["bleu_scores"][key] = {
                    "zero_shot": zs_val,
                    "finetuned": ft_val,
                    "improvement": imp
                }
    
    # Save comparison if output path provided
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"\n{'='*70}")
        print(f"Comparison report saved to: {output_path}")
    
    print(f"\n{'='*70}")
    print("COMPARISON COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
