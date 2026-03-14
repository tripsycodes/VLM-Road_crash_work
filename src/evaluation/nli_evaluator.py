"""Natural Language Inference (NLI) evaluation."""
import torch
from typing import List, Dict, Optional
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class NLIEvaluator:
    """Evaluate text summaries using Natural Language Inference."""
    
    def __init__(self, model_name: str = "roberta-large-mnli", 
                 device: str = "cuda", batch_size: int = 16):
        """
        Initialize NLI evaluator.
        
        Args:
            model_name: Name of pretrained NLI model
            device: Device to run model on
            batch_size: Batch size for inference
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        
        # NLI labels: 0=contradiction, 1=neutral, 2=entailment
        self.label_map = {0: "contradiction", 1: "neutral", 2: "entailment"}
    
    def predict_entailment(self, premise: str, hypothesis: str) -> Dict:
        """
        Predict entailment relationship between premise and hypothesis.
        
        Args:
            premise: Ground truth text (premise)
            hypothesis: Generated text (hypothesis)
            
        Returns:
            Dictionary with prediction and scores
        """
        # Format for NLI: premise and hypothesis
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            predicted_label = torch.argmax(probs, dim=-1).item()
        
        scores = {
            "contradiction": probs[0][0].item(),
            "neutral": probs[0][1].item(),
            "entailment": probs[0][2].item()
        }
        
        return {
            "predicted_label": predicted_label,
            "predicted_class": self.label_map[predicted_label],
            "scores": scores,
            "entailment_prob": scores["entailment"]
        }
    
    def evaluate(self, predictions: List[str], references: List[str]) -> Dict:
        """
        Evaluate predictions against references using NLI.
        
        Args:
            predictions: List of generated text summaries
            references: List of ground truth text summaries
            
        Returns:
            Dictionary with NLI metrics
        """
        if len(predictions) != len(references):
            raise ValueError(
                f"Mismatch: {len(predictions)} predictions, "
                f"{len(references)} references"
            )
        
        entailment_count = 0
        contradiction_count = 0
        neutral_count = 0
        entailment_probs = []
        
        # Process in batches
        for i in range(0, len(predictions), self.batch_size):
            batch_preds = predictions[i:i+self.batch_size]
            batch_refs = references[i:i+self.batch_size]
            
            for pred, ref in zip(batch_preds, batch_refs):
                result = self.predict_entailment(ref, pred)
                
                if result["predicted_class"] == "entailment":
                    entailment_count += 1
                elif result["predicted_class"] == "contradiction":
                    contradiction_count += 1
                else:
                    neutral_count += 1
                
                entailment_probs.append(result["entailment_prob"])
        
        total = len(predictions)
        
        metrics = {
            "entailment_accuracy": entailment_count / total,
            "contradiction_rate": contradiction_count / total,
            "neutral_rate": neutral_count / total,
            "avg_entailment_prob": sum(entailment_probs) / len(entailment_probs),
            "total_samples": total
        }
        
        return metrics

