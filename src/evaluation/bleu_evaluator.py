"""BLEU score evaluation."""
from typing import List, Dict
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sacrebleu import BLEU as SacreBLEU


class BLEUEvaluator:
    """Evaluate text summaries using BLEU scores."""
    
    def __init__(self, max_order: int = 4, smooth: bool = True):
        """
        Initialize BLEU evaluator.
        
        Args:
            max_order: Maximum n-gram order (BLEU-1 to BLEU-N)
            smooth: Whether to use smoothing
        """
        self.max_order = max_order
        self.smooth = smooth
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        return nltk.word_tokenize(text.lower())
    
    def compute_bleu(self, prediction: str, reference: str) -> Dict[str, float]:
        """
        Compute BLEU scores for a single prediction-reference pair.
        
        Args:
            prediction: Generated text summary
            reference: Ground truth text summary
            
        Returns:
            Dictionary with BLEU-1 to BLEU-N scores
        """
        pred_tokens = self.tokenize(prediction)
        ref_tokens = self.tokenize(reference)
        
        scores = {}
        
        if self.smooth:
            smoothing = SmoothingFunction().method1
        else:
            smoothing = None
        
        for n in range(1, self.max_order + 1):
            try:
                score = sentence_bleu(
                    [ref_tokens],
                    pred_tokens,
                    weights=[1.0/n] * n,
                    smoothing_function=smoothing
                )
                scores[f"bleu_{n}"] = score
            except:
                scores[f"bleu_{n}"] = 0.0
        
        # Also compute corpus-level BLEU using sacrebleu
        try:
            sacre_bleu = SacreBLEU()
            corpus_score = sacre_bleu.corpus_score([prediction], [[reference]]).score / 100.0
            scores["bleu_corpus"] = corpus_score
        except:
            scores["bleu_corpus"] = 0.0
        
        return scores
    
    def compute_bleu_batch(self, predictions: List[str], 
                          references: List[str]) -> Dict[str, float]:
        """
        Compute BLEU scores for a batch of predictions.
        
        Args:
            predictions: List of generated text summaries
            references: List of ground truth text summaries
            
        Returns:
            Dictionary with average BLEU scores
        """
        if len(predictions) != len(references):
            raise ValueError(
                f"Mismatch: {len(predictions)} predictions, "
                f"{len(references)} references"
            )
        
        all_scores = {f"bleu_{n}": [] for n in range(1, self.max_order + 1)}
        all_scores["bleu_corpus"] = []
        
        for pred, ref in zip(predictions, references):
            scores = self.compute_bleu(pred, ref)
            for key, value in scores.items():
                all_scores[key].append(value)
        
        # Compute averages
        avg_scores = {
            key: sum(values) / len(values) 
            for key, values in all_scores.items()
        }
        
        return avg_scores

