"""Loss tracking utilities."""
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


class LossTracker:
    """Track and save training/validation losses."""
    
    def __init__(self, loss_file: str = "results/training_loss.json",
                 val_loss_file: str = "results/validation_loss.json"):
        """
        Initialize loss tracker.
        
        Args:
            loss_file: Path to save training loss
            val_loss_file: Path to save validation loss
        """
        self.loss_file = Path(loss_file)
        self.val_loss_file = Path(val_loss_file)
        
        # Create directories
        self.loss_file.parent.mkdir(parents=True, exist_ok=True)
        self.val_loss_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize loss history
        self.training_losses = self._load_json(self.loss_file) or {}
        self.validation_losses = self._load_json(self.val_loss_file) or {}
    
    def _load_json(self, file_path: Path) -> Optional[Dict]:
        """Load JSON file if it exists."""
        if file_path.exists():
            with open(file_path, 'r') as f:
                return json.load(f)
        return None
    
    def _save_json(self, data: Dict, file_path: Path):
        """Save data to JSON file."""
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def log_training_loss(self, epoch: int, loss: float, 
                         metrics: Optional[Dict] = None):
        """
        Log training loss for an epoch.
        
        Args:
            epoch: Epoch number
            loss: Training loss value
            metrics: Optional additional metrics (BLEU, NLI, etc.)
        """
        epoch_key = f"epoch_{epoch}"
        
        # Handle NaN and Inf values
        loss_val = float(loss)
        if not (isinstance(loss_val, float) and (loss_val != loss_val or not (loss_val < float('inf')))):
            # Check if loss is NaN or Inf
            import math
            if math.isnan(loss_val) or math.isinf(loss_val):
                print(f"WARNING: Loss for epoch {epoch} is {loss_val}, logging as None")
                loss_val = None
        
        entry = {
            "loss": loss_val,
            "timestamp": datetime.now().isoformat()
        }
        
        if metrics:
            entry.update({k: (float(v) if not (isinstance(v, float) and (v != v or not (v < float('inf')))) else None) 
                         for k, v in metrics.items()})
        
        self.training_losses[epoch_key] = entry
        self._save_json(self.training_losses, self.loss_file)
    
    def log_validation_loss(self, epoch: int, val_loss: float,
                           metrics: Optional[Dict] = None):
        """
        Log validation loss for an epoch.
        
        Args:
            epoch: Epoch number
            val_loss: Validation loss value
            metrics: Optional additional metrics
        """
        epoch_key = f"epoch_{epoch}"
        
        # Handle NaN and Inf values
        val_loss_val = float(val_loss)
        if not (isinstance(val_loss_val, float) and (val_loss_val != val_loss_val or not (val_loss_val < float('inf')))):
            # Check if loss is NaN or Inf
            import math
            if math.isnan(val_loss_val) or math.isinf(val_loss_val):
                print(f"WARNING: Validation loss for epoch {epoch} is {val_loss_val}, logging as None")
                val_loss_val = None
        
        entry = {
            "val_loss": val_loss_val,
            "timestamp": datetime.now().isoformat()
        }
        
        if metrics:
            entry.update({f"val_{k}": (float(v) if not (isinstance(v, float) and (v != v or not (v < float('inf')))) else None) 
                         for k, v in metrics.items()})
        
        self.validation_losses[epoch_key] = entry
        self._save_json(self.validation_losses, self.val_loss_file)
    
    def get_best_epoch(self, metric: str = "val_loss", 
                      higher_is_better: bool = False) -> int:
        """
        Get epoch with best metric value.
        
        Args:
            metric: Metric name to check
            higher_is_better: Whether higher values are better
            
        Returns:
            Best epoch number
        """
        best_value = None
        best_epoch = 0
        
        for epoch_key, entry in self.validation_losses.items():
            if metric in entry:
                value = entry[metric]
                
                if best_value is None:
                    best_value = value
                    best_epoch = int(epoch_key.split('_')[1])
                elif (higher_is_better and value > best_value) or \
                     (not higher_is_better and value < best_value):
                    best_value = value
                    best_epoch = int(epoch_key.split('_')[1])
        
        return best_epoch
    
    def get_loss_history(self) -> Dict:
        """Get complete loss history."""
        return {
            "training": self.training_losses,
            "validation": self.validation_losses
        }

