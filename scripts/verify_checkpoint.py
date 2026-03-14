#!/usr/bin/env python3
"""
Script to verify that a checkpoint can be loaded and contains valid weights.
"""

import sys
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import get_config
from src.models import LLaVANeXTWrapper

def verify_checkpoint(checkpoint_path: str):
    """Verify a checkpoint can be loaded."""
    print("=" * 60)
    print("CHECKPOINT VERIFICATION")
    print("=" * 60)
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return False
    
    print(f"\nLoading checkpoint: {checkpoint_path}")
    print(f"File size: {checkpoint_path.stat().st_size / (1024**3):.2f} GB")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print("\n✓ Checkpoint loaded successfully")
        print(f"\nCheckpoint contents:")
        print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  - Train Loss: {checkpoint.get('train_loss', 'N/A')}")
        print(f"  - Val Loss: {checkpoint.get('val_loss', 'N/A')}")
        
        # Check model state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"\n✓ Model state dict found")
            print(f"  - Number of parameters: {len(state_dict)}")
            
            # Check for NaN or Inf in weights
            nan_count = 0
            inf_count = 0
            total_params = 0
            
            for key, tensor in state_dict.items():
                if isinstance(tensor, torch.Tensor):
                    total_params += tensor.numel()
                    nan_count += torch.isnan(tensor).sum().item()
                    inf_count += torch.isinf(tensor).sum().item()
            
            print(f"\nWeight Statistics:")
            print(f"  - Total parameters: {total_params:,}")
            print(f"  - NaN values: {nan_count:,}")
            print(f"  - Inf values: {inf_count:,}")
            
            if nan_count > 0 or inf_count > 0:
                print(f"\n⚠ WARNING: Found {nan_count} NaN and {inf_count} Inf values in weights!")
                return False
            else:
                print(f"\n✓ All weights are valid (no NaN or Inf)")
        
        # Check optimizer state
        if 'optimizer_state_dict' in checkpoint:
            print(f"\n✓ Optimizer state dict found")
        
        print("\n" + "=" * 60)
        print("CHECKPOINT VERIFICATION: SUCCESS")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_loading(checkpoint_path: str):
    """Test if we can load the model from checkpoint."""
    print("\n" + "=" * 60)
    print("TESTING MODEL LOADING")
    print("=" * 60)
    
    try:
        # Load config
        config = get_config()
        model_name = config.get("model.vision_model") or config.config["model"]["vision_model"]
        
        print(f"\nLoading base model: {model_name}")
        wrapper = LLaVANeXTWrapper(
            model_name=model_name,
            device="cpu"  # Load on CPU for testing
        )
        
        print("✓ Base model loaded")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            print("\nLoading checkpoint weights into model...")
            wrapper.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("✓ Checkpoint weights loaded successfully")
        
        print("\n" + "=" * 60)
        print("MODEL LOADING TEST: SUCCESS")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR in model loading test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Verify checkpoint")
    parser.add_argument("--checkpoint", type=str, 
                       default="results/checkpoints/checkpoint_epoch_5.pt",
                       help="Path to checkpoint file")
    args = parser.parse_args()
    
    checkpoint_path = project_root / args.checkpoint
    
    success1 = verify_checkpoint(checkpoint_path)
    success2 = test_model_loading(checkpoint_path)
    
    if success1 and success2:
        print("\n✓ All checks passed!")
        sys.exit(0)
    else:
        print("\n✗ Some checks failed")
        sys.exit(1)

