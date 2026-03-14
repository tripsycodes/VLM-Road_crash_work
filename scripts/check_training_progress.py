#!/usr/bin/env python3
"""
Script to check training progress by reading loss files and checkpoint information.
"""

import json
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def check_training_progress():
    """Check and display training progress."""
    results_dir = project_root / "results"
    training_loss_file = results_dir / "training_loss.json"
    validation_loss_file = results_dir / "validation_loss.json"
    checkpoints_dir = results_dir / "checkpoints"
    
    print("=" * 60)
    print("TRAINING PROGRESS MONITOR")
    print("=" * 60)
    
    # Check if loss files exist
    if training_loss_file.exists():
        print(f"\n✓ Training loss file found: {training_loss_file}")
        with open(training_loss_file, 'r') as f:
            training_data = json.load(f)
        
        if training_data:
            print(f"\nTraining Losses:")
            print("-" * 60)
            epochs = sorted([int(k.split('_')[1]) for k in training_data.keys()])
            for epoch in epochs:
                epoch_key = f"epoch_{epoch}"
                loss_info = training_data[epoch_key]
                loss = loss_info.get('loss', 'N/A')
                timestamp = loss_info.get('timestamp', 'N/A')
                print(f"  Epoch {epoch}: Loss = {loss:.4f} (at {timestamp})")
            print(f"\n  Total epochs completed: {len(epochs)}")
        else:
            print("  No training data recorded yet.")
    else:
        print(f"\n✗ Training loss file not found: {training_loss_file}")
        print("  Training may still be in progress or hasn't completed an epoch yet.")
    
    if validation_loss_file.exists():
        print(f"\n✓ Validation loss file found: {validation_loss_file}")
        with open(validation_loss_file, 'r') as f:
            validation_data = json.load(f)
        
        if validation_data:
            print(f"\nValidation Losses:")
            print("-" * 60)
            epochs = sorted([int(k.split('_')[1]) for k in validation_data.keys()])
            for epoch in epochs:
                epoch_key = f"epoch_{epoch}"
                loss_info = validation_data[epoch_key]
                val_loss = loss_info.get('val_loss', 'N/A')
                timestamp = loss_info.get('timestamp', 'N/A')
                print(f"  Epoch {epoch}: Val Loss = {val_loss:.4f} (at {timestamp})")
        else:
            print("  No validation data recorded yet.")
    else:
        print(f"\n✗ Validation loss file not found: {validation_loss_file}")
    
    # Check for checkpoints
    if checkpoints_dir.exists():
        checkpoints = list(checkpoints_dir.glob("*.pt"))
        if checkpoints:
            print(f"\n✓ Found {len(checkpoints)} checkpoint(s):")
            print("-" * 60)
            # Sort by modification time
            checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            for i, ckpt in enumerate(checkpoints[:10], 1):  # Show latest 10
                size_mb = ckpt.stat().st_size / (1024 * 1024)
                mtime = ckpt.stat().st_mtime
                from datetime import datetime
                mtime_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
                print(f"  {i}. {ckpt.name} ({size_mb:.1f} MB, {mtime_str})")
        else:
            print(f"\n✗ No checkpoints found in {checkpoints_dir}")
    else:
        print(f"\n✗ Checkpoints directory not found: {checkpoints_dir}")
    
    print("\n" + "=" * 60)
    print("To monitor in real-time, run: watch -n 5 python scripts/check_training_progress.py")
    print("=" * 60)

if __name__ == "__main__":
    check_training_progress()

