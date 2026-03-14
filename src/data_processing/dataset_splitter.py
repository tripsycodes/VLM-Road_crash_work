"""Dataset splitting utilities."""
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
import shutil


class DatasetSplitter:
    """Split dataset into train/validation/test sets."""
    
    def __init__(self, train_ratio: float = 0.70, val_ratio: float = 0.15, 
                 test_ratio: float = 0.15, random_seed: int = 42):
        """
        Initialize dataset splitter.
        
        Args:
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            test_ratio: Proportion of data for testing
            random_seed: Random seed for reproducibility
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        
        random.seed(random_seed)
    
    def split_videos(self, video_files: List[str]) -> Dict[str, List[str]]:
        """
        Split video files into train/val/test sets.
        
        Args:
            video_files: List of video file paths
            
        Returns:
            Dictionary with 'train', 'val', 'test' keys
        """
        # Shuffle videos
        shuffled = video_files.copy()
        random.shuffle(shuffled)
        
        total = len(shuffled)
        train_end = int(total * self.train_ratio)
        val_end = train_end + int(total * self.val_ratio)
        
        splits = {
            'train': shuffled[:train_end],
            'val': shuffled[train_end:val_end],
            'test': shuffled[val_end:]
        }
        
        return splits
    
    def create_split_directories(self, base_dir: str, splits: Dict[str, List[str]], 
                                copy_files: bool = False) -> Dict[str, Path]:
        """
        Create directory structure for dataset splits.
        
        Args:
            base_dir: Base directory for dataset
            splits: Dictionary with train/val/test file lists
            copy_files: Whether to copy files to split directories
            
        Returns:
            Dictionary mapping split names to directory paths
        """
        base_dir = Path(base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        
        split_dirs = {}
        for split_name, files in splits.items():
            split_dir = base_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            split_dirs[split_name] = split_dir
            
            if copy_files:
                videos_dir = split_dir / "videos"
                videos_dir.mkdir(exist_ok=True)
                
                for file_path in files:
                    src = Path(file_path)
                    if src.exists():
                        dst = videos_dir / src.name
                        shutil.copy2(src, dst)
        
        return split_dirs
    
    def save_split_info(self, output_path: str, splits: Dict[str, List[str]], 
                       annotations: Dict = None):
        """
        Save split information to JSON file.
        
        Args:
            output_path: Path to save split info
            splits: Dictionary with train/val/test file lists
            annotations: Optional annotations dictionary
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        split_info = {
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio,
            'random_seed': self.random_seed,
            'splits': {
                'train': [str(f) for f in splits['train']],
                'val': [str(f) for f in splits['val']],
                'test': [str(f) for f in splits['test']]
            },
            'counts': {
                'train': len(splits['train']),
                'val': len(splits['val']),
                'test': len(splits['test'])
            }
        }
        
        if annotations:
            split_info['annotations'] = annotations
        
        with open(output_path, 'w') as f:
            json.dump(split_info, f, indent=2)
    
    def create_annotation_splits(self, annotations: Dict, splits: Dict[str, List[str]]) -> Dict[str, Dict]:
        """
        Create annotation dictionaries for each split.
        
        Args:
            annotations: Full annotations dictionary
            splits: Dictionary with train/val/test file lists
            
        Returns:
            Dictionary with annotations for each split
        """
        import re
        from pathlib import Path
        
        def extract_video_id(filename: str) -> str:
            """Extract video ID from filename."""
            video_id = Path(filename).stem
            match = re.search(r'(\d+)', video_id)
            if match:
                return match.group(1).zfill(6)
            return video_id
        
        split_annotations = {
            'train': {},
            'val': {},
            'test': {}
        }
        
        for split_name, files in splits.items():
            for file_path in files:
                video_id = extract_video_id(file_path)
                if video_id in annotations:
                    split_annotations[split_name][video_id] = annotations[video_id]
        
        return split_annotations

