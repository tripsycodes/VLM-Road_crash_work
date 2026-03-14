#!/usr/bin/env python3
"""Script to process videos and prepare dataset."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_processing import VideoProcessor, GroundTruthParser, DatasetSplitter
from src.utils import get_config


def main():
    """Main data processing pipeline."""
    config = get_config()
    
    # Get paths from config
    root_dir = Path(config.get('dataset.root_dir'))
    videos_dir = root_dir / config.get('dataset.videos_dir')
    ground_truth_file = root_dir / config.get('dataset.ground_truth_file')
    processed_dir = root_dir / config.get('dataset.processed_dir')
    
    print("=" * 60)
    print("Data Processing Pipeline")
    print("=" * 60)
    
    # Step 1: Get all video files
    print("\n[Step 1] Collecting video files...")
    video_files = list(videos_dir.glob("*.mp4"))
    print(f"Found {len(video_files)} video files")
    
    # Step 2: Process videos (extract frames)
    print("\n[Step 2] Processing videos (extracting frames)...")
    processor = VideoProcessor(
        segment_duration=config.get('dataset.segment_duration', 5),
        frame_interval=config.get('dataset.frame_interval', 5),
        fps=config.get('dataset.fps', 30)
    )
    
    frames_output_dir = processed_dir / "frames"
    results = processor.process_video_batch(
        [str(v) for v in video_files],
        str(frames_output_dir),
        save_frames=True
    )
    
    print(f"Processed: {len(results['processed'])} videos")
    print(f"Failed: {len(results['failed'])} videos")
    
    # Step 3: Parse ground truth
    print("\n[Step 3] Parsing ground truth annotations...")
    parser = GroundTruthParser(str(ground_truth_file))
    annotations = parser.map_videos_to_annotations([str(v) for v in video_files])
    
    stats = parser.get_statistics()
    print(f"Total annotations: {stats['total_videos']}")
    print(f"Average summary length: {stats['avg_summary_length']:.1f} words")
    
    # Save annotations
    annotations_file = processed_dir / "annotations.json"
    parser.save_annotations(str(annotations_file))
    print(f"Saved annotations to: {annotations_file}")
    
    # Step 4: Split dataset
    print("\n[Step 4] Splitting dataset...")
    splitter = DatasetSplitter(
        train_ratio=config.get('dataset.train_ratio', 0.70),
        val_ratio=config.get('dataset.val_ratio', 0.15),
        test_ratio=config.get('dataset.test_ratio', 0.15),
        random_seed=config.get('dataset.random_seed', 42)
    )
    
    splits = splitter.split_videos([str(v) for v in video_files])
    print(f"Train: {len(splits['train'])} videos")
    print(f"Val: {len(splits['val'])} videos")
    print(f"Test: {len(splits['test'])} videos")
    
    # Create split directories
    split_dirs = splitter.create_split_directories(
        str(processed_dir / "splits"),
        splits,
        copy_files=False  # Don't copy, just create structure
    )
    
    # Create annotation splits
    annotation_splits = splitter.create_annotation_splits(annotations, splits)
    
    # Save split info
    split_info_file = processed_dir / "split_info.json"
    splitter.save_split_info(str(split_info_file), splits, annotations)
    print(f"Saved split info to: {split_info_file}")
    
    # Save annotation splits
    for split_name, split_anns in annotation_splits.items():
        ann_file = processed_dir / f"annotations_{split_name}.json"
        parser.save_annotations(str(ann_file))
        # Update with split-specific annotations
        with open(ann_file, 'w') as f:
            import json
            json.dump(split_anns, f, indent=2, ensure_ascii=False)
        print(f"Saved {split_name} annotations to: {ann_file}")
    
    print("\n" + "=" * 60)
    print("Data processing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

