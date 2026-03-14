"""Parse ground truth annotations from Excel file."""
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List
import re


class GroundTruthParser:
    """Parse ground truth text summaries from Excel file."""
    
    def __init__(self, excel_path: str):
        self.excel_path = Path(excel_path)
        if not self.excel_path.exists():
            raise FileNotFoundError(f"Excel file not found: {excel_path}")
        
        self.df = None
        self.annotations = {}
    
    def load_excel(self) -> pd.DataFrame:
        """Load Excel file into DataFrame."""
        try:
            self.df = pd.read_excel(self.excel_path)
            return self.df
        except Exception as e:
            raise ValueError(f"Error loading Excel file: {e}")
    
    def extract_video_id(self, filename: str) -> str:
        """Extract normalized numeric video ID."""
        video_id = Path(filename).stem
        match = re.search(r"(\d+)", video_id)
        if match:
            return match.group(1).zfill(6)
        return video_id
    
    def map_videos_to_annotations(self, video_files: List[str]) -> Dict[str, Dict]:
        """Map video files to their ground truth annotations."""
        if self.df is None:
            self.load_excel()
        
        columns = self.df.columns.tolist()
        print(f"Excel columns: {columns}")

        # ---------- Preferred Columns ----------
        preferred_id_cols = ["Video Number", "Number", "video_id", "Video"]
        preferred_text_cols = ["Explanation", "explanation"]

        video_id_col = None
        text_col = None

        # If preferred id exists, force it
        for c in columns:
            if c in preferred_id_cols:
                video_id_col = c

        # If preferred Explanation exists, force it
        for c in columns:
            if c in preferred_text_cols:
                text_col = c

        # ---------- If not found, fall back to heuristics ----------
        id_patterns = ["id", "video", "video_id", "video_name", "filename", "file"]
        text_patterns = ["text", "summary", "caption", "description", "annotation", "ground_truth"]

        if video_id_col is None or text_col is None:
            for col in columns:
                col_lower = col.lower()
                if video_id_col is None and any(p in col_lower for p in id_patterns):
                    video_id_col = col
                if text_col is None and any(p in col_lower for p in text_patterns):
                    text_col = col

        # ---------- Final Safety Fallback ----------
        if video_id_col is None:
            video_id_col = columns[0]
        if text_col is None:
            # If Explanation wasn't explicitly found but still exists
            if "Explanation" in columns:
                text_col = "Explanation"
            elif len(columns) > 1:
                text_col = columns[1]
            else:
                text_col = columns[0]

        print(f"Using video ID column: {video_id_col}")
        print(f"Using text column (ground truth summary): {text_col}")

        # ---------- Build Annotation Dictionary ----------
        annotations = {}
        for _, row in self.df.iterrows():
            raw_id = str(row[video_id_col]).strip()
            video_id = self.extract_video_id(raw_id)
            text = str(row[text_col]).strip()

            annotations[video_id] = {
                "video_id": video_id,
                "text_summary": text,
                "original_id": raw_id
            }

        # ---------- Map Only Existing Videos ----------
        video_annotations = {}
        for video_file in video_files:
            vid = self.extract_video_id(video_file)

            if vid in annotations:
                video_annotations[vid] = annotations[vid]
            else:
                vid_no_pad = vid.lstrip("0") or "0"
                if vid_no_pad in annotations:
                    video_annotations[vid] = annotations[vid_no_pad]
                else:
                    print(f"Warning: No annotation found for video {video_file} (ID: {vid})")
                    video_annotations[vid] = {
                        "video_id": vid,
                        "text_summary": "",
                        "original_id": video_file
                    }

        self.annotations = video_annotations
        return video_annotations
    
    def save_annotations(self, output_path: str, format: str = "json"):
        """Save annotations to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            with open(output_path, "w") as f:
                json.dump(self.annotations, f, indent=2, ensure_ascii=False)
        elif format == "jsonl":
            with open(output_path, "w") as f:
                for vid, ann in self.annotations.items():
                    f.write(json.dumps({vid: ann}, ensure_ascii=False) + "\n")
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def get_statistics(self) -> Dict:
        """Dataset statistics."""
        if not self.annotations:
            return {}

        lengths = [len(a["text_summary"].split()) for a in self.annotations.values()]
        return {
            "total_videos": len(self.annotations),
            "avg_summary_length": sum(lengths) / len(lengths) if lengths else 0,
            "min_summary_length": min(lengths) if lengths else 0,
            "max_summary_length": max(lengths) if lengths else 0,
            "videos_with_annotations": sum(1 for a in self.annotations.values() if a["text_summary"].strip())
        }
