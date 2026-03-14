"""Temporal prompt generation for video frames."""
from typing import List


class TemporalPromptGenerator:
    """Generate temporal prompts for video frames."""
    
    def __init__(self, format_type: str = "zero_padded", max_frames: int = 30):
        """
        Initialize temporal prompt generator.
        
        Args:
            format_type: Format for temporal prompts ("zero_padded", "decimal", "seconds")
            max_frames: Maximum number of frames
        """
        self.format_type = format_type
        self.max_frames = max_frames
    
    def generate_prompt(self, frame_index: int, total_frames: int, 
                       fps: int = 30, frame_interval: int = 5) -> str:
        """
        Generate temporal prompt for a frame.
        
        Args:
            frame_index: Index of the frame in original video
            total_frames: Total number of frames in video
            fps: Frames per second
            frame_interval: Interval between sampled frames
            
        Returns:
            Temporal prompt string
        """
        if self.format_type == "zero_padded":
            # Format: "00", "05", "10", "15", etc.
            return f"{frame_index:02d}"
        elif self.format_type == "decimal":
            # Format: "0", "5", "10", "15", etc.
            return str(frame_index)
        elif self.format_type == "seconds":
            # Format based on time in seconds
            time_seconds = frame_index / fps
            return f"{time_seconds:.2f}"
        else:
            raise ValueError(f"Unknown format type: {self.format_type}")
    
    def generate_prompts(self, frame_indices: List[int], fps: int = 30) -> List[str]:
        """
        Generate temporal prompts for a list of frame indices.
        
        Args:
            frame_indices: List of frame indices
            fps: Frames per second
            
        Returns:
            List of temporal prompt strings
        """
        prompts = []
        for idx in frame_indices:
            prompt = self.generate_prompt(idx, len(frame_indices), fps)
            prompts.append(prompt)
        return prompts
    
    def interleave_with_tokens(self, visual_tokens: List, temporal_prompts: List[str]) -> List:
        """
        Interleave visual tokens with temporal prompts.
        
        Args:
            visual_tokens: List of visual token representations
            temporal_prompts: List of temporal prompt strings
            
        Returns:
            Interleaved sequence: [t1, v1, t2, v2, ...]
        """
        if len(visual_tokens) != len(temporal_prompts):
            raise ValueError(
                f"Mismatch: {len(visual_tokens)} visual tokens, "
                f"{len(temporal_prompts)} temporal prompts"
            )
        
        interleaved = []
        for token, prompt in zip(visual_tokens, temporal_prompts):
            interleaved.append(prompt)
            interleaved.append(token)
        
        return interleaved

