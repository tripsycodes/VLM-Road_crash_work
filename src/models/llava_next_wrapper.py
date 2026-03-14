"""
Wrapper for LLaVA-NeXT and LLaVA-1.5 model integration.
Supports automatic fallback + correct APIs for both.
"""

import torch
from typing import List, Dict, Any
from PIL import Image
import numpy as np


class LLaVANeXTWrapper:
    def __init__(self, model_name: str = "llava-hf/llava-v1.6-mistral-7b-hf",
                 device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None
        self.is_next = False
        self._load_model()

    def _load_model(self):
        from transformers import (
            LlavaNextProcessor,
            LlavaNextForConditionalGeneration,
            LlavaProcessor,
            LlavaForConditionalGeneration
        )

        # Try to import BitsAndBytesConfig (may not be available in older transformers)
        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            BitsAndBytesConfig = None

        if self.device == "cuda" and torch.cuda.is_available():
            # Clear cache aggressively before loading
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            
            # Get the actual device (respects CUDA_VISIBLE_DEVICES)
            device_id = torch.cuda.current_device()
            device_str = f"cuda:{device_id}"
            
            # Get available memory
            free_memory = None
            if torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(device_id).total_memory / 1e9
                allocated = torch.cuda.memory_allocated(device_id) / 1e9
                reserved = torch.cuda.memory_reserved(device_id) / 1e9
                free_memory = total_memory - reserved
                print(f"GPU {device_id} - Total: {total_memory:.2f}GB, Reserved: {reserved:.2f}GB, Free: {free_memory:.2f}GB")
            
            model_dtype = torch.float16
            use_device_map = True
            
            # Use 8-bit quantization for memory efficiency
            # With proper settings, 8-bit should be stable for training
            quantization_config = None
            use_quantization = False
            try:
                import bitsandbytes as bnb
                if BitsAndBytesConfig is None:
                    raise ImportError("BitsAndBytesConfig not available in transformers")
                use_quantization = True
                # Use 8-bit with optimizations for training stability
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,  # Lower threshold for better stability
                    llm_int8_has_fp16_weight=False  # Keep weights in 8-bit
                )
                print("Using 8-bit quantization with stability optimizations")
            except ImportError:
                print("bitsandbytes not available, using FP16")
            except Exception as e:
                print(f"Quantization setup failed: {e}, using FP16")
                use_quantization = False
        else:
            device_str = "cpu"
            model_dtype = torch.float32
            use_device_map = False
            use_quantization = False
            quantization_config = None
            free_memory = None
            device_id = None

        def _load_with_config(model_class, processor_class, model_name, is_next_model=True):
            """Helper to load model with proper configuration."""
            processor = processor_class.from_pretrained(model_name)

            load_kwargs = {
                "torch_dtype": model_dtype,
                "low_cpu_mem_usage": True
            }

            if use_device_map:
                if use_quantization and quantization_config is not None:
                    load_kwargs["quantization_config"] = quantization_config
                    # Remove torch_dtype when using quantization (handled by config)
                    load_kwargs.pop("torch_dtype", None)
                    # For training, we need all parts on the same GPU (no CPU offloading)
                    # Use device_map with specific device to ensure everything stays on GPU
                    load_kwargs["device_map"] = device_str
                    quant_type = "4-bit" if quantization_config.load_in_4bit else "8-bit"
                    print(f"Loading model to {device_str} with {quant_type} quantization...")
                else:
                    load_kwargs["device_map"] = device_str
                    print(f"Loading model directly to {device_str}...")

            model = model_class.from_pretrained(model_name, **load_kwargs)

            # If device_map was used, model is already on device
            # Otherwise, move to device
            if not use_device_map and self.device != "cpu":
                print(f"Moving model to {device_str}...")
                model = model.to(device_str)
            
            return processor, model

        try:
            print("Trying to load LLaVA-NeXT...")
            self.processor, self.model = _load_with_config(
                LlavaNextForConditionalGeneration,
                LlavaNextProcessor,
                self.model_name,
                is_next_model=True
            )

            self.device = device_str
            self.is_next = True
            print(f"Loaded LLaVA-NeXT model: {self.model_name} on {self.device}")
            
            # Clear cache after loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Failed to load LLaVA-NeXT → Falling back to LLaVA-1.5\n{str(e)[:500]}")
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            torch.cuda.empty_cache()

            fallback_model_name = "llava-hf/llava-1.5-7b-hf"

            try:
                self.processor, self.model = _load_with_config(
                    LlavaForConditionalGeneration,
                    LlavaProcessor,
                    fallback_model_name,
                    is_next_model=False
                )
                
                self.device = device_str
                self.is_next = False
                self.model_name = fallback_model_name
                print(f"Loaded LLaVA-1.5 model: {self.model_name} on {self.device}")
                
                # Clear cache after loading
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e2:
                print(f"Failed to load fallback model: {str(e2)[:500]}")
                raise RuntimeError(f"Could not load either LLaVA-NeXT or LLaVA-1.5. Original error: {str(e)[:500]}")

    # ------------------------------------------------------------------
    def encode_frames(self, frames: List[np.ndarray]) -> Dict[str, torch.Tensor]:
        images = [Image.fromarray(f).convert("RGB") for f in frames]
        inputs = self.processor(images=images, return_tensors="pt")
        return {k: v.to(self.device) for k, v in inputs.items()}

    # ------------------------------------------------------------------
    def generate_caption(self, frames: List[np.ndarray],
                         prompt: str) -> str:

        if len(frames) == 0:
            raise ValueError("No frames provided")

        # Use middle frame (for now)
        rep = frames[len(frames) // 2]
        rep_img = Image.fromarray(rep).convert("RGB")

        formatted_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"

        if self.is_next:
            inputs = self.processor(
                text=formatted_prompt,
                images=rep_img,
                return_tensors="pt"
            )
        else:
            inputs = self.processor(
                images=rep_img,
                text=formatted_prompt,
                return_tensors="pt"
            )

        inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        if self.is_next and "image_sizes" not in inputs:
            w, h = rep_img.size
            inputs["image_sizes"] = torch.tensor([[h, w]], device=self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False
            )

        text = self.processor.decode(outputs[0], skip_special_tokens=True)
        if "ASSISTANT:" in text:
            text = text.split("ASSISTANT:")[-1].strip()

        return text

    # ------------------------------------------------------------------
    def generate_summary(self, frames: List[np.ndarray],
                         task_type: str = "v2t") -> Dict[str, Any]:

        if task_type == "v2t":
            summary_prompt = (
                "You are an expert traffic accident analyst. "
                "Watch this car crash video and generate a factual explanation. "
                "Describe clearly:\n"
                "- vehicles involved\n"
                "- number of vehicles\n"
                "- what each vehicle was doing before the crash\n"
                "- how the crash happened\n"
                "- where the impact occurred\n"
                "- outcome of the crash\n"
                "Write 4-6 coherent sentences like an incident report. "
                "Do NOT hallucinate unseen details."
            )

            summary = self.generate_caption(frames, summary_prompt)
            return {"text_summary": summary}

        elif task_type == "v2v":
            resp = self.generate_caption(
                frames,
                "Identify the key frames useful for summarization. List only frame numbers."
            )
            idx = self._parse_frame_indices(resp)
            return {"video_summary": idx, "response": resp}

        elif task_type == "v2vt":
            resp = self.generate_caption(
                frames,
                "Give both key frames and a textual summary of this car crash video."
            )
            idx = self._parse_frame_indices(resp)
            txt = self._extract_text_summary(resp)
            return {"video_summary": idx, "text_summary": txt, "response": resp}

        else:
            raise ValueError(f"Unknown task: {task_type}")

    # ------------------------------------------------------------------
    def _parse_frame_indices(self, text: str):
        import re
        nums = re.findall(r"\d+", text)
        return [int(n) for n in nums if int(n) < 2000]

    def _extract_text_summary(self, text: str):
        lines = text.split("\n")
        res = []
        for l in lines:
            if not any(c.isdigit() for c in l[:8]):
                res.append(l)
        return " ".join(res).strip()
