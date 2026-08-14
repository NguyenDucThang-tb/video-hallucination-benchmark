from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .base import GenerationConfig, ModelAdapter


@dataclass(frozen=True)
class Qwen25VLConfig:
    checkpoint: str
    local_path: str | None = None
    min_pixels: int | None = None
    max_pixels: int | None = None


class Qwen25VLAdapter(ModelAdapter):
    name = "qwen2.5-vl-7b"

    def __init__(self, checkpoint: str, local_path: str | None = None):
        self.checkpoint = checkpoint
        self.local_path = local_path
        self.model_path = self._resolve_model_path(local_path, checkpoint)

        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=self._is_local_only())
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=self._torch_dtype(),
            device_map="auto",
            local_files_only=self._is_local_only(),
        ).eval()
        self.device = next(self.model.parameters()).device

    def _resolve_model_path(self, local_path: str | None, checkpoint: str) -> str:
        for candidate in (
            local_path,
            os.environ.get("QWEN25_VL_MODEL_DIR"),
            os.environ.get("MODEL_DIR"),
        ):
            if candidate:
                path = Path(candidate).expanduser()
                if path.exists():
                    return str(path)
        return checkpoint

    def _is_local_only(self) -> bool:
        return Path(self.model_path).exists()

    def _torch_dtype(self):
        if self.torch.cuda.is_available():
            if hasattr(self.torch.cuda, "is_bf16_supported") and self.torch.cuda.is_bf16_supported():
                return self.torch.bfloat16
            return self.torch.float16
        return self.torch.float32

    def _frames_to_messages(self, video_frames: np.ndarray, prompt: str) -> list[dict]:
        from PIL import Image

        content = []
        for frame in video_frames:
            content.append({"type": "image", "image": Image.fromarray(np.asarray(frame, dtype=np.uint8))})
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def generate(self, video_frames: np.ndarray, prompt: str, generation_config: GenerationConfig) -> str:
        messages = self._frames_to_messages(video_frames, prompt)
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        images = [item["image"] for item in messages[0]["content"] if item["type"] == "image"]
        inputs = self.processor(
            text=[text],
            images=images,
            return_tensors="pt",
            padding=True,
        )
        inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

        with self.torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=generation_config.max_new_tokens,
                do_sample=False,
                temperature=None,
                num_beams=1,
                use_cache=True,
            )

        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        answer = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return answer.strip()

    def generate_batch(
        self,
        batch_video_frames: list[np.ndarray],
        prompts: list[str],
        generation_config: GenerationConfig,
    ) -> list[str]:
        if len(batch_video_frames) != len(prompts):
            raise ValueError("batch_video_frames and prompts must have the same length")
        if not batch_video_frames:
            return []

        batch_messages = [self._frames_to_messages(video_frames, prompt) for video_frames, prompt in zip(batch_video_frames, prompts)]
        texts = [
            self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            for messages in batch_messages
        ]
        images = [
            [item["image"] for item in messages[0]["content"] if item["type"] == "image"]
            for messages in batch_messages
        ]
        inputs = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )
        inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()

        with self.torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=generation_config.max_new_tokens,
                do_sample=False,
                temperature=None,
                num_beams=1,
                use_cache=True,
            )

        answers = []
        for row_index, prompt_length in enumerate(prompt_lengths):
            generated_ids = output_ids[row_index, int(prompt_length):]
            answer = self.processor.batch_decode(
                generated_ids.unsqueeze(0),
                skip_special_tokens=True,
            )[0]
            answers.append(answer.strip())
        return answers
