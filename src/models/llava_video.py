from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .base import GenerationConfig, ModelAdapter


class LlavaVideoAdapter(ModelAdapter):
    """Base adapter for the official LLaVA-Video Qwen2 checkpoint.

    LLaVA-Video is not the same model family as LLaVA-OneVision or the older
    Video-LLaVA repository.  Its official inference path is the LLaVA-NeXT
    ``llava_qwen`` loader with video tensors passed as ``images=[video]`` and
    ``modalities=["video"]``.
    """

    name = "llava-video-7b"

    def __init__(self, checkpoint: str, local_path: str | None = None):
        self.checkpoint = checkpoint
        self.local_path = local_path
        self.model_path = self._resolve_model_path(local_path, checkpoint)

        try:
            import torch
            from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
            from llava.conversation import conv_templates
            from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
            from llava.model.builder import load_pretrained_model
        except ImportError as exc:
            raise RuntimeError(
                "LLaVA-Video requires the official LLaVA-NeXT Python stack "
                "(`llava.model.builder`, `llava_qwen`). Install it or add its "
                "repository to PYTHONPATH before running this adapter."
            ) from exc

        self.torch = torch
        self.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self.conv_templates = conv_templates
        self.tokenizer_image_token = tokenizer_image_token
        self.loader_stack = "LLaVA-NeXT/llava_qwen"
        self._generation_diagnostics: list[dict] = []

        model_name = get_model_name_from_path(self.model_path)
        if "llava" not in str(model_name).lower() or "qwen" not in str(model_name).lower():
            model_name = "llava_qwen"
        self.model_name = model_name
        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(
            self.model_path,
            None,
            model_name=self.model_name,
            # The official builder accepts only float16/bfloat16 strings.
            # BFloat16 is the documented LLaVA-Video inference dtype.
            torch_dtype="bfloat16",
            device_map="auto",
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.model_dtype = next(self.model.parameters()).dtype

    def _resolve_model_path(self, local_path: str | None, checkpoint: str) -> str:
        for candidate in (local_path, os.environ.get("LLAVA_VIDEO_MODEL_DIR"), os.environ.get("MODEL_DIR")):
            if candidate:
                path = Path(candidate).expanduser()
                if path.exists():
                    return str(path)
        return checkpoint

    def _is_local_only(self) -> bool:
        return Path(self.model_path).exists()

    def _build_prompt(self, prompt: str) -> str:
        template = self.conv_templates.get("qwen_1_5")
        if template is None:
            raise RuntimeError("The installed LLaVA-NeXT stack does not provide conv_template qwen_1_5")
        conversation = template.copy()
        conversation.append_message(conversation.roles[0], self.DEFAULT_IMAGE_TOKEN + "\n" + prompt)
        conversation.append_message(conversation.roles[1], None)
        return conversation.get_prompt()

    def _build_inputs(self, video_frames: np.ndarray, prompt: str) -> tuple[dict, int]:
        video = np.asarray(video_frames, dtype=np.uint8)
        if video.ndim != 4 or video.shape[-1] != 3 or len(video) == 0:
            raise ValueError("video_frames must have shape [frames, height, width, 3]")

        processed = self.image_processor.preprocess(video, return_tensors="pt")["pixel_values"]
        processed = processed.to(device=self.device, dtype=self.model_dtype)
        rendered_prompt = self._build_prompt(prompt)
        input_ids = self.tokenizer_image_token(
            rendered_prompt,
            self.tokenizer,
            self.IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).to(self.device)
        attention_mask = self.torch.ones_like(input_ids, device=self.device)
        self._last_input_audit = {
            "llava_video_checkpoint": self.checkpoint,
            "llava_video_resolved_model_path": self.model_path,
            "llava_video_loader_stack": self.loader_stack,
            "llava_video_model_name": self.model_name,
            "rendered_prompt": rendered_prompt,
            "model_input_keys": ["input_ids", "attention_mask", "images", "modalities"],
            "model_input_shapes": {"input_ids": list(input_ids.shape), "video": list(processed.shape)},
            "vision_tensor_supplied": True,
            "video_modality_supplied": True,
            "video_modality": "video",
            "video_frame_count": int(len(video)),
        }
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "images": [processed],
            "modalities": ["video"],
        }, int(input_ids.shape[1])

    def generate(self, video_frames: np.ndarray, prompt: str, generation_config: GenerationConfig) -> str:
        inputs, prompt_length = self._build_inputs(video_frames, prompt)
        self._generation_diagnostics = [dict(self._last_input_audit)]
        with self.torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=generation_config.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                num_beams=1,
                use_cache=True,
            )
        generated_ids = output_ids[:, prompt_length:]
        answer = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return answer.strip()

    def consume_generation_diagnostics(self, expected_count: int) -> list[dict]:
        diagnostics = self._generation_diagnostics[:expected_count]
        self._generation_diagnostics = self._generation_diagnostics[expected_count:]
        return diagnostics + [{} for _ in range(expected_count - len(diagnostics))]
