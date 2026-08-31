from __future__ import annotations

import inspect
import os
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.methods.dino_heal.fusion import DINOHealConfig, fuse_saliency
from src.methods.positive_feature.enhancement import (
    PositiveFeatureConfig,
    compute_birefnet_foreground,
    enhance_visual_embeddings,
    ensure_birefnet_loaded,
)
from src.methods.season.attention_diagnosis import frame_attention

from .base import GenerationConfig, ModelAdapter, StepOutput, select_decode_input_ids


def cached_mrope_position_ids(attention_mask, rope_deltas):
    """Build one-token Qwen mRoPE positions without expanding the full prefix."""
    if hasattr(attention_mask, "long"):
        text_position = attention_mask.long().cumsum(dim=-1)[:, -1:] - 1
        text_position = text_position.masked_fill(attention_mask[:, -1:] == 0, 0)
        position_ids = text_position.unsqueeze(0).expand(3, -1, -1)
        deltas = rope_deltas.to(device=position_ids.device, dtype=position_ids.dtype)
        if deltas.ndim == 1:
            deltas = deltas.unsqueeze(-1)
        return position_ids + deltas

    mask = np.asarray(attention_mask)
    text_position = np.cumsum(mask, axis=-1)[:, -1:] - 1
    text_position = np.where(mask[:, -1:] == 0, 0, text_position)
    position_ids = np.repeat(text_position[None, ...], 3, axis=0)
    deltas = np.asarray(rope_deltas)
    if deltas.ndim == 1:
        deltas = deltas[:, None]
    return position_ids + deltas


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
        self._configure_padding()
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=self._torch_dtype(),
            device_map="auto",
            local_files_only=self._is_local_only(),
        ).eval()
        self.device = next(self.model.parameters()).device
        self._dino_processor = None
        self._dino_model = None
        self._generation_diagnostics: list[dict] = []

    def _configure_padding(self) -> None:
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            return
        tokenizer.padding_side = "left"
        if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        if hasattr(self.processor, "padding_side"):
            self.processor.padding_side = "left"

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

    def _checkpoint_is_local(self, checkpoint: str) -> bool:
        return Path(checkpoint).expanduser().exists()

    def _torch_dtype(self):
        if self.torch.cuda.is_available():
            if hasattr(self.torch.cuda, "is_bf16_supported") and self.torch.cuda.is_bf16_supported():
                return self.torch.bfloat16
            return self.torch.float16
        return self.torch.float32

    def _attention_configs(self):
        configs = []
        for candidate in (
            getattr(self.model, "config", None),
            getattr(getattr(self.model, "model", None), "config", None),
            getattr(getattr(self.model, "language_model", None), "config", None),
        ):
            if candidate is not None and candidate not in configs:
                configs.append(candidate)
        return configs

    @contextmanager
    def _temporary_attention_implementation(self, implementation: str):
        configs = self._attention_configs()
        originals = [getattr(config, "_attn_implementation", None) for config in configs]
        setter = getattr(self.model, "set_attn_implementation", None)
        try:
            if callable(setter):
                setter(implementation)
            else:
                for config in configs:
                    setattr(config, "_attn_implementation", implementation)
            yield
        finally:
            if callable(setter):
                setter(originals[0] or "sdpa")
            else:
                for config, original in zip(configs, originals):
                    setattr(config, "_attn_implementation", original or "sdpa")

    def _frames_to_messages(self, video_frames: np.ndarray, prompt: str) -> tuple[list[dict], np.ndarray]:
        video = np.asarray(video_frames, dtype=np.uint8)
        if video.ndim != 4 or video.shape[-1] != 3 or len(video) == 0:
            raise ValueError("video_frames must have shape [frames, height, width, 3]")
        messages = [{
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": prompt},
            ],
        }]
        return messages, video

    def _record_input_audit(self, inputs: dict, text: str, frame_count: int) -> dict:
        audit = {
            "rendered_prompt": text,
            "model_input_keys": sorted(inputs),
            "model_input_shapes": {
                key: list(value.shape)
                for key, value in inputs.items()
                if hasattr(value, "shape")
            },
            "vision_tensor_supplied": any(
                inputs.get(key) is not None for key in ("pixel_values", "pixel_values_videos")
            ),
            "video_grid_supplied": inputs.get("video_grid_thw") is not None,
            "video_modality_supplied": inputs.get("pixel_values_videos") is not None,
            "video_frame_count": int(frame_count),
        }
        self._last_input_audit = audit
        return audit

    def generate(self, video_frames: np.ndarray, prompt: str, generation_config: GenerationConfig) -> str:
        messages, video = self._frames_to_messages(video_frames, prompt)
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[text],
            videos=[video],
            return_tensors="pt",
            padding=True,
        )
        inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        prompt_length = int(inputs["attention_mask"].sum(dim=1).item())
        self._generation_diagnostics = [
            self._record_input_audit(inputs, text, len(video_frames))
        ]

        with self.torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=generation_config.max_new_tokens,
                do_sample=False,
                temperature=None,
                num_beams=1,
                use_cache=True,
            )

        generated_ids = output_ids[:, prompt_length:]
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

        message_video_pairs = [
            self._frames_to_messages(video_frames, prompt)
            for video_frames, prompt in zip(batch_video_frames, prompts)
        ]
        batch_messages = [messages for messages, _ in message_video_pairs]
        videos = [video for _, video in message_video_pairs]
        texts = [
            self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            for messages in batch_messages
        ]
        inputs = self.processor(
            text=texts,
            videos=videos,
            return_tensors="pt",
            padding=True,
        )
        inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()
        self._generation_diagnostics = [
            {
                "rendered_prompt": text,
                "model_input_keys": sorted(inputs),
                "model_input_shapes": {
                    key: list(value.shape)
                    for key, value in inputs.items()
                    if hasattr(value, "shape")
                },
                "vision_tensor_supplied": any(
                    inputs.get(key) is not None
                    for key in ("pixel_values", "pixel_values_videos")
                ),
                "video_grid_supplied": inputs.get("video_grid_thw") is not None,
                "video_modality_supplied": inputs.get("pixel_values_videos") is not None,
                "video_frame_count": int(len(frames)),
            }
            for text, frames in zip(texts, batch_video_frames)
        ]

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

    def consume_generation_diagnostics(self, expected_count: int) -> list[dict]:
        values = self._generation_diagnostics
        self._generation_diagnostics = []
        return values if len(values) == expected_count else [{} for _ in range(expected_count)]

    def _prepare_inputs(self, video_frames: np.ndarray, prompt: str) -> dict:
        messages, video = self._frames_to_messages(video_frames, prompt)
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[text],
            videos=[video],
            return_tensors="pt",
            padding=True,
        )
        prepared = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        self._record_input_audit(prepared, text, len(video_frames))
        return prepared

    def _clone_input_value(self, value):
        if hasattr(value, "clone"):
            return value.clone()
        if isinstance(value, np.ndarray):
            return value.copy()
        return value

    def _gaussian_noise_like(self, value, std: float, seed: int):
        if std <= 0:
            return value
        rng = np.random.default_rng(seed)
        if hasattr(value, "detach") and hasattr(value, "dtype"):
            noise = self.torch.from_numpy(
                rng.normal(0.0, std, size=tuple(value.shape)).astype(np.float32)
            ).to(device=value.device, dtype=value.dtype)
            return value + noise
        if isinstance(value, np.ndarray):
            noise = rng.normal(0.0, std, size=value.shape).astype(np.float32)
            return value + noise.astype(value.dtype, copy=False)
        return value

    def _temporal_homogenize_value(self, value, beta: float):
        if beta <= 0 or not hasattr(value, "ndim") or value.ndim < 4:
            return value
        if value.ndim == 4:
            context = value.mean(dim=0, keepdim=True) if hasattr(value, "detach") else value.mean(axis=0, keepdims=True)
        else:
            context = value.mean(dim=1, keepdim=True) if hasattr(value, "detach") else value.mean(axis=1, keepdims=True)
        return (1.0 - beta) * value + beta * context

    def _apply_branch_transform(self, inputs: dict, branch: str, **kwargs) -> dict:
        branch_inputs = {key: self._clone_input_value(value) for key, value in inputs.items()}
        if branch in {"original", "tcd_negative"}:
            # TCDMethod already passes the chronological frame subset. Applying
            # another transform here would invalidate native-video grid metadata.
            return branch_inputs
        if branch == "spatial_negative":
            noise_std = float(kwargs.get("noise_std", 0.1))
            seed = int(kwargs.get("seed", 0))
            for key in ("pixel_values", "pixel_values_videos"):
                if branch_inputs.get(key) is not None:
                    branch_inputs[key] = self._gaussian_noise_like(branch_inputs[key], noise_std, seed)
            return branch_inputs
        if branch == "temporal_homogenized":
            beta = float(kwargs.get("beta", 0.33))
            for key in ("pixel_values", "pixel_values_videos"):
                if branch_inputs.get(key) is not None:
                    branch_inputs[key] = self._temporal_homogenize_value(branch_inputs[key], beta)
            return branch_inputs
        raise NotImplementedError(f"Qwen adapter does not support branch {branch}")

    def _ensure_dino_loaded(self, checkpoint: str, device: str = "cpu"):
        if self._dino_model is not None and self._dino_processor is not None:
            return self._dino_processor, self._dino_model

        from transformers import AutoImageProcessor, AutoModel

        local_only = self._checkpoint_is_local(checkpoint)
        self._dino_processor = AutoImageProcessor.from_pretrained(checkpoint, local_files_only=local_only)
        if device == "cpu":
            self._dino_model = AutoModel.from_pretrained(checkpoint, local_files_only=local_only).to("cpu").eval()
        else:
            self._dino_model = AutoModel.from_pretrained(
                checkpoint,
                torch_dtype=self._torch_dtype(),
                device_map="auto",
                local_files_only=local_only,
            ).eval()
        return self._dino_processor, self._dino_model

    def _ensure_birefnet_loaded(self, checkpoint: str = "ZhengPeng7/BiRefNet", device: str = "cpu"):
        if getattr(self, "_birefnet_model", None) is not None:
            return self._birefnet_model, self._birefnet_transform

        from transformers import AutoModelForImageSegmentation
        from torchvision import transforms

        self._birefnet_model = AutoModelForImageSegmentation.from_pretrained(
            checkpoint,
            trust_remote_code=True,
        ).to(device).eval()
        if str(device).startswith("cpu"):
            self._birefnet_model = self._birefnet_model.float()

        self._birefnet_transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return self._birefnet_model, self._birefnet_transform

    def _compute_dino_saliency(
        self,
        video_frames: np.ndarray,
        checkpoint: str,
        device: str = "cpu",
    ) -> tuple[np.ndarray, dict]:
        from PIL import Image

        processor, dino_model = self._ensure_dino_loaded(checkpoint, device=device)
        images = [Image.fromarray(np.asarray(frame, dtype=np.uint8)) for frame in video_frames]
        dino_inputs = processor(images=images, return_tensors="pt")
        dino_device = next(dino_model.parameters()).device
        dino_inputs = {
            key: value.to(dino_device) if hasattr(value, "to") else value
            for key, value in dino_inputs.items()
        }

        with self.torch.inference_mode():
            outputs = dino_model(**dino_inputs, output_hidden_states=True, return_dict=True)

        tokens = outputs.last_hidden_state[:, 1:, :].float()
        patch_scores = tokens.norm(dim=-1)
        patch_scores = patch_scores / (patch_scores.max(dim=1, keepdim=True).values + 1e-6)
        frame_scores = patch_scores.mean(dim=1).detach().cpu().numpy().astype(np.float32)
        diagnostics = {
            "dino_loaded": True,
            "dino_device": str(dino_device),
            "dino_patch_tokens": int(patch_scores.shape[1]),
            "dino_frame_saliency_mean": float(frame_scores.mean()),
        }
        return frame_scores, diagnostics

    def _coerce_feature_tensor(self, value):
        if value is None:
            return None
        if hasattr(value, "last_hidden_state"):
            return self._coerce_feature_tensor(value.last_hidden_state)
        if hasattr(value, "hidden_states"):
            hidden_states = value.hidden_states
            if isinstance(hidden_states, (tuple, list)) and hidden_states:
                return self._coerce_feature_tensor(hidden_states[-1])
        if isinstance(value, (tuple, list)):
            for item in value:
                tensor = self._coerce_feature_tensor(item)
                if tensor is not None:
                    return tensor
            return None
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            return value
        return None

    def _build_token_scaling(self, total_tokens: int, frame_scores: np.ndarray) -> tuple["torch.Tensor", list[tuple[int, int]]]:
        n_frames = max(int(len(frame_scores)), 1)
        base = max(total_tokens // n_frames, 1)
        remainder = total_tokens % n_frames
        spans: list[tuple[int, int]] = []
        start = 0
        for idx in range(n_frames):
            size = base + (1 if idx < remainder else 0)
            spans.append((start, min(start + size, total_tokens)))
            start += size

        scaling = self.torch.ones((total_tokens,), device=self.device, dtype=self._torch_dtype())
        for idx, (begin, end) in enumerate(spans):
            if begin >= total_tokens:
                break
            end = max(begin + 1, min(end, total_tokens))
            scale_value = 1.0 + float(frame_scores[min(idx, len(frame_scores) - 1)])
            scaling[begin:end] = scale_value
        return scaling, spans

    def _apply_token_scaling_to_output(self, output, scaling):
        tensor = self._coerce_feature_tensor(output)
        if tensor is None:
            return output, False

        scaled = False
        if tensor.ndim == 2:
            token_count = min(int(tensor.shape[0]), int(scaling.shape[0]))
            if token_count > 0:
                tensor[:token_count] = tensor[:token_count] * scaling[:token_count].unsqueeze(-1).to(tensor.dtype)
                scaled = True
        elif tensor.ndim >= 3:
            token_count = min(int(tensor.shape[-2]), int(scaling.shape[0]))
            if token_count > 0:
                view_shape = [1] * tensor.ndim
                view_shape[-2] = token_count
                tensor_slice = tensor[..., :token_count, :]
                tensor[..., :token_count, :] = tensor_slice * scaling[:token_count].view(*view_shape).to(tensor.dtype)
                scaled = True

        if not scaled:
            return output, False

        if hasattr(output, "last_hidden_state"):
            output.last_hidden_state = tensor
            return output, True
        if isinstance(output, tuple):
            output_list = list(output)
            replaced = False
            for idx, item in enumerate(output_list):
                if self._coerce_feature_tensor(item) is not None:
                    output_list[idx] = tensor
                    replaced = True
                    break
            return tuple(output_list), replaced
        return tensor, True

    def prepare_branch(self, video_frames: np.ndarray, prompt: str, branch: str, **kwargs):
        if branch not in {"original", "tcd_negative", "spatial_negative", "temporal_homogenized"}:
            raise NotImplementedError(f"Qwen adapter does not support branch {branch}")
        started = time.perf_counter()
        inputs = self._prepare_inputs(video_frames, prompt)
        inputs = self._apply_branch_transform(inputs, branch, **kwargs)
        return {
            "branch": branch,
            "model_inputs": inputs,
            "past_key_values": None,
            "rope_deltas": None,
            "generated_count": 0,
            "vision_attn": None,
            "profile": bool(kwargs.get("profile", False)),
            "preserve_logits_on_device": bool(kwargs.get("preserve_logits_on_device", False)),
            "diagnostics": {
                **getattr(self, "_last_input_audit", {}),
                "frame_count": int(len(video_frames)),
                "preprocessing_seconds": time.perf_counter() - started,
                "decode_call_count": 0,
                "cache_hit_steps": 0,
                "vision_inputs_supplied_steps": 0,
            },
        }

    def _profile_sync(self, state) -> None:
        if state.get("profile") and self.torch.cuda.is_available():
            self.torch.cuda.synchronize()

    def prepare_positive_branch(self, video_frames: np.ndarray, prompt: str, foreground, **kwargs):
        return self.prepare_branch(video_frames, prompt, branch="original", **kwargs)

    def _sync_generated_tokens(self, state, token_ids: list[int]) -> None:
        if len(token_ids) <= state["generated_count"]:
            return

        new_token_ids = token_ids[state["generated_count"] :]
        inputs = state["model_inputs"]
        token_tensor = self.torch.tensor([new_token_ids], device=self.device, dtype=inputs["input_ids"].dtype)
        inputs["input_ids"] = self.torch.cat([inputs["input_ids"], token_tensor], dim=1)

        if "attention_mask" in inputs:
            extra_mask = self.torch.ones(
                (1, len(new_token_ids)),
                device=self.device,
                dtype=inputs["attention_mask"].dtype,
            )
            inputs["attention_mask"] = self.torch.cat([inputs["attention_mask"], extra_mask], dim=1)

        if "mm_token_type_ids" in inputs:
            extra_token_types = self.torch.zeros(
                (1, len(new_token_ids)),
                device=self.device,
                dtype=inputs["mm_token_type_ids"].dtype,
            )
            inputs["mm_token_type_ids"] = self.torch.cat([inputs["mm_token_type_ids"], extra_token_types], dim=1)

        state["generated_count"] = len(token_ids)

    def _branch_position_ids(self, state, is_first_iteration: bool):
        inputs = state["model_inputs"]
        if not is_first_iteration:
            if state["rope_deltas"] is None:
                raise RuntimeError("Qwen cached decoding is missing branch-specific rope_deltas")
            return cached_mrope_position_ids(inputs["attention_mask"], state["rope_deltas"])

        rope_model = getattr(self.model, "model", None)
        rope_function = getattr(rope_model, "get_rope_index", None)
        if rope_function is None:
            rope_function = getattr(rope_model, "compute_3d_position_ids", None)
        if rope_function is None:
            raise RuntimeError("Qwen model does not expose an mRoPE helper for manual cached decoding")

        rope_kwargs = {
            "input_ids": inputs["input_ids"],
            "mm_token_type_ids": inputs.get("mm_token_type_ids"),
            "image_grid_thw": inputs.get("image_grid_thw"),
            "video_grid_thw": inputs.get("video_grid_thw"),
            "second_per_grid_ts": inputs.get("second_per_grid_ts"),
            "attention_mask": inputs.get("attention_mask"),
        }
        parameters = inspect.signature(rope_function).parameters
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        if not accepts_kwargs:
            rope_kwargs = {key: value for key, value in rope_kwargs.items() if key in parameters}

        rope_result = rope_function(**rope_kwargs)
        if isinstance(rope_result, tuple) and len(rope_result) == 2:
            position_ids, rope_deltas = rope_result
        else:
            position_ids = rope_result
            rope_deltas = getattr(rope_model, "rope_deltas", None)
        if rope_deltas is None:
            raise RuntimeError("Qwen mRoPE helper did not return or store rope_deltas")
        state["rope_deltas"] = rope_deltas.detach().clone()
        return position_ids

    def decode_step(self, state, token_ids: list[int], output_attentions: bool = False) -> StepOutput:
        self._sync_generated_tokens(state, token_ids)
        inputs = state["model_inputs"]
        is_first_iteration = state["past_key_values"] is None
        position_ids = self._branch_position_ids(state, is_first_iteration)
        mm_token_type_ids = inputs.get("mm_token_type_ids")
        if mm_token_type_ids is not None:
            mm_token_type_ids = select_decode_input_ids(mm_token_type_ids, state["past_key_values"])
        prepared = self.model.prepare_inputs_for_generation(
            input_ids=inputs["input_ids"],
            next_sequence_length=None if is_first_iteration else 1,
            past_key_values=state["past_key_values"],
            attention_mask=inputs.get("attention_mask"),
            position_ids=position_ids,
            inputs_embeds=inputs.get("inputs_embeds"),
            pixel_values=inputs.get("pixel_values") if is_first_iteration else None,
            pixel_values_videos=inputs.get("pixel_values_videos") if is_first_iteration else None,
            image_grid_thw=inputs.get("image_grid_thw") if is_first_iteration else None,
            video_grid_thw=inputs.get("video_grid_thw") if is_first_iteration else None,
            second_per_grid_ts=inputs.get("second_per_grid_ts") if is_first_iteration else None,
            mm_token_type_ids=mm_token_type_ids,
            use_cache=True,
            is_first_iteration=is_first_iteration,
        )
        attention_context = (
            self._temporary_attention_implementation("eager")
            if output_attentions
            else nullcontext()
        )
        with attention_context, self.torch.inference_mode():
            outputs = self.model(
                **prepared,
                output_attentions=output_attentions,
                return_dict=True,
            )
        state["past_key_values"] = outputs.past_key_values
        logits = outputs.logits[0, -1].float().detach()
        if not state["preserve_logits_on_device"]:
            logits = logits.cpu().numpy()
        state["diagnostics"]["decode_call_count"] += 1
        state["diagnostics"]["cache_hit_steps"] += int(not is_first_iteration)
        state["diagnostics"]["vision_inputs_supplied_steps"] += int(
            any(prepared.get(key) is not None for key in ("pixel_values", "pixel_values_videos"))
        )
        frame_scores = None
        if output_attentions:
            decoder_attentions = getattr(outputs, "attentions", None)
            if decoder_attentions is None:
                language_output = getattr(outputs, "language_model_output", None)
                decoder_attentions = getattr(language_output, "attentions", None)
            if decoder_attentions is not None:
                frame_scores = self._summarize_frame_attention(
                    decoder_attentions,
                    int(state["diagnostics"]["frame_count"]),
                )
        return StepOutput(logits=logits, frame_attention=frame_scores)

    def _summarize_frame_attention(self, attentions, frame_count: int) -> np.ndarray | None:
        if attentions is None:
            return None
        try:
            attn_layers = []
            if frame_count <= 0:
                return None
            for layer in attentions:
                if layer is None:
                    continue
                layer = layer.detach().float().cpu().numpy()
                if layer.ndim != 4:
                    continue
                per_frame = layer.mean(axis=(0, 1, 3))
                if per_frame.size == 0:
                    continue
                if per_frame.size == frame_count:
                    attn_layers.append(per_frame)
                else:
                    buckets = np.array_split(per_frame, frame_count)
                    attn_layers.append(np.asarray([bucket.mean() if bucket.size else 0.0 for bucket in buckets], dtype=np.float32))
            if not attn_layers:
                return None
            stacked = np.stack(attn_layers, axis=0).astype(np.float32)
            return frame_attention(stacked[:, None, :, None]).astype(np.float32)
        except Exception:
            return None

    def generate_dino_heal(self, video_frames, prompt: str, generation_config: GenerationConfig, config: dict):
        dino_config = DINOHealConfig(
            visual_weight=float(config.get("visual_weight", 0.3)),
            saliency_weight=float(config.get("saliency_weight", 0.7)),
            require_dino=bool(config.get("require_dino", True)),
        )
        checkpoint = config.get("dino_checkpoint", "facebook/dinov2-large")
        dino_device = str(config.get("dino_device", "cpu"))
        diagnostics = {
            "dino_loaded": False,
            "dino_checkpoint": checkpoint,
            "dino_device": dino_device,
        }

        frame_saliency, dino_diag = self._compute_dino_saliency(video_frames, checkpoint, device=dino_device)
        diagnostics.update(dino_diag)

        inputs = self._prepare_inputs(video_frames, prompt)
        diagnostics.update(getattr(self, "_last_input_audit", {}))
        pixel_values = inputs.get("pixel_values_videos")
        if pixel_values is None:
            raise RuntimeError("Qwen inputs are missing pixel_values_videos for DINO-HEAL")

        total_tokens = int(pixel_values.shape[0]) if hasattr(pixel_values, "shape") else len(video_frames)
        n_frames = len(video_frames)
        base = max(total_tokens // max(n_frames, 1), 1)
        remainder = total_tokens % max(n_frames, 1)
        spans = []
        start = 0
        for idx in range(n_frames):
            size = base + (1 if idx < remainder else 0)
            spans.append((start, start + size))
            start += size

        frame_features = []
        patch_saliency = []
        for idx, _span in enumerate(spans):
            frame_features.append(np.zeros((1, 1), dtype=np.float32))
            patch_saliency.append(np.asarray([frame_saliency[idx]], dtype=np.float32))

        features_np = np.stack(frame_features, axis=0)
        saliency_np = np.stack(patch_saliency, axis=0)
        fused_np = fuse_saliency(features_np, saliency_np, dino_config)
        fused_scale = fused_np[..., 0].mean(axis=1)
        fused_scale = np.maximum(fused_scale, 0.0).astype(np.float32)

        scaling = self.torch.ones((total_tokens, 1), device=self.device, dtype=self._torch_dtype())
        for idx, (start, end) in enumerate(spans):
            scaling[start:end] = 1.0 + self.torch.tensor(fused_scale[idx], device=self.device, dtype=scaling.dtype)
        scaling = scaling.squeeze(-1)

        holder = {"applied": False}

        def vision_hook(module, module_inputs, module_output):
            scaled_output, applied = self._apply_token_scaling_to_output(module_output, scaling)
            holder["applied"] = holder["applied"] or applied
            return scaled_output

        try:
            vision_module = (
                getattr(self.model, "visual", None)
                or getattr(self.model, "vision_tower", None)
                or getattr(getattr(self.model, "model", None), "visual", None)
                or getattr(getattr(self.model, "model", None), "vision_tower", None)
            )
            handle = vision_module.register_forward_hook(vision_hook) if vision_module is not None else None
        except Exception:
            handle = None

        try:
            with self.torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=generation_config.max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    num_beams=1,
                    use_cache=True,
                )
        finally:
            if handle is not None:
                handle.remove()

        diagnostics["dino_hook_applied"] = holder["applied"]
        diagnostics["dino_scale_mean"] = float(fused_scale.mean())
        diagnostics["dino_scale_max"] = float(fused_scale.max())
        answer = self.processor.batch_decode(output_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        return answer, diagnostics

    def _get_vision_module(self):
        """Resolve the vision encoder sub-module for hook registration."""
        return (
            getattr(self.model, "visual", None)
            or getattr(self.model, "vision_tower", None)
            or getattr(getattr(self.model, "model", None), "visual", None)
            or getattr(getattr(self.model, "model", None), "vision_tower", None)
        )

    def _compute_dino_patch_saliency(
        self, video_frames, T, P, checkpoint, device="cpu",
    ):
        """Compute [T, P] foreground saliency from DINO patch norms.

        Returns a torch Tensor on ``self.device``.
        """
        from PIL import Image

        n_frames = len(video_frames)
        proc, dino = self._ensure_dino_loaded(checkpoint, device=device)
        images = [Image.fromarray(np.asarray(f, dtype=np.uint8)) for f in video_frames]
        dino_inp = proc(images=images, return_tensors="pt")
        dino_dev = next(dino.parameters()).device
        dino_inp = {k: v.to(dino_dev) if hasattr(v, "to") else v for k, v in dino_inp.items()}

        with self.torch.inference_mode():
            out = dino(**dino_inp, output_hidden_states=True, return_dict=True)

        # [n_frames, dino_patches] — skip CLS token
        scores = out.last_hidden_state[:, 1:, :].float().norm(dim=-1)
        scores = scores / (scores.max(dim=1, keepdim=True).values + 1e-6)
        dino_P = scores.shape[1]

        # 1D interpolate to P patches per frame
        if dino_P != P:
            scores = self.torch.nn.functional.interpolate(
                scores.unsqueeze(1), size=P, mode="linear", align_corners=False,
            ).squeeze(1)

        # Map n_frames → T temporal positions
        if n_frames == T:
            fg = scores
        elif n_frames < T:
            idx = np.rint(np.linspace(0, n_frames - 1, T)).astype(int)
            fg = scores[idx]
        else:
            fg = self.torch.zeros(T, P, device=scores.device, dtype=scores.dtype)
            for t in range(T):
                lo = t * n_frames // T
                hi = max(lo + 1, (t + 1) * n_frames // T)
                fg[t] = scores[lo:hi].mean(0)

        return fg.to(self.device)

    def _compute_birefnet_patch_saliency(
        self, video_frames, T, P, Ht, Wt, checkpoint="ZhengPeng7/BiRefNet", device="cpu",
    ):
        """Compute [T, P] foreground saliency from BiRefNet segmentation masks."""
        from PIL import Image

        n_frames = len(video_frames)
        birefnet, transform = self._ensure_birefnet_loaded(checkpoint, device=device)
        birefnet_device = next(birefnet.parameters()).device

        images = [Image.fromarray(np.asarray(f, dtype=np.uint8)) for f in video_frames]
        inputs = self.torch.stack([transform(img) for img in images]).to(birefnet_device)

        with self.torch.inference_mode():
            outputs = birefnet(inputs)
            # BiRefNet typically returns a list of outputs, with the last one being the final mask
            preds = outputs[-1].sigmoid() if isinstance(outputs, (list, tuple)) else outputs.sigmoid()

        # Resize to Ht, Wt
        preds = self.torch.nn.functional.interpolate(
            preds, size=(Ht, Wt), mode="bilinear", align_corners=False
        ).squeeze(1) # shape: [B, Ht, Wt]

        scores = preds.view(n_frames, -1).float() # [B, P]

        if n_frames == T:
            fg = scores
        elif n_frames < T:
            idx = np.rint(np.linspace(0, n_frames - 1, T)).astype(int)
            fg = scores[idx]
        else:
            fg = self.torch.zeros(T, P, device=scores.device, dtype=scores.dtype)
            for t in range(T):
                lo = t * n_frames // T
                hi = max(lo + 1, (t + 1) * n_frames // T)
                fg[t] = scores[lo:hi].mean(0)

        return fg.to(self.device)

    def generate_positive_feature(self, video_frames, prompt, generation_config, config):
        """Generate text with BiRefNet-based positive visual-feature enhancement.

        Hooks into the vision encoder to apply spatial saliency scaling and
        directed temporal motion evidence, then runs ``model.generate()``.

        Uses BiRefNet for foreground segmentation by default (``use_birefnet=True``).
        """
        use_birefnet = bool(config.get("use_birefnet", True))
        pf_config = PositiveFeatureConfig(
            alpha=float(config.get("alpha", 0.4)),
            alpha_s=float(config.get("alpha_s", 0.4)),
            beta=float(config.get("beta", 0.4)),
            epsilon=float(config.get("epsilon", 1e-6)),
            use_birefnet=use_birefnet,
            birefnet_checkpoint=config.get("birefnet_checkpoint", "ZhengPeng7/BiRefNet"),
            dino_checkpoint=config.get("dino_checkpoint", "facebook/dinov2-large"),
            saliency_device=str(config.get("dino_device", "cpu")),
        )

        # ── prepare inputs ──
        inputs = self._prepare_inputs(video_frames, prompt)
        prompt_length = int(inputs["attention_mask"].sum(dim=1).item())

        # ── determine vision grid ──
        grid_thw = inputs.get("video_grid_thw")
        if grid_thw is None:
            raise RuntimeError("Cannot determine video grid from Qwen processor output")
        T = int(grid_thw[:, 0].sum().item())
        Ht = int(grid_thw[0, 1].item())
        Wt = int(grid_thw[0, 2].item())
        merge_size = int(getattr(getattr(self.processor, "image_processor", None), "merge_size", 1))
        spatial_tokens = Ht * Wt
        if spatial_tokens % (merge_size * merge_size) != 0:
            raise RuntimeError(
                "Qwen video grid is incompatible with the configured spatial merge size"
            )
        P = spatial_tokens // (merge_size * merge_size)

        # ── compute foreground saliency ──
        diagnostics: dict = {
            "positive_feature_mode": "birefnet_vision_hook" if use_birefnet else "dino_vision_hook",
            "positive_feature_hook_applied": False,
            "use_birefnet": use_birefnet,
            "birefnet_loaded": False,
            "dino_loaded": False,
            **dict(getattr(self, "_last_input_audit", {})),
        }

        fg = None
        try:
            if use_birefnet:
                birefnet_model, birefnet_transform = self._ensure_birefnet_loaded(
                    pf_config.birefnet_checkpoint, pf_config.saliency_device
                )
                fg = compute_birefnet_foreground(
                    video_frames, T, P, birefnet_model, birefnet_transform,
                    self.torch, self.device,
                )
                diagnostics["birefnet_loaded"] = True
            else:
                fg = self._compute_dino_patch_saliency(
                    video_frames, T, P, pf_config.dino_checkpoint, pf_config.saliency_device
                )
                diagnostics["dino_loaded"] = True
        except Exception as exc:
            diagnostics["saliency_fallback"] = repr(exc)
            if not use_birefnet:
                raise RuntimeError("DINO foreground extraction failed") from exc
            try:
                fg = self._compute_dino_patch_saliency(
                    video_frames, T, P, pf_config.dino_checkpoint, pf_config.saliency_device
                )
                diagnostics["dino_loaded"] = True
                diagnostics["positive_feature_mode"] = "dino_vision_hook_fallback"
            except Exception as dino_exc:
                raise RuntimeError(
                    "Both BiRefNet and DINO foreground extraction failed"
                ) from dino_exc

        holder: dict = {"applied": False}

        # ── vision hook ──
        def feat_hook(mod, inp, out):
            if hasattr(out, "last_hidden_state"):
                Fv = out.last_hidden_state
            elif hasattr(out, "pooler_output"):
                Fv = out.pooler_output
            elif isinstance(out, tuple):
                Fv = out[0]
            else:
                Fv = out

            orig_shape, orig_dtype = Fv.shape, Fv.dtype
            f = Fv.reshape(-1, Fv.shape[-1]).float()
            n_vis, D = f.shape
            if n_vis != T * P:
                return out  # shape mismatch → skip

            V = f.view(T, P, D)
            V_prime, hook_diag = enhance_visual_embeddings(V, fg, pf_config, self.torch)

            holder["applied"] = True
            holder["diagnostics"] = hook_diag

            result = V_prime.reshape(orig_shape).to(orig_dtype)
            if hasattr(out, "last_hidden_state"):
                out.last_hidden_state = result
                return out
            if hasattr(out, "pooler_output"):
                out.pooler_output = result
                return out
            if isinstance(out, tuple):
                return (result,) + out[1:]
            return result

        vision = self._get_vision_module()
        handle = vision.register_forward_hook(feat_hook) if vision else None
        try:
            with self.torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=generation_config.max_new_tokens,
                    do_sample=False, temperature=None, num_beams=1,
                    use_cache=True,
                )
        finally:
            if handle:
                handle.remove()

        answer = self.processor.batch_decode(
            output_ids[:, prompt_length:], skip_special_tokens=True,
        )[0].strip()

        diagnostics["positive_feature_hook_applied"] = holder["applied"]
        diagnostics.update(holder.get("diagnostics", {}))
        diagnostics.update({
            "grid_T": T, "grid_Ht": Ht, "grid_Wt": Wt,
            "grid_P_after_merge": P,
            "spatial_merge_size": merge_size,
            "alpha": pf_config.alpha, "alpha_s": pf_config.alpha_s, "beta": pf_config.beta,
        })
        return answer, diagnostics

    def forward_positive_feature(
        self,
        video_frames,
        prompt: str,
        foreground_ref: dict,
        alpha: float = 0.4,
        alpha_s: float = 0.4,
        beta: float = 0.4,
        use_evidence: bool = True,
    ) -> dict:
        """Forward pass with positive visual-feature enhancement via vision encoder hook.

        Hooks into the vision encoder output to apply:
        1. Spatial scaling: V' = V * (1 + alpha*fg + alpha_s*persist)
        2. Directed motion evidence: V' += beta * norm_stabilized_diff
        """
        T, Ht, Wt = foreground_ref["grid"]
        P = Ht * Wt
        fg = foreground_ref["fg"].to(self.device).float()   # [T, P]
        persist = fg.mean(0, keepdim=True).expand(T, P)      # [T, P]
        holder = {}
        handles = []

        def feat_hook(mod, inp, out):
            # Step 1: extract the feature tensor
            if hasattr(out, "pooler_output"):
                Fv = out.pooler_output
            elif hasattr(out, "last_hidden_state"):
                Fv = out.last_hidden_state
            elif isinstance(out, tuple):
                Fv = out[0]
            else:
                Fv = out

            sq = (Fv.dim() == 3)
            f = (Fv[0] if sq else Fv).float()
            n_vis, D = f.shape
            assert n_vis == T * P, f"n_vis {n_vis} != T*P {T * P}"
            V = f.view(T, P, D)

            # Spatial: scale by saliency (fg + persist)
            S = alpha * fg + alpha_s * persist
            Vp = V * (1.0 + S.unsqueeze(-1))

            # Directed: add motion evidence
            e = self.torch.zeros_like(V)
            e[1:] = V[1:] - V[:-1]
            e = e * fg.unsqueeze(-1)
            e = e / (e.norm(-1, keepdim=True) + 1e-6) * V.norm(-1, keepdim=True)
            Vp = Vp + beta * e

            holder["delta"] = ((Vp - V).norm(-1) / (V.norm(-1) + 1e-6)).mean().item()

            # Step 2: return in original format
            o = Vp.view(n_vis, D).to(Fv.dtype)
            result = o.unsqueeze(0) if sq else o
            if hasattr(out, "pooler_output"):
                out.pooler_output = result
                return out
            elif hasattr(out, "last_hidden_state"):
                out.last_hidden_state = result
                return out
            elif isinstance(out, tuple):
                return (result,) + out[1:]
            else:
                return result

        try:
            inputs = self._prepare_inputs(video_frames, prompt)

            if use_evidence:
                vision_module = self._get_vision_module()
                if vision_module is not None:
                    handles.append(vision_module.register_forward_hook(feat_hook))

            with self.torch.inference_mode():
                logits = self.model(**inputs, use_cache=False).logits[0, -1].float().cpu()
        finally:
            for h in handles:
                h.remove()

        return {"logits": logits, "delta": holder.get("delta")}

    def token_id_to_text(self, token_id: int) -> str:
        return self.processor.tokenizer.decode([token_id], skip_special_tokens=True)

    def decode_token_ids(self, token_ids: list[int]) -> str:
        return self.processor.tokenizer.decode(token_ids, skip_special_tokens=True).strip()

    def is_eos(self, token_id: int) -> bool:
        return token_id == getattr(self.processor.tokenizer, "eos_token_id", None)

    @property
    def supports_step_logits(self) -> bool:
        return True

    @property
    def supports_frame_attention(self) -> bool:
        return True

    @property
    def supports_vision_layer_hooks(self) -> bool:
        return True

    @property
    def supports_positive_feature_hooks(self) -> bool:
        return True
