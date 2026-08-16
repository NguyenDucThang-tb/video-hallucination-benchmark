from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.methods.dino_heal.fusion import DINOHealConfig, fuse_saliency

from .base import GenerationConfig, ModelAdapter, StepOutput


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
        prompt_length = int(inputs["attention_mask"].sum(dim=1).item())

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

    def _prepare_inputs(self, video_frames: np.ndarray, prompt: str) -> dict:
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
        return {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

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

    def prepare_branch(self, video_frames: np.ndarray, prompt: str, branch: str, **kwargs):
        if branch not in {"original", "tcd_negative", "spatial_negative", "temporal_homogenized"}:
            raise NotImplementedError(f"Qwen adapter does not support branch {branch}")
        inputs = self._prepare_inputs(video_frames, prompt)
        return {
            "branch": branch,
            "model_inputs": inputs,
            "past_key_values": None,
            "generated_count": 0,
            "vision_attn": None,
        }

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

    def decode_step(self, state, token_ids: list[int], output_attentions: bool = False) -> StepOutput:
        self._sync_generated_tokens(state, token_ids)
        inputs = state["model_inputs"]
        prepared = self.model.prepare_inputs_for_generation(
            input_ids=inputs["input_ids"],
            past_key_values=state["past_key_values"],
            attention_mask=inputs.get("attention_mask"),
            inputs_embeds=inputs.get("inputs_embeds"),
            pixel_values=inputs.get("pixel_values"),
            pixel_values_videos=inputs.get("pixel_values_videos"),
            image_grid_thw=inputs.get("image_grid_thw"),
            video_grid_thw=inputs.get("video_grid_thw"),
            second_per_grid_ts=inputs.get("second_per_grid_ts"),
            mm_token_type_ids=inputs.get("mm_token_type_ids"),
            use_cache=True,
            is_first_iteration=state["past_key_values"] is None,
        )
        with self.torch.inference_mode():
            outputs = self.model(
                **prepared,
                output_attentions=output_attentions,
                return_dict=True,
            )
        state["past_key_values"] = outputs.past_key_values
        logits = outputs.logits[0, -1].float().detach().cpu().numpy()
        frame_attention = None
        if output_attentions and getattr(outputs, "attentions", None) is not None:
            frame_attention = self._summarize_frame_attention(outputs.attentions, inputs)
        return StepOutput(logits=logits, frame_attention=frame_attention)

    def _summarize_frame_attention(self, attentions, inputs) -> np.ndarray | None:
        if attentions is None:
            return None
        try:
            attn_layers = []
            pixel_values = inputs.get("pixel_values")
            if pixel_values is None:
                return None
            n_frames = int(pixel_values.shape[0]) if hasattr(pixel_values, "shape") else 1
            for layer in attentions:
                layer = layer.detach().float().cpu().numpy()
                if layer.ndim != 4:
                    continue
                per_frame = layer.mean(axis=(0, 1, 3))
                if per_frame.size == 0:
                    continue
                if per_frame.size == n_frames:
                    attn_layers.append(per_frame)
                else:
                    buckets = np.array_split(per_frame, n_frames)
                    attn_layers.append(np.asarray([bucket.mean() if bucket.size else 0.0 for bucket in buckets], dtype=np.float32))
            if not attn_layers:
                return None
            stacked = np.stack(attn_layers, axis=0).astype(np.float32)
            heads = np.ones((stacked.shape[0], 1), dtype=np.float32)
            return stacked[:, None, :, :]
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
        pixel_values = inputs.get("pixel_values")
        if pixel_values is None:
            raise RuntimeError("Qwen inputs are missing pixel_values for DINO-HEAL")

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

        holder = {"applied": False}

        def vision_hook(module, module_inputs, module_output):
            if hasattr(module_output, "hidden_states"):
                holder["applied"] = True
            return module_output

        try:
            vision_module = getattr(self.model, "visual", None) or getattr(self.model, "vision_tower", None)
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

    def token_id_to_text(self, token_id: int) -> str:
        return self.processor.tokenizer.decode([token_id], skip_special_tokens=True)

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
        return False
