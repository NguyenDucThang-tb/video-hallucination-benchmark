from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .base import GenerationConfig, ModelAdapter, StepOutput
from src.methods.dino_heal.fusion import DINOHealConfig, fuse_saliency


class LlavaOVAdapter(ModelAdapter):
    name = "llava-ov-7b"

    def __init__(self, checkpoint: str, local_path: str | None = None):
        self.checkpoint = checkpoint
        self.local_path = local_path
        self.model_path = self._resolve_model_path(local_path, checkpoint)

        import torch
        from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            local_files_only=self._is_local_only(),
        )
        self._configure_padding()
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=self._torch_dtype(),
            device_map="auto",
            local_files_only=self._is_local_only(),
        ).eval()
        self.device = next(self.model.parameters()).device
        self._dino_processor = None
        self._dino_model = None

    def _resolve_model_path(self, local_path: str | None, checkpoint: str) -> str:
        for candidate in (
            local_path,
            os.environ.get("LLAVA_OV_MODEL_DIR"),
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

    def _configure_padding(self) -> None:
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            return
        tokenizer.padding_side = "left"
        if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        if hasattr(self.processor, "padding_side"):
            self.processor.padding_side = "left"

    def _frames_to_conversation(self, video_frames: np.ndarray, prompt: str) -> tuple[list[dict], list]:
        from PIL import Image

        images = [Image.fromarray(np.asarray(frame, dtype=np.uint8)) for frame in video_frames]
        content = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": prompt})
        conversation = [{"role": "user", "content": content}]
        return conversation, images

    def _build_inputs(self, video_frames: np.ndarray, prompt: str) -> tuple[dict, int]:
        conversation, images = self._frames_to_conversation(video_frames, prompt)
        text = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
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
        return inputs, prompt_length

    def _ensure_dino_loaded(self, checkpoint: str, device: str = "cpu"):
        if self._dino_model is not None and self._dino_processor is not None:
            return self._dino_processor, self._dino_model

        from transformers import AutoImageProcessor, AutoModel

        self._dino_processor = AutoImageProcessor.from_pretrained(
            checkpoint,
            local_files_only=self._checkpoint_is_local(checkpoint),
        )
        model_kwargs = {
            "local_files_only": self._checkpoint_is_local(checkpoint),
        }
        if device == "cpu":
            self._dino_model = AutoModel.from_pretrained(
                checkpoint,
                **model_kwargs,
            ).to("cpu").eval()
        else:
            self._dino_model = AutoModel.from_pretrained(
                checkpoint,
                torch_dtype=self._torch_dtype(),
                device_map="auto",
                **model_kwargs,
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
        frame_scores = patch_scores.mean(dim=1).cpu().numpy().astype(np.float32)
        diagnostics = {
            "dino_loaded": True,
            "dino_device": str(dino_device),
            "dino_patch_tokens": int(patch_scores.shape[1]),
            "dino_frame_saliency_mean": float(frame_scores.mean()),
        }
        return frame_scores, diagnostics

    def _coerce_feature_tensor(self, value):
        if hasattr(value, "float") and hasattr(value, "shape"):
            return value.float()
        if isinstance(value, (list, tuple)):
            tensors = [item for item in value if hasattr(item, "float") and hasattr(item, "shape")]
            if not tensors:
                raise TypeError(f"Unable to extract tensor features from {type(value).__name__}")
            if len(tensors) == 1:
                return tensors[0].float()
            return self.torch.cat([tensor.float() for tensor in tensors], dim=0)
        raise TypeError(f"Unsupported feature container: {type(value).__name__}")

    def _extract_image_features(self, image_outputs):
        if hasattr(image_outputs, "pooler_output") and image_outputs.pooler_output is not None:
            return self._coerce_feature_tensor(image_outputs.pooler_output)
        if hasattr(image_outputs, "last_hidden_state") and image_outputs.last_hidden_state is not None:
            hidden = self._coerce_feature_tensor(image_outputs.last_hidden_state)
            return hidden.reshape(-1, hidden.shape[-1])
        return self._coerce_feature_tensor(image_outputs)

    def _prepare_inputs(self, video_frames: np.ndarray, prompt: str) -> dict:
        inputs, _ = self._build_inputs(video_frames, prompt)
        return inputs

    def generate(self, video_frames: np.ndarray, prompt: str, generation_config: GenerationConfig) -> str:
        inputs, prompt_length = self._build_inputs(video_frames, prompt)

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

        frame_saliency, dino_diag = self._compute_dino_saliency(
            video_frames,
            checkpoint,
            device=dino_device,
        )
        diagnostics.update(dino_diag)

        inputs, prompt_length = self._build_inputs(video_frames, prompt)
        batch_num_images = inputs.get("batch_num_images")
        image_sizes = inputs.get("image_sizes")
        if image_sizes is None:
            raise RuntimeError("LLaVA-OV inputs are missing image_sizes for DINO-HEAL")

        if batch_num_images is None:
            counts = [1] * int(image_sizes.shape[0])
        else:
            counts = [int(x) for x in batch_num_images.detach().cpu().tolist()]

        image_outputs = self.model.get_image_features(
            pixel_values=inputs["pixel_values"],
            image_sizes=image_sizes,
            batch_num_images=batch_num_images,
        )
        image_features = self._extract_image_features(image_outputs)
        feature_lens = []
        for count in counts:
            for _ in range(count):
                feature_lens.append(None)
        # Infer token spans per frame from placeholder count match; fallback to equal chunking.
        total_tokens = int(image_features.shape[0])
        n_frames = len(video_frames)
        base = total_tokens // n_frames
        remainder = total_tokens % n_frames
        spans = []
        start = 0
        for idx in range(n_frames):
            size = base + (1 if idx < remainder else 0)
            spans.append((start, start + size))
            start += size

        frame_features = []
        patch_saliency = []
        for idx, (start, end) in enumerate(spans):
            pooled = image_features[start:end].mean(dim=0, keepdim=True).detach().cpu().numpy()
            frame_features.append(pooled)
            patch_saliency.append(np.asarray([frame_saliency[idx]], dtype=np.float32))

        features_np = np.stack(frame_features, axis=0)
        saliency_np = np.stack(patch_saliency, axis=0)
        fused_np = fuse_saliency(features_np, saliency_np, dino_config)
        fused_scale = fused_np[..., 0].mean(axis=1)
        fused_scale = np.maximum(fused_scale, 0.0).astype(np.float32)

        scaling = self.torch.ones((total_tokens, 1), device=self.device, dtype=image_features.dtype)
        for idx, (start, end) in enumerate(spans):
            scaling[start:end] = 1.0 + self.torch.tensor(
                fused_scale[idx],
                device=self.device,
                dtype=image_features.dtype,
            )

        holder = {"applied": False}

        def projector_hook(module, module_inputs, module_output):
            scaled = module_output * scaling[: module_output.shape[0]].unsqueeze(1)
            holder["applied"] = True
            return scaled

        handle = self.model.model.multi_modal_projector.register_forward_hook(projector_hook)
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
            handle.remove()

        diagnostics["dino_hook_applied"] = holder["applied"]
        diagnostics["dino_scale_mean"] = float(fused_scale.mean())
        diagnostics["dino_scale_max"] = float(fused_scale.max())
        generated_ids = output_ids[:, prompt_length:]
        answer = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return answer, diagnostics

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

        batch_conversations = []
        batch_images = []
        for video_frames, prompt in zip(batch_video_frames, prompts):
            conversation, images = self._frames_to_conversation(video_frames, prompt)
            batch_conversations.append(conversation)
            batch_images.append(images)

        texts = [
            self.processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )
            for conversation in batch_conversations
        ]
        inputs = self.processor(
            text=texts,
            images=batch_images,
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

    def prepare_branch(self, video_frames: np.ndarray, prompt: str, branch: str, **kwargs):
        if branch not in {"original", "tcd_negative"}:
            raise NotImplementedError(f"LLaVA-OV adapter does not support branch {branch}")
        inputs = self._prepare_inputs(video_frames, prompt)
        return {
            "branch": branch,
            "model_inputs": inputs,
            "past_key_values": None,
            "generated_count": 0,
        }

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
            inputs_embeds=inputs.get("inputs_embeds"),
            pixel_values=inputs.get("pixel_values"),
            image_sizes=inputs.get("image_sizes"),
            pixel_values_videos=inputs.get("pixel_values_videos"),
            image_sizes_videos=inputs.get("image_sizes_videos"),
            attention_mask=inputs.get("attention_mask"),
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
        return StepOutput(logits=logits)

    def token_id_to_text(self, token_id: int) -> str:
        return self.processor.tokenizer.decode([token_id], skip_special_tokens=True)

    def is_eos(self, token_id: int) -> bool:
        return token_id == getattr(self.processor.tokenizer, "eos_token_id", None)

    @property
    def supports_step_logits(self) -> bool:
        return True
