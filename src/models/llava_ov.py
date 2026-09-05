from __future__ import annotations

import os
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np

from .base import GenerationConfig, ModelAdapter, StepOutput, select_decode_input_ids
from src.methods.dino_heal.fusion import DINOHealConfig, fuse_saliency
from src.methods.positive_feature.enhancement import (
    PositiveFeatureConfig,
    compute_birefnet_foreground,
    enhance_output_by_frame_saliency,
    enhance_visual_embeddings,
    ensure_birefnet_loaded,
)
from src.methods.season.attention_diagnosis import frame_attention
from src.methods.season.positive_features import FeatureEnhancementConfig
from src.methods.season.vision_homogenization import (
    blend_temporal_hidden,
    frame_mean_context,
    replace_hidden_states,
    unwrap_hidden_states,
)


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
        self._season_attention_layers = (20, 21, 22, 23)
        self._generation_diagnostics = []

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

    def _frames_to_conversation(self, video_frames: np.ndarray, prompt: str) -> tuple[list[dict], np.ndarray]:
        video = np.asarray(video_frames, dtype=np.uint8)
        if video.ndim != 4 or video.shape[-1] != 3 or len(video) == 0:
            raise ValueError("video_frames must have shape [frames, height, width, 3]")
        conversation = [{
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": prompt},
            ],
        }]
        return conversation, video

    def _build_inputs(self, video_frames: np.ndarray, prompt: str) -> tuple[dict, int]:
        conversation, video = self._frames_to_conversation(video_frames, prompt)
        text = self.processor.apply_chat_template(
            conversation,
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
        self._last_input_audit = {
            "rendered_prompt": text,
            "model_input_keys": sorted(inputs),
            "model_input_shapes": {
                key: list(value.shape) for key, value in inputs.items() if hasattr(value, "shape")
            },
            "vision_tensor_supplied": any(
                inputs.get(key) is not None for key in ("pixel_values", "pixel_values_videos")
            ),
            "video_modality_supplied": inputs.get("pixel_values_videos") is not None,
            "video_frame_count": int(len(video_frames)),
        }
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

    def _ensure_birefnet_loaded(self, checkpoint: str = "ZhengPeng7/BiRefNet", device: str = "cpu"):
        """Lazy-load BiRefNet model for foreground segmentation."""
        return ensure_birefnet_loaded(self.__dict__, checkpoint, device, self.torch)

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

    def _scale_projector_output(self, output, scaling):
        if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
            scaled, applied = self._scale_projector_output(output.last_hidden_state, scaling)
            if applied:
                output.last_hidden_state = scaled
            return output, applied

        if isinstance(output, (tuple, list)):
            updated = list(output)
            applied = False
            for idx, item in enumerate(updated):
                scaled, item_applied = self._scale_projector_output(item, scaling)
                if item_applied:
                    updated[idx] = scaled
                    applied = True
            return type(output)(updated), applied

        if not hasattr(output, "shape"):
            return output, False

        tensor = output
        if tensor.ndim == 2:
            token_count = min(int(tensor.shape[0]), int(scaling.shape[0]))
            if token_count <= 0:
                return output, False
            tensor[:token_count] = tensor[:token_count] * scaling[:token_count].unsqueeze(-1).to(tensor.dtype)
            return tensor, True

        if tensor.ndim >= 3:
            token_axis = -2
            token_count = min(int(tensor.shape[token_axis]), int(scaling.shape[0]))
            if token_count <= 0:
                return output, False
            view_shape = [1] * tensor.ndim
            view_shape[token_axis] = token_count
            tensor_slice = tensor.narrow(token_axis, 0, token_count)
            scaled = tensor_slice * scaling[:token_count].view(*view_shape).to(tensor.dtype)
            tensor.narrow(token_axis, 0, token_count).copy_(scaled)
            return tensor, True

        return output, False

    def _prepare_inputs(self, video_frames: np.ndarray, prompt: str) -> dict:
        inputs, _ = self._build_inputs(video_frames, prompt)
        return inputs

    def generate(self, video_frames: np.ndarray, prompt: str, generation_config: GenerationConfig) -> str:
        inputs, prompt_length = self._build_inputs(video_frames, prompt)
        self._generation_diagnostics = [dict(self._last_input_audit)]

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
        pixel_values_videos = inputs.get("pixel_values_videos")
        if pixel_values_videos is None:
            raise RuntimeError("LLaVA-OV inputs are missing pixel_values_videos for DINO-HEAL")

        video_outputs = self.model.get_video_features(pixel_values=pixel_values_videos)
        image_features = self._extract_image_features(video_outputs)
        image_features = image_features.reshape(-1, image_features.shape[-1])
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
            scaled, applied = self._scale_projector_output(module_output, scaling)
            holder["applied"] = holder["applied"] or applied
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

    def generate_positive_feature(
        self,
        video_frames,
        prompt: str,
        generation_config: GenerationConfig,
        config: dict,
    ):
        """Generate text with BiRefNet-based positive visual-feature enhancement.

        Hooks into the multi_modal_projector output to apply:
        1. Spatial saliency scaling using BiRefNet foreground masks + persistence.
        2. Directed temporal motion evidence across consecutive frames.
        3. Fusion: V' = V·(1+S) + β·Diff
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
            foreground_threshold=float(config.get("foreground_threshold", 0.5)),
            foreground_morph_kernel=int(config.get("foreground_morph_kernel", 0)),
            foreground_return_soft=bool(config.get("foreground_return_soft", True)),
            foreground_pair_fusion=str(config.get("foreground_pair_fusion", "mean")),
            foreground_pool_avg_weight=float(config.get("foreground_pool_avg_weight", 1.0)),
        )
        n_frames = len(video_frames)
        diagnostics: dict = {
            "positive_feature_mode": "birefnet_projector_hook" if use_birefnet else "dino_projector_hook",
            "positive_feature_hook_applied": False,
            "use_birefnet": use_birefnet,
        }

        # Compute foreground saliency.
        # LLaVA-OV projector output shape is unknown until the hook fires,
        # so pre-compute fg at frame-level and expand inside the hook.
        fg_per_frame = None
        try:
            if use_birefnet:
                birefnet_model, birefnet_transform = self._ensure_birefnet_loaded(
                    pf_config.birefnet_checkpoint, pf_config.saliency_device
                )
                fg_per_frame = compute_birefnet_foreground(
                    video_frames, n_frames, 1, birefnet_model, birefnet_transform,
                    self.torch, self.device,
                    thr=pf_config.foreground_threshold,
                    kernel=pf_config.foreground_morph_kernel,
                    return_soft=pf_config.foreground_return_soft,
                    avg_weight=pf_config.foreground_pool_avg_weight,
                    pair_fusion=pf_config.foreground_pair_fusion,
                )  # [n_frames, 1]
                diagnostics["birefnet_loaded"] = True
            else:
                frame_saliency, dino_diag = self._compute_dino_saliency(
                    video_frames, pf_config.dino_checkpoint, pf_config.saliency_device
                )
                diagnostics.update(dino_diag)
                fg_per_frame = self.torch.as_tensor(
                    np.asarray(frame_saliency, dtype=np.float32),
                    device=self.device,
                ).unsqueeze(-1)  # [n_frames, 1]
        except Exception as exc:
            diagnostics["saliency_fallback"] = repr(exc)
            if not use_birefnet:
                raise RuntimeError("DINO foreground extraction failed") from exc
            try:
                frame_saliency, dino_diag = self._compute_dino_saliency(
                    video_frames, pf_config.dino_checkpoint, pf_config.saliency_device
                )
                diagnostics.update(dino_diag)
                diagnostics["positive_feature_mode"] = "dino_projector_hook_fallback"
                fg_per_frame = self.torch.as_tensor(
                    np.asarray(frame_saliency, dtype=np.float32),
                    device=self.device,
                ).unsqueeze(-1)
            except Exception as dino_exc:
                raise RuntimeError(
                    "Both BiRefNet and DINO foreground extraction failed"
                ) from dino_exc

        inputs, prompt_length = self._build_inputs(video_frames, prompt)
        if inputs.get("pixel_values_videos") is None:
            raise RuntimeError("LLaVA-OV inputs are missing pixel_values_videos for positive_feature")

        holder = {"applied": False, "diagnostics": {}}

        def projector_hook(module, module_inputs, module_output):
            if not hasattr(module_output, "shape"):
                return module_output

            orig_shape = module_output.shape
            orig_dtype = module_output.dtype
            f = module_output.reshape(-1, module_output.shape[-1]).float()
            n_vis, D = f.shape

            T = n_frames
            P = n_vis // T
            if T * P != n_vis:
                return module_output  # shape mismatch → skip

            V = f.view(T, P, D)

            # Expand fg from [n_frames, 1] to [T, P]
            nonlocal fg_per_frame
            if fg_per_frame.shape[1] == 1:
                fg = fg_per_frame.expand(T, P)
            elif fg_per_frame.shape[1] != P:
                fg = self.torch.nn.functional.interpolate(
                    fg_per_frame.unsqueeze(0).unsqueeze(0),
                    size=(T, P),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).squeeze(0)
            else:
                fg = fg_per_frame

            V_prime, hook_diag = enhance_visual_embeddings(
                V, fg, pf_config, self.torch
            )

            holder["applied"] = True
            holder["diagnostics"] = hook_diag

            return V_prime.reshape(orig_shape).to(orig_dtype)

        projector = getattr(getattr(self.model, "model", None), "multi_modal_projector", None)
        if projector is None:
            projector = getattr(self.model, "multi_modal_projector", None)
        if projector is None:
            raise RuntimeError("LLaVA-OV multi-modal projector not found for positive_feature hook")

        handle = projector.register_forward_hook(projector_hook)
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

        diagnostics["positive_feature_hook_applied"] = holder["applied"]
        diagnostics.update(holder["diagnostics"])
        diagnostics.update({
            "foreground_threshold": pf_config.foreground_threshold,
            "foreground_morph_kernel": pf_config.foreground_morph_kernel,
            "foreground_return_soft": pf_config.foreground_return_soft,
            "foreground_pair_fusion": pf_config.foreground_pair_fusion,
            "foreground_pool_avg_weight": pf_config.foreground_pool_avg_weight,
        })
        if not holder["applied"]:
            raise RuntimeError("LLaVA-OV positive_feature hook did not modify projector output")
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
        batch_videos = []
        for video_frames, prompt in zip(batch_video_frames, prompts):
            conversation, video = self._frames_to_conversation(video_frames, prompt)
            batch_conversations.append(conversation)
            batch_videos.append(video)

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
            videos=batch_videos,
            return_tensors="pt",
            padding=True,
        )
        inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()
        common_shapes = {
            key: list(value.shape) for key, value in inputs.items() if hasattr(value, "shape")
        }
        self._generation_diagnostics = [
            {
                "rendered_prompt": text,
                "model_input_keys": sorted(inputs),
                "model_input_shapes": common_shapes,
                "vision_tensor_supplied": any(
                    inputs.get(key) is not None for key in ("pixel_values", "pixel_values_videos")
                ),
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

    def prepare_branch(self, video_frames: np.ndarray, prompt: str, branch: str, **kwargs):
        supported_branches = {
            "original",
            "tcd_negative",
            "spatial_negative",
            "temporal_homogenized",
        }
        if branch not in supported_branches:
            raise NotImplementedError(f"LLaVA-OV adapter does not support branch {branch}")
        started = time.perf_counter()
        inputs = self._prepare_inputs(video_frames, prompt)
        input_shapes = {
            key: list(value.shape)
            for key, value in inputs.items()
            if hasattr(value, "shape")
        }
        return {
            "branch": branch,
            "model_inputs": inputs,
            "past_key_values": None,
            "generated_count": 0,
            "profile": bool(kwargs.get("profile", False)),
            "preserve_logits_on_device": bool(kwargs.get("preserve_logits_on_device", False)),
            "frame_count": int(len(video_frames)),
            "attention_layers": tuple(kwargs.get("attention_layers", self._season_attention_layers)),
            "homogenization_beta": float(kwargs.get("beta", 0.0)),
            "reference_state": kwargs.get("reference_state"),
            "vision_layer_contexts": {},
            "diagnostics": {
                "frame_count": int(len(video_frames)),
                "preprocessing_seconds": time.perf_counter() - started,
                "input_shapes": input_shapes,
                "decode_steps": [],
                "decode_call_count": 0,
                "cache_hit_steps": 0,
                "vision_inputs_supplied_steps": 0,
                "cuda_sync_seconds": 0.0,
                "vision_hook_calls": 0,
                "vision_context_layers": [],
                "attention_layers_used": [],
                "visual_token_spans": [],
                **dict(getattr(self, "_last_input_audit", {})),
            },
        }

    def enable_season_attention(self, attention_layers: tuple[int, ...]) -> None:
        self._season_attention_layers = tuple(attention_layers)

    def _attention_configs(self):
        configs = []
        for config in (
            getattr(self.model, "config", None),
            getattr(getattr(self.model, "language_model", None), "config", None),
            getattr(getattr(self.model, "model", None), "config", None),
        ):
            if config is not None and id(config) not in {id(item) for item in configs}:
                configs.append(config)
        return configs

    @contextmanager
    def _temporary_attention_implementation(self, implementation: str):
        configs = self._attention_configs()
        originals = [getattr(config, "_attn_implementation", None) for config in configs]
        setter = getattr(self.model, "set_attn_implementation", None)
        restore_value = next((value for value in originals if value), "sdpa")
        try:
            if callable(setter):
                setter(implementation)
            for config in configs:
                setattr(config, "_attn_implementation", implementation)
            yield
        finally:
            if callable(setter):
                setter(restore_value)
            for config, original in zip(configs, originals):
                setattr(config, "_attn_implementation", original or restore_value)

    def _vision_layers(self):
        roots = (
            getattr(self.model, "vision_tower", None),
            getattr(getattr(self.model, "model", None), "vision_tower", None),
        )
        paths = (
            ("vision_model", "encoder", "layers"),
            ("encoder", "layers"),
            ("vision_model", "encoder", "layer"),
            ("encoder", "layer"),
        )
        for root in roots:
            if root is None:
                continue
            for path in paths:
                value = root
                for name in path:
                    value = getattr(value, name, None)
                    if value is None:
                        break
                if value is not None and hasattr(value, "__len__"):
                    return value
        raise RuntimeError("LLaVA-OV vision encoder layers could not be located")

    @contextmanager
    def _season_vision_hooks(self, state):
        branch = state["branch"]
        if branch not in {"original", "temporal_homogenized"}:
            yield
            return

        layers = self._vision_layers()
        handles = []
        for layer_index, layer in enumerate(layers):
            if branch == "original":
                def capture_hook(module, module_inputs, module_output, index=layer_index):
                    hidden = unwrap_hidden_states(module_output)
                    context = frame_mean_context(hidden, state["frame_count"])
                    state["vision_layer_contexts"][index] = context.detach().to("cpu")
                    state["diagnostics"]["vision_hook_calls"] += 1
                    return module_output

                handles.append(layer.register_forward_hook(capture_hook))
            else:
                def homogenize_hook(module, module_inputs, module_output, index=layer_index):
                    reference = state.get("reference_state")
                    if reference is None:
                        raise RuntimeError("SEASON temporal branch is missing original reference state")
                    context = reference["vision_layer_contexts"].get(index)
                    if context is None:
                        raise RuntimeError(
                            f"SEASON original branch did not capture vision layer {index}"
                        )
                    hidden = unwrap_hidden_states(module_output)
                    blended = blend_temporal_hidden(
                        hidden,
                        context,
                        state["homogenization_beta"],
                    )
                    state["diagnostics"]["vision_hook_calls"] += 1
                    return replace_hidden_states(module_output, blended)

                handles.append(layer.register_forward_hook(homogenize_hook))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()
            state["diagnostics"]["vision_context_layers"] = sorted(
                state["vision_layer_contexts"]
                if branch == "original"
                else state["reference_state"]["vision_layer_contexts"]
            )

    def _visual_token_spans(self, state) -> list[tuple[int, int]]:
        input_ids = state["model_inputs"]["input_ids"][0]
        token_id = getattr(self.model.config, "video_token_index", None)
        if token_id is None:
            token_id = getattr(self.processor, "video_token_id", None)
        if token_id is None:
            raise RuntimeError("LLaVA-OV config does not expose video_token_index")
        positions = (input_ids == int(token_id)).nonzero(as_tuple=False).flatten().tolist()
        if not positions:
            raise RuntimeError("LLaVA-OV prompt contains no expanded video tokens")
        spans = []
        start = previous = positions[0]
        for position in positions[1:]:
            if position != previous + 1:
                spans.append((start, previous + 1))
                start = position
            previous = position
        spans.append((start, previous + 1))
        if len(spans) == 1 and state["frame_count"] > 1:
            start, end = spans[0]
            total = end - start
            base = total // state["frame_count"]
            remainder = total % state["frame_count"]
            if base <= 0:
                raise RuntimeError(
                    "Unable to map decoder attention to frames: video-token block "
                    f"has {total} tokens for {state['frame_count']} frames"
                )
            split_spans = []
            cursor = start
            for frame_index in range(state["frame_count"]):
                size = base + (1 if frame_index < remainder else 0)
                split_spans.append((cursor, cursor + size))
                cursor += size
            state["diagnostics"]["visual_token_span_strategy"] = "single_contiguous_block_split_by_frame_order"
            return split_spans
        if len(spans) != state["frame_count"]:
            raise RuntimeError(
                "Unable to map decoder attention to frames: "
                f"found {len(spans)} image-token spans for {state['frame_count']} frames"
            )
        state["diagnostics"]["visual_token_span_strategy"] = "one_contiguous_span_per_frame"
        return spans

    def _extract_frame_attention(self, state, attentions) -> np.ndarray:
        if attentions is None:
            raise RuntimeError(
                "LLaVA-OV returned no decoder attentions; SEASON requires eager attention"
            )
        spans = self._visual_token_spans(state)
        selected = []
        used = []
        for layer_index in state["attention_layers"]:
            if layer_index < 0 or layer_index >= len(attentions):
                raise RuntimeError(
                    f"SEASON attention layer {layer_index} is outside decoder depth {len(attentions)}"
                )
            layer = attentions[layer_index]
            if layer is None or layer.ndim != 4:
                raise RuntimeError(f"Decoder attention layer {layer_index} has invalid shape")
            key_length = int(layer.shape[-1])
            if spans[-1][1] > key_length:
                raise RuntimeError(
                    f"Image token span ends at {spans[-1][1]}, attention key length is {key_length}"
                )
            query = layer[0, :, -1, :].float()
            per_frame = self.torch.stack([
                query[:, start:end].sum(dim=-1) for start, end in spans
            ], dim=-1)
            selected.append(per_frame.detach().cpu().numpy()[:, :, None])
            used.append(layer_index)
        values = np.stack(selected, axis=0)
        state["diagnostics"]["attention_layers_used"] = used
        state["diagnostics"]["visual_token_spans"] = [list(span) for span in spans]
        return frame_attention(values)

    def _profile_sync(self, state) -> None:
        if state.get("profile") and self.torch.cuda.is_available():
            started = time.perf_counter()
            self.torch.cuda.synchronize()
            state["diagnostics"]["cuda_sync_seconds"] += time.perf_counter() - started

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
        step_diagnostics = {
            "step": state["diagnostics"]["decode_call_count"],
            "past_before": state["past_key_values"] is not None,
        }
        self._profile_sync(state)
        started = time.perf_counter()
        self._sync_generated_tokens(state, token_ids)
        self._profile_sync(state)
        step_diagnostics["sync_generated_seconds"] = time.perf_counter() - started
        inputs = state["model_inputs"]
        is_first_iteration = state["past_key_values"] is None
        started = time.perf_counter()
        vision_supplied = False
        prefill_forward_seconds = 0.0
        if output_attentions and is_first_iteration:
            if int(inputs["input_ids"].shape[1]) < 2:
                raise RuntimeError("SEASON needs at least two prompt tokens for preceding-token attention")
            prefix_ids = inputs["input_ids"][:, :-1]
            prefix_mask = inputs.get("attention_mask")
            if prefix_mask is not None:
                prefix_mask = prefix_mask[:, :-1]
            prefix = self.model.prepare_inputs_for_generation(
                input_ids=prefix_ids,
                past_key_values=None,
                inputs_embeds=None,
                pixel_values=inputs.get("pixel_values"),
                image_sizes=inputs.get("image_sizes"),
                pixel_values_videos=inputs.get("pixel_values_videos"),
                image_sizes_videos=inputs.get("image_sizes_videos"),
                attention_mask=prefix_mask,
                use_cache=True,
                is_first_iteration=True,
            )
            vision_supplied = any(
                prefix.get(key) is not None for key in ("pixel_values", "pixel_values_videos")
            )
            prefill_started = time.perf_counter()
            with self._season_vision_hooks(state), self.torch.inference_mode():
                prefix_outputs = self.model(
                    **prefix,
                    output_attentions=False,
                    return_dict=True,
                )
            prefill_forward_seconds = time.perf_counter() - prefill_started
            state["past_key_values"] = prefix_outputs.past_key_values
            del prefix_outputs
            prepared = self.model.prepare_inputs_for_generation(
                input_ids=inputs["input_ids"][:, -1:],
                past_key_values=state["past_key_values"],
                inputs_embeds=None,
                pixel_values=None,
                image_sizes=inputs.get("image_sizes"),
                pixel_values_videos=None,
                image_sizes_videos=inputs.get("image_sizes_videos"),
                attention_mask=inputs.get("attention_mask"),
                use_cache=True,
                is_first_iteration=False,
            )
        else:
            input_ids = select_decode_input_ids(inputs["input_ids"], state["past_key_values"])
            prepared = self.model.prepare_inputs_for_generation(
                input_ids=input_ids,
                past_key_values=state["past_key_values"],
                inputs_embeds=inputs.get("inputs_embeds"),
                pixel_values=inputs.get("pixel_values") if is_first_iteration else None,
                image_sizes=inputs.get("image_sizes"),
                pixel_values_videos=inputs.get("pixel_values_videos") if is_first_iteration else None,
                image_sizes_videos=inputs.get("image_sizes_videos"),
                attention_mask=inputs.get("attention_mask"),
                use_cache=True,
                is_first_iteration=is_first_iteration,
            )
            vision_supplied = any(
                prepared.get(key) is not None for key in ("pixel_values", "pixel_values_videos")
            )
        self._profile_sync(state)
        step_diagnostics["prepare_inputs_seconds"] = time.perf_counter() - started
        step_diagnostics["input_ids_length"] = int(prepared["input_ids"].shape[1])
        step_diagnostics["attention_mask_length"] = int(prepared["attention_mask"].shape[1]) if prepared.get("attention_mask") is not None else None
        step_diagnostics["vision_inputs_supplied"] = vision_supplied
        step_diagnostics["split_attention_prefill"] = bool(output_attentions and is_first_iteration)
        step_diagnostics["prefill_forward_seconds"] = prefill_forward_seconds
        state["diagnostics"]["decode_call_count"] += 1
        state["diagnostics"]["cache_hit_steps"] += int(step_diagnostics["past_before"])
        state["diagnostics"]["vision_inputs_supplied_steps"] += int(step_diagnostics["vision_inputs_supplied"])
        self._profile_sync(state)
        started = time.perf_counter()
        attention_context = (
            self._temporary_attention_implementation("eager")
            if output_attentions
            else nullcontext()
        )
        with attention_context, self._season_vision_hooks(state), self.torch.inference_mode():
            outputs = self.model(
                **prepared,
                output_attentions=output_attentions,
                return_dict=True,
            )
        self._profile_sync(state)
        step_diagnostics["forward_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        state["past_key_values"] = outputs.past_key_values
        logits = outputs.logits[0, -1].float().detach()
        if not state["preserve_logits_on_device"]:
            logits = logits.cpu().numpy()
        step_diagnostics["cache_update_seconds"] = time.perf_counter() - started
        step_diagnostics["past_after"] = state["past_key_values"] is not None
        if state["profile"]:
            state["diagnostics"]["decode_steps"].append(step_diagnostics)
        decoder_attentions = getattr(outputs, "attentions", None)
        if decoder_attentions is None:
            language_output = getattr(outputs, "language_model_output", None)
            decoder_attentions = getattr(language_output, "attentions", None)
        attention = (
            self._extract_frame_attention(state, decoder_attentions)
            if output_attentions
            else None
        )
        return StepOutput(logits=logits, frame_attention=attention)

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
