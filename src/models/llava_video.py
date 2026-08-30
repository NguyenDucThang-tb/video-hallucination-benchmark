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
        # LLaVA-NeXT's builder defaults to FlashAttention2, which is not
        # available in many managed CUDA environments. SDPA is the portable
        # PyTorch fallback; override with LLAVA_VIDEO_ATTN_IMPLEMENTATION if
        # a site-provided flash-attn build is available.
        attn_implementation = os.environ.get("LLAVA_VIDEO_ATTN_IMPLEMENTATION", "sdpa")
        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(
            self.model_path,
            None,
            model_name=self.model_name,
            # The official builder accepts only float16/bfloat16 strings.
            # BFloat16 is the documented LLaVA-Video inference dtype.
            torch_dtype="bfloat16",
            device_map="auto",
            attn_implementation=attn_implementation,
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

    def _build_inputs(self, video_frames: np.ndarray, prompt: str) -> dict:
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
            "llava_video_attention_implementation": os.environ.get(
                "LLAVA_VIDEO_ATTN_IMPLEMENTATION", "sdpa"
            ),
            "rendered_prompt": rendered_prompt,
            "model_input_keys": ["inputs", "attention_mask", "images", "modalities"],
            "model_input_shapes": {"input_ids": list(input_ids.shape), "video": list(processed.shape)},
            "vision_tensor_supplied": True,
            "video_modality_supplied": True,
            "video_modality": "video",
            "video_frame_count": int(len(video)),
        }
        return {
            # LlavaQwenForCausalLM.generate names its token argument `inputs`,
            # unlike the standard Transformers `input_ids` keyword.
            "inputs": input_ids,
            "attention_mask": attention_mask,
            "images": [processed],
            "modalities": ["video"],
        }

    def generate(self, video_frames: np.ndarray, prompt: str, generation_config: GenerationConfig) -> str:
        inputs = self._build_inputs(video_frames, prompt)
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
        answer = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        return answer.strip()

    def consume_generation_diagnostics(self, expected_count: int) -> list[dict]:
        diagnostics = self._generation_diagnostics[:expected_count]
        self._generation_diagnostics = self._generation_diagnostics[expected_count:]
        return diagnostics + [{} for _ in range(expected_count - len(diagnostics))]

    def _sync_branch_tokens(self, state, token_ids: list[int]) -> None:
        if len(token_ids) <= state["generated_count"]:
            return
        new_token_ids = token_ids[state["generated_count"] :]
        inputs = state["model_inputs"]
        token_tensor = self.torch.tensor(
            [new_token_ids], device=self.device, dtype=inputs["input_ids"].dtype
        )
        inputs["input_ids"] = self.torch.cat([inputs["input_ids"], token_tensor], dim=1)
        extra_mask = self.torch.ones(
            (1, len(new_token_ids)),
            device=self.device,
            dtype=inputs["attention_mask"].dtype,
        )
        inputs["attention_mask"] = self.torch.cat(
            [inputs["attention_mask"], extra_mask], dim=1
        )
        state["generated_count"] = len(token_ids)

    def token_id_to_text(self, token_id: int) -> str:
        return self.tokenizer.decode([token_id], skip_special_tokens=True)

    def decode_token_ids(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()

    def is_eos(self, token_id: int) -> bool:
        return token_id == getattr(self.tokenizer, "eos_token_id", None)

    def _checkpoint_is_local(self, checkpoint: str) -> bool:
        return Path(checkpoint).expanduser().exists()

    def _ensure_dino_loaded(self, checkpoint: str, device: str = "cpu"):
        if getattr(self, "_dino_model", None) is not None:
            return self._dino_processor, self._dino_model
        from transformers import AutoImageProcessor, AutoModel

        local_only = self._checkpoint_is_local(checkpoint)
        self._dino_processor = AutoImageProcessor.from_pretrained(
            checkpoint, local_files_only=local_only
        )
        if device == "cpu":
            self._dino_model = AutoModel.from_pretrained(
                checkpoint, local_files_only=local_only
            ).to("cpu").eval()
        else:
            self._dino_model = AutoModel.from_pretrained(
                checkpoint,
                torch_dtype=self.model_dtype,
                device_map="auto",
                local_files_only=local_only,
            ).eval()
        return self._dino_processor, self._dino_model

    def _ensure_birefnet_loaded(self, checkpoint: str = "ZhengPeng7/BiRefNet", device: str = "cpu"):
        """Lazy-load BiRefNet model for foreground segmentation."""
        return ensure_birefnet_loaded(self.__dict__, checkpoint, device, self.torch)

    def _compute_dino_saliency(self, video_frames, checkpoint: str, device: str = "cpu"):
        from PIL import Image

        processor, dino_model = self._ensure_dino_loaded(checkpoint, device)
        images = [Image.fromarray(np.asarray(frame, dtype=np.uint8)) for frame in video_frames]
        dino_inputs = processor(images=images, return_tensors="pt")
        dino_device = next(dino_model.parameters()).device
        dino_inputs = {
            key: value.to(dino_device) if hasattr(value, "to") else value
            for key, value in dino_inputs.items()
        }
        with self.torch.inference_mode():
            outputs = dino_model(**dino_inputs, return_dict=True)
        tokens = outputs.last_hidden_state[:, 1:, :].float()
        scores = tokens.norm(dim=-1)
        scores = scores / (scores.max(dim=1, keepdim=True).values + 1e-6)
        frame_scores = scores.mean(dim=1).cpu().numpy().astype(np.float32)
        return frame_scores, {
            "dino_loaded": True,
            "dino_device": str(dino_device),
            "dino_patch_tokens": int(scores.shape[1]),
            "dino_frame_saliency_mean": float(frame_scores.mean()),
        }

    def _scale_dino_projector_output(self, output, frame_scales):
        if not hasattr(output, "shape"):
            return output, False
        if output.ndim == 2:
            token_axis = 0
        elif output.ndim >= 3:
            token_axis = output.ndim - 2
        else:
            return output, False
        token_count = int(output.shape[token_axis])
        if token_count <= 0:
            return output, False
        scale_indices = np.rint(
            np.linspace(0, len(frame_scales) - 1, token_count)
        ).astype(int)
        scale = self.torch.as_tensor(
            np.asarray(frame_scales, dtype=np.float32)[scale_indices],
            device=output.device,
            dtype=output.dtype,
        )
        view_shape = [1] * output.ndim
        view_shape[token_axis] = token_count
        return output * scale.view(*view_shape), True

    def generate_dino_heal(self, video_frames, prompt: str, generation_config: GenerationConfig, config: dict):
        dino_config = DINOHealConfig(
            visual_weight=float(config.get("visual_weight", 0.3)),
            saliency_weight=float(config.get("saliency_weight", 0.7)),
            require_dino=bool(config.get("require_dino", True)),
        )
        checkpoint = config.get("dino_checkpoint", "facebook/dinov2-large")
        dino_device = str(config.get("dino_device", "cpu"))
        frame_saliency, diagnostics = self._compute_dino_saliency(
            video_frames, checkpoint, dino_device
        )

        # This is the LLaVA-Video equivalent of the repository's existing
        # frame-level DINO-HEAL path; it scales projector tokens by frame order.
        features = np.zeros((len(video_frames), 1, 1), dtype=np.float32)
        fused = fuse_saliency(
            features,
            frame_saliency[:, None],
            dino_config,
        )
        frame_scales = np.maximum(fused[..., 0].mean(axis=1), 0.0).astype(np.float32)
        holder = {"applied": False}

        def projector_hook(module, module_inputs, module_output):
            scaled, applied = self._scale_dino_projector_output(module_output, 1.0 + frame_scales)
            holder["applied"] = holder["applied"] or applied
            return scaled

        vision_model = None
        get_model = getattr(self.model, "get_model", None)
        if callable(get_model):
            vision_model = getattr(get_model(), "mm_projector", None)
        vision_model = vision_model or getattr(self.model, "mm_projector", None)
        if vision_model is None:
            raise RuntimeError("LLaVA-Video projector not found for DINO-HEAL hook")

        handle = vision_model.register_forward_hook(projector_hook)
        try:
            inputs = self._build_inputs(video_frames, prompt)
            diagnostics.update(self._last_input_audit)
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
        finally:
            handle.remove()

        diagnostics.update({
            "dino_hook_applied": holder["applied"],
            "dino_scale_mean": float(frame_scales.mean()),
            "dino_scale_max": float(frame_scales.max()),
        })
        if not holder["applied"] and dino_config.require_dino:
            raise RuntimeError("LLaVA-Video DINO-HEAL hook did not modify projector output")
        answer = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return answer, diagnostics

    def generate_positive_feature(
        self,
        video_frames,
        prompt: str,
        generation_config: GenerationConfig,
        config: dict,
    ):
        """Generate text with BiRefNet-based positive visual-feature enhancement.

        Hooks into the mm_projector output to apply:
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
        )
        n_frames = len(video_frames)
        diagnostics: dict = {
            "positive_feature_mode": "birefnet_projector_hook" if use_birefnet else "dino_projector_hook",
            "positive_feature_hook_applied": False,
            "use_birefnet": use_birefnet,
        }

        # Compute foreground saliency
        # For LLaVA-Video, we don't know exact T,P until the projector fires,
        # so we pre-compute fg at frame-level and resize inside the hook.
        fg_per_frame = None
        try:
            if use_birefnet:
                birefnet_model, birefnet_transform = self._ensure_birefnet_loaded(
                    pf_config.birefnet_checkpoint, pf_config.saliency_device
                )
                # Pre-compute at n_frames x 1 (will be resized in hook)
                fg_per_frame = compute_birefnet_foreground(
                    video_frames, n_frames, 1, birefnet_model, birefnet_transform,
                    self.torch, self.device,
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
            fg_per_frame = self.torch.ones(n_frames, 1, device=self.device, dtype=self.torch.float32)
            diagnostics["saliency_fallback"] = repr(exc)

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

        projector = None
        get_model = getattr(self.model, "get_model", None)
        if callable(get_model):
            projector = getattr(get_model(), "mm_projector", None)
        projector = projector or getattr(self.model, "mm_projector", None)
        if projector is None:
            raise RuntimeError("LLaVA-Video projector not found for positive_feature hook")

        handle = projector.register_forward_hook(projector_hook)
        try:
            inputs = self._build_inputs(video_frames, prompt)
            diagnostics.update(self._last_input_audit)
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
        finally:
            handle.remove()

        diagnostics["positive_feature_hook_applied"] = holder["applied"]
        diagnostics.update(holder["diagnostics"])
        if not holder["applied"]:
            raise RuntimeError("LLaVA-Video positive_feature hook did not modify projector output")
        answer = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return answer, diagnostics

    @property
    def supports_step_logits(self) -> bool:
        return True

    def enable_season_attention(self, attention_layers: tuple[int, ...]) -> None:
        self._season_attention_layers = tuple(attention_layers)

    def _attention_configs(self):
        configs = []
        for value in (
            getattr(self.model, "config", None),
            getattr(getattr(self.model, "model", None), "config", None),
        ):
            if value is not None and id(value) not in {id(item) for item in configs}:
                configs.append(value)
        return configs

    @contextmanager
    def _temporary_attention_implementation(self, implementation: str):
        configs = self._attention_configs()
        originals = [getattr(config, "_attn_implementation", None) for config in configs]
        setter = getattr(self.model, "set_attn_implementation", None)
        restore = next((value for value in originals if value), "sdpa")
        try:
            if callable(setter):
                setter(implementation)
            for config in configs:
                setattr(config, "_attn_implementation", implementation)
            yield
        finally:
            if callable(setter):
                setter(restore)
            for config, original in zip(configs, originals):
                setattr(config, "_attn_implementation", original or restore)

    def _vision_layers(self):
        roots = []
        getter = getattr(self.model, "get_vision_tower", None)
        if callable(getter):
            roots.append(getter())
        roots.extend((
            getattr(self.model, "vision_tower", None),
            getattr(getattr(self.model, "model", None), "vision_tower", None),
        ))
        for root in roots:
            if root is None:
                continue
            for path in (("vision_tower", "vision_model", "encoder", "layers"),
                         ("vision_model", "encoder", "layers"),
                         ("encoder", "layers")):
                value = root
                for name in path:
                    value = getattr(value, name, None)
                    if value is None:
                        break
                if value is not None and hasattr(value, "__len__"):
                    return value
        raise RuntimeError("LLaVA-Video vision encoder layers could not be located")

    @contextmanager
    def _season_vision_hooks(self, state):
        if not state.get("season_enabled", False):
            yield
            return
        branch = state["branch"]
        if branch not in {"original", "temporal_homogenized"}:
            yield
            return
        handles = []
        layers = self._vision_layers()
        for index, layer in enumerate(layers):
            if branch == "original":
                def capture(module, module_inputs, module_output, layer_index=index):
                    hidden = unwrap_hidden_states(module_output)
                    state["vision_layer_contexts"][layer_index] = frame_mean_context(
                        hidden, state["frame_count"]
                    ).detach().to("cpu")
                    state["diagnostics"]["vision_hook_calls"] += 1
                    return module_output
                handles.append(layer.register_forward_hook(capture))
            else:
                def homogenize(module, module_inputs, module_output, layer_index=index):
                    reference = state.get("reference_state")
                    context = reference["vision_layer_contexts"].get(layer_index)
                    if context is None:
                        raise RuntimeError(f"Missing original vision context for layer {layer_index}")
                    hidden = unwrap_hidden_states(module_output)
                    blended = blend_temporal_hidden(
                        hidden, context, state["homogenization_beta"]
                    )
                    state["diagnostics"]["vision_hook_calls"] += 1
                    return replace_hidden_states(module_output, blended)
                handles.append(layer.register_forward_hook(homogenize))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def _visual_token_spans(self, state, key_length: int):
        input_ids = state["model_inputs"]["input_ids"][0]
        image_positions = (input_ids == int(self.IMAGE_TOKEN_INDEX)).nonzero(
            as_tuple=False
        ).flatten().tolist()
        if not image_positions:
            raise RuntimeError("LLaVA-Video prompt contains no image token")
        image_position = image_positions[0]
        text_after = int(input_ids.shape[0]) - image_position - 1
        visual_count = key_length - text_after - image_position
        if visual_count <= 0:
            raise RuntimeError("Unable to determine LLaVA-Video visual token count")
        boundaries = np.rint(
            np.linspace(0, visual_count, state["frame_count"] + 1)
        ).astype(int)
        return [
            (image_position + int(boundaries[i]), image_position + int(boundaries[i + 1]))
            for i in range(state["frame_count"])
        ]

    def _extract_frame_attention(self, state, attentions):
        if attentions is None:
            raise RuntimeError("LLaVA-Video returned no decoder attentions for SEASON")
        key_length = int(attentions[0].shape[-1])
        spans = self._visual_token_spans(state, key_length)
        selected = []
        for layer_index in state["attention_layers"]:
            layer = attentions[layer_index]
            query = layer[0, :, -1, :].float()
            selected.append(np.asarray([
                query[:, start:end].sum(dim=-1).detach().cpu().numpy()
                for start, end in spans
            ]).T[:, :, None])
        state["diagnostics"]["attention_layers_used"] = list(state["attention_layers"])
        state["diagnostics"]["visual_token_spans"] = [list(span) for span in spans]
        return frame_attention(np.stack(selected, axis=0))

    def prepare_branch(self, video_frames: np.ndarray, prompt: str, branch: str, **kwargs):
        supported = {
            "original",
            "tcd_negative",
            "spatial_negative",
            "temporal_homogenized",
        }
        if branch not in supported:
            raise NotImplementedError(f"LLaVA-Video adapter does not support branch {branch}")
        started = time.perf_counter()
        generated_inputs = self._build_inputs(video_frames, prompt)
        season_enabled = "attention_layers" in kwargs
        return {
            "branch": branch,
            "model_inputs": {
                "input_ids": generated_inputs["inputs"],
                "attention_mask": generated_inputs["attention_mask"],
                "images": generated_inputs["images"],
                "modalities": generated_inputs["modalities"],
            },
            "past_key_values": None,
            "generated_count": 0,
            "preserve_logits_on_device": bool(kwargs.get("preserve_logits_on_device", False)),
            "season_enabled": season_enabled,
            "frame_count": int(len(video_frames)),
            "attention_layers": tuple(
                kwargs.get("attention_layers", getattr(self, "_season_attention_layers", ()))
            ),
            "homogenization_beta": float(kwargs.get("beta", 0.0)),
            "reference_state": kwargs.get("reference_state"),
            "vision_layer_contexts": {},
            "diagnostics": {
                "branch": branch,
                "frame_count": int(len(video_frames)),
                "preprocessing_seconds": time.perf_counter() - started,
                "decode_call_count": 0,
                "cache_hit_steps": 0,
                "vision_inputs_supplied_steps": 0,
                "vision_hook_calls": 0,
                **dict(getattr(self, "_last_input_audit", {})),
            },
        }

    def _prepare_multimodal_prefill(self, state):
        inputs = state["model_inputs"]
        prepared = self.model.prepare_inputs_labels_for_multimodal(
            inputs["input_ids"],
            None,
            inputs["attention_mask"],
            None,
            None,
            inputs["images"],
            inputs["modalities"],
            None,
        )
        if not isinstance(prepared, (tuple, list)) or len(prepared) != 6:
            raise RuntimeError(
                "LLaVA-Video multimodal prefill returned an unexpected result"
            )
        _, position_ids, attention_mask, _, inputs_embeds, _ = prepared
        if inputs_embeds is None:
            raise RuntimeError("LLaVA-Video multimodal prefill returned no inputs_embeds")
        if attention_mask is None:
            attention_mask = self.torch.ones(
                inputs_embeds.shape[:2],
                device=inputs_embeds.device,
                dtype=inputs["attention_mask"].dtype,
            )
        if int(attention_mask.shape[1]) != int(inputs_embeds.shape[1]):
            raise RuntimeError(
                "LLaVA-Video multimodal prefill produced mismatched attention mask "
                f"({attention_mask.shape[1]}) and embeddings ({inputs_embeds.shape[1]})"
            )
        inputs["attention_mask"] = attention_mask
        state["diagnostics"]["text_prompt_token_count"] = int(
            inputs["input_ids"].shape[1]
        )
        state["diagnostics"]["multimodal_prefill_token_count"] = int(
            inputs_embeds.shape[1]
        )
        state["diagnostics"]["multimodal_attention_mask_count"] = int(
            attention_mask.shape[1]
        )
        return inputs_embeds, position_ids

    def decode_step(self, state, token_ids: list[int], output_attentions: bool = False) -> StepOutput:
        self._sync_branch_tokens(state, token_ids)
        inputs = state["model_inputs"]
        first_step = state["past_key_values"] is None
        attention_context = (
            self._temporary_attention_implementation("eager")
            if output_attentions
            else nullcontext()
        )
        with attention_context, self._season_vision_hooks(state), self.torch.inference_mode():
            if first_step:
                inputs_embeds, position_ids = self._prepare_multimodal_prefill(state)
                kwargs = {
                    "input_ids": None,
                    "inputs_embeds": inputs_embeds,
                    "position_ids": position_ids,
                    "attention_mask": inputs["attention_mask"],
                    "past_key_values": None,
                    "use_cache": True,
                    "return_dict": True,
                    "output_attentions": output_attentions,
                }
            else:
                kwargs = {
                    "input_ids": select_decode_input_ids(
                        inputs["input_ids"], state["past_key_values"]
                    ),
                    "attention_mask": inputs["attention_mask"],
                    "past_key_values": state["past_key_values"],
                    "use_cache": True,
                    "return_dict": True,
                    "output_attentions": output_attentions,
                }
            outputs = self.model(**kwargs)
        state["past_key_values"] = outputs.past_key_values
        state["diagnostics"]["decode_call_count"] += 1
        state["diagnostics"]["cache_hit_steps"] += int(not first_step)
        state["diagnostics"]["vision_inputs_supplied_steps"] += int(first_step)
        logits = outputs.logits[0, -1].float().detach()
        if not state["preserve_logits_on_device"]:
            logits = logits.cpu().numpy()
        attention = self._extract_frame_attention(state, outputs.attentions) if output_attentions else None
        return StepOutput(logits=logits, frame_attention=attention)

    @property
    def supports_frame_attention(self) -> bool:
        return True

    @property
    def supports_vision_layer_hooks(self) -> bool:
        return True

    @property
    def supports_positive_feature_hooks(self) -> bool:
        return True
