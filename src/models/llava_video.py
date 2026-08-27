from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from .base import GenerationConfig, ModelAdapter, StepOutput, select_decode_input_ids
from src.methods.dino_heal.fusion import DINOHealConfig, fuse_saliency


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

    def prepare_branch(self, video_frames: np.ndarray, prompt: str, branch: str, **kwargs):
        if branch not in {"original", "tcd_negative"}:
            raise NotImplementedError(f"LLaVA-Video adapter does not support branch {branch}")

        started = time.perf_counter()
        generated_inputs = self._build_inputs(video_frames, prompt)
        # The custom LLaVA Qwen forward path uses `input_ids`; only its
        # generate() convenience wrapper calls that argument `inputs`.
        model_inputs = {
            "input_ids": generated_inputs["inputs"],
            "attention_mask": generated_inputs["attention_mask"],
            "images": generated_inputs["images"],
            "modalities": generated_inputs["modalities"],
        }
        return {
            "branch": branch,
            "model_inputs": model_inputs,
            "past_key_values": None,
            "generated_count": 0,
            "preserve_logits_on_device": bool(kwargs.get("preserve_logits_on_device", False)),
            "diagnostics": {
                "branch": branch,
                "frame_count": int(len(video_frames)),
                "preprocessing_seconds": time.perf_counter() - started,
                "decode_call_count": 0,
                "cache_hit_steps": 0,
                "vision_inputs_supplied_steps": 0,
                **dict(getattr(self, "_last_input_audit", {})),
            },
        }

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

    def decode_step(self, state, token_ids: list[int], output_attentions: bool = False) -> StepOutput:
        if output_attentions:
            raise NotImplementedError("LLaVA-Video TCD does not expose decoder attention diagnostics")

        self._sync_branch_tokens(state, token_ids)
        inputs = state["model_inputs"]
        first_step = state["past_key_values"] is None
        input_ids = select_decode_input_ids(inputs["input_ids"], state["past_key_values"])
        model_kwargs = {
            "input_ids": input_ids,
            "attention_mask": inputs["attention_mask"],
            "past_key_values": state["past_key_values"],
            "use_cache": True,
            "return_dict": True,
            "images": inputs["images"] if first_step else None,
            "modalities": inputs["modalities"],
        }
        with self.torch.inference_mode():
            outputs = self.model(**model_kwargs)

        state["past_key_values"] = outputs.past_key_values
        state["diagnostics"]["decode_call_count"] += 1
        state["diagnostics"]["cache_hit_steps"] += int(not first_step)
        state["diagnostics"]["vision_inputs_supplied_steps"] += int(first_step)
        logits = outputs.logits[0, -1].float().detach()
        if not state["preserve_logits_on_device"]:
            logits = logits.cpu().numpy()
        return StepOutput(logits=logits)

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

    @property
    def supports_step_logits(self) -> bool:
        return True
