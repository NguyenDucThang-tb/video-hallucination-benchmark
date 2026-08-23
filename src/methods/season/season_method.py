from __future__ import annotations

import hashlib
from dataclasses import asdict

import numpy as np

from src.methods.base import InferenceMethod, MethodOutput
from src.models.base import GenerationConfig

from .attention_diagnosis import diagnostic_weights
from .config import SeasonConfig
from .contrastive_decoding import season_logits
from .negative_video import spatial_negative


class SeasonMethod(InferenceMethod):
    name = "season"
    implementation_version = "paper-grounded-local-v1"

    def __init__(self, model, config=None):
        super().__init__(model, config)
        values = dict(config or {})
        if "attention_layers" in values:
            values["attention_layers"] = tuple(values["attention_layers"])
        values.pop("batch_size", None)
        values.pop("positive_alpha", None)
        values.pop("positive_alpha_spatial", None)
        values.pop("positive_beta_temporal", None)
        self.season = SeasonConfig(**values)
        enable_attention = getattr(self.model, "enable_season_attention", None)
        if enable_attention is not None:
            enable_attention(self.season.attention_layers)

    def generate(self, video_frames: np.ndarray, prompt: str, generation_config: GenerationConfig) -> MethodOutput:
        required = (
            self.model.supports_step_logits,
            self.model.supports_frame_attention,
            self.model.supports_vision_layer_hooks,
        )
        if not all(required):
            raise RuntimeError(f"SEASON unsupported: adapter {self.model.name} lacks logits/attention/vision hooks")
        frame_count = int(len(video_frames))
        if frame_count != self.season.expected_frame_count:
            raise ValueError(
                f"SEASON requires exactly {self.season.expected_frame_count} frames, got {frame_count}"
            )
        seed = int.from_bytes(hashlib.sha256(prompt.encode()).digest()[:4], "big")
        spatial_frames = spatial_negative(video_frames, self.season.spatial_noise_std, seed)
        original_state = self.model.prepare_branch(
            video_frames,
            prompt,
            branch="original",
            attention_layers=self.season.attention_layers,
            preserve_logits_on_device=True,
        )
        states = {
            "original": original_state,
            "spatial": self.model.prepare_branch(
                spatial_frames,
                prompt,
                branch="spatial_negative",
                attention_layers=self.season.attention_layers,
                preserve_logits_on_device=True,
            ),
            "temporal": self.model.prepare_branch(
                video_frames,
                prompt,
                branch="temporal_homogenized",
                beta=self.season.homogenization_beta,
                attention_layers=self.season.attention_layers,
                reference_state=original_state,
                preserve_logits_on_device=True,
            ),
        }
        generated: list[int] = []
        trace = []
        for step in range(generation_config.max_new_tokens):
            # Original must execute first so its per-layer contexts are available
            # to the temporal-negative branch during the first vision prefill.
            outputs = {}
            for name in ("original", "spatial", "temporal"):
                outputs[name] = self.model.decode_step(
                    states[name], generated, output_attentions=True
                )
                if outputs[name].frame_attention is None:
                    raise RuntimeError(f"SEASON {name} branch returned no frame attention")
            w_s, w_t, d_s, d_t = diagnostic_weights(
                outputs["original"].frame_attention,
                outputs["spatial"].frame_attention,
                outputs["temporal"].frame_attention,
                self.season.epsilon,
            )
            logits = season_logits(
                outputs["original"].logits, outputs["spatial"].logits,
                outputs["temporal"].logits, self.season.alpha, w_s, w_t,
            )
            if logits.__class__.__module__.startswith("torch"):
                token = int(logits.argmax().item())
            else:
                token = int(np.argmax(logits))
            generated.append(token)
            trace.append({
                "step": step,
                "token_id": token,
                "w_spatial": w_s,
                "w_temporal": w_t,
                "d_spatial": d_s,
                "d_temporal": d_t,
                "original_frame_attention": np.asarray(
                    outputs["original"].frame_attention
                ).tolist(),
                "spatial_frame_attention": np.asarray(
                    outputs["spatial"].frame_attention
                ).tolist(),
                "temporal_frame_attention": np.asarray(
                    outputs["temporal"].frame_attention
                ).tolist(),
            })
            if getattr(self.model, "is_eos", lambda _: False)(token):
                break
        text = self.model.decode_token_ids(generated)
        return MethodOutput(text, {
            "season_implementation": self.implementation_version,
            "season_config": asdict(self.season),
            "spatial_negative_seed": seed,
            "spatial_negative_mean_absolute_delta": float(
                np.abs(spatial_frames.astype(np.float32) - video_frames.astype(np.float32)).mean()
            ),
            "frame_count": frame_count,
            "generated_token_count": len(generated),
            "stopped_on_eos": bool(generated and self.model.is_eos(generated[-1])),
            "token_diagnostics": trace,
            "branch_diagnostics": {
                name: state.get("diagnostics", {}) for name, state in states.items()
            },
        })
