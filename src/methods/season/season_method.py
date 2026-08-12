from __future__ import annotations

import hashlib

import numpy as np

from src.methods.base import InferenceMethod, MethodOutput
from src.models.base import GenerationConfig

from .attention_diagnosis import diagnostic_weights
from .config import SeasonConfig
from .contrastive_decoding import season_logits
from .negative_video import spatial_negative


class SeasonMethod(InferenceMethod):
    name = "season"

    def __init__(self, model, config=None):
        super().__init__(model, config)
        values = dict(config or {})
        if "attention_layers" in values:
            values["attention_layers"] = tuple(values["attention_layers"])
        self.season = SeasonConfig(**values)

    def generate(self, video_frames: np.ndarray, prompt: str, generation_config: GenerationConfig) -> MethodOutput:
        required = (
            self.model.supports_step_logits,
            self.model.supports_frame_attention,
            self.model.supports_vision_layer_hooks,
        )
        if not all(required):
            raise RuntimeError(f"SEASON unsupported: adapter {self.model.name} lacks logits/attention/vision hooks")
        seed = int.from_bytes(hashlib.sha256(prompt.encode()).digest()[:4], "big")
        spatial_frames = spatial_negative(video_frames, self.season.spatial_noise_std, seed)
        states = {
            "original": self.model.prepare_branch(video_frames, prompt, branch="original"),
            "spatial": self.model.prepare_branch(spatial_frames, prompt, branch="spatial_negative"),
            "temporal": self.model.prepare_branch(
                video_frames, prompt, branch="temporal_homogenized",
                beta=self.season.homogenization_beta,
            ),
        }
        generated: list[int] = []
        trace = []
        for step in range(generation_config.max_new_tokens):
            outputs = {
                name: self.model.decode_step(state, generated, output_attentions=True)
                for name, state in states.items()
            }
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
            token = int(np.argmax(logits))
            generated.append(token)
            trace.append({"step": step, "w_spatial": w_s, "w_temporal": w_t, "d_spatial": d_s, "d_temporal": d_t})
            if getattr(self.model, "is_eos", lambda _: False)(token):
                break
        text = "".join(self.model.token_id_to_text(token) for token in generated)
        return MethodOutput(text, {"token_diagnostics": trace})
