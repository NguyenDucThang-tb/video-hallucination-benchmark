from __future__ import annotations

from typing import Protocol


class SeasonCapableAdapter(Protocol):
    """Contract required for a paper-faithful SEASON implementation."""

    supports_step_logits: bool
    supports_frame_attention: bool
    supports_vision_layer_hooks: bool
    supports_positive_feature_hooks: bool

    def prepare_branch(self, video_frames, prompt: str, branch: str, **kwargs): ...
    def prepare_positive_branch(self, video_frames, prompt: str, foreground, **kwargs): ...
    def decode_step(self, state, token_ids: list[int], output_attentions: bool = False): ...
    def token_id_to_text(self, token_id: int) -> str: ...
