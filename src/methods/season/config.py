from dataclasses import dataclass


@dataclass(frozen=True)
class SeasonConfig:
    alpha: float = 1.0
    homogenization_beta: float = 0.33
    spatial_noise_std: float = 0.1
    attention_layers: tuple[int, ...] = (20, 21, 22, 23)
    expected_frame_count: int = 8
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if self.alpha < 0:
            raise ValueError("SEASON alpha must be non-negative")
        if not 0.0 <= self.homogenization_beta <= 1.0:
            raise ValueError("SEASON homogenization_beta must be in [0, 1]")
        if self.spatial_noise_std < 0:
            raise ValueError("SEASON spatial_noise_std must be non-negative")
        if not self.attention_layers:
            raise ValueError("SEASON requires at least one decoder attention layer")
        if self.expected_frame_count <= 0:
            raise ValueError("SEASON expected_frame_count must be positive")
        if self.epsilon <= 0:
            raise ValueError("SEASON epsilon must be positive")
