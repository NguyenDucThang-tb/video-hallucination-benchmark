from dataclasses import dataclass


@dataclass(frozen=True)
class SeasonConfig:
    alpha: float = 1.0
    homogenization_beta: float = 0.33
    spatial_noise_std: float = 0.1
    positive_alpha: float = 0.4
    positive_alpha_spatial: float = 0.4
    positive_beta_temporal: float = 0.4
    attention_layers: tuple[int, ...] = (20, 21, 22, 23)
    epsilon: float = 1e-8
