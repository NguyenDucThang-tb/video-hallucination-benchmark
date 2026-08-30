from .enhancement import enhance_output_by_frame_saliency, enhance_tensor_by_frame_saliency

__all__ = [
    "PositiveFeatureMethod",
    "enhance_output_by_frame_saliency",
    "enhance_tensor_by_frame_saliency",
]


def __getattr__(name: str):
    if name == "PositiveFeatureMethod":
        from .positive_feature_method import PositiveFeatureMethod

        return PositiveFeatureMethod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
