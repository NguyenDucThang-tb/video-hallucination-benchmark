__all__ = [
    "PositiveFeatureMethod",
    "enhance_output_by_frame_saliency",
]


def __getattr__(name: str):
    if name == "PositiveFeatureMethod":
        from .positive_feature_method import PositiveFeatureMethod

        return PositiveFeatureMethod
    if name == "enhance_output_by_frame_saliency":
        from .enhancement import enhance_output_by_frame_saliency

        return enhance_output_by_frame_saliency
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
