import pytest

from scripts.run_benchmark import resolve_method_config
from src.experiments.positive_grid import (
    GridPoint,
    experiment_name,
    finalize_grid_rows,
    positive_feature_grid,
    validate_run_diagnostics,
)
from src.methods.positive_feature.enhancement import PositiveFeatureConfig, enhance_visual_embeddings


def test_positive_feature_grid_has_all_109_unique_ablation_points():
    points = positive_feature_grid()
    assert len(points) == 109
    assert len({(point.alpha, point.alpha_s, point.beta) for point in points}) == 109
    assert {point.ablation for point in points} == {
        "foreground_only", "persistence_only", "temporal_only", "spatial", "full",
    }
    assert len({experiment_name("tune", point) for point in points}) == 109


def test_experiment_method_config_overrides_repository_defaults():
    config = resolve_method_config("positive_feature", {
        "method_configs": {"positive_feature": {"alpha": 0.2, "alpha_s": 0.1, "beta": 0.6}}
    })
    assert (config["alpha"], config["alpha_s"], config["beta"]) == (0.2, 0.1, 0.6)
    assert "alpha_spatial" not in config
    assert "beta_temporal" not in config
    assert config["foreground_return_soft"] is True
    assert config["foreground_pair_fusion"] == "mean"
    assert config["foreground_morph_kernel"] == 0
    assert config["foreground_pool_avg_weight"] == 1.0


@pytest.mark.parametrize(
    ("name", "config"),
    [
        ("foreground_only", PositiveFeatureConfig(alpha=0.2, alpha_s=0.0, beta=0.0)),
        ("persistence_only", PositiveFeatureConfig(alpha=0.0, alpha_s=0.2, beta=0.0)),
        ("temporal_only", PositiveFeatureConfig(alpha=0.0, alpha_s=0.0, beta=0.2)),
        ("spatial", PositiveFeatureConfig(alpha=0.2, alpha_s=0.2, beta=0.0)),
        ("full", PositiveFeatureConfig(alpha=0.2, alpha_s=0.2, beta=0.2)),
    ],
)
def test_ablation_diagnostics_match_applied_coefficients(name, config):
    torch = pytest.importorskip("torch")
    values = torch.tensor([
        [[1.0, 0.0], [0.0, 1.0]],
        [[2.0, 0.0], [0.0, 3.0]],
    ])
    foreground = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    enhanced, diagnostics = enhance_visual_embeddings(values, foreground, config, torch)

    assert not torch.allclose(enhanced, values), name
    assert diagnostics["alpha"] == config.alpha
    assert diagnostics["alpha_s"] == config.alpha_s
    assert diagnostics["beta"] == config.beta


def test_zero_coefficients_are_an_exact_identity():
    torch = pytest.importorskip("torch")
    values = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3) + 1
    foreground = torch.ones(2, 2)
    enhanced, diagnostics = enhance_visual_embeddings(
        values, foreground, PositiveFeatureConfig(alpha=0, alpha_s=0, beta=0), torch
    )
    assert torch.equal(enhanced, values)
    assert diagnostics["positive_feature_delta"] == 0.0
    assert diagnostics["positive_feature_direction_delta"] == pytest.approx(0.0)


def test_foreground_context_changes_direction_and_survives_token_normalization():
    torch = pytest.importorskip("torch")
    values = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    foreground = torch.tensor([[1.0, 0.5]])

    enhanced, diagnostics = enhance_visual_embeddings(
        values,
        foreground,
        PositiveFeatureConfig(alpha=0.4, alpha_s=0.0, beta=0.0),
        torch,
    )

    normalized_before = torch.nn.functional.normalize(values, dim=-1)
    normalized_after = torch.nn.functional.normalize(enhanced, dim=-1)
    assert not torch.allclose(normalized_after, normalized_before)
    assert diagnostics["positive_feature_direction_delta"] > 0
    assert diagnostics["foreground_residual_mean_norm"] > 0


def test_persistence_context_changes_direction_and_survives_token_normalization():
    torch = pytest.importorskip("torch")
    values = torch.tensor([
        [[1.0, 0.0], [0.0, 1.0]],
        [[0.0, 1.0], [1.0, 0.0]],
    ])
    foreground = torch.tensor([[1.0, 0.2], [0.8, 1.0]])

    enhanced, diagnostics = enhance_visual_embeddings(
        values,
        foreground,
        PositiveFeatureConfig(alpha=0.0, alpha_s=0.4, beta=0.0),
        torch,
    )

    normalized_before = torch.nn.functional.normalize(values, dim=-1)
    normalized_after = torch.nn.functional.normalize(enhanced, dim=-1)
    assert not torch.allclose(normalized_after, normalized_before)
    assert diagnostics["positive_feature_direction_delta"] > 0
    assert diagnostics["persistence_residual_mean_norm"] > 0


def test_enhancement_reports_mask_distribution_diagnostics():
    torch = pytest.importorskip("torch")
    values = torch.arange(16, dtype=torch.float32).reshape(2, 2, 4) + 1
    foreground = torch.tensor([[0.1, 0.4], [0.6, 0.9]])

    _, diagnostics = enhance_visual_embeddings(
        values,
        foreground,
        PositiveFeatureConfig(alpha=0.2, alpha_s=0.1, beta=0.0),
        torch,
    )

    assert diagnostics["foreground_min"] == pytest.approx(0.1)
    assert diagnostics["foreground_max"] == pytest.approx(0.9)
    assert diagnostics["foreground_std"] > 0
    assert diagnostics["foreground_spatial_std"] > 0
    assert diagnostics["foreground_temporal_std"] > 0
    assert diagnostics["foreground_coverage_at_0p5"] == pytest.approx(0.5)
    assert diagnostics["foreground_p10"] < diagnostics["foreground_p50"]
    assert diagnostics["foreground_p50"] < diagnostics["foreground_p90"]
    assert diagnostics["persistence_std"] > 0


def test_grid_summary_marks_only_complete_runs_as_best_and_worst():
    rows = [
        {"experiment": "bad", "status": "failed", "mean_score": 0.99},
        {"experiment": "low", "status": "complete", "mean_score": 0.2},
        {"experiment": "high", "status": "complete", "mean_score": 0.8},
    ]
    finalized = {row["experiment"]: row for row in finalize_grid_rows(rows)}
    assert finalized["high"]["is_best"] is True
    assert finalized["low"]["is_worst"] is True
    assert finalized["bad"]["rank"] is None


def test_run_diagnostics_validate_hook_and_all_coefficients(tmp_path):
    point = GridPoint("full", 0.2, 0.1, 0.6)
    row = {
        "sample_id": "tsh:1", "model": "m", "method": "positive_feature",
        "benchmark": "vidhalluc", "task": "tsh", "error": None,
        "method_config": {
            "alpha": 0.2,
            "alpha_s": 0.1,
            "beta": 0.6,
            "foreground_threshold": 0.5,
            "foreground_morph_kernel": 0,
            "foreground_return_soft": True,
            "foreground_pair_fusion": "mean",
            "foreground_pool_avg_weight": 1.0,
        },
        "metadata": {
            "positive_feature_hook_applied": True,
            "alpha": 0.2, "alpha_s": 0.1, "beta": 0.6,
            "foreground_threshold": 0.5,
            "foreground_morph_kernel": 0,
            "foreground_return_soft": True,
            "foreground_pair_fusion": "mean",
            "foreground_pool_avg_weight": 1.0,
            "foreground_mean": 0.5,
            "foreground_std": 0.2,
            "foreground_min": 0.1,
            "foreground_max": 0.9,
            "foreground_p10": 0.2,
            "foreground_p50": 0.5,
            "foreground_p90": 0.8,
            "foreground_coverage_at_0p5": 0.5,
            "foreground_spatial_std": 0.2,
            "foreground_temporal_std": 0.1,
            "persistence_std": 0.15,
            "positive_feature_direction_delta": 0.03,
            "foreground_residual_mean_norm": 1.2,
            "persistence_residual_mean_norm": 1.1,
        },
    }
    (tmp_path / "run__records.jsonl").write_text(__import__("json").dumps(row) + "\n")
    assert validate_run_diagnostics(tmp_path, "run", point) == []

    row["metadata"]["beta"] = 0.4
    (tmp_path / "run__records.jsonl").write_text(__import__("json").dumps(row) + "\n")
    assert "diagnostics.beta" in validate_run_diagnostics(tmp_path, "run", point)[0]


def test_run_diagnostics_reject_norm_only_feature_changes(tmp_path):
    point = GridPoint("foreground_only", 0.2, 0.0, 0.0)
    row = {
        "sample_id": "tsh:1",
        "model": "m",
        "method": "positive_feature",
        "benchmark": "vidhalluc",
        "task": "tsh",
        "error": None,
        "method_config": {
            "alpha": 0.2,
            "alpha_s": 0.0,
            "beta": 0.0,
            "foreground_threshold": 0.5,
            "foreground_morph_kernel": 0,
            "foreground_return_soft": True,
            "foreground_pair_fusion": "mean",
            "foreground_pool_avg_weight": 1.0,
        },
        "metadata": {
            "positive_feature_hook_applied": True,
            "alpha": 0.2,
            "alpha_s": 0.0,
            "beta": 0.0,
            "foreground_threshold": 0.5,
            "foreground_morph_kernel": 0,
            "foreground_return_soft": True,
            "foreground_pair_fusion": "mean",
            "foreground_pool_avg_weight": 1.0,
            "foreground_mean": 0.5,
            "foreground_std": 0.2,
            "foreground_min": 0.1,
            "foreground_max": 0.9,
            "foreground_p10": 0.2,
            "foreground_p50": 0.5,
            "foreground_p90": 0.8,
            "foreground_coverage_at_0p5": 0.5,
            "foreground_spatial_std": 0.2,
            "foreground_temporal_std": 0.1,
            "persistence_std": 0.15,
            "positive_feature_direction_delta": 0.0,
            "foreground_residual_mean_norm": 1.2,
            "persistence_residual_mean_norm": 1.1,
        },
    }
    (tmp_path / "run__records.jsonl").write_text(__import__("json").dumps(row) + "\n")

    errors = validate_run_diagnostics(tmp_path, "run", point)
    assert any("did not change feature direction" in error for error in errors)


def test_run_diagnostics_reject_unchanged_logits_when_requested(tmp_path):
    point = GridPoint("foreground_only", 0.2, 0.0, 0.0)
    method_config = {
        "alpha": 0.2,
        "alpha_s": 0.0,
        "beta": 0.0,
        "foreground_threshold": 0.5,
        "foreground_morph_kernel": 0,
        "foreground_return_soft": True,
        "foreground_pair_fusion": "mean",
        "foreground_pool_avg_weight": 1.0,
        "logit_diagnostics": True,
    }
    metadata = {
        **method_config,
        "positive_feature_hook_applied": True,
        "foreground_mean": 0.5,
        "foreground_std": 0.2,
        "foreground_min": 0.1,
        "foreground_max": 0.9,
        "foreground_p10": 0.2,
        "foreground_p50": 0.5,
        "foreground_p90": 0.8,
        "foreground_coverage_at_0p5": 0.5,
        "foreground_spatial_std": 0.2,
        "foreground_temporal_std": 0.1,
        "persistence_std": 0.15,
        "positive_feature_direction_delta": 0.03,
        "foreground_residual_mean_norm": 1.2,
        "persistence_residual_mean_norm": 1.1,
        "positive_feature_logit_mean_abs_delta": 0.0,
        "positive_feature_logit_max_abs_delta": 0.0,
        "positive_feature_logit_cosine_distance": 0.0,
        "positive_feature_logit_top1_changed": False,
        "positive_feature_base_topk_token_ids": [1, 2],
        "positive_feature_enhanced_topk_token_ids": [1, 2],
    }
    row = {
        "sample_id": "tsh:1",
        "model": "m",
        "method": "positive_feature",
        "benchmark": "vidhalluc",
        "task": "tsh",
        "error": None,
        "method_config": method_config,
        "metadata": metadata,
    }
    (tmp_path / "run__records.jsonl").write_text(__import__("json").dumps(row) + "\n")

    errors = validate_run_diagnostics(tmp_path, "run", point)
    assert any("did not change logits" in error for error in errors)
