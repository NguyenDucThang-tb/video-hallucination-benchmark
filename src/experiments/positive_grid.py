from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path


ALPHAS = (0.0, 0.2, 0.4, 0.6, 0.8)
ALPHA_S_VALUES = (0.0, 0.1, 0.2, 0.4, 0.8)
BETAS = (0.0, 0.1, 0.2, 0.4, 0.6, 0.8)


@dataclass(frozen=True)
class GridPoint:
    ablation: str
    alpha: float
    alpha_s: float
    beta: float


def positive_feature_grid(include_baseline: bool = False) -> list[GridPoint]:
    points = []
    points.extend(GridPoint("foreground_only", alpha, 0.0, 0.0) for alpha in ALPHAS[1:])
    points.extend(GridPoint("persistence_only", 0.0, alpha_s, 0.0) for alpha_s in ALPHA_S_VALUES[1:])
    points.extend(GridPoint("temporal_only", 0.0, 0.0, beta) for beta in BETAS[1:])
    points.extend(
        GridPoint("spatial", alpha, alpha_s, 0.0)
        for alpha in ALPHAS[1:]
        for alpha_s in ALPHA_S_VALUES[1:]
    )
    points.extend(
        GridPoint("full", alpha, alpha_s, beta)
        for alpha in ALPHAS[1:]
        for alpha_s in ALPHA_S_VALUES[1:]
        for beta in BETAS[1:]
    )
    if include_baseline:
        points.insert(0, GridPoint("baseline", 0.0, 0.0, 0.0))
    return points


def _number_slug(value: float) -> str:
    return str(value).replace(".", "p")


def experiment_name(prefix: str, point: GridPoint) -> str:
    return (
        f"{prefix}__{point.ablation}__a{_number_slug(point.alpha)}"
        f"__as{_number_slug(point.alpha_s)}__b{_number_slug(point.beta)}"
    )


def metric_scores(metrics: dict) -> dict[str, float | None]:
    output = {
        "tsh": None, "mcq": None, "tph": None, "eventhallusion": None,
        "videohallucer_tph": None,
        "videohallucer_sdh": None,
        "eventhallusion_entire": None,
        "eventhallusion_misleading": None,
        "eventhallusion_mix": None,
        "tempcompass_multi_choice": None,
        "tempcompass_yes_no": None,
        "tempcompass_caption_matching": None,
        "motionbench_action_order": None,
        "motionbench_location_related_motion": None,
        "motionbench_motion_recognition": None,
        "motionbench_motion_related_objects": None,
    }
    for key, result in metrics.items():
        if key.endswith("/vidhalluc"):
            output["tsh"] = result.get("tsh", {}).get("official_accuracy")
            output["mcq"] = result.get("mcq", {}).get("accuracy")
        elif key.endswith("/videohallucer"):
            output["tph"] = result.get("tph", {}).get("accuracy")
            output["videohallucer_tph"] = result.get("tph", {}).get("accuracy")
            output["videohallucer_sdh"] = result.get("sdh", {}).get("accuracy")
        elif key.endswith("/eventhallusion"):
            output["eventhallusion"] = result.get("overall", {}).get("accuracy")
            for task in ("entire", "misleading", "mix"):
                output[f"eventhallusion_{task}"] = result.get(task, {}).get("accuracy")
        elif key.endswith("/tempcompass"):
            tasks = result.get("tasks", {})
            output["tempcompass_multi_choice"] = tasks.get("multi-choice", {}).get(
                "official_accuracy"
            )
            output["tempcompass_yes_no"] = tasks.get("yes_no", {}).get(
                "official_accuracy"
            )
            output["tempcompass_caption_matching"] = tasks.get(
                "caption_matching", {}
            ).get("official_accuracy")
        elif key.endswith("/motionbench"):
            tasks = result.get("tasks", {})
            for task in (
                "action_order",
                "location_related_motion",
                "motion_recognition",
                "motion_related_objects",
            ):
                output[f"motionbench_{task}"] = tasks.get(task, {}).get("accuracy")
    return output


def count_run_records(raw_dir: Path, experiment: str) -> tuple[int, int]:
    latest = {}
    for path in raw_dir.glob(f"{experiment}__*.jsonl"):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (
                    row.get("sample_id"), row.get("model"), row.get("method"),
                    row.get("benchmark"), row.get("task"),
                )
                latest[key] = row
    return len(latest), sum(bool(row.get("error")) for row in latest.values())


def validate_run_diagnostics(
    raw_dir: Path, experiment: str, point: GridPoint
) -> list[str]:
    latest = {}
    for path in raw_dir.glob(f"{experiment}__*.jsonl"):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (
                    row.get("sample_id"), row.get("model"), row.get("method"),
                    row.get("benchmark"), row.get("task"),
                )
                latest[key] = row
    errors = []
    expected = {"alpha": point.alpha, "alpha_s": point.alpha_s, "beta": point.beta}
    preprocessing_fields = (
        "foreground_threshold",
        "foreground_morph_kernel",
        "foreground_return_soft",
        "foreground_pair_fusion",
        "foreground_pool_avg_weight",
    )
    distribution_fields = (
        "foreground_mean",
        "foreground_std",
        "foreground_min",
        "foreground_max",
        "foreground_p10",
        "foreground_p50",
        "foreground_p90",
        "foreground_coverage_at_0p5",
        "foreground_spatial_std",
        "foreground_temporal_std",
        "persistence_std",
        "positive_feature_direction_delta",
        "foreground_residual_mean_norm",
        "persistence_residual_mean_norm",
    )
    if any(row.get("benchmark") == "tempcompass" for row in latest.values()):
        # TempCompass does not require the legacy foreground-distribution
        # diagnostics used by the VidHalluc grid validator.
        preprocessing_fields = ()
        distribution_fields = ()
    for key, row in latest.items():
        if row.get("error"):
            continue
        metadata = row.get("metadata") or {}
        method_config = row.get("method_config") or {}
        if metadata.get("positive_feature_hook_applied") is not True:
            errors.append(f"{key[0]}: positive feature hook was not applied")
        for name, value in expected.items():
            if float(method_config.get(name, float("nan"))) != value:
                errors.append(f"{key[0]}: method_config.{name} does not equal {value}")
            if float(metadata.get(name, float("nan"))) != value:
                errors.append(f"{key[0]}: diagnostics.{name} does not equal {value}")
        for name in preprocessing_fields:
            if name not in method_config:
                errors.append(f"{key[0]}: method_config.{name} is missing")
            elif metadata.get(name) != method_config[name]:
                errors.append(
                    f"{key[0]}: diagnostics.{name} does not match method_config"
                )
        for name in distribution_fields:
            if metadata.get(name) is None:
                errors.append(f"{key[0]}: diagnostics.{name} is missing")
        direction_delta = metadata.get("positive_feature_direction_delta")
        if direction_delta is not None and float(direction_delta) <= 0.0:
            errors.append(
                f"{key[0]}: positive feature enhancement did not change feature direction"
            )
        if method_config.get("logit_diagnostics"):
            for name in (
                "positive_feature_logit_mean_abs_delta",
                "positive_feature_logit_max_abs_delta",
                "positive_feature_logit_cosine_distance",
                "positive_feature_logit_top1_changed",
                "positive_feature_base_topk_token_ids",
                "positive_feature_enhanced_topk_token_ids",
                "positive_feature_hook_output_field",
            ):
                if metadata.get(name) is None:
                    errors.append(f"{key[0]}: diagnostics.{name} is missing")
            logit_delta = metadata.get("positive_feature_logit_max_abs_delta")
            if logit_delta is not None and float(logit_delta) <= 0.0:
                errors.append(
                    f"{key[0]}: positive feature enhancement did not change logits"
                )
            if metadata.get("positive_feature_hook_output_field") != "pooler_output":
                errors.append(
                    f"{key[0]}: Qwen positive feature hook did not target pooler_output"
                )
    return errors


def finalize_grid_rows(rows: list[dict]) -> list[dict]:
    complete = [
        row for row in rows
        if row.get("status") in {"complete", "complete_with_errors"}
        and row.get("mean_score") is not None
    ]
    ranked = sorted(complete, key=lambda row: (-row["mean_score"], row["experiment"]))
    rank_by_name = {row["experiment"]: rank for rank, row in enumerate(ranked, 1)}
    worst_name = ranked[-1]["experiment"] if ranked else None
    best_name = ranked[0]["experiment"] if ranked else None
    for row in rows:
        row["rank"] = rank_by_name.get(row["experiment"])
        row["is_best"] = row["experiment"] == best_name
        row["is_worst"] = row["experiment"] == worst_name
    return rows


def write_grid_csv(rows: list[dict], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "experiment", "ablation", "alpha", "alpha_s", "beta", "status",
        "tsh", "mcq", "tph", "eventhallusion", "mean_score", "record_count",
        "videohallucer_tph", "videohallucer_sdh",
        "eventhallusion_entire", "eventhallusion_misleading", "eventhallusion_mix",
        "tempcompass_multi_choice", "tempcompass_yes_no",
        "tempcompass_caption_matching",
        "motionbench_action_order", "motionbench_location_related_motion",
        "motionbench_motion_recognition", "motionbench_motion_related_objects",
        "expected_records", "failed_records", "return_code", "rank", "is_best", "is_worst", "error",
        "diagnostics_valid",
    ]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({column: row.get(column) for column in columns} for row in rows)
    return destination


def safe_prefix(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()
    if not cleaned:
        raise ValueError("grid prefix must contain at least one alphanumeric character")
    return cleaned
