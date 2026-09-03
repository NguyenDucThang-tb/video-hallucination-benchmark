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
    output = {"tsh": None, "mcq": None, "tph": None, "eventhallusion": None}
    for key, result in metrics.items():
        if key.endswith("/vidhalluc"):
            output["tsh"] = result.get("tsh", {}).get("official_accuracy")
            output["mcq"] = result.get("mcq", {}).get("accuracy")
        elif key.endswith("/videohallucer"):
            output["tph"] = result.get("tph", {}).get("accuracy")
        elif key.endswith("/eventhallusion"):
            output["eventhallusion"] = result.get("overall", {}).get("accuracy")
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
    return errors


def finalize_grid_rows(rows: list[dict]) -> list[dict]:
    complete = [row for row in rows if row.get("status") == "complete" and row.get("mean_score") is not None]
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
