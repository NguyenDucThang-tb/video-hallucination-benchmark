import json
from pathlib import Path

import pytest

from scripts.summarize_eventhallusion_latency import load_groups, percentile, summarize_group


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_percentile_interpolates():
    assert percentile([1.0, 2.0, 3.0], 0.5) == 2.0
    assert percentile([], 0.95) is None


def test_latency_summary_excludes_errors_and_warmup(tmp_path: Path):
    path = tmp_path / "lat_v1_job__model__method__eventhallusion__entire.jsonl"
    base = {
        "model": "llava-ov-7b",
        "method": "base",
        "benchmark": "eventhallusion",
        "task": "entire",
        "is_correct": True,
    }
    write_rows(path, [
        {**base, "sample_id": "1", "runtime_seconds": 9.0, "error": None},
        {**base, "sample_id": "2", "runtime_seconds": 1.0, "error": None},
        {
            **base,
            "sample_id": "3",
            "runtime_seconds": None,
            "error": "missing video",
            "is_correct": None,
        },
    ])

    groups = load_groups(tmp_path, "lat_v1")
    row = summarize_group(groups[("llava-ov-7b", "base", "entire")], warmup=1)

    assert row["records"] == 3
    assert row["errors"] == 1
    assert row["timed_records"] == 2
    assert row["warmup_excluded"] == 1
    assert row["mean_seconds"] == pytest.approx(1.0)
    assert row["accuracy"] == pytest.approx(1.0)
    assert row["status"] == "INCOMPLETE"


def test_latest_duplicate_record_is_used(tmp_path: Path):
    path = tmp_path / "lat_v1_job__model__method__eventhallusion__mix.jsonl"
    base = {
        "model": "qwen2.5-vl-7b",
        "method": "season",
        "benchmark": "eventhallusion",
        "task": "mix",
        "error": None,
    }
    write_rows(path, [
        {**base, "sample_id": "same", "runtime_seconds": 10.0, "is_correct": False},
        {**base, "sample_id": "same", "runtime_seconds": 2.0, "is_correct": True},
    ])

    group = load_groups(tmp_path, "lat_v1")[("qwen2.5-vl-7b", "season", "mix")]
    assert group.records == 1
    assert group.duplicates == 1
    assert group.correct == 1
    assert group.latencies == (2.0,)
