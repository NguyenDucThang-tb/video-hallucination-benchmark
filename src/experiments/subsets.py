from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Callable, Iterable

from src.benchmarks.base import BenchmarkSample


SCHEMA_VERSION = 1


def _select_grouped_samples(
    samples: Iterable[BenchmarkSample],
    unit_key: Callable[[BenchmarkSample], str],
    count: int,
    rng: random.Random,
) -> tuple[list[str], list[str]]:
    groups: dict[str, list[str]] = {}
    for sample in samples:
        groups.setdefault(unit_key(sample), []).append(sample.sample_id)
    if count < 0:
        raise ValueError("subset count must be non-negative")
    if count > len(groups):
        raise ValueError(f"requested {count} units, but only {len(groups)} are available")
    selected_units = sorted(rng.sample(sorted(groups), count))
    sample_ids = sorted(sample_id for unit in selected_units for sample_id in groups[unit])
    return selected_units, sample_ids


def _video_name(sample: BenchmarkSample) -> str:
    return str(sample.metadata.get("video_name") or sample.video_path)


def _resolved(samples: Iterable[BenchmarkSample]) -> list[BenchmarkSample]:
    return [sample for sample in samples if sample.metadata.get("video_resolved", True)]


def build_tuning_subset_manifest(
    *,
    vidhalluc_samples: Iterable[BenchmarkSample],
    videohallucer_samples: Iterable[BenchmarkSample],
    eventhallusion_samples: Iterable[BenchmarkSample],
    seed: int,
    tsh_videos: int = 100,
    mcq_videos: int = 50,
    tph_videos: int = 100,
    event_videos: int = 50,
) -> dict:
    """Choose stable tuning units and materialize their sample IDs.

    VideoHallucer TPH is selected in complete basic/hallucination pairs. Because
    each pair has two records, ``tph_videos`` must map to an exact number of
    complete pairs (normally 100 videos means 50 pairs).
    """
    rng = random.Random(seed)
    vidhalluc = _resolved(vidhalluc_samples)
    video_hallucer = _resolved(videohallucer_samples)
    event = _resolved(eventhallusion_samples)

    tsh_units, tsh_ids = _select_grouped_samples(
        (sample for sample in vidhalluc if sample.task == "tsh"), _video_name, tsh_videos, rng
    )
    mcq_units, mcq_ids = _select_grouped_samples(
        (sample for sample in vidhalluc if sample.task == "mcq"), _video_name, mcq_videos, rng
    )

    tph_groups: dict[str, list[BenchmarkSample]] = {}
    for sample in video_hallucer:
        if sample.task == "tph":
            tph_groups.setdefault(str(sample.metadata.get("pair_id")), []).append(sample)
    tph_groups = {
        key: items
        for key, items in tph_groups.items()
        if len(items) == 2 and {item.metadata.get("branch") for item in items} == {"basic", "hallucination"}
    }
    if tph_videos % 2:
        raise ValueError("tph_videos must be even so complete basic/hallucination pairs are retained")
    tph_pair_count = tph_videos // 2
    tph_units, tph_ids = _select_grouped_samples(
        (sample for items in tph_groups.values() for sample in items),
        lambda sample: str(sample.metadata["pair_id"]),
        tph_pair_count,
        rng,
    )

    event_units, event_ids = _select_grouped_samples(
        event,
        lambda sample: f"{sample.task}:{sample.metadata.get('video_id', sample.sample_id.rsplit(':', 1)[0])}",
        event_videos,
        rng,
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "selections": {
            "vidhalluc/tsh": {
                "unit": "video",
                "requested_units": tsh_videos,
                "selected_units": tsh_units,
                "sample_ids": tsh_ids,
            },
            "vidhalluc/mcq": {
                "unit": "video",
                "requested_units": mcq_videos,
                "selected_units": mcq_units,
                "sample_ids": mcq_ids,
            },
            "videohallucer/tph": {
                "unit": "pair",
                "requested_units": tph_pair_count,
                "requested_videos": tph_videos,
                "selected_units": tph_units,
                "sample_ids": tph_ids,
            },
            "eventhallusion/*": {
                "unit": "video",
                "requested_units": event_videos,
                "selected_units": event_units,
                "sample_ids": event_ids,
            },
        },
    }


def write_subset_manifest(manifest: dict, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def load_subset_manifest(path: str | Path) -> dict:
    source = Path(path)
    manifest = json.loads(source.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported subset manifest schema in {source}")
    if not isinstance(manifest.get("selections"), dict):
        raise ValueError(f"subset manifest has no selections mapping: {source}")
    return manifest


def filter_samples_by_manifest(
    samples: Iterable[BenchmarkSample], benchmark: str, manifest: dict
) -> list[BenchmarkSample]:
    selections = manifest["selections"]
    source = list(samples)
    output: list[BenchmarkSample] = []
    selected_by_task: dict[str, set[str]] = {}
    for sample in source:
        selection = selections.get(f"{benchmark}/{sample.task}") or selections.get(f"{benchmark}/*")
        if selection is None:
            continue
        selected = selected_by_task.setdefault(sample.task, set(selection["sample_ids"]))
        if sample.sample_id in selected:
            output.append(sample)

    expected_ids: set[str] = set()
    for key, selection in selections.items():
        if key.startswith(f"{benchmark}/"):
            expected_ids.update(selection["sample_ids"])
    observed_ids = {sample.sample_id for sample in output}
    missing = expected_ids - observed_ids
    if missing:
        preview = ", ".join(sorted(missing)[:5])
        raise ValueError(f"subset manifest references {len(missing)} missing {benchmark} samples: {preview}")

    if benchmark == "videohallucer":
        pair_count = len({sample.metadata.get("pair_id") for sample in output})
        output = [
            replace(sample, metadata={**sample.metadata, "expected_task_pairs": pair_count})
            for sample in output
        ]
    return output
