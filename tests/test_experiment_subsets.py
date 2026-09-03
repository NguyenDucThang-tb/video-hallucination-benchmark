from pathlib import Path

import pytest

from src.benchmarks.base import BenchmarkSample
from src.experiments.subsets import build_tuning_subset_manifest, filter_samples_by_manifest


def sample(sample_id, benchmark, task, video, **metadata):
    return BenchmarkSample(
        sample_id=sample_id,
        benchmark=benchmark,
        task=task,
        video_path=Path(f"/{video}.mp4"),
        prompt="question",
        ground_truth="yes",
        answer_type="yes_no",
        metadata={"video_name": video, "video_resolved": True, **metadata},
    )


def fixture_samples():
    vidhalluc = [
        *(sample(f"tsh:{i}", "vidhalluc", "tsh", f"tsh-{i}") for i in range(4)),
        *(sample(f"mcq:{i}", "vidhalluc", "mcq", f"mcq-{i}") for i in range(4)),
    ]
    videohallucer = [
        sample(
            f"tph:{i}:{branch}", "videohallucer", "tph", f"tph-{i}-{branch}",
            pair_id=f"tph:{i}", branch=branch, expected_task_pairs=4,
        )
        for i in range(4)
        for branch in ("basic", "hallucination")
    ]
    event = [
        sample(
            f"entire:event-{i}:{question}", "eventhallusion", "entire", f"event-{i}",
            video_id=f"event-{i}",
        )
        for i in range(4)
        for question in range(2)
    ]
    return vidhalluc, videohallucer, event


def test_fixed_subset_is_deterministic_and_keeps_tph_pairs():
    values = fixture_samples()
    kwargs = dict(
        vidhalluc_samples=values[0], videohallucer_samples=values[1],
        eventhallusion_samples=values[2], seed=17, tsh_videos=2,
        mcq_videos=2, tph_videos=4, event_videos=2,
    )
    first = build_tuning_subset_manifest(**kwargs)
    second = build_tuning_subset_manifest(**kwargs)

    assert first == second
    tph = first["selections"]["videohallucer/tph"]
    assert tph["requested_units"] == 2
    assert len(tph["sample_ids"]) == 4
    assert all(sum(item.startswith(pair) for item in tph["sample_ids"]) == 2 for pair in tph["selected_units"])


def test_tph_video_budget_must_preserve_complete_pairs():
    values = fixture_samples()
    with pytest.raises(ValueError, match="must be even"):
        build_tuning_subset_manifest(
            vidhalluc_samples=values[0], videohallucer_samples=values[1],
            eventhallusion_samples=values[2], seed=1, tsh_videos=1,
            mcq_videos=1, tph_videos=3, event_videos=1,
        )


def test_filter_rewrites_videohallucer_subset_denominator():
    values = fixture_samples()
    manifest = build_tuning_subset_manifest(
        vidhalluc_samples=values[0], videohallucer_samples=values[1],
        eventhallusion_samples=values[2], seed=3, tsh_videos=1,
        mcq_videos=1, tph_videos=4, event_videos=1,
    )
    selected = filter_samples_by_manifest(values[1], "videohallucer", manifest)

    assert len(selected) == 4
    assert {item.metadata["expected_task_pairs"] for item in selected} == {2}


def test_subset_generator_excludes_unresolved_videos():
    vidhalluc, videohallucer, event = fixture_samples()
    for item in event:
        if item.metadata["video_id"] == "event-0":
            item.metadata["video_resolved"] = False
    manifest = build_tuning_subset_manifest(
        vidhalluc_samples=vidhalluc, videohallucer_samples=videohallucer,
        eventhallusion_samples=event, seed=2, tsh_videos=1,
        mcq_videos=1, tph_videos=2, event_videos=3,
    )
    assert "entire:event-0" not in manifest["selections"]["eventhallusion/*"]["selected_units"]
