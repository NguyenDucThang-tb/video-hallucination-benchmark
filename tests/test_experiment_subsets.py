from pathlib import Path

import pytest

from src.benchmarks.base import BenchmarkSample
from src.experiments.subsets import (
    build_motionbench_tuning_subset_manifest,
    build_tuning_subset_manifest,
    filter_samples_by_manifest,
)


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


def test_filter_keeps_videohallucer_denominators_separate_per_task():
    samples = [
        sample(
            f"{task}:{index}:{branch}", "videohallucer", task,
            f"{task}-{index}-{branch}", pair_id=f"{task}:{index}",
            branch=branch, expected_task_pairs=99,
        )
        for task, count in (("tph", 2), ("sdh", 3))
        for index in range(count)
        for branch in ("basic", "hallucination")
    ]
    manifest = {
        "selections": {
            "videohallucer/tph": {
                "sample_ids": [item.sample_id for item in samples if item.task == "tph"],
            },
            "videohallucer/sdh": {
                "sample_ids": [item.sample_id for item in samples if item.task == "sdh"],
            },
        }
    }

    selected = filter_samples_by_manifest(samples, "videohallucer", manifest)

    assert {
        task: {item.metadata["expected_task_pairs"] for item in selected if item.task == task}
        for task in ("tph", "sdh")
    } == {"tph": {2}, "sdh": {3}}


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


def test_motionbench_subset_is_deterministic_and_uses_one_fifth_per_task():
    tasks = ("action_order", "motion_recognition")
    samples = [
        sample(
            f"{task}:{index}", "motionbench", task, f"{task}-{index}",
            question_category=task,
            expected_task_records=count,
            expected_category_records=count,
            expected_split_records=16,
        )
        for task, count in (("action_order", 6), ("motion_recognition", 10))
        for index in range(count)
    ]
    kwargs = dict(samples=samples, tasks=tasks, seed=42, fraction=0.2)

    first = build_motionbench_tuning_subset_manifest(**kwargs)
    second = build_motionbench_tuning_subset_manifest(**kwargs)

    assert first == second
    assert len(first["selections"]["motionbench/action_order"]["sample_ids"]) == 2
    assert len(first["selections"]["motionbench/motion_recognition"]["sample_ids"]) == 2


def test_motionbench_filter_rewrites_subset_denominators():
    tasks = ("action_order", "motion_recognition")
    samples = [
        sample(
            f"{task}:{index}", "motionbench", task, f"{task}-{index}",
            question_category=task,
            expected_task_records=5,
            expected_category_records=5,
            expected_split_records=10,
        )
        for task in tasks
        for index in range(5)
    ]
    manifest = build_motionbench_tuning_subset_manifest(
        samples=samples, tasks=tasks, seed=7, fraction=0.2
    )

    selected = filter_samples_by_manifest(samples, "motionbench", manifest)

    assert len(selected) == 2
    assert {item.metadata["expected_task_records"] for item in selected} == {1}
    assert {item.metadata["expected_category_records"] for item in selected} == {1}
    assert {item.metadata["expected_split_records"] for item in selected} == {2}
