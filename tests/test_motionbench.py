import json
from pathlib import Path

from src.benchmarks.motionbench.evaluator import evaluate_motionbench
from src.benchmarks.motionbench.loader import ANSWER_SUFFIX, MotionBenchLoader
from src.benchmarks.motionbench.parsers import parse_motionbench_mcq, polish_answer
from src.data.schema import PredictionRecord


def _write_fixture(root: Path) -> tuple[Path, Path]:
    videos = root / "videos" / "MotionBench" / "self-collected"
    videos.mkdir(parents=True)
    (videos / "dev.mp4").write_bytes(b"")
    (videos / "test.mp4").write_bytes(b"")
    rows = [
        {
            "question_type": "Action Order",
            "video_type": "Gaming",
            "key": "dev",
            "video_path": "dev.mp4",
            "qa": [{
                "uid": "dev-1", "start": None, "end": None, "answer": "B",
                "question": "What happens first?\nA. Sit\nB. Run\nC. Jump",
            }],
        },
        {
            "question_type": "Motion Recognition",
            "video_type": None,
            "key": "test",
            "video_path": "test.mp4",
            "qa": [{
                "uid": "test-1", "start": 0, "end": 2, "answer": "NA",
                "question": "What motion occurs?\nA. Sit\nB. Run",
            }],
        },
    ]
    meta = root / "video_info.meta.jsonl"
    meta.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return meta, root / "videos"


def test_loader_splits_prompts_and_paths(tmp_path):
    meta, videos = _write_fixture(tmp_path)
    dev = list(MotionBenchLoader(meta, videos, ["dev"]).iter_samples())
    test = list(MotionBenchLoader(meta, videos, ["test"]).iter_samples())
    assert [sample.sample_id for sample in dev] == ["dev-1"]
    assert [sample.sample_id for sample in test] == ["test-1"]
    assert dev[0].prompt.endswith(ANSWER_SUFFIX)
    assert dev[0].choices == {"A": "Sit", "B": "Run", "C": "Jump"}
    assert dev[0].metadata["video_resolved"] is True
    assert test[0].metadata["requires_llm_judge"] is True


def test_loader_category_task_is_dev_only(tmp_path):
    meta, videos = _write_fixture(tmp_path)
    samples = list(MotionBenchLoader(meta, videos, ["action_order"]).iter_samples())
    assert len(samples) == 1
    assert samples[0].task == "action_order"
    assert samples[0].metadata["expected_task_records"] == 1


def test_parser_ports_upstream_polish_answer():
    assert polish_answer("B) Run") == "B"
    assert polish_answer("Answer (C) because...") == "C"
    assert parse_motionbench_mcq("A. Sit").value == "A"
    assert parse_motionbench_mcq("The answer is B").status == "unparseable"


def test_evaluator_scores_dev_and_never_scores_test():
    common = dict(
        model="m", method="base", benchmark="motionbench", prompt="q",
        frame_indices=[], raw_output="B", normalized_output="B",
        parser_status="valid", error=None,
    )
    dev = PredictionRecord(
        sample_id="dev-1", task="dev", ground_truth="B", is_correct=True,
        metadata={
            "has_public_ground_truth": True, "question_category": "action_order",
            "expected_task_records": 1, "expected_split_records": 1,
            "expected_category_records": 1,
        }, **common,
    )
    test = PredictionRecord(
        sample_id="test-1", task="test", ground_truth="NA", is_correct=None,
        metadata={
            "has_public_ground_truth": False, "question_category": "motion_recognition",
            "expected_task_records": 1, "expected_split_records": 1,
            "expected_category_records": 1,
        }, **common,
    )
    result = evaluate_motionbench([dev, test])
    assert result["tasks"]["dev"]["accuracy"] == 1.0
    assert result["test"]["accuracy"] is None
    assert result["test"]["status"] == "requires_leaderboard"


def test_vendored_metadata_counts_match_release():
    project = Path(__file__).resolve().parents[1]
    loader = MotionBenchLoader(
        project / "external/MotionBench/data/video_info.meta.jsonl",
        project / "external/MotionBench/videos-not-in-git",
        ["dev", "test"],
    )
    samples = list(loader.iter_samples())
    assert len(samples) == 8052
    assert sum(sample.task == "dev" for sample in samples) == 4018
    assert sum(sample.task == "test" for sample in samples) == 4034
