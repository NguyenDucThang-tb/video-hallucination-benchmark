import json

from src.benchmarks.videomme.evaluator import evaluate_videomme
from src.benchmarks.videomme.loader import PROMPT_PREFIX, VideoMMELoader
from src.data.schema import PredictionRecord
from scripts.run_benchmark import build_plan, instantiate_loader, load_yaml


def _dataset(root):
    (root / "videos").mkdir(parents=True)
    (root / "videos" / "vid-short.mp4").write_bytes(b"fixture")
    rows = [
        {
            "videoID": "vid-short", "question_id": "q1", "duration": "short",
            "domain": "Knowledge", "sub_category": "General", "task_type": "Temporal",
            "question": "Which event occurs?", "options": ["A. First", "B. Second", "C. Third", "D. Fourth"],
            "answer": "B",
        },
        {
            "videoID": "vid-missing", "question_id": "q2", "duration": "long",
            "domain": "Film & TV", "sub_category": "Drama", "task_type": "Object",
            "question": "What is shown?", "options": {"A": "Cat", "B": "Dog", "C": "Bird", "D": "Fish"},
            "answer": "A",
        },
    ]
    (root / "videomme.json").write_text(json.dumps(rows), encoding="utf-8")


def test_loader_reads_official_rows_and_duration_filter(tmp_path):
    _dataset(tmp_path)
    samples = list(VideoMMELoader(tmp_path, ["short"]).iter_samples())
    assert len(samples) == 1
    sample = samples[0]
    assert sample.sample_id == "q1"
    assert sample.task == "short"
    assert sample.ground_truth == "B"
    assert sample.choices == {"A": "First", "B": "Second", "C": "Third", "D": "Fourth"}
    assert sample.video_path.name == "vid-short.mp4"
    assert sample.prompt.startswith(PROMPT_PREFIX)
    assert "subtitles" not in sample.prompt.lower()


def test_loader_keeps_missing_video_as_unresolved_path(tmp_path):
    _dataset(tmp_path)
    sample = list(VideoMMELoader(tmp_path, ["long"]).iter_samples())[0]
    assert sample.metadata["video_resolved"] is False
    assert sample.video_path.name == "vid-missing.mp4"


def test_videomme_evaluator_breaks_down_duration_and_domain():
    common = dict(
        model="qwen2.5-vl-7b", method="base", benchmark="videomme", prompt="q",
        frame_indices=[0], raw_output="B", normalized_output="B", ground_truth="B",
        is_correct=True, parser_status="valid", error=None,
    )
    records = [
        PredictionRecord(
            sample_id="q1", task="short", metadata={"domain": "Knowledge", "sub_category": "General", "task_type": "Temporal"}, **common,
        ),
        PredictionRecord(
            sample_id="q2", task="long", metadata={"domain": "Knowledge", "sub_category": "Objects", "task_type": "Object"}, **common,
        ),
    ]
    metrics = evaluate_videomme(records)
    assert metrics["overall"]["accuracy"] == 1.0
    assert metrics["duration"]["short"]["n"] == 1
    assert metrics["duration"]["long"]["n"] == 1
    assert metrics["domain"]["Knowledge"]["n"] == 2


def test_runner_registers_videomme_config_and_loader_override(tmp_path):
    _dataset(tmp_path)
    config = load_yaml("configs/experiment_qwen25_videomme.yaml")
    config["benchmarks"][0]["data_root"] = str(tmp_path)
    plan = build_plan(config, allow_unvalidated=True)
    assert len(plan) == 15
    assert {job["method"] for job in plan} == {"base", "tcd", "dino_heal", "season", "positive_feature"}
    loader = instantiate_loader("videomme", config["benchmarks"][0])
    assert len(list(loader.iter_samples())) == 2
