import json
from pathlib import Path

from src.benchmarks.tempcompass.evaluator import evaluate_tempcompass
from src.benchmarks.tempcompass.loader import ANSWER_SUFFIXES, TempCompassLoader
from src.benchmarks.tempcompass.parsers import (
    parse_caption_matching,
    parse_multi_choice,
    parse_yes_no,
)
from src.data.schema import PredictionRecord


def _write_fixture(root: Path) -> tuple[Path, Path, Path]:
    questions = root / "questions"
    videos = root / "videos"
    questions.mkdir()
    videos.mkdir()
    (videos / "video_1.mp4").write_bytes(b"")
    common = {"video_1": {"action": []}}
    payloads = {
        "multi-choice.json": {
            "question": "What happens?\nA. runs\nB. sleeps",
            "answer": "A. runs",
        },
        "yes_no.json": {"question": "Does the person run?", "answer": "yes"},
        "caption_matching.json": {
            "question": "Which caption?\nOption 1: runs\nOption 2: sleeps",
            "answer": "Option 1: runs",
        },
        "captioning.json": {
            "question": "Information A: runs\nGenerated Caption:",
            "answer": "A. runs",
        },
    }
    for filename, item in payloads.items():
        data = json.loads(json.dumps(common))
        data["video_1"]["action"] = [item]
        (questions / filename).write_text(json.dumps(data), encoding="utf-8")
    meta = root / "meta_info.json"
    meta.write_text(json.dumps({
        "video_1": {"eval_dim": {"action": {"type": "fine-grained action"}}}
    }), encoding="utf-8")
    return questions, videos, meta


def test_loader_preserves_official_prompts_and_metadata(tmp_path):
    questions, videos, meta = _write_fixture(tmp_path)
    samples = list(TempCompassLoader(questions, videos, meta).iter_samples())

    assert len(samples) == 4
    assert {sample.task for sample in samples} == set(ANSWER_SUFFIXES)
    for sample in samples:
        assert sample.prompt.endswith(ANSWER_SUFFIXES[sample.task])
        assert sample.metadata["video_resolved"] is True
        assert sample.metadata["temporal_aspect"] == "action"
        assert sample.metadata["fine_grained_aspect"] == "fine-grained action"


def test_official_rule_parsers():
    assert parse_multi_choice("A", "A. runs").value == "A"
    assert parse_multi_choice("A) runs", "A. runs").value == "A"
    assert parse_multi_choice("The person runs", "A. runs").status == "requires_judge"
    assert parse_yes_no("Yes, they do.").value == "yes"
    assert parse_yes_no("I think yes.").status == "requires_judge"
    options = [
        {"label": "Option 1", "short": "1", "sentence": "runs"},
        {"label": "Option 2", "short": "2", "sentence": "sleeps"},
    ]
    assert parse_caption_matching("Option 1", options).value == "Option 1"
    assert parse_caption_matching("runs", options).value == "Option 1"


def test_requires_judge_records_resume_without_being_runtime_errors():
    record = PredictionRecord(
        sample_id="1", model="m", method="base", benchmark="tempcompass",
        task="multi-choice", prompt="q", frame_indices=[], raw_output="runs",
        normalized_output=None, ground_truth="A", is_correct=None,
        parser_status="requires_judge", error=None,
        metadata={"temporal_aspect": "action", "fine_grained_aspect": "fine-grained action"},
    )
    assert record.is_valid_for_resume is True
    result = evaluate_tempcompass([record])
    assert result["overall"]["n_requires_llm_judge"] == 1
    assert result["overall"]["official_accuracy"] is None
    assert result["overall"]["rule_only_accuracy"] == 0.0


def test_vendored_official_questions_total_7540():
    project = Path(__file__).resolve().parents[1]
    loader = TempCompassLoader(
        project / "external/TempCompass/questions",
        project / "external/TempCompass/videos-not-in-git",
        project / "external/TempCompass/meta_info.json",
    )
    samples = list(loader.iter_samples())
    assert len(samples) == 7540
    assert sum(sample.task == "multi-choice" for sample in samples) == 1580
    assert sum(sample.task == "yes_no" for sample in samples) == 2453
    assert sum(sample.task == "caption_matching" for sample in samples) == 1503
    assert sum(sample.task == "captioning" for sample in samples) == 2004
