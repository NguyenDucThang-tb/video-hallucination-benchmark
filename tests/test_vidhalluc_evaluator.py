from src.benchmarks.vidhalluc.evaluator import evaluate_classification
from src.data.schema import PredictionRecord


def make(sample_id, task, raw, correct, **metadata):
    return PredictionRecord(
        sample_id=sample_id, model="m", method="base", benchmark="vidhalluc", task=task,
        prompt="q", frame_indices=[], raw_output=raw, normalized_output=raw,
        ground_truth=metadata.pop("ground_truth", "yes"), is_correct=correct,
        parser_status="valid" if correct is not None else "missing", metadata=metadata,
    )


def test_bqa_scores_one_question_only_when_all_clips_are_correct():
    records = [
        make("bqa:1:0:a", "bqa", "Yes", True, section="1", question_index=0, expected_clip_count=2),
        make("bqa:1:0:b", "bqa", "No", False, section="1", question_index=0,
             expected_clip_count=2, ground_truth="yes"),
    ]
    result = evaluate_classification(records)
    assert result["bqa"]["n"] == 1
    assert result["bqa"]["accuracy"] == 0.0


def test_sth_is_not_mislabeled_as_binary_accuracy():
    record = make(
        "sth:1", "sth", "Scene change: Yes, Locations: from room to road.", True,
        scene_change="yes",
    )
    result = evaluate_classification([record])
    assert result["sth"]["accuracy"] is None
    assert result["sth"]["classification_score"] == 0.25
    assert result["avg"]["accuracy"] is None
