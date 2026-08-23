from src.benchmarks.videohallucer.evaluator import pair_accuracy
from src.data.schema import PredictionRecord


def make(branch):
    return PredictionRecord(
        sample_id=f"p:{branch}", model="m", method="base",
        benchmark="videohallucer", task="orh", prompt="q",
        frame_indices=list(range(8)), raw_output="yes", normalized_output="yes",
        ground_truth="yes", is_correct=True, parser_status="valid",
        metadata={"pair_id": "p", "branch": branch, "expected_task_pairs": 2},
    )


def test_missing_annotation_pairs_cannot_inflate_base_accuracy():
    basic = make("basic")
    hallucination = make("hallucination")
    metric = pair_accuracy([basic, hallucination])["orh"]
    assert metric["correct_pairs"] == 1
    assert metric["n_pairs"] == 2
    assert metric["accuracy"] == 0.5
