from src.benchmarks.videohallucer.evaluator import pair_accuracy
from src.data.schema import PredictionRecord


def make(branch, correct, pair="p"):
    return PredictionRecord(
        sample_id=f"{pair}:{branch}", model="m", method="base", benchmark="videohallucer",
        task="orh", prompt="q", frame_indices=list(range(8)), raw_output="yes",
        normalized_output="yes", ground_truth="yes", is_correct=correct,
        parser_status="valid", metadata={"pair_id": pair, "branch": branch},
    )


def test_official_metric_requires_both_pair_branches_correct():
    metric = pair_accuracy([make("basic", True), make("hallucination", False)])
    assert metric["metric"] == "strict_pair_accuracy"
    assert metric["orh"]["accuracy"] == 0.0
    assert pair_accuracy([make("basic", True), make("hallucination", True)])["orh"]["accuracy"] == 1.0
