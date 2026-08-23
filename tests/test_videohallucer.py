from src.benchmarks.videohallucer.evaluator import pair_accuracy
from src.data.schema import PredictionRecord


def make(branch, correct, pair="p", expected_pairs=1):
    return PredictionRecord(
        sample_id=f"{pair}:{branch}", model="m", method="base", benchmark="videohallucer",
        task="orh", prompt="q", frame_indices=list(range(8)), raw_output="yes",
        normalized_output="yes", ground_truth="yes", is_correct=correct,
        parser_status="valid", metadata={
            "pair_id": pair,
            "branch": branch,
            "expected_task_pairs": expected_pairs,
        },
    )


def test_official_metric_requires_both_pair_branches_correct():
    metric = pair_accuracy([make("basic", True), make("hallucination", False)])
    assert metric["metric"] == "strict_pair_accuracy"
    assert metric["orh"]["accuracy"] == 0.0
    assert pair_accuracy([make("basic", True), make("hallucination", True)])["orh"]["accuracy"] == 1.0


def test_incomplete_pair_is_counted_as_incorrect():
    result = pair_accuracy([make("basic", True)])
    assert result["orh"]["n_pairs"] == 1
    assert result["orh"]["n_missing_pairs"] == 1
    assert result["orh"]["accuracy"] == 0.0


def test_annotation_denominator_counts_entirely_missing_pairs_as_incorrect():
    record = make("basic", True, expected_pairs=2)
    result = pair_accuracy([record])
    assert result["orh"]["n_pairs"] == 2
    assert result["orh"]["n_observed_pairs"] == 1
    assert result["orh"]["n_missing_pairs"] == 2
    assert result["orh"]["accuracy"] == 0.0
    assert result["orh"]["denominator_source"] == "annotation_metadata"


def test_unverified_observed_denominator_is_not_reported_as_research_accuracy():
    record = make("basic", True)
    record.metadata.pop("expected_task_pairs")
    result = pair_accuracy([record])["orh"]
    assert result["accuracy"] is None
    assert result["protocol_status"] == "UNVERIFIED_DENOMINATOR"
    assert result["observed_only_accuracy"] == 0.0
