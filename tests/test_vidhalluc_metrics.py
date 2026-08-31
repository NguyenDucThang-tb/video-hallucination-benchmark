from src.benchmarks.vidhalluc.evaluator import evaluate_classification
from src.data.schema import PredictionRecord


def record(sample_id, task, raw, ground_truth, is_correct=None, error=None, **metadata):
    return PredictionRecord(
        sample_id=sample_id, model="m", method="base", benchmark="vidhalluc", task=task,
        prompt="q", frame_indices=[], raw_output=raw, normalized_output=None,
        ground_truth=ground_truth, is_correct=is_correct,
        parser_status="valid" if is_correct is not None else "unparseable",
        error=error, metadata=metadata,
    )


def test_tsh_reports_official_and_diagnostic_denominators_separately():
    result = evaluate_classification([
        record("1", "tsh", "AB", "AB", True),
        record("2", "tsh", "BA", "AB", False),
        record("3", "tsh", "AB.", "AB", None),
    ])["tsh"]
    assert result["official_accuracy"] == 1 / 3
    assert result["valid_only_accuracy"] == 1 / 2
    assert result["parse_coverage"] == 2 / 3
    assert result["unparseable_count"] == 1
    assert result["parsed_AB"] == 1
    assert result["parsed_BA"] == 1
    assert result["empty_output_count"] == 0
    assert result["incorrect"] == 2


def test_none_is_not_mutated_to_false_at_record_level():
    item = record("1", "tsh", "", "AB", None)
    assert item.is_correct is None
    result = evaluate_classification([item])["tsh"]
    assert result["valid_count"] == 0
    assert result["official_accuracy"] == 0.0


def test_tsh_ab_ba_correctness_truth_table():
    cases = [("AB", "AB", 1.0), ("AB", "BA", 0.0), ("BA", "BA", 1.0), ("BA", "AB", 0.0)]
    for ground_truth, prediction, expected in cases:
        metric = evaluate_classification([
            record("one", "tsh", prediction, ground_truth, prediction == ground_truth)
        ])["tsh"]
        assert metric["official_accuracy"] == expected


def test_tsh_single_action_outputs_remain_incorrect_against_order_labels():
    result = evaluate_classification([
        record("a", "tsh", "A", "AB"),
        record("b", "tsh", "B", "BA"),
    ])['tsh']
    assert result["official_accuracy"] == 0.0
    assert result["parsed_AB"] == 0
    assert result["parsed_BA"] == 0
    assert result["parsed_A"] == 1
    assert result["parsed_B"] == 1


def test_tsh_parser_error_is_not_reported_as_runtime_failure():
    item = record("1", "tsh", "AB.", "AB", None, error="no standalone AB/BA answer")
    result = evaluate_classification([item])["tsh"]
    assert result["runtime_failure_count"] == 0


def test_tsh_generation_failure_is_reported_as_runtime_failure():
    item = record(
        "1", "tsh", "", "AB", None, error="generation: RuntimeError('failed')",
        failure_stage="generation",
    )
    item.parser_status = "missing"
    result = evaluate_classification([item])["tsh"]
    assert result["runtime_failure_count"] == 1


def test_sth_without_simcse_is_na_and_avg_stays_na():
    result = evaluate_classification([
        record("1", "sth", "Scene change: Yes, Locations: from room to road.", "yes", True,
               scene_change="yes", locations="from room to road.")
    ])
    assert result["sth"]["official_status"] == "SIMCSE_NOT_AVAILABLE"
    assert result["sth"]["description_accuracy"] is None
    assert result["sth"]["accuracy"] is None
    assert result["avg"]["accuracy"] is None
