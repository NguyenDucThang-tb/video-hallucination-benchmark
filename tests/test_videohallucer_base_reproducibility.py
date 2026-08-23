import csv
import importlib.util
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "scripts/audit_videohallucer_base.py"
SPEC = importlib.util.spec_from_file_location("audit_videohallucer_base", SCRIPT)
AUDIT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(AUDIT)


def test_bundled_annotations_match_current_upstream_inventory_counts():
    root = PROJECT / "external/VideoHallucer/videohallucer_datasets"
    annotations = AUDIT.load_annotations(root)
    inventory, summary = AUDIT.build_inventory(root, annotations)

    assert {task: len(rows) for task, rows in annotations.items()} == {
        "orh": 200,
        "tph": 176,
        "sdh": 200,
        "efh": 200,
        "enfh": 200,
    }
    assert len(inventory) == 1952
    assert all(item["missing_branches"] == 0 for item in summary.values())
    assert all(item["duplicate_pairs"] == 0 for item in summary.values())


def test_inventory_has_exactly_two_named_branches_per_pair():
    root = PROJECT / "external/VideoHallucer/videohallucer_datasets"
    inventory, _ = AUDIT.build_inventory(root, AUDIT.load_annotations(root))
    branches = {}
    for row in inventory:
        branches.setdefault(row["pair_id"], set()).add(row["branch"])
    assert all(value == {"basic", "hallucination"} for value in branches.values())


def test_official_and_strict_parsers_intentionally_differ_on_ambiguous_output():
    raw = "Yes, but maybe no."
    assert AUDIT.official_value(raw, "yes") == "yes"
    assert AUDIT.official_value(raw, "no") == "no"
    assert AUDIT.strict_local_value(raw) == (None, "ambiguous")


def test_missing_branch_stays_in_expected_denominator(tmp_path):
    inventory = [
        {"sample_id": "orh:0:basic", "pair_id": "orh:0", "task": "orh", "branch": "basic", "ground_truth": "yes"},
        {"sample_id": "orh:0:hallucination", "pair_id": "orh:0", "task": "orh", "branch": "hallucination", "ground_truth": "no"},
    ]
    records = [{
        "sample_id": "orh:0:basic", "model": "m", "raw_output": "yes",
        "parser_status": "valid",
    }]
    latest, counts = AUDIT.latest_base_records(records)
    rows = AUDIT.pair_metric_rows(inventory, latest, counts)
    orh = next(row for row in rows if row["model"] == "m" and row["task"] == "orh")
    assert orh["expected_pairs"] == 1
    assert orh["missing_records"] == 1
    assert orh["official_compatible_strict_pair_accuracy"] == 0.0


def test_generated_inventory_is_csv_readable(tmp_path):
    path = tmp_path / "inventory.csv"
    AUDIT.write_csv(path, [{"sample_id": "x", "pair_id": "p"}], ["sample_id", "pair_id"])
    with path.open() as handle:
        assert list(csv.DictReader(handle)) == [{"sample_id": "x", "pair_id": "p"}]
