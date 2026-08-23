import importlib.util
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_videohallucer_base", PROJECT / "scripts/audit_videohallucer_base.py"
)
AUDIT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(AUDIT)


def test_current_upstream_temporal_revision_has_176_pairs():
    root = PROJECT / "external/VideoHallucer/videohallucer_datasets"
    _, summary = AUDIT.build_inventory(root, AUDIT.load_annotations(root))
    assert summary["tph"]["valid_pairs"] == 176
    assert summary["tph"]["missing_branches"] == 0


def test_pair_inventory_is_one_row_per_pair():
    root = PROJECT / "external/VideoHallucer/videohallucer_datasets"
    branches, _ = AUDIT.build_inventory(root, AUDIT.load_annotations(root))
    pairs = AUDIT.build_pair_inventory(branches)
    assert len(pairs) == 976
    assert all(row["pair_valid"] for row in pairs)
