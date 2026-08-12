from src.data.jsonl import append_jsonl, read_jsonl, valid_resume_keys
from src.data.schema import PredictionRecord


def record(**updates):
    values = dict(
        sample_id="1", model="m", method="base", benchmark="b", task="t",
        prompt="q", frame_indices=list(range(8)), raw_output="yes",
        normalized_output="yes", ground_truth="yes", is_correct=True,
        parser_status="valid",
    )
    values.update(updates)
    return PredictionRecord(**values)


def test_jsonl_roundtrip_and_resume(tmp_path):
    path = tmp_path / "predictions.jsonl"
    append_jsonl(path, record())
    append_jsonl(path, record(sample_id="2", parser_status="missing", error="no output"))
    assert read_jsonl(path)[0] == record()
    assert valid_resume_keys(path) == {("1", "m", "base", "b", "t")}
