#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.data.jsonl import append_jsonl, read_jsonl
from src.data.schema import PredictionRecord
from src.evaluation.parsers import parse_yes_no
from src.methods.base import BaseMethod
from src.models.base import GenerationConfig
from src.models.mock import DeterministicSmokeModel


def run_smoke() -> dict:
    frames = np.zeros((8, 16, 16, 3), dtype=np.uint8)
    model = DeterministicSmokeModel()
    output = BaseMethod(model).generate(frames, "Is this a smoke test?", GenerationConfig())
    parsed = parse_yes_no(output.text)
    record = PredictionRecord(
        sample_id="smoke:0", model=model.name, method="base", benchmark="pipeline-smoke",
        task="yes_no", prompt="Is this a smoke test?", frame_indices=list(range(8)),
        raw_output=output.text, normalized_output=parsed.value, ground_truth="yes",
        is_correct=parsed.value == "yes", parser_status=parsed.status,
        metadata={"research_result": False},
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "prediction.jsonl"
        append_jsonl(path, record)
        loaded = read_jsonl(path)
        assert loaded == [record]
    result = {"status": "ok", "adapter": model.name, "research_result": False, "raw_output": output.text}
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run_smoke()
