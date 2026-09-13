from pathlib import Path

from src.benchmarks.vidhalluc.loader import build_sth_prompt, build_tsh_prompt
from src.data.sampler import frame_indices, vidhalluc_frame_indices
from src.data.schema import PredictionRecord
from src.utils.config import load_yaml
from scripts.audit_vidhalluc_tsh_sth import validate_season_table1_records
from scripts.run_benchmark import build_plan, validate_experiment_protocol


def test_public_tsh_prompt_appends_official_sorting_instruction():
    question = "Action A. open\nAction B. close\n"
    assert build_tsh_prompt(question) == question + (
        "Sort these two actions in the order they occur in the video, and return which action "
        "happen before which one. If you only detect one action, return that action."
    )


def test_parser_compatible_tsh_prompt_requires_an_exact_official_token():
    question = "Action A. open\nAction B. close\n"
    prompt = build_tsh_prompt(question, output_protocol="parser_compatible")
    assert prompt.startswith(build_tsh_prompt(question))
    assert "Respond with exactly one of" in prompt
    assert "'AB' if Action A happens before Action B" in prompt
    assert "Do not include any other text." in prompt


def test_binary_order_tsh_prompt_allows_only_ab_or_ba():
    question = "Action A. open\nAction B. close\n"
    prompt = build_tsh_prompt(question, output_protocol="binary_order")
    assert "Respond with exactly one token: AB or BA." in prompt
    assert "Do not answer A or B alone" in prompt


def test_public_sth_prompt_is_preserved_verbatim():
    assert build_sth_prompt() == (
        "Watch the given video and determine if a scene change occurs. "
        "If no change occurs, respond: 'Scene change: No, Locations: None'. "
        "If there is a scene change, respond in the format: "
        "'Scene change: Yes, Locations: from [location1] to [location2].'"
    )


def test_vidhalluc_frame_order_is_chronological_and_capped():
    indices = vidhalluc_frame_indices(2000, 25.0, 32)
    assert len(indices) == 32
    assert indices == sorted(indices)


def test_season_eight_frame_protocol_is_distinct_from_public_vidhalluc():
    assert len(frame_indices(100, 8)) == 8
    assert len(vidhalluc_frame_indices(100, 25.0, 32)) == 4


def test_vidhalluc_tsh_sth_use_controlled_eight_frame_protocol():
    project = Path(__file__).resolve().parents[1]
    benchmark = load_yaml(project / "configs/benchmarks.yaml")["benchmarks"]["vidhalluc"]
    for task in ("tsh", "sth"):
        assert benchmark["task_sampling"][task]["num_frames"] == 8
        assert benchmark["task_sampling"][task]["strategy"] == "uniform"


def test_season_table1_profile_is_explicit_and_valid():
    project = Path(__file__).resolve().parents[1]
    config = load_yaml(project / "configs/vidhalluc_season_table1.yaml")
    validate_experiment_protocol(config)
    benchmark = config["benchmarks"][0]
    assert benchmark["protocol"] == "season_table1"
    assert benchmark["tasks"] == ["sth", "tsh"]
    assert benchmark["tsh_prompt_protocol"] == "official"
    plan = build_plan(config, allow_unvalidated=True)
    assert len(plan) == 24
    assert all(job["protocol"] == "season_table1" for job in plan)


def test_season_table1_rejects_non_official_tsh_prompt():
    config = {
        "name": "bad",
        "benchmarks": [{
            "name": "vidhalluc",
            "tasks": ["tsh"],
            "protocol": "season_table1",
            "tsh_prompt_protocol": "parser_compatible",
        }],
    }
    try:
        validate_experiment_protocol(config)
    except ValueError as exc:
        assert "official TSH prompt" in str(exc)
    else:
        raise AssertionError("non-official TSH prompt must be rejected")


def test_audit_rejects_incomplete_or_untagged_table1_records():
    record = PredictionRecord(
        sample_id="tsh:1",
        model="qwen2.5-vl-7b",
        method="base",
        benchmark="vidhalluc",
        task="tsh",
        prompt=build_tsh_prompt("Question"),
        frame_indices=list(range(8)),
        raw_output="AB",
        normalized_output="AB",
        ground_truth="AB",
        is_correct=True,
        parser_status="valid",
        sampling_config={"num_frames": 8, "strategy": "uniform"},
        generation_config={"do_sample": False},
        metadata={"tsh_prompt_protocol": "official"},
    )
    result = validate_season_table1_records([record])["qwen2.5-vl-7b/base/tsh"]
    assert result["status"] == "INVALID_OR_INCOMPLETE"
    assert result["protocol_tag_errors"] == 1
    assert result["unique_samples"] == 1
