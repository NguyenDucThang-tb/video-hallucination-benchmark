from pathlib import Path

from src.benchmarks.vidhalluc.loader import build_sth_prompt, build_tsh_prompt
from src.data.sampler import frame_indices, vidhalluc_frame_indices
from src.utils.config import load_yaml


def test_public_tsh_prompt_is_preserved_verbatim():
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
