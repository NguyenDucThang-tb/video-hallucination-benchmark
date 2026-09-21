from pathlib import Path

from scripts.run_benchmark import load_model_configs, resolve_method_config
from src.experiments.qwen25_vl_large import (
    BENCHMARK_TASKS,
    METHODS,
    MODELS,
    RESOURCE_PROFILES,
    iter_matrix,
)


def test_large_qwen_models_are_registered():
    configs = load_model_configs()
    assert configs["qwen2.5-vl-32b"]["checkpoint"] == "Qwen/Qwen2.5-VL-32B-Instruct"
    assert configs["qwen2.5-vl-72b"]["checkpoint"] == "Qwen/Qwen2.5-VL-72B-Instruct"
    assert all(configs[name]["adapter"] == "qwen25_vl" for name in MODELS)


def test_large_qwen_matrix_has_all_tasks_methods_and_models():
    assert sum(map(len, BENCHMARK_TASKS.values())) == 22
    matrix = list(iter_matrix())
    assert len(matrix) == 2 * 5 * 22
    assert {row[0] for row in matrix} == set(MODELS)
    assert {row[1] for row in matrix} == set(METHODS)


def test_72b_requests_more_gpus_than_32b():
    assert RESOURCE_PROFILES["qwen2.5-vl-32b"].ngpus == 1
    assert RESOURCE_PROFILES["qwen2.5-vl-72b"].ngpus == 2


def test_large_qwen_season_uses_final_four_layers():
    assert resolve_method_config("season", model_name="qwen2.5-vl-32b")[
        "attention_layers"
    ] == [60, 61, 62, 63]
    assert resolve_method_config("season", model_name="qwen2.5-vl-72b")[
        "attention_layers"
    ] == [76, 77, 78, 79]


def test_large_qwen_positive_transfer_configs():
    expected = {
        None: (0.4, 0.4, 0.8),
        "tempcompass": (0.0, 0.0, 0.6),
        "motionbench": (0.0, 0.0, 0.2),
    }
    for model in MODELS:
        for benchmark, values in expected.items():
            config = resolve_method_config(
                "positive_feature", model_name=model, benchmark=benchmark
            )
            assert (config["alpha"], config["alpha_s"], config["beta"]) == values


def test_large_qwen_pbs_uses_controlled_visual_budget():
    text = Path("pbs/qwen25_vl_large_task_h200.pbs").read_text(encoding="utf-8")
    assert "QWEN25_VL_REQUIRE_GPU_ONLY=1" in text
    assert "QWEN25_VL_MAX_PIXELS=$((384 * 28 * 28))" in text
    assert "official_prompts_controlled_8_frames" in text
