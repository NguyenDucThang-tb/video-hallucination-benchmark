#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from scripts.run_benchmark import instantiate_model, load_method_configs
from src.data.sampler import sample_video
from src.methods.tcd import TCDMethod
from src.models.base import GenerationConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Profile Base and TCD on the same sampled video frames")
    parser.add_argument("--model", choices=["llava-ov-7b", "qwen2.5-vl-7b"], required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--prompt", default="Describe the events in this video.")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, nargs="+", default=[8, 32, 128])
    parser.add_argument("--output", type=Path, default=PROJECT / "results" / "profiles" / "tcd_profile.json")
    return parser.parse_args()


def synchronize(adapter) -> float:
    if not adapter.torch.cuda.is_available():
        return 0.0
    started = time.perf_counter()
    adapter.torch.cuda.synchronize()
    return time.perf_counter() - started


def reset_peak_memory(adapter) -> None:
    if adapter.torch.cuda.is_available():
        adapter.torch.cuda.reset_peak_memory_stats()


def peak_memory_gb(adapter) -> float | None:
    if not adapter.torch.cuda.is_available():
        return None
    return adapter.torch.cuda.max_memory_allocated() / 1e9


def sum_step(diagnostics: dict, key: str) -> float:
    return sum(float(step.get(key, 0.0)) for step in diagnostics.get("decode_steps", []))


def summarize_branch(diagnostics: dict) -> dict:
    steps = diagnostics.get("decode_steps", [])
    step_totals = [
        sum(float(step.get(key, 0.0)) for key in (
            "sync_generated_seconds",
            "prepare_inputs_seconds",
            "forward_seconds",
            "cache_update_seconds",
        ))
        for step in steps
    ]
    return {
        "preprocessing_seconds": diagnostics.get("preprocessing_seconds"),
        "first_token_seconds": steps[0].get("forward_seconds") if steps else None,
        "first_token_total_seconds": step_totals[0] if step_totals else None,
        "subsequent_token_seconds": [step.get("forward_seconds") for step in steps[1:]],
        "subsequent_token_total_seconds": step_totals[1:],
        "prepare_inputs_seconds": sum_step(diagnostics, "prepare_inputs_seconds"),
        "sync_generated_tokens_seconds": sum_step(diagnostics, "sync_generated_seconds"),
        "cache_update_seconds": sum_step(diagnostics, "cache_update_seconds"),
        "cuda_synchronize_seconds": diagnostics.get("cuda_sync_seconds", 0.0),
        "vision_supplied_steps": sum(bool(step.get("vision_inputs_supplied")) for step in steps),
        "input_ids_lengths": [step.get("input_ids_length") for step in steps],
        "mm_token_type_ids_lengths": [step.get("mm_token_type_ids_length") for step in steps],
        "attention_mask_lengths": [step.get("attention_mask_length") for step in steps],
        "past_before": [step.get("past_before") for step in steps],
        "past_after": [step.get("past_after") for step in steps],
    }


def install_vision_timer(adapter, active_branch: dict, vision_times: dict):
    candidates = (
        getattr(adapter.model, "visual", None),
        getattr(adapter.model, "vision_tower", None),
        getattr(getattr(adapter.model, "model", None), "visual", None),
        getattr(getattr(adapter.model, "model", None), "vision_tower", None),
    )
    module = next((candidate for candidate in candidates if candidate is not None), None)
    if module is None:
        return []
    holder = {"started": None}

    def before(_module, _inputs):
        synchronize(adapter)
        holder["started"] = time.perf_counter()

    def after(_module, _inputs, _output):
        synchronize(adapter)
        if holder["started"] is not None:
            branch = active_branch.get("name", "unknown")
            vision_times.setdefault(branch, []).append(time.perf_counter() - holder["started"])

    return [module.register_forward_pre_hook(before), module.register_forward_hook(after)]


def profile_base(adapter, frames, prompt: str, max_new_tokens: int, active_branch: dict) -> dict:
    reset_peak_memory(adapter)
    synchronize(adapter)
    started = time.perf_counter()
    state = adapter.prepare_branch(
        frames,
        prompt,
        branch="original",
        profile=True,
        preserve_logits_on_device=True,
    )
    generated = []
    active_branch["name"] = "original"
    for _ in range(max_new_tokens):
        output = adapter.decode_step(state, generated)
        token = int(output.logits.argmax().item())
        generated.append(token)
        if adapter.is_eos(token):
            break
    synchronize(adapter)
    decode_started = time.perf_counter()
    text = adapter.decode_token_ids(generated)
    token_decode_seconds = time.perf_counter() - decode_started
    total_seconds = time.perf_counter() - started
    return {
        "method": "base",
        "frames": len(frames),
        "max_tokens": max_new_tokens,
        "output_tokens": len(generated),
        "total_seconds": total_seconds,
        "tokens_per_second": len(generated) / total_seconds if total_seconds else None,
        "peak_memory_gb": peak_memory_gb(adapter),
        "token_decode_seconds": token_decode_seconds,
        "output": text,
        "original": summarize_branch(state["diagnostics"]),
    }


def profile_tcd(adapter, frames, prompt: str, max_new_tokens: int, active_branch: dict) -> dict:
    reset_peak_memory(adapter)
    config = dict(load_method_configs()["tcd"])
    config.update({"batch_size": 1, "profile": True})
    method = TCDMethod(adapter, config)
    original_decode = adapter.decode_step

    def tracked_decode(state, token_ids, output_attentions=False):
        active_branch["name"] = state["branch"]
        return original_decode(state, token_ids, output_attentions=output_attentions)

    adapter.decode_step = tracked_decode
    synchronize(adapter)
    started = time.perf_counter()
    try:
        output = method.generate(frames, prompt, GenerationConfig(max_new_tokens=max_new_tokens))
    finally:
        adapter.decode_step = original_decode
    synchronize(adapter)
    total_seconds = time.perf_counter() - started
    output_tokens = int(output.diagnostics["generated_token_count"])
    return {
        "method": "tcd",
        "frames": len(frames),
        "negative_frames": output.diagnostics["negative_frame_count"],
        "negative_frame_positions": output.diagnostics["negative_frame_positions"],
        "max_tokens": max_new_tokens,
        "output_tokens": output_tokens,
        "total_seconds": total_seconds,
        "tokens_per_second": output_tokens / total_seconds if total_seconds else None,
        "peak_memory_gb": peak_memory_gb(adapter),
        "all_masked_fallbacks": output.diagnostics["all_masked_fallbacks"],
        "contrast_seconds": output.diagnostics["contrast_seconds"],
        "token_decode_seconds": output.diagnostics["token_decode_seconds"],
        "output": output.text,
        "original": summarize_branch(output.diagnostics["original_branch"]),
        "negative": summarize_branch(output.diagnostics["negative_branch"]),
    }


def main() -> None:
    args = parse_args()
    if not args.video.exists():
        raise SystemExit(f"Video does not exist: {args.video}")

    load_started = time.perf_counter()
    adapter = instantiate_model(args.model)
    synchronize(adapter)
    model_load_seconds = time.perf_counter() - load_started

    sampling_started = time.perf_counter()
    frames, manifest = sample_video(args.video, num_frames=args.num_frames, strategy="uniform")
    video_sampling_seconds = time.perf_counter() - sampling_started

    active_branch = {"name": "unknown"}
    vision_times: dict[str, list[float]] = {}
    handles = install_vision_timer(adapter, active_branch, vision_times)
    rows = []
    try:
        for max_new_tokens in args.max_new_tokens:
            vision_times.clear()
            base = profile_base(adapter, frames, args.prompt, max_new_tokens, active_branch)
            base["vision_encoder_seconds"] = dict(vision_times)
            rows.append(base)

            vision_times.clear()
            tcd = profile_tcd(adapter, frames, args.prompt, max_new_tokens, active_branch)
            tcd["vision_encoder_seconds"] = dict(vision_times)
            rows.append(tcd)
    finally:
        for handle in handles:
            handle.remove()

    report = {
        "research_result": False,
        "model": args.model,
        "checkpoint": adapter.checkpoint,
        "video": str(args.video.resolve()),
        "prompt": args.prompt,
        "model_load_seconds": model_load_seconds,
        "video_sampling_seconds": video_sampling_seconds,
        "manifest": manifest.to_dict(),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
