#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.data.sampler import sample_video
from src.methods.base import BaseMethod
from src.methods.season.season_method import SeasonMethod
from src.models import GenerationConfig, LlavaOVAdapter
from src.utils.config import load_yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Real-video LLaVA-OV SEASON smoke test")
    parser.add_argument("--video", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--output", default="results/profiles/season_smoke.json")
    return parser.parse_args()


def timed_run(label, method, frames, prompt, generation):
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    output = method.generate(frames, prompt, generation)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    return {
        "label": label,
        "text": output.text,
        "latency_seconds": elapsed,
        "peak_gpu_memory_gb": peak / 1e9,
        "diagnostics": output.diagnostics,
    }


def main():
    args = parse_args()
    if not 1 <= args.steps <= 10:
        raise SystemExit("--steps must be between 1 and 10 for the smoke test")
    model_config = load_yaml(PROJECT / "configs/models.yaml")["models"]["llava-ov-7b"]
    method_config = load_yaml(PROJECT / "configs/methods.yaml")["methods"]["season"]
    model_path = args.model_path or os.environ.get("MODEL_DIR") or model_config.get("local_path")
    model = LlavaOVAdapter(model_config["checkpoint"], model_path)
    frames, manifest = sample_video(args.video, num_frames=8, strategy="uniform")
    generation = GenerationConfig(max_new_tokens=args.steps)

    report = {
        "model": model.name,
        "checkpoint": model.model_path,
        "video": str(Path(args.video).resolve()),
        "prompt": args.prompt,
        "manifest": manifest.to_dict(),
        "frames": int(len(frames)),
        "status": "running",
    }
    print("SEASON smoke test")
    print("-----------------")
    print(f"model: {model.name}")
    print(f"video: {args.video}")
    print(f"frames: {len(frames)}")
    try:
        report["base"] = timed_run(
            "base", BaseMethod(model), frames, args.prompt, generation
        )
        print("base decoding: PASS")
        report["season"] = timed_run(
            "season", SeasonMethod(model, method_config), frames, args.prompt, generation
        )
        diagnostics = report["season"]["diagnostics"]
        branches = diagnostics["branch_diagnostics"]
        for name in ("original", "spatial", "temporal"):
            if branches[name].get("decode_call_count", 0) < 1:
                raise RuntimeError(f"SEASON {name} branch did not execute")
            print(f"{name} forward: PASS")
        if not diagnostics["token_diagnostics"]:
            raise RuntimeError("SEASON produced no token diagnostics")
        if not report["season"]["text"]:
            raise RuntimeError("SEASON decoded empty text")
        print("diagnostic score: PASS")
        print("contrastive logits: PASS")
        print("autoregressive decoding: PASS")
        report["status"] = "PASS"
    except Exception as exc:
        report["status"] = "FAIL"
        report["error"] = {
            "class": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        print(f"result: FAIL ({type(exc).__name__}: {exc})")
    else:
        print("result: PASS")
        print(f"base output: {report['base']['text']!r}")
        print(f"season output: {report['season']['text']!r}")
        print(f"base latency: {report['base']['latency_seconds']:.3f}s")
        print(f"season latency: {report['season']['latency_seconds']:.3f}s")
        print(f"season peak GPU memory: {report['season']['peak_gpu_memory_gb']:.2f} GB")
    finally:
        output_path = PROJECT / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"report: {output_path}")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
