#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.benchmarks.eventhallusion import EventHallusionLoader
from src.benchmarks.vidhalluc import VidHallucLoader
from src.benchmarks.videohallucer import VideoHallucerLoader
from src.experiments.subsets import build_tuning_subset_manifest, write_subset_manifest
from src.utils.config import load_yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Create one fixed random tuning subset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tsh-videos", type=int, default=100)
    parser.add_argument("--mcq-videos", type=int, default=50)
    parser.add_argument(
        "--tph-videos", type=int, default=100,
        help="Number of TPH branch videos; complete two-video pairs are always retained",
    )
    parser.add_argument("--event-videos", type=int, default=50)
    parser.add_argument("--output", default="manifests/positive_feature_tuning_seed42.json")
    return parser.parse_args()


def main():
    args = parse_args()
    configs = load_yaml(PROJECT / "configs" / "benchmarks.yaml")["benchmarks"]
    vh = configs["vidhalluc"]
    vhr = configs["videohallucer"]
    event = configs["eventhallusion"]

    manifest = build_tuning_subset_manifest(
        vidhalluc_samples=VidHallucLoader(vh["data_root"], ["tsh", "mcq"]).iter_samples(),
        videohallucer_samples=VideoHallucerLoader(vhr["data_root"], ["tph"]).iter_samples(),
        eventhallusion_samples=EventHallusionLoader(
            event["questions_root"], event["video_root"], event.get("tasks")
        ).iter_samples(),
        seed=args.seed,
        tsh_videos=args.tsh_videos,
        mcq_videos=args.mcq_videos,
        tph_videos=args.tph_videos,
        event_videos=args.event_videos,
    )
    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT / output
    write_subset_manifest(manifest, output)
    print(json.dumps({
        "manifest": str(output),
        "seed": args.seed,
        "selections": {
            key: {
                "unit": value["unit"],
                "selected_units": len(value["selected_units"]),
                "sample_records": len(value["sample_ids"]),
            }
            for key, value in manifest["selections"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
