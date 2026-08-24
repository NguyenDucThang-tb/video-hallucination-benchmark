#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.utils.config import load_yaml


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def parse_args():
    parser = argparse.ArgumentParser(description="Verify a local VidHalluc dataset without inference")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--dataset-source", default="chaoyuli/VidHalluc")
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--output", default="results/audit/vidhalluc_dataset_verification.json")
    return parser.parse_args()


def annotation_root(root: Path) -> Path:
    return root if (root / "tsh.json").is_file() else root.parent


def duplicate_json_keys(path: Path) -> list[str]:
    pairs = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=lambda value: value)
    return sorted(key for key, count in Counter(str(key) for key, _ in pairs).items() if count > 1)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def video_candidates(root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    if not root.exists():
        return index
    for path in root.rglob("*"):
        if path.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        for alias in {path.name, path.stem}:
            index[alias].append(path)
    return index


def resolve_exact(video_id: object, index: dict[str, list[Path]]) -> list[Path]:
    text = str(video_id or "").strip()
    path = Path(text)
    matches = []
    for alias in (text, path.name, path.stem):
        matches.extend(index.get(alias, []))
    return list(dict.fromkeys(matches))


def verify(root: Path, source: str, revision: str | None) -> dict:
    annotations = annotation_root(root)
    index = video_candidates(root)
    report = {
        "status": "NOT_EXECUTED" if not root.exists() else "VERIFIED_LOCAL_CONTENT_ONLY",
        "dataset_source": source,
        "dataset_revision": revision or "UNVERIFIED",
        "dataset_root": str(root),
        "annotation_root": str(annotations),
        "tsh_annotation_count": 0,
        "sth_annotation_count": 0,
        "missing_tsh_videos": [],
        "missing_sth_videos": [],
        "ambiguous_tsh_videos": {},
        "ambiguous_sth_videos": {},
        "duplicate_video_ids": {"tsh": [], "sth": []},
        "invalid_annotations": [],
        "usable_tsh_count": 0,
        "usable_sth_count": 0,
        "annotation_sha256": {},
        "expected_counts_match": False,
    }
    if not root.exists():
        report["reason"] = "Dataset root does not exist in this environment"
        return report

    for task, filename in (("tsh", "tsh.json"), ("sth", "sth.json")):
        path = annotations / filename
        if not path.is_file():
            report["invalid_annotations"].append({"task": task, "reason": "missing_annotation_file", "path": str(path)})
            continue
        report["annotation_sha256"][filename] = sha256(path)
        report["duplicate_video_ids"][task] = duplicate_json_keys(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        report[f"{task}_annotation_count"] = len(data)
        usable = 0
        for annotation_id, item in data.items():
            reasons = []
            if not isinstance(item, dict):
                report["invalid_annotations"].append({"task": task, "id": annotation_id, "reason": "not_an_object"})
                continue
            if task == "tsh":
                video_id = item.get("video")
                if not str(item.get("Question", "")).strip():
                    reasons.append("missing_question")
                if str(item.get("Correct Answer", "")).strip().upper() not in {"AB", "BA"}:
                    reasons.append("invalid_correct_answer")
            else:
                video_id = annotation_id
                label = str(item.get("Scene change", "")).strip().lower()
                locations = str(item.get("Locations", "")).strip()
                if label not in {"yes", "no"}:
                    reasons.append("invalid_scene_change")
                if not locations:
                    reasons.append("missing_locations")
                if label == "yes" and not (locations.lower().startswith("from ") and " to " in locations.lower()):
                    reasons.append("invalid_transition_locations")
                if label == "no" and locations.lower().rstrip(".") != "none":
                    reasons.append("invalid_no_change_locations")
            matches = resolve_exact(video_id, index)
            if not matches:
                report[f"missing_{task}_videos"].append(str(video_id))
            elif len(matches) > 1:
                report[f"ambiguous_{task}_videos"][str(video_id)] = [str(match) for match in matches]
            if reasons:
                report["invalid_annotations"].append({"task": task, "id": annotation_id, "reasons": reasons})
            if len(matches) == 1 and not reasons:
                usable += 1
        report[f"usable_{task}_count"] = usable

    report["expected_counts_match"] = (
        report["tsh_annotation_count"] == 600 and report["sth_annotation_count"] == 445
    )
    complete = (
        report["expected_counts_match"]
        and not report["missing_tsh_videos"]
        and not report["missing_sth_videos"]
        and not report["ambiguous_tsh_videos"]
        and not report["ambiguous_sth_videos"]
        and not report["duplicate_video_ids"]["tsh"]
        and not report["duplicate_video_ids"]["sth"]
        and not report["invalid_annotations"]
    )
    report["status"] = "MATCH" if complete and revision else ("PARTIAL" if complete else "DATASET_INCOMPLETE")
    if complete and not revision:
        report["reason"] = "Local contents pass structural checks, but dataset revision is unverified"
    return report


def main():
    args = parse_args()
    dataset_root = args.dataset_root
    if dataset_root is None:
        dataset_root = load_yaml(PROJECT / "configs/benchmarks.yaml")["benchmarks"]["vidhalluc"]["data_root"]
    root = Path(dataset_root).expanduser().resolve()
    report = verify(root, args.dataset_source, args.dataset_revision)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
