import json

from src.benchmarks.vidhalluc.loader import VidHallucLoader


def test_vidhalluc_loader_reads_ach_tsh_sth(tmp_path):
    root = tmp_path / "vidhalluc"
    data = root / "data"
    (data / "ACH_videos" / "ACH").mkdir(parents=True)
    (data / "TSH_videos" / "VidHalluc").mkdir(parents=True)
    (data / "STH_videos" / "STH").mkdir(parents=True)

    (data / "ACH_videos" / "ACH" / "clip_a.mp4").write_bytes(b"")
    (data / "TSH_videos" / "VidHalluc" / "video_tsh.mp4").write_bytes(b"")
    (data / "STH_videos" / "STH" / "video_sth.mp4").write_bytes(b"")

    (root / "ach_mcq.json").write_text(json.dumps({
        "1": {
            "clip_a": {
                "Question": "What is happening?",
                "Choices": {"A": "run", "B": "jump", "C": "sit", "D": "sleep"},
                "Correct Answer": "B",
            }
        }
    }))
    (root / "tsh.json").write_text(json.dumps({
        "1": {
            "video": "video_tsh",
            "Question": "Action A. run\\nAction B. jump",
            "Correct Answer": "AB",
        }
    }))
    (root / "sth.json").write_text(json.dumps({
        "video_sth": {
            "Scene change": "Yes",
            "Locations": "from room to street.",
        }
    }))

    loader = VidHallucLoader(data, ["ach", "tsh", "sth"])
    samples = list(loader.iter_samples())

    assert {sample.task for sample in samples} == {"ach", "tsh", "sth"}
    ach = next(sample for sample in samples if sample.task == "ach")
    tsh = next(sample for sample in samples if sample.task == "tsh")
    sth = next(sample for sample in samples if sample.task == "sth")

    assert ach.choices["B"] == "jump"
    assert ach.ground_truth == "B"
    assert tsh.ground_truth == "AB"
    assert sth.metadata["scene_change"] == "yes"
