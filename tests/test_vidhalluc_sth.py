import json

from src.benchmarks.vidhalluc.loader import VidHallucLoader, build_sth_prompt


def test_sth_prompt_matches_upstream_format():
    prompt = build_sth_prompt()
    assert "Scene change: No, Locations: None" in prompt
    assert "Scene change: Yes, Locations: from [location1] to [location2]." in prompt


def test_sth_loader_keeps_scene_and_location_ground_truth(tmp_path):
    root = tmp_path / "vidhalluc"
    data = root / "data"
    data.mkdir(parents=True)
    (data / "scene.mp4").write_bytes(b"")
    (root / "sth.json").write_text(json.dumps({
        "scene": {"Scene change": "Yes", "Locations": "from room to road."}
    }))
    sample = next(VidHallucLoader(data, ["sth"]).iter_samples())
    assert sample.ground_truth == "yes"
    assert sample.metadata["locations"] == "from room to road."

