import json

from scripts.verify_vidhalluc_dataset import verify


def make_dataset(tmp_path, tsh_count=600, sth_count=445):
    root = tmp_path / "vidhalluc"
    data = root / "data"
    data.mkdir(parents=True)
    tsh = {}
    sth = {}
    for index in range(tsh_count):
        video_id = f"tsh_{index:03d}"
        (data / f"{video_id}.mp4").write_bytes(b"video")
        tsh[str(index)] = {
            "video": video_id,
            "Question": "Action A. open\nAction B. close\n",
            "Correct Answer": "AB" if index % 2 == 0 else "BA",
        }
    for index in range(sth_count):
        video_id = f"sth_{index:03d}"
        (data / f"{video_id}.mp4").write_bytes(b"video")
        sth[video_id] = {
            "Scene change": "Yes" if index % 2 == 0 else "No",
            "Locations": "from room to street." if index % 2 == 0 else "None",
        }
    (root / "tsh.json").write_text(json.dumps(tsh))
    (root / "sth.json").write_text(json.dumps(sth))
    return data


def test_dataset_verifier_accepts_complete_release_shape(tmp_path):
    data = make_dataset(tmp_path)
    report = verify(data, "chaoyuli/VidHalluc", "test-revision")
    assert report["status"] == "MATCH"
    assert report["tsh_annotation_count"] == 600
    assert report["sth_annotation_count"] == 445
    assert report["usable_tsh_count"] == 600
    assert report["usable_sth_count"] == 445


def test_dataset_verifier_reports_missing_video_without_reducing_annotations(tmp_path):
    data = make_dataset(tmp_path, tsh_count=1, sth_count=1)
    (data / "tsh_000.mp4").unlink()
    report = verify(data, "chaoyuli/VidHalluc", None)
    assert report["tsh_annotation_count"] == 1
    assert report["usable_tsh_count"] == 0
    assert report["missing_tsh_videos"] == ["tsh_000"]
