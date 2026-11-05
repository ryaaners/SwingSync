"""Synthetic media here validates software behavior, never ML performance."""
import time
from fractions import Fraction

import av
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.swingsync.app import create_app
from backend.swingsync.store import Store
from backend.swingsync.video import inspect_video


@pytest.fixture
def synthetic_video(tmp_path):
    path = tmp_path / "software-fixture.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=25)
        stream.width, stream.height, stream.pix_fmt = 64, 64, "yuv420p"
        stream.codec_context.time_base = Fraction(1, 1000)
        for i, timestamp in enumerate([0, 40, 100, 180]):
            frame = av.VideoFrame.from_ndarray(np.full((64, 64, 3), i * 40, dtype=np.uint8), format="rgb24")
            frame.pts, frame.time_base = timestamp, Fraction(1, 1000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def wait_for_result(client, identifier):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        result = client.get(f"/api/analyses/{identifier}").json()
        if result["status"] not in {"queued", "processing"}:
            return result
        time.sleep(0.01)
    pytest.fail("Local worker did not finish within five seconds")


def send_video(client, data, filename="swing.mp4", **extra):
    return client.post("/api/analyses", files={"file": (filename, data, "video/mp4")},
                       data={"camera_view": "face-on", "handedness": "right", "club": "7 iron", **extra})


def test_actual_variable_frame_timestamps(synthetic_video):
    result = inspect_video(synthetic_video)
    assert result["frame_count"] == 4
    assert result["timestamps_seconds"] == pytest.approx([0, .04, .10, .18])
    assert result["source_timestamps_seconds"] == pytest.approx([0, .04, .10, .18])
    assert result["duration_seconds"] >= .18


def test_upload_inspect_export_delete_and_restart(tmp_path, synthetic_video):
    storage = tmp_path / "storage"
    with TestClient(create_app(storage)) as client:
        assert client.get("/api/health").json()["model"]["status"] == "unavailable"
        response = send_video(client, synthetic_video.read_bytes(), club='<script>alert("x")</script>')
        assert response.status_code == 202, response.text
        identifier = response.json()["id"]
        result = wait_for_result(client, identifier)
        assert result["status"] == "completed", result
        assert result["events"] == result["measurements"] == result["observations"] == []
        assert result["video"]["frame_count"] == 4
        assert result["model"]["status"] == "unavailable"
        assert client.get(f"/api/analyses/{identifier}/video").content == synthetic_video.read_bytes()
        report = client.get(f"/api/analyses/{identifier}/report")
        assert report.status_code == 200
        assert "No event estimates" in report.text
        assert "<script>" not in report.text
        assert "&lt;script&gt;" in report.text
    with TestClient(create_app(storage)) as client:
        assert len(client.get("/api/analyses").json()) == 1
        assert client.delete(f"/api/analyses/{identifier}").status_code == 204
        assert client.get(f"/api/analyses/{identifier}").status_code == 404
        assert not list(storage.glob("*.mp4"))


def test_invalid_uploads_and_cleanup(tmp_path):
    with TestClient(create_app(tmp_path, max_upload_bytes=1024)) as client:
        assert send_video(client, b"").status_code == 400
        assert send_video(client, b"x", "unsafe.txt").status_code == 415
        assert send_video(client, b"x" * 1025).status_code == 413
        assert send_video(client, b"x", camera_view="other").status_code == 422
        response = send_video(client, b"not a video", "../../invalid.mp4")
        result = wait_for_result(client, response.json()["id"])
        assert result["status"] == "failed"
        assert "could not be decoded" in result["error"]
        assert client.get(f"/api/analyses/{result['id']}/video").status_code == 409
        assert client.get(f"/api/analyses/{result['id']}/report").status_code == 409
        assert client.delete(f"/api/analyses/{result['id']}").status_code == 204
    assert not list(tmp_path.glob("*.part"))
    assert not list(tmp_path.glob("*.mp4"))


def test_restart_marks_processing_interrupted_without_relaunch(tmp_path):
    store = Store(tmp_path / "analyses.sqlite3")
    store.save({"id": "old", "status": "processing", "progress": 20, "error": None})
    (tmp_path / "orphan.part").write_bytes(b"partial upload")
    with TestClient(create_app(tmp_path)) as client:
        item = client.get("/api/analyses/old").json()
        assert item["status"] == "interrupted"
        assert "explicitly retry" in item["error"]
        assert not (tmp_path / "orphan.part").exists()


def test_second_worker_refused(tmp_path):
    with TestClient(create_app(tmp_path)):
        with pytest.raises(RuntimeError, match="already has a worker"):
            with TestClient(create_app(tmp_path)):
                pass


def test_reject_foreign_browser_origin_and_host(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/api/health", headers={"Host": "attacker.example"}).status_code == 403
        assert send_video(client, b"bad").status_code == 202
        response = client.post("/api/analyses", headers={"Origin": "https://attacker.example"})
        assert response.status_code == 403


def test_startup_removes_only_generated_unreferenced_media(tmp_path):
    orphan = tmp_path / ("a" * 32 + ".mp4")
    orphan.write_bytes(b"orphan after interrupted save")
    unrelated = tmp_path / "my-recording.mp4"
    unrelated.write_bytes(b"not app generated")
    tracked = tmp_path / ("b" * 32 + ".mp4")
    tracked.write_bytes(b"tracked")
    Store(tmp_path / "analyses.sqlite3").save({
        "id": "b" * 32, "status": "failed", "file_name": tracked.name,
    })
    with TestClient(create_app(tmp_path)) as client:
        assert len(client.get("/api/analyses").json()) == 1
        assert not orphan.exists()
        assert unrelated.exists()
        assert tracked.exists()
