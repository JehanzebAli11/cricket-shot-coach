"""Video-side tests that run without a MediaPipe model.

A fake detector stands in for MediaPipe so we can test decoding, timestamp
handling, missing detections, saving and the overlay writer. A real-model test
runs only if a .task file is present (set POSE_MODEL=/path/to/model.task).
"""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from pose import extract as E  # noqa: E402
from pose import landmarks as L  # noqa: E402
from conftest import make_skeleton  # noqa: E402

W, H, N_FRAMES, FPS = 160, 120, 12, 30.0


@pytest.fixture
def video(tmp_path) -> Path:
    path = tmp_path / "clip.mp4"
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    assert wr.isOpened()
    for i in range(N_FRAMES):
        frame = np.full((H, W, 3), 40, np.uint8)
        cv2.circle(frame, (20 + 8 * i, 60), 10, (255, 255, 255), -1)
        wr.write(frame)
    wr.release()
    return path


class FakeDetector:
    """Returns a skeleton on even frames and nothing on odd frames."""

    def __init__(self, fail_on=None):
        self.timestamps = []
        self.fail_on = fail_on or set()

    def detect(self, rgb, ts):
        assert rgb.shape == (H, W, 3)
        self.timestamps.append(ts)
        i = len(self.timestamps) - 1
        if i in self.fail_on:
            raise RuntimeError("boom")
        if i % 2:
            return None, None
        px = make_skeleton(scale=20, origin=(W / 2, H / 2))
        lms = [SimpleNamespace(x=x / W, y=y / H, z=0.0, visibility=0.9, presence=0.95) for x, y in px]
        world = [SimpleNamespace(x=x / 100, y=y / 100, z=0.0) for x, y in px]
        return lms, world

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


def test_process_video_handles_missing_and_errors(video):
    det = FakeDetector(fail_on={2})
    seq = E.process_video(video, det)
    assert seq.n_frames == N_FRAMES
    assert (seq.width, seq.height) == (W, H)
    assert seq.fps == pytest.approx(FPS, rel=0.01)
    # strictly increasing timestamps, required by MediaPipe VIDEO mode
    assert np.all(np.diff(det.timestamps) > 0)
    expected = np.array([i % 2 == 0 and i != 2 for i in range(N_FRAMES)])
    np.testing.assert_array_equal(seq.detected, expected)
    assert np.isnan(seq.image[1]).all() and np.isnan(seq.image[2]).all()
    assert np.isfinite(seq.image[0]).all()
    px = seq.pixels()
    assert px.shape == (N_FRAMES, 33, 2)
    np.testing.assert_allclose(px[0, L.L_HIP], make_skeleton(20, (W / 2, H / 2))[L.L_HIP], atol=1e-6)


def test_npz_and_json_roundtrip(video, tmp_path):
    seq = E.process_video(video, FakeDetector())
    seq.model = "fake.task"
    seq.meta = {"label": {"shot": "cover_drive"}}
    p = E.save_npz(seq, tmp_path / "k.npz")
    back = E.load_npz(p)
    np.testing.assert_array_equal(back.detected, seq.detected)
    np.testing.assert_allclose(back.image, seq.image, equal_nan=True)
    assert back.model == "fake.task" and back.meta == seq.meta
    j = json.loads(E.save_json(seq, tmp_path / "k.json").read_text())
    assert j["n_frames"] == N_FRAMES and j["frames"][1]["pose"] is None
    assert len(j["frames"][0]["pose"]) == 33 and j["landmark_names"][0] == "nose"


def test_overlay_writes_same_length_video(video, tmp_path):
    from pose.overlay import render_overlay

    out = tmp_path / "o.mp4"
    seq, code = render_overlay(video, out, FakeDetector())
    assert code in ("avc1", "mp4v")
    cap = cv2.VideoCapture(str(out))
    n = 0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
        n += 1
    cap.release()
    assert n == N_FRAMES == seq.n_frames
    # frame 0 has a coloured skeleton; frame 1 only the black/white label
    def coloured(f):
        f = f.astype(int)
        return int(((f.max(axis=2) - f.min(axis=2)) > 60).sum())

    assert coloured(frames[0]) > 50
    assert coloured(frames[1]) < 10


def test_draw_skeleton_none_and_nan():
    from pose.overlay import draw_skeleton

    f = np.zeros((60, 80, 3), np.uint8)
    draw_skeleton(f, None)
    assert f.any()  # label drawn
    g = np.zeros((60, 80, 3), np.uint8)
    lms = np.full((33, 3), np.nan)
    lms[L.L_SHOULDER] = (0.3, 0.3, 0)
    lms[L.L_ELBOW] = (0.5, 0.5, 0)
    draw_skeleton(g, lms, np.ones(33))
    assert g.any()


def test_missing_model_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="pose_landmarker"):
        E.MediaPipeVideoDetector(tmp_path / "nope.task")


def test_batch_script_with_fake_detector(video, tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import batch_process as B

    clips = tmp_path / "clips"
    clips.mkdir()
    video.rename(clips / "c001.mp4")
    (clips / "broken.mp4").write_bytes(b"not a video")
    labels = tmp_path / "labels.csv"
    labels.write_text("clip_id,handedness,shot\nc001,L,cover_drive\n", encoding="utf-8")
    monkeypatch.setattr(B, "MediaPipeVideoDetector", lambda model: FakeDetector())
    out = tmp_path / "out"
    rc = B.main([str(clips), "--model", "x.task", "--out", str(out), "--labels", str(labels), "--json"])
    assert rc == 2  # one clip failed, the batch still finished
    man = json.loads((out / "manifest.json").read_text())
    ok = next(r for r in man if r["clip"] == "c001.mp4")
    bad = next(r for r in man if r["clip"] == "broken.mp4")
    assert "error" in bad
    assert ok["left_handed"] is True and ok["label"]["shot"] == "cover_drive"
    assert (out / ok["keypoints"]).exists() and (out / ok["overlay"]).exists()
    assert (out / "keypoints" / "c001.json").exists()
    norm = np.load(out / ok["normalised"])
    assert norm.shape == (64, 33, 2)
    assert ok["metrics"]["front_knee_bend"] is not None
    # second run skips existing outputs
    rc = B.main([str(clips), "--model", "x.task", "--out", str(out), "--labels", str(labels)])
    man = json.loads((out / "manifest.json").read_text())
    assert next(r for r in man if r["clip"] == "c001.mp4").get("skipped") is True


@pytest.mark.skipif(not os.environ.get("POSE_MODEL"), reason="set POSE_MODEL=/path/to/pose_landmarker_*.task")
def test_real_mediapipe_model(video):
    pytest.importorskip("mediapipe")
    with E.MediaPipeVideoDetector(os.environ["POSE_MODEL"]) as det:
        seq = E.process_video(video, det)
    # synthetic frames contain no person: every frame should be a clean miss
    assert seq.n_frames == N_FRAMES and not seq.detected.any()
