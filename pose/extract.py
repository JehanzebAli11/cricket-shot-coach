"""Run MediaPipe Pose Landmarker over a video and save per-frame keypoints.

Usage::

    python -m pose.extract clip.mp4 --model models/pose_landmarker_full.task \\
        --out out/clip.npz [--json out/clip.json]

The model is a ``.task`` bundle from Google's MediaPipe model storage (see
README, "Models"). Pose Landmarker runs in **VIDEO** mode, which tracks
across frames and needs strictly increasing timestamps.

Saved arrays (frames with no detection are NaN, never zeros):

==============  ============  ==================================================
key             shape         meaning
==============  ============  ==================================================
``image``       (T, 33, 3)    x, y in [0, 1] of width/height; z relative depth
``visibility``  (T, 33)       MediaPipe visibility score
``presence``    (T, 33)       MediaPipe presence score
``world``       (T, 33, 3)    metres, origin at hip centre (MediaPipe world)
``detected``    (T,)          bool, a pose was found in this frame
``timestamps``  (T,)          ms, as passed to the detector
``fps``, ``width``, ``height``, ``source``, ``model``
==============  ============  ==================================================
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Protocol

import numpy as np

from .landmarks import NAMES, NUM_LANDMARKS

# --------------------------------------------------------------------------- #
# Data container
# --------------------------------------------------------------------------- #


@dataclass
class PoseSequence:
    image: np.ndarray
    visibility: np.ndarray
    presence: np.ndarray
    world: np.ndarray
    detected: np.ndarray
    timestamps: np.ndarray
    fps: float
    width: int
    height: int
    source: str = ""
    model: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return int(self.image.shape[0])

    def pixels(self) -> np.ndarray:
        """(T, 33, 2) pixel coordinates, the isotropic frame the angle code expects."""
        return to_pixels(self.image, self.width, self.height)

    def detection_rate(self) -> float:
        return float(self.detected.mean()) if self.n_frames else 0.0


def to_pixels(image_xy: np.ndarray, width: int, height: int) -> np.ndarray:
    """Convert MediaPipe normalised image coords to pixels (drops z)."""
    xy = np.asarray(image_xy, dtype=float)[..., :2].copy()
    xy[..., 0] *= width
    xy[..., 1] *= height
    return xy


def empty_sequence(n: int, fps: float, width: int, height: int, **kw) -> PoseSequence:
    nan3 = np.full((n, NUM_LANDMARKS, 3), np.nan)
    nan1 = np.full((n, NUM_LANDMARKS), np.nan)
    return PoseSequence(nan3.copy(), nan1.copy(), nan1.copy(), nan3.copy(),
                        np.zeros(n, bool), np.zeros(n, np.int64), fps, width, height, **kw)


# --------------------------------------------------------------------------- #
# Detector plumbing
# --------------------------------------------------------------------------- #


class VideoDetector(Protocol):
    """Anything with MediaPipe's ``detect_for_video(mp_image, ts_ms)`` shape.

    ``detect(frame_rgb, ts_ms)`` must return ``(image_landmarks, world_landmarks)``
    as lists of objects with ``x, y, z, visibility, presence`` attributes,
    or ``(None, None)`` when no pose is found.
    """

    def detect(self, frame_rgb: np.ndarray, timestamp_ms: int): ...  # pragma: no cover

    def close(self) -> None: ...  # pragma: no cover


class MediaPipeVideoDetector:
    """Thin wrapper over ``mediapipe.tasks.python.vision.PoseLandmarker`` (VIDEO mode)."""

    def __init__(
        self,
        model_path: str | Path,
        min_pose_detection_confidence: float = 0.5,
        min_pose_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Pose model not found: {model_path}. Download pose_landmarker_lite.task or "
                "pose_landmarker_full.task (see README, 'Models') and pass --model."
            )
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python import vision
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("mediapipe is not installed: pip install mediapipe") from exc
        self._mp = mp
        opts = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,  # the batter; a keeper/umpire in frame is a known risk
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(opts)

    def detect(self, frame_rgb: np.ndarray, timestamp_ms: int):
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                             data=np.ascontiguousarray(frame_rgb))
        res = self._landmarker.detect_for_video(img, int(timestamp_ms))
        if not res.pose_landmarks:
            return None, None
        world = res.pose_world_landmarks[0] if res.pose_world_landmarks else None
        return res.pose_landmarks[0], world

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# --------------------------------------------------------------------------- #
# Video I/O
# --------------------------------------------------------------------------- #


def open_video(path: str | Path):
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if not np.isfinite(fps) or fps <= 1e-3:
        fps = 30.0  # some containers report 0; fall back rather than divide by zero
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, float(fps), w, h


def iter_frames(cap, fps: float) -> Iterator[tuple[int, int, np.ndarray]]:
    """Yield ``(index, timestamp_ms, frame_bgr)`` with strictly increasing timestamps."""
    import cv2

    last_ts = -1
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        pos = cap.get(cv2.CAP_PROP_POS_MSEC)
        ts = int(round(pos)) if pos and pos > 0 else int(round(i * 1000.0 / fps))
        ts = max(ts, last_ts + 1)  # VIDEO mode rejects non-increasing timestamps
        last_ts = ts
        yield i, ts, frame
        i += 1


def _score(lm, name: str) -> float:
    v = getattr(lm, name, None)
    return float(v) if v is not None else np.nan


def _fill(arr: dict, lms, world) -> None:
    """Copy one frame's landmark objects into the per-frame arrays."""
    for j, lm in enumerate(lms[:NUM_LANDMARKS]):
        arr["image"][j] = (lm.x, lm.y, lm.z)
        arr["visibility"][j] = _score(lm, "visibility")
        arr["presence"][j] = _score(lm, "presence")
    if world is not None:
        for j, lm in enumerate(world[:NUM_LANDMARKS]):
            arr["world"][j] = (lm.x, lm.y, lm.z)


def process_video(
    video_path: str | Path,
    detector: VideoDetector,
    on_frame=None,
    max_frames: int | None = None,
) -> PoseSequence:
    """Run ``detector`` over every frame.

    ``on_frame(i, frame_bgr, image_landmarks_or_None)`` is called for each frame;
    the overlay writer uses it so the video is decoded only once.
    """
    import cv2

    cap, fps, w, h = open_video(video_path)
    image, vis, pres, world, det, tss = [], [], [], [], [], []
    try:
        for i, ts, frame in iter_frames(cap, fps):
            if max_frames is not None and i >= max_frames:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                lms, wlms = detector.detect(rgb, ts)
            except Exception as exc:  # noqa: BLE001 - one bad frame must not kill a clip
                print(f"[extract] frame {i}: detector error {exc!r}; treating as no detection",
                      file=sys.stderr)
                lms, wlms = None, None
            arr = {
                "image": np.full((NUM_LANDMARKS, 3), np.nan),
                "visibility": np.full(NUM_LANDMARKS, np.nan),
                "presence": np.full(NUM_LANDMARKS, np.nan),
                "world": np.full((NUM_LANDMARKS, 3), np.nan),
            }
            if lms:
                _fill(arr, lms, wlms)
            image.append(arr["image"])
            vis.append(arr["visibility"])
            pres.append(arr["presence"])
            world.append(arr["world"])
            det.append(bool(lms))
            tss.append(ts)
            if on_frame is not None:
                on_frame(i, frame, arr["image"] if lms else None, arr["visibility"] if lms else None)
    finally:
        cap.release()
    n = len(image)
    if n == 0:
        return empty_sequence(0, fps, w, h, source=str(video_path))
    return PoseSequence(
        np.stack(image), np.stack(vis), np.stack(pres), np.stack(world),
        np.array(det, bool), np.array(tss, np.int64), fps, w, h, source=str(video_path),
    )


# --------------------------------------------------------------------------- #
# Save / load
# --------------------------------------------------------------------------- #


def save_npz(seq: PoseSequence, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, image=seq.image, visibility=seq.visibility, presence=seq.presence,
        world=seq.world, detected=seq.detected, timestamps=seq.timestamps,
        fps=seq.fps, width=seq.width, height=seq.height,
        source=np.array(seq.source), model=np.array(seq.model),
        meta=np.array(json.dumps(seq.meta)), landmark_names=np.array(NAMES),
    )
    return path


def load_npz(path: str | Path) -> PoseSequence:
    with np.load(path, allow_pickle=False) as z:
        return PoseSequence(
            z["image"], z["visibility"], z["presence"], z["world"], z["detected"].astype(bool),
            z["timestamps"], float(z["fps"]), int(z["width"]), int(z["height"]),
            str(z["source"]), str(z["model"]), json.loads(str(z["meta"])),
        )


def _nan_to_none(a: np.ndarray):
    return [None if not np.isfinite(v) else round(float(v), 6) for v in np.ravel(a)]


def to_json_dict(seq: PoseSequence) -> dict:
    frames = []
    for i in range(seq.n_frames):
        if not seq.detected[i]:
            frames.append({"i": i, "t_ms": int(seq.timestamps[i]), "pose": None})
            continue
        frames.append({
            "i": i,
            "t_ms": int(seq.timestamps[i]),
            "pose": [
                {"x": x, "y": y, "z": z, "v": v, "p": p, "wx": wx, "wy": wy, "wz": wz}
                for (x, y, z), v, p, (wx, wy, wz) in zip(
                    (_nan_to_none(r) for r in seq.image[i]),
                    _nan_to_none(seq.visibility[i]),
                    _nan_to_none(seq.presence[i]),
                    (_nan_to_none(r) for r in seq.world[i]),
                )
            ],
        })
    return {
        "source": seq.source, "model": seq.model, "fps": seq.fps,
        "width": seq.width, "height": seq.height, "landmark_names": NAMES,
        "n_frames": seq.n_frames, "detection_rate": round(seq.detection_rate(), 4),
        "meta": seq.meta, "frames": frames,
    }


def save_json(seq: PoseSequence, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_json_dict(seq)), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Extract per-frame pose keypoints from a video.")
    ap.add_argument("video")
    ap.add_argument("--model", required=True, help="Path to pose_landmarker_{lite,full,heavy}.task")
    ap.add_argument("--out", help="Output .npz (default: <video>.npz next to the input)")
    ap.add_argument("--json", help="Also write a JSON file here")
    ap.add_argument("--min-detection", type=float, default=0.5)
    ap.add_argument("--min-tracking", type=float, default=0.5)
    ap.add_argument("--max-frames", type=int)
    a = ap.parse_args(argv)
    out = Path(a.out) if a.out else Path(a.video).with_suffix(".npz")
    with MediaPipeVideoDetector(a.model, a.min_detection, a.min_detection, a.min_tracking) as det:
        seq = process_video(a.video, det, max_frames=a.max_frames)
    seq.model = Path(a.model).name
    save_npz(seq, out)
    if a.json:
        save_json(seq, a.json)
    print(f"{a.video}: {seq.n_frames} frames, pose found in {seq.detection_rate():.0%} -> {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
