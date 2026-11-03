"""Draw the 33-point MediaPipe Pose skeleton on a video and write an MP4.

Usage::

    python -m pose.overlay clip.mp4 --model models/pose_landmarker_full.task \\
        --out out/clip_overlay.mp4 [--npz out/clip.npz]

Frames with no detection are written unchanged, with a small
"no pose detected" label, so the output keeps the same length and timing as
the input. Low-visibility joints are drawn faded instead of dropped.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .landmarks import CONNECTIONS, NUM_LANDMARKS

# BGR colours: left side = blue-ish, right side = orange, centre = white.
_LEFT = (214, 120, 42)
_RIGHT = (52, 104, 235)
_CENTRE = (240, 240, 240)
_LEFT_IDX = {1, 2, 3, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31}
_RIGHT_IDX = {4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32}


def _colour(i: int):
    return _LEFT if i in _LEFT_IDX else _RIGHT if i in _RIGHT_IDX else _CENTRE


def _edge_colour(a: int, b: int):
    if a in _LEFT_IDX and b in _LEFT_IDX:
        return _LEFT
    if a in _RIGHT_IDX and b in _RIGHT_IDX:
        return _RIGHT
    return _CENTRE


def draw_skeleton(
    frame_bgr: np.ndarray,
    image_landmarks: np.ndarray | None,
    visibility: np.ndarray | None = None,
    min_visibility: float = 0.3,
    thickness: int | None = None,
) -> np.ndarray:
    """Draw in place and return ``frame_bgr``.

    ``image_landmarks`` is ``(33, >=2)`` in MediaPipe normalised coordinates,
    or ``None`` for "no detection".
    """
    import cv2

    h, w = frame_bgr.shape[:2]
    if thickness is None:
        thickness = max(2, round(min(w, h) / 240))
    if image_landmarks is None or not np.isfinite(image_landmarks[:, :2]).any():
        cv2.putText(frame_bgr, "no pose detected", (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame_bgr, "no pose detected", (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 1, cv2.LINE_AA)
        return frame_bgr

    pts = np.asarray(image_landmarks, float)[:NUM_LANDMARKS, :2] * (w, h)
    vis = np.ones(NUM_LANDMARKS) if visibility is None else np.nan_to_num(np.asarray(visibility, float), nan=0.0)
    ok = np.isfinite(pts).all(axis=1)

    def faded(c, v):
        if v >= min_visibility:
            return c
        return tuple(int(0.35 * x + 0.65 * 128) for x in c)

    for a, b in CONNECTIONS:
        if ok[a] and ok[b]:
            pa = tuple(int(round(x)) for x in pts[a])
            pb = tuple(int(round(x)) for x in pts[b])
            cv2.line(frame_bgr, pa, pb, faded(_edge_colour(a, b), min(vis[a], vis[b])),
                     thickness, cv2.LINE_AA)
    r = thickness + 1
    for i in range(NUM_LANDMARKS):
        if ok[i]:
            p = tuple(int(round(x)) for x in pts[i])
            cv2.circle(frame_bgr, p, r + 1, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(frame_bgr, p, r, faded(_colour(i), vis[i]), -1, cv2.LINE_AA)
    return frame_bgr


def open_writer(path: str | Path, fps: float, width: int, height: int):
    """Open an MP4 writer. Tries H.264 ('avc1') first, then falls back to 'mp4v'.

    OpenCV's pip wheels usually lack an H.264 encoder, so 'mp4v' (MPEG-4
    Part 2) is the usual result. It plays in VLC and most desktop players. For
    browsers or WhatsApp, re-encode with
    ``ffmpeg -i in.mp4 -c:v libx264 -pix_fmt yuv420p out.mp4``.
    """
    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for code in ("avc1", "mp4v"):
        wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*code), fps, (width, height))
        if wr.isOpened():
            return wr, code
        wr.release()
    raise IOError(f"Could not open an MP4 writer for {path}")


def render_overlay(video_path, out_path, detector, max_frames=None, min_visibility=0.3):
    """Detect and draw in one pass. Returns ``(PoseSequence, fourcc_used)``."""
    from .extract import open_video, process_video

    cap, fps, w, h = open_video(video_path)
    cap.release()
    writer, code = open_writer(out_path, fps, w, h)

    def on_frame(i, frame, lms, vis):
        writer.write(draw_skeleton(frame, lms, vis, min_visibility))

    try:
        seq = process_video(video_path, detector, on_frame=on_frame, max_frames=max_frames)
    finally:
        writer.release()
    return seq, code


def main(argv=None) -> int:
    from .extract import MediaPipeVideoDetector, save_npz

    ap = argparse.ArgumentParser(description="Write an MP4 with the MediaPipe pose skeleton drawn on.")
    ap.add_argument("video")
    ap.add_argument("--model", required=True, help="Path to pose_landmarker_{lite,full,heavy}.task")
    ap.add_argument("--out", help="Output MP4 (default: <video>_overlay.mp4)")
    ap.add_argument("--npz", help="Also save keypoints here (same format as pose.extract)")
    ap.add_argument("--min-detection", type=float, default=0.5)
    ap.add_argument("--min-tracking", type=float, default=0.5)
    ap.add_argument("--min-visibility", type=float, default=0.3, help="Fade joints below this visibility")
    ap.add_argument("--max-frames", type=int)
    a = ap.parse_args(argv)

    out = Path(a.out) if a.out else Path(a.video).with_name(Path(a.video).stem + "_overlay.mp4")
    with MediaPipeVideoDetector(a.model, a.min_detection, a.min_detection, a.min_tracking) as det:
        seq, code = render_overlay(a.video, out, det, a.max_frames, a.min_visibility)
    seq.model = Path(a.model).name
    if a.npz:
        save_npz(seq, a.npz)
    print(f"{a.video}: {seq.n_frames} frames, pose in {seq.detection_rate():.0%}, codec {code} -> {out}")
    if seq.n_frames and seq.detection_rate() < 0.5:
        print("  warning: pose found in under half the frames; check framing and lighting "
              "(docs/filming_protocol.md)", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
