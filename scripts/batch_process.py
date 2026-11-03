#!/usr/bin/env python3
"""Process a folder of clips: keypoints (.npz, optional .json), overlay MP4s,
normalised sequences and per-clip form metrics, plus a run manifest.

    python scripts/batch_process.py clips/ --model models/pose_landmarker_full.task \\
        --out out/ [--labels clips/labels.csv] [--json] [--no-overlay] [--frames 64]

``labels.csv`` (optional) follows docs/filming_protocol.md. Columns used here:
``clip_id`` (file name without extension), ``handedness`` (R/L). Other columns
are copied into the manifest as they are. Clips with no label are treated as
right-handed, and the manifest flags them.

Re-running skips clips whose outputs already exist (``--force`` to redo).
A clip that fails is logged and the batch carries on.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pose import angles as A  # noqa: E402
from pose.extract import MediaPipeVideoDetector, load_npz, process_video, save_json, save_npz  # noqa: E402
from pose.normalise import normalise_sequence  # noqa: E402

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}


def find_clips(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in VIDEO_EXTS and not p.name.startswith("."))


def read_labels(path: Path | None) -> dict[str, dict]:
    if not path:
        return {}
    with open(path, encoding="utf-8", newline="") as fh:
        return {row["clip_id"]: row for row in csv.DictReader(fh)}


def is_left_handed(label: dict | None) -> bool:
    h = (label or {}).get("handedness", "R").strip().upper()
    return h in ("L", "LH", "LEFT")


def metric_summary(norm: np.ndarray) -> dict:
    out = {}
    for name, series in A.summary(norm).items():
        finite = series[np.isfinite(series)]
        out[name] = None if finite.size == 0 else {
            "min": round(float(finite.min()), 3),
            "max": round(float(finite.max()), 3),
            "median": round(float(np.median(finite)), 3),
        }
    return out


def process_clip(clip: Path, root: Path, out: Path, detector_factory, label: dict | None,
                 overlay: bool, write_json: bool, n_frames: int, force: bool) -> dict:
    rel = clip.relative_to(root).with_suffix("")
    npz_path = out / "keypoints" / rel.with_suffix(".npz")
    ov_path = out / "overlays" / (str(rel) + "_overlay.mp4")
    norm_path = out / "normalised" / rel.with_suffix(".npy")
    rec = {"clip": str(clip.relative_to(root)), "clip_id": clip.stem, "labelled": label is not None}
    left = is_left_handed(label)
    rec["left_handed"] = left

    if npz_path.exists() and (ov_path.exists() or not overlay) and not force:
        seq = load_npz(npz_path)
        rec["skipped"] = True
    else:
        t0 = time.time()
        with detector_factory() as det:
            if overlay:
                from pose.overlay import render_overlay

                seq, codec = render_overlay(clip, ov_path, det)
                rec["overlay"] = str(ov_path.relative_to(out))
                rec["codec"] = codec
            else:
                seq = process_video(clip, det)
        seq.meta = {"label": label or {}}
        save_npz(seq, npz_path)
        if write_json:
            save_json(seq, npz_path.with_suffix(".json"))
        rec["seconds"] = round(time.time() - t0, 2)

    rec.update(n_frames=seq.n_frames, fps=seq.fps, detection_rate=round(seq.detection_rate(), 4),
               keypoints=str(npz_path.relative_to(out)))
    if seq.detected.sum() >= 2:
        norm = normalise_sequence(seq.pixels(), left_handed=left, n_frames=n_frames)
        norm_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(norm_path, norm)
        rec["normalised"] = str(norm_path.relative_to(out))
        rec["metrics"] = metric_summary(norm)
    else:
        rec["warning"] = "fewer than 2 frames with a pose; nothing normalised"
    if label:
        rec["label"] = label
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--labels", type=Path)
    ap.add_argument("--json", action="store_true", help="Also write keypoints as JSON")
    ap.add_argument("--no-overlay", action="store_true")
    ap.add_argument("--frames", type=int, default=64, help="Resample normalised sequences to N frames")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)

    clips = find_clips(a.folder)
    if not clips:
        print(f"No video files under {a.folder}", file=sys.stderr)
        return 1
    labels = read_labels(a.labels)
    a.out.mkdir(parents=True, exist_ok=True)
    factory = lambda: MediaPipeVideoDetector(a.model)  # noqa: E731 (fresh tracker per clip)

    manifest, failures = [], 0
    for k, clip in enumerate(clips, 1):
        try:
            rec = process_clip(clip, a.folder, a.out, factory, labels.get(clip.stem),
                               not a.no_overlay, a.json, a.frames, a.force)
            flag = " (skipped, exists)" if rec.get("skipped") else ""
            print(f"[{k}/{len(clips)}] {rec['clip']}: {rec['n_frames']} frames, "
                  f"pose {rec['detection_rate']:.0%}{flag}")
        except Exception as exc:  # noqa: BLE001 - keep going
            failures += 1
            rec = {"clip": str(clip.relative_to(a.folder)), "error": repr(exc)}
            print(f"[{k}/{len(clips)}] {rec['clip']}: FAILED {exc!r}", file=sys.stderr)
        manifest.append(rec)

    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    unlabelled = sum(1 for r in manifest if r.get("labelled") is False)
    print(f"Done: {len(clips) - failures} ok, {failures} failed, {unlabelled} without labels. "
          f"Manifest: {a.out / 'manifest.json'}")
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
