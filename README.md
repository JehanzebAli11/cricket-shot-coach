# cricket-shot-coach

**A phone-video batting coach for club and academy cricketers.** Film a few
deliveries in the nets, and the app shows your skeleton, names the shot, and
points out one or two things to work on, like "front knee straight at
contact" or "head falling outside the line".

This repo is the **week-1 prototype**: pose extraction, normalisation and
joint-angle measurements, plus the filming protocol and consent forms we need
before collecting any data.

> Status: prototype. Pose tooling and its tests are done. No real footage has
> been collected yet.

## What's here

```
pose/
  landmarks.py   33-point MediaPipe Pose topology, left/right swap table (no deps)
  overlay.py     draw the skeleton on a video -> MP4
  extract.py     per-frame keypoints -> .npz / .json
  normalise.py   centre on hips, scale by torso, mirror left-handers, resample (numpy)
  angles.py      front knee bend, head-over-front-knee, trunk lean, shoulder tilt, ... (numpy)
scripts/
  batch_process.py    a folder of clips -> keypoints, overlays, normalised arrays, metrics, manifest
  download_models.sh  fetch the MediaPipe .task models
docs/
  filming_protocol.md      how to film (two tripods, 60 fps, clip window, labels, held-out academy)
  consent_form_en.md       consent form, English
  consent_form_ur.md       consent form, Urdu (needs native-speaker review)
tests/                     pytest, no model or footage needed
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt && pip install -e .
pytest
```

### Models

Pose detection uses **MediaPipe Pose Landmarker** (Tasks API, VIDEO mode).
Download a model bundle from Google's MediaPipe model storage:

```bash
scripts/download_models.sh            # lite + full into models/
scripts/download_models.sh heavy      # optional, slowest and most accurate
```

That script fetches, for example:

```
https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
```

These follow the URL pattern on Google's [Pose Landmarker
page](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker).
Check that page if a download fails. Use `lite` for quick checks and on
phones, and `full` for the dataset. Model files are git-ignored.

## Usage

```bash
# Skeleton overlay (also saves keypoints if --npz is given)
python -m pose.overlay clip.mp4 --model models/pose_landmarker_full.task \
    --out out/clip_overlay.mp4 --npz out/clip.npz

# Keypoints only
python -m pose.extract clip.mp4 --model models/pose_landmarker_full.task \
    --out out/clip.npz --json out/clip.json

# A whole folder (labels.csv per docs/filming_protocol.md)
python scripts/batch_process.py clips/ --model models/pose_landmarker_full.task \
    --out out/ --labels clips/labels.csv
```

```python
from pose.extract import load_npz
from pose.normalise import normalise_sequence
from pose import angles

seq = load_npz("out/clip.npz")
x = normalise_sequence(seq.pixels(), left_handed=False, n_frames=64)   # (64, 33, 2)
angles.front_knee_bend(x)        # degrees per frame, 0 = straight
angles.head_over_front_knee(x)   # torso lengths, 0 = head over front knee
```

Notes:

* Frames with no pose come out as **NaN**, never zeros. The overlay writes
  them unchanged with a "no pose detected" label, so the video keeps its
  length. `normalise_sequence` fills gaps up to 10 frames and leaves longer
  ones as NaN.
* Angles need an isotropic frame. Use `seq.pixels()` (or world landmarks),
  not MediaPipe's 0–1 image coordinates.
* Left-handers are mirrored into right-handers, so "front" is always the
  left side after normalising.
* The overlay MP4 uses H.264 if OpenCV has it, otherwise `mp4v`. For browser
  playback, re-encode with `ffmpeg -i in.mp4 -c:v libx264 -pix_fmt yuv420p out.mp4`.
* `num_poses=1`: if the keeper or umpire is more prominent than the batter,
  MediaPipe may track the wrong person. Frame tightly (see protocol).

## Roadmap

| When | What |
|---|---|
| **Week 1** (this repo) | Pose overlay, keypoint extraction, normalisation, angle helpers, filming protocol, consent forms |
| Weeks 2–4 | Film at 2–3 academies with consent. Label shots. Hold one academy out entirely. |
| Weeks 4–6 | **Shot classifier on keypoints**: a small temporal model (1D-CNN/GRU) on normalised 64-frame sequences. Accuracy measured on the held-out academy. |
| Weeks 5–8 | **Form rules written with a coach**: turn the angle metrics into a handful of clear checks per shot, with thresholds the coach agrees, tested against coach ratings. |
| Weeks 8–11 | **Android app**: on-device MediaPipe, record, then overlay plus 1–2 tips. Works offline. Urdu and English UI. |
| **13 December** | **Launch**: public beta with partner academies. |

## Ground rules

1. **Consent first.** Nobody is filmed without a signed form. Being filmed
   and being shown publicly are **separate** choices. Withdrawal is honoured
   at any time.
2. **Minors.** Under-18s need a parent or guardian's signature **and** their
   own agreement. Faces of minors are blurred in anything public by default.
   Filming happens only at supervised sessions.
3. **No broadcast footage.** No TV, streaming, YouTube or social media clips
   of anyone, professional or not. Only footage we filmed ourselves with
   consent.
4. **Data minimisation.** Pseudonymous IDs in all shared files. Raw video
   stays on encrypted storage, and keypoints are what we share and train on
   wherever possible. Footage, keypoints and models are never committed to
   this repo (`.gitignore` enforces the common file types).
5. **Coaches decide what "good" looks like.** The metrics here are
   measurements, not judgements. Thresholds and advice come from qualified
   coaches.

## License

Code: MIT © 2026 Jehanzeb Ali (see `LICENSE`). The MediaPipe models are
Google's and come under their own licence terms. Check them on the model
page before redistributing.
