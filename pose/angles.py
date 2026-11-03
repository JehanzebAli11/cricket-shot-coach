"""Joint-angle and form-metric helpers for batting.

Conventions
-----------
* Input ``kps`` is ``(T, 33, D)`` (or a single frame ``(33, D)``) in an
  isotropic frame, i.e. pixels or the output of
  :func:`pose.normalise.normalise_sequence`. **Image convention: x to the
  right, y down.**
* "Front" means the side nearer the bowler. For a right-handed batter that is
  the LEFT side of the body. After mirroring left-handers in
  ``normalise_sequence`` every batter is a right-hander, so ``front="left"``
  is the default. Use :func:`front_side` if you skipped mirroring.
* Every function is vectorised over frames and returns NaN where a needed
  landmark is missing.
* These are 2D measurements from one camera. They depend on viewpoint. Knee
  bend and head-over-knee are meant for the **side-on** camera. Shoulder tilt
  and stance width read best **front-on**. See docs/filming_protocol.md.

None of these come with "good" thresholds yet. Those will be agreed with a
coach (see README roadmap).
"""

from __future__ import annotations

import numpy as np

from . import landmarks as L
from .normalise import hip_midpoint, shoulder_midpoint, torso_length

Side = str  # "left" | "right"


def _as_seq(kps: np.ndarray) -> tuple[np.ndarray, bool]:
    kps = np.asarray(kps, dtype=float)
    if kps.ndim == 2:
        return kps[None], True
    if kps.ndim != 3:
        raise ValueError(f"expected (T, 33, D) or (33, D), got {kps.shape}")
    return kps, False


def _ret(x: np.ndarray, single: bool):
    return float(x[0]) if single else x


def front_side(left_handed: bool, mirrored: bool = True) -> Side:
    """Which body side faces the bowler.

    Right-hander: left side. Left-hander: right side, unless the sequence was
    mirrored by ``normalise_sequence(left_handed=True)``, which turns them into
    a right-hander.
    """
    if left_handed and not mirrored:
        return "right"
    return "left"


def back_side(left_handed: bool, mirrored: bool = True) -> Side:
    return "right" if front_side(left_handed, mirrored) == "left" else "left"


def _j(side: Side, part: str) -> int:
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")
    return L.IDX[f"{side}_{part}"]


# --------------------------------------------------------------------------- #
# Generic geometry
# --------------------------------------------------------------------------- #


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle ABC at vertex ``b`` in degrees, in [0, 180]. Works on (..., D) arrays.

    Returns NaN if either segment has zero length or any point is NaN.
    """
    a, b, c = (np.asarray(v, dtype=float) for v in (a, b, c))
    ba, bc = a - b, c - b
    nba = np.linalg.norm(ba, axis=-1)
    nbc = np.linalg.norm(bc, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.sum(ba * bc, axis=-1) / (nba * nbc)
    cos = np.where((nba > 0) & (nbc > 0), cos, np.nan)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def angle_from_vertical(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    """Signed angle (deg) of segment bottom->top from straight up, in an image frame.

    0 means upright. Positive means the top leans towards +x (right in the
    image), negative towards -x.
    """
    v = np.asarray(top, float) - np.asarray(bottom, float)
    # "up" in image coordinates is -y
    return np.degrees(np.arctan2(v[..., 0], -v[..., 1]))


def angle_from_horizontal(p_from: np.ndarray, p_to: np.ndarray) -> np.ndarray:
    """Signed angle (deg) of segment from->to relative to the +x axis, with y down.

    Positive means ``p_to`` is higher in the image than ``p_from``.
    """
    v = np.asarray(p_to, float) - np.asarray(p_from, float)
    return np.degrees(np.arctan2(-v[..., 1], v[..., 0]))


# --------------------------------------------------------------------------- #
# Batting metrics
# --------------------------------------------------------------------------- #


def knee_angle(kps: np.ndarray, side: Side = "left"):
    """Hip-knee-ankle angle in degrees (180 = straight leg)."""
    k, single = _as_seq(kps)
    out = joint_angle(k[:, _j(side, "hip")], k[:, _j(side, "knee")], k[:, _j(side, "ankle")])
    return _ret(out, single)


def front_knee_bend(kps: np.ndarray, front: Side = "left"):
    """Front-knee flexion in degrees: ``180 - knee_angle`` (0 = locked straight).

    Side-on camera. A front-foot drive usually shows a clear bend at contact.
    """
    k, single = _as_seq(kps)
    return _ret(180.0 - knee_angle(k, front), single)


def elbow_angle(kps: np.ndarray, side: Side = "left"):
    """Shoulder-elbow-wrist angle in degrees (180 = straight arm)."""
    k, single = _as_seq(kps)
    out = joint_angle(k[:, _j(side, "shoulder")], k[:, _j(side, "elbow")], k[:, _j(side, "wrist")])
    return _ret(out, single)


def head_over_front_knee(kps: np.ndarray, front: Side = "left", normalise: bool = True):
    """Horizontal offset of the head (nose) from the front knee.

    ``nose_x - front_knee_x``, divided by torso length when ``normalise``
    (unitless, so it compares across players and camera distances). Side-on
    camera. Close to 0 means the head is over the front knee, which coaches
    often want at contact on front-foot shots. The sign depends on which way
    the batter faces in the frame, so use ``abs()`` for "how far off".
    """
    k, single = _as_seq(kps)
    dx = k[:, L.NOSE, 0] - k[:, _j(front, "knee"), 0]
    if normalise:
        t = torso_length(k)
        dx = np.where(t > 0, dx / t, np.nan)
    return _ret(dx, single)


def trunk_lean(kps: np.ndarray):
    """Angle of the hip-mid -> shoulder-mid line from vertical (deg, signed).

    Side-on, this is forward/back lean. Front-on, it is lean to the off/leg side.
    """
    k, single = _as_seq(kps)
    return _ret(angle_from_vertical(shoulder_midpoint(k), hip_midpoint(k)), single)


def shoulder_tilt(kps: np.ndarray, front: Side = "left"):
    """Elevation of the front shoulder above the back shoulder, in degrees.

    In [-90, 90]: 0 = level, positive = front shoulder higher. It does not
    depend on which way the batter faces in the frame (the horizontal gap is
    taken as an absolute value). Front-on camera.
    """
    k, single = _as_seq(kps)
    back = "right" if front == "left" else "left"
    f, b = k[:, _j(front, "shoulder")], k[:, _j(back, "shoulder")]
    rise = b[:, 1] - f[:, 1]  # y down: positive when front is higher
    run = np.abs(f[:, 0] - b[:, 0])
    out = np.degrees(np.arctan2(rise, run))
    return _ret(out, single)


def stance_width(kps: np.ndarray, normalise: bool = True):
    """Distance between the ankles, divided by torso length when ``normalise``.

    Front-on camera for stance width at set-up; side-on for stride length.
    """
    k, single = _as_seq(kps)
    d = np.linalg.norm(k[:, L.L_ANKLE] - k[:, L.R_ANKLE], axis=-1)
    if normalise:
        t = torso_length(k)
        d = np.where(t > 0, d / t, np.nan)
    return _ret(d, single)


def front_elbow_height(kps: np.ndarray, front: Side = "left", normalise: bool = True):
    """How far the front elbow sits above the front shoulder (positive = above).

    The "high front elbow" cue in drives. Divided by torso length when ``normalise``.
    """
    k, single = _as_seq(kps)
    dy = k[:, _j(front, "shoulder"), 1] - k[:, _j(front, "elbow"), 1]  # y down
    if normalise:
        t = torso_length(k)
        dy = np.where(t > 0, dy / t, np.nan)
    return _ret(dy, single)


def summary(kps: np.ndarray, front: Side = "left") -> dict:
    """Per-frame metric arrays in one dict (handy for JSON / plotting)."""
    k, _ = _as_seq(kps)
    return {
        "front_knee_bend": front_knee_bend(k, front),
        "back_knee_angle": knee_angle(k, "right" if front == "left" else "left"),
        "front_elbow_angle": elbow_angle(k, front),
        "head_over_front_knee": head_over_front_knee(k, front),
        "trunk_lean": trunk_lean(k),
        "shoulder_tilt": shoulder_tilt(k, front),
        "stance_width": stance_width(k),
        "front_elbow_height": front_elbow_height(k, front),
    }


__all__ = [
    "front_side", "back_side", "joint_angle", "angle_from_vertical", "angle_from_horizontal",
    "knee_angle", "front_knee_bend", "elbow_angle", "head_over_front_knee", "trunk_lean",
    "shoulder_tilt", "stance_width", "front_elbow_height", "summary",
]
