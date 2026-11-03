"""Make keypoint sequences comparable across players, cameras and clip lengths.

Pure numpy. All functions take an array ``kps`` of shape ``(T, 33, D)`` where
``D`` is 2 (x, y) or 3 (x, y, z), in an **isotropic** coordinate system:
pixels, or MediaPipe world landmarks. Do not feed MediaPipe's normalised image
coordinates straight in: x is divided by width and y by height, so distances
and angles come out distorted. Use :func:`pose.extract.to_pixels` first.

Missing detections are ``NaN`` rows and pass through every step as NaN, so
nothing is silently invented. :func:`interpolate_missing` fills gaps when you
want that.

Pipeline (:func:`normalise_sequence`):

1. Fill short gaps by linear interpolation (optional).
2. Centre: subtract the hip midpoint (per frame), so the batter is at the origin.
3. Scale: divide by torso length (shoulder midpoint to hip midpoint), using the
   clip median by default so the scale doesn't wobble frame to frame.
4. Mirror left-handers (negate x and swap left/right landmarks) so every clip
   looks like a right-hander.
5. Resample to a fixed number of frames.
"""

from __future__ import annotations

import numpy as np

from .landmarks import L_HIP, L_SHOULDER, LR_SWAP, R_HIP, R_SHOULDER


def _check(kps: np.ndarray) -> np.ndarray:
    kps = np.asarray(kps, dtype=float)
    if kps.ndim != 3 or kps.shape[2] not in (2, 3):
        raise ValueError(f"expected (T, J, 2|3) keypoints, got shape {kps.shape}")
    return kps


def hip_midpoint(kps: np.ndarray) -> np.ndarray:
    """(T, D) midpoint of left and right hip."""
    kps = _check(kps)
    return (kps[:, L_HIP] + kps[:, R_HIP]) / 2.0


def shoulder_midpoint(kps: np.ndarray) -> np.ndarray:
    kps = _check(kps)
    return (kps[:, L_SHOULDER] + kps[:, R_SHOULDER]) / 2.0


def torso_length(kps: np.ndarray) -> np.ndarray:
    """(T,) distance from hip midpoint to shoulder midpoint (NaN if missing)."""
    return np.linalg.norm(shoulder_midpoint(kps) - hip_midpoint(kps), axis=-1)


def centre_on_hips(kps: np.ndarray) -> np.ndarray:
    """Translate each frame so the hip midpoint is at the origin."""
    kps = _check(kps)
    return kps - hip_midpoint(kps)[:, None, :]


def scale_by_torso(kps: np.ndarray, per_frame: bool = False, eps: float = 1e-6) -> np.ndarray:
    """Divide coordinates by torso length.

    ``per_frame=False`` (default) uses the median over frames where the torso
    is visible, which is steadier. A clip with no usable torso comes back all
    NaN rather than divided by zero.
    """
    kps = _check(kps)
    t = torso_length(kps)
    if per_frame:
        scale = np.where(t > eps, t, np.nan)[:, None, None]
    else:
        valid = t[np.isfinite(t) & (t > eps)]
        scale = np.median(valid) if valid.size else np.nan
    return kps / scale


def mirror(kps: np.ndarray, about: float = 0.0) -> np.ndarray:
    """Reflect horizontally (x -> 2*about - x) and swap left/right landmarks.

    After :func:`centre_on_hips` use ``about=0``. On raw pixel coordinates use
    ``about=width/2``. Applying it twice returns the input.
    """
    kps = _check(kps).copy()
    kps[..., 0] = 2.0 * about - kps[..., 0]
    return kps[:, LR_SWAP, :]


def interpolate_missing(kps: np.ndarray, max_gap: int | None = None) -> np.ndarray:
    """Linearly fill NaN frames from the nearest detected frames.

    Works per joint and coordinate. Leading and trailing gaps are held at the
    nearest valid value. Gaps longer than ``max_gap`` frames are left as NaN.
    A joint that is never seen stays NaN.
    """
    kps = _check(kps)
    out = kps.copy()
    T = kps.shape[0]
    t = np.arange(T)
    for j in range(kps.shape[1]):
        valid = np.all(np.isfinite(kps[:, j, :]), axis=-1)
        if valid.all() or not valid.any():
            continue
        for d in range(kps.shape[2]):
            out[:, j, d] = np.interp(t, t[valid], kps[valid, j, d])
        if max_gap is not None:
            # put NaN back over gaps that are too long
            missing = ~valid
            start = None
            for i in range(T + 1):
                m = missing[i] if i < T else False
                if m and start is None:
                    start = i
                elif not m and start is not None:
                    if i - start > max_gap:
                        out[start:i, j, :] = np.nan
                    start = None
    return out


def resample(kps: np.ndarray, n_frames: int) -> np.ndarray:
    """Linearly resample along time to exactly ``n_frames`` frames.

    The first and last frames are kept. NaNs spread to neighbouring samples,
    so fill gaps first if you need a dense output.
    """
    kps = _check(kps)
    if n_frames < 1:
        raise ValueError("n_frames must be >= 1")
    T = kps.shape[0]
    if T == 0:
        raise ValueError("empty sequence")
    if T == 1:
        return np.repeat(kps, n_frames, axis=0)
    pos = np.linspace(0.0, T - 1, n_frames)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, T - 1)
    w = (pos - lo)[:, None, None]
    # avoid 0 * NaN when the weight is exactly zero
    a, b = kps[lo], kps[hi]
    return np.where(w == 0, a, np.where(w == 1, b, (1 - w) * a + w * b))


def normalise_sequence(
    kps: np.ndarray,
    left_handed: bool = False,
    n_frames: int | None = 64,
    fill_gaps: bool = True,
    max_gap: int | None = 10,
    per_frame_scale: bool = False,
) -> np.ndarray:
    """Full pipeline: fill gaps, centre, scale, mirror (if left-handed), resample."""
    x = _check(kps)
    if fill_gaps:
        x = interpolate_missing(x, max_gap=max_gap)
    x = centre_on_hips(x)
    x = scale_by_torso(x, per_frame=per_frame_scale)
    if left_handed:
        x = mirror(x, about=0.0)
    if n_frames is not None:
        x = resample(x, n_frames)
    return x


__all__ = [
    "hip_midpoint", "shoulder_midpoint", "torso_length", "centre_on_hips",
    "scale_by_torso", "mirror", "interpolate_missing", "resample", "normalise_sequence",
]
