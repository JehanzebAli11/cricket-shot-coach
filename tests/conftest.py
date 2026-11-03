import numpy as np
import pytest

from pose import landmarks as L


def make_skeleton(scale=100.0, origin=(320.0, 240.0)):
    """A simple upright right-handed batter in image coords (x right, y down).

    Torso length (shoulder-mid to hip-mid) = 1 * scale.
    Left side (front, faces bowler) is at smaller x.
    """
    p = np.full((L.NUM_LANDMARKS, 2), np.nan)
    pts = {
        "nose": (0.0, -1.4),
        "left_shoulder": (-0.3, -1.0), "right_shoulder": (0.3, -1.0),
        "left_elbow": (-0.4, -0.5), "right_elbow": (0.4, -0.5),
        "left_wrist": (-0.4, 0.0), "right_wrist": (0.4, 0.0),
        "left_hip": (-0.2, 0.0), "right_hip": (0.2, 0.0),
        "left_knee": (-0.2, 0.9), "right_knee": (0.2, 0.9),
        "left_ankle": (-0.2, 1.8), "right_ankle": (0.2, 1.8),
    }
    for name, (x, y) in pts.items():
        p[L.IDX[name]] = (origin[0] + x * scale, origin[1] + y * scale)
    # fill the remaining landmarks near their parents so nothing is NaN
    for i in range(L.NUM_LANDMARKS):
        if np.isnan(p[i, 0]):
            p[i] = p[L.NOSE] + (i % 3 - 1, i % 2)
    return p


@pytest.fixture
def skeleton():
    return make_skeleton()


@pytest.fixture
def sequence():
    """10 frames, batter drifting right and growing (camera zooming in)."""
    frames = [make_skeleton(scale=100 + 5 * t, origin=(300 + 4 * t, 240.0)) for t in range(10)]
    return np.stack(frames)
