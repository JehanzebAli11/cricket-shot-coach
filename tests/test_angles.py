import numpy as np
import pytest

from pose import angles as A
from pose import landmarks as L
from pose import normalise as N


def test_joint_angle_basic():
    assert A.joint_angle([1, 0], [0, 0], [0, 1]) == pytest.approx(90)
    assert A.joint_angle([1, 0], [0, 0], [-1, 0]) == pytest.approx(180)
    assert A.joint_angle([1, 0], [0, 0], [1, 0]) == pytest.approx(0)
    assert A.joint_angle([1, 0, 0], [0, 0, 0], [1, 1, 0]) == pytest.approx(45)


def test_joint_angle_degenerate_and_nan():
    assert np.isnan(A.joint_angle([0, 0], [0, 0], [1, 0]))
    assert np.isnan(A.joint_angle([np.nan, 0], [0, 0], [1, 0]))


def test_joint_angle_vectorised():
    a = np.array([[1, 0], [1, 0]])
    b = np.zeros((2, 2))
    c = np.array([[0, 1], [-1, 0]])
    np.testing.assert_allclose(A.joint_angle(a, b, c), [90, 180])


def test_angle_from_vertical_and_horizontal():
    assert A.angle_from_vertical([0, -1], [0, 0]) == pytest.approx(0)  # straight up (y down)
    assert A.angle_from_vertical([1, -1], [0, 0]) == pytest.approx(45)
    assert A.angle_from_vertical([-1, -1], [0, 0]) == pytest.approx(-45)
    assert A.angle_from_horizontal([0, 0], [1, 0]) == pytest.approx(0)
    assert A.angle_from_horizontal([0, 0], [1, -1]) == pytest.approx(45)  # higher in image


def test_straight_leg_has_zero_bend(skeleton):
    assert A.knee_angle(skeleton, "left") == pytest.approx(180)
    assert A.front_knee_bend(skeleton) == pytest.approx(0)


def test_bent_front_knee(skeleton):
    k = skeleton.copy()
    # move the knee forward so hip-knee-ankle makes a right angle
    hip, ankle = k[L.L_HIP], k[L.L_ANKLE]
    mid = (hip + ankle) / 2
    half = np.linalg.norm(ankle - hip) / 2
    k[L.L_KNEE] = mid + np.array([-half, 0])
    assert A.knee_angle(k, "left") == pytest.approx(90)
    assert A.front_knee_bend(k, "left") == pytest.approx(90)
    # back knee untouched
    assert A.knee_angle(k, "right") == pytest.approx(180)


def test_head_over_front_knee(skeleton):
    k = skeleton.copy()
    # nose at x=0, left knee at x=-0.2 (scale 100) -> +20 px, torso 100 -> 0.2
    assert A.head_over_front_knee(k) == pytest.approx(0.2)
    assert A.head_over_front_knee(k, normalise=False) == pytest.approx(20)
    k[L.NOSE, 0] = k[L.L_KNEE, 0]
    assert A.head_over_front_knee(k) == pytest.approx(0)


def test_head_over_knee_scale_invariant():
    from conftest import make_skeleton

    small = make_skeleton(scale=50)
    big = make_skeleton(scale=300, origin=(10, 900))
    assert A.head_over_front_knee(small) == pytest.approx(A.head_over_front_knee(big))


def test_trunk_lean_and_shoulder_tilt(skeleton):
    assert A.trunk_lean(skeleton) == pytest.approx(0)
    assert A.shoulder_tilt(skeleton) == pytest.approx(0)  # level
    k = skeleton.copy()
    k[L.L_SHOULDER, 1] -= 60  # raise front shoulder by the shoulder width
    assert A.shoulder_tilt(k) == pytest.approx(45)
    # same pose facing the other way gives the same tilt
    flipped = N.mirror(k[None], about=320.0)[0]
    assert A.shoulder_tilt(flipped, front="right") == pytest.approx(45)
    k[L.L_SHOULDER, 1] += 120  # now front shoulder lower
    assert A.shoulder_tilt(k) == pytest.approx(-45)


def test_elbow_and_stance(skeleton):
    assert 0 < A.elbow_angle(skeleton, "left") < 180
    assert A.stance_width(skeleton) == pytest.approx(0.4)
    assert A.stance_width(skeleton, normalise=False) == pytest.approx(40)


def test_front_elbow_height(skeleton):
    assert A.front_elbow_height(skeleton) == pytest.approx(-0.5)  # elbow below shoulder
    k = skeleton.copy()
    k[L.L_ELBOW, 1] = k[L.L_SHOULDER, 1] - 10
    assert A.front_elbow_height(k) == pytest.approx(0.1)


def test_front_side_logic():
    assert A.front_side(left_handed=False) == "left"
    assert A.front_side(left_handed=True, mirrored=True) == "left"
    assert A.front_side(left_handed=True, mirrored=False) == "right"
    assert A.back_side(left_handed=True, mirrored=False) == "left"


def test_metrics_agree_for_mirrored_left_hander(sequence):
    rh = sequence.copy()
    rh[:, L.L_KNEE, 0] -= 25
    lh = N.mirror(rh, about=320.0)
    # unmirrored left-hander: front is the right side
    a = A.front_knee_bend(rh, A.front_side(False))
    b = A.front_knee_bend(lh, A.front_side(True, mirrored=False))
    np.testing.assert_allclose(a, b)
    # after normalising, front is "left" for both
    na = N.normalise_sequence(rh, n_frames=None)
    nb = N.normalise_sequence(lh, left_handed=True, n_frames=None)
    np.testing.assert_allclose(A.front_knee_bend(na), A.front_knee_bend(nb))
    np.testing.assert_allclose(A.head_over_front_knee(na), A.head_over_front_knee(nb))


def test_sequence_and_nan(sequence):
    seq = sequence.copy()
    seq[2] = np.nan
    bend = A.front_knee_bend(seq)
    assert bend.shape == (10,)
    assert np.isnan(bend[2]) and np.isfinite(bend[[0, 1, 3]]).all()


def test_summary_keys(sequence):
    s = A.summary(sequence)
    assert set(s) >= {"front_knee_bend", "head_over_front_knee", "trunk_lean", "shoulder_tilt"}
    assert all(v.shape == (10,) for v in s.values())


def test_bad_side(skeleton):
    with pytest.raises(ValueError):
        A.knee_angle(skeleton, "middle")
