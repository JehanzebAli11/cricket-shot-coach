import numpy as np
import pytest

from pose import landmarks as L
from pose import normalise as N


def test_hip_midpoint_and_torso(sequence):
    hm = N.hip_midpoint(sequence)
    assert hm.shape == (10, 2)
    np.testing.assert_allclose(hm[0], [300, 240])
    np.testing.assert_allclose(N.torso_length(sequence), 100 + 5 * np.arange(10))


def test_centre_on_hips(sequence):
    c = N.centre_on_hips(sequence)
    np.testing.assert_allclose(N.hip_midpoint(c), 0, atol=1e-9)
    # relative geometry preserved
    np.testing.assert_allclose(c[3, L.L_KNEE] - c[3, L.L_HIP], sequence[3, L.L_KNEE] - sequence[3, L.L_HIP])


def test_scale_by_torso_per_frame_gives_unit_torso(sequence):
    s = N.scale_by_torso(N.centre_on_hips(sequence), per_frame=True)
    np.testing.assert_allclose(N.torso_length(s), 1.0)


def test_scale_by_torso_median(sequence):
    s = N.scale_by_torso(sequence)
    med = np.median(100 + 5 * np.arange(10))
    np.testing.assert_allclose(s, sequence / med)


def test_scale_without_torso_is_nan():
    k = np.full((3, 33, 2), np.nan)
    assert np.isnan(N.scale_by_torso(k)).all()


def test_mirror_swaps_sides_and_is_involution(skeleton):
    k = N.centre_on_hips(skeleton[None])
    m = N.mirror(k)
    # left knee after mirroring = reflected right knee before
    np.testing.assert_allclose(m[0, L.L_KNEE], [-k[0, L.R_KNEE, 0], k[0, L.R_KNEE, 1]])
    np.testing.assert_allclose(N.mirror(m), k)
    # symmetric skeleton centred on hips maps to itself
    np.testing.assert_allclose(m[0, L.L_ANKLE], k[0, L.L_ANKLE], atol=1e-9)


def test_mirror_about_pixel_centre(skeleton):
    k = skeleton[None]
    m = N.mirror(k, about=320.0)
    assert m[0, L.NOSE, 0] == pytest.approx(640 - k[0, L.NOSE, 0])


def test_left_hander_normalises_to_right_hander(sequence):
    """Mirror a right-hander's clip in the image to fake a left-hander;
    normalising with left_handed=True should give back the right-hander."""
    rh = sequence.copy()
    rh[:, L.L_KNEE, 0] -= 30  # make it asymmetric: front knee pushed forward
    lh = N.mirror(rh, about=320.0)
    a = N.normalise_sequence(rh, left_handed=False, n_frames=None)
    b = N.normalise_sequence(lh, left_handed=True, n_frames=None)
    np.testing.assert_allclose(a, b, atol=1e-9)


def test_resample_lengths_and_endpoints(sequence):
    r = N.resample(sequence, 64)
    assert r.shape == (64, 33, 2)
    np.testing.assert_allclose(r[0], sequence[0])
    np.testing.assert_allclose(r[-1], sequence[-1])
    r2 = N.resample(sequence, 4)
    assert r2.shape == (4, 33, 2)


def test_resample_linear_midpoint():
    k = np.zeros((2, 33, 2))
    k[1] = 10.0
    r = N.resample(k, 3)
    np.testing.assert_allclose(r[1], 5.0)


def test_resample_single_frame_and_errors(skeleton):
    assert N.resample(skeleton[None], 5).shape == (5, 33, 2)
    with pytest.raises(ValueError):
        N.resample(skeleton[None], 0)
    with pytest.raises(ValueError):
        N.resample(np.zeros((0, 33, 2)), 5)


def test_resample_keeps_nan_out_of_exact_frames():
    k = np.zeros((3, 33, 2))
    k[2] = np.nan
    r = N.resample(k, 3)
    assert np.isfinite(r[:2]).all() and np.isnan(r[2]).all()


def test_interpolate_missing_fills_and_respects_max_gap():
    k = np.zeros((8, 33, 2))
    k[:, :, 0] = np.arange(8)[:, None]
    k[2:4] = np.nan  # 2-frame gap
    filled = N.interpolate_missing(k)
    np.testing.assert_allclose(filled[2:4, 0, 0], [2, 3])
    still = N.interpolate_missing(k, max_gap=1)
    assert np.isnan(still[2:4]).all()
    assert np.isfinite(still[[0, 1, 4, 5]]).all()


def test_interpolate_holds_edges_and_never_seen_joint():
    k = np.ones((4, 33, 3))
    k[0] = np.nan
    k[:, 5] = np.nan
    f = N.interpolate_missing(k)
    np.testing.assert_allclose(f[0, 0], 1.0)
    assert np.isnan(f[:, 5]).all()


def test_normalise_sequence_end_to_end(sequence):
    seq = sequence.copy()
    seq[4] = np.nan  # a dropped detection
    out = N.normalise_sequence(seq, n_frames=32)
    assert out.shape == (32, 33, 2)
    assert np.isfinite(out).all()
    np.testing.assert_allclose(N.hip_midpoint(out), 0, atol=1e-9)
    # camera zoom removed: torso length ~1 throughout (median scaling, so within zoom range)
    t = N.torso_length(out)
    assert 0.8 < t.min() and t.max() < 1.2


def test_3d_supported():
    k = np.random.default_rng(0).normal(size=(5, 33, 3))
    out = N.normalise_sequence(k, n_frames=7)
    assert out.shape == (7, 33, 3)


def test_bad_shape():
    with pytest.raises(ValueError):
        N.centre_on_hips(np.zeros((33, 2)))
