from pose import landmarks as L


def test_names_and_indices():
    assert L.NUM_LANDMARKS == 33 == len(L.NAMES)
    assert L.IDX["left_shoulder"] == 11 and L.IDX["right_hip"] == 24
    assert L.IDX["right_foot_index"] == 32


def test_lr_swap_is_an_involution_and_swaps_pairs():
    s = L.LR_SWAP
    assert sorted(s) == list(range(33))
    assert all(s[s[i]] == i for i in range(33))
    assert s[L.NOSE] == L.NOSE
    assert s[L.L_KNEE] == L.R_KNEE and s[L.R_WRIST] == L.L_WRIST
    assert s[L.IDX["mouth_left"]] == L.IDX["mouth_right"]


def test_connections_valid():
    assert len(L.CONNECTIONS) == 35
    assert all(0 <= a < 33 and 0 <= b < 33 for a, b in L.CONNECTIONS)


def test_connections_match_mediapipe_if_installed():
    import pytest

    vision = pytest.importorskip("mediapipe.tasks.python.vision")
    mp_edges = {(c.start, c.end) for c in vision.PoseLandmarksConnections.POSE_LANDMARKS}
    assert mp_edges == set(L.CONNECTIONS)
