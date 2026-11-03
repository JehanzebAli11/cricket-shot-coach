"""MediaPipe Pose (BlazePose) 33-landmark topology, without importing mediapipe.

Keeping this in plain Python means the numpy code (normalise, angles) and its
tests do not need mediapipe or OpenCV installed.
"""

from __future__ import annotations

NUM_LANDMARKS = 33

NAMES = [
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]
assert len(NAMES) == NUM_LANDMARKS

IDX = {name: i for i, name in enumerate(NAMES)}

NOSE = IDX["nose"]
L_SHOULDER, R_SHOULDER = IDX["left_shoulder"], IDX["right_shoulder"]
L_ELBOW, R_ELBOW = IDX["left_elbow"], IDX["right_elbow"]
L_WRIST, R_WRIST = IDX["left_wrist"], IDX["right_wrist"]
L_HIP, R_HIP = IDX["left_hip"], IDX["right_hip"]
L_KNEE, R_KNEE = IDX["left_knee"], IDX["right_knee"]
L_ANKLE, R_ANKLE = IDX["left_ankle"], IDX["right_ankle"]

# Same edges as mediapipe's PoseLandmarksConnections.POSE_LANDMARKS (35 edges).
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
    (29, 31), (30, 32), (27, 31), (28, 32),
]

# Index permutation that swaps every left landmark with its right twin.
# Used when mirroring a left-hander so they look like a right-hander.
LR_SWAP = list(range(NUM_LANDMARKS))
for _i, _n in enumerate(NAMES):
    if _n.startswith("left_"):
        _j = IDX["right_" + _n[len("left_"):]]
        LR_SWAP[_i], LR_SWAP[_j] = _j, _i
    elif _n == "mouth_left":
        LR_SWAP[_i], LR_SWAP[IDX["mouth_right"]] = IDX["mouth_right"], _i
