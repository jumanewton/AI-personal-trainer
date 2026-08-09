"""
Canonical MediaPipe Pose landmark indices and helper utilities.
All other modules import from here so index values live in one place.
"""
from __future__ import annotations
from typing import NamedTuple


class _LM:
    """Landmark index constants matching MediaPipe's 33-point pose model."""
    # Face
    NOSE          = 0
    LEFT_EYE_INNER  = 1;  LEFT_EYE  = 2;  LEFT_EYE_OUTER  = 3
    RIGHT_EYE_INNER = 4;  RIGHT_EYE = 5;  RIGHT_EYE_OUTER = 6
    LEFT_EAR      = 7;    RIGHT_EAR = 8
    MOUTH_LEFT    = 9;    MOUTH_RIGHT = 10
    # Upper body
    LEFT_SHOULDER  = 11;  RIGHT_SHOULDER = 12
    LEFT_ELBOW     = 13;  RIGHT_ELBOW    = 14
    LEFT_WRIST     = 15;  RIGHT_WRIST    = 16
    LEFT_PINKY     = 17;  RIGHT_PINKY    = 18
    LEFT_INDEX     = 19;  RIGHT_INDEX    = 20
    LEFT_THUMB     = 21;  RIGHT_THUMB    = 22
    # Lower body
    LEFT_HIP       = 23;  RIGHT_HIP   = 24
    LEFT_KNEE      = 25;  RIGHT_KNEE  = 26
    LEFT_ANKLE     = 27;  RIGHT_ANKLE = 28
    LEFT_HEEL      = 29;  RIGHT_HEEL  = 30
    LEFT_FOOT_INDEX = 31; RIGHT_FOOT_INDEX = 32

    TOTAL = 33

    # Grouped for convenience
    LOWER_BODY = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
    UPPER_BODY = [11, 12, 13, 14, 15, 16]
    CORE       = [11, 12, 23, 24]


LM = _LM()


# MediaPipe skeleton connections used by both the OpenCV overlay and the web canvas.
POSE_CONNECTIONS: list[tuple[int, int]] = [
    # Face
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    # Left arm
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    # Right arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Left leg
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    # Right leg
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]


class Point3D(NamedTuple):
    x: float
    y: float
    z: float


def world_lm_to_list(world_landmarks) -> list[Point3D]:
    """Convert MediaPipe world NormalizedLandmarkList to list of Point3D."""
    return [Point3D(lm.x, lm.y, lm.z) for lm in world_landmarks]


def image_lm_to_flat(image_landmarks) -> list[float]:
    """
    Convert MediaPipe image NormalizedLandmarkList to a flat list of
    [x0, y0, x1, y1, …, x32, y32] in normalised [0, 1] coordinates.
    """
    result = []
    for lm in image_landmarks:
        result.extend([lm.x, lm.y])
    return result


def mid(a: Point3D, b: Point3D) -> Point3D:
    """Midpoint of two 3-D points."""
    return Point3D((a.x + b.x) / 2, (a.y + b.y) / 2, (a.z + b.z) / 2)
