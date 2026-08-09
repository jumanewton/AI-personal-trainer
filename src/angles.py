"""
3-D joint angle computation from MediaPipe world landmarks.

MediaPipe world landmarks use a right-handed coordinate system:
  - Origin: centre of the hips
  - Y axis: points UP  (positive y = above the hips)
  - X axis: points to the camera's RIGHT (so the person's left side is negative x)
  - Z axis: points TOWARD the camera
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from src.landmarks import LM, Point3D, mid


# ---------------------------------------------------------------------------
# Primitive
# ---------------------------------------------------------------------------

def compute_angle_3d(A: Point3D | Sequence[float],
                     B: Point3D | Sequence[float],
                     C: Point3D | Sequence[float]) -> float:
    """
    Return the angle (in degrees) at vertex B formed by the rays B→A and B→C.
    Safe against degenerate (zero-length) vectors: returns 0.0.
    """
    a = np.asarray(A, dtype=float)
    b = np.asarray(B, dtype=float)
    c = np.asarray(C, dtype=float)

    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-9:
        return 0.0
    cos_val = np.dot(ba, bc) / denom
    return float(math.degrees(math.acos(float(np.clip(cos_val, -1.0, 1.0)))))


def _trunk_lean_angle(lm: list[Point3D]) -> float:
    """
    Forward lean of the torso from vertical (degrees).
    0° = perfectly upright.  90° = horizontal (lying down).

    Uses the spine vector (mid-hip → mid-shoulder) projected onto the
    sagittal plane (y-z plane, ignoring lateral sway) and compared with
    the upward vertical (0, 1, 0).
    """
    hip_mid  = mid(lm[LM.LEFT_HIP],      lm[LM.RIGHT_HIP])
    shld_mid = mid(lm[LM.LEFT_SHOULDER],  lm[LM.RIGHT_SHOULDER])

    spine = np.array([shld_mid.x - hip_mid.x,
                      shld_mid.y - hip_mid.y,
                      shld_mid.z - hip_mid.z], dtype=float)
    norm = np.linalg.norm(spine)
    if norm < 1e-9:
        return 0.0
    # Angle between spine and straight-up vector (0, 1, 0)
    cos_val = spine[1] / norm
    return float(math.degrees(math.acos(float(np.clip(cos_val, -1.0, 1.0)))))


def _shoulder_abduction(lm: list[Point3D], side: str) -> float:
    """
    Shoulder abduction angle (arm raised from side, degrees).
    0° = arm hanging straight down.  90° = arm horizontal.
    Side: 'left' or 'right'.
    """
    if side == "left":
        hip, shoulder, elbow = lm[LM.LEFT_HIP], lm[LM.LEFT_SHOULDER], lm[LM.LEFT_ELBOW]
    else:
        hip, shoulder, elbow = lm[LM.RIGHT_HIP], lm[LM.RIGHT_SHOULDER], lm[LM.RIGHT_ELBOW]

    # Vector from shoulder down to hip (the "body" direction)
    down = np.array([hip.x - shoulder.x, hip.y - shoulder.y, hip.z - shoulder.z], dtype=float)
    # Vector from shoulder out to elbow (the "arm" direction)
    arm  = np.array([elbow.x - shoulder.x, elbow.y - shoulder.y, elbow.z - shoulder.z], dtype=float)
    dn = np.linalg.norm(down); an = np.linalg.norm(arm)
    if dn < 1e-9 or an < 1e-9:
        return 0.0
    cos_val = np.dot(down, arm) / (dn * an)
    # 0° when arm is along down (hanging), 90° when horizontal
    return float(math.degrees(math.acos(float(np.clip(cos_val, -1.0, 1.0)))))


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def compute_all_angles(lm: list[Point3D]) -> dict[str, float]:
    """
    Compute a named dictionary of biomechanical angles from world landmarks.

    Returns
    -------
    dict with keys:
        left_knee, right_knee       — flexion at knee  (straight=180°)
        left_hip,  right_hip        — flexion at hip   (straight=180°)
        left_elbow, right_elbow     — flexion at elbow (straight=180°)
        left_shoulder_abd,
        right_shoulder_abd          — shoulder abduction from hanging (0→90°+)
        trunk_lean                  — forward lean of torso from vertical (0=upright)
    """
    if len(lm) < LM.TOTAL:
        return {}

    return {
        # ── Knee flexion (thigh-knee-ankle) ──────────────────────────────
        "left_knee":  compute_angle_3d(lm[LM.LEFT_HIP],
                                       lm[LM.LEFT_KNEE],
                                       lm[LM.LEFT_ANKLE]),
        "right_knee": compute_angle_3d(lm[LM.RIGHT_HIP],
                                       lm[LM.RIGHT_KNEE],
                                       lm[LM.RIGHT_ANKLE]),

        # ── Hip flexion (shoulder-hip-knee) ──────────────────────────────
        "left_hip":   compute_angle_3d(lm[LM.LEFT_SHOULDER],
                                       lm[LM.LEFT_HIP],
                                       lm[LM.LEFT_KNEE]),
        "right_hip":  compute_angle_3d(lm[LM.RIGHT_SHOULDER],
                                       lm[LM.RIGHT_HIP],
                                       lm[LM.RIGHT_KNEE]),

        # ── Elbow flexion (shoulder-elbow-wrist) ─────────────────────────
        "left_elbow":  compute_angle_3d(lm[LM.LEFT_SHOULDER],
                                        lm[LM.LEFT_ELBOW],
                                        lm[LM.LEFT_WRIST]),
        "right_elbow": compute_angle_3d(lm[LM.RIGHT_SHOULDER],
                                        lm[LM.RIGHT_ELBOW],
                                        lm[LM.RIGHT_WRIST]),

        # ── Shoulder abduction ───────────────────────────────────────────
        "left_shoulder_abd":  _shoulder_abduction(lm, "left"),
        "right_shoulder_abd": _shoulder_abduction(lm, "right"),

        # ── Trunk lean ───────────────────────────────────────────────────
        "trunk_lean": _trunk_lean_angle(lm),
    }
