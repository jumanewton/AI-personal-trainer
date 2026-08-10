"""
Exercise finite-state machines for rep counting and real-time form analysis.

Each FSM class exposes:
    .update(angles, world_lm) -> ExerciseState
    .reset()

The ExerciseState is a snapshot of the current exercise status for a single
frame: phase name, rep count, form score 0-100, and a list of feedback strings.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from src.landmarks import LM, Point3D


# ─────────────────────────────────────────────────────────────────────────────
# Shared data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExerciseState:
    exercise: str      = "none"
    phase: str         = "idle"
    rep: int           = 0
    form_score: float  = 100.0          # 0–100
    feedback: list[str] = field(default_factory=list)
    bad_joints: list[int] = field(default_factory=list)  # landmark indices to highlight red


class ExerciseFSM(Protocol):
    name: str
    def update(self, angles: dict[str, float], wld: list[Point3D]) -> ExerciseState: ...
    def reset(self) -> None: ...


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _form_score(issues: list[str]) -> float:
    return max(0.0, 100.0 - len(issues) * 18.0)


def _dedup(lst: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in lst:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 1. SQUAT
# ─────────────────────────────────────────────────────────────────────────────

class SquatFSM:
    """
    Squat rep counter + form checker.

    Camera position: front-facing (person faces camera).

    Key angles
    ----------
    left_knee / right_knee  : 180° = straight, 90° = parallel, < 90° = deep
    trunk_lean              : 0° = upright, > 40° = excessive forward lean

    Form checks
    -----------
    ✓ Depth           : knee angle must reach < BOTTOM_ANGLE
    ✗ Knee valgus     : knee x-coord moves inside ankle x-coord
    ✗ Excessive lean  : trunk_lean > 40°
    ✗ Asymmetry       : |left_knee - right_knee| > 20°
    """
    name = "Squat"

    # Angle thresholds (degrees)
    STAND_ANGLE   = 160.0
    ENTER_DESCENT = 148.0
    BOTTOM_ANGLE  = 95.0
    EXIT_BOTTOM   = 105.0

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._phase        = "waiting"
        self._rep          = 0
        self._min_knee     = 180.0
        self._rep_issues: list[str] = []
        self._feedback: list[str]   = []

    # ── Internal helpers ──────────────────────────────────────────────────

    def _avg_knee(self, angles: dict) -> float:
        lk = angles.get("left_knee",  180.0)
        rk = angles.get("right_knee", 180.0)
        return (lk + rk) / 2.0

    def _check_form(self, angles: dict, wld: list[Point3D]) -> tuple[list[str], list[int]]:
        issues: list[str]  = []
        bad_joints: list[int] = []

        if len(wld) < LM.TOTAL:
            return issues, bad_joints

        lk_x  = wld[LM.LEFT_KNEE].x;   rk_x  = wld[LM.RIGHT_KNEE].x
        la_x  = wld[LM.LEFT_ANKLE].x;  ra_x  = wld[LM.RIGHT_ANKLE].x

        # Knee valgus: left knee should not drift right of left ankle
        if lk_x > la_x + 0.04:
            issues.append("Left knee caving in")
            bad_joints += [LM.LEFT_KNEE, LM.LEFT_ANKLE]
        if rk_x < ra_x - 0.04:
            issues.append("Right knee caving in")
            bad_joints += [LM.RIGHT_KNEE, LM.RIGHT_ANKLE]

        # Trunk lean
        if angles.get("trunk_lean", 0.0) > 40.0:
            issues.append("Chest falling forward")
            bad_joints += [LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER, LM.LEFT_HIP, LM.RIGHT_HIP]

        # Asymmetric loading
        lk = angles.get("left_knee",  180.0)
        rk = angles.get("right_knee", 180.0)
        if abs(lk - rk) > 22.0:
            issues.append("Uneven weight distribution")

        return issues, bad_joints

    # ── Public API ────────────────────────────────────────────────────────

    def update(self, angles: dict[str, float], wld: list[Point3D]) -> ExerciseState:
        avg = self._avg_knee(angles)
        live_issues, bad_joints = self._check_form(angles, wld)

        # ── Phase transitions ─────────────────────────────────────────────
        if self._phase == "waiting":
            if avg > self.STAND_ANGLE:
                self._phase = "standing"

        elif self._phase == "standing":
            if avg < self.ENTER_DESCENT:
                self._phase = "descending"
                self._rep_issues = []
                self._min_knee   = avg

        elif self._phase == "descending":
            self._min_knee = min(self._min_knee, avg)
            for i in live_issues:
                if i not in self._rep_issues:
                    self._rep_issues.append(i)

            if avg < self.BOTTOM_ANGLE:
                self._phase = "bottom"
            elif avg > self.STAND_ANGLE:
                # Aborted descent — no rep counted, give depth hint
                self._feedback = ["Go deeper! Aim for parallel (90°)"]
                self._phase = "standing"
                self._rep_issues = []

        elif self._phase == "bottom":
            for i in live_issues:
                if i not in self._rep_issues:
                    self._rep_issues.append(i)
            if avg > self.EXIT_BOTTOM:
                self._phase = "ascending"

        elif self._phase == "ascending":
            if avg > self.STAND_ANGLE:
                self._rep += 1
                self._phase = "standing"
                if not self._rep_issues:
                    self._feedback = ["[OK] Perfect rep!"]
                else:
                    self._feedback = self._rep_issues[-3:]
                self._rep_issues = []
                self._min_knee   = 180.0

        # ── Score = live check during active phases ────────────────────────
        if self._phase in ("descending", "bottom", "ascending"):
            score = _form_score(live_issues)
        else:
            score = 100.0

        feedback = _dedup(self._feedback + live_issues)[:4]
        return ExerciseState(self.name, self._phase, self._rep, score, feedback, bad_joints)


# ─────────────────────────────────────────────────────────────────────────────
# 2. PUSH-UP
# ─────────────────────────────────────────────────────────────────────────────

class PushupFSM:
    """
    Push-up rep counter + form checker.

    Best camera position: side-on (90° to the person).

    Key angles
    ----------
    left_elbow / right_elbow : 180° = arms straight, 90° = at the bottom
    left_hip  / right_hip    : should stay ~180° (no sag / pike)
    trunk_lean               : should stay near 0° (horizontal body)

    Form checks
    -----------
    ✓ Depth           : elbow angle < 90°
    ✗ Hip sag         : avg_hip_angle < 155°
    ✗ Pike            : avg_hip_angle > 200° (butt in the air)
    ✗ Flared elbows   : shoulder abduction > 60° at the bottom
    """
    name = "Push-up"

    STRAIGHT_ANGLE = 158.0
    BOTTOM_ANGLE   = 92.0
    ENTER_DESCENT  = 145.0
    EXIT_BOTTOM    = 100.0

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._phase    = "waiting"
        self._rep      = 0
        self._rep_issues: list[str] = []
        self._feedback: list[str]   = []

    def _avg_elbow(self, angles: dict) -> float:
        le = angles.get("left_elbow",  180.0)
        re = angles.get("right_elbow", 180.0)
        return (le + re) / 2.0

    def _check_form(self, angles: dict, wld: list[Point3D]) -> tuple[list[str], list[int]]:
        issues: list[str] = []
        bad: list[int]    = []

        avg_hip = (angles.get("left_hip", 180.0) + angles.get("right_hip", 180.0)) / 2.0
        if avg_hip < 155.0:
            issues.append("Hips sagging — engage your core")
            bad += [LM.LEFT_HIP, LM.RIGHT_HIP]
        elif avg_hip > 210.0:
            issues.append("Hips too high — straighten your body")
            bad += [LM.LEFT_HIP, LM.RIGHT_HIP]

        # Elbow flare: shoulder abduction when arms are bent
        avg_elbow = self._avg_elbow(angles)
        if avg_elbow < 120.0:
            la = angles.get("left_shoulder_abd",  0.0)
            ra = angles.get("right_shoulder_abd", 0.0)
            if (la + ra) / 2.0 > 60.0:
                issues.append("Elbows flaring — keep them close")
                bad += [LM.LEFT_ELBOW, LM.RIGHT_ELBOW]

        return issues, bad

    def update(self, angles: dict[str, float], wld: list[Point3D]) -> ExerciseState:
        avg = self._avg_elbow(angles)
        live_issues, bad_joints = self._check_form(angles, wld)

        if self._phase == "waiting":
            if avg > self.STRAIGHT_ANGLE:
                self._phase = "top"

        elif self._phase == "top":
            if avg < self.ENTER_DESCENT:
                self._phase = "descending"
                self._rep_issues = []

        elif self._phase == "descending":
            for i in live_issues:
                if i not in self._rep_issues: self._rep_issues.append(i)
            if avg < self.BOTTOM_ANGLE:
                self._phase = "bottom"
            elif avg > self.STRAIGHT_ANGLE:
                self._feedback = ["Touch your chest to the floor!"]
                self._phase = "top"
                self._rep_issues = []

        elif self._phase == "bottom":
            for i in live_issues:
                if i not in self._rep_issues: self._rep_issues.append(i)
            if avg > self.EXIT_BOTTOM:
                self._phase = "ascending"

        elif self._phase == "ascending":
            if avg > self.STRAIGHT_ANGLE:
                self._rep += 1
                self._phase = "top"
                self._feedback = ["[OK] Great rep!"] if not self._rep_issues else self._rep_issues[-3:]
                self._rep_issues = []

        score = _form_score(live_issues) if self._phase in ("descending", "bottom", "ascending") else 100.0
        feedback = _dedup(self._feedback + live_issues)[:4]
        return ExerciseState(self.name, self._phase, self._rep, score, feedback, bad_joints)


# ─────────────────────────────────────────────────────────────────────────────
# 3. LUNGE
# ─────────────────────────────────────────────────────────────────────────────

class LungeFSM:
    """
    Forward lunge rep counter + form checker.

    Camera position: front-facing or angled 45°.

    Detects whichever leg steps forward (asymmetric knee angle).

    Form checks
    -----------
    ✓ Depth       : front knee < 100°
    ✗ Trunk lean  : > 20°
    ✗ Back knee   : back knee should not hit the floor (angle < 65°)
    """
    name = "Lunge"

    STAND_ANGLE   = 155.0
    BOTTOM_ANGLE  = 100.0
    ENTER_DESCENT = 140.0
    EXIT_BOTTOM   = 110.0

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._phase    = "waiting"
        self._rep      = 0
        self._rep_issues: list[str] = []
        self._feedback: list[str]   = []

    def _front_knee_angle(self, angles: dict) -> float:
        """Return the angle of the more-bent knee (the lead leg)."""
        lk = angles.get("left_knee",  180.0)
        rk = angles.get("right_knee", 180.0)
        return min(lk, rk)

    def _check_form(self, angles: dict, wld: list[Point3D]) -> tuple[list[str], list[int]]:
        issues: list[str] = []
        bad: list[int]    = []

        if angles.get("trunk_lean", 0.0) > 20.0:
            issues.append("Keep your torso upright")
            bad += [LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER]

        lk = angles.get("left_knee",  180.0)
        rk = angles.get("right_knee", 180.0)
        back_knee = max(lk, rk)
        if back_knee < 65.0:
            issues.append("Back knee too low — lift it slightly")
            if lk > rk:
                bad.append(LM.LEFT_KNEE)
            else:
                bad.append(LM.RIGHT_KNEE)

        return issues, bad

    def update(self, angles: dict[str, float], wld: list[Point3D]) -> ExerciseState:
        front = self._front_knee_angle(angles)
        live_issues, bad_joints = self._check_form(angles, wld)

        if self._phase == "waiting":
            avg = (angles.get("left_knee", 180.0) + angles.get("right_knee", 180.0)) / 2.0
            if avg > self.STAND_ANGLE:
                self._phase = "standing"

        elif self._phase == "standing":
            if front < self.ENTER_DESCENT:
                self._phase = "descending"
                self._rep_issues = []

        elif self._phase == "descending":
            for i in live_issues:
                if i not in self._rep_issues: self._rep_issues.append(i)
            if front < self.BOTTOM_ANGLE:
                self._phase = "bottom"
            elif front > self.STAND_ANGLE:
                self._feedback = ["Lunge deeper!"]
                self._phase = "standing"
                self._rep_issues = []

        elif self._phase == "bottom":
            for i in live_issues:
                if i not in self._rep_issues: self._rep_issues.append(i)
            if front > self.EXIT_BOTTOM:
                self._phase = "ascending"

        elif self._phase == "ascending":
            avg = (angles.get("left_knee", 180.0) + angles.get("right_knee", 180.0)) / 2.0
            if avg > self.STAND_ANGLE:
                self._rep += 1
                self._phase = "standing"
                self._feedback = ["[OK] Good lunge!"] if not self._rep_issues else self._rep_issues[-3:]
                self._rep_issues = []

        score = _form_score(live_issues) if self._phase in ("descending", "bottom", "ascending") else 100.0
        feedback = _dedup(self._feedback + live_issues)[:4]
        return ExerciseState(self.name, self._phase, self._rep, score, feedback, bad_joints)


# ─────────────────────────────────────────────────────────────────────────────
# 4. JUMPING JACK
# ─────────────────────────────────────────────────────────────────────────────

class JumpingJackFSM:
    """
    Jumping jack rep counter.

    Tracks the average shoulder abduction angle.
    Arms must raise past 80° (roughly horizontal) to count as "open".

    Camera position: front-facing.
    """
    name = "Jumping Jack"

    OPEN_ANGLE   = 80.0
    CLOSED_ANGLE = 35.0

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._phase    = "closed"
        self._rep      = 0
        self._feedback: list[str] = []

    def update(self, angles: dict[str, float], wld: list[Point3D]) -> ExerciseState:
        avg_abd = (angles.get("left_shoulder_abd",  0.0) +
                   angles.get("right_shoulder_abd", 0.0)) / 2.0

        if self._phase == "closed" and avg_abd > self.OPEN_ANGLE:
            self._phase = "open"

        elif self._phase == "open" and avg_abd < self.CLOSED_ANGLE:
            self._phase = "closed"
            self._rep  += 1
            self._feedback = [f"Rep {self._rep} [OK]"]

        feedback = self._feedback[-2:]
        return ExerciseState(self.name, self._phase, self._rep, 100.0, feedback, [])


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

EXERCISE_MAP: dict[str, type] = {
    "squat":       SquatFSM,
    "pushup":      PushupFSM,
    "lunge":       LungeFSM,
    "jumpingjack": JumpingJackFSM,
}

EXERCISE_LABELS: dict[str, str] = {
    "squat":       "Squat",
    "pushup":      "Push-up",
    "lunge":       "Lunge",
    "jumpingjack": "Jumping Jack",
}


def make_exercise(key: str) -> ExerciseFSM:
    """Instantiate an exercise FSM by its key string."""
    cls = EXERCISE_MAP.get(key.lower())
    if cls is None:
        raise ValueError(f"Unknown exercise '{key}'. Choices: {list(EXERCISE_MAP)}")
    return cls()
