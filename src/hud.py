"""
OpenCV heads-up display (HUD) renderer for the AI Personal Trainer.

All drawing is done via cv2 with no external font dependencies.
A semi-transparent panel on the right side shows:
  • Exercise name + phase
  • Large rep counter
  • Colour-coded form score bar
  • Key angle readouts
  • Scrolling feedback messages
  • FPS counter
"""
from __future__ import annotations

import time
from collections import deque

import cv2
import numpy as np

from src.exercises import ExerciseState
from src.landmarks import LM, POSE_CONNECTIONS


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette  (OpenCV uses BGR)
# ─────────────────────────────────────────────────────────────────────────────
C_PANEL_BG      = (15,  12,  22)    # very dark navy
C_ACCENT        = (255, 200,  0)    # electric cyan (BGR)
C_TEXT_MAIN     = (235, 230, 245)   # near-white
C_TEXT_DIM      = (140, 130, 155)   # muted
C_PHASE         = (0,   200, 255)   # gold
C_GOOD          = (80,  220,  60)   # green  (BGR)
C_BAD           = (40,   60, 220)   # red    (BGR)
C_WARN          = (0,   160, 255)   # orange (BGR)
C_JOINT_DEFAULT = (255, 190,  50)   # cyan joints (BGR)
C_JOINT_BAD     = (30,   40, 200)   # red joints  (BGR)
C_BONE_DEFAULT  = (200, 140,  30)   # cyan bones
C_BONE_BAD      = (30,   40, 180)   # red bones

PANEL_W         = 230   # width of right-side HUD panel (px)
ALPHA           = 0.82  # panel opacity


# ─────────────────────────────────────────────────────────────────────────────
# Selection screen (shown before exercise is chosen)
# ─────────────────────────────────────────────────────────────────────────────

_SELECT_OPTIONS = [
    ("S", "Squat"),
    ("P", "Push-up"),
    ("L", "Lunge"),
    ("J", "Jumping Jack"),
]


def draw_selection_screen(frame: np.ndarray) -> np.ndarray:
    """Draw an exercise-selection prompt over the frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Dim the whole frame
    cv2.rectangle(overlay, (0, 0), (w, h), (10, 8, 18), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    # Title
    title = "SELECT EXERCISE"
    cv2.putText(frame, title,
                (w // 2 - 180, h // 2 - 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, C_ACCENT, 2, cv2.LINE_AA)

    # Options
    for i, (key, label) in enumerate(_SELECT_OPTIONS):
        y = h // 2 - 50 + i * 52
        cv2.rectangle(frame, (w // 2 - 180, y - 28), (w // 2 + 180, y + 16),
                      (40, 35, 60), -1)
        cv2.rectangle(frame, (w // 2 - 180, y - 28), (w // 2 + 180, y + 16),
                      C_ACCENT, 1)
        text = f"[{key}]  {label}"
        cv2.putText(frame, text,
                    (w // 2 - 160, y + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, C_TEXT_MAIN, 1, cv2.LINE_AA)

    cv2.putText(frame, "Press Q to quit",
                (w // 2 - 100, h // 2 + 170),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_TEXT_DIM, 1, cv2.LINE_AA)
    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Feedback message buffer (auto-expires)
# ─────────────────────────────────────────────────────────────────────────────

class _FeedbackBuffer:
    def __init__(self, ttl: float = 3.0, max_msgs: int = 3) -> None:
        self._ttl  = ttl
        self._max  = max_msgs
        self._msgs: deque[tuple[str, float]] = deque()

    def push(self, messages: list[str]) -> None:
        now = time.monotonic()
        for msg in messages:
            # Deduplicate (reset TTL if same message already present)
            self._msgs = deque((m, t) for m, t in self._msgs if m != msg)
            self._msgs.append((msg, now + self._ttl))
            while len(self._msgs) > self._max:
                self._msgs.popleft()

    def active(self) -> list[str]:
        now = time.monotonic()
        self._msgs = deque((m, t) for m, t in self._msgs if t > now)
        return [m for m, _ in self._msgs]


# ─────────────────────────────────────────────────────────────────────────────
# HUD Renderer
# ─────────────────────────────────────────────────────────────────────────────

class HUDRenderer:
    """
    Stateful renderer — maintains the feedback buffer across frames.

    Call ``render()`` once per frame with the latest state.
    """

    def __init__(self) -> None:
        self._fb   = _FeedbackBuffer()
        self._prev_rep = 0

    def render(
        self,
        frame:      np.ndarray,
        state:      ExerciseState,
        angles:     dict[str, float],
        fps:        float,
        image_lm:   list | None = None,   # normalized image landmarks
    ) -> np.ndarray:
        """
        Draw skeleton overlay + HUD panel.  Returns the annotated frame.
        """
        h, w = frame.shape[:2]

        # Update feedback buffer whenever new feedback arrives
        if state.feedback:
            self._fb.push(state.feedback)
        if state.rep != self._prev_rep and state.rep > 0:
            self._prev_rep = state.rep

        # 1. Draw skeleton
        if image_lm is not None:
            frame = _draw_skeleton(frame, image_lm, state.bad_joints, w, h)

        # 2. Draw HUD panel
        frame = _draw_panel(frame, state, angles, fps,
                            self._fb.active(), w, h)
        return frame


# ─────────────────────────────────────────────────────────────────────────────
# Skeleton drawing
# ─────────────────────────────────────────────────────────────────────────────

def _draw_skeleton(
    frame:      np.ndarray,
    image_lm:   list,           # normalised MediaPipe image landmarks
    bad_joints: list[int],
    w:          int,
    h:          int,
) -> np.ndarray:
    """Draw colour-coded skeleton on the frame."""
    bad_set = set(bad_joints)

    pts = [(int(lm.x * w), int(lm.y * h)) for lm in image_lm]

    # Connections
    for a, b in POSE_CONNECTIONS:
        if a >= len(pts) or b >= len(pts):
            continue
        color = C_BONE_BAD if (a in bad_set or b in bad_set) else C_BONE_DEFAULT
        cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)

    # Joints
    for idx, (x, y) in enumerate(pts):
        color = C_JOINT_BAD if idx in bad_set else C_JOINT_DEFAULT
        cv2.circle(frame, (x, y), 5, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 5, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Panel drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _put(frame, text, x, y, scale=0.55, color=None, thickness=1):
    if color is None:
        color = C_TEXT_MAIN
    cv2.putText(frame, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _form_color(score: float):
    """Interpolate between red and green based on form score."""
    t   = score / 100.0
    r   = int((1 - t) * 200 +  t * 60)
    g   = int((1 - t) * 40  +  t * 220)
    b   = int((1 - t) * 40  +  t * 80)
    return (b, g, r)      # OpenCV BGR


def _draw_panel(
    frame:    np.ndarray,
    state:    ExerciseState,
    angles:   dict[str, float],
    fps:      float,
    feedback: list[str],
    w:        int,
    h:        int,
) -> np.ndarray:
    px = w - PANEL_W   # left edge of panel

    # ── Semi-transparent background ──────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (px, 0), (w, h), C_PANEL_BG, -1)
    cv2.addWeighted(overlay, ALPHA, frame, 1 - ALPHA, 0, frame)

    # Thin accent border on left edge
    cv2.line(frame, (px, 0), (px, h), C_ACCENT, 1)

    cx  = px + 12   # content left margin
    cw  = PANEL_W - 20

    # ── Exercise name ────────────────────────────────────────────────────
    y = 32
    cv2.putText(frame, state.exercise.upper(),
                (cx, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_ACCENT, 2, cv2.LINE_AA)

    # ── Phase ────────────────────────────────────────────────────────────
    y += 28
    phase_display = state.phase.upper().replace("_", " ")
    _put(frame, phase_display, cx, y, 0.52, C_PHASE)

    # ── Divider ──────────────────────────────────────────────────────────
    y += 14
    cv2.line(frame, (cx, y), (w - 12, y), (60, 55, 80), 1)

    # ── Rep counter (large) ──────────────────────────────────────────────
    y += 48
    rep_str = str(state.rep)
    # Shadow
    cv2.putText(frame, rep_str,
                (cx + 2 + (cw - len(rep_str) * 32) // 2, y + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (30, 25, 40), 4, cv2.LINE_AA)
    # Main
    cv2.putText(frame, rep_str,
                (cx + (cw - len(rep_str) * 32) // 2, y),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, C_TEXT_MAIN, 3, cv2.LINE_AA)
    y += 16
    _put(frame, "REPS", cx + (cw - 38) // 2, y, 0.48, C_TEXT_DIM)

    # ── Form score bar ───────────────────────────────────────────────────
    y += 30
    _put(frame, "FORM", cx, y, 0.48, C_TEXT_DIM)
    score_text = f"{state.form_score:.0f}%"
    _put(frame, score_text, w - 12 - len(score_text) * 8, y, 0.48, C_TEXT_DIM)

    y += 10
    bar_w = cw
    bar_h = 10
    cv2.rectangle(frame, (cx, y), (cx + bar_w, y + bar_h), (40, 35, 55), -1)
    fill_w = int(bar_w * state.form_score / 100.0)
    if fill_w > 0:
        fcolor = _form_color(state.form_score)
        cv2.rectangle(frame, (cx, y), (cx + fill_w, y + bar_h), fcolor, -1)
    cv2.rectangle(frame, (cx, y), (cx + bar_w, y + bar_h), (80, 70, 100), 1)

    # ── Divider ──────────────────────────────────────────────────────────
    y += 24
    cv2.line(frame, (cx, y), (w - 12, y), (60, 55, 80), 1)

    # ── Key angles ───────────────────────────────────────────────────────
    y += 18
    _put(frame, "ANGLES", cx, y, 0.45, C_TEXT_DIM)
    angle_items = [
        ("Knee L",   angles.get("left_knee",  0.0)),
        ("Knee R",   angles.get("right_knee", 0.0)),
        ("Hip L",    angles.get("left_hip",   0.0)),
        ("Trunk",    angles.get("trunk_lean", 0.0)),
        ("Elbow L",  angles.get("left_elbow", 0.0)),
    ]
    for label, val in angle_items:
        y += 20
        _put(frame, label, cx, y, 0.46, C_TEXT_DIM)
        val_str = f"{val:.0f} deg"
        _put(frame, val_str, cx + cw - len(val_str) * 9, y, 0.48, C_TEXT_MAIN)

    # ── Divider ──────────────────────────────────────────────────────────
    y += 18
    cv2.line(frame, (cx, y), (w - 12, y), (60, 55, 80), 1)

    # ── Feedback messages ─────────────────────────────────────────────────
    if feedback:
        y += 16
        _put(frame, "FEEDBACK", cx, y, 0.44, C_TEXT_DIM)
        for msg in feedback[:3]:
            y += 20
            is_pos = msg.startswith("[OK]")
            color  = C_GOOD if is_pos else C_WARN
            prefix = "" if is_pos else "! "
            # Wrap long text
            display = (prefix + msg)[:28]
            _put(frame, display, cx, y, 0.43, color)

    # ── FPS (bottom of panel) ─────────────────────────────────────────────
    fps_str = f"FPS {fps:.1f}"
    _put(frame, fps_str, cx, h - 14, 0.44, C_TEXT_DIM)
    _put(frame, "Q=quit  R=reset", cx + 60, h - 14, 0.40, C_TEXT_DIM)

    return frame
