"""
Session recorder: writes per-frame data to a timestamped CSV and a session
summary JSON file on close.

Directory layout
----------------
out/
  session_YYYYMMDD_HHMMSS/
    landmarks.csv   — per-frame row: frame metadata + image landmarks + key angles
    summary.json    — aggregated stats written at session end
"""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path

from src.exercises import ExerciseState
from src.landmarks import Point3D


# CSV column header (order must match _make_row)
_IMG_COLS  = [f"img_{i}_{c}" for i in range(33) for c in ("x", "y")]
_ANGLE_COLS = [
    "angle_left_knee", "angle_right_knee",
    "angle_left_hip",  "angle_right_hip",
    "angle_left_elbow","angle_right_elbow",
    "angle_trunk_lean",
]

HEADER = (
    ["frame", "timestamp_ms", "exercise", "phase", "rep", "form_score"]
    + _IMG_COLS
    + _ANGLE_COLS
)

_ANGLE_KEYS = [
    "left_knee", "right_knee",
    "left_hip",  "right_hip",
    "left_elbow","right_elbow",
    "trunk_lean",
]


class SessionRecorder:
    """
    Opens output files immediately; call `record_frame()` each frame; call
    `close()` (or use as a context manager) to flush + write the summary.
    """

    def __init__(self, out_root: str = "out") -> None:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._dir  = Path(out_root) / f"session_{ts}"
        self._dir.mkdir(parents=True, exist_ok=True)

        csv_path   = self._dir / "landmarks.csv"
        self._file = open(csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(HEADER)

        # Accumulators for summary
        self._n_frames:    int   = 0
        self._form_scores: list[float] = []
        self._max_reps:    int   = 0
        self._exercise:    str   = "unknown"
        self._t_start:     float | None = None
        self._t_last:      float | None = None

    # ── Context-manager support ───────────────────────────────────────────────

    def __enter__(self) -> "SessionRecorder":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── Per-frame recording ───────────────────────────────────────────────────

    def record_frame(
        self,
        frame_idx:    int,
        timestamp_ms: int,
        image_lm_flat: list[float],      # 66 normalised [x,y] values
        angles:        dict[str, float],
        state:         ExerciseState,
    ) -> None:
        """Write one row to the CSV."""
        t_s = timestamp_ms / 1000.0
        if self._t_start is None:
            self._t_start = t_s
        self._t_last = t_s

        angle_vals = [round(angles.get(k, 0.0), 2) for k in _ANGLE_KEYS]
        if not image_lm_flat or len(image_lm_flat) < 66:
            img_vals = [""] * 66
        else:
            img_vals = [round(v, 5) for v in image_lm_flat[:66]]

        row = (
            [frame_idx, timestamp_ms,
             state.exercise, state.phase, state.rep,
             round(state.form_score, 1)]
            + img_vals
            + angle_vals
        )
        self._writer.writerow(row)

        # Accumulate stats
        self._n_frames += 1
        self._form_scores.append(state.form_score)
        self._max_reps  = max(self._max_reps, state.rep)
        self._exercise  = state.exercise

    # ── Session teardown ──────────────────────────────────────────────────────

    def close(self) -> Path:
        """Flush CSV, write summary JSON, return the session directory path."""
        self._file.flush()
        self._file.close()

        if self._n_frames == 0:
            # If 0 frames recorded, clean up empty directory
            import shutil
            shutil.rmtree(self._dir, ignore_errors=True)
            print(f"\n[!] Session cancelled (0 frames recorded). Removed empty directory.")
            return self._dir

        duration_s = (
            round(self._t_last - self._t_start, 1)
            if (self._t_start is not None and self._t_last is not None)
            else 0.0
        )
        avg_form = (
            round(sum(self._form_scores) / len(self._form_scores), 1)
            if self._form_scores else 0.0
        )

        summary = {
            "exercise":    self._exercise,
            "reps":        self._max_reps,
            "duration_s":  duration_s,
            "avg_form":    avg_form,
            "frames":      self._n_frames,
        }
        summary_path = self._dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        print(f"\n[OK] Session saved -> {self._dir}")
        print(f"   Exercise : {self._exercise}")
        print(f"   Reps     : {self._max_reps}")
        print(f"   Avg form : {avg_form:.0f}%")
        print(f"   Duration : {duration_s:.0f}s")

        return self._dir

    @property
    def directory(self) -> Path:
        return self._dir
