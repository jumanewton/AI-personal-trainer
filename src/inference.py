"""
AI Personal Trainer — real-time pose inference pipeline.

Usage
-----
  uv run python src/inference.py --exercise squat
  uv run python src/inference.py --exercise pushup --source 1
  uv run python src/inference.py --exercise auto   # interactive selection

Controls
--------
  Q : quit and save session
  R : reset rep counter for current exercise
  S/P/L/J : switch exercise (only active in 'auto' mode at start)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    PoseLandmarksConnections,
)
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)

from src.angles import compute_all_angles
from src.exercises import EXERCISE_MAP, ExerciseState, make_exercise
from src.hud import HUDRenderer, draw_selection_screen
from src.landmarks import Point3D, image_lm_to_flat, world_lm_to_list
from src.session import SessionRecorder
from src.smoothing import LandmarkSmoother


# ─────────────────────────────────────────────────────────────────────────────
# Model download
# ─────────────────────────────────────────────────────────────────────────────

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
_MODEL_CACHE = Path.home() / ".cache" / "vision" / "pose_landmarker_full.task"


def _resolve_model(path: str | None) -> str:
    if path:
        return path
    _MODEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if not _MODEL_CACHE.exists():
        print("Downloading pose_landmarker_full model …", flush=True)
        urlretrieve(_MODEL_URL, _MODEL_CACHE)
        print("Download complete.", flush=True)
    return str(_MODEL_CACHE)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AI Personal Trainer — real-time pose coach")
    p.add_argument(
        "--exercise", default="auto",
        choices=list(EXERCISE_MAP) + ["auto"],
        help="Exercise to track. 'auto' shows an in-app selection menu.",
    )
    p.add_argument("--source", default=0,
                   help="Camera index (int) or video file / RTSP URL.")
    p.add_argument("--model",  default=None,
                   help="Path to a MediaPipe .task file (default: auto-download full model).")
    p.add_argument("--out",    default="out",
                   help="Root directory for session output folders.")
    p.add_argument("--max_frames", type=int, default=0,
                   help="Stop after N frames (0 = unlimited).")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    # ── Camera ───────────────────────────────────────────────────────────────
    source = int(args.source) if str(args.source).isdigit() else args.source
    cap    = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video source '{args.source}'", file=sys.stderr)
        sys.exit(1)

    # ── MediaPipe (full model → world landmarks available) ────────────────────
    model_path  = _resolve_model(args.model)
    pose_opts   = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionTaskRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # ── Sub-systems ───────────────────────────────────────────────────────────
    smoother  = LandmarkSmoother(min_cutoff=1.0, beta=0.007)
    hud       = HUDRenderer()
    session   = SessionRecorder(out_root=args.out)

    # Exercise FSM — None means we're in selection mode
    fsm = None if args.exercise == "auto" else make_exercise(args.exercise)

    # Default idle state (shown while selecting exercise)
    idle_state = ExerciseState(exercise="Select exercise", phase="—")

    print("Press Q to quit, R to reset reps.")
    if args.exercise == "auto":
        print("Press S=Squat  P=Push-up  L=Lunge  J=Jumping Jack to begin.")

    # ── Main loop ─────────────────────────────────────────────────────────────
    with PoseLandmarker.create_from_options(pose_opts) as detector:
        frame_idx = 0
        t0        = time.monotonic()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            img_h, img_w = frame.shape[:2]

            # ── Selection screen ──────────────────────────────────────────────
            if fsm is None:
                frame = draw_selection_screen(frame)
                cv2.imshow("AI Personal Trainer", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s"):
                    fsm = make_exercise("squat")
                elif key == ord("p"):
                    fsm = make_exercise("pushup")
                elif key == ord("l"):
                    fsm = make_exercise("lunge")
                elif key == ord("j"):
                    fsm = make_exercise("jumpingjack")
                continue

            # ── Pose detection ────────────────────────────────────────────────
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img   = Image(image_format=ImageFormat.SRGB, data=rgb)
            ts_ms    = int(time.time() * 1000)
            results  = detector.detect_for_video(mp_img, ts_ms)

            state  = ExerciseState(exercise=fsm.name, phase="no person detected")
            angles: dict[str, float] = {}
            image_lm = None

            if results.pose_landmarks and results.pose_world_landmarks:
                image_lm  = results.pose_landmarks[0]
                raw_world = world_lm_to_list(results.pose_world_landmarks[0])

                # Smooth in world space
                t_sec    = ts_ms / 1000.0
                smooth_w = smoother.smooth(raw_world, t_sec)

                # Compute angles from smoothed world landmarks
                angles   = compute_all_angles(smooth_w)

                # Update exercise FSM
                state    = fsm.update(angles, smooth_w)

                # Record frame
                img_flat = image_lm_to_flat(image_lm)
                session.record_frame(frame_idx, ts_ms, img_flat, angles, state)

            # ── FPS ───────────────────────────────────────────────────────────
            elapsed = time.monotonic() - t0
            fps     = frame_idx / max(elapsed, 1e-3)

            # ── Render HUD ────────────────────────────────────────────────────
            frame = hud.render(frame, state, angles, fps, image_lm)
            cv2.imshow("AI Personal Trainer", frame)

            frame_idx += 1
            if args.max_frames and frame_idx >= args.max_frames:
                break

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r") and fsm is not None:
                fsm.reset()
                smoother.reset()
                print("Rep counter reset.")

    # ── Teardown ──────────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    session.close()


if __name__ == "__main__":
    main()
