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

import os
import sys

# Suppress MediaPipe/TF C++ log noise before importing mediapipe
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import time
from pathlib import Path
from urllib.request import urlretrieve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
_GUI_AUTO_START_TIMEOUT_S = 5.0


def _resolve_model(path: str | None) -> str:
    if path:
        return path
    _MODEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if not _MODEL_CACHE.exists():
        print("Downloading pose_landmarker_full model...", flush=True)
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
    p.add_argument("--no-gui", action="store_true",
                   help="Disable OpenCV GUI windows and run headless.")
    return p.parse_args()


def _can_show_gui() -> bool:
    if os.environ.get("DISPLAY") is None and os.name != "nt":
        return False
    try:
        ret = cv2.namedWindow("__cv_test__", cv2.WINDOW_NORMAL)
        if ret is not None:
            cv2.destroyWindow("__cv_test__")
            return True
        return False
    except Exception:
        return False


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

    use_gui = not args.no_gui and _can_show_gui()
    selection_started = time.monotonic()

    # If headless and exercise is 'auto', prompt in terminal to pick exercise
    if fsm is None and not use_gui:
        print("\n=== Exercise Selection (Headless / Terminal Mode) ===")
        print("  1) Squat [S]")
        print("  2) Push-up [P]")
        print("  3) Lunge [L]")
        print("  4) Jumping Jack [J]")
        try:
            choice = input("Select exercise (1-4 or S/P/L/J): ").strip().lower()
            key_map = {
                "1": "squat", "s": "squat", "squat": "squat",
                "2": "pushup", "p": "pushup", "pushup": "pushup",
                "3": "lunge", "l": "lunge", "lunge": "lunge",
                "4": "jumpingjack", "j": "jumpingjack", "jumpingjack": "jumpingjack",
            }
            ex_name = key_map.get(choice, "squat")
            print(f"Selected exercise: {ex_name.upper()}\n")
            fsm = make_exercise(ex_name)
        except (KeyboardInterrupt, EOFError):
            print("\nSelection cancelled. Exiting.")
            sys.exit(0)

    if not use_gui:
        print("OpenCV GUI unavailable; running headless. Press Ctrl+C in terminal to stop.")
    else:
        print("Press Q to quit, R to reset reps.")
        if args.exercise == "auto":
            print("Press S=Squat  P=Push-up  L=Lunge  J=Jumping Jack to begin.")
            print(f"If you do nothing, Squat starts automatically after {_GUI_AUTO_START_TIMEOUT_S:.0f}s.")

    # ── Main loop ─────────────────────────────────────────────────────────────
    try:
        with PoseLandmarker.create_from_options(pose_opts) as detector:
            frame_idx = 0
            t0        = time.monotonic()
            last_ts   = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                img_h, img_w = frame.shape[:2]

                # ── Selection screen (GUI mode) ───────────────────────────────────
                if fsm is None:
                    frame = draw_selection_screen(frame)
                    if use_gui:
                        cv2.imshow("AI Personal Trainer", frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (ord("q"), ord("Q")):
                            break
                        elif key in (ord("s"), ord("S")):
                            fsm = make_exercise("squat")
                        elif key in (ord("p"), ord("P")):
                            fsm = make_exercise("pushup")
                        elif key in (ord("l"), ord("L")):
                            fsm = make_exercise("lunge")
                        elif key in (ord("j"), ord("J")):
                            fsm = make_exercise("jumpingjack")
                        elif time.monotonic() - selection_started >= _GUI_AUTO_START_TIMEOUT_S:
                            print("Auto-starting Squat after selection timeout.")
                            fsm = make_exercise("squat")
                        continue
                    if args.max_frames and frame_idx >= args.max_frames:
                        break
                    frame_idx += 1
                    continue

                # ── Pose detection ────────────────────────────────────────────────
                rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img   = Image(image_format=ImageFormat.SRGB, data=rgb)
                ts_ms    = int(time.time() * 1000)
                if ts_ms <= last_ts:
                    ts_ms = last_ts + 1
                last_ts  = ts_ms
                results  = detector.detect_for_video(mp_img, ts_ms)

                state    = ExerciseState(exercise=fsm.name, phase="no person detected")
                angles: dict[str, float] = {}
                image_lm = None
                img_flat = []

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
                    img_flat = image_lm_to_flat(image_lm)

                # Record frame (always record so session timeline is accurate)
                session.record_frame(frame_idx, ts_ms, img_flat, angles, state)

                # ── FPS ───────────────────────────────────────────────────────────
                elapsed = time.monotonic() - t0
                fps     = frame_idx / max(elapsed, 1e-3)

                # ── Render HUD ────────────────────────────────────────────────────
                frame = hud.render(frame, state, angles, fps, image_lm)
                if use_gui:
                    cv2.imshow("AI Personal Trainer", frame)

                frame_idx += 1
                if args.max_frames and frame_idx >= args.max_frames:
                    break

                if use_gui:
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), ord("Q")):
                        print("\nExiting recording gracefully...")
                        break
                    elif key in (ord("r"), ord("R")) and fsm is not None:
                        fsm.reset()
                        smoother.reset()
                        print("Rep counter reset.")
                else:
                    time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nSession stopped by user (KeyboardInterrupt).")
    finally:
        # ── Teardown ──────────────────────────────────────────────────────────────
        cap.release()
        if use_gui:
            cv2.destroyAllWindows()
        session.close()


if __name__ == "__main__":
    main()
