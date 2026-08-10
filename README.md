# 🏋️ AI Personal Trainer

Real-time exercise coaching using **3D pose estimation** (MediaPipe full model).  
The app watches you via webcam, **counts reps**, checks your **form in 3D**, and gives
instant coaching feedback. After each session, a browser-based **Replay Viewer**
lets you scrub through the recording and review joint angles over time.

## Features

| Feature | Details |
|---|---|
| **3D Pose** | `pose_landmarker_full` — real metric world-coordinates (x, y, z in metres) |
| **Exercises** | Squat · Push-up · Lunge · Jumping Jack |
| **Rep Counting** | Hysteresis FSM — robust to camera noise and speed variation |
| **Form Analysis** | Knee valgus · trunk lean · depth · asymmetry · hip sag |
| **Jitter Smoothing** | OneEuro filter on all 33 landmarks × 3 axes |
| **Live HUD** | Phase · rep counter · colour-coded form bar · angle readouts · feedback messages |
| **Session Recording** | Timestamped CSV (79 cols) + summary JSON in `out/session_*/` |
| **Replay Viewer** | Flask SPA with animated skeleton, Chart.js angle plots, timeline scrubber |

## Quick Start

```bash
# Install / sync dependencies (includes Flask)
uv sync

# Record a squat session  (press Q to quit & save)
uv run python src/inference.py --exercise squat

# Or let the app prompt you to choose
uv run python src/inference.py --exercise auto

# Launch the web replay viewer
uv run python web/app.py
# → open http://localhost:5000
```

## Label Studio

Label Studio is not part of the trainer runtime. Keep it in a separate Python 3.12 environment so its dependency chain stays isolated from the main app:

```powershell
uv venv .venv-labelstudio --python 3.12
& .\.venv-labelstudio\Scripts\Activate.ps1
uv pip install --python .\.venv-labelstudio\Scripts\python.exe label-studio
& .\.venv-labelstudio\Scripts\label-studio.exe
```

The main project still uses the regular `.venv` managed by `uv sync`.

## CLI Reference

```
src/inference.py
  --exercise   squat | pushup | lunge | jumpingjack | auto  (default: auto)
  --source     camera index or video file / RTSP URL        (default: 0)
  --model      path to custom .task file                    (default: auto-download)
  --out        output root directory                        (default: out/)
  --max_frames stop after N frames                          (default: unlimited)
```

## Controls

| Key | Action |
|---|---|
| `Q` | Quit and save session |
| `R` | Reset rep counter |
| `S / P / L / J` | Select exercise (auto mode only) |

## Output Format

Each session creates `out/session_YYYYMMDD_HHMMSS/`:

```
landmarks.csv   — 79 columns per frame
  frame, timestamp_ms, exercise, phase, rep, form_score,
  img_0_x, img_0_y, …, img_32_x, img_32_y,   (66 normalised image coords)
  angle_left_knee, …, angle_trunk_lean          (7 key angles)

summary.json    — { exercise, reps, duration_s, avg_form, frames }
```

## Camera Tips

| Exercise | Best Camera Position |
|---|---|
| Squat | Front-facing, full body in frame |
| Push-up | Side-on (90° profile) shows elbow angle best |
| Lunge | Front or 45° angle |
| Jumping Jack | Front-facing |

## Project Structure

```
vision/
├── src/
│   ├── landmarks.py   — MediaPipe index constants + helpers
│   ├── angles.py      — 3D joint angle computation
│   ├── smoothing.py   — OneEuro filter
│   ├── exercises.py   — Squat / Push-up / Lunge / Jumping Jack FSMs
│   ├── session.py     — CSV + JSON session recorder
│   ├── hud.py         — OpenCV HUD renderer
│   └── inference.py   — Main real-time pipeline
├── web/
│   ├── app.py                  — Flask backend
│   ├── templates/index.html    — Replay viewer SPA
│   └── static/pose.js          — Canvas skeleton renderer
└── out/                        — Session recordings (git-ignored)
```

## Roadmap / Next Steps

- [ ] Move classifier (train on session CSVs → identify exercise from motion alone)
- [ ] Multi-person support (`num_poses=4`)
- [ ] TFLite export for Android / iOS
- [ ] Rep quality score trend chart
- [ ] WebSocket live streaming to browser
