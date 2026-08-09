"""
Web Replay Viewer — Flask backend.

Run:
  uv run python web/app.py

Then open:  http://localhost:5000
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, send_from_directory

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

app = Flask(__name__, template_folder="templates", static_folder="static")
OUT_ROOT = Path(__file__).parent.parent / "out"


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/sessions")
def list_sessions():
    """Return a list of available sessions with their summaries."""
    sessions = []
    if OUT_ROOT.exists():
        for d in sorted(OUT_ROOT.iterdir(), reverse=True):
            csv_path = d / "landmarks.csv"
            if not d.is_dir() or not csv_path.exists():
                continue
            summary_path = d / "summary.json"
            summary = {}
            if summary_path.exists():
                try:
                    summary = json.loads(summary_path.read_text())
                except Exception:
                    pass
            sessions.append({"name": d.name, **summary})
    return jsonify(sessions)


@app.route("/api/session/<name>")
def get_session(name: str):
    """Return all CSV rows for a session as JSON."""
    csv_path = OUT_ROOT / name / "landmarks.csv"
    if not csv_path.exists():
        return jsonify({"error": "session not found"}), 404

    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Cast numeric columns
            numeric_row = {}
            for k, v in row.items():
                try:
                    numeric_row[k] = float(v)
                except (ValueError, TypeError):
                    numeric_row[k] = v
            rows.append(numeric_row)
    return jsonify(rows)


@app.route("/api/summary/<name>")
def get_summary(name: str):
    """Return the session summary JSON."""
    summary_path = OUT_ROOT / name / "summary.json"
    if not summary_path.exists():
        return jsonify({"error": "summary not found"}), 404
    return jsonify(json.loads(summary_path.read_text()))


if __name__ == "__main__":
    print("🏋️  AI Personal Trainer — Replay Viewer")
    print("   Open http://localhost:5000 in your browser")
    app.run(debug=False, port=5000)
