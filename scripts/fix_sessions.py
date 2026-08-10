"""
Utility: clean up empty sessions and regenerate missing summary.json files.
Run once: python scripts/fix_sessions.py
"""
import csv
import json
import shutil
from pathlib import Path


OUT_DIR = Path("out")


def main():
    if not OUT_DIR.exists():
        print("No out/ directory found.")
        return

    for session_dir in sorted(OUT_DIR.iterdir()):
        if not session_dir.is_dir():
            continue

        csv_path = session_dir / "landmarks.csv"
        summary_path = session_dir / "summary.json"
        name = session_dir.name

        # Check if CSV exists and has data
        if not csv_path.exists():
            print(f"  [DELETE] {name}: no landmarks.csv")
            shutil.rmtree(session_dir, ignore_errors=True)
            continue

        # Count data rows
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if len(rows) == 0:
            print(f"  [DELETE] {name}: 0 data rows (header only)")
            shutil.rmtree(session_dir, ignore_errors=True)
            continue

        # Has data - check if summary.json exists
        if summary_path.exists():
            print(f"  [OK]     {name}: {len(rows)} rows, summary.json present")
            continue

        # Generate summary.json from CSV data
        exercise = rows[0].get("exercise", "unknown")
        max_rep = 0
        form_scores = []
        t_start = None
        t_end = None

        for row in rows:
            rep = int(row.get("rep", 0) or 0)
            if rep > max_rep:
                max_rep = rep

            score = float(row.get("form_score", 0) or 0)
            form_scores.append(score)

            ts = float(row.get("timestamp_ms", 0) or 0)
            if t_start is None:
                t_start = ts
            t_end = ts

        duration_s = round((t_end - t_start) / 1000, 1) if t_start and t_end else 0.0
        avg_form = round(sum(form_scores) / len(form_scores), 1) if form_scores else 0.0

        summary = {
            "exercise": exercise,
            "reps": max_rep,
            "duration_s": duration_s,
            "avg_form": avg_form,
            "frames": len(rows),
        }

        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"  [REGEN]  {name}: {len(rows)} rows, generated summary.json "
              f"(exercise={exercise}, reps={max_rep}, dur={duration_s}s, form={avg_form}%)")


if __name__ == "__main__":
    main()
