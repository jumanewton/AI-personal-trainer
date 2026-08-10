"""
Scan .py files for non-ASCII characters inside print() or cv2.putText() calls.
These would crash on a Windows console using cp1252 encoding.
"""
import re
import pathlib

BAD = re.compile(r"[\u0080-\uffff]")
RISKY = re.compile(r"(print\s*\(|putText)")

found = 0
for f in sorted(pathlib.Path(".").rglob("*.py")):
    parts = f.parts
    if "__pycache__" in parts or ".venv" in parts or ".venv-labelstudio" in parts:
        continue
    text = f.read_text(encoding="utf-8", errors="replace")
    for i, line in enumerate(text.splitlines(), 1):
        if RISKY.search(line) and BAD.search(line):
            print(f"{f}:{i}: {line.strip()}")
            found += 1

print(f"\nTotal risky lines: {found}")
