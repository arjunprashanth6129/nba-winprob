"""Refresh everything for a new season: results -> play-by-play -> dataset -> model.

Run once after each season ends (or any time) and restart the server:
    python scripts/update.py
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for step in ["fetch_games.py", "fetch_pbp.py", "build_dataset.py", "train.py", "evaluate.py", "season_report.py"]:
    print(f"\n=== {step} ===", flush=True)
    subprocess.run([sys.executable, str(HERE / step)], check=True)
