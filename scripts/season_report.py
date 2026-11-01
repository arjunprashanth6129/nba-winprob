"""Score this season's finished games and cache them for the dashboard's Track record tab.

    python scripts/season_report.py

The server can score new games itself, but each one costs a play-by-play
download, so this cache (models/season_games.json) is refreshed daily by a
GitHub Action and shipped with the app.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "app")]
from winprob.config import SEASON_PATH  # noqa: E402
from live import Service  # noqa: E402


def main():
    svc = Service()
    before = len(svc.season._games)
    svc.season.update()
    if svc.schedule.season is None:
        sys.exit("schedule unavailable")
    SEASON_PATH.write_text(json.dumps(svc.season.dump(), separators=(",", ":"), sort_keys=True))
    print(f"{len(svc.season._games)} games scored ({len(svc.season._games) - before} new) -> {SEASON_PATH}")


if __name__ == "__main__":
    main()
