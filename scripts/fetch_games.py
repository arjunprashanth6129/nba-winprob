"""Step 1: download every finished game (results only) since ELO_FIRST_SEASON.

    python scripts/fetch_games.py
"""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from winprob.config import DATA, ELO_FIRST_SEASON, GAMES_PATH, SEASON_TYPES, current_season, season_str  # noqa: E402
from winprob.nba import season_games  # noqa: E402


def main():
    DATA.mkdir(exist_ok=True)
    frames = []
    for year in range(ELO_FIRST_SEASON, current_season() + 1):
        for st in SEASON_TYPES:
            try:
                g = season_games(year, st)
            except Exception as e:
                print(f"  {season_str(year)} {st}: failed ({e})")
                continue
            print(f"  {season_str(year)} {st:<15} {len(g):>5} games")
            frames.append(g)
            time.sleep(0.8)  # stats.nba.com rate-limits aggressively
    games = pd.concat(frames, ignore_index=True).sort_values(["date", "game_id"]).reset_index(drop=True)
    games.to_csv(GAMES_PATH, index=False)
    print(f"saved {len(games)} games -> {GAMES_PATH}")


if __name__ == "__main__":
    main()
