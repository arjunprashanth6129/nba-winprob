"""Step 3: compute Elo for every game, then turn every cached play-by-play log
into feature rows labelled with the eventual winner.

    python scripts/build_dataset.py
"""
import gzip
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from winprob.config import DATASET_PATH, ELO_PATH, PBP_DIR, PBP_FIRST_SEASON, load_games  # noqa: E402
from winprob.elo import pre_spread, run_elo  # noqa: E402
from winprob.features import FEATURES, game_states  # noqa: E402


def featurize(job):
    gid, season, home_id, away_id, home_pts, away_pts, spread = job
    path = PBP_DIR / str(season) / f"{gid}.json.gz"
    if not path.exists():
        return None, "missing"
    with gzip.open(path, "rt") as f:
        actions = json.load(f)
    states = game_states(actions, home_id, away_id, spread)
    last = states[-1]
    # Guard against truncated or mislabelled logs.
    if (last["home_score"], last["away_score"]) != (home_pts, away_pts):
        return None, "score mismatch"
    last["final"] = True  # a few logs lack the 'game end' event
    X = np.asarray([s["x"] for s in states], dtype=np.float32)
    df = pd.DataFrame(X, columns=FEATURES)
    df["game_id"] = gid
    df["season"] = season
    df["elapsed"] = np.asarray([s["elapsed"] for s in states], dtype=np.float32)
    df["final"] = [s["final"] for s in states]
    df["y"] = np.int8(home_pts > away_pts)
    return df, "ok"


def main():
    games = load_games()
    hist, _ = run_elo(games)
    games = games.merge(hist, on="game_id")
    games.to_parquet(ELO_PATH, index=False)

    window = games[games["season"] >= PBP_FIRST_SEASON]
    jobs = [
        (g.game_id, g.season, g.home_id, g.away_id, g.home_pts, g.away_pts,
         pre_spread(g.elo_home, g.elo_away, g.hca))
        for g in window.itertuples()
    ]
    frames, status = [], {}
    with ProcessPoolExecutor() as pool:
        for df, s in pool.map(featurize, jobs, chunksize=64):
            status[s] = status.get(s, 0) + 1
            if df is not None:
                frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    data.to_parquet(DATASET_PATH, index=False)
    print(f"games: {status}")
    print(f"rows: {len(data):,}  ->  {DATASET_PATH}")
    print(data.groupby("season")["game_id"].nunique().rename("games").to_string())


if __name__ == "__main__":
    main()
