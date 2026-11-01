"""Shared paths and season helpers."""
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PBP_DIR = DATA / "pbp"
MODELS = ROOT / "models"

GAMES_PATH = DATA / "games.csv.gz"         # every finished game since ELO_FIRST_SEASON (small, so it ships with the app)
DATASET_PATH = DATA / "dataset.parquet"    # one row per play-by-play event
ELO_PATH = DATA / "elo_history.parquet"    # pre-game Elo for every game
MODEL_PATH = MODELS / "winprob.pt"         # PyTorch checkpoint (training)
WEIGHTS_PATH = MODELS / "winprob.npz"      # same weights as plain arrays (serving, no PyTorch needed)
METRICS_PATH = MODELS / "metrics.json"
EVAL_PATH = MODELS / "evaluation.json"     # deeper test-season evaluation (scripts/evaluate.py)
SEASON_PATH = MODELS / "season_games.json"  # this season's scored games (scripts/season_report.py)

# Elo needs a few seasons of warm-up before the play-by-play window starts.
ELO_FIRST_SEASON = 2014
# The live CDN has play-by-play from 2019-20 onward.
PBP_FIRST_SEASON = 2019

SEASON_TYPES = ["Regular Season", "PlayIn", "Playoffs"]

REG_PERIOD_SEC = 720
OT_PERIOD_SEC = 300
GAME_SEC = 4 * REG_PERIOD_SEC


def season_str(start_year: int) -> str:
    """2025 -> '2025-26'."""
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def season_of(d: date) -> int:
    """Start year of the NBA season a date falls in (seasons tip off in October)."""
    return d.year if d.month >= 8 else d.year - 1


def current_season() -> int:
    return season_of(date.today())


def load_games() -> pd.DataFrame:
    return pd.read_csv(GAMES_PATH, dtype={"game_id": str}, parse_dates=["date"])
