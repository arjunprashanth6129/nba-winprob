"""Thin wrappers around nba_api with retries, used by both the pipeline and the server."""
import time

import pandas as pd
import requests
from nba_api.live.nba.endpoints import boxscore, playbyplay, scoreboard
from nba_api.live.nba.library.http import NBALiveHTTP
from nba_api.stats.endpoints import leaguegamefinder, scoreboardv3

from .config import season_str

SCHEDULE_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
# The NBA's S3 mirror of the live play-by-play feed: byte-identical JSON, no rate limit.
S3_PBP_URL = "https://nba-prod-us-east-1-mediaops-stats.s3.amazonaws.com/NBA/liveData/playbyplay/playbyplay_{}.json"


def _retry(fn, tries=4, wait=2.0):
    for i in range(tries):
        try:
            return fn()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(wait * (i + 1))


def season_games(start_year: int, season_type: str) -> pd.DataFrame:
    """One row per game: id, date, home/away team ids + tricodes, final points.

    LeagueGameFinder returns one row per team per game; MATCHUP reads
    'BOS vs. NYK' for the home side and 'NYK @ BOS' for the away side.
    """
    def call():
        return leaguegamefinder.LeagueGameFinder(
            season_nullable=season_str(start_year),
            season_type_nullable=season_type,
            league_id_nullable="00",
            player_or_team_abbreviation="T",
            timeout=60,
        ).get_data_frames()[0]

    df = _retry(call)
    if df.empty:
        return pd.DataFrame()
    df = df[df["WL"].notna()]  # unfinished games have no W/L yet
    home = df[df["MATCHUP"].str.contains(" vs. ")]
    away = df[df["MATCHUP"].str.contains(" @ ")]
    cols = ["GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION", "PTS"]
    g = home[cols + ["GAME_DATE"]].merge(away[cols], on="GAME_ID", suffixes=("_H", "_A"))
    g = g.rename(columns={
        "GAME_ID": "game_id", "GAME_DATE": "date",
        "TEAM_ID_H": "home_id", "TEAM_ABBREVIATION_H": "home",
        "TEAM_ID_A": "away_id", "TEAM_ABBREVIATION_A": "away",
        "PTS_H": "home_pts", "PTS_A": "away_pts",
    })
    g["date"] = pd.to_datetime(g["date"])
    g["season"] = start_year
    g["season_type"] = season_type
    return g.drop_duplicates("game_id")


def pbp_actions(game_id: str) -> list[dict]:
    try:
        return _retry(lambda: playbyplay.PlayByPlay(game_id, timeout=20).get_dict()["game"]["actions"], tries=2)
    except Exception:
        r = requests.get(S3_PBP_URL.format(game_id), timeout=15)
        r.raise_for_status()
        return r.json()["game"]["actions"]


def live_boxscore(game_id: str) -> dict:
    return _retry(lambda: boxscore.BoxScore(game_id, timeout=15).get_dict()["game"], tries=2)


def live_scoreboard() -> dict:
    return _retry(lambda: scoreboard.ScoreBoard(timeout=15).get_dict()["scoreboard"], tries=2)


def league_schedule() -> dict:
    """The current season's full schedule, with final scores, from the NBA's CDN.

    Unlike stats.nba.com (which blocks most cloud hosts), the CDN answers from anywhere.
    """
    def call():
        r = requests.get(SCHEDULE_URL, headers=NBALiveHTTP.headers, timeout=20)
        r.raise_for_status()
        return r.json()["leagueSchedule"]
    return _retry(call, tries=2)


def scoreboard_for_date(day: str, timeout=30) -> dict:
    """ScoreboardV3 works for past, present and future dates (YYYY-MM-DD)."""
    return _retry(lambda: scoreboardv3.ScoreboardV3(game_date=day, league_id="00", timeout=timeout).get_dict()["scoreboard"], tries=1)
