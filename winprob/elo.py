"""Team strength via Elo, used as the model's pre-game prior.

Follows FiveThirtyEight's NBA Elo (K=20, a margin-of-victory multiplier that
damps blowouts between mismatched teams, 25% regression to the mean between
seasons), except for home court: 538's 100-point edge predates the post-2020
decline in home advantage (home teams won ~58.5% of games in 2014-19 but ~55%
since), so the edge is fit by maximum likelihood on 2021-25 results instead.
Because ratings carry over season to season, the model has a sensible prior
for opening night of any future season.
"""
from dataclasses import dataclass, field

import pandas as pd

K = 20.0
HOME_ADV = 45.0  # Elo points; best fit on 2021-25 (538 used 100)
MEAN = 1505.0
CARRY = 0.75
ELO_PER_POINT = 28.0  # ~28 Elo points ≈ 1 point of expected margin

BUBBLE_START = pd.Timestamp("2020-07-30")  # 2019-20 restart in Orlando: no home court


def is_neutral(season: int, date: pd.Timestamp) -> bool:
    return season == 2019 and date >= BUBBLE_START


def expected_home(elo_h: float, elo_a: float, hca: float) -> float:
    return 1.0 / (1.0 + 10 ** (-(elo_h + hca - elo_a) / 400.0))


def pre_spread(elo_h: float, elo_a: float, hca: float) -> float:
    """Pre-game expected home margin in points."""
    return (elo_h + hca - elo_a) / ELO_PER_POINT


@dataclass
class EloBook:
    ratings: dict = field(default_factory=dict)
    season: int | None = None

    def roll_to(self, season: int):
        """Regress every team toward the mean once per new season."""
        if self.season is None:
            self.season = season
            return
        while self.season < season:
            self.ratings = {t: CARRY * r + (1 - CARRY) * MEAN for t, r in self.ratings.items()}
            self.season += 1

    def get(self, team_id: int) -> float:
        return self.ratings.get(int(team_id), MEAN)

    def update(self, home_id, away_id, home_pts, away_pts, hca):
        eh, ea = self.get(home_id), self.get(away_id)
        exp = expected_home(eh, ea, hca)
        home_won = home_pts > away_pts
        mov = abs(home_pts - away_pts)
        winner_edge = (eh + hca - ea) if home_won else (ea - eh - hca)
        mult = (mov + 3) ** 0.8 / (7.5 + 0.006 * winner_edge)
        delta = K * mult * ((1.0 if home_won else 0.0) - exp)
        self.ratings[int(home_id)] = eh + delta
        self.ratings[int(away_id)] = ea - delta


def run_elo(games: pd.DataFrame) -> tuple[pd.DataFrame, EloBook]:
    """Walk games chronologically. Returns pre-game ratings per game and the final book."""
    book = EloBook()
    rows = []
    for g in games.sort_values(["date", "game_id"]).itertuples():
        book.roll_to(g.season)
        hca = 0.0 if is_neutral(g.season, g.date) else HOME_ADV
        eh, ea = book.get(g.home_id), book.get(g.away_id)
        rows.append((g.game_id, eh, ea, hca))
        book.update(g.home_id, g.away_id, g.home_pts, g.away_pts, hca)
    hist = pd.DataFrame(rows, columns=["game_id", "elo_home", "elo_away", "hca"])
    return hist, book
