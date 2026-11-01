"""Turn a play-by-play log into model-ready game states.

This is the single source of truth for features: the training pipeline and
the live server both call `game_states`, so what the network sees in
production is exactly what it was trained on.
"""
import math
import re

import numpy as np

from .config import GAME_SEC, OT_PERIOD_SEC, REG_PERIOD_SEC

FEATURES = [
    "diff",           # home score - away score
    "t_min",          # minutes left in regulation (or in the current OT)
    "log_t",          # log(1 + t_min): resolution near the buzzer
    "diff_over_sqrt_t",  # lead scaled by remaining variance (~ a z-score)
    "is_ot",
    "poss",           # +1 home ball, -1 away ball, 0 unknown
    "pre_spread",     # Elo-implied pre-game home margin (points)
    "spread_left",    # pre_spread * share of game left: prior that still applies
    "home_to",        # timeouts left
    "away_to",
]

_CLOCK = re.compile(r"PT(?:(\d+)M)?(?:([\d.]+)S)?")


def clock_seconds(clock: str | None) -> float:
    if not clock:
        return 0.0
    m = _CLOCK.match(clock)
    if not m:
        return 0.0
    return int(m.group(1) or 0) * 60 + float(m.group(2) or 0)


def seconds_left(period: int, clock_s: float) -> float:
    """Seconds left in regulation; inside OT, seconds left in that OT."""
    if period <= 4:
        return (4 - period) * REG_PERIOD_SEC + clock_s
    return clock_s


def elapsed_seconds(period: int, clock_s: float) -> float:
    """Game clock elapsed since tip-off, for plotting on a single x-axis."""
    if period <= 4:
        return (period - 1) * REG_PERIOD_SEC + (REG_PERIOD_SEC - clock_s)
    return GAME_SEC + (period - 5) * OT_PERIOD_SEC + (OT_PERIOD_SEC - clock_s)


class _Timeouts:
    """NBA rules: 7 per game, at most 2 usable in the last 3:00 of the 4th, 2 per OT."""

    def __init__(self):
        self.used = 0
        self.used_late = 0  # in the last 3:00 of Q4
        self.used_ot = {}

    def take(self, period, clock_s):
        if period > 4:
            self.used_ot[period] = self.used_ot.get(period, 0) + 1
        else:
            self.used += 1
            if period == 4 and clock_s <= 180:
                self.used_late += 1

    def remaining(self, period, clock_s):
        if period > 4:
            return max(0, 2 - self.used_ot.get(period, 0))
        left = max(0, 7 - self.used)
        if period == 4 and clock_s <= 180:
            left = min(left, max(0, 2 - self.used_late))
        return left


def feature_row(diff, t_sec, is_ot, poss, spread, home_to, away_to) -> list[float]:
    t_min = t_sec / 60.0
    share_left = 0.0 if is_ot else t_sec / GAME_SEC
    return [
        float(diff),
        t_min,
        math.log1p(t_min),
        diff / math.sqrt(t_min + 1.0),
        float(is_ot),
        float(poss),
        spread,
        spread * share_left,
        float(home_to),
        float(away_to),
    ]


def pregame_state(spread: float) -> dict:
    return {
        "elapsed": 0.0, "period": 1, "clock": "12:00", "home_score": 0, "away_score": 0,
        "x": feature_row(0, GAME_SEC, False, 0, spread, 7, 7), "final": False,
        "description": "Tip-off", "team": None, "action": 0,
    }


def _fmt_clock(s: float) -> str:
    m, sec = divmod(s, 60)
    return f"{int(m)}:{int(sec):02d}" if s >= 60 else f"{s:.1f}"


def game_states(actions: list[dict], home_id: int, away_id: int, spread: float,
                dedupe: bool = True) -> list[dict]:
    """One state per action (after it happens), starting with a pre-game state.

    With dedupe=True, consecutive actions that leave the model inputs
    unchanged (e.g. substitutions) collapse into one row.
    """
    states = [pregame_state(spread)]
    tos = {home_id: _Timeouts(), away_id: _Timeouts()}
    poss = 0
    hs = as_ = 0
    last_key = tuple(states[0]["x"])
    for a in actions:
        period = int(a.get("period") or 1)
        cs = clock_seconds(a.get("clock"))
        atype = a.get("actionType")
        team = a.get("teamId")

        if atype == "timeout" and a.get("subType") == "full" and team in tos:
            tos[team].take(period, cs)
        p = a.get("possession")
        if p == home_id:
            poss = 1
        elif p == away_id:
            poss = -1
        try:
            hs, as_ = int(a.get("scoreHome") or hs), int(a.get("scoreAway") or as_)
        except ValueError:
            pass

        final = atype == "game" and a.get("subType") == "end"
        t = seconds_left(period, cs)
        x = feature_row(hs - as_, t, period > 4, poss, spread,
                        tos[home_id].remaining(period, cs), tos[away_id].remaining(period, cs))
        key = tuple(x)
        if dedupe and key == last_key and not final:
            continue
        last_key = key
        states.append({
            "elapsed": elapsed_seconds(period, cs),
            "period": period,
            "clock": _fmt_clock(cs),
            "home_score": hs,
            "away_score": as_,
            "x": x,
            "final": final,
            "description": a.get("description"),
            "team": a.get("teamTricode"),
            "action": a.get("actionNumber"),
        })
    return states


def matrix(states: list[dict]) -> np.ndarray:
    return np.asarray([s["x"] for s in states], dtype=np.float32)
