"""Live data layer for the dashboard: team ratings, schedules and per-game trackers.

All NBA calls are cached and throttled here so any number of browser tabs
share a single poll of the NBA's servers.
"""
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import date, datetime, timezone

import pandas as pd
from nba_api.stats.static import teams as static_teams

from winprob.config import EVAL_PATH, SEASON_PATH, load_games, season_of
from winprob.elo import ELO_PER_POINT, HOME_ADV, MEAN, EloBook, is_neutral, pre_spread, run_elo
from winprob.features import feature_row, game_states, pregame_state
from winprob.inference import Predictor
from winprob.summary import CALIB_BINS, CHECKPOINTS, explain, summarize
from winprob import nba

LIVE_POLL_SEC = 5
PREGAME_POLL_SEC = 30
SCOREBOARD_TTL_TODAY = 15
SCOREBOARD_TTL_OTHER = 600
RATINGS_TTL = 3 * 3600
SCHEDULE_TTL = 600
ELO_GAME_TYPES = {"002", "004", "005"}  # game-id prefixes: regular season, playoffs, play-in
TEAMS = {t["id"]: t for t in static_teams.get_teams()}  # offline names for historical games


def _team(t):
    return {"id": t["teamId"], "tricode": t["teamTricode"], "name": t.get("teamName", ""),
            "city": t.get("teamCity", ""), "score": t.get("score", 0),
            "record": f"{t['wins']}-{t['losses']}" if "wins" in t else ""}


class Schedule:
    """This season's schedule and results from the NBA CDN.

    The CDN answers from anywhere, unlike stats.nba.com, which blocks most cloud
    hosts, so this is what keeps the deployed app's schedule and ratings current.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._at = 0.0
        self.season = None
        self._by_date, self._by_id = {}, {}

    def _load(self):
        with self._lock:
            if time.time() - self._at < SCHEDULE_TTL:
                return
            try:
                raw = nba.league_schedule()
            except Exception as e:  # keep the last copy; try again in a minute
                print(f"[schedule] fetch failed: {e}")
                self._at = time.time() - SCHEDULE_TTL + 60
                return
            season = int(raw["seasonYear"][:4])
            by_date, by_id = {}, {}
            for d in raw["gameDates"]:
                for g in d["games"]:
                    day = g["gameDateEst"][:10]
                    summary = {
                        "game_id": g["gameId"], "status": g["gameStatus"], "status_text": g["gameStatusText"].strip(),
                        "period": 0, "clock": "", "start_utc": g["gameDateTimeUTC"],
                        "neutral": bool(g.get("isNeutral")), "label": g.get("gameLabel", ""),
                        "series": g.get("seriesText", ""), "home": _team(g["homeTeam"]), "away": _team(g["awayTeam"]),
                        "season": season, "date": day,
                    }
                    by_date.setdefault(day, []).append(summary)
                    by_id[g["gameId"]] = summary
            self.season, self._by_date, self._by_id, self._at = season, by_date, by_id, time.time()

    def covers(self, day: str) -> bool:
        self._load()
        return self.season is not None and season_of(date.fromisoformat(day)) == self.season

    def day(self, day: str) -> list[dict]:
        self._load()
        return deepcopy(self._by_date.get(day, []))

    def game(self, game_id: str) -> dict | None:
        self._load()
        g = self._by_id.get(game_id)
        return deepcopy(g) if g else None

    def results(self) -> pd.DataFrame:
        """Finished games that count toward Elo, shaped like `load_games()`."""
        self._load()
        rows = [(g["game_id"], pd.Timestamp(g["date"]), g["home"]["id"], g["home"]["tricode"], g["home"]["score"],
                 g["away"]["id"], g["away"]["tricode"], g["away"]["score"], g["season"])
                for g in self._by_id.values()
                if g["status"] == 3 and g["game_id"][:3] in ELO_GAME_TYPES and g["home"]["score"] + g["away"]["score"] > 0]
        return pd.DataFrame(rows, columns=["game_id", "date", "home_id", "home", "home_pts",
                                           "away_id", "away", "away_pts", "season"])


def _historical(g) -> dict:
    """Board summary for a finished game from the bundled results table."""
    def team(tid, tri, pts):
        t = TEAMS.get(tid, {})
        return {"id": tid, "tricode": tri, "name": t.get("nickname", ""), "city": t.get("city", ""),
                "score": int(pts), "record": ""}
    return {
        "game_id": g.game_id, "status": 3, "status_text": "Final", "period": 4, "clock": "",
        "start_utc": f"{g.date:%Y-%m-%d}T00:00:00Z", "neutral": is_neutral(g.season, g.date),
        "label": "", "series": "", "season": int(g.season),
        "home": team(int(g.home_id), g.home, g.home_pts), "away": team(int(g.away_id), g.away, g.away_pts),
    }


class Ratings:
    """Elo book kept current: historical games from disk + this season's results from the schedule."""

    def __init__(self, schedule: Schedule):
        self.schedule = schedule
        self._lock = threading.Lock()
        self._book = None
        self._hist = None
        self._games = None
        self._at = 0.0

    def book(self):
        with self._lock:
            if self._book is None or time.time() - self._at > RATINGS_TTL:
                games = load_games()
                fresh = self.schedule.results()  # empty if the CDN is down: serve slightly stale ratings
                if len(fresh):
                    games = pd.concat([games[games.season < self.schedule.season], fresh], ignore_index=True)
                games = games.drop_duplicates("game_id")
                hist, self._book = run_elo(games)
                self._hist = hist.set_index("game_id")
                self._games = games
                self._at = time.time()
                print(f"[ratings] Elo updated through {games.date.max():%Y-%m-%d}")
            return self._book

    def spread_for_game(self, game_id, home_id, away_id, season, neutral=False):
        """Pre-game spread using ratings as they stood before that game, if it's been played."""
        self.book()
        if game_id in self._hist.index:
            r = self._hist.loc[game_id]
            return pre_spread(r.elo_home, r.elo_away, r.hca), r.elo_home, r.elo_away
        return self.spread(home_id, away_id, season, neutral)

    def spread(self, home_id, away_id, season, neutral=False):
        book = self.book()
        # Regress a copy if the requested season hasn't started in our data yet.
        b = EloBook(dict(book.ratings), book.season)
        b.roll_to(season)
        eh, ea = b.get(home_id), b.get(away_id)
        return pre_spread(eh, ea, 0.0 if neutral else HOME_ADV), eh, ea

    def played_on(self, day: str) -> list[dict] | None:
        """Finished games on a date, or None if the bundled history doesn't cover that season."""
        self.book()
        g = self._games
        if not g.season.min() <= season_of(date.fromisoformat(day)) <= g.season.max():
            return None
        return [_historical(r) for r in g[g.date == pd.Timestamp(day)].itertuples()]

    def played(self, game_id: str) -> dict | None:
        self.book()
        hit = self._games[self._games.game_id == game_id]
        return _historical(next(hit.itertuples())) if len(hit) else None

    def table(self) -> list[dict]:
        """Every current team's rating, its 30-day change, and record."""
        book = self.book()
        g, hist = self._games, self._hist
        season = g.season.max()
        recent = g[(g.season == season) & (g.date >= g.date.max() - pd.Timedelta(days=30))].sort_values("date")
        then = {}
        for r in recent.merge(hist, left_on="game_id", right_index=True).itertuples():
            then.setdefault(r.home_id, r.elo_home)
            then.setdefault(r.away_id, r.elo_away)
        cur = g[g.season == season]
        rows = []
        for tid, t in TEAMS.items():
            elo = book.get(tid)
            mine = cur[(cur.home_id == tid) | (cur.away_id == tid)]
            won = ((mine.home_id == tid) & (mine.home_pts > mine.away_pts)) | ((mine.away_id == tid) & (mine.away_pts > mine.home_pts))
            rows.append({"id": tid, "tricode": t["abbreviation"], "name": t["nickname"], "city": t["city"],
                         "elo": round(elo, 1), "vs_avg": round((elo - MEAN) / ELO_PER_POINT, 1),
                         "change_30d": round(elo - then.get(tid, elo), 1),
                         "wins": int(won.sum()), "losses": int(len(mine) - won.sum())})
        rows.sort(key=lambda r: -r["elo"])
        return rows


class Service:
    def __init__(self):
        self.predictor = Predictor()
        self.schedule = Schedule()
        self.ratings = Ratings(self.schedule)
        self.season = SeasonTracker(self)
        try:
            self.typical_timeouts = json.loads(EVAL_PATH.read_text())["typical_timeouts"]
        except Exception:
            self.typical_timeouts = [7] * 6 + [6] * 11 + [5] * 11 + [4] * 10 + [3] * 6 + [2] * 3 + [1]
        self._boards = {}       # date -> (fetched_at, games)
        self._meta = {}         # game_id -> game summary
        self._trackers = {}
        self._lock = threading.Lock()

    # ---------- schedule ----------
    def scoreboard(self, day: str) -> list[dict]:
        today = date.today().isoformat()
        ttl = SCOREBOARD_TTL_TODAY if day >= today or day == self._live_date() else SCOREBOARD_TTL_OTHER
        with self._lock:
            hit = self._boards.get(day)
        if hit and time.time() - hit[0] < ttl:
            return hit[1]

        # status: 1 scheduled, 2 live, 3 final
        if self.schedule.covers(day):
            games = self.schedule.day(day)
        elif (past := self.ratings.played_on(day)) is not None:
            games = past
        else:
            games = self._stats_scoreboard(day)
        self._overlay_live(day, games)
        with self._lock:
            for g in games:
                self._meta.setdefault(g["game_id"], {}).update(g)
        self._probs(games)
        with self._lock:
            self._boards[day] = (time.time(), games)
        return games

    def _stats_scoreboard(self, day):
        """Seasons outside the schedule and the bundled history (stats.nba.com; may be unreachable from cloud hosts)."""
        try:
            sb = nba.scoreboard_for_date(day, timeout=10)
        except Exception as e:
            print(f"[board] ScoreboardV3 {day} failed: {e}")
            return []
        return [{
            "game_id": g["gameId"], "status": g["gameStatus"],
            "status_text": g["gameStatusText"].strip(), "period": g.get("period", 0),
            "clock": g.get("gameClock", ""), "start_utc": g.get("gameTimeUTC"),
            "neutral": bool(g.get("isNeutral")), "label": g.get("gameLabel", ""),
            "series": g.get("seriesText", ""),
            "home": _team(g["homeTeam"]), "away": _team(g["awayTeam"]),
            "season": season_of(date.fromisoformat(day)),
        } for g in sb["games"]]

    def _live_date(self):
        try:
            return self._live_board()["gameDate"]
        except Exception:
            return None

    _live_cache = (0.0, None)

    def _live_board(self):
        at, board = self._live_cache
        if board is None or time.time() - at > SCOREBOARD_TTL_TODAY:
            board = nba.live_scoreboard()
            self._live_cache = (time.time(), board)
        return board

    def _overlay_live(self, day, games):
        """ScoreboardV3 lags during games; the live CDN scoreboard has real-time scores."""
        try:
            board = self._live_board()
        except Exception:
            return
        live = {g["gameId"]: g for g in board["games"]}
        for g in games:
            lg = live.get(g["game_id"])
            if not lg:
                continue
            g.update(status=lg["gameStatus"], status_text=lg["gameStatusText"].strip(),
                     period=lg["period"], clock=lg.get("gameClock", ""))
            g["home"]["score"], g["away"]["score"] = lg["homeTeam"]["score"], lg["awayTeam"]["score"]

    def _probs(self, games):
        """Pre-game and current P(home win) for each game on a board."""
        spreads = [self.ratings.spread_for_game(g["game_id"], g["home"]["id"], g["away"]["id"], g["season"], g["neutral"])[0]
                   for g in games]
        pre = self.predictor.score_states([pregame_state(s) for s in spreads]) if games else []
        for g, spread, p in zip(games, spreads, pre):
            g["pre_spread"] = round(spread, 1)
            g["pregame_prob"] = round(p, 4)
            g["home_win_prob"] = p
            if g["status"] == 3:
                g["home_win_prob"] = 1.0 if g["home"]["score"] > g["away"]["score"] else 0.0
            elif g["status"] == 2:
                tr = self.tracker(g["game_id"])
                tr.refresh()
                if tr.probs:
                    g["home_win_prob"] = tr.probs[-1]

    # ---------- what-if ----------
    def whatif(self, diff, t_sec, ot, poss, strength, home_court, hto, ato) -> dict:
        """The model on a made-up game state, plus its whole lead x time surface."""
        hc = HOME_ADV / ELO_PER_POINT if home_court else 0.0
        spread = strength + hc
        x = feature_row(diff, t_sec, ot, poss, spread, hto, ato)
        typ = self.typical_timeouts
        minutes = list(range(48, -1, -1))  # minutes left in regulation
        leads = list(range(-20, 21))
        rows = [feature_row(d, m * 60, False, poss, spread, typ[min(47, 48 - m)], typ[min(47, 48 - m)])
                for m in minutes for d in leads]
        curve_rows = [feature_row(d, t_sec, ot, poss, spread, hto, ato) for d in leads]
        surf = self.predictor.predict(rows).reshape(len(minutes), len(leads))
        return {
            "p": round(float(self.predictor.predict([x])[0]), 4),
            "spread": round(spread, 2),
            "explain": explain(x, self.predictor, hc),
            "leads": leads, "minutes": minutes,
            "surface": [[round(float(v), 3) for v in r] for r in surf],
            "lead_curve": [round(float(v), 4) for v in self.predictor.predict(curve_rows)],
        }

    # ---------- games ----------
    def tracker(self, game_id: str) -> "GameTracker":
        with self._lock:
            tr = self._trackers.get(game_id)
            if tr is None:
                tr = self._trackers[game_id] = GameTracker(self, game_id)
        return tr

    def meta(self, game_id: str) -> dict:
        m = self._meta.get(game_id) or self.schedule.game(game_id) or self.ratings.played(game_id)
        if m:
            self._meta[game_id] = m
            return m
        # Deep link to a game we haven't listed yet: build its summary from the live box score.
        box = nba.live_boxscore(game_id)
        d = date.fromisoformat(box["gameTimeUTC"][:10])
        m = {
            "game_id": game_id, "status": box["gameStatus"], "status_text": box["gameStatusText"].strip(),
            "period": box.get("period", 0), "clock": box.get("gameClock", ""), "start_utc": box["gameTimeUTC"],
            "neutral": False, "label": "", "series": "", "season": season_of(d),
            "home": _team(box["homeTeam"]), "away": _team(box["awayTeam"]),
        }
        self._meta[game_id] = m
        return m


class GameTracker:
    """Holds the scored timeline for one game and refreshes it while the game is live."""

    def __init__(self, svc: Service, game_id: str):
        self.svc = svc
        self.game_id = game_id
        self.meta = svc.meta(game_id)
        self.spread, self.elo_home, self.elo_away = svc.ratings.spread_for_game(
            game_id, self.meta["home"]["id"], self.meta["away"]["id"], self.meta["season"], self.meta["neutral"])
        self.home_court = self.spread - (self.elo_home - self.elo_away) / ELO_PER_POINT
        self.states, self.probs = [], []
        self.stats, self.explanation = {}, None
        self.version = 0
        self.finished = False
        self._fetched = 0.0
        self._lock = threading.Lock()
        self.refresh(force=True)

    def refresh(self, force=False):
        with self._lock:
            # Before tip-off the play-by-play feed doesn't exist yet, so check less often.
            interval = LIVE_POLL_SEC if len(self.states) > 1 or self.meta["status"] != 1 else PREGAME_POLL_SEC
            if self.finished or (not force and time.time() - self._fetched < interval):
                return self.version
            self._fetched = time.time()
            try:
                actions = nba.pbp_actions(self.game_id)
            except Exception:
                actions = []  # not started yet (the CDN 403s until tip-off) or a blip
            home, away = self.meta["home"]["id"], self.meta["away"]["id"]
            states = game_states(actions, home, away, self.spread)
            if len(states) != len(self.states) or (states and states[-1]["x"] != self.states[-1]["x"]):
                self.states = states
                self.probs = self.svc.predictor.score_states(states)
                self.stats = summarize(states, self.probs)
                live = [s for s in states if not s["final"]]
                if live:
                    s = live[-1]
                    self.explanation = {"per": s["period"], "clk": s["clock"],
                                        "steps": explain(s["x"], self.svc.predictor, self.home_court)}
                self.version += 1
            if states and states[-1]["final"]:
                self.finished = True
                self.meta["status"] = 3
            elif len(states) > 1:
                self.meta["status"] = max(self.meta["status"], 2)
            return self.version

    def points(self, start=0, end=None):
        out = []
        for s, p in zip(self.states[start:end], self.probs[start:end]):
            out.append({
                "t": round(s["elapsed"], 1), "p": round(p, 4), "hs": s["home_score"], "as": s["away_score"],
                "per": s["period"], "clk": s["clock"], "desc": s["description"], "team": s["team"],
            })
        return out

    def header(self):
        m = self.meta
        return {
            "game_id": self.game_id, "home": m["home"], "away": m["away"], "status": m["status"],
            "status_text": m["status_text"], "start_utc": m["start_utc"], "series": m.get("series", ""),
            "label": m.get("label", ""), "neutral": m["neutral"],
            "elo_home": round(self.elo_home), "elo_away": round(self.elo_away),
            "pre_spread": round(self.spread, 1), "pregame_prob": self.probs[0] if self.probs else None,
            "home_court": round(self.home_court, 2), "finished": self.finished,
            "stats": self.stats, "explain": self.explanation,
        }


class SeasonTracker:
    """The model's record on this season's finished games.

    All of it is out of sample: the network was trained on earlier seasons, and
    each game's Elo prior only uses results from before that game tipped off.
    Scored games are cached (and shipped in models/season_games.json by
    scripts/season_report.py), so only games since then need their play-by-play.
    """

    TTL = 600

    def __init__(self, svc: "Service"):
        self.svc = svc
        self._lock = threading.Lock()
        try:
            self._games = json.loads(SEASON_PATH.read_text())["games"]
        except Exception:
            self._games = {}
        self._report, self._at = None, 0.0

    def _score(self, r):
        """Summarise one finished game from its play-by-play; None if the log is unusable."""
        try:
            actions = nba.pbp_actions(r.game_id)
        except Exception:
            return None
        spread, _, _ = self.svc.ratings.spread_for_game(r.game_id, r.home_id, r.away_id, r.season)
        states = game_states(actions, int(r.home_id), int(r.away_id), spread)
        last = states[-1]
        if (last["home_score"], last["away_score"]) != (r.home_pts, r.away_pts):
            return None
        last["final"] = True  # a few logs lack the 'game end' event
        return summarize(states, self.svc.predictor.score_states(states))

    def update(self) -> pd.DataFrame:
        """Score any finished games not yet cached. Returns this season's results."""
        res = self.svc.schedule.results().sort_values(["date", "game_id"])
        todo = [r for r in res.itertuples() if r.game_id not in self._games]
        if todo:
            with ThreadPoolExecutor(12) as pool:
                for r, s in zip(todo, pool.map(self._score, todo)):
                    if s:
                        self._games[r.game_id] = s
        return res

    def dump(self) -> dict:
        return {"season": self.svc.schedule.season, "games": self._games}

    def report(self) -> dict:
        with self._lock:
            if self._report is None or time.time() - self._at > self.TTL:
                self._report = self._build(self.update())
                self._at = time.time()
            return self._report

    def _build(self, res: pd.DataFrame) -> dict:
        if not len(res):
            return {"season": self.svc.schedule.season, "games": 0}
        spreads = [self.svc.ratings.spread_for_game(r.game_id, r.home_id, r.away_id, r.season)[0] for r in res.itertuples()]
        pre = self.svc.predictor.score_states([pregame_state(s) for s in spreads])

        rows = []
        for r, spread, p in zip(res.itertuples(), spreads, pre):
            s = self._games.get(r.game_id, {})
            hw = bool(r.home_pts > r.away_pts)
            rows.append({
                "game_id": r.game_id, "date": f"{r.date:%Y-%m-%d}",
                "home": {"id": int(r.home_id), "tricode": r.home, "score": int(r.home_pts)},
                "away": {"id": int(r.away_id), "tricode": r.away, "score": int(r.away_pts)},
                "pregame": round(p, 4), "pre_spread": round(spread, 1), "home_won": hw,
                "hit": 1.0 if (p > 0.5) == hw else 0.5 if p == 0.5 else 0.0,
                "tracked": bool(s), "winner_min": s.get("winner_min"), "excitement": s.get("excitement"),
                "lead_changes": s.get("lead_changes"), "at": s.get("at"),
            })
        n = len(rows)
        y = [1.0 if g["home_won"] else 0.0 for g in rows]
        ps = [g["pregame"] for g in rows]
        clip = lambda v: min(max(v, 1e-6), 1 - 1e-6)  # noqa: E731
        pregame = {
            "games": n, "correct": sum(g["hit"] for g in rows),
            "accuracy": sum(g["hit"] for g in rows) / n,
            "home_team_accuracy": sum(y) / n,  # baseline: always pick the home team
            "log_loss": -sum(math.log(clip(p if t else 1 - p)) for p, t in zip(ps, y)) / n,
            "brier": sum((p - t) ** 2 for p, t in zip(ps, y)) / n,
        }

        tracked = [g for g in rows if g["tracked"]]
        calls = []
        for key, label, _ in CHECKPOINTS:
            got = [(g["at"][key], g["home_won"]) for g in tracked if g["at"] and g["at"].get(key) is not None]
            if got:
                hits = sum(1.0 if (p > 0.5) == hw else 0.5 if p == 0.5 else 0.0 for p, hw in got)
                calls.append({"key": key, "label": label, "games": len(got), "accuracy": hits / len(got)})

        st = [self._games[g["game_id"]]["states"] for g in tracked]
        ns = sum(s["n"] for s in st) or 1
        calib = [[0.0, 0.0, 0] for _ in range(CALIB_BINS)]
        for s in st:
            for b, (a, w, k) in zip(calib, s["calib"]):
                b[0] += a
                b[1] += w
                b[2] += k
        states = {
            "n": ns, "log_loss": sum(s["log_loss"] for s in st) / ns, "brier": sum(s["brier"] for s in st) / ns,
            "accuracy": sum(s["hits"] for s in st) / ns,
            "calibration": [{"pred": a / k, "actual": w / k, "n": k} for a, w, k in calib if k],
        }

        by_date, cum_hit, cum_home, k = [], 0.0, 0.0, 0
        for d, grp in pd.DataFrame(rows).groupby("date", sort=True):
            cum_hit += grp.hit.sum()
            cum_home += grp.home_won.sum()
            k += len(grp)
            by_date.append({"date": d, "games": k, "accuracy": cum_hit / k, "home_team": cum_home / k,
                            "day_correct": float(grp.hit.sum()), "day_games": int(len(grp))})

        def top(key, rev=False, need=None, k=5):
            pool = [g for g in rows if need is None or g.get(need) is not None]
            return sorted(pool, key=key, reverse=rev)[:k]

        winner_pre = lambda g: g["pregame"] if g["home_won"] else 1 - g["pregame"]  # noqa: E731
        return {
            "season": self.svc.schedule.season,
            "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "games": n, "tracked": len(tracked),
            "avg_swing": sum(g["excitement"] for g in tracked) / len(tracked) if tracked else None,
            "pregame": pregame, "calls": calls, "states": states, "by_date": by_date,
            "upsets": top(winner_pre, need=None),
            "comebacks": top(lambda g: g["winner_min"], need="winner_min"),
            "thrillers": top(lambda g: g["excitement"], rev=True, need="excitement"),
            "recent": list(reversed(rows[-60:])),
        }
