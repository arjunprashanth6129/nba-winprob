"""Game-level numbers derived from a scored timeline.

Shared by the live game view, the season tracker and the evaluation script, so
"accuracy at halftime" means the same thing everywhere.
"""
import math

from .features import feature_row

# Game-clock checkpoints (seconds elapsed since tip-off) at which a game is "called".
CHECKPOINTS = [
    ("tipoff", "Tip-off", 0),
    ("q1", "End of Q1", 720),
    ("half", "Halftime", 1440),
    ("q3", "End of Q3", 2160),
    ("m5", "5:00 left", 2580),
    ("m1", "1:00 left", 2820),
]
CALIB_BINS = 10


def _clip(p):
    return min(max(p, 1e-6), 1 - 1e-6)


def summarize(states: list[dict], probs: list[float]) -> dict:
    """Swings, leads and how well the model called this game (once it's over)."""
    if not states:
        return {}
    last = states[-1]
    done = bool(last.get("final"))
    home_won = last["home_score"] > last["away_score"]

    lead_changes = ties = 0
    prev_sign = 0
    big_home = big_away = 0
    for s in states[1:]:
        d = s["home_score"] - s["away_score"]
        big_home, big_away = max(big_home, d), max(big_away, -d)
        sign = (d > 0) - (d < 0)
        if sign == 0 and prev_sign != 0:
            ties += 1
        if sign != 0:
            if prev_sign != 0 and sign != prev_sign:
                lead_changes += 1
            prev_sign = sign

    live = [p for s, p in zip(states, probs) if not s.get("final")]
    excitement = sum(abs(b - a) for a, b in zip(probs, probs[1:]))
    out = {
        "final": done,
        "pregame": round(probs[0], 4),
        "lead_changes": lead_changes,
        "ties": ties,
        "largest_lead": {"home": big_home, "away": big_away},
        "excitement": round(excitement, 3),  # total win-probability movement, in "games"
        "home_min": round(min(live), 4) if live else None,
        "home_max": round(max(live), 4) if live else None,
        # Share of the game clock each side spent as the model's favourite.
        "home_fav_share": round(_fav_share(states, probs), 3),
    }
    if not done:
        return out

    out["home_won"] = home_won
    y = 1.0 if home_won else 0.0
    out["winner_min"] = round(min(p if home_won else 1 - p for p in live), 4) if live else None
    at = {}
    for key, _, t in CHECKPOINTS:
        p = None
        for s, q in zip(states, probs):
            if s["elapsed"] > t or s.get("final"):
                break
            p = q
        at[key] = round(p, 4) if p is not None else None
    out["at"] = at
    # Per-state scoring, aggregated so seasons can be summed game by game.
    ll = br = hits = 0.0
    calib = [[0.0, 0.0, 0] for _ in range(CALIB_BINS)]
    for p in live:
        ll -= math.log(_clip(p if home_won else 1 - p))
        br += (p - y) ** 2
        hits += 1.0 if (p > 0.5) == home_won else 0.5 if p == 0.5 else 0.0
        b = calib[min(int(p * CALIB_BINS), CALIB_BINS - 1)]
        b[0] += p
        b[1] += y
        b[2] += 1
    out["states"] = {"n": len(live), "log_loss": round(ll, 4), "brier": round(br, 4), "hits": hits,
                     "calib": [[round(a, 4), b, n] for a, b, n in calib]}
    return out


def _fav_share(states, probs):
    total = home = 0.0
    for (s, p), nxt in zip(zip(states, probs), states[1:]):
        dt = max(0.0, nxt["elapsed"] - s["elapsed"])
        total += dt
        if p > 0.5:
            home += dt
        elif p == 0.5:
            home += dt / 2
    return home / total if total else 0.5


def explain(x: list[float], predictor, home_court: float) -> list[dict]:
    """Break a state's probability into steps that add up to it.

    Starts from a coin flip and switches the model's inputs on one group at a
    time: score and clock, possession, timeouts, home court, team strength.
    Each step's size is how much that group moved the probability given the
    ones before it (the order matters a little; this is the natural one).
    `home_court` is the part of the pre-game spread (points) due to home court.
    """
    diff, t_sec, is_ot, poss, spread, hto, ato = x[0], x[1] * 60, bool(x[4]), x[5], x[6], x[8], x[9]
    even_to = (hto + ato) / 2
    steps = [
        ("score_clock", "Score & clock", dict(poss=0, spread=0.0, home_to=even_to, away_to=even_to)),
        ("possession", "Possession", dict(poss=poss, spread=0.0, home_to=even_to, away_to=even_to)),
        ("timeouts", "Timeouts left", dict(poss=poss, spread=0.0, home_to=hto, away_to=ato)),
        ("home_court", "Home court", dict(poss=poss, spread=home_court, home_to=hto, away_to=ato)),
        ("strength", "Team strength (Elo)", dict(poss=poss, spread=spread, home_to=hto, away_to=ato)),
    ]
    rows = [feature_row(diff, t_sec, is_ot, **kw) for _, _, kw in steps]
    ps = [float(p) for p in predictor.predict(rows)]
    out, prev = [], 0.5
    for (key, label, _), p in zip(steps, ps):
        out.append({"key": key, "label": label, "delta": round(p - prev, 4), "p": round(p, 4)})
        prev = p
    return out
