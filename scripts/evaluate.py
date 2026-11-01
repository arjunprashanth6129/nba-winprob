"""Step 5: deeper evaluation of the trained model, for the dashboard's Model page.

    python scripts/evaluate.py

Reads the saved weights (no retraining) and the dataset, scores the held-out
test season with the network, the logistic-regression baseline and pre-game
Elo, and writes models/evaluation.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from winprob.config import DATASET_PATH, EVAL_PATH, METRICS_PATH, WEIGHTS_PATH, load_games, season_str  # noqa: E402
from winprob.elo import ELO_PER_POINT, HOME_ADV  # noqa: E402
from winprob.features import FEATURES, feature_row  # noqa: E402
from winprob.inference import Predictor  # noqa: E402
from winprob.summary import CHECKPOINTS  # noqa: E402

MODELS_ = [("neural_net", "p_nn"), ("logistic_regression", "p_lr"), ("elo_pregame_only", "p_elo")]


def ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def acc(y, p):
    return float(np.mean(np.where(p == 0.5, 0.5, (p > 0.5) == y)))


def with_raw(df, **raw):
    """Feature matrix after replacing raw inputs, recomputing the derived features."""
    d = {k: df[k].to_numpy(np.float64) for k in ["diff", "t_min", "is_ot", "poss", "pre_spread", "home_to", "away_to"]}
    d.update(raw)
    t = d["t_min"]
    share = np.where(d["is_ot"] > 0, 0.0, t * 60 / 2880)
    cols = {
        "diff": d["diff"], "t_min": t, "log_t": np.log1p(t), "diff_over_sqrt_t": d["diff"] / np.sqrt(t + 1),
        "is_ot": d["is_ot"], "poss": d["poss"], "pre_spread": d["pre_spread"],
        "spread_left": d["pre_spread"] * share, "home_to": d["home_to"], "away_to": d["away_to"],
    }
    return np.column_stack([cols[f] for f in FEATURES]).astype(np.float32)


def main():
    metrics = json.loads(METRICS_PATH.read_text())
    test = int(metrics["test_season"][:4])
    val = int(metrics["val_season"][:4])
    pred = Predictor(WEIGHTS_PATH)

    df = pd.read_parquet(DATASET_PATH)
    tr = df[df.season < min(val, test)]
    te = df[df.season == test]
    X = lambda d: d[FEATURES].to_numpy(np.float32)  # noqa: E731

    # Same baseline as train.py: logistic regression on standardised features.
    lr = LogisticRegression(max_iter=500)
    sub = tr.sample(min(len(tr), 1_500_000), random_state=0)
    lr.fit((X(sub) - pred.mean) / pred.std, sub.y)

    ev = te[~te.final].copy()
    ev["p_nn"] = pred.predict(X(ev))
    ev["p_lr"] = lr.predict_proba((X(ev) - pred.mean) / pred.std)[:, 1]
    ev["p_elo"] = 1 / (1 + 10 ** (-ev.pre_spread * ELO_PER_POINT / 400))
    y = ev.y.to_numpy().astype(float)

    # ---------- skill vs. naive forecasts ----------
    base_rate = float(tr.groupby("game_id").y.first().mean())  # home win rate in training
    br_nn = brier(y, ev.p_nn.to_numpy())
    skill = {
        "base_rate": round(base_rate, 4),
        "brier_coin_flip": 0.25,
        "brier_home_rate": round(brier(y, np.full(len(y), base_rate)), 4),
        "brier_skill_vs_home_rate": round(1 - br_nn / brier(y, np.full(len(y), base_rate)), 4),
        "brier_skill_vs_elo": round(1 - br_nn / brier(y, ev.p_elo.to_numpy()), 4),
    }

    # ---------- calibration ----------
    p = ev.p_nn.to_numpy()
    edges = np.linspace(0, 1, 21)
    b = np.clip(np.digitize(p, edges) - 1, 0, 19)
    calib = []
    ece = 0.0
    for k in range(20):
        m = b == k
        if not m.any():
            continue
        calib.append({"lo": round(edges[k], 2), "hi": round(edges[k + 1], 2), "pred": round(float(p[m].mean()), 4),
                      "actual": round(float(y[m].mean()), 4), "n": int(m.sum())})
        ece += m.mean() * abs(p[m].mean() - y[m].mean())

    # ---------- how often the favourite wins, by confidence ----------
    conf = np.maximum(p, 1 - p)
    fav_won = np.where(p >= 0.5, y, 1 - y)
    confidence = []
    for lo, hi in [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 0.99), (0.99, 1.0001)]:
        m = (conf >= lo) & (conf < hi)
        confidence.append({"lo": lo, "hi": min(hi, 1.0), "share": round(float(m.mean()), 4),
                           "said": round(float(conf[m].mean()), 4), "won": round(float(fav_won[m].mean()), 4),
                           "n": int(m.sum())})

    # ---------- error through the game, all three models ----------
    minute = np.where(ev.is_ot > 0, 48, np.minimum(ev.elapsed // 60, 47)).astype(int)
    by_minute = []
    for mnt in range(49):
        m = minute == mnt
        if not m.any():
            continue
        row = {"minute": mnt, "n": int(m.sum())}
        for name, col in MODELS_:
            row[name] = round(ll(y[m], ev[col].to_numpy()[m]), 4)
        by_minute.append(row)

    # ---------- which inputs matter: permutation importance ----------
    # Shuffle one raw input across the test season (re-deriving the features
    # built from it) and measure how much worse the log loss gets.
    groups = {
        "score_margin": ["diff"], "time_left": ["t_min"], "possession": ["poss"],
        "elo_spread": ["pre_spread"], "timeouts": ["home_to", "away_to"], "overtime": ["is_ot"],
    }

    def importance_on(rows, n=400_000):
        rng = np.random.default_rng(0)
        samp = rows.sample(min(len(rows), n), random_state=0)
        ys = samp.y.to_numpy().astype(float)
        base = ll(ys, pred.predict(with_raw(samp)))
        out = []
        for g, cols in groups.items():
            hits = []
            for _ in range(3):
                idx = rng.permutation(len(samp))
                raw = {c: samp[c].to_numpy(np.float64)[idx] for c in cols}
                hits.append(ll(ys, pred.predict(with_raw(samp, **raw))) - base)
            out.append({"group": g, "delta_log_loss": round(float(np.mean(hits)), 5)})
        return sorted(out, key=lambda r: -r["delta_log_loss"])

    importance = importance_on(ev)
    crunch = ev[((ev.is_ot > 0) | (ev.elapsed >= 2160)) & (ev.t_min <= 2) & (ev["diff"].abs() <= 6)]
    importance_crunch = importance_on(crunch)

    # ---------- game-level calls ----------
    games = te.groupby("game_id").agg(y=("y", "first")).reset_index()
    calls = []
    for key, label, t in CHECKPOINTS:
        snap = ev[ev.elapsed <= t].groupby("game_id").tail(1).set_index("game_id")
        row = {"key": key, "label": label, "games": int(len(snap))}
        for name, col in MODELS_:
            pp, yy = snap[col].to_numpy(), snap.y.to_numpy().astype(float)
            row[name] = {"accuracy": round(acc(yy, pp), 4), "log_loss": round(ll(yy, pp), 4)}
        calls.append(row)

    # Comebacks: how low did the eventual winner's win probability get?
    winner_p = np.where(ev.y == 1, ev.p_nn, 1 - ev.p_nn)
    wmin = pd.Series(winner_p, index=ev.index).groupby(ev.game_id).min()
    comebacks = [{"below": thr, "games": int((wmin < thr).sum()), "share": round(float((wmin < thr).mean()), 4)}
                 for thr in [0.5, 0.3, 0.2, 0.1, 0.05]]

    # ---------- stability across seasons ----------
    seasons = []
    for s, g in df[~df.final].groupby("season"):
        pp = pred.predict(X(g))
        yy = g.y.to_numpy().astype(float)
        role = "test" if s == test else "validation" if s == val else "train" if s < min(val, test) else "live"
        seasons.append({"season": season_str(int(s)), "role": role, "games": int(g.game_id.nunique()),
                        "log_loss": round(ll(yy, pp), 4), "brier": round(brier(yy, pp), 4), "accuracy": round(acc(yy, pp), 4)})

    # ---------- what the network learned ----------
    def P(rows):
        return [round(float(v), 4) for v in pred.predict(np.asarray(rows, np.float32))]

    # Typical timeouts left at each minute of regulation, so the curves sit on realistic states.
    reg = ev[ev.is_ot == 0]
    typical = reg.groupby(np.minimum(reg.elapsed // 60, 47).astype(int)).home_to.median().to_dict()

    def tos(t):
        return float(typical[min(47, int((2880 - t) // 60))])

    q4 = list(range(720, -1, -15))
    possession_value = {"t_sec": q4, "pp": [round(a - b, 4) for a, b in zip(
        P([feature_row(0, t, False, 1, 0.0, tos(t), tos(t)) for t in q4]),
        P([feature_row(0, t, False, -1, 0.0, tos(t), tos(t)) for t in q4]))]}
    game = list(range(2880, -1, -60))
    prior_fade = {"t_sec": game, "curves": {
        f"{s:+d}": P([feature_row(0, t, False, 0, float(s), tos(t), tos(t)) for t in game])
        for s in [3, 6, 9]}}

    # Home-court edge over time (game level), the reason the Elo home edge was refit.
    allg = load_games()
    home_by_season = [{"season": season_str(int(s)), "home_win": round(float((g.home_pts > g.away_pts).mean()), 4), "games": int(len(g))}
                      for s, g in allg.groupby("season") if len(g) > 200]

    n_params = sum(w.size + b_.size for w, b_ in pred.layers)
    out = {
        "test_season": metrics["test_season"],
        "n_params": int(n_params),
        "states_scored": int(len(ev)),
        "games_scored": int(len(games)),
        "skill": skill,
        "ece": round(float(ece), 4),
        "calibration_fine": calib,
        "confidence": confidence,
        "by_minute": by_minute,
        "importance": importance,
        "importance_crunch": importance_crunch,
        "calls": calls,
        "comebacks": comebacks,
        "seasons": seasons,
        "possession_value": possession_value,
        "prior_fade": prior_fade,
        "home_by_season": home_by_season,
        "elo": {"home_adv": HOME_ADV, "elo_per_point": ELO_PER_POINT},
        "typical_timeouts": [typical[m] for m in range(48)],  # by minute elapsed, regulation
    }
    EVAL_PATH.write_text(json.dumps(out, indent=1))
    print(f"ECE {ece:.4f}  Brier skill vs home-rate {skill['brier_skill_vs_home_rate']:.3f}  params {n_params:,}")
    for c in calls:
        print(f"  {c['label']:<10} NN {c['neural_net']['accuracy']:.3f}  LR {c['logistic_regression']['accuracy']:.3f}  Elo {c['elo_pregame_only']['accuracy']:.3f}")
    print("importance:", ", ".join(f"{r['group']} {r['delta_log_loss']:.4f}" for r in importance))
    print("crunch:    ", ", ".join(f"{r['group']} {r['delta_log_loss']:.4f}" for r in importance_crunch))
    print(f"-> {EVAL_PATH}")


if __name__ == "__main__":
    main()
