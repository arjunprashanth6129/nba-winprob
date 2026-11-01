"""Step 4: train the win-probability network and evaluate it on a held-out season.

Split is by season so no game leaks across sets, and rolls forward each year:
    test = last completed season, val = the one before, train = everything earlier.
    (As of the 2026-27 season: train 2019-20..2023-24, val 2024-25, test 2025-26.)

    python scripts/train.py [--epochs 20]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from winprob.config import DATASET_PATH, METRICS_PATH, MODEL_PATH, MODELS, WEIGHTS_PATH, current_season, season_str  # noqa: E402
from winprob.elo import ELO_PER_POINT  # noqa: E402
from winprob.features import FEATURES, feature_row  # noqa: E402
from winprob.inference import Predictor  # noqa: E402
from winprob.model import WinProbNet, export_weights  # noqa: E402

TEST_SEASON = current_season() - 1
VAL_SEASON = TEST_SEASON - 1
HIDDEN = (128, 64, 32)


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def scores(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "accuracy": float(((p > 0.5) == y).mean()),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
        "n": int(len(y)),
    }


def train(Xtr, ytr, Xva, yva, epochs, dev):
    mean, std = Xtr.mean(0), Xtr.std(0) + 1e-6
    norm = lambda a: torch.from_numpy((a - mean) / std).float()  # noqa: E731
    xt, yt = norm(Xtr).to(dev), torch.from_numpy(ytr).float().to(dev)
    xv, yv = norm(Xva).to(dev), torch.from_numpy(yva).float().to(dev)

    torch.manual_seed(0)
    net = WinProbNet(len(FEATURES), HIDDEN).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.BCEWithLogitsLoss()
    bs = 8192
    best, best_state, patience = float("inf"), None, 0
    history = []
    for ep in range(epochs):
        t0 = time.time()
        net.train()
        perm = torch.randperm(len(xt), device=dev)
        total = 0.0
        for i in range(0, len(xt), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(net(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step()
        net.eval()
        with torch.no_grad():
            vl = sum(lossf(net(xv[i:i + 65536]), yv[i:i + 65536]).item() * len(yv[i:i + 65536])
                     for i in range(0, len(xv), 65536)) / len(xv)
        tl = total / len(xt)
        history.append({"epoch": ep + 1, "train_loss": tl, "val_loss": vl})
        print(f"  epoch {ep + 1:>2}  train {tl:.4f}  val {vl:.4f}  ({time.time() - t0:.1f}s)", flush=True)
        if vl < best - 1e-5:
            best, patience = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            patience += 1
            if patience >= 4:
                print("  early stop")
                break
    return best_state, mean, std, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--val", type=int, default=VAL_SEASON, help="validation season start year")
    ap.add_argument("--test", type=int, default=TEST_SEASON, help="test season start year")
    args = ap.parse_args()
    val_season, test_season = args.val, args.test
    dev = device()
    print(f"device: {dev}")

    df = pd.read_parquet(DATASET_PATH)
    tr = df[df.season < min(val_season, test_season)]
    va = df[df.season == val_season]
    te = df[df.season == test_season]
    print(f"rows  train {len(tr):,} ({tr.game_id.nunique()} games)  "
          f"val {len(va):,}  test {len(te):,} ({te.game_id.nunique()} games)")

    X = lambda d: d[FEATURES].to_numpy(np.float32)  # noqa: E731
    state, mean, std, history = train(X(tr), tr.y.to_numpy().copy(), X(va), va.y.to_numpy().copy(), args.epochs, dev)

    MODELS.mkdir(exist_ok=True)
    checkpoint = {
        "state_dict": state, "mean": mean.tolist(), "std": std.tolist(),
        "features": FEATURES, "hidden": list(HIDDEN),
        "train_seasons": sorted(int(s) for s in tr.season.unique()),
    }
    torch.save(checkpoint, MODEL_PATH)
    export_weights(checkpoint, WEIGHTS_PATH)
    print(f"saved -> {MODEL_PATH}, {WEIGHTS_PATH}")

    # ---------- evaluation on the untouched test season ----------
    pred = Predictor(WEIGHTS_PATH)  # evaluate exactly what the server will run
    ev = te[~te.final].copy()  # final buzzer rows are trivially 0/1
    ev["p_nn"] = pred.predict(X(ev))

    lr = LogisticRegression(max_iter=500)
    sub = tr.sample(min(len(tr), 1_500_000), random_state=0)
    lr.fit((X(sub) - mean) / std, sub.y)
    ev["p_lr"] = lr.predict_proba((X(ev) - mean) / std)[:, 1]
    ev["p_elo"] = 1 / (1 + 10 ** (-ev.pre_spread * ELO_PER_POINT / 400))  # pre-game only

    y = ev.y.to_numpy()
    overall = {name: scores(y, ev[col].to_numpy())
               for name, col in [("neural_net", "p_nn"), ("logistic_regression", "p_lr"), ("elo_pregame_only", "p_elo")]}

    ev["phase"] = np.select(
        [ev.is_ot == 1, ev.elapsed < 720, ev.elapsed < 1440, ev.elapsed < 2160],
        ["OT", "Q1", "Q2", "Q3"], "Q4")
    by_phase = {ph: {m: scores(g.y.to_numpy(), g[c].to_numpy()) for m, c in [("neural_net", "p_nn"), ("logistic_regression", "p_lr")]}
                for ph, g in ev.groupby("phase")}

    # Crunch time is where a nonlinear model should earn its keep: possession,
    # timeouts and intentional fouling interact in ways a linear logit can't express.
    late = (ev.is_ot == 1) | (ev.elapsed >= 2160)
    margin = ev["diff"].abs()
    clutch = []
    for label, mins, lead in [("Last 5 min, within 10", 5, 10), ("Last 2 min, within 6", 2, 6), ("Last 30 s, within 3", 0.5, 3)]:
        g = ev[late & (ev.t_min <= mins) & (margin <= lead)]
        clutch.append({"slice": label, **{m: scores(g.y.to_numpy(), g[c].to_numpy())
                                          for m, c in [("neural_net", "p_nn"), ("logistic_regression", "p_lr")]}})

    minute = np.minimum((ev.elapsed // 60).astype(int), 47)
    acc_by_min = (ev.assign(m=minute, hit=((ev.p_nn > 0.5) == ev.y)).groupby("m")["hit"].mean())

    bins = np.clip((ev.p_nn * 10).astype(int), 0, 9)
    calib = (ev.assign(b=bins).groupby("b").agg(pred=("p_nn", "mean"), actual=("y", "mean"), n=("y", "size")).reset_index())

    # What the network learned: P(home win) vs. lead at several clock points,
    # evenly matched teams, home ball, all timeouts.
    curves = {}
    for label, tsec in [("12:00 left", 720), ("5:00 left", 300), ("1:00 left", 60), ("0:10 left", 10)]:
        leads = list(range(-15, 16))
        rows = [feature_row(d, tsec, False, 0, 0.0, 2 if tsec <= 180 else 4, 2 if tsec <= 180 else 4) for d in leads]
        curves[label] = {"lead": leads, "p": [round(float(p), 4) for p in pred.predict(np.asarray(rows, np.float32))]}

    metrics = {
        "trained_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "train_seasons": [season_str(int(s)) for s in sorted(tr.season.unique())],
        "val_season": season_str(val_season),
        "test_season": season_str(test_season),
        "rows": {"train": len(tr), "val": len(va), "test": len(te)},
        "games": {"train": int(tr.game_id.nunique()), "val": int(va.game_id.nunique()), "test": int(te.game_id.nunique())},
        "features": FEATURES,
        "architecture": f"MLP {len(FEATURES)} → " + " → ".join(map(str, HIDDEN)) + " → 1 (SiLU, sigmoid output)",
        "history": history,
        "overall": overall,
        "by_phase": by_phase,
        "clutch": clutch,
        "accuracy_by_minute": [round(float(v), 4) for v in acc_by_min.reindex(range(48)).ffill().to_numpy()],
        "calibration": calib.round(4).to_dict(orient="records"),
        "lead_curves": curves,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"\nTest season {season_str(test_season)} (excluding final-buzzer rows):")
    for name, s in overall.items():
        print(f"  {name:<22} logloss {s['log_loss']:.4f}  brier {s['brier']:.4f}  acc {s['accuracy']:.3f}  auc {s['auc']:.3f}")
    print("\nBy phase (neural net):")
    for ph in ["Q1", "Q2", "Q3", "Q4", "OT"]:
        if ph in by_phase:
            s = by_phase[ph]["neural_net"]
            print(f"  {ph:<3} logloss {s['log_loss']:.4f}  brier {s['brier']:.4f}  acc {s['accuracy']:.3f}")
    print("\nCrunch time (log loss, NN vs logistic):")
    for c in clutch:
        print(f"  {c['slice']:<24} {c['neural_net']['log_loss']:.4f} vs {c['logistic_regression']['log_loss']:.4f}  (n={c['neural_net']['n']:,})")
    print(f"metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
