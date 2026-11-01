# Courtside: Live NBA Win Probability

A neural network that estimates, after every play, the probability that each
team wins an NBA game, served on a real-time dashboard over WebSockets.

It's trained on play-by-play from 2019-20 onward (8,600+ games, 3.45M game states) from the
NBA's own data feeds, and it works for any game: live games stream as they
happen, finished games can be replayed play by play, and upcoming games
(including future seasons) get a pre-game estimate.

**Stack:** `nba_api` · `pandas` · `PyTorch` · `Flask` · WebSockets (`flask-sock`)

**Live demo:** https://nba-winprob.vercel.app

### The dashboard

- **Games:** pick any date. Each game shows the win-probability chart with the
  biggest swings marked, a one-line summary, a "why this number" breakdown
  (from 50/50, how much the score and clock, possession, timeouts, home court
  and team strength each move the estimate), game-flow stats, and how the model
  called the game at tip-off, halftime, end of Q3 and in the final minutes.
- **Track record:** every finished game of the current season, scored after the
  fact. These are games the model never trained or tuned on. It shows the pre-game
  pick record against always picking the home team, accuracy at each stage, live
  calibration, upsets, comebacks, the most dramatic games, and current Elo
  ratings for all 30 teams.
- **The model:** the pipeline and network architecture, the 10 inputs and
  which ones matter (permutation importance), a what-if calculator with a
  lead-by-minute heatmap, what the network learned (value of possession, how
  the favourite's edge fades), and the full test-season evaluation: calibration,
  error by minute against the baselines, confidence buckets, comebacks, and
  results for every season.

## Results

Scored on the **2025-26 season**, which the model never saw during training or tuning
(1,315 games, 532k game states):

| model | log loss | Brier | accuracy | AUC |
|---|---|---|---|---|
| **Neural net (this project)** | **0.4408** | **0.1470** | **77.9%** | **0.868** |
| Logistic regression, same features | 0.4412 | 0.1470 | 78.0% | 0.868 |
| Pre-game Elo only | 0.6019 | 0.2072 | 68.1% | 0.730 |

- **Calibrated:** when the model says 70%, that team wins about 70% of the time.
  The average gap between stated and actual win rates is 1.1 percentage points.
- **Brier skill score 40%:** 40% less squared error than always predicting the
  home team's 55% base rate.
- **Accuracy climbs through the game:** 69% in Q1, 80% in Q3, 89% in Q4.
- **Crunch time is where the neural network earns its keep.** Over a whole
  game it ties a well-engineered logistic regression, because features like
  lead ÷ √time already make most game states close to linear. But in close
  late-game situations, where possession, timeouts and intentional fouling
  interact, the network is clearly better:

| situation | NN log loss | LR log loss | improvement |
|---|---|---|---|
| last 5 min, within 10 pts | 0.3531 | 0.3570 | 1.1% |
| last 2 min, within 6 pts | 0.3410 | 0.3479 | 2.0% |
| last 30 s, within 3 pts | 0.3861 | 0.4058 | **4.9%** |

- **Home court has shrunk.** FiveThirtyEight's Elo used a 100-point home edge;
  fitting it to 2021-25 results gives 45. That's consistent with home teams
  winning ~55% of games since 2020, down from ~58.5% before.

---

## How it works

```
 nba_api ──► game results (2014→now) ──► Elo team ratings ─┐
         └─► play-by-play (2019→now) ──► game states ──────┴─► features ─► MLP ─► P(home win)
                                                                                  │
 live CDN ──► poll every 5s ──► same feature code ──► same model ──► WebSocket ──► dashboard
```

### 1. Data (`scripts/fetch_games.py`, `scripts/fetch_pbp.py`)
- **Game results** from `LeagueGameFinder` for every regular-season, play-in
  and playoff game since 2014-15 (~15k games). These drive the Elo ratings.
- **Play-by-play** for every game since 2019-20 (~8.7k games, ~550 events each),
  in the format of the NBA live-data feed (`nba_api.live`). That's the same feed
  the live dashboard reads, so training and serving see identical data. The
  CDN rate-limits bulk downloads, so the bulk pull uses the NBA's S3 mirror
  of the same JSON (verified byte-identical), with the throttled CDN as a
  fallback. Downloads are threaded, cached per game, and resumable.

### 2. Team strength prior (`winprob/elo.py`)
A FiveThirtyEight-style Elo system: K = 20, margin-of-victory multiplier, and
25% regression to the mean between seasons. Home-court edge is refit to 45
Elo points (538 used 100), which matches the post-2020 decline in home advantage.
The 2020 Orlando "bubble" games are treated as neutral-site. Because ratings
carry over from season to season, the model has a sensible prior on opening
night of any future season. During the season the server re-fetches results
every few hours, so ratings stay current.

### 3. Features (`winprob/features.py`)
One function turns a play-by-play log into model inputs. Both the training
pipeline and the live server call it, so there's no train/serve skew.

| feature | why |
|---|---|
| score margin (home − away) | the main signal |
| minutes left, `log(1 + minutes)` | resolution near the buzzer |
| margin / √(minutes + 1) | a lead's strength scales with remaining variance, roughly a z-score |
| possession (+1 / −1) | worth ~1 point, decisive late |
| timeouts left (each team) | tracked per NBA rules: 7 per game, max 2 in the last 3:00, 2 per OT |
| Elo pre-game spread | who's better, in points |
| spread × share of game left | the prior should fade as the game plays out |
| overtime flag | OT is a 5-minute coin-flip with its own dynamics |

### 4. Model (`winprob/model.py`, `scripts/train.py`)
- MLP `10 → 128 → 64 → 32 → 1`, SiLU activations, sigmoid output, binary
  cross-entropy loss, AdamW with a cosine learning-rate schedule, early stopping.
- **Split by season, never by row**, so no game leaks between sets. The test set is
  the most recent completed season, which the model never sees during training
  or tuning.
- Baselines: logistic regression on the same features, and pre-game Elo alone.
- Evaluation: log loss, Brier score, accuracy, AUC, per-quarter breakdown, and
  a calibration curve (when it says 70%, does the team win 70% of the time?).
  All of it shows on the dashboard's **Model** tab.

### 5. Real-time dashboard (`app/`)
- **Flask** serves the page; a **WebSocket** (`/ws`) pushes updates.
- One `GameTracker` per game polls the NBA live feed every 5 seconds, no matter
  how many browsers are watching. It re-scores the timeline and pushes only the
  new plays to each client. If the NBA revises earlier plays (score corrections
  happen), it detects that and resends the full timeline.
- **Replay mode** streams a finished game through the same socket at 30–300×
  speed, so you can demo it when no games are on.
- **Serverless fallback.** Vercel's functions can't hold a WebSocket open, so
  there the page polls `/api/board` and `/api/game` every 5 seconds. A small
  client-side shim emits the exact messages the socket would, so the rest of
  the UI doesn't know the difference. Responses carry `s-maxage` headers, so
  Vercel's CDN lets all viewers share one poll of the NBA's servers.
- **No PyTorch at serving time.** Training saves the weights as NumPy arrays
  too, and `winprob/inference.py` runs the forward pass (a few matrix
  multiplies) in NumPy. Outputs match PyTorch to 1e-7, and the test-set
  metrics are computed with this exact code path.
- **Cloud-safe data sources.** stats.nba.com blocks most cloud hosts, so the
  server reads this season's schedule and results from the NBA's CDN, and past
  seasons from a bundled results table (`data/games.csv.gz`).
- Any date works: past dates show results, today shows live games, future dates
  show the schedule with pre-game win probabilities.

## Running it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-train.txt   # requirements.txt alone is enough for the dashboard

# one-time data + training (~20 min, mostly downloading)
python scripts/fetch_games.py
python scripts/fetch_pbp.py
python scripts/build_dataset.py
python scripts/train.py
python scripts/evaluate.py       # deeper test-season evaluation for the Model tab
python scripts/season_report.py  # score this season's games for the Track record tab

# dashboard
python app/server.py        # → http://localhost:5050
```

Each new season, run `python scripts/update.py` to pull the latest data and
retrain (the train/test split moves forward automatically). Between retrains,
live team ratings update on their own.

### Deploying (Vercel)

```bash
vercel deploy --prod
```

Vercel detects the Flask app in `app/server.py` and installs only
`requirements.txt` (no PyTorch). `.vercelignore` leaves out the raw
play-by-play and the training dataset. The app ships with the trained weights
(`models/winprob.npz`), the metrics, and the game results it needs for Elo.
The project is linked to this GitHub repo, so every push to `main` redeploys.

A GitHub Action (`.github/workflows/refresh-season.yml`) runs
`scripts/season_report.py` every morning and commits newly scored games, which
keeps the Track record tab current. The server also scores any games newer
than that cache on demand.

## Project layout

```
winprob/          core library
  config.py       paths, season helpers
  nba.py          nba_api wrappers with retries
  elo.py          team ratings
  features.py     play-by-play → feature rows (shared by training and serving)
  model.py        PyTorch network (training) + weight export
  inference.py    NumPy forward pass used by the server
  summary.py      game-level stats, checkpoint calls, "why this number"
scripts/          data pipeline, training, evaluation, season report
app/
  server.py       Flask routes + WebSocket protocol
  live.py         schedule, ratings, per-game trackers, season tracker, what-if
  templates/      the page
  static/         vanilla JS + hand-built SVG charts
    app.js        navigation, live transport, games view
    record.js     track record view
    model.js      model view and what-if calculator
    charts.js     line, column, heatmap, bar and waterfall charts
```

## Limitations and next steps
- The CDN has no play-by-play for ~260 games from 2019-20; those are skipped.
- On Vercel, updates arrive by 5-second polling rather than WebSocket push
  (run locally for the WebSocket version).
- Player-level information (injuries, who's on the floor) isn't modelled.
  Lineup-aware ratings would be the biggest upgrade.
- Pre-game strength is Elo only; blending in betting lines or
  rest/travel data would sharpen early-game estimates.
