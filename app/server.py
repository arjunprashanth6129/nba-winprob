"""Flask + WebSocket server for the live win-probability dashboard.

    python app/server.py            # http://localhost:5050

On hosts without WebSockets (Vercel's serverless functions), the dashboard
falls back to polling /api/board and /api/game, which return the same data.

WebSocket protocol (/ws, JSON messages)
  client -> server
    {"op": "board", "date": "YYYY-MM-DD"}             subscribe to a day's games
    {"op": "watch", "game_id": "...", "replay": bool, "speed": 60}
    {"op": "replay", "playing": bool, "speed": 60}     replay controls
  server -> client
    {"type": "board", "date", "games": [...]}          on request, then every 15s
    {"type": "game", "header", "points", "replay"}     full timeline (reset)
    {"type": "points", "header", "points"}             new plays appended
"""
import bisect
import json
import os
import sys
import threading
import time
from datetime import date
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from flask_sock import Sock

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]  # works as a script and when imported (Vercel)
from winprob.config import EVAL_PATH, METRICS_PATH  # noqa: E402
from live import Service  # noqa: E402

BOARD_PUSH_SEC = 15
# Vercel sets VERCEL=1. Its functions can't hold a socket open, so the page polls instead.
WEBSOCKET = not os.environ.get("VERCEL")

app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 300
sock = Sock(app)
svc = Service()


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/metrics")
def metrics():
    return _cached(app.response_class(METRICS_PATH.read_text(), mimetype="application/json"), 3600)


@app.get("/api/evaluation")
def evaluation():
    return _cached(app.response_class(EVAL_PATH.read_text(), mimetype="application/json"), 3600)


@app.get("/api/season")
def season():
    """The model's record on this season's finished games."""
    return _cached(jsonify(svc.season.report()), 600)


@app.get("/api/ratings")
def ratings():
    return _cached(jsonify(svc.ratings.table()), 600)


@app.get("/api/whatif")
def whatif():
    """Score a made-up game state: /api/whatif?diff=-2&t=45&poss=1&strength=0&hc=1&hto=1&ato=2&ot=0"""
    a = request.args
    num = lambda k, d, lo, hi: min(max(float(a.get(k, d)), lo), hi)  # noqa: E731
    ot = a.get("ot") == "1"
    out = svc.whatif(diff=num("diff", 0, -40, 40), t_sec=num("t", 2880, 0, 300 if ot else 2880), ot=ot,
                     poss=int(num("poss", 0, -1, 1)), strength=num("strength", 0, -20, 20), home_court=a.get("hc", "1") == "1",
                     hto=num("hto", 7, 0, 7), ato=num("ato", 7, 0, 7))
    return _cached(jsonify(out), 86400)


@app.get("/api/today")
def today():
    """The NBA's current game day (US Eastern), which can differ from the viewer's local date."""
    return _cached(jsonify({"date": svc._live_date() or date.today().isoformat(), "websocket": WEBSOCKET}), 60)


@app.get("/api/board")
def board():
    day = request.args.get("date") or date.today().isoformat()
    games = svc.scoreboard(day)
    return _cached(jsonify(games), 15 if any(g["status"] != 3 for g in games) else 600)


@app.get("/api/game/<game_id>")
def game(game_id):
    tr = svc.tracker(game_id)
    tr.refresh()
    return _cached(jsonify({"header": tr.header(), "points": tr.points()}), 3600 if tr.finished else 4)


def _cached(resp, seconds):
    """Let a CDN in front of the app (Vercel's) share responses, so viewers share one NBA poll."""
    resp.headers["Cache-Control"] = f"public, max-age=0, s-maxage={seconds}, stale-while-revalidate={seconds}"
    return resp


def _sig(state):
    return (state["elapsed"], state["home_score"], state["away_score"])


class Session:
    """State for one browser connection."""

    def __init__(self, ws):
        self.ws = ws
        self.board_date = None
        self.board_at = 0.0
        self.tr = None
        self.sent = 0
        self.last_sig = None
        self.version = -1
        self.replay = False
        self.playing = True
        self.speed = 60.0
        self.rclock = 0.0
        self.last_tick = time.time()

    def send(self, obj):
        self.ws.send(json.dumps(obj, separators=(",", ":")))

    def handle(self, msg):
        op = msg.get("op")
        if op == "board":
            self.board_date = msg.get("date") or date.today().isoformat()
            self.push_board()
        elif op == "watch":
            self.tr = svc.tracker(msg["game_id"])
            self.replay = bool(msg.get("replay"))
            self.speed = float(msg.get("speed", self.speed))
            self.playing = True
            self.rclock = 0.0
            self.version = self.tr.refresh()
            self.sent = 1 if self.replay else len(self.tr.states)
            self.last_sig = _sig(self.tr.states[self.sent - 1]) if self.tr.states else None
            self.send({"type": "game", "header": self.tr.header(), "points": self.tr.points(0, self.sent),
                       "replay": self.replay_state()})
        elif op == "replay" and self.tr:
            self.playing = bool(msg.get("playing", self.playing))
            self.speed = float(msg.get("speed", self.speed))
            if msg.get("restart"):
                self.rclock, self.sent, self.replay = 0.0, 1, True
                self.last_sig = _sig(self.tr.states[0])
                self.send({"type": "game", "header": self.tr.header(), "points": self.tr.points(0, 1),
                           "replay": self.replay_state()})

    def replay_state(self):
        return {"on": self.replay, "playing": self.playing, "speed": self.speed, "clock": self.rclock}

    def push_board(self):
        try:
            games = svc.scoreboard(self.board_date)
            self.send({"type": "board", "date": self.board_date, "games": games})
        except Exception as e:
            self.send({"type": "error", "where": "board", "message": str(e)[:200]})
        self.board_at = time.time()

    def tick(self):
        now = time.time()
        dt, self.last_tick = now - self.last_tick, now
        if self.board_date and now - self.board_at > BOARD_PUSH_SEC:
            self.push_board()
        if not self.tr:
            return
        if self.replay:
            if self.playing:
                self.rclock += dt * self.speed
            times = [s["elapsed"] for s in self.tr.states]
            n = max(1, bisect.bisect_right(times, self.rclock))
            if n >= len(times):
                self.replay = False  # caught up: the replay has become the full timeline
            if n > self.sent:
                self.send({"type": "points", "header": self.tr.header(), "points": self.tr.points(self.sent, n),
                           "replay": self.replay_state()})
                self.sent = n
                self.last_sig = _sig(self.tr.states[n - 1])
            return
        v = self.tr.refresh()
        if v == self.version:
            return
        self.version = v
        states = self.tr.states
        if 0 < self.sent <= len(states) and _sig(states[self.sent - 1]) == self.last_sig:
            self.send({"type": "points", "header": self.tr.header(), "points": self.tr.points(self.sent)})
        else:  # the NBA revised earlier plays: resend everything
            self.send({"type": "game", "header": self.tr.header(), "points": self.tr.points(),
                       "replay": self.replay_state()})
        self.sent = len(states)
        self.last_sig = _sig(states[-1]) if states else None


@sock.route("/ws")
def ws(ws):
    s = Session(ws)
    while True:
        raw = ws.receive(timeout=0.25)
        if raw:
            try:
                s.handle(json.loads(raw))
            except Exception as e:
                s.send({"type": "error", "message": str(e)[:200]})
        s.tick()


if __name__ == "__main__":
    threading.Thread(target=svc.ratings.book, daemon=True).start()  # warm Elo in the background
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5050)), threaded=True, debug=False)
