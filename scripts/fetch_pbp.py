"""Step 2: download play-by-play for every game since PBP_FIRST_SEASON.

Each game is cached as data/pbp/<season>/<game_id>.json.gz, so the script
can be stopped and resumed.

    python scripts/fetch_pbp.py [--workers 6]
"""
import argparse
import gzip
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from winprob.config import PBP_DIR, PBP_FIRST_SEASON, load_games  # noqa: E402
from winprob.nba import S3_PBP_URL  # noqa: E402
from nba_api.live.nba.endpoints import playbyplay  # noqa: E402

KEEP = ("clock", "period", "actionType", "subType", "teamId", "possession", "scoreHome", "scoreAway")


def path_for(season: int, game_id: str) -> Path:
    return PBP_DIR / str(season) / f"{game_id}.json.gz"


MISSING = PBP_DIR / "missing.txt"  # games the CDN never served, skipped on later runs


class Throttle:
    """Shared across workers. The CDN answers rate-limited requests with an empty
    body, so on a failure every worker pauses together, longer each time, instead of
    racing through games and wrongly marking them missing."""

    def __init__(self, spacing=0.8):
        self.lock = threading.Lock()
        self.spacing = spacing
        self.next_slot = 0.0
        self.paused_until = 0.0
        self.strikes = 0
        self.blocked_at = 0.0

    def wait(self):
        with self.lock:
            now = time.time()
            slot = max(now, self.next_slot, self.paused_until)
            self.next_slot = slot + self.spacing
        time.sleep(max(0.0, slot - time.time()))

    def ok(self):
        with self.lock:
            if self.strikes:
                print(f"  CDN back after {time.time() - self.blocked_at:.0f}s", flush=True)
            self.strikes = 0


    def failed(self):
        with self.lock:
            if self.strikes == 0:
                self.blocked_at = time.time()
            self.strikes += 1
            pause = min(300, 15 * 2 ** min(self.strikes - 1, 5))
            if time.time() + pause > self.paused_until:
                self.paused_until = time.time() + pause
                print(f"  CDN refused a request; all workers pausing {pause}s", flush=True)


THROTTLE = Throttle()


PROBE_GAME = "0022300500"  # a game known to exist on the CDN


def _get(game_id):
    return playbyplay.PlayByPlay(game_id, timeout=20).get_dict()["game"]["actions"]


def cdn_reachable() -> bool:
    try:
        THROTTLE.wait()
        _get(PROBE_GAME)
        return True
    except Exception:
        return False


# The NBA mirrors the live-data feed to S3. Same JSON, byte for byte, but not
# behind the CDN's rate limiter, so it's the fast path for bulk downloads.


def fetch_s3(game_id: str) -> list[dict]:
    for attempt in range(4):
        try:
            r = requests.get(S3_PBP_URL.format(game_id), timeout=30)
            if r.status_code in (403, 404):
                raise FileNotFoundError(game_id)  # not mirrored: genuinely missing
            r.raise_for_status()
            return r.json()["game"]["actions"]
        except FileNotFoundError:
            raise
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))


def fetch_one(season: int, game_id: str, source: str = "s3") -> str:
    out = path_for(season, game_id)
    if out.exists():  # another downloader got here first
        return game_id
    if source == "s3":
        raw = fetch_s3(game_id)
    while source == "cdn":
        THROTTLE.wait()
        try:
            raw = _get(game_id)
            THROTTLE.ok()
            break
        except Exception:
            # If a known-good game still loads, this one simply isn't on the CDN.
            # Otherwise we're rate-limited: wait it out however long it takes.
            if cdn_reachable():
                raise
            THROTTLE.failed()
    actions = [{k: a.get(k) for k in KEEP} for a in raw]
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    with gzip.open(tmp, "wt") as f:
        json.dump(actions, f, separators=(",", ":"))
    tmp.rename(out)
    return game_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["s3", "cdn"], default="s3")
    ap.add_argument("--workers", type=int, default=None, help="default: 12 for s3, 2 for cdn")
    ap.add_argument("--retry-missing", action="store_true")
    ap.add_argument("--reverse", action="store_true", help="newest games first (to run two sources at once)")
    args = ap.parse_args()

    games = load_games()
    games = games[games["season"] >= PBP_FIRST_SEASON]
    skip = set() if args.retry_missing or not MISSING.exists() else set(MISSING.read_text().split())
    todo = [(r.season, r.game_id) for r in games.itertuples()
            if not path_for(r.season, r.game_id).exists() and r.game_id not in skip]
    if args.reverse:
        todo.reverse()
    PBP_DIR.mkdir(parents=True, exist_ok=True)
    print(f"{len(games)} games in window, {len(todo)} to download")

    done = failed = 0
    t0 = time.time()
    workers = args.workers or (12 if args.source == "s3" else 2)
    with ThreadPoolExecutor(workers) as pool:
        futs = {pool.submit(fetch_one, s, g, args.source): g for s, g in todo}
        for fut in as_completed(futs):
            try:
                fut.result()
                done += 1
            except Exception as e:
                failed += 1
                print(f"  {futs[fut]} failed: {e!r}"[:160])
                with open(MISSING, "a") as f:
                    f.write(futs[fut] + "\n")
            if (done + failed) % 250 == 0:
                rate = (done + failed) / (time.time() - t0)
                print(f"  {done + failed}/{len(todo)}  ({rate:.1f} games/s, {failed} failed)", flush=True)
    print(f"finished: {done} downloaded, {failed} failed")


if __name__ == "__main__":
    main()
