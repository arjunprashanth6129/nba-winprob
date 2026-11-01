// Courtside dashboard: navigation, live transport, game list and the live/replay game view.
(function () {
  const { $, h, pct, tile, logo, verdict } = UI;
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode */ } },
  };

  const S = {
    ws: null, websocket: true, wsFails: 0, nbaToday: null, date: null, games: [], gameId: null, view: "games",
    header: null, points: [], replay: { on: false, playing: true, speed: 60 }, chart: null, autoPicked: false,
  };
  const CHECKPOINTS = [["Tip-off", 0], ["End of Q1", 720], ["Halftime", 1440], ["End of Q3", 2160], ["5:00 left", 2580], ["1:00 left", 2820]];

  // ------------------------------------------------------------ theme + views
  const savedTheme = store.get("theme");
  if (savedTheme) document.documentElement.dataset.theme = savedTheme;
  $("theme").onclick = () => {
    const next = document.documentElement.dataset.theme === "light" ? "dark" : "light";
    document.documentElement.dataset.theme = next;
    store.set("theme", next);
    window.dispatchEvent(new Event("resize"));
    if (S.chart) S.chart.redraw();
    document.dispatchEvent(new Event("themechange"));
  };

  function showView(view) {
    S.view = view;
    document.querySelectorAll(".tab").forEach((x) => { const on = x.dataset.view === view; x.classList.toggle("active", on); x.setAttribute("aria-selected", on); });
    for (const v of ["games", "record", "model"]) $("view-" + v).hidden = v !== view;
    if (view === "record") window.RecordView.load();
    if (view === "model") window.ModelView.load();
    if (view !== "games") window.scrollTo(0, 0);
    updateHash();
  }
  document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => showView(b.dataset.view)));

  // ------------------------------------------------------------ helpers
  function addDays(iso, n) {
    const d = new Date(iso + "T12:00:00Z");
    d.setUTCDate(d.getUTCDate() + n);
    return d.toISOString().slice(0, 10);
  }
  function localTime(utc) {
    return new Date(utc).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  }
  function statusLine(g) {
    if (g.status === 1) return localTime(g.start_utc);
    return g.status_text;
  }
  const fav = (p, home, away) => (p >= 0.5 ? home : away);
  const favPct = (p) => pct(Math.max(p, 1 - p));
  const hitOf = (p, homeWon) => (p === 0.5 ? 0.5 : (p > 0.5) === homeWon ? 1 : 0);

  // ------------------------------------------------------------ websocket
  function connect() {
    if (!S.websocket) return usePolling();
    const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
    let opened = false;
    S.ws = ws;
    setConn("Connecting", "");
    ws.onopen = () => {
      opened = true;
      setConn("Live", "ok");
      resubscribe();
    };
    ws.onclose = () => {
      if (!opened && ++S.wsFails >= 2) return usePolling();  // a proxy that blocks WebSockets
      setConn("Reconnecting", "bad");
      setTimeout(connect, 2000);
    };
    ws.onmessage = (e) => onMessage(JSON.parse(e.data));
  }
  function resubscribe() {
    if (S.date) send({ op: "board", date: S.date });
    if (S.gameId) send({ op: "watch", game_id: S.gameId, replay: false });
  }
  function usePolling() {
    S.ws = Poller();
    setConn("Live", "ok");
    $("conn").title = "Updates every 5 s over HTTP (this host doesn't support WebSockets)";
    resubscribe();
  }

  // Stands in for the WebSocket where there isn't one (Vercel): takes the same
  // ops and emits the same messages as app/server.py's Session, built from polling
  // /api/board and /api/game. Replay runs here in the browser.
  function Poller() {
    const P = { date: null, boardAt: 0, gid: null, header: null, full: [], sent: 0, sig: null, fetchedAt: 0,
                busy: false, replay: false, playing: true, speed: 60, rclock: 0, last: performance.now() };
    const sig = (p) => (p ? `${p.t}|${p.hs}|${p.as}` : null);
    const replayState = () => ({ on: P.replay, playing: P.playing, speed: P.speed, clock: P.rclock });
    const reset = () => {
      P.sig = sig(P.full[P.sent - 1]);
      onMessage({ type: "game", header: P.header, points: P.full.slice(0, P.sent), replay: replayState() });
    };
    const append = (n, withReplay) => {
      const m = { type: "points", header: P.header, points: P.full.slice(P.sent, n) };
      if (withReplay) m.replay = replayState();
      P.sent = n; P.sig = sig(P.full[n - 1]);
      onMessage(m);
    };

    async function board() {
      const day = P.date;
      P.boardAt = Date.now();
      try {
        const r = await fetch(`/api/board?date=${day}`);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        onMessage({ type: "board", date: day, games: await r.json() });
      } catch (e) { onMessage({ type: "error", where: "board", message: String(e) }); }
    }

    async function game(first) {
      const gid = P.gid;
      P.busy = true; P.fetchedAt = Date.now();
      try {
        const r = await fetch(`/api/game/${gid}`);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const g = await r.json();
        if (gid !== P.gid) return;
        P.header = g.header; P.full = g.points;
        if (first) { P.sent = P.replay ? Math.min(1, P.full.length) : P.full.length; reset(); }
        else if (!P.replay) {
          if (P.sent > 0 && P.sent <= P.full.length && sig(P.full[P.sent - 1]) === P.sig) {
            if (P.full.length > P.sent) append(P.full.length, false);
          } else { P.sent = P.full.length; reset(); }  // the NBA revised earlier plays
        }
      } catch (e) { console.warn("poll:", e); }
      finally { P.busy = false; }
    }

    function tick() {
      const now = performance.now(), dt = (now - P.last) / 1000;
      P.last = now;
      if (P.date && Date.now() - P.boardAt > 15000) board();
      if (!P.gid || !P.header) return;
      if (P.replay && P.full.length) {
        if (P.playing) P.rclock += dt * P.speed;
        let n = 1;
        while (n < P.full.length && P.full[n].t <= P.rclock) n++;
        if (n >= P.full.length) P.replay = false;  // caught up: the replay is now the full timeline
        if (n > P.sent) append(n, true);
      }
      if (!P.header.finished && !P.busy && Date.now() - P.fetchedAt > 5000) game(false);
    }
    setInterval(tick, 250);

    return {
      readyState: 1,
      send(raw) {
        const m = JSON.parse(raw);
        if (m.op === "board") { P.date = m.date; board(); }
        else if (m.op === "watch") {
          Object.assign(P, { gid: m.game_id, header: null, full: [], sent: 0, replay: !!m.replay,
                             playing: true, rclock: 0, speed: +(m.speed || P.speed) });
          game(true);
        } else if (m.op === "replay" && P.header) {
          if ("playing" in m) P.playing = !!m.playing;
          if (m.speed) P.speed = +m.speed;
          if (m.restart && P.full.length) { Object.assign(P, { rclock: 0, sent: 1, replay: true }); reset(); }
        }
      },
    };
  }
  function send(obj) { if (S.ws && S.ws.readyState === 1) S.ws.send(JSON.stringify(obj)); }
  function setConn(text, cls) { const c = $("conn"); c.className = "conn " + cls; c.lastElementChild.textContent = text; }

  function onMessage(m) {
    if (m.type === "board") {
      if (m.date !== S.date) return;
      S.games = m.games;
      renderBoard();
      autoPick();
    } else if (m.type === "game") {
      if (m.header.game_id !== S.gameId) return;
      S.header = m.header; S.points = m.points; S.replay = m.replay;
      renderGame(true);
    } else if (m.type === "points") {
      if (m.header.game_id !== S.gameId) return;
      S.header = m.header;
      if (m.replay) S.replay = m.replay;
      const before = S.points.length;
      S.points = S.points.concat(m.points);
      renderGame(false, before);
    } else if (m.type === "error") {
      console.warn("server:", m.message);
      if (m.where === "board") { $("games").classList.remove("loading"); }
    }
  }

  // ------------------------------------------------------------ date + board
  function setDate(iso) {
    S.date = iso;
    $("date").value = iso;
    document.querySelectorAll(".presets button").forEach((b) => b.classList.toggle("on", S.nbaToday && addDays(S.nbaToday, +b.dataset.offset) === iso));
    $("games").classList.add("loading");
    send({ op: "board", date: iso });
    updateHash();
  }
  $("date").onchange = (e) => e.target.value && setDate(e.target.value);
  $("prev-day").onclick = () => setDate(addDays(S.date, -1));
  $("next-day").onclick = () => setDate(addDays(S.date, 1));
  document.querySelectorAll(".presets button").forEach((b) => (b.onclick = () => setDate(addDays(S.nbaToday, +b.dataset.offset))));

  // With nothing selected, open the most interesting game of the day so the page is never empty.
  function autoPick() {
    if (S.gameId || S.autoPicked || !S.games.length) return;
    S.autoPicked = true;
    const live = S.games.filter((g) => g.status === 2);
    const done = S.games.filter((g) => g.status === 3);
    const close = (g) => Math.abs(g.home.score - g.away.score);
    const pick = live[0] || done.sort((a, b) => close(a) - close(b))[0] || S.games[0];
    watch(pick.game_id);
  }

  function renderDaySummary() {
    const el = $("day-summary");
    const done = S.games.filter((g) => g.status === 3 && g.pregame_prob != null);
    const live = S.games.filter((g) => g.status === 2).length;
    const upcoming = S.games.filter((g) => g.status === 1).length;
    el.hidden = !S.games.length;
    el.replaceChildren();
    if (done.length) {
      const hits = done.reduce((a, g) => a + hitOf(g.pregame_prob, g.home.score > g.away.score), 0);
      el.append(h("div.ds-main", null, h("b", `${hits % 1 ? hits.toFixed(1) : hits} of ${done.length}`), " pre-game picks right"));
    }
    const bits = [];
    if (live) bits.push(`${live} live`);
    if (upcoming) bits.push(`${upcoming} upcoming`);
    if (done.length) bits.push(`${done.length} final`);
    el.append(h("div.ds-sub", bits.join(" · ")));
  }

  function renderBoard() {
    const list = $("games");
    list.classList.remove("loading");
    list.replaceChildren();
    renderDaySummary();
    if (!S.games.length) {
      list.appendChild(h("div.list-empty", "No games on this date."));
      return;
    }
    const order = { 2: 0, 1: 1, 3: 2 };
    const games = [...S.games].sort((a, b) => order[a.status] - order[b.status] || a.start_utc.localeCompare(b.start_utc));
    for (const g of games) {
      const card = h("button.gcard" + (g.game_id === S.gameId ? ".sel" : ""), { onclick: () => watch(g.game_id) });
      for (const side of ["away", "home"]) {
        const t = g[side], other = g[side === "away" ? "home" : "away"];
        const lg = h("span"); logo(lg, t, "mini-logo");
        const row = h("div.row", null, lg, h("span.tri", t.tricode), h("span.muted.small", t.record));
        if (g.status !== 1) row.appendChild(h("span.sc" + (g.status === 3 && t.score < other.score ? ".lose" : ""), t.score));
        card.appendChild(row);
      }
      const bar = h("div.minibar", null,
        h("div", { style: { background: "var(--away)", flexGrow: 1 - g.home_win_prob } }),
        h("div", { style: { background: "var(--home)", flexGrow: g.home_win_prob } }));
      card.appendChild(bar);

      const foot = h("div.foot");
      if (g.status === 2) foot.append(h("span.live-pill", "LIVE"), h("span", g.status_text));
      else foot.append(h("span", statusLine(g)));
      if (g.status === 3 && g.pregame_prob != null) {
        const homeWon = g.home.score > g.away.score;
        foot.append(h("span.pct", null, `Pick ${fav(g.pregame_prob, g.home, g.away).tricode} ${favPct(g.pregame_prob)} `, verdict(hitOf(g.pregame_prob, homeWon))));
      } else {
        foot.append(h("span.pct", `${fav(g.home_win_prob, g.home, g.away).tricode} ${favPct(g.home_win_prob)}`));
      }
      card.appendChild(foot);
      list.appendChild(card);
    }
  }

  // ------------------------------------------------------------ game view
  function watch(gameId) {
    S.gameId = gameId; S.points = []; S.header = null;
    $("empty").hidden = true; $("game").hidden = false;
    setMode("live", false);
    send({ op: "watch", game_id: gameId, replay: false });
    renderBoard();
    updateHash();
  }

  function setMode(mode, sendIt = true) {
    document.querySelectorAll("#mode button").forEach((b) => b.classList.toggle("on", b.dataset.mode === mode));
    $("replay-ctl").hidden = mode !== "replay";
    if (!sendIt) return;
    if (mode === "replay") send({ op: "watch", game_id: S.gameId, replay: true, speed: +$("speed").value });
    else send({ op: "watch", game_id: S.gameId, replay: false });
    $("play").textContent = "❚❚";
  }
  document.querySelectorAll("#mode button").forEach((b) => (b.onclick = () => !b.disabled && setMode(b.dataset.mode)));
  $("play").onclick = () => {
    if (!S.replay.on) { send({ op: "replay", restart: true, playing: true, speed: +$("speed").value }); $("play").textContent = "❚❚"; return; }
    const playing = !S.replay.playing;
    S.replay.playing = playing;
    send({ op: "replay", playing });
    $("play").textContent = playing ? "❚❚" : "▶";
  };
  $("restart").onclick = () => { send({ op: "replay", restart: true, playing: true, speed: +$("speed").value }); $("play").textContent = "❚❚"; };
  $("speed").onchange = () => send({ op: "replay", speed: +$("speed").value });

  const isFinal = () => !S.replay.on && (S.header.finished || S.header.status === 3);

  function swings(pts) {
    const deltas = pts.map((p, i) => (i ? p.p - pts[i - 1].p : 0));
    const top = deltas.map((d, i) => [Math.abs(d), i]).filter(([d, i]) => i > 0 && pts[i].desc).sort((a, b) => b[0] - a[0]).slice(0, 6);
    return { deltas, top: top.map(([, i]) => i) };
  }

  function renderGame(reset, before = 0) {
    const h0 = S.header, pts = S.points;
    if (!h0) return;
    const last = pts[pts.length - 1] || { p: h0.pregame_prob ?? 0.5, hs: 0, as: 0 };
    const prev = pts.length > 1 ? pts[pts.length - 2] : null;

    if (reset) {
      logo(document.querySelector('.logo[data-side="away"]'), h0.away);
      logo(document.querySelector('.logo[data-side="home"]'), h0.home);
      $("away-name").textContent = `${h0.away.city} ${h0.away.name}`.trim();
      $("home-name").textContent = `${h0.home.city} ${h0.home.name}`.trim();
      $("away-rec").textContent = h0.away.record ? `${h0.away.record} · Away` : "Away";
      $("home-rec").textContent = h0.home.record ? `${h0.home.record} · Home` : "Home";
      $("away-tri").textContent = h0.away.tricode;
      $("home-tri").textContent = h0.home.tricode;
      $("series").textContent = [h0.label, h0.series].filter(Boolean).join(" · ") || (h0.neutral ? "Neutral site" : "");
      Charts.legend($("wp-legend"), [{ name: `${h0.home.tricode} (home)`, color: "var(--home)" }, { name: `${h0.away.tricode} (away)`, color: "var(--away)" }], true);
      renderPregame(h0);
      const notStarted = h0.status === 1 && pts.length <= 1;
      document.querySelector('#mode button[data-mode="replay"]').disabled = notStarted;
    }

    $("away-score").textContent = last.as;
    $("home-score").textContent = last.hs;
    const st = $("status");
    st.replaceChildren();
    if (S.replay.on) st.textContent = `Replay · ${Charts.periodName(last.per || 1)} ${last.clk || ""}`;
    else if (isFinal()) st.textContent = "Final" + (last.per > 4 ? (last.per === 5 ? "/OT" : `/${last.per - 4}OT`) : "");
    else if (pts.length > 1) st.append(h("span.live-pill", "LIVE"), ` ${Charts.periodName(last.per)} ${last.clk}`);
    else st.textContent = `Tip-off ${localTime(h0.start_utc)}`;

    $("home-prob").textContent = pct(last.p);
    $("away-prob").textContent = pct(1 - last.p);
    $("bar-home").style.flexGrow = last.p;
    $("bar-away").style.flexGrow = 1 - last.p;
    const d = prev ? last.p - prev.p : 0;
    const fmt = (v) => (Math.abs(v) < 0.0005 ? "" : (v > 0 ? "▲ " : "▼ ") + Math.abs(100 * v).toFixed(1));
    $("home-delta").textContent = fmt(d); $("home-delta").className = d > 0 ? "up" : "down";
    $("away-delta").textContent = fmt(-d); $("away-delta").className = d < 0 ? "up" : "down";

    const sw = swings(pts);
    const marks = sw.top.slice(0, 3).map((i, k) => ({ t: pts[i].t, p: pts[i].p, n: k + 1 }));
    if (!S.chart) S.chart = Charts.winProbChart($("wp-chart"), S.chartState = { points: pts, home: h0.home.tricode, away: h0.away.tricode });
    Object.assign(S.chartState, { points: pts, home: h0.home.tricode, away: h0.away.tricode, marks });
    S.chart.redraw();

    renderHeadline(h0, pts, last);
    renderWhy(h0);
    renderFlow(h0, pts);
    renderCalls(h0, pts);
    renderPlays(reset ? pts.length : pts.length - before, sw);
  }

  function renderHeadline(hd, pts, last) {
    const el = $("headline");
    const H = hd.home.tricode, A = hd.away.tricode;
    const pre = hd.pregame_prob;
    el.replaceChildren();
    if (pts.length <= 1 && !isFinal()) {
      if (pre == null) return;
      el.append("The model makes ", h("b", `${fav(pre, H, A)} a ${favPct(pre)} favourite`),
        ` before tip-off, from Elo team ratings${hd.neutral ? "" : " and home court"}. It updates after every play once the game starts.`);
      return;
    }
    const lead = last.hs - last.as;
    if (isFinal()) {
      const homeWon = last.hs > last.as, W = homeWon ? H : A;
      const wp = pts.filter((p) => p.p > 0 && p.p < 1).map((p) => (homeWon ? p.p : 1 - p.p));
      const low = wp.length ? Math.min(...wp) : null;
      const preW = homeWon ? pre : 1 - pre;
      el.append(h("b", `${W} won ${Math.max(last.hs, last.as)}–${Math.min(last.hs, last.as)}`),
        `. Before tip-off the model gave them ${pct(preW)}`,
        low != null ? `, and their lowest point was ${pct(low)}` : "", ".",
        low != null && low < 0.2 ? " A real comeback." : preW < 0.35 ? " An upset by the model's numbers." : "");
      return;
    }
    const leader = lead > 0 ? H : A;
    const situation = lead === 0 ? "tied" : `${leader} up ${Math.abs(lead)}`;
    const when = last.per <= 4 ? `${last.clk} left in ${Charts.periodName(last.per)}` : `${last.clk} left in ${Charts.periodName(last.per)}`;
    el.append(h("b", `${fav(last.p, H, A)} has a ${favPct(last.p)} chance to win`), `: ${situation} with ${when}.`);
    if (S.replay.on) el.append(" (replay)");
  }

  function renderWhy(hd) {
    const el = $("why");
    if (S.replay.on) {
      el.replaceChildren(h("p.muted", "Shown for the live state. Finish or leave the replay to see it."));
      $("why-at").textContent = "";
      return;
    }
    const ex = hd.explain;
    if (!ex) { el.replaceChildren(h("p.muted", "Available once the model has scored a state.")); return; }
    if (isFinal()) {
      $("why-at").textContent = "at the final play before the buzzer";
    } else {
      $("why-at").textContent = ex.per === 1 && ex.clk === "12:00" ? "before tip-off" : `at ${Charts.periodName(ex.per)} ${ex.clk}`;
    }
    Charts.waterfall(el, ex.steps, { home: hd.home.tricode, away: hd.away.tricode });
  }

  function flowStats(pts) {
    let changes = 0, ties = 0, prevSign = 0, bigH = 0, bigA = 0, swing = 0, favH = 0, total = 0;
    for (let i = 1; i < pts.length; i++) {
      const d = pts[i].hs - pts[i].as, sign = Math.sign(d);
      bigH = Math.max(bigH, d); bigA = Math.max(bigA, -d);
      if (sign === 0 && prevSign !== 0) ties++;
      if (sign !== 0) { if (prevSign !== 0 && sign !== prevSign) changes++; prevSign = sign; }
      swing += Math.abs(pts[i].p - pts[i - 1].p);
      const dt = Math.max(0, pts[i].t - pts[i - 1].t), p = pts[i - 1].p;
      total += dt; favH += p > 0.5 ? dt : p === 0.5 ? dt / 2 : 0;
    }
    return { changes, ties, bigH, bigA, swing, favShare: total ? favH / total : 0.5 };
  }

  function renderFlow(hd, pts) {
    const el = $("flow");
    el.replaceChildren();
    if (pts.length <= 1) { el.appendChild(h("p.muted", "Starts at tip-off.")); return; }
    const f = flowStats(pts), H = hd.home.tricode, A = hd.away.tricode;
    tile(el, "Lead changes", f.changes, `${f.ties} time${f.ties === 1 ? "" : "s"} tied`);
    tile(el, "Largest lead", `${H} ${f.bigH} · ${A} ${f.bigA}`, "points");
    tile(el, "Time as favourite", `${H} ${Math.round(100 * f.favShare)}%`, `${A} ${Math.round(100 * (1 - f.favShare))}% of the game clock`);
    const avg = window.RecordView && window.RecordView.avgSwing();
    tile(el, "Total swing", f.swing.toFixed(2), avg ? `win-prob movement · season avg ${avg.toFixed(2)}` : "sum of every win-prob change",
      { title: "Adds up every change in win probability. 1.00 = one full swing from 0% to 100%." });
  }

  function renderCalls(hd, pts) {
    const el = $("calls");
    el.replaceChildren();
    const done = isFinal();
    const now = pts.length ? pts[pts.length - 1].t : 0;
    const homeWon = done ? pts[pts.length - 1].hs > pts[pts.length - 1].as : null;
    for (const [label, t] of CHECKPOINTS) {
      let p = null;
      if (pts.length > 1 || t === 0) {
        for (const q of pts) { if (q.t > t || q.p === 0 || q.p === 1) break; p = q.p; }
        if (t === 0 && hd.pregame_prob != null) p = hd.pregame_prob;
      }
      const reached = t === 0 || now >= t || done;
      const cell = h("div.call" + (reached && p != null ? "" : ".pending"));
      cell.append(h("div.k", label));
      if (reached && p != null) {
        cell.append(h("div.v", null, h("i", { style: { background: p >= 0.5 ? "var(--home)" : "var(--away)" } }), `${fav(p, hd.home.tricode, hd.away.tricode)} ${favPct(p)}`));
        if (done) cell.append(verdict(hitOf(p, homeWon)));
      } else cell.append(h("div.v.muted", "—"));
      el.appendChild(cell);
    }
  }

  function playItem(pt, delta, hd, n) {
    const c = h("span.chip");
    if (Math.abs(delta) < 0.0005) c.textContent = "±0.0";
    else c.append(h("i", { style: { background: delta > 0 ? "var(--home)" : "var(--away)" } }), `${delta > 0 ? hd.home.tricode : hd.away.tricode} +${Math.abs(100 * delta).toFixed(1)}`);
    const when = h("span.when", `${Charts.periodName(pt.per)} ${pt.clk}`);
    const desc = h("span.desc", null, n ? h("b.num", n) : null, pt.desc || "", h("small", `${hd.away.tricode} ${pt.as} – ${pt.hs} ${hd.home.tricode}`));
    return h("li", null, when, desc, c);
  }

  function renderPlays(newCount, sw) {
    const hd = S.header, pts = S.points, deltas = sw.deltas;
    const feed = $("feed");
    feed.replaceChildren();
    const start = Math.max(1, pts.length - 150);
    for (let i = pts.length - 1; i >= start; i--) {
      if (!pts[i].desc) continue;
      const li = playItem(pts[i], deltas[i], hd);
      if (newCount && newCount < 20 && i >= pts.length - newCount) li.classList.add("new");
      feed.appendChild(li);
    }
    $("play-count").textContent = pts.length > 1 ? `${pts.length - 1} events` : "";

    const list = $("swings");
    list.replaceChildren();
    if (!sw.top.length) list.appendChild(h("li.muted", "No plays yet."));
    sw.top.forEach((i, k) => list.appendChild(playItem(pts[i], deltas[i], hd, k < 3 ? k + 1 : null)));
  }

  function renderPregame(hd) {
    const el = $("pregame");
    el.replaceChildren();
    const f = hd.pre_spread >= 0 ? hd.home.tricode : hd.away.tricode;
    tile(el, `${hd.home.tricode} Elo`, hd.elo_home, "team strength rating");
    tile(el, `${hd.away.tricode} Elo`, hd.elo_away, "team strength rating");
    const spread = Math.abs(hd.pre_spread) < 1 ? "Pick'em" : `${f} −${Math.abs(hd.pre_spread).toFixed(1)}`;
    tile(el, "Elo spread", spread, hd.neutral ? "neutral site" : `incl. ${hd.home_court ? hd.home_court.toFixed(1) : "1.6"} pts home court`);
    if (hd.pregame_prob != null) {
      const p = hd.pregame_prob;
      tile(el, "Pre-game win prob", `${fav(p, hd.home.tricode, hd.away.tricode)} ${favPct(p)}`, "model at tip-off");
    }
  }

  // ------------------------------------------------------------ boot + routing
  function updateHash() {
    const p = new URLSearchParams();
    if (S.view !== "games") p.set("view", S.view);
    if (S.date) p.set("date", S.date);
    if (S.gameId) p.set("game", S.gameId);
    history.replaceState(null, "", "#" + p.toString());
  }

  // Other views open games through this.
  window.App = {
    openGame(gameId, day) {
      showView("games");
      if (day && day !== S.date) setDate(day);
      watch(gameId);
      window.scrollTo(0, 0);
    },
    showView,
  };

  async function boot() {
    const params = new URLSearchParams(location.hash.slice(1));
    try {
      const t = await (await fetch("/api/today")).json();
      S.nbaToday = t.date; S.websocket = t.websocket !== false;
    } catch { S.nbaToday = new Date().toISOString().slice(0, 10); }
    S.date = params.get("date") || S.nbaToday;
    $("date").value = S.date;
    setDate(S.date);
    connect();
    if (params.get("game")) watch(params.get("game"));
    const view = params.get("view");
    if (view === "record" || view === "model") showView(view);
  }
  boot();
})();
