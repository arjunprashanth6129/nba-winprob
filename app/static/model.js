// "The model" view: how it works, what it learned, how well it predicts, and a what-if calculator.
(function () {
  const { $, h, pct, tile, table, num, getJSON } = UI;
  const NS = "http://www.w3.org/2000/svg";
  let loaded = false;

  const pctAxis = (v) => Math.round(v * 100) + "%";
  const COLORS = { neural_net: "var(--s1)", logistic_regression: "var(--s2)", elo_pregame_only: "var(--s3)" };
  const NAMES = { neural_net: "Neural net", logistic_regression: "Logistic regression", elo_pregame_only: "Pre-game Elo only" };

  // What each input is, in the order the network sees them (winprob/features.py).
  const FEATURE_INFO = {
    diff: ["Score margin", "Home score minus away score.", "The main signal."],
    t_min: ["Minutes left", "Minutes left in regulation, or in the current overtime.", "The same lead means more later."],
    log_t: ["log(1 + minutes left)", "Time left on a log scale.", "More resolution in the final minutes, where things change fastest."],
    diff_over_sqrt_t: ["Lead ÷ √(minutes + 1)", "Margin scaled by how much randomness is left.", "Roughly a z-score: how many 'standard deviations' the lead is."],
    is_ot: ["Overtime", "1 during overtime.", "OT is five short minutes with its own dynamics."],
    poss: ["Possession", "+1 home ball, −1 away ball.", "Worth about a point; decisive late."],
    pre_spread: ["Elo spread", "Pre-game expected home margin from team ratings.", "Who's better, in points."],
    spread_left: ["Spread × time left", "The spread times the share of the game remaining.", "Team strength should fade as the game plays out."],
    home_to: ["Home timeouts", "Timeouts the home team has left.", "Tracked with NBA rules: 7, max 2 in the last 3:00, 2 per OT."],
    away_to: ["Away timeouts", "Timeouts the away team has left.", "Lets a trailing team stop the clock."],
  };
  const GROUPS = { score_margin: "Score margin", time_left: "Time left", elo_spread: "Elo team strength", possession: "Possession", timeouts: "Timeouts", overtime: "Overtime flag" };

  function svgEl(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (parent) parent.appendChild(e);
    return e;
  }
  function observe(el, draw) {
    new ResizeObserver(() => draw()).observe(el);
    draw();
  }

  async function load() {
    if (loaded) return;
    loaded = true;
    let m, e;
    try { [m, e] = await Promise.all([getJSON("/api/metrics"), getJSON("/api/evaluation")]); }
    catch (err) { $("model-blurb").textContent = `Couldn't load the model report (${err.message}).`; loaded = false; return; }
    intro(m, e);
    pipeline(m, e);
    network(m, e);
    inputs(m, e);
    tryIt(e);
    learned(m, e);
    ability(m, e);
    calibration(m, e);
    training(m, e);
    document.querySelectorAll("#subnav a").forEach((a) => (a.onclick = (ev) => {
      ev.preventDefault();
      document.querySelector(a.getAttribute("href")).scrollIntoView({ behavior: "smooth", block: "start" });
    }));
  }

  // ------------------------------------------------------------ intro
  function intro(m, e) {
    const nn = m.overall.neural_net, lr = m.overall.logistic_regression, elo = m.overall.elo_pregame_only;
    const totalGames = e.seasons.filter((s) => s.role !== "live").reduce((a, s) => a + s.games, 0);
    $("model-blurb").textContent =
      `After every play, the model reads the state of the game (score, clock, possession, timeouts and how strong each team is) ` +
      `and outputs the probability that the home team wins. It's a small neural network trained on ${(m.rows.train / 1e6).toFixed(1)} million game states ` +
      `from ${m.train_seasons[0]} to ${m.train_seasons[m.train_seasons.length - 1]}, tuned on ${m.val_season}, and scored on all of ${m.test_season}, ` +
      `a season it never saw. ${num(totalGames)} games of play-by-play in all.`;
    const t = $("model-tiles");
    t.replaceChildren();
    tile(t, "Test log loss", nn.log_loss.toFixed(4), `logistic ${lr.log_loss.toFixed(4)} · Elo only ${elo.log_loss.toFixed(4)}`, { accent: true });
    tile(t, "Brier skill score", pct(e.skill.brier_skill_vs_home_rate, 0), `less squared error than always guessing the ${pct(e.skill.base_rate, 0)} home win rate`);
    tile(t, "Calibration error", `${(100 * e.ece).toFixed(1)} pts`, "average gap between stated and actual win rates");
    tile(t, "Picks the winner", pct(nn.accuracy), `of all game states · ${pct(m.by_phase.Q4.neural_net.accuracy, 0)} in the 4th quarter`);
    tile(t, "Parameters", num(e.n_params), "10 → 128 → 64 → 32 → 1");
  }

  // ------------------------------------------------------------ pipeline
  function pipeline(m, e) {
    const games = e.seasons.filter((s) => s.role !== "live").reduce((a, s) => a + s.games, 0);
    const steps = [
      ["1", "Play-by-play", `${num(games)} games since ${m.train_seasons[0]}, ~550 events each (shots, fouls, turnovers, timeouts), from the NBA's live feed via nba_api.`],
      ["2", "Team ratings", `Elo ratings from every result since 2014-15. The rating gap plus a ${e.elo.home_adv}-point home edge becomes a pre-game point spread (${e.elo.elo_per_point} Elo ≈ 1 point).`],
      ["3", "Game states", "One function turns each event into 10 numbers: lead, time left, possession, timeouts, the Elo spread. Training and the live site share it, so they never disagree."],
      ["4", "Neural network", `A 4-layer network with ${num(e.n_params)} weights, trained in PyTorch to minimise log loss. Its output is P(home team wins).`],
      ["5", "Live dashboard", "A Flask server polls the NBA's live feed every 5 seconds, scores the new plays, and pushes them to your browser."],
    ];
    const el = $("pipeline");
    el.replaceChildren();
    steps.forEach(([n, t, d], i) => {
      el.appendChild(h("div.step", null, h("div.n", n), h("div.t", t), h("div.d", d)));
      if (i < steps.length - 1) el.appendChild(h("div.arrow", { "aria-hidden": "true" }, "→"));
    });
  }

  // ------------------------------------------------------------ network
  function network(m, e) {
    const layers = [
      { n: 10, label: "Inputs", sub: "10 features" },
      { n: 128, label: "Hidden", sub: "128 · SiLU" },
      { n: 64, label: "Hidden", sub: "64 · SiLU" },
      { n: 32, label: "Hidden", sub: "32 · SiLU" },
      { n: 1, label: "Output", sub: "sigmoid" },
    ];
    const names = m.features.map((f) => FEATURE_INFO[f][0]);
    const container = $("arch");
    $("arch-sub").textContent = `${num(e.n_params)} learnable weights`;
    observe(container, () => {
      const W = container.clientWidth, H = 330;
      if (!W) return;
      container.replaceChildren();
      const svg = svgEl("svg", { width: W, height: H, role: "img", "aria-label": "Network architecture: 10 inputs, hidden layers of 128, 64 and 32, one output" }, container);
      const left = Math.min(150, W * 0.3), right = Math.min(110, W * 0.2);
      const xs = layers.map((_, i) => left + (i * (W - left - right)) / (layers.length - 1));
      const top = 18, bottom = H - 46;
      const shown = layers.map((L) => Math.min(L.n, 10));
      const ys = shown.map((k) => Array.from({ length: k }, (_, j) => (k === 1 ? (top + bottom) / 2 : top + (j * (bottom - top)) / (k - 1))));
      for (let i = 0; i < layers.length - 1; i++)
        for (const y1 of ys[i]) for (const y2 of ys[i + 1])
          svgEl("line", { x1: xs[i], y1, x2: xs[i + 1], y2, stroke: "var(--grid)", "stroke-width": 1 }, svg);
      layers.forEach((L, i) => {
        ys[i].forEach((y, j) => {
          const gap = L.n > 10 && j === 4;
          if (gap) { svgEl("text", { x: xs[i], y: y + 5, "text-anchor": "middle", class: "axis-label" }, svg).textContent = "⋮"; return; }
          svgEl("circle", { cx: xs[i], cy: y, r: L.n === 1 ? 9 : 6, fill: i === 0 ? "var(--s3)" : i === layers.length - 1 ? "var(--home)" : "var(--s1)", stroke: "var(--surface-1)", "stroke-width": 2 }, svg);
          if (i === 0) svgEl("text", { x: xs[i] - 12, y: y + 4, "text-anchor": "end", class: "axis-label" }, svg).textContent = names[j];
        });
        svgEl("text", { x: xs[i], y: H - 26, "text-anchor": "middle", class: "direct-label" }, svg).textContent = L.label;
        svgEl("text", { x: xs[i], y: H - 10, "text-anchor": "middle", class: "axis-label" }, svg).textContent = L.sub;
      });
      svgEl("text", { x: xs[4] + 16, y: (top + bottom) / 2 + 4, class: "direct-label" }, svg).textContent = "P(home wins)";
    });

    const best = m.history.reduce((b, r) => (r.val_loss < b.val_loss ? r : b), m.history[0]);
    const rows = [
      ["Training data", `${num(m.games.train)} games · ${(m.rows.train / 1e6).toFixed(2)}M states · ${m.train_seasons[0]} to ${m.train_seasons[m.train_seasons.length - 1]}`],
      ["Validation", `${m.val_season} · ${num(m.games.val)} games, used only to decide when to stop`],
      ["Test", `${m.test_season} · ${num(m.games.test)} games, untouched until the final evaluation`],
      ["Split", "By season, never by row, so plays from one game never leak across sets"],
      ["Inputs", "Standardised to mean 0, std 1 using training-set statistics"],
      ["Loss", "Binary cross-entropy (log loss) on the final result"],
      ["Optimiser", "AdamW · learning rate 0.002 · weight decay 0.0001"],
      ["Schedule", "Cosine learning-rate decay · mini-batches of 8,192 states"],
      ["Regularisation", "Dropout 5% · weight decay · early stopping (patience 4)"],
      ["Epochs", `Best validation loss at epoch ${best.epoch}, stopped after ${m.history.length}`],
      ["Serving", "Weights exported to NumPy; outputs match PyTorch to 1e-7"],
    ];
    const kv = $("train-setup");
    kv.replaceChildren();
    for (const [k, v] of rows) kv.append(h("dt", k), h("dd", v));
  }

  // ------------------------------------------------------------ inputs
  function inputs(m, e) {
    table($("features"), ["Input", "What it is", "Why it helps"], m.features.map((f) => {
      const [name, what, why] = FEATURE_INFO[f];
      return [h("b", name), what, why];
    }));
    const help = {
      importance: "How much worse the test-season log loss gets when one input is scrambled across games. Over a whole game, the score dominates, then team strength and the clock. Possession and timeouts look tiny here because they only matter in a handful of close, late moments (see Value of having the ball, below).",
      importance_crunch: "Same test, on the last 2 minutes of games within 6 points. Team strength barely matters any more: the score and the clock decide it.",
    };
    const draw = (key) => {
      Charts.hbars($("importance"), e[key].map((r) => ({
        label: GROUPS[r.group] || r.group, value: Math.max(0, r.delta_log_loss), color: "var(--s1)",
        tip: (t, div) => { div("tv", GROUPS[r.group], t); div("tm", `Log loss +${r.delta_log_loss.toFixed(4)} when scrambled`, t); },
      })), { fmt: (v) => "+" + (v >= 0.01 ? v.toFixed(3) : v.toFixed(4)) });
      $("imp-help").textContent = help[key];
      document.querySelectorAll("#imp-seg button").forEach((b) => b.classList.toggle("on", b.dataset.k === key));
    };
    document.querySelectorAll("#imp-seg button").forEach((b) => (b.onclick = () => draw(b.dataset.k)));
    draw("importance");
  }

  // ------------------------------------------------------------ try it
  function tryIt(e) {
    const st = { lead: -2, q: 4, clock: 45, poss: 1, strength: 0, hc: true, hto: 1, ato: 2 };
    const box = $("try-controls");
    const ctl = {};
    const fmtClock = (s) => `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")}`;

    function slider(key, label, min, max, step, fmt) {
      const out = h("output");
      const input = h("input", { type: "range", min, max, step, value: st[key], "aria-label": label });
      input.oninput = () => { st[key] = +input.value; sync(); };
      ctl[key] = { input, out, fmt };
      box.appendChild(h("label.ctl", null, h("span.lbl", null, label, out), input));
    }
    function seg(key, label, opts) {
      const s = h("div.seg");
      opts.forEach(([v, t]) => s.appendChild(h("button", { "data-v": v, onclick: () => { st[key] = v; sync(); } }, t)));
      ctl[key] = { seg: s };
      box.appendChild(h("div.ctl", null, h("span.lbl", label), s));
    }
    box.replaceChildren();
    const presets = h("div.presets.try-presets");
    [["Tip-off, even teams", { lead: 0, q: 1, clock: 720, poss: 0, strength: 0, hc: true, hto: 7, ato: 7 }],
     ["Down 3, 10 s left, ball", { lead: -3, q: 4, clock: 10, poss: 1, strength: 0, hc: true, hto: 1, ato: 1 }],
     ["Up 10 at halftime", { lead: 10, q: 3, clock: 720, poss: 0, strength: 0, hc: true, hto: 5, ato: 5 }],
     ["Big underdog, tied late", { lead: 0, q: 4, clock: 120, poss: -1, strength: -8, hc: true, hto: 2, ato: 2 }]]
      .forEach(([t, s]) => presets.appendChild(h("button", { onclick: () => { Object.assign(st, s); sync(); } }, t)));
    box.appendChild(presets);
    slider("lead", "Score", -25, 25, 1, (v) => (v === 0 ? "Tied" : v > 0 ? `Home up ${v}` : `Away up ${-v}`));
    seg("q", "Period", [[1, "Q1"], [2, "Q2"], [3, "Q3"], [4, "Q4"], [5, "OT"]]);
    slider("clock", "Clock", 0, 720, 1, (v) => `${fmtClock(v)} left`);
    seg("poss", "Ball", [[1, "Home"], [0, "Jump ball"], [-1, "Away"]]);
    slider("strength", "Team strength", -12, 12, 0.5, (v) => (v === 0 ? "Even" : v > 0 ? `Home ${v} pts better` : `Away ${-v} pts better`));
    seg("hc", "Home court", [[true, "On"], [false, "Neutral"]]);
    slider("hto", "Home timeouts", 0, 7, 1, (v) => `${v}`);
    slider("ato", "Away timeouts", 0, 7, 1, (v) => `${v}`);

    let timer = null, seq = 0, last = null;
    function sync() {
      const maxClock = st.q === 5 ? 300 : 720;
      ctl.clock.input.max = maxClock;
      st.clock = Math.min(st.clock, maxClock);
      for (const [k, c] of Object.entries(ctl)) {
        if (c.input) { c.input.value = st[k]; c.out.textContent = c.fmt(st[k]); }
        else c.seg.querySelectorAll("button").forEach((b) => b.classList.toggle("on", String(st[k]) === b.dataset.v));
      }
      clearTimeout(timer);
      timer = setTimeout(fetchIt, 90);
    }
    async function fetchIt() {
      const ot = st.q === 5;
      const t = ot ? st.clock : (4 - st.q) * 720 + st.clock;
      const q = new URLSearchParams({ diff: st.lead, t, ot: ot ? 1 : 0, poss: st.poss, strength: st.strength, hc: st.hc ? 1 : 0, hto: st.hto, ato: st.ato });
      const my = ++seq;
      try {
        const r = await getJSON("/api/whatif?" + q);
        if (my !== seq) return;
        last = r;
        draw(r, t / 60, ot);
      } catch (err) { $("try-p").textContent = "–"; console.warn(err); }
    }
    function draw(r, minutesLeft, ot) {
      $("try-p").textContent = pct(r.p);
      $("try-bar-home").style.flexGrow = r.p;
      $("try-bar-away").style.flexGrow = 1 - r.p;
      Charts.waterfall($("try-why"), r.explain, { home: "Home", away: "Away" });
      Charts.lineChart($("try-curve"), {
        label: "Home win probability for every lead at this moment",
        series: [{ name: "home wins", color: "var(--home)", pts: r.leads.map((l, i) => [l, r.lead_curve[i]]) }],
        marks: [{ x: Math.max(-20, Math.min(20, st.lead)), y: r.lead_curve[r.leads.indexOf(Math.max(-20, Math.min(20, st.lead)))], color: "var(--home)", label: pct(r.p) }],
        x: { min: -20, max: 20, ticks: [-20, -10, 0, 10, 20], fmt: (v) => (v > 0 ? "+" + v : "" + v), tipFmt: (v) => (v === 0 ? "Tied" : v > 0 ? `Home up ${v}` : `Away up ${-v}`) },
        y: { min: 0, max: 1, ticks: [0, 0.25, 0.5, 0.75, 1], fmt: pctAxis, tipFmt: (v) => pct(v) },
      });
      Charts.heatmap($("try-heat"), {
        label: "Home win probability by lead and minutes left", xs: r.leads, ys: r.minutes, z: r.surface, home: "Home", away: "Away",
        marker: ot ? null : { x: st.lead, y: minutesLeft },
        onPick: (lead, mins) => {  // minutes left in regulation -> quarter + clock
          const q = mins === 0 ? 4 : 4 - Math.floor((mins - 1) / 12);
          Object.assign(st, { lead, q, clock: (mins - (4 - q) * 12) * 60 });
          sync();
        },
      });
    }
    $("grad").style.background = `linear-gradient(90deg, ${[0, 0.25, 0.5, 0.75, 1].map((p) => Charts.divergingFill(p)).join(",")})`;
    document.addEventListener("themechange", () => last && draw(last, st.q === 5 ? 0 : ((4 - st.q) * 720 + st.clock) / 60, st.q === 5));
    sync();
  }

  // ------------------------------------------------------------ what it learned
  function learned(m, e) {
    const colors = ["var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)"];
    const curves = Object.entries(m.lead_curves).map(([name, c], i) => ({ name, color: colors[i], pts: c.lead.map((l, j) => [l, c.p[j]]) }));
    Charts.lineChart($("ch-lead"), {
      label: "Home win probability by lead", series: curves,
      x: { min: -15, max: 15, ticks: [-15, -10, -5, 0, 5, 10, 15], fmt: (v) => (v > 0 ? "+" + v : "" + v), tipFmt: (v) => `Home lead ${v > 0 ? "+" + v : v}` },
      y: { min: 0, max: 1, ticks: [0, 0.25, 0.5, 0.75, 1], fmt: pctAxis, tipFmt: (v) => pct(v) },
    });
    Charts.legend($("lg-lead"), curves);

    const pv = e.possession_value;
    const ppts = pv.t_sec.map((t, i) => [(720 - t) / 60, pv.pp[i]]);
    const pmax = Math.ceil(Math.max(...pv.pp) * 20) / 20;
    Charts.lineChart($("ch-poss"), {
      label: "Value of possession in a tied 4th quarter",
      series: [{ name: "with the ball − without", color: "var(--s1)", pts: ppts }],
      x: { min: 0, max: 12, ticks: [0, 3, 6, 9, 12], fmt: (v) => `${12 - v}:00`, tipFmt: (v) => { const s = Math.round((12 - v) * 60); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")} left in Q4`; } },
      y: { min: 0, max: pmax, ticks: [0, pmax / 2, pmax], fmt: (v) => "+" + Math.round(v * 100) + " pts", tipFmt: (v) => "+" + (100 * v).toFixed(1) + " pts" },
    });

    $("poss-help").textContent = `Win probability with the ball minus without it, in a tied game. Worth about ${(100 * pv.pp[0]).toFixed(1)} points of win probability at the start of the 4th quarter, rising to ${(100 * pv.pp[pv.pp.length - 1]).toFixed(1)} in the final seconds.`;

    const pf = e.prior_fade;
    const fx = pf.t_sec.map((t) => (2880 - t) / 60);
    const fade = Object.entries(pf.curves).map(([k, ps], i) => ({ name: `favoured by ${k.slice(1)}`, color: colors[i], pts: fx.map((x, j) => [x, ps[j]]) }));
    Charts.lineChart($("ch-fade"), {
      label: "Pre-game favourite's win probability in a tied game, over time", series: fade,
      x: { min: 0, max: 48, ticks: [0, 12, 24, 36, 48], fmt: (v) => (v === 0 ? "Tip" : v === 48 ? "End" : `Q${v / 12 + 1}`), tipFmt: (v) => `${48 - v} min left, tied` },
      y: { min: 0.5, max: 0.85, ticks: [0.5, 0.6, 0.7, 0.8], fmt: pctAxis, tipFmt: (v) => pct(v) },
    });
    Charts.legend($("lg-fade"), fade);
    const nine = pf.curves["+9"];
    if (nine) $("fade-help").textContent = `Team strength matters less as the game plays out. With the score tied, a 9-point favourite is ${pct(nine[0], 0)} at tip-off but ${pct(nine[nine.length - 2], 0)} with a minute left.`;

    const hs = e.home_by_season;
    Charts.columnChart($("ch-home"), {
      label: "Home team win rate by season", cats: hs.map((s) => s.season.slice(2)),
      series: [{ name: "home win rate", color: "var(--s1)", values: hs.map((s) => s.home_win) }],
      y: { min: 0.5, max: 0.62, ticks: [0.5, 0.54, 0.58, 0.62], fmt: pctAxis, tipFmt: (v) => pct(v) },
      ref: { y: 0.5, label: "" }, tipCat: (i) => hs[i].season, tipExtra: (i, t) => Charts.div("tm", `${num(hs[i].games)} games`, t),
    });
    const before = hs.filter((s) => +s.season.slice(0, 4) < 2019), after = hs.filter((s) => +s.season.slice(0, 4) >= 2019);
    const avg = (a) => a.reduce((x, s) => x + s.home_win * s.games, 0) / a.reduce((x, s) => x + s.games, 0);
    $("home-help").textContent = `Home teams won ${pct(avg(before))} of games from ${before[0].season} to ${before[before.length - 1].season}, but ${pct(avg(after))} since. ` +
      `So the Elo home-court edge was refit to ${e.elo.home_adv} rating points, about ${(e.elo.home_adv / e.elo.elo_per_point).toFixed(1)} points on the scoreboard (FiveThirtyEight used 100).`;
  }

  // ------------------------------------------------------------ predictive ability
  function ability(m, e) {
    $("test-label").textContent = `· ${m.test_season}, ${num(m.games.test)} games never seen in training`;
    const base = e.skill.brier_home_rate;
    table($("cmp"), ["Model", "Log loss", "Brier", "Brier skill", "Accuracy", "AUC"],
      Object.entries(m.overall).map(([k, s]) => [{ neural_net: "Neural net (this model)", logistic_regression: "Logistic regression, same inputs", elo_pregame_only: "Pre-game Elo only" }[k],
        s.log_loss.toFixed(4), s.brier.toFixed(4), pct(1 - s.brier / base, 1), pct(s.accuracy), s.auc.toFixed(3)]),
      { cls: (i) => (i === 0 ? "best" : "") });

    const bm = e.by_minute;
    const series = Object.keys(NAMES).map((k) => ({ name: NAMES[k], color: COLORS[k], pts: bm.map((r) => [r.minute + 0.5, r[k]]) }));
    Charts.lineChart($("ch-llmin"), {
      label: "Log loss by minute of the game", series,
      x: { min: 0, max: 49, ticks: [0, 12, 24, 36, 48], fmt: (v) => (v === 0 ? "Tip" : v === 48 ? "OT" : `Q${v / 12 + 1}`), tipFmt: (v) => (v >= 48 ? "Overtime" : `Minute ${Math.floor(v) + 1}`) },
      y: { min: 0, max: 0.75, ticks: [0, 0.25, 0.5, 0.75], fmt: (v) => v.toFixed(2), tipFmt: (v) => v.toFixed(4) },
      extra: (i, t) => Charts.div("tm", `${num(bm[i].n)} states`, t),
    });
    Charts.legend($("lg-llmin"), series);

    const cs = Object.keys(NAMES).map((k) => ({ name: NAMES[k], color: COLORS[k], values: e.calls.map((c) => c[k].accuracy) }));
    Charts.columnChart($("ch-calls-test"), {
      label: "How often each model's favourite won, by stage", cats: e.calls.map((c) => c.label), series: cs,
      y: { min: 0.5, max: 1, ticks: [0.5, 0.6, 0.7, 0.8, 0.9, 1], fmt: pctAxis, tipFmt: (v) => pct(v) },
      tipExtra: (i, t) => Charts.div("tm", `${num(e.calls[i].games)} games`, t),
    });
    Charts.legend($("lg-calls-test"), cs, true);

    table($("conf"), ["Model's confidence", "Share of states", "It said (avg)", "Favourite won", "States"],
      e.confidence.map((c) => [`${Math.round(c.lo * 100)}–${Math.round(c.hi * 100)}%`, pct(c.share), pct(c.said), pct(c.won), num(c.n)]));
    table($("comebacks"), ["Winner's chance fell below", "Games", "Share of games"],
      e.comebacks.map((c) => [pct(c.below, 0), num(c.games), pct(c.share)]));
    const ten = e.comebacks.find((c) => c.below === 0.1);
    if (ten) $("comeback-help").textContent = `A calibrated model has to be wrong sometimes. In ${pct(ten.share)} of test-season games, about one in ten, the eventual winner's chance fell below 10% at some point.`;

    const cl = $("clutch");
    if (m.clutch) table(cl, ["Situation", "NN log loss", "LR log loss", "NN better by", "States"], m.clutch.map((c) => {
      const nn = c.neural_net, lr = c.logistic_regression;
      return [c.slice, nn.log_loss.toFixed(4), lr.log_loss.toFixed(4), `${(100 * (lr.log_loss - nn.log_loss) / lr.log_loss).toFixed(1)}%`, num(nn.n)];
    }));
    table($("phase"), ["Phase", "NN log loss", "LR log loss", "NN accuracy", "States"],
      ["Q1", "Q2", "Q3", "Q4", "OT"].filter((p) => m.by_phase[p]).map((p) => {
        const v = m.by_phase[p];
        return [p, v.neural_net.log_loss.toFixed(4), v.logistic_regression.log_loss.toFixed(4), pct(v.neural_net.accuracy), num(v.neural_net.n)];
      }));
  }

  // ------------------------------------------------------------ calibration
  function calibration(m, e) {
    const cal = e.calibration_fine;
    $("ece-label").textContent = `test season · average gap ${(100 * e.ece).toFixed(1)} pts`;
    Charts.lineChart($("ch-calib"), {
      label: "Calibration", ref: [[0, 0], [1, 1]],
      series: [{ name: "actual home win rate", color: "var(--s1)", dots: true, pts: cal.map((c) => [c.pred, c.actual]) }],
      x: { min: 0, max: 1, ticks: [0, 0.25, 0.5, 0.75, 1], fmt: pctAxis, tipFmt: (v) => `Model said ${pct(v)}` },
      y: { min: 0, max: 1, ticks: [0, 0.25, 0.5, 0.75, 1], fmt: pctAxis, tipFmt: (v) => pct(v) },
      extra: (i, t) => Charts.div("tm", `${num(cal[i].n)} states`, t),
    });
    const total = cal.reduce((a, c) => a + c.n, 0);
    Charts.columnChart($("ch-hist"), {
      label: "Distribution of predicted probabilities", cats: cal.map((c) => Math.round(c.lo * 100) + "%"),
      series: [{ name: "of all states", color: "var(--s1)", values: cal.map((c) => c.n / total) }],
      y: { min: 0, max: Math.ceil((Math.max(...cal.map((c) => c.n)) / total) * 50) / 50, ticks: [0, 0.04, 0.08, 0.12].filter((v) => v <= Math.ceil((Math.max(...cal.map((c) => c.n)) / total) * 50) / 50), fmt: pctAxis, tipFmt: (v) => pct(v) },
      tipCat: (i) => `Model said ${Math.round(cal[i].lo * 100)}–${Math.round(cal[i].hi * 100)}%`,
      tipExtra: (i, t) => Charts.div("tm", `${num(cal[i].n)} states · home won ${pct(cal[i].actual)}`, t),
    });
  }

  // ------------------------------------------------------------ training
  function training(m, e) {
    const hist = m.history;
    const lossSeries = [
      { name: "train", color: "var(--s1)", pts: hist.map((r) => [r.epoch, r.train_loss]) },
      { name: `validation (${m.val_season})`, color: "var(--s2)", pts: hist.map((r) => [r.epoch, r.val_loss]) },
    ];
    const all = hist.flatMap((r) => [r.train_loss, r.val_loss]);
    const lo = Math.floor(Math.min(...all) * 100) / 100, hi = Math.ceil(Math.max(...all) * 100) / 100;
    const best = hist.reduce((b, r) => (r.val_loss < b.val_loss ? r : b), hist[0]);
    Charts.lineChart($("ch-loss"), {
      label: "Training loss", series: lossSeries,
      marks: [{ x: best.epoch, y: best.val_loss, color: "var(--s2)", label: "kept" }],
      x: { min: 1, max: Math.max(2, hist.length), ticks: hist.map((r) => r.epoch), fmt: (v) => "" + v, tipFmt: (v) => `Epoch ${v}` },
      y: { min: lo, max: hi, ticks: [lo, (lo + hi) / 2, hi], fmt: (v) => v.toFixed(3), tipFmt: (v) => v.toFixed(4) },
    });
    Charts.legend($("lg-loss"), lossSeries);
    const role = { train: "Train", validation: "Validation", test: "Test", live: "Live (so far)" };
    table($("seasons"), ["Season", "Used for", "Games", "Log loss", "Brier", "Accuracy"],
      e.seasons.map((s) => [s.season, role[s.role], num(s.games), s.log_loss.toFixed(4), s.brier.toFixed(4), pct(s.accuracy)]),
      { cls: (i) => (e.seasons[i].role === "test" ? "best" : "") });
  }

  window.ModelView = { load };
})();
