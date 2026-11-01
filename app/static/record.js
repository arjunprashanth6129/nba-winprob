// Track record view: how the model is doing on this season's games (all out of sample).
(function () {
  const { $, h, pct, tile, table, verdict, getJSON, num } = UI;
  let report = null, loading = null;

  const shortDate = (iso) => new Date(iso + "T12:00:00Z").toLocaleDateString([], { month: "short", day: "numeric", timeZone: "UTC" });
  const matchup = (g) => `${g.away.tricode} ${g.away.score} @ ${g.home.tricode} ${g.home.score}`;
  const winner = (g) => (g.home_won ? g.home : g.away);
  const hitOf = (p, homeWon) => (p === 0.5 ? 0.5 : (p > 0.5) === homeWon ? 1 : 0);
  const favLabel = (g, p) => `${(p >= 0.5 ? g.home : g.away).tricode} ${pct(Math.max(p, 1 - p), 0)}`;

  function load() {
    if (!loading) {
      loading = Promise.all([getJSON("/api/season"), getJSON("/api/evaluation").catch(() => null), getJSON("/api/ratings").catch(() => null)])
        .then(([r, ev, ratings]) => { report = r; render(r, ev, ratings); document.dispatchEvent(new Event("seasonloaded")); })
        .catch((e) => { $("rec-loading").textContent = `Couldn't load the season record (${e.message}).`; loading = null; });
    }
    return loading;
  }

  function render(r, ev, ratings) {
    $("rec-season").textContent = r.season ? `${r.season}-${String(r.season + 1).slice(2)}` : "";
    if (!r.games) { $("rec-loading").textContent = "No finished games yet this season."; return; }
    $("rec-loading").hidden = true;
    $("rec-body").hidden = false;

    const pg = r.pregame, calls = Object.fromEntries(r.calls.map((c) => [c.key, c]));
    const tiles = $("rec-tiles");
    tiles.replaceChildren();
    const correct = pg.correct % 1 ? pg.correct.toFixed(1) : pg.correct;
    tile(tiles, "Pre-game picks", `${correct}–${+(pg.games - pg.correct).toFixed(1)}`, `${pct(pg.accuracy)} right · always picking the home team: ${pct(pg.home_team_accuracy)}`, { accent: true });
    if (calls.half) tile(tiles, "Called at halftime", pct(calls.half.accuracy), `favourite went on to win, ${calls.half.games} games`);
    if (calls.q3) tile(tiles, "Called entering the 4th", pct(calls.q3.accuracy), `favourite went on to win, ${calls.q3.games} games`);
    tile(tiles, "Every play", pct(r.states.accuracy), `log loss ${r.states.log_loss.toFixed(3)} · ${num(r.states.n)} game states`);
    tile(tiles, "Games scored", `${r.tracked}`, `of ${r.games} finished · updated ${new Date(r.updated).toLocaleString([], { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" })}`);

    // running pick accuracy
    const bd = r.by_date;
    const testPick = ev && ev.calls.find((c) => c.key === "tipoff");
    const series = [
      { name: "Model's pre-game pick", color: "var(--s1)", pts: bd.map((d, i) => [i, d.accuracy]) },
      { name: "Always pick the home team", color: "var(--ref)", dash: "4 4", pts: bd.map((d, i) => [i, d.home_team]) },
    ];
    if (testPick) series.push({ name: `Model on the ${ev.test_season} test season (${pct(testPick.neural_net.accuracy)})`, color: "var(--s3)", dash: "2 4", pts: bd.map((d, i) => [i, testPick.neural_net.accuracy]) });
    const lo = Math.max(0, Math.floor(Math.min(...series.flatMap((s) => s.pts.map((p) => p[1]))) * 10) / 10 - 0.05);
    const yt = []; for (let v = Math.ceil(lo * 10) / 10; v <= 1.0001; v += 0.1) yt.push(+v.toFixed(1));
    const every = Math.max(1, Math.ceil(bd.length / 7));
    Charts.lineChart($("ch-cum"), {
      label: "Running pre-game pick accuracy", series,
      x: { min: 0, max: Math.max(1, bd.length - 1), ticks: bd.map((d, i) => i).filter((i) => i % every === 0), fmt: (i) => shortDate(bd[i].date),
           tipFmt: (i) => `Through ${shortDate(bd[i].date)} · ${bd[i].games} games` },
      y: { min: lo, max: 1, ticks: yt, fmt: (v) => Math.round(v * 100) + "%", tipFmt: (v) => pct(v) },
      extra: (i, t) => Charts.div("tm", `That day: ${bd[i].day_correct} of ${bd[i].day_games} right`, t),
    });
    Charts.legend($("lg-cum"), series);

    // calls by stage, vs. test season
    const cats = r.calls.map((c) => c.label);
    const cs = [{ name: "This season", color: "var(--s1)", values: r.calls.map((c) => c.accuracy) }];
    if (ev) cs.push({ name: `${ev.test_season} test season`, color: "var(--ref)", values: r.calls.map((c) => (ev.calls.find((x) => x.key === c.key) || {}).neural_net?.accuracy ?? null) });
    Charts.columnChart($("ch-calls"), {
      label: "Accuracy of the model's call by stage", cats, series: cs,
      y: { min: 0.5, max: 1, ticks: [0.5, 0.6, 0.7, 0.8, 0.9, 1], fmt: (v) => Math.round(v * 100) + "%", tipFmt: (v) => pct(v) },
      tipExtra: (i, t) => Charts.div("tm", `${r.calls[i].games} games this season`, t),
    });
    Charts.legend($("lg-calls"), cs, true);

    const cal = r.states.calibration;
    Charts.lineChart($("ch-rcal"), {
      label: "Calibration this season", ref: [[0, 0], [1, 1]],
      series: [{ name: "actual home win rate", color: "var(--s1)", dots: true, pts: cal.map((c) => [c.pred, c.actual]) }],
      x: { min: 0, max: 1, ticks: [0, 0.25, 0.5, 0.75, 1], fmt: (v) => Math.round(v * 100) + "%", tipFmt: (v) => `Model said ${pct(v)}` },
      y: { min: 0, max: 1, ticks: [0, 0.25, 0.5, 0.75, 1], fmt: (v) => Math.round(v * 100) + "%", tipFmt: (v) => pct(v) },
      extra: (i, t) => Charts.div("tm", `${num(cal[i].n)} states`, t),
    });

    highlights($("hl-upsets"), r.upsets, (g) => `${winner(g).tricode} had ${pct(g.home_won ? g.pregame : 1 - g.pregame, 0)}`);
    highlights($("hl-comebacks"), r.comebacks, (g) => `${winner(g).tricode} fell to ${pct(g.winner_min, 1)}`);
    highlights($("hl-thrillers"), r.thrillers, (g) => `swing ${g.excitement.toFixed(2)} · ${g.lead_changes} lead changes`);

    if (ratings) renderRatings(ratings);

    renderRecent(r.recent, 15);
  }

  function renderRecent(games, shown) {
    const list = games.slice(0, shown);
    table($("rec-games"), ["Date", "Game", "Pre-game pick", "", "Halftime call", "", "Winner's low point", "Total swing"],
      list.map((g) => {
        const half = g.at && g.at.half;
        return [shortDate(g.date), matchup(g), favLabel(g, g.pregame), verdict(hitOf(g.pregame, g.home_won)),
          half != null ? favLabel(g, half) : "—", half != null ? verdict(hitOf(half, g.home_won)) : "",
          g.winner_min != null ? pct(g.winner_min, 1) : "—", g.excitement != null ? g.excitement.toFixed(2) : "—"];
      }),
      { onclick: (i) => window.App.openGame(list[i].game_id, list[i].date) });
    const more = $("rec-more");
    more.hidden = games.length <= shown;
    more.textContent = `Show ${games.length - shown} more`;
    more.onclick = () => renderRecent(games, games.length);
  }

  function highlights(el, games, note) {
    el.replaceChildren();
    for (const g of games) {
      el.appendChild(h("li", { onclick: () => window.App.openGame(g.game_id, g.date), tabindex: 0,
                               onkeydown: (e) => e.key === "Enter" && window.App.openGame(g.game_id, g.date) },
        h("div.m", matchup(g)), h("div.n", null, h("span", note(g)), h("span.muted", shortDate(g.date)))));
    }
  }

  function renderRatings(rows) {
    const el = $("ratings");
    el.replaceChildren();
    const max = Math.max(...rows.map((r) => Math.abs(r.vs_avg)));
    const half = Math.ceil(rows.length / 2);
    [rows.slice(0, half), rows.slice(half)].forEach((part, k) => {
      const col = h("div");
      Charts.hbars(col, part.map((r, i) => ({
        label: `${k * half + i + 1}. ${r.tricode} ${r.wins}-${r.losses}`, value: r.vs_avg, color: "var(--s1)",
        tip: (t, div) => { div("tv", `${r.city} ${r.name}`, t); div("tm", `Elo ${r.elo.toFixed(0)} · ${r.vs_avg > 0 ? "+" : ""}${r.vs_avg} pts vs. average`, t);
                           div("tm", `${r.change_30d >= 0 ? "▲" : "▼"} ${Math.abs(r.change_30d).toFixed(0)} Elo over the last 30 days`, t); },
      })), { max, diverging: true, fmt: (v) => (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v).toFixed(1) });
      el.appendChild(col);
    });
  }

  // The Games view compares each game's total swing to the season average.
  window.RecordView = { load, avgSwing: () => report && report.avg_swing };
  setTimeout(load, 1500);  // warm it in the background
})();
