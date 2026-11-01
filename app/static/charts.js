// Small SVG chart kit: win-probability chart, line, column, heatmap, bar and waterfall charts.
// Every chart has a hover layer; they share one tooltip element.
(function () {
  const NS = "http://www.w3.org/2000/svg";
  const tip = () => document.getElementById("tip");

  function el(tag, attrs = {}, parent) {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (parent) parent.appendChild(e);
    return e;
  }

  function showTip(evt, build) {
    const t = tip();
    t.replaceChildren();
    build(t);
    t.hidden = false;
    const pad = 14, w = t.offsetWidth, h = t.offsetHeight;
    let x = evt.clientX + pad, y = evt.clientY + pad;
    if (x + w > window.innerWidth - 8) x = evt.clientX - w - pad;
    if (y + h > window.innerHeight - 8) y = evt.clientY - h - pad;
    t.style.left = Math.max(8, x) + "px";
    t.style.top = Math.max(8, y) + "px";
  }
  function hideTip() { tip().hidden = true; }

  function div(cls, text, parent) {
    const d = document.createElement("div");
    if (cls) d.className = cls;
    if (text != null) d.textContent = text;
    if (parent) parent.appendChild(d);
    return d;
  }

  function nearest(xs, x) {
    let lo = 0, hi = xs.length - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (xs[m] < x) lo = m; else hi = m; }
    return Math.abs(xs[lo] - x) <= Math.abs(xs[hi] - x) ? lo : hi;
  }

  function observe(container, draw) {
    if (container._ro) container._ro.disconnect();
    const ro = new ResizeObserver(() => draw());
    ro.observe(container);
    container._ro = ro;
  }

  // ---------------------------------------------------------------- win probability
  // points: [{t, p, hs, as, per, clk, desc, team}], p = P(home wins)
  function quarterTicks(maxT) {
    const ticks = [];
    for (let q = 0; q < 4; q++) ticks.push({ at: q * 720, mid: q * 720 + 360, label: "Q" + (q + 1) });
    for (let s = 2880, k = 1; s < maxT - 1; s += 300, k++) ticks.push({ at: s, mid: s + 150, label: "OT" + (k > 1 ? k : "") });
    return ticks;
  }

  function winProbChart(container, state) {
    function draw() {
      const { points, home, away, onHover } = state;
      const W = container.clientWidth, H = container.clientHeight;
      if (!W || !H) return;
      const m = { l: 40, r: 12, t: 12, b: 26 };
      const iw = W - m.l - m.r, ih = H - m.t - m.b;
      const last = points.length ? points[points.length - 1].t : 0;
      const maxT = Math.max(2880, last > 2880 ? 2880 + Math.ceil((last - 2880) / 300) * 300 : 2880);
      const X = (t) => m.l + (t / maxT) * iw;
      const Y = (p) => m.t + (1 - p) * ih;

      container.replaceChildren();
      const svg = el("svg", { width: W, height: H, role: "img", "aria-label": "Win probability over the game" }, container);
      const uid = container.id || "wp";

      // grid: horizontal 25/50/75, vertical period boundaries
      const ticks = quarterTicks(maxT);
      for (const p of [0, 0.25, 0.5, 0.75, 1]) {
        el("line", { x1: m.l, x2: m.l + iw, y1: Y(p), y2: Y(p), stroke: p === 0.5 ? "var(--axis)" : "var(--grid)", "stroke-width": 1 }, svg);
        const lab = el("text", { x: m.l - 8, y: Y(p) + 4, "text-anchor": "end", class: "axis-label" }, svg);
        lab.textContent = Math.round(Math.max(p, 1 - p) * 100) + "%";
      }
      for (const tk of ticks) {
        if (tk.at > 0) el("line", { x1: X(tk.at), x2: X(tk.at), y1: m.t, y2: m.t + ih, stroke: "var(--grid)", "stroke-width": 1 }, svg);
        const lab = el("text", { x: X(tk.mid), y: H - 8, "text-anchor": "middle", class: "axis-label" }, svg);
        lab.textContent = tk.label;
      }

      // direct labels for the two sides
      const hl = el("text", { x: m.l + 8, y: m.t + 16, class: "direct-label" }, svg);
      hl.textContent = (home || "HOME") + " favoured ↑";
      const al = el("text", { x: m.l + 8, y: m.t + ih - 8, class: "direct-label" }, svg);
      al.textContent = (away || "AWAY") + " favoured ↓";

      if (points.length) {
        const defs = el("defs", {}, svg);
        const top = el("clipPath", { id: uid + "-top" }, defs);
        el("rect", { x: 0, y: 0, width: W, height: Y(0.5) }, top);
        const bot = el("clipPath", { id: uid + "-bot" }, defs);
        el("rect", { x: 0, y: Y(0.5), width: W, height: H }, bot);

        let line = "";
        points.forEach((pt, i) => { line += (i ? "L" : "M") + X(pt.t).toFixed(1) + "," + Y(pt.p).toFixed(1); });
        const area = line + `L${X(last).toFixed(1)},${Y(0.5)}L${X(points[0].t).toFixed(1)},${Y(0.5)}Z`;

        for (const [clip, color] of [["top", "var(--home)"], ["bot", "var(--away)"]]) {
          el("path", { d: area, fill: color, style: "fill-opacity: var(--fill-alpha)", "clip-path": `url(#${uid}-${clip})` }, svg);
          el("path", { d: line, fill: "none", stroke: color, "stroke-width": 2, "stroke-linejoin": "round", "clip-path": `url(#${uid}-${clip})` }, svg);
        }
        if (points.length === 1) {
          const w = el("text", { x: m.l + iw / 2, y: Y(0.5) - 10, "text-anchor": "middle", class: "axis-label" }, svg);
          w.textContent = "Waiting for tip-off · pre-game estimate shown";
        }
        // numbered markers for the biggest swings (matching the list below the chart)
        for (const mk of state.marks || []) {
          const g = el("g", { class: "mark" }, svg);
          el("circle", { cx: X(mk.t), cy: Y(mk.p), r: 8, fill: "var(--surface-1)", stroke: "var(--text-secondary)", "stroke-width": 1.5 }, g);
          el("text", { x: X(mk.t), y: Y(mk.p) + 3.5, "text-anchor": "middle", class: "mark-label" }, g).textContent = mk.n;
        }
        // current-state marker
        const lp = points[points.length - 1];
        el("circle", { cx: X(lp.t), cy: Y(lp.p), r: 4.5, fill: lp.p >= 0.5 ? "var(--home)" : "var(--away)", stroke: "var(--surface-1)", "stroke-width": 2 }, svg);
      }

      // hover layer
      const cross = el("line", { y1: m.t, y2: m.t + ih, stroke: "var(--muted)", "stroke-width": 1, visibility: "hidden" }, svg);
      const dot = el("circle", { r: 5, stroke: "var(--surface-1)", "stroke-width": 2, visibility: "hidden" }, svg);
      const hit = el("rect", { x: m.l, y: m.t, width: iw, height: ih, fill: "transparent" }, svg);
      const xs = points.map((p) => p.t);
      hit.addEventListener("pointermove", (evt) => {
        if (!points.length) return;
        const r = svg.getBoundingClientRect();
        const t = ((evt.clientX - r.left - m.l) / iw) * maxT;
        const pt = points[nearest(xs, t)];
        cross.setAttribute("x1", X(pt.t)); cross.setAttribute("x2", X(pt.t)); cross.setAttribute("visibility", "visible");
        dot.setAttribute("cx", X(pt.t)); dot.setAttribute("cy", Y(pt.p));
        dot.setAttribute("fill", pt.p >= 0.5 ? "var(--home)" : "var(--away)"); dot.setAttribute("visibility", "visible");
        showTip(evt, (t) => {
          const fav = pt.p >= 0.5;
          div("tv", `${fav ? home : away} ${(100 * (fav ? pt.p : 1 - pt.p)).toFixed(1)}%`, t);
          div("tm", `${periodName(pt.per)} ${pt.clk} · ${away} ${pt.as} – ${pt.hs} ${home}`, t);
          if (pt.desc) div("td", pt.desc, t);
        });
        if (onHover) onHover(pt);
      });
      hit.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); dot.setAttribute("visibility", "hidden"); hideTip(); });
    }
    observe(container, draw);
    draw();
    return { redraw: draw };
  }

  function periodName(per) { return per <= 4 ? "Q" + per : per === 5 ? "OT" : "OT" + (per - 4); }

  // ---------------------------------------------------------------- generic lines
  // opts: {series: [{name, color, pts: [[x, y]], dots}], x: {min, max, ticks, fmt}, y: {min, max, ticks, fmt},
  //        ref: [[x1, y1], [x2, y2]], xLabel}
  function lineChart(container, opts) {
    function draw() {
      const W = container.clientWidth, H = container.clientHeight;
      if (!W || !H) return;
      const m = { l: 46, r: 14, t: 10, b: 30 };
      const iw = W - m.l - m.r, ih = H - m.t - m.b;
      const X = (v) => m.l + ((v - opts.x.min) / (opts.x.max - opts.x.min)) * iw;
      const Y = (v) => m.t + (1 - (v - opts.y.min) / (opts.y.max - opts.y.min)) * ih;
      container.replaceChildren();
      const svg = el("svg", { width: W, height: H, role: "img", "aria-label": opts.label || "chart" }, container);

      opts.y.ticks.forEach((v) => {
        el("line", { x1: m.l, x2: m.l + iw, y1: Y(v), y2: Y(v), stroke: v === opts.y.min ? "var(--axis)" : "var(--grid)" }, svg);
        el("text", { x: m.l - 8, y: Y(v) + 4, "text-anchor": "end", class: "axis-label" }, svg).textContent = opts.y.fmt(v);
      });
      opts.x.ticks.forEach((v) => {
        el("text", { x: X(v), y: H - 10, "text-anchor": "middle", class: "axis-label" }, svg).textContent = opts.x.fmt(v);
      });
      if (opts.ref) {
        const [[a, b], [c, d]] = opts.ref;
        el("line", { x1: X(a), y1: Y(b), x2: X(c), y2: Y(d), stroke: "var(--axis)", "stroke-dasharray": "4 4" }, svg);
      }
      for (const s of opts.series) {
        const d = s.pts.map(([x, y], i) => (i ? "L" : "M") + X(x).toFixed(1) + "," + Y(y).toFixed(1)).join("");
        el("path", { d, fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round", "stroke-dasharray": s.dash || "none" }, svg);
        if (s.dots) s.pts.forEach(([x, y]) => el("circle", { cx: X(x), cy: Y(y), r: 4, fill: s.color, stroke: "var(--surface-1)", "stroke-width": 2 }, svg));
      }

      for (const mk of opts.marks || []) {
        el("circle", { cx: X(mk.x), cy: Y(mk.y), r: 6, fill: mk.color, stroke: "var(--surface-1)", "stroke-width": 2 }, svg);
        if (mk.label) el("text", { x: X(mk.x) + 10, y: Y(mk.y) - 8, class: "direct-label" }, svg).textContent = mk.label;
      }
      if (opts.yLabel) el("text", { x: m.l + 4, y: m.t + 12, class: "axis-label" }, svg).textContent = opts.yLabel;

      const xs = opts.series[0].pts.map((p) => p[0]);
      const cross = el("line", { y1: m.t, y2: m.t + ih, stroke: "var(--muted)", visibility: "hidden" }, svg);
      const dots = opts.series.map((s) => el("circle", { r: 4.5, fill: s.color, stroke: "var(--surface-1)", "stroke-width": 2, visibility: "hidden" }, svg));
      const hit = el("rect", { x: m.l, y: m.t, width: iw, height: ih, fill: "transparent" }, svg);
      hit.addEventListener("pointermove", (evt) => {
        const r = svg.getBoundingClientRect();
        const xv = opts.x.min + ((evt.clientX - r.left - m.l) / iw) * (opts.x.max - opts.x.min);
        const i = nearest(xs, xv);
        const x = xs[i];
        cross.setAttribute("x1", X(x)); cross.setAttribute("x2", X(x)); cross.setAttribute("visibility", "visible");
        opts.series.forEach((s, k) => {
          const p = s.pts[i];
          if (!p) return;
          dots[k].setAttribute("cx", X(p[0])); dots[k].setAttribute("cy", Y(p[1])); dots[k].setAttribute("visibility", "visible");
        });
        showTip(evt, (t) => {
          div("tm", (opts.x.tipFmt || opts.x.fmt)(x), t);
          opts.series.forEach((s) => {
            const p = s.pts[i];
            if (!p) return;
            const row = div("tr", null, t);
            const k = document.createElement("i"); k.style.background = s.color; row.appendChild(k);
            const b = document.createElement("b"); b.textContent = (opts.y.tipFmt || opts.y.fmt)(p[1]); row.appendChild(b);
            row.appendChild(document.createTextNode(s.name));
          });
          if (opts.extra) opts.extra(i, t);
        });
      });
      hit.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); dots.forEach((d) => d.setAttribute("visibility", "hidden")); hideTip(); });
    }
    observe(container, draw);
    draw();
  }

  function legend(container, items, box = false) {
    container.replaceChildren();
    for (const it of items) {
      const s = document.createElement("span");
      const i = document.createElement("i");
      if (box) i.className = "box";
      i.style.background = it.color;
      s.appendChild(i);
      s.appendChild(document.createTextNode(it.name));
      container.appendChild(s);
    }
  }

  // ---------------------------------------------------------------- columns
  // opts: {cats: [label], series: [{name, color, values}], y: {min, max, ticks, fmt, tipFmt}, ref: {y, label}, tipExtra(i, t)}
  function columnChart(container, opts) {
    function draw() {
      const W = container.clientWidth, H = container.clientHeight;
      if (!W || !H) return;
      const m = { l: 46, r: 10, t: 10, b: 30 };
      const iw = W - m.l - m.r, ih = H - m.t - m.b;
      const n = opts.cats.length, k = opts.series.length;
      const Y = (v) => m.t + (1 - (v - opts.y.min) / (opts.y.max - opts.y.min)) * ih;
      const band = iw / n, gap = 2;
      const bw = Math.max(2, Math.min(28, (band * 0.72 - gap * (k - 1)) / k));
      container.replaceChildren();
      const svg = el("svg", { width: W, height: H, role: "img", "aria-label": opts.label || "column chart" }, container);
      opts.y.ticks.forEach((v) => {
        el("line", { x1: m.l, x2: m.l + iw, y1: Y(v), y2: Y(v), stroke: v === opts.y.min ? "var(--axis)" : "var(--grid)" }, svg);
        el("text", { x: m.l - 8, y: Y(v) + 4, "text-anchor": "end", class: "axis-label" }, svg).textContent = opts.y.fmt(v);
      });
      const every = Math.ceil(n / Math.max(1, Math.floor(iw / 46)));
      opts.cats.forEach((c, i) => {
        if (i % every) return;
        el("text", { x: m.l + band * (i + 0.5), y: H - 10, "text-anchor": "middle", class: "axis-label" }, svg).textContent = c;
      });
      const base = Y(Math.max(opts.y.min, 0));
      opts.series.forEach((s, j) => s.values.forEach((v, i) => {
        if (v == null) return;
        const x = m.l + band * (i + 0.5) - (k * bw + (k - 1) * gap) / 2 + j * (bw + gap);
        const y = Y(v), hgt = Math.max(1, base - y);
        // rounded data end, square at the baseline
        const r = Math.min(4, bw / 2, hgt);
        el("path", { d: `M${x},${base}V${y + r}Q${x},${y} ${x + r},${y}H${x + bw - r}Q${x + bw},${y} ${x + bw},${y + r}V${base}Z`, fill: s.color }, svg);
      }));
      if (opts.ref) {
        el("line", { x1: m.l, x2: m.l + iw, y1: Y(opts.ref.y), y2: Y(opts.ref.y), stroke: "var(--text-secondary)", "stroke-dasharray": "4 4" }, svg);
        if (opts.ref.label) el("text", { x: m.l + iw - 4, y: Y(opts.ref.y) - 6, "text-anchor": "end", class: "direct-label" }, svg).textContent = opts.ref.label;
      }
      const hi = el("rect", { y: m.t, height: ih, fill: "var(--text-primary)", opacity: 0.06, visibility: "hidden" }, svg);
      const hit = el("rect", { x: m.l, y: m.t, width: iw, height: ih, fill: "transparent" }, svg);
      hit.addEventListener("pointermove", (evt) => {
        const r = svg.getBoundingClientRect();
        const i = Math.min(n - 1, Math.max(0, Math.floor((evt.clientX - r.left - m.l) / band)));
        hi.setAttribute("x", m.l + band * i); hi.setAttribute("width", band); hi.setAttribute("visibility", "visible");
        showTip(evt, (t) => {
          div("tm", opts.tipCat ? opts.tipCat(i) : opts.cats[i], t);
          opts.series.forEach((s) => {
            if (s.values[i] == null) return;
            const row = div("tr", null, t);
            const kk = document.createElement("i"); kk.className = "box"; kk.style.background = s.color; row.appendChild(kk);
            const b = document.createElement("b"); b.textContent = (opts.y.tipFmt || opts.y.fmt)(s.values[i]); row.appendChild(b);
            row.appendChild(document.createTextNode(s.name));
          });
          if (opts.tipExtra) opts.tipExtra(i, t);
        });
      });
      hit.addEventListener("pointerleave", () => { hi.setAttribute("visibility", "hidden"); hideTip(); });
    }
    observe(container, draw);
    draw();
  }

  // ---------------------------------------------------------------- heatmap
  // P(home win) over lead (x) and minutes left (y). Diverging: away hue < 50% < home hue, neutral midpoint.
  // opts: {xs, ys, z: [row per y][col per x], marker: {x, y}, home, away, onPick(x, y)}
  function divergingFill(p) {
    const k = Math.min(1, Math.abs(p - 0.5) * 2);
    const amt = Math.round(100 * Math.pow(k, 0.8));
    return `color-mix(in oklab, ${p >= 0.5 ? "var(--home)" : "var(--away)"} ${amt}%, var(--div-mid))`;
  }
  function heatmap(container, opts) {
    function draw() {
      const W = container.clientWidth, H = container.clientHeight;
      if (!W || !H) return;
      const m = { l: 46, r: 10, t: 8, b: 30 };
      const iw = W - m.l - m.r, ih = H - m.t - m.b;
      const nx = opts.xs.length, ny = opts.ys.length;
      const cw = iw / nx, ch = ih / ny;
      container.replaceChildren();
      const svg = el("svg", { width: W, height: H, role: "img", "aria-label": opts.label || "heatmap" }, container);
      opts.z.forEach((row, j) => row.forEach((p, i) => {
        el("rect", { x: m.l + i * cw, y: m.t + j * ch, width: cw + 0.6, height: ch + 0.6, style: `fill:${divergingFill(p)}` }, svg);
      }));
      // y: minutes left (top = tip-off), x: lead
      opts.ys.forEach((v, j) => {
        if (v % 12) return;
        el("text", { x: m.l - 8, y: m.t + j * ch + ch / 2 + 4, "text-anchor": "end", class: "axis-label" }, svg).textContent = v === 48 ? "Tip" : v === 0 ? "End" : `${v}m`;
        if (v && v !== 48) el("line", { x1: m.l, x2: m.l + iw, y1: m.t + j * ch + ch / 2, y2: m.t + j * ch + ch / 2, stroke: "var(--surface-1)", "stroke-opacity": 0.5 }, svg);
      });
      opts.xs.forEach((v, i) => {
        if (v % 5) return;
        el("text", { x: m.l + i * cw + cw / 2, y: H - 10, "text-anchor": "middle", class: "axis-label" }, svg).textContent = v > 0 ? "+" + v : v;
      });
      const mid = opts.xs.indexOf(0);
      if (mid >= 0) el("line", { x1: m.l + mid * cw + cw / 2, x2: m.l + mid * cw + cw / 2, y1: m.t, y2: m.t + ih, stroke: "var(--surface-1)", "stroke-opacity": 0.6 }, svg);
      if (opts.marker) {
        const i = opts.xs.indexOf(Math.max(opts.xs[0], Math.min(opts.xs[nx - 1], Math.round(opts.marker.x))));
        const j = opts.ys.indexOf(Math.max(0, Math.min(48, Math.round(opts.marker.y))));
        if (i >= 0 && j >= 0) el("circle", { cx: m.l + i * cw + cw / 2, cy: m.t + j * ch + ch / 2, r: 6, fill: "none", stroke: "var(--text-primary)", "stroke-width": 2 }, svg);
      }
      const box = el("rect", { width: cw, height: ch, fill: "none", stroke: "var(--text-primary)", "stroke-width": 1.5, visibility: "hidden" }, svg);
      const hit = el("rect", { x: m.l, y: m.t, width: iw, height: ih, fill: "transparent", style: opts.onPick ? "cursor:pointer" : "" }, svg);
      const cell = (evt) => {
        const r = svg.getBoundingClientRect();
        return [Math.min(nx - 1, Math.max(0, Math.floor((evt.clientX - r.left - m.l) / cw))),
                Math.min(ny - 1, Math.max(0, Math.floor((evt.clientY - r.top - m.t) / ch)))];
      };
      hit.addEventListener("pointermove", (evt) => {
        const [i, j] = cell(evt);
        box.setAttribute("x", m.l + i * cw); box.setAttribute("y", m.t + j * ch); box.setAttribute("visibility", "visible");
        const p = opts.z[j][i], lead = opts.xs[i];
        showTip(evt, (t) => {
          const fav = p >= 0.5;
          div("tv", `${fav ? opts.home : opts.away} ${(100 * (fav ? p : 1 - p)).toFixed(1)}%`, t);
          div("tm", `${opts.home} ${lead > 0 ? "up " + lead : lead < 0 ? "down " + -lead : "tied"} · ${opts.ys[j]} min left`, t);
          if (opts.onPick) div("td", "Click to load this state", t);
        });
      });
      hit.addEventListener("pointerleave", () => { box.setAttribute("visibility", "hidden"); hideTip(); });
      if (opts.onPick) hit.addEventListener("click", (evt) => { const [i, j] = cell(evt); opts.onPick(opts.xs[i], opts.ys[j]); });
    }
    observe(container, draw);
    draw();
  }

  // ---------------------------------------------------------------- horizontal bars (HTML)
  // items: [{label, value, color, note, tip}]; opts: {max, fmt, diverging}
  function hbars(container, items, opts = {}) {
    container.replaceChildren();
    container.classList.add("hbars");
    if (opts.diverging) container.classList.add("div");
    const max = opts.max || Math.max(...items.map((d) => Math.abs(d.value)), 1e-9);
    for (const d of items) {
      const row = div("hb", null, container);
      div("hb-l", d.label, row);
      const track = div("hb-t", null, row);
      const bar = div("hb-b", null, track);
      const w = (Math.abs(d.value) / max) * (opts.diverging ? 50 : 100);
      bar.style.width = Math.max(0.5, w) + "%";
      bar.style.background = d.color || "var(--s1)";
      if (opts.diverging) bar.style[d.value >= 0 ? "left" : "right"] = "50%";
      div("hb-v", (opts.fmt || String)(d.value), row);
      if (d.tip) {
        row.addEventListener("pointermove", (evt) => showTip(evt, (t) => d.tip(t, div)));
        row.addEventListener("pointerleave", hideTip);
      }
    }
  }

  // ---------------------------------------------------------------- waterfall (HTML)
  // From a coin flip to the model's number, one input group at a time.
  // steps: [{label, delta, p}] with p = P(home) after the step; teams: {home, away}
  function waterfall(container, steps, teams) {
    container.replaceChildren();
    container.classList.add("wf");
    const row = (label, from, to, valueText, cls) => {
      const r = div("wf-r" + (cls ? " " + cls : ""), null, container);
      div("wf-l", label, r);
      const t = div("wf-t", null, r);
      div("wf-mid", null, t);
      const lo = Math.min(from, to), hi = Math.max(from, to);
      const b = div("wf-b", null, t);
      b.style.left = 100 * lo + "%";
      b.style.width = Math.max(0.4, 100 * (hi - lo)) + "%";
      b.style.background = to >= from ? "var(--home)" : "var(--away)";
      if (cls === "total") b.style.background = to >= 0.5 ? "var(--home)" : "var(--away)";
      div("wf-v", valueText, r);
    };
    row("Coin flip", 0.5, 0.5, "50.0%", "start");
    let prev = 0.5;
    for (const s of steps) {
      const d = s.p - prev;
      const who = d >= 0 ? teams.home : teams.away;
      row(s.label, prev, s.p, Math.abs(d) < 0.0005 ? "±0.0" : `${who} +${(100 * Math.abs(d)).toFixed(1)}`);
      prev = s.p;
    }
    const fav = prev >= 0.5 ? teams.home : teams.away;
    row("Model", 0.5, prev, `${fav} ${(100 * Math.max(prev, 1 - prev)).toFixed(1)}%`, "total");
  }

  window.Charts = { winProbChart, lineChart, columnChart, heatmap, hbars, waterfall, legend, periodName, divergingFill, showTip, hideTip, div };
})();
