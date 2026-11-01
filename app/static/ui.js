// Small DOM helpers shared by the Games, Track record and Model views.
(function () {
  const $ = (id) => document.getElementById(id);

  // h("div.card", {onclick}, child, "text", ...) -> element
  function h(tag, attrs, ...kids) {
    const [name, ...cls] = tag.split(".");
    const e = document.createElement(name || "div");
    if (cls.length) e.className = cls.join(" ");
    if (attrs && (typeof attrs !== "object" || attrs instanceof Node || Array.isArray(attrs))) { kids.unshift(attrs); attrs = null; }
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v == null || v === false) continue;
      if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
      else if (k === "style" && typeof v === "object") Object.assign(e.style, v);
      else if (k === "class") e.className += (e.className ? " " : "") + v;
      else e.setAttribute(k, v === true ? "" : v);
    }
    for (const k of kids.flat()) if (k != null && k !== false) e.append(k instanceof Node ? k : String(k));
    return e;
  }

  const pct = (p, d = 1) => (100 * p).toFixed(d) + "%";
  const pts = (v, d = 1) => (v > 0 ? "+" : v < 0 ? "−" : "±") + Math.abs(v).toFixed(d);
  const num = (v) => v.toLocaleString();

  function tile(parent, k, v, s, opts = {}) {
    const t = h("div.tile" + (opts.accent ? ".accent" : ""), null, h("div.k", k), h("div.v", v));
    if (s) t.appendChild(h("div.s", s));
    if (opts.title) t.title = opts.title;
    parent.appendChild(t);
    return t;
  }

  // rows: array of arrays (cells may be Nodes); opts.head, opts.cls(rowIndex) -> class, opts.onclick(rowIndex)
  function table(el, head, rows, opts = {}) {
    el.replaceChildren();
    const tr = el.insertRow();
    head.forEach((t) => tr.appendChild(h("th", t)));
    rows.forEach((r, i) => {
      const row = el.insertRow();
      const c = opts.cls && opts.cls(i);
      if (c) row.className = c;
      if (opts.onclick) { row.classList.add("link"); row.onclick = () => opts.onclick(i); }
      r.forEach((v) => { const td = row.insertCell(); td.append(v instanceof Node ? v : String(v)); });
    });
  }

  const logoUrl = (id) => `https://cdn.nba.com/logos/nba/${id}/primary/L/logo.svg`;
  function logo(container, team, cls) {
    container.replaceChildren();
    const img = new Image();
    img.alt = team.tricode;
    img.src = logoUrl(team.id);
    if (cls) img.className = cls;
    img.onerror = () => {
      const m = h("div.mono" + (cls ? "." + cls : ""));
      if (cls !== "mini-logo") m.textContent = team.tricode;  // card rows already show the tricode
      img.replaceWith(m);
    };
    container.appendChild(img);
  }

  // Legend swatch + label, for inline keys inside text.
  const key = (color, label) => h("span.ikey", null, h("i", { style: { background: color } }), label);

  // ✓ / ✗ marker that never relies on colour alone.
  const verdict = (hit) => h("span.verdict." + (hit === 1 ? "hit" : hit === 0 ? "miss" : "push"), hit === 1 ? "✓" : hit === 0 ? "✗" : "½");

  async function getJSON(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
    return r.json();
  }

  window.UI = { $, h, pct, pts, num, tile, table, logo, key, verdict, getJSON };
})();
