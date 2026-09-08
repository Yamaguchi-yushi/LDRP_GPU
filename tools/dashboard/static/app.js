/* LDRP experiment dashboard
 *
 * GET /api/train  学習の進捗 (キャッシュを読むだけなので即返る)
 * GET /api/eval   評価結果 (results/summary.csv)
 * POST /api/collect  全マシンを収集 (重いのでバックグラウンド)
 */
"use strict";

const COLORS = ["#0969da", "#1a7f37", "#9a6700", "#cf222e",
                "#8250df", "#0f6b6b", "#bc4c00"];
let TRAIN = null, EVAL = null, tab = "train";
let evalSortCol = null, evalSortAsc = true;

const $ = id => document.getElementById(id);
const esc = s => String(s == null ? "" : s)
  .replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const M = v => v == null ? "?" : (v / 1e6).toFixed(2);

function when(iso) {
  if (!iso) return "-";
  const d = typeof iso === "number" ? new Date(iso * 1000) : new Date(iso);
  if (isNaN(d)) return "-";
  const p = n => String(n).padStart(2, "0");
  return `${p(d.getMonth() + 1)}/${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
}
function dur(s) {
  if (s == null) return "-";
  s = Math.max(0, s | 0);
  return s < 3600 ? Math.round(s / 60) + "m"
    : Math.floor(s / 3600) + "h" + String(Math.round((s % 3600) / 60)).padStart(2, "0") + "m";
}
function num(v, nd) {
  if (v == null) return "-";
  const s = (nd == null ? (Math.abs(v) >= 1 ? v.toFixed(2) : v.toFixed(4)) : v.toFixed(nd));
  return s.indexOf(".") >= 0 ? s.replace(/\.?0+$/, "") : s;
}
async function api(path, body) {
  const r = await fetch(path, body
    ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
    : {});
  return r.json();
}
function fillSelect(el, values, label) {
  const cur = el.value;
  el.innerHTML = `<option value="">${label || "(all)"}</option>` +
    values.map(v => `<option>${esc(v)}</option>`).join("");
  if (values.map(String).includes(cur)) el.value = cur;
}
const uniq = (arr, k) => [...new Set(arr.map(d => d[k]).filter(v => v != null && v !== ""))]
  .sort((a, b) => typeof a === "number" ? a - b : String(a).localeCompare(String(b)));

/* ── タブ ─────────────────────────────────────────────── */
function setTab(name) {
  tab = name;
  $("pane-train").hidden = name !== "train";
  $("pane-eval").hidden = name !== "eval";
  $("tab-train").classList.toggle("on", name === "train");
  $("tab-eval").classList.toggle("on", name === "eval");
  if (name === "eval" && !EVAL) loadEval();
}
$("tab-train").onclick = () => setTab("train");
$("tab-eval").onclick = () => setTab("eval");

/* ── 学習の進捗 ───────────────────────────────────────── */
async function loadTrain() {
  TRAIN = await api("/api/train");
  const runs = TRAIN.runs || [];
  fillSelect($("t-machine"), uniq(runs, "machine"));
  fillSelect($("t-map"), uniq(runs, "map"));
  fillSelect($("t-agents"), uniq(runs, "agents"));
  fillSelect($("t-algo"), uniq(runs, "algo"));
  renderStamp();
  renderTrain();
}
["t-machine", "t-map", "t-agents", "t-algo", "t-state"]
  .forEach(id => $(id).onchange = renderTrain);

function renderStamp() {
  const s = TRAIN || {};
  $("stamp").textContent = "収集 " + when(s.collected_at) + (s.busy ? "  (収集中…)" : "");
  $("errors").innerHTML = (s.errors || [])
    .map(e => `<div class="err">⚠ ${esc(e)}</div>`).join("");
  $("collect").disabled = !!s.busy;
}

function trainRows() {
  const g = id => $(id).value;
  return (TRAIN.runs || []).filter(r =>
    (!g("t-machine") || r.machine === g("t-machine")) &&
    (!g("t-map") || r.map === g("t-map")) &&
    (!g("t-agents") || String(r.agents) === g("t-agents")) &&
    (!g("t-algo") || r.algo === g("t-algo")) &&
    (!g("t-state") || r.state === g("t-state")));
}

function renderTrain() {
  if (!TRAIN) return;
  const R = trainRows();
  const c = {};
  R.forEach(r => c[r.state] = (c[r.state] || 0) + 1);
  $("t-counts").textContent =
    Object.entries(c).map(([k, v]) => `${k}=${v}`).join("  ") + `   計 ${R.length}`;

  // machines
  const byM = {};
  R.forEach(r => {
    const m = byM[r.machine] || (byM[r.machine] = { run: 0, done: 0, bad: 0, eta: null });
    if (r.state === "running") { m.run++; if (r.eta && (!m.eta || r.eta < m.eta)) m.eta = r.eta; }
    else if (r.state === "done") m.done++;
    else if (["stalled", "failed", "short"].includes(r.state)) m.bad++;
  });
  $("t-mach").innerHTML =
    "<tr><th>machine</th><th>実行中</th><th>予約待ち</th><th>完了</th>"
    + "<th>異常</th><th>次の完了</th></tr>"
    + (Object.keys(byM).sort().map(k => {
      const m = byM[k], b = (TRAIN.machines || {})[k];
      const res = b ? `+${b.reserved}${b.unknown ? "?" : ""} <span class="mut">(train.py x${b.batches})</span>` : "-";
      return `<tr><td>${esc(k)}</td><td class="num">${m.run}</td>
        <td>${res}</td><td class="num">${m.done}</td>
        <td class="num ${m.bad ? "err" : "mut"}">${m.bad}</td>
        <td class="mut">${when(m.eta)}</td></tr>`;
    }).join("") || `<tr><td class="mut" colspan="6">なし</td></tr>`);

  // running now
  const run = R.filter(r => r.state === "running")
    .sort((a, b) => (a.eta || "9") < (b.eta || "9") ? -1 : 1);
  $("t-run").innerHTML =
    "<tr><th>進捗</th><th>t_env</th><th>残り</th><th>終了予定</th><th>machine</th>"
    + "<th>条件</th><th>seed</th><th>経過</th></tr>"
    + (run.length ? run.map(r => `<tr>
        <td><span class="pb"><i style="width:${((r.progress || 0) * 100).toFixed(0)}%"></i></span>
            ${((r.progress || 0) * 100).toFixed(0)}%</td>
        <td class="mut num">${M(r.t_last)}/${M(r.t_max)}M</td>
        <td class="num">${dur(r.remaining_sec)}</td><td>${when(r.eta)}</td>
        <td>${esc(r.machine)}</td>
        <td>${r.agents}ag ${esc(r.map)} ${esc(r.algo)} ${esc(r.setting)}</td>
        <td class="mut">${esc(r.seed)}${r.in_plan ? "" : ' <span class="wrn">計画外</span>'}</td>
        <td class="mut">${esc(r.duration)}</td></tr>`).join("")
      : `<tr><td class="mut" colspan="8">なし</td></tr>`);

  // conditions — **表は計画 (tools/plan.md) から作る**。実績はそこに埋める。
  // 計画に無い run は表に出さない (running セクションには出る)
  const WANT_DEFAULT = 5;
  const ALGO = a => String(a || "").toUpperCase();
  const g = id => $(id).value;
  const plan = (TRAIN.plan || []).filter(c =>
    (!g("t-map") || c.map === g("t-map")) &&
    (!g("t-agents") || String(c.agents) === g("t-agents")) &&
    (!g("t-algo") || c.algo === g("t-algo")) &&
    (!g("t-machine") || c.slots.some(s => s.machine === g("t-machine")
                                       || (s.run && s.run.machine === g("t-machine")))) &&
    (!g("t-state") || c.slots.some(s => s.run && s.run.state === g("t-state"))));

  const nOut = (TRAIN.runs || []).filter(r => !r.in_plan).length;
  let html = "";
  if (!TRAIN.plan || !TRAIN.plan.length)
    html += `<div class="wrn">計画ファイルが読めていません: ${esc(TRAIN.plan_file || "")}</div>`;
  else if (nOut)
    html += `<div class="mut">計画外の run: ${nOut} 件 `
          + `<span class="mut">(表には出しません。running / machines には出ます)</span></div>`;

  let head = null;
  plan.forEach(c => {
    if (c.label !== head) {
      if (head !== null) html += `</table></div>`;
      head = c.label;
      html += `<h3>${esc(head)}</h3><div class="wrap"><table class="cond">
        <tr><th>seed</th><th>machine</th><th>setting</th><th>algorithm</th>
            <th>task arrival</th><th>task assign</th><th>dynamic</th><th>状態</th></tr>`;
    }
    const want = c.want || WANT_DEFAULT;
    const done = c.slots.filter(s => s.run && s.run.state === "done"
                                     && !s.run.odd_params).length;
    // 5 seed そろっていれば、失敗した run は出さない (メモにも書かない運用に合わせる)。
    // 何件隠したかは条件行に出すので、黙って消えるわけではない
    const filled = done >= want;
    let slots = c.slots, hidden = 0;
    if (filled) {
      const before = slots.length;
      slots = slots.filter(s => s.run && !s.run.odd_params
                                && (s.run.state === "done" || s.run.state === "running"));
      hidden = before - slots.length;
    }
    const odd = filled ? 0 : c.slots.filter(s => s.run && s.run.odd_params).length;
    const tmax = slots.find(s => s.run && s.run.t_max_ok === false);
    const dyn = c.dynamic == null ? "" : (c.dynamic ? "T" : "F");

    html += `<tr class="condrow"><td></td><td></td>
      <td>${esc(c.setting)}</td><td>${esc(ALGO(c.algo))}</td>
      <td>${esc(c.task_arrival)}</td><td>${esc(c.task_assign || "TP")}</td>
      <td>${esc(dyn)}</td>
      <td class="${filled ? "c-ok" : "wrn"}">${done}/${want} done${
        odd ? ` <span class="wrn">⚠要再実行 ${odd}</span>` : ""}${
        tmax ? ` <span class="wrn">⚠t_max</span>` : ""}${
        hidden ? ` <span class="mut">(失敗 ${hidden} 件を非表示)</span>` : ""}</td></tr>`;

    slots.forEach(sl => {
      const r = sl.run;
      let st;
      if (!r) st = `<span class="mut">未実行</span>`;
      else {
        const pct = ((r.progress || 0) * 100).toFixed(0);
        const steps = `${M(r.t_last)}M / ${M(r.t_max)}M`;
        if (r.state === "done")
          st = `<span class="c-ok">✔ done</span> <span class="mut">${M(r.t_max)}M</span>`;
        else if (r.state === "running")
          st = `<span class="pb"><i style="width:${pct}%"></i></span> ${pct}%`
             + ` <span class="steps">${steps}</span>`
             + ` <span class="mut">残り ${dur(r.remaining_sec)} → ${when(r.eta)}</span>`;
        else
          st = `<span class="err">✖ ${esc(r.state)}</span>`
             + ` <span class="mut">${steps} で停止</span>`;
        if (r.odd_params) st += ` <span class="wrn">params✗</span>`;
      }
      const sr = "";
      html += `<tr><td class="${sl.unplanned_seed ? "wrn" : (r ? "" : "mut")}">${
          esc(sl.seed || "—")}${sl.unplanned_seed ? " *" : ""}</td>
        <td class="mut">${esc((r && r.machine) || sl.machine || "")}</td>
        <td colspan="4"></td><td>${esc(sr)}</td><td>${st}</td></tr>`;
    });
  });
  if (head !== null) html += `</table></div>`;
  $("t-conds").innerHTML = html || `<div class="mut">計画に一致する条件がありません</div>`;
}

/* ── 評価結果 ─────────────────────────────────────────── */
async function loadEval() {
  EVAL = await api("/api/eval");
  if (!EVAL.available) {
    $("e-tbl").innerHTML = `<tr><td class="err">${esc(EVAL.error)}</td></tr>`;
    return;
  }
  const C = EVAL.conditions;
  fillSelect($("e-map"), uniq(C, "map"));
  fillSelect($("e-n"), uniq(C, "n"));
  fillSelect($("e-env"), uniq(C, "env"));
  fillSelect($("e-planner"), uniq(C, "planner"));
  fillSelect($("e-tag"), uniq(C, "method_tag"));
  fillSelect($("e-alloc"), uniq(C, "allocator"));
  $("e-metric").innerHTML = (EVAL.metrics || []).map(m => `<option>${esc(m)}</option>`).join("");
  const pref = (EVAL.metrics || []).indexOf("task_completion");
  if (pref >= 0) $("e-metric").selectedIndex = pref;
  renderEval();
}
["e-map", "e-n", "e-env", "e-planner", "e-tag", "e-alloc", "e-metric", "e-log"]
  .forEach(id => $(id).onchange = renderEval);

function evalRows() {
  const g = id => $(id).value;
  return (EVAL.conditions || []).filter(d =>
    (!g("e-map") || d.map === g("e-map")) &&
    (!g("e-n") || String(d.n) === g("e-n")) &&
    (!g("e-env") || d.env === g("e-env")) &&
    (!g("e-planner") || d.planner === g("e-planner")) &&
    (!g("e-tag") || d.method_tag === g("e-tag")) &&
    (!g("e-alloc") || d.allocator === g("e-alloc")));
}
window.evalSortBy = k => {
  if (evalSortCol === k) evalSortAsc = !evalSortAsc;
  else { evalSortCol = k; evalSortAsc = true; }
  renderEval();
};

function renderEval() {
  if (!EVAL || !EVAL.available) return;
  const metric = $("e-metric").value;
  let rows = evalRows();
  if (evalSortCol) {
    rows = rows.slice().sort((a, b) => {
      const va = evalSortCol === "metric" ? ((a.metrics[metric] || {}).mean) : a[evalSortCol];
      const vb = evalSortCol === "metric" ? ((b.metrics[metric] || {}).mean) : b[evalSortCol];
      const x = va == null ? -Infinity : va, y = vb == null ? -Infinity : vb;
      return (x < y ? -1 : x === y ? 0 : 1) * (evalSortAsc ? 1 : -1);
    });
  }
  $("e-counts").textContent = rows.length + " 条件";

  const cols = [["map", "map"], ["n", "N"], ["env", "env"], ["planner", "planner"],
                ["method_tag", "tag"], ["allocator", "alloc"], ["metric", metric]];
  $("e-tbl").innerHTML =
    "<tr>" + cols.map(([k, l]) =>
      `<th class="sortable" onclick="evalSortBy('${k}')">${esc(l)}${evalSortCol === k ? (evalSortAsc ? " ▲" : " ▼") : ""}</th>`).join("")
    + "<th>n</th><th>per-seed</th></tr>"
    + rows.map(d => {
      const st = d.metrics[metric] || {};
      const cls = st.n === 1 ? "one" : (st.n < 5 ? "thin" : "");
      return `<tr><td>${esc(d.map)}</td><td class="num">${d.n}</td><td>${esc(d.env)}</td>
        <td>${esc(d.planner)}</td><td>${esc(d.method_tag || "-")}</td>
        <td>${esc(d.allocator)}</td>
        <td class="num">${num(st.mean)} ± ${num(st.std)}</td>
        <td class="num ${cls}">${st.n == null ? "-" : st.n}</td>
        <td class="mut">${esc((st.per_seed || []).map(v => num(v)).join("  "))}</td></tr>`;
    }).join("");
  drawChart(rows, metric);
}

function drawChart(rows, metric) {
  const host = $("e-chart"), log = $("e-log").checked;
  const pts = [];
  rows.forEach(d => {
    const st = d.metrics[metric];
    if (st && st.mean != null)
      pts.push({ x: d.n, mean: st.mean, per: st.per_seed || [],
                 key: `${d.planner}${d.method_tag ? "_" + d.method_tag : ""}/${d.allocator}` });
  });
  if (!pts.length) { host.innerHTML = ""; return; }

  const W = 760, H = 300, L = 64, R = 14, T = 14, B = 34;
  const xs = [...new Set(pts.map(p => p.x))].sort((a, b) => a - b);
  let vals = [];
  pts.forEach(p => { vals.push(p.mean); p.per.forEach(v => vals.push(v)); });
  if (log) vals = vals.filter(v => v > 0);
  let lo = Math.min(...vals), hi = Math.max(...vals);
  if (lo === hi) { lo -= 1; hi += 1; }
  const lg = v => Math.log10(Math.max(v, 1e-9));
  const sx = x => L + (xs.length < 2 ? (W - L - R) / 2
    : (xs.indexOf(x) / (xs.length - 1)) * (W - L - R));
  const sy = v => log
    ? H - B - ((lg(v) - lg(lo)) / (lg(hi) - lg(lo))) * (H - T - B)
    : H - B - ((v - lo) / (hi - lo)) * (H - T - B);

  const keys = [...new Set(pts.map(p => p.key))].sort();
  let g = `<line x1="${L}" y1="${T}" x2="${L}" y2="${H - B}" stroke="var(--line)"/>`
        + `<line x1="${L}" y1="${H - B}" x2="${W - R}" y2="${H - B}" stroke="var(--line)"/>`;
  for (let i = 0; i <= 4; i++) {
    const v = log ? Math.pow(10, lg(lo) + i / 4 * (lg(hi) - lg(lo))) : lo + i / 4 * (hi - lo);
    const y = sy(v);
    g += `<line x1="${L}" y1="${y}" x2="${W - R}" y2="${y}" stroke="var(--line)" stroke-dasharray="2 3"/>`
       + `<text x="${L - 6}" y="${y + 4}" text-anchor="end" fill="var(--mut)" font-size="10">${num(v)}</text>`;
  }
  xs.forEach(x => {
    g += `<text x="${sx(x)}" y="${H - B + 16}" text-anchor="middle" fill="var(--mut)" font-size="10">${x}agent</text>`;
  });
  keys.forEach((k, i) => {
    const c = COLORS[i % COLORS.length];
    const line = pts.filter(p => p.key === k).sort((a, b) => a.x - b.x);
    if (line.length > 1)
      g += `<polyline fill="none" stroke="${c}" stroke-width="1.6" points="${
        line.map(p => `${sx(p.x)},${sy(p.mean)}`).join(" ")}"/>`;
    line.forEach(p => {
      p.per.forEach(v => {
        if (!log || v > 0)
          g += `<circle cx="${sx(p.x) + (i - keys.length / 2) * 3}" cy="${sy(v)}" r="2" fill="${c}" opacity="0.35"/>`;
      });
      g += `<circle cx="${sx(p.x)}" cy="${sy(p.mean)}" r="3.5" fill="${c}"/>`;
    });
  });
  host.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="100%">${g}</svg>`
    + `<div class="legend mut">`
    + keys.map((k, i) => `<span><i class="dot" style="background:${COLORS[i % COLORS.length]}"></i>${esc(k)}</span>`).join("")
    + `　<span>小さい点 = seed ごとの値</span></div>`;
}

/* ── 収集・自動更新 ───────────────────────────────────── */
$("collect").onclick = async () => {
  $("collect").disabled = true;
  await api("/api/collect", {});
  poll();
};
async function poll() {
  const st = await api("/api/status");
  const wasBusy = TRAIN && TRAIN.busy;
  if (st.busy) { $("stamp").textContent = "収集中…"; $("collect").disabled = true;
                 setTimeout(poll, 3000); return; }
  if (wasBusy || !TRAIN) await loadTrain(); else renderStamp();
}
setInterval(() => { if ($("auto").checked) loadTrain(); }, 60000);

loadTrain();
