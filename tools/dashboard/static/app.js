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
// train.py の予約から起動された run なら「何本中何本目か」(例: 2/5)。予約外は "-"
function batchNo(r) {
  if (!r.batch_pos || !r.batch_total) return `<span class="mut">-</span>`;
  return `<span title="train.py の予約 ${r.batch_total} 本のうち ${r.batch_pos} 本目">`
       + `${r.batch_pos}/${r.batch_total}</span>`;
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
  $("pane-curve").hidden = name !== "curve";
  $("tab-train").classList.toggle("on", name === "train");
  $("tab-eval").classList.toggle("on", name === "eval");
  $("tab-curve").classList.toggle("on", name === "curve");
  if (name === "eval" && !EVAL) loadEval();
  if (name === "curve") loadCurves();
}
$("tab-train").onclick = () => setTab("train");
$("tab-eval").onclick = () => setTab("eval");
$("tab-curve").onclick = () => setTab("curve");

/* ── 学習の進捗 ───────────────────────────────────────── */
// 絞り込みの値の取り出し方。**計画の行 (cond) と run で同じ表記に揃える**。
// 両者で書き方が違うものがあり (t_max は 100.0 と 100、setting は空白区切りと
// "→" 区切り)、揃えないと片方の値を選んだとき、もう片方が全部消える
const normSetting = v => String(v == null ? "" : v).replace(/→/g, " ").replace(/\s+/g, " ").trim();
const FV = {
  "t-tmax":    { cond: c => c.t_max_m == null ? "" : `${Number(c.t_max_m)}M`,
                 run:  r => r.group_m == null ? "" : `${Number(r.group_m)}M` },
  "t-setting": { cond: c => normSetting(c.setting),  run: r => normSetting(r.setting) },
  "t-arrival": { cond: c => c.task_arrival || "",    run: r => r.task_arrival || "" },
  // 割当学習なし = 空欄。表では "TP" と出しているので合わせる
  "t-assign":  { cond: c => c.task_assign || "TP",   run: r => r.task_assign || "TP" },
  // 計画表に reassign 列が無い (= 問わない) 条件は空。選んだときは表から外れる
  "t-reassign": { cond: c => c.reassign == null ? "" : (c.reassign ? "T" : "F"),
                  run:  r => r.reassign ? "T" : "F" },
};
const FV_IDS = Object.keys(FV);
// 選択肢は **表に出るものだけ** から集める (計画の条件 + 計画に載った run + 実行中)。
// 全 run から集めると、古い探索の値 (t_max 5M など) が並び、選ぶと表が空になる。
// 数字は数の順 (20M < 100M) に並べる
const fvOptions = id => [...new Set([
    ...(TRAIN.plan || []).map(FV[id].cond),
    ...(TRAIN.runs || []).filter(r => r.in_plan || r.state === "running")
                         .map(FV[id].run)].filter(v => v !== ""))]
  .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));

async function loadTrain() {
  TRAIN = await api("/api/train");
  const runs = TRAIN.runs || [];
  fillSelect($("t-plan"), uniq(TRAIN.plan || [], "plan"));
  fillSelect($("t-machine"), uniq(runs, "machine"));
  fillSelect($("t-map"), uniq(runs, "map"));
  fillSelect($("t-agents"), uniq(runs, "agents"));
  fillSelect($("t-algo"), uniq(runs, "algo"));
  FV_IDS.forEach(id => fillSelect($(id), fvOptions(id)));
  renderStamp();
  renderTrain();
  loadAutorun();
}
// t-plan はここに入っていなかったため、計画を選んでも 60 秒後の自動更新まで
// 表が変わらなかった。絞り込みは全部同じ扱いにする
["t-plan", "t-machine", "t-map", "t-agents", "t-algo", "t-state", ...FV_IDS]
  .forEach(id => $(id).onchange = renderTrain);

// SSH で繋がらないホスト。**異常ではなく「確認できない」**。表示している値は
// 最後に見えた時点のもので、学習が止まったとは限らない
const UNREACH = () => (TRAIN && TRAIN.unreachable) || {};
const unkChip = (label, extra) => {
  const u = UNREACH()[label];
  if (!u) return "";
  return `<span class="unk" title="${esc(label)} に接続できません (${esc(u.error || "")})。`
    + `${when(u.since)} から確認できていません">確認できません</span>${extra || ""}`;
};

function renderStamp() {
  const s = TRAIN || {};
  $("stamp").textContent = "収集 " + when(s.collected_at) + (s.busy ? "  (収集中…)" : "");
  $("errors").innerHTML = (s.errors || [])
    .map(e => `<div class="err">⚠ ${esc(e)}</div>`).join("")
    // 繋がらないホストは赤にしない。最後に見えた状態を出しているだけだと分かるようにする
    + Object.entries(s.unreachable || {}).map(([k, u]) =>
      `<div class="mut">🔌 ${esc(k)} に接続できないため、現在の状況は確認できません`
      + ` (${when(u.since)} から)。${esc(k)} の表示は最後に確認できた時点のものです</div>`).join("");
  $("collect").disabled = !!s.busy;
}

function trainRows() {
  const g = id => $(id).value;
  return (TRAIN.runs || []).filter(r =>
    (!g("t-machine") || r.machine === g("t-machine")) &&
    (!g("t-map") || r.map === g("t-map")) &&
    (!g("t-agents") || String(r.agents) === g("t-agents")) &&
    (!g("t-algo") || r.algo === g("t-algo")) &&
    (!g("t-state") || r.state === g("t-state")) &&
    FV_IDS.every(id => !g(id) || FV[id].run(r) === g(id)));
}

// 開いたままにしておく差分パネル。innerHTML を作り直しても消えないよう、
// 条件そのものから作ったキーで覚える (index だと絞り込みでずれる)
// 収集経路の内部名をそのまま出すと "drop" が「落ちている」と読めてしまうので、
// 画面には日本語を出す (値そのものは API のまま)
const VIA = { local: "このPC", ssh: "SSH", drop: "共有フォルダ" };

const OPEN_DIFF = new Set();

// conditions 表の列幅。見出し (map × 台数 × t_max) ごとに別の <table> なので、
// 自動レイアウトだと表ごとに中身の長さで列幅が決まり、縦線が揃わない
// (例: setting が "safe" だけの表と "8x5_2 10M aoba00_2 5M" を含む表で 17 文字ずれる)。
// 全表で同じ colgroup + table-layout:fixed にして位置を固定する。
// 幅は 2026-09 時点の実データの最大長 + 見出しの長さから決めた。状態だけ残りを取る
// dynamic の列は出さない (どの計画も F だけ)。代わりに reassign を全部の表に出す
// rows = その章の条件。setting が既定の幅 (22ch) に収まらない章 (LaRe の多段の系列など) は
// setting を全部見せる幅に広げ、代わりに algorithm と task arrival を中身ぎりぎりまで詰める
const condCols = (rows) => {
  const len = s => [...String(s || "")].length;
  const maxOf = (k, d) => Math.max(d, ...(rows || []).map(c => len(c[k])));
  // ✅ の目印を付けた setting はその分 (空白 + 絵文字 = 3ch) 長く見る
  const setLen = Math.max(0, ...(rows || []).map(c => len(c.setting) + (c.mark ? 3 : 0)));
  const wide = setLen > 22;
  const setW = wide ? setLen + 1 : 22;
  const algoW = wide ? Math.max(6, maxOf("algo", 4)) : 12;
  const arrW = wide ? Math.max(6, maxOf("task_arrival", 4)) : 28;
  return `<colgroup>
  <col style="width:calc(12ch + 16px)"><col style="width:calc(8ch + 16px)">
  <col style="width:calc(${setW}ch + 16px)"><col style="width:calc(${algoW}ch + 16px)">
  <col style="width:calc(${arrW}ch + 16px)"><col style="width:calc(12ch + 16px)">
  <col style="width:calc(9ch + 16px)"><col></colgroup>`;
};
// 再割当ありの条件だけ末尾に印を足す (再割当なしは今までと同じ鍵のまま = 開閉状態などを引き継ぐ)
const condKey = c => [c.plan, c.label, c.setting, c.algo,
                      c.task_assign || "TP", c.dynamic ? 1 : 0].join("|")
                     + (c.reassign ? "|R" : "");

// クリックは委譲で受ける。表は再描画のたびに作り直されるので個別の onclick は付けない
document.addEventListener("click", ev => {
  const b = ev.target.closest && ev.target.closest("[data-diff]");
  if (!b) return;
  // data-diff は **描画ごとの通し番号**。条件名をそのまま入れると空白や "|" が
  // 混ざってセレクタのエスケープが要るので、番号で引いて実体はそこから読む
  const n = b.getAttribute("data-diff");
  const row = document.querySelector('tr.diffrow[data-n="' + n + '"]');
  if (!row) return;
  const show = row.hidden;
  row.hidden = !show;
  const k = row.getAttribute("data-key");
  if (show) { OPEN_DIFF.add(k); row.scrollIntoView({ block: "nearest" }); }
  else OPEN_DIFF.delete(k);
  document.querySelectorAll('[data-diff="' + n + '"]')
    .forEach(el => el.setAttribute("aria-expanded", show ? "true" : "false"));
});

// 「実行中」の条件名から、計画表の同じ run の行へ飛ぶ。
// 条件が 84 個あるので、目で探すと毎回 1 分かかる
function gotoCondRow(el) {
  const uid = el.getAttribute("data-goto");
  // uid には ":" と "/" が入る (sacred のパス) ので、セレクタには入れず走査で引く
  const row = [...document.querySelectorAll("#t-conds tr[data-uid]")]
    .find(tr => tr.getAttribute("data-uid") === uid);
  if (!row) {
    // 絞り込みで隠れているか、計画外で表に出ていない。
    // 文字を書き換えると条件名が消えるので、脇に一言足して戻す
    if (el.querySelector(".gotomiss")) return;
    const note = document.createElement("span");
    note.className = "mut gotomiss";
    note.textContent = " (表に無し)";
    el.appendChild(note);
    setTimeout(() => note.remove(), 1600);
    return;
  }
  // 見出しが sticky で上端に貼り付いているぶん、center にしないと隠れる
  row.scrollIntoView({ behavior: "smooth", block: "center" });
  document.querySelectorAll("tr.flash").forEach(t => t.classList.remove("flash"));
  row.classList.add("flash");
  setTimeout(() => row.classList.remove("flash"), 2000);
}

document.addEventListener("click", ev => {
  const el = ev.target.closest && ev.target.closest("[data-goto]");
  if (el) gotoCondRow(el);
});
// span なので Enter / Space は自分で拾う (button なら既定で効くもの)
document.addEventListener("keydown", ev => {
  if (ev.key !== "Enter" && ev.key !== " ") return;
  const el = ev.target.closest && ev.target.closest("[data-goto]");
  if (!el) return;
  ev.preventDefault();
  gotoCondRow(el);
});

function renderTrain() {
  if (!TRAIN) return;
  // 表を作り直す前に、いま書いている最中のコメント欄を控えておく
  const keptNote = grabNoteFocus();
  const R = trainRows();
  const c = {};
  R.forEach(r => c[r.state] = (c[r.state] || 0) + 1);
  // 状態の並びを固定する。オブジェクトのキー順だと実行のたびに入れ替わって読みにくい
  const ORDER = ["running", "done", "short", "stalled", "failed", "unknown"];
  const LBL = { running: "実行中", done: "完了", short: "途中終了",
                stalled: "停止", failed: "失敗", unknown: "不明" };
  $("t-counts").textContent =
    ORDER.filter(k => c[k]).map(k => `${LBL[k]} ${c[k]}`).join("  ")
    + `   計 ${R.length}`;

  // machines
  const byM = {};
  R.forEach(r => {
    const m = byM[r.machine] || (byM[r.machine] = { run: 0, done: 0, bad: 0, eta: null,
                                                    planDone: 0, planSaved: 0 });
    if (r.state === "running") { m.run++; if (r.eta && (!m.eta || r.eta < m.eta)) m.eta = r.eta; }
    else if (r.state === "done") {
      m.done++;
      // 回収は計画内の run だけが対象なので、計画外を分母に入れない。
      // 入れると「全部回収済みでも 112/182」になり、未回収との区別が付かない
      // LaRe モデル学習の枠だけに入る run (方策を保管しない) も分母に入れない
      if (r.in_plan && r.policy_plan !== false) {
        m.planDone++; if ((r.saved || []).length) m.planSaved++;
      }
    } else if (["stalled", "failed", "short"].includes(r.state)) m.bad++;
  });
  // 「情報がいつのものか」を machine 表の主役にする。これが無いと
  // 「そのマシンが黙っている」ことに気づけず、古い情報を現状だと誤読する
  $("t-mach").innerHTML =
    // 「予約待ち」列は train.py が予約ファイルを書かないと埋まらないので出さない
    // (design/run_collector.md §12.3 が入ったら戻す)
    // 数の列は th にも .num を付ける。td だけ right、th は left のままだと
    // 見出しと数字が左右に離れて、どの列の数か読み取れなくなる
    "<tr><th>machine</th><th>いつの情報か</th><th class=\"num\">実行中</th>"
    + "<th class=\"num\">完了</th><th class=\"num\">モデル回収</th>"
    + "<th class=\"num\">異常</th><th>次の完了</th></tr>"
    + (Object.keys(byM).sort().map(k => {
      const m = byM[k], b = (TRAIN.machines || {})[k];
      let fresh = `<span class="mut">-</span>`;
      if (b && b.data_age_sec != null) {
        const via = VIA[b.via] || b.via || "";
        // 列を短く保つ。経路と「いつもの間隔」は title に逃がし、
        // 本体は経過時間だけにする (長い文字列で表が崩れるため)
        const tip = via + (b.interval_sec ? ` / 通常 ${dur(b.interval_sec)} 間隔` : "");
        fresh = `<span class="${b.stale_data ? "err" : "mut"}" title="${esc(tip)}">`
              + `${dur(b.data_age_sec)} 前</span>`
              + (via ? ` <span class="mut sm">${esc(via)}</span>` : "");
      }
      // 「**計画内の**完了 run のうち何本のモデルを手元に持っているか」。
      // 分母を全完了にすると計画外 (別マップ・mmpp 導入前の設定など) が混ざり、
      // 回収が済んでいても埋まらない
      const nd = m.planDone, ns = m.planSaved;
      const skip = m.done - nd;
      const tip = `方策を保管する計画内の完了 ${nd} 本のうち ${ns} 本を回収済み`
                + (skip ? ` (計画外・LaRe モデル学習の完了 ${skip} 本は対象外)` : "");
      const got = nd ? `<span class="${ns >= nd ? "c-ok" : "wrn"}" title="${esc(tip)}">${ns}/${nd}</span>`
                     : `<span class="mut" title="${esc(tip)}">-</span>`;
      if (UNREACH()[k]) fresh = unkChip(k, b && b.data_age_sec != null
        ? ` <span class="mut sm">最終 ${dur(b.data_age_sec)} 前</span>` : "");
      return `<tr><td>${esc(k)}</td><td>${fresh}</td><td class="num">${m.run}</td>
        <td class="num">${m.done}</td><td class="num">${got}</td>
        <td class="num ${m.bad ? "err" : "mut"}">${m.bad}</td>
        <td class="mut">${UNREACH()[k] ? "-" : when(m.eta)}</td></tr>`;
    }).join("") || `<tr><td class="mut" colspan="7">なし</td></tr>`);

  renderLoad();

  // 情報が古いホストがあれば見出しで知らせる (run の状態と混ぜない)
  const quiet = Object.entries(TRAIN.machines || {})
    .filter(([k, b]) => b.stale_data && !UNREACH()[k]);   // 繋がらないホストは上の灰色の行で知らせる
  $("t-stale").innerHTML = quiet.map(([k, b]) => {
    const norm = b.interval_sec
      ? `通常は ${dur(b.interval_sec)} 間隔で届きます。` : "";
    return `<div class="err">⚠ ${esc(k)} から ${dur(b.data_age_sec)} 情報が届いていません。`
      + `${norm}表示は最後に見えた時点のもので、いま動いているかは不明です`
      + `<span class="mut"> — そのマシンがスリープ / 停止しているか、`
      + `iCloud の同期が止まっている可能性があります</span></div>`;
  }).join("");

  // running now。observed_at 基準にしたので "running" は
  // **最後に観測できた時点で走っていた** という意味になる。
  // 情報が古い場合はそのことを行に添える (止まったとは断定しない)
  const run = R.filter(r => r.state === "running")
    .sort((a, b) => (a.eta || "9") < (b.eta || "9") ? -1 : 1);
  $("t-run").innerHTML =
    "<tr><th>進捗</th><th class=\"num\">t_env</th><th class=\"num\">残り</th>"
    + "<th>終了予定</th><th>machine</th><th class=\"num\">予約</th><th>計画</th>"
    + "<th>条件</th><th>seed</th><th>経過</th><th>コメント</th></tr>"
    // 繋がらないホストの行は薄くし、残り・終了予定は出さない (最後に見えた値からの推測になるため)
    + (run.length ? run.map(r => `<tr${UNREACH()[r.machine] ? ' class="unkrow"' : ""}>
        <td><span class="pb"><i style="width:${((r.progress || 0) * 100).toFixed(0)}%"></i></span>
            ${((r.progress || 0) * 100).toFixed(0)}%</td>
        <td class="mut num">${M(r.t_last)}/${M(r.t_max)}M</td>
        <td class="num">${UNREACH()[r.machine] ? "-" : dur(r.remaining_sec)}</td><td>${
          UNREACH()[r.machine] ? unkChip(r.machine) :
          // 一時停止中 (Ctrl-Z) は再開の時刻が分からないので予定時刻は出さない
          r.paused ? `<span class="wrn" title="プロセスが一時停止しています (ps の状態 T)。再開すれば残り時間で終わります">⏸ 一時停止中</span>`
                   : when(r.eta)}</td>
        <td>${esc(r.machine)}</td><td class="num">${batchNo(r)}</td>
        <td>${(r.plans || []).length ? esc(r.plans.join(", ")) : '<span class="mut">-</span>'}</td>
        <td>${(cond => r.in_plan
            // ボタンを置くと行が騒がしくなるので、条件名そのものを押させる。
            // ただの文字だと押せると気づけないので、点線の下線で示す
            ? `<span class="goto" role="button" tabindex="0" data-goto="${esc(r.uid)}"`
              + ` title="計画表のこの run の行へ移動">${cond}</span>`
            : cond)(`${r.agents}ag ${esc(r.map)} ${esc(r.algo)} ${esc(r.setting)}`)}</td>
        <td class="mut">${esc(r.seed)}${r.in_plan ? "" : ' <span class="wrn">計画外</span>'}${
          UNREACH()[r.machine] ? " " + unkChip(r.machine)
          : (r.stale_data ? ` <span class="wrn" title="このホストからの情報が古い">情報 ${dur(r.data_age_sec)} 前</span>` : "")}</td>
        <td class="mut">${esc(r.duration)}</td>
        <td>${noteCell(r)}</td></tr>`).join("")
      : `<tr><td class="mut" colspan="11">なし</td></tr>`);

  renderAttention();

  // conditions — **表は計画 (tools/plan.md) から作る**。実績はそこに埋める。
  // 計画に無い run は表に出さない (running セクションには出る)
  const WANT_DEFAULT = 5;
  const ALGO = a => String(a || "").toUpperCase();
  const g = id => $(id).value;
  const plan = (TRAIN.plan || []).filter(c =>
    (!g("t-plan") || c.plan === g("t-plan")) &&
    (!g("t-map") || c.map === g("t-map")) &&
    (!g("t-agents") || String(c.agents) === g("t-agents")) &&
    (!g("t-algo") || c.algo === g("t-algo")) &&
    (!g("t-machine") || c.slots.some(s => s.machine === g("t-machine")
                                       || (s.run && s.run.machine === g("t-machine")))) &&
    (!g("t-state") || c.slots.some(s => s.run && s.run.state === g("t-state"))) &&
    FV_IDS.every(id => !g(id) || FV[id].cond(c) === g(id)));

  const nOut = (TRAIN.runs || []).filter(r => !r.in_plan).length;
  let html = "";
  if (!TRAIN.plan || !TRAIN.plan.length)
    html += `<div class="wrn">計画ファイルが読めていません: ${esc(TRAIN.plan_file || "")}</div>`;
  else if (nOut)
    html += `<div class="mut">計画外の run: ${nOut} 件 `
          + `<span class="mut">(表には出しません。running / machines には出ます)</span></div>`;

  let head = null, planHead = null, groupHead = null, diffN = 0;
  const WARN = [];     // 上の「パラメータの警告」一覧に出すもの (表の描画と同じ判定で集める)
  const multi = new Set(plan.map(c => c.plan)).size > 1;
  plan.forEach(c => {
    // 複数の計画を読んでいるときだけ計画名の見出しを出す (1 枚運用では邪魔になる)
    if (multi && c.plan !== planHead) {
      if (head !== null) { html += `</table></div>`; head = null; }
      planHead = c.plan;
      groupHead = null;
      html += `<h2 class="planhead">${esc(c.plan)}</h2>`;
    }
    // 計画ファイルの "# ..." (LaRe モデル学習 / 方策学習 など) の区切り
    if (c.group && c.group !== groupHead) {
      if (head !== null) { html += `</table></div>`; head = null; }
      groupHead = c.group;
      html += `<div class="grouphead">${esc(c.group)}</div>`;
    }
    if (c.label !== head) {
      if (head !== null) html += `</table></div>`;
      head = c.label;
      // マップ / 台数 / t_max を別々の span にして列を揃える (label は行の同一判定用)
      html += `<h3><span class="g-map">${esc(c.map || "")}</span>`
            + `<span class="g-n">${c.agents} agent</span>`
            + `<span class="g-t">${c.t_max_m}M</span></h3>`
            + `<div class="wrap"><table class="cond">${condCols(plan.filter(x =>
                x.plan === c.plan && x.label === c.label && x.group === c.group))}
        <tr>${c.collect === "lare"
              ? `<th colspan="2">LaRe モデル</th>` : `<th>seed</th><th>machine</th>`}<th>setting</th><th>algorithm</th>
            <th>task arrival</th><th>task assign</th><th>reassign</th><th>状態</th></tr>`;
    }
    const NC = 8;                // 表の列の数 (隠し行の colspan に使う)
    const want = c.want || WANT_DEFAULT;
    // 5 seed に数えてよい run: 設定が割れていない (params✗ でない) かつ 手で除外していない
    const usable = r => r && !r.odd_params && !r.excluded;
    const done = c.slots.filter(s => s.run && s.run.state === "done"
                                     && usable(s.run)).length;
    // 5 seed そろっていれば、失敗した run は出さない (メモにも書かない運用に合わせる)。
    // 何件隠したかは条件行に出すので、黙って消えるわけではない
    // LaRe モデル学習の条件は 1 本で完了するので、完了していても設定の確認を出す
    // (方策の表は 5 seed 揃ったら余分な run を使わないので確認を引っ込める)
    const lare = c.collect === "lare";
    const filled = lare ? false : done >= want;
    let slots = c.slots, hidden = 0;
    if (filled) {
      const before = slots.length;
      slots = slots.filter(s => s.run && usable(s.run)
                                && (s.run.state === "done" || s.run.state === "running"));
      hidden = before - slots.length;
    }
    // モデルを手元に保管できている本数。学習の完了とは別の軸なので分けて出す
    const saved = c.slots.filter(s => s.run && s.run.state === "done"
                                      && usable(s.run)
                                      && s.run.saved && s.run.saved.length).length;
    const kept = Math.min(saved, want);
    const allDone = filled && kept >= want;     // 学習もモデルも揃って「完了」
    const odd = filled ? 0 : c.slots.filter(s => s.run && s.run.odd_params).length;
    const tmax = slots.find(s => s.run && s.run.t_max_ok === false);
    // 計画表に reassign 列が無い計画 (plan_AAMAS) は「問わない」
    const ra = c.reassign == null
      ? `<span class="mut" title="計画表に reassign の列がありません (問わない)">-</span>`
      : (c.reassign ? "T" : "F");

    // 差分があるときだけ押せるボタンを出す。押すと下の隠し行が開く。
    // **5 seed そろっている条件では出さない**。揃っていれば余分な run は使わないので、
    // パラメータが割れていても直す必要がない (失敗行を隠すのと同じ運用)
    const hasDiff = !filled && !!(c.param_diff && c.param_diff.length);
    const dkey = condKey(c);
    const dn = hasDiff ? ++diffN : 0;
    const opened = OPEN_DIFF.has(dkey);
    const diffBtn = hasDiff
      ? ` <button type="button" class="diffbtn" data-diff="${dn}"`
        + ` aria-expanded="${opened ? "true" : "false"}"`
        + ` title="どのパラメータが違うか出す">違いを見る (${c.param_diff.length})</button>`
      : "";

    // 全体比較: 表の他の条件 (同じ algo / dynamic) と設定が違うもの。
    // 条件内の params✗ とは別物なので、色と文言を分ける (こちらは「数から外さない」)
    // 5 seed そろった条件では出さない (params✗ と同じ運用。揃っていれば回し直さない)
    const G = filled ? [] : (c.global_diff || []);
    const gn = G.length ? ++diffN : 0;
    const gkey = "g|" + dkey;
    const gopened = OPEN_DIFF.has(gkey);
    const gBtn = G.length
      ? ` <button type="button" class="diffbtn gdiffbtn" data-diff="${gn}"`
        + ` aria-expanded="${gopened ? "true" : "false"}"`
        + ` title="表の他の条件と設定が違います。学習の本数には数えたままです">`
        + `他と違う (${G.length})</button>`
      : (c.global_dismissed
          ? ` <span class="mut sm" title="全体との差を確認済みにしています">`
            + `他との差 ${c.global_dismissed} 件 確認済み `
            + `<a href="#" class="gundo" data-conds="${esc(JSON.stringify(c.global_conds || []))}">取り消す</a></span>`
          : "");

    if (hasDiff) WARN.push({ kind: "split", c, n: dn });
    if (G.length) WARN.push({ kind: "global", c, n: gn, G });

    // 「LaRe モデル学習」の条件は seed を並べず、モデルファイルがあるかだけ出す。
    // 設定の確認 (違いを見る / 他と違う) は学習 run の設定で方策の表と同じように出す
    if (lare) html += lareRow(c, diffBtn + gBtn);
    else html += `<tr class="condrow"><td></td><td></td>
      <td title="${esc(c.setting)}">${esc(c.setting)}${c.mark ? " ✅" : ""}</td><td>${esc(ALGO(c.algo))}</td>
      <td>${esc(c.task_arrival)}</td><td>${esc(c.task_assign || "TP")}</td>
      <td>${ra}</td>
      <td class="st">${
        `<span class="${filled ? "c-ok" : "wrn"}">学習 ${done}/${want}</span>`}${
        `<span class="${kept >= want ? "c-ok" : (kept ? "wrn" : "mut")}">モデル ${kept}/${want}</span>`}${
        allDone ? ` <span class="c-ok">✔ 完了</span>` : ""}${
        odd ? ` <span class="wrn">⚠要再実行 ${odd}</span>` : ""}${
        tmax ? ` <span class="wrn">⚠t_max</span>` : ""}${
        hidden ? ` <span class="mut">(失敗・除外 ${hidden} 件を非表示)</span>` : ""}${
        diffBtn}${gBtn}${c.ar_id ? ` <button type="button" class="arbtn" data-ar="${esc(c.ar_id)}"`
          + ` title="この条件の学習・評価を自動実行に頼む">依頼</button>` : ""}</td></tr>`;

    // 全体との差の中身。キーごとに「この条件 / 他」を並べ、問題なければ消せる
    if (G.length) {
      html += `<tr class="diffrow gdiffrow" data-n="${gn}" data-key="${esc(gkey)}"${
        gopened ? "" : " hidden"}>
        <td colspan="${NC}">
        <div class="wrn">表の他の条件 (同じ algo・dynamic) と設定が違います。`
        + `意図した違いなら「問題なし」で消せます</div>
        <table class="pdiff"><tr><th>key</th><th>この条件</th><th>他の条件 (多数派)</th><th></th></tr>`
        + G.map(x => `<tr><td>${esc(x.key)}</td>
            <td class="wrn">${esc(fmtVal(x.actual))} <span class="mut">(${x.n_this} run)</span></td>
            <td>${esc(fmtVal(x.expected))} <span class="mut">(${x.n_major}/${x.n_group} run)</span></td>
            <td><button type="button" class="ack" data-dismiss="${esc(x.dismiss_key)}"
                 title="この差は意図したものなので警告を消す">問題なし</button></td></tr>`).join("")
        + `</table></td></tr>`;
    }

    // パラメータが割れている条件は、どのキーがどう違うかを隠し行に持つ。
    // ハッシュだけ出しても「何を直して回し直すか」が分からない
    if (hasDiff) {
      const hs = c.param_hashes || [];
      html += `<tr class="diffrow" data-n="${dn}" data-key="${esc(dkey)}"${
        opened ? "" : " hidden"}>
        <td colspan="${NC}">
        <div class="wrn">パラメータが ${hs.length} 通りに割れています `
        + `(${c.param_diff.length} キー)</div>
        ${splitTable(c)}
        <div class="mut">seed: `
        + hs.map(h => `${esc(h.hash)} = ${h.seeds.map(esc).join(", ")}`).join(" / ")
        + `</div></td></tr>`;
    }

    if (lare) return;            // seed の行は出さない
    slots.forEach(sl => {
      const r = sl.run;
      let st;
      // train.py が「これから回す」と予約している枠。sacred のディレクトリは
      // まだ無いので run は付かない。未実行と分けないと、埋まっているマシンに
      // 二重で投入してしまう
      if (!r && sl.pending)
        st = `<span class="pend" title="train.py がこの条件をあと何本か回す予定です${
              sl.pending_on ? " (" + esc(sl.pending_on) + ")" : ""}。seed は起動時に決まるので、この行の seed 番号とは限りません">⏳ 実行待ち${
              sl.pending_on ? ` <span class="mut sm">${esc(sl.pending_on)}</span>` : ""}</span>`;
      else if (!r) st = `<span class="mut">未実行</span>`;
      else {
        const pct = ((r.progress || 0) * 100).toFixed(0);
        const steps = `${M(r.t_last)}M / ${M(r.t_max)}M`;
        // 「何ステップまで行ったか」の隣に「いつ終わったか」を必ず出す。
        // 終了時刻が無いと、同じ条件の seed が何日にまたがって回ったか分からない
        const fin = r.stop_at
          ? ` <span class="fin" title="${esc(r.stop_at)}">${when(r.stop_at)} 完了</span>` : "";
        const stopped = r.stop_at
          ? ` <span class="fin" title="${esc(r.stop_at)}">${when(r.stop_at)}</span>` : "";
        // モデルを手元に持っているか。done なのに無いものを見つけられるようにする
        let got = "";
        if (r.state === "done") {
          if (r.saved && r.saved.length)
            got = ` <span class="c-ok" title="保管済み: ${esc(r.saved.join(", "))}">`
                + `📦${r.saved.length > 1 ? " " + r.saved.join("+") : ""}</span>`;
          // サーバが古いとこれらのキー自体が無い。"モデル無し" と断定すると
          // 全行が誤表示になるので、未定義は「不明」として区別する
          else if (r.has_model === undefined)
            got = ` <span class="mut" title="サーバを再起動すると分かります">-</span>`;
          else if (!r.has_model)
            got = ` <span class="mut" title="この run にはモデルファイルが残っていません">モデル無し</span>`;
          else
            got = ` <span class="wrn" title="取りに行けば回収できます">未回収</span>`;
        }
        if (r.state === "done")
          st = `<span class="c-ok">✔ done</span> <span class="mut">${M(r.t_max)}M</span>${fin}${got}`;
        else if (r.state === "running" && UNREACH()[r.machine])
          // 繋がらない間は進み具合も終了予定も分からない。最後に見えた値だけ添える
          st = unkChip(r.machine, ` <span class="mut">最後に見えた時点で ${pct}% (${steps})</span>`);
        else if (r.state === "running")
          st = `<span class="pb"><i style="width:${pct}%"></i></span> ${pct}%`
             + ` <span class="steps">${steps}</span>`
             + (r.paused
                 ? ` <span class="wrn">⏸ 一時停止中</span> <span class="mut">(再開すれば残り ${dur(r.remaining_sec)})</span>`
                 : ` <span class="mut">残り ${dur(r.remaining_sec)} → ${when(r.eta)} 終了予定</span>`);
        else
          st = `<span class="err">✖ ${esc(r.state)}</span>`
             + ` <span class="mut">${steps} で停止</span>${stopped}`;
        // params✗ の横から直接開けるようにする (条件行まで目を動かさずに済む)
        if (r.odd_params)
          st += ` <span class="wrn">params✗</span>${hasDiff
            ? ` <button type="button" class="diffbtn" data-diff="${dn}"`
              + ` aria-expanded="${OPEN_DIFF.has(dkey) ? "true" : "false"}"`
              + ` title="どのパラメータが違うか出す">違いを見る</button>` : ""}`;
        // 手動除外。報酬の計算式を変えた前後など、config に残らない違いで
        // 使えなくなった run を 5 seed から外す (表示は残し、いつでも戻せる)
        st = r.excluded
          ? `<span class="exl" title="5 seed に数えていません (${esc(r.excluded_at || "")})">除外</span> `
            + `<span class="exltxt">${st}</span>`
            + ` <button type="button" class="ack" data-excl="${esc(r.uid)}" data-on="0"`
            + ` title="5 seed に数え直す">戻す</button>`
          : st + ` <button type="button" class="ack exlbtn" data-excl="${esc(r.uid)}" data-on="1"`
            + ` title="この run を 5 seed に数えない (設定が config に残らない形で違うときなど)">除外</button>`;
      }
      // 表に無い seed は薄字の "+" で補足するだけ。黄色くするのは suspect のときだけ
      // (状態 / params✗ / t_max は別の欄に出ているので、ここで重ねて警告しない)
      // data-uid: 「実行中」の表から、この行へ飛ぶための目印
      html += `<tr${r ? ` data-uid="${esc(r.uid)}"` : ""}${r && r.excluded ? ' class="excl"' : ""}><td class="${sl.suspect ? "wrn" : (r ? "" : "mut")}">${
          esc(sl.seed || "—")}${sl.suspect ? ' <span class="wrn" title="表には 5 seed 書いてあるのに、別の seed も完了しています">*</span>'
            : (sl.unplanned_seed ? ' <span class="mut" title="計画表に書かれていない seed">+</span>' : "")}</td>
        <td class="mut">${esc((r && r.machine) || sl.machine || "")}</td>
        <td colspan="5" class="notecell">${r ? noteCell(r) : ""}</td>
        <td class="st">${st}</td></tr>`;
    });
  });
  if (head !== null) html += `</table></div>`;
  $("t-conds").innerHTML = html || `<div class="mut">計画に一致する条件がありません</div>`;
  renderParamWarn(WARN);
  restoreNoteFocus(keptNote);
}

/* ── 評価結果 ─────────────────────────────────────────── */
async function loadEval() {
  EVAL = await api("/api/eval");
  if (!EVAL.available) {
    $("e-tbl").innerHTML = `<tr><td class="err">${esc(EVAL.error)}</td></tr>`;
    return;
  }
  const C = EVAL.conditions;
  // 計画の選択肢: seed のモデルが属する計画 + どの計画にも無いもの
  const P = new Set();
  let noPlan = false;
  C.forEach(d => (d.seeds || []).forEach(s => {
    s.plans.forEach(p => P.add(p));
    if (!s.plans.length) noPlan = true;
  }));
  fillSelect($("e-plan"), [...[...P].sort(), ...(noPlan ? [NOPLAN] : [])]);
  fillSelect($("e-map"), uniq(C, "map"));
  fillSelect($("e-n"), uniq(C, "n"));
  fillSelect($("e-env"), uniq(C, "env"));
  fillSelect($("e-planner"), uniq(C, "planner"));
  fillSelect($("e-tag"), uniq(C, "method_tag"));
  fillSelect($("e-alloc"), uniq(C, "allocator"));
  fillSelect($("e-trained"), [...new Set(C.map(EF.reassign))].sort());
  fillSelect($("e-exre"), [...new Set(C.map(EF.env_reassign))].sort());
  fillSelect($("e-arrival"), [...new Set(C.map(EF.arrival))].sort(natCmp));
  // 比較に使える要素のチェックボックス (前に選んでいたものは残す)
  $("e-cmp").innerHTML = EVAL_FACTORS.map(([k, l]) =>
    `<label><input type="checkbox" value="${k}"${CMP.has(k) ? " checked" : ""}> ${esc(l)}</label>`).join("");
  $("e-cmp").querySelectorAll("input").forEach(el => el.onchange = () => {
    if (el.checked) CMP.add(el.value); else CMP.delete(el.value);
    renderEval();
  });
  $("e-metric").innerHTML = (EVAL.metrics || []).map(m => `<option>${esc(m)}</option>`).join("");
  const pref = (EVAL.metrics || []).indexOf("task_completion");
  if (pref >= 0) $("e-metric").selectedIndex = pref;
  renderEval();
}
["e-plan", "e-map", "e-n", "e-env", "e-planner", "e-tag", "e-alloc", "e-trained", "e-exre", "e-arrival",
 "e-metric", "e-log"].forEach(id => $(id).onchange = renderEval);

// 評価の条件を作っている要素。[キー, 見出し, 表示する値]
// 比較の選択欄・比較表の列と行・絞り込みで共通に使う
const EVAL_FACTORS = [
  ["map", "map", d => d.map],
  ["n", "N", d => String(d.n)],
  ["env", "env", d => d.env],
  ["planner", "planner", d => d.planner],
  ["method_tag", "tag", d => d.method_tag || "-"],
  ["allocator", "alloc", d => d.allocator],
  ["reassign", "trained", d => d.reassign || "base"],
  ["env_reassign", "exec reassign", d => d.env_reassign ? "T" : "F"],
  ["arrival", "arrival", d => d.arrival || "fixed"],
  ["dynamic", "dyn", d => d.dynamic ? "T" : "F"],
];
const EF = Object.fromEntries(EVAL_FACTORS.map(([k, , f]) => [k, f]));
// 比較表で、値が同じでも行の見出しに必ず出す要素 (基本の要素)
const EVAL_BASE = ["map", "n", "env", "planner", "method_tag", "allocator"];
const CMP = new Set();            // 比較に選んでいる要素のキー
// "5" と "10"、"bern0.05" と "bern0.1" を数の大きさの順に並べる
const natCmp = (a, b) => String(a).localeCompare(String(b), undefined, { numeric: true });

// 計画で絞ったときは、その計画の seed だけで平均・標準偏差を取り直す。
// summary.csv の値は全 seed をまとめたもので、計画ごとに seed の数が違う
// (AAMAS は 5 seed、reassign は seed0 だけ、arrival_range は範囲を変えた run) ため
const NOPLAN = "(計画外)";
// LaRe モデル学習の条件 1 行。src/lare/path/models/ にあれば完了、checkpoints/ だけなら
// コピー待ち、どちらにも無ければ学習中の run の進み具合か「未作成」
function lareRow(c, btns) {
  const F = c.lare_files || [];
  const inModels = F.find(f => f.where === "models");
  const inCkpt = F.find(f => f.where === "checkpoints");
  const inFetched = F.find(f => f.where === "fetched");     // 他のマシンから回収したもの
  const run = (c.slots || []).map(s => s.run).find(r => r && r.state === "running");
  const doneRun = (c.slots || []).map(s => s.run).find(r => r && r.state === "done");
  // 学習した PC。記録のある run (完了 > 実行中) を優先し、無ければ計画表の
  // LaRe model 列の括弧書き (例: "✔ models (GPU2)") を使う
  const prod = (c.slots || []).map(s => s.run).find(r => r && r.state === "done") || run;
  const noteM = /[(（]([^)）]+)[)）]/.exec(c.lare_note || "");
  const pc = prod ? prod.machine : (noteM ? noteM[1] : "");
  const pcTag = pc ? ` <span class="mut sm" title="学習した PC">${esc(pc)}</span>` : "";
  let file = `<span class="mut">-</span>`, st;
  if (inModels) {
    file = `<span class="c-ok" title="src/lare/path/models/${esc(inModels.name)}">✔ models/</span>`;
    st = `<span class="c-ok">✔ あり</span> <span class="mut sm">${esc(inModels.name)}</span>`;
  } else if (inCkpt) {
    file = `<span class="wrn" title="src/lare/path/checkpoints/${esc(inCkpt.name)}">checkpoints/</span>`;
    st = `<span class="wrn" title="使う前に models/ へコピーしてください">checkpoints/ のみ</span>`
       + ` <span class="mut sm">${esc(inCkpt.name)}</span>`;
  } else if (inFetched) {
    const p = `src/lare/path/fetched/${inFetched.machine || ""}/${inFetched.name}`;
    file = `<span class="wrn" title="${esc(p)}">fetched/</span>`;
    st = `<span class="wrn" title="使う前に models/ へコピーしてください: ${esc(p)}">`
       + `取得済み (${esc(inFetched.machine || "?")})</span>`
       + ` <span class="mut sm">${esc(inFetched.name)}</span>`;
  } else if (doneRun) {
    // 学習は終わったが、モデルがまだこの Mac に届いていない (次の定期回収で取ってくる)
    st = `<span class="wrn" title="次の定期回収 (launchd) で src/lare/path/fetched/ に取ってきます。`
       + `共有フォルダのマシンは、相手の collect_runs.py が新しい版 (git pull 済み) である必要があります">`
       + `学習完了・未回収 (${esc(doneRun.machine || "?")})</span>`;
  } else if (run) {
    st = `<span class="pb"><i style="width:${((run.progress || 0) * 100).toFixed(0)}%"></i></span> `
       + `${((run.progress || 0) * 100).toFixed(0)}% <span class="mut sm">${esc(run.machine || "")}</span>`;
  } else {
    st = `<span class="mut">未作成</span>`;
  }
  return `<tr class="condrow"><td colspan="2">${file}${pcTag}</td>
    <td title="${esc(c.setting)}">${esc(c.setting)}${c.mark ? " ✅" : ""}</td><td>${esc(String(c.algo || "").toUpperCase())}</td>
    <td>${esc(c.task_arrival)}</td><td>${esc(c.task_assign || "TP")}</td>
    <td>${c.reassign == null ? "-" : (c.reassign ? "T" : "F")}</td>
    <td class="st">${st}${btns || ""}</td></tr>`;
}

function planView(d, plan) {
  const ss = (d.seeds || []).filter(s =>
    plan === NOPLAN ? !s.plans.length : s.plans.includes(plan));
  if (!ss.length) return null;
  const metrics = {};
  (EVAL.metrics || []).forEach(m => {
    const v = ss.map(s => s.vals[m]).filter(x => x != null);
    if (!v.length) return;
    const n = v.length, mean = v.reduce((a, b) => a + b, 0) / n;
    // 標本標準偏差 (aggregate.py と同じ ddof=1)
    const std = n >= 2 ? Math.sqrt(v.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1)) : null;
    metrics[m] = { n, mean, std, sem: std == null ? null : std / Math.sqrt(n), per_seed: v };
  });
  return Object.assign({}, d, { metrics });
}

function evalRows() {
  const g = id => $(id).value;
  const plan = g("e-plan");
  const base = plan ? (EVAL.conditions || []).map(d => planView(d, plan)).filter(Boolean)
                    : (EVAL.conditions || []);
  return base.filter(d =>
    (!g("e-map") || d.map === g("e-map")) &&
    (!g("e-n") || String(d.n) === g("e-n")) &&
    (!g("e-env") || d.env === g("e-env")) &&
    (!g("e-planner") || d.planner === g("e-planner")) &&
    (!g("e-tag") || d.method_tag === g("e-tag")) &&
    (!g("e-alloc") || d.allocator === g("e-alloc")) &&
    (!g("e-trained") || EF.reassign(d) === g("e-trained")) &&
    (!g("e-exre") || EF.env_reassign(d) === g("e-exre")) &&
    (!g("e-arrival") || EF.arrival(d) === g("e-arrival")));
}
window.evalSortBy = k => {
  if (evalSortCol === k) evalSortAsc = !evalSortAsc;
  else { evalSortCol = k; evalSortAsc = true; }
  renderEval();
};

// 比べるための列。**表示している結果の中で値が 2 通り以上あるときだけ**出す
// (1 通りしか無ければ比べる対象ではないので出さない。絞り込みを変えると列も変わる)。
// map / N / env / planner / tag / alloc の基本の列は、値が同じでも常に出す
// [キー, 見出し, 表示する値]
const EVAL_OPT_COLS = [
  ["reassign", "trained", d => d.reassign || "base"],
  ["env_reassign", "exec reassign", d => d.env_reassign ? "T" : "F"],
  ["arrival", "arrival", d => d.arrival || "fixed"],
  ["dynamic", "dyn", d => d.dynamic ? "T" : "F"],
];

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
  const opt = EVAL_OPT_COLS.filter(c => new Set(rows.map(c[2])).size > 1);
  if (CMP.size) {                  // 比較する要素が選ばれていれば比較表にする
    renderCompare(rows, metric, EVAL_FACTORS.filter(f => CMP.has(f[0])));
    drawChart(rows, metric, opt);
    return;
  }

  const cols = [["map", "map"], ["n", "N"], ["env", "env"], ["planner", "planner"],
                ["method_tag", "tag"], ["allocator", "alloc"],
                ...opt.map(c => [c[0], c[1]]), ["metric", metric]];
  $("e-tbl").innerHTML =
    "<tr>" + cols.map(([k, l]) =>
      `<th class="sortable${["n", "metric"].includes(k) ? " num" : ""}" onclick="evalSortBy('${k}')">${esc(l)}${evalSortCol === k ? (evalSortAsc ? " ▲" : " ▼") : ""}</th>`).join("")
    + "<th class=\"num\">n</th><th>per-seed</th></tr>"
    + rows.map(d => {
      const st = d.metrics[metric] || {};
      const cls = st.n === 1 ? "one" : (st.n < 5 ? "thin" : "");
      return `<tr><td>${esc(d.map)}</td><td class="num">${d.n}</td><td>${esc(d.env)}</td>
        <td>${esc(d.planner)}</td><td>${esc(d.method_tag || "-")}</td>
        <td>${esc(d.allocator)}</td>${
        opt.map(c => `<td>${esc(c[2](d))}</td>`).join("")}
        <td class="num">${num(st.mean)} ± ${num(st.std)}</td>
        <td class="num ${cls}">${st.n == null ? "-" : st.n}</td>
        <td class="mut">${esc((st.per_seed || []).map(v => num(v)).join("  "))}</td></tr>`;
    }).join("");
  drawChart(rows, metric, opt);
}

// 比較表: 比べる要素の値 (の組み合わせ) を列に、それ以外の要素が同じ条件を 1 行にまとめる。
// 行の見出しには、基本の要素と、表示中に 2 通り以上ある要素を出す (同じ値しか無い要素は省く)
function renderCompare(rows, metric, cmpF) {
  const cmpKeys = cmpF.map(f => f[0]);
  const rowF = EVAL_FACTORS.filter(f => !cmpKeys.includes(f[0]) &&
    (EVAL_BASE.includes(f[0]) || new Set(rows.map(f[2])).size > 1));
  const colKey = d => cmpF.map(f => f[2](d)).join(" / ");
  const rowKey = d => rowF.map(f => f[2](d)).join("\u0001");
  const cols = [...new Set(rows.map(colKey))].sort(natCmp);
  const groups = new Map();
  rows.forEach(d => {
    const k = rowKey(d);
    if (!groups.has(k)) groups.set(k, { d, cells: {} });
    groups.get(k).cells[colKey(d)] = d;
  });
  const keys = [...groups.keys()].sort(natCmp);
  const cell = d => {
    const st = d && d.metrics[metric];
    if (!st || st.mean == null) return null;
    return st;
  };
  $("e-tbl").innerHTML =
    "<tr>" + rowF.map(f => `<th>${esc(f[1])}</th>`).join("")
    + cols.map(c => `<th class="cmpcol num" title="${esc(cmpF.map(f => f[1]).join(" / "))}">${esc(c)}</th>`).join("")
    + "</tr>"
    + keys.map(k => {
      const g = groups.get(k);
      const sts = cols.map(c => cell(g.cells[c]));
      // 行の中で平均が一番大きいものを太字にする (指標によっては小さい方が良いので、目安として)
      const means = sts.filter(Boolean).map(st => st.mean);
      const best = means.length > 1 ? Math.max(...means) : null;
      return "<tr>" + rowF.map(f => `<td>${esc(f[2](g.d))}</td>`).join("")
        + sts.map(st => st
          ? `<td class="cmpcell${st.mean === best ? " best" : ""}"`
            + ` title="per-seed: ${esc((st.per_seed || []).map(v => num(v)).join("  "))}">`
            + `${num(st.mean)}${st.std == null ? "" : " ± " + num(st.std)}`
            + ` <span class="mut ${st.n === 1 ? "one" : (st.n < 5 ? "thin" : "")}">(${st.n})</span></td>`
          : `<td class="cmpcell mut">-</td>`).join("")
        + "</tr>";
    }).join("");
}

function drawChart(rows, metric, opt) {
  const host = $("e-chart"), log = $("e-log").checked;
  const pts = [];
  rows.forEach(d => {
    const st = d.metrics[metric];
    if (st && st.mean != null)
      pts.push({ x: d.n, mean: st.mean, per: st.per_seed || [],
                 // 表に出している違い (再割当・到着など) も系列の名前に入れる。入れないと別の条件の点が 1 本の線に混ざる
                 key: `${d.planner}${d.method_tag ? "_" + d.method_tag : ""}/${d.allocator}`
                      + (opt || []).map(c => "/" + c[2](d)).join("") });
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
// ── 全体比較 (他の条件と設定が違う) ──────────────────────────────────
// config の値をそのまま見せる。null = その run の config にキーが無い (古い run)
function fmtVal(v) {
  if (v === null || v === undefined) return "(記録なし)";
  if (v === true) return "True";
  if (v === false) return "False";
  return typeof v === "object" ? JSON.stringify(v) : String(v);
}

// 「問題なし」/「取り消す」。状態が変わると件数も変わるので、表ごと取り直す
document.addEventListener("click", async ev => {
  const b = ev.target.closest && ev.target.closest("[data-dismiss]");
  if (b) {
    b.disabled = true;
    const res = await api("/api/dismiss_diff", { key: b.getAttribute("data-dismiss") });
    if (apiFailed(res, "「問題なし」の保存")) { b.disabled = false; return; }
    return loadTrain();
  }
  const u = ev.target.closest && ev.target.closest(".gundo");
  if (u) {
    ev.preventDefault();
    let conds = [];
    try { conds = JSON.parse(u.getAttribute("data-conds") || "[]"); } catch (e) { /* 空のまま */ }
    const res = await api("/api/dismiss_diff", { undo: true, conds });
    if (res && res.error) return apiFailed(res, "取り消し");
    return loadTrain();
  }
});

// サーバが古いまま (app.py を変えて再起動していない) だと新しい API は 404 になる。
// 黙って失敗させると「押しても何も起きない」ように見えるので、必ず知らせる
function apiFailed(res, what) {
  if (res && !res.error && res.ok !== false) return false;
  const msg = (res && res.error) || "応答がありません";
  alert(`${what}に失敗しました: ${msg}\n`
        + "ダッシュボードを再起動していない場合は、再起動してからもう一度試してください。");
  return true;
}

// ── 手動除外 ───────────────────────────────────────────────────────
document.addEventListener("click", async ev => {
  const b = ev.target.closest && ev.target.closest("[data-excl]");
  if (!b) return;
  b.disabled = true;
  const res = await api("/api/exclude", { uid: b.getAttribute("data-excl"),
                                          on: b.getAttribute("data-on") === "1" });
  if (apiFailed(res, "除外")) { b.disabled = false; return; }
  // 評価用フォルダのモデルも動かしたので、何をしたかを知らせる
  // (気づかないまま評価すると「seed が 1 本足りない」理由が分からなくなる)
  const on = b.getAttribute("data-on") === "1";
  if ((res.moved || []).length)
    alert((on ? "評価の対象から外しました (run.py が拾わなくなります):\n"
              : "評価の対象に戻しました:\n") + res.moved.join("\n"));
  if ((res.move_errors || []).length)
    alert("モデルの改名に失敗したものがあります:\n" + res.move_errors.join("\n"));
  loadTrain();
});

// ── seed ごとのコメント ─────────────────────────────────────────────
// 保存先は tools/.run_notes.json (uid ごとに 1 本)。どのブラウザから見ても
// 同じものが出る。run が消えない限り done / failed どちらでも残る。
// running now と conditions のどちらから書いても同じ 1 本を編集する。
function noteCell(r) {
  return `<input class="note" data-uid="${esc(r.uid)}" value="${esc(r.note || "")}"
           placeholder="コメント" title="この seed のメモ。Enter か他をクリックで保存">`;
}

// 表は毎回作り直すので、個々の input ではなくタブ全体に 1 つ委譲する
// (running now / conditions のどちらの欄でも同じ処理が動く)
$("pane-train").addEventListener("change", async e => {
  const el = e.target;
  if (!el.classList || !el.classList.contains("note")) return;
  const uid = el.dataset.uid;
  const res = await api("/api/note", { uid, text: el.value });
  // 失敗時に res.text で上書きすると、打った文字が消える。残したまま知らせる
  if (apiFailed(res, "コメントの保存")) return;
  el.value = res.text || "";                   // サーバ側の整形 (trim/長さ) を反映
  // 同じ run の欄が 2 つの表に出ていることがある。両方に反映する
  const r = (TRAIN.runs || []).find(x => x.uid === uid);
  if (r) r.note = el.value;                    // 次の再描画で消えないように控える
  $("pane-train").querySelectorAll(`.note[data-uid="${CSS.escape(uid)}"]`)
    .forEach(o => { if (o !== el) o.value = el.value; });
  el.classList.add("saved");
  setTimeout(() => el.classList.remove("saved"), 900);
});

// Enter で確定 (change が飛ぶのでフォーカスを外すだけでよい)
$("pane-train").addEventListener("keydown", e => {
  if (e.key === "Enter" && e.target.classList
      && e.target.classList.contains("note")) e.target.blur();
});

// 60 秒ごとの自動更新で innerHTML を作り直すため、入力中だと打った字が
// 消えてしまう。描き直す前に控えて、あとで戻す
function grabNoteFocus() {
  const a = document.activeElement;
  return a && a.classList && a.classList.contains("note")
    ? { uid: a.dataset.uid, val: a.value, s: a.selectionStart, e: a.selectionEnd } : null;
}

function restoreNoteFocus(keep) {
  if (!keep) return;
  const el = $("pane-train").querySelector(`.note[data-uid="${CSS.escape(keep.uid)}"]`);
  if (!el) return;
  el.value = keep.val;
  el.focus();
  try { el.setSelectionRange(keep.s, keep.e); } catch (err) { /* 型が違えば諦める */ }
}

// 条件内で割れているパラメータの表。**どの設定の組が正しいか**を見出しに出す。
// 基準 = 学習を完了した (数えてよい) run がいちばん多い組。同数なら run の総数、
// それでも同じなら決めない (どちらも「?」)。列ごとに状態別の本数を添えるので、
// 「失敗した run だけが違う設定」なのか「完了した run どうしで割れている」のかが分かる
function splitTable(c) {
  const hs = c.param_hashes || [];
  const runs = (c.slots || []).map(s => s.run).filter(Boolean);
  const stat = hs.map(h => {
    const rs = runs.filter(r => r.param_hash === h.hash);
    const n = st => rs.filter(r => r.state === st).length;
    return { done: rs.filter(r => r.state === "done" && !r.excluded).length,
             total: h.seeds.length, running: n("running"),
             bad: rs.filter(r => r.state !== "done" && r.state !== "running").length };
  });
  const score = x => [x.done, x.total];
  let ref = -1;
  stat.forEach((x, i) => {
    if (ref < 0 || score(x)[0] > score(stat[ref])[0]
        || (score(x)[0] === score(stat[ref])[0] && x.total > stat[ref].total)) ref = i;
  });
  // 1 位が同点なら基準を決めない
  if (ref >= 0 && stat.some((x, i) => i !== ref && x.done === stat[ref].done
                                     && x.total === stat[ref].total)) ref = -1;
  const head = (h, i) => {
    const x = stat[i];
    const tag = ref < 0 ? `<span class="wrn">? 同数</span>`
      : (i === ref ? `<span class="c-ok">✔ 基準 (多数派)</span>`
                   : `<span class="err">✖ 違う</span>`);
    const cnt = [`完了 ${x.done}`, x.running ? `実行中 ${x.running}` : "",
                 x.bad ? `失敗・停止 ${x.bad}` : ""].filter(Boolean).join(" / ");
    return `<th title="seed: ${esc(h.seeds.join(", "))}">${tag}<br>`
         + `<span class="mut">${esc(h.hash)} · ${x.total} seed</span><br>`
         + `<span class="mut">${cnt}</span></th>`;
  };
  const cell = (v, i) => {
    const txt = v === "-" ? `<span title="この run の config にこのキーが無い (古い版で学習した run)">(キー無し)</span>`
                          : esc(v);
    const cls = ref < 0 ? "" : (i === ref ? "c-ok" : "wrn");
    return `<td class="${cls}">${txt}</td>`;
  };
  return `<table class="pdiff"><tr><th>key</th>${hs.map(head).join("")}</tr>`
    + c.param_diff.map(d => `<tr><td>${esc(d.key)}</td>${d.vals.map(cell).join("")}</tr>`).join("")
    + `</table>`;
}

// ── 稼働状況 (CPU / メモリ / GPU) ─────────────────────────────────────
// collect_runs.host_stats() が収集のたびに測る値。**マシン全体 (全アカウント)** の合計で、
// 共有マシンで他の人が使っている分も含む。「空き」を見て、あと何本回すかを決める
const pctCls = v => v == null ? "mut" : (v >= 90 ? "err" : (v >= 60 ? "wrn" : "c-ok"));
const meter = (v, txt) => v == null ? `<span class="mut">-</span>`
  : `<span class="pb" title="${Math.round(v)}%"><i class="${pctCls(v)}" style="width:${Math.min(100, v)}%"></i></span>`
    + ` <span class="${pctCls(v)}">${Math.round(v)}%</span>${txt ? ` <span class="mut sm">${txt}</span>` : ""}`;
const gb = mb => mb == null ? "?" : (mb / 1024).toFixed(1);

function renderLoad() {
  const H = TRAIN.host_stats || {};
  const ks = Object.keys(H).sort();
  if (!ks.length) {
    $("t-load").innerHTML = `<tr><td class="mut">まだ値がありません (次の収集で入ります)</td></tr>`;
    return;
  }
  const now = Date.now();
  $("t-load").innerHTML =
    "<tr><th>machine</th><th>いつの値か</th><th>CPU</th><th>メモリ</th>"
    + "<th>GPU 使用率</th><th>VRAM</th><th>空き</th><th class=\"num\">自分の学習</th></tr>"
    + ks.map(k => {
      const h = H[k];
      const age = h.at ? (now - Date.parse(h.at)) / 1000 : null;
      const memPct = h.mem_total_mb ? 100 * h.mem_used_mb / h.mem_total_mb : null;
      const g = (h.gpus || [])[0];
      const vPct = g && g.mem_total_mb ? 100 * g.mem_used_mb / g.mem_total_mb : null;
      // 空き: 何本足せるかの判断材料。CPU は「使われていないコア数」に直す
      const idleCores = h.cpu_pct == null || !h.cpu_count ? null
        : h.cpu_count * (100 - h.cpu_pct) / 100;
      const free = [
        idleCores == null ? "" : `CPU ${idleCores.toFixed(1)} / ${h.cpu_count} コア`,
        h.mem_total_mb ? `メモリ ${gb(h.mem_total_mb - h.mem_used_mb)} GB` : "",
        g ? `VRAM ${gb(g.mem_total_mb - g.mem_used_mb)} GB` : "",
      ].filter(Boolean).join(" / ");
      // 自分の学習 1 本あたりの VRAM (取れるマシンだけ)。空き VRAM と比べて足せる本数の目安にする
      const per = h.gpu_mem_mine_mb != null && h.train_procs
        ? ` <span class="mut sm" title="このリポジトリの学習 1 本あたりの VRAM">(1 本 ${gb(h.gpu_mem_mine_mb / h.train_procs)} GB)</span>` : "";
      // 繋がらないホストは値を薄く出し、「確認できません」と添える (赤くはしない)
      const unr = !!UNREACH()[k];
      return `<tr${unr ? ' class="unkrow"' : ""}><td>${esc(k)}</td>
        <td>${unr ? unkChip(k, ` <span class="mut sm">最終 ${age == null ? "-" : dur(age) + " 前"}</span>`)
                  : `<span class="${age != null && age > 3600 ? "err" : "mut"}">${age == null ? "-" : dur(age) + " 前"}</span>`}</td>
        <td>${meter(h.cpu_pct, h.load1 != null ? `load ${h.load1}` : "")}</td>
        <td>${meter(memPct, h.mem_total_mb ? `${gb(h.mem_used_mb)} / ${gb(h.mem_total_mb)} GB` : "")}</td>
        <td>${g ? meter(g.util, `${Math.round(g.temp)}°C`) : `<span class="mut">GPU なし</span>`}</td>
        <td>${g ? meter(vPct, `${gb(g.mem_used_mb)} / ${gb(g.mem_total_mb)} GB`) : ""}</td>
        <td>${esc(free)}</td>
        <td class="num">${h.train_procs ?? "-"}${per}</td></tr>`;
    }).join("");
}

// ── パラメータの警告 (一覧) ─────────────────────────────────────────
// 条件の表では、警告が各条件の行に散っていて見落とす。ここに集めて上に出す。
// 中身の判定は表と同じ (5 seed そろった条件の params✗ は出さない)。
// 条件名を押すと、表のその条件の差分パネルを開いてそこへ移動する
function renderParamWarn(W) {
  $("t-warn-h").hidden = !W.length;
  $("t-warn-n").textContent = W.length ? `${W.length} 件` : "";
  if (!W.length) { $("t-warn").innerHTML = ""; return; }
  const ALGO = a => String(a || "").toUpperCase();
  const cond = c => `<span class="attmain">${esc(c.map)} ${c.agents}台 ${esc(ALGO(c.algo))}</span>`
    + ` <span class="mut"><b>env:</b>${esc(c.setting)}</span>`
    + ` <span class="mut"><b>到着:</b>${esc(c.task_arrival)}</span>`
    + ` <span class="mut"><b>割当:</b>${esc(c.task_assign || "TP")}</span>`
    + (c.reassign ? ` <span class="mut"><b>再割当</b>あり</span>` : "")
    + (c.dynamic ? ` <span class="mut"><b>動的台数</b>あり</span>` : "");
  $("t-warn").innerHTML =
    "<tr><th>種類</th><th>計画</th><th>条件</th><th>違うパラメータ</th></tr>"
    + W.map(w => {
      const c = w.c;
      let kind, keys;
      if (w.kind === "split") {
        // 条件の中で seed ごとに設定が割れている。値はハッシュ (= 設定の組) ごとに並べる
        kind = `<span class="wrn" title="同じ条件の seed どうしで設定が違います">params✗ 条件内で割れ</span>`;
        keys = splitTable(c);
      } else {
        kind = `<span class="wrn" title="表の他の条件 (同じ algo・dynamic) と設定が違います。5 seed には数えたままです">他の条件と違う</span>`;
        keys = w.G.map(x => `<div><b>${esc(x.key)}</b> `
          + `この条件 <span class="wrn">✖ ${esc(fmtVal(x.actual))}</span>`
          + ` <span class="mut">(${x.n_this} run)</span>`
          + ` ／ 他の条件の多数派 <span class="c-ok">✔ ${esc(fmtVal(x.expected))}</span>`
          + ` <span class="mut">(${x.n_major}/${x.n_group} run)</span>`
          + ` <button type="button" class="ack" data-dismiss="${esc(x.dismiss_key)}"`
          + ` title="この差は意図したものなので警告を消す">問題なし</button></div>`).join("");
      }
      return `<tr><td>${kind}</td><td class="mut">${esc(c.plan || "")}</td>
        <td><span class="goto" role="button" tabindex="0" data-wopen="${w.n}"
             title="表のこの条件へ移動して差分を開く">${cond(c)}</span></td>
        <td class="wkeys">${keys}</td></tr>`;
    }).join("");
}

// 一覧の条件名 → 表の差分パネルを開いて移動。data-wopen は表の data-diff と同じ通し番号
function openWarnRow(el) {
  const n = el.getAttribute("data-wopen");
  const row = document.querySelector('tr.diffrow[data-n="' + n + '"]');
  if (!row) return;
  row.hidden = false;
  OPEN_DIFF.add(row.getAttribute("data-key"));
  document.querySelectorAll('[data-diff="' + n + '"]')
    .forEach(b => b.setAttribute("aria-expanded", "true"));
  // 差分パネルの 1 つ上が条件行。見出しが sticky なので center に寄せる
  const target = row.previousElementSibling || row;
  target.scrollIntoView({ behavior: "smooth", block: "center" });
  document.querySelectorAll("tr.flash").forEach(t => t.classList.remove("flash"));
  target.classList.add("flash");
  setTimeout(() => target.classList.remove("flash"), 2000);
}
document.addEventListener("click", ev => {
  const el = ev.target.closest && ev.target.closest("[data-wopen]");
  if (el) openWarnRow(el);
});
document.addEventListener("keydown", ev => {
  if (ev.key !== "Enter" && ev.key !== " ") return;
  const el = ev.target.closest && ev.target.closest("[data-wopen]");
  if (!el) return;
  ev.preventDefault();
  openWarnRow(el);
});

// ── 要確認 (異常終了) ───────────────────────────────────────────────
// running now の下に置く。OK を押すとサーバ側 (tools/.acked_runs.json) に
// 記録されるので、再読み込みしても別のブラウザから開いても出てこない。
// 同じ run が回し直して**また落ちた**ときは停止時刻が変わるので再び出る。
const ATT_LABEL = { failed: "異常終了", stalled: "停止 (更新なし)", short: "途中終了" };
const ATT_WHY = {
  failed: "プロセスが落ちています。ログを確認してください",
  stalled: "プロセスは残っていますが t_env が進んでいません",
  short: "t_max に届かないまま終わっています",
};

// 条件を conditions 表と同じ語彙で並べる。どれを回し直せばよいか、
// この 1 行だけ見て計画表の行を特定できるようにする
function attCond(r) {
  const chip = (label, v, cls) =>
    v == null || v === "" ? ""
      : ` <span class="${cls || "mut"}"><b>${label}</b>${esc(v)}</span>`;
  return `<span class="attmain">${esc(r.agents)}台 ${esc(r.map)} `
       + `${esc(String(r.algo || "").toUpperCase())}</span>`
       + chip("env:", r.setting)
       + chip("tag:", r.method_tag)
       + chip("LaRe:", r.lare_mode)
       + chip("到着:", r.task_arrival)
       + chip("割当:", r.task_assign || "TP")
       + (r.reassign ? ` <span class="mut"><b>再割当</b>あり</span>` : "")
       + (r.dynamic_agents ? ` <span class="mut"><b>動的台数</b>あり</span>` : "");
}

function renderAttention() {
  const A = TRAIN.attention || [];
  $("t-att-h").hidden = !A.length;
  if (!A.length) { $("t-att").innerHTML = ""; return; }
  $("t-att").innerHTML =
    "<tr><th>状態</th><th>machine</th><th>seed</th><th>条件</th>"
    + "<th class=\"num\">進捗</th><th>停止</th><th>経過</th><th></th></tr>"
    + A.map(r => {
      // 計画上の M と実際の t_max がずれていれば、回し直す前に気づけるようにする
      const tmax = (r.group_m != null && r.t_max != null
                    && Math.abs(r.t_max / 1e6 - r.group_m) > 0.6)
        ? ` <span class="wrn" title="計画は ${r.group_m}M です">⚠t_max</span>` : "";
      return `<tr>
        <td><span class="err" title="${esc(ATT_WHY[r.state] || "")}">✖ ${
            esc(ATT_LABEL[r.state] || r.state)}</span>${
            r.odd_params ? ' <span class="wrn" title="他の seed とパラメータが違います">params✗</span>' : ""}</td>
        <td>${esc(r.machine)}</td>
        <td class="mut">${esc(r.seed)}</td>
        <td>${attCond(r)}</td>
        <td class="num">${M(r.t_last)}/${M(r.t_max)}M${tmax}
            <span class="mut">(${((r.progress || 0) * 100).toFixed(0)}%)</span></td>
        <td class="mut">${when(r.stop_at || r.last_seen)}</td>
        <td class="mut">${esc(r.duration || "")}</td>
        <td><button class="ack" data-ack="${esc(r.ack_key)}"
             title="確認済みにしてこの一覧から消す (run 自体は残る)">OK</button></td>
      </tr>`;
    }).join("");
}

// 行の OK。表は毎回描き直すので、個々のボタンではなく表に 1 つ委譲する
$("t-att").addEventListener("click", async e => {
  const key = e.target.dataset && e.target.dataset.ack;
  if (!key) return;
  e.target.disabled = true;
  const res = await api("/api/ack", { keys: [key] });
  if (res && res.error) { e.target.disabled = false; return apiFailed(res, "確認済みにする"); }
  TRAIN.attention = (TRAIN.attention || []).filter(r => r.ack_key !== key);
  renderAttention();
});

$("t-att-all").addEventListener("click", async e => {
  const n = (TRAIN.attention || []).length;
  if (!n || !confirm(`要確認 ${n} 件をすべて確認済みにします。よろしいですか`)) return;
  e.target.disabled = true;
  await api("/api/ack", { keys: (TRAIN.attention || []).map(r => r.ack_key) });
  TRAIN.attention = [];
  renderAttention();
  e.target.disabled = false;
});

async function poll() {
  const st = await api("/api/status");
  const wasBusy = TRAIN && TRAIN.busy;
  if (st.busy) { $("stamp").textContent = "収集中…"; $("collect").disabled = true;
                 setTimeout(poll, 3000); return; }
  if (wasBusy || !TRAIN) await loadTrain(); else renderStamp();
}
/* ── 学習曲線 ─────────────────────────────────────────── */
// 計画の章 (計画 x マップ x 台数 x t_max) ごとに 1 枚の図。線 1 本 = run 1 本
// (「条件で平均」なら条件 1 本)。値は sacred の metrics.json を間引いたもの
let CURVE = null;
let CURVE_WANT = "";                     // URL の #curve/<計画名> で開く計画
const CURVE_HIDDEN = new Set();          // 凡例で消した線 (章ラベル|系列キー)
const CURVE_PALETTE = ["#0969da", "#cf222e", "#1a7f37", "#8250df", "#bc4c00", "#0f6b6b",
                       "#9a6700", "#e85aad", "#57606a", "#2da44e", "#6639ba", "#fb8f44"];
// 「アルゴリズム × setting」の色: アルゴリズムで色相 (図をまたいで固定)、setting で同じ色相の濃淡。
// 濃淡の順は 標準 → 明るい → 暗い → 中間 (safe = LaRe なしを標準の色にする)
const ALGO_FAMILIES = [
  ["#0969da", "#7cc0ff", "#06306b", "#3b8eea"],   // 青
  ["#cf222e", "#ff9492", "#82071e", "#e5534b"],   // 赤
  ["#1a7f37", "#6fdd8b", "#0b4a1f", "#2da44e"],   // 緑
  ["#8250df", "#c297ff", "#3e1f79", "#a371f7"],   // 紫
  ["#bc4c00", "#ffb070", "#762c00", "#e16f24"],   // 橙
  ["#0f6b6b", "#5fd3d3", "#073e3e", "#1b9999"],   // 青緑
];
const ALGO_FAMILY_OF = { QMIX: 0, MAPPO: 1, MAT: 2, MAT_DEC: 3, QPLEX: 4, IQL: 5, VDN: 5, TRANSF_QMIX: 4, HAPPO: 3 };
// 16 進の色 <-> HSL。濃淡を必要な段数だけ作るのに使う
function hexToHsl(hex) {
  const n = parseInt(hex.slice(1), 16), r = (n >> 16 & 255) / 255, g = (n >> 8 & 255) / 255, b = (n & 255) / 255;
  const mx = Math.max(r, g, b), mn = Math.min(r, g, b), l = (mx + mn) / 2, d = mx - mn;
  if (!d) return [0, 0, l];
  const sat = l > 0.5 ? d / (2 - mx - mn) : d / (mx + mn);
  const h = mx === r ? ((g - b) / d + (g < b ? 6 : 0)) : mx === g ? (b - r) / d + 2 : (r - g) / d + 4;
  return [h * 60, sat, l];
}
function hslToHex(h, sat, l) {
  const f = k => { const a = sat * Math.min(l, 1 - l), t = (k + h / 30) % 12;
                   return Math.round(255 * (l - a * Math.max(-1, Math.min(t - 3, 9 - t, 1)))); };
  return "#" + [f(0), f(8), f(4)].map(v => v.toString(16).padStart(2, "0")).join("");
}
// 基準の色から n 段の濃淡 (標準 → 明るい → 暗い → もっと明るい → …)。4 段までは手で選んだ表を使う
function shadesOf(family, n) {
  if (n <= family.length) return family;
  const [h, sat, l] = hexToHsl(family[0]), out = [family[0]];
  for (let k = 1; out.length < n; k++) {
    const step = Math.ceil(k / 2) * Math.min(0.12, 0.5 / n);
    out.push(hslToHex(h, sat, Math.max(0.15, Math.min(0.85, l + (k % 2 ? step : -step)))));
  }
  return out;
}
// 見分けやすい色を n 色。12 色までは CURVE_PALETTE、それより多ければ色相を黄金角でずらして作る
function categorical(i, n) {
  if (n <= CURVE_PALETTE.length) return CURVE_PALETTE[i];
  return hslToHex((i * 137.508) % 360, 0.65, i % 2 ? 0.55 : 0.4);
}
const settingOrder = (a, b) => (a === "safe" ? -1 : b === "safe" ? 1 : a.localeCompare(b, undefined, { numeric: true }));
const METRIC_LABEL = {
  test_task_completion_mean: "配送数 (test_task_completion_mean)",
  test_goal_mean: "ゴール数 (test_goal_mean)",
  test_collision_mean: "衝突数 (test_collision_mean)",
  test_return_mean: "return (test_return_mean。LaRe ありは LaRe の報酬なので条件間で比べられない)",
};
const curVer = s => { const m = / F(\d+)$/.exec(normSetting(s.setting)); return m ? "F" + m[1] : "F1"; };
const curRe = s => s.reassign == null ? "-" : (s.reassign ? "T" : "F");
// 絞り込み・色分け・凡例に使えるパラメータ。**計画の中で値が 2 通り以上あるものだけ**画面に出す
// (計画ごとに振っている軸が違う: lare_factor は事前学習の経路と版、arrival_range は到着率の範囲)。
// cond: true = 条件を決める軸 (凡例の名前・「条件で平均」のまとめ方に使う)
const CURVE_FIELDS = [
  { k: "setting",  label: "setting",  cond: true,  get: s => normSetting(s.setting) },
  { k: "ver",      label: "版",       cond: false, get: curVer },
  { k: "algo",     label: "algo",     cond: true,  get: s => String(s.algo || "").toUpperCase() },
  { k: "arrival",  label: "arrival",  cond: true,  get: s => s.task_arrival || "-" },
  { k: "assign",   label: "assign",   cond: true,  get: s => s.task_assign || "TP" },
  { k: "reassign", label: "reassign", cond: true,  get: s => "re " + curRe(s) },
  { k: "dynamic",  label: "dynamic",  cond: true,  get: s => s.dynamic ? "dyn" : "no dyn" },
  { k: "machine",  label: "machine",  cond: false, get: s => s.machine || "?" },
  { k: "state",    label: "状態",     cond: false, get: s => s.state },
];
const CF_SEL = {};                       // 絞り込みで選んでいる値 (計画を変えても残す)
let CF_VARY = [];                        // 今の計画で値が割れているパラメータ
const cfOf = k => CURVE_FIELDS.find(f => f.k === k);
// 凡例の名前: 割れている「条件の軸」だけを並べる (全部同じなら setting だけ)
const curCond = s => {
  const ks = CF_VARY.filter(f => f.cond);
  return (ks.length ? ks : [cfOf("setting")]).map(f => f.get(s)).join(" / ");
};

async function loadCurves() {
  // 2 回目以降 (自動の取り直し) は表示を消さない。消すと 1 分おきに画面がちらつく
  if (!CURVE) $("c-body").innerHTML = `<div class="mut">読み込み中…</div>`;
  CURVE = await api("/api/curves");
  const plans = uniq(CURVE.chapters, "plan");
  const pl = $("c-plan"), cur = pl.value;
  pl.innerHTML = plans.map(p => `<option>${esc(p)}</option>`).join("");
  // URL で指定された計画 (#curve/<計画名>) > 今選んでいる計画 > lare_factor > 一覧の最初
  if (plans.includes(CURVE_WANT)) { pl.value = CURVE_WANT; CURVE_WANT = ""; }
  else if (plans.includes(cur)) pl.value = cur;
  else if (plans.includes("lare_factor")) pl.value = "lare_factor";
  const ms = $("c-metric"), mcur = ms.value;
  ms.innerHTML = CURVE.metrics.map(m => `<option value="${m}">${esc(METRIC_LABEL[m] || m)}</option>`).join("");
  ms.value = CURVE.metrics.includes(mcur) ? mcur : "test_task_completion_mean";
  fillCurveFilters();
  renderCurves();
}
// 計画が決まったら、その計画で値が割れているパラメータを調べて絞り込み・色・線種の選択肢を作る
function fillCurveFilters() {
  const chs = CURVE.chapters.filter(ch => ch.plan === $("c-plan").value
    && ($("c-lare").checked || ch.collect !== "lare"));
  const S = chs.flatMap(ch => ch.series.map(s => Object.assign({ map: ch.map, agents: ch.agents }, s)));
  fillSelect($("c-map"), uniq(chs, "map"));
  fillSelect($("c-n"), uniq(chs, "agents"));
  CF_VARY = CURVE_FIELDS.filter(f => new Set(S.map(f.get)).size > 1);
  $("c-dyn").innerHTML = CF_VARY.map(f => {
    const vals = [...new Set(S.map(f.get))].sort();
    const cur = vals.includes(CF_SEL[f.k]) ? CF_SEL[f.k] : "";
    CF_SEL[f.k] = cur;
    return `<span><label>${esc(f.label)}</label><select data-cf="${f.k}">`
      + `<option value="">(all)</option>`
      + vals.map(v => `<option${v === cur ? " selected" : ""}>${esc(v)}</option>`).join("")
      + `</select></span>`;
  }).join("") || `<span class="mut">この計画で振っているパラメータはありません</span>`;
  // 色・線種に選べるのも、割れているパラメータだけ
  const opts = CF_VARY.filter(f => f.k !== "state");
  const col = $("c-color"), ccur = col.value;
  const both = opts.some(f => f.k === "algo") && opts.some(f => f.k === "setting");
  col.innerHTML = `<option value="auto">おまかせ</option>`
    + (both ? `<option value="algo_setting">algo (色) × setting (濃淡)</option>` : "")
    + opts.map(f => `<option value="${f.k}">${esc(f.label)}</option>`).join("")
    + `<option value="cond">条件ごと</option>`;
  // 既定の色: algo か setting が割れていれば「おまかせ」(章ごとに、アルゴリズムが複数なら algo で色相・
  // setting で濃淡、1 つなら LaRe の経路ごとに別の色・版で濃淡)。それ以外は割れている最初の「条件の軸」
  const firstCond = opts.find(f => f.cond);
  col.value = col.dataset.user && [...col.options].some(o => o.value === ccur) ? ccur
    : (opts.some(f => f.k === "algo" || f.k === "setting") ? "auto"
       : (firstCond ? firstCond.k : "cond"));
  // 既定の線種: 色に使っていない assign > reassign。自分で選んだら (data-user) それを残す
  const dsh = $("c-dash"), dcur = dsh.value;
  dsh.innerHTML = `<option value="">なし</option>`
    + opts.filter(f => f.k !== col.value
                       && !(["auto", "algo_setting"].includes(col.value) && ["algo", "setting", "ver"].includes(f.k)))
          .map(f => `<option value="${f.k}">${esc(f.label)}</option>`).join("");
  const dflt = ["assign", "reassign"].find(k => k !== col.value && opts.some(f => f.k === k)) || "";
  dsh.value = dsh.dataset.user && [...dsh.options].some(o => o.value === dcur) ? dcur : dflt;
}
// 移動平均 (前後 w/2 点)。端は取れる点だけで平均する
function smooth(ys, w) {
  if (w <= 1) return ys;
  const h = Math.floor(w / 2), out = [];
  for (let i = 0; i < ys.length; i++) {
    let s = 0, n = 0;
    for (let j = Math.max(0, i - h); j <= Math.min(ys.length - 1, i + h); j++)
      if (ys[j] != null) { s += ys[j]; n++; }
    out.push(n ? s / n : null);
  }
  return out;
}
// x (t_env) の位置の値。前後 3 点の平均 (評価は 1 点ごとに揺れるので)。届いていなければ null
function valueAt(xs, ys, x) {
  if (!xs.length || xs[xs.length - 1] < x * 0.97) return null;
  let i = 0, best = Infinity;
  xs.forEach((v, k) => { const d = Math.abs(v - x); if (d < best) { best = d; i = k; } });
  const sl = ys.slice(Math.max(0, i - 3), i + 4).filter(v => v != null);
  return sl.length ? sl.reduce((a, b) => a + b, 0) / sl.length : null;
}
// 同じ条件の run を 1 本にする。共通の x の格子に線形補間して平均する。
// 横軸は最も長い run の端まで。各点ではそこまで届いている run だけで平均する
// (短い run に合わせると、途中で止めた run や学習中の seed があるだけで、終わった run の後半が見えなくなる)
function meanSeries(list) {
  const end = Math.max(...list.map(s => s.xs[s.xs.length - 1]));
  const grid = [];
  for (let i = 0; i <= 200; i++) grid.push(end * i / 200);
  const interp = (xs, ys, x) => {
    if (x <= xs[0]) return ys[0];
    for (let k = 1; k < xs.length; k++)
      if (xs[k] >= x) { const t = (x - xs[k - 1]) / (xs[k] - xs[k - 1] || 1);
                        return ys[k - 1] + t * (ys[k] - ys[k - 1]); }
    return null;                               // この run はここまで届いていない
  };
  // 平均に加えて seed の範囲 (最小・最大) と標本標準偏差も持つ (図の帯に使う)
  const ys = [], mn = [], mx = [], sd = [];
  grid.forEach(x => {
    const v = list.map(s => interp(s.xs, s.ys, x)).filter(y => y != null && !isNaN(y));
    if (!v.length) { ys.push(null); mn.push(null); mx.push(null); sd.push(null); return; }
    const m = v.reduce((a, b) => a + b, 0) / v.length;
    ys.push(m); mn.push(Math.min(...v)); mx.push(Math.max(...v));
    sd.push(v.length > 1 ? Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / (v.length - 1)) : 0);
  });
  return { xs: grid, ys, ymin: mn, ymax: mx, ysd: sd };
}

// その run が作る LaRe モデルの名前 (元にしたモデル + この章のマップ_台数 学習量)。
// 例: setting "8x5_2 10M 8x5_3 5M" の 8x5 5agent 5M の章 -> "8x5_2 10M→8x5_3 5M→8x5_5 5M (F1)"
function lareMadeOf(s, ch) {
  const base = normSetting(s.setting).replace(/ F\d+$/, "");
  const toks = (base === "safe" || base === "unsafe") ? [] : base.split(" ");
  const parts = [];
  for (let i = 0; i + 1 < toks.length; i += 2) parts.push(toks[i] + " " + toks[i + 1]);
  parts.push(`${String(ch.map || "").replace(/^map_/, "")}_${ch.agents} ${Number(ch.t_max_m)}M`);
  return parts.join("→") + ` (${curVer(s)})`;
}
// LaRe モデル学習の章 (台数ごと) を、マップ x 学習量 (t_max) ごとの 1 章にまとめる。
// 学習量が違うと横軸の長さが違うので、別の図にする
function mergeLareChapters(chs) {
  const out = [], byMap = {};
  chs.forEach(ch => {
    if (ch.collect !== "lare") { out.push(ch); return; }
    const k = (ch.group || "") + "|" + ch.map + "|" + Number(ch.t_max_m);
    if (!byMap[k]) {
      byMap[k] = { plan: ch.plan, group: ch.group, map: ch.map, collect: "lare", merged: true,
                   label: ch.map + " " + Number(ch.t_max_m) + "M LaRe (merged)", agents: null,
                   t_max_m: Number(ch.t_max_m) || 0, series: [] };
      out.push(byMap[k]);
    }
    const m = byMap[k];
    ch.series.forEach(s => m.series.push(Object.assign({}, s, { agents: ch.agents, made: lareMadeOf(s, ch) })));
  });
  return out;
}

function renderCurves() {
  if (!CURVE) return;
  const metric = $("c-metric").value, w = +$("c-smooth").value, avg = $("c-mean").checked;
  const keep = s => CF_VARY.every(f => !CF_SEL[f.k] || f.get(s) === CF_SEL[f.k]);
  const chs = CURVE.chapters.filter(ch => ch.plan === $("c-plan").value
    && (!$("c-map").value || ch.map === $("c-map").value)
    && (!$("c-n").value || String(ch.agents) === $("c-n").value)
    && ($("c-lare").checked || ch.collect !== "lare"));
  let html = "", nRun = 0, nNo = 0, group = null;
  const per = $("c-per-agent").checked;
  const chs2 = $("c-lare").checked && $("c-lare-merge").checked ? mergeLareChapters(chs) : chs;
  chs2.forEach(ch => {
    let ss = ch.series.filter(keep);
    if (!ss.length) return;
    const runs = ss.slice();                  // 評価表用 (曲線の無い run も、平均でまとめる前の run も入れる)
    // 曲線のある run だけ図に出す。無いもの (取得前・旧版のマシン) は数だけ知らせる
    const noCurve = ss.filter(s => !(s.m && s.m[metric]));
    nNo += noCurve.length;
    // 台数で割る: まとめた章では run ごとの台数 (s.agents)、ふつうの章は章の台数
    const nag = s => Number(s.agents || ch.agents) || 1;
    ss = ss.filter(s => s.m && s.m[metric]).map(s => Object.assign({}, s, {
      xs: s.m[metric][0],
      ys: per ? s.m[metric][1].map(v => v == null ? null : v / nag(s)) : s.m[metric][1],
      key: (ch.merged ? s.made : curCond(s)) + (avg ? "" : ` / ${s.machine} ${s.seed}`) }));
    if (avg) {
      const by = {};
      const ck = s => ch.merged ? s.made : curCond(s);
      ss.forEach(s => (by[ck(s)] = by[ck(s)] || []).push(s));
      ss = Object.values(by).map(l => Object.assign({}, l[0], meanSeries(l),
        { key: ck(l[0]), n: l.length, state: l.every(s => s.state === "done") ? "done" : "running",
          machine: [...new Set(l.map(s => s.machine))].join("/"),
          // 表の学習の列を seed ごとの値から 平均 ± 標準偏差 にするため、元の run を持っておく
          members: l, seeds: l.map(s => s.seed) }));
    }
    if (!ss.length && !noCurve.length) return;
    nRun += ss.length;
    if (ch.group && ch.group !== group) { group = ch.group; html += `<div class="grouphead">${esc(group)}</div>`; }
    html += curveChapter(ch, ss, noCurve, metric, w, runs);
  });
  if (!html) html = `<div class="mut">この絞り込みで曲線のある run がありません`
    + ` (曲線は収集のたびに取りに行きます。始めたばかりなら次の収集を待ってください)</div>`;
  $("c-body").innerHTML = html;
  $("c-counts").textContent = `${nRun} 本` + (nNo ? ` (曲線未取得 ${nNo} 本)` : "")
    + (CURVE.collected_at ? `  収集 ${when(CURVE.collected_at)}` : "");
}

// ── 学習曲線の章に付ける評価表 ──
// 行 = 凡例と同じ条件 (seed はまとめて 平均 ± sd)、列 = 評価時の到着率。
// 評価時の reassign・割当は表の上の選択欄で 1 つに絞る (列に並べると読めなくなるため)
const EVAL_LOWER = new Set(["collision_rate", "task_dropped", "deadhead_ratio"]);
const arrOrder = a => a === "fixed" ? -1 : a === "mmpp" ? 99 : (parseFloat(String(a).replace("bern", "")) || 50);
// run 1 本の評価結果から、選択欄に合うものを 到着率 -> 値 で取り出す
// 返り値: 実行時 reassign ("off" / "on") -> 到着率 -> その評価の全指標 ({task_completion: …})
function runEvalVals(s, nag) {
  const al = $("c-ealloc").value;
  const want = al || String(s.task_assign || "TP").toLowerCase();   // 既定は学習時と同じ割当
  const out = { off: {}, on: {}, rsv: { off: {}, on: {} } };
  (s.ev || []).forEach(e => {
    if (e.n !== nag || e.allocator !== want) return;
    const side = e.env_reassign ? "on" : "off";
    if (e.reserved) {                       // 予約 (まだ結果の無い条件を評価中)
      if (!(e.arrival in out.rsv[side])) out.rsv[side][e.arrival] = e.reserved;
      return;
    }
    const o = out[side];
    if (!(e.arrival in o)) o[e.arrival] = e.vals;
  });
  return out;
}
// 評価表のかたまり = チェックした指標 x チェックした実行時 reassign (選択欄の並び順)
const evMetrics = () => [...document.querySelectorAll("input.c-em:checked")].map(x => x.value);
const evReassigns = () => [...document.querySelectorAll("input.c-er:checked")].map(x => x.value);
const evBlocks = () => evMetrics().flatMap(m => evReassigns().map(re => ({ m, re })));
// 平均 ± sd (標本標準偏差)。値が 1 つなら sd は出さない
function evStat(xs) {
  if (!xs.length) return null;
  const m = xs.reduce((a, b) => a + b, 0) / xs.length;
  const sd = xs.length > 1 ? Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / (xs.length - 1)) : null;
  return { m, sd, n: xs.length, xs };
}

function curveChapter(ch, ss, noCurve, metric, w, runs) {
  const tmax = (ch.t_max_m || 1) * 1e6;
  const colorF = cfOf($("c-color").value), dashF = cfOf($("c-dash").value);
  const colorKey = s => ch.merged ? s.made : (colorF ? colorF.get(s) : s.key);
  const ckeys = [...new Set(ss.map(colorKey))].sort();
  let color = s => CURVE_PALETTE[ckeys.indexOf(colorKey(s)) % CURVE_PALETTE.length];
  const cmode = $("c-color").value;
  if ((cmode === "auto" || cmode === "algo_setting") && !ch.merged) {
    // おまかせ: 章の中身で決める
    //   アルゴリズムが複数 -> アルゴリズムで色相 (図をまたいで固定)、setting で濃淡
    //   アルゴリズムが 1 つ -> LaRe の経路 (setting から版を除いたもの) ごとに別の色、版 (F1/F2/F3) で濃淡
    const fa = cfOf("algo"), fs = cfOf("setting");
    const algos = [...new Set(ss.map(fa.get))];
    const multiAlgo = algos.length > 1 || cmode === "algo_setting";
    const chain = s => normSetting(s.setting).replace(/ F\d+$/, "") || "-";
    const hueKey = multiAlgo ? fa.get : chain;
    const shadeKey = multiAlgo ? fs.get : curVer;
    const hues = [...new Set(ss.map(hueKey))].sort(settingOrder);
    const sub = {};
    ss.forEach(s => (sub[hueKey(s)] = sub[hueKey(s)] || new Set()).add(shadeKey(s)));
    const base = h => multiAlgo
      ? ALGO_FAMILIES[(h in ALGO_FAMILY_OF ? ALGO_FAMILY_OF[h]
                       : Object.keys(ALGO_FAMILY_OF).length + hues.indexOf(h)) % ALGO_FAMILIES.length]
      : [categorical(hues.indexOf(h), hues.length)];
    color = s => {
      const h = hueKey(s), l = [...sub[h]].sort(settingOrder);
      const f = shadesOf(base(h), l.length);
      return f[l.indexOf(shadeKey(s)) % f.length];
    };
  }
  // 線種: 選んだパラメータの 1 つ目の値 (並べ替えて先頭) を実線、それ以外を破線
  // (assign は基準の TP を実線にする。reassign は "re F" が先頭に来る)
  const dvals = dashF ? [...new Set(ss.map(dashF.get))].sort() : [];
  const dfirst = dvals.includes("TP") ? "TP" : dvals[0];
  const dash = s => dashF && dashF.get(s) !== dfirst ? "6 4" : "";
  const hid = s => CURVE_HIDDEN.has(ch.label + "|" + s.key);
  ss.sort((a, b) => a.key.localeCompare(b.key));
  ss.forEach(s => { s.sy = smooth(s.ys, w); });
  // 同じ条件の seed が 2 本以上あるときの帯 (最小〜最大 / 平均 ± 標準偏差)。平均の線と同じ平滑化をかける
  const band = $("c-band").value;
  ss.forEach(s => {
    s.blo = s.bhi = null;
    if (!band || !(s.n > 1) || !s.ymin) return;
    if (band === "sd") {
      s.blo = smooth(s.ys.map((m, i) => m == null ? null : m - s.ysd[i]), w);
      s.bhi = smooth(s.ys.map((m, i) => m == null ? null : m + s.ysd[i]), w);
    } else {
      s.blo = smooth(s.ymin, w);
      s.bhi = smooth(s.ymax, w);
    }
  });

  const W = 820, H = 320, L = 52, R = 16, T = 12, B = 32;
  const vis = ss.filter(s => !hid(s));
  const vals = vis.flatMap(s => [...s.sy, ...(s.blo || []), ...(s.bhi || [])].filter(v => v != null));
  let lo = vals.length ? Math.min(...vals) : 0, hi = vals.length ? Math.max(...vals) : 1;
  if (lo === hi) { lo -= 1; hi += 1; }
  const pad = (hi - lo) * 0.05; lo -= pad; hi += pad;
  const sx = x => L + (x / tmax) * (W - L - R);
  const sy = v => H - B - ((v - lo) / (hi - lo)) * (H - T - B);
  let g = `<line x1="${L}" y1="${H - B}" x2="${W - R}" y2="${H - B}" stroke="var(--line)"/>`;
  for (let i = 0; i <= 4; i++) {
    const v = lo + i / 4 * (hi - lo), y = sy(v);
    g += `<line x1="${L}" y1="${y}" x2="${W - R}" y2="${y}" stroke="var(--line)" stroke-dasharray="2 3"/>`
       + `<text x="${L - 6}" y="${y + 4}" text-anchor="end" fill="var(--mut)" font-size="10">${num(v, hi - lo < 2 ? 2 : 1)}</text>`;
  }
  for (let i = 0; i <= 5; i++) {
    const x = tmax * i / 5;
    g += `<text x="${sx(x)}" y="${H - B + 16}" text-anchor="middle" fill="var(--mut)" font-size="10">${num(x / 1e6, 1)}M</text>`;
  }
  vis.forEach(s => {
    if (!s.blo) return;
    const up = [], dn = [];
    s.xs.forEach((x, i) => {
      if (s.blo[i] == null || s.bhi[i] == null) return;
      const X = sx(Math.min(x, tmax)).toFixed(1);
      up.push(`${X},${sy(s.bhi[i]).toFixed(1)}`);
      dn.push(`${X},${sy(s.blo[i]).toFixed(1)}`);
    });
    if (up.length > 1)
      g += `<polygon fill="${color(s)}" fill-opacity="0.15" stroke="none" points="${up.concat(dn.reverse()).join(" ")}">`
         + `<title>${esc(s.key)}: ${s.n} seed の${band === "sd" ? "平均 ± 標準偏差" : "範囲 (最小〜最大)"}</title></polygon>`;
  });
  vis.forEach(s => {
    const pts = s.xs.map((x, i) => s.sy[i] == null ? null : `${sx(Math.min(x, tmax)).toFixed(1)},${sy(s.sy[i]).toFixed(1)}`)
      .filter(Boolean).join(" ");
    g += `<polyline fill="none" stroke="${color(s)}" stroke-width="1.6" stroke-dasharray="${dash(s)}"`
       + ` points="${pts}"><title>${esc(s.key)}</title></polyline>`;
    if (s.state === "running") {              // 学習中の先端に印
      const i = s.xs.length - 1;
      if (s.sy[i] != null)
        g += `<circle cx="${sx(Math.min(s.xs[i], tmax))}" cy="${sy(s.sy[i])}" r="3" fill="${color(s)}"><title>${esc(s.key)} (学習中)</title></circle>`;
    }
  });

  const legend = ss.map(s => {
    const k = ch.label + "|" + s.key;
    return `<span class="cleg${hid(s) ? " off" : ""}" data-ckey="${esc(k)}">`
      + `<svg width="22" height="8"><line x1="0" y1="4" x2="22" y2="4" stroke="${color(s)}" stroke-width="2"`
      + ` stroke-dasharray="${dash(s)}"/></svg>${esc(s.key)}${
        // 条件でまとめたときは seed を並べる (どの run の平均かが分かるように)
        s.seeds ? ` <span class="mut sm">(${s.seeds.length > 1 ? s.seeds.length + " seed: " : "seed "}${
          esc(s.seeds.join(", "))})</span>` : ""}</span>`;
  }).join("");

  // 表: t_max の 10 / 25 / 50 / 75 / 100% の時点の値 (前後 3 点の平均。平滑化はかけない)
  const fr = [0.1, 0.25, 0.5, 0.75, 1.0];
  const better = metric === "test_collision_mean" ? -1 : 1;
  // 学習の列の値: 条件でまとめた行は seed ごとの値の 平均 ± 標準偏差 (評価の列と同じ evStat)。
  // 平均の曲線から読むと、一番短い seed の位置で切れて 100% の値が出ないため
  const tstat = (s, f) => s.members
    ? evStat(s.members.map(m => valueAt(m.xs, m.ys, f * tmax)).filter(v => v != null))
    : (v => v == null ? null : { m: v, sd: null, n: 1, xs: [v] })(valueAt(s.xs, s.ys, f * tmax));
  const bestOf = fr.map(f => {
    const vs = vis.map(s => (tstat(s, f) || {}).m).filter(v => v != null);
    return vs.length ? (better > 0 ? Math.max(...vs) : Math.min(...vs)) : null;
  });
  // 表は 1 つ: 行 = 図の線 (「条件で平均」なら条件、外せば run)、列 = 学習 (t_max の各時点) | 評価 (到着率)
  const ckey = s => ch.merged ? s.made : curCond(s);
  const avg = $("c-mean").checked;
  const evOn = $("c-evshow").checked;
  const nagOf = s => Number(s.agents || ch.agents);
  // 行ごとの評価: 平均なら同じ条件の run (曲線の無い run も含む) を seed として平均 ± sd、run の行ならその run の値
  const evRuns = s => avg ? (runs || []).filter(r => ckey(r) === ckey(s)) : [s];
  const evVals = ss.map(s => evRuns(s).map(r => runEvalVals(r, nagOf(r))));
  let ecols = [...new Set(evVals.flatMap(l => l.flatMap(v => [...Object.keys(v.off), ...Object.keys(v.on),
                                                                ...Object.keys(v.rsv.off), ...Object.keys(v.rsv.on)])))]
    .sort((a, b) => arrOrder(a) - arrOrder(b));
  // 評価結果がまだ無い章は、既定の評価条件 (default.yaml の eval_arrivals) を列にした空の欄にする
  const evEmpty = !ecols.length;
  if (evEmpty) ecols = (CURVE.eval_arrivals || []).slice().sort((a, b) => arrOrder(a) - arrOrder(b));
  // かたまり (指標 x 実行時 reassign) ごとに到着率の列を作り、かたまりの間を縦線で区切る
  const blocks = evOn ? evBlocks() : [];
  // estats[行][かたまり][到着率]
  // 予約: 結果の無い欄に「…」を出す (行の run のどれかが予約されていれば)
  const eres = evVals.map(l => blocks.map(b => ecols.map(c => l.map(v => v.rsv[b.re][c]).find(Boolean) || null)));
  const estats = evVals.map(l => blocks.map(b => ecols.map(c =>
    evStat(l.map(v => v[b.re][c] && v[b.re][c][b.m]).filter(v => v != null)))));
  const ebest = blocks.map((b, k) => ecols.map((_c, i) => {
    const ms = estats.filter((_r, j) => !hid(ss[j])).map(r => r[k][i] && r[k][i].m).filter(v => v != null);
    return ms.length > 1 ? (EVAL_LOWER.has(b.m) ? Math.min(...ms) : Math.max(...ms)) : null;
  }));
  // 小数の桁はかたまりごと (1 未満の値しかない指標は 3 桁)
  const end = blocks.map((_b, k) => {
    const ms = estats.flatMap(r => r[k]).filter(Boolean).map(c => Math.abs(c.m));
    return ms.length && Math.max(...ms) < 1 ? 3 : ms.some(v => v < 10) ? 2 : 1;
  });
  // 表示する列の並び: 指標 > 到着率 > 実行時 reassign (bern0.1 の下に off / on)。gap = 区切り線の太さ
  const ems = evOn ? evMetrics() : [], ers = evOn ? evReassigns() : [];
  const ecolSeq = ems.flatMap(m => ecols.flatMap((_c, i) => ers.map((re, r) => ({
    k: blocks.findIndex(b => b.m === m && b.re === re), i,
    gap: i === 0 && r === 0 ? " egap" : r === 0 ? " agap" : "" }))));
  const rsvTitle = rv => `評価中: ${rv.host || "?"} (${new Date(rv.until * 1000).toLocaleString("ja-JP",
    { month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" })} ごろまで)`;
  const evCell = (c, k, i, nrun, gap, rv) => !c
    ? (rv ? `<td class="num mut${gap}" title="${esc(rsvTitle(rv))}">…</td>` : `<td class="num mut${gap}">-</td>`)
    : `<td class="num${gap}${c.m === ebest[k][i] ? " cbest" : ""}" title="${esc(c.xs.map(x => num(x, end[k])).join(", "))}">`
      + `${num(c.m, end[k])}${c.sd != null ? `<span class="mut sm"> ±${num(c.sd, end[k])}</span>` : ""}`
      + (c.n < nrun ? `<span class="mut sm"> (${c.n})</span>` : "") + `</td>`;

  const rows = ss.map((s, j) => {
    // 状態は左端に短く: 完了 = ✓、学習中 = 進み具合 (%)
    return `<tr${hid(s) ? ` class="mut"` : ""}>`
      + `<td class="cst">${s.state === "done" ? `<span class="c-ok" title="done">✓</span>` : `<span title="${esc(s.state)}">${Math.round((s.progress || 0) * 100)}%</span>`}</td>`
      + `<td><i class="dot" style="background:${color(s)}"></i>${esc(s.key)}</td>`
      + fr.map((f, i) => { const c = tstat(s, f), v = c ? c.m : null;
          return `<td class="num${i === 0 ? " egap" : ""}${v != null && v === bestOf[i] && vis.length > 1 ? " cbest" : ""}"`
            + `${c && c.xs.length > 1 ? ` title="${esc(c.xs.map(x => num(x, 1)).join(", "))}"` : ""}>`
            + (v == null ? `<span class="mut">-</span>`
               : num(v, 1) + (c.sd != null ? `<span class="mut sm"> ±${num(c.sd, 1)}</span>` : "")
                 + (s.members && c.n < s.members.length ? `<span class="mut sm"> (${c.n})</span>` : ""))
            + `</td>`; }).join("")
      + ecolSeq.map(o => evCell(estats[j][o.k][o.i], o.k, o.i, evRuns(s).length, o.gap, eres[j][o.k][o.i])).join("") + `</tr>`;
  }).join("");
  const condHead = (ch.merged ? "作る LaRe モデル (版)"
    : (CF_VARY.filter(f => f.cond).map(f => f.label).join(" / ") || "setting"))
    + (avg ? "" : " / machine seed");
  const mlabel = METRIC_LABEL[metric] || metric;
  // 見出しは 3 段 (評価の指標 / 到着率 / 実行時 reassign)。評価が無ければ 2 段
  const ev3 = ecolSeq.length > 0, hr = ev3 ? 3 : 2;
  const head = `<tr><th rowspan="${hr}" class="cst">状態</th><th rowspan="${hr}">${esc(condHead)}</th>`
    + `<th class="grp egap" colspan="${fr.length}">学習: ${esc(mlabel)} <span class="mut sm">(test 平均${avg ? "。seed の平均 ± 標準偏差" : ""})</span></th>`
    + ems.map(m => `<th class="grp egap" colspan="${ecols.length * ers.length}">評価: ${esc(m)}${EVAL_LOWER.has(m) ? ` <span class="mut sm">低いほど良い</span>` : ""}</th>`).join("")
    + `</tr><tr>${fr.map((f, i) => `<th rowspan="${hr - 1}" class="num${i === 0 ? " egap" : ""}">${num(f * ch.t_max_m, 1)}M</th>`).join("")}`
    + ems.map(() => ecols.map((c, i) => `<th class="grp${i === 0 ? " egap" : " agap"}" colspan="${ers.length}">${esc(c)}</th>`).join("")).join("")
    + (ev3 ? `</tr><tr>` + ecolSeq.map(o => `<th class="num mut sm${o.gap}" title="実行時 reassign">re ${esc(blocks[o.k].re)}</th>`).join("") : "")
    + `</tr>`;
  const note = evOn ? `<div class="mut sm">評価 (平均 ± sd) の条件: 割当 ${esc($("c-ealloc").value || "学習と同じ")}`
    + (avg ? "・seed = 同じ条件の run" : "") + "。セルにマウスを乗せると seed ごとの値"
    + (evEmpty ? ` <span class="wrn">評価結果なし (評価前、または保管リポジトリに無い run)</span>` : "") + `</div>` : "";

  const svg = `<svg viewBox="0 0 ${W} ${H}" width="100%">${g}</svg>`;
  // 凡例は図の右。1 項目 1 行 (折り返さない) で縦に並べ、LEG_ROWS 行を超えたら列を足す
  const LEG_ROWS = 15;
  const body = !ss.length ? "" : `<div class="cpair"><div class="cchart">${svg}</div>`
    + `<div class="legend vlegend" style="grid-template-rows:repeat(${Math.min(ss.length, LEG_ROWS)}, auto)">${legend}</div></div>`
    + note + `<div class="wrap"><table class="ctbl">${head}${rows}</table></div>`;

  return `<h3><span class="g-map">${esc(ch.map || "")}</span>`
    + (ch.merged ? `<span class="g-n">LaRe モデル学習 (台数をまとめた図)</span><span class="g-t">${ch.t_max_m}M</span>`
                 : `<span class="g-n">${ch.agents} agent</span><span class="g-t">${ch.t_max_m}M</span>`)
    + ($("c-per-agent").checked ? `<span class="mut sm"> 1 台あたり</span>` : "") + `</h3>`
    + body
    + (noCurve.length ? `<div class="mut sm">曲線をまだ取れていない run: ${noCurve.map(s =>
        esc(curCond(s) + " / " + s.machine)).join(", ")}</div>` : "");
}

document.addEventListener("click", ev => {
  const el = ev.target.closest && ev.target.closest("[data-ckey]");
  if (!el) return;
  const k = el.getAttribute("data-ckey");
  if (CURVE_HIDDEN.has(k)) CURVE_HIDDEN.delete(k); else CURVE_HIDDEN.add(k);
  renderCurves();
});
// 計画ごとに作り直す絞り込み (c-dyn の中の select) は委譲で受ける
document.addEventListener("change", ev => {
  const el = ev.target.closest && ev.target.closest("[data-cf]");
  if (!el) return;
  CF_SEL[el.getAttribute("data-cf")] = el.value;
  renderCurves();
});
$("c-plan").addEventListener("change", () => { fillCurveFilters(); renderCurves(); });
$("c-lare").addEventListener("change", () => { fillCurveFilters(); renderCurves(); });
$("c-lare-merge").addEventListener("change", () => {
  if ($("c-lare-merge").checked && !$("c-lare").checked) { $("c-lare").checked = true; fillCurveFilters(); }
  renderCurves();
});
$("c-per-agent").addEventListener("change", renderCurves);
// 評価の指標のチェック。選んだものはこのブラウザに覚えておく
try {
  const v = localStorage.getItem("ldrp.curve.emetrics");
  if (v != null) document.querySelectorAll("input.c-em").forEach(x => { x.checked = v.split(",").includes(x.value); });
} catch (e) {}
document.querySelectorAll("input.c-em").forEach(x => x.addEventListener("change", () => {
  try { localStorage.setItem("ldrp.curve.emetrics", [...document.querySelectorAll("input.c-em:checked")].map(y => y.value).join(",")); } catch (e) {}
  renderCurves();
}));
// 実行時 reassign のチェック (off / on)。こちらも覚えておく
try {
  const v = localStorage.getItem("ldrp.curve.ereassign");
  if (v != null) document.querySelectorAll("input.c-er").forEach(x => { x.checked = v.split(",").includes(x.value); });
} catch (e) {}
document.querySelectorAll("input.c-er").forEach(x => x.addEventListener("change", () => {
  try { localStorage.setItem("ldrp.curve.ereassign", [...document.querySelectorAll("input.c-er:checked")].map(y => y.value).join(",")); } catch (e) {}
  renderCurves();
}));
// 評価表の表示・非表示。選んだ状態はこのブラウザに覚えておく (読めない環境でも動くよう try で包む)
try { if (localStorage.getItem("ldrp.curve.evshow") === "0") $("c-evshow").checked = false; } catch (e) {}
const syncEvOpts = () => { $("c-evopts").hidden = !$("c-evshow").checked; };
syncEvOpts();
$("c-evshow").addEventListener("change", () => {
  try { localStorage.setItem("ldrp.curve.evshow", $("c-evshow").checked ? "1" : "0"); } catch (e) {}
  syncEvOpts();
  renderCurves();
});
$("c-color").addEventListener("change", () => { $("c-color").dataset.user = "1"; fillCurveFilters(); renderCurves(); });
$("c-dash").addEventListener("change", () => { $("c-dash").dataset.user = "1"; });
["c-metric", "c-map", "c-n", "c-dash", "c-smooth", "c-mean", "c-band",
 "c-ealloc"]
  .forEach(id => $(id).addEventListener("change", renderCurves));

setInterval(() => { if ($("auto").checked) loadTrain(); }, 60000);
// 学習曲線タブを開いている間は 1 分おきに取り直す (サーバ側の収集は 5 分おき)。
// 学習中の run の曲線を追えるようにするため
setInterval(() => { if ($("auto").checked && tab === "curve") loadCurves(); }, 60000);

loadTrain();
// URL の末尾が #curve / #curve/<計画名> / #eval なら、そのタブで開く (ブックマーク用)
if (location.hash.startsWith("#curve")) {
  CURVE_WANT = decodeURIComponent(location.hash.split("/")[1] || "");
  // #curve/<計画名>/lare で、LaRe モデル学習をマップごとにまとめた図を開く
  if (location.hash.split("/")[2] === "lare") { $("c-lare").checked = true; $("c-lare-merge").checked = true; }
  setTab("curve");
}
else if (location.hash === "#eval") setTab("eval");


// ---------------------------------------------------------------------------
// 自動実行 (tools/autorun.py) の依頼。投入そのものは autorun.py tick (15 分おき) が行う
// ---------------------------------------------------------------------------
let AUTORUN = null;
async function loadAutorun() {
  try { AUTORUN = await api("/api/autorun"); } catch (e) { AUTORUN = null; }
  renderAutorun();
}
function renderAutorun() {
  const tbl = $("ar-tbl"), ctl = $("ar-ctl");
  if (!tbl) return;
  if (!AUTORUN || !AUTORUN.hosts) {
    tbl.innerHTML = `<tr><td class="mut">サーバを再起動すると使えます</td></tr>`;
    ctl.innerHTML = "";
    return;
  }
  const on = AUTORUN.hosts.filter(h => h.enabled).map(h => h.label);
  ctl.innerHTML = `<button type="button" class="ack" id="ar-pause">${AUTORUN.paused ? "再開" : "一時停止"}</button>`
    + ` <button type="button" class="ack" id="ar-tick" title="15 分おきの実行を待たずに 1 回投入する">今すぐ投入</button>`
    + (AUTORUN.paused ? ` <span class="wrn">一時停止中 (新しい投入をしません)</span>` : "")
    + (on.length ? ` <span class="mut sm">使うマシン: ${esc(on.join(", "))} / 学習は全体で ${AUTORUN.max_train_total} 本まで</span>`
                 : ` <span class="wrn sm">投入できるマシンがありません (collect_config.yaml の autorun.enabled)</span>`);
  const R = AUTORUN.requests || [];
  const ST = { open: "受付中", done: "完了", cancelled: "取り消し", error: "エラー" };
  tbl.innerHTML = "<tr><th>依頼</th><th>種類</th><th>条件</th><th class=\"num\">本数</th><th>マシン</th><th>状態</th><th></th></tr>"
    + (R.length ? R.map(r => `<tr${r.status === "open" ? "" : ' class="mut"'}>
        <td>${esc(r.id)}</td>
        <td>${r.kind === "train" ? "学習" + (r.then_eval ? "+評価" : "") : "評価"}</td>
        <td class="sm">${esc([].concat(r.cond_text || []).join(" / "))}</td>
        <td class="num">${r.kind === "train" ? `${r.launched}/${r.count}` : (r.models ? r.models + " モデル" : "-")}</td>
        <td>${esc((r.hosts && r.hosts.length ? r.hosts : (r.machines || ["自動"])).join(", "))}</td>
        <td>${esc(ST[r.status] || r.status)}${r.note ? ` <span class="wrn sm" title="${esc(r.note)}">⚠</span>` : ""}</td>
        <td>${r.status === "open" ? `<button type="button" class="ack ar-cancel" data-id="${esc(r.id)}"`
             + ` title="まだ投入していない分を取り消す (動いている run は止めない)">取り消す</button>` : ""}</td></tr>`).join("")
      : `<tr><td class="mut" colspan="7">依頼はありません</td></tr>`);
}
// 条件行の「依頼」: 下に小さな入力欄を開く
function arForm(id) {
  const hosts = (AUTORUN && AUTORUN.hosts) || [];
  return `<tr class="arrow" data-ar-row="${esc(id)}"><td colspan="8">
    <span class="arform">学習 <input type="number" class="ar-n" min="1" max="10" value="1"> 本
    マシン <select class="ar-m"><option value="">自動</option>${hosts.map(h =>
      `<option value="${esc(h.label)}"${h.enabled ? "" : " disabled"}>${esc(h.label)}${h.enabled ? "" : " (無効)"}</option>`).join("")}</select>
    <label><input type="checkbox" class="ar-ev"> 終わったら評価</label>
    <button type="button" class="ack ar-train" data-id="${esc(id)}">学習を依頼</button>
    <button type="button" class="ack ar-eval" data-id="${esc(id)}" title="この条件の完了した run を評価する (評価済みの条件は飛ばす)">評価を依頼</button>
    <span class="mut sm ar-msg"></span></span></td></tr>`;
}
document.addEventListener("click", async ev => {
  const t = ev.target;
  if (t.classList.contains("arbtn")) {
    const tr = t.closest("tr"), id = t.getAttribute("data-ar");
    const open = tr.nextElementSibling && tr.nextElementSibling.getAttribute("data-ar-row") === id;
    if (open) tr.nextElementSibling.remove();
    else tr.insertAdjacentHTML("afterend", arForm(id));
    return;
  }
  if (t.classList.contains("ar-train") || t.classList.contains("ar-eval")) {
    const row = t.closest("tr"), msg = row.querySelector(".ar-msg");
    const m = row.querySelector(".ar-m").value;
    const body = { kind: t.classList.contains("ar-train") ? "train" : "eval", ids: [t.getAttribute("data-id")],
                   n: Number(row.querySelector(".ar-n").value) || 1,
                   then_eval: row.querySelector(".ar-ev").checked, machines: m ? [m] : [] };
    t.disabled = true;
    const res = await api("/api/autorun/add", body);
    msg.textContent = res.ok ? `${res.ids.join(", ")} を受け付けました (次の投入で実行)` : `失敗: ${res.error || ""}`;
    t.disabled = false;
    loadAutorun();
    return;
  }
  if (t.classList.contains("ar-cancel")) {
    await api("/api/autorun/cancel", { id: t.getAttribute("data-id") });
    loadAutorun();
    return;
  }
  if (t.id === "ar-pause") {
    await api("/api/autorun/pause", { on: !(AUTORUN && AUTORUN.paused) });
    loadAutorun();
    return;
  }
  if (t.id === "ar-tick") {
    t.disabled = true;
    await api("/api/autorun/tick", {});
    t.textContent = "投入中… (結果は依頼の表に出ます)";
    setTimeout(loadAutorun, 60000);
  }
});
