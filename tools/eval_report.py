#!/usr/bin/env python3
"""LDRP: 方策評価の集計結果 (results/summary.csv) をダッシュボードで見る.

`aggregate.py` が出した summary.csv を読み、条件を分解して

  - 端末で見る表            --format table
  - ブラウザで見る 1 枚 HTML  --format html
  - 他のツールが読む JSON    --format json

を出す。集計そのもの (重複検出 / ddof=1 / seed 数) は aggregate.py が
既に持っているので、ここでは再実装しない。summary.csv を読むだけ。

    python tools/eval_report.py --format html -o eval.html && open eval.html
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import OrderedDict

# planner 名には "_" が入る (mat_dec / transf_qmix) ので、長い順に最長一致させる。
# `safe_mat_dec_dbct_base_tp` を "_" で切っても mat_dec+dbct と mat+dec_dbct を
# 区別できないため。新しい planner を足したらここにも足す
DEFAULT_PLANNERS = ("transf_qmix", "mat_dec", "qplex", "mappo", "qmix",
                    "iql", "vdn", "mat", "pbs")
REASSIGN_TOKENS = ("base", "reassign")
DEFAULT_METRICS = ("task_completion", "time_sec")

COND_HEAD_RE = re.compile(r"^map_(?P<map>.+?)/(?P<n>\d+)agent/(?P<rest>.+)$")
TRAIN_RE = re.compile(r"^train(\d+)$")


def parse_condition(cond, planners=DEFAULT_PLANNERS):
    """runner.py の _condition_id() を分解する.

        {map}/{N}agent/{env}_{planner}[_{tag}]_{reassign}_{assigner}[_train{N}][_envreassign]

    古い summary.csv には reassign / tag が無い ({env}_{planner}_{assigner}) ので、
    "base"/"reassign" が見つかるかどうかで新旧を判定する。
    """
    m = COND_HEAD_RE.match(cond)
    if not m:
        return None
    out = {"condition": cond, "map": "map_" + m.group("map"),
           "n": int(m.group("n")), "method_tag": "", "reassign": "",
           "allocator": "", "train_n": None, "env_reassign": False}
    rest = m.group("rest")

    for env in ("unsafe", "safe"):
        if rest.startswith(env + "_"):
            out["env"] = env
            rest = rest[len(env) + 1:]
            break
    else:
        out["env"] = ""

    for pl in sorted(planners, key=len, reverse=True):
        if rest == pl or rest.startswith(pl + "_"):
            out["planner"] = pl
            rest = rest[len(pl):].lstrip("_")
            break
    else:
        out["planner"] = rest.split("_")[0]
        rest = rest[len(out["planner"]):].lstrip("_")

    tokens = [t for t in rest.split("_") if t]
    if tokens and tokens[-1] == "envreassign":
        out["env_reassign"] = True
        tokens.pop()
    if tokens:
        tm = TRAIN_RE.match(tokens[-1])
        if tm:
            out["train_n"] = int(tm.group(1))
            tokens.pop()

    idx = next((i for i, t in enumerate(tokens) if t in REASSIGN_TOKENS), None)
    if idx is None:
        # 旧形式: reassign タグが無い。残りは assigner だけ
        out["allocator"] = "_".join(tokens)
    else:
        out["method_tag"] = "_".join(tokens[:idx])
        out["reassign"] = tokens[idx]
        out["allocator"] = "_".join(tokens[idx + 1:])
    return out


def load_summary(path, planners=DEFAULT_PLANNERS):
    """summary.csv を 条件 -> {metric: stats} にまとめる."""
    rows = OrderedDict()
    metrics = []
    with open(os.path.expanduser(path)) as f:
        for r in csv.DictReader(f):
            cond = r.get("condition")
            if not cond:
                continue
            if cond not in rows:
                parsed = parse_condition(cond, planners)
                if parsed is None:
                    sys.stderr.write("[warn] cannot parse condition: %s\n" % cond)
                    continue
                parsed["metrics"] = {}
                rows[cond] = parsed
            metric = r.get("metric")
            if metric and metric not in metrics:
                metrics.append(metric)

            def num(x):
                try:
                    v = float(x)
                except (TypeError, ValueError):
                    return None
                return None if v != v else v          # nan を落とす

            per = [num(v) for v in (r.get("per_seed") or "").split(";") if v != ""]
            rows[cond]["metrics"][metric] = {
                "n": int(r.get("n_seeds") or 0), "mean": num(r.get("mean")),
                "std": num(r.get("std")), "sem": num(r.get("sem")),
                "per_seed": [v for v in per if v is not None],
            }
    return list(rows.values()), metrics


def apply_filters(rows, args):
    def keep(d):
        if args.map and d["map"] not in args.map:
            return False
        if args.n and d["n"] not in args.n:
            return False
        if args.planner and d["planner"] not in args.planner:
            return False
        if args.allocator and d["allocator"] not in args.allocator:
            return False
        if args.env and d["env"] != args.env:
            return False
        if args.method_tag is not None and d["method_tag"] != args.method_tag:
            return False
        if args.min_seeds and max(
                (m["n"] for m in d["metrics"].values()), default=0) < args.min_seeds:
            return False
        return True
    return [d for d in rows if keep(d)]


def fmt(v, nd=None):
    if v is None:
        return "-"
    if nd is None:
        nd = 2 if abs(v) >= 1 else 4
    s = "%.*f" % (nd, v)
    return s.rstrip("0").rstrip(".") if "." in s else s


def sort_key(d):
    return (d["map"], d["n"], d["planner"], d["method_tag"], d["allocator"])


def render_table(rows, metrics):
    head = ["map", "N", "env", "planner", "tag", "alloc"]
    for m in metrics:
        head += [m, "sd", "n"]
    cells = [head]
    for d in sorted(rows, key=sort_key):
        line = [d["map"], str(d["n"]), d["env"], d["planner"],
                d["method_tag"] or "-", d["allocator"]]
        for m in metrics:
            st = d["metrics"].get(m)
            if st is None:
                line += ["-", "-", "-"]
            else:
                line += [fmt(st["mean"]), fmt(st["std"]), str(st["n"])]
        cells.append(line)
    widths = [max(len(r[i]) for r in cells) for i in range(len(head))]
    out = []
    for i, r in enumerate(cells):
        out.append("  ".join(c.ljust(widths[j]) for j, c in enumerate(r)))
        if i == 0:
            out.append("-" * len(out[0]))
    thin = [d for d in rows if any(m["n"] < 5 for m in d["metrics"].values())]
    if thin:
        out.append("")
        out.append("[warn] seed が 5 本に満たない条件が %d 件:" % len(thin))
        for d in sorted(thin, key=sort_key)[:10]:
            n = max((m["n"] for m in d["metrics"].values()), default=0)
            out.append("    %s  n=%d" % (d["condition"], n))
        if len(thin) > 10:
            out.append("    ... +%d" % (len(thin) - 10))
    return "\n".join(out)


HTML = r"""<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LDRP evaluation</title>
<style>
:root{--bg:#fff;--fg:#111;--mut:#666;--line:#e3e3e3;--card:#fafafa;
      --ok:#1a7f37;--warn:#9a6700;--bad:#cf222e;--ac:#0969da}
@media (prefers-color-scheme:dark){
:root{--bg:#0d1117;--fg:#e6edf3;--mut:#8b949e;--line:#30363d;--card:#161b22;
      --ok:#3fb950;--warn:#d29922;--bad:#f85149;--ac:#58a6ff}}
*{box-sizing:border-box}
body{margin:0;padding:16px;background:var(--bg);color:var(--fg);
     font:13px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace}
h1{font-size:15px;margin:0 0 2px}
.sub{color:var(--mut);margin-bottom:10px}
.bar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;
     border:1px solid var(--line);border-radius:7px;padding:8px 10px;
     background:var(--card);margin-bottom:12px}
label{color:var(--mut);margin-right:3px}
select,input{font:inherit;padding:2px 5px;border:1px solid var(--line);
             border-radius:4px;background:var(--bg);color:var(--fg)}
table{border-collapse:collapse;width:100%}
td,th{padding:3px 8px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
th{color:var(--mut);font-weight:normal;cursor:pointer;user-select:none}
th:hover{color:var(--fg)}
.num{text-align:right;font-variant-numeric:tabular-nums}
.wrap{overflow-x:auto}
.thin{color:var(--warn)}
.one{color:var(--bad)}
.mut{color:var(--mut)}
svg{max-width:100%;height:auto}
.legend span{margin-right:12px}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:4px}
</style></head><body>
<h1>LDRP evaluation</h1>
<div class="sub" id="src"></div>

<div class="bar">
  <span><label>map</label><select id="f-map"></select></span>
  <span><label>N</label><select id="f-n"></select></span>
  <span><label>env</label><select id="f-env"></select></span>
  <span><label>planner</label><select id="f-planner"></select></span>
  <span><label>tag</label><select id="f-tag"></select></span>
  <span><label>alloc</label><select id="f-alloc"></select></span>
  <span><label>metric</label><select id="f-metric"></select></span>
  <span><label><input type="checkbox" id="f-log"> log 軸</label></span>
  <span class="mut" id="count"></span>
</div>

<div id="chart"></div>
<div class="wrap"><table id="tbl"></table></div>

<script>
const DATA = __DATA__;
const METRICS = __METRICS__;
const SRC = __SRC__;
const COLORS = ["#0969da","#1a7f37","#9a6700","#cf222e","#8250df","#0f6b6b","#bc4c00"];
let sortCol = null, sortAsc = true;

const esc = s => String(s==null?"":s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const uniq = k => [...new Set(DATA.map(d=>d[k]).filter(v=>v!==""&&v!=null))].sort(
  (a,b)=> (typeof a==="number") ? a-b : String(a).localeCompare(String(b)));

function fill(id, key, label){
  const el = document.getElementById(id);
  el.innerHTML = `<option value="">${label}</option>` +
    uniq(key).map(v=>`<option value="${esc(v)}">${esc(v)}</option>`).join("");
  el.onchange = render;
}
fill("f-map","map","(all)"); fill("f-n","n","(all)"); fill("f-env","env","(all)");
fill("f-planner","planner","(all)"); fill("f-tag","method_tag","(all)");
fill("f-alloc","allocator","(all)");
document.getElementById("f-metric").innerHTML =
  METRICS.map(m=>`<option>${esc(m)}</option>`).join("");
document.getElementById("f-metric").onchange = render;
document.getElementById("f-log").onchange = render;
document.getElementById("src").textContent = SRC;

function filtered(){
  const g = id => document.getElementById(id).value;
  return DATA.filter(d =>
    (!g("f-map") || d.map===g("f-map")) &&
    (!g("f-n") || String(d.n)===g("f-n")) &&
    (!g("f-env") || d.env===g("f-env")) &&
    (!g("f-planner") || d.planner===g("f-planner")) &&
    (!g("f-tag") || d.method_tag===g("f-tag")) &&
    (!g("f-alloc") || d.allocator===g("f-alloc")));
}

function statOf(d, m){ return (d.metrics||{})[m]; }
const f = (v,nd) => v==null ? "-" :
  (nd==null ? (Math.abs(v)>=1?v.toFixed(2):v.toFixed(4)) : v.toFixed(nd))
    .replace(/\.?0+$/,"");

function render(){
  const metric = document.getElementById("f-metric").value;
  let rows = filtered();
  if(sortCol){
    rows = rows.slice().sort((a,b)=>{
      const va = sortCol==="metric" ? (statOf(a,metric)||{}).mean : a[sortCol];
      const vb = sortCol==="metric" ? (statOf(b,metric)||{}).mean : b[sortCol];
      const r = (va==null?-Infinity:va) < (vb==null?-Infinity:vb) ? -1 :
                (va===vb ? 0 : 1);
      return sortAsc ? r : -r;
    });
  }
  document.getElementById("count").textContent = rows.length + " 条件";

  const cols = [["map","map"],["n","N"],["env","env"],["planner","planner"],
                ["method_tag","tag"],["allocator","alloc"],["metric",metric]];
  document.getElementById("tbl").innerHTML =
    "<tr>" + cols.map(([k,l])=>`<th onclick="sortBy('${k}')">${esc(l)}${sortCol===k?(sortAsc?" ▲":" ▼"):""}</th>`).join("")
    + "<th>n</th><th>per-seed</th></tr>"
    + rows.map(d=>{
        const st = statOf(d, metric) || {};
        const cls = st.n===1 ? "one" : (st.n<5 ? "thin" : "");
        const ps = (st.per_seed||[]).map(v=>f(v)).join("  ");
        return `<tr>
          <td>${esc(d.map)}</td><td class="num">${d.n}</td><td>${esc(d.env)}</td>
          <td>${esc(d.planner)}</td><td>${esc(d.method_tag||"-")}</td>
          <td>${esc(d.allocator)}</td>
          <td class="num">${f(st.mean)} ± ${f(st.std)}</td>
          <td class="num ${cls}">${st.n==null?"-":st.n}</td>
          <td class="mut">${esc(ps)}</td></tr>`;
      }).join("");
  drawChart(rows, metric);
}
function sortBy(k){ if(sortCol===k) sortAsc=!sortAsc; else {sortCol=k; sortAsc=true;} render(); }

function drawChart(rows, metric){
  const host = document.getElementById("chart");
  const pts = [];
  rows.forEach(d=>{ const st=statOf(d,metric); if(st&&st.mean!=null)
    pts.push({x:d.n, mean:st.mean, per:st.per_seed||[],
              key:`${d.planner}${d.method_tag?"_"+d.method_tag:""}/${d.allocator}`}); });
  if(!pts.length){ host.innerHTML=""; return; }

  const log = document.getElementById("f-log").checked;
  const W=760,H=300,L=64,R=14,T=14,B=34;
  const xs=[...new Set(pts.map(p=>p.x))].sort((a,b)=>a-b);
  let vals=[]; pts.forEach(p=>{ vals.push(p.mean); p.per.forEach(v=>vals.push(v)); });
  if(log) vals = vals.filter(v=>v>0);
  let lo=Math.min(...vals), hi=Math.max(...vals);
  if(lo===hi){ lo=lo-1; hi=hi+1; }
  const sx = x => L + (xs.length<2 ? (W-L-R)/2
                   : (xs.indexOf(x)/(xs.length-1))*(W-L-R));
  const sy = v => { if(log){ const a=Math.log10(Math.max(lo,1e-9)), b=Math.log10(hi);
                             return H-B-((Math.log10(Math.max(v,1e-9))-a)/(b-a))*(H-T-B); }
                    return H-B-((v-lo)/(hi-lo))*(H-T-B); };

  const keys=[...new Set(pts.map(p=>p.key))].sort();
  let g="";
  // 軸
  g += `<line x1="${L}" y1="${T}" x2="${L}" y2="${H-B}" stroke="var(--line)"/>`;
  g += `<line x1="${L}" y1="${H-B}" x2="${W-R}" y2="${H-B}" stroke="var(--line)"/>`;
  for(let i=0;i<=4;i++){
    const v = log ? Math.pow(10, Math.log10(Math.max(lo,1e-9)) + i/4*(Math.log10(hi)-Math.log10(Math.max(lo,1e-9))))
                  : lo + i/4*(hi-lo);
    const y = sy(v);
    g += `<line x1="${L}" y1="${y}" x2="${W-R}" y2="${y}" stroke="var(--line)" stroke-dasharray="2 3"/>`;
    g += `<text x="${L-6}" y="${y+4}" text-anchor="end" fill="var(--mut)" font-size="10">${f(v)}</text>`;
  }
  xs.forEach(x=>{ g += `<text x="${sx(x)}" y="${H-B+16}" text-anchor="middle" fill="var(--mut)" font-size="10">${x}agent</text>`; });

  keys.forEach((k,i)=>{
    const c=COLORS[i%COLORS.length];
    const line=pts.filter(p=>p.key===k).sort((a,b)=>a.x-b.x);
    if(line.length>1)
      g += `<polyline fill="none" stroke="${c}" stroke-width="1.6" points="${
        line.map(p=>`${sx(p.x)},${sy(p.mean)}`).join(" ")}"/>`;
    line.forEach(p=>{
      p.per.forEach(v=>{ if(!log||v>0)
        g += `<circle cx="${sx(p.x)+(i-keys.length/2)*3}" cy="${sy(v)}" r="2" fill="${c}" opacity="0.35"/>`; });
      g += `<circle cx="${sx(p.x)}" cy="${sy(p.mean)}" r="3.5" fill="${c}"/>`;
    });
  });
  host.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="100%">${g}</svg>`
    + `<div class="legend mut">` + keys.map((k,i)=>
        `<span><i class="dot" style="background:${COLORS[i%COLORS.length]}"></i>${esc(k)}</span>`).join("")
    + `　<span class="mut">小さい点 = seed ごとの値</span></div>`;
}
render();
</script></body></html>
"""


def render_html(rows, metrics, src):
    keys = ("condition", "map", "n", "env", "planner", "method_tag",
            "reassign", "allocator", "train_n", "env_reassign", "metrics")
    data = [dict((k, d.get(k)) for k in keys) for d in sorted(rows, key=sort_key)]
    return (HTML.replace("__DATA__", json.dumps(data, ensure_ascii=False))
                .replace("__METRICS__", json.dumps(metrics, ensure_ascii=False))
                .replace("__SRC__", json.dumps(src, ensure_ascii=False)))


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Show the aggregated evaluation results (results/summary.csv).")
    p.add_argument("-s", "--summary", default="results/summary.csv",
                   help="aggregate.py --csv output (default: results/summary.csv)")
    p.add_argument("-f", "--format", default="table",
                   choices=["table", "html", "json"])
    p.add_argument("-o", "--out", default=None, help="write to this file")
    p.add_argument("--metrics", default=None,
                   help="comma separated metric names (default: %s)"
                        % ",".join(DEFAULT_METRICS))
    p.add_argument("--all-metrics", action="store_true",
                   help="use every metric found in the file")
    p.add_argument("--map", action="append", default=None)
    p.add_argument("--n", action="append", type=int, default=None)
    p.add_argument("--planner", action="append", default=None)
    p.add_argument("--allocator", action="append", default=None)
    p.add_argument("--env", default=None, choices=["safe", "unsafe"])
    p.add_argument("--method-tag", default=None,
                   help="filter by method tag (pass '' for no tag)")
    p.add_argument("--min-seeds", type=int, default=None,
                   help="only conditions with at least this many seeds")
    p.add_argument("--planner-names", default=None,
                   help="comma separated planner names used to split the condition key")
    args = p.parse_args(argv)

    path = os.path.expanduser(args.summary)
    if not os.path.exists(path):
        p.error("summary not found: %s (run aggregate.py --csv first, "
                "or pass --summary)" % path)

    planners = tuple(x.strip() for x in args.planner_names.split(",")) \
        if args.planner_names else DEFAULT_PLANNERS
    rows, found = load_summary(path, planners)
    if not rows:
        p.error("no condition parsed from %s" % path)

    if args.all_metrics:
        metrics = found
    elif args.metrics:
        metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    else:
        metrics = [m for m in DEFAULT_METRICS if m in found] or found[:2]
    missing = [m for m in metrics if m not in found]
    if missing:
        sys.stderr.write("[warn] not in %s: %s\n" % (path, ", ".join(missing)))
        metrics = [m for m in metrics if m in found]

    rows = apply_filters(rows, args)
    sys.stderr.write("[eval] %d condition(s), metrics: %s\n"
                     % (len(rows), ", ".join(metrics)))

    if args.format == "json":
        text = json.dumps({"source": path, "metrics": metrics,
                           "conditions": sorted(rows, key=sort_key)},
                          ensure_ascii=False, indent=1)
    elif args.format == "html":
        text = render_html(rows, metrics, "%s  (%d 条件)" % (path, len(rows)))
    else:
        text = render_table(rows, metrics)

    if args.out:
        with open(os.path.expanduser(args.out), "w") as f:
            f.write(text + "\n")
        sys.stderr.write("[eval] wrote %s\n" % args.out)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
