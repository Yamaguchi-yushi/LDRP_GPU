#!/usr/bin/env python3
"""LDRP の実験 (方策学習・方策評価) を、依頼に応じて複数のマシンで自動実行する.

この Mac (白) が司令塔。各マシンには ssh で投入し、投入したら手を離す。LLM は使わない。
設計: design/auto_runner.md

  依頼を入れる
    python tools/autorun.py find aoba00 7agent "8x5_5 10M aoba00_5 10M" TP     # 計画表の行を探す (id が出る)
    python tools/autorun.py add train <id> -n 2 --then-eval                    # その条件を 2 本学習 (終わったら評価)
    python tools/autorun.py add eval <id> [<id> ...]                           # その条件の done の run を評価
  様子を見る / 止める
    python tools/autorun.py status            # 依頼ごとの進み具合
    python tools/autorun.py hosts             # マシンごとの空き
    python tools/autorun.py cancel q3         # 依頼を取り消す (動いている run は止めない)
    python tools/autorun.py pause | resume    # 新しい投入を止める / 再開
  実行 (launchd が 15 分おきに呼ぶ)
    python tools/autorun.py tick [--dry-run]  # 依頼を見て、空いているマシンに投入する

状態は tools/.autorun/ に置く (queue.json = 依頼、ledger.jsonl = 投入の記録、PAUSE = 一時停止)。
マシンごとの設定は tools/collect_config.yaml の各 host の `autorun:` (enabled: true のマシンにだけ投入する)。
"""

import argparse
import datetime
import hashlib
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
import urllib.request

TOOLS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(TOOLS)
sys.path.insert(0, TOOLS)
import collect_runs as CR      # noqa: E402
import plan as PLAN            # noqa: E402
import eval_lare_factor as EV  # noqa: E402

STATE = os.path.join(TOOLS, ".autorun")
QUEUE = os.path.join(STATE, "queue.json")
LEDGER = os.path.join(STATE, "ledger.jsonl")
PAUSE = os.path.join(STATE, "PAUSE")
CONFIG = os.path.join(TOOLS, "collect_config.yaml")
DASHBOARD = "http://127.0.0.1:8765"
REMOTE_DIR = ".ldrp/autorun"            # 各マシンのホーム直下 (ジョブの JSON・ログ・実行役)

# 評価のコードとして写すときに除くもの (学習結果・ログ・モデルは送らない)
SYNC_EXCLUDES = [".git/", ".venv/", "__pycache__/", "/logs/", "/results/", "/train_results/",
                 "/src/epymarl/results/", "/src/epymarl/tmp_results/", "/src/all_policy/models/",
                 "/src/task_assign/models/", "/src/lare/path/checkpoints/",
                 "/src/lare/task/checkpoints/", "/tools/.*", "/tools/dist/"]
# 評価の結果を左右するファイル。写したあと手元とハッシュが一致することを確かめる
EVAL_CODE = ["test.py", "runner.py", "src/main/drp_env/drp_env.py",
             "src/main/drp_env/wrapper/safe_marl.py", "src/config/default.yaml",
             "src/all_policy/policy.py", "src/all_policy/policy_runner.py",
             "src/task_assign/task_policy/tp.py", "src/task_assign/task_policy/ppo.py"]
EVAL_SEC_PER_COND = 720.0       # 1 条件 (1000 ep x 1000 step) の時間。2026-11 の実測 (Mac・GPU1 とも 11〜12 分)
DEFAULT_RATE = 500.0            # 学習の速さの実績が無いときの仮の値 (steps/秒)
LAUNCH_GRACE_SEC = 45 * 60      # 投入してから run が見えるまでの猶予 (この間は「動いている」に数える)


# ---------------------------------------------------------------------------
# 小物
# ---------------------------------------------------------------------------

def now_utc():
    return datetime.datetime.now(datetime.timezone.utc)


def parse_dt(s):
    if not s:
        return None
    try:
        d = datetime.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=datetime.timezone.utc)


def fmt_dt(d):
    return d.astimezone().strftime("%m-%d %H:%M") if d else "-"


def fmt_dur(sec):
    sec = int(sec or 0)
    return "%dd%02dh" % (sec // 86400, sec % 86400 // 3600) if sec >= 86400 else "%dh%02dm" % (sec // 3600, sec % 3600 // 60)


def load_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (IOError, OSError, ValueError):
        return default


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(path + ".tmp", path)


def ledger():
    out = []
    try:
        with open(LEDGER) as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    except (IOError, OSError):
        pass
    return out


def ledger_add(rec):
    os.makedirs(STATE, exist_ok=True)
    rec = dict(rec, at=now_utc().isoformat())
    with open(LEDGER, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def dashboard():
    with urllib.request.urlopen(DASHBOARD + "/api/train", timeout=120) as f:
        return json.load(f)


def load_conf():
    return CR.load_config(CONFIG) if os.path.exists(CONFIG) else {}


# ---------------------------------------------------------------------------
# マシン
# ---------------------------------------------------------------------------

def hosts_of(conf):
    """autorun: のある host を {label, ssh, ..., python, train_repo, eval_repo, max_*} にして返す."""
    out = []
    for h in conf.get("hosts") or []:
        a = h.get("autorun")
        if not isinstance(a, dict):
            continue
        d = {k: h.get(k) for k in ("label", "ssh", "port", "connect_timeout", "ssh_options")}
        d.update(a)
        d["local"] = h.get("ssh") in (None, "", "local")
        out.append(d)
    return out


def ssh_base(h):
    cmd = ["ssh", "-o", "BatchMode=yes"]
    for opt in h.get("ssh_options") or []:
        cmd += ["-o", opt]
    cmd += ["-o", "ConnectTimeout=%d" % int(h.get("connect_timeout") or 10)]
    if h.get("port"):
        cmd += ["-p", str(h["port"])]
    return cmd


def run_on(h, script, stdin=None, timeout=120):
    """host でシェルのコマンドを実行して (rc, stdout, stderr) を返す."""
    if h["local"]:
        cmd = ["bash", "-c", script]
    else:
        cmd = ssh_base(h) + [h["ssh"], script]
    try:
        p = subprocess.run(cmd, input=stdin, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except (subprocess.TimeoutExpired, OSError) as e:
        return 255, "", str(e)


def rsync_to(h, srcs, dest, extra=None, timeout=600):
    """手元のファイル (群) を host の dest に送る. local なら cp と同じ."""
    target = dest if h["local"] else "%s:%s" % (h["ssh"], dest)
    cmd = ["rsync", "-az"] + (extra or [])
    if not h["local"]:
        cmd += ["-e", " ".join(ssh_base(h))]
    cmd += list(srcs) + [target]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError("rsync to %s failed: %s" % (h["label"], p.stderr.strip()[-300:]))


def match_launches(led, dash, by_id):
    """投入した学習 (ledger) -> その run (ダッシュボードの行) の対応.

    seed は epymarl (sacred) が決めるので、投入した側は知らない。そこで
    「同じマシン・同じ条件で、投入した時刻より後に始まった run」を、投入の古い順に 1 本ずつ割り当てる。
    返り値: ledger の job id -> run (まだ見えていなければ入らない)
    """
    out, used = {}, set()
    trains = sorted((x for x in led if x.get("kind") == "train"), key=lambda x: x.get("at") or "")
    for x in trains:
        cond = by_id.get(x.get("cond"))
        t0 = parse_dt(x.get("at"))
        if cond is None or t0 is None:
            continue
        cands = [r for r in dash.get("runs") or []
                 if r.get("machine") == x.get("host") and r["uid"] not in used
                 and (parse_dt(r.get("start_at")) or t0 - datetime.timedelta(days=1))
                 >= t0 - datetime.timedelta(minutes=2)
                 and PLAN.matches(cond, r)]
        if cands:
            r = min(cands, key=lambda r: parse_dt(r.get("start_at")))
            used.add(r["uid"])
            out[x["job"]] = r
    return out


NVSMI = "$(command -v nvidia-smi || echo /usr/lib/wsl/lib/nvidia-smi)"
PROBE = ("echo '#eval'; pgrep -f '[t]est.py map_' | wc -l; "
         "echo '#cpu'; getconf _NPROCESSORS_ONLN; "
         "echo '#load'; (cut -d' ' -f1 /proc/loadavg 2>/dev/null || sysctl -n vm.loadavg | awk '{print $2}'); "
         "echo '#mem'; awk '/MemAvailable|MemTotal/ {print int($2/1024)}' /proc/meminfo 2>/dev/null; "
         "echo '#gpu'; " + NVSMI + " --query-gpu=memory.used,memory.total,utilization.gpu "
         "--format=csv,noheader,nounits 2>/dev/null; "
         "echo '#apps'; " + NVSMI + " --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null; "
         "echo '#ps'; ps -eo pid=,ppid=,pcpu=,rss=,args= 2>/dev/null; true")


def _sections(out):
    sec, cur = {}, None
    for line in out.splitlines():
        if line.startswith("#") and line[1:].strip().isalpha():
            cur = line[1:].strip()
            sec[cur] = []
        elif cur is not None and line.strip():
            sec[cur].append(line.strip())
    return sec


def host_state(h, dash, led, matched=None):
    """host の今の状態: 学習の本数と空く時刻・評価の test.py 数・CPU / メモリ / GPU の空き.

    CPU の空き = コア数 x cpu_target (既定 0.8) - load average (他の人の分も含む)
    メモリの空き = MemAvailable - 余裕 (1 割か 2GB の大きいほう)。Mac は測らない (None)
    ついでに、autorun が投入して動いている学習の「1 本が使う量」を測る (st["measured"])
    """
    label = h["label"]
    st = {"label": label, "reachable": label not in (dash.get("unreachable") or {}),
          "train_eta": [], "pending_train": 0, "n_eval": 0, "cpu_free": None, "mem_free_mb": None,
          "gpu_free_mb": None, "gpu_util": None, "measured": {}}
    for r in dash.get("runs") or []:
        if r.get("machine") == label and r.get("state") == "running":
            st["train_eta"].append(parse_dt(r.get("eta")) or now_utc() + datetime.timedelta(days=1))
    # 投入したがまだ run が見えていない学習 (猶予の間は動いているものとして数える)
    mine_running = []
    for x in led:
        if x.get("kind") != "train" or x.get("host") != label:
            continue
        run = (matched or {}).get(x.get("job"))
        if run is None:
            age = (now_utc() - parse_dt(x.get("at"))).total_seconds()
            if age < LAUNCH_GRACE_SEC:
                st["pending_train"] += 1
                st["train_eta"].append(now_utc() + datetime.timedelta(seconds=float(x.get("est_sec") or 0)))
        elif run.get("state") == "running" and (run.get("progress") or 0) > 0.01:
            mine_running.append(x)
    st["n_train"] = len(st["train_eta"])
    rc, out, _ = run_on(h, PROBE, timeout=60)
    if rc == 255:
        st["reachable"] = False
        return st
    sec = _sections(out)
    num = lambda xs, k=0: float(xs[k]) if len(xs) > k else None
    try:
        st["n_eval"] = int(num(sec.get("eval", []) or ["0"]))
        cpu, load = num(sec.get("cpu", [])), num(sec.get("load", []))
        if cpu and load is not None:
            st["cpu_count"] = cpu
            st["cpu_free"] = cpu * float(h.get("cpu_target") or 0.8) - load
        mem = [float(x) for x in sec.get("mem", [])]
        if len(mem) == 2:                         # MemTotal, MemAvailable の順
            total, avail = mem
            st["mem_free_mb"] = avail - max(2000.0, 0.1 * total)
    except ValueError:
        pass
    free, util = None, None
    for l in sec.get("gpu", []):
        try:
            used, total, u = [float(x) for x in l.split(",")]
            free = (free or 0) + total - used
            util = max(util or 0, u)
        except ValueError:
            pass
    st["gpu_free_mb"], st["gpu_util"] = free, util
    # autorun の学習 1 本が使う量 (実行役 autorun_train.py から下のプロセス全部の合計)
    if mine_running:
        procs = {}
        for l in sec.get("ps", []):
            parts = l.split(None, 4)
            if len(parts) == 5:
                try:
                    procs[int(parts[0])] = (int(parts[1]), float(parts[2]), float(parts[3]), parts[4])
                except ValueError:
                    pass
        gmem = {}
        for l in sec.get("apps", []):
            try:
                pid, m = [x.strip() for x in l.split(",")]
                gmem[int(pid)] = float(m)
            except ValueError:
                pass
        kids = {}
        for pid, (pp, _c, _r, _a) in procs.items():
            kids.setdefault(pp, []).append(pid)
        for x in mine_running:
            root = [pid for pid, v in procs.items() if "jobs/%s.json" % x["job"] in v[3]]
            if not root:
                continue
            tree, stack = set(), list(root)
            while stack:
                p = stack.pop()
                if p not in tree:
                    tree.add(p)
                    stack += kids.get(p, [])
            st["measured"][x["job"]] = {"cpu": sum(procs[p][1] for p in tree) / 100.0,
                                        "mem": sum(procs[p][2] for p in tree) / 1024.0,
                                        "gpu": sum(gmem.get(p, 0.0) for p in tree)}
    return st


# ---------------------------------------------------------------------------
# 1 本が使う量 (footprint)
# ---------------------------------------------------------------------------
FOOTPRINT = os.path.join(STATE, "footprint.json")
EVAL_CPU = 1.0                  # 評価 1 本 = test.py 1 本 = 1 コア (メモリは環境 1 つ ~0.5GB なので見ない)
# マップごとの (1 機の観測の次元, ノード数 = 行動の数)。onehot_fov で実測 (2026-11-10)
MAP_DIMS = {"map_5x4": (40, 20), "map_8x5": (80, 40), "map_aoba00": (86, 43), "map_aoba01": (36, 18)}
LEARNER_MB = 2500.0             # 学習本体 (torch・CUDA の初期化・モデル) のメモリ
WORKER_MB = 150.0               # parallel runner の環境プロセス 1 つ (環境 1 つは ~100MB を実測)
GPU_BASE_MB = 1500.0            # CUDA のコンテキスト + モデル


def _train_shape(rec):
    cfg, env = rec.get("cfg") or {}, rec.get("env") or {}
    m = re.search(r"(\d+)agent_(map_\w+?)-v", env.get("key", ""))
    n, mp = (int(m.group(1)), m.group(2)) if m else (10, "?")
    return cfg, env, n, mp


def fp_key(rec):
    cfg, _env, n, mp = _train_shape(rec)
    return "train|%s|%s|%s" % (cfg.get("name"), mp, n)


def estimate_train(rec):
    """学習 1 本が使う量の見積もり {cpu, mem, gpu} (設定から計算する).

    メモリ = 学習本体 + リプレイバッファ + 環境プロセス
      リプレイバッファ = buffer_size エピソード x (time_limit+1) ステップ x 1 ステップの大きさ
        1 ステップ = 4 バイト x (観測 N*obs + 状態 N*obs + 選べる行動 N*nodes + 行動 one-hot N*nodes) + 8 バイト x N
        QMIX (buffer 5000) は台数に比例して大きい: 8x5 で 5 台 ~12GB / 10 台 ~24GB、aoba00 7 台 ~18GB
        MAPPO / MAT (buffer 32) は小さい (~0.1GB)
      環境プロセス = parallel runner なら batch_size_run 個 (MAPPO / MAT は 32 個)
    buffer_cpu_only: False ならバッファは GPU に置かれる
    """
    cfg, env, n, mp = _train_shape(rec)
    obs, nodes = MAP_DIMS.get(mp, (90, 45))           # 知らないマップは大きめに見る
    step = 4.0 * (2 * n * obs + 2 * n * nodes) + 8.0 * n
    t = int(env.get("time_limit") or cfg.get("episode_limit") or 500) + 1
    buf_mb = float(cfg.get("buffer_size") or 32) * t * step / 2 ** 20
    parallel = cfg.get("runner") == "parallel"
    workers = int(cfg.get("batch_size_run") or 1) if parallel else 1
    on_gpu = cfg.get("buffer_cpu_only") is False and cfg.get("use_cuda", True)
    mem = LEARNER_MB + (0.0 if on_gpu else buf_mb) + workers * WORKER_MB
    cpu = 1.5 + (min(workers, 32) * 0.25 if parallel else 0.0)   # 環境プロセスは学習の間は待つので 1/4 コアずつ
    gpu = GPU_BASE_MB + (buf_mb if on_gpu else 0.0)
    return {"cpu": cpu, "mem": mem, "gpu": gpu}


def footprint(rec, db):
    """学習 1 本が使う量 {cpu, mem, gpu}.

    同じ (アルゴリズム, マップ, 台数) の実測があればそれ。同じアルゴリズム・マップで台数だけ違う実測があれば、
    見積もりの比 (台数で変わる分) で換算する。どちらも無ければ見積もり。実測を使うときは 1.2 倍の余裕を見る。
    """
    est = estimate_train(rec)
    got = db.get(fp_key(rec))
    if got and got.get("n"):
        return {"cpu": got["cpu"] * 1.2, "mem": got["mem"] * 1.2, "gpu": got["gpu"] * 1.2, "measured": "same"}
    cfg, _env, n, mp = _train_shape(rec)
    near = [(k, v) for k, v in db.items()
            if k.startswith("train|%s|%s|" % (cfg.get("name"), mp)) and v.get("n") and v.get("est")]
    if near:
        k, v = min(near, key=lambda kv: abs(int(kv[0].rsplit("|", 1)[1]) - n))
        r = lambda f: (v[f] / max(v["est"][f], 1e-6)) * est[f] * 1.2
        return {"cpu": r("cpu"), "mem": r("mem"), "gpu": r("gpu"), "measured": k}
    return dict(est, measured=None)


def update_footprints(states, led, recs_by_job, dry_run):
    """測った量を (アルゴリズム, マップ, 台数) ごとに平均して残す (指数移動平均)."""
    db = load_json(FOOTPRINT, {})
    for st in states.values():
        for job, v in st.get("measured", {}).items():
            rec = recs_by_job.get(job)
            if rec is None:
                continue
            k = fp_key(rec)
            est = estimate_train(rec)
            old = db.get(k)
            if old:
                a = 0.3
                db[k] = {"cpu": (1 - a) * old["cpu"] + a * v["cpu"], "mem": (1 - a) * old["mem"] + a * v["mem"],
                         "gpu": (1 - a) * old["gpu"] + a * v["gpu"], "n": old["n"] + 1, "est": est}
            else:
                db[k] = dict(v, n=1, est=est)
    if not dry_run:
        save_json(FOOTPRINT, db)
    return db


# ---------------------------------------------------------------------------
# 計画の条件
# ---------------------------------------------------------------------------

def cond_key(c):
    return "|".join(str(x) for x in (c.get("plan"), c.get("map"), c.get("agents"), "%g" % float(c.get("t_max_m") or 0),
                                      c.get("setting"), c.get("algo"), c.get("task_arrival"),
                                      c.get("task_assign") or "", c.get("reassign"), c.get("dynamic")))


def cond_id(c):
    return hashlib.sha1(cond_key(c).encode("utf-8")).hexdigest()[:6]


def cond_text(c):
    flag = lambda v: "-" if v is None else ("T" if v else "F")
    return "%s | %s %dagent %gM | %s | %s | %s | %s | re:%s dyn:%s" % (
        c.get("plan"), (c.get("map") or "*").replace("map_", ""), c.get("agents"), c.get("t_max_m"),
        c.get("setting"), c.get("algo"), c.get("task_arrival"), c.get("task_assign") or "TP",
        flag(c.get("reassign")), flag(c.get("dynamic")))


def load_conds(dash):
    paths = [p if os.path.isabs(p) else os.path.join(REPO, p) for p in dash.get("plan_files") or []]
    conds = PLAN.parse_plans(paths)
    for c in conds:
        c["id"] = cond_id(c)
    return conds


def runs_of(cond, dash):
    return [r for r in dash.get("runs") or [] if PLAN.matches(cond, r) and not r.get("excluded")]


def find_conds(conds, words):
    """語をすべて含む条件. 「7台」= 7agent、→ / -> は空白と同じ扱い."""
    norm = lambda s: re.sub(r"\s+", " ", re.sub(r"→|->", " ", s)).strip().lower()
    toks = []
    for w in words:
        w = re.sub(r"(\d+)\s*台", r"\1agent", norm(w))
        toks.append(w)
    out = []
    for c in conds:
        hay = norm(cond_text(c) + " " + c["id"] + " " + (c.get("map") or ""))
        if all(t in hay for t in toks):
            out.append(c)
    return out


# ---------------------------------------------------------------------------
# 依頼
# ---------------------------------------------------------------------------

def load_queue():
    return load_json(QUEUE, [])


def new_req_id(q):
    n = max([int(r["id"][1:]) for r in q if re.match(r"^q\d+$", r.get("id", ""))] or [0])
    return "q%d" % (n + 1)


# ---------------------------------------------------------------------------
# 学習ジョブ (雛形 run の config を写す。seed は書かず、train.py と同じく epymarl に決めさせる)
# ---------------------------------------------------------------------------

def run_records():
    """collect のキャッシュ (run ごとの config.json の中身) を uid で引けるようにする."""
    out = {}
    for r in CR.read_cache(os.path.join(TOOLS, ".run_cache.jsonl")):
        if r.get("uid"):
            out[r["uid"]] = r
    return out


def pick_template(cond, dash, recs):
    """同じ条件の実績 run (done を優先、なければ running) の記録を返す."""
    cands = [r for r in runs_of(cond, dash) if r.get("state") in ("done", "running") and r["uid"] in recs]
    if not cands:
        return None
    cands.sort(key=lambda r: (r.get("state") != "done", -(parse_dt(r.get("start_at")) or now_utc()).timestamp()))
    return recs[cands[0]["uid"]]


_YAML_CACHE = {}


def remote_defaults(h, algo):
    """host の学習用リポジトリにある epymarl の既定値 (default + envs/gymma + algs/<algo>)."""
    import yaml
    key = (h["label"], algo)
    if key in _YAML_CACHE:
        return _YAML_CACHE[key]
    base = "%s/src/epymarl/src/config" % h["train_repo"]
    sep = "\n---8<---\n"
    script = "cat %s/default.yaml; printf %s; cat %s/envs/gymma.yaml; printf %s; cat %s/algs/%s.yaml" % (
        base, shlex.quote(sep), base, shlex.quote(sep), base, algo)
    rc, out, err = run_on(h, script, timeout=60)
    parts = out.split(sep.strip("\n"))
    if rc != 0 or len(parts) != 3:
        raise RuntimeError("cannot read epymarl config on %s: %s" % (h["label"], err.strip()[-200:]))
    d = {}
    for p in parts:
        d.update(yaml.safe_load(p) or {})
    _YAML_CACHE[key] = d
    return d


def _fmt(v):
    if isinstance(v, bool) or v is None:
        return str(v)
    if isinstance(v, (int, float)):
        return repr(v)
    s = str(v)
    if re.search(r"[\s\"'$`\\]", s):
        raise ValueError("value with spaces or quotes cannot be passed safely: %r" % s)
    return '"%s"' % s


def build_train_command(h, rec):
    """雛形の config から、host の既定値と違う項目だけを並べた epymarl のコマンドを作る.

    既定値 + 並べた項目 = 雛形の config になることを確かめてから返す (違えば例外)。
    """
    cfg, env = rec.get("cfg") or {}, rec.get("env") or {}
    algo = cfg.get("name")
    defaults = remote_defaults(h, algo)
    denv = defaults.get("env_args") or {}
    skip = {"seed", "env_args", "name", "env"}
    # t_max はダッシュボードの条件判定に要るので、既定値と同じでも必ず書く
    over = [(k, v) for k, v in sorted(cfg.items())
            if k not in skip and not isinstance(v, (dict, list))
            and (k == "t_max" or k not in defaults or defaults[k] != v)]
    # env_args は train.py と同じく全部書く (ダッシュボードは書かれた値で条件を判定するため)。
    # ただし host の gymma.yaml に無い項目は、雛形が同じマシンの run のときだけ通す
    # (別のマシンのコードにしか無い引数だと、環境の生成で落ちる)
    eover = [(k, v) for k, v in sorted(env.items()) if not isinstance(v, (dict, list))]
    unknown = [k for k, _v in eover if k not in denv]
    if unknown and rec.get("machine") != h["label"]:
        raise RuntimeError("env_args %s are not in %s's gymma.yaml (template is from %s)"
                           % (", ".join(unknown), h["label"], rec.get("machine")))
    # 検算: 既定値に上書きを重ねると雛形と同じになるか
    merged = dict(defaults)
    merged.update(dict(over))
    bad = [k for k, v in cfg.items() if k not in skip and not isinstance(v, (dict, list)) and merged.get(k) != v]
    menv = dict(denv)
    menv.update(dict(eover))
    bad += ["env_args." + k for k, v in env.items() if not isinstance(v, (dict, list)) and menv.get(k) != v]
    if bad:
        raise RuntimeError("template does not round-trip on %s: %s" % (h["label"], ", ".join(bad[:5])))
    toks = ["%s=%s" % (k, _fmt(v)) for k, v in over] + ["env_args.%s=%s" % (k, _fmt(v)) for k, v in eover]
    cmd = "%s src/epymarl/src/main.py --config=%s --env-config=gymma with %s" % (
        h["python"], algo, " ".join(toks))
    return cmd


def check_command_condition(cmd, cond, conf):
    """作ったコマンドをダッシュボードと同じ derive() に通し、依頼の条件に判定されるか."""
    rec = CR.parse_train_command(cmd)
    rec.update({"uid": "autorun:check", "machine": "autorun"})
    d = CR.derive(rec, conf.get("stale_minutes") or 90, conf.get("method_tag_by_lare_mode"),
                  conf.get("expected_t_max"), lare_chain=conf.get("lare_chain"))
    return PLAN.matches(cond, d)


def ensure_lare_model(h, env):
    """事前学習した LaRe のモデルを使う学習なら、そのファイルを host に置く (手元の models/ から送る)."""
    names = []
    if env.get("use_lare_path") and env.get("use_pretrained_lare_path"):
        names.append(env.get("pretrained_lare_path_model_name"))
    if env.get("use_lare_path") and env.get("use_finetuning_lare_path"):
        names.append(env.get("finetuning_lare_path_model_name"))
    for n in [x for x in names if x]:
        script = "ls %s/src/lare/path/models/%s %s/src/lare/path/checkpoints/%s 2>/dev/null | head -1" % (
            h["train_repo"], shlex.quote(n), h["train_repo"], shlex.quote(n))
        rc, out, _ = run_on(h, script)
        if out.strip():
            continue
        local = os.path.join(REPO, "src", "lare", "path", "models", n)
        if not os.path.exists(local):
            raise RuntimeError("LaRe model %s is not on %s nor in this Mac's models/" % (n, h["label"]))
        rsync_to(h, [local], "%s/src/lare/path/models/" % h["train_repo"])


def launch_train(h, req, cond, rec, conf, dry_run):
    cmd = build_train_command(h, rec)
    if not check_command_condition(cmd, cond, conf):
        raise RuntimeError("generated command is not judged as the requested condition")
    job_id = "%s_%s_%s" % (req["id"], h["label"], time.strftime("%m%d%H%M%S"))
    if dry_run:
        return {"cmd": cmd, "job": job_id}
    ensure_lare_model(h, rec.get("env") or {})
    run_on(h, "mkdir -p ~/%s/jobs ~/%s/logs" % (REMOTE_DIR, REMOTE_DIR))
    rsync_to(h, [os.path.join(TOOLS, "autorun_train.py")], "~/%s/" % REMOTE_DIR if not h["local"]
             else os.path.expanduser("~/%s/" % REMOTE_DIR))
    job = {"id": job_id, "commands": [cmd], "max_processes": 1}
    rc, _, err = run_on(h, "cat > ~/%s/jobs/%s.json" % (REMOTE_DIR, job_id), stdin=json.dumps(job))
    if rc != 0:
        raise RuntimeError("cannot write job on %s: %s" % (h["label"], err.strip()))
    script = ("cd %s && nohup %s ~/%s/autorun_train.py ~/%s/jobs/%s.json > ~/%s/logs/%s.out 2>&1 < /dev/null &"
              % (shlex.quote(h["train_repo"]), h["python"], REMOTE_DIR, REMOTE_DIR, job_id, REMOTE_DIR, job_id))
    rc, _, err = run_on(h, script)
    if rc != 0:
        raise RuntimeError("launch failed on %s: %s" % (h["label"], err.strip()))
    rc, head, _ = run_on(h, "cd %s && git rev-parse --short HEAD 2>/dev/null; git status --porcelain 2>/dev/null | wc -l"
                         % shlex.quote(h["train_repo"]))
    return {"cmd": cmd, "job": job_id, "code": " ".join(head.split())}


# ---------------------------------------------------------------------------
# 評価ジョブ
# ---------------------------------------------------------------------------

def eval_targets(req, conds, dash):
    """評価の依頼が対象にする done の run: uid -> {map, n, assign}."""
    out = {}
    uids = set(req.get("uids") or [])
    for c in conds:
        if c["id"] not in (req.get("conds") or []) and not uids:
            continue
        for r in runs_of(c, dash):
            if r.get("state") != "done" or (uids and r["uid"] not in uids):
                continue
            out[r["uid"]] = {"map": r["map"], "n": int(r["agents"]),
                             "assign": (r.get("task_assign") or "TP").lower()}
    return out


def eval_jobs(req, conds, dash, finished, python):
    runs = eval_targets(req, conds, dash)
    arr = req.get("arrivals") or EV.ARRIVALS
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):        # 「未公開」などの細かい表示は要らない
        return EV.make_jobs(runs, "~/LDRP_models", arrivals_by_map=arr,
                            exec_reassign=req.get("reassign"), finished=finished, python=python)


def sync_eval_code(h):
    """評価のコードを host の eval_repo に写し、結果を左右するファイルが手元と同じか確かめる."""
    if h["local"] and os.path.realpath(h["eval_repo"]) == os.path.realpath(REPO):
        return
    ex = []
    for e in SYNC_EXCLUDES:
        ex += ["--exclude", e]
    rsync_to(h, [REPO + "/"], h["eval_repo"] + "/", extra=ex)
    local = {}
    for p in EVAL_CODE:
        with open(os.path.join(REPO, p), "rb") as f:
            local[p] = hashlib.md5(f.read()).hexdigest()
    rc, out, _ = run_on(h, "cd %s && md5sum %s" % (shlex.quote(h["eval_repo"]), " ".join(EVAL_CODE)))
    remote = dict((l.split()[1], l.split()[0]) for l in out.splitlines() if len(l.split()) == 2)
    bad = [p for p in EVAL_CODE if remote.get(p) != local[p]]
    if bad:
        raise RuntimeError("evaluation code differs on %s after sync: %s" % (h["label"], ", ".join(bad)))


def launch_eval(h, req, jobs, n_par, dry_run):
    job_id = "%s_%s_%s" % (req["id"], h["label"], time.strftime("%m%d%H%M"))
    if dry_run:
        return {"job": job_id, "models": [j["name"] for j in jobs]}
    sync_eval_code(h)
    for j in jobs:                                          # モデルを評価側の名前のまま置く
        for (src, d), rel in zip(j["files"], j["dirs"]):
            if h["local"]:
                if not EV.place(src, os.path.join(h["eval_repo"], rel), j["name"]):
                    raise RuntimeError("a different file named %s is already in %s" % (j["name"], rel))
            else:
                rsync_to(h, [src], "%s/%s/" % (h["eval_repo"], rel))
    payload = [dict(j, files=[]) for j in jobs]
    rc, _, err = run_on(h, "mkdir -p ~/%s/jobs && cat > ~/%s/jobs/%s.json" % (REMOTE_DIR, REMOTE_DIR, job_id),
                        stdin=json.dumps(payload))
    if rc != 0:
        raise RuntimeError("cannot write job on %s: %s" % (h["label"], err.strip()))
    script = ("cd %s && mkdir -p logs && CUDA_VISIBLE_DEVICES= nohup %s -u tools/eval_lare_factor.py "
              "--jobs-file ~/%s/jobs/%s.json -j %d --tag %s --host %s > logs/autorun_%s.out 2>&1 < /dev/null &"
              % (shlex.quote(h["eval_repo"]), h["python"], REMOTE_DIR, job_id, n_par, job_id,
                 shlex.quote(h["label"]), job_id))
    rc, _, err = run_on(h, script)
    if rc != 0:
        raise RuntimeError("launch failed on %s: %s" % (h["label"], err.strip()))
    return {"job": job_id, "models": [j["name"] for j in jobs]}


# ---------------------------------------------------------------------------
# 所要時間の見積もり
# ---------------------------------------------------------------------------

def train_rates(dash):
    """(machine, algo, map, agents) -> steps/秒 (実績の中央値)."""
    acc = {}
    for r in dash.get("runs") or []:
        st, t = parse_dt(r.get("start_at")), r.get("t_last")
        end = parse_dt(r.get("stop_at")) if r.get("state") == "done" else parse_dt(r.get("last_seen"))
        if not st or not end or not t or (end - st).total_seconds() < 3600:
            continue
        acc.setdefault((r.get("machine"), r.get("algo"), r.get("map"), r.get("agents")), []).append(
            t / (end - st).total_seconds())
    return dict((k, statistics.median(v)) for k, v in acc.items())


def train_sec(rates, host, rec):
    cfg = rec.get("cfg") or {}
    key = (cfg.get("env_args") or rec.get("env") or {}).get("key", "")
    m = re.search(r"(\d+)agent_(map_\w+?)-v", key)
    n, mp = (int(m.group(1)), m.group(2)) if m else (None, None)
    algo, t_max = cfg.get("name"), float(cfg.get("t_max") or 0)
    r = rates.get((host, algo, mp, n))
    if r is None:                           # 同じマシン・マップの別台数 (台数に反比例とみなす)
        near = [(k[3], v) for k, v in rates.items() if k[:3] == (host, algo, mp) and k[3]]
        if near and n:
            r = statistics.median(v * k / n for k, v in near)
    if r is None:                           # 別のマシンの同じ条件
        same = [v for k, v in rates.items() if k[1:] == (algo, mp, n)]
        r = statistics.median(same) if same else DEFAULT_RATE
    return t_max / max(r, 1.0)


# ---------------------------------------------------------------------------
# tick: 依頼を見て投入する
# ---------------------------------------------------------------------------

def fits(st, fp, h):
    """1 本が使う量 fp がマシンの空きに収まるか. 測れていない資源 (None) は見ない."""
    if st["cpu_free"] is not None and st["cpu_free"] < fp["cpu"]:
        return False
    if st["mem_free_mb"] is not None and st["mem_free_mb"] < fp["mem"]:
        return False
    if st["gpu_free_mb"] is not None and fp["gpu"] and st["gpu_free_mb"] < fp["gpu"]:
        return False
    return True


def take(st, fp):
    """投入した分だけ空きを減らす (同じ tick で続けて入れるときのため)."""
    for k, f in (("cpu_free", "cpu"), ("mem_free_mb", "mem"), ("gpu_free_mb", "gpu")):
        if st.get(k) is not None:
            st[k] -= fp[f]


def tick(dry_run=False, verbose=True):
    say = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    if os.path.exists(PAUSE) and not dry_run:
        say("[autorun] paused (%s). nothing launched." % PAUSE)
        return
    q = load_queue()
    if not any(r.get("status") == "open" for r in q):
        say("[autorun] no open requests.")
        return
    conf = load_conf()
    dash = dashboard()
    conds = load_conds(dash)
    by_id = dict((c["id"], c) for c in conds)
    led = ledger()
    recs = run_records()
    hosts = [h for h in hosts_of(conf) if h.get("enabled") or dry_run]
    if not hosts:
        say("[autorun] no host has autorun.enabled: true in %s" % CONFIG)
        return
    matched = match_launches(led, dash, by_id)
    states = dict((h["label"], host_state(h, dash, led, matched)) for h in hosts)
    rates = train_rates(dash)
    # 測った「1 本が使う量」を記録 (投入の記録 -> その条件の雛形 config)
    recs_by_job = {}
    for x in led:
        c = by_id.get(x.get("cond"))
        if x.get("kind") == "train" and c is not None:
            r = pick_template(c, dash, recs)
            if r is not None:
                recs_by_job[x["job"]] = r
    fpdb = update_footprints(states, led, recs_by_job, dry_run)
    finished = EV.busy(os.path.join(REPO, "logs"))     # 評価済み + 他で予約中 (手で起動した評価も含む)

    # --- 依頼の状態を更新し、まだ投入していない分を「単位」にする ---
    units = []
    for req in q:
        if req.get("status") != "open":
            continue
        mine = [x for x in led if x.get("req") == req["id"]]
        if req["kind"] == "train":
            cond = by_id.get(req["cond"])
            if cond is None:
                req["status"], req["note"] = "error", "condition not found in plans"
                continue
            launched = [x for x in mine if x.get("kind") == "train"]
            got = [matched.get(x["job"]) for x in launched]
            if len(launched) >= req["count"]:
                if all(g and g.get("state") in ("done", "failed") for g in got):
                    if req.get("then_eval") and not req.get("eval_req"):
                        ev = {"id": new_req_id(q), "kind": "eval", "conds": [cond["id"]],
                              "uids": [g["uid"] for g in got if g.get("state") == "done"],
                              "arrivals": req.get("arrivals"), "reassign": req.get("eval_reassign") or [False, True],
                              "machines": req.get("machines"), "status": "open",
                              "created": now_utc().isoformat(), "from": req["id"]}
                        q.append(ev)
                        req["eval_req"] = ev["id"]
                        say("[autorun] %s finished training -> evaluation request %s" % (req["id"], ev["id"]))
                    req["status"] = "done"
                continue
            rec = pick_template(cond, dash, recs)
            if rec is None:
                req["note"] = "no finished/running run of this condition to copy parameters from"
                continue
            req.pop("note", None)
            for _ in range(req["count"] - len(launched)):
                units.append({"kind": "train", "req": req, "cond": cond, "rec": rec})
        else:
            targets = eval_targets(req, conds, dash)
            man = EV.load_manifest(os.path.expanduser("~/LDRP_models"))
            unpublished = [u for u in targets if u not in man]
            jobs = eval_jobs(req, conds, dash, finished, None)
            inflight = set(m for x in mine if x.get("kind") == "eval"
                           and (now_utc() - parse_dt(x["at"])).total_seconds() < 2 * float(x.get("est_sec") or 86400)
                           for m in x.get("models") or [])
            todo = [j for j in jobs if j["name"] not in inflight]
            if unpublished:
                req["note"] = "%d run(s) wait for their model to be published (collect --fetch-models)" % len(unpublished)
            elif not targets:
                req["note"] = "no finished run to evaluate yet"
            else:
                req.pop("note", None)
            if not jobs and not inflight and not unpublished and targets:
                req["status"] = "done"
                continue
            if todo:
                units.append({"kind": "eval", "req": req, "jobs": todo})

    # --- 並べる: 長い連鎖 (学習 + 続く評価) から ---
    def est(u, label):
        if u["kind"] == "train":
            sec = train_sec(rates, label, u["rec"])
            if u["req"].get("then_eval"):
                sec += 12 * EVAL_SEC_PER_COND
            return sec
        return sum(j["n_cond"] for j in u["jobs"]) * EVAL_SEC_PER_COND / float(next(
            (h.get("eval_speed") or 1.0) for h in hosts if h["label"] == label))
    for u in units:
        u["prio"] = max(est(u, h["label"]) for h in hosts)
    units.sort(key=lambda u: -u["prio"])

    # 空き: 学習スロットごとの空く時刻
    free_at = {}
    for h in hosts:
        st = states[h["label"]]
        busy = sorted(st["train_eta"])
        cap = int(h.get("max_train") or 0)
        if cap <= 0:
            free_at[h["label"]] = []
        elif len(busy) >= cap:                  # k 番目の枠が空くのは、len(busy)-cap+k 本目が終わったとき
            free_at[h["label"]] = busy[len(busy) - cap:]
        else:
            free_at[h["label"]] = [now_utc()] * (cap - len(busy)) + busy
    eval_free = dict((h["label"], max(0, int(h.get("max_eval") or 0) - states[h["label"]]["n_eval"])) for h in hosts)

    # GPU は他の人と共有しているので、autorun が同時に動かす学習の総数に上限を置く
    total_cap = int(conf.get("autorun_max_train_total") or 2)
    n_auto = sum(1 for x in led if x.get("kind") == "train"
                 and ((matched.get(x["job"]) or {}).get("state") == "running"
                      or (x["job"] not in matched
                          and (now_utc() - parse_dt(x["at"])).total_seconds() < LAUNCH_GRACE_SEC)))
    launched_any = False
    for u in units:
        allowed = [h for h in hosts if states[h["label"]]["reachable"]
                   and (not u["req"].get("machines") or h["label"] in u["req"]["machines"])]
        if u["kind"] == "train":
            best = None
            for h in allowed:
                if not free_at[h["label"]]:
                    continue
                start = free_at[h["label"]][0]
                fin = max(start, now_utc()) + datetime.timedelta(seconds=train_sec(rates, h["label"], u["rec"]))
                # 今すぐ入れる枠なら、1 本が使う量がマシンの空きに収まるかを見る (収まらなければ今は入れない)
                st = states[h["label"]]
                fp = footprint(u["rec"], fpdb)
                if start <= now_utc() and not fits(st, fp, h):
                    continue
                # 他の人が GPU を使っている (使用率が高い) ときは今は入れない
                gutil, umax = states[h["label"]].get("gpu_util"), h.get("max_gpu_util")
                if start <= now_utc() and umax is not None and gutil is not None and gutil > float(umax):
                    continue
                if best is None or fin < best[1]:
                    best = (h, fin, start)
            if best is None:
                continue
            h, fin, start = best
            free_at[h["label"]].pop(0)
            if start <= now_utc() and n_auto >= total_cap:
                say("[autorun] %s waits: %d training run(s) by autorun already running (autorun_max_train_total=%d)"
                    % (u["req"]["id"], n_auto, total_cap))
                free_at[h["label"]].insert(0, start)
                continue
            if start > now_utc():                       # 速いマシンが空くのを待つほうが早く終わる
                free_at[h["label"]].append(fin)
                free_at[h["label"]].sort()
                say("[autorun] %s waits for %s (free %s, finish %s)" % (u["req"]["id"], h["label"],
                                                                      fmt_dt(start), fmt_dt(fin)))
                continue
            try:
                info = launch_train(h, u["req"], u["cond"], u["rec"], conf, dry_run)
            except Exception as e:
                u["req"]["note"] = str(e)[:300]
                say("[autorun] %s: cannot launch on %s: %s" % (u["req"]["id"], h["label"], e))
                continue
            free_at[h["label"]].append(fin)
            free_at[h["label"]].sort()
            n_auto += 1
            take(states[h["label"]], footprint(u["rec"], fpdb))
            fp = footprint(u["rec"], fpdb)
            say("[autorun] %s train %s on %s  (est. finish %s; needs %.1f cores, %.1f GB RAM, %.1f GB GPU%s)%s" % (
                u["req"]["id"], cond_text(u["cond"]), h["label"], fmt_dt(fin), fp["cpu"], fp["mem"] / 1024,
                fp["gpu"] / 1024, ", measured" if fp.get("measured") else ", estimated",
                "  [dry-run]" if dry_run else ""))
            if dry_run:
                say("    $ " + info["cmd"])
            else:
                ledger_add({"kind": "train", "req": u["req"]["id"], "host": h["label"], "cond": u["cond"]["id"],
                            "job": info["job"], "cmd": info["cmd"], "code": info.get("code"),
                            "est_sec": (fin - now_utc()).total_seconds()})
                launched_any = True
        else:
            # 評価は 1 本 = 1 コア。設定の上限 (max_eval) と CPU の空きの小さいほうまで並列にする
            # (メモリは環境 1 つ ~0.5GB で台数にほぼよらないので見ない)
            ec = EVAL_CPU
            def eval_room(h):
                st = states[h["label"]]
                room = eval_free[h["label"]]
                if st["cpu_free"] is not None:
                    room = min(room, int(st["cpu_free"] // ec))
                return max(0, room)
            cand = [h for h in allowed if eval_room(h) > 0]
            if not cand:
                continue
            h = max(cand, key=lambda x: eval_room(x) * float(x.get("eval_speed") or 1.0))
            n_par = min(eval_room(h), len(u["jobs"]))
            jobs = [dict(j, cmd=[h["python"]] + j["cmd"][1:]) for j in u["jobs"]]
            try:
                info = launch_eval(h, u["req"], jobs, n_par, dry_run)
            except Exception as e:
                u["req"]["note"] = str(e)[:300]
                say("[autorun] %s: cannot launch evaluation on %s: %s" % (u["req"]["id"], h["label"], e))
                continue
            eval_free[h["label"]] -= n_par
            take(states[h["label"]], {"cpu": ec * n_par, "mem": 0.0, "gpu": 0.0})
            n_cond = sum(j["n_cond"] for j in jobs)
            sec = -(-len(jobs) // n_par) * max(j["n_cond"] for j in jobs) * EVAL_SEC_PER_COND / float(h.get("eval_speed") or 1.0)
            say("[autorun] %s eval %d models / %d conditions on %s (%d parallel, est. %s)%s" % (
                u["req"]["id"], len(jobs), n_cond, h["label"], n_par, fmt_dur(sec), "  [dry-run]" if dry_run else ""))
            if not dry_run:
                ledger_add({"kind": "eval", "req": u["req"]["id"], "host": h["label"], "job": info["job"],
                            "models": info["models"], "n_cond": n_cond, "est_sec": sec})
                launched_any = True
    if not dry_run:
        save_json(QUEUE, q)
    if not launched_any and not dry_run:
        say("[autorun] nothing launched this time.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_find(args):
    dash = dashboard()
    conds = load_conds(dash)
    hits = find_conds(conds, args.words)
    if args.policy_only:
        hits = [c for c in hits if c.get("collect") != "lare"]
    if not hits:
        print("no condition matches: %s" % " ".join(args.words))
        return 1
    for c in hits[:args.limit]:
        rs = runs_of(c, dash)
        cnt = lambda s: sum(1 for r in rs if r.get("state") == s)
        print("%s  %s  [done %d, running %d, failed %d]" % (c["id"], cond_text(c), cnt("done"), cnt("running"),
                                                           cnt("failed") + cnt("stalled")))
    if len(hits) > args.limit:
        print("... %d more (add words to narrow down)" % (len(hits) - args.limit))
    return 0


def _arrivals_arg(text):
    """"8x5=fixed,mmpp,bernoulli:0.05;aoba00=fixed,..." -> {"map_8x5": [...], ...}."""
    out = {}
    for part in text.split(";"):
        if "=" in part:
            m, v = part.split("=", 1)
            m = m.strip()
            out[m if m.startswith("map_") else "map_" + m] = [x.strip() for x in v.split(",") if x.strip()]
    return out


def add_requests(kind, conds, n=1, then_eval=False, machines=None, arrivals=None, reassign=None):
    """依頼を足す (コマンドとダッシュボードの両方から使う). conds: 計画の条件 (id 付き) のリスト."""
    q = load_queue()
    made = []
    if kind == "train":
        for c in conds:
            req = {"id": new_req_id(q), "kind": "train", "cond": c["id"], "cond_text": cond_text(c),
                   "count": int(n), "then_eval": bool(then_eval), "arrivals": arrivals,
                   "eval_reassign": reassign, "machines": machines or None, "status": "open",
                   "created": now_utc().isoformat()}
            q.append(req)
            made.append(req)
    else:
        req = {"id": new_req_id(q), "kind": "eval", "conds": [c["id"] for c in conds],
               "cond_text": [cond_text(c) for c in conds], "arrivals": arrivals,
               "reassign": reassign if reassign is not None else [False, True], "machines": machines or None,
               "status": "open", "created": now_utc().isoformat()}
        q.append(req)
        made.append(req)
    save_json(QUEUE, q)
    return made


def cancel_requests(ids):
    q = load_queue()
    hit = []
    for r in q:
        if r["id"] in ids and r.get("status") == "open":
            r["status"] = "cancelled"
            hit.append(r["id"])
    save_json(QUEUE, q)
    return hit


def set_pause(on):
    os.makedirs(STATE, exist_ok=True)
    if on:
        open(PAUSE, "w").close()
    elif os.path.exists(PAUSE):
        os.remove(PAUSE)


def summary(limit_closed=8):
    """ダッシュボード用の軽い要約 (ssh もダッシュボードの API も使わない)."""
    q = load_queue()
    led = ledger()
    out = []
    closed = [r for r in q if r.get("status") != "open"][-limit_closed:]
    for r in [r for r in q if r.get("status") == "open"] + closed[::-1]:
        mine = [x for x in led if x.get("req") == r["id"]]
        d = {"id": r["id"], "kind": r["kind"], "status": r.get("status"), "note": r.get("note"),
             "cond_text": r.get("cond_text"), "machines": r.get("machines"), "created": r.get("created"),
             "hosts": sorted(set(x["host"] for x in mine))}
        if r["kind"] == "train":
            d.update({"count": r.get("count"), "launched": len(mine), "then_eval": r.get("then_eval")})
        else:
            d.update({"models": len(set(m for x in mine for m in x.get("models") or []))})
        out.append(d)
    conf = load_conf()
    hosts = [{"label": h["label"], "enabled": bool(h.get("enabled")), "max_train": h.get("max_train"),
              "max_eval": h.get("max_eval")} for h in hosts_of(conf)]
    return {"requests": out, "paused": os.path.exists(PAUSE), "hosts": hosts,
            "max_train_total": conf.get("autorun_max_train_total") or 2}


def cmd_add(args):
    dash = dashboard()
    conds = load_conds(dash)
    by_id = dict((c["id"], c) for c in conds)
    for i in args.ids:
        if i not in by_id:
            print("unknown condition id: %s (use `find` to get ids)" % i)
            return 1
    machines = [m.strip() for m in args.machines.split(",")] if args.machines else None
    arrivals = _arrivals_arg(args.arrivals) if args.arrivals else None
    reassign = [x.strip().lower() in ("t", "true", "1") for x in args.reassign.split(",")] if args.reassign else None
    made = add_requests(args.kind, [by_id[i] for i in args.ids], args.n, args.then_eval, machines,
                        arrivals, reassign)
    for r in made:
        what = ("train x%d%s" % (r["count"], " + eval" if r.get("then_eval") else "")) if r["kind"] == "train" else "eval"
        print("%s accepted: %s  %s" % (r["id"], what, r.get("cond_text")))
    print("It is launched at the next tick (every 15 min), or now with: python tools/autorun.py tick")
    return 0


def cmd_status(args):
    q = load_queue()
    if not q:
        print("no requests.")
        return 0
    led = ledger()
    try:
        dash = dashboard()
    except Exception:
        dash = {"runs": []}
    try:
        by_id = dict((c["id"], c) for c in load_conds(dash))
    except Exception:
        by_id = {}
    matched = match_launches(led, dash, by_id)
    finished = EV.evaluated(os.path.join(REPO, "logs"))
    if os.path.exists(PAUSE):
        print("** paused: no new launches (python tools/autorun.py resume) **")
    for r in q:
        if not args.all and r.get("status") not in ("open",) and r["id"] not in args.ids:
            continue
        if args.ids and r["id"] not in args.ids:
            continue
        mine = [x for x in led if x.get("req") == r["id"]]
        if r["kind"] == "train":
            parts = []
            etas = []
            for x in mine:
                g = matched.get(x.get("job"))
                if g is None:
                    parts.append("%s:starting" % x["host"])
                else:
                    parts.append("%s:%s %d%% seed=%s" % (x["host"], g.get("state"),
                                                         round(100 * (g.get("progress") or 0)), g.get("seed")))
                    if g.get("state") == "running" and g.get("eta"):
                        etas.append(parse_dt(g["eta"]))
            print("%s [%s] train %d/%d launched%s  %s" % (
                r["id"], r["status"], len(mine), r["count"], " +eval" if r.get("then_eval") else "", r.get("cond_text")))
            if parts:
                print("    " + ", ".join(parts) + ("   finish ~%s" % fmt_dt(max(etas)) if etas else ""))
        else:
            models = sorted(set(m for x in mine for m in x.get("models") or []))
            done_c = sum(1 for m in models for k in finished if k[0] == m)
            total_c = sum(x.get("n_cond") or 0 for x in mine)
            hosts = ",".join(sorted(set(x["host"] for x in mine))) or "-"
            print("%s [%s] eval %d models on %s, %d/%d conditions done" % (
                r["id"], r["status"], len(models), hosts, min(done_c, total_c), total_c))
        if r.get("note"):
            print("    note: %s" % r["note"])
    return 0


def cmd_cancel(args):
    hit = cancel_requests(args.ids)
    for i in hit:
        print("%s cancelled (runs already launched keep running)" % i)
    return 0 if hit else 1


def cmd_hosts(args):
    conf = load_conf()
    dash = dashboard()
    led = ledger()
    matched = match_launches(led, dash, dict((c["id"], c) for c in load_conds(dash)))
    for h in hosts_of(conf):
        st = host_state(h, dash, led, matched)
        f = lambda v, fmt="%d": "-" if v is None else fmt % v
        print("%-5s %-8s reachable=%s  train %d/%s  eval %d/%s  free: cpu %s cores, mem %s MB, gpu %s MB (util %s%%)" % (
            h["label"], "enabled" if h.get("enabled") else "disabled", st["reachable"], st["n_train"],
            h.get("max_train"), st["n_eval"], h.get("max_eval"), f(st["cpu_free"], "%.1f"),
            f(st["mem_free_mb"]), f(st["gpu_free_mb"]), f(st.get("gpu_util"))))
    db = load_json(FOOTPRINT, {})
    if db:
        print("measured per training run (cpu cores / mem MB / gpu MB):")
        for k, v in sorted(db.items()):
            print("  %-36s %.1f / %d / %d  (n=%d)" % (k, v["cpu"], v["mem"], v["gpu"], v["n"]))
    print("autorun_max_train_total = %s (training runs autorun keeps running at once, all machines)"
          % (conf.get("autorun_max_train_total") or 2))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description="Run LDRP training / evaluation requests on several machines.")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("find", help="find plan conditions (all words must match)")
    f.add_argument("words", nargs="+")
    f.add_argument("--limit", type=int, default=20)
    f.add_argument("--policy-only", action="store_true", help="skip the LaRe-model groups")
    a = sub.add_parser("add", help="add a request")
    a.add_argument("kind", choices=["train", "eval"])
    a.add_argument("ids", nargs="+", help="condition ids from `find`")
    a.add_argument("-n", type=int, default=1, help="number of training runs (train)")
    a.add_argument("--then-eval", action="store_true", help="evaluate the runs when training finishes (train)")
    a.add_argument("--machines", help="comma separated host labels to use (default: all enabled)")
    a.add_argument("--arrivals", help='e.g. "8x5=fixed,mmpp,bernoulli:0.05;aoba00=fixed,mmpp,bernoulli:0.01"')
    a.add_argument("--reassign", help="execution-time reassign to evaluate, e.g. F,T (default both)")
    s = sub.add_parser("status", help="show requests")
    s.add_argument("ids", nargs="*")
    s.add_argument("--all", action="store_true", help="include finished / cancelled requests")
    c = sub.add_parser("cancel", help="cancel requests (launched runs keep running)")
    c.add_argument("ids", nargs="+")
    sub.add_parser("hosts", help="show machines and free slots")
    sub.add_parser("pause", help="stop launching new jobs")
    sub.add_parser("resume", help="resume launching")
    t = sub.add_parser("tick", help="launch what can be launched now")
    t.add_argument("--dry-run", action="store_true", help="show what would be launched (all hosts, nothing started)")
    args = p.parse_args(argv)

    if args.cmd == "find":
        return cmd_find(args)
    if args.cmd == "add":
        return cmd_add(args)
    if args.cmd == "status":
        return cmd_status(args)
    if args.cmd == "cancel":
        return cmd_cancel(args)
    if args.cmd == "hosts":
        return cmd_hosts(args)
    if args.cmd == "pause":
        set_pause(True)
        print("paused. running jobs are not touched.")
        return 0
    if args.cmd == "resume":
        set_pause(False)
        print("resumed.")
        return 0
    if args.cmd == "tick":
        tick(dry_run=args.dry_run)
        return 0


if __name__ == "__main__":
    sys.exit(main())
