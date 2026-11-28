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

ssh で届かないマシン (黒) は共有フォルダ (collect_config.yaml の drop_root) でつなぐ:
  投入   司令塔が <drop_root>/<label>/autorun/inbox/ にジョブを置き、そのマシンの
         tools/autorun_worker.py (launchd) が拾って起動する。空き具合は worker が status.json に書く
  依頼   司令塔でないマシン (collect_config.yaml の autorun_commander が自分でない) で add / cancel /
         pause を実行すると <drop_root>/_autorun/requests/ に置かれ、司令塔の次の tick で取り込まれる。
         find / status / hosts は司令塔が tick ごとに書き出す _autorun/ の写しを読む
使い方の全体: tools/AUTORUN.md
"""

import argparse
import contextlib
import datetime
import glob
import hashlib
import json
import os
import re
import shlex
import shutil
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
# 共有フォルダのマシンは、worker が拾う (launchd 5 分おき) + 共有フォルダの同期 + run の書き出し (15 分おき)
# のぶん遅れて見えるので、猶予を長くとる
DROP_GRACE_SEC = 120 * 60
DROP_STALE_SEC = 40 * 60        # worker の status.json がこれより古ければ、そのマシンには投入しない
COMMANDER_DIR = "_autorun"      # <drop_root>/_autorun/ = 司令塔が書き出す写し + 他のマシンからの依頼


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
        dd = CR.drop_dir_for(h, conf.get("drop_root"))
        d["drop"] = bool(dd) and h.get("ssh") in (None, "")
        d["local"] = not d["drop"] and h.get("ssh") in (None, "", "local")
        if d["drop"]:
            # python・リポジトリの場所は、そのマシンの worker が status.json に書いたものを使う
            d["drop_dir"] = os.path.join(dd, "autorun")
            ds = drop_status(d)
            for k in ("python", "train_repo", "eval_repo"):
                if not d.get(k):
                    d[k] = ds.get(k)
        out.append(d)
    return out


def drop_status(h):
    """共有フォルダのマシンの worker が書いた status.json ({} なら未設置)."""
    return load_json(os.path.join(h["drop_dir"], "status.json"), {})


def drop_fresh(ds):
    at = parse_dt(ds.get("at"))
    return at is not None and (now_utc() - at).total_seconds() < DROP_STALE_SEC


def drop_put(h, sub, name, obj=None, src=None):
    """共有フォルダのマシンへファイルを置く (書きかけを拾われないよう、.tmp に書いてから名前を変える)."""
    d = os.path.join(h["drop_dir"], sub)
    os.makedirs(d, exist_ok=True)
    dst = os.path.join(d, name)
    tmp = os.path.join(d, "." + name + ".tmp")
    if src is not None:
        shutil.copyfile(src, tmp)
    else:
        with open(tmp, "w") as f:
            json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, dst)
    return dst


def grace_of(x):
    return float(x.get("grace") or LAUNCH_GRACE_SEC)


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
          "gpu_free_mb": None, "gpu_util": None, "measured": {}, "n_train_algo": {}}
    for r in dash.get("runs") or []:
        if r.get("machine") == label and r.get("state") == "running":
            st["train_eta"].append(parse_dt(r.get("eta")) or now_utc() + datetime.timedelta(days=1))
            a = str(r.get("algo") or "").lower()
            st["n_train_algo"][a] = st["n_train_algo"].get(a, 0) + 1
    # 投入したがまだ run が見えていない学習 (猶予の間は動いているものとして数える)
    mine_running = []
    st["our_jobs"] = []                 # このマシンで autorun の学習が動いている (or 起動待ち) のジョブ
    for x in led:
        if x.get("kind") != "train" or x.get("host") != label:
            continue
        if x.get("killed"):             # 手で止めたジョブ (台帳に killed を書いたもの) は数えない
            continue
        run = (matched or {}).get(x.get("job"))
        if run is None:
            age = (now_utc() - parse_dt(x.get("at"))).total_seconds()
            if age < grace_of(x):
                st["pending_train"] += 1
                st["train_eta"].append(now_utc() + datetime.timedelta(seconds=float(x.get("est_sec") or 0)))
                st["our_jobs"].append(x["job"])
                m = re.search(r"--config=(\w+)", x.get("cmd") or "")
                if m:
                    a = m.group(1).lower()
                    st["n_train_algo"][a] = st["n_train_algo"].get(a, 0) + 1
        elif run.get("state") == "running":
            st["our_jobs"].append(x["job"])
            if (run.get("progress") or 0) > 0.01:
                mine_running.append(x)
    st["n_train"] = len(st["train_eta"])
    if h.get("drop"):
        # ssh で届かないマシン: worker が数分おきに PROBE と同じものを status.json に書いている
        ds = drop_status(h)
        st["reachable"] = drop_fresh(ds)
        st["drop_errors"] = ds.get("errors") or []
        if not st["reachable"]:
            return st
        out = ds.get("probe") or ""
    else:
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
    メモリは「実測」と「見積もり (リプレイバッファが埋まりきったとき)」の大きいほう。QMIX はバッファ
    (5000 エピソード) が埋まるにつれて増えるので、学習の初めに測った値だと少なすぎる (2026-11-14)
    """
    est = estimate_train(rec)
    got = db.get(fp_key(rec))
    if got and got.get("n"):
        return {"cpu": got["cpu"] * 1.2, "mem": max(got["mem"] * 1.2, est["mem"]), "gpu": got["gpu"] * 1.2,
                "measured": "same"}
    cfg, _env, n, mp = _train_shape(rec)
    near = [(k, v) for k, v in db.items()
            if k.startswith("train|%s|%s|" % (cfg.get("name"), mp)) and v.get("n") and v.get("est")]
    if not near:
        # 同じマップの実測も無ければ、同じアルゴリズムの別のマップの実測 (見積もりとの比) を使う。
        # 見積もりの CPU (MAPPO 9.5 コア) は実測 (3〜4 コア) よりずっと大きく、これだけで入らなくなるため
        near = [(k, v) for k, v in db.items()
                if k.startswith("train|%s|" % cfg.get("name")) and v.get("n") and v.get("est")]
    if near:
        k, v = min(near, key=lambda kv: abs(int(kv[0].rsplit("|", 1)[1]) - n))
        r = lambda f: (v[f] / max(v["est"][f], 1e-6)) * est[f] * 1.2
        return {"cpu": r("cpu"), "mem": max(r("mem"), est["mem"]), "gpu": r("gpu"), "measured": k}
    return dict(est, measured=None)


def mem_growth(st, recs_by_job, db):
    """このマシンで動いている autorun の学習が、これから増やすメモリの合計 (MB).

    1 本ごとに「最後の見積もり (footprint) − 今の実測」(実測が無ければ見積もり全部)。
    今の空き (mem_free_mb) からこれを引いたものが、本当に使ってよい空き。
    """
    total = 0.0
    # 1 つのジョブに依頼を順番に並べたもの (q19+q20 など) は同じ job が 2 回入るので、1 回だけ数える
    for job in sorted(set(st.get("our_jobs") or [])):
        rec = recs_by_job.get(job)
        if rec is None:
            continue
        full = footprint(rec, db)["mem"]
        now_mb = (st.get("measured", {}).get(job) or {}).get("mem") or 0.0
        total += max(0.0, full - now_mb)
    return total


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


LARE_NAME_KEYS = ("pretrained_lare_path_model_name", "finetuning_lare_path_model_name")


def template_for(req, cond, dash, recs, by_id):
    """依頼の雛形. 同じ条件の実績が無く、依頼に like (別の条件の id) と lare_model (ファイル名) があれば、
    その条件の雛形の LaRe モデル名だけを差し替えたものを使う (新しい LaRe モデルの最初の 1 本用).

    差し替えた結果が依頼した条件に判定されるかは launch_train の check_command_condition で確かめる。
    """
    rec = pick_template(cond, dash, recs)
    if rec is not None or not req.get("like") or not (req.get("lare_model") or req.get("factor_version")
                                                       or req.get("set")):
        return rec
    base_cond = by_id.get(req["like"])
    base = pick_template(base_cond, dash, recs) if base_cond is not None else None
    if base is None:
        return None
    env = dict(base.get("env") or {})
    if not env.get("use_lare_path"):
        return None
    if req.get("lare_model"):
        for k in LARE_NAME_KEYS:
            if env.get(k):
                env[k] = req["lare_model"]
    # 評価関数の版: 指定 (--factor-version) が優先。無ければ使う LaRe モデルの名前 (_PATH_F2_ など) から。
    # 版 1 は書かない (今までの run と同じコマンドにする)
    ver = req.get("factor_version")
    if ver is None and req.get("lare_model"):
        m = re.search(r"_PATH_F(\d+)_", req["lare_model"])
        ver = int(m.group(1)) if m else 1
    if ver is not None:
        if int(ver) == 1:
            env.pop("lare_path_factor_version", None)
        else:
            env["lare_path_factor_version"] = int(ver)
    # --set 名前=値: 雛形の env_args を上書き (台数の違う条件を作るときの env の key など)
    env.update(req.get("set") or {})
    # 学習の長さ: --like の条件と章の t_max が違えば、章の長さにする (端数 50000 などは雛形のまま)。
    # 例: 10M の章の run を雛形に、15M の章 (デコーダの更新を 5M から始める追加学習) を出すとき (2026-11-27)
    cfg = dict(base.get("cfg") or {})
    want = (cond or {}).get("t_max_m")
    if want and cfg.get("t_max"):
        t = int(cfg["t_max"])
        tail = t - int(round(t / 1e6)) * 1000000
        if int(round(t / 1e6)) != int(round(float(want))):
            cfg["t_max"] = int(round(float(want) * 1e6)) + tail
    # GPU 版にだけある装置の指定。cpu なら外しても同じ (Mac のコードには無く、残すと env の生成で落ちる)
    if env.get("lare_device") in (None, "cpu"):
        env.pop("lare_device", None)
    return dict(base, cfg=cfg, env=env, uid=None)


_YAML_CACHE = {}


def remote_defaults(h, algo):
    """host の学習用リポジトリにある epymarl の既定値 (default + envs/gymma + algs/<algo>)."""
    import yaml
    key = (h["label"], algo)
    if key in _YAML_CACHE:
        return _YAML_CACHE[key]
    if h.get("drop"):
        ec = drop_status(h).get("epymarl_config") or {}
        texts = [ec.get("default"), ec.get("gymma"), (ec.get("algs") or {}).get(algo)]
        if any(t is None for t in texts):
            raise RuntimeError("epymarl config for %s is not in %s's status.json" % (algo, h["label"]))
        d = {}
        for t in texts:
            d.update(yaml.safe_load(t) or {})
        _YAML_CACHE[key] = d
        return d
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


def cpu_only(h, rec):
    """この host ではこのアルゴリズムを CPU だけで学習するか (autorun: の cpu_algos: [mappo] など)."""
    algo = ((rec.get("cfg") or {}).get("name") or "").lower()
    return algo in [str(a).lower() for a in (h.get("cpu_algos") or [])]


def host_footprint(h, rec, db):
    """footprint() に、CPU だけで学習する host では GPU を使わないことを反映したもの."""
    fp = footprint(rec, db)
    if cpu_only(h, rec):
        fp = dict(fp, gpu=0.0)
    return fp


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
    # --beat-interval: sacred の heartbeat (既定 10 秒) と torch の計算が重なると SIGSEGV / SIGFPE で
    # 落ちることがある (GPU2 の q22, q59 で実測)。host の autorun: に beat_interval: 900 があれば間隔を広げる
    beat = " --beat-interval=%d" % int(h["beat_interval"]) if h.get("beat_interval") else ""
    cmd = "%s src/epymarl/src/main.py --config=%s --env-config=gymma%s with %s" % (
        h["python"], algo, beat, " ".join(toks))
    if cpu_only(h, rec):
        # GPU を見せずに起動する (train.py のコメントにある CPU 実行と同じ)。epymarl は GPU が無いと CPU で動く。
        # host の cpu_algos を消せば元に戻る (一時的な処置, 2026-11-12)
        cmd = 'CUDA_VISIBLE_DEVICES="" ' + cmd
    return cmd


def check_command_condition(cmd, cond, conf):
    """作ったコマンドをダッシュボードと同じ derive() に通し、依頼の条件に判定されるか."""
    rec = CR.parse_train_command(cmd)
    rec.update({"uid": "autorun:check", "machine": "autorun"})
    d = CR.derive(rec, conf.get("stale_minutes") or 90, conf.get("method_tag_by_lare_mode"),
                  conf.get("expected_t_max"), lare_chain=conf.get("lare_chain"))
    return PLAN.matches(cond, d)


def ensure_lare_model(h, env):
    """事前学習した LaRe のモデルを使う学習なら、そのファイルを host に置く (手元の models/ から送る).

    共有フォルダのマシンには直接置けないので、共有フォルダの files/ に置き、
    worker に「train_repo のどこへ写すか」を渡す。返り値: その一覧 (ssh のマシンでは空)
    """
    names = []
    if env.get("use_lare_path") and env.get("use_pretrained_lare_path"):
        names.append(env.get("pretrained_lare_path_model_name"))
    if env.get("use_lare_path") and env.get("use_finetuning_lare_path"):
        names.append(env.get("finetuning_lare_path_model_name"))
    # 環境 (drp_env) が事前学習モデルを読むのは src/lare/path/models/ だけ。checkpoints/ にあっても読まない。
    # そこで「host の models/ にある」まで揃えてから起動する (2026-11-14: 以前は checkpoints/ にあれば良しとしていた)
    #   1. host の models/ にある            → そのまま
    #   2. この Mac の models/ にある         → host の models/ へ送る
    #   3. host の checkpoints/ にだけある    → host の中で checkpoints/ から models/ へコピー
    #      (その LaRe モデルを作った学習が host で終わったばかりで、まだこの Mac に回収されていないとき)
    #   どれでもなければ、まだ使えないので投入しない (次の tick で見直す)
    attach = []
    if h.get("drop"):
        ds = drop_status(h)
        new_worker = "lare_checkpoints" in ds        # 新しい worker は models/ と checkpoints/ を分けて知らせる
        have = set(ds.get("lare_models") or []) if new_worker else set()
        ckpt = set(ds.get("lare_checkpoints") or [])
        copy = []
        for n in [x for x in names if x and x not in have]:
            local = os.path.join(REPO, "src", "lare", "path", "models", n)
            if os.path.exists(local):
                drop_put(h, "files", n, src=local)
                attach.append({"src": "files/" + n, "dest": "src/lare/path/models/" + n})
            elif n in ckpt:
                copy.append(n)
            else:
                raise RuntimeError("LaRe model %s is not in %s's models/ nor in this Mac's models/ (not ready yet)"
                                   % (n, h["label"]))
        return {"files": attach, "copy_lare_from_checkpoints": copy}
    for n in [x for x in names if x]:
        lp = "%s/src/lare/path" % h["train_repo"]
        q = shlex.quote(n)
        rc, out, _ = run_on(h, "if [ -s %s/models/%s ]; then echo models; elif [ -s %s/checkpoints/%s ]; then echo ckpt; fi"
                            % (lp, q, lp, q))
        where = out.strip()
        if where == "models":
            continue
        local = os.path.join(REPO, "src", "lare", "path", "models", n)
        if os.path.exists(local):
            rsync_to(h, [local], "%s/models/" % lp)
        elif where == "ckpt":
            rc, _, err = run_on(h, "cp -n %s/checkpoints/%s %s/models/%s" % (lp, q, lp, q))
            if rc != 0:
                raise RuntimeError("cannot copy LaRe model %s into models/ on %s: %s" % (n, h["label"], err.strip()))
        else:
            raise RuntimeError("LaRe model %s is not on %s nor in this Mac's models/ (not ready yet)" % (n, h["label"]))
    return {"files": [], "copy_lare_from_checkpoints": []}


def launch_train(h, req, cond, rec, conf, dry_run, mem_mb=None):
    cmd = build_train_command(h, rec)
    if not check_command_condition(cmd, cond, conf):
        raise RuntimeError("generated command is not judged as the requested condition")
    job_id = "%s_%s_%s" % (req["id"], h["label"], time.strftime("%m%d%H%M%S"))
    if dry_run:
        return {"cmd": cmd, "job": job_id}
    attach = ensure_lare_model(h, rec.get("env") or {})
    if h.get("drop"):
        # worker が拾って autorun_train.py で起動する (ssh のマシンと同じ ~/.ldrp/autorun/jobs/<id>.json に写す)
        drop_put(h, "inbox", job_id + ".json", {"id": job_id, "kind": "train", "commands": [cmd],
                                                "max_processes": 1, "files": attach["files"],
                                                "copy_lare_from_checkpoints": attach["copy_lare_from_checkpoints"]})
        return {"cmd": cmd, "job": job_id, "code": drop_status(h).get("train_head")}
    run_on(h, "mkdir -p ~/%s/jobs ~/%s/logs" % (REMOTE_DIR, REMOTE_DIR))
    rsync_to(h, [os.path.join(TOOLS, "autorun_train.py")], "~/%s/" % REMOTE_DIR if not h["local"]
             else os.path.expanduser("~/%s/" % REMOTE_DIR))
    job = {"id": job_id, "commands": [cmd], "max_processes": 1}
    if mem_mb is not None:
        job["mem_mb"] = round(mem_mb)       # 最後まで膨らんだときの見積もり (GPU 側の控えの worker が見る)
    rc, _, err = run_on(h, "cat > ~/%s/jobs/%s.json" % (REMOTE_DIR, job_id), stdin=json.dumps(job))
    if rc != 0:
        raise RuntimeError("cannot write job on %s: %s" % (h["label"], err.strip()))
    # { ... & } で nohup だけを背後に回す。`a && b > out &` と書くと && の列全体が背後のシェルになり、
    # そのシェルが ssh の出力をつかんだまま残るので、ssh がジョブの終わりまで戻らない (2026-11-11 GPU1 で実測)
    script = ("cd %s && { nohup %s ~/%s/autorun_train.py ~/%s/jobs/%s.json > ~/%s/logs/%s.out 2>&1 < /dev/null & }"
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
    local = eval_code_hashes()
    rc, out, _ = run_on(h, "cd %s && md5sum %s" % (shlex.quote(h["eval_repo"]), " ".join(EVAL_CODE)))
    remote = dict((l.split()[1], l.split()[0]) for l in out.splitlines() if len(l.split()) == 2)
    bad = [p for p in EVAL_CODE if remote.get(p) != local[p]]
    if bad:
        raise RuntimeError("evaluation code differs on %s after sync: %s" % (h["label"], ", ".join(bad)))


def eval_code_hashes(root=REPO):
    out = {}
    for p in EVAL_CODE:
        with open(os.path.join(root, p), "rb") as f:
            out[p] = hashlib.md5(f.read()).hexdigest()
    return out


def code_bundle(h):
    """評価のコード (ssh のマシンに rsync するのと同じ範囲) を tar.gz にして共有フォルダの files/ に置く.

    中身が同じなら作り直さない (名前 = 中身のハッシュ)。古い束は 2 つだけ残して消す。返り値: files/ からの名前
    """
    stage = os.path.join(STATE, "code_stage")
    ex = []
    for e in SYNC_EXCLUDES:
        ex += ["--exclude", e]
    subprocess.run(["rsync", "-a", "--delete"] + ex + [REPO + "/", stage + "/"], check=True,
                   capture_output=True, timeout=600)
    m = hashlib.md5()
    for dp, dns, fns in sorted(os.walk(stage)):
        dns.sort()
        for fn in sorted(fns):
            p = os.path.join(dp, fn)
            m.update(os.path.relpath(p, stage).encode("utf-8"))
            with open(p, "rb") as f:
                m.update(hashlib.md5(f.read()).digest())
    name = "code_%s.tar.gz" % m.hexdigest()[:12]
    files = os.path.join(h["drop_dir"], "files")
    if not os.path.exists(os.path.join(files, name)):
        tmp = os.path.join(STATE, name)
        # COPYFILE_DISABLE: Mac の tar が ._ で始まる属性ファイルを混ぜないようにする
        subprocess.run(["tar", "-czf", tmp, "-C", stage, "."], check=True, capture_output=True,
                       env=dict(os.environ, COPYFILE_DISABLE="1"), timeout=600)
        drop_put(h, "files", name, src=tmp)
        os.remove(tmp)
        olds = sorted(glob.glob(os.path.join(files, "code_*.tar.gz")), key=os.path.getmtime)[:-2]
        for p in olds:
            os.remove(p)
    return name


def launch_eval(h, req, jobs, n_par, dry_run):
    job_id = "%s_%s_%s" % (req["id"], h["label"], time.strftime("%m%d%H%M"))
    if dry_run:
        return {"job": job_id, "models": [j["name"] for j in jobs]}
    if h.get("drop"):
        # コードの束・モデル・ジョブを共有フォルダに置き、worker に展開と起動を任せる
        code = code_bundle(h)
        place = []
        for j in jobs:
            for (src, _d), rel in zip(j["files"], j["dirs"]):
                sub = "files/models/" + rel.replace("/", "_")
                drop_put(h, sub, j["name"], src=src)
                place.append({"src": sub + "/" + j["name"], "dest": rel + "/" + j["name"]})
        drop_put(h, "inbox", job_id + ".json", {
            "id": job_id, "kind": "eval", "code": "files/" + code, "eval_hashes": eval_code_hashes(),
            "files": place, "jobs": [dict(j, files=[]) for j in jobs], "n_par": n_par, "host": h["label"]})
        return {"job": job_id, "models": [j["name"] for j in jobs]}
    sync_eval_code(h)
    for j in jobs:                                          # モデルを評価側の名前のまま置く
        for (src, d), rel in zip(j["files"], j["dirs"]):
            if h["local"]:
                if not EV.place(src, os.path.join(h["eval_repo"], rel), j["name"]):
                    raise RuntimeError("a different file named %s is already in %s" % (j["name"], rel))
            else:
                # rsync は置き場の親フォルダまでは作らない (models/safe/ が無いマシンで失敗する)
                run_on(h, "mkdir -p %s" % shlex.quote("%s/%s" % (h["eval_repo"], rel)))
                rsync_to(h, [src], "%s/%s/" % (h["eval_repo"], rel))
    payload = [dict(j, files=[]) for j in jobs]
    rc, _, err = run_on(h, "mkdir -p ~/%s/jobs && cat > ~/%s/jobs/%s.json" % (REMOTE_DIR, REMOTE_DIR, job_id),
                        stdin=json.dumps(payload))
    if rc != 0:
        raise RuntimeError("cannot write job on %s: %s" % (h["label"], err.strip()))
    script = ("cd %s && mkdir -p logs && { CUDA_VISIBLE_DEVICES= nohup %s -u tools/eval_lare_factor.py "
              "--jobs-file ~/%s/jobs/%s.json -j %d --tag %s --host %s > logs/autorun_%s.out 2>&1 < /dev/null & }"
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


# ---------------------------------------------------------------------------
# 控え (standby, 2026-11-18〜): 司令塔 (白) が寝ている・外出していても、GPU が自分で次の学習を入れる
#
#   白の tick (GPU に届くとき)
#     1. GPU の ~/.ldrp/autorun/commander_seen を新しくし (= 司令塔がいる印)、
#        GPU が自分で起動した記録 (standby_started.jsonl) を読んで台帳に取り込む
#     2. 通常どおり投入する
#     3. まだ入っていない学習のうち先頭から数件を、その GPU の控え (standby.json) に書く
#        (1 つのジョブは 1 台の GPU の控えにだけ載せる)
#   GPU の autorun_standby.py (cron 5 分おき)
#     commander_seen が 30 分以上古い (= 白が来ていない) ときだけ、控えを先頭から、今の空き
#     (本数・メモリ・CPU) に入る分を 1 回に 1 本起動し、standby_started.jsonl に残す
#   二重投入を防ぐため、白から届かない GPU の控えに載っているジョブは、控えを書いてから
#   autorun_standby_lease_hours (既定 24 時間) の間は他のマシンに入れない。GPU 側もその期限を
#   過ぎた控えは起動しない
# ---------------------------------------------------------------------------
STANDBY_STATE = os.path.join(STATE, "standby.json")    # 白側の写し: label -> {"at", "jobs": [...]}
STANDBY_ABSENT_SEC = 30 * 60


def standby_hosts(conf):
    """控えを置く host (ssh で届く Linux のマシンで、autorun.standby が 1 以上)."""
    return [h for h in hosts_of(conf) if h.get("enabled") and not h.get("drop") and not h["local"]
            and int(h.get("standby") or 0) > 0]


def standby_lease_sec(conf):
    return float(conf.get("autorun_standby_lease_hours") or 24) * 3600


def _standby_shell(inner):
    """~/.ldrp/autorun で、GPU 側の worker と同じロック (standby.lock) を取って inner を実行するシェル."""
    d = "~/%s" % REMOTE_DIR
    return ("mkdir -p %s && cd %s && if command -v flock >/dev/null 2>&1; then flock standby.lock sh -c %s; "
            "else sh -c %s; fi" % (d, d, shlex.quote(inner), shlex.quote(inner)))


def standby_sync(hosts, say, dry_run):
    """司令塔がいる印を GPU に付け、GPU が自分で起動した学習を台帳に取り込む. 返り値: 届いた host の label."""
    reached = set()
    known = set(x.get("job") for x in ledger())
    for h in hosts:
        inner = ("true" if dry_run else "touch commander_seen") + "; cat standby_started.jsonl 2>/dev/null; true"
        rc, out, _ = run_on(h, _standby_shell(inner), timeout=60)
        if rc != 0:
            continue
        reached.add(h["label"])
        for line in out.splitlines():
            try:
                x = json.loads(line)
            except ValueError:
                continue
            if x.get("job") in known or dry_run:
                continue
            os.makedirs(STATE, exist_ok=True)
            rec = {"kind": "train", "req": x.get("req"), "host": h["label"], "cond": x.get("cond"),
                   "job": x["job"], "cmd": x.get("cmd"), "code": None, "est_sec": x.get("est_sec"),
                   "grace": LAUNCH_GRACE_SEC, "at": x.get("at"), "by": "standby"}
            with open(LEDGER, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            known.add(x["job"])
            say("[autorun] %s started %s by itself while the commander was away (at %s)"
                % (h["label"], x.get("req"), fmt_dt(parse_dt(x.get("at")))))
    return reached


def standby_held(hosts, reached, conf):
    """白から届かない GPU の控えに載っていて、まだ期限内のジョブ: 依頼 id -> 本数 (他のマシンに入れない)."""
    held = {}
    saved = load_json(STANDBY_STATE, {})
    lease = standby_lease_sec(conf)
    for h in hosts:
        if h["label"] in reached:
            continue
        rec = saved.get(h["label"]) or {}
        at = parse_dt(rec.get("at"))
        if at is None or (now_utc() - at).total_seconds() >= lease:
            continue
        for j in rec.get("jobs") or []:
            held[j["req"]] = held.get(j["req"], 0) + 1
    return held


def standby_write(hosts, reached, units, conf, fpdb, rates, say, taken=None, plan=None):
    """まだ入っていない学習を、届いた GPU の控えに書く (units は順番どおり。launched のものは除く).
    返り値: 控えに載せた単位の番号 (共有フォルダのマシンの控えと重ねないため)."""
    saved = load_json(STANDBY_STATE, {})
    taken = set() if taken is None else taken      # 控えに載せた単位 (1 つの単位は 1 台にだけ)
    model_ok = {}
    order_h = sorted((h for h in hosts if h["label"] in reached),
                     key=lambda h: (int(h.get("train_priority") or 0), hosts.index(h)))
    for h in order_h:
        jobs = []
        cap = int(h.get("standby") or 0)
        for i, u in enumerate(units):
            if len(jobs) >= cap:
                break
            if plan is not None and i not in plan.get(h["label"], ()):
                continue                    # 控えの割り振り (standby_plan) で別のマシンに配った
            if u["kind"] != "train" or u.get("launched") or i in taken or u["req"].get("series"):
                continue
            if u["req"].get("machines") and h["label"] not in u["req"]["machines"]:
                continue
            if int(h.get("max_train") or 0) <= 0:
                continue
            fp = host_footprint(h, u["rec"], fpdb)
            try:
                cmd = build_train_command(h, u["rec"])
                if not check_command_condition(cmd, u["cond"], conf):
                    continue
                env = u["rec"].get("env") or {}
                key = (h["label"], env.get("pretrained_lare_path_model_name"),
                       env.get("finetuning_lare_path_model_name"))
                if key not in model_ok:
                    try:
                        ensure_lare_model(h, env)       # LaRe のモデルを GPU の models/ に揃えておく
                        model_ok[key] = True
                    except Exception:
                        model_ok[key] = False
                if not model_ok[key]:
                    continue
            except Exception:
                continue
            rid = u["req"]["id"]
            jobs.append({"job": "%s_%s_%s_s%d" % (rid, h["label"], time.strftime("%m%d%H%M%S"), len(jobs)),
                         "req": rid, "cond": u["cond"]["id"], "cmd": cmd,
                         "algo": str((u["rec"].get("cfg") or {}).get("name") or "").lower(),
                         "mem_mb": round(fp["mem"]), "cpu": round(fp["cpu"], 2),
                         "est_sec": train_sec(rates, h["label"], u["rec"])})
            taken.add(i)
        at = now_utc().isoformat()
        body = {"at": at, "lease_sec": standby_lease_sec(conf), "absent_sec": STANDBY_ABSENT_SEC,
                "host": h["label"], "python": h["python"], "train_repo": h["train_repo"],
                "max_train": int(h.get("max_train") or 0), "mem_reserve_mb": h.get("mem_reserve_mb"),
                "cpu_target": float(h.get("cpu_target") or 0.8), "trust_max_train": bool(h.get("trust_max_train")),
                "max_train_by_algo": h.get("max_train_by_algo") or {}, "jobs": jobs}
        try:
            rsync_to(h, [os.path.join(TOOLS, "autorun_train.py"), os.path.join(TOOLS, "autorun_standby.py")],
                     "~/%s/" % REMOTE_DIR)
        except Exception as e:
            say("[autorun] standby: cannot copy the worker to %s: %s" % (h["label"], e))
            continue
        inner = "cat > standby.json.tmp && mv standby.json.tmp standby.json"
        rc, _, err = run_on(h, _standby_shell(inner), stdin=json.dumps(body, ensure_ascii=False), timeout=60)
        if rc != 0:
            say("[autorun] standby: cannot write on %s: %s" % (h["label"], err.strip()[:200]))
            continue
        # GPU の crontab に worker を登録 (登録済みなら何もしない)。cron が動いていなければ知らせる
        line = ("*/5 * * * * %s $HOME/%s/autorun_standby.py >> $HOME/%s/standby.log 2>&1"
                % (h["python"], REMOTE_DIR, REMOTE_DIR))
        rc, out, err = run_on(h, "(crontab -l 2>/dev/null | grep -q autorun_standby.py) || "
                                 "{ (crontab -l 2>/dev/null; echo %s) | crontab - && echo installed; }; "
                                 "pgrep -x cron >/dev/null || pgrep -x crond >/dev/null || echo no-cron"
                              % shlex.quote(line), timeout=60)
        if "installed" in out:
            say("[autorun] standby: registered the worker in %s's crontab" % h["label"])
        if rc != 0 or "no-cron" in out:
            say("[autorun] standby: WARNING cron is not running on %s (the standby is never started there): %s"
                % (h["label"], (err or out).strip()[:200]))
        saved[h["label"]] = {"at": at, "jobs": [{"job": j["job"], "req": j["req"]} for j in jobs]}
        if jobs:
            say("[autorun] standby on %s (started there only if the commander is away): %s"
                % (h["label"], ", ".join(j["req"] for j in jobs)))
    save_json(STANDBY_STATE, saved)
    return taken


# ---------------------------------------------------------------------------
# 共有フォルダのマシン (黒・M2) の控え (2026-11-28〜)。GPU の控えと同じ考え方を共有フォルダで行う
#
#   白の tick
#     1. <drop_dir>/commander_seen.json を新しくする (= 司令塔がいる印。iCloud で届く)
#     2. worker が控えから自分で起動したジョブ (<drop_dir>/started/<job>.json) を台帳に取り込む
#     3. 通常どおり投入する
#     4. まだ入っていない学習のうち先頭から数件を、ジョブの形のまま <drop_dir>/standby/ に置く
#        (GPU の控えに載せたものとは重ねない)
#   worker (tools/autorun_worker.py, 5 分おき)
#     commander_seen.json が DROP_ABSENT_SEC より古い (= 白が来ていない) ときだけ、控えを先頭から見て、
#     本数 (max_train / max_train_by_algo) に入る最初の 1 本を started/ に移して起動する
#   白が留守から戻った直後は、worker が起動した記録が iCloud でまだ届いていないことがある。そこで
#   戻ってから DROP_RETURN_HOLD_SEC の間は、控えに載っていた依頼を他のマシンに入れず、控えも書き換えない
# ---------------------------------------------------------------------------
DROP_ABSENT_SEC = 60 * 60           # これより長く白の印が新しくならなければ、worker が控えを起動してよい
DROP_RETURN_HOLD_SEC = 30 * 60      # 留守から戻った直後に、控えを据え置く長さ (iCloud の同期待ち)
LAST_TICK = os.path.join(STATE, "last_tick.json")


def standby_plan(gpu_hosts, reached, drop_hosts, holding, units, conf):
    """控えに載せる学習を、マシンの学習枠 (max_train) に比例するように順番に配る (2026-11-28).

    前は GPU の控えから先に埋めていたので、件数を増やすと 1 台 (GPU1) が全部取り、留守の間に
    黒・M2 が空いたままになった。先頭の学習から順に、「控えの数 / 学習枠」が一番小さいマシンへ配る
    (同じなら train_priority の小さいほう)。返り値: label -> 単位の番号の集合."""
    cand = [h for h in gpu_hosts if h["label"] in reached]
    if not holding:
        cand += drop_hosts
    load = dict((h["label"], 0) for h in cand)
    plan = dict((h["label"], set()) for h in cand)
    for i, u in enumerate(units):
        if u["kind"] != "train" or u.get("launched") or u["req"].get("series"):
            continue
        algo = str((u["rec"].get("cfg") or {}).get("name") or "").lower()
        ok = []
        for h in cand:
            cap_t = int(h.get("max_train") or 0)
            if load[h["label"]] >= int(h.get("standby") or 0) or cap_t <= 0:
                continue
            if (h.get("max_train_by_algo") or {}).get(algo) == 0:
                continue
            if u["req"].get("machines") and h["label"] not in u["req"]["machines"]:
                continue
            try:
                if not check_command_condition(build_train_command(h, u["rec"]), u["cond"], conf):
                    continue
            except Exception:
                continue
            ok.append(h)
        if not ok:
            continue
        h = min(ok, key=lambda x: (load[x["label"]] / float(int(x.get("max_train") or 1)),
                                   int(x.get("train_priority") or 0), cand.index(x)))
        load[h["label"]] += 1
        plan[h["label"]].add(i)
    return plan


def drop_standby_hosts(conf):
    """控えを置く共有フォルダのマシン (autorun.standby が 1 以上)."""
    return [h for h in hosts_of(conf) if h.get("enabled") and h.get("drop") and int(h.get("standby") or 0) > 0]


def drop_standby_begin(hosts, say, dry_run):
    """tick の最初: 司令塔がいる印を置き、留守から戻ったかを判定し、worker が起動した控えを台帳に取り込む.

    返り値: 控えを据え置く (他に入れない・書き換えない) なら True."""
    if not hosts:
        return False
    st = load_json(LAST_TICK, {})
    now = now_utc()
    prev = parse_dt(st.get("at"))
    hold_until = parse_dt(st.get("hold_until"))
    if prev is not None and (now - prev).total_seconds() > DROP_ABSENT_SEC:
        hold_until = now + datetime.timedelta(seconds=DROP_RETURN_HOLD_SEC)
        say("[autorun] back after %s away: keeping the shared-folder standby for %d min (waiting for iCloud)"
            % (fmt_dur((now - prev).total_seconds()), DROP_RETURN_HOLD_SEC // 60))
    holding = hold_until is not None and now < hold_until
    if dry_run:
        return holding
    saved = load_json(STANDBY_STATE, {})
    known = set(x.get("job") for x in ledger())
    for h in hosts:
        drop_put(h, "", "commander_seen.json", {"at": now.isoformat()})
        for j in (saved.get(h["label"]) or {}).get("jobs") or []:
            if j["job"] in known:
                continue
            p = os.path.join(h["drop_dir"], "started", j["job"] + ".json")
            job = load_json(p, None) if os.path.exists(p) else None
            if not job or job.get("error"):
                continue
            rec = {"kind": "train", "req": j["req"], "host": h["label"], "cond": j.get("cond"),
                   "job": j["job"], "cmd": (job.get("commands") or [None])[0], "code": None,
                   "est_sec": j.get("est_sec"), "grace": DROP_GRACE_SEC,
                   "at": job.get("started_at") or now.isoformat(), "by": "standby"}
            ledger_add(rec)
            known.add(j["job"])
            say("[autorun] %s started %s by itself while the commander was away (at %s)"
                % (h["label"], j["req"], fmt_dt(parse_dt(rec["at"]))))
    save_json(LAST_TICK, {"at": now.isoformat(),
                          "hold_until": hold_until.isoformat() if holding else None})
    return holding


def drop_standby_held(hosts, holding):
    """据え置き中に、共有フォルダのマシンの控えに載っていて、まだ台帳に無いジョブ: 依頼 id -> 本数."""
    held = {}
    if not holding:
        return held
    saved = load_json(STANDBY_STATE, {})
    known = set(x.get("job") for x in ledger())
    for h in hosts:
        for j in (saved.get(h["label"]) or {}).get("jobs") or []:
            if j["job"] not in known:
                held[j["req"]] = held.get(j["req"], 0) + 1
    return held


def drop_standby_write(hosts, units, conf, rates, say, taken, holding, plan=None):
    """まだ入っていない学習を、共有フォルダのマシンの控え (<drop_dir>/standby/) にジョブの形で置く."""
    if holding:
        return
    saved = load_json(STANDBY_STATE, {})
    for h in hosts:
        d = os.path.join(h["drop_dir"], "standby")
        os.makedirs(d, exist_ok=True)
        for p in glob.glob(os.path.join(d, "*.json")):       # 前の控えは置き直す (起動されたものは started/ にある)
            os.remove(p)
        jobs = []
        cap = int(h.get("standby") or 0)
        for i, u in enumerate(units):
            if len(jobs) >= cap:
                break
            if plan is not None and i not in plan.get(h["label"], ()):
                continue                    # 控えの割り振り (standby_plan) で別のマシンに配った
            if u["kind"] != "train" or u.get("launched") or i in taken or u["req"].get("series"):
                continue
            if u["req"].get("machines") and h["label"] not in u["req"]["machines"]:
                continue
            algo = str((u["rec"].get("cfg") or {}).get("name") or "").lower()
            if int(h.get("max_train") or 0) <= 0 or (h.get("max_train_by_algo") or {}).get(algo) == 0:
                continue
            try:
                cmd = build_train_command(h, u["rec"])
                if not check_command_condition(cmd, u["cond"], conf):
                    continue
                attach = ensure_lare_model(h, u["rec"].get("env") or {})
            except Exception:
                continue
            rid = u["req"]["id"]
            job_id = "%s_%s_%s_s%d" % (rid, h["label"], time.strftime("%m%d%H%M%S"), len(jobs))
            est = train_sec(rates, h["label"], u["rec"])
            drop_put(h, "standby", job_id + ".json",
                     {"id": job_id, "kind": "train", "commands": [cmd], "max_processes": 1,
                      "files": attach["files"], "copy_lare_from_checkpoints": attach["copy_lare_from_checkpoints"],
                      "req": rid, "algo": algo, "seq": len(jobs)})
            jobs.append({"job": job_id, "req": rid, "cond": u["cond"]["id"], "est_sec": est})
            taken.add(i)
        drop_put(h, "standby", "_meta.json",
                 {"at": now_utc().isoformat(), "lease_sec": standby_lease_sec(conf), "absent_sec": DROP_ABSENT_SEC,
                  "max_train": int(h.get("max_train") or 0), "max_train_by_algo": h.get("max_train_by_algo") or {}})
        saved[h["label"]] = {"at": now_utc().isoformat(), "jobs": jobs}
        if jobs:
            say("[autorun] standby on %s (started there only if the commander is away): %s"
                % (h["label"], ", ".join(j["req"] for j in jobs)))
    save_json(STANDBY_STATE, saved)


def tick(dry_run=False, verbose=True):
    say = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    if is_client():
        say("[autorun] this machine is not the commander (%s). requests are forwarded; "
            "jobs for this machine are started by tools/autorun_worker.py." % commander_label())
        return
    if not dry_run:
        ingest_requests(say)
        drop_cancelled()
    try:
        _tick(dry_run, say)
    finally:
        if not dry_run:
            publish(say)


LAST_STATES = {}        # 直近の tick で測ったマシンの状態 (publish で他のマシンに見せる)


def _tick(dry_run, say):
    conf = load_conf()
    # 控え (standby): 先に GPU へ「司令塔がいる」印を付け、留守の間に GPU が自分で起動した学習を台帳に取り込む
    sb_hosts = standby_hosts(conf)
    sb_reached = standby_sync(sb_hosts, say, dry_run) if sb_hosts else set()
    dsb_hosts = drop_standby_hosts(conf)
    dsb_holding = drop_standby_begin(dsb_hosts, say, dry_run)
    if os.path.exists(PAUSE) and not dry_run:
        standby_write(sb_hosts, sb_reached, [], conf, {}, {}, say)     # 止めている間は GPU にも起動させない
        drop_standby_write(dsb_hosts, [], conf, {}, say, set(), dsb_holding)
        say("[autorun] paused (%s). nothing launched." % PAUSE)
        return
    q = load_queue()
    if not any(r.get("status") == "open" for r in q):
        if not dry_run:
            standby_write(sb_hosts, sb_reached, [], conf, {}, {}, say)
            drop_standby_write(dsb_hosts, [], conf, {}, say, set(), dsb_holding)
        say("[autorun] no open requests.")
        return
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
    LAST_STATES.update(states)
    rates = train_rates(dash)
    # 測った「1 本が使う量」を記録 (投入の記録 -> その条件の雛形 config)
    recs_by_job = {}
    for x in led:
        c = by_id.get(x.get("cond"))
        if x.get("kind") == "train" and c is not None:
            r = template_for(next((y for y in q if y.get("id") == x.get("req")), {}), c, dash, recs, by_id)
            if r is not None:
                recs_by_job[x["job"]] = r
    fpdb = update_footprints(states, led, recs_by_job, dry_run)
    finished = EV.busy(os.path.join(REPO, "logs"))     # 評価済み + 他で予約中 (手で起動した評価も含む)

    # 白から届かない GPU の控えに載っている学習は、期限までは他のマシンに入れない (二重投入を防ぐ)
    sb_held = standby_held(sb_hosts, sb_reached, conf)
    for rid, n in drop_standby_held(dsb_hosts, dsb_holding).items():
        sb_held[rid] = sb_held.get(rid, 0) + n
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
            rec = template_for(req, cond, dash, recs, by_id)
            if rec is None:
                req["note"] = ("no finished/running run of this condition to copy parameters from"
                               + (" (nor of the --like condition)" if req.get("like") else
                                  "; use --like <id> --lare-model <file> for a new LaRe model"))
                continue
            req.pop("note", None)
            n_held = min(sb_held.get(req["id"], 0), req["count"] - len(launched))
            if n_held:
                say("[autorun] %s: %d run(s) held for the standby of an unreachable GPU" % (req["id"], n_held))
            for _ in range(req["count"] - len(launched) - n_held):
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
            # 全条件の結果 ([RESULT]) が手元に揃ったら完了。投入からの経過時間 (inflight) は待たない
            # (待つと、評価が終わってもダッシュボードの依頼欄に「受付中」が見積もりの 2 倍の時間残る)
            all_done = (targets and not unpublished
                        and not eval_jobs(req, conds, dash, EV.evaluated(os.path.join(REPO, "logs")), None))
            if all_done or (not jobs and not inflight and not unpublished and targets):
                req["status"] = "done"
                continue
            if todo:
                units.append({"kind": "eval", "req": req, "jobs": todo})

    # 共有フォルダのマシンで起動に失敗したジョブは、その依頼の note に出す (上で消した note より後に書く)
    open_by_id = dict((r["id"], r) for r in q if r.get("status") == "open")
    for st in states.values():
        for e in st.get("drop_errors") or []:
            at = parse_dt(e.get("at"))
            if at is None or (now_utc() - at).total_seconds() > 86400:
                continue
            r = open_by_id.get(str(e.get("job", "")).split("_")[0])
            if r is not None:
                r["note"] = "%s: %s" % (st["label"], str(e.get("error"))[:300])

    # 順番: 依頼の並び (queue.json の順 = 頼んだ順。`autorun.py move` で並べ替えられる) のまま。
    # 前は「長いものから」並べ、先のマシンの枠まで押さえていたが、2026-11-16 から
    # 「今空いている枠に、先頭から入るものを入れる」だけにした (下の割り当てを参照)
    rank = {}                                   # まだ投入していない依頼の中での順位 (表示用)
    for u in units:
        rank.setdefault(u["req"]["id"], len(rank) + 1)

    eval_free = dict((h["label"], max(0, int(h.get("max_eval") or 0) - states[h["label"]]["n_eval"])) for h in hosts)
    n_train = dict((h["label"], len(states[h["label"]]["train_eta"])) for h in hosts)   # 今動いている学習の本数

    # 他の人と共有している GPU マシン (autorun の shared: true) では、autorun が同時に動かす学習の総数に
    # 上限を置く。占有できるマシン (黒など) は数えない
    total_cap = int(conf.get("autorun_max_train_total") or 2)
    shared = set(h["label"] for h in hosts if h.get("shared"))
    n_auto = sum(1 for x in led if x.get("kind") == "train" and x.get("host") in shared
                 and ((matched.get(x["job"]) or {}).get("state") == "running"
                      or (x["job"] not in matched
                          and (now_utc() - parse_dt(x["at"])).total_seconds() < grace_of(x))))
    # マシンの優先順位 (小さいほど先)。学習は train_priority、評価は eval_priority
    prio = lambda h, k: int(h.get(k) or 0)
    # 同じ series の依頼は、同時に 1 本だけ動かす (1 本終わったら次)。動いている / 起動待ちのものがあれば待つ
    req_series = dict((r["id"], r.get("series")) for r in q if r.get("series"))
    busy_series = set()
    for x in led:
        sname = req_series.get(x.get("req"))
        if not sname or x.get("kind") != "train":
            continue
        run = matched.get(x["job"])
        if (run is not None and run.get("state") == "running") or (
                run is None and (now_utc() - parse_dt(x["at"])).total_seconds() < grace_of(x)):
            busy_series.add(sname)
    launched_any = False
    max_wait = float(conf.get("autorun_max_wait_hours") or 12) * 3600
    kept = {}                   # マシン -> 依頼 id。長く待っている依頼のために取っておく (後ろの依頼を入れない)
    order_l = [x["label"] for x in hosts]

    def cannot_start(h, u):
        """このマシンで今すぐ始められないなら、その理由 (短い文字列)。始められるなら None."""
        lab = h["label"]
        st_h = states[lab]
        cap = int(h.get("max_train") or 0)
        if n_train[lab] >= cap:
            return "full %d/%d" % (n_train[lab], cap)
        # アルゴリズムごとの上限 (max_train_by_algo: {qmix: 1} など)。QMIX は 1 本でメモリを多く使う
        algo = str((u["rec"].get("cfg") or {}).get("name") or "").lower()
        cap_a = (h.get("max_train_by_algo") or {}).get(algo)
        if cap_a is not None and st_h["n_train_algo"].get(algo, 0) >= int(cap_a):
            return "%s %d/%d" % (algo, st_h["n_train_algo"].get(algo, 0), int(cap_a))
        fp = host_footprint(h, u["rec"], fpdb)
        # trust_max_train: true の host は空きの見積もりを見ず、max_train の本数だけで決める
        # (黒・M2 は MAPPO を 2 本確実に回せるが、見積もり (1 本 9.5 コア) では 1 本も入らないため)
        if not h.get("trust_max_train") and not fits(st_h, fp, h):
            if st_h["cpu_free"] is not None and st_h["cpu_free"] < fp["cpu"]:
                return "cpu %.1f free < %.1f" % (st_h["cpu_free"], fp["cpu"])
            if st_h["mem_free_mb"] is not None and st_h["mem_free_mb"] < fp["mem"]:
                return "memory %.1f GB free < %.1f" % (st_h["mem_free_mb"] / 1024, fp["mem"] / 1024)
            return "gpu memory"
        # 他の人と共有のマシンでは、動いている学習がこれから増やす分も先に取っておき、さらに
        # mem_reserve_mb を残す (他の人の学習が増えても落ちないように)
        if st_h.get("mem_free_mb") is not None and h.get("mem_reserve_mb") is not None:
            room = (st_h["mem_free_mb"] - mem_growth(st_h, recs_by_job, fpdb) - fp["mem"]
                    - float(h["mem_reserve_mb"]))
            if room < 0:
                return "memory short by %.1f GB" % (-room / 1024)
        elif h.get("mem_reserve_mb") is not None:
            return "free memory unknown"        # 空きが測れていないときは入れない
        # 他の人が GPU を使っている (使用率が高い) ときは今は入れない (CPU だけで学習するなら見ない)
        gutil, umax = st_h.get("gpu_util"), h.get("max_gpu_util")
        if umax is not None and gutil is not None and gutil > float(umax) and not cpu_only(h, u["rec"]):
            return "gpu busy %d%%" % gutil
        if lab in shared and n_auto >= total_cap:
            return "autorun total %d/%d" % (n_auto, total_cap)
        return None

    for u in units:
        allowed = [h for h in hosts if states[h["label"]]["reachable"]
                   and (not u["req"].get("machines") or h["label"] in u["req"]["machines"])]
        if u["kind"] == "train" and u["req"].get("series") in busy_series:
            continue
        if u["kind"] == "train":
            allowed = [h for h in allowed if int(h.get("max_train") or 0) > 0]    # 学習しないマシン (白) は除く
            # --- 割り当て (2026-11-16〜): 依頼の順番どおりに、今空いている枠へ入れる ---
            # 先頭の依頼が今どこにも入らなければ (メモリが足りない・枠が埋まっている)、後ろの依頼で
            # 入るものを先に入れる。先のマシンの枠を押さえて待つことはしない。
            # ただし autorun_max_wait_hours (既定 12 時間) 以上待った依頼には、いちばん早く空きそうな
            # マシンを 1 台取っておき、後ろの依頼でそのマシンを埋めない (後回しにされ続けないように)
            rid = u["req"]["id"]
            ok, why = [], {}
            for h in allowed:
                r = cannot_start(h, u)
                if r is None and kept.get(h["label"], rid) != rid:
                    r = "kept for %s" % kept[h["label"]]
                if r is None:
                    ok.append(h)
                else:
                    why[h["label"]] = r
            # 届かないマシンも理由に出す (候補から黙って消えると、マシンが空いているのに気づけない)
            for h in hosts:
                if (not states[h["label"]]["reachable"] and int(h.get("max_train") or 0) > 0
                        and (not u["req"].get("machines") or h["label"] in u["req"]["machines"])):
                    why[h["label"]] = "unreachable"
            if not ok:
                since = parse_dt(u["req"].get("blocked_since"))
                if since is None:
                    since = now_utc()
                    if not dry_run:
                        u["req"]["blocked_since"] = since.isoformat()
                waited = (now_utc() - since).total_seconds()
                if waited >= max_wait and rid not in kept.values():
                    # 空になっても入らないマシン (動いている学習が 0 本) は取っておいても無駄なので除く
                    cand = [h for h in allowed if h["label"] not in kept and n_train[h["label"]] > 0
                            and not str(why.get(h["label"], "")).startswith("gpu busy")]
                    if cand:
                        hk = min(cand, key=lambda h: (min(states[h["label"]]["train_eta"]),
                                                      order_l.index(h["label"])))
                        kept[hk["label"]] = rid
                        why[hk["label"]] += " (kept for this request)"
                msg = "; ".join("%s: %s" % kv for kv in why.items()) or "no allowed machine is reachable"
                if waited >= 3600:
                    msg += "  [waiting %.0f h]" % (waited / 3600)
                if not dry_run:
                    u["req"]["wait_reason"] = msg
                say("[autorun] %s waits (#%d in queue)  %s" % (rid, rank.get(rid, 0), msg))
                continue
            # 入れられるマシンが複数あれば、優先度の高い段 (train_priority が小さい) → 動いている学習が
            # 少ないマシン → 設定ファイルで先に書かれたマシンの順。黒と M2 を同じ段にすると
            # 黒 1 → M2 1 → 黒 2 の順に埋まる (MAPPO の CPU 学習は 1 台に 2 本より 2 台に 1 本ずつのほうが速い)
            top = min(prio(x, "train_priority") for x in ok)
            h = min((x for x in ok if prio(x, "train_priority") == top),
                    key=lambda x: (n_train[x["label"]], order_l.index(x["label"])))
            fin = now_utc() + datetime.timedelta(seconds=train_sec(rates, h["label"], u["rec"]))
            try:
                info = launch_train(h, u["req"], u["cond"], u["rec"], conf, dry_run,
                                    mem_mb=host_footprint(h, u["rec"], fpdb)["mem"])
            except Exception as e:
                u["req"]["note"] = str(e)[:300]
                say("[autorun] %s: cannot launch on %s: %s" % (rid, h["label"], e))
                continue
            u["launched"] = True
            if not dry_run:
                u["req"].pop("blocked_since", None)
                u["req"].pop("wait_reason", None)
            n_train[h["label"]] += 1
            states[h["label"]]["train_eta"].append(fin)
            if u["req"].get("series"):
                busy_series.add(u["req"]["series"])
            na = states[h["label"]]["n_train_algo"]
            a_new = str((u["rec"].get("cfg") or {}).get("name") or "").lower()
            na[a_new] = na.get(a_new, 0) + 1          # 同じ tick で同じアルゴリズムを続けて入れすぎないように
            if h["label"] in shared:
                n_auto += 1
            fp = host_footprint(h, u["rec"], fpdb)
            take(states[h["label"]], fp)
            say("[autorun] %s train %s on %s  (est. finish %s; needs %.1f cores, %.1f GB RAM, %.1f GB GPU%s)%s" % (
                rid, cond_text(u["cond"]), h["label"], fmt_dt(fin), fp["cpu"], fp["mem"] / 1024,
                fp["gpu"] / 1024, ", measured" if fp.get("measured") else ", estimated",
                "  [dry-run]" if dry_run else ""))
            if dry_run:
                say("    $ " + info["cmd"])
            else:
                ledger_add({"kind": "train", "req": rid, "host": h["label"], "cond": u["cond"]["id"],
                            "job": info["job"], "cmd": info["cmd"], "code": info.get("code"),
                            "est_sec": (fin - now_utc()).total_seconds(),
                            "grace": DROP_GRACE_SEC if h.get("drop") else LAUNCH_GRACE_SEC})
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
            # 優先度 (eval_priority) の最も高い段の中で、空きが多く速いマシン
            top = min(prio(x, "eval_priority") for x in cand)
            cand = [x for x in cand if prio(x, "eval_priority") == top]
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
        plan = standby_plan(sb_hosts, sb_reached, dsb_hosts, dsb_holding, units, conf)
        taken = standby_write(sb_hosts, sb_reached, units, conf, fpdb, rates, say, plan=plan) if sb_hosts else set()
        drop_standby_write(dsb_hosts, units, conf, rates, say, taken, dsb_holding, plan=plan)
    if not launched_any and not dry_run:
        say("[autorun] nothing launched this time.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def client_conds():
    conds = published("conds.json")
    if not conds:
        raise SystemExit("no conds.json from the commander yet (%s/conds.json). "
                         "has the commander run `autorun.py tick`?" % shared_dir())
    return conds


def cmd_find(args):
    if is_client():
        dash, conds = None, client_conds()
    else:
        dash = dashboard()
        conds = load_conds(dash)
    hits = find_conds(conds, args.words)
    if args.policy_only:
        hits = [c for c in hits if c.get("collect") != "lare"]
    if not hits:
        print("no condition matches: %s" % " ".join(args.words))
        return 1
    for c in hits[:args.limit]:
        if dash is None:
            k = c.get("counts") or {}
            n = (k.get("done", 0), k.get("running", 0), k.get("failed", 0))
        else:
            rs = runs_of(c, dash)
            cnt = lambda s: sum(1 for r in rs if r.get("state") == s)
            n = (cnt("done"), cnt("running"), cnt("failed") + cnt("stalled"))
        print("%s  %s  [done %d, running %d, failed %d]" % ((c["id"], cond_text(c)) + n))
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


# ---------------------------------------------------------------------------
# 司令塔と他のマシン (共有フォルダの _autorun/ でやり取りする)
# ---------------------------------------------------------------------------

def commander_label(conf=None):
    conf = conf if conf is not None else load_conf()
    return conf.get("autorun_commander") or conf.get("machine")


def is_client(conf=None):
    """このマシンが司令塔でないか (collect_config.yaml の autorun_commander が自分の machine と違う)."""
    conf = conf if conf is not None else load_conf()
    c = conf.get("autorun_commander")
    return bool(c) and c != conf.get("machine")


def shared_dir(conf=None):
    conf = conf if conf is not None else load_conf()
    root = conf.get("drop_root")
    return os.path.join(os.path.expanduser(root), COMMANDER_DIR) if root else None


def send_request(op):
    """司令塔への依頼 (add / cancel / pause) を共有フォルダに置く. 取り込まれるのは司令塔の次の tick."""
    conf = load_conf()
    d = shared_dir(conf)
    if not d:
        raise RuntimeError("drop_root is not set in %s" % CONFIG)
    op = dict(op, origin=conf.get("machine"), at=now_utc().isoformat())
    # 同じ秒に続けて送っても上書きしないよう、末尾に乱数を付ける (名前順 = 送った順は保つ)
    name = "%s_%s_%s.json" % (now_utc().strftime("%Y%m%d%H%M%S%f"), conf.get("machine") or "host",
                              os.urandom(3).hex())
    os.makedirs(os.path.join(d, "requests"), exist_ok=True)
    tmp = os.path.join(d, "requests", "." + name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(op, f, ensure_ascii=False)
    os.replace(tmp, os.path.join(d, "requests", name))


def published(name, default=None):
    """司令塔が書き出した写し (conds.json / summary.json / hosts.json)."""
    d = shared_dir()
    return load_json(os.path.join(d, name), default) if d else default


def ingest_requests(say):
    """他のマシンから届いた依頼を queue に取り込み、requests/done/ に結果つきで移す."""
    d = shared_dir()
    if not d:
        return
    files = sorted(glob.glob(os.path.join(d, "requests", "*.json")))
    by_id = None
    for p in files:
        op = load_json(p, None)
        if op is None:                      # 同期の途中など。次の tick でもう一度読む
            continue
        try:
            if op.get("op") == "add":
                if by_id is None:
                    by_id = dict((c["id"], c) for c in load_conds(dashboard()))
                picked = [by_id[i] for i in op.get("ids") or [] if i in by_id]
                if not picked:
                    raise RuntimeError("unknown condition id(s): %s" % ", ".join(op.get("ids") or []))
                made = add_requests(op.get("kind") or "train", picked, op.get("n") or 1, op.get("then_eval"),
                                    op.get("machines"), op.get("arrivals"), op.get("reassign"),
                                    origin=op.get("origin"), like=op.get("like"), lare_model=op.get("lare_model"),
                                    factor_version=op.get("factor_version"), series=op.get("series"),
                                    env_set=op.get("set"))
                result = "accepted: " + ", ".join(r["id"] for r in made)
            elif op.get("op") == "cancel":
                result = "cancelled: " + ", ".join(cancel_requests(op.get("ids") or [])) or "nothing to cancel"
            elif op.get("op") == "pause":
                set_pause(bool(op.get("on")))
                result = "paused" if op.get("on") else "resumed"
            else:
                raise RuntimeError("unknown op %r" % op.get("op"))
        except Exception as e:
            result = "error: %s" % e
        say("[autorun] request from %s: %s %s" % (op.get("origin"), op.get("op"), result))
        os.makedirs(os.path.join(d, "requests", "done"), exist_ok=True)
        with open(os.path.join(d, "requests", "done", os.path.basename(p)), "w") as f:
            json.dump(dict(op, result=result, done_at=now_utc().isoformat()), f, ensure_ascii=False)
        os.remove(p)


def drop_cancelled():
    """取り消した依頼のうち、共有フォルダのマシンがまだ拾っていないジョブを inbox から消す."""
    ids = [r["id"] for r in load_queue() if r.get("status") == "cancelled"]
    if not ids:
        return
    for h in hosts_of(load_conf()):
        if h.get("drop"):
            for i in ids:
                for p in glob.glob(os.path.join(h["drop_dir"], "inbox", i + "_*.json")):
                    os.remove(p)


def publish(say):
    """他のマシンの find / status / hosts 用に、計画の条件・依頼の要約・マシンの状態を共有フォルダに書き出す."""
    d = shared_dir()
    if not d:
        return
    try:
        os.makedirs(d, exist_ok=True)
        try:
            dash = dashboard()
            out = []
            for c in load_conds(dash):
                rs = runs_of(c, dash)
                cnt = lambda *ss: sum(1 for r in rs if r.get("state") in ss)
                out.append(dict(c, counts={"done": cnt("done"), "running": cnt("running"),
                                           "failed": cnt("failed", "stalled")}))
            save_json(os.path.join(d, "conds.json"), json.loads(json.dumps(out, default=str)))
        except Exception as e:
            say("[autorun] publish conds: %s" % e)
        save_json(os.path.join(d, "summary.json"),
                  dict(summary(), commander=commander_label(), at=now_utc().isoformat()))
        if LAST_STATES:
            keep = ("label", "reachable", "n_train", "n_eval", "cpu_free", "mem_free_mb", "gpu_free_mb", "gpu_util")
            save_json(os.path.join(d, "hosts.json"),
                      {"at": now_utc().isoformat(),
                       "hosts": [dict((k, st.get(k)) for k in keep) for st in LAST_STATES.values()]})
    except (IOError, OSError) as e:
        say("[autorun] publish: %s" % e)


def add_requests(kind, conds, n=1, then_eval=False, machines=None, arrivals=None, reassign=None, origin=None,
                 like=None, lare_model=None, factor_version=None, series=None, env_set=None):
    """依頼を足す (コマンドとダッシュボードの両方から使う). conds: 計画の条件 (id 付き) のリスト.

    司令塔でないマシンでは、共有フォルダ経由で司令塔に送る (返り値の id は "(sent)")。
    like / lare_model: 実績の無い条件を、別の条件 (like) の設定で LaRe モデルだけ替えて学習する (template_for)
    """
    if is_client():
        send_request({"op": "add", "kind": kind, "ids": [c["id"] for c in conds], "n": int(n),
                      "then_eval": bool(then_eval), "machines": machines or None, "arrivals": arrivals,
                      "reassign": reassign, "like": like, "lare_model": lare_model,
                      "factor_version": factor_version, "series": series, "set": env_set})
        return [{"id": "(sent)", "kind": kind, "count": int(n), "then_eval": bool(then_eval),
                 "cond_text": [cond_text(c) for c in conds] if kind == "eval" else cond_text(conds[0])}]
    q = load_queue()
    made = []
    if kind == "train":
        for c in conds:
            req = {"id": new_req_id(q), "kind": "train", "cond": c["id"], "cond_text": cond_text(c),
                   "count": int(n), "then_eval": bool(then_eval), "arrivals": arrivals,
                   "eval_reassign": reassign, "machines": machines or None, "status": "open",
                   "created": now_utc().isoformat(), "origin": origin}
            if like and (lare_model or factor_version or env_set):
                req.update(like=like, lare_model=lare_model, factor_version=factor_version)
                if env_set:
                    req["set"] = env_set
            if series:
                req["series"] = series
            q.append(req)
            made.append(req)
    else:
        req = {"id": new_req_id(q), "kind": "eval", "conds": [c["id"] for c in conds],
               "cond_text": [cond_text(c) for c in conds], "arrivals": arrivals,
               "reassign": reassign if reassign is not None else [False, True], "machines": machines or None,
               "status": "open", "created": now_utc().isoformat(), "origin": origin}
        q.append(req)
        made.append(req)
    save_json(QUEUE, q)
    return made


@contextlib.contextmanager
def queue_lock():
    """依頼の一覧 (queue.json) を書き換えるコマンド同士 (tick・add・cancel・move) を 1 つずつにする.

    tick は数分かかり、その間に add / cancel すると、tick が最後に古い一覧で上書きして変更が消える
    (2026-11-14 に起きた)。fcntl.flock は同じファイルを開いた別プロセスとの排他。プロセスが落ちれば自動で外れる。
    """
    import fcntl
    os.makedirs(STATE, exist_ok=True)
    f = open(os.path.join(STATE, "queue.lock"), "w")
    try:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print("waiting for another autorun command (probably a tick) to finish ...", flush=True)
            fcntl.flock(f, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(f, fcntl.LOCK_UN)
        f.close()


def move_requests(ids, where, ref=None):
    """依頼を並べ替える. where = top / bottom / before / after (before・after は ref の依頼の前・後)."""
    q = load_queue()
    by = dict((r["id"], r) for r in q)
    bad = [i for i in list(ids) + ([ref] if ref else []) if i not in by]
    if bad:
        raise ValueError("unknown request id: %s" % ", ".join(bad))
    moving = [by[i] for i in ids]
    rest = [r for r in q if r["id"] not in ids]
    if where == "top":
        opens = [i for i, r in enumerate(rest) if r.get("status") == "open"]
        pos = opens[0] if opens else len(rest)
    elif where == "bottom":
        pos = len(rest)
    else:
        if ref in ids:
            raise ValueError("%s is one of the requests being moved" % ref)
        pos = next(i for i, r in enumerate(rest) if r["id"] == ref) + (1 if where == "after" else 0)
    q = rest[:pos] + moving + rest[pos:]
    save_json(QUEUE, q)
    return [r for r in q if r.get("status") == "open"]


def cancel_requests(ids):
    if is_client():
        send_request({"op": "cancel", "ids": list(ids)})
        return list(ids)
    q = load_queue()
    hit = []
    for r in q:
        if r["id"] in ids and r.get("status") == "open":
            r["status"] = "cancelled"
            hit.append(r["id"])
    save_json(QUEUE, q)
    return hit


def set_pause(on):
    if is_client():
        send_request({"op": "pause", "on": bool(on)})
        return
    os.makedirs(STATE, exist_ok=True)
    if on:
        open(PAUSE, "w").close()
    elif os.path.exists(PAUSE):
        os.remove(PAUSE)


def summary(limit_closed=0):
    """ダッシュボード用の軽い要約 (ssh もダッシュボードの API も使わない).

    司令塔でないマシンでは、司令塔が書き出した写しを返す (remote に書き出した時刻)。
    終わった依頼 (完了・取り消し・エラー) は既定では出さない (2026-11-12 ユーザーの希望)。
    記録は queue.json に残り、`autorun.py status --all` で見られる
    """
    if is_client():
        s = published("summary.json") or {"requests": [], "paused": False, "hosts": []}
        s["remote"] = {"commander": s.get("commander"), "at": s.get("at")}
        return s
    q = load_queue()
    led = ledger()
    out = []
    # [-0:] はリスト全体になるので、0 件のときは明示的に空にする
    closed = [r for r in q if r.get("status") != "open"][-limit_closed:] if limit_closed > 0 else []
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
    conds = client_conds() if is_client() else load_conds(dashboard())
    by_id = dict((c["id"], c) for c in conds)
    for i in args.ids:
        if i not in by_id:
            print("unknown condition id: %s (use `find` to get ids)" % i)
            return 1
    machines = [m.strip() for m in args.machines.split(",")] if args.machines else None
    arrivals = _arrivals_arg(args.arrivals) if args.arrivals else None
    reassign = [x.strip().lower() in ("t", "true", "1") for x in args.reassign.split(",")] if args.reassign else None
    env_set = {}
    for kv in args.set or []:
        if "=" not in kv:
            print("--set needs NAME=VALUE (got %s)" % kv)
            return 1
        k, v = kv.split("=", 1)
        try:
            env_set[k.strip()] = json.loads(v)         # 数値・true/false・null は型つきで
        except ValueError:
            env_set[k.strip()] = v                     # それ以外は文字列
    if bool(args.like) != bool(args.lare_model or args.factor_version or env_set):
        print("--like goes with --lare-model, --factor-version and/or --set")
        return 1
    if args.like and (args.kind != "train" or args.like not in by_id):
        print("--like needs `add train` and a condition id from `find`")
        return 1
    made = add_requests(args.kind, [by_id[i] for i in args.ids], args.n, args.then_eval, machines,
                        arrivals, reassign, like=args.like, lare_model=args.lare_model,
                        factor_version=args.factor_version, series=args.series, env_set=env_set or None)
    for r in made:
        what = ("train x%d%s" % (r["count"], " + eval" if r.get("then_eval") else "")) if r["kind"] == "train" else "eval"
        print("%s accepted: %s  %s" % (r["id"], what, r.get("cond_text")))
    if is_client():
        print("Sent to the commander (%s) through the shared folder. It is accepted at its next tick (every 15 min)."
              % commander_label())
    else:
        print("It is launched at the next tick (every 15 min), or now with: python tools/autorun.py tick")
    return 0


def cmd_status_client(args):
    s = summary()
    print("(copy written by the commander %s at %s)" % (s["remote"].get("commander"),
                                                      fmt_dt(parse_dt(s["remote"].get("at")))))
    if s.get("paused"):
        print("** paused: no new launches **")
    for r in s.get("requests") or []:
        if not args.all and r.get("status") != "open" and r["id"] not in args.ids:
            continue
        if args.ids and r["id"] not in args.ids:
            continue
        if r["kind"] == "train":
            print("%s [%s] train %d/%d launched%s on %s  %s" % (
                r["id"], r["status"], r.get("launched") or 0, r.get("count") or 0,
                " +eval" if r.get("then_eval") else "", ",".join(r.get("hosts") or []) or "-", r.get("cond_text")))
        else:
            print("%s [%s] eval %d models on %s" % (r["id"], r["status"], r.get("models") or 0,
                                                   ",".join(r.get("hosts") or []) or "-"))
        if r.get("note"):
            print("    note: %s" % r["note"])
    return 0


def cmd_status(args):
    if is_client():
        return cmd_status_client(args)
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
            if r.get("wait_reason") and r["status"] == "open" and len(mine) < r["count"]:
                print("    waits: %s" % r["wait_reason"])
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


def cmd_order(args):
    """まだ投入していない依頼を、投入する順に出す (投入済みのものは status で見る)."""
    led = ledger()
    q = []
    for r in load_queue():
        if r.get("status") != "open":
            continue
        n = sum(1 for x in led if x.get("req") == r["id"] and x.get("kind") == r["kind"])
        if r["kind"] == "train" and n >= r["count"]:
            continue
        q.append((r, n))
    if not q:
        print("no request is waiting to be launched.")
        return 0
    sb = {}                             # 依頼 id -> 控えに載っている GPU (白が来ないときだけそこで起動される)
    for lab, rec in load_json(STANDBY_STATE, {}).items():
        for j in rec.get("jobs") or []:
            sb.setdefault(j["req"], []).append(lab)
    for i, (r, n) in enumerate(q, 1):
        state = ("%d/%d launched" % (n, r["count"])) if r["kind"] == "train" else "eval"
        mach = (" only " + ",".join(r["machines"])) if r.get("machines") else ""
        print("#%-3d %-4s %-14s %s%s" % (i, r["id"], state, r.get("cond_text") or "", mach))
        if r.get("wait_reason") and (r["kind"] != "train" or n < r["count"]):
            print("            waits: %s" % r["wait_reason"])
        if r["id"] in sb:
            print("            standby on %s (started there if the commander is away)" % ",".join(sb[r["id"]]))
    return 0


def cmd_move(args):
    where = "top" if args.top else "bottom" if args.bottom else "before" if args.before else "after"
    try:
        move_requests(args.ids, where, args.before or args.after)
    except ValueError as e:
        print(e)
        return 1
    return cmd_order(args)


def cmd_cancel(args):
    hit = cancel_requests(args.ids)
    for i in hit:
        print("%s cancelled (runs already launched keep running)" % i)
    return 0 if hit else 1


def cmd_hosts(args):
    if is_client():
        h = published("hosts.json") or {}
        print("(copy written by the commander %s at %s)" % (commander_label(), fmt_dt(parse_dt(h.get("at")))))
        f = lambda v, fmt="%d": "-" if v is None else fmt % v
        for st in h.get("hosts") or []:
            print("%-5s reachable=%s  train %d  eval %d  free: cpu %s cores, mem %s MB, gpu %s MB (util %s%%)" % (
                st["label"], st.get("reachable"), st.get("n_train") or 0, st.get("n_eval") or 0,
                f(st.get("cpu_free"), "%.1f"), f(st.get("mem_free_mb")), f(st.get("gpu_free_mb")),
                f(st.get("gpu_util"))))
        return 0
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


def cmd_pull(args):
    """共有フォルダのマシン (黒・M2 など) に git pull --ff-only をさせる / その結果を見る.

    worker が次に動いたとき (5 分おき) に train_repo で git pull する。動いている学習・評価は止めない。
    ssh のマシン (GPU1・GPU2) は対象外 (GPU 版のリポジトリは別に管理する)。
    """
    hosts = dict((h["label"], h) for h in hosts_of(load_conf()) if h.get("drop"))
    labels = args.machines or sorted(hosts)
    bad = [m for m in labels if m not in hosts]
    if bad:
        print("not a shared-folder machine: %s (choose from %s)" % (", ".join(bad), ", ".join(sorted(hosts))))
        return 1
    for m in labels:
        h = hosts[m]
        if args.check:
            done = sorted(glob.glob(os.path.join(h["drop_dir"], "started", "pull_*.json")))
            waiting = glob.glob(os.path.join(h["drop_dir"], "inbox", "pull_*.json"))
            last = load_json(done[-1], {}) if done else {}
            print("%-4s last: %s  %s%s  (code now: %s)" % (
                m, last.get("id", "-"), last.get("result") or "",
                ("ERROR " + last["error"]) if last.get("error") else "",
                drop_status(h).get("train_head")))
            if waiting:
                print("      waiting in inbox: %s" % ", ".join(os.path.basename(w) for w in waiting))
            continue
        job_id = "pull_%s_%s" % (m, time.strftime("%m%d%H%M%S"))
        drop_put(h, "inbox", job_id + ".json", {"id": job_id, "kind": "pull", "host": m})
        print("%s: git pull is queued (%s). The worker runs it within ~5 min; check with `autorun.py pull --check`"
              % (m, job_id))
    return 0


def cmd_stop(args):
    """評価の依頼 (qN) で起動した評価を止める. 親 (eval_lare_factor.py) だけを止め、動いている test.py は最後まで走らせる.

    共有フォルダのマシン (黒・M2) は inbox に stop の指示を置き、worker が次に動いたとき (5 分おき) に止める。
    ssh のマシン (GPU1・GPU2) はその場で止める。依頼そのものは残るので、要らなければ cancel も行う。
    """
    if is_client():
        print("stop works only on the commander")
        return 1
    hosts = dict((h["label"], h) for h in hosts_of(load_conf()))
    jobs = [x for x in ledger() if x.get("kind") == "eval" and x.get("req") in args.ids
            and (not args.machines or x.get("host") in args.machines)]
    if not jobs:
        print("no evaluation launched for %s" % ", ".join(args.ids))
        return 1
    for x in jobs:
        h = hosts.get(x.get("host"))
        if h is None:
            print("%s: unknown host %s" % (x["job"], x.get("host")))
            continue
        if h.get("drop"):
            sid = "stop_%s_%s" % (h["label"], time.strftime("%m%d%H%M%S"))
            drop_put(h, "inbox", sid + ".json", {"id": sid, "kind": "stop", "target": x["job"], "host": h["label"]})
            print("%s on %s: stop is queued (%s). The worker runs it within ~5 min; the result is in "
                  "started/%s.json" % (x["job"], h["label"], sid, sid))
        else:
            # 先頭が python の行だけ (この命令自身のシェル・awk の行は外れる)。
            # 親だけに SIGTERM を送るので子の test.py は走り続ける
            cmd = ("ps -eo pid=,args= | awk '$2 ~ /python/ && /eval_lare_factor[.]py/ && /jobs\\/%s[.]json/ {print $1}'"
                   " | xargs -r kill -TERM; echo done" % x["job"])
            rc, out, err = run_on(h, cmd)
            print("%s on %s: %s" % (x["job"], h["label"], "SIGTERM sent" if rc == 0 else "failed: " + err.strip()))
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
    a.add_argument("--like", metavar="ID", help="(train) copy the parameters of this condition's runs when the "
                                                "requested condition has no run yet (use with --lare-model)")
    a.add_argument("--lare-model", metavar="FILE", help="(train, with --like) LaRe-Path model file name to use "
                                                        "instead of the one in the --like runs")
    a.add_argument("--series", metavar="NAME", help="(train) requests with the same series name run one at a "
                                                    "time, in the order they were added")
    a.add_argument("--factor-version", type=int, metavar="N", help="(train, with --like) LaRe-Path evaluation "
                   "function version (env_args.lare_path_factor_version) to use instead of the --like runs' one")
    a.add_argument("--set", action="append", metavar="NAME=VALUE",
                   help="(train, with --like) override an env_args value of the --like runs, e.g. "
                        "--set key=drp_env:drp_safe-2agent_map_8x5-v2 (repeatable)")
    s = sub.add_parser("status", help="show requests")
    s.add_argument("ids", nargs="*")
    s.add_argument("--all", action="store_true", help="include finished / cancelled requests")
    c = sub.add_parser("cancel", help="cancel requests (launched runs keep running)")
    c.add_argument("ids", nargs="+")
    sub.add_parser("order", help="show open requests in the order they are launched")
    m = sub.add_parser("move", help="change the order of requests (launch earlier / later)")
    m.add_argument("ids", nargs="+")
    mg = m.add_mutually_exclusive_group(required=True)
    mg.add_argument("--top", action="store_true", help="move to the front")
    mg.add_argument("--bottom", action="store_true", help="move to the end")
    mg.add_argument("--before", metavar="ID", help="put just before this request")
    mg.add_argument("--after", metavar="ID", help="put just after this request")
    sub.add_parser("hosts", help="show machines and free slots")
    g = sub.add_parser("pull", help="make shared-folder machines (e.g. 黒, M2) run git pull --ff-only")
    g.add_argument("machines", nargs="*", help="host labels (default: all shared-folder machines)")
    g.add_argument("--check", action="store_true", help="show the result of the last pull instead")
    st = sub.add_parser("stop", help="stop evaluations launched for these requests (running test.py runs finish)")
    st.add_argument("ids", nargs="+")
    st.add_argument("--machines", type=lambda s: [x for x in s.split(",") if x], default=None,
                    help="only these hosts (comma separated)")
    sub.add_parser("pause", help="stop launching new jobs")
    sub.add_parser("resume", help="resume launching")
    t = sub.add_parser("tick", help="launch what can be launched now")
    t.add_argument("--dry-run", action="store_true", help="show what would be launched (all hosts, nothing started)")
    args = p.parse_args(argv)

    if args.cmd == "find":
        return cmd_find(args)
    if args.cmd in ("add", "cancel", "move") or (args.cmd == "tick" and not args.dry_run):
        with queue_lock():
            return _main_locked(args)
    if args.cmd == "order":
        return cmd_order(args)
    return _main_locked(args)


def _main_locked(args):
    if args.cmd == "add":
        return cmd_add(args)
    if args.cmd == "move":
        return cmd_move(args)
    if args.cmd == "status":
        return cmd_status(args)
    if args.cmd == "cancel":
        return cmd_cancel(args)
    if args.cmd == "hosts":
        return cmd_hosts(args)
    if args.cmd == "pull":
        return cmd_pull(args)
    if args.cmd == "stop":
        return cmd_stop(args)
    if args.cmd == "pause":
        set_pause(True)
        print(("sent to the commander: " if is_client() else "") + "paused. running jobs are not touched.")
        return 0
    if args.cmd == "resume":
        set_pause(False)
        print(("sent to the commander: " if is_client() else "") + "resumed.")
        return 0
    if args.cmd == "tick":
        tick(dry_run=args.dry_run)
        return 0


if __name__ == "__main__":
    sys.exit(main())
