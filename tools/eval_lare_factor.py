#!/usr/bin/env python3
"""LaRe 因子計画 (tools/plans/plan_lare_factor.md) の方策を評価する.

方策学習の表 (MAPPO) のうち、規定のステップまで学習した run (state=done) だけを選び、
保管リポジトリ (~/LDRP_models) の評価用モデルを手元に置いてから test.py を並列で回す。

  - run の状態はダッシュボード (/api/train) から読む (計画表と run の対応はダッシュボードと同じ)
  - モデルは src/all_policy/models/safe / src/task_assign/models/safe にコピーする (既存は上書きしない)
  - 評価ログは logs/<日時>_lare_factor_eval/ に置く。ダッシュボードは logs/ の [RESULT] 行の
    model= を manifest.jsonl と突き合わせるので、終わった条件から「学習曲線」タブの評価表に出る
  - 1 本終わるたびに aggregate.py --csv で results/summary.csv を作り直す (「評価」タブ用)
  - もう評価済みの (モデル, 到着, 実行時 reassign) は飛ばすので、何度起動し直してもよい
    (学習が新しく終わった run は、起動し直せば対象に入る)

    python tools/eval_lare_factor.py --dry-run     # 対象と実行するコマンドを見るだけ
    nohup python tools/eval_lare_factor.py > logs/eval_lare_factor.out 2>&1 &
"""

import argparse
import filecmp
import glob
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH_DIR = os.path.join(REPO, "src", "all_policy", "models", "safe")
TASK_DIR = os.path.join(REPO, "src", "task_assign", "models", "safe")

# マップごとの評価条件。p はマップの処理能力 (台数 / 1 タスクの所要ステップ) に対する
# 負荷率が 2 マップでほぼ同じ範囲 (7 台で ~0.15〜0.9) になるように選んだ
ARRIVALS = {
    "map_8x5":    ["fixed", "mmpp", "bernoulli:0.05", "bernoulli:0.1", "bernoulli:0.15", "bernoulli:0.2"],
    "map_aoba00": ["fixed", "mmpp", "bernoulli:0.01", "bernoulli:0.02", "bernoulli:0.04", "bernoulli:0.06"],
}
EXEC_REASSIGN = [False, True]
CHAPTER_ORDER = [("map_8x5", 7), ("map_8x5", 10), ("map_aoba00", 7), ("map_aoba00", 10)]


def arrival_label(a):
    """test.py の到着の書き方 -> runner.py _condition_id の書き方 (bern0.05 / mmpp / fixed)."""
    mode, _, p = a.partition(":")
    return "bern%g" % float(p) if mode == "bernoulli" else mode


def done_runs(url):
    """計画 lare_factor の方策学習 run のうち、完了したものを (uid, 章の情報) で返す."""
    with urllib.request.urlopen(url, timeout=60) as f:
        t = json.load(f)
    out = {}
    for c in t.get("plan") or []:
        if "lare_factor" not in str(c.get("plan")) or c.get("collect") == "lare":
            continue
        for s in c.get("slots") or []:
            r = s.get("run")
            if r and r.get("state") == "done" and not r.get("excluded"):
                out[r["uid"]] = {"map": c["map"], "n": int(c["agents"]),
                                 "setting": c.get("setting"),
                                 "assign": (c.get("task_assign") or "TP").lower()}
    return out


def load_manifest(models_repo):
    man = {}
    with open(os.path.join(models_repo, "manifest.jsonl")) as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("uid") and r.get("eval_name"):
                man[r["uid"]] = r
    return man


def parse_eval_name(name, map_name, n):
    """map_8x5_7_mappo_dbct_8x5_5-10M_ppo_reassign_seed0.th -> (planner, method_tag, ppo?, reassign, seed)."""
    m = re.match(r"^%s_%d_(?P<planner>[a-z_]+?)_(?P<rest>.+)_(?P<re>base|reassign)_seed(?P<seed>\d+)\.th$"
                 % (re.escape(map_name), n), name)
    if not m:
        raise ValueError("cannot parse eval name: " + name)
    rest, ppo = m.group("rest"), False
    if rest == "ppo" or rest.endswith("_ppo"):
        ppo, rest = True, rest[:-len("ppo")].rstrip("_")
    return m.group("planner"), rest, ppo, m.group("re"), int(m.group("seed"))


def place(src, dst_dir, name):
    """モデルを評価側に置く。同名の別ファイルがあれば False (上書きしない)."""
    dst = os.path.join(dst_dir, name)
    if os.path.exists(dst):
        return filecmp.cmp(src, dst, shallow=False)
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy2(src, dst)
    return True


# --- 評価の予約 ---------------------------------------------------------------
# 評価を始めるとき、これから回す (モデル, 到着, 実行時 reassign) を logs/<回>/reservations.txt に
# "[RESERVED] ..." の行で書く。.txt なので別マシンのものも eval_logs の回収で白に届く。
# 他の評価 (手で起動したもの・tools/autorun.py) はこれを見て同じ条件を回さない。
# 予約が効くのは until (見込みの 1.5 倍) までで、かつその回のログが 45 分以内に更新されている間だけ
# (評価が落ちたら自然に外れる)。aggregate.py は "[RESULT] " の行しか読まないので集計には影響しない。
RESERVED = "[RESERVED] "
RESERVE_FACTOR = 1.5
ACTIVE_SEC = 45 * 60
SEC_PER_COND = 720.0


def reserve(batch_dir, j, host=None):
    """ジョブ j の条件を予約する."""
    until = time.time() + j["n_cond"] * SEC_PER_COND * RESERVE_FACTOR
    assign = j["cmd"][6] if len(j["cmd"]) > 6 else ""
    arrivals = next((t[len("arrival="):].split(",") for t in j["cmd"] if str(t).startswith("arrival=")), [])
    reassign = next((t[len("allow_reassign="):].split(",") for t in j["cmd"] if str(t).startswith("allow_reassign=")), [])
    with open(os.path.join(batch_dir, "reservations.txt"), "a") as f:
        for a in arrivals:
            for e in reassign:
                f.write("%smodel=%s arrival=%s reassign=%d n=%d allocator=%s host=%s until=%d\n" % (
                    RESERVED, j["name"], arrival_label(a), 1 if e.lower() in ("1", "true") else 0,
                    j["n"], assign, host or socket.gethostname().split(".")[0], until))


def reservations(logs_root):
    """今効いている予約の一覧 [{model, arrival, env_reassign, n, allocator, host, until}]."""
    out, now = [], time.time()
    for p in glob.glob(os.path.join(logs_root, "**", "reservations.txt"), recursive=True):
        d = os.path.dirname(p)
        try:
            active = max(os.path.getmtime(q) for q in glob.glob(os.path.join(d, "**", "*.txt"), recursive=True))
        except ValueError:
            continue
        if now - active > ACTIVE_SEC:
            continue
        with open(p, errors="replace") as f:
            for line in f:
                if not line.startswith(RESERVED):
                    continue
                kv = dict(t.split("=", 1) for t in line[len(RESERVED):].split() if "=" in t)
                try:
                    until = float(kv.get("until", 0))
                except ValueError:
                    continue
                if until < now:
                    continue
                out.append({"model": kv.get("model"), "arrival": kv.get("arrival"),
                            "env_reassign": kv.get("reassign") == "1", "n": int(kv.get("n") or 0),
                            "allocator": kv.get("allocator"), "host": kv.get("host"), "until": until})
    return out


def busy(logs_root):
    """評価済み + 予約中の (model, 到着, 実行時 reassign). 新しく回すときはこれを飛ばす."""
    done = evaluated(logs_root)
    return done | set((r["model"], r["arrival"], r["env_reassign"]) for r in reservations(logs_root))


def evaluated(logs_root):
    """もう評価済みの (model, 到着, 実行時 reassign) の集合 (logs/ の [RESULT] 行から)."""
    done = set()
    for p in glob.glob(os.path.join(logs_root, "**", "*.txt"), recursive=True):
        try:
            with open(p, errors="replace") as f:
                for line in f:
                    if not line.startswith("[RESULT] "):
                        continue
                    kv = dict(t.split("=", 1) for t in line[9:].split() if "=" in t)
                    cond = kv.get("condition", "")
                    env_re = cond.endswith("_envreassign")
                    if env_re:
                        cond = cond[:-len("_envreassign")]
                    m = re.search(r"_(bern[0-9.e-]+|mmpp)$", cond)
                    done.add((kv.get("model"), m.group(1) if m else "fixed", env_re))
        except (IOError, OSError):
            pass
    return done


def build_jobs(args):
    runs = done_runs(args.dashboard.rstrip("/") + "/api/train")
    return make_jobs(runs, args.models_repo)


def make_jobs(runs, models_repo, arrivals_by_map=None, exec_reassign=None, finished=None, python=None):
    """評価するモデルの一覧を作る (tools/autorun.py からも使う).

    runs: uid -> {"map", "n", "assign"}。arrivals_by_map / exec_reassign を省けばこのファイルの既定値。
    評価済みの (モデル, 到着, 実行時 reassign) は飛ばす。
    """
    arrivals_by_map = arrivals_by_map or ARRIVALS
    exec_reassign = EXEC_REASSIGN if exec_reassign is None else list(exec_reassign)
    man = load_manifest(os.path.expanduser(models_repo))
    if finished is None:
        finished = busy(os.path.join(REPO, "logs"))
    jobs, seen = [], set()
    for uid, c in runs.items():
        r = man.get(uid)
        if not r:
            print("[skip] not published yet: %s" % uid)
            continue
        name = r["eval_name"]
        if name in seen:
            continue
        seen.add(name)
        planner, tag, ppo, re_tag, seed = parse_eval_name(name, c["map"], c["n"])
        assign = "ppo" if ppo else "tp"
        if assign != c["assign"]:
            print("[skip] assigner mismatch: %s (plan %s)" % (name, c["assign"]))
            continue
        base = os.path.join(os.path.expanduser(models_repo), r["path"])
        files = [(os.path.join(base, name), PATH_DIR)]
        if ppo:
            files.append((os.path.join(base.replace("/path/", "/task/", 1), name), TASK_DIR))
        if any(not os.path.exists(s) for s, _ in files):
            print("[skip] model file missing: %s" % name)
            continue
        arrivals = [a for a in arrivals_by_map.get(c["map"], [])
                    if any((name, arrival_label(a), e) not in finished for e in exec_reassign)]
        if not arrivals:
            continue
        cmd = [python or sys.executable, "-u", "test.py", c["map"], str(c["n"]), planner, assign, tag, re_tag, "",
               "model_seed=%d" % seed, "use_safe_env=True", "use_dynamic_agents=False",
               "arrival=" + ",".join(arrivals),
               "allow_reassign=" + ",".join(str(e) for e in exec_reassign)]
        order = (CHAPTER_ORDER.index((c["map"], c["n"])) if (c["map"], c["n"]) in CHAPTER_ORDER else 99,
                 tag != "safe", tag, assign, re_tag)
        jobs.append({"name": name, "map": c["map"], "n": c["n"], "files": files, "uid": uid,
                     "cmd": cmd, "order": order, "arrivals": arrivals, "reassign": exec_reassign,
                     "dirs": [os.path.relpath(d, REPO) for _s, d in files],
                     "n_cond": len(arrivals) * len(exec_reassign)})
    jobs.sort(key=lambda j: j["order"])
    return jobs


def aggregate():
    subprocess.run([sys.executable, "aggregate.py", "--csv", os.path.join("results", "summary.csv")],
                   cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser(description="Evaluate finished policies of plan_lare_factor.md.")
    ap.add_argument("-j", "--jobs", type=int, default=3, help="number of parallel test.py processes")
    ap.add_argument("--dashboard", default="http://127.0.0.1:8765")
    ap.add_argument("--models-repo", default="~/LDRP_models")
    ap.add_argument("--dry-run", action="store_true", help="list the jobs without running them")
    ap.add_argument("--host", default=None, help="machine label written in reservations (default: hostname)")
    ap.add_argument("--tag", default="lare_factor_eval",
                    help="name of the log folder: logs/<date>_<tag>/ (tools/autorun.py passes the request id)")
    ap.add_argument("--export-jobs", metavar="JSON",
                    help="write the job list (with model files placed under DIR/models/) and exit")
    ap.add_argument("--jobs-file", metavar="JSON",
                    help="run the jobs in this file (models must already be in src/*/models/safe)")
    args = ap.parse_args()

    if args.jobs_file:
        # 別マシンで回す: ダッシュボードも保管リポジトリも見ない。モデルは置いてある前提
        with open(args.jobs_file) as f:
            jobs = json.load(f)
        finished = busy(os.path.join(REPO, "logs"))
        for j in jobs:
            j["files"] = []
            j["cmd"][0] = sys.executable
            missing = [os.path.join(d, j["name"]) for d in j["dirs"]
                       if not os.path.exists(os.path.join(REPO, d, j["name"]))]
            if missing:
                print("[skip] model file missing: %s" % missing, flush=True)
                j["n_cond"] = 0
                continue
            # 起動し直したときは、まだ終わっていない到着だけを回す
            ers = j.get("reassign", EXEC_REASSIGN)
            left = [a for a in j["arrivals"]
                    if any((j["name"], arrival_label(a), e) not in finished for e in ers)]
            j["cmd"] = [t if not str(t).startswith("arrival=") else "arrival=" + ",".join(left)
                        for t in j["cmd"]]
            j["n_cond"] = len(left) * len(ers)
        jobs = [j for j in jobs if j["n_cond"]]
    else:
        jobs = build_jobs(args)
    if args.export_jobs:
        out_dir = os.path.dirname(os.path.abspath(args.export_jobs))
        for j in jobs:
            j["dirs"] = []
            for src, d in j["files"]:
                sub = os.path.relpath(d, REPO)
                os.makedirs(os.path.join(out_dir, "models", sub), exist_ok=True)
                shutil.copy2(src, os.path.join(out_dir, "models", sub, j["name"]))
                j["dirs"].append(sub)
            j["files"] = []
        with open(args.export_jobs, "w") as f:
            json.dump(jobs, f, indent=1)
        print("[export] %d jobs -> %s (models under %s/models)" % (len(jobs), args.export_jobs, out_dir))
        return 0
    total = sum(j["n_cond"] for j in jobs)
    print("[plan] %d models, %d conditions (1000 ep x 1000 steps each), %d parallel"
          % (len(jobs), total, args.jobs), flush=True)
    for j in jobs:
        print("  %-10s N=%-2d %-70s %d cond" % (j["map"], j["n"], j["name"], j["n_cond"]))
    if args.dry_run or not jobs:
        return 0

    batch_dir = os.path.join(REPO, "logs", time.strftime("%Y%m%d-%H%M%S") + "_" + args.tag)
    os.makedirs(batch_dir)
    with open(os.path.join(batch_dir, "batch_info.txt"), "x") as f:
        f.write("run_tag = %s\nstarted = %s\n" % (args.tag, time.strftime("%Y-%m-%d %H:%M:%S")))
        f.write("arrivals = %s\nexec_reassign = %s\nmodels = %d\n" % (ARRIVALS, EXEC_REASSIGN, len(jobs)))

    # 1 プロセス 1 コアに抑え、優先度も下げる (PC が重くならないように)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    queue, running, n_done = list(jobs), [], 0
    while queue or running:
        while queue and len(running) < args.jobs:
            j = queue.pop(0)
            if not all(place(s, d, j["name"]) for s, d in j["files"]):
                print("[skip] a different file with the same name is already in models/: %s" % j["name"],
                      flush=True)
                continue
            log_dir = os.path.join(batch_dir, j["map"], "safe", "mappo", "%dagent" % j["n"])
            os.makedirs(log_dir, exist_ok=True)
            f = open(os.path.join(log_dir, j["name"][:-3] + ".txt"), "x")
            p = subprocess.Popen(["nice", "-n", "10"] + j["cmd"], cwd=REPO, env=env,
                                 stdout=f, stderr=subprocess.STDOUT)
            running.append((p, f, j))
            reserve(batch_dir, j, args.host)
            print("[start] %s  %s" % (time.strftime("%m-%d %H:%M"), j["name"]), flush=True)
        time.sleep(30)
        for p, f, j in list(running):
            if p.poll() is None:
                continue
            f.close()
            running.remove((p, f, j))
            n_done += 1
            print("[%s] %s  %s  (%d/%d)" % ("done" if p.returncode == 0 else "FAILED rc=%d" % p.returncode,
                                           time.strftime("%m-%d %H:%M"), j["name"], n_done, len(jobs)),
                  flush=True)
            aggregate()
    print("[finished] %s" % time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
