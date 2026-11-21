#!/usr/bin/env python3
"""司令塔 (白) が来ていないときに、GPU が自分で次の学習を起動する控えの worker (2026-11-18〜).

GPU の cron から 5 分おきに呼ぶ (tools/autorun.py が ~/.ldrp/autorun/ に写す):

    */5 * * * * <python> ~/.ldrp/autorun/autorun_standby.py >> ~/.ldrp/autorun/standby.log 2>&1

1 回の実行でやること (~/.ldrp/autorun/standby.lock を取ってから):
  1. commander_seen (白が tick のたびに新しくする) が absent_sec (30 分) より新しければ何もしない
  2. standby.json (白が書いた控え) が lease_sec (24 時間) より古ければ何もしない
     (白はその期限まで、控えのジョブを他のマシンに入れない。期限後は白が別のマシンに入れうるので起動しない)
  3. 控えを先頭から見て、まだ起動していないもののうち、今の空きに入る最初の 1 本を起動し、
     standby_started.jsonl に 1 行残す (白が次に来たとき台帳に取り込む)
     空きの判定は白と同じ: 本数 (max_train, max_train_by_algo)・メモリ (MemAvailable から余裕と
     mem_reserve_mb と、動いている学習がこれから膨らむ分を引く)・CPU (trust_max_train なら見ない)

標準ライブラリだけで動く。白の設定は standby.json に入っている。
"""
import datetime
import fcntl
import glob
import json
import os
import subprocess
import sys
import time

D = os.path.expanduser("~/.ldrp/autorun")
STANDBY = os.path.join(D, "standby.json")
STARTED = os.path.join(D, "standby_started.jsonl")
SEEN = os.path.join(D, "commander_seen")
LOCK = os.path.join(D, "standby.lock")
RECENT_SEC = 15 * 60            # 起動してからこの間は、load average に CPU の使用がまだ出ていないとみなす


def log(msg):
    print("[standby %s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg), flush=True)


def now_utc():
    return datetime.datetime.now(datetime.timezone.utc)


def parse_dt(s):
    try:
        d = datetime.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return d if d.tzinfo else d.replace(tzinfo=datetime.timezone.utc)


def read_started():
    out = []
    try:
        with open(STARTED) as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    except (IOError, OSError):
        pass
    return out


def processes():
    """自分のプロセス: pid -> (ppid, rss_mb, args)."""
    p = subprocess.run(["ps", "-u", str(os.getuid()), "-o", "pid=,ppid=,rss=,args="],
                       capture_output=True, text=True)
    out = {}
    for line in p.stdout.splitlines():
        parts = line.split(None, 3)
        if len(parts) == 4:
            try:
                out[int(parts[0])] = (int(parts[1]), float(parts[2]) / 1024.0, parts[3])
            except ValueError:
                pass
    return out


def training_runs(procs):
    """動いている epymarl の学習 (python src/epymarl/src/main.py ...) の --config= 名の一覧."""
    algos = []
    for _pp, _rss, args in procs.values():
        t = args.split()
        if len(t) >= 2 and os.path.basename(t[0]).startswith("python") and t[1].endswith("src/epymarl/src/main.py"):
            a = [x.split("=", 1)[1] for x in t if x.startswith("--config=")]
            algos.append(a[0].lower() if a else "?")
    return algos


def growth_mb(procs):
    """autorun が起動して動いている学習が、見積もり (job の mem_mb) までこれから膨らむ分の合計."""
    kids = {}
    for pid, (pp, _r, _a) in procs.items():
        kids.setdefault(pp, []).append(pid)
    total = 0.0
    for path in glob.glob(os.path.join(D, "jobs", "*.json")):
        try:
            with open(path) as f:
                job = json.load(f)
        except (IOError, OSError, ValueError):
            continue
        if not isinstance(job, dict):           # 評価のジョブはリスト形式 (学習のジョブだけ見る)
            continue
        try:
            mem = float(job.get("mem_mb") or 0)
        except (TypeError, ValueError):
            continue
        root = [pid for pid, v in procs.items() if "autorun_train.py" in v[2] and path.split("/")[-1] in v[2]]
        if not root or mem <= 0:
            continue
        tree, stack = set(), list(root)
        while stack:
            p = stack.pop()
            if p not in tree:
                tree.add(p)
                stack += kids.get(p, [])
        total += max(0.0, mem - sum(procs[p][1] for p in tree))
    return total


def meminfo():
    vals = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            vals[k] = float(v.split()[0]) / 1024.0
    return vals.get("MemTotal"), vals.get("MemAvailable")


def launch(sb, j):
    os.makedirs(os.path.join(D, "jobs"), exist_ok=True)
    os.makedirs(os.path.join(D, "logs"), exist_ok=True)
    path = os.path.join(D, "jobs", j["job"] + ".json")
    with open(path, "w") as f:
        json.dump({"id": j["job"], "commands": [j["cmd"]], "max_processes": 1, "mem_mb": j.get("mem_mb")}, f)
    out = open(os.path.join(D, "logs", j["job"] + ".out"), "w")
    subprocess.Popen([sb["python"], os.path.join(D, "autorun_train.py"), path], cwd=sb["train_repo"],
                     stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)


def main():
    os.makedirs(D, exist_ok=True)
    with open(LOCK, "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            with open(STANDBY) as f:
                sb = json.load(f)
        except (IOError, OSError, ValueError):
            return 0
        try:
            seen_age = time.time() - os.path.getmtime(SEEN)
        except OSError:
            seen_age = float("inf")
        if seen_age < float(sb.get("absent_sec") or 1800):
            return 0                                        # 司令塔が来ている
        at = parse_dt(sb.get("at"))
        if at is None or (now_utc() - at).total_seconds() >= float(sb.get("lease_sec") or 86400):
            return 0                                        # 控えの期限切れ
        started = read_started()
        done = set(x.get("job") for x in started)
        todo = [j for j in sb.get("jobs") or [] if j["job"] not in done]
        if not todo:
            return 0
        procs = processes()
        algos = training_runs(procs)
        total, avail = meminfo()
        free = avail - max(2000.0, 0.1 * total) - growth_mb(procs)
        reserve = float(sb.get("mem_reserve_mb") or 0)
        with open("/proc/loadavg") as f:
            load = float(f.read().split()[0])
        recent_cpu = sum(float(x.get("cpu") or 0) for x in started
                         if (now_utc() - (parse_dt(x.get("at")) or now_utc())).total_seconds() < RECENT_SEC)
        cpu_free = (os.cpu_count() or 1) * float(sb.get("cpu_target") or 0.8) - load - recent_cpu
        why = []
        for j in todo:
            if len(algos) >= int(sb.get("max_train") or 0):
                why.append("full %d/%d" % (len(algos), int(sb.get("max_train") or 0)))
                break
            cap_a = (sb.get("max_train_by_algo") or {}).get(j.get("algo"))
            if cap_a is not None and algos.count(j.get("algo")) >= int(cap_a):
                why.append("%s: %s %d/%d" % (j["req"], j.get("algo"), algos.count(j.get("algo")), int(cap_a)))
                continue
            room = free - float(j.get("mem_mb") or 0) - reserve
            if room < 0:
                why.append("%s: memory short by %.1f GB" % (j["req"], -room / 1024))
                continue
            if not sb.get("trust_max_train") and cpu_free < float(j.get("cpu") or 0):
                why.append("%s: cpu %.1f free < %.1f" % (j["req"], cpu_free, float(j.get("cpu") or 0)))
                continue
            launch(sb, j)
            rec = {"job": j["job"], "req": j["req"], "cond": j.get("cond"), "cmd": j["cmd"],
                   "est_sec": j.get("est_sec"), "cpu": j.get("cpu"), "at": now_utc().isoformat()}
            with open(STARTED, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            log("commander away for %.0f min -> started %s (%s)" % (seen_age / 60, j["req"], j["job"]))
            return 0                                        # 1 回に 1 本 (メモリや load に出るのを待つ)
        if why:
            log("commander away for %.0f min; nothing fits: %s" % (seen_age / 60, "; ".join(why)))
        # 古い記録 (2 週間より前) を消す
        keep = [x for x in started if (now_utc() - (parse_dt(x.get("at")) or now_utc())).days < 14]
        if len(keep) < len(started):
            with open(STARTED + ".tmp", "w") as f:
                for x in keep:
                    f.write(json.dumps(x, ensure_ascii=False) + "\n")
            os.replace(STARTED + ".tmp", STARTED)
    return 0


if __name__ == "__main__":
    sys.exit(main())
