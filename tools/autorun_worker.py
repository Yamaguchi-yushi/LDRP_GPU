#!/usr/bin/env python3
"""ssh で届かないマシン (黒など) で、司令塔 (tools/autorun.py) が共有フォルダに置いたジョブを起動する受け取り役.

launchd から 5 分おきに 1 回ずつ呼ぶ (登録: python tools/autorun_worker.py --install)。1 回の実行でやること:
  1. <drop_root>/<label>/autorun/inbox/*.json を started/ に移して起動する
       train: 添付の LaRe モデルを train_repo に置き、autorun_train.py で epymarl を起動 (Mac ではスリープ防止つき)
       eval : コードの束を eval_repo に展開し (中身が変わったときだけ)、方策モデルを置いて
              tools/eval_lare_factor.py を CPU で起動
       pull : train_repo で git pull --ff-only
       stop : 評価のジョブ target の親 (eval_lare_factor.py) を止める (動いている test.py は最後まで走る)
  2. eval_repo/logs/ の *.txt を共有フォルダの eval_logs/ に写す (司令塔の collect が手元の logs/ に取り込む)
  3. このマシンの空き具合などを status.json に書く (司令塔はこれを ssh の代わりに読む)

設定は tools/collect_config.yaml (このマシンのもの):
  machine: 黒                       # このマシンのラベル (司令塔の collect_config.yaml の label と同じ)
  drop_root: ~/Library/Mobile Documents/com~apple~CloudDocs/LDRP_runs
  autorun_commander: 白             # 司令塔のラベル (これがあると autorun.py の add などが司令塔に送られる)
  autorun_worker:                   # 省略可。省略時は下の既定値
    python: /opt/anaconda3/envs/ldrp/bin/python   # 既定: この worker を動かしている Python
    train_repo: ~/LDRP                            # 既定: この worker があるリポジトリ
    eval_repo: ~/LDRP_eval                        # 既定: ~/LDRP_eval (評価用のコードの写しを置く場所)

使い方の全体: tools/AUTORUN.md
"""

import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

TOOLS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(TOOLS)
sys.path.insert(0, TOOLS)
import autorun as AR        # noqa: E402  (PROBE / EVAL_CODE などを司令塔と揃える)

HOME_DIR = os.path.expanduser("~/" + AR.REMOTE_DIR)       # ~/.ldrp/autorun (ジョブ JSON・ログ)
ERRORS = os.path.join(HOME_DIR, "worker_errors.json")


def log(msg):
    print("[worker %s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg), flush=True)


def settings():
    conf = AR.load_conf()
    label, root = conf.get("machine"), conf.get("drop_root")
    if not label or not root:
        raise SystemExit("set `machine:` and `drop_root:` in %s" % AR.CONFIG)
    w = conf.get("autorun_worker") or {}
    ex = lambda p: os.path.abspath(os.path.expanduser(p))
    return {"label": label,
            "base": os.path.join(ex(root), str(label), "autorun"),
            "python": w.get("python") or sys.executable,
            "train_repo": ex(w.get("train_repo") or REPO),
            "eval_repo": ex(w.get("eval_repo") or "~/LDRP_eval"),
            "keep_local": w.get("keep_local")}


SF_DATALESS = 0x40000000           # macOS: 中身がまだ手元に無い iCloud のファイル (ls -lO で dataless と出る)
DOWNLOAD_WAIT_SEC = 120             # 1 回の実行でダウンロードを待つ長さ。間に合わなければ次の回にもう一度


def is_dataless(path):
    try:
        return bool(getattr(os.stat(path), "st_flags", 0) & SF_DATALESS)
    except OSError:
        return False


def ensure_local(paths, wait=DOWNLOAD_WAIT_SEC):
    """iCloud のファイルを手元にダウンロードさせて待つ. 返り値: まだダウンロードされていないパス.

    launchd から動く worker では、ファイルを開いても iCloud が自動でダウンロードしない
    (自分のシェルからなら開いたときにダウンロードされる。2026-11-12 黒で確認)。そこで brctl download で頼む。
    """
    todo = [p for p in paths if is_dataless(p)]
    for p in todo:
        subprocess.run(["brctl", "download", p], capture_output=True, timeout=60)
    end = time.time() + wait
    while todo and time.time() < end:
        time.sleep(3)
        todo = [p for p in todo if is_dataless(p)]
    return todo


def job_paths(s, job):
    """ジョブが共有フォルダから読むファイル (コードの束・添付)."""
    out = [os.path.join(s["base"], f["src"]) for f in job.get("files") or []]
    if job.get("code"):
        out.append(os.path.join(s["base"], job["code"]))
    return out


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def put_file(src, dst):
    """共有フォルダの添付を置く. 同名で中身の違うファイルがあれば上書きせず止める."""
    if os.path.exists(dst):
        if md5(src) != md5(dst):
            raise RuntimeError("a different file already exists: %s" % dst)
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def spawn(cmd, cwd, out_path, env=None):
    """切り離して起動する (worker や launchd が終わっても止まらない)."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "a") as out:
        subprocess.Popen(cmd, cwd=cwd, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         env=env, start_new_session=True)


def start_train(s, job):
    for f in job.get("files") or []:
        put_file(os.path.join(s["base"], f["src"]), os.path.join(s["train_repo"], f["dest"]))
    # このマシンの checkpoints/ にしか無い LaRe モデルを models/ にコピーする (環境は models/ からしか読まない)
    lp = os.path.join(s["train_repo"], "src", "lare", "path")
    for name in job.get("copy_lare_from_checkpoints") or []:
        put_file(os.path.join(lp, "checkpoints", name), os.path.join(lp, "models", name))
    path = os.path.join(HOME_DIR, "jobs", job["id"] + ".json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"id": job["id"], "commands": job["commands"],
                   "max_processes": job.get("max_processes") or 1}, f)
    cmd = [s["python"], os.path.join(TOOLS, "autorun_train.py"), path]
    if shutil.which("caffeinate"):          # Mac: 学習の間はスリープしない (-i = アイドルでのスリープを止める)
        cmd = ["caffeinate", "-i"] + cmd
    spawn(cmd, s["train_repo"], os.path.join(HOME_DIR, "logs", job["id"] + ".out"))


def start_eval(s, job):
    er = s["eval_repo"]
    os.makedirs(er, exist_ok=True)
    mark = os.path.join(er, ".autorun_code")
    code = os.path.basename(job["code"])
    have = open(mark).read().strip() if os.path.exists(mark) else ""
    if have != code:
        subprocess.run(["tar", "-xzf", os.path.join(s["base"], job["code"]), "-C", er], check=True)
        with open(mark, "w") as f:
            f.write(code)
    bad = [p for p, h in (job.get("eval_hashes") or {}).items()
           if not os.path.exists(os.path.join(er, p)) or md5(os.path.join(er, p)) != h]
    if bad:
        raise RuntimeError("evaluation code differs after unpacking: %s" % ", ".join(bad))
    for f in job.get("files") or []:
        put_file(os.path.join(s["base"], f["src"]), os.path.join(er, f["dest"]))
    path = os.path.join(HOME_DIR, "jobs", job["id"] + ".json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(job["jobs"], f)
    cmd = [s["python"], "-u", "tools/eval_lare_factor.py", "--jobs-file", path, "-j", str(job.get("n_par") or 1),
           "--tag", job["id"], "--host", s["label"]]
    if shutil.which("caffeinate"):
        cmd = ["caffeinate", "-i"] + cmd
    # 評価は CPU だけで回す (GPU マシンと同じ。結果が Mac と一致することを確認済み)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    spawn(cmd, er, os.path.join(er, "logs", "autorun_%s.out" % job["id"]), env=env)


def take_jobs(s):
    """inbox のジョブを started/ に移してから起動する (移してから起動するので、2 回起動しない)."""
    errors = AR.load_json(ERRORS, [])
    for p in sorted(glob.glob(os.path.join(s["base"], "inbox", "*.json"))):
        # 中身が手元に無い (dataless) ままだと読めないので、先にダウンロードさせる
        if ensure_local([p]):
            log("waiting for iCloud to download %s" % os.path.basename(p))
            continue
        job = AR.load_json(p, None)
        if job is None:                     # 同期の途中など。次の回にもう一度読む
            log("cannot read %s yet (still syncing?)" % os.path.basename(p))
            continue
        # コードの束・モデルも揃ってから started/ に移す (揃う前に移すと、失敗扱いになって二度と試さない)
        missing = ensure_local([x for x in job_paths(s, job) if os.path.exists(x)])
        absent = [x for x in job_paths(s, job) if not os.path.exists(x)]
        if missing or absent:
            log("waiting for iCloud for %s: %s" % (job.get("id"), ", ".join(
                os.path.basename(x) for x in missing + absent)))
            continue
        started = os.path.join(s["base"], "started")
        os.makedirs(started, exist_ok=True)
        dst = os.path.join(started, os.path.basename(p))
        os.replace(p, dst)
        try:
            kind = job.get("kind")
            if kind == "pull":
                job["result"] = git_pull(s)
            elif kind == "stop":
                job["result"] = stop_eval(job["target"])
            else:
                (start_train if kind == "train" else start_eval)(s, job)
            job["started_at"] = AR.now_utc().isoformat()
            log("started %s (%s)" % (job["id"], job.get("kind")))
        except Exception as e:
            job["error"] = str(e)[:500]
            errors = (errors + [{"job": job["id"], "error": job["error"], "at": AR.now_utc().isoformat()}])[-10:]
            log("cannot start %s: %s" % (job["id"], e))
        with open(dst, "w") as f:
            json.dump(job, f, ensure_ascii=False, indent=1)
    os.makedirs(HOME_DIR, exist_ok=True)
    with open(ERRORS, "w") as f:
        json.dump(errors, f, ensure_ascii=False)
    return errors


def running_trains():
    """このマシンで動いている学習 (epymarl の main.py) の本数を、アルゴリズムごとに数える.

    終わったのに抜けきれず CPU 0% で残っているプロセス (黒で実測) は数えない:
    CPU を 2% 以上使っているか、起動から 15 分以内のものだけを「動いている」とみなす。"""
    p = subprocess.run(["ps", "-eo", "etime=,pcpu=,args="], capture_output=True, text=True, errors="replace")
    out = {}
    for line in p.stdout.splitlines():
        parts = line.split(None, 2)
        if len(parts) != 3 or "src/epymarl/src/main.py" not in parts[2]:
            continue
        tok = parts[2].split()
        if not os.path.basename(tok[0]).startswith("python"):
            continue                        # caffeinate や sh の行は数えない (中の python を数える)
        try:
            cpu = float(parts[1])
        except ValueError:
            cpu = 0.0
        sec = AR.CR._etime_sec(parts[0])
        if cpu < 2.0 and not (sec is not None and sec < 15 * 60):
            continue
        algo = next((t.split("=", 1)[1] for t in tok if t.startswith("--config=")), "?").lower()
        out[algo] = out.get(algo, 0) + 1
    return out


def take_standby(s):
    """司令塔 (白) が来ていないとき、控え (standby/) から 1 本だけ学習を起動する (1 回の実行で 1 本まで).

    白の tick が commander_seen.json を新しくしている間は何もしない。控えの中身・上限は白が standby/_meta.json に書く。"""
    d = os.path.join(s["base"], "standby")
    meta_p, seen_p = os.path.join(d, "_meta.json"), os.path.join(s["base"], "commander_seen.json")
    if not os.path.exists(meta_p):
        return
    if ensure_local([x for x in (meta_p, seen_p) if os.path.exists(x)]):
        return
    meta, seen = AR.load_json(meta_p, {}), AR.load_json(seen_p, {})
    now = AR.now_utc()
    seen_at, meta_at = AR.parse_dt(seen.get("at")), AR.parse_dt(meta.get("at"))
    absent = float(meta.get("absent_sec") or 3600)
    if seen_at is not None and (now - seen_at).total_seconds() < absent:
        return                              # 白がいる
    if meta_at is None or (now - meta_at).total_seconds() > float(meta.get("lease_sec") or 86400):
        return                              # 控えが古すぎる (白がその後で別のマシンに入れたかもしれない)
    run = running_trains()
    if sum(run.values()) >= int(meta.get("max_train") or 0):
        return
    caps = meta.get("max_train_by_algo") or {}
    files = sorted(p for p in glob.glob(os.path.join(d, "*.json")) if not os.path.basename(p).startswith("_"))
    jobs = []
    for p in files:
        if ensure_local([p]):
            continue
        job = AR.load_json(p, None)
        if job:
            jobs.append((int(job.get("seq") or 0), p, job))
    for _seq, p, job in sorted(jobs, key=lambda x: x[0]):
        algo = str(job.get("algo") or "").lower()
        if algo in caps and run.get(algo, 0) >= int(caps[algo]):
            continue
        if ensure_local([x for x in job_paths(s, job) if os.path.exists(x)]) or \
                [x for x in job_paths(s, job) if not os.path.exists(x)]:
            log("standby %s: waiting for iCloud" % job.get("id"))
            continue
        started = os.path.join(s["base"], "started")
        os.makedirs(started, exist_ok=True)
        dst = os.path.join(started, os.path.basename(p))
        os.replace(p, dst)                  # 移してから起動する (2 回起動しない)
        try:
            start_train(s, job)
            job["started_at"] = AR.now_utc().isoformat()
            job["by"] = "standby"
            log("started %s from the standby (commander away since %s)" % (job["id"], seen.get("at")))
        except Exception as e:
            job["error"] = str(e)[:500]
            log("cannot start standby %s: %s" % (job["id"], e))
        with open(dst, "w") as f:
            json.dump(job, f, ensure_ascii=False, indent=1)
        return


def stop_eval(target):
    """評価のジョブ target (例 q49_黒_11201725) の親 (tools/eval_lare_factor.py) に SIGTERM を送る (司令塔の `autorun.py stop`).

    親だけを止めるので、動いている test.py (子) は最後まで走ってログに [RESULT] を書く。次のモデルには進まない。
    学習のジョブは対象外 (autorun_train.py を止めても学習本体が残るため)。
    """
    p = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True, errors="replace")
    key = "jobs/%s.json" % target
    pids = []
    for line in p.stdout.splitlines():
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        tok = parts[1].split()
        # 先頭が python で、引数に eval_lare_factor.py とこのジョブのファイルがあるものだけ
        # (caffeinate は中の python が終われば一緒に終わる。シェルの行は先頭が sh・zsh なので外れる)
        if not (os.path.basename(tok[0]).startswith("python")
                and any(t.endswith("eval_lare_factor.py") for t in tok)
                and any(t.endswith(key) for t in tok)):
            continue
        if int(parts[0]) != os.getpid():
            pids.append(int(parts[0]))
    if not pids:
        return "no running evaluation for %s" % target
    for pid in pids:
        os.kill(pid, 15)
    return "sent SIGTERM to %s (running test.py children keep going)" % ", ".join(str(x) for x in pids)


# git pull で手元の版を優先するファイル (マシンごとに書き換えて使うもの)。
# collect_config.yaml の autorun_worker.keep_local で変えられる
KEEP_LOCAL = ["train.py"]


def git_pull(s):
    """train_repo で git pull --ff-only する (司令塔の `autorun.py pull <マシン>` から).

    --ff-only: 手元にコミットしていない変更やずれた履歴があってマージが要るときは、何もせずに失敗する
    (勝手にマージ・上書きしない)。失敗は例外にして worker_errors.json と started/ のジョブに残す。
    動いている学習・評価は止めない (読み込み済みのコードで続く。次に起動するものから新しいコードになる)。
    """
    repo = s["train_repo"]
    git = lambda *a: subprocess.run(["git", "-C", repo] + list(a), capture_output=True, text=True, timeout=300)
    before = git("rev-parse", "--short", "HEAD").stdout.strip()
    # マシンごとに書き換えて使うファイル (train.py) は手元の版を優先する。
    # 手元の変更があると pull が止まるので、写しを取って戻し、pull のあとで手元の版を書き戻す。
    # 上流の新しい版は <名前>.upstream として横に置く (見比べて、要る変更だけ手で取り込む)
    kept = {}
    for rel in s.get("keep_local") or KEEP_LOCAL:
        path = os.path.join(repo, rel)
        if os.path.exists(path) and git("diff", "--quiet", "HEAD", "--", rel).returncode == 1:
            with open(path, "rb") as f:
                kept[rel] = f.read()
            bak = os.path.join(HOME_DIR, "keep_local", "%s.%s" % (rel.replace("/", "_"), time.strftime("%Y%m%d%H%M%S")))
            os.makedirs(os.path.dirname(bak), exist_ok=True)
            with open(bak, "wb") as f:
                f.write(kept[rel])                      # 念のための写し (書き戻しに失敗しても残る)
            git("checkout", "--", rel)
    r = git("pull", "--ff-only")
    # 共有フォルダで送ったモデル (git に入っていない) が、あとで git に入ると pull が止まる。
    # 上流の版と同じ中身のものだけ退避してやり直す (違う中身なら何もしない)。2026-11-14 M2 で実際に起きた
    blocked = []
    if r.returncode != 0 and "untracked working tree files would be overwritten" in (r.stdout + r.stderr):
        lines = (r.stdout + r.stderr).splitlines()
        k = next(n for n, l in enumerate(lines) if "untracked working tree files" in l)
        blocked = [l.strip() for l in lines[k + 1:] if l.startswith(("\t", "  ")) and l.strip()]
        moved = []
        for rel in blocked:
            up = subprocess.run(["git", "-C", repo, "show", "FETCH_HEAD:%s" % rel], capture_output=True, timeout=60)
            path = os.path.join(repo, rel)
            if up.returncode != 0 or not os.path.exists(path):
                break
            with open(path, "rb") as f:
                if f.read() != up.stdout:
                    break
            bak = os.path.join(HOME_DIR, "keep_local", "%s.%s" % (rel.replace("/", "_"), time.strftime("%Y%m%d%H%M%S")))
            os.makedirs(os.path.dirname(bak), exist_ok=True)
            shutil.move(path, bak)
            moved.append((path, bak))
        else:
            r = git("pull", "--ff-only")
        for path, bak in moved:                     # 置き直されなかったときは元に戻す
            if not os.path.exists(path):
                shutil.move(bak, path)
    after = git("rev-parse", "--short", "HEAD").stdout.strip()
    notes = []
    for rel, data in kept.items():
        path = os.path.join(repo, rel)
        with open(path, "rb") as f:
            upstream = f.read()
        if upstream != data and before != after:
            with open(path + ".upstream", "wb") as f:
                f.write(upstream)
            notes.append("%s kept local (upstream copy: %s.upstream)" % (rel, rel))
        else:
            notes.append("%s kept local" % rel)
        with open(path, "wb") as f:
            f.write(data)
    out = (r.stdout + r.stderr).strip()
    if r.returncode != 0:
        raise RuntimeError("git pull failed (%s): %s%s" % (before, out[-400:],
                                                           ("; " + "; ".join(notes)) if notes else ""))
    log("git pull: %s -> %s%s" % (before, after, ("  (" + "; ".join(notes) + ")") if notes else ""))
    return "%s -> %s%s" % (before, after, ("  (" + "; ".join(notes) + ")") if notes else "")


def copy_eval_logs(s):
    src = os.path.join(s["eval_repo"], "logs")
    if os.path.isdir(src):
        subprocess.run(["rsync", "-a", "--include", "*/", "--include", "*.txt", "--exclude", "*",
                        src + "/", os.path.join(s["base"], "eval_logs") + "/"], capture_output=True)


def read(path):
    try:
        with open(path) as f:
            return f.read()
    except (IOError, OSError):
        return None


def write_status(s, errors):
    tr, er = s["train_repo"], s["eval_repo"]
    cfg = os.path.join(tr, "src", "epymarl", "src", "config")
    algs = dict((os.path.basename(p)[:-5], read(p)) for p in glob.glob(os.path.join(cfg, "algs", "*.yaml")))
    # 環境が事前学習モデルを読むのは models/ だけなので、models/ と checkpoints/ を分けて知らせる
    # (以前は合わせて lare_models に入れていたため、checkpoints/ にしか無いモデルも「ある」とみなされていた)
    lare, ckpt = [], []
    for sub, out in (("models", lare), ("checkpoints", ckpt)):
        out += [os.path.basename(p) for p in glob.glob(os.path.join(tr, "src", "lare", "path", sub, "*.pth"))]
    head = subprocess.run("git rev-parse --short HEAD 2>/dev/null; git status --porcelain 2>/dev/null | wc -l",
                          shell=True, cwd=tr, capture_output=True, text=True).stdout
    probe = subprocess.run(["bash", "-c", AR.PROBE], capture_output=True, text=True, timeout=60).stdout
    st = {"at": AR.now_utc().isoformat(), "label": s["label"], "python": s["python"],
          "train_repo": tr, "eval_repo": er, "probe": probe, "train_head": " ".join(head.split()),
          "epymarl_config": {"default": read(os.path.join(cfg, "default.yaml")),
                             "gymma": read(os.path.join(cfg, "envs", "gymma.yaml")), "algs": algs},
          "lare_models": sorted(set(lare)), "lare_checkpoints": sorted(set(ckpt)), "errors": errors}
    os.makedirs(s["base"], exist_ok=True)
    tmp = os.path.join(s["base"], ".status.json.tmp")
    with open(tmp, "w") as f:
        json.dump(st, f, ensure_ascii=False)
    os.replace(tmp, os.path.join(s["base"], "status.json"))


LABEL = "com.ldrp.autorun-worker"
PLIST = os.path.expanduser("~/Library/LaunchAgents/%s.plist" % LABEL)


def install(interval, calendar=False):
    """launchd に登録する (この Python・このリポジトリのパスで plist を作る).

    AbandonProcessGroup: worker が終わっても、起動した学習・評価を launchd に止めさせない
    calendar: StartInterval (起動からの秒数) の代わりに StartCalendarInterval (毎時 0, 5, 10, ... 分) で
              起動する。ログイン中に登録した StartInterval のジョブが一度も発火しない Mac (M2 で実測) 向け
    """
    if calendar:
        step = max(1, interval // 60)
        minutes = "".join("<dict><key>Minute</key><integer>%d</integer></dict>" % m for m in range(0, 60, step))
        timer = "<key>StartCalendarInterval</key>\n  <array>%s</array>" % minutes
    else:
        timer = "<key>StartInterval</key><integer>%d</integer>" % interval
    out = os.path.join(HOME_DIR, "worker.log")
    os.makedirs(HOME_DIR, exist_ok=True)
    plist = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>%s</string>
  <key>ProgramArguments</key>
  <array><string>%s</string><string>%s</string></array>
  <key>WorkingDirectory</key><string>%s</string>
  %s
  <key>RunAtLoad</key><true/>
  <key>AbandonProcessGroup</key><true/>
  <key>StandardOutPath</key><string>%s</string>
  <key>StandardErrorPath</key><string>%s</string>
</dict>
</plist>
""" % (LABEL, sys.executable, os.path.abspath(__file__), REPO, timer, out, out)
    os.makedirs(os.path.dirname(PLIST), exist_ok=True)
    subprocess.run(["launchctl", "unload", PLIST], capture_output=True)
    with open(PLIST, "w") as f:
        f.write(plist)
    subprocess.run(["launchctl", "load", PLIST], check=True)
    print("installed %s (every %d s%s). log: %s" % (PLIST, interval, ", on the clock" if calendar else "", out))


def uninstall():
    subprocess.run(["launchctl", "unload", PLIST], capture_output=True)
    if os.path.exists(PLIST):
        os.remove(PLIST)
    print("uninstalled %s (jobs already started keep running)" % PLIST)


def main(argv=None):
    p = argparse.ArgumentParser(description="Start autorun jobs placed in the shared folder (for machines "
                                            "the commander cannot reach by ssh). Run every few minutes.")
    p.add_argument("--status-only", action="store_true", help="only write status.json (start nothing)")
    p.add_argument("--install", action="store_true", help="register with launchd (macOS) and exit")
    p.add_argument("--interval", type=int, default=300, help="seconds between runs for --install")
    p.add_argument("--calendar", action="store_true",
                   help="for --install: start at fixed minutes of the hour (StartCalendarInterval) "
                        "instead of StartInterval")
    p.add_argument("--uninstall", action="store_true", help="remove the launchd job and exit")
    args = p.parse_args(argv)
    if args.install:
        settings()                          # 設定が無ければここで止める
        return install(args.interval, args.calendar) or 0
    if args.uninstall:
        return uninstall() or 0
    s = settings()
    errors = AR.load_json(ERRORS, []) if args.status_only else take_jobs(s)
    if not args.status_only:
        try:
            take_standby(s)
        except Exception as e:              # 控えの失敗で、評価ログの写しや status.json を止めない
            log("standby failed: %s" % e)
    copy_eval_logs(s)
    write_status(s, errors)
    return 0


if __name__ == "__main__":
    sys.exit(main())
