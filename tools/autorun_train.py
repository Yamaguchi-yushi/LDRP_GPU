#!/usr/bin/env python3
"""tools/autorun.py が各マシンで起動する学習の実行役 (train.py と同じことをする).

    cd <学習用リポジトリ> && python ~/.ldrp/autorun/autorun_train.py <job.json>

job.json: {"id": "...", "commands": ["python src/epymarl/src/main.py ...", ...], "max_processes": 1}

train.py と同じく
  - epymarl のコマンドを shell で起動し、max_processes 本ずつ回す
  - ~/.ldrp/batch_<pid>.json に「総本数・開始済み本数・コマンド」を書く
    (ダッシュボードはこれを train.py のバッチとして「実行待ち」に数える。ファイル名が *train.py
     なので ps の検出にも掛かる)
train.py 自体は書き換えずに済むよう、train.py の実行部分だけをここに写してある。
標準ライブラリだけで動く (どのマシンの Python でもよい)。
"""
import json
import os
import subprocess
import sys
import time

SLOT = os.path.expanduser("~/.ldrp/batch_%d.json" % os.getpid())


def publish(total, started, cmd=None):
    try:
        os.makedirs(os.path.dirname(SLOT), exist_ok=True)
        rec = {"pid": os.getpid(), "total": total, "started": started,
               "updated": time.time(), "autorun": True}
        if cmd:
            rec["cmd"] = cmd
        with open(SLOT, "w") as f:
            json.dump(rec, f)
    except OSError:
        pass


def main():
    with open(sys.argv[1]) as f:
        job = json.load(f)
    cmds = job["commands"]
    maxp = int(job.get("max_processes") or 1)
    env = dict(os.environ)
    env.setdefault("MPLBACKEND", "Agg")
    # WSL のマシン (GPU2) は gui_task.py が tkagg を強制するので、DISPLAY が無いと落ちる (go.sh と同じ対処)
    if not env.get("DISPLAY") and os.path.exists("/tmp/.X11-unix/X0"):
        env["DISPLAY"] = ":0"
    publish(len(cmds), 0)
    running = []
    for i, cmd in enumerate(cmds):
        print("[start] %s %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), cmd), flush=True)
        running.append(subprocess.Popen(cmd, shell=True, env=env))
        publish(len(cmds), i + 1, cmd)
        while len(running) >= maxp:
            running = [p for p in running if p.poll() is None]
            time.sleep(5)
    for p in running:
        p.wait()
    print("[finished] %s" % time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    try:
        os.remove(SLOT)
    except OSError:
        pass


if __name__ == "__main__":
    main()
