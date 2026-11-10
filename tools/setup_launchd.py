#!/usr/bin/env python3
"""tools の定期実行を macOS の launchd に登録する (研究室配布版).

    python tools/setup_launchd.py collect           # 集約するマシン: 15 分ごとに全マシンを収集
    python tools/setup_launchd.py quick             # 集約するマシン: 2 分ごとの軽い完了チェック
    python tools/setup_launchd.py mini              # 集約するマシン: ログイン時に進捗パネルを出す
    python tools/setup_launchd.py export --machine laptop
                                                    # SSH が通らないマシン: 15 分ごとに共有フォルダへ書き出す
    python tools/setup_launchd.py collect --dry-run # 登録せず、作る plist を表示するだけ
    python tools/setup_launchd.py remove collect    # 登録を解除する

plist は**このスクリプトを動かした python とリポジトリの場所から作る**ので、
conda のパスやユーザー名を書き換える必要はない。PyYAML が入った python で動かすこと
(収集とダッシュボードが設定ファイルを読むため)。

Linux (GPU など) では launchd が無いので cron を使う。README の「定期実行」を参照。
"""

import argparse
import os
import plistlib
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
AGENTS = os.path.expanduser("~/Library/LaunchAgents")
STATE = os.path.expanduser("~/.ldrp")
LABEL = "com.ldrp.%s"


def jobs(args):
    py = sys.executable
    cr = os.path.join(HERE, "collect_runs.py")
    conf = os.path.join(HERE, "collect_config.yaml")
    cache = os.path.join(HERE, ".run_cache.jsonl")
    home = os.path.expanduser("~")
    return {
        # 全マシンの収集 -> 完了の通知 -> モデル回収 -> 保管 -> 共有フォルダの後始末 -> 要約
        "collect": {
            "args": [py, cr, "-c", conf, "--cache", cache, "--notify",
                     "--fetch-models", os.path.join(home, "models_inbox"),
                     "--publish-models", os.path.join(home, "LDRP_models"),
                     "--purge-drop", "--format", "status",
                     "-o", os.path.join(STATE, "status.txt")],
            "interval": 900, "log": "/tmp/ldrp_collect.log",
        },
        # 実行中の run だけを読み直す軽い確認 (完了をすぐ知るため)
        "quick": {
            "args": [py, cr, "-c", conf, "--cache", cache, "--quick", "--notify"],
            "interval": 120, "log": "/tmp/ldrp_quick.log",
        },
        # デスクトップの進捗パネル。落ちても起こし直す
        "mini": {
            "args": [py, os.path.join(HERE, "dashboard", "mini.py")],
            "keepalive": True, "log": "/tmp/ldrp_mini.log",
        },
        # SSH が通らないマシン: 自分の run と完了モデルを共有フォルダへ書き出す
        "export": {
            "args": [py, cr, "--export", os.path.expanduser(args.drop),
                     "--machine", args.machine or "", "--repo", REPO],
            "interval": 900, "log": "/tmp/ldrp_export.log",
        },
    }


def build(name, job):
    d = {"Label": LABEL % name, "ProgramArguments": job["args"],
         "WorkingDirectory": REPO, "RunAtLoad": True,
         "StandardOutPath": job["log"], "StandardErrorPath": job["log"]}
    if job.get("interval"):
        d["StartInterval"] = job["interval"]
    if job.get("keepalive"):
        d["KeepAlive"] = True
    return d


def launchctl(*a):
    return subprocess.call(["launchctl"] + list(a))


def main(argv=None):
    p = argparse.ArgumentParser(description="register LDRP tools jobs with launchd (macOS)")
    p.add_argument("job", nargs="+", help="collect / quick / mini / export, or: remove <job>")
    p.add_argument("--machine", help="label of this machine (export only; must match hosts: in "
                                     "the collecting machine's collect_config.yaml)")
    p.add_argument("--drop", default="~/Library/Mobile Documents/com~apple~CloudDocs/LDRP_runs",
                   help="shared folder for export (default: iCloud Drive/LDRP_runs)")
    p.add_argument("--dry-run", action="store_true", help="print the plist and do nothing")
    args = p.parse_args(argv)

    if sys.platform != "darwin":
        sys.stderr.write("launchd is macOS only. On Linux use cron (see tools/README.md).\n")
        return 1

    remove = args.job[0] == "remove"
    names = args.job[1:] if remove else args.job
    table = jobs(args)
    for name in names:
        if name not in table:
            sys.stderr.write("unknown job %r (choose from %s)\n" % (name, ", ".join(table)))
            return 1
        path = os.path.join(AGENTS, (LABEL % name) + ".plist")
        if remove:
            launchctl("unload", path)
            if os.path.exists(path):
                os.remove(path)
            print("removed %s" % path)
            continue
        if name == "export" and not args.machine:
            sys.stderr.write("export needs --machine <label> (the label the collecting "
                             "machine uses for this machine)\n")
            return 1
        if name in ("collect", "quick") and not os.path.exists(os.path.join(HERE, "collect_config.yaml")):
            sys.stderr.write("tools/collect_config.yaml not found. Copy the example first:\n"
                             "  cp tools/collect_config.example.yaml tools/collect_config.yaml\n")
            return 1
        plist = build(name, table[name])
        if args.dry_run:
            print("--- %s" % path)
            print(plistlib.dumps(plist).decode("utf-8"))
            continue
        os.makedirs(AGENTS, exist_ok=True)
        os.makedirs(STATE, exist_ok=True)
        if os.path.exists(path):
            launchctl("unload", path)          # 作り直すときは一度外す
        with open(path, "wb") as f:
            plistlib.dump(plist, f)
        launchctl("load", path)
        print("registered %s  (log: %s)" % (path, table[name]["log"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
