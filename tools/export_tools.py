#!/usr/bin/env python3
"""tools を研究室配布用の LDRP (テンプレート) へ書き出す.

    python tools/export_tools.py ~/src/LDRP-template            # 書き出す
    python tools/export_tools.py ~/src/LDRP-template --dry-run  # 何を書くかだけ表示
    python tools/export_tools.py --check                        # 書き出し対象の検査だけ

配布するのは**コードと汎用の見本だけ**。自分専用のもの (設定・計画表・キャッシュ・
自分用の plist) は持っていかない。README と設定の見本は
tools/dist/ にある配布用のものに差し替える。

**書き出す前に、接続先などが混ざっていないかを検査する。** 見つかったら書き出さずに止まる。
配布先のリポジトリは他の人からも見えるため (2026-09-28 に、手順書へ書いた GPU の
IP アドレスとユーザー名を公開リポジトリへ push していたことが見つかった)。

書き出したあとの commit / push は自分で行う (このスクリプトは git を操作しない)。
"""

import argparse
import getpass
import os
import re
import shutil
import socket
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# (配布先での相対パス, このリポジトリでの元ファイル)
FILES = [
    ("tools/collect_runs.py", "collect_runs.py"),
    ("tools/plan.py", "plan.py"),
    ("tools/eval_report.py", "eval_report.py"),
    ("tools/setup_launchd.py", "setup_launchd.py"),
    ("tools/dashboard/app.py", "dashboard/app.py"),
    ("tools/dashboard/mini.py", "dashboard/mini.py"),
    ("tools/dashboard/static/app.js", "dashboard/static/app.js"),
    ("tools/dashboard/static/style.css", "dashboard/static/style.css"),
    ("tools/dashboard/templates/index.html", "dashboard/templates/index.html"),
    # 配布用に差し替えるもの
    ("tools/README.md", "dist/README.md"),
    ("tools/collect_config.example.yaml", "dist/collect_config.example.yaml"),
    ("tools/plan.example.md", "dist/plan.example.md"),
    # AI エージェント向けの作業ガイド。配布先では tools/CLAUDE.md にする
    # (Claude Code は作業中のディレクトリの CLAUDE.md を自動で読む)。
    # 手元で CLAUDE.md という名前にすると、自分が tools/dist/ を触るたびに
    # 配布用の指示が読み込まれてしまうので、手元では AI_GUIDE.md で持つ
    ("tools/CLAUDE.md", "dist/AI_GUIDE.md"),
]

# 配布先の .gitignore に必ず入れるもの (接続先・計画・収集結果)
GITIGNORE = [
    "# tools (collect_runs.py / dashboard) — 接続先と収集結果は commit しない",
    "tools/collect_config.yaml",
    "tools/plans/",
    "tools/.run_cache.jsonl",
    "tools/.run_cache.jsonl.tmp",
    "tools/.host_seen.json",
    "tools/*.bak*",
    "tools/**/__pycache__/",
    "models_inbox/",
]

# 公開してはいけないものの見分け方
PRIVATE_IP = re.compile(r"\b(?:10\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])|192\.168)\.\d{1,3}\.\d{1,3}\b")
USER_AT_HOST = re.compile(r"\b[a-z_][a-z0-9_.-]{1,31}@(?:\d{1,3}\.){3}\d{1,3}\b")
HOME_PATH = re.compile(r"/(?:Users|home)/([A-Za-z0-9_.-]+)")
TOKEN = re.compile(r"\b(?:secret_|ntn_|ghp_|github_pat_|sk-)[A-Za-z0-9_]{16,}")
# 見本としてわざと書いている名前は許す
HOME_OK = {"USERNAME", "you", "user", "<user>", "name"}


def forbidden_words():
    """このマシン固有で、配布物に出てはいけない語 (自分のユーザー名・ホスト名)."""
    words = set()
    try:
        words.add(getpass.getuser())
    except Exception:
        pass
    host = socket.gethostname().split(".")[0]
    if host:
        words.add(host)
    return {w for w in words if len(w) >= 4}


def scan_text(text):
    """公開してはいけないものを探す。[(行番号, 種類, 該当部分)] を返す."""
    found = []
    words = forbidden_words()
    for n, line in enumerate(text.splitlines(), 1):
        for m in PRIVATE_IP.finditer(line):
            found.append((n, "private IP", m.group(0)))
        for m in USER_AT_HOST.finditer(line):
            found.append((n, "user@IP", m.group(0)))
        for m in HOME_PATH.finditer(line):
            if m.group(1) not in HOME_OK:
                found.append((n, "home path", m.group(0)))
        for m in TOKEN.finditer(line):
            found.append((n, "token", m.group(0)[:12] + "..."))
        low = line.lower()
        for w in words:
            if w.lower() in low:
                found.append((n, "local name", w))
    return found


def check(sources):
    bad = 0
    for dst, src in sources:
        with open(src, encoding="utf-8", errors="replace") as f:
            hits = scan_text(f.read())
        for n, kind, what in hits:
            bad += 1
            sys.stderr.write("  %s:%d  %-11s %s\n" % (os.path.relpath(src, HERE), n, kind, what))
    return bad


def update_gitignore(dest, dry_run):
    path = os.path.join(dest, ".gitignore")
    have = open(path).read().splitlines() if os.path.exists(path) else []
    add = [l for l in GITIGNORE if l not in have]
    if add and not dry_run:
        with open(path, "a") as f:
            f.write(("\n" if have and have[-1].strip() else "") + "\n".join(add) + "\n")
    return add


def main(argv=None):
    p = argparse.ArgumentParser(description="export the LDRP tools into a template LDRP clone")
    p.add_argument("dest", nargs="?", help="root of the template LDRP clone")
    p.add_argument("--dry-run", action="store_true", help="show what would be written")
    p.add_argument("--check", action="store_true", help="only scan the files for private data")
    p.add_argument("--force", action="store_true",
                   help="overwrite files that differ in the destination without asking")
    args = p.parse_args(argv)

    sources = [(dst, os.path.join(HERE, src)) for dst, src in FILES]
    missing = [s for _d, s in sources if not os.path.exists(s)]
    if missing:
        sys.stderr.write("missing source files:\n  " + "\n  ".join(missing) + "\n")
        return 1

    sys.stderr.write("[check] scanning %d files for private data...\n" % len(sources))
    bad = check(sources)
    if bad:
        sys.stderr.write("[check] %d finding(s). Nothing was written. Remove them first "
                         "(the destination repository is visible to others).\n" % bad)
        return 2
    sys.stderr.write("[check] clean\n")
    if args.check:
        return 0
    if not args.dest:
        p.error("dest is required unless --check is given")

    dest = os.path.abspath(os.path.expanduser(args.dest))
    if not os.path.isdir(os.path.join(dest, "src")) or not os.path.exists(os.path.join(dest, "train.py")):
        sys.stderr.write("%s does not look like an LDRP checkout (no src/ or train.py)\n" % dest)
        return 1

    changed = []
    for dst, src in sources:
        out = os.path.join(dest, dst)
        new = open(src, "rb").read()
        if os.path.exists(out) and open(out, "rb").read() == new:
            continue
        changed.append(dst)
        if args.dry_run:
            continue
        if os.path.exists(out) and not args.force:
            # 配布先で誰かが手を入れていたら黙って上書きしない
            sys.stderr.write("[export] %s exists and differs; rerun with --force to overwrite\n" % dst)
            return 1
        os.makedirs(os.path.dirname(out), exist_ok=True)
        shutil.copyfile(src, out)
    added = update_gitignore(dest, args.dry_run)

    verb = "would write" if args.dry_run else "wrote"
    print("[export] %s %d file(s) into %s" % (verb, len(changed), dest))
    for c in changed:
        print("   %s" % c)
    if added:
        print("[export] %s .gitignore entries:" % ("would add" if args.dry_run else "added"))
        for a in added:
            print("   %s" % a)
    if not args.dry_run:
        print("\n次に配布先で確認してから commit / push する:\n"
              "  cd %s && git status && git diff --stat" % dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
