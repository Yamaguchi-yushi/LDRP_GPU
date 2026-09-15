import argparse
import csv
import glob
import math
import os
from collections import OrderedDict

import numpy as np

PREFIX = "[RESULT] "
META_KEYS = ("condition", "model_seed", "n_ep")


def parse_log(text):
    """テキストから [RESULT] 行を全部拾って dict のリストにする."""
    recs = []
    for line in text.splitlines():
        if not line.startswith(PREFIX):
            continue
        try:
            recs.append(dict(t.split("=", 1) for t in line[len(PREFIX):].split()))
        except ValueError:
            print(f"  [warn] skipping malformed [RESULT] line: {line[:80]}")
    return recs


def load_runs(root):
    """root 配下の *.txt を読み、条件 -> {seed: rec} にまとめる."""
    conditions = OrderedDict()
    mtimes = {}
    paths = sorted(glob.glob(os.path.join(root, "**", "*.txt"), recursive=True), key=lambda p: os.path.getmtime(p))
    for path in paths:
        mt = os.path.getmtime(path)
        with open(path, "r", errors="replace") as f:
            for rec in parse_log(f.read()):
                cond = rec.get("condition", "(unknown)")
                seed = rec.get("model_seed", "0")
                bucket = conditions.setdefault(cond, OrderedDict())
                key = (cond, seed)
                if seed in bucket:
                    old = mtimes.get(key, 0)
                    if mt < old:
                        continue  # 古い方は無視
                    # 同じ条件 x 同じ seed が 2 回出た = 再実行 or 条件名の衝突.
                    # 黙って 2 seed として数えると std が偽物になるので警告して上書き
                    print(f"  [warn] duplicate entry for {cond} seed{seed}; "
                          f"keeping the last one ({path})")
                bucket[seed] = rec
                mtimes[key] = mt
    return conditions


def metric_names(recs, requested):
    if requested:
        return requested
    names = []
    for r in recs:
        for k in r:
            if k not in META_KEYS and k not in names:
                names.append(k)
    return names


def summarize(values):
    n = len(values)
    if n == 0:
        return None
    arr = np.asarray(values, dtype=float)
    # ddof=1 = 標本標準偏差. 手元の seed は「あり得た学習結果」からの標本なので,
    # 既定の ddof=0 (母標準偏差) だとばらつきを系統的に過小評価する
    std = float(arr.std(ddof=1)) if n >= 2 else float("nan")
    return {"n_seeds": n, "mean": float(arr.mean()), "std": std,
            "sem": std / math.sqrt(n) if n >= 2 else float("nan"),
            "per_seed": [float(v) for v in arr]}


def main():
    ap = argparse.ArgumentParser(
        description="Aggregate [RESULT] lines over seeds and report mean / std / sem.")
    ap.add_argument("--root", default="logs",
                    help="root directory to search for evaluation logs")
    ap.add_argument("--csv", default=None,
                    help="path to write the aggregated results as CSV")
    ap.add_argument("--metrics", default=None,
                    help="comma-separated metric names (default: every metric found)")
    ap.add_argument("--show-seeds", action="store_true",
                    help="also print the per-seed values")
    args = ap.parse_args()

    requested = args.metrics.split(",") if args.metrics else None
    conditions = load_runs(args.root)
    if not conditions:
        print(f"[aggregate] no [RESULT] line found under {args.root}")
        return

    rows = []
    for cond, bucket in conditions.items():
        recs = list(bucket.values())
        seeds = sorted(bucket, key=lambda s: int(s) if s.isdigit() else s)
        print(f"\n=== {cond} ===")
        print(f"  seeds: {seeds}  ({len(recs)} runs)")
        if len(recs) < 2:
            print("  [warn] only one seed available; std cannot be computed")
        n_eps = {r.get("n_ep") for r in recs}
        if len(n_eps) > 1:
            print(f"  [warn] episode count differs across seeds: {sorted(n_eps)}")
        print(f"  {'metric':<28}{'mean':>12}{'std':>12}{'sem':>10}")
        for m in metric_names(recs, requested):
            vals, missing = [], 0
            for r in recs:
                if m in r:
                    try:
                        vals.append(float(r[m]))
                    except ValueError:
                        missing += 1
                else:
                    missing += 1
            if missing:
                print(f"  [warn] '{m}' missing in {missing} seed(s); those runs are excluded")
            st = summarize(vals)
            if st is None:
                continue
            print(f"  {m:<28}{st['mean']:>12.4f}{st['std']:>12.4f}{st['sem']:>10.4f}")
            if args.show_seeds:
                print(f"  {'':<28}per-seed: " + ", ".join(f"{v:.4f}" for v in st["per_seed"]))
            rows.append(OrderedDict([
                ("condition", cond), ("metric", m), ("n_seeds", st["n_seeds"]),
                ("mean", st["mean"]), ("std", st["std"]), ("sem", st["sem"]),
                ("per_seed", ";".join(f"{v:.6g}" for v in st["per_seed"])),
            ]))

    if args.csv and rows:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n[aggregate] CSV written to: {args.csv}")


if __name__ == "__main__":
    main()
