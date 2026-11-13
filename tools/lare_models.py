#!/usr/bin/env python3
"""LaRe-Path のモデル (.pth) がどこにあるかを調べる.

計画表の「LaRe モデル学習」の条件は、方策の run ではなく
**LaRe のモデルファイルがあるかどうか**で埋まったかを判定する。

ファイル名から次の 3 つを読む (drp_env.py の _lare_build_save_filename の規則):
  - 系列 (chain)  "8x5_2 10M→aoba00_2 5M"  (map_X_Nagents_SM の並び)
  - 因子の版       "_PATH_F2_" があれば 2、無ければ 1
  - アルゴリズム   最初の "<ALGO>_PATH" の ALGO (QMIX など)

    python tools/lare_models.py            # 一覧を表示
"""

import os
import re
import sys

TOOLS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(TOOLS)
# 探す順 = 優先順。models/ が「整理済み」、checkpoints/ はこの Mac の autosave の出力、
# fetched/<マシン>/ は他のマシンから回収したもの (collect_runs.py の fetch_lare_models)
LARE_DIRS = (("models", os.path.join(REPO, "src", "lare", "path", "models")),
             ("checkpoints", os.path.join(REPO, "src", "lare", "path", "checkpoints")),
             ("fetched", os.path.join(REPO, "src", "lare", "path", "fetched")))

SEG_RE = re.compile(r"map_(.+?)_(\d+)agents_([\d.]+)M")
VER_RE = re.compile(r"_PATH_F(\d+)_")
ALGO_RE = re.compile(r"(?:^|_)([A-Z][A-Z0-9]*)_PATH")
ARROW_RE = re.compile(r"\s*(?:→|->|=>)\s*|\s+")


def parse_name(name):
    """ファイル名 -> {chain, version, algo}. LaRe-Path のファイルでなければ None."""
    if not name.endswith(".pth") or "_PATH" not in name:
        return None
    ms = list(SEG_RE.finditer(name))
    segs = ["%s_%s %gM" % (m.group(1), m.group(2), float(m.group(3))) for m in ms]
    if not segs:
        return None
    v = VER_RE.search(name)
    a = ALGO_RE.search(name)
    chain = "→".join(segs)
    # 手で編集したモデル (..._checkpoint_waitedit.pth) は系列の後ろに印を付け、元のモデルと分ける
    # (collect_runs.pretty_lare_models と同じ規則)
    tail = re.sub(r"^_(?:checkpoint|final)", "", re.sub(r"\.pth$", "", name[ms[-1].end():])).strip("_ ")
    if tail:
        chain = "%s %s" % (chain, tail)
    return {"chain": chain, "version": int(v.group(1)) if v else 1,
            "algo": a.group(1).lower() if a else None}


def norm_chain(text):
    """"8x5_2 10M→aoba00_2 5M" と "8x5_2 10M aoba00_2 5M" を同じにする."""
    toks = [t for t in ARROW_RE.split(str(text or "").strip()) if t]
    return " ".join(toks)


def _listing(where, d):
    """(ファイル名, パス, マシン) の列。fetched/ だけは 1 段下のマシン名のフォルダを見る."""
    try:
        names = sorted(os.listdir(d))
    except OSError:
        return []
    if where != "fetched":
        return [(n, os.path.join(d, n), None) for n in names]
    out = []
    for m in names:
        sub = os.path.join(d, m)
        if not os.path.isdir(sub):
            continue
        try:
            out += [(n, os.path.join(sub, n), m) for n in sorted(os.listdir(sub))]
        except OSError:
            continue
    return out


def scan():
    """[{name, where, path, chain, version, algo, mtime, machine}] を返す."""
    out = []
    for where, d in LARE_DIRS:
        for n, p, machine in _listing(where, d):
            info = parse_name(n)
            if not info:
                continue
            try:
                st = os.stat(p)
            except OSError:
                continue
            if st.st_size == 0:
                continue                      # 書きかけ / 壊れたコピー
            info.update(name=n, where=where, path=p, mtime=st.st_mtime, machine=machine)
            out.append(info)
    return out


SETTING_VER_RE = re.compile(r"\s+F(\d+)\s*$")


def expected_of(cond):
    """計画の条件 -> できるはずの LaRe モデルの (chain, version, algo).

    setting は「元にしたモデルの系列 (+ 版)」。scratch は "safe F1" のように系列が無い。
    この run で作られるモデルは、その系列の後ろに「この章のマップ_台数 t_max」が付く。
    """
    setting = str(cond.get("setting") or "")
    m = SETTING_VER_RE.search(setting)
    version = int(m.group(1)) if m else 1
    src = SETTING_VER_RE.sub("", setting).strip()
    if src.lower() in ("safe", "unsafe", ""):
        src = ""
    mp = str(cond.get("map") or "").replace("map_", "", 1)
    stage = "%s_%s %gM" % (mp, cond.get("agents"), float(cond.get("t_max_m") or 0))
    chain = norm_chain((src + " " + stage) if src else stage)
    return chain, version, (cond.get("algo") or None)


def find_for(cond, files=None):
    """条件に合うファイルを優先順 (models/ が先) で返す."""
    files = scan() if files is None else files
    chain, version, algo = expected_of(cond)
    hit = [f for f in files
           if norm_chain(f["chain"]) == chain and f["version"] == version
           and (algo is None or f["algo"] in (None, algo))]
    order = dict((w, i) for i, (w, _d) in enumerate(LARE_DIRS))
    return sorted(hit, key=lambda f: (order.get(f["where"], 9), -f["mtime"]))


MODELS_DIR = LARE_DIRS[0][1]
# 自動で models/ に置かない計画 (追跡用・検討中のもの)
AUTO_INSTALL_SKIP_PLANS = ("watch",)


def auto_install(conds, rows, matches, dry_run=False, log=None):
    """計画内の LaRe モデル学習が完了したら、できたモデルを src/lare/path/models/ にコピーする.

    conds:   計画の条件 (plan.parse_plans の結果)。collect: lare の条件だけ見る
    rows:    run (collect_runs.derive の結果)。条件に一致する done の run があるものだけ置く
             (同じ名前の古いファイルを、学習の終わっていない条件に置かないため)
    matches: plan.matches
    置くのは「その条件が作るモデル」= 系列の最後の段がこの章の t_max のファイルだけ
    (途中の 0.5M ごとのファイルは名前の学習量が違うので当たらない)。
    models/ に同じ名前があれば何もしない (上書きしない)。コピーなので元は残る。
    返り値: 置いたファイル名のリスト
    """
    import shutil
    log = log or (lambda m: sys.stderr.write(m + "\n"))
    files = scan()
    in_models = set(f["name"] for f in files if f["where"] == "models")
    done = [d for d in rows if d.get("state") == "done"]
    placed = []
    for c in conds:
        if c.get("collect") != "lare" or c.get("plan") in AUTO_INSTALL_SKIP_PLANS:
            continue
        hit = find_for(c, files)
        if not hit or hit[0]["where"] == "models":
            continue                              # 無い / もう置いてある
        if not any(matches(c, d) for d in done):
            continue                              # この条件の学習がまだ終わっていない
        f = hit[0]
        if f["name"] in in_models:
            continue
        dst = os.path.join(MODELS_DIR, f["name"])
        log("[lare] install %s (%s%s) -> src/lare/path/models/%s"
            % (f["name"], f["where"], "/" + f["machine"] if f.get("machine") else "",
               " (dry run)" if dry_run else ""))
        if not dry_run:
            shutil.copy2(f["path"], dst)
            os.chmod(dst, 0o644)                  # iCloud 経由のファイルは 600 になっていることがある
            in_models.add(f["name"])
        placed.append(f["name"])
    return placed


CHAIN_SEG_RE = re.compile(r"^\S+_\d+ [\d.]+M$")
WHERE_LABEL = {"models": "✔ models", "checkpoints": "checkpoints/", "fetched": "fetched/"}
NOTE_PC_RE = re.compile(r"[(（]([^)）]+)[)）]")


def _uses_chain(setting):
    """方策学習の行の setting -> (使う LaRe の系列, 版). LaRe を使わない行 (safe など) は None."""
    s = str(setting or "").replace("✅", "").strip()
    m = SETTING_VER_RE.search(s)
    version = int(m.group(1)) if m else 1
    src = SETTING_VER_RE.sub("", s).strip()
    toks = norm_chain(src).split(" ")
    # "8x5_5 10M" の 2 語ずつが系列の段。最後に 1 語余れば手で編集した印 (waitedit など)
    segs = [" ".join(toks[i:i + 2]) for i in range(0, len(toks) - len(toks) % 2, 2)]
    if not segs or not all(CHAIN_SEG_RE.match(x) for x in segs):
        return None
    return norm_chain(src), version


def _row_cells(line):
    """"| a | b |" -> ["a", "b"] (前後の空白を除く)."""
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _set_cell(line, idx, text):
    """表の行の idx 番目のセルを text に置き換える (他のセルの書き方はそのまま)."""
    parts = line.split("|")                 # 先頭と末尾は "|" の外側 (空文字)
    k = idx + 1
    if k >= len(parts) - 1:
        return line
    parts[k] = " %s " % text if text else "  "
    return "|".join(parts)


def update_plan_marks(conds, rows, matches, dry_run=False, log=None):
    """計画表の LaRe モデルの印を、手元のファイルの有無に合わせて書き直す.

    - LaRe モデル学習の行 (collect: lare): 先頭の「LaRe model」列を
        "✔ models (PC)" / "checkpoints/ (PC)" / "fetched/ (PC)" / 空 にする。
        PC は書いてある括弧書きを残し、無ければその条件の完了 run のマシン (fetched はそのマシン)
    - 方策学習の行 (同じ計画ファイルの LaRe を使う行): 使う LaRe モデルが src/lare/path/models/ に
        あれば setting の後ろに ✅、無ければ ✅ を外す
    LaRe モデル学習のグループがある計画ファイルだけを対象にする。中身が変わったファイルだけ書く。
    返り値: 書き換えたファイルのリスト
    """
    log = log or (lambda m: sys.stderr.write(m + "\n"))
    files = scan()
    in_models = [f for f in files if f["where"] == "models"]
    done = [d for d in rows if d.get("state") == "done"]
    by_file = {}
    for c in conds:
        if c.get("plan_file") and c.get("source_line"):
            by_file.setdefault(c["plan_file"], []).append(c)
    changed = []
    for path, cs in by_file.items():
        if not any(c.get("collect") == "lare" for c in cs):
            continue
        with open(path, encoding="utf-8") as f:
            text = f.read()
        lines = text.split("\n")
        for c in cs:
            i = c["source_line"] - 1
            if not (0 <= i < len(lines)) or not lines[i].lstrip().startswith("|"):
                continue
            # この行の表の見出し (列の位置を知るため)
            head = next((_row_cells(lines[k]) for k in range(i - 1, -1, -1)
                         if lines[k].lstrip().startswith("|")
                         and any(x.lower() in ("seed", "lare model") for x in _row_cells(lines[k]))), None)
            if head is None:
                continue
            cols = [x.lower() for x in head]
            cells = _row_cells(lines[i])
            if c.get("collect") == "lare":
                if "lare model" not in cols:
                    continue
                hit = find_for(c, files)
                where = hit[0]["where"] if hit else None
                note = NOTE_PC_RE.search(c.get("lare_note") or "")
                pc = note.group(1) if note else None
                if not pc and where:
                    run = next((d for d in done if matches(c, d)), None)
                    pc = (run or {}).get("machine") or (hit[0].get("machine") if where == "fetched" else None)
                new = WHERE_LABEL[where] + (" (%s)" % pc if pc else "") if where else ""
                idx = cols.index("lare model")
            else:
                if "setting" not in cols:
                    continue
                idx = cols.index("setting")
                cur = cells[idx] if idx < len(cells) else ""
                use = _uses_chain(cur)
                if use is None:
                    continue
                chain, version = use
                ok = any(norm_chain(f["chain"]) == chain and f["version"] == version for f in in_models)
                base = cur.replace("✅", "").strip()
                new = base + " ✅" if ok else base
            if idx < len(cells) and cells[idx] != new:
                lines[i] = _set_cell(lines[i], idx, new)
        out = "\n".join(lines)
        if out != text:
            log("[lare] plan marks updated: %s%s" % (os.path.relpath(path, REPO), " (dry run)" if dry_run else ""))
            if not dry_run:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(out)
            changed.append(path)
    return changed


def main():
    if "--plan-marks" in sys.argv:
        # 計画表の印だけを今すぐ書き直す (--dry-run で表示のみ)。run の情報は収集のキャッシュから
        sys.path.insert(0, TOOLS)
        import plan as PLAN
        import collect_runs as CR
        rows = []
        for r in CR.read_cache(os.path.join(TOOLS, ".run_cache.jsonl")):
            try:
                rows.append(CR.derive(r, 90))
            except Exception:
                continue
        update_plan_marks(PLAN.parse_plans(PLAN.find_plans()), rows, PLAN.matches,
                          dry_run="--dry-run" in sys.argv)
        return 0
    for f in scan():
        print("%-12s F%d %-5s %-45s %s" % (f["where"], f["version"], f["algo"] or "-",
                                         f["chain"], f["name"]))


if __name__ == "__main__":
    sys.exit(main())
