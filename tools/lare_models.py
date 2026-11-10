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
    segs = ["%s_%s %gM" % (m.group(1), m.group(2), float(m.group(3)))
            for m in SEG_RE.finditer(name)]
    if not segs:
        return None
    v = VER_RE.search(name)
    a = ALGO_RE.search(name)
    return {"chain": "→".join(segs), "version": int(v.group(1)) if v else 1,
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


def main():
    for f in scan():
        print("%-12s F%d %-5s %-45s %s" % (f["where"], f["version"], f["algo"] or "-",
                                         f["chain"], f["name"]))


if __name__ == "__main__":
    sys.exit(main())
