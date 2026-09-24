#!/usr/bin/env python3
"""実験計画 (tools/plan.md) を読む.

Notion で管理している表を**そのまま貼り付けられる**形式にしてある。
書き直さなくてよいように、見出しと表の構造だけを頼りに解釈する。

    ## 5agent   80M                     <- 見出し: 台数と t_max
    | seed | machine | setting | algorithm | task arrival | task assign | reassign |
    | --- | --- | --- | --- | --- | --- | --- |
    |  |  | safe | MAPPO | bernoulli, mmpp | TP |  |     <- 条件行 (seed が空)
    | 337863318 | GPU1 |  |  |  |  |  |                  <- seed 行
    | 343618845 | GPU1 |  |  |  |  |  |

マップは見出しに書いてもよいし (`## 5agent map_aoba00  80M`)、
`<!-- map: map_aoba00 -->` で以降の既定を決めてもよい。
**書かなければ「どのマップでも一致」**として扱う (推測しない)。
"""

import re
import sys

HEAD_RE = re.compile(
    r"^\s{0,3}#{1,6}\s+(?P<n>\d+)\s*agent\s*(?P<map>[A-Za-z0-9_]*?)\s*"
    r"(?P<t>[\d.]+)\s*M\s*$", re.I)
MAP_HINT_RE = re.compile(r"<!--\s*map:\s*([A-Za-z0-9_]+)\s*-->", re.I)
# 章の前に置かれた裸のマップ名 (メモでは "aoba00" / "8x5" とだけ書かれている)
BARE_MAP_RE = re.compile(r"^\s*(?:map_)?([A-Za-z0-9][A-Za-z0-9_]{0,19})\s*$")
ARROW_RE = re.compile(r"\s*(?:→|->|=>)\s*")
# 区切り行 "| --- | --- |". ハイフンを必須にしないと、空欄だけの行
# "|  |  |  |" まで区切りと誤判定して「未実行の枠」が消える
SEP_RE = re.compile(r"^\|[\s:|-]*-[\s:|-]*\|$")

# "MAT (fixed)" のような手書きの注記を落として algo 名だけにする
ALGO_NOTE_RE = re.compile(r"\s*[（(].*?[)）]\s*")


def norm_map(name):
    """メモの "aoba00" と env の "map_aoba00" を揃える."""
    if not name:
        return None
    n = str(name).strip()
    return n if n.startswith("map_") else "map_" + n


def norm_setting(text):
    """setting の表記ゆれを吸収する.

    メモには "8x5_2 10M 8x5_3 5M" (空白) と "8x5_2 10M→aoba00_2 5M" (矢印) が
    混在している。collect 側は矢印で作るので、両方を空白区切りに寄せて比べる。
    """
    if text is None:
        return None
    t = ARROW_RE.sub(" ", str(text)).strip()
    return " ".join(t.split()) or None


def norm_algo(text):
    a = ALGO_NOTE_RE.sub("", str(text or "")).strip().lower()
    return a.replace("-", "_") or None


def norm_assign(text):
    """表の task assign 列を derive() の task_assign に合わせる.
    TP は既定のタスク割当なので、collect 側では空文字になる。"""
    t = str(text or "").strip().upper()
    if t in ("", "TP", "-"):
        return ""
    return t                                     # "PPO" など


def norm_reassign(text):
    """T/F を bool に。空欄と見出し語 "reassign" は「指定なし」= None."""
    t = str(text or "").strip().upper()
    if t in ("", "-", "REASSIGN"):
        return None
    return t in ("T", "TRUE", "YES", "1")


def _cells(line):
    if not line.startswith("|"):
        return None
    body = line.strip()
    if body.endswith("|"):
        body = body[:-1]
    return [c.strip() for c in body[1:].split("|")]


def parse_plan(path):
    """計画ファイルを読んで条件のリストにする.

    返り値の 1 件:
      {agents, map (None = 任意), t_max_m, setting, algo, task_arrival,
       task_assign, reassign, seeds: [...], want: int, source_line}
    """
    conds = []
    section = None            # (agents, map, t_max_m)
    default_map = None
    cols = None
    cur = None

    with open(path) as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.rstrip("\n")

            hint = MAP_HINT_RE.search(line)
            if hint:
                default_map = norm_map(hint.group(1))
                continue

            h = HEAD_RE.match(line)
            if h:
                section = (int(h.group("n")),
                           norm_map(h.group("map")) or default_map,
                           float(h.group("t")))
                cols, cur = None, None
                continue

            # 章の前に単独で置かれたマップ名 ("aoba00" / "8x5")
            if line.strip() and not line.lstrip().startswith(("|", "#", ">")):
                b = BARE_MAP_RE.match(line)
                if b:
                    default_map = norm_map(b.group(1))
                    section, cols, cur = None, None, None
                continue

            cells = _cells(line)
            if cells is None or section is None:
                continue
            if SEP_RE.match(line.strip()):
                continue
            if cols is None:
                if any(c.lower() == "seed" for c in cells):
                    cols = [c.lower() for c in cells]
                continue

            def get(name):
                try:
                    return cells[cols.index(name)]
                except (ValueError, IndexError):
                    return ""

            seed, machine = get("seed"), get("machine")
            setting, algo = get("setting"), get("algorithm")
            # reassign は列名が無いことがある (メモの 7 列目は見出しが空). その場合は
            # 条件行の最終セルに "reassign" と書いてある列を使う
            reas_raw = get("reassign")
            if not reas_raw and cols and len(cells) > len(
                    [c for c in cols if c]) - 1 and len(cells) == len(cols):
                reas_raw = cells[-1] if not cols[-1] else ""

            if not seed and not machine and (setting or algo):
                cur = {"agents": section[0], "map": section[1],
                       "t_max_m": section[2],
                       "setting": norm_setting(setting),
                       "algo": norm_algo(algo),
                       "task_arrival": get("task arrival") or None,
                       "task_assign": norm_assign(get("task assign")),
                       "reassign": norm_reassign(reas_raw),
                       "seeds": [], "source_line": lineno}
                conds.append(cur)
            elif cur is not None:
                # seed 行。**完全に空の行も「未実行の枠」として数える**
                # (メモでは空行が「あと何本回すか」を表している)。
                # reassign が seed ごとに違う表があるので seed 側にも持たせる
                cur["seeds"].append({"seed": seed or None,
                                     "machine": machine or None,
                                     "reassign": norm_reassign(reas_raw)})

    for c in conds:
        c["want"] = len(c["seeds"]) or 5
    return conds


def cond_key(agents, map_name, t_max_m, algo, setting, arrival, assign, reassign):
    """計画と実績を突き合わせるキー. map が None の計画はマップを問わない."""
    return (agents, map_name, round(float(t_max_m)), algo or None,
            setting or None, arrival or None, assign or "", bool(reassign))


def run_key(d):
    """collect_runs.derive() の結果からキーを作る."""
    return cond_key(d.get("agents"), d.get("map"),
                    round((d.get("t_max") or 0) / 1e6),
                    d.get("algo"), d.get("setting"), d.get("task_arrival"),
                    d.get("task_assign") or "", d.get("reassign"))


def matches(cond, d):
    """計画の 1 条件が run d に一致するか. 計画側が None の項目は問わない."""
    if cond["agents"] != d.get("agents"):
        return False
    if cond["map"] and cond["map"] != d.get("map"):
        return False
    if round(cond["t_max_m"]) != round((d.get("t_max") or 0) / 1e6):
        return False
    if cond["algo"] and cond["algo"] != (d.get("algo") or ""):
        return False
    if cond["setting"] and cond["setting"] != norm_setting(d.get("setting")):
        return False
    if cond["task_arrival"] and cond["task_arrival"] != (d.get("task_arrival") or ""):
        return False
    if cond["task_assign"] != (d.get("task_assign") or ""):
        return False
    # reassign は「指定なし (None)」なら問わない。seed 単位で違う表があるため
    if cond["reassign"] is not None and cond["reassign"] != bool(d.get("reassign")):
        return False
    return True


def label(cond):
    parts = ["%dagent" % cond["agents"]]
    if cond["map"]:
        parts.append(cond["map"])
    parts.append("%gM" % cond["t_max_m"])
    return " ".join(parts)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "tools/plan.md"
    conds = parse_plan(path)
    print("%s: %d 条件" % (path, len(conds)))
    for c in conds:
        seeds = [s["seed"] for s in c["seeds"] if s["seed"]]
        print("  %-26s %-24s %-8s %-16s assign=%-4s reas=%s  seed %d/%d"
              % (label(c), c["setting"], c["algo"], c["task_arrival"],
                 c["task_assign"] or "TP",
                 "-" if c["reassign"] is None else ("T" if c["reassign"] else "F"),
                 len(seeds), c["want"]))
