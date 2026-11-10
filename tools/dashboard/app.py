#!/usr/bin/env python3
"""LDRP 実験ダッシュボード (make_graph と同じ Flask + templates/ + static/ 構成).

    python tools/dashboard/app.py
    -> http://127.0.0.1:8765

タブ:
  学習   各マシンの run の条件と進捗・終了予定・seed 充足状況
  評価   results/summary.csv の集計結果

設計の要点:

  - **GET は待たせない**。全ホストへの ssh は 1〜2 分かかる (GPU2 は load 34 まで
    上がる) ので、画面表示はキャッシュ (tools/.run_cache.jsonl) を読むだけにする。
    収集は POST /api/collect か launchd に任せる。
  - 集計・状態判定・条件の分解は collect_runs.py / eval_report.py を import して
    使う。ここには**再実装しない**。
  - flask があれば flask、無ければ標準ライブラリの http.server で動く。
    ルーティングとハンドラは共通なので、どちらでも同じ画面になる。
"""

import argparse
import datetime
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(HERE)
REPO = os.path.dirname(TOOLS)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(TOOLS, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_repo(name):
    """リポジトリ直下のスクリプト (aggregate.py など) を読み込む."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# 表示順。ベースライン -> 提案 の順に並ぶようにする
ALGO_ORDER = ("qmix", "mappo", "mat", "mat_dec", "qplex", "iql", "vdn", "transf_qmix")

# 計画の枠 (5 seed) を埋められる run の状態。failed / stalled / short は
# 「回し直しが要る」ので枠を占めさせない (占めると未実行の数が実態より減る)
SLOT_STATES = ("done", "running")

# 「要確認」に出す状態。collect_runs.py の通知と同じ集合にしておく
# (通知で知らせたものが画面にも残る = 見落としの経路を 1 本にする)
ATTENTION_STATES = ("failed", "stalled", "short")

# OK を押した run を覚えておくファイル。tools/.host_seen.json と同じ置き方
ACK_PATH = os.path.join(TOOLS, ".acked_runs.json")

# 手で「5 seed に数えない」にした run。{uid: 除外した時刻}。
# 報酬の計算式を変えたときなど、config に残らない違いで使えなくなった run を
# 本数・多数決・設定の差分・全体比較のすべてから外す (表示は残して取り消せる)
EXCLUDE_PATH = os.path.join(TOOLS, ".run_excluded.json")
# 除外した run の評価用モデルに付ける印。run.py の model_seeds="auto" は
# <stem>_seed{K}.th にしか一致しないので、これを付ければ評価の対象から外れる
EXCLUDED_SUFFIX = ".excluded"

# seed ごとの手書きメモ。{uid: "本文"}。run が消えない限り残す
NOTE_PATH = os.path.join(TOOLS, ".run_notes.json")
# マシンごとの稼働率 (CPU / メモリ / GPU) の最新値。ダッシュボードを再起動しても
# 直前の値を出せるようにファイルにも置く (中身は collect_runs.host_stats の戻り値)
HOST_STATS_PATH = os.path.join(TOOLS, ".host_stats.json")
# 学習曲線のキャッシュ (collect_runs.fetch_metric_curves が書く)。gitignore
CURVE_CACHE_PATH = os.path.join(TOOLS, ".curve_cache.json")
NOTE_MAX = 300      # 表の 1 列に収まる長さ。超える分は切る

# --- 全体比較 (条件をまたいで設定を比べる) ---------------------------------
# 条件内の多数決 (collect_runs.odd_param_runs) は、ある条件の 5 seed が**揃って**
# 古い設定だと何も言えない。表に載っている run 同士を条件をまたいで比べて、
# その穴を埋める。
#
# 比べないキー = 条件そのものを決めるもの (条件ごとに違って当然):
#   map / 台数 (env.key), t_max, algo (cfg.name), seed, LaRe 設定 (setting 列),
#   タスク到着の方式 (task arrival 列), 割当学習 (task assign 列), 動的台数 (dynamic 列),
#   学習時の再割当 (reassign 列。plan_reassign.md で条件として振っている)
GLOBAL_SKIP_KEYS = frozenset((
    "env.key", "cfg.t_max", "cfg.name", "cfg.env_args", "cfg.seed", "env.seed",
    "env.task_arrival", "env.randomize_task_arrival",
    "env.use_lare_path", "env.use_lare_path_training",
    "env.use_pretrained_lare_path", "env.pretrained_lare_path_model_name",
    "env.use_finetuning_lare_path", "env.finetuning_lare_path_model_name",
    "cfg.train_task_assigner", "env.use_dynamic_agents",
    "env.allow_reassign_before_pickup",
    # 学習時のタスク発生率の範囲。計画表の task arrival 列 (p=下限-上限) で
    # 条件として振っている (plan_arrival_range.md)。MMPP の 2 相も範囲の両端にそろえる。
    # 条件の中で割れたときの params✗ は今までどおり出る
    "env.rand_p_min", "env.rand_p_max", "env.task_p_low", "env.task_p_high",
    # LaRe の評価関数の版。計画表の setting 列 (F1 / F2 / F3) で条件として振っている
    "env.lare_path_factor_version",
))
GLOBAL_MIN_GROUP = 3    # 比べる相手がこれ未満なら多数派を決めない
GLOBAL_MAJORITY = 0.5   # 多数派とみなす割合 (これ未満 = 意図的に振っているキー)
# 「この差は問題ない」と判断したものを覚えておくファイル
DISMISS_PATH = os.path.join(TOOLS, ".param_dismissed.json")


def _norm(v):
    return json.dumps(v, sort_keys=True, default=str)


def _raw_param(d, k):
    """表示用に生の値を引く.

    param_fields は False / None を「キーが無い」に潰す (ハッシュを安定させるため)。
    判定はそちらに合わせるが、表示で null と出すと use_rnn=False が「記録なし」に
    見えてしまうので、画面には config に書かれていた値そのものを出す。
    """
    pre, key = k.split(".", 1)
    src = (d.get("cfg") or {}) if pre == "cfg" else (d.get("env") or {})
    # 報酬を渡していない run は drp_env の既定値で学習している (「記録なし」ではない)
    if k == "env.reward_list" and src.get(key) is None:
        return CR.DEFAULT_REWARD_LIST
    return src.get(key)


def _pdiff_val(key, v):
    """条件内の差分表に出す値. "-" は画面で「キー無し (古い版)」と出る."""
    if v is None:
        # 報酬は既定値のとき param_fields がキーを落とす。古い版ではなく既定値
        return str(CR.DEFAULT_REWARD_LIST) if key == "env.reward_list" else "-"
    return str(v)


def dismiss_key(cond, key, actual):
    """確認済みにする鍵. 値まで入れておけば、別の値に変わったとき再び出る."""
    return "%s|%s|%s" % (cond, key, _norm(actual))


def global_param_diff(rows):
    """表に載っている run 同士で、条件をまたいで設定を比べる.

    **同じ algo・同じ dynamic 設定の run はハイパーパラメータと env 設定が揃って
    いるはず**、という前提でキーごとに多数派の値を決め、そこから外れている条件を返す。
    algo をまたがないのは、mappo と qmix で lr が違うのは設計どおりだから。
    dynamic で分けるのは、動的台数の run にしか無いキー (min_active_agents など) が
    あるから。

    返り値: {condition: [{key, expected, actual, n_this, n_major, n_group}, ...]}
    """
    from collections import Counter, defaultdict
    groups = defaultdict(list)
    for d in rows:
        if d.get("state") in ("done", "running") and d.get("cfg"):
            groups[(d.get("algo"), bool(d.get("dynamic_agents")))].append(d)
    out = defaultdict(list)
    for ds in groups.values():
        if len(ds) < GLOBAL_MIN_GROUP:
            continue
        # dropped: その run では使われないとして param_fields が外したキー
        # (fixed の run の mmpp_ratio など)。そのキーの比較からはその run を除く。
        # 除かないと「無い」が 1 つの値として多数派 (0.5 など) と違って見える
        fields, dropped = [], []
        for d in ds:
            dr = set()
            fields.append(CR.param_fields(d.get("cfg"), d.get("env"), dropped=dr))
            dropped.append(dr)
        keys = set().union(*fields) - GLOBAL_SKIP_KEYS - set(CR.HASH_IGNORE_FIELDS)
        for k in sorted(keys):
            use = [(d, f) for d, f, dr in zip(ds, fields, dropped) if k not in dr]
            if len(use) < GLOBAL_MIN_GROUP:
                continue
            vals = [_norm(f.get(k)) for _d, f in use]
            count = Counter(vals)
            if len(count) < 2:
                continue
            major, n_major = count.most_common(1)[0]
            if n_major < len(use) * GLOBAL_MAJORITY:
                continue
            ref = use[vals.index(major)][0]
            off = defaultdict(list)
            for (d, _f), v in zip(use, vals):
                if v != major:
                    off[(d["condition"], v)].append(d)
            for (cond, _v), xs in off.items():
                out[cond].append({
                    "key": k,
                    "expected": _raw_param(ref, k),
                    "actual": _raw_param(xs[0], k),
                    "n_this": len(xs), "n_major": n_major, "n_group": len(use),
                })
    return out


def ack_key(r):
    """確認済みかどうかを見分ける鍵.

    uid だけにすると、同じ run を回し直して**また落ちた**ときに前回の OK が
    効いたままになり、二度目の失敗に気づけない。停止時刻まで入れておけば
    新しい失敗は別物として必ず出てくる。
    """
    return "%s|%s|%s" % (r.get("uid"), r.get("state"),
                         r.get("stop_at") or r.get("last_seen") or "")


def load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (IOError, OSError, ValueError):
        return {}


def save_json(obj, path):
    tmp = path + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(obj, f, ensure_ascii=False, indent=1, sort_keys=True)
        os.replace(tmp, path)
    except (IOError, OSError) as e:
        sys.stderr.write("[warn] could not save %s: %s\n" % (path, e))


def load_acks(path=ACK_PATH):
    return load_json(path)


def save_acks(acks, path=ACK_PATH):
    save_json(acks, path)


def _rank(seq, v):
    try:
        return seq.index(v)
    except ValueError:
        return len(seq)


CR = _load("collect_runs")
ER = _load("eval_report")
PLAN = _load("plan")
LM = _load("lare_models")        # LaRe モデル (.pth) の置き場所を調べる
AR = _load("autorun")            # 依頼された学習・評価の自動実行 (依頼の追加・取り消しだけここから)


# ---------------------------------------------------------------------------
# データ
# ---------------------------------------------------------------------------

class State(object):
    """収集結果を持ち回す。GET はここを読むだけにする."""

    def _auto_collect_loop(self):
        """--collect-every 分ごとに裏で収集する.

        画面の再描画 (60 秒) はキャッシュを読み直すだけなので、これが無いと
        開きっぱなしでも中身は起動時のまま古くなる。
        """
        n = max(1, int(self.args.collect_every))
        while True:
            # 繋がらないホストがあるときは 2 分後に試し直す (スリープ明け・VPN の再接続待ちで
            # 一時的に失敗しただけなら、15 分も「確認できません」のままにしない)
            time.sleep(min(n * 60, 120) if self.unreachable else n * 60)
            if not self.busy:
                self.collect()

    def __init__(self, args):
        self.args = args
        self.lock = threading.RLock()
        self.busy = False
        self.collected_at = None
        self.errors = []
        self.batches = []          # train.py のバッチ。キャッシュには入れない
        # SSH で繋がらなかったホスト {label: {"since": 最初に失敗した時刻, "error": 理由}}。
        # 「異常」ではなく「確認できない」として画面に出す (その間の表示は最後に見えた時点のもの)
        self.unreachable = {}
        # 共有フォルダの runs.jsonl の mtime。変わっていなければ読み飛ばす
        self.drop_mtime = {}

    # --- 設定 -----------------------------------------------------------
    def conf(self):
        path = os.path.expanduser(self.args.config)
        conf = CR.load_config(path) if os.path.exists(path) else {}
        CR.set_eval_naming(conf.get("eval_naming"))    # 評価用モデル名の形式
        return conf

    # --- 計画 -----------------------------------------------------------
    def plan_files(self):
        """読む計画ファイルの一覧.

        --plan が明示されていればそれ、無ければ collect_config.yaml の plans:、
        それも無ければ tools/plans/*.md → tools/plan.md の順に探す。
        """
        spec = self.args.plan
        if not spec:
            conf = CR.load_config(os.path.expanduser(self.args.config)) or {}
            spec = conf.get("plans")
        return PLAN.find_plans(spec)

    def pending_by_cond(self, conds):
        """train.py がこれから回す予定を、実験計画の条件ごとに数える.

        sacred は run が **始まって初めて** ディレクトリを作るので、5 本連続実行の
        2 本目を回している時点では 1 本しか見えない。マシンが埋まっているのに
        表は「未実行 4」のままになり、二重に投入してしまう。
        train.py が書いた予約 (`~/.ldrp/batch_<pid>.json` の cmd) を条件に
        対応づけて、残り (total - started) を「実行待ち」として枠に出す。

        条件の判定は実績の run と **同じ derive() 1 か所** で行う。
        コマンド行のキー名は config.json と同じなので、そのまま通せる。
        """
        out = {}
        conf = self.conf()
        for b in (self.batches or []):
            total, started = b.get("total"), b.get("started")
            cmd = b.get("train_cmd")
            if not cmd or total is None or started is None:
                continue                  # 予約を書いていない = 本数が分からない
            left = int(total) - int(started)
            if left <= 0:
                continue
            rec = CR.parse_train_command(cmd)
            rec.update({"uid": b.get("uid"), "machine": b.get("machine")})
            try:
                d = CR.derive(rec, conf.get("stale_minutes") or 90,
                              conf.get("method_tag_by_lare_mode"),
                              conf.get("expected_t_max"),
                              lare_chain=conf.get("lare_chain"))
            except Exception:             # 解析できない予約で画面を壊さない
                continue
            for i, c in enumerate(conds):
                if PLAN.matches(c, d):
                    e = out.setdefault(i, {"n": 0, "machines": []})
                    e["n"] += left
                    if b.get("machine") not in e["machines"]:
                        e["machines"].append(b.get("machine"))
                    break
        return out

    def plan_view(self, rows, shaped, excl=None):
        """**表は計画から作る**。実績はそこに埋めていく.

        計画に無い run は conditions 表には出さない (実行中セクションには出る)。
        こうすると「あと何を回せばよいか」が表そのものになる。

        slots に入れる run は **dashboard_data() が整形したもの** (shaped)。
        derive() の生レコードを入れると odd_params などが欠けて画面側が壊れる。
        """
        paths = self.plan_files()
        if not paths:
            return [], set()
        try:
            conds = PLAN.parse_plans(paths)
        except Exception as e:
            sys.stderr.write("[plan] %s: %s\n" % (type(e).__name__, e))
            return [], set()

        excl = excl or {}
        used = set()
        out = []
        # パラメータが多数派とずれている run。印を付ける判断に使う
        odd_uids = set(d["uid"] for d in CR.odd_param_runs(rows))
        pending = self.pending_by_cond(conds)
        # 「LaRe モデル学習」の条件 (collect: lare) はモデルファイルの有無で判定する。
        # 置き場所の一覧は 1 回だけ読む
        lare_files = LM.scan() if any(c.get("collect") == "lare" for c in conds) else []
        # seed -> その seed を表に明記している条件の添字。別の計画 (例: AAMAS) の表に
        # seed が書いてある run は、その計画のもの。1 seed だけ借りている計画
        # (例: plan_reassign.md) の枠の下に「余分な完了 run *」として並べない
        written = {}
        for ci, c in enumerate(conds):
            for sl in c["seeds"]:
                if sl.get("seed"):
                    written.setdefault(str(sl["seed"]), []).append(ci)
        for ci, c in enumerate(conds):
            hit = [d for d in rows if PLAN.matches(c, d)]
            by_seed = {}
            for d in hit:
                by_seed.setdefault(str(d.get("seed")), []).append(d)

            slots = []
            blanks = []                       # seed 欄が空のままの枠 (添字)
            for sl in c["seeds"]:
                sd = str(sl["seed"]) if sl["seed"] else None
                run = None
                if sd and by_seed.get(sd):
                    # 除外した run では枠を埋めない。同じ seed で回し直した run が
                    # あればそちらを入れ、無ければ枠は空 (= 未実行) のまま。
                    # 除外した run は by_seed に残るので、下の leftover で枠の下に出る
                    cands = by_seed[sd]
                    d = next((x for x in cands if x["uid"] not in excl), None)
                    if d is not None:
                        cands.remove(d)
                        used.add(d["uid"])
                        run = shaped.get(d["uid"])
                slots.append({"seed": sd, "machine": sl.get("machine"),
                              "reassign": sl.get("reassign"), "run": run})
                if sd is None:
                    blanks.append(len(slots) - 1)

            # 計画に seed が書かれていない run。**まず空いている枠を埋める**。
            # 枠と別に並べると 5 seed の条件が「未実行 5 行 + 実績 2 行」の 7 行に
            # なり、あと何本回せばよいのかが表から読めなくなる (実測で 84 条件中
            # 46 条件が seed 欄を空けたまま自動検出に任せている)。
            # 枠より多い分だけを下に足す。
            leftover = sorted((d for ds in by_seed.values() for d in ds),
                              key=lambda d: (str(d.get("state") not in SLOT_STATES),
                                             str(d.get("start_time") or ""),
                                             str(d.get("seed"))))
            # 印 (*) を付けるのは **不具合が疑わしいときだけ**。状態 / params✗ /
            # t_max は別の欄に既に出ているので、ここで黄色にするのは「表に 5 seed
            # 書いてあるのに、さらに別の seed が回っている」場合 (取り違えか二重実行)。
            full = sum(1 for sl in c["seeds"] if sl.get("seed")) >= c["want"]
            planned = set(str(sl["seed"]) for sl in c["seeds"] if sl.get("seed"))
            extra = []
            for d in leftover:
                used.add(d["uid"])
                if full and any(cj != ci and PLAN.matches(conds[cj], d)
                                for cj in written.get(str(d.get("seed")), [])):
                    continue              # 別の計画の表に載っている run
                gone = d["uid"] in excl
                row = {"seed": str(d.get("seed")), "machine": d.get("machine"),
                       "run": shaped.get(d["uid"]),
                       # 表に無い seed = 補足情報。除外した run は計画の seed のまま
                       # 下に来るので、実際に計画に無いかで判定する
                       "unplanned_seed": str(d.get("seed")) not in planned,
                       # 除外済みは「別の seed も完了している」の疑いに数えない
                       "suspect": bool(full and d.get("state") == "done" and not gone)}
                # 失敗・停止した run は枠を埋めない。埋めてしまうと 5 行のうち
                # 1 行が失敗で占められ、「あと何本回せばよいか」が読めなくなる。
                # 消さずに下へ出す (どの seed が落ちたかは見たいため)
                if blanks and d.get("state") in SLOT_STATES and not gone:
                    i = blanks.pop(0)
                    row["reassign"] = slots[i].get("reassign")
                    slots[i] = row
                else:
                    extra.append(row)

            # train.py がこれから回す分を「実行待ち」として置く。**run が入って
            # いない枠すべて**が対象 (seed が明記された枠も含む)。train.py は
            # seed を自分で引くので予約とその seed 番号は結び付かないが、
            # 埋めたいのは「この条件はあと何本要るか」なので枠の別は問わない。
            # 枠より多く予約されていても表は 5 行のまま。
            p = pending.get(ci)
            if p:
                empty = [i for i, s in enumerate(slots) if not s.get("run")]
                for i in empty[:p["n"]]:
                    slots[i]["pending"] = True
                    slots[i]["pending_on"] = "/".join(x for x in p["machines"] if x)
            # 同じ条件のはずなのにパラメータが割れていたら、**どのキーが違うか**を渡す。
            # ハッシュだけでは何を直せばよいか分からない
            pdiff, phashes = [], {}
            # 比べるのは **5 seed に数える run** (完了・実行中で、除外していないもの) だけ。
            # 失敗・停止した run は数えないので、その設定が違っていても直す必要がない
            # (odd_param_runs / 全体比較と同じ対象にそろえる)
            hit_live = [d for d in hit if d["uid"] not in excl
                        and d.get("state") in SLOT_STATES]
            if len(set(d.get("param_hash") for d in hit_live)) > 1:
                diffs, seeds = CR.param_diff(hit_live)
                order = sorted(seeds, key=lambda h: (-len(seeds[h]), str(h)))
                phashes = [{"hash": h, "seeds": sorted(seeds[h])} for h in order]
                pdiff = [{"key": k,
                          "vals": [_pdiff_val(k, v.get(h)) for h in order]}
                         for k, v in diffs]

            out.append({
                "param_diff": pdiff, "param_hashes": phashes,
                "plan": c.get("plan"), "plan_file": c.get("plan_file"),
                "group": c.get("group"), "group_order": c.get("group_order") or 0,
                "collect": c.get("collect"), "mark": bool(c.get("mark")),
                "lare_note": c.get("lare_note"),
                "lare_files": ([{"where": f["where"], "name": f["name"],
                                 "machine": f.get("machine")}
                                for f in LM.find_for(c, lare_files)]
                               if c.get("collect") == "lare" else None),
                "label": PLAN.label(c), "map": c["map"], "agents": c["agents"],
                "t_max_m": c["t_max_m"], "setting": c["setting"],
                "algo": c["algo"], "task_arrival": c["task_arrival"],
                "task_assign": c["task_assign"], "reassign": c["reassign"],
                "dynamic": c.get("dynamic"),
                "want": c["want"], "slots": slots + extra,
            })
        # 見出しは マップ -> 台数 -> t_max。
        # 表の中は**列の左から順に** setting -> algorithm -> arrival -> assign -> dynamic。
        # 列内の並びは意味のある順にする (safe が基準、QMIX->MAPPO->MAT、TP->PPO、F->T)
        out.sort(key=lambda c: (
            str(c.get("plan") or ""),
            c.get("group_order") or 0,          # "# ..." のグループは書いた順
            str(c["map"]), c["agents"], c["t_max_m"],
            0 if c["setting"] == "safe" else 1, str(c["setting"]),
            _rank(ALGO_ORDER, c["algo"]), str(c["algo"] or ""),
            str(c["task_arrival"] or ""),
            0 if not c["task_assign"] else 1, str(c["task_assign"] or ""),
            1 if c.get("reassign") else 0,      # 再割当なし (F) を先に
            1 if c.get("dynamic") else 0,
        ))
        return out, used

    # --- 学習 -----------------------------------------------------------
    def train(self):
        conf = self.conf()
        raw = CR.read_cache(os.path.expanduser(self.args.cache))
        raw = [r for r in raw if r.get("kind") not in ("batch", "host")]
        stale = conf.get("stale_minutes") or 90
        rows = [CR.derive(r, stale, conf.get("method_tag_by_lare_mode"),
                          conf.get("expected_t_max"),
                          lare_chain=conf.get("lare_chain"))
                for r in CR.dedupe(raw)]
        min_steps = conf.get("min_steps", 1e6)
        if min_steps:
            # 短い run (動作確認用の t_max=数万) は表を埋めるので既定で落とす。
            # ただし **走っている間は必ず見せる**。報酬設計を変えた直後の試し
            # 実行のように、計画に無く t_max も小さい run こそ進捗を見たい。
            # 終われば消えるが、そのときは計画外の件数として数えられる
            rows = [d for d in rows
                    if (d.get("t_max") or 0) >= min_steps
                    or d.get("state") == "running"]
        with self.lock:
            data = CR.dashboard_data(
                rows, self.batches, self.errors,
                saved=CR.load_saved_models(self.args.models_repo),
                cadence=CR.record_host_seen(rows, hosts=self.conf().get("hosts")))
            # 除外した run は**判定に一切使わない**。多数決に残すと、たとえば旧報酬の
            # 3 本を外しても、回し直した新しい 2 本のほうが少数派として params✗ になる。
            # odd_params は dashboard_data() が全 run で決めているので、ここで決め直す
            excl = load_json(EXCLUDE_PATH)
            live = [d for d in rows if d["uid"] not in excl]
            odd = set(d["uid"] for d in CR.odd_param_runs(live))
            for r in data["runs"]:
                r["excluded"] = r["uid"] in excl
                r["excluded_at"] = excl.get(r["uid"])
                r["odd_params"] = r["uid"] in odd
            shaped = dict((r["uid"], r) for r in data["runs"])
            plan, used = self.plan_view(rows, shaped, excl)
            # run ごとの「どの計画の枠に入っているか」(running now に出す)。
            # 1 つの run が複数の計画に数えられることがある (AAMAS と reassign 等)
            plans_of = {}
            # 方策モデルを保管する枠に入っている run。「LaRe モデル学習」の枠
            # (collect: lare) は LaRe の .pth が目当てで方策は保管しないので、
            # machine 表の「モデル回収」の分母に入れない (入れると永久に満たない)
            policy_uids = set()
            for c in plan:
                for sl in c.get("slots") or []:
                    u = (sl.get("run") or {}).get("uid")
                    if u and c.get("collect") != "lare":
                        policy_uids.add(u)
                    if u and c.get("plan") and c["plan"] not in plans_of.setdefault(u, []):
                        plans_of[u].append(c["plan"])
            for r in data["runs"]:
                r["in_plan"] = r["uid"] in used
                r["policy_plan"] = r["uid"] in policy_uids
                r["plans"] = plans_of.get(r["uid"], [])
            self.attach_global_diff(plan, [d for d in live if d["uid"] in used])
            # 自動実行 (tools/autorun.py) の依頼で条件を指すための id
            for c in plan:
                try:
                    c["ar_id"] = AR.cond_id(c)
                except Exception:
                    pass
            data["plan"] = plan
            data["plan_files"] = self.plan_files()
            data["plan_file"] = ", ".join(data["plan_files"])
            data["collected_at"] = self.collected_at
            data["busy"] = self.busy
            data["unreachable"] = dict(self.unreachable)
            data["cache"] = os.path.expanduser(self.args.cache)
            notes = load_json(NOTE_PATH)
            for r in data["runs"]:
                r["note"] = notes.get(r["uid"], "")
            data["attention"] = self.attention(data["runs"])
            data["host_stats"] = load_json(HOST_STATS_PATH)
        return data

    # --- 全体比較 -------------------------------------------------------
    def attach_global_diff(self, plan, rows):
        """計画の各条件に、全体と違う設定 (global_diff) を付ける.

        比べる相手は**表に載っている run だけ** (計画外の試し実行や古い探索を
        混ぜると多数派がぶれる)。確認済みにしたもの (DISMISS_PATH) は外し、
        件数だけ n_dismissed として残す (取り消せるように)。
        """
        gdiff = global_param_diff(rows)
        dismissed = load_json(DISMISS_PATH)
        for c in plan:
            conds = set(s["run"]["condition"] for s in (c.get("slots") or [])
                        if s.get("run") and s["run"].get("condition"))
            live, n_dis = [], 0
            for cond in sorted(conds):
                for x in gdiff.get(cond, []):
                    x = dict(x, cond=cond,
                             dismiss_key=dismiss_key(cond, x["key"], x["actual"]))
                    if x["dismiss_key"] in dismissed:
                        n_dis += 1
                    else:
                        live.append(x)
            c["global_diff"] = live
            c["global_dismissed"] = n_dis
            c["global_conds"] = sorted(conds)     # 取り消し用 (確認済みの鍵の頭)

    def dismiss_diff(self, body):
        """全体との差を「問題なし」にする. body: {"key": dismiss_key}.

        {"undo": true} を付けると取り消す (この条件の確認済みを全部戻す)。
        """
        body = body or {}
        dismissed = load_json(DISMISS_PATH)
        if body.get("undo"):
            conds = body.get("conds") or [body.get("cond", "")]
            prefixes = tuple("%s|" % c for c in conds if c)
            keys = [k for k in dismissed if prefixes and k.startswith(prefixes)]
            for k in keys:
                dismissed.pop(k, None)
            save_json(dismissed, DISMISS_PATH)
            return {"undone": len(keys)}
        k = body.get("key")
        if not k:
            return {"ok": False, "error": "key is required"}
        dismissed[k] = CR.now_utc().isoformat()
        save_json(dismissed, DISMISS_PATH)
        return {"ok": True}

    # --- 手動除外 -------------------------------------------------------
    def exclude(self, body):
        """run を「5 seed に数えない」にする / 戻す.

        body: {"uid": ..., "on": true}  (false で戻す)
        """
        body = body or {}
        uid = body.get("uid")
        if not uid:
            return {"ok": False, "error": "uid is required"}
        on = bool(body.get("on", True))
        excl = load_json(EXCLUDE_PATH)
        if on:
            excl[uid] = CR.now_utc().isoformat()
        else:
            excl.pop(uid, None)
        save_json(excl, EXCLUDE_PATH)
        moved, errors = self._move_eval_models(uid, exclude=on)
        return {"ok": True, "uid": uid, "excluded": uid in excl,
                "moved": moved, "move_errors": errors}

    def _eval_names(self, uid):
        """この run が評価用フォルダに置かれたときのファイル名 (保管リポジトリの manifest から)."""
        path = os.path.join(os.path.expanduser(self.args.models_repo), "manifest.jsonl")
        names = []
        try:
            with open(path, "r") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    if r.get("uid") == uid and r.get("eval_name"):
                        names.append(r["eval_name"])
        except (IOError, OSError):
            pass
        return names

    def _move_eval_models(self, uid, exclude):
        """評価用フォルダのモデルに .excluded を付けて評価の対象から外す / 戻す.

        消さずに改名するだけなので「戻す」で元どおりになる。経路方策と割当方策は
        **同じ名前で別フォルダ**に置かれている (collect_runs.install_models)。
        戻す先に同じ名前のファイルが既にある場合は上書きしない。
        """
        moved, errors = [], []
        for name in self._eval_names(uid):
            for kind, rel in sorted(CR.EVAL_MODEL_DIRS.items()):
                d = os.path.join(REPO, rel)
                live = os.path.join(d, name)
                off = live + EXCLUDED_SUFFIX
                src, dst = (live, off) if exclude else (off, live)
                if not os.path.exists(src):
                    continue
                if os.path.exists(dst):
                    # 除外で空いた番号は回し直した run に引き継がれる
                    # (collect_runs.load_seed_index)。その後で「戻す」と同じ名前が
                    # 2 つになるので、上書きせずに理由を知らせる
                    if not exclude:
                        msg = ("%s: この番号は回し直した run が使っているので、"
                               "モデルは戻していません" % name)
                        if msg not in errors:
                            errors.append(msg)
                    continue
                try:
                    os.replace(src, dst)
                    moved.append("%s/%s" % (kind, os.path.basename(dst)))
                except OSError as e:
                    errors.append("%s: %s" % (name, e))
        return moved, errors

    # --- seed ごとのメモ -------------------------------------------------
    def note(self, body):
        """conditions 表に手で書いたコメントを保存する.

        body: {"uid": ..., "text": ...}。空文字なら削除する
        (空のキーを残すとファイルが使い回すたびに膨らむだけ)
        """
        body = body or {}
        uid = body.get("uid")
        if not uid:
            return {"ok": False, "error": "uid is required"}
        text = str(body.get("text") or "").strip()[:NOTE_MAX]
        notes = load_json(NOTE_PATH)
        if text:
            notes[uid] = text
        else:
            notes.pop(uid, None)
        save_json(notes, NOTE_PATH)
        return {"ok": True, "uid": uid, "text": text}

    # --- 要確認リスト ---------------------------------------------------
    def attention(self, runs):
        """異常終了した run のうち、まだ OK を押していないものを返す.

        OK 済みでも run 自体は conditions / counts にそのまま残る。ここで
        消えるのは「気づいてほしい」という呼びかけだけ。
        """
        acks = load_acks()
        out = []
        for r in runs:
            if r.get("state") not in ATTENTION_STATES:
                continue
            key = ack_key(r)
            if key in acks:
                continue
            out.append(dict(r, ack_key=key))
        # 新しく落ちたものほど上。時刻が無いものは末尾へ
        out.sort(key=lambda r: r.get("stop_at") or r.get("last_seen") or "",
                 reverse=True)
        return out

    def ack(self, body):
        """OK が押された run を確認済みにする.

        body: {"keys": [ack_key, ...]}  /  {"all": true} で表示中の全部
        """
        body = body or {}
        keys = list(body.get("keys") or [])
        if body.get("all"):
            keys += [r["ack_key"] for r in self.train()["attention"]]
        if not keys:
            return {"acked": 0}
        acks = load_acks()
        now = CR.now_utc().isoformat()
        for k in keys:
            acks[k] = now
        save_acks(acks)
        return {"acked": len(keys)}

    # --- 評価 -----------------------------------------------------------
    def eval(self):
        path = os.path.expanduser(self.args.summary)
        if not os.path.exists(path):
            return {"source": path, "available": False, "metrics": [],
                    "conditions": [],
                    "error": "summary.csv がありません (aggregate.py --csv を実行してください)"}
        rows, metrics = ER.load_summary(path)
        self._attach_seed_plans(rows, metrics)
        return {"source": path, "available": True, "metrics": metrics,
                "conditions": sorted(rows, key=ER.sort_key)}

    def _attach_seed_plans(self, rows, metrics):
        """条件ごとに、seed ごとの値と「そのモデルがどの計画のものか」を付ける.

        summary.csv は seed をまとめた平均しか持たない。計画 (tools/plans/*.md) ごとに
        見るには、どの seed がどの計画の run かが要る。評価ログの [RESULT] 行の
        model= (評価用のファイル名) を、保管リポジトリの manifest.jsonl の
        eval_name -> plans で引く。画面側でその計画の seed だけで平均を取り直す。
        (例: reassign 計画は seed0 だけ、arrival_range は範囲を変えた run の seed)
        """
        plans_of = {}
        man = os.path.join(os.path.expanduser(self.args.models_repo), "manifest.jsonl")
        try:
            with open(man, "r") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    if r.get("eval_name"):
                        plans_of.setdefault(r["eval_name"], set()).update(r.get("plans") or [])
        except (IOError, OSError):
            return
        try:
            AG = _load_repo("aggregate")
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()):   # 集計のログは画面に要らない
                runs = AG.load_runs(os.path.join(REPO, "logs"))
        except Exception as e:                  # 読めなくても評価タブ自体は出す
            sys.stderr.write("[eval] seed plans: %s: %s\n" % (type(e).__name__, e))
            return
        for d in rows:
            seeds = []
            for seed, rec in (runs.get(d["condition"]) or {}).items():
                vals = {}
                for m in metrics:
                    try:
                        vals[m] = float(rec[m])
                    except (KeyError, ValueError):
                        pass
                seeds.append({"seed": seed, "model": rec.get("model"),
                              "plans": sorted(plans_of.get(rec.get("model"), ())),
                              "vals": vals})
            d["seeds"] = seeds

    # --- 収集 (重い。バックグラウンドで回す) -----------------------------
    def collect(self):
        with self.lock:
            if self.busy:
                return False
            self.busy = True
        threading.Thread(target=self._collect_worker, daemon=True).start()
        return True

    def _collect_worker(self):
        try:
            conf = self.conf()
            hosts = conf.get("hosts")
            if not hosts:
                # 設定が読めていないのに os.uname()[1] を使うと、白の run が
                # ホスト名ラベルで二重に溜まる (uid にマシン名が入るので
                # dedupe では消えない)。収集せずに知らせて止める
                msg = ("collect_config.yaml から hosts を読めませんでした"
                       " (PyYAML が無い python で起動していませんか)")
                sys.stderr.write("[collect] %s\n" % msg)
                with self.lock:
                    self.errors = [("config", msg)]
                return
            # collect_runs の「失敗した ssh ホストは 10 分試さない」は、1 回の実行で何度も
            # 呼ぶ CLI 用。ダッシュボードは 15 分おきにしか収集しないので毎回ちゃんと試す。
            # 残しておくと、スリープ明けなどに 1 回失敗しただけで次の収集が "skipped" になり、
            # 繋がっているのに「確認できません」が続き、本当の失敗理由も隠れる (2026-10-17)
            CR._SSH_BACKOFF.clear()
            raw, errors = CR.collect(hosts, conf.get("tail_bytes") or 65536,
                                     self.args.ssh_timeout,
                                     drop_root=conf.get("drop_root"),
                                     skip_unchanged=self.drop_mtime)
            batches = [r for r in raw if r.get("kind") == "batch"]
            hstats = [r for r in raw if r.get("kind") == "host"]
            runs = [r for r in raw if r.get("kind") not in ("batch", "host")]
            if hstats:
                # 届かなかったマシンの値は前回のものを残す (いつの値かは "at" で分かる)
                hs = load_json(HOST_STATS_PATH)
                for h in hstats:
                    hs[h.get("machine")] = h
                try:
                    with open(HOST_STATS_PATH + ".tmp", "w") as f:
                        json.dump(hs, f, ensure_ascii=False)
                    os.replace(HOST_STATS_PATH + ".tmp", HOST_STATS_PATH)
                except (IOError, OSError) as e:
                    sys.stderr.write("[host] %s\n" % e)

            # 別マシンで回した方策評価のログを logs/ に取ってくる (host に eval_logs: があるときだけ)
            errors = list(errors) + CR.pull_eval_logs(hosts)

            cache = os.path.expanduser(self.args.cache)
            merged = CR.dedupe(CR.read_cache(cache) + runs)
            stale = conf.get("stale_minutes") or 90
            rows = [CR.derive(r, stale, conf.get("method_tag_by_lare_mode"),
                              conf.get("expected_t_max"),
                              lare_chain=conf.get("lare_chain"))
                    for r in merged]
            by_uid = dict((d["uid"], d["state"]) for d in rows)
            for r in merged:
                r["_state"] = by_uid.get(r.get("uid"))
            CR.write_cache(cache, merged)

            with self.lock:
                # SSH のホストで run が 1 件も取れなかったものは「繋がらない」。
                # 赤いエラーではなく「確認できません」として別に持つ
                ssh_labels = set(h["label"] for h in hosts
                                 if not h.get("drop")
                                 and h.get("ssh") not in (None, "", "local"))
                got = set(r.get("machine") for r in raw)
                down = dict((a, b) for a, b in errors
                            if a in ssh_labels and a not in got)
                now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
                for label in ssh_labels:
                    if label in down:
                        prev = self.unreachable.get(label) or {}
                        self.unreachable[label] = {"since": prev.get("since") or now_iso,
                                                   "error": down[label]}
                    else:
                        self.unreachable.pop(label, None)
                self.errors = ["%s: %s" % (a, b) for a, b in errors if a not in down]
                # 繋がらなかったホストの予約 (train.py のバッチ) は前回の分を残す。
                # 消すと計画表の「⏳ 実行待ち」が「未実行」に化け、空いていると誤読する
                self.batches = batches + [b for b in (self.batches or [])
                                          if b.get("machine") in down]
                self.collected_at = time.time()
            # 学習曲線 (「学習曲線」タブ)。計画に入っている run のうち、進んだものだけ取り直す。
            # 繋がらなかったホストは待たされるだけなので飛ばす
            try:
                self.update_curves([h for h in hosts if h["label"] not in down], rows,
                                   conf.get("drop_root"))
            except Exception as e:
                sys.stderr.write("[curves] %s: %s\n" % (type(e).__name__, e))
        except Exception as e:                      # 画面を落とさない
            with self.lock:
                self.errors = ["collect failed: %s: %s" % (type(e).__name__, e)]
            sys.stderr.write("[collect] %s: %s\n" % (type(e).__name__, e))
        finally:
            with self.lock:
                self.busy = False

    def update_curves(self, hosts, rows, drop_root):
        """計画に入っている run の学習曲線を CURVE_CACHE_PATH に貯める (変わった run だけ)."""
        paths = self.plan_files()
        if not paths:
            return
        conds = PLAN.parse_plans(paths)
        planned = [d for d in rows if any(PLAN.matches(c, d) for c in conds)]
        CR.fetch_metric_curves(hosts, planned, CURVE_CACHE_PATH, drop_root=drop_root,
                               timeout=self.args.ssh_timeout)
        # 計画内の LaRe モデル学習が完了していれば、できたモデルを models/ に置く
        # (launchd の定期回収と同じ処理。models/ にあるものは上書きしない)
        try:
            LM.auto_install(conds, rows, PLAN.matches)
        except Exception as e:
            sys.stderr.write("[lare] install: %s: %s\n" % (type(e).__name__, e))

    def curves(self):
        """「学習曲線」タブ用: 計画の章 (計画 x グループ x マップ x 台数 x t_max) ごとに、
        そこに入っている run の曲線を並べる。条件と run の対応は train() の計画表と同じ."""
        t = self.train()
        cache = load_json(CURVE_CACHE_PATH)
        evals = self._curve_evals()
        chapters, order = {}, []
        for c in t.get("plan") or []:
            key = (c.get("plan"), c.get("group_order") or 0, c.get("label"))
            if key not in chapters:
                chapters[key] = {"plan": c.get("plan"), "group": c.get("group"),
                                 "label": c.get("label"), "map": c.get("map"),
                                 "agents": c.get("agents"), "t_max_m": c.get("t_max_m"),
                                 "collect": c.get("collect"), "series": []}
                order.append(key)
            for s in c.get("slots") or []:
                r = s.get("run")
                # 失敗・短すぎる run は曲線を取らない (枠を埋めないので比較にも使わない)
                if not r or r.get("excluded") or r.get("state") not in ("done", "running", "stalled"):
                    continue
                cv = cache.get(r.get("uid")) or {}
                m = {}
                for k, xy in (cv.get("m") or {}).items():
                    # 画面に送る点は 300 点まで、値は小数 3 桁 (全計画で 12MB -> 数 MB)
                    xs, ys = xy[0], xy[1]
                    step = max(1, (len(xs) + 299) // 300)
                    m[k] = [xs[::step] + ([xs[-1]] if (len(xs) - 1) % step else []),
                            [None if v is None else round(v, 3) for v in
                             ys[::step] + ([ys[-1]] if (len(ys) - 1) % step else [])]]
                chapters[key]["series"].append({
                    "uid": r.get("uid"), "machine": r.get("machine"), "seed": r.get("seed"),
                    "state": r.get("state"), "progress": r.get("progress"),
                    "t_last": r.get("t_last"), "t_max": r.get("t_max"),
                    "setting": c.get("setting"), "mark": bool(c.get("mark")),
                    "algo": c.get("algo"), "task_arrival": c.get("task_arrival"),
                    "task_assign": c.get("task_assign") or "TP",
                    "reassign": c.get("reassign"), "dynamic": c.get("dynamic"),
                    "m": m, "ev": evals.get(r.get("uid")) or []})
        return {"chapters": [chapters[k] for k in order],
                "metrics": list(CR.CURVE_KEYS), "collected_at": t.get("collected_at"),
                "eval_arrivals": self._default_eval_arrivals()}

    @staticmethod
    def _default_eval_arrivals():
        """評価表の列の既定 (評価結果がまだ無い章で、空の表の列にする).

        src/config/default.yaml の eval_arrivals を、評価の条件名と同じ書き方
        (runner.py: bern{p:g} / mmpp / fixed) にして返す
        """
        try:
            import yaml
            with open(os.path.join(REPO, "src", "config", "default.yaml")) as f:
                arr = (yaml.safe_load(f) or {}).get("eval_arrivals") or []
        except Exception:
            arr = []
        out = []
        for a in arr:
            mode, _, p = str(a).partition(":")
            if mode == "bernoulli" and p:
                out.append("bern%g" % float(p))
            elif mode in ("mmpp", "fixed"):
                out.append(mode)
        return out or ["bern0.02", "bern0.05", "bern0.1", "mmpp"]

    # 学習曲線タブの評価表に出す指標 (全部送ると重いので絞る)
    CURVE_EVAL_METRICS = ("task_completion", "task_completion_per_agent", "collision_rate",
                          "task_dropped", "goal_account", "deadhead_ratio")

    def _curve_evals(self):
        """run の uid -> その run の方策を評価した結果のリスト.

        保管リポジトリの manifest.jsonl (uid -> eval_name) と、評価ログ (logs/) の
        model= (評価用のファイル名) を突き合わせる。1 つのモデルは評価の条件
        (到着率・実行時の reassign・割当・台数) ごとに何度も評価されるので、条件を
        分解して返す。画面側で 列 = 到着率、選択欄 = reassign・割当 にする
        """
        name_of = {}
        man = os.path.join(os.path.expanduser(self.args.models_repo), "manifest.jsonl")
        try:
            with open(man, "r") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    if r.get("uid") and r.get("eval_name"):
                        name_of[r["uid"]] = r["eval_name"]
        except (IOError, OSError):
            return {}
        try:
            AG = _load_repo("aggregate")
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()):
                runs = AG.load_runs(os.path.join(REPO, "logs"))
        except Exception as e:                  # 読めなくても曲線は出す
            sys.stderr.write("[curves] eval: %s: %s\n" % (type(e).__name__, e))
            return {}
        by_model = {}
        for cond, seeds in runs.items():
            p = ER.parse_condition(cond)
            if p is None:
                continue
            for rec in seeds.values():
                vals = {}
                for m in self.CURVE_EVAL_METRICS:
                    try:
                        vals[m] = round(float(rec[m]), 3)
                    except (KeyError, ValueError):
                        pass
                by_model.setdefault(rec.get("model"), []).append({
                    "n": p["n"], "arrival": p["arrival"], "env_reassign": p["env_reassign"],
                    "allocator": p["allocator"], "planner": p.get("planner"), "vals": vals})
        # 評価の予約 (tools/eval_lare_factor.py が書く。結果がまだ無い欄に「…」を出すだけ)
        try:
            EVL = _load("eval_lare_factor")
            for rv in EVL.reservations(os.path.join(REPO, "logs")):
                by_model.setdefault(rv["model"], []).append({
                    "n": rv["n"], "arrival": rv["arrival"], "env_reassign": rv["env_reassign"],
                    "allocator": rv["allocator"], "vals": {},
                    "reserved": {"host": rv["host"], "until": rv["until"]}})
        except Exception as e:
            sys.stderr.write("[curves] reservations: %s: %s\n" % (type(e).__name__, e))
        return {uid: by_model[name] for uid, name in name_of.items() if name in by_model}

    # --- 自動実行の依頼 (tools/autorun.py。投入そのものは autorun.py tick が行う) ---
    def autorun(self):
        return AR.summary()

    def autorun_add(self, body):
        ids = set(body.get("ids") or [])
        conds = PLAN.parse_plans(self.plan_files())
        for c in conds:
            c["id"] = AR.cond_id(c)
        picked = [c for c in conds if c["id"] in ids]
        if not picked:
            return {"ok": False, "error": "condition not found"}
        machines = [m for m in (body.get("machines") or []) if m]
        made = AR.add_requests(body.get("kind") or "train", picked, int(body.get("n") or 1),
                               bool(body.get("then_eval")), machines or None)
        return {"ok": True, "ids": [r["id"] for r in made]}

    def autorun_cancel(self, body):
        return {"ok": True, "cancelled": AR.cancel_requests([body.get("id")])}

    def autorun_pause(self, body):
        AR.set_pause(bool(body.get("on")))
        return {"ok": True}

    def autorun_tick(self):
        """「今すぐ投入」: autorun.py tick を別プロセスで 1 回動かす (ssh があるので画面は待たない)."""
        log = os.path.join(TOOLS, ".autorun", "tick.log")
        os.makedirs(os.path.dirname(log), exist_ok=True)
        with open(log, "a") as f:
            subprocess.Popen([sys.executable, os.path.join(TOOLS, "autorun.py"), "tick"],
                             cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
        return {"ok": True}

    def status(self):
        with self.lock:
            return {"busy": self.busy, "collected_at": self.collected_at,
                    "errors": self.errors}

    def mini(self):
        """デスクトップの小さいパネル用の要約 (1KB 未満).

        train() は 1.8MB あり、常駐パネルが毎分取りにくるには重い。
        中身は同じ計算なので、要るものだけ抜いて返す。
        """
        d = self.train()
        plan = d.get("plan") or []
        full = part = todo = 0
        need = running = 0
        for c in plan:
            want = c.get("want") or 5
            slots = (c.get("slots") or [])[:want]
            done = sum(1 for s in slots
                       if s.get("run") and s["run"].get("state") == "done")
            kept = sum(1 for s in slots
                       if s.get("run") and (s["run"].get("saved") or []))
            run = sum(1 for s in slots
                      if s.get("run") and s["run"].get("state") == "running")
            need += max(0, want - done - run)
            running += run
            if done >= want and kept >= want:
                full += 1
            elif done or run:
                part += 1
            else:
                todo += 1
        # マシンごとの「最後に何を終わらせたか」。実行中が 0 でも、直前に
        # 終わったのが 3 日前なら手が空いている = 投入すべき、と判断できる
        last = {}
        nxt = {}        # マシンごとに「次に終わる予定の実行中 run」
        paused = {}     # マシンごとの一時停止中 (Ctrl-Z) の run
        for r in d.get("runs") or []:
            k = r.get("machine")
            if r.get("state") == "running" and r.get("paused"):
                paused.setdefault(k, []).append(r)
            elif r.get("state") == "running" and r.get("eta"):
                if k not in nxt or r["eta"] < nxt[k]["eta"]:
                    nxt[k] = r
            if r.get("state") != "done" or not r.get("stop_at"):
                continue
            if k not in last or r["stop_at"] > last[k]["stop_at"]:
                last[k] = r
        def what(r):
            return "%sag %s %s" % (r.get("agents"), str(r.get("map") or "").replace("map_", ""),
                                   r.get("algo"))

        machines = []
        for name, m in sorted((d.get("machines") or {}).items()):
            r = last.get(name)
            e = nxt.get(name)
            machines.append({
                "name": name, "running": m.get("running") or 0,
                "age": m.get("data_age_sec"), "stale": bool(m.get("stale_data")),
                "last_done": r.get("stop_at") if r else None,
                "last_what": what(r) if r else None,
                # 次の終了予定。実測ペースから出した見込みなので、情報が古いホストでは当てにならない
                "next_eta": e.get("eta") if e else None,
                "next_what": what(e) if e else None,
                # 一時停止中の本数と、再開したときに最も早く終わるものの残り時間
                "paused": len(paused.get(name, [])),
                "paused_remaining": min((x.get("remaining_sec") for x in paused.get(name, [])
                                         if x.get("remaining_sec") is not None), default=None),
                "paused_what": what(paused[name][0]) if paused.get(name) else None,
            })
        return {"conditions": len(plan), "full": full, "part": part, "todo": todo,
                "need": need, "running": running, "machines": machines,
                "collected_at": self.collected_at, "busy": self.busy}


# ---------------------------------------------------------------------------
# ルーティング (flask / 標準ライブラリのどちらからも同じものを呼ぶ)
# ---------------------------------------------------------------------------

def make_routes(state):
    return {
        ("GET", "/api/train"): lambda body: state.train(),
        ("GET", "/api/eval"): lambda body: state.eval(),
        ("GET", "/api/curves"): lambda body: state.curves(),
        ("GET", "/api/status"): lambda body: state.status(),
        ("GET", "/api/mini"): lambda body: state.mini(),
        ("POST", "/api/collect"): lambda body: {"started": state.collect()},
        ("POST", "/api/ack"): lambda body: state.ack(body),
        ("POST", "/api/note"): lambda body: state.note(body),
        ("POST", "/api/dismiss_diff"): lambda body: state.dismiss_diff(body),
        ("POST", "/api/exclude"): lambda body: state.exclude(body),
        ("GET", "/api/autorun"): lambda body: state.autorun(),
        ("POST", "/api/autorun/add"): lambda body: state.autorun_add(body),
        ("POST", "/api/autorun/cancel"): lambda body: state.autorun_cancel(body),
        ("POST", "/api/autorun/pause"): lambda body: state.autorun_pause(body),
        ("POST", "/api/autorun/tick"): lambda body: state.autorun_tick(),
    }


def serve_flask(state, host, port):
    from flask import Flask, jsonify, render_template, request
    app = Flask(__name__, template_folder=os.path.join(HERE, "templates"),
                static_folder=os.path.join(HERE, "static"))
    routes = make_routes(state)
    # stdlib 側と同じく、開発中は常に取り直させる
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

    @app.after_request
    def _nocache(resp):
        resp.headers["Cache-Control"] = "no-store, must-revalidate"
        return resp

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/<path:rest>", methods=["GET", "POST"])
    def api(rest):
        fn = routes.get((request.method, "/api/" + rest))
        if fn is None:
            return jsonify({"error": "not found"}), 404
        body = request.get_json(silent=True) or {}
        return jsonify(fn(body))

    sys.stderr.write("[dashboard] flask  http://%s:%d\n" % (host, port))
    app.run(host=host, port=port, threaded=True, debug=False)


def serve_stdlib(state, host, port):
    """flask が無いとき用。templates/index.html に Jinja 構文が無いのでそのまま返せる."""
    import mimetypes
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    routes = make_routes(state)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, data, ctype):
            body = data if isinstance(data, bytes) else data.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            # 開発中のツールなので **常に取り直させる**。
            # Cache-Control も ETag も付けないと、ブラウザが発見的に
            # app.js を握り続け、直したはずの画面が変わらない
            self.send_header("Cache-Control", "no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(body)

        def do_HEAD(self):
            # 501 Unsupported method ('HEAD') を返さないようにする (curl -I 用)
            self.do_GET()

        def _file(self, path, ctype=None):
            try:
                with open(path, "rb") as f:
                    data = f.read()
            except OSError:
                return self._send(404, b"not found", "text/plain")
            self._send(200, data, ctype or
                       (mimetypes.guess_type(path)[0] or "application/octet-stream"))

        def _api(self, method):
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                body = {}
            fn = routes.get((method, self.path.split("?")[0]))
            if fn is None:
                return self._send(404, json.dumps({"error": "not found"}),
                                  "application/json")
            self._send(200, json.dumps(fn(body), ensure_ascii=False, default=str),
                       "application/json; charset=utf-8")

        def do_GET(self):
            p = self.path.split("?")[0]
            if p in ("/", "/index.html"):
                return self._file(os.path.join(HERE, "templates", "index.html"),
                                  "text/html; charset=utf-8")
            if p.startswith("/static/"):
                rel = os.path.normpath(p[len("/static/"):]).lstrip(os.sep)
                return self._file(os.path.join(HERE, "static", rel))
            if p.startswith("/api/"):
                return self._api("GET")
            self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path.startswith("/api/"):
                return self._api("POST")
            self._send(404, b"not found", "text/plain")

    try:
        srv = ThreadingHTTPServer((host, port), H)
    except OSError as e:
        return port_in_use(host, port, e)
    sys.stderr.write("[dashboard] stdlib http.server  http://%s:%d\n"
                     "[dashboard] (pip install flask すると flask で動きます)\n"
                     % (host, port))
    srv.serve_forever()


def port_in_use(host, port, err):
    """ポートが埋まっているときに **何をすればよいか** を出す.

    素の OSError だと Flask の ImportError を巻き込んだ二重トレースバックになり、
    「flask が無いのが原因」と誤読しやすい。原因と対処だけを出す。
    """
    import errno
    if getattr(err, "errno", None) != errno.EADDRINUSE:
        raise err
    sys.stderr.write("[dashboard] port %d on %s is already in use.\n" % (port, host))
    try:
        out = subprocess.check_output(
            ["lsof", "-nP", "-iTCP:%d" % port, "-sTCP:LISTEN"],
            stderr=subprocess.DEVNULL).decode("utf-8", "replace").splitlines()
        for line in out[1:]:
            f = line.split()
            if len(f) > 1:
                sys.stderr.write("[dashboard]   pid %s (%s) is holding it\n"
                                 % (f[1], f[0]))
    except (OSError, subprocess.CalledProcessError):
        pass
    sys.stderr.write("[dashboard] either open http://%s:%d (it is probably the "
                     "same dashboard), stop that pid, or pass --port\n"
                     % (host, port))
    return 1


def main(argv=None):
    p = argparse.ArgumentParser(description="LDRP experiment dashboard.")
    p.add_argument("-c", "--config", default=os.path.join(TOOLS, "collect_config.yaml"))
    p.add_argument("--cache", default=os.path.join(TOOLS, ".run_cache.jsonl"),
                   help="collect_runs.py --cache と同じファイル")
    p.add_argument("-p", "--plan", action="append", default=None,
                   help="実験計画 (Notion の表をそのまま貼れる). glob 可。"
                        "複数指定でき、省略時は tools/plans/*.md -> tools/plan.md")
    p.add_argument("--models-repo", default="~/LDRP_models",
                   help="方策モデルの保管リポジトリ。manifest.jsonl を読んで "
                        "「回収済みか」を表示する")
    p.add_argument("-s", "--summary", default=os.path.join(REPO, "results", "summary.csv"))
    p.add_argument("--host", default="127.0.0.1", help="既定は localhost のみ")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--ssh-timeout", type=int, default=180)
    p.add_argument("--collect-on-start", action="store_true",
                   help="起動時に 1 回収集する (既定はキャッシュを読むだけ)")
    p.add_argument("--collect-every", type=int, default=5, metavar="MIN",
                   help="この分数ごとに裏で収集する (0 で無効。既定 5)。"
                        "学習曲線を学習中に追うため 5 分にしてある (GPU は ssh で 10 秒ほど)。"
                        "共有フォルダのホスト (黒) は相手が 15 分おきに書き出すので、それより細かくはならない")
    args = p.parse_args(argv)

    state = State(args)
    if args.collect_on_start:
        state.collect()
    if args.collect_every:
        threading.Thread(target=state._auto_collect_loop, daemon=True).start()
        sys.stderr.write("[dashboard] auto collect every %d min\n" % args.collect_every)

    try:
        import flask                                     # noqa: F401
    except ImportError:
        return serve_stdlib(state, args.host, args.port)
    return serve_flask(state, args.host, args.port)


if __name__ == "__main__":
    sys.exit(main())
