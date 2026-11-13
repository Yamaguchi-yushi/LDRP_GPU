#!/usr/bin/env python3
"""LDRP: 学習 run をマシン横断で収集し、進捗・完遂状況を集計して Notion に同期する.

2 つのモードがあり、併用できる (どちらも同じ Notion DB に upsert される):

  push : 各マシン上で `--hosts local --machine GPU1 --notion` を cron 実行する。
         SSH 不要。Notion token だけあればよい。
  pull : Mac から `--config tools/collect_config.yaml --notion` を実行する。
         各ホストへ ssh し、このスクリプト自身を stdin で送り込んで走らせる
         (リモートには何もインストールしない。stdlib のみで動く)。

低メモリ設計: sacred の config.json / run.json (数 KB) と cout.txt の末尾
(既定 64KB) しか読まない。巨大な metrics.json (5MB) / info.json (2.5MB) は
一切開かない。
"""

from __future__ import print_function

import argparse
import collections
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------

# repo ルートからの相対で、sacred の出力が置かれる場所
DEFAULT_SACRED_SUBDIRS = (
    "src/epymarl/results/sacred",
    "src/epymarl/tmp_results/sacred",
)

# 学習済みモデルの置き場 (repo ルートからの相対).
# epymarl は save_path = "{local_results_path}/models/{unique_token}/{t_env}" に保存し、
# local_results_path は既定 "results" = 起動時 cwd (= repo ルート) からの相対
DEFAULT_MODEL_SUBDIRS = (
    "results/models",
    "src/epymarl/results/models",
    "src/epymarl/tmp_results/models",
)

# config.json から拾うキー (これ以外は捨ててメモリと転送量を抑える)
CFG_KEYS = (
    "name", "seed", "t_max", "env", "label", "checkpoint_path", "use_cuda",
    "train_task_assigner", "test_interval", "save_model_interval",
    "batch_size_run", "runner",
)
ENV_KEYS = (
    "key", "time_limit", "state_repre_flag",
    "use_lare_path", "use_lare_path_training",
    "use_pretrained_lare_path", "pretrained_lare_path_model_name",
    "use_finetuning_lare_path", "finetuning_lare_path_model_name",
    "use_lare_task", "use_lare_task_training",
    "randomize_task_arrival", "task_arrival", "task_density", "mmpp_ratio",
    "rand_p_min", "rand_p_max", "task_p_high", "task_p_low", "task_switch_prob",
    "use_dynamic_agents", "randomize_initial_active",
    "min_active_agents", "max_active_agents",
    "allow_reassign_before_pickup", "episode_seed_base",
)

# epymarl のログ行:  "[INFO 18:46:07] my_main t_env: 10050500 / 10050000"
T_ENV_RE = re.compile(r"t_env:\s*(\d+)\s*/\s*(\d+)")
# LaRe 側のログ行:  "t_env:   10050000 | Episode:    20100"
T_ENV_ALT_RE = re.compile(r"t_env:\s*(\d+)\s*\|")
# LaRe モデル名の 1 区間: "Safe_map_8x5_2agents_10.0M" -> ("8x5", "2", "10.0")
LARE_SEG_RE = re.compile(r"map_(.+?)_(\d+)agents_([\d.]+)M")
# 環境キー: "drp_env:drp_safe-7agent_map_8x5-v2"
ENV_KEY_RE = re.compile(r"drp(_safe)?-(\d+)agent_(.+?)-v\d")

# method_tag (評価用モデルのファイル名に入る条件ラベル) の既定ルール。
# 「事前学習した LaRe モデルを使ったか」で dbct / safe を分ける。
# 非 Safe 環境で学習したモデルは design/multi_seed_eval.md の運用に合わせて "unsafe"。
# 別の軸で分けたくなったら collect_config.yaml の method_tag_by_lare_mode で上書きする
DEFAULT_METHOD_TAG_BY_LARE_MODE = {
    "pretrained": "dbct",
    "finetuning": "dbct",
    "scratch": "safe",
    "scratch(frozen)": "safe",
    "off": "safe",
}

STATE_ORDER = ("failed", "stalled", "short", "running", "done", "unknown")
STATE_MARK = {
    "done": "OK", "running": "RUN", "stalled": "STALL",
    "short": "SHORT", "failed": "FAIL", "unknown": "?",
}


# ===========================================================================
# ここから scanner (リモートでも実行される。stdlib のみ / 出力は必ず ASCII JSON)
# ===========================================================================

def _read_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _tail_text(path, nbytes):
    """ファイル末尾 nbytes だけを読む (巨大ログでもメモリを食わない)."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            if size > nbytes:
                f.seek(size - nbytes)
            data = f.read()
    except OSError:
        return ""
    return data.decode("utf-8", "replace")


# metrics.json の "steps": [...] 配列。中身は素の整数 (= その記録の t_env) だけ。
# values (小数) や timestamps (文字列) は見ない。以前は「配列要素の整数」を全部拾って
# いたので、末尾に steps が無いと values の整数を t_env と取り違えるおそれがあった
STEPS_ARRAY_RE = re.compile(r'"steps":\s*\[([^\]]*)\]')

# 末尾を読む量の上限。記録が増えると、最後の指標 1 つ分 (steps / timestamps / values)
# だけで 256KB を超え、末尾 256KB に steps が入らなくなる
# (GPU1 の mappo aoba00 10 台、1.2 億ステップで 1 指標 332KB の実測)。
# steps が見つかるまで 2 倍ずつ広げ、ここで打ち切る
METRICS_TAIL_MAX = 8 * 1024 * 1024


def _t_env_from_metrics(path, nbytes, cap):
    """cout.txt が空のときの保険。metrics.json の末尾だけを読み、
    steps 配列の中で t_max を超えない最大の整数 (= 直近の t_env) を拾う。

    metrics.json は数十 MB になるので全体はロードしない。末尾 nbytes に
    steps 配列が 1 つも入っていなければ、読む量を 2 倍ずつ増やして読み直す
    (上限 METRICS_TAIL_MAX)。
    """
    try:
        size = os.path.getsize(path)
    except OSError:
        return None
    try:
        limit = float(cap) * 1.01 if cap else None
    except (TypeError, ValueError):
        limit = None
    n = nbytes
    while True:
        text = _tail_text(path, n)
        best = None
        for arr in STEPS_ARRAY_RE.finditer(text):
            for tok in arr.group(1).split(","):
                tok = tok.strip()
                if not tok.isdigit():
                    continue
                v = int(tok)
                if limit is not None and v > limit:
                    continue
                if best is None or v > best:
                    best = v
        if best is not None or n >= size or n >= METRICS_TAIL_MAX:
            return best
        n *= 2


def _pick(d, keys):
    if not isinstance(d, dict):
        return {}
    return dict((k, d[k]) for k in keys if k in d)


# 実験条件としての同一性から外すキー (run ごとに変わる / 条件と無関係)
HASH_EXCLUDE_TOP = frozenset((
    "seed", "checkpoint_path", "load_step", "local_results_path", "label",
    "repeat_id", "evaluate", "render", "save_replay", "use_tensorboard",
    "use_cuda", "save_model", "save_model_interval",
    "test_interval",    # テストの間隔は学習曲線の点の間隔だけで、学習には効かない
))
# lare_device は cpu / cuda の実行環境であって実験条件ではない。
# GPU 機と Mac で同じ条件を回すと必ず割れるので外す
HASH_EXCLUDE_ENV = frozenset(("seed", "lare_device"))


# 「有効化フラグが off のとき無視されるキー」. 消し忘れが残っていても
# 同一条件と判定できるようにする (実際に use_lare_path=False の run に
# use_finetuning_lare_path=True が残っていた事例がある)
HASH_DEPENDENT = (
    ("use_lare_path", ("use_lare_path_training", "use_pretrained_lare_path",
                       "pretrained_lare_path_model_name", "use_finetuning_lare_path",
                       "finetuning_lare_path_model_name")),
    ("use_pretrained_lare_path", ("pretrained_lare_path_model_name",)),
    ("use_finetuning_lare_path", ("finetuning_lare_path_model_name",)),
    ("use_dynamic_agents", ("randomize_initial_active", "min_active_agents",
                            "max_active_agents", "initial_active_num")),
    ("use_lare_task", ("use_lare_task_training",)),
)


# drp_env.DrpEnv.__init__ の reward_list の既定値。train.py は reward_list を渡して
# いないので、config にキーが無い run はこの値で学習している
DEFAULT_REWARD_LIST = {"goal": 100, "collision": -10, "wait": -10, "move": -1}


def _norm_reward_list(v):
    """reward_list を比べられる形にする. 既定値と同じなら None (= キーが無い run と同一視).

    sacred には書いたままの数が残るので、-10 と -10. を同じ値にするため float にそろえる。
    """
    if not isinstance(v, dict):
        return v
    try:
        v = dict((k, float(x)) for k, x in v.items())
    except (TypeError, ValueError):
        return v
    if v == dict((k, float(x)) for k, x in DEFAULT_REWARD_LIST.items()):
        return None
    return v


def param_fields(cfg_full, env_full, dropped=None):
    """実験条件として意味を持つパラメータだけに正規化した dict を返す.

    param_hash も条件間の差分表示も**必ずこれを通す**。別々に正規化すると
    「ハッシュは同じなのに差分が出る」ような食い違いが起きる。

    dropped に set を渡すと、「この run では使われない」として規則で外したキー
    ("env.mmpp_ratio" など) を入れて返す。条件をまたいで比べるとき (dashboard の
    global_param_diff)、外したキーを「値が無い」と比べて警告しないために使う。
    """
    top = dict((k, v) for k, v in (cfg_full or {}).items()
               if k not in HASH_EXCLUDE_TOP)
    env = dict((k, v) for k, v in (env_full or {}).items()
               if k not in HASH_EXCLUDE_ENV)

    def drop(k):
        # キーが無い古い run でも「この run では使われない」ことは同じなので記録する
        if dropped is not None:
            dropped.add("env." + k)
        env.pop(k, None)

    # 有効化フラグが off のとき無視されるキーを落とす
    for flag, dependents in HASH_DEPENDENT:
        if not env.get(flag, False):
            for k in dependents:
                drop(k)
    # 事前学習 (固定) が指定されていると追加学習の指定は無視される
    # (drp_env.py の _init_lare_path: finetuning = ... and not pretrained)
    if env.get("use_pretrained_lare_path") and env.get("pretrained_lare_path_model_name"):
        for k in ("use_finetuning_lare_path", "finetuning_lare_path_model_name"):
            drop(k)
    # 動的台数の初期台数: ランダムに引くなら max_active_agents、引かないなら
    # initial_active_num だけが使われる (drp_env.py の reset)。min_active_agents は
    # 稼働を止めてよい下限としても使う (ppo.py) ので残す
    if env.get("use_dynamic_agents"):
        drop("initial_active_num" if env.get("randomize_initial_active")
             else "max_active_agents")
    # randomize_task_arrival が真なら固定値側は使われない
    if env.get("randomize_task_arrival"):
        for k in ("task_arrival", "task_density"):
            drop(k)
    # task_arrival="fixed" は毎ステップ 1 件で、task_density を使わない
    # (EE_map.create_tasklist の mode == 'fixed' の分岐)
    # mmpp も task_density を使わない (p_high / p_low だけ)
    elif env.get("task_arrival", "fixed") in ("fixed", "mmpp"):
        drop("task_density")
    if not env.get("randomize_task_arrival"):
        # ランダム化の範囲は、randomize_task_arrival=True のときだけ読まれる
        # (drp_env.py の reset → _sample_task_arrival)
        for k in ("mmpp_ratio", "rand_p_min", "rand_p_max"):
            drop(k)
        # MMPP の値は task_arrival="mmpp" のときだけ使う (EE_map.create_tasklist)。
        # ランダム化ありでは MMPP が引かれうるので残す
        if env.get("task_arrival", "fixed") != "mmpp":
            for k in ("task_p_high", "task_p_low", "task_switch_prob"):
                drop(k)
    # 環境報酬 (goal / collision / wait / move)。既定値から変えた run だけキーが残り、
    # 同じ条件の他の seed や他の条件と違えば params✗ / 他と違う の警告になる
    if "reward_list" in env:
        rl = _norm_reward_list(env.get("reward_list"))
        if rl is None:
            env.pop("reward_list")
        else:
            env["reward_list"] = rl
    # max_active_agents は「開始時に稼働台数を引くときの上限」で、env が台数 N
    # (環境名の drp_safe-{N}agent) に丸める。未指定 (None) も N 以上も挙動は同じ
    # (drp_env.py の reset: hi = agent_num if max is None else min(agent_num, max))。
    # 同じ値として扱わないと、行を消した run と max=N を書いた run が別条件に見える。
    # N 未満のときだけ本当に条件が違うので残す
    if "max_active_agents" in env:
        m = ENV_KEY_RE.search(str(env.get("key") or ""))
        n = int(m.group(2)) if m else None
        mx = env.get("max_active_agents")
        try:
            if mx is None or (n is not None and int(mx) >= n):
                env.pop("max_active_agents", None)
        except (TypeError, ValueError):
            pass
    # None / False は「キーが無い」と同一視する。config のスキーマは版ごとに増えており、
    # 古い run にはキー自体が存在しない。新設フラグの既定は False / None なので、
    # これを潰さないと「後から足したキーの有無」だけで差が出る。
    # (0 や "" は falsy だが意図した設定値なので残す)
    out = {}
    for pre, d in (("cfg.", top), ("env.", env)):
        for k, v in d.items():
            if v is not None and v is not False:
                out[pre + k] = v
    return out


# env 側でハードコードされており、epymarl の config に出ていても挙動を変えないキー
HASH_IGNORE_FIELDS = frozenset((
    "cfg.task_num",        # env が self.task_num = 10 を固定で持つ (drp_env.py ~168 行)
))


def param_diff(rows):
    """同じ条件のはずの run 群で **値が割れているキーだけ** を抜き出す.

    返り値: [(key, {param_hash: 値}), ...] をキー名順に。
    param_fields() を通すので、ハッシュの判定と必ず一致する
    (別々に正規化すると「ハッシュは同じなのに差分が出る」ことが起きる)。
    """
    by_hash = {}
    seeds = {}
    for r in rows:
        h = r.get("param_hash")
        seeds.setdefault(h, []).append(str(r.get("seed")))
        if h not in by_hash:
            by_hash[h] = param_fields(r.get("cfg"), r.get("env"))
    keys = set()
    for f in by_hash.values():
        keys |= set(f)
    out = []
    for k in sorted(keys - set(HASH_IGNORE_FIELDS)):
        vals = dict((h, f.get(k)) for h, f in by_hash.items())
        uniq = set(json.dumps(v, sort_keys=True, default=str) for v in vals.values())
        if len(uniq) > 1:
            out.append((k, vals))
    return out, seeds


def fmt_param_diff(rows, limit=8, indent="    "):
    """param_diff を人が読める行のリストにする."""
    diffs, seeds = param_diff(rows)
    if not diffs:
        return []
    order = sorted(seeds, key=lambda h: (-len(seeds[h]), str(h)))
    lines = [indent + "%-26s %s" % ("(key)", "  ".join(
        "%s[%d]" % (h, len(seeds[h])) for h in order))]
    for k, vals in diffs[:limit]:
        cells = []
        for h in order:
            v = vals.get(h)
            t = "-" if v is None else str(v)
            cells.append(t if len(t) <= 22 else t[:19] + "...")
        lines.append(indent + "%-26s %s" % (k, "  ".join(cells)))
    if len(diffs) > limit:
        lines.append(indent + "... 他 %d キー (--diff-params で全部出す)"
                     % (len(diffs) - limit))
    return lines


def param_hash(cfg_full, env_full):
    """学習パラメータの同一性を表す 8 桁。seed とパスは含めない."""
    import hashlib
    f = dict((k, v) for k, v in param_fields(cfg_full, env_full).items()
             if k not in HASH_IGNORE_FIELDS)
    blob = json.dumps(f, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:8]


def _root_tag(sacred_root):
    """.../src/epymarl/results/sacred -> 'results'."""
    parent = os.path.basename(os.path.dirname(os.path.abspath(sacred_root)))
    return parent or "sacred"


def build_model_index(repo, model_subdirs):
    """repo 配下の models/<unique_token> を 1 度だけ列挙する.

    unique_token = "{algo}_seed{seed}_{env_key}_{起動時刻}" (src/epymarl/src/run.py)
    なので、seed まで含めた前綴りで sacred の run と 1:1 に紐づけられる。
    """
    index = []
    for sub in model_subdirs:
        root = sub if os.path.isabs(sub) else os.path.join(repo, sub)
        if not os.path.isdir(root):
            continue
        try:
            names = os.listdir(root)
        except OSError:
            continue
        for name in names:
            if os.path.isdir(os.path.join(root, name)):
                index.append((name, sub, os.path.join(root, name)))
    return index


DEFAULT_TB_SUBDIRS = (
    "src/epymarl/results/tb_logs",
    "src/epymarl/tmp_results/tb_logs",
    "results/tb_logs",
)


def build_tb_index(repo, tb_subdirs):
    """repo 配下の tb_logs/<unique_token> を列挙する.

    ディレクトリ名は models/ と同じ unique_token なので、find_model と
    同じ前綴りで sacred の run に紐づけられる。
    """
    index = []
    for sub in tb_subdirs or ():
        root = sub if os.path.isabs(sub) else os.path.join(repo, sub)
        if not os.path.isdir(root):
            continue
        try:
            names = os.listdir(root)
        except OSError:
            continue
        for name in names:
            full = os.path.join(root, name)
            if os.path.isdir(full):
                index.append((name, full))
    return index


def find_tb_dir(index, algo, seed, env_key):
    """run に対応する tb_logs ディレクトリを返す (find_model と同じ規則)."""
    if not index or seed in (None, "") or not algo:
        return None
    cands = [c for c in index if c[0].startswith("%s_seed%s_" % (algo, seed))]
    if env_key:
        keyed = [c for c in cands if env_key in c[0]]
        if keyed:
            cands = keyed
    if not cands:
        return None
    cands.sort(key=lambda c: c[0])
    return cands[-1]                      # 再実行があれば新しい方 (末尾が時刻)


def read_scalars(tb_dir, metrics=None, max_points=0):
    """event ファイルから scalar を読む -> {tag: [(wall, step, value), ...]}.

    tensorboard が無い環境 (自己送出先のリモート等) では黙って空を返す。
    run の収集そのものは止めない。
    """
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        return None                       # None = 読めなかった (空 {} と区別する)
    try:
        ea = event_accumulator.EventAccumulator(tb_dir, size_guidance={"scalars": 0})
        ea.Reload()
        tags = ea.Tags()["scalars"]
    except Exception:
        return None
    out = {}
    for t in tags:
        if metrics and t not in metrics:
            continue
        try:
            pts = [(e.wall_time, e.step, e.value) for e in ea.Scalars(t)]
        except Exception:
            continue
        if max_points and len(pts) > max_points:
            # 等間隔に間引く。最後の点は必ず残す (収束値が消えると困る)
            step = len(pts) / float(max_points)
            idx = sorted(set([int(i * step) for i in range(max_points)] + [len(pts) - 1]))
            pts = [pts[i] for i in idx]
        out[t] = pts
    return out


def find_model(index, repo, algo, seed, env_key):
    """run に対応するモデルディレクトリの「最終ステップ」を返す."""
    if not index or seed in (None, "") or not algo:
        return None
    prefix = "%s_seed%s_" % (algo, seed)
    cands = [c for c in index if c[0].startswith(prefix)]
    if env_key:
        keyed = [c for c in cands if env_key in c[0]]
        if keyed:
            cands = keyed
    if not cands:
        return None
    cands.sort(key=lambda c: c[0])
    name, sub, full = cands[-1]           # 同 seed の再実行があれば新しい方 (時刻が末尾)

    steps = []
    try:
        for d in os.listdir(full):
            if d.isdigit() and os.path.isdir(os.path.join(full, d)):
                steps.append(int(d))
    except OSError:
        return None
    if not steps:
        return None
    step = max(steps)

    files = []
    step_dir = os.path.join(full, str(step))
    for dirpath, _dirnames, filenames in os.walk(step_dir):
        for fn in filenames:
            fp = os.path.join(dirpath, fn)
            try:
                size = os.path.getsize(fp)
            except OSError:
                continue
            files.append([os.path.relpath(fp, repo), size])
    if not files:
        return None
    return {
        "dir": os.path.relpath(full, repo),
        "sub": sub,
        "step": step,
        "n_steps": len(steps),
        "files": sorted(files),
    }


# ---- LaRe-Path のモデル (.pth) ------------------------------------------------
# LaRe を学習した run (scratch / 追加学習) は、方策とは別に
# src/lare/path/checkpoints/ へ LaRe のモデルを保存する (drp_env.py の autosave)。
# ファイル名は run の設定から決まるので、名前の規則と更新時刻で run と結ぶ:
#   scratch : {ALGO}_PATH_[F{v}_]{Safe_}{map}_{N}agents_{S}M_{checkpoint|final}.pth
#   追加学習: FT_{元のモデル名}_{Safe_}{map}_{N}agents_{S}M_{checkpoint|final}.pth
# 同じ条件を同じマシンで回し直すと同じ名前で上書きされるので、
# 「run の開始から終了 (+10 分) の間に書かれたもの」だけをその run のものとみなす
LARE_PATH_CKPT_DIR = os.path.join("src", "lare", "path", "checkpoints")
LARE_STEP_RE = re.compile(r"_([\d.]+)M_(checkpoint|final)\.pth$")
LARE_VER_RE = re.compile(r"_PATH_F(\d+)_")
_LARE_LS = {}


def _lare_listdir(d):
    """checkpoints/ の一覧 (走査 1 回の中では使い回す)."""
    if d not in _LARE_LS:
        try:
            _LARE_LS[d] = sorted(os.listdir(d))
        except OSError:
            _LARE_LS[d] = []
    return _LARE_LS[d]


def _lare_source_base(name):
    """drp_env._lare_get_source_base_name と同じ: 拡張子・接尾辞・先頭の FT_ を外す."""
    n = os.path.basename(str(name or ""))
    if n.endswith(".pth"):
        n = n[:-4]
    n = n.replace("_final", "").replace("_checkpoint", "")
    while n.startswith("FT_"):
        n = n[3:]
    return n


def find_lare_model(repo, env, start_time, end_time):
    """run が保存した LaRe-Path のモデルのうち最後のもの. 無ければ None.

    返り値: {"name", "rel" (repo からの相対), "size", "mtime", "step_m"}
    """
    env = env or {}
    if not repo or not env.get("use_lare_path"):
        return None
    if env.get("use_pretrained_lare_path") and env.get("pretrained_lare_path_model_name"):
        return None                       # 固定して使うだけ。保存しない
    if env.get("lare_path_autosave") is False:
        return None
    m = ENV_KEY_RE.search(str(env.get("key") or ""))
    if not m:
        return None
    safe_seg = "Safe_" if m.group(1) else ""
    stage = "%s%s_%sagents_" % (safe_seg, m.group(3), m.group(2))
    fin = env.get("use_finetuning_lare_path") and env.get("finetuning_lare_path_model_name")
    try:
        version = int(env.get("lare_path_factor_version") or 1)
    except (TypeError, ValueError):
        version = 1
    explicit = env.get("lare_path_autosave_path")
    if explicit:
        d = os.path.dirname(os.path.join(repo, os.path.expanduser(str(explicit))))
        names = [os.path.basename(str(explicit))]
    else:
        d = os.path.join(repo, LARE_PATH_CKPT_DIR)
        names = _lare_listdir(d)
    t0 = parse_dt(start_time)
    t1 = parse_dt(end_time)
    lo = t0.timestamp() - 60 if t0 else None
    hi = t1.timestamp() + 600 if t1 else None
    best = None
    for n in names:
        sm = LARE_STEP_RE.search(n)
        if not sm:
            continue
        head = n[:sm.start() + 1]                     # "..._{N}agents_" まで
        if not head.endswith(stage):
            continue
        if fin:
            if not n.startswith("FT_%s_%s" % (_lare_source_base(fin), stage)):
                continue
        else:
            if n.startswith("FT_") or "_PATH_" not in n:
                continue
            vm = LARE_VER_RE.search(n)
            if (int(vm.group(1)) if vm else 1) != version:
                continue
        p = os.path.join(d, n)
        try:
            st = os.stat(p)
        except OSError:
            continue
        if st.st_size == 0:
            continue
        if (lo is not None and st.st_mtime < lo) or (hi is not None and st.st_mtime > hi):
            continue
        key = (float(sm.group(1)), sm.group(2) == "final")
        if best is None or key > best[0]:
            best = (key, n, st)
    if best is None:
        return None
    (step_m, _final), n, st = best
    return {"name": n, "rel": os.path.relpath(os.path.join(d, n), repo),
            "size": st.st_size, "mtime": st.st_mtime, "step_m": step_m}


def scan_run_dir(run_dir, machine, root_tag, tail_bytes, repo=None, model_index=None):
    """sacred の run ディレクトリ 1 つを読んでレコード dict を返す."""
    cfg = _read_json(os.path.join(run_dir, "config.json"))
    if cfg is None:
        return None
    run = _read_json(os.path.join(run_dir, "run.json")) or {}

    run_id = os.path.basename(run_dir.rstrip("/"))
    env_dir = os.path.basename(os.path.dirname(run_dir.rstrip("/")))
    algo_dir = os.path.basename(os.path.dirname(os.path.dirname(run_dir.rstrip("/"))))

    cout = os.path.join(run_dir, "cout.txt")
    tail = _tail_text(cout, tail_bytes)
    t_last = None
    hits = T_ENV_RE.findall(tail) or T_ENV_ALT_RE.findall(tail)
    if hits:
        last = hits[-1]
        t_last = int(last[0] if isinstance(last, tuple) else last)

    t_source = "cout" if t_last is not None else None
    if t_last is None:
        # GPU 機のように stdout をシェルでリダイレクトしていると cout.txt が 0 byte
        t_last = _t_env_from_metrics(os.path.join(run_dir, "metrics.json"),
                                     nbytes=262144, cap=cfg.get("t_max"))
        if t_last is not None:
            t_source = "metrics"

    mtime = None
    for f in ("cout.txt", "metrics.json", "run.json"):
        try:
            mtime = max(mtime or 0, os.path.getmtime(os.path.join(run_dir, f)))
        except OSError:
            pass

    # 学習パラメータは whitelist せず全文を持つ (1 run 2.5KB. 490 run でも 1.2MB).
    # 落とすと再現できない: lr / gamma / batch_size / epsilon_* / mixer /
    # hypernet_* / obs_agent_id / exclude_station_from_tasks など
    cfg_full = dict((k, v) for k, v in cfg.items() if k != "env_args")
    env_full = dict(cfg.get("env_args") or {})

    model = None
    if model_index is not None:
        model = find_model(model_index, repo, cfg.get("name"), cfg.get("seed"),
                           (cfg.get("env_args") or {}).get("key"))

    if t_last is None and model and model.get("step") is not None:
        # cout.txt が 0 byte で、かつ metrics.json が pretty-print で巨大な環境
        # (GPU 機は 29MB) では末尾 256KB が最後の values 配列だけで埋まり
        # steps の整数に届かない。最後のチェックポイント名 = 保存時の t_env なので
        # これを最後の保険にする (save_model_interval の粒度だけ切り捨てられる)
        t_last = model["step"]
        t_source = "model"

    host = run.get("host") or {}
    rec = {
        "repo": repo,
        "model": model,
        "uid": "%s:%s/%s/%s/%s" % (machine, root_tag, algo_dir, env_dir, run_id),
        "machine": machine,
        "hostname": host.get("hostname"),
        "root_tag": root_tag,
        "run_dir": run_dir,
        "run_id": run_id,
        "algo_dir": algo_dir,
        "env_dir": env_dir,
        "sacred_status": run.get("status"),
        "start_time": run.get("start_time"),
        "stop_time": run.get("stop_time"),
        "heartbeat": run.get("heartbeat"),
        "fail_trace": (run.get("fail_trace") or [])[-1:] if run.get("fail_trace") else None,
        "t_last": t_last,
        "t_source": t_source,
        "finished_log": "Finished Training" in tail,
        "cout_mtime": mtime,
        "cfg": cfg_full,
        "env": env_full,
        "param_hash": param_hash(cfg_full, env_full),
    }
    # 学習プロセスの状態。"stopped" = Ctrl-Z 等で一時停止中 / "running" / None = 不明
    # train.py の予約から起動されていれば、その何本目か (batch_pos / batch_total) も付ける
    # LaRe を学習した run は、LaRe のモデル (.pth) も run に結び付けておく
    try:
        rec["lare"] = find_lare_model(repo, env_full, run.get("start_time"),
                                      run.get("stop_time") or run.get("heartbeat"))
    except Exception:                      # 走査そのものは止めない
        rec["lare"] = None
    proc = (find_training_proc(cfg_full, env_full, run.get("start_time"))
            if run.get("status") == "RUNNING" else None)
    rec["proc_state"] = (None if proc is None else
                         "stopped" if proc["stat"].startswith("T") else "running")
    pos = batch_position(proc)
    rec["batch_pos"], rec["batch_total"], rec["batch_pid"] = pos if pos else (None, None, None)
    return rec


def root_tag_of_run(run_dir):
    """<...>/<results|tmp_results>/sacred/<algo>/<env>/<run_id> から "results" を取る."""
    d = os.path.abspath(run_dir)
    sacred_root = os.path.dirname(os.path.dirname(os.path.dirname(d)))
    return _root_tag(sacred_root)


# ---- 学習プロセスの状態 (一時停止の検出) -------------------------------------
# Ctrl-Z で一時停止すると sacred の heartbeat も止まるので、90 分後には「停止 (stalled)」
# と区別できなくなる。ps の状態 (T = stopped) を見て「一時停止中」と判定する。
# sacred の run.json に pid は残らないので、起動時刻と起動引数で run とプロセスを結ぶ
_PROC_CACHE = {"t": 0.0, "procs": None, "ppid": {}}
PS_MAIN_RE = re.compile(r"^(?:\S*/)?python[0-9.]*\s+(?:-\S+\s+)*\S*main\.py\b")


def _training_procs(ttl=30.0):
    """epymarl の main.py (python 本体) のプロセス一覧。同じ走査中は使い回す."""
    now = time.time()
    if _PROC_CACHE["procs"] is not None and now - _PROC_CACHE["t"] < ttl:
        return _PROC_CACHE["procs"]
    procs = []
    try:
        # 日本語ロケールだと lstart の月・曜日が日本語になるので英語に固定する
        # 共有 GPU では他のユーザーも学習を回しているので、自分のプロセスだけを見る
        # (見ないと他人の run を自分の一時停止と取り違える。GPU1 で他ユーザーの main.py 6 本を確認)
        out = subprocess.check_output(["ps", "-U", str(os.getuid()), "-o", "pid=,ppid=,stat=,lstart=,args="],
                                      stderr=subprocess.DEVNULL,
                                      env=dict(os.environ, LC_ALL="C", LANG="C"))
    except (OSError, subprocess.CalledProcessError):
        out = b""
    ppid_of = {}                  # 全プロセスの pid -> 親 pid (train.py までたどるため)
    for line in out.decode("utf-8", "replace").splitlines():
        tok = line.split(None, 8)
        if len(tok) < 9:
            continue
        pid, ppid, stat, args = tok[0], tok[1], tok[2], tok[8]
        ppid_of[pid] = ppid
        # /bin/sh -c python ... のシェルは除き、python 本体だけを見る
        if not PS_MAIN_RE.search(args):
            continue
        try:
            started = time.mktime(time.strptime(" ".join(tok[3:8]), "%a %b %d %H:%M:%S %Y"))
        except ValueError:
            started = None
        procs.append({"pid": pid, "stat": stat, "start": started, "args": args})
    _PROC_CACHE.update(t=now, procs=procs, ppid=ppid_of)
    return procs


def _ancestors(pid, depth=4):
    """pid 自身と、その親を depth 代までたどった pid の列 (文字列)."""
    ppid_of = _PROC_CACHE.get("ppid") or {}
    out, cur = [], str(pid)
    for _ in range(depth + 1):
        if not cur or cur in out or cur in ("0", "1"):
            break
        out.append(cur)
        cur = ppid_of.get(cur)
    return out


def batch_position(proc):
    """学習プロセスが train.py の予約の何本目かを返す. (本目, 総本数, バッチ pid) か None.

    main.py の親 (shell=True なら /bin/sh -c、その親) をたどり、~/.ldrp/batch_<pid>.json
    を書いた train.py を探す。
    - 予約ファイルに "pids" (起動順の Popen の pid) があれば、その並びの位置がそのまま答え
    - 無ければ (古い train.py)、起動済み本数 started から「自分より後に起動して
      まだ動いている同じバッチの run」の数を引く。後から起動した run が先に終わって
      いると数え落とすので、その場合だけ 1 つ大きく出る
    """
    if not proc:
        return None
    chain = _ancestors(proc["pid"])
    d = os.path.expanduser(BATCH_DIR)
    for anc in chain[1:]:
        info = _read_json(os.path.join(d, "batch_%s.json" % anc))
        if not info or not info.get("total"):
            continue
        total = int(info["total"])
        pids = [str(x) for x in info.get("pids") or []]
        for c in chain[:chain.index(anc)]:
            if c in pids:
                return pids.index(c) + 1, total, int(anc)
        started = int(info.get("started") or 0)
        if not started:
            return None
        later = 0
        for p in _training_procs():
            if p is proc or p["start"] is None or proc["start"] is None:
                continue
            if anc in _ancestors(p["pid"])[1:] and p["start"] > proc["start"]:
                later += 1
        return max(1, started - later), total, int(anc)
    return None


def match_training_proc(cfg, env, start_time):
    """sacred の run に対応する学習プロセスの状態を返す.

    "stopped" = 一時停止中 (ps の STAT が T) / "running" / None = 見つからない。
    """
    best = find_training_proc(cfg, env, start_time)
    if best is None:
        return None
    return "stopped" if best["stat"].startswith("T") else "running"


def find_training_proc(cfg, env, start_time):
    """sacred の run に対応する学習プロセス (_training_procs の 1 要素) を探す. 無ければ None.

    起動引数 (アルゴリズム・環境名・t_max) が合うプロセスのうち、起動時刻が sacred の
    start_time に最も近いものを採る。同じ条件を同時に回していても取り違えない。
    """
    algo = (cfg or {}).get("name")
    key = (env or {}).get("key")
    if not algo or not key:
        return None
    t_max = (cfg or {}).get("t_max")
    cand = []
    for p in _training_procs():
        a = p["args"]
        if not re.search(r"--config=%s(\s|$)" % re.escape(str(algo)), a):
            continue
        if str(key) not in a.replace('"', "").replace("'", ""):
            continue
        if t_max and not re.search(r"(^|\s)t_max=%s(\s|$)" % int(t_max), a):
            continue
        cand.append(p)
    if not cand:
        return None
    best = cand[0]
    t0 = None
    if start_time:
        import calendar
        dt = parse_dt(start_time)
        if dt is not None:
            t0 = calendar.timegm(dt.utctimetuple())
    if t0 is not None:
        timed = [p for p in cand if p["start"] is not None]
        if timed:
            # 学習プロセスは sacred の run より**先に**起動する。環境の準備に時間がかかると
            # 差が大きくなる (GPU2 の mappo aoba00 7 台 32 並列で 51 分の実測、2026-10-07)。
            # 以前は「前後 15 分以内」で判定していて、この run を見落としていた。
            # run より後に起動したプロセス (= 別の run) を除き、その中で一番遅く起動したもの
            # (= run の開始に一番近いもの) を採る
            before = [p for p in timed if p["start"] <= t0 + 60]
            if not before:
                return None
            best = max(before, key=lambda p: p["start"])
            if t0 - best["start"] > 6 * 3600:
                # 6 時間以上前に起動したプロセス = 別の run のもの。当て推量はしない
                return None
    return best


BATCH_DIR = "~/.ldrp"
PS_TRAIN_RE = re.compile(r"(?:^|/)python[0-9.]*\s+(?:-\S+\s+)*(\S*train\.py)\b")

# epymarl のコマンド行から key=value を拾う。値は素・"..."・'...' のいずれも取る
CMD_KV_RE = re.compile(r"([A-Za-z_][\w.]*)=(\"[^\"]*\"|'[^']*'|\S+)")
CMD_CONFIG_RE = re.compile(r"--config=(\S+)")


def _cmd_value(raw):
    """コマンド行の値を config.json に載っているのと同じ型に戻す."""
    v = raw.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        return v[1:-1]
    if v in ("True", "False"):
        return v == "True"
    if v in ("None", "null"):
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def parse_train_command(cmd):
    """train.py が起動する epymarl のコマンド行を {cfg, env} に開く.

    `--config=qmix ... t_max=100050000 env_args.key="drp_env:drp_safe-7agent_..."`
    のキー名は config.json のものと同じなので、**derive() をそのまま通せる**形に
    戻せば、条件の判定 (setting / method_tag / task_assign / dynamic) を
    実績の run と同じ 1 か所の実装で行える。

    予約 (まだ sacred のディレクトリが無い run) を実験計画の表に並べるために使う。
    """
    cfg, env = {}, {}
    if not cmd:
        return {"cfg": cfg, "env": env}
    m = CMD_CONFIG_RE.search(cmd)
    if m:
        cfg["name"] = m.group(1)
    for k, raw in CMD_KV_RE.findall(cmd):
        if k.startswith("--"):
            continue
        val = _cmd_value(raw)
        if k.startswith("env_args."):
            env[k[len("env_args."):]] = val
        elif "." not in k:
            cfg[k] = val
    return {"cfg": cfg, "env": env}
ETIME_RE = re.compile(r"^(?:(\d+)-)?(?:(\d+):)?(\d+):(\d+)$")


def _etime_sec(text):
    m = ETIME_RE.match(text.strip())
    if not m:
        return None
    dd, hh, mm, ss = (int(x) if x else 0 for x in m.groups())
    return ((dd * 24 + hh) * 60 + mm) * 60 + ss


def scan_batches(machine):
    """train.py のバッチ実行を拾う.

    2 段構え:
      1. ~/.ldrp/batch_<pid>.json  train.py が書き出した予約 (総本数が分かる)
      2. ps で見つけた train.py    予約が無くても「動いている」ことは分かる

    sacred は run が始まって初めてディレクトリを作るので、これが無いと
    「5 本連続実行の 2 本目まで来た」状態が 1 本しか見えない。
    """
    live = {}
    try:
        # 自分のプロセスだけを見る。共有 GPU で他のユーザーの train.py を
        # 自分の「実行予定」に数えないため
        out = subprocess.check_output(["ps", "-U", str(os.getuid()), "-o", "pid,etime,args"],
                                      stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        out = b""
    for line in out.decode("utf-8", "replace").splitlines()[1:]:
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        pid, etime, args = parts
        if "main.py" in args or not PS_TRAIN_RE.search(args):
            continue
        try:
            live[int(pid)] = (_etime_sec(etime), args.strip()[:200])
        except ValueError:
            continue

    out = []
    d = os.path.expanduser(BATCH_DIR)
    seen = set()
    if os.path.isdir(d):
        for name in sorted(os.listdir(d)):
            if not (name.startswith("batch_") and name.endswith(".json")):
                continue
            info = _read_json(os.path.join(d, name)) or {}
            pid = info.get("pid")
            if pid not in live:
                continue                      # 死んだバッチの置き土産は無視する
            seen.add(pid)
            elapsed, cmd = live[pid]
            out.append({
                "kind": "batch", "uid": "batch:%s:%s" % (machine, pid),
                "machine": machine, "pid": pid, "source": "file",
                "total": info.get("total"), "started": info.get("started"),
                "elapsed_sec": elapsed, "cmd": cmd,
                # train.py が書いた epymarl のコマンド行。これがあると
                # 「どの条件を何本待っているか」まで分かる。条件の判定は
                # 白側で derive() を通して行う (計画表を持っているのは白だけ)
                "train_cmd": info.get("cmd"),
            })
    for pid, (elapsed, cmd) in sorted(live.items()):
        if pid in seen:
            continue
        out.append({
            "kind": "batch", "uid": "batch:%s:%s" % (machine, pid),
            "machine": machine, "pid": pid, "source": "ps",
            "total": None, "started": None, "elapsed_sec": elapsed, "cmd": cmd,
        })
    return out


def _run_text(cmd, timeout=10):
    """コマンドの標準出力を文字列で返す. 無い・失敗したら None (止まらない)."""
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL,
                                       timeout=timeout).decode("utf-8", "replace")
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def _cpu_times():
    """/proc/stat の全 CPU 合計 (busy, total). Linux 以外は None."""
    try:
        with open("/proc/stat") as f:
            v = [int(x) for x in f.readline().split()[1:]]
    except (IOError, OSError, ValueError):
        return None
    idle = v[3] + (v[4] if len(v) > 4 else 0)          # idle + iowait
    return sum(v) - idle, sum(v)


def _my_train_pids(repos):
    """このリポジトリで動いている学習 (epymarl の main.py) と、その子プロセスの PID.

    GPU2 は研究室で 1 つのアカウントを共有しているので、ユーザー名では
    自分の学習を見分けられない。**作業ディレクトリがこのリポジトリの中**かで判定する。
    子プロセス (parallel runner の環境ワーカー) も CPU を食うので一緒に数える。
    Linux (/proc) 以外では、自分のユーザーの main.py を数えるだけにする。
    返り値: (main.py の PID の集合, 子も含めた PID の集合)
    """
    roots = [os.path.realpath(os.path.expanduser(r)) for r in repos or []]
    if not os.path.isdir("/proc"):
        out = _run_text(["ps", "-U", str(os.getuid()), "-o", "pid=,args="]) or ""
        mains = set()
        for ln in out.splitlines():
            f = ln.split(None, 1)
            if len(f) == 2 and "main.py" in f[1] and "epymarl" in f[1]:
                mains.add(int(f[0]))
        return mains, set(mains)
    parent, mains = {}, set()
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        pid = int(name)
        try:
            with open("/proc/%d/stat" % pid) as f:
                st = f.read()
            parent[pid] = int(st[st.rfind(")") + 2:].split()[1])
            with open("/proc/%d/cmdline" % pid, "rb") as f:
                argv = f.read().decode("utf-8", "replace").split("\0")
            # train.py は "/bin/sh -c python ... main.py" で起動するので、
            # sh ではなく python 本体だけを数える
            if not (os.path.basename(argv[0]).startswith("python")
                    and any(a.endswith("main.py") for a in argv[1:3])):
                continue
            cwd = os.path.realpath("/proc/%d/cwd" % pid)
        except (IOError, OSError, ValueError, IndexError):
            continue
        if any(cwd == r or cwd.startswith(r + os.sep) for r in roots):
            mains.add(pid)
    # fork した環境ワーカーも親と同じ cmdline (main.py) を持つ。
    # 親が main.py でないものだけを「学習 1 本」と数える
    allp = set(mains)
    mains = set(p for p in mains if parent.get(p) not in mains)
    for pid in parent:
        q, seen = pid, 0
        while q in parent and q not in allp and seen < 50:
            q, seen = parent[q], seen + 1
        if q in allp:
            allp.add(pid)
    return mains, allp


def host_stats(machine, repos=None):
    """このマシンの CPU / メモリ / GPU の使用状況 ({"kind": "host"} レコード).

    測り方は GPU 側にある sysstat.sh と同じ (nvidia-smi / /proc/stat / meminfo)。
    共有マシンなので「このリポジトリの学習」とそれ以外を分けて数える。
    これがあると「あと何本回せるか」を白のダッシュボードで見積もれる。
    どの値も取れなければ None のまま返す (収集そのものは止めない)。
    """
    mains, mine = _my_train_pids(repos)
    st = {"kind": "host", "uid": "host:%s" % machine, "machine": machine,
          "at": datetime.utcnow().isoformat() + "+00:00",
          "cpu_count": os.cpu_count(), "load1": None, "cpu_pct": None,
          "cpu_pct_mine": None, "mem_total_mb": None, "mem_used_mb": None,
          "train_procs": len(mains), "gpus": [],
          "gpu_mem_mine_mb": None, "gpu_mem_others_mb": None}
    try:
        st["load1"] = round(os.getloadavg()[0], 2)
    except (OSError, AttributeError):
        pass

    # CPU 使用率 (全体): /proc/stat を 0.5 秒あけて 2 回読んだ差 (その瞬間の値)
    a = _cpu_times()
    if a:
        time.sleep(0.5)
        b = _cpu_times()
        if b and b[1] > a[1]:
            st["cpu_pct"] = round(100.0 * (b[0] - a[0]) / (b[1] - a[1]), 1)
    # 自分の学習の CPU。top の 2 回目の表示 (= 直近 0.5 秒の値) を使う。
    # ps の %CPU は起動からの平均なので、長く回している学習では実態とずれる
    out = _run_text(["top", "-b", "-n", "2", "-d", "0.5", "-w", "200"]) if mine else None
    if out and "PID" in out:
        lines = out[out.rfind("PID"):].splitlines()
        head = lines[0].split()
        if "%CPU" in head and head[0] == "PID":
            i = head.index("%CPU")
            tot = 0.0
            for ln in lines[1:]:
                f = ln.split()
                try:
                    if len(f) > i and int(f[0]) in mine:
                        tot += float(f[i].replace(",", "."))
                except ValueError:
                    pass
            if st["cpu_count"]:
                st["cpu_pct_mine"] = round(tot / st["cpu_count"], 1)   # 全体比の %

    # mac には /proc も top -b も無い。ps の %CPU (mac では直近の減衰平均) で代える
    if st["cpu_pct"] is None and st["cpu_count"]:
        out = _run_text(["ps", "-A", "-o", "pid=,%cpu="])
        if out:
            tot = mtot = 0.0
            for ln in out.splitlines():
                f = ln.split()
                try:
                    v = float(f[1].replace(",", "."))
                except (ValueError, IndexError):
                    continue
                tot += v
                if int(f[0]) in mine:
                    mtot += v
            st["cpu_pct"] = round(min(100.0, tot / st["cpu_count"]), 1)
            st["cpu_pct_mine"] = round(mtot / st["cpu_count"], 1)

    # メモリ: Linux は /proc/meminfo、mac は sysctl + vm_stat
    try:
        with open("/proc/meminfo") as f:
            mi = dict((l.split(":")[0], int(l.split()[1])) for l in f if ":" in l)
        st["mem_total_mb"] = mi["MemTotal"] // 1024
        st["mem_used_mb"] = (mi["MemTotal"] - mi.get("MemAvailable", mi.get("MemFree", 0))) // 1024
    except (IOError, OSError, KeyError, ValueError, IndexError):
        tot = _run_text(["sysctl", "-n", "hw.memsize"])
        vm = _run_text(["vm_stat"])
        if tot and vm:
            try:
                page = int(re.search(r"page size of (\d+)", vm).group(1))
                cnt = dict((k.strip(), int(v.strip().rstrip(".")))
                           for k, v in (l.split(":") for l in vm.splitlines()[1:] if ":" in l))
                free = (cnt.get("Pages free", 0) + cnt.get("Pages inactive", 0)
                        + cnt.get("Pages speculative", 0)) * page
                st["mem_total_mb"] = int(tot) // (1 << 20)
                st["mem_used_mb"] = (int(tot) - free) // (1 << 20)
            except (AttributeError, ValueError):
                pass

    # GPU: 全体の使用率と VRAM。WSL2 (GPU2) では nvidia-smi が PATH に無いので、
    # sysstat.sh と同じ場所を見る
    nvsmi = "nvidia-smi"
    if os.path.exists("/usr/lib/wsl/lib/nvidia-smi"):
        nvsmi = "/usr/lib/wsl/lib/nvidia-smi"
    out = _run_text([nvsmi, "--query-gpu=index,name,utilization.gpu,"
                     "memory.used,memory.total,temperature.gpu",
                     "--format=csv,noheader,nounits"])
    for ln in (out or "").splitlines():
        f = [x.strip() for x in ln.split(",")]
        if len(f) < 6:
            continue
        try:
            st["gpus"].append({"index": int(f[0]), "name": f[1], "util": float(f[2]),
                               "mem_used_mb": float(f[3]), "mem_total_mb": float(f[4]),
                               "temp": float(f[5])})
        except ValueError:
            continue
    # プロセスごとの VRAM。WSL2 では "[N/A]" になり分けられない (その場合は None のまま)
    out = _run_text([nvsmi, "--query-compute-apps=pid,used_memory",
                     "--format=csv,noheader,nounits"])
    mine_mb, others_mb, ok = 0.0, 0.0, False
    for ln in (out or "").splitlines():
        f = [x.strip() for x in ln.split(",")]
        try:
            pid, mb = int(f[0]), float(f[1])
        except (ValueError, IndexError):
            continue
        ok = True
        if pid in mine:
            mine_mb += mb
        else:
            others_mb += mb
    if ok:
        st["gpu_mem_mine_mb"], st["gpu_mem_others_mb"] = mine_mb, others_mb
    return st


def iter_run_dirs(sacred_root):
    """<sacred_root>/<algo>/<env_key>/<run_id>/ を列挙する."""
    if not os.path.isdir(sacred_root):
        return
    for algo in sorted(os.listdir(sacred_root)):
        algo_p = os.path.join(sacred_root, algo)
        if not os.path.isdir(algo_p) or algo.startswith("_"):
            continue
        for env_key in sorted(os.listdir(algo_p)):
            env_p = os.path.join(algo_p, env_key)
            if not os.path.isdir(env_p) or env_key.startswith("_"):
                continue
            for run_id in sorted(os.listdir(env_p)):
                run_p = os.path.join(env_p, run_id)
                if run_id.startswith("_") or not os.path.isdir(run_p):
                    continue
                yield run_p


def scan(repos, sacred_subdirs, machine, tail_bytes, emit, model_subdirs=None,
         curve_metrics=None, curve_max_points=0, tb_subdirs=None, with_host=True):
    """repos 配下を走査し、レコードを 1 件ずつ emit に渡す (溜め込まない).

    curve_metrics を渡すと、学習曲線も {"kind":"curve"} レコードとして emit する。
    **抽出は event ファイルのある側で行い、点列だけを持ち帰る**
    (event ファイルは 1 run 5MB あり、運ぶには重い)。
    """
    for b in scan_batches(machine):
        emit(b)
    try:
        if with_host:
            emit(host_stats(machine, repos))
    except Exception as e:                    # 稼働率が取れなくても run の収集は続ける
        sys.stderr.write("[host] %s: %s\n" % (type(e).__name__, e))
    n = 0
    for repo in repos:
        repo = os.path.abspath(os.path.expanduser(repo))
        model_index = build_model_index(
            repo, model_subdirs or list(DEFAULT_MODEL_SUBDIRS))
        tb_index = (build_tb_index(repo, tb_subdirs or list(DEFAULT_TB_SUBDIRS))
                    if curve_metrics else [])
        for sub in sacred_subdirs:
            root = sub if os.path.isabs(sub) else os.path.join(repo, sub)
            if not os.path.isdir(root):
                continue
            tag = _root_tag(root)
            for run_dir in iter_run_dirs(root):
                rec = scan_run_dir(run_dir, machine, tag, tail_bytes,
                                   repo=repo, model_index=model_index)
                if rec is None:
                    continue
                emit(rec)
                n += 1
                if not curve_metrics:
                    continue
                cfg = rec.get("cfg") or {}
                hit = find_tb_dir(tb_index, cfg.get("name") or rec.get("algo_dir"),
                                  cfg.get("seed"),
                                  (rec.get("env") or {}).get("key"))
                if not hit:
                    continue
                token, tb_dir = hit
                sc = read_scalars(tb_dir, curve_metrics, curve_max_points)
                if not sc:
                    continue
                for t, pts in sc.items():
                    emit({"kind": "curve", "uid": rec["uid"], "machine": machine,
                          "token": token, "tag": t, "points": pts})
    return n


def cmd_scan(args):
    """--scan: JSONL を stdout に吐く (リモート実行時のエントリポイント)."""
    def emit(rec):
        sys.stdout.write(json.dumps(rec, ensure_ascii=True, default=str) + "\n")

    if args.scan_dir:
        # 完了検知用の軽い問い合わせ。指定された run だけを読み、
        # モデル索引 (models/ の listdir) は作らない
        for d in args.scan_dir:
            d = os.path.expanduser(d)
            rec = scan_run_dir(d, args.machine, root_tag_of_run(d), args.tail_bytes,
                               repo=None, model_index=None)
            if rec is not None:
                emit(rec)
        sys.stdout.flush()
        return 0

    metrics = set(m for m in (args.scan_curves or "").split(",") if m)
    scan(args.repo, args.sacred_subdir or list(DEFAULT_SACRED_SUBDIRS),
         args.machine, args.tail_bytes, emit,
         model_subdirs=args.model_subdir or list(DEFAULT_MODEL_SUBDIRS),
         curve_metrics=metrics or None,
         curve_max_points=args.curve_max_points,
         tb_subdirs=args.tb_subdir or list(DEFAULT_TB_SUBDIRS))
    sys.stdout.flush()
    return 0


# ===========================================================================
# ここからローカル側 (集約・整形・Notion)
# ===========================================================================

def parse_dt(s):
    """時刻文字列を aware な UTC の datetime にする.

    受ける形は 2 通り:
      - sacred が書く naive UTC   "2026-09-08T07:05:15.123456"
      - こちらが書く aware な UTC "2026-09-08T07:05:15+00:00"

    後者は s[:26] で切ると "+00:00" が残って strptime が落ちるので、
    先に fromisoformat を試す (秒が割り切れる時刻は .%f が付かず 25 文字になる)。
    """
    if not s:
        return None
    from datetime import timezone
    s = str(s).replace("Z", "")
    dt = None
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
            try:
                dt = datetime.strptime(s[:26], fmt)
                break
            except ValueError:
                continue
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def now_utc():
    from datetime import timezone
    return datetime.now(timezone.utc)


def fmt_local(dt):
    if dt is None:
        return ""
    return dt.astimezone().strftime("%m/%d %H:%M")


def fmt_dur(td):
    if td is None:
        return ""
    s = int(td.total_seconds())
    if s < 0:
        return ""
    return "%dh%02dm" % (s // 3600, (s % 3600) // 60)


def pretty_lare_models(name):
    """'FT_QMIX_PATH_Safe_map_8x5_2agents_10.0M_Safe_map_aoba00_2agents_5.0M_...'
    -> '8x5_2 10M→aoba00_2 5M' (元の管理表の setting 列の表記に合わせる)."""
    if not name:
        return ""
    parts = []
    last = None
    for m in LARE_SEG_RE.finditer(str(name)):
        steps = float(m.group(3))
        parts.append("%s_%s %gM" % (m.group(1), m.group(2), steps))
        last = m
    if not parts:
        return str(name)
    # 元の管理表の表記に合わせて矢印で繋ぐ (例: "8x5_2 10M→aoba00_2 5M")
    out = "→".join(parts)
    # 手で編集したモデルの印 (例: ..._checkpoint_waitedit.pth -> "8x5_5 10M waitedit")。
    # 付けないと元のモデルの run と同じ setting・同じ評価用モデル名になる
    mark = lare_edit_mark(str(name)[last.end():])
    return "%s %s" % (out, mark) if mark else out


def lare_edit_mark(tail):
    """LaRe のファイル名の最後の段より後ろ ("_checkpoint_waitedit.pth" など) から、
    編集の印 ("waitedit") を取り出す。印が無ければ ""."""
    t = re.sub(r"\.pth$", "", str(tail or ""))
    t = re.sub(r"^_(?:checkpoint|final)", "", t)
    return t.strip("_ ")


def parse_steps(v):
    """"20.0M" / "500K" / 20000000 のいずれも受ける."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    t = str(v).strip().upper()
    try:
        if t.endswith("M"):
            return float(t[:-1]) * 1e6
        if t.endswith("K"):
            return float(t[:-1]) * 1e3
        return float(t)
    except ValueError:
        return None


def lookup_expected_t_max(table, map_name, agents):
    """条件ごとに想定している t_max を引く. map 別 -> default の順に見る."""
    if not table or agents is None:
        return None
    key = "%sagent" % agents
    for scope in (map_name, "default"):
        sec = table.get(scope) if scope else None
        if isinstance(sec, dict) and key in sec:
            return parse_steps(sec[key])
    return None


def _num(v):
    """config.json の値が文字列で入っていることがあるので数値に寄せる."""
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None


# 観測からこれだけ経ったら「情報が古い」と印を付ける (既定 3 時間)。
# drop は 1 時間ごとのエクスポート + iCloud 同期なので、2 回落としたら気づけるあたり
DATA_STALE_SEC = 3 * 3600


def norm_chain(text):
    """LaRe 系列の表記ゆれを吸収する ("→" と空白のどちらでも同じ扱い)."""
    t = re.sub(r"\s*(?:→|->|=>)\s*", " ", str(text or "")).strip()
    return " ".join(t.split())


def recent_rate(rec, t_ref, t_last):
    """記録点から直近 15 分ぶんのペース (step/sec) を出す。区間が 1 つも無ければ None.

    最新の点から古い方へ区間をたどり、合計が RATE_WINDOW_SEC (15 分) に届くまで
    ステップ数と秒数を足し合わせて割る。
    - 2 分おきの quick-check だと 1 区間が短く、t_env は log 間隔 (5 万 step) 刻みでしか
      進まないので、区間 1 つずつのペースは 0 と 2 倍を行き来する。15 分束ねればならされる
    - 学習マシン側が止まっていた (蓋を閉じた / Ctrl-Z) 区間は heartbeat が飛ぶので、
      「長くて (PAUSE_GAP_SEC 以上) 遅い (普段のペースの半分未満)」区間として飛ばし、
      その前の区間で 15 分を埋める。収集側 (白) が閉じていただけの長い区間は
      ペースが普段どおりなので飛ばさない
    """
    pts = []
    for b, t in rec.get("progress_samples") or []:
        dt = parse_dt(b)
        if dt is not None:
            pts.append((dt, int(t)))
    if t_ref is not None and t_last:
        pts.append((t_ref, int(t_last)))
    pts = sorted(set(pts))
    iv = []                                   # (秒, ステップ) 古い順
    for (b0, t0), (b1, t1) in zip(pts, pts[1:]):
        sec = (b1 - b0).total_seconds()
        if sec > 0 and t1 >= t0:
            iv.append((sec, t1 - t0))
    if not iv:
        return None
    # 普段のペース = 短い区間をまとめたペース。短い区間が無ければ停止の判定はしない
    short = [(s, n) for s, n in iv if s < PAUSE_GAP_SEC]
    s_sum = sum(s for s, _n in short)
    base = (sum(n for _s, n in short) / s_sum) if s_sum > 0 else None
    acc_sec = acc_steps = 0
    for sec, n in reversed(iv):
        if base and sec >= PAUSE_GAP_SEC and n / sec < 0.5 * base:
            continue                          # 学習マシンが止まっていた区間
        acc_sec += sec
        acc_steps += n
        if acc_sec >= RATE_WINDOW_SEC:
            break
    if acc_sec <= 0 or acc_steps <= 0:
        return None
    return acc_steps / acc_sec


def derive(rec, stale_minutes, method_tag_map=None, expected_t_max=None,
           data_stale_sec=DATA_STALE_SEC, lare_chain=None):
    """スキャン結果から Notion 列に対応する派生フィールドを作る."""
    cfg, env = rec.get("cfg") or {}, rec.get("env") or {}
    d = dict(rec)

    key = env.get("key") or rec.get("env_dir") or ""
    m = ENV_KEY_RE.search(str(key))
    d["safe"] = bool(m and m.group(1))
    d["agents"] = int(m.group(2)) if m else None
    d["map"] = m.group(3) if m else str(key)

    d["algo"] = cfg.get("name") or rec.get("algo_dir")
    d["seed"] = cfg.get("seed")
    if d["seed"] in (None, ""):
        d["seed"] = "run%s" % rec.get("run_id")
    t_max = _num(cfg.get("t_max"))
    d["t_max"] = t_max
    d["t_max_m"] = round(t_max / 1e6, 2) if t_max else None
    if not t_max:
        d["group_m"] = None
    elif t_max >= 1e6:
        d["group_m"] = int(round(t_max / 1e6))
    else:
        d["group_m"] = float("%.3g" % (t_max / 1e6))

    # setting 列 (Notion の表記に合わせる: 素なら "safe" / 事前学習ありならその由来).
    # use_lare_path=False のとき pretrained/finetuning フラグは残っていても
    # 無視されるので、必ず use_lare_path を先に見る
    env_label = "safe" if d["safe"] else "unsafe"
    if not env.get("use_lare_path", False):
        d["setting"] = env_label
        d["lare_mode"] = "off"
    elif env.get("use_finetuning_lare_path"):
        d["setting"] = pretty_lare_models(env.get("finetuning_lare_path_model_name"))
        d["lare_mode"] = "finetuning"
    elif env.get("use_pretrained_lare_path"):
        d["setting"] = pretty_lare_models(env.get("pretrained_lare_path_model_name"))
        d["lare_mode"] = "pretrained"
    else:
        d["setting"] = env_label
        d["lare_mode"] = ("scratch" if env.get("use_lare_path_training")
                          else "scratch(frozen)")
    # LaRe-Path の評価関数の版 (env_args.lare_path_factor_version)。版 1 (今まで) は
    # 表記を変えない。2 以降は setting に " F2" のように付け、LaRe なしの run や
    # 版 1 の run と同じ条件に数えられないようにする (design/lare_factor_versions.md)
    fver = int(env.get("lare_path_factor_version") or 1) if env.get("use_lare_path") else 1
    d["lare_factor_version"] = fver
    if fver != 1:
        d["setting"] = "%s F%d" % (d["setting"], fver)
    elif d["lare_mode"] in ("scratch", "scratch(frozen)"):
        # 版 1 の scratch (LaRe を run の中で一から学習) も " F1" を付ける。
        # 付けないと LaRe なしの run と同じ "safe" になり、計画表で区別できない。
        # use_lare_path_training=False (= "scratch(frozen)") でも LaRe のデコーダは学習される
        # (このフラグは方策の報酬を LaRe に差し替えるかだけ)。LaRe モデル学習の run は
        # この設定で回すので、同じく F1 を付ける (2026-10-27)
        d["setting"] = "%s F1" % d["setting"]

    # task arrival 列
    if env.get("randomize_task_arrival"):
        # 学習時の bernoulli p の範囲 U(rand_p_min, rand_p_max) を必ず付ける。
        # 範囲を変えた学習 (plan_arrival_range.md) と区別するため。
        # キーが無い古い run は drp_env の既定 (0.01, 0.10) で回っている
        lo = env.get("rand_p_min")
        hi = env.get("rand_p_max")
        lo = 0.01 if lo is None else float(lo)
        hi = 0.10 if hi is None else float(hi)
        d["task_arrival"] = "bernoulli, mmpp p=%.2f-%.2f" % (lo, hi)
    else:
        d["task_arrival"] = str(env.get("task_arrival") or "")

    # task assign 列
    d["task_assign"] = "PPO" if cfg.get("train_task_assigner") else ""

    d["dynamic_agents"] = bool(env.get("use_dynamic_agents"))
    # 元の管理表の reassign 列 (T/F)
    d["reassign"] = bool(env.get("allow_reassign_before_pickup"))

    # 進捗 / 状態
    t_last = _num(rec.get("t_last"))
    d["t_last"] = t_last
    d["progress"] = (min(t_last / float(t_max), 1.0)
                     if (t_last and t_max) else (1.0 if t_last and not t_max else None))
    start, stop = parse_dt(rec.get("start_time")), parse_dt(rec.get("stop_time"))
    beat = parse_dt(rec.get("heartbeat"))
    d["start_dt"], d["stop_dt"], d["beat_dt"] = start, stop, beat

    # そのホストを最後に観測できた時刻。
    # observed_at が無いのは observed_at 導入前に書かれたキャッシュ。
    # そこで "今" を入れると **鮮度 0 時間** に化けて「届いていない」を隠すので、
    # 状態判定だけ従来どおり now にフォールバックし、鮮度は **不明 (None)** にする
    obs = parse_dt(rec.get("observed_at"))
    d["observed_dt"] = obs
    d["observed_via"] = rec.get("observed_via")
    # 情報そのものの古さ。**run の状態とは別の軸**として持つ。
    # これを state に混ぜると「止まった」と「届いていない」が区別できなくなる
    d["data_age_sec"] = (None if obs is None
                         else max(0.0, (now_utc() - obs).total_seconds()))
    if obs is None:
        obs = now_utc()

    status = (rec.get("sacred_status") or "").upper()
    # epymarl は `while t_env <= t_max` を抜けた直後に "Finished Training" を出す
    # (src/epymarl/src/run.py). t_env のログは log_interval 毎にしか出ないので、
    # 最終行が t_max をわずかに下回るのは正常。1% (最低 50k step) の余裕を見る
    margin = max(t_max * 0.01, 50000) if t_max else 0
    reached = bool(rec.get("finished_log")) or bool(
        t_last and t_max and t_last >= t_max - margin)
    if status == "COMPLETED":
        # sacred の COMPLETED は「学習ループを正常に抜けた」= t_max 到達を意味する。
        # ログが取れていない (cout.txt が空 & metrics も読めない) run はそれを信じる。
        # ログがあるのに t_max に届いていない run だけを short として炙り出す
        if reached or t_last is None:
            d["state"] = "done"
            d["progress"] = 1.0 if reached else d["progress"]
        else:
            d["state"] = "short"
    elif status in ("FAILED", "INTERRUPTED"):
        d["state"] = "failed"
    elif status == "RUNNING":
        last = beat or stop or start
        # **観測時刻**を基準にする。"今" と比べると、drop ホストのエクスポートが
        # 遅れたぶんだけ heartbeat が古く見え、走っている run が stalled に化ける
        stale = (last is None) or (obs - last > timedelta(minutes=stale_minutes))
        d["state"] = "stalled" if stale else "running"
        # Ctrl-Z 等で一時停止中なら heartbeat が止まっていても「実行中」のまま扱う。
        # 再開すれば続きから進むので、止まった (stalled) と同じにすると枠が空いて見え、
        # 同じ条件を二重に投入してしまう
        if rec.get("proc_state") == "stopped":
            d["state"] = "running"
    else:
        d["state"] = "unknown"

    # 観測が古いホストの run は「そう見えているだけ」かもしれない、と印を付ける。
    # state は変えない (集計やモデル公開のゲートに波及させないため)
    d["stale_data"] = bool(d["state"] in ("running", "stalled")
                           and d["data_age_sec"] is not None
                           and d["data_age_sec"] > data_stale_sec)

    if method_tag_map is None:
        tag_map = DEFAULT_METHOD_TAG_BY_LARE_MODE
    else:
        # YAML では裸の off / on が bool になるのでキーを文字列に正規化する
        tag_map = dict(("off" if k is False else "on" if k is True else str(k), v)
                       for k, v in method_tag_map.items())
    if not d["safe"]:
        d["method_tag"] = "unsafe"
    else:
        d["method_tag"] = tag_map.get(d["lare_mode"], "")
        # map ごとに「正規の LaRe 系列」を決めてあるなら、それ以外の系列は
        # 別のタグにする。全部 dbct にすると、事前学習の経路が違うのに
        # 同じファイル名になって取り違える (実測で 3 stem が衝突していた)
        canon = (lare_chain or {}).get(d.get("map"))
        if canon and d["lare_mode"] in ("pretrained", "finetuning"):
            if norm_chain(d.get("setting")) != norm_chain(canon):
                d["method_tag"] = "%s_%s" % (d["method_tag"] or "dbct",
                                             _slug(d.get("setting"), "alt"))
    # LaRe-Path の評価関数の版 2 以降は、評価用モデルの名前のタグにも版を付ける
    # (例: safe -> safe_f2)。付けないと LaRe なし / 版 1 のモデルと同じ名前になる
    # 版 1 の scratch も _f1 を付ける (LaRe なしの評価用モデルと同じ名前にしないため)
    if (d.get("lare_factor_version", 1) != 1
            or d.get("lare_mode") in ("scratch", "scratch(frozen)")):
        d["method_tag"] = "%s_f%d" % (d["method_tag"] or "safe", d.get("lare_factor_version", 1))
    # 学習時の到着率の範囲が既定 U(0.01, 0.10) と違う run は、タグに範囲を足す
    # (例: safe -> safe_p0.01-0.15)。足さないと評価用のファイル名が既定の run と
    # 同じ stem になり、範囲の違いが seed の番号でしか区別できない。評価の条件名にも
    # タグが入るので、集計で別の範囲の結果が同じ条件の seed として平均されなくなる
    # (tools/plans/plan_arrival_range.md)
    if env.get("randomize_task_arrival"):
        lo = env.get("rand_p_min")
        hi = env.get("rand_p_max")
        lo = 0.01 if lo is None else float(lo)
        hi = 0.10 if hi is None else float(hi)
        if (lo, hi) != (0.01, 0.10):
            d["method_tag"] = "%s_p%g-%g" % (d["method_tag"] or "safe", lo, hi)

    # 想定 t_max との照合. モデルの選択には使わない (使うのは run の t_max) が、
    # 「30M 回すつもりだったのに 10M で止めた run」を取り違えないための確認用
    exp = lookup_expected_t_max(expected_t_max, d.get("map"), d.get("agents"))
    d["t_max_expected"] = exp
    if exp is None or not t_max:
        d["t_max_ok"] = None
    else:
        # t_max は 20050000 のように +50000 されている運用なので M 単位で丸めて比べる
        d["t_max_ok"] = round(t_max / 1e6) == round(exp / 1e6)

    # ハッシュは手元の param_fields で計算し直す。drop のホストは自分の (古いかもしれない)
    # collect_runs.py で計算して書き出すので、照合の規則を変えたときに食い違う
    d["param_hash"] = (param_hash(rec.get("cfg"), rec.get("env"))
                       if rec.get("cfg") is not None else rec.get("param_hash"))
    d["model"] = rec.get("model")
    d["lare"] = rec.get("lare")
    d["curve_file"] = rec.get("curve_file")      # 共有フォルダ経由の学習曲線 (export_curves_drop)
    d["repo"] = rec.get("repo")
    d["duration"] = fmt_dur((stop or beat or now_utc()) - start) if start else ""
    d["last_seen"] = beat or stop or start

    # 学習終了予定時刻. 「開始 -> 最後に進捗が分かった時点」で実測ペースを出し、
    # 残りステップを割る。heartbeat と start_time はどちらも sacred が書く UTC なので
    # 時計系が揃う (cout.txt の mtime はリモートの時計なので使わない)。
    # 進行中の run 以外は出さない (停止した run のペースから作った予定時刻は誤解を招く)。
    d["rate"] = None            # step/sec
    d["remaining_sec"] = None
    d["eta"] = None
    d["rate_source"] = None     # "recent" = 直近の区間 / "average" = 開始からの平均
    d["paused"] = bool(rec.get("proc_state") == "stopped" and d["state"] == "running")
    t_ref = beat or stop
    if d["state"] == "running" and start and t_ref and t_last and t_max:
        elapsed = (t_ref - start).total_seconds()
        if elapsed > 0 and t_last > 0 and t_last < t_max:
            # 直近の区間ペースを優先する。開始からの平均は、一時停止やスリープで
            # 止まっていた時間まで含むので、終了予定が何日も先に出てしまう
            rate = recent_rate(rec, t_ref, t_last)
            d["rate_source"] = "recent" if rate else None
            if not rate:
                rate = t_last / elapsed
                d["rate_source"] = "average"
            if rate > 0:
                d["rate"] = rate
                d["remaining_sec"] = (t_max - t_last) / rate
                # 一時停止中は「いつ再開するか」が分からないので時刻は出さない
                # (残り時間だけ出す。再開すればその時点から残り時間で終わる)
                if not d["paused"]:
                    d["eta"] = t_ref + timedelta(seconds=d["remaining_sec"])
    # map を必ず含める。入れないと 3agent 8M iql が map_5x4 / map_8x5 / map_aoba01 を
    # 1 条件に混ぜてしまい、「done seeds」が別マップ混在の一覧になる
    d["condition"] = "%sagent %s %sM | %s | %s | %s | %s" % (
        d["agents"], d["map"], d["group_m"], d["algo"], d["setting"],
        d["task_arrival"], d["task_assign"] or "-")
    # 再割当ありは別条件。入れないと、再割当なしで 5 seed そろった条件に
    # 再割当ありの run が混ざり、params✗ (同じ条件なのに設定が違う) の判定で
    # どちらかが少数派扱いになる。なしのときは何も足さない (既存の条件名と互換)
    if d["reassign"]:
        d["condition"] += " | reassign"
    return d


# ---------------------------------------------------------------------------
# 収集 (local / ssh)
# ---------------------------------------------------------------------------

try:
    from shlex import quote as _quote
except ImportError:                                   # Python 2 系リモート向けの保険
    from pipes import quote as _quote


def _script_source():
    with open(os.path.abspath(__file__), "rb") as f:
        return f.read()


# ---- 学習曲線 (sacred の metrics.json) -----------------------------------------
# ダッシュボードの「学習曲線」タブ用。TensorBoard の event ファイルは読まない
# (リモートに tensorboard が無いと黙って空になる)。sacred の metrics.json は
# 1 run 数十 MB あるので、**読むのは変わった run だけ**にする (fetch_metric_curves の
# キャッシュ、export_drop の状態ファイル)。持ち帰るのは間引いた点列だけ
CURVE_KEYS = ("test_task_completion_mean", "test_goal_mean",
              "test_collision_mean", "test_return_mean")
CURVE_MAX_POINTS = 600
CURVE_CACHE_NAME = ".curve_cache.json"


def _curve_val(v):
    """metrics.json の値は普通は float。numpy が jsonpickle された形 ({"value": ...}) も受ける."""
    if isinstance(v, dict):
        v = v.get("value", v.get("py/reduce"))
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def read_metric_curves(run_dir, keys=CURVE_KEYS, max_points=CURVE_MAX_POINTS):
    """{指標: [[t_env, ...], [値, ...]]} を返す. metrics.json が読めなければ None."""
    try:
        with open(os.path.join(run_dir, "metrics.json")) as f:
            m = json.load(f)
    except (OSError, IOError, ValueError):
        return None
    out = {}
    for k in keys:
        e = m.get(k)
        if not isinstance(e, dict):
            continue
        st, va = e.get("steps") or [], e.get("values") or []
        n = min(len(st), len(va))
        if not n:
            continue
        if n > max_points > 1:
            idx = sorted(set(int(round(i * (n - 1) / float(max_points - 1)))
                             for i in range(max_points)))
        else:
            idx = range(n)
        out[k] = [[st[i] for i in idx], [_curve_val(va[i]) for i in idx]]
    return out


def cmd_dump_curves(dirs):
    """--dump-curves: run ディレクトリごとに 1 行の JSON を吐く (ssh 越しに呼ぶ)."""
    for d in dirs:
        d = os.path.expanduser(d)
        sys.stdout.write(json.dumps({"run_dir": d, "m": read_metric_curves(d)},
                                    ensure_ascii=True) + "\n")
    sys.stdout.flush()
    return 0


def _curve_file_name(uid):
    import hashlib
    return "curves/%s.json" % hashlib.sha1(uid.encode("utf-8")).hexdigest()[:16]


def export_curves_drop(records, drop_dir, state_path=None, dry_run=False):
    """共有フォルダのホスト (黒) 側: 学習曲線を drop_dir/curves/ に書き出す.

    metrics.json の更新時刻が前回の書き出しから変わった run だけを読み直す
    (状態は state_path に持つ)。書き出した場所は rec["curve_file"] に入れ、
    runs.jsonl に載せて集約側 (白) に伝える。
    """
    state_path = os.path.expanduser(state_path or "~/.ldrp/curve_export_state.json")
    try:
        with open(state_path) as f:
            state = json.load(f)
    except (OSError, IOError, ValueError):
        state = {}
    n_new = 0
    for rec in records:
        if rec.get("kind") in ("batch", "host") or not rec.get("run_dir"):
            continue
        p = os.path.join(rec["run_dir"], "metrics.json")
        try:
            mt = os.path.getmtime(p)
        except OSError:
            continue
        rel = _curve_file_name(rec["uid"])
        dst = os.path.join(drop_dir, rel)
        if state.get(rec["uid"]) == mt and os.path.exists(dst):
            rec["curve_file"] = rel
            continue
        if dry_run:
            n_new += 1
            continue
        m = read_metric_curves(rec["run_dir"])
        if not m:
            continue
        try:
            os.makedirs(os.path.dirname(dst))
        except OSError:
            pass
        tmp = dst + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"uid": rec["uid"], "m": m}, f)
        os.replace(tmp, dst)
        state[rec["uid"]] = mt
        rec["curve_file"] = rel
        n_new += 1
    if not dry_run:
        try:
            os.makedirs(os.path.dirname(state_path))
        except OSError:
            pass
        with open(state_path + ".tmp", "w") as f:
            json.dump(state, f)
        os.replace(state_path + ".tmp", state_path)
    sys.stderr.write("[export] %d learning curve(s) updated%s\n"
                     % (n_new, " (dry run)" if dry_run else ""))
    return n_new


def fetch_metric_curves(hosts, rows, cache_path, drop_root=None, timeout=600):
    """集約側 (白): rows の run の学習曲線を集めて cache_path に貯める.

    取り直すのは「まだ無い」「前回から進んだ (t_last が変わった)」「状態が変わった」run だけ。
    返り値: キャッシュ全体 {uid: {"t": t_last, "state": 状態, "m": {指標: [[x], [y]]}}}
    """
    try:
        with open(cache_path) as f:
            cache = json.load(f)
    except (OSError, IOError, ValueError):
        cache = {}
    host_by_label = dict((h["label"], h) for h in hosts)
    need = {}
    for d in rows:
        if d.get("state") not in ("done", "running", "stalled") or not d.get("run_dir"):
            continue
        c = cache.get(d["uid"])
        if c and c.get("t") == d.get("t_last") and c.get("state") == d.get("state"):
            continue
        need.setdefault(d.get("machine"), []).append(d)
    got = 0
    for label, ds in need.items():
        host = host_by_label.get(label)
        if host is None:
            continue
        res = {}                                   # uid -> m
        if host.get("drop"):
            base = drop_dir_for(host, drop_root)
            for d in ds:
                if not base or not d.get("curve_file"):
                    continue                       # 相手のスクリプトが旧版 (まだ書き出していない)
                p = os.path.join(base, d["curve_file"])
                try:
                    tmp = p + ".read"
                    copy_cloud_file(p, tmp)
                    with open(tmp) as f:
                        res[d["uid"]] = json.load(f).get("m")
                    os.remove(tmp)
                except (OSError, IOError, ValueError):
                    continue
        elif host.get("ssh") in (None, "", "local"):
            for d in ds:
                res[d["uid"]] = read_metric_curves(d["run_dir"])
        else:
            by_dir = dict((d["run_dir"], d["uid"]) for d in ds)
            dirs = list(by_dir)
            for i in range(0, len(dirs), 40):      # 引数が長くなりすぎないよう分ける
                args = []
                for d in dirs[i:i + 40]:
                    args += ["--dump-curves", d]     # action="append" なので 1 つずつ付ける
                recs, err = _collect_ssh_run(host, host.get("python") or "python3",
                                             args, timeout, False, False)
                if err:
                    sys.stderr.write("[curves] %s: %s\n" % (label, err[:300]))
                    break
                for r in recs:
                    if r.get("run_dir") in by_dir:
                        res[by_dir[r["run_dir"]]] = r.get("m")
        for d in ds:
            if d["uid"] not in res:
                continue                           # 届かなかった。次の収集で取り直す
            if not res[d["uid"]] and d.get("state") == "running":
                continue                           # 書き込み中の metrics.json を読んだだけ。次で取り直す
            # metrics.json が無い・壊れている run も「読んだ」と覚える (毎回取りに行かない)。
            # 進めば t_last が変わるので取り直される
            cache[d["uid"]] = {"t": d.get("t_last"), "state": d.get("state"),
                               "m": res[d["uid"]] or {}}
            got += 1
    if got:
        tmp = cache_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(cache, f)
        os.replace(tmp, cache_path)
    if need:
        sys.stderr.write("[curves] updated %d / %d run(s)\n"
                         % (got, sum(len(v) for v in need.values())))
    return cache


def collect_local(host, tail_bytes, curves=None, curve_max_points=0):
    recs = []
    scan(host.get("repos") or ["."],
         host.get("sacred_subdirs") or list(DEFAULT_SACRED_SUBDIRS),
         host["label"], tail_bytes, recs.append,
         model_subdirs=host.get("model_subdirs") or list(DEFAULT_MODEL_SUBDIRS),
         curve_metrics=curves, curve_max_points=curve_max_points,
         tb_subdirs=host.get("tb_subdirs") or list(DEFAULT_TB_SUBDIRS),
         # 返却済みマシンのバックアップ (M2) を白で読むときは、白の稼働率を
         # M2 の値として出さない
         with_host=host.get("host_stats", True))
    return recs, None


def multiplex_options(enabled):
    """ssh の接続多重化オプション.

    完了チェックのように短い問い合わせを繰り返すとき、毎回 TCP と認証をやり直すと
    高負荷のホストでは数十秒かかる。マスタ接続を使い回すと 2 回目以降はほぼ無料になる。
    ControlPersist で放置すれば自然に閉じるので、常駐プロセスは残さない。
    """
    if not enabled:
        return []
    sock = os.path.expanduser("~/.ssh/.ldrp-cm-%r@%h-%p")
    return ["-o", "ControlMaster=auto", "-o", "ControlPath=" + sock,
            "-o", "ControlPersist=10m"]


def collect_ssh(host, tail_bytes, timeout, verbose=False, only_dirs=None,
                multiplex=False, curves=None, curve_max_points=0):
    """このスクリプト自身を stdin で送り込み、リモートで --scan させる."""
    quote = _quote
    py = host.get("python") or "python3"
    remote_args = ["--scan", "--machine", host["label"],
                   "--tail-bytes", str(tail_bytes)]
    if curves:
        # 抽出はリモートで行い、点列だけ持ち帰る (event ファイルは 1 run 5MB)
        remote_args += ["--scan-curves", ",".join(sorted(curves))]
        if curve_max_points:
            remote_args += ["--curve-max-points", str(curve_max_points)]
        for sub in host.get("tb_subdirs") or []:
            remote_args += ["--tb-subdir", sub]
    if only_dirs:
        for d in only_dirs:
            remote_args += ["--scan-dir", d]
        return _collect_ssh_run(host, py, remote_args, timeout, verbose, multiplex)
    for r in host.get("repos") or ["~/LDRP"]:
        remote_args += ["--repo", r]
    for sub in host.get("sacred_subdirs") or []:
        remote_args += ["--sacred-subdir", sub]
    for sub in host.get("model_subdirs") or []:
        remote_args += ["--model-subdir", sub]
    return _collect_ssh_run(host, py, remote_args, timeout, verbose, multiplex)


def pull_eval_logs(hosts, repo=None, timeout=120, verbose=False, drop_root=None):
    """別マシンで回した方策評価のログ (*.txt) を手元の logs/ に取ってくる.

    collect_config.yaml の host に `eval_logs: ~/LDRP_eval/logs` のように書いたものだけが対象。
    共有フォルダのマシン (drop) は、tools/autorun_worker.py が <drop_root>/<label>/autorun/eval_logs/ に
    写したものを取る (eval_logs: の指定は要らない)。
    ダッシュボードは手元の logs/ の [RESULT] 行しか読まないので、取ってくれば評価表に出る。
    何か新しく届いたときだけ aggregate.py --csv で results/summary.csv を作り直す。
    返り値: [(host label, エラー文)]
    """
    repo = repo or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    errors, changed = [], False
    for h in hosts:
        src = h.get("eval_logs")
        dd = drop_dir_for(h, drop_root) if h.get("ssh") in (None, "") else None
        if dd:
            src = os.path.join(dd, "autorun", "eval_logs")
            if not os.path.isdir(src):
                continue                    # まだ一度も評価を回していないマシン
            cmd = ["rsync", "-a", "--itemize-changes", "--include", "*/", "--include", "*.txt",
                   "--exclude", "*", src + "/", os.path.join(repo, "logs") + "/"]
        elif not src or h.get("ssh") in (None, "", "local"):
            continue
        else:
            rel = src[2:] if src.startswith("~/") else src       # rsync のリモート側は ~ を展開しない
            ssh = ["ssh", "-o", "BatchMode=yes"]
            for opt in h.get("ssh_options") or []:
                ssh += ["-o", opt]
            ssh += ["-o", "ConnectTimeout=%d" % int(h.get("connect_timeout", 8))]
            if h.get("port"):
                ssh += ["-p", str(h["port"])]
            cmd = ["rsync", "-az", "--itemize-changes", "-e", " ".join(ssh),
                   "--include", "*/", "--include", "*.txt", "--exclude", "*",
                   "%s:%s/" % (h["ssh"], rel.rstrip("/")), os.path.join(repo, "logs") + "/"]
        if verbose:
            sys.stderr.write("  $ %s\n" % " ".join(cmd))
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except (subprocess.TimeoutExpired, OSError) as e:
            errors.append((h.get("label"), "eval log pull: %s" % e))
            continue
        if p.returncode == 23 and "No such file or directory" in p.stderr:
            continue                        # まだ一度も評価を回していないマシン
        if p.returncode != 0:
            errors.append((h.get("label"), "eval log pull: rsync rc=%d %s"
                           % (p.returncode, p.stderr.strip()[-200:])))
            continue
        # 届いたファイルは "<f+++" / ">f.st" のように f で始まる行になる
        got = [l for l in p.stdout.splitlines() if l[1:2] == "f"]
        if got:
            changed = True
            sys.stderr.write("[eval-logs] %s: %d file(s) updated\n" % (h.get("label"), len(got)))
    if changed:
        subprocess.run([sys.executable, os.path.join(repo, "aggregate.py"),
                        "--root", os.path.join(repo, "logs"),
                        "--csv", os.path.join(repo, "results", "summary.csv")],
                       cwd=repo, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return errors


def _collect_ssh_run(host, py, remote_args, timeout, verbose, multiplex):
    quote = _quote
    remote_cmd = "%s - %s" % (py, " ".join(quote(a) for a in remote_args))

    # ssh は「最初に指定されたオプションが勝つ」ので、host 側の指定を先に置く
    cmd = ["ssh", "-o", "BatchMode=yes"]
    for opt in host.get("ssh_options") or []:
        cmd += ["-o", opt]
    cmd += ["-o", "ConnectTimeout=%d" % int(host.get("connect_timeout", 8))]
    cmd += multiplex_options(multiplex)
    if host.get("port"):
        cmd += ["-p", str(host["port"])]
    cmd += [host["ssh"], remote_cmd]

    if verbose:
        sys.stderr.write("  $ %s\n" % " ".join(cmd))
    try:
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate(_script_source(), timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        return [], "ssh timeout (%ss)" % timeout
    except OSError as e:
        return [], "ssh failed: %s" % e

    recs = []
    for line in out.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                recs.append(json.loads(line))
            except ValueError:
                pass
    if p.returncode != 0 and not recs:
        msg = err.decode("utf-8", "replace").strip().splitlines()
        return [], "exit %s: %s" % (p.returncode, msg[-1] if msg else "")
    return recs, None


def drop_dir_for(host, drop_root):
    """host が読み書きする共有フォルダのパスを決める."""
    d = host.get("drop")
    if d in (None, False, ""):
        return None
    base = drop_root if d is True else d
    if not base:
        return None
    return os.path.join(os.path.abspath(os.path.expanduser(base)), str(host["label"]))


def collect_drop(host, drop_root, stale_hours=24):
    """共有フォルダに置かれた runs.jsonl を読む (SSH の要らないホスト用)."""
    d = drop_dir_for(host, drop_root)
    if d is None:
        return [], "no drop directory configured"
    path = os.path.join(d, "runs.jsonl")
    if not os.path.exists(path):
        return [], "no runs.jsonl in %s (has that machine run --export yet?)" % d

    # iCloud は「ストレージを最適化」で実体を退避する (ls -lO が dataless)。
    # 読めば自動で落ちてくるが、時間がかかったり失敗したりするので、
    # 先に明示的にダウンロードを要求しておく。brctl が無い環境では何もしない
    try:
        subprocess.call(["brctl", "download", path],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        pass

    # iCloud が同期している最中のファイルを読むと Errno 11 (Resource deadlock avoided)
    # で落ちる。以前はこれで収集全体が止まっていた (launchd のログに 334 回)。
    # 少し待って読み直し、それでもだめならこのホストだけ諦めて次の回に回す
    recs = None
    for attempt in range(3):
        try:
            got = []
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("{"):
                        try:
                            got.append(json.loads(line))
                        except ValueError:
                            pass
            recs = got
            break
        except OSError as e:
            err = e
            time.sleep(2 * (attempt + 1))
    if recs is None:
        return [], "cannot read %s (iCloud busy?): %s" % (path, err)
    mtime = os.path.getmtime(path)
    # 書き出された時刻をそのまま観測時刻にする。相手が --export した瞬間の
    # スナップショットなので、それ以降の経過は「情報が届いていない」であって
    # 「run が止まった」ではない
    from datetime import timezone
    stamp_observed(recs, datetime.fromtimestamp(mtime, timezone.utc), "drop")
    age_h = (now_utc().timestamp() - mtime) / 3600.0
    if stale_hours and age_h > stale_hours:
        sys.stderr.write("[warn] %s: drop data is %.1f h old (that machine may have "
                         "stopped exporting, or the folder is not syncing)\n"
                         % (host["label"], age_h))
    return recs, None


def export_drop(records, rows, drop_dir, what="path", with_optimizer=False,
                max_mb=2000.0, dry_run=False, with_mixer=False,
                with_critic=False):
    """自分の run 一覧とモデルを共有フォルダへ書き出す (--export).

    共有フォルダは iCloud Drive / Dropbox / NFS / USB など何でもよい。
    集約側はこのフォルダを読むだけなので、SSH は要らない。
    """
    import shutil
    by_uid = dict((d["uid"], d) for d in rows)
    models_root = os.path.join(drop_dir, "models")
    copied, total = 0, 0.0

    # --- 1 巡目: 何をコピーするか決めるだけ (まだ書かない) -------------------
    # export_dir は runs.jsonl に載せる必要があるので、ここで確定させる
    todo = []
    for rec in records:
        d = by_uid.get(rec.get("uid"))
        model = rec.get("model")
        if d is None or not model:
            continue
        files = select_model_files(model, what, with_optimizer, with_mixer,
                                   with_critic)
        if not files:
            continue
        dest_name = model_dest_name(d)
        out_dir = os.path.join(models_root, dest_name)
        # 集約側が「共有フォルダのどこを見ればよいか」を辿れるようにしておく
        rec["model"]["export_dir"] = os.path.join("models", dest_name)
        if os.path.isdir(out_dir) and os.listdir(out_dir):
            continue
        size_mb = sum(z for _r, _s, z in files) / 1e6
        if max_mb and total + size_mb > max_mb:
            continue
        total += size_mb
        copied += 1
        todo.append((rec, out_dir, files))

    # LaRe-Path のモデル (.pth)。1 本 25KB 程度なので容量の上限には数えない。
    # ファイル名に条件が全部入っているので、共有フォルダでは lare/ の下に平らに置く。
    # export_path は runs.jsonl に載せるので、進捗ファイルより先に決める
    lare_todo = []
    for rec in records:
        lr = rec.get("lare")
        if not lr or rec.get("uid") not in by_uid:
            continue
        lr["export_path"] = os.path.join("lare", lr["name"])
        dst = os.path.join(drop_dir, lr["export_path"])
        try:
            if os.path.getsize(dst) == lr.get("size"):
                continue                      # 同じものが置いてある
        except OSError:
            pass
        lare_todo.append((os.path.join(rec.get("repo") or ".", lr["rel"]), dst))

    # 学習曲線 (ダッシュボードの「学習曲線」タブ用)。書き出した場所を runs.jsonl に
    # 載せるので、進捗ファイルより先に済ませる。変わった run だけ読むので 2 回目以降は軽い
    try:
        export_curves_drop(records, drop_dir, dry_run=dry_run)
    except Exception as e:                    # 曲線が書けなくても進捗は必ず届ける
        sys.stderr.write("[export] curves: %s: %s\n" % (type(e).__name__, e))

    # --- 進捗ファイルを **先に** 書く ---------------------------------------
    # 440KB の runs.jsonl を 90MB のモデル転送の後ろに置くと、iCloud の
    # アップロードが詰まったときに進捗だけ先に届けられない。実測で
    # 「黒の進捗が 3 時間遅れる」原因になっていたのでこの順序にする。
    # モデルのコピーが途中で失敗しても、進捗は必ず最新になる
    if not dry_run:
        try:
            os.makedirs(drop_dir)
        except OSError:
            pass
        tmp = os.path.join(drop_dir, "runs.jsonl.tmp")
        with open(tmp, "w") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=True, default=str) + "\n")
        os.replace(tmp, os.path.join(drop_dir, "runs.jsonl"))

    # --- 2 巡目: モデルを運ぶ (時間がかかってよい) --------------------------
    moved = 0
    if not dry_run:
        for rec, out_dir, files in todo:
            try:
                for rel, sub, _size in files:
                    src = os.path.join(rec.get("repo") or ".", rel)
                    dst = os.path.join(out_dir, sub)
                    try:
                        os.makedirs(os.path.dirname(dst))
                    except OSError:
                        pass
                    shutil.copy2(src, dst)
                moved += 1
            except (OSError, IOError) as e:
                # 1 件失敗しても残りは運ぶ。次回の --export で拾い直される
                sys.stderr.write("[export] skipped %s: %s\n"
                                 % (os.path.basename(out_dir), e))

    lare_moved = 0
    if not dry_run:
        for src, dst in lare_todo:
            try:
                try:
                    os.makedirs(os.path.dirname(dst))
                except OSError:
                    pass
                shutil.copy2(src, dst)
                lare_moved += 1
            except (OSError, IOError) as e:
                sys.stderr.write("[export] skipped LaRe %s: %s\n"
                                 % (os.path.basename(dst), e))

    sys.stderr.write("[export] %d run(s), %d model dir(s) newly copied (%.1f MB), "
                     "%d LaRe model(s) -> %s%s\n"
                     % (len(records), moved if not dry_run else copied,
                        total, lare_moved if not dry_run else len(lare_todo),
                        drop_dir, " (dry run)" if dry_run else ""))
    return copied


def stamp_observed(recs, when, source):
    """レコードに **そのホストを観測できた時刻** を刻む.

    状態判定の基準をここに揃える。"今" で判定すると、drop ホストの
    エクスポートが遅れたぶんだけ heartbeat が古く見え、健全に走っている run が
    stalled に化ける (実測: エクスポート間隔 60 分に対ししきい値 90 分で余裕 30 分)。
    観測時刻で判定すれば、情報が届かなくなっても**最後に見えた時点の判定で凍る**。

    リモート側のスクリプトは触らない。drop はファイルの mtime、ssh/local は
    収集した瞬間を白側で刻むので、黒や M2 が旧版のままでも動く。
    """
    iso = when.isoformat()
    for r in recs:
        r["observed_at"] = iso
        r["observed_via"] = source
    return recs


# 到達できなかった ssh ホストを、しばらく試さないでおくための覚え書き。
# VPN を切っている間は毎回 connect_timeout 秒ずつ待たされるので、
# 1 度失敗したら次はしばらく飛ばす (プロセス内でだけ効く)
_SSH_BACKOFF = {}
SSH_BACKOFF_SEC = 600


def collect(hosts, tail_bytes, timeout, verbose=False, drop_root=None,
            skip_unchanged=None, curves=None, curve_max_points=0):
    """各ホストから run を集める.

    skip_unchanged: {label: mtime} を渡すと、共有フォルダの runs.jsonl が
    その時刻から変わっていないホストを読み飛ばす (stat だけで済む)。
    相手は 1 時間おきにしか書かないので、毎回読み直す意味がない。
    """
    all_recs, errors = [], []
    now = time.time()
    for host in hosts:
        label = host["label"]
        if host.get("drop"):
            if skip_unchanged is not None:
                d = drop_dir_for(host, drop_root)
                f = os.path.join(d, "runs.jsonl") if d else None
                try:
                    mt = os.path.getmtime(f) if f else None
                except OSError:
                    mt = None
                if mt is not None and skip_unchanged.get(label) == mt:
                    continue                  # 前回から変わっていない
                if mt is not None:
                    skip_unchanged[label] = mt
            recs, err = collect_drop(host, drop_root)
        elif host.get("ssh") in (None, "", "local"):
            recs, err = collect_local(host, tail_bytes, curves, curve_max_points)
            stamp_observed(recs, now_utc(), "local")
        else:
            until = _SSH_BACKOFF.get(label, 0)
            if until > now:
                errors.append((label, "skipped for %d more s (last attempt failed)"
                               % int(until - now)))
                continue
            recs, err = collect_ssh(host, tail_bytes, timeout, verbose,
                                    curves=curves,
                                    curve_max_points=curve_max_points)
            stamp_observed(recs, now_utc(), "ssh")
            if err and not recs:
                _SSH_BACKOFF[label] = now + SSH_BACKOFF_SEC
            else:
                _SSH_BACKOFF.pop(label, None)
        if err:
            errors.append((label, err))
            sys.stderr.write("[warn] %s: %s\n" % (label, err))
        else:
            sys.stderr.write("[info] %s: %d runs\n" % (label, len(recs)))
        all_recs.extend(recs)
    return all_recs, errors


# ---------------------------------------------------------------------------
# 出力
# ---------------------------------------------------------------------------

TABLE_COLS = [
    ("state", "state", 6), ("seed", "seed", 10), ("machine", "machine", 8),
    ("agents", "N", 3), ("map", "map", 10), ("algo", "algo", 9),
    ("setting", "setting", 22), ("task_arrival", "arrival", 16),
    ("task_assign", "assign", 6), ("prog", "progress", 18),
    ("duration", "elapsed", 8), ("eta", "eta", 20), ("last", "last seen", 12),
]


def _prog_cell(d):
    t_last, t_max = d.get("t_last"), d.get("t_max")
    if not t_max:
        return "-"
    return "%5.1f%% %s/%sM" % ((d["progress"] or 0) * 100,
                               ("%.2f" % (t_last / 1e6)) if t_last else "?",
                               "%g" % (t_max / 1e6))


def eta_cell(d):
    """残り時間と終了予定時刻を 1 セルにまとめる (例: "12h30m -> 09/01 03:40")."""
    if d.get("eta") is None:
        return ""
    return "%s -> %s" % (fmt_dur(timedelta(seconds=d["remaining_sec"])),
                         fmt_local(d["eta"]))


def row_values(d):
    return {
        "state": STATE_MARK.get(d["state"], "?"),
        "seed": str(d.get("seed") or ""),
        "machine": str(d.get("machine") or ""),
        "agents": str(d.get("agents") or ""),
        "map": str(d.get("map") or ""),
        "algo": str(d.get("algo") or ""),
        "setting": str(d.get("setting") or ""),
        "task_arrival": str(d.get("task_arrival") or ""),
        "task_assign": str(d.get("task_assign") or "-"),
        "prog": _prog_cell(d),
        "duration": str(d.get("duration") or ""),
        "eta": eta_cell(d),
        "last": fmt_local(d.get("last_seen")),
    }


def render_table(rows):
    widths = dict((k, max(w, len(h))) for k, h, w in TABLE_COLS)
    cells = [row_values(d) for d in rows]
    for c in cells:
        for k, _h, _w in TABLE_COLS:
            widths[k] = max(widths[k], len(c[k]))
    out = ["  ".join(h.ljust(widths[k]) for k, h, _w in TABLE_COLS)]
    out.append("-" * len(out[0]))
    for c in cells:
        out.append("  ".join(c[k].ljust(widths[k]) for k, _h, _w in TABLE_COLS))
    return "\n".join(out)


MD_COLS = [("seed", "seed"), ("machine", "machine"), ("setting", "setting"),
           ("algo", "algorithm"), ("task_arrival", "task arrival"),
           ("task_assign", "task assign"), ("state", "status"), ("prog", "progress"),
           ("eta", "eta")]


def render_markdown(rows):
    """Notion に貼り付けるとそのまま表になる Markdown."""
    out, cur = [], None
    for d in rows:
        grp = "%sagent %s   %sM" % (d.get("agents"), d.get("map"), d.get("group_m"))
        if grp != cur:
            cur = grp
            out.append("")
            out.append("## " + grp)
            out.append("")
            out.append("| " + " | ".join(h for _k, h in MD_COLS) + " |")
            out.append("|" + "|".join(["---"] * len(MD_COLS)) + "|")
        v = row_values(d)
        out.append("| " + " | ".join(v[k] or " " for k, _h in MD_COLS) + " |")
    return "\n".join(out).strip() + "\n"


def render_csv(rows, stream):
    import csv
    keys = ["state", "machine", "hostname", "seed", "agents", "map", "algo",
            "setting", "lare_mode", "task_arrival", "task_assign", "t_max",
            "t_last", "progress", "duration", "remaining_sec", "eta",
            "sacred_status", "uid", "run_dir"]
    w = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
    w.writeheader()
    for d in rows:
        w.writerow(dict((k, d.get(k)) for k in keys))


def odd_param_runs(rows):
    """条件ごとに多数派の param_hash を決め、そこから外れた run を返す.

    「同じ条件のはずなのに設定が違う run」= 使えないので学習し直す対象。
    充足数のカウントからも外す (4/5 done でも 1 本が別条件なら実質 4 本)。
    """
    from collections import defaultdict
    by_cond = defaultdict(lambda: defaultdict(list))
    for d in rows:
        # running も含める。17 時間回してから「設定が違った」と分かるより、
        # 走り出してすぐ気づけるほうがよい
        if d.get("state") in ("done", "running") and d.get("param_hash"):
            by_cond[d["condition"]][d["param_hash"]].append(d)
    odd = []
    for _cond, groups in by_cond.items():
        if len(groups) < 2:
            continue
        ranked = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        for _h, ds in ranked[1:]:          # 先頭が多数派、残りが少数派
            odd.extend(ds)
    return odd


def batch_summary(batches):
    """マシンごとの (バッチ数, 予約残り, 総本数が分かっているか) を返す."""
    out = {}
    for b in batches or []:
        m = b.get("machine")
        n_batch, reserved, known = out.get(m, (0, 0, True))
        total, started = b.get("total"), b.get("started")
        if total is None:
            known = False
        else:
            reserved += max(0, int(total) - int(started or 0))
        out[m] = (n_batch + 1, reserved, known)
    return out


def render_status(rows, batches=None, max_rows=3):
    """常駐パネル向けの短い表示.

    1 行目 = 要約 (メニューバー 1 行に収まる長さ), 2 行目 = "---",
    以降 = 実行中の一覧 (終了予定の早い順). この形は xbar のプラグイン出力
    そのままなので、メニューバー常駐にもターミナル常駐にも同じものが使える。
    """
    from collections import Counter
    counts = Counter(d["state"] for d in rows)
    far = now_utc() + timedelta(days=3650)
    running = sorted([d for d in rows if d["state"] == "running"],
                     key=lambda d: d.get("eta") or far)

    bsum = batch_summary(batches)
    reserved_total = sum(v[1] for v in bsum.values())
    unknown = any(not v[2] for v in bsum.values())

    nxt = next((d for d in running if d.get("eta")), None)
    head = "▶ %d" % len(running)
    if reserved_total or unknown:
        head += " +%d%s" % (reserved_total, "?" if unknown else "")
    if nxt:
        head += "  next %s" % fmt_local(nxt["eta"])
    bad = counts["stalled"] + counts["failed"] + counts["short"]
    if bad:
        head += "  ⚠ %d" % bad
    n_tmax = len([d for d in rows if d.get("t_max_ok") is False])
    if n_tmax:
        head += "  t_max✗ %d" % n_tmax
    n_odd = len(odd_param_runs(rows))
    if n_odd:
        head += "  params✗ %d" % n_odd

    lines = [head, "---"]

    # マシン別の集約. 台数が増えても行数がマシン数で頭打ちになる
    machines = sorted(set([d.get("machine") for d in running] + list(bsum)),
                      key=lambda m: str(m))
    for m in machines:
        mine = [d for d in running if d.get("machine") == m]
        n_batch, reserved, known = bsum.get(m, (0, 0, True))
        cell = "%-6s %2d run" % (m, len(mine))
        if n_batch:
            # 予約ファイルを出していないバッチが混ざっていたら "?" を付ける
            cell += "  +%d%s wait" % (reserved, "" if known else "?")
        else:
            cell += "         "
        e = next((d["eta"] for d in mine if d.get("eta")), None)
        if e:
            cell += "   next %s" % fmt_local(e)
        if n_batch:
            cell += "   (train.py x%d)" % n_batch
        lines.append(cell)
    if not machines:
        lines.append("no run in progress")

    if running:
        lines.append("---")
    for d in running[:max_rows]:
        eta = ("%s → %s" % (fmt_dur(timedelta(seconds=d["remaining_sec"])),
                            fmt_local(d["eta"]))) if d.get("eta") else "eta ?"
        lines.append("%3.0f%%  %-7s %2sag %-11s %-5s @%-5s  %s" % (
            (d.get("progress") or 0) * 100, d.get("algo"), d.get("agents"),
            d.get("map"), "%gM" % (d.get("group_m") or 0), d.get("machine"), eta))
    if len(running) > max_rows:
        lines.append("... +%d more" % (len(running) - max_rows))

    if bad:
        lines.append("---")
        lines.append("  ".join("%s %d" % (STATE_MARK[k], counts[k])
                               for k in ("stalled", "failed", "short") if counts[k]))
    lines.append("done %d / %d runs   updated %s"
                 % (counts["done"], len(rows), fmt_local(now_utc())))
    return "\n".join(lines)


# ダッシュボードの「条件」パネルに出す学習パラメータ.
# config 全文 (86 キー) を埋めると HTML が重くなるので、条件を判断できる分だけ。
# 全文は --format jsonl に出る
DASH_CFG_KEYS = ("t_max", "lr", "gamma", "batch_size", "buffer_size", "hidden_dim",
                 "mixer", "epsilon_anneal_time", "target_update_interval_or_tau",
                 "use_rnn", "obs_agent_id", "obs_last_action", "standardise_rewards",
                 "batch_size_run", "runner", "train_task_assigner")
DASH_ENV_KEYS = ("key", "time_limit", "state_repre_flag", "use_lare_path",
                 "use_lare_path_training", "use_pretrained_lare_path",
                 "pretrained_lare_path_model_name", "use_finetuning_lare_path",
                 "finetuning_lare_path_model_name", "randomize_task_arrival",
                 "task_arrival", "task_density", "mmpp_ratio", "rand_p_min",
                 "rand_p_max", "task_switch_prob", "use_dynamic_agents",
                 "randomize_initial_active", "min_active_agents",
                 "max_active_agents", "exclude_station_from_tasks",
                 "allow_reassign_before_pickup")


def dash_params(d):
    """1 run 分の主要パラメータを "cfg.lr" 形式の平たい dict にする."""
    out = {}
    for pre, src, keys in (("", d.get("cfg") or {}, DASH_CFG_KEYS),
                           ("env.", d.get("env") or {}, DASH_ENV_KEYS)):
        for k in keys:
            if k in src:
                out[pre + k] = src[k]
    return out


def overdue_sec(interval_sec):
    """このホストが「遅れている」と言ってよい経過時間.

    いつもの間隔の 2 倍。ただし短すぎると同期のゆらぎで誤検知するので
    下限 20 分、上限は従来の固定しきい値 (3 時間) を超えないようにする。
    間隔がまだ分からないホストは従来どおり 3 時間。
    """
    if not interval_sec:
        return DATA_STALE_SEC
    return max(1200.0, min(float(interval_sec) * 2.0, DATA_STALE_SEC))


def dashboard_data(rows, batches, errors, saved=None, cadence=None):
    """HTML に埋め込む JSON を組み立てる.

    saved = load_saved_models() の結果。渡すと各 run に
    「モデルを手元に回収済みか」が付く。
    """
    odd = set(d["uid"] for d in odd_param_runs(rows))
    saved = saved or {}
    cadence = cadence or {}
    runs = []
    for d in rows:
        runs.append({
            "uid": d["uid"], "machine": d.get("machine"), "seed": str(d.get("seed")),
            "condition": d.get("condition"), "map": d.get("map"),
            "agents": d.get("agents"), "algo": d.get("algo"),
            "group_m": d.get("group_m"), "t_max": d.get("t_max"),
            "setting": d.get("setting"), "lare_mode": d.get("lare_mode"),
            "method_tag": d.get("method_tag"), "task_arrival": d.get("task_arrival"),
            "task_assign": d.get("task_assign"), "reassign": d.get("reassign"),
            "dynamic_agents": d.get("dynamic_agents"),
            "state": d.get("state"),
            "progress": d.get("progress"), "t_last": d.get("t_last"),
            "duration": d.get("duration"),
            "eta": d["eta"].isoformat() if d.get("eta") else None,
            "remaining_sec": d.get("remaining_sec"),
            "paused": d.get("paused"),
            # train.py の予約の何本目か (例: 2 / 5)。予約から起動していなければ None
            "batch_pos": d.get("batch_pos"), "batch_total": d.get("batch_total"),
            "last_seen": d["last_seen"].isoformat() if d.get("last_seen") else None,
            # 実際に終わった / 止まった時刻。last_seen は heartbeat 優先なので、
            # 「いつ完了したか」を出すには stop_time そのものが要る
            "stop_at": d["stop_dt"].isoformat() if d.get("stop_dt") else None,
            "start_at": d["start_dt"].isoformat() if d.get("start_dt") else None,
            "param_hash": d.get("param_hash"), "odd_params": d["uid"] in odd,
            "t_max_ok": d.get("t_max_ok"), "t_max_expected": d.get("t_max_expected"),
            "run_dir": d.get("run_dir"), "params": dash_params(d),
            "stale_data": d.get("stale_data"),
            "data_age_sec": d.get("data_age_sec"),
            "observed_via": d.get("observed_via"),
            # モデルを手元 (保管リポジトリ) に回収できているか。
            # done なのに saved が空なら、まだ取ってきていない
            "saved": sorted(saved.get(d["uid"], {})),
            "has_model": bool(d.get("model")),
        })
    mach = {}

    def _m(label):
        return mach.setdefault(label, {"batches": 0, "reserved": 0,
                                       "unknown": False, "observed_at": None,
                                       "data_age_sec": None, "via": None,
                                       "runs": 0, "running": 0, "stale_data": False,
                                       "done": 0, "saved": 0})

    # ホストごとの **情報の鮮度**。run の状態とは別軸で持ち、画面の先頭に出す。
    # これが無いと「そのマシンが黙っている」ことに気づけない (実測: M2 が
    # 5 日エクスポートしていないのに、画面上は run が stalled と出るだけだった)
    for d in rows:
        m = _m(d.get("machine"))
        m["runs"] += 1
        if d.get("state") == "running":
            m["running"] += 1
        if d.get("state") == "done":
            m["done"] = m.get("done", 0) + 1
            if saved.get(d["uid"]):
                m["saved"] = m.get("saved", 0) + 1
        age = d.get("data_age_sec")
        if age is not None and (m["data_age_sec"] is None or age < m["data_age_sec"]):
            m["data_age_sec"] = age
            m["observed_at"] = (d["observed_dt"].isoformat()
                                if d.get("observed_dt") else None)
            m["via"] = d.get("observed_via")
    for label, m in mach.items():
        cad = cadence.get(label) or {}
        # 「いつもの間隔」が意味を持つのは共有フォルダ経由だけ。
        # local / ssh の観測時刻は **こちらが収集した時刻** なので、
        # そこから学習した間隔は「ツールを何分おきに回したか」でしかない
        m["interval_sec"] = (cad.get("interval_sec")
                             if (m.get("via") == "drop" or cad.get("declared"))
                             else None)
        m["interval_n"] = cad.get("n") or 0
        m["overdue_sec"] = overdue_sec(m["interval_sec"])
        m["stale_data"] = bool(m["data_age_sec"] is not None
                               and m["data_age_sec"] > m["overdue_sec"])

    for b in batches or []:
        m = _m(b.get("machine"))
        m["batches"] += 1
        if b.get("total") is None:
            m["unknown"] = True
        else:
            m["reserved"] += max(0, int(b["total"]) - int(b.get("started") or 0))
    return {"generated": now_utc().isoformat(), "runs": runs,
            "machines": mach, "errors": errors or []}


DASH_HTML = r"""<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LDRP training</title>
<style>
:root{--bg:#fff;--fg:#111;--mut:#666;--line:#e3e3e3;--card:#fafafa;
      --ok:#1a7f37;--run:#0969da;--warn:#9a6700;--bad:#cf222e;--idle:#8c959f}
@media (prefers-color-scheme:dark){
:root{--bg:#0d1117;--fg:#e6edf3;--mut:#8b949e;--line:#30363d;--card:#161b22;
      --ok:#3fb950;--run:#58a6ff;--warn:#d29922;--bad:#f85149;--idle:#6e7681}}
*{box-sizing:border-box}
body{margin:0;padding:16px;background:var(--bg);color:var(--fg);
     font:13px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace}
h1{font-size:15px;margin:0 0 2px}h2{font-size:12px;color:var(--mut);
   text-transform:uppercase;letter-spacing:.06em;margin:20px 0 8px}
.sub{color:var(--mut);margin-bottom:10px}
.bar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;
     border:1px solid var(--line);border-radius:7px;padding:8px 10px;
     background:var(--card);margin-bottom:6px}
label{color:var(--mut);margin-right:3px}
select{font:inherit;padding:2px 5px;border:1px solid var(--line);border-radius:4px;
       background:var(--bg);color:var(--fg)}
table{border-collapse:collapse;width:100%}
td,th{padding:3px 8px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
th{color:var(--mut);font-weight:normal}
.wrap{overflow-x:auto}
.card{border:1px solid var(--line);border-radius:7px;padding:9px 11px;
      margin-bottom:7px;background:var(--card)}
.chip{display:inline-block;padding:1px 7px;margin:2px 3px 2px 0;border-radius:10px;
      border:1px solid var(--line);font-size:12px}
.c-done{color:var(--ok);border-color:var(--ok)}
.c-run{color:var(--run);border-color:var(--run)}
.c-bad{color:var(--bad);border-color:var(--bad)}
.c-odd{color:var(--warn);border-color:var(--warn);border-style:dashed}
.c-miss{color:var(--idle);border-color:var(--idle);border-style:dashed}
.mut{color:var(--mut)}.err{color:var(--bad)}.wrn{color:var(--warn)}
.pb{display:inline-block;width:110px;height:8px;border-radius:4px;
    background:var(--line);overflow:hidden;vertical-align:middle}
.pb>i{display:block;height:100%;background:var(--run)}
.right{margin-left:auto}
details>summary{cursor:pointer;color:var(--mut);outline:none}
.kv{display:grid;grid-template-columns:auto 1fr;gap:0 12px;margin-top:6px;
    font-size:12px}
.kv b{font-weight:normal;color:var(--mut)}
.diff{color:var(--warn)}
</style></head><body>
<h1>LDRP training</h1>
<div class="sub" id="stamp"></div>
<div id="errors"></div>

<div class="bar">
  <span><label>machine</label><select id="f-machine"></select></span>
  <span><label>map</label><select id="f-map"></select></span>
  <span><label>N</label><select id="f-agents"></select></span>
  <span><label>algo</label><select id="f-algo"></select></span>
  <span><label>状態</label><select id="f-state">
    <option value="">(all)</option><option>done</option><option>running</option>
    <option>stalled</option><option>failed</option><option>short</option></select></span>
  <span class="right mut" id="counts"></span>
</div>

<h2>machines</h2><div class="wrap"><table id="mach"></table></div>
<h2>running now</h2><div class="wrap"><table id="run"></table></div>
<h2>conditions</h2><div id="conds"></div>

<script>
const D = __DATA__;
const esc = s => String(s==null?"":s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const M = v => v==null ? "?" : (v/1e6).toFixed(2);
function when(iso){ if(!iso) return "-"; const d=new Date(iso);
  return `${String(d.getMonth()+1).padStart(2,"0")}/${String(d.getDate()).padStart(2,"0")} `
       + `${String(d.getHours()).padStart(2,"0")}:${String(d.getMinutes()).padStart(2,"0")}`; }
function dur(s){ if(s==null) return "-"; s=Math.max(0,s|0);
  return s<3600 ? Math.round(s/60)+"m" : Math.floor(s/3600)+"h"+String(Math.round(s%3600/60)).padStart(2,"0")+"m"; }
const uniq = k => [...new Set(D.runs.map(r=>r[k]).filter(v=>v!=null&&v!==""))]
  .sort((a,b)=> typeof a==="number" ? a-b : String(a).localeCompare(String(b)));

function fill(id,k){ const el=document.getElementById(id);
  el.innerHTML = `<option value="">(all)</option>` + uniq(k).map(v=>`<option>${esc(v)}</option>`).join("");
  el.onchange = render; }
fill("f-machine","machine"); fill("f-map","map"); fill("f-agents","agents"); fill("f-algo","algo");
document.getElementById("f-state").onchange = render;
document.getElementById("stamp").textContent = "収集 " + when(D.generated);
document.getElementById("errors").innerHTML =
  (D.errors||[]).map(e=>`<div class="err">⚠ ${esc(e)}</div>`).join("");

function sel(id){ return document.getElementById(id).value; }
function rows(){ return D.runs.filter(r=>
  (!sel("f-machine")||r.machine===sel("f-machine")) &&
  (!sel("f-map")||r.map===sel("f-map")) &&
  (!sel("f-agents")||String(r.agents)===sel("f-agents")) &&
  (!sel("f-algo")||r.algo===sel("f-algo")) &&
  (!sel("f-state")||r.state===sel("f-state"))); }

function render(){
  const R = rows();
  const c = {}; R.forEach(r=>c[r.state]=(c[r.state]||0)+1);
  document.getElementById("counts").textContent =
    Object.entries(c).map(([k,v])=>k+"="+v).join("  ") + `   計 ${R.length}`;

  // machines
  const byM = {};
  R.forEach(r=>{ const m=byM[r.machine]||(byM[r.machine]={run:0,done:0,bad:0,eta:null});
    if(r.state==="running"){ m.run++; if(r.eta&&(!m.eta||r.eta<m.eta)) m.eta=r.eta; }
    else if(r.state==="done") m.done++;
    else if(["stalled","failed","short"].includes(r.state)) m.bad++; });
  document.getElementById("mach").innerHTML =
    "<tr><th>machine</th><th>実行中</th><th>予約待ち</th><th>完了</th><th>異常</th><th>次の完了</th></tr>"
    + Object.keys(byM).sort().map(k=>{ const m=byM[k], b=(D.machines||{})[k];
        const res = b ? `+${b.reserved}${b.unknown?"?":""} (train.py x${b.batches})` : "-";
        return `<tr><td>${esc(k)}</td><td>${m.run}</td><td class="mut">${res}</td>
                <td>${m.done}</td><td class="${m.bad?"err":"mut"}">${m.bad}</td>
                <td class="mut">${when(m.eta)}</td></tr>`; }).join("");

  // running
  const run = R.filter(r=>r.state==="running")
    .sort((a,b)=>(a.eta||"9")<(b.eta||"9")?-1:1);
  document.getElementById("run").innerHTML =
    "<tr><th>進捗</th><th>t_env</th><th>残り</th><th>終了予定</th><th>machine</th>"
    + "<th>条件</th><th>seed</th><th>経過</th></tr>"
    + (run.length ? run.map(r=>`<tr>
        <td><span class="pb"><i style="width:${((r.progress||0)*100).toFixed(0)}%"></i></span>
            ${((r.progress||0)*100).toFixed(0)}%</td>
        <td class="mut">${M(r.t_last)}/${M(r.t_max)}M</td>
        <td>${dur(r.remaining_sec)}</td><td>${when(r.eta)}</td>
        <td>${esc(r.machine)}</td>
        <td>${r.agents}ag ${esc(r.map)} ${esc(r.algo)} ${esc(r.setting)}</td>
        <td class="mut">${esc(r.seed)}</td><td class="mut">${esc(r.duration)}</td></tr>`).join("")
       : `<tr><td class="mut" colspan="8">なし</td></tr>`);

  // conditions
  const byC = {};
  R.forEach(r=>(byC[r.condition]||(byC[r.condition]=[])).push(r));
  document.getElementById("conds").innerHTML = Object.keys(byC).sort().map(cond=>{
    const rs = byC[cond];
    const done = rs.filter(r=>r.state==="done" && !r.odd_params);
    const running = rs.filter(r=>r.state==="running");
    const odd = rs.filter(r=>r.odd_params);
    const bad = rs.filter(r=>["stalled","failed","short"].includes(r.state));
    const want = 5, missing = Math.max(0, want - done.length - running.length);
    const chips = []
      .concat(done.map(r=>`<span class="chip c-done" title="done">${esc(r.seed)}</span>`))
      .concat(running.map(r=>`<span class="chip c-run" title="${((r.progress||0)*100).toFixed(0)}% @${esc(r.machine)}">${esc(r.seed)}</span>`))
      .concat(odd.map(r=>`<span class="chip c-odd" title="params differ (${esc(r.param_hash)})">${esc(r.seed)}</span>`))
      .concat(bad.map(r=>`<span class="chip c-bad" title="${esc(r.state)} @${esc(r.machine)}">${esc(r.seed)}</span>`))
      .concat(Array.from({length:missing},()=>`<span class="chip c-miss">?</span>`));
    const tmax = rs.find(r=>r.t_max_ok===false);
    const ref = (done[0]||running[0]||rs[0]).params||{};
    const keys = [...new Set(rs.flatMap(r=>Object.keys(r.params||{})))].sort();
    const tbl = keys.map(k=>{
      const vs = [...new Set(rs.map(r=>JSON.stringify((r.params||{})[k])))];
      const cls = vs.length>1 ? "diff" : "";
      return `<b>${esc(k)}</b><span class="${cls}">${esc(vs.map(v=>v==="undefined"?"-":JSON.parse(v)).join("  |  "))}</span>`;
    }).join("");
    return `<div class="card">
      <div class="bar" style="border:0;background:none;padding:0;margin-bottom:4px">
        <b>${esc(cond)}</b>
        <span class="right ${done.length>=want?"":"wrn"}">${done.length}/${want} done</span></div>
      <div>${chips.join("")}</div>
      ${odd.length?`<div class="wrn">⚠ 要再実行 ${odd.length}: 他の seed と学習パラメータが違う</div>`:""}
      ${tmax?`<div class="wrn">⚠ t_max ${M(tmax.t_max)}M != 想定 ${M(tmax.t_max_expected)}M</div>`:""}
      <details><summary>学習パラメータ (${keys.length})</summary><div class="kv">${tbl}</div></details>
    </div>`;
  }).join("");
}
render();
</script></body></html>
"""


def render_dashboard(rows, batches, errors):
    return DASH_HTML.replace("__DATA__", json.dumps(
        dashboard_data(rows, batches, errors), ensure_ascii=False, default=str))


def render_summary(rows, diff_limit=8):
    from collections import Counter, defaultdict
    by_cond = defaultdict(list)
    for d in rows:
        by_cond[d["condition"]].append(d)
    lines = []
    for cond in sorted(by_cond):
        rs = by_cond[cond]
        c = Counter(r["state"] for r in rs)
        seeds = sorted(str(r.get("seed")) for r in rs if r.get("state") == "done")
        lines.append("%-70s  n=%d  %s" % (
            cond, len(rs),
            " ".join("%s=%d" % (STATE_MARK[s], c[s]) for s in STATE_ORDER if c[s])))
        if seeds:
            lines.append("    done seeds: %s" % ", ".join(seeds))
        # 同じ条件のはずなのに学習パラメータが割れていたら知らせる
        by_hash = defaultdict(list)
        for r in rs:
            by_hash[r.get("param_hash")].append(str(r.get("seed")))
        bad_tmax = [r for r in rs if r.get("t_max_ok") is False]
        if bad_tmax:
            r0 = bad_tmax[0]
            lines.append("    [warn] t_max %gM != expected %gM  (%d run)"
                         % ((r0.get("t_max") or 0) / 1e6,
                            (r0.get("t_max_expected") or 0) / 1e6, len(bad_tmax)))
        if len(by_hash) > 1:
            lines.append("    [warn] params differ across seeds: "
                         + "  ".join("%s(%d)" % (h, len(v))
                                     for h, v in sorted(by_hash.items(),
                                                        key=lambda kv: -len(kv[1]))))
            # ハッシュだけ出しても何を直せばよいか分からないので、
            # **割れているキーと値** をそのまま並べる
            lines.extend(fmt_param_diff(rs, limit=diff_limit))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 学習済みモデルの回収
# ---------------------------------------------------------------------------

def _rel_under_step(model, rel):
    """model["dir"]/<step>/ より下の相対パスを返す (例: "path/agent.th")."""
    head = "%s/%s/" % (model["dir"].replace(os.sep, "/"), model["step"])
    rel = rel.replace(os.sep, "/")
    return rel[len(head):] if rel.startswith(head) else os.path.basename(rel)


def select_model_files(model, what="path", with_optimizer=False,
                       with_mixer=False, with_critic=False):
    """回収するファイルを絞る.

    epymarl の新しい保存形は <step>/path/{agent,mixer,opt}.th と <step>/task/*.th。
    古い保存形は <step>/ 直下に .th が並ぶので、path/ が無ければ全部を対象にする。
    既定は "all" = 経路方策と PPO タスク割当方策の両方
    (TP / FIFO の run には task/ が無いので自然に経路だけになる)。
    opt.th / agent_opt.th / critic_opt.th (optimizer state) は評価に要らないので既定で外す。

    mixer.th と critic.th も既定で外す。学習を再開するときにしか使わず、評価側
    (src/all_policy / test.py / runner.py) はどこからも読まない。それでいて
    agent.th の 15 倍あり (実測 平均 1.24MB 対 80KB)、共有フォルダの 93% を
    占めていた。再開したくなったら元のマシンに残っている。
    """
    picked = []
    has_path = any(_rel_under_step(model, r).startswith("path/") for r, _ in model["files"])
    for rel, size in model["files"]:
        sub = _rel_under_step(model, rel)
        if what == "path" and has_path and not sub.startswith("path/"):
            continue
        if not with_optimizer and os.path.basename(sub).endswith("opt.th"):
            continue
        if not with_mixer and os.path.basename(sub) == "mixer.th":
            continue
        # critic も学習専用。評価側はどこからも読まない
        # (MAT ですら agent.th の中の decoder.* だけ使い critic.* は捨てる:
        #  src/all_policy/mat_policy_runner.py)
        if not with_critic and os.path.basename(sub) == "critic.th":
            continue
        picked.append((rel, sub, size))
    return picked


def model_dest_name(d):
    return "%s_%s_%sagents_seed%s_step%s" % (
        d.get("algo"), d.get("map"), d.get("agents"), d.get("seed"),
        (d.get("model") or {}).get("step"))


def eval_model_stem(d):
    """評価側 (src/all_policy/policy.py の model_stem) と同じ stem を組み立てる。

    stem = {map}_{N}_{path_planner}[_{method_tag}][_{task_assign}][_dyn]_{reassign_tag}

    軸の選び方: 実験計画の 84 条件が **重複なく 84 通りの名前になる最小の組**。
      map / N / planner / method_tag      36 通り (衝突 36)
      + task_assign                       72 通り (衝突 12)
      + dynamic                           84 通り (衝突 0)  <- これ
    入れない軸と理由:
      task_arrival  計画内は "bernoulli, mmpp p=<下限>-<上限>" (学習時の p の範囲)
      reassign      末尾の reassign_tag で区別する (再割当ありで学習 = reassign / なし = base)
      LaRe 系列     map + N が系列を一意に決めるので method_tag で足りる
                    (計画外の系列は manifest.jsonl 側で判別する)
      t_max         map + N から決まる

    TP かつ dynamic 無しのときは何も足さないので、**既存のファイル名と互換**。
    """
    parts = ["%s_%s_%s" % (d.get("map"), d.get("agents"), d.get("algo"))]
    if d.get("method_tag"):
        parts.append(d["method_tag"])
    assign = (d.get("task_assign") or "").strip()
    if assign and assign.upper() != "TP":       # TP は既定なので付けない
        parts.append(assign.lower())
    if d.get("dynamic_agents"):
        parts.append("dyn")
    # reassign_tag。再割当ありで学習した run は "reassign" にして、再割当なしの
    # モデル (…_base_seedK.th) を上書きしないようにする (test.py の引数 reassign で読み分ける)
    parts.append("reassign" if d.get("reassign") else "base")
    return "_".join(parts)


# 評価用モデルのファイル名の形式。collect_config.yaml の `eval_naming` で切り替える。
#   "template" (既定): テンプレートの LDRP (研究室配布版) の評価が読む形。
#                      src/all_policy/policy.py が models/safe/{map}_{N}_{algo}.th を開くので、
#                      seed 番号 0 はその名前、1 以降は _seed{i} を付けて並べる
#   "ldrp":            この LDRP の評価 (policy.py の model_stem / resolve_model_path) が読む形。
#                      {map}_{N}_{algo}[_{tag}][_{assign}][_dyn]_base_seed{i}.th
# どちらでも「どの run を同じ条件として seed 番号を振るか」は eval_model_stem() で決める
# (ファイル名の形式だけが変わる)。
EVAL_NAMINGS = ("template", "ldrp")
EVAL_NAMING = "template"


def set_eval_naming(name):
    """設定の eval_naming を反映する。未知の値は既定のまま警告する."""
    global EVAL_NAMING
    name = str(name or "template").strip().lower()
    if name not in EVAL_NAMINGS:
        sys.stderr.write("[config] unknown eval_naming %r; using 'template' (choose from %s)\n"
                         % (name, ", ".join(EVAL_NAMINGS)))
        name = "template"
    EVAL_NAMING = name
    return name


def eval_model_filename(d, seed_index=None):
    """評価側がそのまま読めるファイル名。形式は EVAL_NAMING で決まる.

    末尾の seed は **学習時の生 seed ではなく 0 始まりの通し番号**。
    評価側は model_seed=0 (= 番号 0) を既定で読むので、生 seed (9 桁) を入れると
    噛み合わない。生 seed は manifest.jsonl に残す。
    """
    n = d.get("seed") if seed_index is None else seed_index
    if EVAL_NAMING == "ldrp":
        return "%s_seed%s.th" % (eval_model_stem(d), n)
    base = "%s_%s_%s" % (d.get("map"), d.get("agents"), d.get("algo"))
    return (base + ".th") if str(n) == "0" else "%s_seed%s.th" % (base, n)


def assign_seed_indexes(planned, repo_dir=None, conds=None, planned_only=True):
    """保管する run に stem ごとの通し番号を割り当てて {stem: {生 seed: 番号}} を返す.

    既に manifest に載っているものはその番号を引き継ぐ。新規は **学習を開始した順**
    に 0,1,2,... と振るので、表に並べた順とファイル名の番号が一致する。

    **採番の対象は「実際に保管する run」だけ**にする。計画外の run にも番号を
    配ってしまうと、若い番号がそちらに食われて計画内の条件が seed5..9 のように
    飛び番になる (実測で 40 条件中 14 条件がそうなっていた)。
    評価側は model_seed=0 が既定なので、0 から詰まっていないと困る。
    """
    table = load_seed_index(repo_dir) if repo_dir else {}
    fresh = [t for t in planned if t[0].get("state") == "done"]
    if planned_only and conds:
        fresh = [t for t in fresh if plans_of(t[0], conds)]
    fresh.sort(key=lambda t: (str(t[0].get("start_time") or ""), str(t[0].get("seed"))))

    # 1 巡目: **実験計画の表に書いてある行順**をそのまま番号にする。
    # seed0 が表の 1 行目に対応するので、ファイル名から表を引ける
    for d, _o, _f in fresh:
        stem = eval_model_stem(d)
        tab = table.setdefault(stem, {})
        key = str(d.get("seed"))
        if key in tab:
            continue
        pos = plan_row_of(d, conds)
        if pos is not None and pos not in tab.values():
            tab[key] = pos

    # 2 巡目: 表に無い seed は空いている番号へ詰める
    for d, _o, _f in fresh:
        tab = table.setdefault(eval_model_stem(d), {})
        key = str(d.get("seed"))
        if key in tab:
            continue
        used = set(tab.values())
        n = 0
        while n in used:
            n += 1
        tab[key] = n
    return table


def plan_row_of(d, conds):
    """この run の seed が、計画表の何行目に書いてあるか (0 始まり). 無ければ None."""
    if not conds:
        return None
    import plan as PLAN
    want = str(d.get("seed"))
    for c in conds:
        if not PLAN.matches(c, d):
            continue
        for i, sl in enumerate(c.get("seeds") or []):
            if sl.get("seed") and str(sl["seed"]) == want:
                return i
    return None


def check_seed_order(planned, conds, seed_idx):
    """計画表とモデルの対応を突き合わせ、ずれていたら知らせる.

    - 表に無い seed のモデル (表を書き足し忘れ)
    - 表にあるのにモデルが無い seed (回収できていない)
    どちらも黙って進むと「seed0 が表の何行目か」が分からなくなる。
    """
    if not conds:
        return
    import plan as PLAN
    from collections import defaultdict
    got = defaultdict(set)
    for d, _o, _f in planned:
        if d.get("state") == "done":
            got[id(None)]  # noqa - placeholder to keep defaultdict typed
    have_seed = defaultdict(set)
    for d, _o, _f in planned:
        if d.get("state") != "done":
            continue
        for c in conds:
            if PLAN.matches(c, d):
                have_seed[id(c)].add(str(d.get("seed")))

    off_table, missing = [], []
    for c in conds:
        planned_seeds = [str(sl["seed"]) for sl in (c.get("seeds") or []) if sl.get("seed")]
        have = have_seed.get(id(c), set())
        for sd in sorted(have - set(planned_seeds)):
            off_table.append((PLAN.label(c), c.get("algo"), sd))
        for sd in planned_seeds:
            if sd not in have:
                missing.append((PLAN.label(c), c.get("algo"), sd))
    if off_table:
        sys.stderr.write("[seed] %d model(s) whose seed is not written in the plan "
                         "(numbered after the table rows):\n" % len(off_table))
        for lab, algo, sd in off_table[:8]:
            sys.stderr.write("    %-26s %-6s seed %s\n" % (lab, algo, sd))
    if missing:
        sys.stderr.write("[seed] %d seed(s) in the plan have no model yet:\n"
                         % len(missing))
        for lab, algo, sd in missing[:8]:
            sys.stderr.write("    %-26s %-6s seed %s\n" % (lab, algo, sd))


SEEN_PATH = "tools/.host_seen.json"

# ダッシュボードで「5 seed に数えない」にした run。{uid: 除外した時刻}。
# tools/dashboard/app.py の EXCLUDE_PATH と同じファイル (書くのはあちら、読むのはこちら)。
# launchd から呼ばれても場所がずれないよう、このファイルの場所から決める
EXCLUDE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".run_excluded.json")


def load_excluded(path=EXCLUDE_PATH):
    try:
        with open(path, "r") as f:
            return json.load(f) or {}
    except (IOError, OSError, ValueError):
        return {}
SEEN_KEEP = 40


def record_host_seen(rows, path=SEEN_PATH, hosts=None):
    """ホストごとの観測時刻を積み、**いつもの更新間隔** を出す.

    固定の 3 時間しきい値だと、1 時間おきに出しているホストの
    「1 回落とした」を 3 時間気づけない。実測の間隔を覚えておけば
    その 2 倍で気づける (15 分間隔に変えたホストは 30 分で気づく)。

    返り値: {label: {"interval_sec": 中央値 or None, "n": サンプル数}}
    """
    path = os.path.expanduser(path)
    hist = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                hist = json.load(f) or {}
        except (OSError, ValueError):
            hist = {}

    for d in rows:
        obs = d.get("observed_dt")
        label = d.get("machine")
        if not obs or not label:
            continue
        seen = hist.setdefault(str(label), [])
        iso = obs.isoformat()
        if seen and seen[-1] == iso:
            continue                      # 同じスナップショットは 1 回だけ
        if iso not in seen:
            seen.append(iso)
            seen.sort()
            del seen[:-SEEN_KEEP]

    try:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(hist, f, indent=1)
        os.replace(tmp, path)
    except OSError:
        pass

    # 設定に書いてあれば実測より優先する。学習を待たずに効かせたいため
    declared = {}
    for h in hosts or []:
        v = h.get("export_interval_min")
        if v:
            declared[str(h.get("label"))] = float(v) * 60.0

    out = {}
    for label in set(hist) | set(declared):
        ts = [parse_dt(x) for x in hist.get(label, [])]
        ts = [t for t in ts if t]
        gaps = sorted((ts[i + 1] - ts[i]).total_seconds()
                      for i in range(len(ts) - 1))
        med = gaps[len(gaps) // 2] if gaps else None
        out[label] = {"interval_sec": declared.get(label) or med,
                      "n": len(ts), "declared": label in declared}
    return out


def load_saved_models(repo_dir):
    """保管リポジトリの manifest から {uid: {"path": 名前, "task": 名前}} を作る.

    「この run のモデルはもう手元にあるか」を画面で示すために使う。
    manifest は追記なので同じ uid が複数回出うる。後勝ちで最新を採る。
    """
    out = {}
    path = os.path.join(os.path.expanduser(repo_dir or ""), "manifest.jsonl")
    if not repo_dir or not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            uid = rec.get("uid")
            if not uid:
                continue
            got = {}
            for rel in rec.get("files") or []:
                # files は "{計画名}/{path|task}/..." (計画名が無い古い記録もある)。
                # 先頭だけ見ると計画名を拾ってしまうので、path/task の segment を探す
                for seg in str(rel).split("/"):
                    if seg in SUBSYS_ROOT:
                        got[seg] = rel
                        break
            if got:
                out[uid] = got
    return out


def load_seed_index(repo_dir, excluded=None):
    """manifest.jsonl から stem -> {生 seed: 通し番号} を復元する。

    **一度振った番号は変えない**。評価スクリプトの --model-seed も論文の表も
    その番号を指しているので、後から詰め直すと過去の記録とずれる。

    ただし**ダッシュボードで除外した run の番号は空き番号として返す**。
    除外した run は使わないので、その番号 (= 計画表の行) を回し直した run に
    引き継がせる。こうすると評価用のモデル (…_seed3.th) も評価ログ
    (…_seed3.txt) も同じ名前で上書きされ、集計に古い結果が残らない。
    除外していない run の番号は動かないので、上の原則とは矛盾しない。
    excluded を渡さなければ EXCLUDE_PATH を読む。
    """
    if excluded is None:
        excluded = load_excluded()
    table = {}
    path = os.path.join(repo_dir, "manifest.jsonl")
    if not os.path.exists(path):
        return table
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            stem, idx = rec.get("eval_stem"), rec.get("seed_index")
            if stem is None or idx is None:
                continue
            if rec.get("uid") in excluded:
                continue                  # この番号は空き扱い (回し直した run が入る)
            table.setdefault(stem, {})[str(rec.get("seed"))] = int(idx)
    return table


# 評価側がモデルを探す場所。経路と割当で木が分かれる。
# ファイル名 (eval_model_filename) は両者で同じなので、同じ run から出た
# 2 本が同じ名前で並ぶ
EVAL_MODEL_DIRS = {
    "path": "src/all_policy/models/safe",
    "task": "src/task_assign/models/safe",
}


def task_file_in(out_dir):
    """回収したモデルの中から、タスク割当方策の state_dict を選ぶ.

    PPO 割当は <step>/task/agent.th に保存される
    (src/task_assign/task_policy/ppo.py の save_models)。
    TP / FIFO は規則ベースなので、そもそもファイルが無い。
    """
    cand = os.path.join(out_dir, "task", "agent.th")
    return cand if os.path.exists(cand) else None


def policy_file_in(out_dir):
    """回収したモデルの中から、評価側が読む RNNAgent の state_dict を選ぶ.

    新しい保存形は <step>/path/agent.th、古い保存形は <step>/ 直下に
    {map}_{N}_{algo}.th が置かれる。mixer / critic / optimizer は評価に使わない。
    """
    for cand in (os.path.join(out_dir, "path", "agent.th"),
                 os.path.join(out_dir, "agent.th")):
        if os.path.exists(cand):
            return cand
    for root in (os.path.join(out_dir, "path"), out_dir):
        if not os.path.isdir(root):
            continue
        others = [f for f in sorted(os.listdir(root))
                  if f.endswith(".th")
                  and not f.endswith("opt.th")
                  and os.path.basename(f) not in ("mixer.th", "critic.th")]
        if len(others) == 1:
            return os.path.join(root, others[0])
    return None


def _fetch_local(repo, files, tmp_dir):
    import shutil
    for rel, _sub, _size in files:
        src = os.path.join(repo, rel)
        dst = os.path.join(tmp_dir, rel)
        try:
            os.makedirs(os.path.dirname(dst))
        except OSError:
            pass
        shutil.copy2(src, dst)


def _fetch_ssh(host, repo, files, tmp_dir, timeout):
    """ssh 越しに tar を 1 本のストリームで受ける.

    ファイル名を NUL 区切りで tar の stdin に渡すので、unique_token に含まれる
    空白や ':' があっても壊れない。tar の出力はローカル tar へ直結するので、
    転送内容が Python のメモリに載ることはない。
    """
    remote = "tar -cf - -C %s --null -T -" % _quote(repo)
    cmd = ["ssh", "-o", "BatchMode=yes"]
    for opt in host.get("ssh_options") or []:
        cmd += ["-o", opt]
    cmd += ["-o", "ConnectTimeout=%d" % int(host.get("connect_timeout", 8))]
    if host.get("port"):
        cmd += ["-p", str(host["port"])]
    cmd += [host["ssh"], remote]

    untar = subprocess.Popen(["tar", "-xf", "-", "-C", tmp_dir], stdin=subprocess.PIPE)
    try:
        send = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=untar.stdin,
                                stderr=subprocess.PIPE)
        untar.stdin.close()
        names = b"".join((rel + "\0").encode("utf-8") for rel, _s, _z in files)
        _out, err = send.communicate(names, timeout=timeout)
        rc_untar = untar.wait()
    except subprocess.TimeoutExpired:
        send.kill()
        untar.kill()
        return "tar over ssh timed out (%ss)" % timeout
    if send.returncode != 0:
        msg = err.decode("utf-8", "replace").strip().splitlines()
        return "tar over ssh failed: %s" % (msg[-1] if msg else send.returncode)
    if rc_untar != 0:
        return "local tar exited %s" % rc_untar
    return None


def _record_fetch(dest, d, out_dir, files):
    """回収記録 (manifest.jsonl) を追記する.

    設置コマンド案 (install_hints.sh) はここでは書かない。ファイル名の通し番号は
    回収対象がすべて出そろってからでないと決まらないため (write_install_hints)。
    """
    with open(os.path.join(dest, "manifest.jsonl"), "a") as f:
        f.write(json.dumps({
            "uid": d["uid"], "machine": d.get("machine"),
            "algo": d.get("algo"), "map": d.get("map"),
            "agents": d.get("agents"), "seed": d.get("seed"),
            "t_max": d.get("t_max"), "step": (d.get("model") or {}).get("step"),
            "method_tag": d.get("method_tag"), "param_hash": d.get("param_hash"),
            "dest": os.path.relpath(out_dir, dest),
            "files": [sub for _r, sub, _z in files],
        }, ensure_ascii=True, default=str) + "\n")



def copy_cloud_file(src, dst):
    """iCloud 上のファイルを安全にコピーする.

    macOS の shutil.copy2 は内部で fcopyfile を使うが、実体が退避された
    ファイル (ls -lO が dataless) だと
      OSError: [Errno 11] Resource deadlock avoided
    で落ちる。先に実体を落とし、それでも駄目なら素の読み書きで運ぶ。
    """
    import shutil
    try:
        subprocess.call(["brctl", "download", src],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        pass
    try:
        shutil.copy2(src, dst)
        return
    except OSError:
        pass                                  # fcopyfile が使えない。手で運ぶ
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        shutil.copyfileobj(fi, fo, 1024 * 1024)
    try:
        shutil.copystat(src, dst)
    except OSError:
        pass


def copy_cloud_file_checked(src, dst, tries=3, wait=5.0):
    """copy_cloud_file の後で大きさを確かめ、合わなければやり直す.

    同期の途中 (実体がまだ届いていない) のファイルをコピーすると、例外を出さずに
    **0 バイトのファイル**ができることがある (実測: 黒 24 件・M2 5 件)。
    回収済みの判定は「ディレクトリが空でないか」なので、0 バイトのまま残ると
    二度と取り直されない。合わなければ消して False を返す (次の回で取り直す)。
    """
    import time
    want = os.path.getsize(src)
    for k in range(tries):
        try:
            copy_cloud_file(src, dst)
        except OSError:
            pass
        if os.path.exists(dst) and os.path.getsize(dst) == want and want > 0:
            return True
        if k + 1 < tries:
            time.sleep(wait)
    try:
        os.remove(dst)
    except OSError:
        pass
    return False


def local_copy_ok(path, size=None):
    """回収済みのファイルが使える状態か (0 バイトでない・大きさが記録と一致)."""
    try:
        got = os.path.getsize(path)
    except OSError:
        return False
    return got > 0 and (not size or got == size)


def _fetch_drop(host, drop_root, items):
    """共有フォルダに置かれたモデルをコピーする (--export 済みのホスト用)."""
    import shutil
    base = drop_dir_for(host, drop_root)
    if base is None:
        return "no drop directory configured"
    for d, out_dir, files in items:
        export_dir = (d.get("model") or {}).get("export_dir")
        if not export_dir:
            continue
        for _rel, sub, _size in files:
            src = os.path.join(base, export_dir, sub)
            if not os.path.exists(src):
                continue
            dst = os.path.join(out_dir, sub)
            try:
                os.makedirs(os.path.dirname(dst))
            except OSError:
                pass
            if not copy_cloud_file_checked(src, dst):
                sys.stderr.write("[fetch] %s: %s/%s did not copy completely "
                                 "(still syncing?); will retry next round\n"
                                 % (host["label"], export_dir, sub))
    return None


PURGE_MARKER = ".fetched"


# 色覚バリアフリー配色 (PAAMS の data/ で使っていたもの)。手法に色が無いときの既定
CURVE_PALETTE = ("#03AF7A", "#005AFF", "#FF0000", "#F6AA00",
                 "#4DC4FF", "#804000", "#990099", "#FFF100")
_META_BAD_CHARS = re.compile(r'[<>:"/\\|?*]')
# condition.key は図のファイル名の接頭辞になるので、スキーマで文字集合が決まっている
_COND_BAD_CHARS = re.compile(r"[^A-Za-z0-9._+@~-]+")
# make_graph は -tag- の後ろを [A-Za-z0-9_]+ で取るので、それ以外の tag は扱えない
METRIC_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")


def metric_names(spec_metrics):
    """curves.metrics は文字列の列でも {name, y_label, ...} の列でもよい。名前だけ取る."""
    out = []
    for m in spec_metrics or []:
        n = m.get("name") if isinstance(m, dict) else m
        if n:
            out.append(str(n))
    return out


# make_graph の docs/interchange/*.schema.json が受け付ける語彙。
# 外れた値は **落とさずそのまま書き写す** (仕様は「書き写すだけ」) が、
# make_graph 側で無言で推測にフォールバックするので、ここで知らせる
METRIC_KINDS = ("rate", "ratio", "fraction", "count", "value", "length")
METRIC_BETTER = ("up", "down")


def metric_specs(spec_metrics):
    """_meta.json の metrics[] にそのまま書ける形 (dict の列) に揃える."""
    out = []
    for m in spec_metrics or []:
        e = dict(m) if isinstance(m, dict) else {"name": str(m)}
        n = e.get("name")
        if not METRIC_NAME_RE.match(str(n or "")):
            sys.stderr.write("[curves] metric %r has characters make_graph "
                             "cannot parse out of a file name\n" % n)
        if e.get("kind") and e["kind"] not in METRIC_KINDS:
            sys.stderr.write("[curves] %s: kind=%r is not one of %s; make_graph "
                             "will fall back to guessing from the name\n"
                             % (n, e["kind"], "/".join(METRIC_KINDS)))
        if e.get("better") and e["better"] not in METRIC_BETTER:
            sys.stderr.write("[curves] %s: better=%r is not 'up' or 'down'\n"
                             % (n, e["better"]))
        out.append(e)
    return out


def _meta_dir(name):
    """手法フォルダ名。make_graph が書き出し時に潰す文字だけ '_' にする.

    _meta.json の methods[].dir は**フォルダ名と完全一致**が必須なので、
    ここで決めた名前をフォルダにもメタにも同じく使う。
    """
    return _META_BAD_CHARS.sub("_", str(name or "")).strip() or "method"


def _git_short_head(repo="."):
    try:
        out = subprocess.check_output(["git", "-C", repo, "rev-parse", "--short", "HEAD"],
                                      stderr=subprocess.DEVNULL, timeout=5)
        return out.decode("ascii", "replace").strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _match_field(d, k):
    key = {"setting": "method_tag", "algo": "algo", "task_assign": "task_assign",
           "dynamic": "dynamic_agents", "map": "map", "agents": "agents",
           "method_tag": "method_tag"}.get(k, k)
    return d.get(key)


def curve_method_of(d, methods):
    """この run がどの手法フォルダに入るかを返す (dict) or None.

    methods[] は make_graph の curve_spec.json と同じ形:
      {dir, label, color, order, match: {algo, setting, task_assign, dynamic}}
    旧形式 {name, ...} も受ける (name を dir と label の両方に使う)。
    match の setting は method_tag (safe / dbct / dbct_xxx) と比べる。
    """
    for i, m in enumerate(methods or []):
        mt = m.get("match") or {}
        ok = True
        for k, want in mt.items():
            got = _match_field(d, k)
            if k == "dynamic":
                ok = bool(got) == bool(want)
            elif k == "task_assign":
                w = (str(want) if want is not None else "").upper() or "TP"
                g = (got or "").upper() or "TP"
                ok = (g == w)
            else:
                ok = (str(got) == str(want))
            if not ok:
                break
        if ok:
            name = m.get("dir") or m.get("name") or "method%d" % i
            return {"dir": _meta_dir(name),
                    "label": m.get("label") or m.get("name") or _meta_dir(name),
                    "color": m.get("color"),
                    "order": m.get("order", i),
                    "match": dict(mt)}
    return None


def _mechanical_method(d):
    """手法の指定が無いときの機械的な手法名。条件の軸だけで決める (判断を挟まない)."""
    parts = [str(d.get("algo")), str(d.get("method_tag") or "safe"),
             (d.get("task_assign") or "TP").upper()]
    if d.get("dynamic_agents"):
        parts.append("dyn")
    dir_ = _meta_dir("_".join(parts))
    return {"dir": dir_, "label": dir_, "color": None, "order": None,
            "match": {"algo": d.get("algo"), "setting": d.get("method_tag") or "safe",
                      "task_assign": d.get("task_assign") or "",
                      "dynamic": bool(d.get("dynamic_agents"))}}


GROUP_AXES = ("map", "agents", "t_max", "setting", "algo", "task_assign", "dynamic")


def _bump(counter, key):
    counter[key] = counter.get(key, 0) + 1


def _majority(counter):
    """多数派の値と、割れているかどうか."""
    if not counter:
        return None, False
    v = max(counter.items(), key=lambda kv: (kv[1], str(kv[0])))[0]
    return v, len(counter) > 1


def cond_key(d, grouping=None):
    """条件フォルダ名。grouping にある軸だけで作る (既定は map x agents = 1 図).

    make_graph は条件フォルダ名を「置き場所」としか見ない (_meta.json があるため)
    ので自由に付けてよいが、PAAMS のときと同じ `{map}-v2_{N}agent` に揃えておく。
    """
    g = list(grouping or ("map", "agents"))
    parts = []
    for ax in g:
        if ax == "map":
            m = str(d.get("map") or "").replace("map_", "")
            parts.append("%s-v2" % m if m else "")
        elif ax == "agents":
            parts.append("%sagent" % d["agents"] if d.get("agents") else "")
        elif ax == "t_max":
            tm = _num(d.get("t_max"))
            parts.append("%gM" % (tm / 1e6) if tm else "")
        elif ax == "setting":
            parts.append(str(d.get("method_tag") or "safe"))
        elif ax == "algo":
            parts.append(str(d.get("algo")))
        elif ax == "task_assign":
            parts.append((d.get("task_assign") or "TP").upper())
        elif ax == "dynamic":
            parts.append("dyn" if d.get("dynamic_agents") else "fixed")
    return _COND_BAD_CHARS.sub("-", "_".join(x for x in parts if x)).strip("-") \
        or "condition"


def curve_filter_ok(d, filt):
    """curve_spec.json の filter。空・未指定の軸は「制限なし」."""
    if not filt:
        return True
    def has(key, got, norm=None):
        want = filt.get(key)
        if not want:
            return True
        if norm:
            return norm(got) in [norm(w) for w in want]
        return got in want
    m = str(d.get("map") or "")
    if filt.get("maps") and not (m in filt["maps"]
                                 or m.replace("map_", "") in filt["maps"]):
        return False
    if filt.get("agents") and _num(d.get("agents")) not in [_num(a) for a in filt["agents"]]:
        return False
    if not has("settings", d.get("method_tag") or "safe"):
        return False
    if not has("algos", d.get("algo")):
        return False
    if filt.get("task_assigns") and (d.get("task_assign") or "TP").upper() not in \
            [str(t or "TP").upper() for t in filt["task_assigns"]]:
        return False
    if filt.get("dynamic") and bool(d.get("dynamic_agents")) not in \
            [bool(x) for x in filt["dynamic"]]:
        return False
    return True


def build_metrics_catalog(rows, curves, grouping=None):
    """metrics_catalog.json を組み立てる (make_graph の指標ピッカーの材料).

    仕様は make_graph の docs/interchange/metrics_catalog.schema.json。
    value_min/max を入れておくと make_graph が指標の種類を **名前ではなく
    実測値から** 決められる。無いと test_collision_mean (1 エピソードあたりの
    衝突「回数」) が「率」と誤認され、y 軸が [0,1] に固定されて曲線が軸外へ消える。
    """
    by_uid = dict((d["uid"], d) for d in rows)
    per_tag = {}
    for c in curves:
        t = c["tag"]
        if not METRIC_NAME_RE.match(t):
            continue          # make_graph のファイル名解析を通らない tag は載せない
        e = per_tag.setdefault(t, {"uids": set(), "pts": [], "bytes": [],
                                   "lo": None, "hi": None, "first": None})
        e["uids"].add(c["uid"])
        e["pts"].append(len(c["points"]))
        e["bytes"].append(sum(len("%s,%s,%s\n" % p) for p in c["points"]))
        vals = [p[2] for p in c["points"]]
        if vals:
            lo, hi = min(vals), max(vals)
            e["lo"] = lo if e["lo"] is None else min(e["lo"], lo)
            e["hi"] = hi if e["hi"] is None else max(e["hi"], hi)
        if e["first"] is None and c["uid"] in by_uid:
            e["first"] = cond_key(by_uid[c["uid"]], grouping)

    def median(xs):
        xs = sorted(xs)
        return xs[len(xs) // 2] if xs else 0

    mets = []
    for t in sorted(per_tag):
        e = per_tag[t]
        m = {"name": t, "n_runs": len(e["uids"]),
             "points_per_run": int(median(e["pts"])),
             "bytes_per_run": int(median(e["bytes"])),
             "is_eval": t.startswith("test_")}
        if e["lo"] is not None:
            m["value_min"], m["value_max"] = float(e["lo"]), float(e["hi"])
        if e["first"]:
            m["first_seen"] = e["first"]
        mets.append(m)
    return {"version": 1,
            "generated_by": "ldrp collect_runs.py --scan",
            "generated_at": now_utc().isoformat(),
            "n_runs_scanned": len(set(c["uid"] for c in curves)),
            "n_conditions": len(set(cond_key(by_uid[c["uid"]], grouping)
                                    for c in curves if c["uid"] in by_uid)),
            "metrics": mets}


def export_curves(rows, curves, dest, methods=None, spec_metrics=None, conds=None,
                  dry_run=False, repo=".", filt=None, grouping=None,
                  x_label=None, require_seeds=0):
    """学習曲線を make_graph が読めるフォルダ構成で書き出し、条件ごとに _meta.json (v2) を書く.

        {dest}/{map}-v2_{N}agent/            <- 1 図 = 1 条件フォルダ (_meta.json はここ)
            {手法 dir}/run-{token}/run-{token}-tag-{metric}.csv

    命名の制約 (make_graph の実装で確認済み):
      - `-tag-{metric}` の**後ろに英字を足さない**。detect_metric が
        [A-Za-z0-9_]+ を貪欲に取るので、指標が手法ごとに分裂する
      - unique_token の ':' は '_' に置換する (フォルダ名に使えない)
      - ファイル名は 150 バイトまで (clean_basename が切り詰める)
      - methods[].dir はフォルダ名と完全一致 (ずれると黙って既定色になる)
    手法はフォルダで表し、ファイル名には入れない。

    _meta.json v2 の仕様は make_graph の docs/interchange/meta.schema.json。
    condition.t_max を必ず入れる: 条件ごとに学習ステップ数が違うので、
    make_graph が x 軸の上限と「途中で止まった run」の判定に使う。
    """
    dest = os.path.abspath(os.path.expanduser(dest))
    want = set(metric_names(spec_metrics)) or None
    by_uid = dict((d["uid"], d) for d in rows)
    written, skipped_run, skipped_method, skipped_filter = 0, 0, 0, 0
    figs = {}
    for c in curves:
        d = by_uid.get(c.get("uid"))
        if d is None or d.get("state") != "done":
            skipped_run += 1
            continue
        if want and c.get("tag") not in want:
            continue
        if conds is not None and not plans_of(d, conds):
            skipped_run += 1
            continue
        if not curve_filter_ok(d, filt):
            skipped_filter += 1
            continue
        if methods:
            m = curve_method_of(d, methods)
            if m is None:
                skipped_method += 1
                continue
        else:
            m = _mechanical_method(d)
        fig = cond_key(d, grouping)
        token = str(c.get("token") or "").replace(":", "_")
        rd = os.path.join(dest, fig, m["dir"], "run-" + token)
        fn = clean_curve_name("run-%s-tag-%s.csv" % (token, c["tag"]))

        F = figs.setdefault(fig, {"map": {}, "agents": {}, "setting": {},
                                  "task_assign": {}, "dynamic": {},
                                  "t_max": {}, "tags": set(), "methods": {}})
        _bump(F["map"], str(d.get("map") or "").replace("map_", ""))
        _bump(F["agents"], d.get("agents"))
        _bump(F["setting"], str(d.get("method_tag") or "safe"))
        _bump(F["task_assign"], (d.get("task_assign") or "TP").upper())
        _bump(F["dynamic"], bool(d.get("dynamic_agents")))
        tm = _num(d.get("t_max"))
        if tm:
            _bump(F["t_max"], int(tm))
        F["tags"].add(c["tag"])
        M = F["methods"].setdefault(m["dir"], {
            "label": m["label"], "color": m["color"], "order": m["order"],
            "match": m["match"], "seeds": set(), "runs": set(), "stems": set()})
        M["runs"].add("run-" + token)
        M["seeds"].add(d.get("seed"))
        M["stems"].add(eval_model_stem(d))

        if dry_run:
            written += 1
            continue
        try:
            os.makedirs(rd)
        except OSError:
            pass
        with open(os.path.join(rd, fn), "w") as f:
            f.write("Wall time,Step,Value\n")
            for wall, step, val in c["points"]:
                f.write("%s,%s,%s\n" % (wall, step, val))
        written += 1

    # ---- 条件ごとに _meta.json (v2) ----------------------------------------
    # 仕様: make_graph の docs/interchange/meta.schema.json
    commit = _git_short_head(repo)
    mspecs = metric_specs(spec_metrics)
    # 色が指定されていない手法に割り当てるパレットの添字。条件ごとに数えると
    # 「5agent には MAPPO があるが 7agent には無い」で同じ手法の色がずれるので、
    # この export に出てくる手法全体で 1 回だけ決める
    allm = {}
    for F in figs.values():
        for dir_, M in F["methods"].items():
            o = M["order"]
            cur = allm.get(dir_, "?")
            if cur == "?" or (o is not None and (cur is None or o < cur)):
                allm[dir_] = o
    pal_ix = dict((d, i) for i, d in enumerate(
        sorted(allm, key=lambda d: (allm[d] is None, allm[d] if allm[d] is not None else 0, d))))

    for fig, F in sorted(figs.items()):
        # t_max: 同じ条件なら計画上は 1 つ。割れていたら多数派を採り、知らせる
        t_max, split = _majority(F["t_max"])
        if split:
            sys.stderr.write("[curves] %s: t_max differs across runs %s; using %s\n"
                             % (fig, sorted(F["t_max"]), t_max))
        # map / agents はこの条件で一意のときだけ書く (grouping 次第で混ざりうる)
        mp, mp_split = _majority(F["map"])
        ag, ag_split = _majority(F["agents"])

        # methods[]: order が無いものは dir 名順で後ろに詰める。色が無ければパレット
        items = sorted(F["methods"].items(),
                       key=lambda kv: (kv[1]["order"] is None,
                                       kv[1]["order"] if kv[1]["order"] is not None else 0,
                                       kv[0]))
        meths = []
        for i, (dir_, M) in enumerate(items):
            stems = sorted(x for x in M["stems"] if x)
            if len(stems) > 1:
                sys.stderr.write("[curves] %s/%s: %d model stems in one method folder "
                                 "(match is looser than the plan): %s\n"
                                 % (fig, dir_, len(stems), stems))
            seeds = []
            for sd in M["seeds"]:
                try:
                    seeds.append(int(sd))
                except (TypeError, ValueError):
                    seeds.append(str(sd))
            if require_seeds and len(M["runs"]) < require_seeds:
                sys.stderr.write("[curves] %s/%s: %d seed(s), fewer than the "
                                 "required %d\n"
                                 % (fig, dir_, len(M["runs"]), require_seeds))
            entry = {"dir": dir_, "label": M["label"],
                     "color": M["color"] or CURVE_PALETTE[pal_ix.get(dir_, i)
                                                          % len(CURVE_PALETTE)],
                     "order": i,
                     "seeds": sorted(seeds, key=lambda x: (isinstance(x, str), x)),
                     "runs": sorted(M["runs"]),
                     "match": M["match"]}
            if stems:
                entry["model_stem"] = stems[0]
            if len(stems) > 1:
                entry["model_stems"] = stems
            meths.append(entry)

        # metrics[]: 仕様があれば **その順序で**、この条件に実在する tag だけ
        if mspecs:
            metrics = [dict(m) for m in mspecs if m.get("name") in F["tags"]]
        else:
            metrics = [{"name": t} for t in sorted(F["tags"])]

        # null はスキーマを通らない (t_max/agents は integer, model_stem は string)
        # ので、値が無いキーは書かない
        cond = {"key": fig}
        if t_max:
            cond["t_max"] = int(t_max)
        if not mp_split and mp:
            cond["map"] = mp
        if not ag_split and (_num(ag) or 0) > 0:
            cond["agents"] = int(_num(ag))
        # 軸がこの条件で一意に決まるときだけ書く。grouping が既定 (map x agents)
        # なら algo/setting は手法ごとに違うので、当然ここには出ない
        for ax, key in (("setting", "setting"), ("task_assign", "task_assign"),
                        ("dynamic", "dynamic")):
            v, split2 = _majority(F[ax])
            if v is not None and not split2:
                cond[key] = v
        meta = {
            "version": 2,
            "generated_by": "ldrp collect_runs.py",
            "generated_at": now_utc().isoformat(),
            "ldrp_commit": commit,
            "spec_version": 1,
            "condition": cond,
            "x_label": x_label or "Training steps",
            "metrics": metrics,
            "methods": meths,
        }
        if not mp_split and mp:
            meta["map_name"] = mp
        if not ag_split and ag is not None:
            # スキーマ上 condition.agents は整数、トップレベルは文字列
            meta["agent_count"] = str(ag)
        if not dry_run:
            fd = os.path.join(dest, fig)
            try:
                os.makedirs(fd)
            except OSError:
                pass
            tmp = os.path.join(fd, "_meta.json.tmp")
            with open(tmp, "w") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            os.replace(tmp, os.path.join(fd, "_meta.json"))

    sys.stderr.write("[curves] %d csv written into %d condition(s) -> %s%s\n"
                     % (written, len(figs), dest, " (dry run)" if dry_run else ""))
    if skipped_run:
        sys.stderr.write("[curves] %d skipped: run is not done or not in a plan\n"
                         % skipped_run)
    if skipped_method:
        sys.stderr.write("[curves] %d skipped: no method matched\n" % skipped_method)
    if skipped_filter:
        sys.stderr.write("[curves] %d skipped by the spec filter\n" % skipped_filter)
    return figs


def clean_curve_name(name, max_len=150):
    """make_graph の clean_basename と同じ 150 バイト制限に合わせる."""
    root, ext = os.path.splitext(name)
    b = root.encode("utf-8")
    budget = max(1, max_len - len(ext.encode("utf-8")))
    if len(b) > budget:
        root = b[:budget].decode("utf-8", "ignore")
    return root + ext


def purge_drop_models(planned, hosts, drop_root, dry_run=False):
    """共有フォルダから **回収済みのモデルだけ** を消す.

    iCloud の容量を空けるための後始末。次の 2 つを必ず守る:

    1. **ローカルに同じサイズのファイルがあることを確認してから消す**。
       1 つでも欠けていたらその run はまるごと残す (消し損じより取り逃しを避ける)
    2. 空にせず `.fetched` を 1 個置く。送信側の重複判定は
       「共有フォルダにディレクトリが残っているか」(export_drop の
       os.listdir) なので、空にすると**次の --export で再送される**。
       マーカーを残せば送信側を一切変更せずに再送を止められる
    """
    bases = {}
    for h in hosts or []:
        b = drop_dir_for(h, drop_root)
        if b:
            bases[h["label"]] = b
    removed, freed, kept, already = 0, 0, 0, 0
    for d, out_dir, files in planned:
        base = bases.get(d.get("machine"))
        export_dir = (d.get("model") or {}).get("export_dir")
        if not base or not export_dir:
            continue
        src_dir = os.path.join(base, export_dir)
        if not os.path.isdir(src_dir):
            continue
        targets, ok = [], True
        for _rel, sub, _z in files:
            src = os.path.join(src_dir, sub)
            if not os.path.exists(src):
                continue
            local = os.path.join(out_dir, sub)
            if not os.path.exists(local) or \
                    os.path.getsize(local) != os.path.getsize(src):
                ok = False
                break
            targets.append(src)
        if not ok:
            kept += 1
            continue
        if not targets:
            already += 1                      # 前の回で消し済み (.fetched だけ残っている)
            continue
        for t in targets:
            freed += os.path.getsize(t)
            if not dry_run:
                try:
                    os.remove(t)
                except OSError:
                    pass
        removed += 1
        if not dry_run:
            try:
                with open(os.path.join(src_dir, PURGE_MARKER), "w") as f:
                    f.write("fetched %s\n" % now_utc().isoformat())
            except OSError:
                pass
    sys.stderr.write("[purge] %d model dir(s) cleared (%.1f MB freed), "
                     "%d already cleared, %d kept because the local copy did not match%s\n"
                     % (removed, freed / 1e6, already, kept,
                        " (dry run)" if dry_run else ""))
    return removed


def fetch_models(hosts, rows, dest, what="path", with_optimizer=False,
                 max_mb=2000.0, timeout=900, dry_run=False, drop_root=None,
                 with_mixer=False, with_critic=False):
    """done の run の最終ステップのモデルを dest に集める.

    dest/{machine}/{algo}_{map}_{N}agents_seed{seed}_step{t_env}/path/agent.th
    という形に整えて置く (リモートの unique_token は空白・':' 入りで扱いづらいため)。
    既に置いてあるものは再取得しない。
    """
    import shutil
    import tempfile

    dest = os.path.abspath(os.path.expanduser(dest))
    by_host = {}
    planned, present, skipped, total, deferred = [], [], 0, 0, 0
    for d in rows:
        model = d.get("model")
        if not model:
            continue
        out_dir = os.path.join(dest, str(d.get("machine")), model_dest_name(d))
        files = select_model_files(model, what, with_optimizer, with_mixer,
                                   with_critic)
        # 置いてあるファイルが 0 バイト・大きさ違いなら取り直す (上書きになる)。
        # 置いていないファイルは問わない (送信側で欠けている run を毎回取りに行かないため)
        broken = [sub for _r, sub, z in files
                  if os.path.exists(os.path.join(out_dir, sub))
                  and not local_copy_ok(os.path.join(out_dir, sub), z)]
        if os.path.isdir(out_dir) and os.listdir(out_dir) and not broken:
            skipped += 1
            present.append((d, out_dir, files))
            continue
        if broken:
            sys.stderr.write("[fetch] re-fetching %s: incomplete local copy (%s)\n"
                             % (os.path.relpath(out_dir, dest), ", ".join(broken)))
        if not files:
            continue
        size_mb = sum(z for _r, _s, z in files) / 1e6
        if max_mb and total + size_mb > max_mb:
            deferred += 1
            continue
        total += size_mb
        planned.append((d, out_dir, files))
        by_host.setdefault(d.get("machine"), []).append((d, out_dir, files))

    sys.stderr.write("[fetch] %d run(s) to fetch (%.1f MB), %d already present\n"
                     % (len(planned), total, skipped))
    if deferred:
        sys.stderr.write("[fetch] %d run(s) left out by the %.0f MB budget "
                         "(raise --max-fetch-mb to get the rest)\n" % (deferred, max_mb))
    if dry_run or not planned:
        for d, out_dir, files in planned:
            sys.stderr.write("  would fetch %s (%d files)\n"
                             % (os.path.relpath(out_dir, dest), len(files)))
        return present + planned, 0

    try:
        os.makedirs(dest)
    except OSError:
        pass
    host_by_label = dict((h["label"], h) for h in hosts)
    fetched = 0
    for label, items in by_host.items():
        host = host_by_label.get(label)
        if host is None:
            sys.stderr.write("[fetch] %s: no host entry, skipped\n" % label)
            continue
        if host.get("drop"):
            err = _fetch_drop(host, drop_root, items)
            if err:
                sys.stderr.write("[fetch] %s: %s\n" % (label, err))
                continue
            for d, out_dir, files in items:
                if os.path.isdir(out_dir) and os.listdir(out_dir):
                    fetched += 1
                    _record_fetch(dest, d, out_dir, files)
            continue

        repo = items[0][0].get("repo") or (host.get("repos") or ["."])[0]
        tmp_dir = tempfile.mkdtemp(prefix=".fetch_", dir=dest)
        try:
            flat = [f for _d, _o, files in items for f in files]
            if host.get("ssh") in (None, "", "local"):
                _fetch_local(repo, flat, tmp_dir)
                err = None
            else:
                err = _fetch_ssh(host, repo, flat, tmp_dir, timeout)
            if err:
                sys.stderr.write("[fetch] %s: %s\n" % (label, err))
                continue
            for d, out_dir, files in items:
                got = 0
                for rel, sub, _size in files:
                    src = os.path.join(tmp_dir, rel)
                    if not os.path.exists(src):
                        continue
                    dst = os.path.join(out_dir, sub)
                    try:
                        os.makedirs(os.path.dirname(dst))
                    except OSError:
                        pass
                    shutil.move(src, dst)
                    got += 1
                if not got:
                    continue
                fetched += 1
                _record_fetch(dest, d, out_dir, files)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    sys.stderr.write("[fetch] fetched %d run(s) into %s\n" % (fetched, dest))
    return present + planned, fetched


# 他のマシンで学習した LaRe-Path のモデルの置き場 (gitignore)。
# 使うときは src/lare/path/models/ へコピーする (pretrained / finetuning は models/ から読む)。
# checkpoints/ に混ぜないのは、この Mac で同じ条件を回したときの autosave と
# 同じ名前になって上書きし合うため
LARE_FETCH_DIR = os.path.join("src", "lare", "path", "fetched")


def fetch_lare_models(hosts, rows, dest_repo, timeout=300, dry_run=False, drop_root=None):
    """完了した run の LaRe-Path のモデルを {dest_repo}/src/lare/path/fetched/{マシン}/ に集める.

    共有フォルダのホスト (黒) は lare/ に書き出されたものを、ssh のホスト (GPU) は
    tar で取る。この Mac (local) で同じリポジトリのものは手元にあるので取らない。
    """
    import shutil
    import tempfile
    dest_repo = os.path.abspath(os.path.expanduser(dest_repo))
    host_by_label = dict((h["label"], h) for h in hosts)
    todo = {}
    for d in rows:
        lr = d.get("lare")
        if d.get("state") != "done" or not lr:
            continue
        host = host_by_label.get(d.get("machine"))
        if host is None:
            continue
        local = not host.get("drop") and host.get("ssh") in (None, "", "local")
        if local and os.path.abspath(os.path.expanduser(d.get("repo") or "")) == dest_repo:
            continue                              # この Mac の checkpoints/ にある
        dst = os.path.join(dest_repo, LARE_FETCH_DIR, str(d.get("machine")), lr["name"])
        try:
            if os.path.getsize(dst) == lr.get("size"):
                continue                          # 取得済み
        except OSError:
            pass
        todo.setdefault(d.get("machine"), []).append((d, lr, dst))
    n_want = sum(len(v) for v in todo.values())
    if dry_run or not n_want:
        for label, items in todo.items():
            for _d, lr, _dst in items:
                sys.stderr.write("  would fetch LaRe %s from %s\n" % (lr["name"], label))
        if n_want:
            sys.stderr.write("[lare] %d LaRe model(s) to fetch (dry run)\n" % n_want)
        return 0

    def _put(src, dst, copy):
        try:
            os.makedirs(os.path.dirname(dst))
        except OSError:
            pass
        copy(src, dst)

    got = 0
    for label, items in todo.items():
        host = host_by_label[label]
        if host.get("drop"):
            base = drop_dir_for(host, drop_root)
            for _d, lr, dst in items:
                # 相手のスクリプトが旧版だと export_path が無い (まだ書き出されていない)
                src = os.path.join(base or "", lr.get("export_path") or "")
                if not base or not lr.get("export_path") or not os.path.exists(src):
                    continue
                try:
                    _put(src, dst, copy_cloud_file)
                    got += 1
                except (OSError, IOError) as e:
                    sys.stderr.write("[lare] %s: %s: %s\n" % (label, lr["name"], e))
            continue
        if host.get("ssh") in (None, "", "local"):
            for d, lr, dst in items:
                src = os.path.join(os.path.expanduser(d.get("repo") or ""), lr["rel"])
                try:
                    _put(src, dst, shutil.copy2)
                    got += 1
                except (OSError, IOError) as e:
                    sys.stderr.write("[lare] %s: %s: %s\n" % (label, lr["name"], e))
            continue
        # ssh: リポジトリごとに tar 1 本で受ける
        by_repo = {}
        for d, lr, dst in items:
            repo = d.get("repo") or (host.get("repos") or ["."])[0]
            by_repo.setdefault(repo, []).append((lr, dst))
        for repo, its in by_repo.items():
            tmp_dir = tempfile.mkdtemp(prefix=".lare_fetch_")
            try:
                err = _fetch_ssh(host, repo,
                                 [(lr["rel"], lr["rel"], lr["size"]) for lr, _dst in its],
                                 tmp_dir, timeout)
                if err:
                    sys.stderr.write("[lare] %s: %s\n" % (label, err))
                    continue
                for lr, dst in its:
                    src = os.path.join(tmp_dir, lr["rel"])
                    if os.path.exists(src):
                        _put(src, dst, shutil.move)
                        got += 1
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)
    sys.stderr.write("[lare] fetched %d / %d LaRe model(s) into %s\n"
                     % (got, n_want, os.path.join(dest_repo, LARE_FETCH_DIR)))
    return got


SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _slug(text, default="none"):
    """ディレクトリ名に使える文字だけにする.

    "8x5_2 10M→aoba00_2 5M" -> "8x5_2-10M-aoba00_2-5M"
    """
    out = SLUG_RE.sub("-", str(text or "").strip()).strip("-")
    return out or default


SUBSYS_ROOT = {"path": "path", "task": "task"}


def subsystem_of(sub):
    """モデルファイルの相対パスから、経路 (path) か タスク割当 (task) かを返す.

    epymarl の保存形は <step>/path/agent.th と <step>/task/agent.th。
    先頭のディレクトリ名がそのまま系統になる (それ以外は経路とみなす)。
    """
    head = str(sub or "").replace(os.sep, "/").split("/")[0]
    return SUBSYS_ROOT.get(head, "path")


UNPLANNED_DIR = "_unplanned"


def plans_of(d, conds):
    """この run が一致する計画名のリスト。どれにも一致しなければ空."""
    if not conds:
        return []
    import plan as PLAN                      # 白でしか使わないので遅延 import
    return sorted(set(c["plan"] for c in conds if PLAN.matches(c, d)))


def publish_dir(d, sub="path/agent.th", plan_name=None):
    """保管用リポジトリ内の配置先ディレクトリ.

        {計画名}/{path|task}/{map}/{N}agent/{setting}/{algo}/{task_assign}[_dyn]/

    先頭を計画名にすると「AAMAS で使うモデル一式」がフォルダごと取り出せる。
    どの計画にも一致しない run は _unplanned/ へ (捨てずに残す)。

    経路方策は path/ 配下、タスク割当方策は task/ 配下。**その下の階層と
    ファイル名は完全に同じ**にしてあるので、同じ run から出た 2 本が
    同じ名前で並ぶ (どちらがどの run のものか一目で対応が取れる)。

    **実験計画の表と同じ並び** (map -> 台数 -> setting -> algorithm -> assign
    -> dynamic) にしてある。表で隣にある条件がディレクトリでも隣に来る。

    setting と task_assign を入れているのは、これが無いと別条件が同じ
    ディレクトリに落ちるため。実測で
      map_8x5_7_qmix_safe    <- TP と PPO が同居
      map_aoba00_7_qmix_dbct <- 別々の LaRe 系列が同居
    が起きていた。
    """
    assign = _slug(d.get("task_assign") or "TP")
    if d.get("dynamic_agents"):
        assign += "_dyn"
    return os.path.join(_slug(plan_name or UNPLANNED_DIR, UNPLANNED_DIR),
                        subsystem_of(sub), str(d.get("map")),
                        "%sagent" % d.get("agents"),
                        _slug(d.get("setting"), "safe"), str(d.get("algo")),
                        assign)


def publish_names(d, sub, seed_index=None):
    """保管用リポジトリでのファイル名を決める.

    方策本体は**評価側がそのまま読める名前**にする (cp するだけで済む)。
    評価に必要なのは agent.th 1 本だけ (QMIX も MAT も単一の .th を torch.load する)。
    mixer / critic は学習専用なので、拾ってあれば接尾辞を付けて併置する。
    step と param_hash はファイル名に入れず manifest.jsonl に持つ。
    """
    eval_name = eval_model_filename(d, seed_index)
    base = os.path.basename(sub)
    if base == "agent.th":
        return eval_name
    stem = eval_name[:-3] if eval_name.endswith(".th") else eval_name
    return "%s.%s" % (stem, base)          # 例: ..._seed123.mixer.th


def publish_models(planned, repo_dir, dry_run=False, hint=True, seed_idx=None,
                   conds=None, planned_only=False):
    """回収したモデルを保管用リポジトリの階層へ配置する.

    git の commit / push はしない (外向きの操作なので明示的にやってもらう)。
    """
    import shutil
    repo_dir = os.path.abspath(os.path.expanduser(repo_dir))
    published = skipped = 0
    lines = []
    refused, tmax_warn, unplanned, lare_only, broken_src = 0, 0, 0, 0, 0
    # 「LaRe モデル学習」の条件 (計画表の <!-- collect: lare -->) は LaRe の .pth が
    # 目当てで、方策モデルは使わない。そこにしか一致しない run の方策は保管しない
    lare_conds = [c for c in (conds or []) if c.get("collect") == "lare"]
    conds = [c for c in (conds or []) if c.get("collect") != "lare"] if lare_conds else conds

    if seed_idx is None:
        seed_idx = assign_seed_indexes(planned, repo_dir, conds, planned_only)
    # 保管済みのファイルが、どの run のものか (manifest の files -> uid)。
    # 除外した run の番号は空き番号として回し直した run に引き継がれる (load_seed_index)。
    # そのとき同じ名前のファイルが「除外した run のもの」として残っているので、
    # 名前があるだけで「保管済み」とみなすと、回し直した run がいつまでも保管されない
    owner = {}
    excluded_uids = set(load_excluded())
    mpath = os.path.join(repo_dir, "manifest.jsonl")
    if os.path.exists(mpath):
        with open(mpath) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                for rel in rec.get("files") or []:
                    owner[rel] = rec.get("uid")
    if planned_only and not conds:
        # 計画が 1 条件も読めていないのに planned_only で走らせると、
        # 「計画に無い」と判定されて **1 つも保管されない**。
        # 設定ミスで静かに空になる方が危ないので、全部出す側に倒す
        sys.stderr.write("[publish] no plan conditions loaded; publishing "
                         "everything under %s/ instead of skipping all\n"
                         % UNPLANNED_DIR)
        planned_only = False

    for d, out_dir, files in planned:
        # 異常終了した run のモデルは保管しない。--fetch-state を広げて調査した
        # ときにも巻き込まれないよう、ここでも必ず弾く
        if d.get("state") != "done":
            refused += 1
            continue
        if d.get("t_max_ok") is False:
            tmax_warn += 1
        stem = eval_model_stem(d)
        s_idx = seed_idx.get(stem, {}).get(str(d.get("seed")))
        # この run が属する計画。複数に一致したらそれぞれへ置く
        # (ファイルの中身は同じなので git は 1 blob しか持たない)
        names = plans_of(d, conds)
        if not names and lare_conds and plans_of(d, lare_conds):
            lare_only += 1
            continue
        if planned_only and not names:
            unplanned += 1
            continue
        targets = names or [UNPLANNED_DIR]
        # 経路 (path/) と タスク割当 (task/) は別の木に置くので、
        # 配置先はファイルごとに決める
        srcs = []
        for _r, sub, _z in files:
            if os.path.basename(sub).endswith("opt.th"):
                continue
            src = os.path.join(out_dir, sub)
            # 0 バイト (回収が途中で切れたもの) は置かない。次の回収で取り直される
            if os.path.exists(src) and not local_copy_ok(src):
                broken_src += 1
                continue
            # dry run では回収前なのでファイルがまだ無い。配置先だけ見せる
            if os.path.exists(src) or dry_run:
                for pn in targets:
                    srcs.append((src, publish_dir(d, sub, pn),
                                 publish_names(d, sub, s_idx)))
        if not srcs:
            continue
        # 置き換えるのは「同じ名前のファイルの持ち主が除外した run」のときと、
        # 置いてあるものが回収元と大きさが違う (0 バイトで置かれた) ときだけ
        if all(os.path.exists(os.path.join(repo_dir, r, n))
               and owner.get("%s/%s" % (r, n)) not in excluded_uids
               and (dry_run or not os.path.exists(s)
                    or os.path.getsize(os.path.join(repo_dir, r, n)) == os.path.getsize(s))
               for s, r, n in srcs):
            skipped += 1
            continue
        for _s, r, n in srcs:
            sys.stderr.write("  %s/%s\n" % (r, n))
        published += 1
        rel = srcs[0][1]
        if dry_run:
            continue
        for src, r, name in srcs:
            dst_dir = os.path.join(repo_dir, r)
            try:
                os.makedirs(dst_dir)
            except OSError:
                pass
            shutil.copy2(src, os.path.join(dst_dir, name))
        lines.append({
            "uid": d["uid"], "machine": d.get("machine"),
            "path": rel, "eval_name": eval_model_filename(d, s_idx),
            "eval_stem": stem, "seed_index": s_idx, "plans": names,
            "t_max_expected": d.get("t_max_expected"), "t_max_ok": d.get("t_max_ok"),
            "map": d.get("map"), "agents": d.get("agents"), "algo": d.get("algo"),
            "seed": d.get("seed"), "t_max": d.get("t_max"),
            "step": (d.get("model") or {}).get("step"),
            "method_tag": d.get("method_tag"), "lare_mode": d.get("lare_mode"),
            "setting": d.get("setting"), "task_arrival": d.get("task_arrival"),
            "task_assign": d.get("task_assign"), "param_hash": d.get("param_hash"),
            "files": ["%s/%s" % (r, n) for _s, r, n in srcs],
        })
    if lines and not dry_run:
        try:
            os.makedirs(repo_dir)
        except OSError:
            pass
        with open(os.path.join(repo_dir, "manifest.jsonl"), "a") as f:
            for rec in lines:
                f.write(json.dumps(rec, ensure_ascii=True, default=str) + "\n")
    # stem には task_assign と dynamic まで入れてあるので、**実験計画の中では
    # 衝突しない**。残るのは LaRe の事前学習系列が違うのに method_tag が同じ
    # "dbct" になる場合だけ (計画内は map+N が系列を一意に決めるので起きない)。
    # 平らな models/safe/ へ install すると同じ通し番号の列に混ざるので知らせる
    stems = {}
    for d, _o, _f in planned:
        if d.get("state") != "done":
            continue
        key = (d.get("setting"), d.get("task_assign") or "TP",
               bool(d.get("dynamic_agents")))
        stems.setdefault(eval_model_stem(d), set()).add(key)
    mixed = dict((k, v) for k, v in stems.items() if len(v) > 1)
    if mixed:
        sys.stderr.write("[publish] %d stem(s) cover more than one LaRe chain; "
                         "directories are separate but the FILE NAMES collide "
                         "(method_tag alone cannot tell the chains apart):\n"
                         % len(mixed))
        for k in sorted(mixed):
            sys.stderr.write("    %s\n" % k)
            for setting, assign, dyn in sorted(map(lambda x: tuple(map(str, x)),
                                                   mixed[k])):
                sys.stderr.write("        setting=%-24s assign=%-4s dynamic=%s\n"
                                 % (setting, assign, dyn))

    sys.stderr.write("[publish] %d placed, %d already there -> %s%s\n"
                     % (published, skipped, repo_dir,
                        " (dry run)" if dry_run else ""))
    if broken_src:
        sys.stderr.write("[publish] %d file(s) skipped: empty local copy "
                         "(will be placed after it is re-fetched)\n" % broken_src)
    if unplanned:
        sys.stderr.write("[publish] %d run(s) skipped: not in any plan\n" % unplanned)
    if lare_only:
        sys.stderr.write("[publish] %d run(s) skipped: LaRe-model runs (policy not kept)\n"
                         % lare_only)
    if refused:
        sys.stderr.write("[publish] %d run(s) skipped: not finished cleanly\n" % refused)
    if tmax_warn:
        sys.stderr.write("[publish] %d run(s) have t_max different from expected_t_max; "
                         "placed anyway (check tools/collect_config.yaml)\n" % tmax_warn)
    if published and not dry_run and hint:
        sys.stderr.write("[publish] commit is left to you:\n"
                         "  cd %s && git add -A && git commit -m 'add models' && git push\n"
                         % repo_dir)
    return published


GIT_FILE_LIMIT_MB = 45.0     # GitHub は 50MB で警告、100MB で push を拒否する


def _git(repo, *args, **kw):
    p = subprocess.Popen(["git", "-C", repo] + list(args),
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out, _ = p.communicate(timeout=kw.get("timeout", 180))
    return p.returncode, out.decode("utf-8", "replace").strip()


def publish_commit(repo_dir, n_added, push=False):
    """保管用リポジトリを commit する (push は明示指定のときだけ).

    自動で回すことを想定しているので、
      - 変更が無ければ何もしない
      - GitHub が拒否するサイズのファイルがあれば中止する
    """
    repo_dir = os.path.abspath(os.path.expanduser(repo_dir))
    if not os.path.isdir(os.path.join(repo_dir, ".git")):
        sys.stderr.write("[publish] %s is not a git repository; skipping commit\n"
                         % repo_dir)
        return 1

    big = []
    for root, _dirs, files in os.walk(repo_dir):
        if ".git" in root.split(os.sep):
            continue
        for f in files:
            fp = os.path.join(root, f)
            try:
                mb = os.path.getsize(fp) / 1e6
            except OSError:
                continue
            if mb > GIT_FILE_LIMIT_MB:
                big.append((mb, os.path.relpath(fp, repo_dir)))
    if big:
        big.sort(reverse=True)
        sys.stderr.write("[publish] refusing to commit: %d file(s) over %.0f MB\n"
                         % (len(big), GIT_FILE_LIMIT_MB))
        for mb, rel in big[:3]:
            sys.stderr.write("            %.1f MB  %s\n" % (mb, rel))
        return 1

    rc, out = _git(repo_dir, "status", "--porcelain")
    if rc != 0:
        sys.stderr.write("[publish] git status failed: %s\n" % out)
        return 1
    if not out:
        sys.stderr.write("[publish] nothing to commit\n")
        return 0

    rc, out = _git(repo_dir, "add", "-A")
    if rc != 0:
        sys.stderr.write("[publish] git add failed: %s\n" % out)
        return 1
    rc, out = _git(repo_dir, "commit", "-m",
                   "add %d model(s) collected by tools/collect_runs.py" % n_added)
    if rc != 0:
        sys.stderr.write("[publish] git commit failed: %s\n" % out)
        return 1
    sys.stderr.write("[publish] committed %d model(s)\n" % n_added)

    if not push:
        sys.stderr.write("[publish] not pushed (pass --publish-push to push)\n")
        return 0
    rc, out = _git(repo_dir, "remote")
    if rc != 0 or not out:
        sys.stderr.write("[publish] no remote configured; not pushed\n")
        return 1
    rc, out = _git(repo_dir, "push")
    if rc != 0:
        sys.stderr.write("[publish] git push failed: %s\n" % out.splitlines()[-1:])
        return 1
    sys.stderr.write("[publish] pushed\n")
    return 0


def write_install_hints(planned, dest, seed_idx):
    """評価側へ置くための cp コマンド案を install_hints.sh に書き出す.

    通し番号は回収対象が出そろってからでないと決まらないので、
    _record_fetch (1 件ずつ) ではなくここでまとめて書く。
    """
    path = os.path.join(dest, "install_hints.sh")
    lines = []
    for d, out_dir, _files in planned:
        policy = policy_file_in(out_dir)
        if not policy:
            continue
        s_idx = seed_idx.get(eval_model_stem(d), {}).get(str(d.get("seed")))
        lines.append("cp %s src/all_policy/models/safe/%s\n"
                     % (_quote(policy), eval_model_filename(d, s_idx)))
    if not lines:
        return
    with open(path, "w") as f:
        f.write("#!/bin/sh\n# generated by collect_runs.py --fetch-models\n")
        for l in sorted(lines):
            f.write(l)


def install_models(planned, models_dir, overwrite=False, dry_run=False,
                   seed_idx=None, task_dir=None):
    """回収したモデルを評価側が読む名前で置く.

    経路方策 -> models_dir            (既定 src/all_policy/models/safe)
    割当方策 -> task_dir              (既定 src/task_assign/models/safe)

    **ファイル名は両者で同じ**。同じ run から出た 2 本が同じ名前になるので、
    評価時に「どの経路方策とどの割当方策が対か」を名前だけで追える。
    TP / FIFO の run には割当ファイルが無いので経路だけ置かれる。
    """
    import shutil
    if seed_idx is None:
        seed_idx = assign_seed_indexes(planned)
    dirs = {"path": os.path.abspath(os.path.expanduser(models_dir)),
            "task": os.path.abspath(os.path.expanduser(
                task_dir or EVAL_MODEL_DIRS["task"]))}
    installed = collections.Counter()
    skipped = missing = 0
    for d, out_dir, _files in planned:
        s_idx = seed_idx.get(eval_model_stem(d), {}).get(str(d.get("seed")))
        name = eval_model_filename(d, s_idx)
        found = [("path", policy_file_in(out_dir)), ("task", task_file_in(out_dir))]
        if all(src is None for _k, src in found) and not dry_run:
            missing += 1
            continue
        for kind, src in found:
            dst = os.path.join(dirs[kind], name)
            if src is None:
                if dry_run and kind == "path":
                    # dry run では回収前なのでファイルがまだ無い。付く名前だけ見せる
                    sys.stderr.write("  (not fetched yet) -> %s/%s\n" % (kind, name))
                    installed[kind] += 1
                continue
            if os.path.exists(dst) and not overwrite:
                skipped += 1
                continue
            sys.stderr.write("  %s/agent.th -> %s/%s\n" % (kind, kind, name))
            if not dry_run:
                try:
                    os.makedirs(dirs[kind])
                except OSError:
                    pass
                shutil.copy2(src, dst)
            installed[kind] += 1
    sys.stderr.write("[install] %d path, %d task installed; %d already there, "
                     "%d run(s) without any model file%s\n"
                     % (installed["path"], installed["task"], skipped, missing,
                        " (dry run)" if dry_run else ""))
    return sum(installed.values())


# ---------------------------------------------------------------------------
# Notion
# ---------------------------------------------------------------------------

NOTION_VERSION = "2022-06-28"
NOTION_API = "https://api.notion.com/v1"

# our field -> (Notion property name, type)
NOTION_SCHEMA = [
    ("seed",         "seed",         "title"),
    ("machine",      "machine",      "select"),
    ("setting",      "setting",      "rich_text"),
    ("algo",         "algorithm",    "select"),
    ("task_arrival", "task arrival", "rich_text"),
    ("task_assign",  "task assign",  "rich_text"),
    ("agents",       "agents",       "number"),
    ("map",          "map",          "select"),
    ("t_max_m",      "steps (M)",    "number"),
    ("t_last",       "t_env",        "number"),
    ("progress",     "progress",     "number"),
    ("state",        "status",       "select"),
    ("lare_mode",    "lare mode",    "select"),
    ("method_tag",   "method tag",   "select"),
    ("param_hash",   "param hash",   "rich_text"),
    ("t_max_ok",     "t_max ok",     "checkbox"),
    ("duration",     "elapsed",      "rich_text"),
    ("eta",          "eta",          "date"),
    ("last_seen",    "last seen",    "date"),
    ("run_dir",      "run dir",      "rich_text"),
    ("uid",          "run_uid",      "rich_text"),
]


DEFAULT_TOKEN_FILES = (
    "~/.config/ldrp/notion_token",
    "~/.ldrp/notion_token",
)


def resolve_notion_token(token_env, token_file=None, script_dir=None):
    """Notion token を env -> 明示ファイル -> 既定ファイルの順に探す.

    cron / systemd から動かすと環境変数を渡しにくいので、ファイルも見る。
    トークンをコマンドライン引数で受け取らないのは、ps に出てしまうため。
    """
    tok = os.environ.get(token_env)
    if tok:
        return tok.strip(), "env:%s" % token_env
    cands = list(DEFAULT_TOKEN_FILES)
    if script_dir:
        cands.insert(0, os.path.join(script_dir, ".notion_token"))
    if token_file:
        cands.insert(0, token_file)
    for c in cands:
        c = os.path.expanduser(c)
        if os.path.exists(c):
            with open(c) as f:
                tok = f.read().strip()
            if tok:
                return tok, c
    return None, None


def notion_check(token, db_id):
    """token と DB へのアクセスを確かめ、足りないプロパティを報告する."""
    try:
        me = notion_request(token, "GET", "/users/me")
    except RuntimeError as e:
        sys.stderr.write("[notion] token rejected: %s\n" % e)
        return 1
    name = (me.get("bot") or {}).get("owner", {}).get("type") or me.get("name") or "?"
    sys.stderr.write("[notion] token ok (integration: %s)\n" % (me.get("name") or name))
    if not db_id:
        sys.stderr.write("[notion] no database id configured\n")
        return 1
    try:
        db = notion_request(token, "GET", "/databases/%s" % db_id)
    except RuntimeError as e:
        sys.stderr.write("[notion] cannot read the database: %s\n" % e)
        sys.stderr.write("[notion] did you add this integration to the page "
                         "(page ... -> Connections)?\n")
        return 1
    title = "".join(t.get("plain_text", "") for t in db.get("title") or [])
    props = db.get("properties") or {}
    sys.stderr.write("[notion] database ok: %r (%d properties)\n" % (title, len(props)))
    missing, wrong = [], []
    for _field, pname, kind in NOTION_SCHEMA:
        got = props.get(pname)
        if got is None:
            missing.append(pname)
        elif got.get("type") != kind:
            wrong.append("%s (is %s, expected %s)" % (pname, got.get("type"), kind))
    if missing:
        sys.stderr.write("[notion] missing properties: %s\n" % ", ".join(missing))
    if wrong:
        sys.stderr.write("[notion] wrong property types: %s\n" % ", ".join(wrong))
    if missing or wrong:
        sys.stderr.write("[notion] add them in Notion, or make a fresh database with "
                         "--notion-create-db <PAGE_ID>\n")
        return 1
    sys.stderr.write("[notion] all properties present\n")
    return 0


def notion_request(token, method, path, payload=None):
    import urllib.error
    import urllib.request
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(NOTION_API + path, data=data, method=method.upper())
    req.add_header("Authorization", "Bearer " + token)
    req.add_header("Notion-Version", NOTION_VERSION)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        raise RuntimeError("Notion API %s %s -> %s: %s" % (method, path, e.code, body))


# 作成時にだけ効く表示オプション (既存 DB のプロパティ書式は変更しない)
NOTION_PROP_OPTS = {"progress": {"format": "percent"}}


def notion_value(kind, value):
    if value is None or value == "":
        return {"title": {"title": []}, "rich_text": {"rich_text": []},
                "select": {"select": None}, "number": {"number": None},
                "date": {"date": None}, "checkbox": {"checkbox": False}}[kind]
    if kind == "title":
        return {"title": [{"text": {"content": str(value)[:2000]}}]}
    if kind == "rich_text":
        return {"rich_text": [{"text": {"content": str(value)[:2000]}}]}
    if kind == "select":
        return {"select": {"name": str(value)[:100]}}
    if kind == "number":
        return {"number": float(value)}
    if kind == "date":
        dt = value if isinstance(value, datetime) else parse_dt(value)
        return {"date": {"start": dt.isoformat()} if dt else None}
    if kind == "checkbox":
        return {"checkbox": bool(value)}
    raise ValueError(kind)


def notion_plain(prop):
    """既存ページのプロパティを比較用のプレーン値に落とす."""
    if not prop:
        return None
    t = prop.get("type")
    if t in ("title", "rich_text"):
        return "".join(x.get("plain_text", "") for x in prop.get(t) or []) or None
    if t == "select":
        return (prop.get("select") or {}).get("name")
    if t == "number":
        return prop.get("number")
    if t == "date":
        return (prop.get("date") or {}).get("start")
    if t == "checkbox":
        return prop.get("checkbox")
    return None


def notion_field_plain(kind, value):
    """比較用のプレーン値 (notion_plain と同じ空間に揃える)."""
    if value is None or value == "":
        return False if kind == "checkbox" else None
    if kind in ("title", "rich_text"):
        return str(value)[:2000]
    if kind == "select":
        return str(value)[:100]
    if kind == "number":
        return float(value)
    if kind == "date":
        dt = value if isinstance(value, datetime) else parse_dt(value)
        return dt.isoformat() if dt else None
    if kind == "checkbox":
        return bool(value)
    return None


def notion_page_props(d):
    """(API へ送る properties, 比較用のプレーン値) を返す."""
    props, plains = {}, {}
    for field, name, kind in NOTION_SCHEMA:
        props[name] = notion_value(kind, d.get(field))
        plains[name] = notion_field_plain(kind, d.get(field))
    return props, plains


def notion_create_db(token, parent_page_id, title):
    props = {}
    for _field, name, kind in NOTION_SCHEMA:
        props[name] = ({"title": {}} if kind == "title"
                       else {kind: NOTION_PROP_OPTS.get(name, {})})
    res = notion_request(token, "POST", "/databases", {
        "parent": {"type": "page_id", "page_id": parent_page_id},
        "title": [{"type": "text", "text": {"content": title}}],
        "properties": props,
    })
    return res["id"]


def notion_fetch_index(token, db_id):
    """DB 全ページを 1 度だけ取得し uid -> (page_id, plain props) を作る."""
    index, cursor = {}, None
    while True:
        payload = {"page_size": 100}
        if cursor:
            payload["start_cursor"] = cursor
        res = notion_request(token, "POST", "/databases/%s/query" % db_id, payload)
        for page in res.get("results", []):
            props = page.get("properties") or {}
            uid = notion_plain(props.get("run_uid"))
            if uid:
                index[uid] = (page["id"],
                              dict((n, notion_plain(props.get(n)))
                                   for _f, n, _k in NOTION_SCHEMA))
        if not res.get("has_more"):
            break
        cursor = res.get("next_cursor")
    return index


def notion_sync(token, db_id, rows, dry_run=False, sleep=0.34):
    import time
    sys.stderr.write("[notion] fetching existing pages...\n")
    index = notion_fetch_index(token, db_id)
    sys.stderr.write("[notion] %d existing pages\n" % len(index))
    created = updated = skipped = 0
    for d in rows:
        uid = d["uid"]
        props, plains = notion_page_props(d)
        if uid not in index:
            if not dry_run:
                notion_request(token, "POST", "/pages",
                               {"parent": {"database_id": db_id}, "properties": props})
                time.sleep(sleep)
            created += 1
            continue
        page_id, old = index[uid]
        changed = {}
        for _field, name, kind in NOTION_SCHEMA:
            a, b = old.get(name), plains.get(name)
            if kind == "date":
                if parse_dt(a) == parse_dt(b):
                    continue
            elif kind == "number":
                if (a is None and b is None) or (
                        a is not None and b is not None and abs(a - b) < 1e-9):
                    continue
            elif (a or None) == (b or None):
                continue
            changed[name] = props[name]
        if not changed:
            skipped += 1
            continue
        if not dry_run:
            notion_request(token, "PATCH", "/pages/" + page_id, {"properties": changed})
            time.sleep(sleep)
        updated += 1
    return created, updated, skipped


def bootstrap_push(host, token, db_id, python=None, remote_dir="~/.ldrp",
                   interval_min=60, install_cron=False, timeout=60):
    """ホストに push 一式 (スクリプト + token + 起動スクリプト) を置く.

    リポジトリを git pull させなくても済むように、このスクリプト自身を送り込む。
    crontab への登録は install_cron=True のときだけ行い、既定では行を表示するに留める。
    """
    label = host["label"]
    py = python or host.get("python") or "python3"
    repo = (host.get("repos") or ["~/LDRP"])[0]
    rd = remote_dir

    runner = "\n".join([
        "#!/bin/sh",
        "# LDRP: このマシンの run を Notion へ push する (tools/collect_runs.py --bootstrap が生成)",
        "exec %s %s/collect_runs.py \\" % (py, rd),
        "    --hosts local --machine %s --repo %s \\" % (_quote(label), _quote(repo)),
        "    --notion --notion-db %s \\" % _quote(db_id),
        "    --notion-token-file %s/notion_token \\" % rd,
        "    --format none",
        "",
    ])
    cron_line = "%d * * * * %s/push.sh >> /tmp/ldrp_push.log 2>&1" % (
        0 if interval_min >= 60 else 0, rd)
    if interval_min < 60:
        cron_line = "*/%d * * * * %s/push.sh >> /tmp/ldrp_push.log 2>&1" % (
            interval_min, rd)

    remote_cmd = " && ".join([
        "mkdir -p %s" % rd,
        "cat > %s/collect_runs.py" % rd,
    ])
    if host.get("ssh") in (None, "", "local"):
        sys.stderr.write("[bootstrap] %s is local; nothing to copy\n" % label)
        return 0

    def run(cmd, stdin_bytes=None):
        base = ["ssh", "-o", "BatchMode=yes"]
        for opt in host.get("ssh_options") or []:
            base += ["-o", opt]
        base += ["-o", "ConnectTimeout=%d" % int(host.get("connect_timeout", 8))]
        if host.get("port"):
            base += ["-p", str(host["port"])]
        base += [host["ssh"], cmd]
        pr = subprocess.Popen(base, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
        out, err = pr.communicate(stdin_bytes, timeout=timeout)
        return pr.returncode, out, err

    rc, _o, err = run(remote_cmd, _script_source())
    if rc != 0:
        sys.stderr.write("[bootstrap] %s: copying the script failed: %s\n"
                         % (label, err.decode("utf-8", "replace").strip()))
        return 1

    rc, _o, err = run("cat > %s/notion_token && chmod 600 %s/notion_token" % (rd, rd),
                      (token + "\n").encode("utf-8"))
    if rc != 0:
        sys.stderr.write("[bootstrap] %s: writing the token failed: %s\n"
                         % (label, err.decode("utf-8", "replace").strip()))
        return 1

    rc, _o, err = run("cat > %s/push.sh && chmod 755 %s/push.sh" % (rd, rd),
                      runner.encode("utf-8"))
    if rc != 0:
        sys.stderr.write("[bootstrap] %s: writing push.sh failed: %s\n"
                         % (label, err.decode("utf-8", "replace").strip()))
        return 1

    rc, out, err = run("%s/push.sh 2>&1 | tail -5" % rd)
    sys.stderr.write("[bootstrap] %s: first push ->\n%s\n"
                     % (label, out.decode("utf-8", "replace").rstrip()))

    if install_cron:
        add = ("( crontab -l 2>/dev/null | grep -v '%s/push.sh' ; echo %s ) | crontab -"
               % (rd, _quote(cron_line)))
        rc, _o, err = run(add)
        if rc != 0:
            sys.stderr.write("[bootstrap] %s: installing the cron entry failed: %s\n"
                             % (label, err.decode("utf-8", "replace").strip()))
            return 1
        sys.stderr.write("[bootstrap] %s: cron entry installed\n" % label)
    else:
        sys.stderr.write("[bootstrap] %s: add this to `crontab -e` on that machine:\n"
                         "  %s\n" % (label, cron_line))
    return 0


# ---------------------------------------------------------------------------
# 完了チェック (軽い問い合わせ)
# ---------------------------------------------------------------------------

def read_cache(path):
    recs = []
    if path and os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("{"):
                    try:
                        recs.append(json.loads(line))
                    except ValueError:
                        pass
    return recs


def write_cache(path, recs):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=True, default=str) + "\n")
    os.replace(tmp, path)


def quick_check(conf, hosts, cache_path, tail_bytes, timeout, stale, tag_map,
                verbose=False):
    """キャッシュ上で実行中の run だけを読み直し、完了/異常を検知する.

    フル収集は 490 run 分の config.json を読むので重いが、こちらは実行中の
    数件だけ。モデル索引も作らないので、2 分おきに回しても負荷にならない。
    """
    cached = read_cache(cache_path)
    if not cached:
        sys.stderr.write("[quick] no cache yet; run a full collect first\n")
        return 0

    by_uid = dict((r.get("uid"), r) for r in cached)
    prev_states = dict((r.get("uid"), r.get("_state")) for r in cached
                       if r.get("_state"))
    targets = {}
    for r in cached:
        if r.get("_state") == "running" and r.get("run_dir"):
            targets.setdefault(r.get("machine"), []).append(r["run_dir"])
    if not targets:
        sys.stderr.write("[quick] nothing is running\n")
        return 0

    host_by_label = dict((h["label"], h) for h in hosts)
    fresh = []
    for label, dirs in targets.items():
        host = host_by_label.get(label)
        if host is None or host.get("drop"):
            continue
        if host.get("ssh") in (None, "", "local"):
            for d in dirs:
                rec = scan_run_dir(d, label, root_tag_of_run(d), tail_bytes,
                                   repo=None, model_index=None)
                if rec is not None:
                    fresh.append(rec)
            continue
        recs, err = collect_ssh(host, tail_bytes, timeout, verbose,
                                only_dirs=dirs, multiplex=True)
        if err:
            sys.stderr.write("[quick] %s: %s\n" % (label, err))
        fresh.extend(recs)

    if not fresh:
        return 0

    # 軽い問い合わせでは repo / model を取らないので、キャッシュの値を残す。
    # 終了予定の記録点も derive() より先に引き継ぐ (直近ペースで予定を出すため)
    for r in fresh:
        old_rec = by_uid.get(r.get("uid")) or {}
        for k in ("repo", "model"):
            if not r.get(k) and old_rec.get(k):
                r[k] = old_rec[k]
        carry_progress_samples(r, old_rec)

    rows = [derive(r, stale, tag_map, lare_chain=conf.get("lare_chain"))
            for r in fresh]
    n = notify_transitions(rows, prev_states)

    state_by_uid = dict((d["uid"], d["state"]) for d in rows)
    for r in fresh:
        r["_state"] = state_by_uid.get(r.get("uid"))
        by_uid[r["uid"]] = r
    write_cache(cache_path, list(by_uid.values()))

    changed = [d for d in rows if prev_states.get(d["uid"]) != d["state"]]
    sys.stderr.write("[quick] checked %d running run(s), %d changed\n"
                     % (len(fresh), len(changed)))
    return n


# ---------------------------------------------------------------------------
# 状態が変わったときの通知
# ---------------------------------------------------------------------------

def _osa_quote(text):
    """AppleScript の文字列リテラルに埋め込める形にする."""
    return '"%s"' % str(text).replace("\\", "\\\\").replace('"', '\\"')


def notify(title, subtitle, body, sound=None):
    """デスクトップ通知を出す (macOS -> Linux -> 標準エラー の順に落とす)."""
    # AppleScript の文字列に改行は入れられないので 1 行に潰す
    body = " / ".join(str(body).splitlines())
    if sys.platform == "darwin":
        script = "display notification %s with title %s subtitle %s" % (
            _osa_quote(body), _osa_quote(title), _osa_quote(subtitle))
        if sound:
            script += " sound name %s" % _osa_quote(sound)
        try:
            subprocess.call(["osascript", "-e", script])
            return True
        except OSError:
            pass
    else:
        try:
            subprocess.call(["notify-send", "%s: %s" % (title, subtitle), body])
            return True
        except OSError:
            pass
    sys.stderr.write("[notify] %s: %s -- %s\n" % (title, subtitle, body))
    return False


NOTIFY_PROBLEM_STATES = ("failed", "stalled", "short")


def one_line(d):
    return "seed %s @%s (%s台 %s %sM %s)" % (
        d.get("seed"), d.get("machine"), d.get("agents"), d.get("map"),
        d.get("group_m"), d.get("algo"))


def notify_transitions(rows, prev_states, max_lines=3):
    """前回の収集からの状態変化を通知する.

    prev_states が空のとき (初回 / キャッシュ無し) は何も出さない。
    そうしないと既存の数百件が全部「今 done になった」ものとして飛んでくる。
    """
    if not prev_states:
        return 0
    finished, problems = [], []
    for d in rows:
        was = prev_states.get(d["uid"])
        if not was or was == d["state"]:
            continue
        if d["state"] == "done":
            finished.append(d)
        elif d["state"] in NOTIFY_PROBLEM_STATES:
            problems.append(d)

    def summary(items):
        head = [one_line(d) for d in items[:max_lines]]
        if len(items) > max_lines:
            head.append("他 %d 件" % (len(items) - max_lines))
        return " / ".join(head)

    # デスクトップ通知は自分だけが見るものなので日本語で出す
    # (CLAUDE.md の「出力は英語」はログ・例外・CLI ヘルプが対象)
    if finished:
        notify("LDRP", "学習が終了しました (%d 件)" % len(finished),
               summary(finished), sound="Glass")
    if problems:
        notify("LDRP", "要確認の run があります (%d 件)" % len(problems),
               summary(problems), sound="Basso")
    return len(finished) + len(problems)


# ---------------------------------------------------------------------------
# config / CLI
# ---------------------------------------------------------------------------

def load_config(path):
    if not path:
        return {}
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        sys.stderr.write("[warn] config not found: %s\n" % path)
        return {}
    with open(path, "r") as f:
        text = f.read()
    if path.endswith(".json"):
        return json.loads(text)
    try:
        import yaml
    except ImportError:
        # ここで黙って {} を返すと、呼び出し側が「設定が無い」と判断して
        # ホスト定義ごと失われる (実測: ラベルが os.uname()[1] に化けて
        # 同じ run が二重にキャッシュへ入った)。落ちないが必ず知らせる
        # 案内に特定の環境のパスを書かない (配布先ではパスが人ごとに違う)
        sys.stderr.write(
            "[error] PyYAML is missing, so %s cannot be read.\n"
            "[error]   this python is %s. Run with a python that has PyYAML\n"
            "[error]   (e.g. the conda env you train with), or: pip install pyyaml\n"
            % (path, sys.executable))
        return {}
    return yaml.safe_load(text) or {}


def local_host_in(conf):
    """設定ファイルの中の「このマシン」のエントリを返す (無ければ None).

    ssh: local と書かれたホストがこのマシン。--hosts local のときに
    このラベルを使わないと、同じ run が別ラベルで二重に溜まる
    (実測: 白 167 run と os.uname()[1] 167 run が完全重複していた)。
    uid にマシン名が入るので dedupe では消えない。
    """
    for h in conf.get("hosts") or []:
        if not h.get("drop") and h.get("ssh") in (None, "", "local"):
            return h
    return None


def build_hosts(conf, args):
    if args.hosts:
        labels = [h.strip() for h in args.hosts.split(",") if h.strip()]
        if labels == ["local"]:
            me = local_host_in(conf) or {}
            return [{"label": (args.machine or conf.get("machine")
                               or me.get("label") or os.uname()[1]),
                     "ssh": "local",
                     "repos": (args.repo or me.get("repos")
                               or conf.get("repos") or [os.getcwd()]),
                     "sacred_subdirs": me.get("sacred_subdirs"),
                     "model_subdirs": me.get("model_subdirs")}]
        hosts = [h for h in conf.get("hosts", []) if h.get("label") in labels]
        missing = set(labels) - set(h["label"] for h in hosts)
        if missing:
            sys.stderr.write("[warn] unknown host label(s): %s\n" % ", ".join(sorted(missing)))
        return hosts
    hosts = conf.get("hosts") or []
    if not hosts:
        hosts = [{"label": args.machine or conf.get("machine") or os.uname()[1],
                  "ssh": "local", "repos": args.repo or [os.getcwd()]}]
    return hosts


# 終了予定を出すための「(heartbeat 時刻, t_env)」の記録点。収集のたびに 1 点ずつ増える
PROGRESS_SAMPLES_MAX = 120      # 1 run あたり保持する点数の上限 (2 分おきの quick-check で約 4 時間分)
PROGRESS_SAMPLES_KEEP_SEC = 3 * 3600   # 最新の点からこれより古い点は捨てる (heartbeat 基準)
PROGRESS_SAMPLES_MIN = 12       # ただし直近のこの点数は古くても残す (停止より前の区間で 15 分を埋めるため)
RATE_WINDOW_SEC = 15 * 60       # 終了予定のペースを出す区間の長さ
PAUSE_GAP_SEC = 30 * 60         # この長さ以上で遅い区間は「学習マシンが止まっていた」とみなす


def _progress_point(rec):
    """このレコードが示す (heartbeat, t_env) の 1 点。どちらかが無ければ None."""
    beat = (rec or {}).get("heartbeat")
    t = (rec or {}).get("t_last")
    if not beat or t in (None, ""):
        return None
    try:
        return [str(beat), int(t)]
    except (TypeError, ValueError):
        return None


def carry_progress_samples(new, old):
    """古いレコードの記録点を新しいレコードへ引き継ぎ、今回の 1 点を足す.

    終了予定を「開始からの平均ペース」で出すと、一時停止 (Ctrl-Z) やスリープで
    止まっていた時間まで平均に入り、何日も先の予定になる (白で 5.5 時間 → 65 時間と
    出ていた)。直近の区間のペースを使えるように、収集のたびに点を残しておく。
    一時停止中は heartbeat も止まるので、止まっていた時間に点は増えない。
    """
    if new is None:
        return new
    pts = {}
    for b, t in list((old or {}).get("progress_samples") or []) + \
            list(new.get("progress_samples") or []):
        pts[str(b)] = int(t)
    for rec in (old, new):
        pt = _progress_point(rec)
        if pt:
            pts[pt[0]] = pt[1]
    if pts:
        seq = [[b, t] for b, t in sorted(pts.items())][-PROGRESS_SAMPLES_MAX:]
        # heartbeat 基準で古い点を捨てる (壁時計ではないので、停止中に点が消えることはない)
        newest = parse_dt(seq[-1][0])
        if newest is not None:
            cut = newest - timedelta(seconds=PROGRESS_SAMPLES_KEEP_SEC)
            keep = [p for p in seq[:-PROGRESS_SAMPLES_MIN]
                    if (parse_dt(p[0]) or newest) >= cut]
            seq = keep + seq[-PROGRESS_SAMPLES_MIN:]
        new["progress_samples"] = seq
    return new


def dedupe(rows):
    """同じ uid が複数回来たら最後の 1 件を残す (記録点は古い方から引き継ぐ)."""
    seen = {}
    for d in rows:
        prev = seen.get(d["uid"])
        if prev is not None and prev is not d:
            carry_progress_samples(d, prev)
        seen[d["uid"]] = d
    return list(seen.values())


def sort_key(d):
    return (d.get("agents") or 0, d.get("group_m") or 0, str(d.get("algo")),
            str(d.get("setting")), str(d.get("task_assign")), str(d.get("seed")))


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Collect LDRP training runs across machines and sync them to Notion.")
    p.add_argument("--scan", action="store_true",
                   help="scanner mode: emit one JSON line per run to stdout (used over ssh)")
    p.add_argument("--dump-curves", action="append", default=None, metavar="RUN_DIR",
                   help="emit the learning curves (metrics.json) of these sacred run "
                        "directories as JSON lines (used over ssh by the dashboard)")
    p.add_argument("--repo", action="append", default=None,
                   help="repository root to scan (repeatable)")
    p.add_argument("--sacred-subdir", action="append", default=None,
                   help="sacred dir relative to repo root (repeatable)")
    p.add_argument("--model-subdir", action="append", default=None,
                   help="saved-model dir relative to repo root (repeatable)")
    p.add_argument("--scan-curves", default="",
                   help="scanner mode: also emit learning curves for these comma "
                        "separated scalar tags (extracted where the event files are)")
    p.add_argument("--curve-max-points", type=int, default=0,
                   help="thin each curve down to this many points (0 = keep all)")
    p.add_argument("--tb-subdir", action="append",
                   help="where tb_logs live, relative to --repo")
    p.add_argument("--scan-dir", action="append", default=None,
                   help="scan only these sacred run directories (repeatable); "
                        "used by the quick finished-run check")
    p.add_argument("--machine", default=None, help="label of this machine (e.g. GPU1)")
    p.add_argument("--tail-bytes", type=int, default=65536,
                   help="bytes of cout.txt tail to read (default: 65536)")

    p.add_argument("--export", default=None, metavar="DIR",
                   help="write this machine's runs and finished models into a shared "
                        "folder (iCloud Drive, Dropbox, NFS, ...) so a machine with no "
                        "ssh access to it can still collect them")
    p.add_argument("--drop-root", default=None,
                   help="shared folder that hosts with 'drop: true' read from")

    p.add_argument("-c", "--config", default=None, help="path to collect_config.yaml")
    p.add_argument("--hosts", default=None,
                   help="comma separated host labels to visit, or 'local'")
    p.add_argument("--ssh-timeout", type=int, default=180, help="per-host ssh timeout (s)")
    p.add_argument("--stale-minutes", type=int, default=None,
                   help="a RUNNING run with no heartbeat for this long is 'stalled'")
    p.add_argument("--format", default="table",
                   choices=["table", "markdown", "csv", "jsonl", "summary",
                            "status", "html", "none"])
    p.add_argument("-o", "--out", default=None,
                   help="write the rendered output to this file")
    p.add_argument("--state", default=None,
                   help="only show these states (comma separated: done,running,stalled,short,failed)")
    p.add_argument("--since-days", type=float, default=None,
                   help="only include runs started within this many days")
    p.add_argument("--min-steps", type=float, default=None,
                   help="drop runs whose t_max is below this (default: 1e6, "
                        "so short debug runs are hidden; pass 0 to keep everything)")
    p.add_argument("--diff-params", action="store_true",
                   help="in --format summary, list every differing parameter "
                        "(default: the first 8 per condition)")
    p.add_argument("--check", action="store_true",
                   help="exit 1 if any run is stalled/failed/short")
    p.add_argument("--cache", default=None,
                   help="merge with (and write back to) this JSONL cache so offline hosts persist")
    p.add_argument("--quick", action="store_true",
                   help="only re-read the runs the cache says are running, notify on "
                        "the ones that finished, and exit; cheap enough to run every "
                        "couple of minutes")
    p.add_argument("--notify", action="store_true",
                   help="pop up a desktop notification when a run finishes or breaks "
                        "(compares against the cache; implies --cache)")
    p.add_argument("-v", "--verbose", action="store_true")

    p.add_argument("--fetch-models", default=None, metavar="DIR",
                   help="download the final policy model of finished runs into DIR")
    p.add_argument("--fetch-state", default="done",
                   help="which states to fetch models for (default: done)")
    p.add_argument("--fetch-what", default="all", choices=["path", "all"],
                   help="'all' = path policy and task assigner (default), "
                        "'path' = path policy only")
    p.add_argument("--fetch-optimizer", action="store_true",
                   help="also fetch opt.th (optimizer state; not needed to evaluate)")
    p.add_argument("--export-curves", metavar="DIR", nargs="?", const="",
                   help="write learning curves as CSV under DIR, laid out for "
                        "make_graph ({cond}/{method}/run-{token}/*.csv). "
                        "with no DIR, the spec's out_root is used")
    p.add_argument("--curve-spec", "--spec", metavar="JSON", dest="curve_spec",
                   help="curve_spec.json from make_graph (metrics / methods / "
                        "filter / grouping). falls back to the 'curves' section "
                        "of the config")
    p.add_argument("--metrics-catalog", metavar="JSON",
                   help="write the list of available scalar tags (with a size "
                        "estimate per run) so make_graph can offer a picker")
    p.add_argument("--purge-drop", action="store_true",
                   help="after a successful fetch, delete the copied model files "
                        "from the shared folder to free iCloud space (leaves a "
                        ".fetched marker so the sender does not re-upload them)")
    p.add_argument("--fetch-critic", action="store_true",
                   help="also fetch critic.th (value function; training only)")
    p.add_argument("--fetch-mixer", action="store_true",
                   help="also fetch mixer.th (QMIX mixer; only needed to resume "
                        "training, and 15x the size of agent.th)")
    p.add_argument("--max-fetch-mb", type=float, default=2000.0,
                   help="stop fetching once this much has been queued (default: 2000)")
    p.add_argument("--fetch-timeout", type=int, default=900,
                   help="per-host timeout for the model transfer (s)")
    p.add_argument("--fetch-dry-run", action="store_true",
                   help="list what would be fetched without transferring")
    p.add_argument("--publish-models", default=None, metavar="REPO",
                   help="place fetched models into the archive repository as "
                        "path/{map}/{N}agent/{planner}_{tag}/{name}__{step}_{hash}/ "
                        "and append to its manifest.jsonl (does not commit or push)")
    p.add_argument("--plan", action="append", default=None, metavar="GLOB",
                   help="experiment plan file(s); used to group published models "
                        "by plan name (default: tools/plans/*.md -> tools/plan.md)")
    p.add_argument("--publish-all", action="store_true",
                   help="also publish runs that match no plan (they go under "
                        "%s/). default: only runs that are in a plan" % UNPLANNED_DIR)
    p.add_argument("--publish-commit", action="store_true",
                   help="commit the archive repository after placing models "
                        "(does nothing when there is no change)")
    p.add_argument("--publish-push", action="store_true",
                   help="also push. implies --publish-commit")
    p.add_argument("--install-models", action="store_true",
                   help="also copy each fetched policy into the evaluation model dir "
                        "under the name test.py expects")
    p.add_argument("--models-dir", default=EVAL_MODEL_DIRS["path"],
                   help="where --install-models puts the path policies")
    p.add_argument("--task-models-dir", default=EVAL_MODEL_DIRS["task"],
                   help="where --install-models puts the task-assigner policies")
    p.add_argument("--overwrite-installed", action="store_true",
                   help="overwrite an evaluation model file that already exists")

    p.add_argument("--notion", action="store_true", help="upsert rows into the Notion database")
    p.add_argument("--notion-db", default=None, help="Notion database id")
    p.add_argument("--notion-token-env", default="NOTION_TOKEN",
                   help="env var holding the Notion integration token")
    p.add_argument("--notion-create-db", default=None, metavar="PAGE_ID",
                   help="create a new database under this Notion page and exit")
    p.add_argument("--notion-dry-run", action="store_true",
                   help="report what would be written without calling the write API")
    p.add_argument("--notion-token-file", default=None,
                   help="read the token from this file instead of the environment "
                        "(cron friendly; ~/.config/ldrp/notion_token is tried by default)")
    p.add_argument("--notion-check", action="store_true",
                   help="verify the token and the database, then exit")
    p.add_argument("--bootstrap", default=None, metavar="LABELS",
                   help="install this script, the token and a push.sh on the named hosts "
                        "so each machine writes to Notion on its own ('all' for every host)")
    p.add_argument("--bootstrap-install-cron", action="store_true",
                   help="also add the crontab entry on those hosts (otherwise the line "
                        "is only printed)")
    p.add_argument("--bootstrap-interval-min", type=int, default=60,
                   help="how often the pushed cron entry runs (default: 60 min)")
    args = p.parse_args(argv)

    if args.dump_curves:
        return cmd_dump_curves(args.dump_curves)
    if args.scan:
        args.machine = args.machine or os.uname()[1]
        args.repo = args.repo or [os.getcwd()]
        return cmd_scan(args)

    conf = load_config(args.config)
    nconf = conf.get("notion") or {}
    token_env = args.notion_token_env or nconf.get("token_env") or "NOTION_TOKEN"
    token, token_src = resolve_notion_token(
        token_env, args.notion_token_file or nconf.get("token_file"),
        script_dir=os.path.dirname(os.path.abspath(__file__)))
    db_id = args.notion_db or nconf.get("database_id")

    if args.notion_check:
        if not token:
            p.error("no Notion token (set %s, or put it in ~/.config/ldrp/notion_token)"
                    % token_env)
        sys.stderr.write("[notion] token from %s\n" % token_src)
        return notion_check(token, db_id)

    if args.bootstrap:
        if not token:
            p.error("no Notion token (set %s, or put it in ~/.config/ldrp/notion_token)"
                    % token_env)
        if not db_id:
            p.error("no Notion database id (--notion-db or notion.database_id)")
        labels = [x.strip() for x in args.bootstrap.split(",") if x.strip()]
        targets = [h for h in (conf.get("hosts") or [])
                   if labels == ["all"] or h.get("label") in labels]
        if not targets:
            p.error("no matching hosts in the config")
        rc = 0
        for h in targets:
            if h.get("drop") or h.get("ssh") in (None, "", "local"):
                sys.stderr.write("[bootstrap] %s: skipped (not an ssh host)\n" % h["label"])
                continue
            rc |= bootstrap_push(h, token, db_id,
                                 interval_min=args.bootstrap_interval_min,
                                 install_cron=args.bootstrap_install_cron)
        return rc

    if args.notion_create_db:
        if not token:
            p.error("no Notion token (set %s, or put it in ~/.config/ldrp/notion_token)"
                    % token_env)
        new_id = notion_create_db(token, args.notion_create_db,
                                  nconf.get("title") or "LDRP runs")
        print("created database: %s" % new_id)
        print("put it in collect_config.yaml as notion.database_id")
        return 0

    if args.export:
        label = args.machine or conf.get("machine") or os.uname()[1]
        host = {"label": label, "ssh": "local",
                "repos": args.repo or conf.get("repos") or [os.getcwd()]}
        records, _err = collect_local(host, args.tail_bytes or 65536)
        min_steps = (args.min_steps if args.min_steps is not None
                     else conf.get("min_steps", 1e6))
        tag_map = conf.get("method_tag_by_lare_mode")
        stale_m = args.stale_minutes or conf.get("stale_minutes") or 90
        records = [r for r in dedupe(records)]
        rows = [derive(r, stale_m, tag_map, lare_chain=conf.get("lare_chain"))
                for r in records]
        if min_steps:
            # batch (train.py の実行予定) は t_max を持たないので、この足切りに
            # 巻き込まれて消える。**予定は共有フォルダ経由でも届けたい** ので残す
            # (これが無いと黒 / M2 の「あと何本回すか」が白から一切見えない)
            keep = set(d["uid"] for d in rows
                       if (d.get("t_max") or 0) >= min_steps
                       or d.get("kind") in ("batch", "host"))
            records = [r for r in records if r["uid"] in keep]
            rows = [d for d in rows if d["uid"] in keep]
        want = set(x.strip() for x in (args.fetch_state or "done").split(",") if x.strip())
        export_drop(records, [d for d in rows if d["state"] in want],
                    os.path.join(os.path.abspath(os.path.expanduser(args.export)), label),
                    what=args.fetch_what, with_optimizer=args.fetch_optimizer,
                    max_mb=args.max_fetch_mb, dry_run=args.fetch_dry_run,
                    with_mixer=args.fetch_mixer,
                    with_critic=args.fetch_critic)
        return 0

    if args.quick:
        cache_path = os.path.expanduser(
            args.cache or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       ".run_cache.jsonl"))
        quick_check(conf, build_hosts(conf, args), cache_path,
                    args.tail_bytes or conf.get("tail_bytes") or 65536,
                    args.ssh_timeout,
                    args.stale_minutes or conf.get("stale_minutes") or 90,
                    conf.get("method_tag_by_lare_mode"), args.verbose)
        return 0

    if args.notify and not args.cache:
        args.cache = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  ".run_cache.jsonl")

    stale = args.stale_minutes or conf.get("stale_minutes") or 90
    set_eval_naming(conf.get("eval_naming"))
    tail_bytes = args.tail_bytes or conf.get("tail_bytes") or 65536
    hosts = build_hosts(conf, args)
    if not hosts:
        p.error("no hosts to visit")

    drop_root = args.drop_root or conf.get("drop_root")

    # 別マシンで回した方策評価のログを logs/ に取ってくる (host に eval_logs: があるときだけ)
    for label, err in pull_eval_logs(hosts, verbose=args.verbose, drop_root=drop_root):
        sys.stderr.write("[eval-logs] %s: %s\n" % (label, err))

    # 学習曲線の仕様。--curve-spec (make_graph が書く) が優先、無ければ config
    spec = {}
    if args.curve_spec:
        spec = _read_json(os.path.expanduser(args.curve_spec)) or {}
    if not spec:
        spec = conf.get("curves") or {}
    if args.export_curves == "":
        args.export_curves = spec.get("out_root")
        if not args.export_curves:
            p.error("--export-curves needs a directory "
                    "(the spec has no out_root)")
    want_curves = None
    if args.export_curves or args.metrics_catalog:
        want_curves = set(metric_names(spec.get("metrics"))) or None
        if args.metrics_catalog:
            want_curves = None            # カタログは全タグを見たい

    raw, errors = collect(hosts, tail_bytes, args.ssh_timeout, args.verbose,
                          drop_root=drop_root,
                          curves=want_curves if (args.export_curves or
                                                 args.metrics_catalog) else None,
                          curve_max_points=int(spec.get("max_points") or 0))

    # 前回の state はキャッシュに書き込んである (_state). 再導出すると
    # 「到達できないホストの running が時間経過で stalled に変わる」等でぶれるので、
    # 保存時点の値をそのまま比較に使う
    # バッチ (train.py の実行予定) は「いまの状態」なのでキャッシュしない。
    # 古い予約が残ると、終わったバッチの枠がいつまでも埋まって見える
    batches = [r for r in raw if r.get("kind") == "batch"]
    curves = [r for r in raw if r.get("kind") == "curve"]
    raw = [r for r in raw if r.get("kind") not in ("batch", "curve", "host")]

    prev_states = {}
    cache_path = os.path.expanduser(args.cache) if args.cache else None
    if cache_path:
        cached = []
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("{"):
                        try:
                            rec = json.loads(line)
                        except ValueError:
                            continue
                        cached.append(rec)
                        if rec.get("_state"):
                            prev_states[rec.get("uid")] = rec["_state"]
        raw = dedupe(cached + raw)

    tag_map = conf.get("method_tag_by_lare_mode")
    exp_t_max = conf.get("expected_t_max")
    rows = [derive(r, stale, tag_map, exp_t_max,
                   lare_chain=conf.get("lare_chain")) for r in dedupe(raw)]

    if args.notify:
        notify_transitions(rows, prev_states)

    if cache_path:
        state_by_uid = dict((d["uid"], d["state"]) for d in rows)
        tmp = cache_path + ".tmp"
        with open(tmp, "w") as f:
            for r in raw:
                r["_state"] = state_by_uid.get(r.get("uid"))
                f.write(json.dumps(r, ensure_ascii=True, default=str) + "\n")
        os.replace(tmp, cache_path)

    min_steps = args.min_steps if args.min_steps is not None else conf.get("min_steps", 1e6)
    if min_steps:
        rows = [d for d in rows if (d.get("t_max") or 0) >= min_steps]
    if args.since_days:
        cut = now_utc() - timedelta(days=args.since_days)
        rows = [d for d in rows if d.get("start_dt") and d["start_dt"] >= cut]
    if args.state:
        want = set(s.strip() for s in args.state.split(","))
        rows = [d for d in rows if d["state"] in want]
    rows.sort(key=sort_key)

    if args.format == "markdown":
        text = render_markdown(rows)
    elif args.format == "summary":
        text = render_summary(rows, diff_limit=10000 if args.diff_params else 8)
    elif args.format == "status":
        text = render_status(rows, batches)
    elif args.format == "html":
        text = render_dashboard(rows, batches,
                                ["%s: %s" % (a, b) for a, b in errors])
    elif args.format == "jsonl":
        text = "\n".join(json.dumps(d, ensure_ascii=True, default=str) for d in rows)
    elif args.format == "csv":
        try:
            from io import StringIO
        except ImportError:
            from StringIO import StringIO
        buf = StringIO()
        render_csv(rows, buf)
        text = buf.getvalue()
    elif args.format == "none":
        text = ""
    else:
        text = render_table(rows)

    if text:
        if args.out:
            with open(os.path.expanduser(args.out), "w") as f:
                f.write(text + "\n")
            sys.stderr.write("[info] wrote %s\n" % args.out)
        else:
            print(text)

    from collections import Counter
    counts = Counter(d["state"] for d in rows)
    sys.stderr.write("[info] %d runs  %s\n" % (
        len(rows), "  ".join("%s=%d" % (STATE_MARK[s], counts[s])
                             for s in STATE_ORDER if counts[s])))

    if args.metrics_catalog:
        cat = build_metrics_catalog(rows, curves, spec.get("grouping"))
        out = os.path.expanduser(args.metrics_catalog)
        try:
            os.makedirs(os.path.dirname(os.path.abspath(out)))
        except OSError:
            pass
        with open(out, "w") as f:
            json.dump(cat, f, indent=2, ensure_ascii=False)
        sys.stderr.write("[catalog] %d metric(s) from %d run(s) -> %s\n"
                         % (len(cat["metrics"]), cat["n_runs_scanned"], out))

    if args.export_curves:
        plan_conds = None
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import plan as _PLAN
            plan_conds = _PLAN.parse_plans(_PLAN.find_plans(args.plan))
        except Exception as e:
            sys.stderr.write("[curves] plan not loaded (%s: %s); "
                             "every done run is exported\n" % (type(e).__name__, e))
        export_curves(rows, curves, args.export_curves,
                      methods=spec.get("methods"),
                      spec_metrics=spec.get("metrics"),
                      conds=plan_conds,
                      repo=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      filt=spec.get("filter"),
                      grouping=spec.get("grouping"),
                      x_label=spec.get("x_label"),
                      require_seeds=int(spec.get("require_seeds") or 0))

    if args.fetch_models or args.fetch_dry_run:
        want = set(x.strip() for x in (args.fetch_state or "done").split(",") if x.strip())
        # 除外した run は取ってこない。ここで外せば保管 (publish)・評価用フォルダへの
        # 配置 (install)・共有フォルダの後始末 (purge) もすべて対象から外れる
        excl = load_excluded()
        targets = [d for d in rows if d["state"] in want and d["uid"] not in excl]
        n_skip = sum(1 for d in rows if d["state"] in want and d["uid"] in excl)
        if n_skip:
            sys.stderr.write("[fetch] skipping %d excluded run(s) listed in %s\n"
                             % (n_skip, EXCLUDE_PATH))
        # LaRe-Path のモデルは方策と別の置き場 (このリポジトリの src/lare/path/fetched/)
        try:
            fetch_lare_models(hosts, targets,
                              os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              timeout=args.fetch_timeout,
                              dry_run=args.fetch_dry_run or not args.fetch_models,
                              drop_root=drop_root)
            # 計画内の LaRe モデル学習が完了していれば、できたモデルを models/ に置く
            # (tools/lare_models.py の auto_install。models/ にあるものは上書きしない)
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import plan as _PLAN
            import lare_models as _LM
            _conds = _PLAN.parse_plans(_PLAN.find_plans(args.plan))
            _LM.auto_install(_conds, targets, _PLAN.matches,
                             dry_run=args.fetch_dry_run or not args.fetch_models)
            # 計画表の LaRe モデルの印を手元のファイルに合わせる (置いた直後のファイルも反映)
            _LM.update_plan_marks(_PLAN.parse_plans(_PLAN.find_plans(args.plan)), targets, _PLAN.matches,
                                  dry_run=args.fetch_dry_run)
        except Exception as e:
            sys.stderr.write("[lare] %s: %s\n" % (type(e).__name__, e))
        planned, _n = fetch_models(
            hosts, targets, args.fetch_models or "models_inbox",
            what=args.fetch_what, with_optimizer=args.fetch_optimizer,
            max_mb=args.max_fetch_mb, timeout=args.fetch_timeout,
            dry_run=args.fetch_dry_run or not args.fetch_models,
            drop_root=drop_root, with_mixer=args.fetch_mixer,
            with_critic=args.fetch_critic)
        # ファイル名の通し番号は回収対象がすべて出そろってから 1 回だけ決める。
        # publish と install で別々に採番すると番号がずれる
        # 計画は白でしか読まない (リモートへ送るスクリプトには影響しない)
        plan_conds = None
        if args.publish_models:
            try:
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                import plan as _PLAN
                plan_conds = _PLAN.parse_plans(_PLAN.find_plans(args.plan))
            except Exception as e:
                sys.stderr.write("[publish] plan not loaded (%s: %s); "
                                 "everything goes under %s/\n"
                                 % (type(e).__name__, e, UNPLANNED_DIR))
        # 採番は計画を読んだあと。計画外に若い番号を食われないようにする
        seed_idx = assign_seed_indexes(planned, args.publish_models, plan_conds,
                                       not args.publish_all)
        check_seed_order(planned, plan_conds, seed_idx)
        if args.fetch_models and not args.fetch_dry_run:
            write_install_hints(planned, args.fetch_models, seed_idx)
        if args.publish_models:
            dry = args.fetch_dry_run or not args.fetch_models
            auto = args.publish_commit or args.publish_push
            n_pub = publish_models(planned, args.publish_models, dry_run=dry,
                                   hint=not auto, seed_idx=seed_idx,
                                   conds=plan_conds,
                                   planned_only=not args.publish_all)
            if auto and not dry:
                publish_commit(args.publish_models, n_pub, push=args.publish_push)
        # 共有フォルダの後始末は **配置が終わってから**。
        # publish より先に消すと、配置に失敗したとき原本が無くなる
        if args.purge_drop and args.fetch_models and not args.fetch_dry_run:
            purge_drop_models(planned, hosts, drop_root)

        if args.install_models:
            install_models(planned, args.models_dir,
                           overwrite=args.overwrite_installed,
                           dry_run=args.fetch_dry_run or not args.fetch_models,
                           seed_idx=seed_idx, task_dir=args.task_models_dir)

    if args.notion or args.notion_dry_run:
        if not token:
            p.error("no Notion token (set %s, or put it in ~/.config/ldrp/notion_token)"
                    % token_env)
        if not db_id:
            p.error("no Notion database id (--notion-db or notion.database_id)")
        c, u, s = notion_sync(token, db_id, rows, dry_run=args.notion_dry_run)
        sys.stderr.write("[notion] created=%d updated=%d unchanged=%d%s\n" % (
            c, u, s, " (dry run)" if args.notion_dry_run else ""))

    bad = sum(counts[s] for s in ("stalled", "failed", "short"))
    if args.check and (bad or errors):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
