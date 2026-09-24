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
import json
import os
import re
import subprocess
import sys
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


# JSON 配列の要素になっている整数だけを拾う。
# "[500, 1000, ...]" の形にだけ当てることで、文字列中の "2026-08-19T..." (年) や
# 小数 (json は 1024.0 と必ず小数点付きで書く) を確実に除外する
ARRAY_INT_RE = re.compile(r"[\[,]\s*(\d{4,})(?=\s*[,\]])")


def _t_env_from_metrics(path, nbytes, cap):
    """cout.txt が空のときの保険。metrics.json の末尾だけを読み、
    t_max を超えない最大の整数 (= 直近の t_env) を拾う。

    metrics.json は 1〜5MB あるので絶対に全体をロードしない。
    steps 配列の値は素の整数、values は小数、timestamps は文字列なので、
    「配列要素の整数」だけを見れば t_env に当たる。
    """
    text = _tail_text(path, nbytes)
    if not text:
        return None
    try:
        limit = float(cap) * 1.01 if cap else None
    except (TypeError, ValueError):
        limit = None
    best = None
    for m in ARRAY_INT_RE.finditer(text):
        v = int(m.group(1))
        if limit is not None and v > limit:
            continue
        if best is None or v > best:
            best = v
    return best


def _pick(d, keys):
    if not isinstance(d, dict):
        return {}
    return dict((k, d[k]) for k in keys if k in d)


# 実験条件としての同一性から外すキー (run ごとに変わる / 条件と無関係)
HASH_EXCLUDE_TOP = frozenset((
    "seed", "checkpoint_path", "load_step", "local_results_path", "label",
    "repeat_id", "evaluate", "render", "save_replay", "use_tensorboard",
    "use_cuda", "save_model", "save_model_interval",
))
HASH_EXCLUDE_ENV = frozenset(("seed",))

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
)


def param_fields(cfg_full, env_full):
    """実験条件として意味を持つパラメータだけに正規化した dict を返す.

    param_hash も条件間の差分表示も**必ずこれを通す**。別々に正規化すると
    「ハッシュは同じなのに差分が出る」ような食い違いが起きる。
    """
    top = dict((k, v) for k, v in (cfg_full or {}).items()
               if k not in HASH_EXCLUDE_TOP)
    env = dict((k, v) for k, v in (env_full or {}).items()
               if k not in HASH_EXCLUDE_ENV)
    # 有効化フラグが off のとき無視されるキーを落とす
    for flag, dependents in HASH_DEPENDENT:
        if not env.get(flag, False):
            for k in dependents:
                env.pop(k, None)
    # randomize_task_arrival が真なら固定値側は使われない
    if env.get("randomize_task_arrival"):
        for k in ("task_arrival", "task_density"):
            env.pop(k, None)
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
    return {
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


def root_tag_of_run(run_dir):
    """<...>/<results|tmp_results>/sacred/<algo>/<env>/<run_id> から "results" を取る."""
    d = os.path.abspath(run_dir)
    sacred_root = os.path.dirname(os.path.dirname(os.path.dirname(d)))
    return _root_tag(sacred_root)


BATCH_DIR = "~/.ldrp"
PS_TRAIN_RE = re.compile(r"(?:^|/)python[0-9.]*\s+(?:-\S+\s+)*(\S*train\.py)\b")
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
        out = subprocess.check_output(["ps", "-eo", "pid,etime,args"],
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


def scan(repos, sacred_subdirs, machine, tail_bytes, emit, model_subdirs=None):
    """repos 配下を走査し、レコードを 1 件ずつ emit に渡す (溜め込まない)."""
    for b in scan_batches(machine):
        emit(b)
    n = 0
    for repo in repos:
        repo = os.path.abspath(os.path.expanduser(repo))
        model_index = build_model_index(
            repo, model_subdirs or list(DEFAULT_MODEL_SUBDIRS))
        for sub in sacred_subdirs:
            root = sub if os.path.isabs(sub) else os.path.join(repo, sub)
            if not os.path.isdir(root):
                continue
            tag = _root_tag(root)
            for run_dir in iter_run_dirs(root):
                rec = scan_run_dir(run_dir, machine, tag, tail_bytes,
                                   repo=repo, model_index=model_index)
                if rec is not None:
                    emit(rec)
                    n += 1
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

    scan(args.repo, args.sacred_subdir or list(DEFAULT_SACRED_SUBDIRS),
         args.machine, args.tail_bytes, emit,
         model_subdirs=args.model_subdir or list(DEFAULT_MODEL_SUBDIRS))
    sys.stdout.flush()
    return 0


# ===========================================================================
# ここからローカル側 (集約・整形・Notion)
# ===========================================================================

def parse_dt(s):
    """sacred の時刻は naive UTC。aware な datetime に変換する."""
    if not s:
        return None
    s = str(s).replace("Z", "")
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.strptime(s[:26], fmt)
            break
        except ValueError:
            continue
    else:
        return None
    try:
        from datetime import timezone
        return dt.replace(tzinfo=timezone.utc)
    except ImportError:
        return dt


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
    for m in LARE_SEG_RE.finditer(str(name)):
        steps = float(m.group(3))
        parts.append("%s_%s %gM" % (m.group(1), m.group(2), steps))
    # 元の管理表の表記に合わせて矢印で繋ぐ (例: "8x5_2 10M→aoba00_2 5M")
    return "→".join(parts) if parts else str(name)


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


def derive(rec, stale_minutes, method_tag_map=None, expected_t_max=None):
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

    # task arrival 列
    if env.get("randomize_task_arrival"):
        d["task_arrival"] = "bernoulli, mmpp"
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
        stale = (last is None) or (now_utc() - last > timedelta(minutes=stale_minutes))
        d["state"] = "stalled" if stale else "running"
    else:
        d["state"] = "unknown"

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

    # 想定 t_max との照合. モデルの選択には使わない (使うのは run の t_max) が、
    # 「30M 回すつもりだったのに 10M で止めた run」を取り違えないための確認用
    exp = lookup_expected_t_max(expected_t_max, d.get("map"), d.get("agents"))
    d["t_max_expected"] = exp
    if exp is None or not t_max:
        d["t_max_ok"] = None
    else:
        # t_max は 20050000 のように +50000 されている運用なので M 単位で丸めて比べる
        d["t_max_ok"] = round(t_max / 1e6) == round(exp / 1e6)

    d["param_hash"] = rec.get("param_hash")
    d["model"] = rec.get("model")
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
    t_ref = beat or stop
    if d["state"] == "running" and start and t_ref and t_last and t_max:
        elapsed = (t_ref - start).total_seconds()
        if elapsed > 0 and t_last > 0 and t_last < t_max:
            rate = t_last / elapsed
            if rate > 0:
                d["rate"] = rate
                d["remaining_sec"] = (t_max - t_last) / rate
                d["eta"] = t_ref + timedelta(seconds=d["remaining_sec"])
    # map を必ず含める。入れないと 3agent 8M iql が map_5x4 / map_8x5 / map_aoba01 を
    # 1 条件に混ぜてしまい、「done seeds」が別マップ混在の一覧になる
    d["condition"] = "%sagent %s %sM | %s | %s | %s | %s" % (
        d["agents"], d["map"], d["group_m"], d["algo"], d["setting"],
        d["task_arrival"], d["task_assign"] or "-")
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


def collect_local(host, tail_bytes):
    recs = []
    scan(host.get("repos") or ["."],
         host.get("sacred_subdirs") or list(DEFAULT_SACRED_SUBDIRS),
         host["label"], tail_bytes, recs.append,
         model_subdirs=host.get("model_subdirs") or list(DEFAULT_MODEL_SUBDIRS))
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
                multiplex=False):
    """このスクリプト自身を stdin で送り込み、リモートで --scan させる."""
    quote = _quote
    py = host.get("python") or "python3"
    remote_args = ["--scan", "--machine", host["label"],
                   "--tail-bytes", str(tail_bytes)]
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


def _collect_ssh_run(host, py, remote_args, timeout, verbose, multiplex):
    quote = _quote
    remote_cmd = "%s - %s" % (py, " ".join(quote(a) for a in remote_args))

    # ssh は「最初に指定されたオプションが勝つ」ので、host 側の指定を先に置く
    cmd = ["ssh", "-o", "BatchMode=yes"]
    for opt in host.get("ssh_options") or []:
        cmd += ["-o", opt]
    cmd += ["-o", "ConnectTimeout=%d" % int(host.get("connect_timeout", 10))]
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

    recs = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("{"):
                try:
                    recs.append(json.loads(line))
                except ValueError:
                    pass
    age_h = (now_utc().timestamp() - os.path.getmtime(path)) / 3600.0
    if stale_hours and age_h > stale_hours:
        sys.stderr.write("[warn] %s: drop data is %.1f h old (that machine may have "
                         "stopped exporting, or the folder is not syncing)\n"
                         % (host["label"], age_h))
    return recs, None


def export_drop(records, rows, drop_dir, what="path", with_optimizer=False,
                max_mb=2000.0, dry_run=False):
    """自分の run 一覧とモデルを共有フォルダへ書き出す (--export).

    共有フォルダは iCloud Drive / Dropbox / NFS / USB など何でもよい。
    集約側はこのフォルダを読むだけなので、SSH は要らない。
    """
    import shutil
    by_uid = dict((d["uid"], d) for d in rows)
    models_root = os.path.join(drop_dir, "models")
    copied, total = 0, 0.0

    for rec in records:
        d = by_uid.get(rec.get("uid"))
        model = rec.get("model")
        if d is None or not model:
            continue
        files = select_model_files(model, what, with_optimizer)
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
        if dry_run:
            continue
        for rel, sub, _size in files:
            src = os.path.join(rec.get("repo") or ".", rel)
            dst = os.path.join(out_dir, sub)
            try:
                os.makedirs(os.path.dirname(dst))
            except OSError:
                pass
            shutil.copy2(src, dst)

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

    sys.stderr.write("[export] %d run(s), %d model dir(s) newly copied (%.1f MB) -> %s%s\n"
                     % (len(records), copied, total, drop_dir,
                        " (dry run)" if dry_run else ""))
    return copied


def collect(hosts, tail_bytes, timeout, verbose=False, drop_root=None):
    all_recs, errors = [], []
    for host in hosts:
        label = host["label"]
        if host.get("drop"):
            recs, err = collect_drop(host, drop_root)
        elif host.get("ssh") in (None, "", "local"):
            recs, err = collect_local(host, tail_bytes)
        else:
            recs, err = collect_ssh(host, tail_bytes, timeout, verbose)
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


def dashboard_data(rows, batches, errors):
    """HTML に埋め込む JSON を組み立てる."""
    odd = set(d["uid"] for d in odd_param_runs(rows))
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
            "state": d.get("state"),
            "progress": d.get("progress"), "t_last": d.get("t_last"),
            "duration": d.get("duration"),
            "eta": d["eta"].isoformat() if d.get("eta") else None,
            "remaining_sec": d.get("remaining_sec"),
            "last_seen": d["last_seen"].isoformat() if d.get("last_seen") else None,
            "param_hash": d.get("param_hash"), "odd_params": d["uid"] in odd,
            "t_max_ok": d.get("t_max_ok"), "t_max_expected": d.get("t_max_expected"),
            "run_dir": d.get("run_dir"), "params": dash_params(d),
        })
    mach = {}
    for b in batches or []:
        m = mach.setdefault(b.get("machine"), {"batches": 0, "reserved": 0,
                                               "unknown": False})
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


def render_summary(rows):
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
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 学習済みモデルの回収
# ---------------------------------------------------------------------------

def _rel_under_step(model, rel):
    """model["dir"]/<step>/ より下の相対パスを返す (例: "path/agent.th")."""
    head = "%s/%s/" % (model["dir"].replace(os.sep, "/"), model["step"])
    rel = rel.replace(os.sep, "/")
    return rel[len(head):] if rel.startswith(head) else os.path.basename(rel)


def select_model_files(model, what="path", with_optimizer=False):
    """回収するファイルを絞る.

    epymarl の新しい保存形は <step>/path/{agent,mixer,opt}.th と <step>/task/*.th。
    古い保存形は <step>/ 直下に .th が並ぶので、path/ が無ければ全部を対象にする。
    opt.th / agent_opt.th / critic_opt.th (optimizer state) は評価に要らないので既定で外す。
    """
    picked = []
    has_path = any(_rel_under_step(model, r).startswith("path/") for r, _ in model["files"])
    for rel, size in model["files"]:
        sub = _rel_under_step(model, rel)
        if what == "path" and has_path and not sub.startswith("path/"):
            continue
        if not with_optimizer and os.path.basename(sub).endswith("opt.th"):
            continue
        picked.append((rel, sub, size))
    return picked


def model_dest_name(d):
    return "%s_%s_%sagents_seed%s_step%s" % (
        d.get("algo"), d.get("map"), d.get("agents"), d.get("seed"),
        (d.get("model") or {}).get("step"))


def eval_model_filename(d):
    """評価側 (src/all_policy/policy.py の model_stem) が探すファイル名を組み立てる。

    stem = {map}_{model_n}_{path_planner}[_{method_tag}]_{reassign_tag}
    reassign_tag は学習時に決まる軸ではないので常に "base"
    (reassign 側のモデルが要るときは手で名前を変える)。
    """
    tag = "_%s" % d["method_tag"] if d.get("method_tag") else ""
    return "%s_%s_%s%s_base_seed%s.th" % (d.get("map"), d.get("agents"),
                                          d.get("algo"), tag, d.get("seed"))


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
    cmd += ["-o", "ConnectTimeout=%d" % int(host.get("connect_timeout", 10))]
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
    """回収記録 (manifest.jsonl) と設置コマンド案 (install_hints.sh) を追記する."""
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
    policy = policy_file_in(out_dir)
    if policy:
        with open(os.path.join(dest, "install_hints.sh"), "a") as f:
            f.write("cp %s src/all_policy/models/safe/%s\n"
                    % (_quote(policy), eval_model_filename(d)))


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
            shutil.copy2(src, dst)
    return None


def fetch_models(hosts, rows, dest, what="path", with_optimizer=False,
                 max_mb=2000.0, timeout=900, dry_run=False, drop_root=None):
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
        files = select_model_files(model, what, with_optimizer)
        if os.path.isdir(out_dir) and os.listdir(out_dir):
            skipped += 1
            present.append((d, out_dir, files))
            continue
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


def publish_dir(d):
    """保管用リポジトリ内の配置先ディレクトリ.

        path/{map}/{N}agent/{planner}_{method_tag}/
    """
    tag = d.get("method_tag") or ""
    planner_dir = "%s_%s" % (d.get("algo"), tag) if tag else str(d.get("algo"))
    return os.path.join("path", str(d.get("map")), "%sagent" % d.get("agents"),
                        planner_dir)


def publish_names(d, sub):
    """保管用リポジトリでのファイル名を決める.

    方策本体は**評価側がそのまま読める名前**にする (cp するだけで済む)。
    評価に必要なのは agent.th 1 本だけ (QMIX も MAT も単一の .th を torch.load する)。
    mixer / critic は学習専用なので、拾ってあれば接尾辞を付けて併置する。
    step と param_hash はファイル名に入れず manifest.jsonl に持つ。
    """
    eval_name = eval_model_filename(d)
    base = os.path.basename(sub)
    if base == "agent.th":
        return eval_name
    stem = eval_name[:-3] if eval_name.endswith(".th") else eval_name
    return "%s.%s" % (stem, base)          # 例: ..._seed123.mixer.th


def publish_models(planned, repo_dir, dry_run=False, hint=True):
    """回収したモデルを保管用リポジトリの階層へ配置する.

    git の commit / push はしない (外向きの操作なので明示的にやってもらう)。
    """
    import shutil
    repo_dir = os.path.abspath(os.path.expanduser(repo_dir))
    published = skipped = 0
    lines = []
    refused, tmax_warn = 0, 0
    for d, out_dir, files in planned:
        # 異常終了した run のモデルは保管しない。--fetch-state を広げて調査した
        # ときにも巻き込まれないよう、ここでも必ず弾く
        if d.get("state") != "done":
            refused += 1
            continue
        if d.get("t_max_ok") is False:
            tmax_warn += 1
        rel = publish_dir(d)
        dst_dir = os.path.join(repo_dir, rel)
        srcs = []
        for _r, sub, _z in files:
            if os.path.basename(sub).endswith("opt.th"):
                continue
            src = os.path.join(out_dir, sub)
            # dry run では回収前なのでファイルがまだ無い。配置先だけ見せる
            if os.path.exists(src) or dry_run:
                srcs.append((src, publish_names(d, sub)))
        if not srcs:
            continue
        if all(os.path.exists(os.path.join(dst_dir, n)) for _s, n in srcs):
            skipped += 1
            continue
        sys.stderr.write("  %s/%s\n" % (rel, srcs[0][1]))
        published += 1
        if dry_run:
            continue
        try:
            os.makedirs(dst_dir)
        except OSError:
            pass
        for src, name in srcs:
            shutil.copy2(src, os.path.join(dst_dir, name))
        lines.append({
            "uid": d["uid"], "machine": d.get("machine"),
            "path": rel, "eval_name": eval_model_filename(d),
            "t_max_expected": d.get("t_max_expected"), "t_max_ok": d.get("t_max_ok"),
            "map": d.get("map"), "agents": d.get("agents"), "algo": d.get("algo"),
            "seed": d.get("seed"), "t_max": d.get("t_max"),
            "step": (d.get("model") or {}).get("step"),
            "method_tag": d.get("method_tag"), "lare_mode": d.get("lare_mode"),
            "setting": d.get("setting"), "task_arrival": d.get("task_arrival"),
            "task_assign": d.get("task_assign"), "param_hash": d.get("param_hash"),
            "files": [n for _s, n in srcs],
        })
    if lines and not dry_run:
        try:
            os.makedirs(repo_dir)
        except OSError:
            pass
        with open(os.path.join(repo_dir, "manifest.jsonl"), "a") as f:
            for rec in lines:
                f.write(json.dumps(rec, ensure_ascii=True, default=str) + "\n")
    sys.stderr.write("[publish] %d placed, %d already there -> %s%s\n"
                     % (published, skipped, repo_dir,
                        " (dry run)" if dry_run else ""))
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


def install_models(planned, models_dir, overwrite=False, dry_run=False):
    """回収したモデルを評価側が読む名前で src/all_policy/models/safe/ に置く."""
    import shutil
    models_dir = os.path.abspath(os.path.expanduser(models_dir))
    installed = skipped = missing = 0
    for d, out_dir, _files in planned:
        src = policy_file_in(out_dir)
        if src is None and not dry_run:
            missing += 1
            continue
        dst = os.path.join(models_dir, eval_model_filename(d))
        if src is None:
            # dry run では回収前なのでファイルがまだ無い。付く名前だけ見せる
            sys.stderr.write("  (not fetched yet) -> %s\n" % os.path.basename(dst))
            installed += 1
            continue
        if os.path.exists(dst) and not overwrite:
            skipped += 1
            continue
        sys.stderr.write("  %s -> %s\n" % (os.path.basename(src), os.path.basename(dst)))
        if not dry_run:
            try:
                os.makedirs(models_dir)
            except OSError:
                pass
            shutil.copy2(src, dst)
        installed += 1
    sys.stderr.write("[install] %d installed, %d already there, %d without a policy file%s\n"
                     % (installed, skipped, missing, " (dry run)" if dry_run else ""))
    return installed


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
        base += ["-o", "ConnectTimeout=%d" % int(host.get("connect_timeout", 10))]
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

    # 軽い問い合わせでは repo / model を取らないので、キャッシュの値を残す
    for r in fresh:
        old_rec = by_uid.get(r.get("uid")) or {}
        for k in ("repo", "model"):
            if not r.get(k) and old_rec.get(k):
                r[k] = old_rec[k]

    rows = [derive(r, stale, tag_map) for r in fresh]
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
    return "seed %s @%s (%sagent %s %sM %s)" % (
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
            head.append("+%d more" % (len(items) - max_lines))
        return " / ".join(head)

    if finished:
        notify("LDRP", "Training finished (%d)" % len(finished),
               summary(finished), sound="Glass")
    if problems:
        notify("LDRP", "Runs need attention (%d)" % len(problems),
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
    import yaml
    return yaml.safe_load(text) or {}


def build_hosts(conf, args):
    if args.hosts:
        labels = [h.strip() for h in args.hosts.split(",") if h.strip()]
        if labels == ["local"]:
            return [{"label": args.machine or conf.get("machine") or os.uname()[1],
                     "ssh": "local",
                     "repos": args.repo or conf.get("repos") or [os.getcwd()]}]
        hosts = [h for h in conf.get("hosts", []) if h.get("label") in labels]
        missing = set(labels) - set(h["label"] for h in hosts)
        if missing:
            sys.stderr.write("[warn] unknown host label(s): %s\n" % ", ".join(sorted(missing)))
        return hosts
    hosts = conf.get("hosts") or []
    if not hosts:
        hosts = [{"label": args.machine or os.uname()[1], "ssh": "local",
                  "repos": args.repo or [os.getcwd()]}]
    return hosts


def dedupe(rows):
    """同じ uid が複数回来たら最後の 1 件を残す."""
    seen = {}
    for d in rows:
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
    p.add_argument("--repo", action="append", default=None,
                   help="repository root to scan (repeatable)")
    p.add_argument("--sacred-subdir", action="append", default=None,
                   help="sacred dir relative to repo root (repeatable)")
    p.add_argument("--model-subdir", action="append", default=None,
                   help="saved-model dir relative to repo root (repeatable)")
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
    p.add_argument("--fetch-what", default="path", choices=["path", "all"],
                   help="'path' = policy only (default), 'all' = task assigner too")
    p.add_argument("--fetch-optimizer", action="store_true",
                   help="also fetch opt.th (optimizer state; not needed to evaluate)")
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
    p.add_argument("--publish-commit", action="store_true",
                   help="commit the archive repository after placing models "
                        "(does nothing when there is no change)")
    p.add_argument("--publish-push", action="store_true",
                   help="also push. implies --publish-commit")
    p.add_argument("--install-models", action="store_true",
                   help="also copy each fetched policy into the evaluation model dir "
                        "under the name test.py expects")
    p.add_argument("--models-dir", default="src/all_policy/models/safe",
                   help="where --install-models puts the files")
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
        rows = [derive(r, stale_m, tag_map) for r in records]
        if min_steps:
            keep = set(d["uid"] for d in rows if (d.get("t_max") or 0) >= min_steps)
            records = [r for r in records if r["uid"] in keep]
            rows = [d for d in rows if d["uid"] in keep]
        want = set(x.strip() for x in (args.fetch_state or "done").split(",") if x.strip())
        export_drop(records, [d for d in rows if d["state"] in want],
                    os.path.join(os.path.abspath(os.path.expanduser(args.export)), label),
                    what=args.fetch_what, with_optimizer=args.fetch_optimizer,
                    max_mb=args.max_fetch_mb, dry_run=args.fetch_dry_run)
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
    tail_bytes = args.tail_bytes or conf.get("tail_bytes") or 65536
    hosts = build_hosts(conf, args)
    if not hosts:
        p.error("no hosts to visit")

    drop_root = args.drop_root or conf.get("drop_root")
    raw, errors = collect(hosts, tail_bytes, args.ssh_timeout, args.verbose,
                          drop_root=drop_root)

    # 前回の state はキャッシュに書き込んである (_state). 再導出すると
    # 「到達できないホストの running が時間経過で stalled に変わる」等でぶれるので、
    # 保存時点の値をそのまま比較に使う
    # バッチ (train.py の実行予定) は「いまの状態」なのでキャッシュしない。
    # 古い予約が残ると、終わったバッチの枠がいつまでも埋まって見える
    batches = [r for r in raw if r.get("kind") == "batch"]
    raw = [r for r in raw if r.get("kind") != "batch"]

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
    rows = [derive(r, stale, tag_map, exp_t_max) for r in dedupe(raw)]

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
        text = render_summary(rows)
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

    if args.fetch_models or args.fetch_dry_run:
        want = set(x.strip() for x in (args.fetch_state or "done").split(",") if x.strip())
        targets = [d for d in rows if d["state"] in want]
        planned, _n = fetch_models(
            hosts, targets, args.fetch_models or "models_inbox",
            what=args.fetch_what, with_optimizer=args.fetch_optimizer,
            max_mb=args.max_fetch_mb, timeout=args.fetch_timeout,
            dry_run=args.fetch_dry_run or not args.fetch_models,
            drop_root=drop_root)
        if args.publish_models:
            dry = args.fetch_dry_run or not args.fetch_models
            auto = args.publish_commit or args.publish_push
            n_pub = publish_models(planned, args.publish_models, dry_run=dry,
                                   hint=not auto)
            if auto and not dry:
                publish_commit(args.publish_models, n_pub, push=args.publish_push)
        if args.install_models:
            install_models(planned, args.models_dir,
                           overwrite=args.overwrite_installed,
                           dry_run=args.fetch_dry_run or not args.fetch_models)

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
