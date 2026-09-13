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


# 表示順。ベースライン -> 提案 の順に並ぶようにする
ALGO_ORDER = ("qmix", "mappo", "mat", "mat_dec", "qplex", "iql", "vdn", "transf_qmix")


def _rank(seq, v):
    try:
        return seq.index(v)
    except ValueError:
        return len(seq)


CR = _load("collect_runs")
ER = _load("eval_report")
PLAN = _load("plan")


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
            time.sleep(n * 60)
            if not self.busy:
                self.collect()

    def __init__(self, args):
        self.args = args
        self.lock = threading.RLock()
        self.busy = False
        self.collected_at = None
        self.errors = []
        self.batches = []          # train.py のバッチ。キャッシュには入れない
        # 共有フォルダの runs.jsonl の mtime。変わっていなければ読み飛ばす
        self.drop_mtime = {}

    # --- 設定 -----------------------------------------------------------
    def conf(self):
        path = os.path.expanduser(self.args.config)
        return CR.load_config(path) if os.path.exists(path) else {}

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

    def plan_view(self, rows, shaped):
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

        used = set()
        out = []
        for c in conds:
            hit = [d for d in rows if PLAN.matches(c, d)]
            by_seed = {}
            for d in hit:
                by_seed.setdefault(str(d.get("seed")), []).append(d)

            slots = []
            for sl in c["seeds"]:
                sd = str(sl["seed"]) if sl["seed"] else None
                run = None
                if sd and by_seed.get(sd):
                    d = by_seed[sd].pop(0)
                    used.add(d["uid"])
                    run = shaped.get(d["uid"])
                slots.append({"seed": sd, "machine": sl.get("machine"),
                              "reassign": sl.get("reassign"), "run": run})
            # 計画に無い seed で回っているもの (捨てずに「計画外の seed」として出す)
            extra = []
            for sd, ds in by_seed.items():
                for d in ds:
                    used.add(d["uid"])
                    extra.append({"seed": sd, "machine": d.get("machine"),
                                  "run": shaped.get(d["uid"]), "unplanned_seed": True})
            # 同じ条件のはずなのにパラメータが割れていたら、**どのキーが違うか**を渡す。
            # ハッシュだけでは何を直せばよいか分からない
            pdiff, phashes = [], {}
            if len(set(d.get("param_hash") for d in hit)) > 1:
                diffs, seeds = CR.param_diff(hit)
                order = sorted(seeds, key=lambda h: (-len(seeds[h]), str(h)))
                phashes = [{"hash": h, "seeds": sorted(seeds[h])} for h in order]
                pdiff = [{"key": k,
                          "vals": ["-" if v.get(h) is None else str(v.get(h))
                                   for h in order]}
                         for k, v in diffs]

            out.append({
                "param_diff": pdiff, "param_hashes": phashes,
                "plan": c.get("plan"), "plan_file": c.get("plan_file"),
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
            str(c["map"]), c["agents"], c["t_max_m"],
            0 if c["setting"] == "safe" else 1, str(c["setting"]),
            _rank(ALGO_ORDER, c["algo"]), str(c["algo"] or ""),
            str(c["task_arrival"] or ""),
            0 if not c["task_assign"] else 1, str(c["task_assign"] or ""),
            1 if c.get("dynamic") else 0,
        ))
        return out, used

    # --- 学習 -----------------------------------------------------------
    def train(self):
        conf = self.conf()
        raw = CR.read_cache(os.path.expanduser(self.args.cache))
        raw = [r for r in raw if r.get("kind") != "batch"]
        stale = conf.get("stale_minutes") or 90
        rows = [CR.derive(r, stale, conf.get("method_tag_by_lare_mode"),
                          conf.get("expected_t_max")) for r in CR.dedupe(raw)]
        min_steps = conf.get("min_steps", 1e6)
        if min_steps:
            rows = [d for d in rows if (d.get("t_max") or 0) >= min_steps]
        with self.lock:
            data = CR.dashboard_data(
                rows, self.batches, self.errors,
                saved=CR.load_saved_models(self.args.models_repo),
                cadence=CR.record_host_seen(rows, hosts=self.conf().get("hosts")))
            shaped = dict((r["uid"], r) for r in data["runs"])
            plan, used = self.plan_view(rows, shaped)
            for r in data["runs"]:
                r["in_plan"] = r["uid"] in used
            data["plan"] = plan
            data["plan_files"] = self.plan_files()
            data["plan_file"] = ", ".join(data["plan_files"])
            data["collected_at"] = self.collected_at
            data["busy"] = self.busy
            data["cache"] = os.path.expanduser(self.args.cache)
        return data

    # --- 評価 -----------------------------------------------------------
    def eval(self):
        path = os.path.expanduser(self.args.summary)
        if not os.path.exists(path):
            return {"source": path, "available": False, "metrics": [],
                    "conditions": [],
                    "error": "summary.csv がありません (aggregate.py --csv を実行してください)"}
        rows, metrics = ER.load_summary(path)
        return {"source": path, "available": True, "metrics": metrics,
                "conditions": sorted(rows, key=ER.sort_key)}

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
            raw, errors = CR.collect(hosts, conf.get("tail_bytes") or 65536,
                                     self.args.ssh_timeout,
                                     drop_root=conf.get("drop_root"),
                                     skip_unchanged=self.drop_mtime)
            batches = [r for r in raw if r.get("kind") == "batch"]
            runs = [r for r in raw if r.get("kind") != "batch"]

            cache = os.path.expanduser(self.args.cache)
            merged = CR.dedupe(CR.read_cache(cache) + runs)
            stale = conf.get("stale_minutes") or 90
            rows = [CR.derive(r, stale, conf.get("method_tag_by_lare_mode"),
                              conf.get("expected_t_max")) for r in merged]
            by_uid = dict((d["uid"], d["state"]) for d in rows)
            for r in merged:
                r["_state"] = by_uid.get(r.get("uid"))
            CR.write_cache(cache, merged)

            with self.lock:
                self.batches = batches
                self.errors = ["%s: %s" % (a, b) for a, b in errors]
                self.collected_at = time.time()
        except Exception as e:                      # 画面を落とさない
            with self.lock:
                self.errors = ["collect failed: %s: %s" % (type(e).__name__, e)]
            sys.stderr.write("[collect] %s: %s\n" % (type(e).__name__, e))
        finally:
            with self.lock:
                self.busy = False

    def status(self):
        with self.lock:
            return {"busy": self.busy, "collected_at": self.collected_at,
                    "errors": self.errors}


# ---------------------------------------------------------------------------
# ルーティング (flask / 標準ライブラリのどちらからも同じものを呼ぶ)
# ---------------------------------------------------------------------------

def make_routes(state):
    return {
        ("GET", "/api/train"): lambda body: state.train(),
        ("GET", "/api/eval"): lambda body: state.eval(),
        ("GET", "/api/status"): lambda body: state.status(),
        ("POST", "/api/collect"): lambda body: {"started": state.collect()},
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
    p.add_argument("--collect-every", type=int, default=15, metavar="MIN",
                   help="この分数ごとに裏で収集する (0 で無効。既定 15)。"
                        "共有フォルダ側が 1 時間おきにしか更新されないので、"
                        "これより短くしても情報は増えない")
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
