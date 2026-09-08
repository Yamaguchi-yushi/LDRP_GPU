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


CR = _load("collect_runs")
ER = _load("eval_report")
PLAN = _load("plan")


# ---------------------------------------------------------------------------
# データ
# ---------------------------------------------------------------------------

class State(object):
    """収集結果を持ち回す。GET はここを読むだけにする."""

    def __init__(self, args):
        self.args = args
        self.lock = threading.RLock()
        self.busy = False
        self.collected_at = None
        self.errors = []
        self.batches = []          # train.py のバッチ。キャッシュには入れない

    # --- 設定 -----------------------------------------------------------
    def conf(self):
        path = os.path.expanduser(self.args.config)
        return CR.load_config(path) if os.path.exists(path) else {}

    # --- 計画 -----------------------------------------------------------
    def plan_view(self, rows, shaped):
        """**表は計画から作る**。実績はそこに埋めていく.

        計画に無い run は conditions 表には出さない (実行中セクションには出る)。
        こうすると「あと何を回せばよいか」が表そのものになる。

        slots に入れる run は **dashboard_data() が整形したもの** (shaped)。
        derive() の生レコードを入れると odd_params などが欠けて画面側が壊れる。
        """
        path = os.path.expanduser(self.args.plan)
        if not os.path.exists(path):
            return [], set()
        try:
            conds = PLAN.parse_plan(path)
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
            out.append({
                "label": PLAN.label(c), "map": c["map"], "agents": c["agents"],
                "t_max_m": c["t_max_m"], "setting": c["setting"],
                "algo": c["algo"], "task_arrival": c["task_arrival"],
                "task_assign": c["task_assign"], "reassign": c["reassign"],
                "dynamic": c.get("dynamic"),
                "want": c["want"], "slots": slots + extra,
            })
        # マップ名 -> 台数 -> t_max の順に並べる
        out.sort(key=lambda c: (str(c["map"]), c["agents"], c["t_max_m"],
                                str(c["setting"]), str(c["algo"])))
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
            data = CR.dashboard_data(rows, self.batches, self.errors)
            shaped = dict((r["uid"], r) for r in data["runs"])
            plan, used = self.plan_view(rows, shaped)
            for r in data["runs"]:
                r["in_plan"] = r["uid"] in used
            data["plan"] = plan
            data["plan_file"] = os.path.expanduser(self.args.plan)
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
            hosts = conf.get("hosts") or [
                {"label": conf.get("machine") or os.uname()[1], "ssh": "local",
                 "repos": [REPO]}]
            raw, errors = CR.collect(hosts, conf.get("tail_bytes") or 65536,
                                     self.args.ssh_timeout,
                                     drop_root=conf.get("drop_root"))
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
            self.end_headers()
            self.wfile.write(body)

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

    sys.stderr.write("[dashboard] stdlib http.server  http://%s:%d\n"
                     "[dashboard] (pip install flask すると flask で動きます)\n"
                     % (host, port))
    ThreadingHTTPServer((host, port), H).serve_forever()


def main(argv=None):
    p = argparse.ArgumentParser(description="LDRP experiment dashboard.")
    p.add_argument("-c", "--config", default=os.path.join(TOOLS, "collect_config.yaml"))
    p.add_argument("--cache", default=os.path.join(TOOLS, ".run_cache.jsonl"),
                   help="collect_runs.py --cache と同じファイル")
    p.add_argument("-p", "--plan", default=os.path.join(TOOLS, "plan.md"),
                   help="実験計画 (Notion の表をそのまま貼れる)")
    p.add_argument("-s", "--summary", default=os.path.join(REPO, "results", "summary.csv"))
    p.add_argument("--host", default="127.0.0.1", help="既定は localhost のみ")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--ssh-timeout", type=int, default=180)
    p.add_argument("--collect-on-start", action="store_true",
                   help="起動時に 1 回収集する (既定はキャッシュを読むだけ)")
    args = p.parse_args(argv)

    state = State(args)
    if args.collect_on_start:
        state.collect()

    try:
        import flask                                     # noqa: F401
    except ImportError:
        return serve_stdlib(state, args.host, args.port)
    return serve_flask(state, args.host, args.port)


if __name__ == "__main__":
    sys.exit(main())
