#!/usr/bin/env python3
"""デスクトップに常駐する小さい進捗パネル。クリックでダッシュボードを開く。

    python tools/dashboard/mini.py

付箋のように常に最前面に浮かべておき、実験の進み具合を横目で見るためのもの。
細かい話はダッシュボードで見るので、ここに出すのは **次に手を動かすかどうかの
判断に要るものだけ** にする:

    ✔ 完了した条件 / 全条件      ← 論文に載る本数
    残り                         ← あと何 seed 回せばよいか
    ▶ 実行中                     ← いま計算機が遊んでいないか
    ⚠ 沈黙しているマシン          ← 気づかないと何日も無駄になる

データはダッシュボードの /api/mini (1KB 未満) から取る。サーバが落ちていれば
~/.ldrp/status.txt (launchd が 15 分ごとに書く) に落とし、それも無ければ
「停止中」と出す。**パネルのためにキャッシュを読み直すことはしない** (毎分
688 run を解析すると、見るだけのものが一番重い処理になってしまう)。

  クリック            ダッシュボードをブラウザで開く (落ちていれば起動してから)
  ドラッグ            移動。位置は ~/.ldrp/mini_pos.json に覚える
  右クリック          メニュー (いま収集 / 位置のリセット / 終了)
"""

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
STATE_DIR = os.path.expanduser("~/.ldrp")
POS_PATH = os.path.join(STATE_DIR, "mini_pos.json")
STATUS_TXT = os.path.join(STATE_DIR, "status.txt")

BG = "#1b1d23"
FG = "#e8eaed"
MUT = "#8b93a1"
OK = "#4ade80"
WARN = "#fbbf24"
ERR = "#f87171"


def fetch(url, timeout=4):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return None


def parse_dt(iso):
    if not iso:
        return None
    import datetime
    try:
        return datetime.datetime.fromisoformat(str(iso))
    except ValueError:
        return None


def since(iso):
    """ISO 文字列から「何秒前か」。読めなければ None."""
    import datetime
    t = parse_dt(iso)
    if t is None:
        return None
    now = datetime.datetime.now(t.tzinfo) if t.tzinfo else datetime.datetime.now()
    return max(0.0, (now - t).total_seconds())


def when(iso):
    """終了時刻を地元時間で `09/05 12:58`。記録は UTC なので必ず変換する."""
    t = parse_dt(iso)
    if t is None:
        return None
    if t.tzinfo is not None:
        t = t.astimezone()
    return t.strftime("%m/%d %H:%M")


def dur(sec):
    if sec is None:
        return "-"
    sec = int(sec)
    if sec < 3600:
        return "%dm" % (sec // 60)
    if sec < 86400:
        return "%dh%02dm" % (sec // 3600, (sec % 3600) // 60)
    return "%dd" % (sec // 86400)


def from_status_txt():
    """サーバが落ちているときの代替。status.txt の最終行から done 数だけ拾う."""
    try:
        with open(STATUS_TXT) as f:
            text = f.read()
        age = time.time() - os.path.getmtime(STATUS_TXT)
    except OSError:
        return None
    run = 0
    for line in text.splitlines():
        if line.startswith("▶"):
            try:
                run = int(line.split()[1])
            except (IndexError, ValueError):
                pass
            break
    return {"offline": True, "running": run, "age": age}


class Panel(object):
    def __init__(self, url, every, opacity):
        import tkinter as tk
        self.tk = tk
        self.url = url.rstrip("/")
        self.every = max(15, every) * 1000
        self.data = None

        self.root = tk.Tk()
        self.root.title("LDRP")
        self.root.configure(bg=BG)
        self.root.overrideredirect(True)          # 枠なし = 付箋のように見せる
        self.root.attributes("-topmost", True)
        try:
            self.root.attributes("-alpha", opacity)
        except tk.TclError:
            pass

        pad = tk.Frame(self.root, bg=BG, padx=12, pady=9)
        pad.pack(fill="both", expand=True)

        self.l_head = tk.Label(pad, text="LDRP", bg=BG, fg=MUT,
                               font=("Helvetica Neue", 10), anchor="w")
        self.l_main = tk.Label(pad, text="…", bg=BG, fg=FG,
                               font=("Helvetica Neue", 20, "bold"), anchor="w")
        self.l_sub = tk.Label(pad, text="", bg=BG, fg=MUT,
                              font=("Helvetica Neue", 11), anchor="w", justify="left")
        self.l_head.pack(fill="x")
        self.l_main.pack(fill="x")
        self.l_sub.pack(fill="x")

        # マシンごとの行。実行中の本数と「最後に終わったのはいつ・何か」。
        # 実行中 0 でも直前の完了が何日も前なら、そのマシンは遊んでいる
        self.grid = tk.Frame(pad, bg=BG)
        self.grid.pack(fill="x", pady=(7, 0))
        self.rows = []
        for i in range(8):                 # マシンは増えうるので多めに用意しておく
            name = tk.Label(self.grid, bg=BG, fg=FG, anchor="w",
                            font=("Helvetica Neue", 11, "bold"))
            run = tk.Label(self.grid, bg=BG, fg=MUT, anchor="e",
                           font=("Helvetica Neue", 11))
            when = tk.Label(self.grid, bg=BG, fg=MUT, anchor="w",
                            font=("Helvetica Neue", 11))
            name.grid(row=i, column=0, sticky="w", padx=(0, 8))
            run.grid(row=i, column=1, sticky="e", padx=(0, 8))
            when.grid(row=i, column=2, sticky="w")
            self.rows.append((name, run, when))
        self.grid.grid_columnconfigure(2, weight=1)

        self.l_warn = tk.Label(pad, text="", bg=BG, fg=WARN,
                               font=("Helvetica Neue", 10), anchor="w", justify="left")
        self.l_warn.pack(fill="x", pady=(6, 0))

        widgets = [self.root, pad, self.grid, self.l_head, self.l_main,
                   self.l_sub, self.l_warn]
        for r in self.rows:
            widgets.extend(r)
        for w in widgets:
            w.bind("<Button-1>", self.press)
            w.bind("<B1-Motion>", self.drag)
            w.bind("<ButtonRelease-1>", self.release)
            w.bind("<Button-2>", self.menu)
            w.bind("<Button-3>", self.menu)

        self.m = tk.Menu(self.root, tearoff=0)
        self.m.add_command(label="ダッシュボードを開く", command=self.open_dash)
        self.m.add_command(label="いま収集する", command=self.collect)
        self.m.add_separator()
        self.m.add_command(label="位置をリセット", command=self.reset_pos)
        self.m.add_command(label="終了", command=self.root.destroy)

        self.restore_pos()
        self.refresh()

    # -- 位置 -------------------------------------------------------------
    def restore_pos(self):
        x, y = None, None
        try:
            with open(POS_PATH) as f:
                p = json.load(f)
            x, y = int(p["x"]), int(p["y"])
        except (OSError, ValueError, KeyError):
            pass
        if x is None:
            # 既定は右上。メニューバーに被らないよう少し下げる
            self.root.update_idletasks()
            x = self.root.winfo_screenwidth() - 260
            y = 60
        self.root.geometry("+%d+%d" % (x, y))

    def save_pos(self):
        try:
            os.makedirs(STATE_DIR, exist_ok=True)
            with open(POS_PATH, "w") as f:
                json.dump({"x": self.root.winfo_x(), "y": self.root.winfo_y()}, f)
        except OSError:
            pass

    def reset_pos(self):
        try:
            os.remove(POS_PATH)
        except OSError:
            pass
        self.restore_pos()

    # -- マウス -----------------------------------------------------------
    def press(self, e):
        self._dx, self._dy = e.x_root - self.root.winfo_x(), e.y_root - self.root.winfo_y()
        self._moved = False

    def drag(self, e):
        self._moved = True
        self.root.geometry("+%d+%d" % (e.x_root - self._dx, e.y_root - self._dy))

    def release(self, e):
        # ドラッグしたのか、ただ押したのかで分ける。付箋を動かすたびに
        # ブラウザが開くと使い物にならない
        if getattr(self, "_moved", False):
            self.save_pos()
        else:
            self.open_dash()

    def menu(self, e):
        try:
            self.m.tk_popup(e.x_root, e.y_root)
        finally:
            self.m.grab_release()

    # -- 動作 -------------------------------------------------------------
    def open_dash(self):
        if self.data is None or self.data.get("offline"):
            self.start_server()
        subprocess.Popen(["open", self.url])

    def start_server(self):
        """落ちていたら起動する。conda の python を使う (PyYAML が要る)."""
        py = "/opt/anaconda3/envs/ldrp/bin/python"
        if not os.path.exists(py):
            py = sys.executable
        try:
            subprocess.Popen([py, os.path.join(HERE, "app.py")],
                             cwd=REPO, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        except OSError:
            return
        for _ in range(20):                # 立ち上がるまで待つ (最大 10 秒)
            time.sleep(0.5)
            if fetch(self.url + "/api/status", timeout=1):
                return

    def collect(self):
        try:
            req = urllib.request.Request(self.url + "/api/collect", data=b"{}",
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=4).read()
        except (urllib.error.URLError, OSError):
            pass

    # -- 表示 -------------------------------------------------------------
    def set_rows(self, machines):
        for i, (l_name, l_run, l_when) in enumerate(self.rows):
            if i >= len(machines):
                for w in (l_name, l_run, l_when):
                    w.grid_remove()
                continue
            m = machines[i]
            for w in (l_name, l_run, l_when):
                w.grid()
            l_name.config(text=m["name"],
                          fg=ERR if m.get("stale") else FG)
            n = m.get("running") or 0
            l_run.config(text=("▶%d" % n) if n else "—",
                         fg=OK if n else MUT)
            ago = since(m.get("last_done"))
            if ago is None:
                l_when.config(text="完了なし", fg=MUT)
            else:
                # 直前の完了が古いほど「遊んでいる」。3 日以上は黄色で示す
                l_when.config(text="%s  %s" % (when(m.get("last_done")),
                                               m.get("last_what") or ""),
                              fg=WARN if (not n and ago > 3 * 86400) else MUT)

    def refresh(self):
        d = fetch(self.url + "/api/mini") or from_status_txt()
        self.data = d
        if d is None:
            self.l_head.config(text="LDRP", fg=MUT)
            self.l_main.config(text="停止中", fg=ERR)
            self.l_sub.config(text="クリックで起動")
            self.set_rows([])
            self.l_warn.config(text="")
        elif d.get("offline"):
            self.l_head.config(text="LDRP  (サーバ停止中)", fg=MUT)
            self.l_main.config(text="▶ %d" % d["running"], fg=FG)
            self.l_sub.config(text="%s前の記録  クリックで起動" % dur(d["age"]))
            self.set_rows([])
            self.l_warn.config(text="")
        else:
            self.l_head.config(
                text="LDRP  %s" % ("収集中…" if d.get("busy") else "実験の進捗"),
                fg=MUT)
            self.l_main.config(text="%d / %d 条件" % (d["full"], d["conditions"]),
                               fg=OK if d["full"] >= d["conditions"] else FG)
            self.l_sub.config(text="残り %d seed    ▶ 実行中 %d"
                                   % (d["need"], d["running"]))
            machines = d.get("machines") or []
            self.set_rows(machines)
            quiet = [m for m in machines if m.get("stale")]
            idle = [m for m in machines if not m.get("stale") and not m.get("running")]
            msg = ""
            if quiet:
                msg = "⚠ %s が沈黙 (%s)" % (
                    " / ".join(m["name"] for m in quiet[:2]),
                    dur(max(m.get("age") or 0 for m in quiet)))
            elif idle:
                msg = "・空き: %s" % " ".join(m["name"] for m in idle)
            self.l_warn.config(text=msg, fg=ERR if quiet else MUT)
        self.root.after(self.every, self.refresh)

    def run(self):
        self.root.mainloop()


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(
        description="small always-on-top progress panel for the LDRP dashboard")
    p.add_argument("--url", default="http://127.0.0.1:8765",
                   help="dashboard URL (default: %(default)s)")
    p.add_argument("--every", type=int, default=60,
                   help="refresh interval in seconds (default: %(default)s)")
    p.add_argument("--opacity", type=float, default=0.93,
                   help="0.0-1.0 (default: %(default)s)")
    args = p.parse_args(argv)
    try:
        import tkinter                                          # noqa: F401
    except ImportError:
        sys.stderr.write("tkinter is not available in this python\n")
        return 1
    Panel(args.url, args.every, args.opacity).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
