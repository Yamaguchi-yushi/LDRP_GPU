#!/usr/bin/env python3
"""LLM が書いた LaRe の評価関数 (evaluation_func) を検査する.

design/lare_ldrp_prompt.md §5.1 の (3) 検査に当たる。LLM の応答 (JSON) から
Functions を取り出して実行し、プロンプト v1 の観測 (v2 の並び) のダミーで動かす。

    python tools/lare_factor_check.py design/prompts/responses/v1/candidate1.json
    python tools/lare_factor_check.py design/prompts/responses/v2/candidate1.json --layout v2

--layout はプロンプトの版 (観測の並び) に合わせる。v1: 6N+2NK+4K+21、evaluation_func(obs, apsp)。
v2: 7N+4NK+6K+26、evaluation_func(obs, apsp, adj)。

検査すること:
  - 例外が出ない / 戻り値が (batch, 1) の配列のリスト / NaN・inf が出ない
  - 因子がちょうど 10 個
  - 地図・台数を変えたダミー観測で動き、因子ごとの値の範囲が大きく変わらない

ダミー観測は実際の地図 (ノード座標・辺の長さ・最短路) を使い、機体の位置・段階・
イベントなどを乱数で作る。値の範囲の比較用で、学習中の分布とは違う。
誤りがあれば、そのままチャットに送れる英文 ("There is an error in your previous
answer. Error: ...") を最後に出す。
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
N_FACTORS = 10
CAPACITY = 10          # env の task_num (待てるタスクの上限)
SPEED = 5.0

# 検査に使う (地図, 台数, エピソード長)
CASES = [("map_8x5", 5, 1000), ("map_8x5", 10, 1000), ("map_aoba00", 7, 1000),
         ("map_aoba00", 10, 500), ("map_5x4", 3, 500)]


def load_map(map_name):
    """地図のノード座標・最短路の表・直径を、env と同じグラフから作る."""
    import networkx as nx
    sys.path.insert(0, os.path.join(REPO, "src", "main"))
    import gym
    import drp_env  # noqa: F401  (gym に環境を登録する)
    env = gym.make("drp_env:drp_safe-2agent_%s-v2" % map_name,
                   state_repre_flag="onehot_fov", time_limit=10, task_flag=True,
                   task_list=None, use_lare_path=False, verbose=False) \
        if "verbose" in gym.make.__code__.co_varnames else None
    if env is None:
        env = gym.make("drp_env:drp_safe-2agent_%s-v2" % map_name,
                       state_repre_flag="onehot_fov", time_limit=10, task_flag=True,
                       task_list=None, use_lare_path=False)
    e = env.unwrapped
    n = e.n_nodes
    coords = np.array([e.pos[i] for i in range(n)], dtype=float)
    apsp = np.full((n, n), np.inf)
    for s, d in nx.all_pairs_dijkstra_path_length(e.G, weight="weight"):
        for t, v in d.items():
            apsp[s, t] = v
    diam = float(np.max(apsp[np.isfinite(apsp)]))
    apsp[~np.isfinite(apsp)] = diam
    edges = [(u, v) for u, v in e.G.edges()]
    adj = np.zeros((n, n))
    for u, v, w in e.G.edges(data="weight"):
        adj[u, v] = adj[v, u] = float(w)
    return coords, apsp, diam, edges, float(e.colli_distan_value), adj


def random_location(rng, n, edges):
    """ノードの上 (one-hot) か辺の上 (両端に比率)."""
    loc = np.zeros(n)
    if rng.random() < 0.6:
        loc[rng.integers(n)] = 1.0
    else:
        u, v = edges[rng.integers(len(edges))]
        f = rng.uniform(0.05, 0.95)          # u から v へ f だけ進んだ
        loc[u], loc[v] = 1.0 - f, f
    return loc


def make_obs(rng, B, coords, apsp, diam, edges, cd, A, T):
    """プロンプト v1 の並び (6N + 2NK + 4K + 21) のダミー観測."""
    n = len(coords)
    K = A - 1
    rows = []
    for _ in range(B):
        phase_id = rng.integers(3)                    # 0 待機 / 1 ピックアップへ / 2 配達へ
        prev = random_location(rng, n, edges)
        curr = prev.copy() if rng.random() < 0.3 else random_location(rng, n, edges)
        dest = np.zeros(n)
        deliv = np.zeros(n)
        events = np.zeros(2)
        if phase_id != 0:
            dest[rng.integers(n)] = 1.0
            deliv[rng.integers(n) if phase_id == 1 else int(np.argmax(dest))] = 1.0
            if phase_id == 2 and rng.random() < 0.1:
                events[0] = 1.0                       # ピックアップした直後
        elif rng.random() < 0.1:
            events[1] = 1.0                           # 配達した直後
        phase = np.eye(3)[phase_id]
        since = 0.0 if phase_id == 0 else float(rng.integers(0, 200))
        age = 0.0 if phase_id == 0 else since + float(rng.integers(0, 100))
        others = []
        for _k in range(K):
            p = random_location(rng, n, edges)
            c = p.copy() if rng.random() < 0.3 else random_location(rng, n, edges)
            others += [p, c]
        others_phase = np.concatenate([np.eye(3)[rng.integers(3)] for _ in range(K)]) if K else np.zeros(0)
        others_active = np.ones(K)
        demand = np.array([float(rng.integers(0, CAPACITY + 1)), float(rng.integers(0, A + 1))])
        coll = np.array([float(rng.random() < 0.02), float(rng.random() < 0.02)])
        wait = 0.0 if phase_id == 0 else float(rng.integers(0, 50))
        override = float(rng.random() < 0.2)
        meta = np.array([diam, n, A, T, CAPACITY, float(rng.integers(0, T))])
        row = np.concatenate([prev, curr, dest, phase, deliv, events,
                              [since, age, 1.0],
                              np.concatenate(others) if K else np.zeros(0),
                              others_phase, others_active, demand, [cd], coll,
                              [wait, override], coords.reshape(-1), meta])
        assert len(row) == 6 * n + 2 * n * K + 4 * K + 21
        rows.append(row)
    return np.stack(rows)


def make_obs_v2(rng, B, coords, apsp, diam, edges, cd, A, T):
    """プロンプト v2 の並び (7N + 4NK + 6K + 26) のダミー観測."""
    n = len(coords)
    K = A - 1
    nbr = {i: [] for i in range(n)}
    for u, v in edges:
        nbr[u].append(v)
        nbr[v].append(u)

    def agent(rng):
        ph = rng.integers(3)
        prev = random_location(rng, n, edges)
        curr = prev.copy() if rng.random() < 0.3 else random_location(rng, n, edges)
        dest = np.zeros(n)
        nxt = np.zeros(n)
        if ph != 0:
            dest[rng.integers(n)] = 1.0
        if not np.allclose(prev, curr):
            a = int(np.argmax(curr))
            nxt[nbr[a][rng.integers(len(nbr[a]))] if nbr[a] else a] = 1.0
        return ph, prev, curr, dest, nxt

    rows = []
    for _ in range(B):
        ph, prev, curr, dest, nxt = agent(rng)
        deliv = np.zeros(n)
        events = np.zeros(2)
        if ph != 0:
            deliv[rng.integers(n) if ph == 1 else int(np.argmax(dest))] = 1.0
            if ph == 2 and rng.random() < 0.1:
                events[0] = 1.0
        elif rng.random() < 0.1:
            events[1] = 1.0
        since = 0.0 if ph == 0 else float(rng.integers(0, 200))
        age = 0.0 if ph == 0 else since + float(rng.integers(0, 100))
        ideal = 0.0 if ph == 0 else float(rng.uniform(0, 2 * diam))
        opos, ophase, oact, odest, onext, oover, owait = [], [], [], [], [], [], []
        for _k in range(K):
            oph, p, c, d, x = agent(rng)
            opos += [p, c]
            ophase.append(np.eye(3)[oph])
            oact.append(1.0)
            odest.append(d)
            onext.append(x)
            oover.append(float(rng.random() < 0.2))
            owait.append(0.0 if oph == 0 else float(rng.integers(0, 50)))
        cat = lambda xs: np.concatenate(xs) if xs else np.zeros(0)
        delivered = float(rng.integers(0, 150))
        dropped = float(rng.integers(0, 50))
        row = np.concatenate([
            prev, curr, dest, nxt, np.eye(3)[ph], deliv, events,
            [since, age, 1.0, ideal],
            cat(opos), cat(ophase), np.asarray(oact), cat(odest), cat(onext),
            np.asarray(oover), np.asarray(owait),
            [float(rng.integers(0, CAPACITY + 1)), float(rng.integers(0, A + 1)), cd,
             float(rng.random() < 0.02), float(rng.random() < 0.02),
             0.0 if ph == 0 else float(rng.integers(0, 50)), float(rng.random() < 0.2),
             float(rng.integers(0, 3)), delivered, dropped],
            coords.reshape(-1),
            [diam, n, A, T, CAPACITY, float(rng.integers(0, T)), float(rng.uniform(0.01, 0.2))]])
        assert len(row) == 7 * n + 4 * n * K + 6 * K + 26
        rows.append(row)
    return np.stack(rows)


def load_func(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    data = json.loads(text)
    ns = {}
    exec(data["Functions"], ns)
    return ns["evaluation_func"], data


def check(func, B=256, seed=0, quiet=False, layout="v1"):
    """全ケースで動かし、(誤りの文, ケースごとの因子の範囲) を返す."""
    rng = np.random.default_rng(seed)
    stats = {}
    for map_name, A, T in CASES:
        coords, apsp, diam, edges, cd, adj = load_map(map_name)
        if layout == "v2":
            obs = make_obs_v2(rng, B, coords, apsp, diam, edges, cd, A, T)
        else:
            obs = make_obs(rng, B, coords, apsp, diam, edges, cd, A, T)
        try:
            out = func(obs, apsp, adj) if layout == "v2" else func(obs, apsp)
        except Exception as e:                       # 例外の内容をそのまま LLM に返す
            return "%s on %s with %d agents: %s: %s" % (
                "Exception", map_name, A, type(e).__name__, e), stats
        if not isinstance(out, (list, tuple)):
            return "The function must return a list of factor arrays, got %s." % type(out).__name__, stats
        if len(out) != N_FACTORS:
            return "The function must return exactly %d factors, got %d." % (N_FACTORS, len(out)), stats
        rng_case = []
        for i, f in enumerate(out):
            f = np.asarray(f)
            if f.shape != (B, 1):
                return ("The shape of factor %d should be (batch_size, 1) = (%d, 1), got %s "
                        "(map %s, %d agents)." % (i, B, f.shape, map_name, A)), stats
            if not np.all(np.isfinite(f)):
                return "Factor %d contains NaN or inf (map %s, %d agents)." % (i, map_name, A), stats
            rng_case.append((float(f.min()), float(f.mean()), float(f.max())))
        stats["%s/%dag" % (map_name, A)] = rng_case
    return None, stats


def main():
    ap = argparse.ArgumentParser(description="Check an LLM-generated LaRe evaluation function.")
    ap.add_argument("response", help="JSON file of the LLM response (with a 'Functions' field)")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layout", choices=["v1", "v2"], default="v1",
                    help="observation layout of the prompt version (v2 also passes adj)")
    args = ap.parse_args()

    func, data = load_func(args.response)
    err, stats = check(func, args.batch, args.seed, layout=args.layout)
    if stats:
        cases = list(stats)
        print("factor ranges (min / mean / max) per case:")
        print("  #  " + "  ".join("%-24s" % c for c in cases))
        for i in range(N_FACTORS):
            cells = []
            for c in cases:
                if i < len(stats[c]):
                    lo, me, hi = stats[c][i]
                    cells.append("%7.3f %7.3f %7.3f   " % (lo, me, hi))
            print("  %-2d " % i + "".join(cells))
    if err:
        print("\nFAILED")
        print("Message to send back to the LLM:")
        print("There is an error in your previous answer. Error: " + err)
        return 1
    print("\nPASSED: %d factors, shape (batch, 1), finite on %d cases" % (N_FACTORS, len(stats)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
