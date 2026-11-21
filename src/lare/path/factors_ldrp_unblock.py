"""LaRe-Path 評価関数 unblock (LDRP 用, 手書き). 版 4 で使う.

Claude が版 1〜3 の分析 (時刻ごとの完了数・デコーダの probe) を踏まえて手で書いたもの。
観測は版 3 と同じ obs_ldrp_v2.build_ldrp_v2_obs_for_agent (長さ 7N + 4NK + 6K + 26) を使う。
関数の形も版 3 と同じ evaluation_func(observation, apsp, adj)。

分析から取り入れた方針:
  1. 全体の値 (backlog 等) を入れない — 機体を区別できず、信用割当に使えない
  2. 時間・距離の尺度を地図に合わせる — 版 3 の stall = tanh(wait/5) は aoba00 (辺が最長 40 ステップ) で飽和した
  3. 待機を「無駄な待機」と「塞がれた待機」に分ける — 版 3 は待機を一律に罰し、遠回りが増えた
  4. 近さそのものは罰しない — 版 1 の近接ペナルティは 8x5 で譲り合いの膠着 (deadlock) を招いた。
     代わりに「次の一歩の競合」と「にらみ合い (互いに相手の次の一歩を塞ぐ)」を見る
  5. 空いている機体にも意味のある因子を持たせる — 版 2/3 は空き機体の因子がほぼ全部 0 で、
     デコーダが空き機体を高く評価していた。idle フラグと「他機の通り道を塞いでいるか」を入れる

返す因子 (10 個):
  progress, event, delay, wait_free, wait_long, standoff, safety, conflict, in_way, idle
"""

import numpy as np


def evaluation_func(observation, apsp, adj):
    obs = np.asarray(observation, dtype=np.float64)
    if obs.ndim == 1:
        obs = obs[None, :]
    B, D = obs.shape
    eps = 1e-8
    SPEED = 5.0
    rows = np.arange(B)

    # ---------- sizes from meta (last 7 values) ----------
    N = int(round(obs[0, D - 6]))
    A = int(round(obs[0, D - 5]))
    K = max(A - 1, 0)
    diam = np.maximum(obs[:, D - 7], eps)
    diam_steps = diam / SPEED                       # 地図の端から端までの所要ステップ (8x5 ~38, aoba00 ~124)

    apsp = np.asarray(apsp, dtype=np.float64).reshape(N, N)
    adj = np.asarray(adj, dtype=np.float64).reshape(N, N)
    fin = np.isfinite(apsp)
    big = (np.max(apsp[fin]) if np.any(fin) else 1.0) * 2.0 + 1.0
    apsp = np.where(fin, apsp, big)
    adj = np.where(np.isfinite(adj), adj, 0.0)
    tol = 1e-4 * diam + 1e-6

    o0 = 5 * N + 9
    P = o0 + 4 * N * K + 6 * K

    # ---------- own part ----------
    own_prev = obs[:, 0:N]
    own_curr = obs[:, N:2 * N]
    dest = obs[:, 2 * N:3 * N]
    phase = obs[:, 4 * N:4 * N + 3]
    deliv = obs[:, 4 * N + 3:5 * N + 3]
    ev_pick = obs[:, 5 * N + 3] > 0.5
    ev_deliv = obs[:, 5 * N + 4] > 0.5
    ssa = np.maximum(obs[:, 5 * N + 5], 0.0)
    active = obs[:, 5 * N + 7] > 0.5
    ideal_len = np.maximum(obs[:, 5 * N + 8], 0.0)

    own_coll = obs[:, P + 3]
    wait_cnt = np.maximum(obs[:, P + 5], 0.0)       # 連続待機ステップ数 (動くと 0 に戻る)
    override = obs[:, P + 6]

    def locate(loc):
        """位置 (one-hot or 辺上の 2 ノードの重み) -> (近い端 a, 遠い端 b, 重み, 辺長, 辺上か)."""
        s = loc.sum(axis=-1, keepdims=True)
        locn = loc / np.maximum(s, eps)
        order = np.argsort(-locn, axis=-1)
        a = order[..., 0]
        b = order[..., 1] if N > 1 else order[..., 0]
        wa = np.take_along_axis(locn, a[..., None], axis=-1)[..., 0]
        wb = np.take_along_axis(locn, b[..., None], axis=-1)[..., 0]
        L = adj[a, b]
        L = np.where(L > 0, L, apsp[a, b])
        on_edge = (wb > 1e-3) & (a != b)
        return a, b, wa, wb, L, on_edge

    def loc_dist(loc, tgt):
        a, b, wa, wb, L, on_edge = locate(loc)
        da = (1.0 - wa) * L + apsp[a, tgt]
        db = (1.0 - wb) * L + apsp[b, tgt]
        return np.where(on_edge, np.minimum(da, db), apsp[a, tgt])

    def next_hops(a, b, wa, wb, L, on_edge, tgt):
        """最短経路で次に向かうノードの集合 (..., N) の bool. 辺上なら目的地側の端 1 つ."""
        to_t = np.moveaxis(apsp[:, tgt], 0, -1)                        # (..., N): 各ノード -> tgt
        adj_row = adj[a]                                               # (..., N)
        base = apsp[a, tgt][..., None]
        t = tol.reshape((B,) + (1,) * (a.ndim - 1))[..., None]
        sp = (adj_row > 0) & (np.abs(adj_row + to_t - base) < t)
        da = (1.0 - wa) * L + apsp[a, tgt]
        db = (1.0 - wb) * L + apsp[b, tgt]
        hop = np.where(da <= db, a, b)
        edge = np.arange(N) == hop[..., None]
        return np.where(on_edge[..., None], edge, sp)

    has_dest = dest.sum(axis=1) > 0.5
    has_deliv = deliv.sum(axis=1) > 0.5
    busy = active & has_dest & ((phase[:, 1] + phase[:, 2]) > 0.5)
    idle = active & ~busy
    has_event = ev_pick | ev_deliv
    dest_node = np.argmax(dest, axis=1)
    deliv_node = np.argmax(deliv, axis=1)
    cur = locate(own_curr)
    cur_a, cur_on_edge = cur[0], cur[5]
    stalled = busy & ((wait_cnt > 0.5) | (override > 0.5))

    # F1 progress: このステップで目的地に近づいた量 / 1 ステップの最大移動量 (高いほど良い)
    target = np.where(has_event, cur_a, dest_node)
    d_prev = loc_dist(own_prev, target)
    d_curr = loc_dist(own_curr, target)
    f_progress = np.where(active & (busy | has_event),
                          np.clip((d_prev - d_curr) / SPEED, -1.0, 1.0), 0.0)

    # F2 event: 配達 1.0, ピックアップ 0.5 (高いほど良い)
    f_event = np.where(active, np.clip(0.5 * ev_pick + 1.0 * ev_deliv, 0.0, 1.0), 0.0)

    # F3 delay: 割当からの経過 + 残りの所要 が理想の所要をどれだけ超えたか (低いほど良い)
    #   分母を「理想の所要」と「地図の直径の 1/4」の大きい方にして、短いタスクで過敏にならないようにする
    d_dest = loc_dist(own_curr, dest_node)
    extra = np.where((phase[:, 1] > 0.5) & has_deliv, apsp[dest_node, deliv_node], 0.0)
    remaining_steps = (d_dest + extra) / SPEED
    ideal_steps = ideal_len / SPEED
    overrun = np.maximum(ssa + remaining_steps - ideal_steps, 0.0)
    f_delay = np.where(busy, np.tanh(overrun / np.maximum(ideal_steps, 0.25 * diam_steps)), 0.0)

    # ---------- own next hops and conflicts with others ----------
    my_hops = next_hops(*cur, dest_node) & (busy & (d_dest > tol))[:, None]   # (B, N)

    f_standoff = np.zeros(B)
    f_conflict = np.zeros(B)
    f_in_way = np.zeros(B)
    blocked = np.zeros(B, dtype=bool)
    if K > 0:
        opos = obs[:, o0:o0 + 2 * N * K].reshape(B, K, 2, N)
        o_curr = opos[:, :, 1, :]
        o_phase = obs[:, o0 + 2 * N * K:o0 + 2 * N * K + 3 * K].reshape(B, K, 3)
        o_act = obs[:, o0 + 2 * N * K + 3 * K:o0 + 2 * N * K + 4 * K] > 0.5
        o_dest = obs[:, o0 + 2 * N * K + 4 * K:o0 + 3 * N * K + 4 * K].reshape(B, K, N)
        o_next = obs[:, o0 + 3 * N * K + 4 * K:o0 + 4 * N * K + 4 * K].reshape(B, K, N)
        o_override = obs[:, o0 + 4 * N * K + 4 * K:o0 + 4 * N * K + 5 * K]
        o_wait = obs[:, o0 + 4 * N * K + 5 * K:o0 + 4 * N * K + 6 * K]
        o_busy = o_act & ((o_phase[:, :, 1] + o_phase[:, :, 2]) > 0.5) & (o_dest.sum(axis=2) > 0.5)
        o_stalled = o_busy & ((o_wait > 0.5) | (o_override > 0.5))

        oc = locate(o_curr)                                            # 各 (B, K)
        o_a, o_on_edge = oc[0], oc[5]
        o_dn = np.argmax(o_dest, axis=2)
        o_hops = next_hops(*oc, o_dn) & o_busy[:, :, None]              # (B, K, N)
        o_nv = o_next.sum(axis=2) > 0.5
        o_nn = np.argmax(o_next, axis=2)

        # 自分の次の一歩のノードに他機がいる (ノード上で止まっている) か、同じノードへ向かっている
        o_at_node = o_act & ~o_on_edge
        occupies_my_hop = o_at_node & my_hops[rows[:, None], o_a]
        same_target = o_act & o_nv & my_hops[rows[:, None], o_nn]
        n_conf = (occupies_my_hop | same_target).sum(axis=1).astype(np.float64)

        # F8 conflict: 次の一歩の競合 (低いほど良い). 近さそのものではなく「行きたい所が塞がっているか」
        f_conflict = np.where(busy, 1.0 - 0.5 ** n_conf, 0.0)
        blocked = busy & ((n_conf > 0) | (override > 0.5))

        # F6 standoff: にらみ合い. 自分も相手も止まっていて、相手が自分の次の一歩に居座り、
        #   自分が相手の次の一歩に居座っている (deadlock の芽. 低いほど良い)
        me_at_node = ~cur_on_edge
        i_block_k = me_at_node[:, None] & o_hops[rows[:, None], np.arange(K)[None, :], cur_a[:, None]]
        mutual = stalled[:, None] & o_stalled & occupies_my_hop & i_block_k
        f_standoff = 1.0 - 0.5 ** mutual.sum(axis=1).astype(np.float64)

        # F9 in_way: 自分がノード上で止まっていて、止まっている他機の次の一歩になっている (低いほど良い)
        #   空き機体にも意味のある因子. 「どいてあげる」ことを評価できる
        my_still = active & me_at_node & ((wait_cnt > 0.5) | idle)
        n_in_way = (my_still[:, None] & o_stalled & i_block_k).sum(axis=1).astype(np.float64)
        f_in_way = 1.0 - 0.5 ** n_in_way

    # F4 wait_free: 塞がれていないのに待機した (無駄な待機. 低いほど良い). 1 ステップごとの印
    f_wait_free = np.where(busy & (wait_cnt > 0.5) & ~blocked, 1.0, 0.0)

    # F5 wait_long: 連続待機の長さ (地図の尺度に合わせる. 低いほど良い)
    #   8x5 で ~4 ステップ、aoba00 で ~12 ステップ待つと 0.76 (tanh(1))
    f_wait_long = np.where(busy, np.tanh(wait_cnt / np.maximum(3.0, 0.1 * diam_steps)), 0.0)

    # F7 safety: 安全層による取り消し or 衝突 (低いほど良い)
    f_safety = np.where(active, np.clip(np.maximum(override, own_coll), 0.0, 1.0), 0.0)

    # F10 idle: タスクを持たない稼働中の機体. 他の因子が 0 になる状態をデコーダが「良い」と取り違えないための印
    f_idle = idle.astype(np.float64)

    factors = [f_progress, f_event, f_delay, f_wait_free, f_wait_long,
               f_standoff, f_safety, f_conflict, f_in_way, f_idle]
    return [np.nan_to_num(np.asarray(f, dtype=np.float64), nan=0.0, posinf=1.0, neginf=-1.0).reshape(B, 1)
            for f in factors]
