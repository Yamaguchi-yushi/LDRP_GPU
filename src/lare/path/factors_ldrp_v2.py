"""LaRe-Path 評価関数 v2 (LDRP 用, LLM が生成). 版 3 で使う.

LLM (Claude Opus 5.5, 思考の強さ: 高) がプロンプト v2 から生成した 3 候補を、同じ LLM が
1 つにまとめたもの。人の手は加えていない (Functions の文字列をそのまま貼った)。
  プロンプト:   design/prompts/lare_ldrp_prompt_v2.txt
  元の応答:     design/prompts/responses/v2/summarized.json
  生成の記録:   design/prompts/responses/v2/README.md
観測の並びは obs_ldrp_v2.build_ldrp_v2_obs_for_agent (長さ 7N + 4NK + 6K + 26) を前提にしている。
関数の形は evaluation_func(observation, apsp, adj)。
返す因子 (10 個): progress, remaining, event, delay, stall, safety, proximity, conflict, block, backlog
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
    C = np.maximum(obs[:, D - 3], 1.0)

    apsp = np.asarray(apsp, dtype=np.float64).reshape(N, N)
    adj = np.asarray(adj, dtype=np.float64).reshape(N, N)
    fin = np.isfinite(apsp)
    big = (np.max(apsp[fin]) if np.any(fin) else 1.0) * 2.0 + 1.0
    apsp = np.where(fin, apsp, big)
    adj = np.where(np.isfinite(adj), adj, 0.0)

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

    # ---------- global part ----------
    waiting = np.maximum(obs[:, P], 0.0)
    coll_dist = np.maximum(obs[:, P + 2], 0.0)
    own_coll = obs[:, P + 3]
    wait_cnt = np.maximum(obs[:, P + 5], 0.0)
    override = obs[:, P + 6]
    dropped_now = np.maximum(obs[:, P + 7], 0.0)
    coords = obs[:, P + 10:P + 10 + 2 * N].reshape(B, N, 2)

    def locate(loc):
        s = loc.sum(axis=1, keepdims=True)
        locn = loc / np.maximum(s, eps)
        order = np.argsort(-locn, axis=1)
        a = order[:, 0]
        if N > 1:
            b = order[:, 1]
            wb = locn[rows, b]
        else:
            b = order[:, 0]
            wb = np.zeros(B)
        wa = locn[rows, a]
        L = adj[a, b]
        L = np.where(L > 0, L, apsp[a, b])
        on_edge = (wb > 1e-3) & (a != b)
        return a, b, wa, wb, L, on_edge

    def loc_dist(loc, tgt):
        a, b, wa, wb, L, on_edge = locate(loc)
        da = (1.0 - wa) * L + apsp[a, tgt]
        db = (1.0 - wb) * L + apsp[b, tgt]
        return np.where(on_edge, np.minimum(da, db), apsp[a, tgt])

    has_dest = dest.sum(axis=1) > 0.5
    has_deliv = deliv.sum(axis=1) > 0.5
    busy = active & has_dest & ((phase[:, 1] + phase[:, 2]) > 0.5)
    idle = active & ~busy
    has_event = ev_pick | ev_deliv
    dest_node = np.argmax(dest, axis=1)
    deliv_node = np.argmax(deliv, axis=1)
    cur_a, cur_b, cur_wa, cur_wb, cur_L, cur_on_edge = locate(own_curr)

    # F1: progress toward the current target this step (fraction of max speed, higher is better)
    target = np.where(has_event, cur_a, dest_node)
    d_prev = loc_dist(own_prev, target)
    d_curr = loc_dist(own_curr, target)
    prog_valid = active & (busy | has_event)
    f_progress = np.where(prog_valid, np.clip((d_prev - d_curr) / SPEED, -1.0, 1.0), 0.0)

    # F2: remaining task distance / (2 * diameter) (lower is better)
    d_dest = loc_dist(own_curr, dest_node)
    heading_pick = (phase[:, 1] > 0.5) & has_deliv
    extra = np.where(heading_pick, apsp[dest_node, deliv_node], 0.0)
    remaining = d_dest + extra
    f_remaining = np.where(busy, np.clip(remaining / (2.0 * diam), 0.0, 1.0), 0.0)

    # F3: task events, delivery 1.0 and pickup 0.5 (higher is better)
    f_event = np.where(active, np.clip(0.5 * ev_pick.astype(np.float64) + 1.0 * ev_deliv.astype(np.float64), 0.0, 1.0), 0.0)

    # F4: estimated task-time overrun relative to the ideal time (lower is better)
    ideal_steps = ideal_len / SPEED
    est_total = ssa + remaining / SPEED
    overrun = np.maximum(est_total - ideal_steps, 0.0)
    f_delay = np.where(busy, np.tanh(overrun / np.maximum(ideal_steps, 1.0)), 0.0)

    # F5: stalling while holding a task (lower is better)
    f_stall = np.where(busy, np.tanh(wait_cnt / 5.0), 0.0)

    # F6: own safety override or collision (lower is better)
    f_safety = np.where(active, np.clip(np.maximum(override, own_coll), 0.0, 1.0), 0.0)

    # ---------- shortest-path next-hop set from the current location ----------
    tol = 1e-4 * diam + 1e-6
    adj_row = adj[cur_a]
    to_d = apsp[:, dest_node].T
    base = apsp[cur_a, dest_node][:, None]
    sp_mask = (adj_row > 0) & (np.abs(adj_row + to_d - base) < tol[:, None])
    da_c = (1.0 - cur_wa) * cur_L + apsp[cur_a, dest_node]
    db_c = (1.0 - cur_wb) * cur_L + apsp[cur_b, dest_node]
    edge_hop = np.where(da_c <= db_c, cur_a, cur_b)
    edge_mask = np.zeros((B, N), dtype=bool)
    edge_mask[rows, edge_hop] = True
    hop_mask = np.where(cur_on_edge[:, None], edge_mask, sp_mask)
    hop_mask = hop_mask & (busy & (d_dest > tol))[:, None]

    f_proximity = np.zeros(B)
    f_conflict = np.zeros(B)
    f_block = np.zeros(B)
    if K > 0:
        kk = np.arange(K)
        opos = obs[:, o0:o0 + 2 * N * K].reshape(B, K, 2, N)
        o_curr = opos[:, :, 1, :]
        o_phase = obs[:, o0 + 2 * N * K:o0 + 2 * N * K + 3 * K].reshape(B, K, 3)
        o_act = obs[:, o0 + 2 * N * K + 3 * K:o0 + 2 * N * K + 4 * K] > 0.5
        o_dest = obs[:, o0 + 2 * N * K + 4 * K:o0 + 3 * N * K + 4 * K].reshape(B, K, N)
        o_next = obs[:, o0 + 3 * N * K + 4 * K:o0 + 4 * N * K + 4 * K].reshape(B, K, N)
        o_override = obs[:, o0 + 4 * N * K + 4 * K:o0 + 4 * N * K + 5 * K]
        o_wait = obs[:, o0 + 4 * N * K + 5 * K:o0 + 4 * N * K + 6 * K]
        o_busy = o_act & ((o_phase[:, :, 1] + o_phase[:, :, 2]) > 0.5) & (o_dest.sum(axis=2) > 0.5)

        # F7: proximity to the nearest active agent (lower is better)
        own_w = own_curr / np.maximum(own_curr.sum(axis=1, keepdims=True), eps)
        my_pos = np.einsum('bn,bnc->bc', own_w, coords)
        o_w = o_curr / np.maximum(o_curr.sum(axis=2, keepdims=True), eps)
        o_pos = np.einsum('bkn,bnc->bkc', o_w, coords)
        dist = np.sqrt(np.sum((o_pos - my_pos[:, None, :]) ** 2, axis=2))
        close = np.clip(1.0 - (dist - coll_dist[:, None]) / (2.0 * SPEED), 0.0, 1.0)
        close = np.where(o_act, close, 0.0)
        f_proximity = np.where(active, close.max(axis=1), 0.0)

        # F8: conflicts on the own next hop (lower is better)
        o_nv = o_next.sum(axis=2) > 0.5
        o_nn = np.argmax(o_next, axis=2)
        o_a = np.argmax(o_curr, axis=2)
        o_wa = o_w[rows[:, None], kk[None, :], o_a]
        hop_at_nn = hop_mask[rows[:, None], o_nn]
        hop_at_a = hop_mask[rows[:, None], o_a]
        own_support = own_curr > 1e-6
        nn_in_mine = own_support[rows[:, None], o_nn]
        same_tgt = o_nv & hop_at_nn
        head_on = o_nv & nn_in_mine & hop_at_a
        standing = (~o_nv) & (o_wa > 0.5) & hop_at_a
        n_conf = (o_act & (same_tgt | head_on | standing)).sum(axis=1).astype(np.float64)
        occ = np.clip((o_curr * o_act[:, :, None]).sum(axis=1), 0.0, 1.0)
        hasm = hop_mask.any(axis=1)
        occ_m = np.where(hop_mask, occ, np.inf)
        all_occ = np.where(hasm, np.min(np.where(hasm[:, None], occ_m, 0.0), axis=1), 0.0)
        conflict = np.maximum(1.0 - 0.5 ** n_conf, np.clip(all_occ, 0.0, 1.0))
        f_conflict = np.where(busy, conflict, 0.0)

        # F9: blocking others: nearby stuck/cancelled agents, idle agent in busy agents' way (lower is better)
        near = o_act & (dist < (coll_dist + 2.0 * SPEED)[:, None])
        o_stuck = (o_override > 0.5) | (o_busy & (o_wait > 0.5))
        n_near_blocked = (near & o_stuck).sum(axis=1).astype(np.float64)
        o_dn = np.argmax(o_dest, axis=2)
        toward_me = (o_nv & (o_nn == cur_a[:, None])) | (o_dn == cur_a[:, None])
        idle_hits = (idle[:, None] & o_busy & toward_me).sum(axis=1).astype(np.float64)
        prox_busy = np.where(o_busy, close, 0.0).max(axis=1)
        blk = 1.0 - 0.5 ** (n_near_blocked + idle_hits)
        blk = np.where(idle, np.maximum(blk, prox_busy), blk)
        f_block = np.where(active, np.clip(blk, 0.0, 1.0), 0.0)

    # F10: system backlog pressure (lower is better)
    f_backlog = np.clip(0.7 * np.clip(waiting / C, 0.0, 1.0) + 0.3 * np.minimum(dropped_now, 1.0), 0.0, 1.0)

    factors = [f_progress, f_remaining, f_event, f_delay, f_stall,
               f_safety, f_proximity, f_conflict, f_block, f_backlog]
    return [np.nan_to_num(np.asarray(f, dtype=np.float64), nan=0.0, posinf=1.0, neginf=-1.0).reshape(B, 1) for f in factors]
