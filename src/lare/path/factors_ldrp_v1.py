"""LaRe-Path 評価関数 v2 (LDRP 用, LLM が生成).

LLM (Claude Opus 5.5, 思考の強さ: 高) がプロンプト v1 から生成した 3 候補を、同じ LLM が
1 つにまとめたもの。人の手は加えていない (Functions の文字列をそのまま貼った)。
  プロンプト:   design/prompts/lare_ldrp_prompt_v1.txt
  元の応答:     design/prompts/responses/v1/summarized.json
  生成の記録:   design/prompts/responses/v1/README.md
観測の並びは obs_ldrp.build_ldrp_obs_for_agent (19 区間) を前提にしている。
返す因子 (10 個): deliver, pickup, progress, route, override, stall, delay, proximity, blocker, collision
"""

import numpy as np

def evaluation_func(observation, apsp):
    obs = np.asarray(observation, dtype=np.float64)
    if obs.ndim == 1:
        obs = obs[None, :]
    B = obs.shape[0]
    eps = 1e-8
    SPEED = 5.0
    meta = obs[0, -6:]
    N = max(int(round(meta[1])), 1)
    A = max(int(round(meta[2])), 1)
    K = max(A - 1, 0)
    D = np.asarray(apsp, dtype=np.float64).reshape(N, N)
    finite = np.isfinite(D)
    diam = float(meta[0])
    if (not np.isfinite(diam)) or diam <= eps:
        diam = float(np.max(D[finite])) if np.any(finite) else 1.0
    diam = max(diam, eps)
    D = np.where(finite, D, 2.0 * diam)
    rows = np.arange(B)

    o0 = 4 * N + 8
    P = o0 + 2 * N * K + 4 * K
    own_prev = obs[:, 0:N]
    own_curr = obs[:, N:2 * N]
    dest = obs[:, 2 * N:3 * N]
    phase = obs[:, 3 * N:3 * N + 3]
    delivery = obs[:, 3 * N + 3:4 * N + 3]
    picked = obs[:, 4 * N + 3] > 0.5
    delivered = obs[:, 4 * N + 4] > 0.5
    steps_since_assign = np.maximum(obs[:, 4 * N + 5], 0.0)
    active = obs[:, 4 * N + 7] > 0.5
    coll_dist = np.maximum(obs[:, P + 2], 0.0)
    coll_self = obs[:, P + 3] > 0.5
    wait_count = np.maximum(obs[:, P + 5], 0.0)
    override = obs[:, P + 6] > 0.5
    xy = obs[0, P + 7:P + 7 + 2 * N].reshape(N, 2)

    def locate(loc):
        valid = loc.sum(axis=1) > 0.5
        if N >= 2:
            order = np.argsort(-loc, axis=1)
            i1 = order[:, 0]
            i2 = order[:, 1]
        else:
            i1 = np.zeros(B, dtype=int)
            i2 = i1.copy()
        w1 = loc[rows, i1]
        w2 = loc[rows, i2]
        on_edge = (w2 > 1e-6) & (i1 != i2)
        tot = np.maximum(w1 + np.where(on_edge, w2, 0.0), eps)
        w1n = w1 / tot
        w2n = np.where(on_edge, w2 / tot, 0.0)
        L = np.linalg.norm(xy[i1] - xy[i2], axis=1)
        return i1, i2, w1n, w2n, on_edge, L, valid

    def dist_to(info, tgt):
        i1, i2, w1, w2, on_edge, L, valid = info
        d1 = w2 * L + D[i1, tgt]
        d2 = w1 * L + D[i2, tgt]
        use2 = on_edge & (d2 < d1)
        d = np.where(use2, d2, d1)
        nxt = np.where(use2, i2, i1)
        d_nxt = np.where(use2, w1 * L, w2 * L)
        return np.where(valid, d, 0.0), nxt, np.where(valid, d_nxt, 0.0)

    to_pick = phase[:, 1] > 0.5
    carry = phase[:, 2] > 0.5
    has_dest = dest.sum(axis=1) > 0.5
    has_task = active & (to_pick | carry) & has_dest
    dest_idx = np.argmax(dest, axis=1)
    has_deliv = delivery.sum(axis=1) > 0.5
    deliv_idx = np.argmax(delivery, axis=1)
    event = picked | delivered

    prev_info = locate(own_prev)
    curr_info = locate(own_curr)
    curr_valid = curr_info[6]

    # 1, 2) task events
    f_deliver = delivered.astype(np.float64)
    f_pickup = picked.astype(np.float64)

    # 3) progress toward the target of this move, relative to the max feasible move, [-1, 1]
    tgt = np.where(event, curr_info[0], dest_idx)
    d_prev, _, _ = dist_to(prev_info, tgt)
    d_now, _, _ = dist_to(curr_info, tgt)
    denom = np.maximum(np.minimum(SPEED, d_prev), eps)
    ok = active & (has_task | event) & prev_info[6] & curr_valid & (d_prev > 1e-6)
    f_progress = np.where(ok, np.clip((d_prev - d_now) / denom, -1.0, 1.0), 0.0)

    # 4) remaining route of the current task, [-1, 0]
    d_dest, nxt, d_to_nxt = dist_to(curr_info, dest_idx)
    extra = np.where(to_pick & has_deliv, D[dest_idx, deliv_idx], 0.0)
    f_route = np.where(has_task & curr_valid, -np.clip((d_dest + extra) / (2.0 * diam), 0.0, 1.0), 0.0)

    # 5) move cancelled by the safety layer, {-1, 0}
    f_override = np.where(active & override, -1.0, 0.0)

    # 6) stalling while holding a task (lost travel distance vs. diameter), [-1, 0]
    f_stall = np.where(has_task, -np.tanh(wait_count * SPEED / diam), 0.0)

    # 7) service delay since assignment, -0.5 at the worst-case ideal trip time, [-1, 0]
    s = steps_since_assign * SPEED
    f_delay = np.where(has_task, -s / (s + 2.0 * diam), 0.0)

    # 8) proximity to the nearest other agent, 9) parked agent on the route ahead
    f_prox = np.zeros(B)
    f_blocker = np.zeros(B)
    if K > 0 and P <= obs.shape[1]:
        oc = obs[:, o0:o0 + 2 * N * K].reshape(B, K, 2, N)[:, :, 1, :]
        oph = obs[:, o0 + 2 * N * K:o0 + 2 * N * K + 3 * K].reshape(B, K, 3)
        oact = obs[:, o0 + 2 * N * K + 3 * K:P]
        osum = oc.sum(axis=2)
        present = osum > 0.5
        opos = np.einsum('bkn,nc->bkc', oc, xy) / np.maximum(osum, eps)[:, :, None]
        msum = np.maximum(own_curr.sum(axis=1), eps)
        mypos = (own_curr @ xy) / msum[:, None]
        dd = np.linalg.norm(opos - mypos[:, None, :], axis=2)
        dd = np.where(present, dd, 1e12)
        dmin = dd.min(axis=1)
        margin = np.clip((dmin - coll_dist) / (2.0 * SPEED), 0.0, 1.0)
        f_prox = np.where(active & curr_valid & (dmin < 1e11), -(1.0 - margin), 0.0)

        parked = present & ((oph[:, :, 0] > 0.5) | (oact < 0.5))
        o_node = np.argmax(oc, axis=2)
        a_no = D[nxt[:, None], o_node]
        a_od = D[o_node, dest_idx[:, None]]
        a_nd = D[nxt, dest_idx][:, None]
        tol = 1e-6 + 1e-3 * diam
        on_path = (a_no + a_od) <= (a_nd + tol)
        lookahead = np.maximum(3.0 * SPEED + 2.0 * coll_dist, eps)[:, None]
        close = np.clip(1.0 - (d_to_nxt[:, None] + a_no) / lookahead, 0.0, 1.0)
        blk = np.where(parked & on_path, close, 0.0).max(axis=1)
        f_blocker = np.where(has_task & curr_valid, -blk, 0.0)

    # 10) own collision, {-1, 0}
    f_collision = np.where(coll_self, -1.0, 0.0)

    factors = [f_deliver, f_pickup, f_progress, f_route, f_override,
               f_stall, f_delay, f_prox, f_blocker, f_collision]
    return [np.nan_to_num(np.asarray(f, dtype=np.float64), nan=0.0, posinf=1.0, neginf=-1.0).reshape(B, 1) for f in factors]
