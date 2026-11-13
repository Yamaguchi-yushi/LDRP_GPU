"""LaRe-Path 用の観測 (LDRP 版 v2, 版 3 で使う) を機体ごとに作る.

並びは design/prompts/lare_ldrp_prompt_v2.txt の OBSERVATION FORM と同じ。
評価関数 (factors_ldrp_v2.evaluation_func) はこの並びを前提に書かれている。
並びを変えるときはプロンプトも変え、評価関数を生成し直すこと。
長さは 7N + 4NK + 6K + 26 (K = A - 1)。

    自分:   [own_prev N][own_curr N][dest N][next_node N][phase 3][delivery N][events 2]
            [steps_since_assign 1][task_age 1][active 1][ideal_task_length 1]
    他の機体: [others_pos 2NK][others_phase 3K][others_active K][others_dest NK]
            [others_next NK][others_override K][others_wait K]
    全体:   [demand 2][collision_distance 1][collision_info 2][wait_count 1][safety_override 1]
            [dropped_this_step 1][delivered_total 1][dropped_total 1]
            [node_coords 2N][meta 7 = diameter, N, A, T, C, t, arrival_rate]

版 2 (obs_ldrp.py) の記録に加えて、env 側で持つ値 (drp_env.py で記録する):
    _lare_next[i]         = このステップで向かったノード (SafeEnv の後)。止まったら None
    _lare_ideal_len[i]    = 今のタスクを割り当てた時点の位置 → ピックアップ → 配達先 の最短の道のり
    _lare_dropped_step    = このステップで捨てたタスクの数
"""

import numpy as np

from .encoder import get_node_coordinates_flat
from .obs_ldrp import _phase_of, _is_active


def _arrival_rate(env):
    """1 ステップあたりの期待到着数 (env の info["arrival_density"] と同じ定義)."""
    arrival = getattr(env, "task_arrival", "fixed")
    if arrival == "mmpp":
        return float((env.task_p_high + env.task_p_low) / 2.0)
    if arrival == "bernoulli":
        return float(env.task_density)
    return 1.0


def build_ldrp_v2_obs_for_agent(env, agent_id, graph_diameter,
                                prev_onehot_position, current_colliding_pairs):
    n = env.n_nodes
    A = env.agent_num
    eye3 = np.eye(3)
    override = getattr(env, "_lare_override", None)

    def curr_onehot(i):
        oh = np.asarray(env.obs_onehot[i]).flatten()
        return oh[:n].copy() if oh.size >= n else np.zeros(n)

    def prev_onehot(i):
        if prev_onehot_position is not None:
            return np.asarray(prev_onehot_position[i], dtype=float).copy()
        return curr_onehot(i)

    def node_onehot(node):
        v = np.zeros(n)
        if node is not None and 0 <= int(node) < n:
            v[int(node)] = 1.0
        return v

    def dest_of(i):
        return node_onehot(env.goal_array[i]) if env.assigned_tasks[i] else np.zeros(n)

    def over_of(i):
        return float(override[i]) if override is not None else 0.0

    i = agent_id
    task = env.assigned_tasks[i]
    phase = _phase_of(env, i)
    step = float(env.step_account)

    delivery = node_onehot(task[1]) if task else np.zeros(n)
    events = np.asarray(env._lare_events[i], dtype=float)
    if task:
        since = max(0.0, step - float(env._lare_assign_step[i]))
        age = max(0.0, step - float(env._lare_task_created[i]))
        ideal = float(env._lare_ideal_len[i])
    else:
        since, age, ideal = 0.0, 0.0, 0.0
    active = 1.0 if _is_active(env, i) else 0.0

    others = [j for j in range(A) if j != i]
    others_pos = []
    for j in others:
        others_pos += [prev_onehot(j), curr_onehot(j)]
    others_phase = [eye3[_phase_of(env, j)] for j in others]
    others_active = [1.0 if _is_active(env, j) else 0.0 for j in others]
    others_dest = [dest_of(j) for j in others]
    others_next = [node_onehot(env._lare_next[j]) for j in others]
    others_over = [over_of(j) for j in others]
    others_wait = [float(env.wait_count[j]) for j in others]

    def cat(parts):
        return np.concatenate(parts) if parts else np.zeros(0)

    waiting = float(sum(1 for v in env.assigned_list if v == -1))
    idle = float(sum(1 for j in range(A) if _is_active(env, j) and not env.assigned_tasks[j]))

    self_coll, others_coll = 0.0, 0.0
    for pair in current_colliding_pairs or []:
        if i in pair:
            self_coll = 1.0
        else:
            others_coll = 1.0

    meta = np.array([float(graph_diameter), float(n), float(A), float(env.time_limit),
                     float(env.task_num), step, _arrival_rate(env)])

    return np.concatenate([
        prev_onehot(i), curr_onehot(i), dest_of(i), node_onehot(env._lare_next[i]),
        eye3[phase], delivery, events, [since, age, active, ideal],
        cat(others_pos), cat(others_phase), np.asarray(others_active, dtype=float),
        cat(others_dest), cat(others_next),
        np.asarray(others_over, dtype=float), np.asarray(others_wait, dtype=float),
        [waiting, idle, float(env.colli_distan_value), self_coll, others_coll,
         float(env.wait_count[i]), over_of(i),
         float(env._lare_dropped_step), float(env.task_completion), float(env.task_dropped_count)],
        get_node_coordinates_flat(env), meta,
    ]).astype(np.float64)
