"""LaRe-Path 用の観測 (LDRP 版, 19 区間) を機体ごとに作る.

並びは design/prompts/lare_ldrp_prompt_v1.txt の OBSERVATION FORM と同じ。
評価関数 v2 (factors_ldrp_v1.evaluation_func) はこの並びを前提に書かれている。
並びを変えるときはプロンプトも変え、評価関数を生成し直すこと。

    [own_prev N][own_curr N][dest N][phase 3][delivery N][events 2]
    [steps_since_assign 1][task_age 1][active 1]
    [others_pos (A-1)*2N][others_phase (A-1)*3][others_active (A-1)]
    [demand 2][collision_distance 1][collision_info 2][wait_count 1][safety_override 1]
    [node_coords 2N][meta 6 = diameter, N, A, T, C, t]

env 側で持つ値 (drp_env.py で記録する):
    _lare_events[i]       = [このステップにピックアップした, 配達した]
    _lare_assign_step[i]  = 今のタスクを割り当てられたステップ
    _lare_task_created[i] = 今のタスクが発生したステップ
    _lare_override[i]     = このステップ SafeEnv に動きを止められたか (safe_marl.py が書く)
"""

import numpy as np

from .encoder import get_node_coordinates_flat


def _phase_of(env, i):
    """0 待機中 / 1 ピックアップへ / 2 配達へ."""
    task = env.assigned_tasks[i]
    if not task:
        return 0
    return 2 if env.goal_array[i] == task[1] else 1


def _is_active(env, i):
    if getattr(env, "use_dynamic_agents", False):
        return bool(env.active[i])
    return True


def build_ldrp_obs_for_agent(env, agent_id, graph_diameter,
                             prev_onehot_position, current_colliding_pairs):
    n = env.n_nodes
    A = env.agent_num
    eye3 = np.eye(3)

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

    i = agent_id
    task = env.assigned_tasks[i]
    phase = _phase_of(env, i)
    step = float(env.step_account)

    dest = node_onehot(env.goal_array[i]) if task else np.zeros(n)
    delivery = node_onehot(task[1]) if task else np.zeros(n)
    events = np.asarray(env._lare_events[i], dtype=float)
    if task:
        since = max(0.0, step - float(env._lare_assign_step[i]))
        age = max(0.0, step - float(env._lare_task_created[i]))
    else:
        since, age = 0.0, 0.0
    active = 1.0 if _is_active(env, i) else 0.0

    others_pos, others_phase, others_active = [], [], []
    for j in range(A):
        if j == i:
            continue
        others_pos += [prev_onehot(j), curr_onehot(j)]
        others_phase.append(eye3[_phase_of(env, j)])
        others_active.append(1.0 if _is_active(env, j) else 0.0)

    waiting = float(sum(1 for v in env.assigned_list if v == -1))
    idle = float(sum(1 for j in range(A) if _is_active(env, j) and not env.assigned_tasks[j]))

    self_coll, others_coll = 0.0, 0.0
    for pair in current_colliding_pairs or []:
        if i in pair:
            self_coll = 1.0
        else:
            others_coll = 1.0

    override = getattr(env, "_lare_override", None)
    over = float(override[i]) if override is not None else 0.0

    meta = np.array([float(graph_diameter), float(n), float(A), float(env.time_limit),
                     float(env.task_num), step])

    return np.concatenate([
        prev_onehot(i), curr_onehot(i), dest, eye3[phase], delivery, events,
        [since, age, active],
        np.concatenate(others_pos) if others_pos else np.zeros(0),
        np.concatenate(others_phase) if others_phase else np.zeros(0),
        np.asarray(others_active, dtype=float),
        [waiting, idle, float(env.colli_distan_value), self_coll, others_coll,
         float(env.wait_count[i]), over],
        get_node_coordinates_flat(env), meta,
    ]).astype(np.float64)
