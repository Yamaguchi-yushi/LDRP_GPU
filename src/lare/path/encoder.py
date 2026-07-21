"""LaRe-Path encoder: 10-factor evaluation function for path planning.

Mirrors the 10-factor evaluation_func in Safe-TSL-DBCT
(MARL4DRP/drp_env/reward_model/LLMrd/fallback_functions/evaluation_func.py).

Factors (returned in this order):
  1. prog_goal           : decrease in shortest-path distance to the goal
  2. in_collision        : 1 if this agent is in a colliding pair, else 0
  3. others_in_collision : 1 if other agents are colliding (this agent not involved), else 0
  4. wait_norm           : agent's wait counter
  5. dist_goal_norm      : current distance to goal, normalized by graph diameter
  6. min_sep_norm        : minimum euclidean distance to any other agent, normalized
  7. avg_sep_norm        : average euclidean distance to other agents, normalized
  8. safety_margin       : min_sep / collision_distance (clipped)
  9. collision_risk      : 1 if min_sep < collision_distance * 2, else 0
 10. at_goal             : 1 if dist_goal < eps, else 0
"""

import heapq
import numpy as np


FACTOR_NUMBER = 10


def evaluation_func(observation, apsp, eps=1e-6):
    """Compute the 10 latent factors for a batch of LaRe-compatible observations.

    The observation layout (per row) matches `_get_lare_compatible_obs`:
        [own_prev (N), own_curr (N), goal (N),
         others_curr_raw ((A-1)*N*2),
         collision_distance (1), collision_info (2), wait_count (1),
         nodes_flat (2*N), edges_flat (3*E),
         graph_diameter (1), N (1), A (1), E (1)]
    """
    B = observation.shape[0]

    A = int(observation[0, -2])
    N = int(observation[0, -3])
    graph_diameter = float(observation[0, -4])

    size_others_curr = (A - 1) * N * 2
    size_nodes_flat = N * 2

    idx = 0
    own_prev = observation[:, idx : idx + N]; idx += N
    own_curr = observation[:, idx : idx + N]; idx += N
    goal = observation[:, idx : idx + N]; idx += N
    others_curr_raw = observation[:, idx : idx + size_others_curr]; idx += size_others_curr
    collision_distance = observation[:, idx : idx + 1]; idx += 1
    collision_info = observation[:, idx : idx + 2]; idx += 2
    wait_count = observation[:, idx : idx + 1]; idx += 1
    nodes_flat = observation[:, idx : idx + size_nodes_flat]; idx += size_nodes_flat

    nodes = nodes_flat.reshape(B, N, 2)

    def estimate_partial_distance(pos_vec, goal_node):
        nz = np.where(pos_vec > 1e-8)[0]
        if len(nz) == 1:
            return float(apsp[nz[0], goal_node])
        if len(nz) == 2:
            i, j = int(nz[0]), int(nz[1])
            wi, wj = pos_vec[i], pos_vec[j]
            s = wi + wj + eps
            alpha = wj / s
            Di = apsp[i, goal_node]
            Dj = apsp[j, goal_node]
            return (1 - alpha) * Di + alpha * Dj
        if len(nz) == 0:
            return graph_diameter
        pivot = int(np.argmax(pos_vec))
        return float(apsp[pivot, goal_node])

    agent_curr_pos = np.sum(own_curr[:, :, None] * nodes, axis=1)

    others_curr_pos_list = []
    for k in range(A - 1):
        start_idx = k * N * 2 + N
        end_idx = start_idx + N
        other_curr = others_curr_raw[:, start_idx:end_idx]
        others_curr_pos_list.append(np.sum(other_curr[:, :, None] * nodes, axis=1))

    if len(others_curr_pos_list) > 0:
        others_curr_pos = np.stack(others_curr_pos_list, axis=1)
    else:
        others_curr_pos = np.zeros((B, 1, 2))

    dist_goal = np.zeros((B, 1))
    dist_goal_prev = np.zeros((B, 1))
    for b in range(B):
        goal_node = int(np.argmax(goal[b]))
        dist_goal[b, 0] = estimate_partial_distance(own_curr[b], goal_node)
        dist_goal_prev[b, 0] = estimate_partial_distance(own_prev[b], goal_node)

    dist_goal = np.where(np.isinf(dist_goal), graph_diameter, dist_goal)
    dist_goal_prev = np.where(np.isinf(dist_goal_prev), graph_diameter, dist_goal_prev)

    prog_goal = dist_goal_prev - dist_goal
    at_goal = (dist_goal < eps).astype(float)
    dist_goal_norm = dist_goal / (graph_diameter + eps)

    if A > 1:
        sep = np.linalg.norm(agent_curr_pos[:, None, :] - others_curr_pos, axis=2)
        min_sep = np.min(sep, axis=1, keepdims=True)
        avg_sep = np.mean(sep, axis=1, keepdims=True)
    else:
        min_sep = np.full((B, 1), graph_diameter)
        avg_sep = np.full((B, 1), graph_diameter)

    min_sep_norm = min_sep / (graph_diameter + eps)
    avg_sep_norm = avg_sep / (graph_diameter + eps)
    safety_margin = np.clip(min_sep / (collision_distance + eps), 0, 100)
    collision_risk = (min_sep < collision_distance * 2).astype(float)

    wait_norm = wait_count.astype(float)
    in_collision = collision_info[:, 0:1]
    others_in_collision = collision_info[:, 1:2]

    prog_goal = np.nan_to_num(prog_goal, nan=0.0)
    dist_goal_norm = np.nan_to_num(dist_goal_norm, nan=1.0)
    min_sep_norm = np.nan_to_num(min_sep_norm, nan=0.0)
    avg_sep_norm = np.nan_to_num(avg_sep_norm, nan=0.0)
    safety_margin = np.nan_to_num(safety_margin, nan=0.0)

    # print("====================================================")
    # print(f"prog_goal: {prog_goal.flatten()}")
    # print(f"in_collision: {in_collision.flatten()}")
    # print(f"others_in_collision: {others_in_collision.flatten()}")
    # print(f"wait_norm: {wait_norm.flatten()}")
    # print(f"dist_goal_norm: {dist_goal_norm.flatten()}")
    # print(f"min_sep_norm: {min_sep_norm.flatten()}")
    # print(f"avg_sep_norm: {avg_sep_norm.flatten()}")
    # print(f"safety_margin: {safety_margin.flatten()}")
    # print(f"collision_risk: {collision_risk.flatten()}")
    # print(f"at_goal: {at_goal.flatten()}")

    return [
        prog_goal,
        in_collision,
        others_in_collision,
        wait_norm,
        dist_goal_norm,
        min_sep_norm,
        avg_sep_norm,
        safety_margin,
        collision_risk,
        at_goal,
    ]


def compute_graph_diameter(env):
    """Diameter as the longest shortest-path distance between any two nodes."""
    n_nodes = len(env.G.nodes())
    max_dist = 0.0
    has_finite = False

    for i in range(n_nodes):
        dist = {v: float("inf") for v in env.G.nodes()}
        dist[i] = 0.0
        pq = [(0.0, i)]
        visited = set()
        while pq:
            d, v = heapq.heappop(pq)
            if v in visited:
                continue
            visited.add(v)
            for nb in env.G.neighbors(v):
                p1, p2 = env.pos[v], env.pos[nb]
                w = float(np.linalg.norm(np.array(p1) - np.array(p2)))
                nd = d + w
                if nd < dist[nb]:
                    dist[nb] = nd
                    heapq.heappush(pq, (nd, nb))
        for v in env.G.nodes():
            if dist[v] != float("inf"):
                has_finite = True
                if dist[v] > max_dist:
                    max_dist = dist[v]

    return max(max_dist, 1.0) if has_finite else 100.0


def get_node_coordinates_flat(env):
    """Flat node coords: [x0, y0, x1, y1, ...] sorted by node id."""
    coords = []
    for node_id in sorted(env.pos.keys()):
        x, y = env.pos[node_id]
        coords.extend([float(x), float(y)])
    return np.array(coords, dtype=float)


def build_lare_obs_for_agent(env, agent_id, graph_diameter,
                             prev_onehot_position, current_colliding_pairs):
    """Build the LaRe-compatible observation row for a single agent.

    Layout matches Safe-TSL-DBCT's `_get_lare_compatible_obs`:
      [own_prev (N), own_curr (N), goal (N),
       others_curr_raw ((A-1)*N*2),
       collision_distance (1), collision_info (2), wait_count (1),
       nodes_flat (2*N), edges_flat (3*E),
       graph_diameter (1), N (1), A (1), E (1)]
    """
    n_nodes = env.n_nodes
    agent_num = env.agent_num

    def _agent_curr_onehot(i):
        oh = np.asarray(env.obs_onehot[i]).flatten()
        return oh[:n_nodes].copy() if oh.size >= n_nodes else np.zeros(n_nodes)

    def _agent_goal_onehot(i):
        oh = np.asarray(env.obs_onehot[i]).flatten()
        if oh.size >= 2 * n_nodes:
            return oh[n_nodes : 2 * n_nodes].copy()
        out = np.zeros(n_nodes)
        gn = int(env.goal_array[i])
        if 0 <= gn < n_nodes:
            out[gn] = 1.0
        return out

    own_prev = prev_onehot_position[agent_id].copy() if prev_onehot_position is not None else _agent_curr_onehot(agent_id)
    own_curr = _agent_curr_onehot(agent_id)
    goal = _agent_goal_onehot(agent_id)

    others_parts = []
    for j in range(agent_num):
        if j == agent_id:
            continue
        prev_j = prev_onehot_position[j].copy() if prev_onehot_position is not None else _agent_curr_onehot(j)
        curr_j = _agent_curr_onehot(j)
        others_parts.append(prev_j)
        others_parts.append(curr_j)
    if len(others_parts) > 0:
        others_curr_raw = np.concatenate(others_parts)
    else:
        others_curr_raw = np.array([], dtype=float)

    collision_distance = np.array([float(env.colli_distan_value)])

    self_involved = 0
    others_exist = 0
    if current_colliding_pairs:
        for pair in current_colliding_pairs:
            if agent_id in pair:
                self_involved = 1
            else:
                others_exist = 1
    collision_info = np.array([self_involved, others_exist], dtype=float)

    wait_count = np.array([float(env.wait_count[agent_id])])

    nodes_flat = get_node_coordinates_flat(env)
    diameter = np.array([float(graph_diameter)])
    meta = np.array([int(n_nodes), int(agent_num), int(len(env.G.edges()))], dtype=float)

    return np.concatenate([
        own_prev, own_curr, goal,
        others_curr_raw,
        collision_distance, collision_info, wait_count,
        nodes_flat,
        diameter, meta,
    ])
