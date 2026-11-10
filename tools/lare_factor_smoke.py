"""LaRe-Path の評価関数の版を、短いエピソードで動かして確かめる (design/lare_factor_versions.md §5).

    python tools/lare_factor_smoke.py <リポジトリの場所> <版 (0 = 指定しない)> <LaRe を使うか 0/1>

機体は最短路で各自の目的地へ向かう (方策の代わり)。版 2・3 では、ピックアップ・配達の印の数と、
10 因子の平均・最小・最大を出す。版 3 では観測の長さと、版 3 で足した記録 (向かったノード・
捨てたタスク・理想の道のり) も出す。乱数を固定しているので、同じ版・同じコードなら報酬の合計は毎回同じ。
"""
import sys, os, numpy as np, networkx as nx
root = sys.argv[1]; version = int(sys.argv[2]); lare = sys.argv[3] == "1"
sys.path.insert(0, os.path.join(root, "src", "main")); sys.path.insert(0, os.path.join(root, "src")); sys.path.insert(0, root)
import gym, drp_env
kw = dict(state_repre_flag='onehot_fov', time_limit=200, task_flag=True, task_list=None,
          use_lare_path=lare, use_lare_path_training=True, use_pretrained_lare_path=False,
          use_finetuning_lare_path=False, lare_path_min_buffer=1, lare_path_update_freq=1,
          lare_path_batch_size=2, lare_path_autosave=False, task_arrival="bernoulli", task_density=0.1)
if version: kw["lare_path_factor_version"] = version
import torch as _t; _t.manual_seed(0)
env = gym.make('drp_env:drp_safe-5agent_map_8x5-v2', **kw); e = env.unwrapped
paths = dict(nx.all_pairs_dijkstra_path(e.G, weight="weight"))
np.random.seed(0); import torch; torch.manual_seed(0)
tot_r = 0.0; facs = []; ev = np.zeros(2); nxt = 0; drop = 0; ideal = []
for ep in range(2):
    env.reset(); done = False
    while not done:
        act = []
        for i in range(5):
            if e.current_goal[i] is not None: act.append(int(e.current_goal[i])); continue
            s, g = int(e.current_start[i]), int(e.goal_array[i])
            p = paths[s][g]; act.append(p[1] if len(p) > 1 else s)
        obs, r, term, info = env.step({'pass': act, 'task': None})
        tot_r += float(sum(r)); done = all(term)
        if lare and version >= 2:
            ev += np.array(e._lare_events).sum(0)
            facs.append(e.lare_path_module.compute_factors(e._lare_prev_onehot_pos, e._lare_current_colliding_pairs))
        if lare and version >= 3:
            nxt += sum(1 for v in e._lare_next if v is not None)
            drop += e._lare_dropped_step
            ideal += [e._lare_ideal_len[k] for k in range(5) if e.assigned_tasks[k]]
print("version", version, "lare", lare, "reward_sum %.4f" % tot_r, "completion", e.task_completion)
if facs:
    F = np.concatenate(facs)
    names = {2: ['deliver','pickup','progress','route','override','stall','delay','prox','blocker','collision'],
             3: ['progress','remaining','event','delay','stall','safety','prox','conflict','block','backlog']}[version]
    print("events pick/deliver", ev)
    if version >= 3:
        spec = e.lare_path_module._factor_spec
        row = spec["build_obs"](e, 0, e.lare_path_module.graph_diameter, e._lare_prev_onehot_pos, [])
        N, K = e.n_nodes, 4
        print("obs length %d (expected 7N+4NK+6K+26 = %d)" % (len(row), 7 * N + 4 * N * K + 6 * K + 26))
        print("moved agent-steps %d  dropped %d (env total %d)  ideal_len mean %.1f max %.1f"
              % (nxt, drop, e.task_dropped_count, np.mean(ideal), np.max(ideal)))
    for n, m, lo, hi in zip(names, F.mean(0), F.min(0), F.max(0)): print("  %-9s mean %6.3f  min %6.2f  max %6.2f" % (n, m, lo, hi))
