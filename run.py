import subprocess
import time
import os
import sys
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.all_policy.policy import model_stem, list_model_seeds

map_name = [
    #"map_5x4",
    "map_8x5",
    #"map_aoba00",
    #"map_aoba01",
]

agent_num = [
    # 3,
    # 4,
    # 5,
    # 6,
    7,
    # 8,
    # 9,
    10,
    # 11,
    # 12,
    # 13,
    # 14,
    # 15,
]

path_planner = [
    #"iql",
    # "qmix",
    #"vdn",
    "mappo",
    # "qplex",
    #"happo",
    #"mat",
    #"mat_dec",
    #"pbs",
]

task_assigner = [
    "fifo",
    "tp",
]

method_tag = [
    #"",
    "safe",
    "ours",
    # "dbct",
]

reassign_before_pickup = [
    "base",
    #"reassign",
]

mat_model_agent_num = [     # mat_decのモデルを学習した際のエージェント数を指定する．
     "",
    #"4",
    # 7,
    #"8",
    #10,
]

use_safe_env = [
    "True",
    #"False",
]

model_seed = "auto"

measure_runtime = True
maxpurocesses = 1 if measure_runtime else 5

command = []
skipped = []
seen = set()

for i, j, k, l, m, n, o, u in product(map_name, agent_num, path_planner, task_assigner,
                                      method_tag, reassign_before_pickup,
                                      mat_model_agent_num, use_safe_env):
    model_n = int(o) if (k == "mat_dec" and str(o)) else int(j)
    key = (i, j, k, l, m, n, u, model_n if k == "mat_dec" else "")
    if key in seen:
        continue
    seen.add(key)
    stem = model_stem(i, model_n, k, m, n)
    seeds = list_model_seeds(stem) if model_seed == "auto" else [int(model_seed)]
    if not seeds:
        skipped.append(stem)
        continue
    for s in seeds:
        command.append([sys.executable, "-u", "test.py",
                        str(i), str(j), str(k), str(l), str(m), str(n), str(o),
                        f"model_seed={s}", f"use_safe_env={u}"])
for stem in sorted(set(skipped)):
    print(f"Skipped: {stem} (no model found)")
print(f"Total commands to run: {len(command)}")

"""
command = [
    ["python3", "test.py", "map_aoba00", "4", "pbs", "tp"]
]
"""
"""
for cmd in command:
    with open("logs/" + str(cmd[2]) + "_" + str(cmd[3]) + "_" + str(cmd[4]) + "_" + str(cmd[5]) + ".txt", "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

"""
running_processes = []

for cmd in command:
    named = dict(t.split("=", 1) for t in cmd[10:] if "=" in t)
    env_dir = "safe" if named.get("use_safe_env", "True").lower() in ("1", "true", "yes") else "unsafe"
    log_dir = "logs/" + str(cmd[3]) + "/" + env_dir + "/" + str(cmd[5]) + "/" + str(cmd[4]) + "agent"
    os.makedirs(log_dir, exist_ok=True)
    method_suffix = f"_{cmd[7]}" if len(cmd) > 7 and cmd[7] else ""
    reassign_suffix = f"_{cmd[8]}" if len(cmd) > 8 and cmd[8] else ""
    train_n = cmd[9] if len(cmd) > 9 and cmd[9] else cmd[4]
    seed_suffix = f"_seed{named['model_seed']}" if "model_seed" in named else ""
    log_name = f"{cmd[3]}_{cmd[4]}_{cmd[5]}_{cmd[6]}{method_suffix}{reassign_suffix}_{train_n}{seed_suffix}.txt"
    with open(os.path.join(log_dir, log_name), "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    running_processes.append((proc ,cmd))
    print("Started:", cmd, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    while len(running_processes) >= maxpurocesses:
        for p,c in running_processes[:]:
            if p.poll() is not None:
                print("Finished:", c)
                running_processes.remove((p,c))
        time.sleep(0.1)

for p, c in running_processes:
    p.wait()
    print("Finished:", c)

print ("\n======== All runs completed. =======")
subprocess.run([sys.executable, "aggregate.py", "--csv", "results/summary.csv"])
#"""