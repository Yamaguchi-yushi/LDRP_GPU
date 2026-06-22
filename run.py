import subprocess
import time
import os
import sys

map_name = [
    #"map_5x4",
    "map_8x5",
    #"map_aoba00",
    #"map_aoba01",
]

agent_num = [
    #3,
    4,
    #5,
]

path_planner = [
    #"iql",
    #"qmix",
    #"vdn",
    "mappo",
    #"qplex",
    #"happo",
    #"mat",
    #"matdec",
    #"pbs",
]

method_tag = [
    #"",
    "safe",
    "ours",
]

task_assigner = [
    "fifo",
    "tp",
]
#"""
command = [
    [sys.executable, "test.py", str(i), str(j), str(k), str(l), str(m)]
    for i in map_name
    for j in agent_num
    for k in path_planner
    for l in task_assigner
    for m in method_tag
]
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
maxpurocesses = 5
running_processes = []

#logファイルのパス変更ver
for cmd in command:
    log_dir = "logs/" + str(cmd[2]) + "/safe"
    os.makedirs(log_dir, exist_ok=True)
    method_suffix = f"_{cmd[6]}" if len(cmd) > 6 and cmd[6] else ""
    with open(log_dir + "/" + str(cmd[2]) + "_" + str(cmd[3]) + "_" + str(cmd[4]) + "_" + str(cmd[5]) + method_suffix + ".txt", "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    running_processes.append((proc ,cmd))
    print("Started:", cmd, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    while len(running_processes) >= maxpurocesses:
        for p,c in running_processes[:]:
            if p.poll() is not None:
                print("Finished:", c)
                running_processes.remove((p,c))
        time.sleep(0.1)

#ver2
"""
for cmd in command:
    with open("logs/" + str(cmd[2]) + "_" + str(cmd[3]) + "_" + str(cmd[4]) + "_" + str(cmd[5]) + ".txt", "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    running_processes.append((proc ,cmd))

    while len(running_processes) >= maxpurocesses:
        for p,c in running_processes[:]:
            if p.poll() is not None:
                print("Finished:", c)
                running_processes.remove((p,c))
        time.sleep(0.1)
"""

for p, c in running_processes:
    p.wait()
    print("Finished:", c)
#"""