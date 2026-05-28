import subprocess
import time
"""
command = [
    ["python3", "test.py", "map_5x4", "3", "pbs", "tp"]
]
"""
command = [
'python3 src/main.py --config=qmix --env-config=gymma with env_args.time_limit=100 env_args.key="drp_env:drp_safe-4agent_map_8x5-v2" env_args.state_repre_flag="onehot_fov" > train_results/qmix_drp_safe-4agent_map_8x5-v2.txt 2>&1'
]

num_runs = 1
maxpurocesses = 1
running_processes = []

for i in range(num_runs):
    #algとmap，実行step数確認，drp_envのpbs用の変更箇所
    #iql,aoba00,16050000,unsafe
    command = (
        f'python src/epymarl/src/main.py --config=qmix --env-config=gymma '
        f'with env_args.time_limit=500 '
        f'env_args.key="drp_env:drp_safe-4agent_map_8x5-v2" '
        f'env_args.state_repre_flag="onehot_fov" '
        f'> train_results/{i} 2>&1'
        )
    proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    running_processes.append(proc)

    while len(running_processes) >= maxpurocesses:
        for p in running_processes[:]:
            if p.poll() is not None:
                running_processes.remove(p)
        time.sleep(0.1)

for p in running_processes:
    p.wait()

print("All runs completed.")
