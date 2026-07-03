import subprocess
import time
"""
command = [
    ["python3", "test.py", "map_5x4", "3", "pbs", "tp"]
]
"""
command = [
'python3 src/main.py --config=qmix --env-config=gymma with env_args.time_limit=500 env_args.key="drp_env:drp_safe-4agent_map_aoba00-v2" env_args.state_repre_flag="onehot_fov" > train_results/qmix_drp_safe-4agent_map_8x5-v2.txt 2>&1'
]

num_runs = 1
maxpurocesses = 1
running_processes = []

for i in range(num_runs):
    #algとmap，実行step数確認，drp_envのpbs用の変更箇所
    #iql,aoba00,16050000,unsafe
    command = (
        f'python src/epymarl/src/main.py --config=qplex --env-config=gymma '
        f'with env_args.time_limit=500 '
        f't_max=50050000'
        f'env_args.key="drp_env:drp_safe-3agent_map_aoba00-v2" '
        f'env_args.state_repre_flag="onehot_fov" '
        f'env_args.use_lare_path=False '
        f'env_args.use_lare_path_training=True '
        f'env_args.use_pretrained_lare_path=True '
        f'env_args.pretrained_lare_path_model_name="FT_QMIX_PATH_Safe_map_8x5_2agents_10.0M_Safe_map_aoba00_2agents_5.0M_checkpoint.pth" '
        f'env_args.use_finetuning_lare_path=False '
        f'env_args.finetuning_lare_path_model_name="QMIX_PATH_Safe_map_8x5_2agents_5.0M_checkpoint.pth" '
        )
    
    # GPUを使用するMARLアルゴリズムをCPUで実行する場合
    # command = (
    #     f'CUDA_VISIBLE_DEVICES="" '
    #     f'python src/epymarl/src/main.py --config=mappo --env-config=gymma '
    #     f'with env_args.time_limit=500 '
    #     f't_max=50050000'
    #     f'env_args.key="drp_env:drp_safe-3agent_map_aoba00-v2" '
    #     f'env_args.state_repre_flag="onehot_fov" '
    #     f'env_args.use_lare_path=False '
    #     f'env_args.use_lare_path_training=True '
    #     f'env_args.use_pretrained_lare_path=True '
    #     f'env_args.pretrained_lare_path_model_name="FT_QMIX_PATH_Safe_map_8x5_2agents_10.0M_Safe_map_aoba00_2agents_5.0M_checkpoint.pth" '
    #     f'env_args.use_finetuning_lare_path=False '
    #     f'env_args.finetuning_lare_path_model_name="QMIX_PATH_Safe_map_8x5_2agents_5.0M_checkpoint.pth" '
    #     )

    proc = subprocess.Popen(command, shell=True)
    running_processes.append(proc)

    while len(running_processes) >= maxpurocesses:
        for p in running_processes[:]:
            if p.poll() is not None:
                running_processes.remove(p)
        time.sleep(0.1)

for p in running_processes:
    p.wait()

print("All runs completed.")
