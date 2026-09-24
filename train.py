import subprocess
import time
import os
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
        f'python src/epymarl/src/main.py --config=mat --env-config=gymma --beat-interval=3600 '
        f'with env_args.time_limit=500 '
        f't_max=80050000 '
        f'env_args.key="drp_env:drp_safe-5agent_map_aoba00-v2" '
        f'env_args.state_repre_flag="onehot_fov" '
        f'env_args.use_lare_path=False '
        f'env_args.use_lare_path_training=True '
        f'env_args.use_pretrained_lare_path=True '
        f'env_args.pretrained_lare_path_model_name="FT_QMIX_PATH_Safe_map_8x5_2agents_10.0M_Safe_map_aoba00_2agents_5.0M_checkpoint.pth" '
        f'env_args.use_finetuning_lare_path=False '
        f'env_args.finetuning_lare_path_model_name="QMIX_PATH_Safe_map_8x5_2agents_5.0M_checkpoint.pth" '
        f'env_args.allow_reassign_before_pickup=False '
        # --- タスク到着のランダム化 (学習用) ---------------------------------
        # True: エピソード毎に bernoulli(p ランダム) / mmpp を引き直す。
        #       1 つの方策を全シナリオで評価するために必要
        #       (シナリオ毎に学習すると方策が需要水準を暗黙に知ることになる)。
        # False: 下の task_arrival / task_density 等の固定値で回す (従来動作)。
        f'env_args.randomize_task_arrival=True '
        f'env_args.mmpp_ratio=0.5 '        # MMPP を引く確率 (残りが bernoulli)
        f'env_args.rand_p_min=0.01 '       # bernoulli p の下限 (全マップ共通)
        f'env_args.rand_p_max=0.10 '       # bernoulli p の上限 (全マップ共通)
        # p の範囲はマップ共通。1 タスクの所要ステップがマップで 3.2 倍違うが、
        # この範囲ならどのマップも疎〜密を経験する
        #   map_8x5   (32.1 step/タスク): 0.32 〜 3.21 件/タスクサイクル
        #   map_aoba00(103.0 step/タスク): 1.03 〜 10.3 件/タスクサイクル
        # --- MMPP のパラメータ (ランダム化 ON でもこの値が使われる) ------------
        f'env_args.task_p_high=0.10 '      # 密相 (= rand_p_max)
        f'env_args.task_p_low=0.01 '       # 疎相 (= rand_p_min)
        f'env_args.task_switch_prob=0.01 ' # 相転換 5 回/ep (= 5 / time_limit=500)
        # ※ 相の長さ = 1/switch_prob = 100 step。map_aoba00 では約 1 タスクサイクル
        #    分しかないが、経路方策は需要を観測しないので問題ない。
        #    フリート方策の学習・評価 (time_limit=3000) では 0.0017 にすること。
        # --- ランダム化 OFF のときだけ使う固定値 ------------------------------
        f'env_args.task_arrival="bernoulli" '   # 'fixed' or 'bernoulli' or 'mmpp'
        f'env_args.task_density=0.3 '
        # --- 動的フリートサイジング (既定 False = 全台稼働の従来動作) ----------
        f'env_args.use_dynamic_agents=True '
        f'env_args.randomize_initial_active=True '
        f'env_args.min_active_agents=1 '
        f'env_args.max_active_agents=5 '
        # 同時学習 (True で PPO タスク割当器を経路方策と同時に学習)
        f'train_task_assigner=True '
        )

    # GPUを使用するMARLアルゴリズムをCPUで実行する場合
    # command = (
    #     f'CUDA_VISIBLE_DEVICES="" '
    #     f'python src/epymarl/src/main.py --config=qmix --env-config=gymma '
    #     f'with env_args.time_limit=500 '
    #     f't_max=20050000 '
    #     f'env_args.key="drp_env:drp_safe-5agent_map_8x5-v2" '
    #     f'env_args.state_repre_flag="onehot_fov" '
    #     f'env_args.use_lare_path=True '
    #     f'env_args.use_lare_path_training=True '
    #     f'env_args.use_pretrained_lare_path=True '
    #     f'env_args.pretrained_lare_path_model_name="FT_QMIX_PATH_Safe_map_8x5_2agents_10.0M_Safe_map_8x5_3agents_5.0M_checkpoint.pth" '
    #     f'env_args.use_finetuning_lare_path=False '
    #     f'env_args.finetuning_lare_path_model_name="QMIX_PATH_Safe_map_8x5_2agents_5.0M_checkpoint.pth" '
    #     f'env_args.allow_reassign_before_pickup=False '
    #     f'env_args.randomize_task_arrival=True '
    #     f'env_args.mmpp_ratio=0.5 '
    #     f'env_args.rand_p_min=0.01 '
    #     f'env_args.rand_p_max=0.10 '
    #     f'env_args.task_p_high=0.10 '
    #     f'env_args.task_p_low=0.01 '
    #     f'env_args.task_switch_prob=0.01 '
    #     f'env_args.task_arrival="fixed" '
    #     f'env_args.task_density=0.02 '
    #     f'env_args.use_dynamic_agents=False '
    #     f'env_args.randomize_initial_active=False '
    #     f'env_args.min_active_agents=2 '
    #     f'env_args.max_active_agents=5 '
    #     f'train_task_assigner=False '
    #     )


    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    proc = subprocess.Popen(command, shell=True, env=env)
    running_processes.append(proc)

    while len(running_processes) >= maxpurocesses:
        for p in running_processes[:]:
            if p.poll() is not None:
                running_processes.remove(p)
        time.sleep(0.1)

for p in running_processes:
    p.wait()

print("All runs completed.")
