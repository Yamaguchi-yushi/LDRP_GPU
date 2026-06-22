import warnings
warnings.filterwarnings("ignore")

import yaml
import gym
import sys
import numpy as np
from argparse import Namespace
import argparse
from runner import Runner


if __name__ == "__main__":
    reward_list = {
        "goal": 100,
        "collision": -100,
        "wait": -10,
        "move": -1,
    }

    with open("./src/config/default.yaml", 'r') as file:
        config_dict = yaml.safe_load(file)
    config = Namespace(**config_dict)

    if len(sys.argv) > 1:
        #map,agent,path,task,[method_tag]
        config.map_name = sys.argv[1]
        config.agent_num = int(sys.argv[2])
        config.path_planner = sys.argv[3]
        config.task_assigner = sys.argv[4]
        if len(sys.argv) > 5:
            config.method_tag = sys.argv[5]

    env_name = "drp_env:drp_safe-" + str(config.agent_num) + "agent_" + config.map_name + "-v2"
    #env_name = "drp_env:drp_safe-" + str(config.agent_num) + "agent_" + config.map_name + "-v2"

    # Optionally forward LaRe-Path params from config (no-op when use_lare_path=false).
    lare_path_keys = [
        "use_lare_path",
        "use_lare_path_training",
        "lare_path_factor_dim",
        "lare_path_decoder_hidden_dim",
        "lare_path_decoder_n_layers",
        "lare_path_use_transformer",
        "lare_path_transformer_heads",
        "lare_path_transformer_depth",
        "lare_path_buffer_capacity",
        "lare_path_min_buffer",
        "lare_path_update_freq",
        "lare_path_batch_size",
        "lare_path_lr",
        "use_pretrained_lare_path",
        "pretrained_lare_path_model_name",
        "use_finetuning_lare_path",
        "finetuning_lare_path_model_name",
        "lare_path_autosave",
        "lare_path_autosave_path",
        "lare_path_save_dir",
        "lare_path_save_freq_steps",
        # LaRe-Task (System B)
        "use_lare_task",
        "use_lare_task_training",
        "lare_task_factor_dim",
        "lare_task_decoder_hidden_dim",
        "lare_task_decoder_n_layers",
        "lare_task_buffer_capacity",
        "lare_task_min_buffer",
        "lare_task_update_freq",
        "lare_task_batch_size",
        "lare_task_lr",
        "use_pretrained_lare_task",
        "pretrained_lare_task_model_name",
        "use_finetuning_lare_task",
        "finetuning_lare_task_model_name",
        "lare_task_autosave",
        "lare_task_autosave_path",
        "lare_task_save_dir",
        "lare_task_save_freq_steps",
    ]
    lare_kwargs = {k: getattr(config, k) for k in lare_path_keys if hasattr(config, k)}

    # path_planner が PBS のときだけ pbs_mode=True にする.
    # PBS は待機 agent の予定も path 計画に反映するため current_goal を非 None に
    # 保つ必要があるが、それ以外 (QMIX/IQL/VDN/MAA2C) では None のままにして
    # SafeEnv の保護を機能させる. 詳細は CLAUDE.md「SafeEnv と PBS のトレードオフ」.
    pbs_mode = (getattr(config, "path_planner", "") == "pbs")

    env = gym.make(
        env_name,
        state_repre_flag="onehot_fov",
        reward_list=reward_list,
        time_limit=config.time_limit,
        task_flag=True,
        task_list=None,
        pbs_mode=pbs_mode,
        **lare_kwargs,
    )
    """
    with open("./config/algo/" + config.algo + ".yaml", 'r') as file:
        config_dict = yaml.safe_load(file)
    config = Namespace(**config_dict)
    """
    runner = Runner(config, env, reward_list)
    runner.run()
    runner.finish()