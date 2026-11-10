import warnings
warnings.filterwarnings("ignore")

import yaml
from itertools import product
import gym
import sys
import numpy as np
from argparse import Namespace
import argparse
import torch
from runner import Runner
import os
from src.all_policy.policy import model_stem_form_args, resolve_model_path, PATH_MODELS_DIR, TASK_MODELS_DIR


def _model_file(config, models_dir):
    """この条件で読むモデルのファイル名 (見出し用)。無ければ (not found) を付けて返す."""
    if config.path_planner == "mat_dec" and getattr(config, "mat_model_agent_num", None):
        model_n = config.mat_model_agent_num
    else:
        model_n = config.agent_num
    stem = model_stem_form_args(config, model_n)
    seed = int(getattr(config, "model_seed", 0) or 0)
    try:
        return os.path.basename(resolve_model_path(stem, seed, models_dir))
    except (FileNotFoundError, ValueError):
        return f"{stem}_seed{seed}.th (not found)"


def print_condition_header(k, total, config, arrival, reassign_flag, model_tag, training):
    """各条件の評価を始める前に、その条件を見やすく並べて出す."""
    mode, _, p = arrival.partition(":")
    if mode == "bernoulli":
        arr = f"bernoulli (p={p})"
    elif mode == "mmpp":
        arr = (f"mmpp (p_high={config.task_p_high}, p_low={config.task_p_low}, "
               f"switch_prob={config.task_switch_prob})")
    elif mode == "fixed":
        arr = "fixed (1 task / step)"
    else:
        arr = mode

    if config.path_planner == "pbs":
        path_model = "- (pbs: search-based, no model)"
    else:
        path_model = _model_file(config, PATH_MODELS_DIR)
    if config.task_assigner != "ppo":
        task_model = f"- ({config.task_assigner}: rule-based, no model)"
    elif training:
        task_model = "- (training from scratch)"
    elif getattr(config, "ppo_task_checkpoint_path", ""):
        task_model = f"{config.ppo_task_checkpoint_path} (ppo_task_checkpoint_path)"
    else:
        task_model = _model_file(config, TASK_MODELS_DIR)

    line = "=" * 80
    print(f"\n{line}", flush=True)
    print(f"Condition {k}/{total}{'  [TRAINING]' if training else ''}")
    print(f"  map / agents    : {config.map_name} / {config.agent_num}")
    print(f"  path planner    : {config.path_planner}  (method_tag={getattr(config, 'method_tag', '') or '-'}, trained: {model_tag})")
    print(f"  task assigner   : {config.task_assigner}")
    print(f"  model seed      : {getattr(config, 'model_seed', 0)}")
    print(f"  path model      : {path_model}")
    print(f"  task model      : {task_model}")
    print(f"  task arrival    : {arr}")
    print(f"  reassign        : {'allowed' if reassign_flag else 'not allowed'} at execution")
    print(f"  env             : safe={getattr(config, 'use_safe_env', True)}, "
          f"dynamic_agents={getattr(config, 'use_dynamic_agents', False)}")
    print(f"  episodes        : {config.test_num} x {config.time_limit} steps "
          f"(eval_seed={config.eval_seed}, episode_seed_base={config.episode_seed_base})")
    print(line, flush=True)


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
        #map,agent,path,task,[method_tag],[mat_model_agent_num]
        config.map_name = sys.argv[1]
        config.agent_num = int(sys.argv[2])
        config.path_planner = sys.argv[3]
        config.task_assigner = sys.argv[4]
        for tok in sys.argv[5:]:
            if tok == "":
                continue
            if "=" in tok:
                key, val = tok.split("=", 1)
                if key == "model_seed":
                    config.model_seed = int(val)
                elif key == "use_safe_env":
                    config.use_safe_env = val.lower() in ("1", "true", "yes") 
                elif key == "use_dynamic_agents":
                    config.use_dynamic_agents = val.lower() in ("1", "true", "yes")
                elif key == "arrival":
                    config.eval_arrivals = val.split(",")
                elif key == "allow_reassign":
                    config.eval_allow_reassign = [v.lower() in ("1", "true", "yes") for v in val.split(",")]
                else:
                    raise ValueError(f"Unknown key in argument: {key}")
            elif tok in ("base", "reassign"):
                config.reassign_before_pickup = tok
            elif tok.isdigit():
                config.mat_model_agent_num = int(tok)
            else:
                config.method_tag = tok

    use_safe_env = bool(getattr(config, "use_safe_env", True))
    prefix = "drp_safe-" if use_safe_env else "drp-"
    env_name = f"drp_env:{prefix}{config.agent_num}agent_{config.map_name}-v2"
    config.env_name = env_name
    print(f"[test] env={env_name} method_tag={getattr(config, 'method_tag', '') or '(none)'}", flush=True)

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
        # タスク到着プロセスのランダム化 (学習用) / エピソード単位 seed (評価用)
        "randomize_task_arrival",
        "mmpp_ratio",
        "rand_p_min",
        "rand_p_max",
        "episode_seed_base",
    ]
    lare_kwargs = {k: getattr(config, k) for k in lare_path_keys if hasattr(config, k)}

    dynamic_agent_keys = [
        "use_dynamic_agents",
        "randomize_initial_active",
        "min_active_agents",
        "max_active_agents",
        "initial_active_num",
        "exclude_station_from_tasks",
    ]
    dynamic_agent_kwargs = {k: getattr(config, k) for k in dynamic_agent_keys if hasattr(config, k)}

    # path_planner が PBS のときだけ pbs_mode=True にする.
    # PBS は待機 agent の予定も path 計画に反映するため current_goal を非 None に
    # 保つ必要があるが、それ以外 (QMIX/IQL/VDN/MAA2C) では None のままにして
    # SafeEnv の保護を機能させる. 詳細は CLAUDE.md「SafeEnv と PBS のトレードオフ」.
    pbs_mode = (getattr(config, "path_planner", "") == "pbs")

    model_tag = getattr(config, "reassign_before_pickup", "base")
    training = bool(getattr(config, "train_task_assigner", False))
    if training and config.task_assigner != "ppo":
        raise ValueError("train_task_assigner is True but task_assigner is not 'ppo'.")
    
    if training:
        arrivals = [config.task_arrival]
        reassign_flags = [bool(config.allow_reassign_before_pickup)]
    else:
        arrivals = list(config.eval_arrivals)
        reassign_flags = list(config.eval_allow_reassign)
    for arrival in arrivals:
        if arrival.startswith("bernoulli") and ":" not in arrival:
            raise ValueError(f"'{arrival}' needs a probability, e.g. bernoulli:0.05")

    combos = list(product(arrivals, reassign_flags))
    for k, (arrival, reassign_flag) in enumerate(combos, 1):
        print_condition_header(k, len(combos), config, arrival, reassign_flag, model_tag, training)
        mode, _, p = arrival.partition(":")
        config.task_arrival = mode
        if p:
            config.task_density = float(p)
        arrival_kwargs = {k: getattr(config, k) for k in 
                          ("task_arrival", "task_density", "task_p_high", "task_p_low", "task_switch_prob")
                          if hasattr(config, k)}
        config.allow_reassign_before_pickup = reassign_flag
        np.random.seed(config.seed if training else config.eval_seed)
        torch.manual_seed(config.seed)
        env = gym.make(
            env_name,
            state_repre_flag="onehot_fov",
            reward_list=reward_list,
            time_limit=config.time_limit,
            task_flag=True,
            task_list=None,
            pbs_mode=pbs_mode,
            allow_reassign_before_pickup=reassign_flag,
            **lare_kwargs,
            **dynamic_agent_kwargs,
            **arrival_kwargs
        )
        """
        with open("./config/algo/" + config.algo + ".yaml", 'r') as file:
            config_dict = yaml.safe_load(file)
        config = Namespace(**config_dict)
        """
        runner = Runner(config, env, reward_list, training=training)
        runner.run()
        runner.finish()