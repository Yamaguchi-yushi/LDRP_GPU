import torch
import numpy as np
from collections import deque
import math
import os
import matplotlib.pyplot as plt
from copy import deepcopy
import time
import sys

import yaml

from src.policy import Policy
from src.all_policy.policy_manager import PolicyManager
from src.task_assign.task_manager import TaskManager


def _resolve_running_steps(args):
    """running_steps == -1 のとき epymarl の t_max を流用する.

    target: 経路計画の MARL (epymarl) 学習ステップと PPO タスク割当の学習打ち切り
    ステップを揃えたいとき、`running_steps: -1` と書くだけで自動連動させる.
    フォールバック: epymarl の config が読めない場合はデフォルト 20_000_000.
    """
    rs = getattr(args, "running_steps", None)
    if rs is None or int(rs) >= 0:
        return int(rs) if rs is not None else 20_000_000
    # rs == -1 (or any negative) -> read epymarl t_max
    cfg_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "src", "epymarl", "src", "config", "default.yaml",
    )
    try:
        with open(cfg_path, "r") as f:
            t_max = int(yaml.safe_load(f).get("t_max", 20_000_000))
        print(f"[runner] running_steps=-1 -> using epymarl t_max={t_max}")
        return t_max
    except Exception as e:
        print(f"[runner] running_steps=-1 fallback: failed to read t_max ({e}); using 20000000")
        return 20_000_000

class Runner():
    def __init__(self, args, env, reward_list, training=False):

        # Prepare directories
        self.args = args
        self.args.task_num = env.task_num
        self.args.node_num = env.n_nodes

        self.env = env
        self.reward_list = reward_list
        self.test_num = args.test_num
        self.training = training
        if training:
            self.test_mode = False
        else:
            self.test_mode = True

        self.episode_length = self.env.time_limit
        self.current_step = 0
        self.env_step = 0
        self.current_episode = 0
        self.max_step = _resolve_running_steps(args)
        self.batch_size = args.batch_size
        self.check_interval = 100000
        args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.info_buffer = deque(maxlen=self.test_num)
        self.path_planner = PolicyManager(self.args)
        self.task_manager = TaskManager(self.args.task_assigner, self.args)
        self.both_policy_manager = Policy(self.args)

    def get_avail_actions(self):
        avail_actions = []
        for i in range(self.env.n_agents):
            avail_actions.append(self.env.get_avail_agent_actions(i, self.env.n_actions)[0])
        avail_actions = torch.tensor(avail_actions, dtype=torch.int32)

        return avail_actions
    
    def run_episode(self):

        obs_n = self.env.reset()
        done = False
        episode_score = 0
        env_step = 0
        #実験用
        tmp_step = 0
        tmp_goal = self.env.goal_array.copy()
        self.tmp_flag = False

        
        
        while not done:
            """
            agents_action = self.path_planner.policy(obs_n, self.env)
            task_assign = self.task_manager.assign_task(self.env)
            joint_action = {"pass": agents_action, "task": task_assign}
            """
            self.env.reset_prepick_assignments()
            joint_action = self.both_policy_manager.policy(obs_n, self.env)

            next_obs_n, rew_n, terminated_n, info = self.env.step(joint_action)

            done = all(terminated_n)

            #報酬をバッファへ
            if self.training:
                # LaRe-Task: when the task decoder is trained, replace the env-reward sum
                # with the proxy reward for this step's assignment decisions.
                if (
                    isinstance(info, dict)
                    and info.get("lare_task_is_trained", False)
                ):
                    task_reward = float(info.get("lare_task_proxy_reward", 0.0))
                else:
                    task_reward = float(sum(rew_n))
                self.task_manager.task_assigner.buffer_add_rewards(task_reward, done)

            episode_score += np.mean(rew_n)
                            
            env_step += 1
            obs_n = deepcopy(next_obs_n)

            if self.env.goal_array != tmp_goal:
                tmp_goal = self.env.goal_array.copy()
                tmp_step = 0
            else:
                tmp_step += 1
            if tmp_step > 40:
                self.tmp_flag = True

            #"""
            # if True:
            #     #print("############################")
            #     print("step:", env_step)
            #     #print("agents_action:", agents_action)
            #     print("current_start:", self.env.current_start)
            #     print("current_goal:", self.env.current_goal)
            #     print("goal_array:", self.env.goal_array)
            #     print("obs", self.env.obs)
            #     #print("current_tasklist:", self.env.current_tasklist)
            #     #print("assigned_tasks:", self.env.assigned_tasks)
            #     #print("assigned_list:", self.env.assigned_list)
            #     #print("task_assign:", task_assign)
            #     print("############################")

            # #"""

        return episode_score, env_step, info

    def run(self):

        step_tmp = 0
        #強化学習用
        if self.training:
            self.task_Agent.task_assigner.set_test_mode(False)
            while self.current_step < self.max_step:
                episode_score, env_step, info = self.run_episode()
                self.info_buffer.append(info)
                self.current_step += env_step
                step_tmp += env_step

                self.task_Agent.task_assigner.process_end_episode()

                #training
                
                if self.task_Agent.task_assigner.update_ready():
                    a_loss, c_loss, e_loss = self.task_Agent.task_assigner.update()
                
                #log
                if step_tmp > self.check_interval:
                    print("Current step:", self.current_step)
                    print("a_loss:", a_loss.numpy(), "\nc_loss:", c_loss.numpy(), "\ne_loss:", e_loss.numpy())
                    print("Average task completion:", np.mean([info["task_completion"] for info in self.info_buffer]))
                    step_tmp = 0
                

            self.test_mode = True


        #実行ループ
        times = []
        tmp_list = []
        self.info_buffer = deque(maxlen=self.test_num)
        if self.test_mode:
            for i in range(self.test_num):
                start = time.perf_counter()
                episode_score, env_step, info = self.run_episode()
                end = time.perf_counter()

                self.info_buffer.append(info)
                #print(info["goal_account"])
                tmp_list.append(self.tmp_flag)

                times.append(end - start)
                if (i+1) % 10 == 0:
                    print(f"Test Episode {i+1}/{self.test_num} completed.")

        steps = [info["step"] for info in self.info_buffer]
        goal_account = [info["goal_account"] for info in self.info_buffer]
        task_completion = [info["task_completion"] for info in self.info_buffer]
        full_completion = [info["task_completion"] for info in self.info_buffer if not info["collision"]]
        non_lock_completion = [info["task_completion"] for idx, info in enumerate(self.info_buffer) if tmp_list[idx]==False]
        total = len(self.info_buffer)
        collision_count = total - len(full_completion)
        collision_rate = collision_count / total if total > 0 else 0.0
        non_collision_mean = np.mean(full_completion) if full_completion else 0.0

        print("=== 集計結果 ===")
        print(f"Total test episodes: {total}")
        print(f"Average steps:       {np.mean(steps):.1f}")

        print("--- タスク配送 ---")
        print(f"Average task completion (全エピソード): {np.mean(task_completion):.2f}")
        print(f"衝突なし平均配送             ({len(full_completion)} ep): {non_collision_mean:.2f}")
        print(f"最高値: {np.max(task_completion)}")
        print(f"最低値: {np.min(task_completion)}")

        print("--- エピソード終了理由 ---")
        print(f"衝突終了: {collision_count}/{total} ({collision_rate*100:.1f}%)")

        print("--- 実行時間 ---")
        print(f"合計:   {np.sum(times):.2f} 秒")
        print(f"平均/ep: {np.mean(times):.2f} 秒")
        #print("ロックなし", np.mean(non_lock_completion), len(non_lock_completion))

        return

    def finish(self):
        self.env.close()

