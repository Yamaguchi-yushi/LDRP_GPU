import os
from .policy_runner import PolicyRunner
import torch
import numpy as np

runner = None


def get_model_path(env):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    filename = f"{env.map_name}_{env.agent_num}_qmix.th"
    path = os.path.join(base_dir, "models", filename)

    return path

def policy(obs, env):
    global runner

    if runner is None:
        runner = PolicyRunner(
            model_path=get_model_path(env),
            input_shape=len(obs[0]),
            n_actions=env.n_actions,
            agent_num=env.agent_num
        )
    
    actions = []
    for agi in range(env.agent_num):
        _, avail_actions = env.get_avail_agent_actions(agi, env.n_actions)
        action = runner.get_action(agi, obs[agi], avail_actions)
        actions.append(action)

    return actions

class MARLPolicy():
    def __init__(self, args):
        self.args = args
        self.path_planner = args.path_planner
        self.method_tag = getattr(args, "method_tag", "") or ""
        self.model_reassign_tag = getattr(args, "reassign_before_pickup", "base")
        self.mat_model_agent_num = getattr(args, "mat_model_agent_num", None)
        self.runner = None
    
    def get_model_path(self, env):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        suffix = f"_{self.method_tag}" if self.method_tag else ""
        if self.path_planner == "mat_dec" and self.mat_model_agent_num is not None:
            model_n = self.mat_model_agent_num
        else:
            model_n = env.agent_num
        filename = f"{env.map_name}_{model_n}_{self.path_planner}{suffix}_{self.model_reassign_tag}.th"
        path = os.path.join(base_dir, "models", "safe", filename)
        return path
    
    def policy(self, obs, env):
        #agent_idをtrueにしている場合，以下が必要
        #identity = np.eye(env.agent_num)
        #obs = np.concatenate([obs, identity], axis=1)

        if self.runner is None:
            if self.path_planner == "mat_dec":
                from .mat_policy_runner import MatPolicyRunner
                self.runner = MatPolicyRunner(
                    model_path=self.get_model_path(env),
                    input_shape=len(obs[0]),
                    n_actions=env.n_actions,
                    agent_num=env.agent_num
                )
            else:
                self.runner = PolicyRunner(
                    model_path=self.get_model_path(env),
                    input_shape=len(obs[0]),
                    n_actions=env.n_actions,
                    agent_num=env.agent_num
                )
        
        actions = []
        for agi in range(env.agent_num):
            _, avail_actions = env.get_avail_agent_actions(agi, env.n_actions)
            action = self.runner.get_action(agi, obs[agi], avail_actions)
            actions.append(action)

        return actions

    def reset_hidden(self, ag_idx=None):
        """
        エピソード開始時 / エージェント再投入時に RNN のhidden state を戻す
        """
        if self.runner is not None and hasattr(self.runner, "reset_hidden"):
            self.runner.reset_hidden(ag_idx)
    
