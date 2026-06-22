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
        self.runner = None
    
    def get_model_path(self, env):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        suffix = f"_{self.method_tag}" if self.method_tag else ""
        filename = f"{env.map_name}_{env.agent_num}_{self.path_planner}{suffix}.th"
        path = os.path.join(base_dir, "models", "safe", filename)

        return path
    
    def policy(self, obs, env):
        #agent_idをtrueにしている場合，以下が必要
        #identity = np.eye(env.agent_num)
        #obs = np.concatenate([obs, identity], axis=1)

        if self.runner is None:
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
    
