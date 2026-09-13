import os
import re
from .policy_runner import PolicyRunner
import torch
import numpy as np

runner = None

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "safe")

def model_stem(map_name, model_n, path_planner, method_tag="", reassign_tag="base"):
    suffix = f"_{method_tag}" if method_tag else ""
    return f"{map_name}_{model_n}_{path_planner}{suffix}_{reassign_tag}"

def list_model_seeds(stem, models_dir=MODELS_DIR):
    if not os.path.isdir(models_dir):
        return []
    pat = re.compile(rf"^{re.escape(stem)}_seed(\d+)\.th$")
    seeds = [int(m.group(1)) for m in (pat.match(n) for n in os.listdir(models_dir)) if m]
    if not seeds and os.path.exists(os.path.join(models_dir, stem + ".th")):
        seeds = [0]
    return sorted(seeds)

def resolve_model_path(stem, model_seed=0, models_dir=MODELS_DIR):
    cand = os.path.join(models_dir, f"{stem}_seed{model_seed}.th")
    if os.path.exists(cand):
        return cand
    legacy = os.path.join(models_dir, stem + ".th")
    if os.path.exists(legacy):
        if int(model_seed) != 0:
            raise ValueError(f"Legacy model {legacy} exists, but model_seed={model_seed} is requested")
        return legacy
    raise FileNotFoundError(f"Model file not found for stem={stem}, model_seed={model_seed} in {models_dir}")


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
        self.model_seed = int(getattr(args, "model_seed", 0) or 0)
        self.resolved_model_path = None
        self.runner = None

    def get_model_stem(self, env):
        if self.path_planner == "mat_dec" and self.mat_model_agent_num is not None:
            model_n = self.mat_model_agent_num
        else:
            model_n = env.agent_num
        return model_stem(env.map_name, model_n, self.path_planner,
                          self.method_tag, self.model_reassign_tag)
    
    def get_model_path(self, env):
        self.resolved_model_path = resolve_model_path(self.get_model_stem(env), self.model_seed)
        return self.resolved_model_path
    
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
    
