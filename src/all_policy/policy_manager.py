from .policy import MARLPolicy
from .pbs import PBS

class PolicyManager():
    def __init__(self, args):
        self.args = args
        if args.path_planner == "pbs":
            self.path_planner = PBS(args)
        else:
            self.path_planner = MARLPolicy(args)
    
    def policy(self, obs, env):
        actions = self.path_planner.policy(obs, env)
        return actions

    def reset_hidden(self, ag_idx=None):
        """
        エピソード開始時 / エージェント再投入時に RNN のhidden state を戻す
        """
        if hasattr(self.path_planner, "reset_hidden"):
            self.path_planner.reset_hidden(ag_idx)