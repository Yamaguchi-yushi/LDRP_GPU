import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical

from modules.critics.mat import init_
from modules.critics.mat_full import SelfAttention

class MATAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(MATAgent, self).__init__()
        self.args = args

        assert isinstance(input_shape, int), "MAT does not support image obs for the time being!"

        self.n_actions = args.n_actions
        self.input_shape = input_shape
        self.decoder = Decoder(args.n_actions, args.n_block, args.n_embd, args.n_head)
        self.critic = None
        self.device = None

    def get_actions(self, ep_batch, t, obs, available_actions=None, deterministic=False):
        """
        obs: (n_agents, obs_dim)
        available_actions: (n_agents, n_actions)
        """
        v_loc, obs_rep = self.critic.encode(ep_batch, t)
        if available_actions is not None:
            available_actions = available_actions.to(obs_rep.device)

        output_action, output_action_log = self.discrete_autoregressive_act(
            obs_rep, available_actions, deterministic)

        return output_action, output_action_log, v_loc

    def discrete_autoregressive_act(self, obs_rep, available_actions=None, deterministic=False):
        """
        agent 0 -> agent 1 -> ... -> agent n の順に行動を決定する
        """
        batch_size, n_agent = obs_rep.shape[0], obs_rep.shape[1] 

        shifted_action = torch.zeros((batch_size, n_agent, self.n_actions + 1),
                                    dtype=torch.float32, device=obs_rep.device)
        shifted_action[:, 0, 0] = 1.0  # agent0の前の行動は "no action" とする
        output_action = torch.zeros((batch_size, n_agent, 1),
                                    dtype=torch.long, device=obs_rep.device)
        output_action_log = torch.zeros_like(output_action, dtype=torch.float32)

        for i in range(n_agent):
            logit = self.decoder(shifted_action, obs_rep)[:, i, :]
            if available_actions is not None:
                logit[available_actions[:, i, :] == 0] = -1e10

            distri = Categorical(logits=logit)
            action = distri.probs.argmax(dim=-1) if deterministic else distri.sample()

            output_action[:, i, :] = action.unsqueeze(-1)
            output_action_log[:, i, :] = distri.log_prob(action).unsqueeze(-1)
            if i + 1 < n_agent:
                shifted_action[:, i + 1, 1:] = F.one_hot(action, num_classes=self.n_actions).float()

        return output_action, output_action_log

    def evaluate_actions(self, ep_batch, t, agent_inputs, actions, available_actions):
        # 学習時に呼ばれる．　rolloutと同じ分布を再構成して log_prob, entropy valueを計算する 
        v_loc, obs_rep = self.critic.encode(ep_batch, t)
        actions = actions.long().to(obs_rep.device)
        if available_actions is not None:
            available_actions = available_actions.to(obs_rep.device)

        action_log, entropy = self.discrete_parallel_act(obs_rep, actions, available_actions)

        return action_log, v_loc, entropy

    def discrete_parallel_act(self, obs_rep, actions, available_actions=None):
        """
        学習時： 記録済み行動列を１個右にずらして一括入力
        """
        batch_size, n_agent = obs_rep.shape[0], obs_rep.shape[1] 

        one_hot_action = F.one_hot(actions.squeeze(-1), num_classes=self.n_actions).float()
        shifted_action = torch.zeros((batch_size, n_agent, self.n_actions + 1),
                                    dtype=torch.float32, device=obs_rep.device)
        shifted_action[:, 0, 0] = 1.0  # agent0の前の行動は "no action" とする
        shifted_action[:, 1:, 1:] = one_hot_action[:, :-1, :]

        logit = self.decoder(shifted_action, obs_rep)
        if available_actions is not None:
            logit[available_actions == 0] = -1e10

        distri = Categorical(logits=logit)
        action_log = distri.log_prob(actions.squeeze(-1)).unsqueeze(-1)
        entropy = distri.entropy().unsqueeze(-1)

        return action_log, entropy

class Decoder(nn.Module):
    """
    行動列 (右シフト済み) + obs_rep -> 各 agent の行動 logits。
    """
    def __init__(self, action_dim, n_block, n_embd, n_head):
        super(Decoder, self).__init__()

        self.n_embd = n_embd
        self.action_encoder = nn.Sequential(
            init_(nn.Linear(action_dim + 1, n_embd, bias=False), activate=True),
            nn.GELU())
        self.ln = nn.LayerNorm(n_embd)
        self.blocks = nn.ModuleList([DecodeBlock(n_embd, n_head) for _ in range(n_block)])
        self.head = nn.Sequential(
            init_(nn.Linear(n_embd, n_embd), activate=True),
            nn.GELU(),
            nn.LayerNorm(n_embd),
            init_(nn.Linear(n_embd, action_dim)))

    def forward(self, shifted_action, obs_rep):
        x = self.ln(self.action_encoder(shifted_action))   
        for block in self.blocks:
            x = block(x, obs_rep)                    
        return self.head(x)

class DecodeBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super(DecodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)
        self.attn1 = SelfAttention(n_embd, n_head, masked=True)
        self.attn2 = SelfAttention(n_embd, n_head, masked=True)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * n_embd, n_embd)))

    def forward(self, x, rep_enc):
        x = self.ln1(x + self.attn1(x, x, x))
        x = self.ln2(rep_enc+ self.attn2(key=x, value=x, query=rep_enc))
        x = self.ln3(x + self.mlp(x))
        return x

