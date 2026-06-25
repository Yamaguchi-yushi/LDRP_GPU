import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class MLPMATAgent(nn.Module):

    def __init__(self, input_shape, args):
        super(MLPMATAgent, self).__init__()
        self.args = args

        # Currently, only vector-based input is supported
        self.is_image = False
        assert isinstance(input_shape, int), "MAT-DEC does not support image obs for the time being!"

        self.n_agent = args.n_agents
        self.input_shape = input_shape
        self.decoder = Decoder(input_shape, args.n_actions, args.n_embd)
        self.critic = None
        self.device = None

    def forward(self, obs, action, available_actions=None):
        # obs: (batch, n_agent, obs_dim)
        # action: (batch, n_agent, 1)
        # available_actions: (batch, n_agent, act_dim)

        action = action.long()
        action_log, entropy = self.discrete_parallel_act(obs, action, available_actions)

        return action_log, entropy

    def discrete_parallel_act(self, obs, action, available_actions=None):

        logit = self.decoder(obs)
        if available_actions is not None:
            logit[available_actions == 0] = -1e10

        distri = Categorical(logits=logit)
        action_log = distri.log_prob(action.squeeze(-1)).unsqueeze(-1)
        entropy = distri.entropy().unsqueeze(-1)

        return action_log, entropy

    def get_actions(self, ep_batch, t, obs, available_actions=None, deterministic=False):

        batch_size = np.shape(obs)[0]
        v_loc = self.critic(ep_batch, t)
        output_action, output_action_log = self.discrete_autoregreesive_act(
            obs,
            batch_size,
            available_actions,
            deterministic
        )

        return output_action, output_action_log, v_loc

    def discrete_autoregreesive_act(
            self,
            obs,
            batch_size,
            available_actions=None,
            deterministic=False
    ):

        output_action = torch.zeros((batch_size, self.n_agent, 1), dtype=torch.long, device=self.device)
        output_action_log = torch.zeros_like(output_action, dtype=torch.float32)

        for i in range(self.n_agent):
            logit = self.decoder(obs)[:, i, :]
            if available_actions is not None:
                logit[available_actions[:, i, :] == 0] = -1e10

            distri = Categorical(logits=logit)
            action = distri.probs.argmax(dim=-1) if deterministic else distri.sample()
            action_log = distri.log_prob(action)

            output_action[:, i, :] = action.unsqueeze(-1)
            output_action_log[:, i, :] = action_log.unsqueeze(-1)

        return output_action, output_action_log

    def get_values(self, obs):

        v_tot = self.critic(obs)

        return v_tot

    def evaluate_actions(self, ep_batch, t, agent_inputs, actions, available_actions):

        v_loc = self.critic(ep_batch, t)
        action_log, entropy = self.forward(agent_inputs, actions, available_actions)

        return action_log, v_loc, entropy


class Decoder(nn.Module):

    def __init__(self, obs_dim, action_dim, n_embd):

        super(Decoder, self).__init__()

        self.action_dim = action_dim
        self.n_embd = n_embd

        self.mlp = nn.Sequential(
            nn.LayerNorm(obs_dim),
            init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU(),
            nn.LayerNorm(n_embd),
            init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(),
            nn.LayerNorm(n_embd),
            init_(nn.Linear(n_embd, action_dim))
        )

    def forward(self, obs):

        logit = self.mlp(obs)

        return logit


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module


def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)