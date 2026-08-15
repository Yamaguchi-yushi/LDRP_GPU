import math
import torch as th
import torch.nn as nn
from torch.nn import functional as F

from .mat import init_

class SelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, masked=False):
        super(SelfAttention, self).__init__()

        assert n_embd % n_head == 0
        self.masked = masked
        self.n_head = n_head
        self.key = init_(nn.Linear(n_embd, n_embd))
        self.query = init_(nn.Linear(n_embd, n_embd))
        self.value = init_(nn.Linear(n_embd, n_embd))
        self.proj = init_(nn.Linear(n_embd, n_embd))

    def forward(self, key, value, query):
        B, L, D = query.size()

        k = self.key(key).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)
        q = self.query(query).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)
        v = self.value(value).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if self.masked:
            causal = th.tril(th.ones(L, L, dtype=th.bool, device=att.device))
            att = att.masked_fill(~causal, float("-inf"))
        att = F.softmax(att, dim=-1)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, L, D)
        return self.proj(y)

class EncodeBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super(EncodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, masked=False)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * n_embd, n_embd)))

    def forward(self, x):
        x = self.ln1(x + self.attn(x, x, x))
        x = self.ln2(x + self.mlp(x))
        return x

class Encoder(nn.Module):
    def __init__(self, obs_dim, n_block, n_embd, n_head):
        super(Encoder, self).__init__()

        self.n_embd = n_embd
        self.obs_encoder = nn.Sequential(
            nn.LayerNorm(obs_dim),
            init_(nn.Linear(obs_dim, n_embd), activate=True),
            nn.GELU())
        self.ln = nn.LayerNorm(n_embd)
        self.blocks = nn.Sequential(*[EncodeBlock(n_embd, n_head) for _ in range(n_block)])
        self.head = nn.Sequential(
            init_(nn.Linear(n_embd, n_embd), activate=True),
            nn.GELU(),
            nn.LayerNorm(n_embd),
            init_(nn.Linear(n_embd, 1)))

    def forward(self, obs):
        x = self.obs_encoder(obs)
        obs_rep = self.blocks(self.ln(x))
        v_loc = self.head(obs_rep)
        return v_loc, obs_rep

class MATFullCritic(nn.Module):
    def __init__(self, scheme, args):
        super(MATFullCritic, self).__init__()

        self.args = args
        self.n_agents = args.n_agents

        assert not args.obs_agent_id, \
        "obs_agent_id=True is not supported by MAT: the agent-id one-hot fixes the " \
            "encoder input dim to obs_dim + n_agents, which breaks zero-shot transfer to " \
            "a different number of agents. Set obs_agent_id: False in mat.yaml "

        self.input_shape = scheme["obs"]["vshape"]
        assert isinstance(self.input_shape, int), "MAT does not support image obs for the time being!"

        self.encoder = Encoder(self.input_shape, args.n_block, args.n_embd, args.n_head)

    def forward(self, batch, t =None):
        v_loc, _ = self.encode(batch, t=t)
        return v_loc

    def encode(self, batch, t=None):
        ts = slice(None) if t is None else slice(t, t+1)
        inputs = batch["obs"][:, ts]
        inputs = inputs.reshape(-1, self.n_agents, self.input_shape)
        inputs = inputs.to(next(self.encoder.parameters()).device)
        return self.encoder(inputs)