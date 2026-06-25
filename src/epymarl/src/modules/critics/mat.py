import math
import torch as th
import torch.nn as nn
from torch.nn import functional as F


class MATCritic(nn.Module):
    def __init__(self, scheme, args):
        super(MATCritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.is_image = False

        self.input_shape = self._get_input_shape(scheme)

        self.encoder = Encoder(self.input_shape, args.n_block, args.n_embd, args.n_head, self.n_agents)

    def forward(self,  batch, t=None):
        inputs, bs, max_t = self._build_inputs(batch, t=t)
        inputs = inputs.reshape(-1, self.n_agents, self.input_shape)

        v_loc = self.encoder(inputs)

        return v_loc

    def _build_inputs(self, batch, t=None):
        bs = batch["batch_size"]
        max_t = batch["max_seq_length"] if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []
        # observations
        inputs.append(batch["obs"][:, ts])

        # observation
        assert not (self.is_image is True and self.args.obs_individual_obs is True), \
            "In case of state image, obs_individual_obs is not supported."
        if self.args.obs_individual_obs:
            inputs.append(batch["obs"][:, ts])

        # actions (masked out by agent)
        actions = batch["actions_onehot"][:, ts].view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
        agent_mask = (1 - th.eye(self.n_agents, device=batch["device"]))
        agent_mask = agent_mask.view(-1, 1).repeat(1, self.n_actions).view(self.n_agents, -1)
        inputs.append(actions * agent_mask.unsqueeze(0).unsqueeze(0))

        # last actions
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, 0:1]).view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1))
            elif isinstance(t, int):
                inputs.append(batch["actions_onehot"][:, slice(t-1, t)].view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1))
            else:
                last_actions = th.cat([th.zeros_like(batch["actions_onehot"][:, 0:1]),
                                       batch["actions_onehot"][:, :-1]],
                                      dim=1)
                last_actions = last_actions.view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
                inputs.append(last_actions)

        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch["device"]).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))

        if self.is_image is False:
            inputs = th.cat([x.reshape(bs, max_t, self.n_agents, -1) for x in inputs], dim=-1)
        else:
            inputs[1] = th.cat([x.reshape(bs, max_t, self.n_agents, -1) for x in inputs[1:]], dim=-1)
            del inputs[2:]
            assert len(inputs) == 2, "length of inputs: {}".format(len(inputs))
        return inputs, bs, max_t

    def _get_input_shape(self, scheme):
        # state
        input_shape = scheme["obs"]["vshape"]
        if isinstance(input_shape, int):
            # observation
            if self.args.obs_individual_obs:
                input_shape += scheme["obs"]["vshape"]
            # actions
            input_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents
            # last action
            if self.args.obs_last_action:
                input_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents
            # agent id
            if self.args.obs_agent_id:
                input_shape += self.n_agents
        elif isinstance(input_shape, tuple):
            assert self.args.obs_individual_obs is False, "In case of state image, obs_individual_obs is not supported."
            self.is_image = True
            input_shape = [input_shape, 0]
            input_shape[1] += scheme["actions_onehot"]["vshape"][0] * self.n_agents
            if self.args.obs_last_action:
                input_shape[1] += scheme["actions_onehot"]["vshape"][0] * self.n_agents
            if self.args.obs_agent_id:
                input_shape[1] += self.n_agents
            input_shape = tuple(input_shape)
        return input_shape


class Encoder(nn.Module):

    def __init__(self, obs_dim, n_block, n_embd, n_head, n_agent):
        super(Encoder, self).__init__()

        self.obs_dim = obs_dim
        self.n_embd = n_embd
        self.n_agent = n_agent

        self.obs_encoder = nn.Sequential(nn.LayerNorm(obs_dim),
                                         init_(nn.Linear(obs_dim, n_embd), activate=True),
                                         nn.GELU()
                                         )

        self.ln = nn.LayerNorm(n_embd)
        self.blocks = nn.Sequential(*[EncodeBlock(n_embd, n_head, n_agent) for _ in range(n_block)])
        self.head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True),
                                  nn.GELU(),
                                  nn.LayerNorm(n_embd),
                                  init_(nn.Linear(n_embd, 1))
                                  )

    def forward(self, obs):
        # obs: (batch, n_agent, obs_dim)
        obs_embeddings = self.obs_encoder(obs)
        x = obs_embeddings

        rep = self.blocks(self.ln(x))
        v_loc = self.head(rep)

        return v_loc


class EncodeBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, n_agent):
        super(EncodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        # self.attn = SelfAttention(n_embd, n_head, n_agent, masked=True)
        self.attn = SelfAttention(n_embd, n_head, n_agent, masked=False)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * n_embd, n_embd))
        )

    def forward(self, x):
        x = self.ln1(x + self.attn(x, x, x))
        x = self.ln2(x + self.mlp(x))
        return x


class SelfAttention(nn.Module):

    def __init__(self, n_embd, n_head, n_agent, masked=False):
        super(SelfAttention, self).__init__()

        assert n_embd % n_head == 0
        self.masked = masked
        self.n_head = n_head
        # key, query, value projections for all heads
        self.key = init_(nn.Linear(n_embd, n_embd))
        self.query = init_(nn.Linear(n_embd, n_embd))
        self.value = init_(nn.Linear(n_embd, n_embd))
        # output projection
        self.proj = init_(nn.Linear(n_embd, n_embd))
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", th.tril(th.ones(n_agent + 1, n_agent + 1))
                             .view(1, 1, n_agent + 1, n_agent + 1)
                             )

        self.att_bp = None

    def forward(self, key, value, query):
        B, L, D = query.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(key).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        q = self.query(query).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        v = self.value(value).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)

        # causal attention: (B, nh, L, hs) x (B, nh, hs, L) -> (B, nh, L, L)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # self.att_bp = F.softmax(att, dim=-1)

        if self.masked:
            att = att.masked_fill(self.mask[:, :, :L, :L] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        y = att @ v  # (B, nh, L, L) x (B, nh, L, hs) -> (B, nh, L, hs)
        y = y.transpose(1, 2).contiguous().view(B, L, D)  # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)
        return y

 
def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module


def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)