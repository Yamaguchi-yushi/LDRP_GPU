import torch 
import numpy as np

#from epymarl.src.modules.agents.rnn_agent import RNNAgent  # EPyMARLそのまま使う
from .rnn_agent import RNNAgent  # ローカルのRNNAgentを使用


class DummyArgs:
    def __init__(self, hidden_dim=64, n_actions=None, use_rnn=False):
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions
        self.use_rnn = use_rnn


class PolicyRunner:
    def __init__(self, model_path, input_shape, n_actions, agent_num):
        # --- 1. Pre-load the checkpoint and auto-detect the training architecture ---
        state_dict = torch.load(model_path, map_location='cpu')

        # Detect use_rnn: True if GRUCell
        ckpt_use_rnn = any(k.startswith('rnn.weight_ih') for k in state_dict.keys())

        if "fc1.weight" not in state_dict:
            raise ValueError(f"[PolicyRunner]{model_path}: 'fc1.weight' not found"
                             f"(checkpoint is not RNNAgent compatible)")
        ckpt_input_shape = state_dict["fc1.weight"].shape[1]

        # --- 2. Compute the obs-width gap and decide whether padding is needed ---
        self.agent_num = agent_num
        self.input_diff = ckpt_input_shape - input_shape

        if self.input_diff < 0:
            raise ValueError(f"[PolicyRunner] checkpoint input_shape={ckpt_input_shape} < "
                             f"env input_shape={input_shape}. Trimming the obs is not supported.")
        if self.input_diff > 0 and self.input_diff != agent_num:
            raise ValueError(f"[PolicyRunner] input dim mismatch: "
                             f"ckpt={ckpt_input_shape}, env={input_shape}, diff={self.input_diff} "
                             f"(does not match agent_num={agent_num}). "
                             f"Check obs_agent_id / obs_last_action used at training time."
            )
        
        # --- 3. Rebuild RNNAgent with the detected architecture and load weights ---
        self.args = DummyArgs(hidden_dim=64, n_actions=n_actions, use_rnn=ckpt_use_rnn)
        self.agent = RNNAgent(ckpt_input_shape, self.args)
        self.agent.load_state_dict(state_dict)
        self.agent.eval()
        self.hidden_states = [self.agent.init_hidden() for _ in range(agent_num)]

        pad_info = (f"+agent_id onehot (+{self.input_diff})" if self.input_diff>0 else "no padding")
        print(f"[PolicyRunner] Loaded {model_path} | input_shape={input_shape} | "
                f"ckpt_input_shape={ckpt_input_shape} | {pad_info} | use_rnn={ckpt_use_rnn}")

    def get_action(self, ag_idx, obs, avail_actions):
         # Pad the obs with agent_id one-hot encoding
        if self.input_diff > 0:
            agent_id_onehot = np.zeros(self.agent_num, dtype=np.float32)
            agent_id_onehot[ag_idx] = 1.0
            obs = np.concatenate([np.asarray(obs, dtype=np.float32), agent_id_onehot])

        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        h_in = self.hidden_states[ag_idx]

        q_values, h_out = self.agent(obs_tensor, h_in)
        self.hidden_states[ag_idx] = h_out.detach()  # detachでグラフ切って次に備える

        q_numpy = q_values.squeeze(0).detach().numpy()
        masked_q = [q_numpy[a] if a in avail_actions else -np.inf for a in range(len(q_numpy))]
        # if agents act bad behavior, give the agents the second-good action from masked_q
        #
        #
        #####################################

        return int(np.argmax(masked_q))