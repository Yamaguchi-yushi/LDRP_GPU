import os
import numpy as np
import datetime
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import DataLoader, TensorDataset

import copy

class Buffer():
    def __init__(self, buffer_size, n_envs, obs_shape, action_dim, device, args=None):
        self.buffer_size = buffer_size
        self.n_envs = n_envs
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.device = device
        #self.pos = 0
        self.agent_num = args.agent_num if args else 1 
        self.reset_buffer()

    def reset_buffer(self):
        self.steps      = [[] for _ in range(self.n_envs)]
        self.states     = [[] for _ in range(self.n_envs)]
        self.actions    = [[] for _ in range(self.n_envs)]
        self.values     = [[] for _ in range(self.n_envs)]
        self.log_probs  = [[] for _ in range(self.n_envs)]
        self.masks      = [[] for _ in range(self.n_envs)]
        self.rewards    = [[] for _ in range(self.n_envs)]
        self.dones      = [[] for _ in range(self.n_envs)]
        self.returns    = [[] for _ in range(self.n_envs)]

    def n_samples(self):
        return sum(len(s) for s in self.steps)

    def add_actions(self, step_idx, state, action, log_prob, entropy, value,
                    mask=None, env_idx=0):
        if self.n_samples() >= self.buffer_size:
            return                                  
        self.steps[env_idx].append(step_idx)
        self.states[env_idx].append(state)
        self.actions[env_idx].append(action)
        self.log_probs[env_idx].append(log_prob)
        self.values[env_idx].append(value)
        self.masks[env_idx].append(
            mask.detach() if mask is not None
            else torch.zeros(self.action_dim, device=self.device))

    def add_rewards(self, reward, done, env_idx=0):
        self.rewards[env_idx].append(reward)
        self.dones[env_idx].append(done)

    def compute_returns(self, gamma=0.99):
        """各 env で return-to-go を計算し, 行動を打ったステップの値を割り当てる.

        step_idx は「その決定の直後に実行されるステップの 0 始まり index」で,
        rewards[env_idx][step_idx] がその報酬になる。
        """
        for e in range(self.n_envs):
            rews, dones = self.rewards[e], self.dones[e]
            T = len(rews)
            rtg, R = [0.0] * T, 0.0
            for t in reversed(range(T)):
                if dones[t]:
                    R = 0.0                         # エピソード境界で割引を切る
                R = rews[t] + gamma * R
                rtg[t] = R
            # まだ return が付いていない行動だけを処理する。steps は update() まで
            # 積み上がり続けるので, 全体を舐めると前エピソードの行動に return が
            # 二重に付き, 行動数と return 数がずれる
            for s in self.steps[e][len(self.returns[e]):]:
                # 収集途中で打ち切られた場合に備えて index を範囲内に丸める
                self.returns[e].append(rtg[min(s, T - 1)] if T > 0 else 0.0)
            self.rewards[e], self.dones[e] = [], []

    def get_tensors(self, device):
        """env 別ストリームを 1 本に連結してから学習用テンソルにする."""
        flat = lambda xs: [v for e in range(self.n_envs) for v in xs[e]]
        states    = torch.stack(flat(self.states)).to(device)
        actions   = torch.tensor(flat(self.actions)).to(device)
        log_probs = torch.stack(flat(self.log_probs)).detach().to(device)
        values    = torch.stack(flat(self.values)).detach().to(device).squeeze()
        returns   = torch.tensor(flat(self.returns), dtype=torch.float32).to(device)
        masks     = torch.stack(flat(self.masks)).to(device)
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return states, actions, log_probs, returns, advantages, masks


class PPO(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PPO, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.policy_layer = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

        self.value_layer = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.apply(self.orthogonal_init)

    def forward(self, x):
        policy = self.policy_layer(x)
        value = self.value_layer(x)
        return policy, value
    
    def orthogonal_init(self,m):
        if isinstance(m, nn.Linear):
            init.orthogonal_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)

    def save_model(self, path, extra=None):
        """ネットワーク重みを path に保存する.

        extra に dict を渡すと payload にマージされる (optimizer state 等を
        PPOAgent 側から差し込むため. ファイル書き込みは 1 回で済ませる).
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        payload = {
            "model_state_dict": self.state_dict(),
            "input_dim": self.input_dim,    
            "output_dim": self.output_dim,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    def load_model(self, path):
        """save_model で書いた payload を読み，重みを自身へ流し込んで payload を返す.

        戻り値の payload は，optimizer state 等を PPOAgent 側で復元するために使う.
        """
        payload = torch.load(path, map_location="cpu")

        if payload.get("input_dim") != self.input_dim or payload.get("output_dim") != self.output_dim:
            raise ValueError(f"Model dimensions mismatch: {payload.get('input_dim')}x{payload.get('output_dim')} vs {self.input_dim}x{self.output_dim}")
        self.load_state_dict(payload["model_state_dict"])
        return payload

class PPOAgent():
    def __init__(self, args):
        input_dim = args.agent_num * args.node_num * 2 + args.task_num * args.node_num + args.agent_num 
        output_dim = args.task_num * args.agent_num 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PPO(input_dim, output_dim).to(self.device)
        self.buffer = Buffer(args.buffer_size, args.n_envs, (input_dim,), output_dim, self.device, args)
        self.args = args
        self.test_mode = True
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate)
        self.actor_optimizer = torch.optim.Adam(self.model.policy_layer.parameters(), lr=args.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.model.value_layer.parameters(), lr=args.learning_rate)

        # save and load model
        self.update_count = 0
        self.total_steps = 0
        self._last_saved_step = 0

        self.save_model_flag = bool(getattr(args, "save_ppo_task_model", False))
        self.save_interval = int(getattr(args, "ppo_task_save_interval", 500_000))
        self.results_path = getattr(args, "local_results_path", "results")

        name = f"ppo_task_{getattr(args, 'path_planner', 'unknown')}"
        seed = getattr(args, "seed", 0)
        env_key = getattr(args, "env_name", "unknown")
        self.unique_token = f"{name}_seed{seed}_env{env_key}_{datetime.datetime.now()}"

        ckpt = getattr(args, "ppo_task_checkpoint_path", "")
        if ckpt:
            self.load_models(self._resolve_checkpoint_dir(ckpt))

    def update(self):
        # Implement the PPO update logic here
        states, actions, log_probs, returns, advantages, masks = self.buffer.get_tensors(self.device)
        dataset = TensorDataset(states, actions, log_probs, returns, advantages, masks)
        loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)

        sums = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0,
                "approx_kl": 0.0, "clip_frac": 0.0}

        n_batched = 0

        for _ in range(self.args.epochs):
            for batch in loader:
                s, a, old_logp, r, adv, m = batch
                logits, values = self.model(s)
                # 行動選択時と同じマスクを当てて分布を揃える。
                # これが無いと収集時 (マスク後) と更新時 (生 logits) で別分布になり,
                # 重要度比が意味を成さなくなる
                logits = logits.masked_fill(m == 1, float("-inf"))
                dist = Categorical(logits=logits)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(a)

                ratio = (new_log_probs - old_logp).exp()
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.args.clip_epsilon, 1 + self.args.clip_epsilon) * adv
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = (r - values.squeeze(-1)).pow(2).mean()

                with torch.no_grad():
                    log_ratio = new_log_probs - old_logp
                    approx_kl = ((log_ratio.exp() - 1) - log_ratio).mean()
                    clip_frac = ((ratio - 1.0).abs() > self.args.clip_epsilon).float().mean()

                sums["actor_loss"] += float(actor_loss)
                sums["critic_loss"] += float(critic_loss)
                sums["entropy"] += float(entropy)
                sums["approx_kl"] += float(approx_kl)
                sums["clip_frac"] += float(clip_frac)
                n_batched += 1

                actor_loss -= self.args.entropy_coef * entropy
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

        with torch.no_grad():
            _, v_all = self.model(states)
            v_all = v_all.squeeze(-1)
            explained_var = 1.0 - (returns - v_all).var() / (returns.var() + 1e-8)

        n = max(1, n_batched)  # avoid div by zero
        stats = {k: v / n for k, v in sums.items()}
        stats["explained_variance"] = float(explained_var)
        stats["adv_std"] = float(advantages.std())
        stats["n_samples"] = float(states.shape[0])

        self.update_count += 1
        self.buffer_reset()

        return stats

    def assign_task(self, env):
        return self._assign(env, env_idx=0, step_idx=env.step_account,
                            test_mode=self.test_mode)

    def assign_task_from_state(self, task_state, env_idx, step_idx, test_mode):
        from types import SimpleNamespace
        if task_state is None:
            return [-1] * self.args.agent_num
        return self._assign(SimpleNamespace(**task_state), env_idx=env_idx, step_idx=step_idx,
                            test_mode=test_mode)
    

    def _assign(self, env, env_idx, step_idx, test_mode):
        current_tasklist = copy.deepcopy(env.current_tasklist)  
        assigned_tasklist = copy.deepcopy(env.assigned_tasks)
        assigned_list = copy.deepcopy(env.assigned_list)

        agent_num = env.n_agents
        task_num = env.task_num

        task_assign = [-1 for _ in range(agent_num)]  # -1で初期化


        len_current_task = sum(
                    1 for j in range(len(current_tasklist))
                    if j >= len(assigned_list) or assigned_list[j] == -1
                )
        for _ in range(agent_num):
            if not any(len(t) == 0 for t in assigned_tasklist):
                break
            if len_current_task <= 0:
                break

            state = self.create_state(env, current_tasklist, assigned_tasklist)
            state = state.clone().detach().to(self.device)
            policy, value = self.model(state)
            #マスク
            #task持ちのエージェント
            mask = torch.zeros_like(policy)
            for k in range(agent_num):
                if len(assigned_tasklist[k]) > 0 or (
                    getattr(env, "use_dynamic_agents", False) and (not env.active[k] or env.pending_off[k])):
                    mask[task_num * k:task_num * (k + 1)] = 1
            #task数が少ないとき
            for j in range(env.task_num):
                if j > len(current_tasklist) - 1:
                    mask[[task_num * k + j for k in range(agent_num)]] = 1
            #使用済みのtask
            for j in range(len(current_tasklist)):
                taken = (current_tasklist[j][0] == -1) or (j < len(assigned_list) and assigned_list[j] != -1)
                if taken:
                    mask[[task_num * k + j for k in range(agent_num)]] = 1

            if bool((mask == 1).all()):
                break

            policy = policy.masked_fill(mask.bool(), float('-inf'))
            policy = F.softmax(policy, dim=-1)

            if test_mode:# 実行用
                action = policy.argmax().item()
            else:# 学習用
                dist = Categorical(policy)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                entropy = dist.entropy()
                action = action.item()

                # mask を渡して update() 側で同じ分布を再構成できるようにする
                self.buffer.add_actions(
                    step_idx, state, action, log_prob, entropy, value, mask,
                    env_idx=env_idx
                )
        
            #actionをみてtask_assignを決定
            #currentとassignedを更新
            q, r = divmod(action, env.task_num)
            assigned_tasklist[q] = list(current_tasklist[r])  # エージェントqにタスクrを割り当て
            task_assign[q] = r
            if r < len(assigned_list):
                assigned_list[r] = q  # タスクrがエージェントqに割り当てられたことを記録
            current_tasklist[r][0] = -1  # タスクを割り当てたので、タスクリストから削除
            len_current_task -= 1

        return task_assign
    
    def create_state(self, env, current_tasklist, assigned_tasklist):
        current_tasklist = copy.deepcopy(current_tasklist)#[[4,2,-1],[1,5,-1]][s,g,time]->s,gのonehotに
        assigned_tasklist = copy.deepcopy(assigned_tasklist)#[[4,2,-1]]->エージェントごとのonehotに
        onehot_obs = copy.deepcopy(env.obs_onehot)
        tensor_list = [torch.tensor(lst, dtype=torch.float32) for lst in onehot_obs]
        state = torch.cat(tensor_list, dim=0)

        #current_tasklistのonehot化
        for _ in range(env.task_num):
            task_tensor = torch.zeros(env.n_nodes, dtype=torch.float32)
            if len(current_tasklist) > 0:
                task = current_tasklist.pop(0)
                if task[0] != -1:
                    task_tensor[task[0]] = 1
                    task_tensor[task[1]] = -1 
                else:
                    pass

            state = torch.cat((state, task_tensor), dim=0)
        #assignedされているエージェントを可視化
        assigned = []
        for i in range(env.n_agents):
            if len(assigned_tasklist[i]) > 0:
                assigned.append(1)
            else:
                assigned.append(0)
        assigned_tensor = torch.tensor(assigned, dtype=torch.float32)
        state = torch.cat((state, assigned_tensor), dim=0)

        return state
        
    def update_ready(self):
        # steps は env 別の入れ子リストなので len() では env 本数になる。
        # 全 env 合計のサンプル数で判定する
        return self.buffer.n_samples() >= self.buffer.buffer_size

    def buffer_add_rewards(self, reward, done, env_idx=0):
        self.buffer.add_rewards(reward, done, env_idx=env_idx)

    def buffer_reset(self):
        self.buffer.reset_buffer()

    def set_test_mode(self, mode_tf):
        self.test_mode = mode_tf

    #エピソード終了時の処理
    def process_end_episode(self):
        self.buffer.compute_returns(self.args.gamma)

    # モデルの保存先
    def _run_dir(self):
        # 保存ルート： results/models/{unique_token}/
        return os.path.join(self.results_path, "models", self.unique_token)

    def _resolve_checkpoint_dir(self, ckpt_path):
        if not os.path.isdir(ckpt_path):
            raise ValueError(f"PPO Task checkpoint dir not found: {ckpt_path}")
        if os.path.exists(os.path.join(ckpt_path, "agent.th")):
            return ckpt_path
        steps = [int(n) for n in os.listdir(ckpt_path) 
                 if n.isdigit() and os.path.isdir(os.path.join(ckpt_path, n))]
        if not steps:
            raise ValueError(f"No valid checkpoint directories found in: {ckpt_path}")
        want = int(getattr(self.args, "ppo_task_load_step", 0))
        chosen = max(steps) if want == 0 else min(steps, key=lambda x: abs(x - want))
        step_dir = os.path.join(ckpt_path, str(chosen))
        task_dir = os.path.join(step_dir, "task")
        return task_dir if os.path.exists(os.path.join(task_dir, "agent.th")) else step_dir

    # saveとload
    def set_total_steps(self, total_steps):
        self.total_steps = int(total_steps)

    def save_models(self, path):
        """
        pathディレクトリに agent.th と opt.th を保存する. 
        """
        os.makedirs(path, exist_ok=True)
        self.model.save_model(os.path.join(path, "agent.th"), extra={
            "update_count": self.update_count,
            "total_steps": self.total_steps,
            "agent_num": getattr(self.args, "agent_num", None),
            "task_num": getattr(self.args, "task_num", None),
            "node_num": getattr(self.args, "node_num", None),
            "map_name": getattr(self.args, "map_name", None),
            "path_planner": getattr(self.args, "path_planner", None),
        })
        torch.save({"actor": self.actor_optimizer.state_dict(),
                    "critic": self.critic_optimizer.state_dict()},
                    os.path.join(path, "opt.th"))
        print(f"[PPOAgent] Saved model and optimizer state to {path}")
        return path

    def load_models(self, path, load_optimizer=True):
        """path から復元する"""
        payload = self.model.load_model(os.path.join(path, "agent.th"))
        self.model.to(self.device)
        opt_path = os.path.join(path, "opt.th")

        if load_optimizer and os.path.exists(opt_path):
            opt = torch.load(opt_path, map_location="cpu")
            self.actor_optimizer.load_state_dict(opt["actor"])
            self.critic_optimizer.load_state_dict(opt["critic"])
        self.update_count = int(payload.get("update_count", 0))
        self.total_steps = int(payload.get("total_steps", 0))
        self._last_saved_steps = self.total_steps
        print(f"[PPO-Task] Loading model from {path}")
        return payload

    def maybe_save_models(self, total_steps):
        """total_steps が save_freq_steps を超えたらモデルを保存する"""
        self.total_steps = int(total_steps)
        if not self.save_model_flag:
            return None
        if self._last_saved_step != 0 and \
              (self.total_steps - self._last_saved_step) < self.save_interval:
                return None
        try:
            path = self.save_models(
                os.path.join(self._run_dir(), str(self.total_steps), "task"))
            self._last_saved_step = self.total_steps
            return path
        except Exception as e:
            print(f"[PPO-Task] Failed to save model at step {self.total_steps}: {e}")
            return None 
