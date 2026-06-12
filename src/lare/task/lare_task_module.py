"""LaRe-Task main module (System B).

Mirrors LaRePathModule but operates at the *assignment-decision* granularity:
  - decisions are sparse (K_episode << T_episode)
  - the training target R_task is the number of completed tasks at episode end
  - the decoder is small (K is small)

Three operating modes (matching LaRe-Path):
  1. use_lare_task=False                                 -> baseline (no LaRe code path)
  2. use_lare_task=True, frozen=False                    -> train decoder online
  3. use_lare_task=True, frozen=True (pretrained loaded) -> use loaded decoder, no training
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import nn

from .buffer import TaskEpisodeBuffer
from .decoder import TaskRewardDecoder
from .encoder import (
    FACTOR_NUMBER,
    build_assignment_state,
    build_env_info,
    factors_as_array,
)


@dataclass
class LaReTaskConfig:
    factor_dim: int = FACTOR_NUMBER
    decoder_hidden_dim: int = 64
    decoder_n_layers: int = 2
    buffer_capacity: int = 512
    min_buffer: int = 32
    update_freq: int = 16
    batch_size: int = 32
    learning_rate: float = 5e-4
    # MARL4DRP の evaluation_period 方式に整合 (Path 側と対称).
    # update_freq ごとに評価期間を発火し、train_epochs 個の連続エピソード末尾で
    # fresh batch を 1 step ずつ流す.
    train_epochs: int = 50
    use_lare_training: bool = True
    frozen: bool = False
    autosave_path: Optional[object] = None  # str or zero-arg callable
    # 保存間引きしきい値. 前回保存時の累積環境ステップから save_freq_steps 以上
    # 進んだ時だけ保存. 0 にすると毎更新ごとに保存.
    save_freq_steps: int = 500_000
    device: str = "auto"  # "auto", "cpu", or "cuda"


class LaReTaskModule:
    def __init__(self, env, config: Optional[LaReTaskConfig] = None, graph_diameter: Optional[float] = None):
        self.env = env
        self.cfg = config if config is not None else LaReTaskConfig()
        self.factor_dim = int(self.cfg.factor_dim)
        self.n_agents = int(env.agent_num)

        if self.cfg.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.cfg.device)

        self.graph_diameter = float(graph_diameter) if graph_diameter is not None else self._compute_graph_diameter()

        self.decoder = TaskRewardDecoder(
            factor_dim=self.factor_dim,
            hidden_dim=self.cfg.decoder_hidden_dim,
            n_layers=self.cfg.decoder_n_layers,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.decoder.parameters(), lr=self.cfg.learning_rate, weight_decay=1e-5)
        self.loss_fn = nn.MSELoss(reduction="mean")

        self.buffer = TaskEpisodeBuffer(capacity=self.cfg.buffer_capacity, factor_dim=self.factor_dim)

        self.is_trained = False
        self.is_pretrained = False
        self.frozen = bool(self.cfg.frozen)
        self.use_lare_training = bool(self.cfg.use_lare_training)

        self.episode_count = 0
        self.update_count = 0
        self.last_loss = None
        # save_freq_steps の throttle 用. 0 で初期化 → 最初の保存は
        # save_freq_steps を超えた時点. 0 step ではセーブしない.
        self._last_saved_step = 0

        # MARL4DRP の evaluation_period 状態機械 (Path 側と対称).
        self._evaluation_active = False
        self._evaluation_count = 0

        # Per-step bookkeeping (filled by record_step_assignments).
        self._last_step_proxy = 0.0
        self._step_assignment_count = 0

    def _compute_graph_diameter(self):
        # Local import keeps task module independent of path module's import chain
        # while still reusing the same routine.
        try:
            from src.lare.path.encoder import compute_graph_diameter
            return compute_graph_diameter(self.env)
        except Exception:
            return 100.0

    # -------------------- per-step API --------------------

    def record_step_assignments(self, env, decisions):
        """Record the assignment decisions made *this env step*.

        decisions: list of dicts, each one describing a single assignment with keys:
          agent_id, pickup, dropoff, agent_prev_goal, agent_was_idle, wait_steps,
          unassigned_after, agent_loads_after, n_assignments_step
        Returns the proxy reward for this step (a scalar). 0.0 when no decisions were made
        OR when the decoder has not been trained yet.
        """
        self._step_assignment_count = len(decisions)
        if not decisions:
            self._last_step_proxy = 0.0
            return self._last_step_proxy

        factor_vectors = []
        for d in decisions:
            env_info = build_env_info(
                env=env,
                graph_diameter=self.graph_diameter,
                agent_id=d["agent_id"],
                pickup_node=d["pickup"],
                dropoff_node=d["dropoff"],
                agent_prev_goal=d.get("agent_prev_goal"),
            )
            assign_state = build_assignment_state(
                env=env,
                graph_diameter=self.graph_diameter,
                agent_id=d["agent_id"],
                pickup_node=d["pickup"],
                agent_loads_after=d["agent_loads_after"],
                wait_steps=d["wait_steps"],
                agent_was_idle=d["agent_was_idle"],
                unassigned_after=d["unassigned_after"],
                n_assignments_step=d["n_assignments_step"],
            )
            f = factors_as_array(assign_state, env_info)
            factor_vectors.append(f)
            self.buffer.add_decision(f)

        if self.is_trained:
            with torch.no_grad():
                self.decoder.eval()
                z = torch.from_numpy(np.stack(factor_vectors, axis=0)).float().to(self.device)
                r_hat = self.decoder(z).squeeze(-1).cpu().numpy()
            proxy = float(np.sum(r_hat))
        else:
            proxy = 0.0
        self._last_step_proxy = proxy
        return proxy

    def consume_step_proxy_reward(self):
        """Return the most recent step's proxy reward; resets to 0 after read."""
        v = self._last_step_proxy
        self._last_step_proxy = 0.0
        return v

    def end_episode(self, r_task):
        appended = self.buffer.end_episode(r_task)
        if not appended:
            return
        self.episode_count += 1

        if self.frozen:
            return

        # MARL4DRP の評価期間方式 (Path 側と対称).
        if not self._evaluation_active:
            if (
                len(self.buffer) >= self.cfg.min_buffer
                and self.episode_count % max(1, self.cfg.update_freq) == 0
            ):
                self._evaluation_active = True
                self._evaluation_count = 0

        if self._evaluation_active:
            self._update()
            self._evaluation_count += 1
            if self._evaluation_count >= max(1, self.cfg.train_epochs):
                self._evaluation_active = False
                self._evaluation_count = 0

    # -------------------- training --------------------

    def _update(self):
        if self.frozen:
            return
        sample = self.buffer.sample_batch(self.cfg.batch_size)
        if sample is None:
            return
        factors_np, ks_np, returns_np = sample
        factors = torch.from_numpy(factors_np).to(self.device)
        ks = torch.from_numpy(ks_np).to(self.device)
        returns = torch.from_numpy(returns_np).to(self.device)

        b, max_k, _ = factors.shape
        idx = torch.arange(max_k, device=self.device)[None, :]
        mask = (idx < ks[:, None]).float()

        # 1 batch sample + 1 optimizer step. 連続更新は end_episode 側の
        # 評価期間ループが管理する (train_epochs ep で本関数が連続呼出).
        self.decoder.train()
        r_hat = self.decoder(factors).squeeze(-1)  # (b, max_k)
        r_hat_masked = r_hat * mask
        pred_return = r_hat_masked.sum(dim=1)
        loss = self.loss_fn(pred_return, returns)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.is_trained = True
        self.update_count += 1
        self.last_loss = float(loss.detach().cpu().item())

        if self.cfg.autosave_path:
            try:
                # 保存頻度 throttle (詳細は Path 側と同じ).
                current_step = int(getattr(self.env, "_lare_total_step_account", 0))
                freq = int(max(0, self.cfg.save_freq_steps))
                if freq == 0 or (current_step - self._last_saved_step) >= freq:
                    target = self.cfg.autosave_path
                    if callable(target):
                        target = target()
                    self.save_model(target)
                    self._last_saved_step = current_step
            except Exception as e:
                print(f"[LaRe-Task] autosave failed: {e}")

    # -------------------- save / load --------------------

    def save_model(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        torch.save({
            "decoder_state_dict": self.decoder.state_dict(),
            "factor_dim": self.factor_dim,
            "n_agents": self.n_agents,
            "update_count": self.update_count,
            "last_loss": self.last_loss,
        }, path)

    def load_model(self, path, freeze=True):
        if not os.path.exists(path):
            raise FileNotFoundError(f"LaRe-Task model file not found: {path}")
        payload = torch.load(path, map_location=self.device)
        saved = int(payload.get("factor_dim", self.factor_dim))
        if saved != self.factor_dim:
            raise ValueError(f"factor_dim mismatch: model={saved}, current={self.factor_dim}")
        self.decoder.load_state_dict(payload["decoder_state_dict"])
        self.decoder.eval()

        self.is_trained = True
        self.is_pretrained = True
        self.frozen = bool(freeze)
        self.update_count = int(payload.get("update_count", 0))
        self.last_loss = payload.get("last_loss", None)
