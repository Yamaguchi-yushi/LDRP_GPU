"""LaRe-Path main module: ties encoder, decoder, optional transformer, and buffer.

Usage from the env:
  module = LaRePathModule(env, config)
  on each env.step():
      factors, proxy_rewards = module.step(prev_onehot_position, current_colliding_pairs)
      module.record_step(factors, env_reward_sum=sum(ri_array))
      ...
      if done: module.end_episode()  # may trigger a decoder update

When the decoder has not been trained yet, `proxy_rewards` is None and the env
should fall back to the original reward.

Three operating modes:
  1. use_lare_path=False                                 -> baseline (no LaRe code path)
  2. use_lare_path=True, frozen=False                    -> train decoder online; proxy after first update
  3. use_lare_path=True, frozen=True (pretrained loaded) -> use loaded decoder, no training
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import nn

from .encoder import (
    FACTOR_NUMBER,
    build_lare_obs_for_agent,
    compute_graph_diameter,
    evaluation_func,
    precompute_edge_info,
)
from .decoder import PathRewardDecoder
from .transformer import TimeAgentTransformer


@dataclass
class LaRePathConfig:
    factor_dim: int = FACTOR_NUMBER
    decoder_hidden_dim: int = 64
    decoder_n_layers: int = 3
    use_transformer: bool = False
    transformer_heads: int = 4
    transformer_depth: int = 2
    transformer_seq_length: int = 100
    buffer_capacity: int = 512
    seq_length: int = 100
    min_buffer: int = 64
    update_freq: int = 32
    batch_size: int = 32
    learning_rate: float = 5e-4
    # MARL4DRP の evaluation_period 方式に整合: update_freq (= 128) ごとに発火する
    # 評価期間中、連続する train_epochs 個のエピソード末尾で fresh batch をサンプル
    # して 1 step ずつ更新する (= MARL4DRP の evaluation_episodes=50 と同じ).
    train_epochs: int = 50
    use_lare_training: bool = True
    # Pretrained-model mode. When frozen=True the decoder is used in eval-only mode
    # and end_episode() will NOT trigger optimizer updates.
    frozen: bool = False
    # Auto-save the decoder after each successful update (training mode only).
    # Either a fixed string path, or a zero-arg callable returning a path
    # (callable form lets the caller embed mutable state like step counts).
    autosave_path: Optional[object] = None
    # 保存を間引くしきい値: 前回保存時の累積ステップから save_freq_steps 以上進んだ
    # ときだけ実際にファイルへ書き出す. デコーダ学習自体は update_freq 通り走る
    # ので、学習頻度 ≠ 保存頻度 を独立に制御できる. 0 にすると毎更新ごとに保存.
    save_freq_steps: int = 500_000
    device: str = "auto"


class LaRePathModule:
    def __init__(self, env, config: Optional[LaRePathConfig] = None):
        self.env = env
        self.cfg = config if config is not None else LaRePathConfig()

        self.edge_info_cache = precompute_edge_info(env)
        self.graph_diameter = compute_graph_diameter(env)

        self.factor_dim = self.cfg.factor_dim
        self.n_agents = env.agent_num
        self.seq_length = self.cfg.seq_length

        if self.cfg.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.cfg.device)

        self.decoder = PathRewardDecoder(
            factor_dim=self.factor_dim,
            hidden_dim=self.cfg.decoder_hidden_dim,
            n_layers=self.cfg.decoder_n_layers,
        ).to(self.device)

        self.use_transformer = bool(self.cfg.use_transformer)
        if self.use_transformer:
            self.transformer = TimeAgentTransformer(
                emb=self.factor_dim,
                heads=self.cfg.transformer_heads,
                depth=self.cfg.transformer_depth,
                seq_length=self.cfg.transformer_seq_length,
                n_agents=self.n_agents,
                agent=True,
            ).to(self.device)
            params = list(self.decoder.parameters()) + list(self.transformer.parameters())
        else:
            self.transformer = None
            params = list(self.decoder.parameters())

        self.optimizer = torch.optim.Adam(params, lr=self.cfg.learning_rate, weight_decay=1e-5)
        self.loss_fn = nn.MSELoss(reduction="mean")

        from .buffer import PathEpisodeBuffer
        self.buffer = PathEpisodeBuffer(
            capacity=self.cfg.buffer_capacity,
            seq_length=self.seq_length,
            n_agents=self.n_agents,
            factor_dim=self.factor_dim,
        )

        self.is_trained = False
        self.is_pretrained = False
        self.episode_count = 0
        self.update_count = 0
        self.last_loss = None
        self.use_lare_training = bool(self.cfg.use_lare_training)
        self.frozen = bool(self.cfg.frozen)
        # save_freq_steps での保存間引き用. 前回保存時の累積環境ステップ.
        # 0 で初期化 → 最初の保存は save_freq_steps (= デフォ 0.5M) を超えた時点.
        # 0 step ではセーブしない.
        self._last_saved_step = 0

        # MARL4DRP の evaluation_period 状態機械 (drp_env.py:1340-1382 相当).
        # update_freq ごとに True にセット → 連続する train_epochs ep で 1 step ずつ
        # 更新 → カウンタが train_epochs に達したら False に戻す.
        self._evaluation_active = False
        self._evaluation_count = 0

    def compute_factors(self, prev_onehot_position, current_colliding_pairs):
        """Compute the (n_agents, factor_dim) factor matrix for the current step.

        prev_onehot_position: (n_agents, n_nodes) array of pre-step positions.
        current_colliding_pairs: list of agent-id pairs that collided this step (or None).
        """
        rows = []
        for i in range(self.n_agents):
            obs_row = build_lare_obs_for_agent(
                self.env, i, self.edge_info_cache, self.graph_diameter,
                prev_onehot_position, current_colliding_pairs,
            )
            rows.append(obs_row)
        batch_obs = np.stack(rows, axis=0)
        factor_list = evaluation_func(batch_obs)
        factor_arr = np.concatenate(factor_list, axis=-1).astype(np.float32)
        return factor_arr

    def proxy_rewards(self, factors_per_agent):
        """Decoder forward for one step. Returns shape (n_agents,) numpy array.

        Returns None if the decoder has not been trained yet.
        """
        if not self.is_trained:
            return None
        with torch.no_grad():
            self.decoder.eval()
            z = torch.from_numpy(factors_per_agent).float().to(self.device)
            r_hat = self.decoder(z).squeeze(-1)
        return r_hat.detach().cpu().numpy()

    def record_step(self, factors_per_agent, env_reward_sum):
        self.buffer.add_step(factors_per_agent, env_reward_sum)

    def end_episode(self):
        self.buffer.end_episode()
        self.episode_count += 1

        # Frozen / pretrained mode: never update the decoder.
        if self.frozen:
            return

        # MARL4DRP の評価期間方式 (drp_env.py:1340-1382 相当):
        #   1) update_freq ごとに評価期間を開始 (バッファが十分に溜まっていれば)
        #   2) 評価期間中は毎エピソード末尾で fresh batch を 1 回 _update() する
        #   3) train_epochs 個のエピソードを消化したら評価期間を閉じる
        # トリガと最初の update は同じ呼び出しで起きる (= 128 ep 目も update が走る).
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

    def _update(self):
        if self.frozen:
            return
        sample = self.buffer.sample_batch(self.cfg.batch_size)
        if sample is None:
            return
        factors_np, lengths_np, returns_np = sample

        factors = torch.from_numpy(factors_np).to(self.device)
        lengths = torch.from_numpy(lengths_np).to(self.device)
        returns = torch.from_numpy(returns_np).to(self.device)

        b, n_a, t, _ = factors.shape
        time_idx = torch.arange(t, device=self.device)[None, :]
        mask = (time_idx < lengths[:, None]).float()
        mask = mask.unsqueeze(1).expand(b, n_a, t)

        # MARL4DRP の train_step に対応: 1 batch sample + 1 optimizer step.
        # 連続更新回数 (= MARL4DRP の evaluation_episodes) は end_episode 側の
        # 評価期間ループが管理する (train_epochs 個のエピソードで本関数が連続呼出).
        self.decoder.train()
        if self.transformer is not None:
            self.transformer.train()
            z = self.transformer(factors).unsqueeze(-1)
            r_hat = self.decoder(z.squeeze(-1).unsqueeze(-1).expand(b, n_a, t, self.factor_dim))
        else:
            r_hat = self.decoder(factors)
        r_hat = r_hat.squeeze(-1)
        r_hat_masked = r_hat * mask
        pred_return = r_hat_masked.sum(dim=[1, 2])

        loss = self.loss_fn(pred_return, returns)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.is_trained = True
        self.update_count += 1
        self.last_loss = float(loss.detach().cpu().item())

        if self.cfg.autosave_path:
            try:
                # 保存頻度の throttle: 前回保存時の累積ステップから save_freq_steps
                # 以上進んでいる時だけ実際に保存. デコーダ更新自体は update_freq
                # 通り続行するので学習頻度には影響しない.
                current_step = int(getattr(self.env, "_lare_total_step_account", 0))
                freq = int(max(0, self.cfg.save_freq_steps))
                if freq == 0 or (current_step - self._last_saved_step) >= freq:
                    target = self.cfg.autosave_path
                    if callable(target):
                        target = target()
                    self.save_model(target)
                    self._last_saved_step = current_step
            except Exception as e:
                print(f"[LaRe-Path] autosave failed: {e}")

    def save_model(self, path):
        """Persist decoder (and optional transformer) weights + minimal metadata."""
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        payload = {
            "decoder_state_dict": self.decoder.state_dict(),
            "transformer_state_dict": (
                self.transformer.state_dict() if self.transformer is not None else None
            ),
            "factor_dim": self.factor_dim,
            "n_agents": self.n_agents,
            "use_transformer": self.use_transformer,
            "update_count": self.update_count,
            "last_loss": self.last_loss,
        }
        torch.save(payload, path)

    def load_model(self, path, freeze=True):
        """Load decoder weights from disk. With freeze=True (default) further
        training is disabled and the proxy reward is used immediately."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"LaRe-Path model file not found: {path}")
        payload = torch.load(path, map_location=self.device)

        if "decoder_state_dict" in payload:
            # -- LDRP format --
            saved_factor_dim = int(payload.get("factor_dim", self.factor_dim))
            if saved_factor_dim != self.factor_dim:
                raise ValueError(
                    f"factor_dim mismatch: model={saved_factor_dim}, current={self.factor_dim}"
                )
            decoder_sd = payload["decoder_state_dict"]
            transformer_sd = payload.get("transformer_state_dict")
            update_count = int(payload.get("update_count", 0))
            last_loss = payload.get("last_loss", None)
            source = "LDRP"
        elif "model_state_dict" in payload:
            # -- MARL4DRP format --
            decoder_sd = self._convert_marl4drp_state_dict(payload["model_state_dict"])
            transformer_sd = None
            update_count = int(payload.get("total_training_steps", 0))
            last_loss = None
            source = "MARL4DRP"
        else:
            raise KeyError("Unrecognized model file format: missing expected keys")
        
        self.decoder.load_state_dict(decoder_sd)
        self.decoder.eval()
        if self.transformer is not None and transformer_sd is not None:
            self.transformer.load_state_dict(transformer_sd)
            self.transformer.eval()

        self.is_trained = True
        self.is_pretrained = True
        self.frozen = bool(freeze)
        self.update_count = update_count
        self.last_loss = last_loss
        print(f"[LaRe-Path] Loaded weights from {source} format ({len(decoder_sd)} tensors)")

    def _convert_marl4drp_state_dict(self, marl4drp_sd):
        """Convert MARL4DRP's saved state dict to LDRP's decoder state dict format.

        This is a best-effort conversion that relies on the two models having
        identical architecture and compatible PyTorch versions. It will attempt
        to extract the decoder weights from the MARL4DRP checkpoint, which may
        contain additional parameters for other components. If the architectures
        differ significantly, this conversion may fail or produce incorrect results.
        """
        target_sd = self.decoder.state_dict()
        src_keys = sorted(marl4drp_sd.keys())
        tgt_keys = sorted(target_sd.keys())

        if len(src_keys) != len(tgt_keys):
            raise ValueError(
                f"State dict key count mismatch: MARL4DRP has {len(src_keys)} keys, "
                f"but decoder expects {len(tgt_keys)} keys."
            )
        
        converted_sd = {}
        for sk, tk in zip(src_keys, tgt_keys):
            sv = marl4drp_sd[sk]
            tv = target_sd[tk]
            if tuple(sv.shape) != tuple(tv.shape):
                raise ValueError(
                    f"Shape mismatch for key '{tk}': MARL4DRP shape {tuple(sv.shape)} vs "
                    f"decoder shape {tuple(tv.shape)}"
                )
            converted_sd[tk] = sv
        return converted_sd
