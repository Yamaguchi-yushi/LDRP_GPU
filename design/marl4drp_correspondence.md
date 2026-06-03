# MARL4DRP との LaRe 機能ファイル対応表

LDRP の LaRe 統合は MARL4DRP (`/Users/yamaguchiyuushi/MARL4DRP/`) を参照実装として行った。
本書は **どの機能がどのファイルに対応するか** を記録する。実装の差分・設計思想の違いも併記。

詳細な設計判断は [lare_integration.md](lare_integration.md) を参照。

---

## 目次

1. [対応表 (機能別)](#1-対応表-機能別)
2. [設計思想の差](#2-設計思想の差)
3. [未移植の機能](#3-未移植の機能)
4. [移植時の注意点](#4-移植時の注意点)

---

## 1. 対応表 (機能別)

### 1.1 10 因子エンコーダ (factor extraction)

env の state を 10 次元の因子ベクトルに変換する関数。

| 役割 | MARL4DRP | LDRP |
|---|---|---|
| Path 用 factor 関数 (env state → 10 次元 factor) | `drp_env/reward_model/LLMrd/fallback_functions/evaluation_func.py` | `src/lare/path/encoder.py` の `evaluation_func()` |
| マップ別の特殊 factor 関数 (オプション) | `LLMrd/pre_fallback_functions/map_*_*agents.py` | (LDRP では未移植) |
| Task 用 factor 関数 | (MARL4DRP は Task LaRe 持たず) | `src/lare/task/encoder.py` の `evaluation_func_task()` |

### 1.2 デコーダ (factor → reward)

10 因子から 1 ステップ報酬を予測する MLP。

| 役割 | MARL4DRP | LDRP |
|---|---|---|
| Path デコーダ MLP | `drp_env/reward_model/LLMrd/factor_reward_model.py` (`FactorRewardModel`) | `src/lare/path/decoder.py` (`PathRewardDecoder`) |
| Task デコーダ MLP | (なし) | `src/lare/task/decoder.py` |
| factor + state → reward 統合層 | `LLMrd/factor_reward_decompose.py` (`FactorRewardDecomposer`) | (LDRP は module 側に直接統合) |

### 1.3 モジュール / オーケストレータ (バッファ + 学習を回す本体)

評価期間管理・update トリガ・autosave 制御。

| 役割 | MARL4DRP | LDRP |
|---|---|---|
| Path 学習ループの司令塔 | `drp_env/drp_env.py` の `initialize_lare_system()`, `perform_episode_update()`, `evaluation_period` 状態機械 (1340-1382 行) | `src/lare/path/lare_path_module.py` (`LaRePathModule` クラス) |
| Task 学習ループの司令塔 | (なし) | `src/lare/task/lare_task_module.py` |

**重要な違い**: MARL4DRP は env 内部に LaRe 制御を直接書いている。
LDRP は `LaRePathModule` という独立クラスに切り出して env からは委譲する設計。
これにより env コードの肥大化を防ぎ、Path / Task の対称的な実装が可能になっている。

### 1.4 エピソードバッファ (経験データの保管)

| 役割 | MARL4DRP | LDRP |
|---|---|---|
| エピソードバッファ実装 | `epymarl/src/utils/replay_memory.py` (`ReplayMemory_episode`) | `src/lare/path/buffer.py` (`PathEpisodeBuffer`), `src/lare/task/buffer.py` |
| 置換戦略 | circular buffer (`position % capacity`) | `deque(maxlen=capacity)` (FIFO) |
| memory 分離 (goal/collision/timeup) | `goal_memory`, `collision_memory`, `timeup_memory` を別々に持つ | 単一 buffer のみ (未移植) |
| サンプリング | termination タイプ別 60/20/20 weighted | uniform random |

### 1.5 訓練ステップ (loss + gradient)

| 役割 | MARL4DRP | LDRP |
|---|---|---|
| train_step 関数 (1 batch の forward + backward + optimizer.step) | `epymarl/src/utils/util.py` `make_train_step()` (163-280 行) | `src/lare/path/lare_path_module.py:200-254` `_update()` メソッド |

両者とも MSE 損失 (`pred_return` と `episode_return` の二乗誤差) で同一。
LDRP は `reduction='mean'` で正規化しているが、ターゲットスケール (-5000〜-10000) のため loss が 10⁵〜10⁸ に達しやすい。

### 1.6 Transformer (AREL Time-Agent Attention, オプション)

時系列とエージェント間の attention で credit を分解。

| 役割 | MARL4DRP | LDRP |
|---|---|---|
| 時系列・エージェント間の注意機構 | `drp_env/reward_model/arel/transformers.py` | `src/lare/path/transformer.py` (`TimeAgentTransformer`) |
| 基本 attention モジュール | `drp_env/reward_model/arel/modules.py` | `src/lare/shared/attention.py` |
| utility (positional encoding 等) | `drp_env/reward_model/arel/util.py` | (transformer.py 内に統合) |

LDRP デフォルトでは `use_transformer=False` (= MLP のみ)。

### 1.7 MARD (Shapley Attention for assignment, 未移植)

タスク割当のクレジット分解に Shapley 値ベースの attention を使う仕組み。

| 役割 | MARL4DRP | LDRP |
|---|---|---|
| 割当クレジット用 Shapley attention | `drp_env/reward_model/mard/mard.py` | (未移植。Task LaRe で別アプローチを採用) |
| MARD の attention モジュール | `drp_env/reward_model/mard/modules.py` | (なし) |
| 状態正規化 | `drp_env/reward_model/mard/norm.py` | (なし) |

### 1.8 LLM プロンプト (factor 関数の自動生成)

GPT に factor 関数を生成させる仕組み。

| 役割 | MARL4DRP | LDRP |
|---|---|---|
| GPT に factor 関数を生成させるコード | `drp_env/reward_model/LLMrd/factor_chat_with_gpt.py` | (未移植。fallback の evaluation_func を直接使用) |
| プロンプトテンプレート | `drp_env/reward_model/LLMrd/prompt_template.py` | (なし) |

### 1.9 env 統合 (step フックポイント)

env の step() に LaRe の呼び出しを埋め込む箇所。

| 役割 | MARL4DRP | LDRP |
|---|---|---|
| env.step() 内 LaRe 呼び出し箇所 | `drp_env/drp_env.py` の `step()` 内 (1431 行〜)、collision 後の reward 置換 (1601-1619 行)、reset() 内の evaluation_period 管理 (1340-1382 行) | `src/main/drp_env/drp_env.py:879-882` で proxy 報酬置換、step 内の `_lare_total_step_account += 1` |
| env 設定 | `use_lare_reward`, `use_lare_training`, `use_pretrained_model`, `pretrained_model_name` | `use_lare_path`, `use_lare_path_training`, `use_pretrained_lare_path`, `pretrained_lare_path_model_name` 等 |

### 1.10 ベース報酬モデルの抽象化

| 役割 | MARL4DRP | LDRP |
|---|---|---|
| 共通基底クラス | `drp_env/reward_model/base_reward_model.py` | (LDRP では Path/Task で抽象化していない) |
| module レジストリ | `drp_env/reward_model/__init__.py` | (LDRP は分離設計でレジストリ不要) |

---

## 2. 設計思想の差

| 軸 | MARL4DRP | LDRP |
|---|---|---|
| 構造 | env (drp_env.py) に LaRe ロジックが内蔵 | LaRe を別パッケージ (`src/lare/`) に切り出し、env はフックポイントのみ |
| Task LaRe | 無し (Path 用のみ) | Path / Task の 2 系統を独立に実装 |
| memory 分離 | termination タイプ別に 3 バッファ + weighted sampling | 単一 buffer + uniform sampling |
| LLM 連携 | GPT で factor 関数を自動生成可 | fallback を直接使用 (LLM 未移植) |
| MARD (Shapley) | 割当に Shapley attention 使用可 | 未移植 (Task LaRe で別アプローチ) |

### LDRP 側の主な変更理由

1. **モジュール分離**: env コードに reward model ロジックを書くと、env が肥大化して保守困難になる。LDRP は env はフック (`step()` 内 1 行) のみで、本体は `LaRePathModule` に切り出し
2. **Path/Task 対称化**: 割当系の LaRe (Task) を後から追加できるよう、Path と同じパターン (`encoder.py + decoder.py + module.py + buffer.py`) を `src/lare/task/` に複製
3. **シンプルな buffer**: termination 分離は学習安定化に効くが、実装が複雑化するため第一バージョンでは省略

---

## 3. 未移植の機能

MARL4DRP にあって LDRP に持ってきていない機能の一覧。後日の移植候補。

| # | 機能 | 効果 (推定) | 移植コスト |
|---|---|---|---|
| 1 | termination 別 memory 分離 (`goal_memory`, `collision_memory`, `timeup_memory`) + **60/20/20 weighted sampling** | デコーダの安定学習。レア事象 (collision) の signal を強調 | 中 (buffer.py + module.py の変更) |
| 2 | MARD (Shapley attention) | 割当 credit 分解の精度向上 | 高 (新規モジュール) |
| 3 | LLM による factor 自動生成 | factor の自動探索。マップ別の最適化 | 高 (OpenAI API + プロンプト管理) |
| 4 | マップ別 `pre_fallback_functions` | マップ特化の factor 設計 | 低 (ファイルコピー + register) |
| 5 | `use_separete_memory` モード切り替え | memory 分離の on/off | (1 と一緒に) |

**特に 1 は現在の loss 振動 (10⁵〜10⁸) を改善する可能性が高い** ため、最優先で移植検討の価値あり。

---

## 4. 移植時の注意点

### 4.1 update_count の引き継ぎ

MARL4DRP の `total_training_steps` を LDRP の `update_count` として読む変換が `lare_path_module.py:295` にある。
finetuning で MARL4DRP の `.pth` を読むと `update_count` が数百万から始まる。

### 4.2 state_dict のキー名差

LDRP は `lare_path_module.py:314-343` の `_convert_marl4drp_state_dict()` でキー名を位置的にマッピング。
**両者の MLP 構造 (層数・hidden_dim) が完全に一致している必要がある**。デフォルトは両者とも `n_layers=3, hidden_dim=64, factor_dim=10` で揃っている。

### 4.3 評価期間トリガの違い

LDRP は MARL4DRP の `evaluation_period` 状態機械を `lare_path_module.py:185-198` に移植済み。
ただし MARL4DRP デフォルト値 `reward_model_update_freq=256, evaluation_episodes=16` (= 6.25% update) に対し、
LDRP デフォルトは `update_freq=128, train_epochs=50` (= 39% update) なので、**学習頻度が約 6 倍多い**。
変更したい場合は `default.yaml` の `lare_path_update_freq` と、`LaRePathConfig.train_epochs` (現状 hardcode) を調整。

### 4.4 env 統合の差

MARL4DRP は `drp_env.py` 内に LaRe ロジックを直書きしているので、env 改変時に LaRe 部分も同時に触る必要がある。
LDRP は **env と LaRe が疎結合** なので、env 改変時に LaRe を考慮しなくてよい (フック呼び出しの 1 行だけ確認すれば OK)。

---

## 更新履歴

- 2026-06-XX (初版): 1.2M ファインチューニング失敗を契機に、対応関係と未移植機能を整理
