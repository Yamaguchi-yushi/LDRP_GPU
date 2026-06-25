# GPU 最適化設計書

VRAM 使用量の削減と GPU 利用効率の改善に関する設計メモ。

---

## 目次

1. [エピソードバッファの VRAM → RAM 移行](#1-エピソードバッファの-vram--ram-移行)
2. [LaRe on CUDA がワーカープロセスの VRAM を占有する問題](#2-lare-on-cuda-がワーカープロセスの-vram-を占有する問題)

---

## 1. エピソードバッファの VRAM → RAM 移行

### 背景

RTX 5080 (16 GB VRAM) で `batch_size_run: 32` の MAPPO を学習すると VRAM が 12 GB 超に達する。
16 GB のうち 77% が埋まった状態で `use_rnn: True` に変更すると OOM のリスクがある。

### 現状の VRAM 内訳 (batch_size_run=32 実測)

| 要因 | サイズ | 原因 |
|---|---|---|
| メインプロセス (モデル・バッファ) | 938 MiB | バッファが CUDA テンソル |
| ワーカープロセス × 32 | 334 MiB × 32 = 10,688 MiB | use_lare_path=True 時、各ワーカーが DrpEnv 内で LaRe を CUDA に初期化 (セクション 2) |
| **合計** | **≈ 11,626 MiB** | |

### 問題の構造

**問題 A: エピソードバッファが VRAM に置かれている**

[src/epymarl/src/runners/parallel_runner.py:54](../src/epymarl/src/runners/parallel_runner.py#L54):

```python
self.new_batch = partial(EpisodeBatch, ..., device=self.args.device)
# use_cuda: True のとき args.device = "cuda" → バッファ全体が VRAM に乗る
```

バッファに必要なのは「学習時に GPU に転送できること」だけ。
待機中は CPU メモリに置いておけば十分。

**問題 B: ワーカープロセスが LaRe モジュールを CUDA に初期化する**

→ セクション 2 で詳述。`use_lare_path=False` の場合は発生しない。
なお `parallel_runner.py:9` はすでに `spawn` 方式を使用済みのため、
fork による CUDA コンテキスト継承の問題は存在しない。

### 対策案

#### 対策 A: バッファを CPU に置き、学習時だけ GPU に転送 (優先度: 高)

**変更ファイル: `src/epymarl/src/runners/parallel_runner.py`**

```python
# 変更前
self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size,
                         self.episode_limit + 1,
                         preprocess=preprocess, device=self.args.device)

# 変更後
self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size,
                         self.episode_limit + 1,
                         preprocess=preprocess, device="cpu")  # 常に CPU
```

**変更ファイル: `src/epymarl/src/learners/ppo_learner.py`**

```python
def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
    batch = batch.to(self.args.device)  # ← 学習開始時に GPU へ転送
    # 以下は変更なし
    rewards = batch["reward"][:, :-1]
    ...
```

- `EpisodeBatch` には既に `.to(device)` メソッドが実装済み ([episode_buffer.py:80-85](../src/epymarl/src/components/episode_buffer.py#L80))
- 転送タイミングは学習 (update) 呼び出し時のみ。ロールアウト収集中はずっと RAM

**期待削減量**: メインプロセスのバッファ分 (エピソード数 × ステップ数 × obs サイズに依存。数百 MiB 程度)

### 実装優先順位

| 対策 | 実装難度 | VRAM 削減量 | 推奨 |
|---|---|---|---|
| A: バッファを CPU | 低 (2 行変更) | 数百 MiB | ◎ |
| LaRe ワーカー CPU 化 (セクション 2) | 中 | 最大 10.7 GB | LaRe 使用時に対応 |

### 影響範囲

- **対策 A**: `parallel_runner.py` 1 行 + `ppo_learner.py` 1 行。QMIX 等の他 learner は `batch.to()` が不要なため、MAPPO (ppo_learner) のみ変更で可。
- 学習・推論の **挙動は不変**。速度はロールアウト収集時の CPU↔GPU 転送が増えるが、転送はバッチ単位なので影響は軽微。

### 未決事項

- QMIX も同じ `parallel_runner` を使う場合、QMIX の learner にも `.to(device)` が必要になるか確認
- `use_rnn: True` 有効化後の VRAM 削減量を実測して記録する

---

## 2. LaRe on CUDA がワーカープロセスの VRAM を占有する問題

### 背景

`use_lare_path=True` で `batch_size_run: 32` の MAPPO を動かすと VRAM が 12 GB 超に達する。
`use_lare_path=False` にするだけで 938 MiB まで下がる。

### 原因

`parallel_runner.py` はすでに `spawn` 方式でワーカーを生成している ([parallel_runner.py:9](../src/epymarl/src/runners/parallel_runner.py#L9))。
fork による CUDA コンテキスト継承は発生しない。

問題は **ワーカー自身が DrpEnv を生成するときに `DrpEnv.__init__` の中で LaRe モジュールを CUDA に初期化**することにある。

```
ワーカー1: env_fn() → DrpEnv.__init__() → _init_lare_path() → LaRe on CUDA → 334 MiB
ワーカー2: env_fn() → DrpEnv.__init__() → _init_lare_path() → LaRe on CUDA → 334 MiB
...× 32
```

各ワーカーは独立した LaRe モジュールを持ち、それぞれ CUDA メモリを確保する。
ワーカーは環境シミュレーションと報酬計算を担当するため、現状 LaRe をワーカーから切り離せない。

### 対策案

#### 対策 A: ワーカーでは LaRe を CPU に置く (優先度: 高)

`DrpEnv.__init__` で LaRe を初期化するときにデバイスを判定し、
ワーカープロセス（= `is_cuda_available()` を見ても判定できないため別の方法が必要）では CPU を使う。

実装候補:
- `DrpEnv` に `lare_device` 引数を追加し、gym register の kwargs で `"cpu"` を渡す
- または環境変数 (`LARE_DEVICE=cpu`) でワーカー側を制御する

**期待削減量**: 334 MiB × `batch_size_run` = `batch_size_run: 32` で最大 **10.7 GB**

#### 対策 B: ワーカーでは LaRe を無効化し、メインプロセスだけで報酬整形する (優先度: 中)

ワーカーは raw reward を返し、メインプロセスの `ParallelRunner` 側で LaRe の報酬整形を適用する。
LaRe のバッファ更新・学習もメインプロセスに集約する。

- ワーカーの env_args に `use_lare_path=False` を強制
- `ParallelRunner.run()` の報酬受取後に `lare_module.compute_reward(batch)` を呼ぶ

より設計が綺麗だが、現状の「LaRe は env.step() 内で完結」という構造を変える必要がある。

### 影響範囲

- 対策 A: `DrpEnv.__init__` と `_init_lare_path` / `_init_lare_task` のデバイス指定、gym register の kwargs
- 対策 B: `parallel_runner.py` の報酬処理 + `drp_env.py` の LaRe 呼び出し構造
- `use_lare_path=False` の挙動は不変

### 備考

`use_lare_path=False`（従来の報酬設計）で動かす場合はこの問題は発生しない。
LaRe を使う実験を始める前に対策 A を先に入れることを推奨。

---

## 3. MAT (Multi-Agent Transformer) の実装

### 背景

MAPPO は QMIX と比べて性能が劣る傾向がある。MAT (Multi-Agent Transformer, Wen et al. NeurIPS 2022) は cooperative MARL 向けの transformer ベースの on-policy 手法で、MAPPO より高い性能が報告されている。

### 現状

epymarl 経由で QMIX / IQL / VDN / MAPPO / MAA2C が利用可能。MAT は未実装。

### 実装方針

参照実装: PyMARLzoo++ (`src/modules/agents/mat_agent.py` 等)

- **追加ファイル**:
  - `src/epymarl/src/modules/agents/mat_agent.py` — transformer encoder-decoder エージェント
  - `src/epymarl/src/controllers/mat_controller.py` — 全エージェント一括処理用コントローラ
  - `src/epymarl/src/learners/mat_learner.py` — on-policy 学習ループ (MAPPO ベースに transformer 対応を追加)
  - `src/epymarl/src/config/algs/mat.yaml` — MAT 用ハイパーパラメータ (`use_rnn: False`, `d_model: 64`, `n_head: 4`)
- **アーキテクチャ概要**:
  - Encoder: 全エージェントの観測を `[bs, n_agents, obs_dim]` で同時入力し self-attention → latent
  - Decoder: agent 0→1→…→N-1 の順に causal mask 付き cross-attention で行動を自己回帰生成
  - RNN 不要 → シーケンス全体を一括処理

### コントローラの変更が必要な理由

現在の `basic_controller.py` は `agent.forward(inputs, hidden_states)` を 1 ステップずつ呼ぶ設計。MAT は全エージェントを同時に処理するため、`mat_controller.py` を新規作成し `controllers/__init__.py` に登録する。既存コントローラは無変更。

### LDRP 固有の注意点

- **avail_actions マスク**: デコーダの最終 linear 出力に `avail_actions` を掛けてから softmax する (MAPPO と同じパターン)。cross-attention には影響なし
- **エージェント順序**: 固定順序 (0→N-1) で自己回帰。同質エージェントのため順序バイアスは小さいが、PyMARLzoo++ のランダム順序シャッフル option の採否を検討
- **二重行動 (経路＋タスク割当)**: v1 は経路選択のみ MAT でデコード。タスク割当統合は v2 以降
- **CPU buffer 対応**: `mat_learner.py` の `train()` 冒頭で `batch = batch.to(self.args.device)` を入れること (`ppo_learner.py` のパターンを踏襲)

### 実装進捗 (2026-06-25 時点) ── 中断・再開メモ

参照実装は **PyMARLzoo++ (`AILabDsUnipi/pymarlzooplus`)** の **MAT-Dec** を採用。当初計画 (上記「実装方針」) と実体が一部異なるので以下が確定事項。

#### MAT-Dec の構成 (当初計画との差分)

- actor は **分散 (Dec)**: `mlp_mat_agent.py` の `Decoder` は **エージェントごとの素の MLP** (エージェント間 attention なし)。当初計画の「encoder-decoder 自己回帰」とは違い、行動生成だけ `discrete_autoregreesive_act` で逐次サンプルするが、ネット自体は MLP
- critic は **集中**: `modules/critics/mat.py` の `Encoder` が **エージェント軸の self-attention** (`masked=False`) で全 obs を集約し各エージェントの value を出す
- 学習: **PPO + GAE + PopArt**。actor と critic を **1 つの optimizer で同時更新** (MAPPO のような critic 専用 optimizer や target_critic は持たない)
- yaml キー実体: `mac:"mat_mac"` / `agent:"mlp_mat"` / `learner:"mat_learner"` / `critic_type:"mat_critic"` / `extra_in_buffer:["log_probs","values"]` / `use_popart:True` / `standardise_returns,rewards:False` (PopArt と二重正規化になるため False 固定。mat_learner 冒頭で assert 強制)

#### 完了済み (import スモークテスト合格)

| ファイル | 状態 |
|---|---|
| `src/epymarl/src/modules/critics/mat.py` (MATCritic/Encoder/SelfAttention) | 新規追加済 (参照実装と一致) |
| `src/epymarl/src/modules/agents/mlp_mat_agent.py` (MLPMATAgent/Decoder) | 新規追加済 |
| `src/epymarl/src/controllers/mat_controller.py` (MATMAC) | 新規追加済 |
| `src/epymarl/src/learners/mat_learner.py` (MATLearner) | 新規追加済。import パスを `pymarlzooplus.xxx` → LDRP 形式に修正済 |
| `src/epymarl/src/components/standarize_stream.py` に `PopArt` クラス追記 | 済。**import 追加要注意**: `th`/`nn`/`np` を使うので先頭に `import torch as th` / `import torch.nn as nn` / `import numpy as np` を追加済 (元は `import torch` のみで、放置すると import 時 NameError → 全アルゴ起動不能だった) |
| REGISTRY 登録 4 つ: `learners`(mat_learner) / `controllers`(mat_mac) / `agents`(mlp_mat) / `critics`(mat_critic) | 済 |
| `src/epymarl/src/config/algs/mat_dec.yaml` | 作成済 (必須キー充足。参照との差は buffer/batch=32, ppo_epoch=10 のチューニングのみ。`target_update_interval_or_tau` は MAT では未使用) |

#### 未完了 ── ★ここから再開 (C 群: 共有コア改修)

MAT-Dec は `select_actions` がタプル `(actions, extra_returns)` を返し、`extra_in_buffer` で `log_probs`/`values` を buffer に貯める設計。LDRP の epymarl は旧インターフェース (単一返し・extra_in_buffer 無し) なので、**共有コアの改修が必須**。これが無いと MATMAC のタプル返しで runner が落ち、学習データも貯まらない。

**重要 (当初計画の「既存無変更」は誤り)**: `select_actions` を持つ mac は **3 つ** あり (`BasicMAC` / `NonSharedMAC` / `MADDPGMAC`、いずれも独立クラス)、runner を一律タプル展開にするなら **3 つ全部** をタプル化しないと `_ns` 系 (iql_ns/qmix_ns/vdn_ns/coma_ns/ia2c_ns/ippo_ns/maa2c_ns/mappo_ns/pac_ns/pac_dcg_ns) と maddpg が壊れる。

| # | ファイル | 改修内容 |
|---|---|---|
| C-1 | `components/episode_buffer.py` の `__getitem__` (str 分岐) | `elif item == 'batch_size': return self.batch_size` / `'max_seq_length'` / `'device'` を追加 (MATCritic と learner の dict アクセス用) |
| C-2a | `controllers/basic_controller.py:24` | `return chosen_actions` → `return chosen_actions, {}` |
| C-2b | `controllers/non_shared_controller.py:22` | 同上 (`return chosen_actions, {}`) |
| C-2c | `controllers/maddpg_controller.py:65` (`select_actions` のみ。`target_actions` は変更しない) | 同上 |
| C-3 | `runners/episode_runner.py:67,99` | `actions = ...` → `actions, _ = ...` |
| C-4 | `runners/parallel_runner.py:109` | `actions, extra_returns = ...` に展開し、直後の `actions_chosen` に `if "log_probs" in self.args.extra_in_buffer: actions_chosen["log_probs"]=extra_returns["log_probs"].unsqueeze(1)` / `values` 同様を追加 |
| C-5 | `run.py:107` 直後 (`scheme={...}` の後) | `if "log_probs" in args.extra_in_buffer: scheme["log_probs"]={"vshape":(1,),"group":"agents"}` / `values` 同様 |
| C-6 | `config/default.yaml` | `extra_in_buffer: []` を追加 (未定義だと C-4/C-5 で AttributeError) |

**不変条件の保証**: C-1/C-5/C-6 はガード付き/純粋追加で既存に無影響。C-2/C-3/C-4 は「テンソルをタプルで包んで即展開」するだけで、既存アルゴは `extra={}` + `extra_in_buffer=[]` のため格納分岐を素通り → **scheme・buffer・乱数消費すべて不変 = 数値的にビット一致**。ただし上記のとおり **mac 3 つ全部のタプル化が前提条件**。

#### 残タスク (C 群完了後)

- 検証スニペット (CLAUDE.md 末尾) の要領で `lare_*_min_buffer=1` 等を下げ、`--config=mat_dec` で 1〜2 エピソード回して learner が train まで到達するか確認
- 動作確認後、MANUAL.md 更新履歴に「MAT-Dec 追加」を追記 (公開フラグ `--config=mat_dec` の新設として)
- PC→GPU 取り込み時は gpu_changes.md にも記録

### 影響範囲

- **新規ファイルのみ追加…ではない (要注意)**: 上記 C 群のとおり `episode_buffer.py` / `basic_controller.py` / `non_shared_controller.py` / `maddpg_controller.py` / `episode_runner.py` / `parallel_runner.py` / `run.py` / `default.yaml` の **共有コア 8 ファイルを改修**する。ただし既存アルゴリズムの挙動は数値的に不変 (上記「不変条件の保証」参照)
- LaRe との組み合わせ: 報酬は epymarl wrapper (`envs/__init__.py` の `sum(reward)`) でチームスカラーに集約されてから buffer に入るため、MAT-Dec も QMIX/MAPPO と同じ経路で消費。LaRe-Path proxy も同様に集約される。PopArt は報酬スケールが時間変化する LaRe と好相性 (固定正規化より有利)。LaRe-Task は task 割当用で MARL の報酬経路に乗らず無干渉
- `train.py` の `--config=mat_dec` で呼び出せるようにする

---

## 4. HAPPO の実装

### 背景

HAPPO (Heterogeneous-Agent PPO, Zhong et al. ICLR 2022) は MAPPO をエージェントごとの逐次更新に拡張したもの。後続エージェントの advantage を直前エージェントの policy 変化量で補正することで単調性能改善を保証する。MAT 論文でも比較対象として使われているため投稿に必要。

### 実装方針

参照実装: PyMARLzoo++ (`src/learners/happo_learner.py`)

- **追加ファイル**:
  - `src/epymarl/src/learners/happo_learner.py` — `ppo_learner.py` をベースに逐次更新ループを追加
  - `src/epymarl/src/config/algs/happo.yaml` — `mappo.yaml` ベースに `use_individual_optimizer: True` を追加

`ppo_learner.py` との差分は「全エージェント同時更新 → agent 0..N-1 を順番に更新し、直前エージェントの ratio で advantage を補正」の部分のみ。runner / buffer / agent は MAPPO と共通。

### 問題点: パラメータ共有との非整合

HAPPO の理論保証は「各エージェントが独自パラメータを持つ」前提。epymarl の標準設定（全エージェントでパラメータ共有）のまま適用すると、agent_i の更新が残りのエージェントの重みも書き換えてしまい逐次補正が意味をなさなくなる。

| 選択肢 | 実装コスト | メモリ | 論理的正確さ |
|---|---|---|---|
| 非共有パラメータ | agent ごとに独立した重み | n_agents 倍 | ◎ 論文通り |
| 共有パラメータ + 逐次更新 | 補正なしで順番に更新 | 現状維持 | △ MAPPO の亜種 |

LDRP は同質エージェントなのでパラメータ共有が自然。**論文では「共有パラメータ + 逐次更新」として MAPPO との ablation 用途で使う位置づけとし、理論保証の前提が異なる点を明記する**。

### CPU buffer 対応

`happo_learner.py` でも `batch = batch.to(self.args.device)` を `train()` 冒頭に入れる。

### 異種エージェント環境への拡張（配送＋割り当て同時学習）

現在は配送エージェントのみに MARL を適用し、タスク割り当ては PPO/TP/FIFO を独立に使っている。将来的に両者を同時学習する場合の主な選択肢：

| アルゴリズム | 概要 | LDRP 適用難易度 |
|---|---|---|
| **HAPPO** | エージェントごとに独立方策・逐次更新。異種エージェントの理論的本命 | 低（MAPPO の拡張） |
| HATRPO | HAPPO の TRPO 版。信頼領域制約がより厳格 | 高 |
| MAT | アテンションで異種観測・行動を padding/masking で吸収可能 | 中（行動空間の統一が必要） |
| RODE | 役割を自律学習。割り当て役・配送役の分離を自然に学習できる可能性 | 高 |

**同時学習が難しい主な理由**:
- 配送エージェントはノード選択（連続ステップ）、割り当てエージェントはマッチング（イベント駆動）で**タイムスケールが異なる**
- タスク完了報酬を割り当て判断と経路実行のどちらに帰属させるかが不明確（クレジット代入問題）
- 現在の epymarl は同一行動空間を前提としており、異種行動空間への対応には改造が必要

現実的な方針: **まず配送エージェントのみ HAPPO/MAT で改善し、割り当ては PPO のまま維持する**。両者の同時学習は階層型強化学習（HRL）として別途検討する。

---

## 5. QPLEX の実装

### 背景

QPLEX (Wang et al. ICLR 2021) は QMIX の mixing network を Duplex Dueling 構造に置き換えたもの。個々の Q 値を V(s) と A(s,a) に分解し IGM 整合性を維持したまま表現力を上げる。「価値分解の天井」として MAT との比較に使う。

### 実装方針

参照実装: PyMARLzoo++ (`src/modules/mixers/dmaq_qatten.py`)

- **追加ファイル**:
  - `src/epymarl/src/modules/mixers/qplex.py` — `dmaq_qatten.py` を移植 (QMixer の上位互換として差し込む)
  - `src/epymarl/src/config/algs/qplex.yaml` — `qmix.yaml` ベースに `mixer: "qplex"`, `n_head: 4`, `qplex_embed_dim: 64` を追加

**Learner**: 既存の `q_learner.py` をそのまま使う。`mixer` を `qplex` に差し替えるだけで動くはず。

### 注意点

- QPLEX は off-policy (QMIX と同じ replay buffer 運用)。runner / buffer 変更不要
- Q-Attention は O(n_agents²) の計算を含むが n_agents=4 では問題なし
- `modules/mixers/__init__.py` に `"qplex": QPLEXMixer` を登録することを忘れずに

---

最終更新: 2026-06-14
