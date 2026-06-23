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

### 影響範囲

- `src/epymarl/` 配下に新規ファイルのみ追加。既存アルゴリズム (QMIX/MAPPO 等) は無変更
- LaRe との組み合わせ: on-policy のため LaRe バッファとの相性は MAPPO と同様の課題がある (セクション 2 の集約対策が前提)
- `train.py` の `--config=mat` で呼び出せるようにする

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
