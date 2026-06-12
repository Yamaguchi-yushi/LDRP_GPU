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

epymarl に MAT 用のエージェントとラーナーを追加する。

- **参照実装**: [PKU-MARL/Multi-Agent-Transformer](https://github.com/PKU-MARL/Multi-Agent-Transformer) (論文著者実装)
- **追加ファイル**:
  - `src/epymarl/src/modules/agents/mat_agent.py` — transformer encoder-decoder エージェント
  - `src/epymarl/src/learners/mat_learner.py` — on-policy 学習ループ (MAPPO ベースに transformer 対応を追加)
  - `src/epymarl/src/config/algs/mat.yaml` — MAT 用ハイパーパラメータ
- **アーキテクチャ概要**:
  - Encoder: 全エージェントの観測を同時に処理 (self-attention)
  - Decoder: エージェントごとに順番に行動を生成 (autoregressive, cross-attention)
  - RNN 不要 → シーケンス全体を一括処理

### 影響範囲

- `src/epymarl/` 配下に新規ファイルのみ追加。既存アルゴリズム (QMIX/MAPPO 等) は無変更
- LaRe との組み合わせ: on-policy のため LaRe バッファとの相性は MAPPO と同様の課題がある (セクション 2 の集約対策が前提)
- `train.py` の `--config=mat` で呼び出せるようにする

---

最終更新: 2026-06-12
