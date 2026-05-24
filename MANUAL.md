# LDRP プロジェクト説明書

LDRP リポジトリに対して加えた **大きな変更の使い方** をまとめた人間向け説明書。
今のところ収録範囲は **LaRe 統合 (System A: 経路計画 / System B: タスク割当)** のみ。今後の大きな変更はこのファイルに章を追加し、末尾の [更新履歴](#10-更新履歴) に1エントリ追加していく方針。

設計の背景・理論は [design/lare_integration.md](design/lare_integration.md) を、LDRP 全体の概要は [GUIDE.md](GUIDE.md) を参照。設計書一覧は [design/](design/) 以下。

---

## 目次

1. [概要](#1-概要)
2. [クイックスタート](#2-クイックスタート)
3. [4つの動作モード](#3-4つの動作モード)
4. [設定パラメータ一覧](#4-設定パラメータ一覧)
5. [保存ファイルの命名規則](#5-保存ファイルの命名規則)
6. [エンコーダの 10 因子](#6-エンコーダの-10-因子)
7. [コード構成](#7-コード構成)
8. [既存コードへの変更点](#8-既存コードへの変更点)
9. [運用 Tips & トラブルシューティング](#9-運用-tips--トラブルシューティング)
10. [更新履歴](#10-更新履歴)

---

## 1. 概要

LDRP の単一スカラー報酬 (`goal/collision/wait/move`) を、**学習可能な潜在報酬モデル** で置き換える機能を追加しました。Safe-TSL-DBCT の実装パターンを踏襲しつつ、**Path 用** と **Task 用** で完全に独立したサブシステムにしてあります。

| | System A: **LaRe-Path** (経路計画) | System B: **LaRe-Task** (タスク割当) |
|---|---|---|
| 評価対象 | 各ステップの移動行動 (IQL/QMIX/VDN/MAA2C などの epymarl 学習) | 各タスク割当決定 (PPOタスク割当器) |
| 訓練ターゲット R | エピソード中の環境報酬合計 (Σ_t Σ_i r_env) | エピソード末の `task_completion` |
| 因子数 | 10 (Safe-TSL-DBCT と同じ) | 10 (本実装で新規設計) |
| 決定頻度 | 毎ステップ (高頻度 T 回) | アイドル時のみ (疎 K ≪ T) |
| 保存先 | `src/lare/path/saved_models/` | `src/lare/task/saved_models/` |
| 保存ファイル名トークン | `PATH` | `TASK` |

**最重要事項**: フラグを全て `false` にすれば LDRP の挙動は元通り変わりません。LaRe を有効化したときだけコード経路に入ります。

---

## 2. クイックスタート

### 2.1 何もしないモード (デフォルト = 既存挙動)

```yaml
# src/config/default.yaml の関連抜粋
use_lare_path: false
use_lare_task: false
```

→ `python test.py map_8x5 4 qmix tp` などで従来通り動きます。

### 2.2 Path をオンライン学習

```yaml
use_lare_path: true
lare_path_autosave: true     # 学習中チェックポイント自動保存
```

実行例 (epymarl 経由で学習する場合は `train.py` を参照):
```bash
python test.py map_8x5 4 qmix ppo
```

エピソード経過とともに `src/lare/path/saved_models/Safe_QMIX_PATH_map_8x5_4agents_X.XM_checkpoint.pth` が自動更新されます。

### 2.3 Task もオンライン学習

```yaml
use_lare_task: true
lare_task_autosave: true
```

`runner.py` の `task_assigner.buffer_add_rewards(...)` がデコーダ学習後は proxy 報酬を流し込むようになります (PPOタスク割当時のみ効果あり)。

### 2.4 学習済みモデルで動かす (再学習しない)

```yaml
use_lare_path: true
use_pretrained_lare_path: true
pretrained_lare_path_model_path: "Safe_QMIX_PATH_map_8x5_4agents_2.0M_checkpoint.pth"

use_lare_task: true
use_pretrained_lare_task: true
pretrained_lare_task_model_path: "Safe_QMIX_TASK_map_8x5_4agents_2.0M_checkpoint.pth"
```

→ 重みをロード → デコーダは凍結 (`frozen=True`) → 推論専用。

### 2.5 既存モデルから追加学習 (ファインチューニング)

```yaml
use_lare_path: true
use_finetuning_lare_path: true
finetuning_lare_path_model_path: "Safe_QMIX_PATH_map_8x5_4agents_2.0M_checkpoint.pth"
lare_path_autosave: true
```

→ 重みロード後も学習継続。新しい checkpoint のファイル名は `FT_Safe_<元>_..._checkpoint.pth`。

---

## 3. 4つの動作モード

各システム (Path / Task) で **4モード** が選べます。両者は独立に組み合わせ可能です。

| モード | 必要フラグ | デコーダ | 目的 |
|---|---|---|---|
| **1. Baseline** | `use_lare_*: false` | なし | 完全に従来挙動 |
| **2. Scratch** | `use_lare_*: true` | 新規初期化 + オンライン学習 | ゼロから学習 |
| **3. Pretrained** | `use_lare_*: true` + `use_pretrained_lare_*: true` + path指定 | ロード + 凍結 | 推論専用 (再学習なし) |
| **4. Finetuning** | `use_lare_*: true` + `use_finetuning_lare_*: true` + path指定 | ロード + 学習継続 | 既存モデルを追加学習 |

> Pretrained と Finetuning を同時に true にすると **Pretrained が優先** されます (Safe-TSL-DBCT と同挙動)。

### モード判定ログ

env 初期化時に標準出力に出ます。例:
```
[LaRe-Path] Initialized (mode=scratch, training=True, factors=10)
[LaRe-Task][PRETRAINED] Loaded /path/to/model.pth - frozen (inference only)
[LaRe-Task][FINETUNE] Loaded /path/to/model.pth - trainable (finetuning)
```

---

## 4. 設定パラメータ一覧

`src/config/default.yaml` で全て指定可能。`gym.make(...)` の kwargs として直接渡すこともできます ([test.py](test.py) で自動転送)。

### 4.1 共通スイッチ

| キー | 型 | デフォルト | 説明 |
|---|---|---|---|
| `use_lare_path` | bool | false | System A 全体のマスタースイッチ |
| `use_lare_path_training` | bool | true | false にするとデコーダ学習はするが報酬置換しない (検証用) |
| `use_lare_task` | bool | false | System B 全体のマスタースイッチ |
| `use_lare_task_training` | bool | true | 同上 (Task 用) |

### 4.2 ハイパーパラメータ (Path 側)

| キー | デフォルト | 説明 |
|---|---|---|
| `lare_path_factor_dim` | 10 | エンコーダ出力次元 (Safe-TSL-DBCT 互換は 10) |
| `lare_path_decoder_hidden_dim` | 64 | MLP 隠れ層サイズ |
| `lare_path_decoder_n_layers` | 3 | MLP 総層数 |
| `lare_path_use_transformer` | false | AREL Time-Agent Transformer を挟むか |
| `lare_path_transformer_heads` | 4 | (Transformer 有効時のみ) |
| `lare_path_transformer_depth` | 2 | (同上) |
| `lare_path_buffer_capacity` | 512 | エピソードバッファのリングサイズ |
| `lare_path_min_buffer` | 64 | 学習開始の最小エピソード数 |
| `lare_path_update_freq` | 32 | N エピソードごとに更新 |
| `lare_path_batch_size` | 32 | バッチサイズ |
| `lare_path_lr` | 0.0005 | Adam 学習率 |

### 4.3 ハイパーパラメータ (Task 側)

| キー | デフォルト | 説明 |
|---|---|---|
| `lare_task_factor_dim` | 10 | (本実装で設計した 10 因子) |
| `lare_task_decoder_hidden_dim` | 64 | |
| `lare_task_decoder_n_layers` | 2 | (K ≪ T なので Path より浅め) |
| `lare_task_buffer_capacity` | 512 | |
| `lare_task_min_buffer` | 32 | |
| `lare_task_update_freq` | 16 | |
| `lare_task_batch_size` | 32 | |
| `lare_task_lr` | 0.0005 | |

### 4.4 ロード/保存

| キー | デフォルト | 説明 |
|---|---|---|
| `use_pretrained_lare_path` | false | Pretrained モードに入る |
| `pretrained_lare_path_model_path` | null | ロード対象パス (絶対 or 相対 or ファイル名のみ) |
| `use_finetuning_lare_path` | false | Finetuning モードに入る |
| `finetuning_lare_path_model_path` | null | ロード対象パス |
| `lare_path_autosave` | false | 学習中の自動保存 (scratch / finetuning でのみ動作) |
| `lare_path_autosave_path` | null | 明示的な保存パス (固定文字列で1ファイル上書き) |
| `lare_path_save_dir` | null | 保存先ディレクトリ (null なら `src/lare/path/saved_models/`) |

Task 側もキー名を `lare_task_*` に置き換えただけで同じ構造です。

#### モデルパスの解決順

`pretrained_lare_*_model_path` / `finetuning_lare_*_model_path` は次の順に探索します:

1. 絶対パス → そのまま
2. 相対パス → カレントディレクトリ
3. リポジトリルート相対
4. `src/lare/{path,task}/saved_models/<ファイル名>`
5. 上記4つの末尾に `.pth` を足したバリエーション

---

## 5. 保存ファイルの命名規則

Safe-TSL-DBCT の流儀に揃えています。

### Path

| 状況 | ファイル名 |
|---|---|
| Scratch (Safe env) | `Safe_{ALGO}_PATH_{map}_{N}agents_{X.X}M_checkpoint.pth` |
| Scratch (非Safe env) | `{ALGO}_PATH_{map}_{N}agents_{X.X}M_checkpoint.pth` |
| Finetuning (Safe env) | `FT_Safe_{source_base}_{map}_{N}agents_{X.X}M_checkpoint.pth` |
| Finetuning (非Safe env) | `FT_{source_base}_{map}_{N}agents_{X.X}M_checkpoint.pth` |

### Task

`PATH` を `TASK` に置き換えるだけ。例: `Safe_QMIX_TASK_map_8x5_4agents_1.5M_checkpoint.pth`。

### トークンの中身

- **Safe**: 環境クラスが `SafeEnv` (= `drp_safe-...`) のとき "Safe"。それ以外は空文字 (アンダースコアごと省略)。
- **ALGO**: `--config=qmix` 等の CLI 引数を大文字化。検出できないと `UNKNOWN`。
- **{X.X}M**: env が観測した累積ステップ数 (`_lare_total_step_account`) を 100万単位で表示。各保存タイミングで再計算されます (1.0M, 1.1M, 2.0M…)。
- **source_base**: Finetuning 元ファイル名から `.pth` / `_final` / `_checkpoint` / 先頭 `FT_` を剥いだもの。

### 保存先ディレクトリ

```
LDRP/
└── src/lare/
    ├── path/saved_models/    ← Path モデルの自動保存先
    └── task/saved_models/    ← Task モデルの自動保存先
```

`lare_path_save_dir` / `lare_task_save_dir` で上書き可能。

---

## 6. エンコーダの 10 因子

各因子は概ね [0, 1] に正規化し、**高いほど望ましい状態** に揃えています。

### 6.1 Path (`src/lare/path/encoder.py`)

Safe-TSL-DBCT の `evaluation_func` をそのまま移植:

| # | 因子 | 意味 |
|---|---|---|
| 1 | `prog_goal` | 1ステップでゴールに近づいた距離 |
| 2 | `in_collision` | 自分が衝突に関与している (0/1) |
| 3 | `others_in_collision` | 他者間で衝突がある (0/1) |
| 4 | `wait_norm` | 待機カウンタ |
| 5 | `dist_goal_norm` | ゴールまでの距離 / グラフ直径 |
| 6 | `min_sep_norm` | 最近隣エージェントとの距離 (正規化) |
| 7 | `avg_sep_norm` | 平均他者距離 (正規化) |
| 8 | `safety_margin` | min_sep / collision_distance (クリップ) |
| 9 | `collision_risk` | min_sep < 衝突距離×2 ? (0/1) |
| 10 | `at_goal` | ゴール到達フラグ (0/1) |

### 6.2 Task (`src/lare/task/encoder.py`)

本実装で新規設計した 10 因子。1 つの "agent q にタスク r を割り当てる" 決定に対して計算します:

| # | 因子 | 意味 |
|---|---|---|
| 1 | `pickup_proximity` | 1 - dist(agent, pickup) / 直径 (近いほど良) |
| 2 | `delivery_efficiency` | 1 - dist(pickup, dropoff) / 直径 (短いほど良) |
| 3 | `wait_time_norm` | wait_steps / time_limit (古いタスクほど高) |
| 4 | `load_balance` | 1 - std(agent_loads_after) / max_imbalance |
| 5 | `idle_assignment` | アイドル → 1.0 / 占有 → 0.0 |
| 6 | `closest_agent_match` | この agent が pickup に対して最近隣か |
| 7 | `queue_drain` | 1 - 残未割当数 / task_num |
| 8 | `low_redirect_cost` | 1 - dist(prev_goal, pickup) / 直径 |
| 9 | `urgency_response` | 古いタスクへの非線形重み |
| 10 | `batch_assignment_density` | この step の割当数 / agent_num |

LLM 生成版に差し替えたい場合は `evaluation_func_task` を別関数に差し替えれば同じインタフェースで動きます (将来拡張用)。

---

## 7. コード構成

### 新規追加ファイル

```
src/lare/
├── __init__.py
├── path/
│   ├── __init__.py
│   ├── encoder.py            # 10 因子 evaluation_func + obs ビルダー + Dijkstra/直径計算
│   ├── decoder.py            # PathRewardDecoder (MLP)
│   ├── transformer.py        # AREL Time-Agent Transformer (オプション)
│   ├── buffer.py             # PathEpisodeBuffer (step 単位, 固定形状)
│   └── lare_path_module.py   # LaRePathModule + LaRePathConfig
├── task/
│   ├── __init__.py
│   ├── encoder.py            # 10 因子 evaluation_func_task + 状態ビルダー
│   ├── decoder.py            # TaskRewardDecoder (MLP)
│   ├── buffer.py             # TaskEpisodeBuffer (assignment 単位, 可変 K)
│   └── lare_task_module.py   # LaReTaskModule + LaReTaskConfig
└── shared/
    ├── __init__.py
    └── attention.py          # SelfAttention / TransformerBlock 共通実装
```

### 変更したファイル

```
src/main/drp_env/drp_env.py    # __init__ にフラグ追加, step() にフック挿入, 命名ヘルパ群
src/main/drp_env/__init__.py   # drp_safe 登録に use_lare_path: False を追加 (ドキュメント目的)
src/config/default.yaml        # LaRe 系キーを追加
runner.py                      # PPO の buffer_add_rewards を Task proxy 報酬に分岐
test.py                        # config から LaRe キーを gym.make に転送
```

LaRe-Path/Task 以外の既存ロジック (PBS, IQL推論, PPO, Safe wrapper など) は **触っていません**。

---

## 8. 既存コードへの変更点

### `DrpEnv.__init__`

新規 kwargs を追加。**全てデフォルト値で false / null** なので呼び出し側の変更不要:

```python
DrpEnv(
    ...,
    # System A
    use_lare_path=False, use_lare_path_training=True,
    lare_path_factor_dim=10, lare_path_decoder_hidden_dim=64, ...
    use_pretrained_lare_path=False, pretrained_lare_path_model_path=None,
    use_finetuning_lare_path=False, finetuning_lare_path_model_path=None,
    lare_path_autosave=False, lare_path_autosave_path=None, lare_path_save_dir=None,
    # System B
    use_lare_task=False, use_lare_task_training=True, ...
)
```

### `DrpEnv.step` (System A 用フック)

```python
# step 開始時: 移動前の onehot 位置をスナップショット
self._lare_prev_onehot_pos = self._lare_capture_prev_onehot_pos()

# 衝突判定後: 衝突ペアを抽出 (encoder で使用)
self._lare_current_colliding_pairs = self._lare_compute_colliding_pairs(self.obs_prepare)

# ri_array が確定した直後: 因子計算 → record_step → 学習済みなら proxy 報酬で置換
if self.use_lare_path and self.lare_path_module is not None:
    factors = self.lare_path_module.compute_factors(...)
    self.lare_path_module.record_step(factors, sum(ri_array))
    if self.lare_path_module.is_trained:
        ri_array = list(self.lare_path_module.proxy_rewards(factors))

# done 時: end_episode (バッファに R_path を確定し、必要なら更新+autosave)
if all(self.terminated):
    self.lare_path_module.end_episode()
```

### `DrpEnv.step` (System B 用フック)

タスク割り当てロジック内で **割り当てが確定する瞬間** にスナップを取り、`record_step_assignments` に渡します:

```python
lare_task_decisions = []
for i in range(agent_num):
    if (assigned_tasks[i] == [] or i in assigned_list) and task_assign[i] != -1:
        # ① 割当 *直前* の状態を保存
        was_idle = (assigned_tasks[i] == [])
        prev_goal = goal_array[i] if not was_idle else None
        wait_steps = step_account - _lare_task_creation_steps[r]
        # ② 既存の割当処理
        assigned_tasks[i] = current_tasklist[r]; goal_array[i] = ...
        # ③ 決定情報を蓄積
        lare_task_decisions.append({...})

# まとめて encoder/decoder へ
if self.use_lare_task and decisions:
    full = [{**d, "agent_loads_after": ..., "unassigned_after": ..., "n_assignments_step": K} for d in decisions]
    self.lare_task_module.record_step_assignments(self, full)

# step 戻り値: info に proxy 報酬と is_trained フラグを載せる
info["lare_task_proxy_reward"] = self.lare_task_module.consume_step_proxy_reward()
info["lare_task_is_trained"] = self.lare_task_module.is_trained

# done 時: R_task = task_completion で end_episode
if all(self.terminated):
    self.lare_task_module.end_episode(self.task_completion)
```

タスク作成・除去と同期して `_lare_task_creation_steps` (current_tasklist と並行配列) を維持しています。

### `runner.py`

PPO に渡す報酬だけを分岐:

```python
if self.training:
    if info.get("lare_task_is_trained", False):
        task_reward = float(info.get("lare_task_proxy_reward", 0.0))
    else:
        task_reward = float(sum(rew_n))
    self.task_manager.task_assigner.buffer_add_rewards(task_reward, done)
```

`use_lare_task=False` のときは info にキーが入らないので `False` 分岐で `sum(rew_n)` が使われます = **既存挙動維持**。

---

## 9. 運用 Tips & トラブルシューティング

### 9.1 Path/Task の独立性

両モジュールは別ファイル・別オプティマイザ・別バッファ・別保存先で完全に独立です。同時オン可:

```yaml
use_lare_path: true
use_lare_task: true
```

→ 環境ステップごとに Path が proxy 報酬を返し、Task は割当発生時のみ proxy 報酬を `info` に出力。

### 9.2 baseline 動作確認

`use_lare_path: false`, `use_lare_task: false` で次のチェックポイントを通る:
- `env.unwrapped.lare_path_module is None`
- `env.unwrapped.lare_task_module is None`
- `info` に `lare_*` キーが入らない
- `runner.py` で `task_reward = sum(rew_n)` 経路を通る

### 9.3 学習が進んでるか確認

```python
mod = env.unwrapped.lare_path_module
print(mod.is_trained, mod.update_count, mod.last_loss)
# True 5 0.034...   ← 学習が進んでいる
```

学習開始まで `min_buffer` 分のエピソードが必要なので、短い検証では `lare_*_min_buffer: 2` などに下げてください。

### 9.4 ファイルが見つからない

ロードログ:
```
[LaRe-Path][PRETRAINED] Model not found in: ['xxx', '/abs/xxx', '...']
[LaRe-Path][PRETRAINED] Falling back to scratch training.
```

候補リストが表示されるので、それを参考にパスを修正してください。`.pth` 拡張子無しでも自動補完されます。

### 9.5 命名衝突

Path と Task は **同名アルゴリズム/マップでも衝突しません**:
- `Safe_QMIX_PATH_map_8x5_4agents_1.0M_checkpoint.pth` → `src/lare/path/saved_models/`
- `Safe_QMIX_TASK_map_8x5_4agents_1.0M_checkpoint.pth` → `src/lare/task/saved_models/`

### 9.6 性能

`compute_factors` は内部で agent ごと/step ごとに Dijkstra を呼ぶため、大規模マップ + 多エージェント + Path/Task 同時オンだと 1step 数十 ms オーダになることがあります。長エピソードでは `lare_path_use_transformer: false` を保ち、Transformer は学習効果を見たいときだけ true にしてください。

### 9.7 epymarl 学習との連動

epymarl の `_GymmaWrapper.step()` は `sum(reward)` で集約するので、env が ri_array を proxy 化した時点で IQL/QMIX 等の学習ターゲットも自動で proxy 報酬になります。学習スクリプト ([train.py](train.py)) 側の改修は不要です。

### 9.8 既知の制限

- LaRe-Path の Transformer 経路 (`lare_path_use_transformer=true`) は実装済みですが、デコーダの forward での次元の取り回しに簡略化があるため (`(b,n_a,t,1)→展開`)、本格的に AREL の貢献分解を使いたい場合は学習ループの再検討が必要です。**まず MLP のみ (デフォルト false) で十分動きます。**
- Task 側の MARD/Shapley クレジット (設計書 §7-4) は未実装。代わりにシンプルな per-decision MLP で割り当て品質を回帰します。Shapley を入れたい場合は `src/lare/task/` に `mard.py` を追加し、`LaReTaskModule._update` の前段に挟んでください。

---

## 付録: 設定例 (用途別)

### A. 既存実験を続行 (LaRe オフ)
```yaml
use_lare_path: false
use_lare_task: false
```

### B. ゼロから学習 + 自動保存
```yaml
use_lare_path: true
lare_path_autosave: true
use_lare_task: true
lare_task_autosave: true
```

### C. 学習済み Path のみ載せて試す
```yaml
use_lare_path: true
use_pretrained_lare_path: true
pretrained_lare_path_model_path: "Safe_QMIX_PATH_map_8x5_4agents_2.0M_checkpoint.pth"
use_lare_task: false
```

### D. 既存 Path をファインチューニング + Task は新規学習
```yaml
use_lare_path: true
use_finetuning_lare_path: true
finetuning_lare_path_model_path: "Safe_QMIX_PATH_map_8x5_4agents_1.0M_checkpoint.pth"
lare_path_autosave: true
use_lare_task: true
lare_task_autosave: true
```

### E. デコーダ学習はするが報酬は置換しない (実験対照群)
```yaml
use_lare_path: true
use_lare_path_training: false   # 学習はする, 環境の報酬は元のまま
```

---

## 10. 更新履歴

大きな変更を加えたら **最上部に1エントリ追記** する。フォーマットは下のテンプレに揃える。

### テンプレート (コピペ用)

```markdown
### YYYY-MM-DD <変更タイトル>
- **意図**: なぜこの変更が必要だったか
- **変わること**: 挙動・出力・互換性が具体的に何がどう変わるか (フラグ名、ファイル名、API 形状など)
- **触ったファイル**: 主な編集対象
- **互換性**: 既存挙動・既存ファイルへの影響 (なければ「影響なし」)
```

「大きな変更」の定義は [CLAUDE.md](CLAUDE.md) を参照。小さな変更 (リファクタ・コメント・typo 修正) は記録しない。

---

### 2026-05-19 SafeEnv を待機 agent にも有効化 (PBS 用修正を無効化)

- **意図**: 開発者からの情報「`drp_env.py` の待機分岐にある `self.current_goal_prepare[i] = action_i` (PBS のため、と書かれていた行) をコメントアウトすると安全制御が機能する」を反映。SafeEnv の保護範囲を待機 agent にも広げて衝突を減らす
- **変わること**:
  - `src/main/drp_env/drp_env.py`: 待機分岐の `self.current_goal_prepare[i] = action_i` をコメントアウト。待機 agent の `current_goal` が再び `None` になり、SafeEnv の `if self.current_goal[i] == None:` ガードが機能する
  - `src/main/drp_env/wrapper/safe_marl.py`: 2026-05-13 に入れていた無限ループ対策 (`if joint_action[i] != self.current_start[i]:` チェック) を元に戻した。上記の根本原因が解消したため不要
  - **MARL 系 path planner (QMIX/IQL/VDN/MAA2C) では衝突がさらに減少する見込み**
- **触ったファイル**: `src/main/drp_env/drp_env.py`, `src/main/drp_env/wrapper/safe_marl.py`, `CLAUDE.md` (経緯メモ)
- **互換性 / 注意**:
  - **PBS path planner を使う場合は要再検証**。当該行は元々 PBS のために追加されたものなので、PBS の挙動に影響する可能性あり
  - 再度 PBS を使いたい場合は drp_env.py 該当行をアンコメントすれば元に戻る (= ただし安全制御は再びバイパスされる)
  - 両立は今のところ困難という既知の制約 (詳細は CLAUDE.md「SafeEnv と PBS のトレードオフ」)

---

### 2026-05-22 設計書を `design/` フォルダに集約 + 複数タスク化を独立設計書に切り出し

- **意図**: ルート直下に並んでいた設計書 (`DESIGN_*.md`) を整理し、設計書が増えても見通しが効くようにする。あわせて、ALMA 設計書 §15 にあった「複数タスク保持エージェントへの拡張」は ALMA 採用判断と独立した将来課題なので別ファイルへ分離
- **変わること**:
  - 新規フォルダ `design/` を作成
  - `DESIGN_LaRe_Integration.md` → `design/lare_integration.md` (リネーム + 移動)
  - `DESIGN_ALMA_Integration.md` → `design/alma_integration.md` (リネーム + 移動)
  - 新規 `design/multi_task_agents.md`: ALMA 設計書 §15 の内容をそのまま移植 (Gerkey-Mararic 分類 / 環境改修スコープ / 3 案比較 / 関連研究 / ALMA・LaRe との互換性 / ロードマップ)
  - `design/alma_integration.md` から §15 を削除し、`design/multi_task_agents.md` への参照リンク 1 行に置換
  - `CLAUDE.md` / `MANUAL.md` の設計書パスを新パスに更新
- **触ったファイル**: `design/lare_integration.md` (移動), `design/alma_integration.md` (移動 + §15 削除 + 相互参照リンク更新), `design/multi_task_agents.md` (新規), `CLAUDE.md`, `MANUAL.md`
- **互換性**: 実装・挙動への影響なし (ドキュメント整理のみ)。外部から旧パス `DESIGN_*.md` を参照しているリンクは切れる

---

### 2026-05-15 conda env セットアップスクリプト追加

- **意図**: ldrp conda env を新環境 (別マシン / clone 直後) でも再現できるようにする。手作業の `conda create` + `pip install` 手順を 1 コマンド化
- **変わること**:
  - `requirements.txt` 新規作成: 動作確認済みのバージョン (gym 0.26.2, numpy 1.26.4, torch 2.8.0, networkx 3.2.1, PyYAML 6.0.3, matplotlib 3.8.2) を pin
  - `setup_env.sh` 新規作成: 1 コマンドで conda env "ldrp" を作成 + 依存インストール + ローカル `drp` パッケージを editable で配置 + 動作確認まで実行
  - 使い方: `./setup_env.sh` (新規作成) / `./setup_env.sh --recreate` (作り直し) / `LDRP_ENV_NAME=foo ./setup_env.sh` (env 名変更)
- **触ったファイル**: `requirements.txt` (新規), `setup_env.sh` (新規)
- **互換性**: 既存 ldrp env には影響なし。スクリプト実行時にデフォルトで既存環境は保持される (`--recreate` 指定時のみ削除)

---

### 2026-05-14 タスク割当のデバッグ出力フラグ追加

- **意図**: タスク割当が発生したステップで「候補タスク一覧」「エージェント状態」「割当結果」が見えると、`tp` / `fifo` / `ppo` の挙動確認・原因切り分けが手早くできる
- **変わること**:
  - `src/config/default.yaml` に新キー `debug_task_assign: false` を追加 (デフォルト無効、既存挙動と同一)
  - `True` のとき `TaskManager.assign_task()` が **実際に何かを割当てたステップだけ** stdout に下記フォーマットで出力:

    ```text
    [task_assign] step=<n> method=<tp|fifo|ppo|ppo_v1>
      current_tasklist (idx: [start,goal,time]  status):
        ...
      agents (idx: start->goal  status):
        ...
      result: [..., ..., ...]  (a0<-tX, a1<-none, ...)
    ```

  - 全 -1 (誰にも割当てなかった) のステップではサイレント
- **触ったファイル**: `src/config/default.yaml`, `src/task_assign/task_manager.py`
- **互換性**: `debug_task_assign=false` (デフォルト) のとき挙動は完全に従来通り。`Random`/`TP`/`PPOAgent` 等の個別実装には触っていない

---

### 2026-05-13 SafeEnv の衝突回避 while ループの無限ループバグ修正

- **意図**: `src/main/drp_env/wrapper/safe_marl.py` の `step()` 内 while-do ループが、ある配置 (4 agents の密集衝突など) で **値が変わらないのに `do=True` を立て続けて無限ループ** に陥っていた。`python test.py` が応答停止 → Ctrl+C で Fatal GIL error になる事象の根本原因
- **変わること**:
  - `joint_action[i]` の値が **実際に変わったときのみ** `do = True` をセットするように修正 (2 箇所: 同一目的地衝突回避、正面衝突回避)
  - バグ条件に当たらない既存ケースの挙動は完全互換
  - バグ条件に当たるケースは正しく有限時間で収束する
- **触ったファイル**: `src/main/drp_env/wrapper/safe_marl.py` のみ
- **互換性**: 後方互換 (誤動作していたケースが正しく動くようになるだけ)
- **補足**: 同じバグ構造が Safe-TSL-DBCT (`~/MARL4DRP/drp_env/SafeMarlEnv/env_wrapper.py`) にも存在するが、向こうはタスク機能オフ + 衝突即終了で顕在化しなかった。LDRP では task_flag=True で長時間動かすため引き当てやすかった

---

### 2026-05-10 default.yaml 並び替え + `running_steps: -1` で t_max 連動

- **意図**: 実験のたびに変えるキー (利用フラグ・モデルパス) を上部、固定で良いハイパーパラメータを下部に分けて視認性を上げる。Safe-TSL-DBCT のコンストラクタ並びと同じ思想。さらに、PPO タスク割当の打ち切りを epymarl の MARL 学習と揃える簡易な指定方法を追加
- **変わること**:
  - `src/config/default.yaml` を 3 ゾーン構成に再編 (実験設定 / LaRe 利用設定 / ハイパーパラメータ)。**キー名・値は変更なし、並びだけ変更**
  - `running_steps: -1` を指定すると `runner.py` 内で `src/epymarl/src/config/default.yaml` の `t_max` を読みに行き、その値が `self.max_step` になる
  - `runner.py` に `_resolve_running_steps(args)` ヘルパ追加。yaml 読み込み失敗時は `20_000_000` にフォールバック
- **触ったファイル**: `src/config/default.yaml`, `runner.py`
- **互換性**: 既存の `running_steps: <正の数>` の挙動は完全に同一。`-1` を使った時だけ動作が変わる (新機能)

---

### 2026-05-10 プロジェクト規約整備 (CLAUDE.md / MANUAL.md / subagent)

- **意図**: Claude Code とユーザー間の作業ルールをファイルに固定し、毎回再導出するコストをなくす。LDRP に固有の不変条件 (use_lare_*=False で従来挙動完全一致など) も明文化
- **変わること**:
  - `CLAUDE.md` 新規作成: 開発環境固定 (Python 3.9, ldrp conda env), ファイル地図, 不変条件, ユーザー嗜好 (変更時は意図+変わることをセットで記述), 禁止事項
  - `LaRe_Manual.md` → `MANUAL.md` にリネーム。冒頭をプロジェクト全体の説明書として再フレーム化し、末尾に `更新履歴` セクションを新設。今後の大きな変更はここに追記
  - `.claude/agents/marl4drp-lookup.md` 新規作成: Safe-TSL-DBCT (`/Users/yamaguchiyuushi/MARL4DRP/`) への参照クエリを Haiku モデルで実行する読み取り専用 subagent。LaRe 系の参照作業でメイン context を節約する用途
  - `.gitignore` 緩和: `.claude/` 全体ではなく `settings.local.json` のみ無視に変更。`.claude/agents/` を共有可能に
- **触ったファイル**: `CLAUDE.md` (新規), `MANUAL.md` (LaRe_Manual.md からリネーム), `.claude/agents/marl4drp-lookup.md` (新規), `.gitignore`
- **互換性**: `MANUAL.md` リネームで `LaRe_Manual.md` への外部リンク・引用は壊れる。実コード挙動には影響なし

---

### 2026-05-10 LaRe-Task (System B) 統合 + Path/Task 命名規則確定

- **意図**: タスク割当の PPO が `sum(rew_n)` (経路報酬和) を学習信号にしていた問題を解消。割当決定品質を反映する潜在報酬で置き換える。同時に Path 側の命名も統一
- **変わること**:
  - 新フラグ `use_lare_task` (デフォルト false) でON/OFF。`use_pretrained_lare_task` / `use_finetuning_lare_task` で 4 モード対応
  - 10 因子のタスク評価関数 (本実装で新規設計): pickup_proximity, delivery_efficiency, wait_time_norm, load_balance, idle_assignment, closest_agent_match, queue_drain, low_redirect_cost, urgency_response, batch_assignment_density
  - `runner.py` が `info["lare_task_is_trained"]==True` の時だけ proxy 報酬を PPO に流す。それ以外は従来 `sum(rew_n)`
  - 保存ファイル命名から `LARE` を削除し、`{Safe_}{ALGO}_{PATH|TASK}_{map}_{N}agents_{X.X}M_checkpoint.pth` に統一
  - 非Safe env での先頭 `_` も削除
- **触ったファイル**: `src/lare/task/` 全体, `src/main/drp_env/drp_env.py`, `runner.py`, `src/config/default.yaml`, `test.py`
- **互換性**: 全フラグ false で従来挙動維持。既存命名で保存した checkpoint がある場合、新命名と混在するが読込パス解決は両方を試す

### 2026-05-10 LaRe-Path (System A) 統合

- **意図**: 環境スカラー報酬 (goal/collision/wait/move) を、Safe-TSL-DBCT と同じ 10 因子の潜在報酬モデルで置き換え、IQL/QMIX 系の学習効率を改善
- **変わること**:
  - 新フラグ `use_lare_path` (デフォルト false) で 4 モード (off / scratch / pretrained / finetuning) 切替
  - true 時のみ env.step() 内で proxy 報酬に置換、false 時は従来挙動と完全一致
  - 保存先: `src/lare/path/saved_models/{Safe_}{ALGO}_PATH_..._checkpoint.pth`
- **触ったファイル**: `src/lare/path/` 全体, `src/lare/shared/`, `src/main/drp_env/drp_env.py`, `src/main/drp_env/__init__.py`, `src/config/default.yaml`, `test.py`
- **互換性**: 既存挙動は全フラグ false で維持

---

最終更新: 2026-05-10
