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
pretrained_lare_path_model_name: "Safe_QMIX_PATH_map_8x5_4agents_2.0M_checkpoint.pth"

use_lare_task: true
use_pretrained_lare_task: true
pretrained_lare_task_model_name: "Safe_QMIX_TASK_map_8x5_4agents_2.0M_checkpoint.pth"
```

→ 重みをロード → デコーダは凍結 (`frozen=True`) → 推論専用。

### 2.5 既存モデルから追加学習 (ファインチューニング)

```yaml
use_lare_path: true
use_finetuning_lare_path: true
finetuning_lare_path_model_name: "Safe_QMIX_PATH_map_8x5_4agents_2.0M_checkpoint.pth"
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
| `pretrained_lare_path_model_name` | null | ロード対象パス (絶対 or 相対 or ファイル名のみ) |
| `use_finetuning_lare_path` | false | Finetuning モードに入る |
| `finetuning_lare_path_model_name` | null | ロード対象パス |
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
    use_pretrained_lare_path=False, pretrained_lare_path_model_name=None,
    use_finetuning_lare_path=False, finetuning_lare_path_model_name=None,
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
pretrained_lare_path_model_name: "Safe_QMIX_PATH_map_8x5_4agents_2.0M_checkpoint.pth"
use_lare_task: false
```

### D. 既存 Path をファインチューニング + Task は新規学習
```yaml
use_lare_path: true
use_finetuning_lare_path: true
finetuning_lare_path_model_name: "Safe_QMIX_PATH_map_8x5_4agents_1.0M_checkpoint.pth"
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

### 2026-11-14 LaRe-Path の評価関数を番号で切り替える (`lare_path_factor_version`, 版 1〜3)

- **意図**: LaRe-Path の評価関数 (因子) を、DRP 用の 10 因子 (MARL4DRP の評価関数) と、LLM が LDRP 用に生成した評価関数とで比べられるようにする。評価関数ごとに前提の観測の並びが違うので、観測の作り方と評価関数を 1 組にして番号で選ぶ
- **変わること**:
  - 新しい設定キー `lare_path_factor_version` (env_args / `gym.make` の引数、既定 1)
    - `1`: DRP 用の 10 因子 (今までと同じ)
    - `2`: LLM 生成の評価関数 v1 (`src/lare/path/factors_ldrp_v1.py`) + 19 区間の観測 (`obs_ldrp.py`)
    - `3`: LLM 生成の評価関数 v2 (`src/lare/path/factors_ldrp_v2.py`) + 長さ 7N + 4NK + 6K + 26 の観測 (`obs_ldrp_v2.py`)。評価関数は辺の長さの表 (adj) も使う
    - 登録に無い番号は `gym.make` の時点で `ValueError: unknown lare_path_factor_version` で止まる
  - **LaRe-Path の保存ファイル名**: 版 2 以降は `PATH` の後ろに版が入る (例: `QMIX_PATH_F2_Safe_map_8x5_5agents_1.0M_checkpoint.pth`)。版 1 は今までどおり
  - 保存するファイルに `factor_version` を書く。読み込み時に版を照合し、違う版のデコーダは `ValueError: factor_version mismatch` で止まる (版の書かれていない今までのファイル・MARL4DRP 形式は版 1 として扱う)
  - 版 2・3 の学習済みデコーダはまだ無いので、`use_pretrained_lare_path=False`・`use_finetuning_lare_path=False` (scratch) で回す。env の既定は pretrained が ON なので、指定し忘れると起動時に止まる
  - SafeEnv が機体ごとに「このステップ動きを止めたか」を env (`_lare_override`) に書くようになった (版 2・3 の観測で使う)
- **触ったファイル**: 新規 `src/lare/path/factor_versions.py` / `obs_ldrp.py` / `obs_ldrp_v2.py` / `factors_ldrp_v1.py` / `factors_ldrp_v2.py`。変更 `src/lare/path/lare_path_module.py`、`src/main/drp_env/drp_env.py`、`src/main/drp_env/wrapper/safe_marl.py`、`train.py` (使い方のコメントのみ)。設計は [design/lare_factor_versions.md](design/lare_factor_versions.md)
- **互換性**: `lare_path_factor_version` を指定しない (= 1) ときの因子・報酬・保存ファイル名・読み込みは今までと同じ (`tools/lare_factor_smoke.py` で変更前と同じ報酬の合計を確認)。保存ファイルに `factor_version` のキーが 1 つ増える (今までのコードで読んでも無視される)。ダッシュボードは版 2・3 の run を `safe F2` / `safe F3` として別の条件に数える

---

### 2026-10-16 LaRe のモデルが無い・読めないときは、学習を始めずに止める

- **意図**: 指定した pretrained / finetuning の LaRe モデルが無い・読めないと、今までは警告を出すだけで **LaRe を切って (環境の報酬で) 学習が進んだ**。読めないときは scratch に切り替わった。config.json には「LaRe あり (事前学習あり)」と残るので、後から見分けられない。2026-10-15 に GPU で、未配置のモデルを指定した 4 本がこうなっていた
- **変わること**:
  - `use_pretrained_lare_path` / `use_finetuning_lare_path` で指定したモデルが `src/lare/path/models/` に無いと、`gym.make` の時点で `FileNotFoundError` で止まる (学習が始まらない)
  - モデルはあるが読めない (壊れている・デコーダの形が違う) と `ValueError: [LaRe-Path][PRETRAINED] Load failed: …` で止まる (今までは「falling back to scratch」の警告だけ)
  - LaRe-Task (`use_pretrained_lare_task` / `use_finetuning_lare_task`) も同じ
  - それ以外の想定外の初期化の失敗は、今までどおり「Failed to initialize, falling back to env reward」で環境の報酬に戻る
- **触ったファイル**: `src/main/drp_env/drp_env.py` (`_init_lare_path` / `_load_lare_path_weights` / `_init_lare_task` / `_load_lare_task_weights`)。設計は [design/lare_factor_versions.md §4.5](design/lare_factor_versions.md)
- **互換性**: **破壊的変更**。モデル名の打ち間違い・置き忘れがある train.py は、今までは (LaRe なしで) 動いていたが、これからは起動直後に止まる。モデルが読める run・scratch・LaRe なしの挙動は変わらない。GPU 版 (LDRP_GPU) は別に直す必要がある

---

### 2026-10-14 学習時の到着率の範囲を、評価用モデルの名前 (method_tag) に入れる

- **意図**: 学習時の到着率の範囲を変えた run (tools/plans/plan_arrival_range.md) が、既定の範囲の run と同じ名前の列 (`map_8x5_7_mappo_safe_base_seed5.th` など) に seed の番号違いで入っていた。番号の付き方が条件ごとにばらばらで、run.py で範囲をそろえて評価できず、評価結果も別の範囲が同じ条件の seed として平均されてしまう
- **変わること**:
  - collect_runs.py が決める method_tag に、既定 U(0.01, 0.10) 以外で学習した run だけ範囲を足す (例: `safe` → `safe_p0.01-0.15`)。評価用モデルの名前は `map_8x5_7_mappo_safe_p0.01-0.15_base_seed0.th` になり、範囲ごとに seed0 から数える
  - 評価の条件名にもタグが入る (`…/safe_mappo_safe_p0.01-0.15_base_tp_…`) ので、集計・ダッシュボードの評価タブで別の条件として扱われる
  - run.py では `method_tag = ["safe", "safe_p0.01-0.15", "safe_p0.01-0.2", "safe_p0.05-0.2"]` のように並べて評価する (タグの数字は `%g` 表記なので 0.20 は `0.2`)
  - 保管リポジトリ (`~/LDRP_models/arrival_range/`) の 36 本を新しい名前で置き直し、評価用の置き場 (`src/all_policy/models/safe/` と `src/task_assign/models/safe/`) にも設置した
- **触ったファイル**: `tools/collect_runs.py` (`derive`)
- **互換性**: 既定の範囲で学習した run (AAMAS / reassign の計画) の名前は変わらない。保管リポジトリの古い名前 (`…_seed5.th` など、到着率範囲の run) のファイルと manifest.jsonl の行は消した

---

### 2026-10-07 評価のログを回ごとのフォルダに分けて残す

- **意図**: run.py はログを同じ名前に書いていたので、同じ条件・同じ seed を回し直すと前の結果が消えていた。また、ログの名前からは、どの回 (どの設定) で回したものかが分からなかった
- **変わること**:
  - run.py は、起動するたびに `logs/<起動日時 (秒まで)>_<回の名前>/` のフォルダを作り、その下に今までと同じ形 (`map_8x5/safe/mappo/7agent/…_seed0.txt`) でログを書く
  - 回の名前は、run.py の `run_tag` に書けばそれを、空なら一覧から自動で作る (例: `8x5_N5-7-10_mappo_tp-ppo_safe_trainReassign-off-on_p0.01-0.2-mmpp_execReassign-on-off`)
  - 回のフォルダに `batch_info.txt` (run.py の一覧・model_seed・git のコミット・コマンド数)、`summary.csv` (この回だけの集計)、`summary_all.csv` (この回が終わった時点の全体の集計の写し) を残す
  - `results/summary.csv` (ダッシュボードの評価タブが読む) は、今までどおり `logs/` の下の全部のログから作り直す。同じ条件・同じ seed は新しい方を使う
  - 上書きの守り: 同じ名前のフォルダがあれば止まり、ログは「新しく作るときだけ書ける」開き方 (`"x"`) にした
  - aggregate.py の `duplicate entry` の警告は、「古い結果を N 件置き換えた」の 1 行にまとめた
- **触ったファイル**: `run.py`, `aggregate.py`
- **互換性**: 今ある日時無しのログ (`logs/map_8x5/…`) はそのまま残り、集計にも使われる (更新時刻が古いので、新しい評価が優先される)。ログを直接読んでいるスクリプトがあれば、置き場所が 1 段深くなる点に注意

---

### 2026-10-02 評価のタスク到着を複数指定して 1 回で回す (確率固定 + MMPP)

- **意図**: 評価 (test.py) が到着のモードと確率を env に渡しておらず、env の既定 `task_arrival="fixed"` (毎ステップ 1 件、待ち行列は常に満杯) で評価していた。学習 (Bernoulli p=0.01〜0.10 / MMPP) と大きく違う負荷だった。学習と同じ範囲の到着で評価し、到着ごとの性能を出せるようにする
- **変わること**:
  - run.py に一覧 `task_arrival` (例 `"bernoulli:0.02"`, `"mmpp"`, `"fixed"`) と `allow_reassign` (`"False"` / `"True"`) を追加。run.py はこれを `arrival=a,b,...` / `allow_reassign=...` の 1 つの引数にまとめて test.py に渡し、**test.py 1 回で「到着 × 再割当の有無」の全組み合わせを順に回す**。ログは 1 条件 × 1 seed につき 1 本のまま (中に複数の `[RESULT]` 行)
  - test.py が引数 `arrival=` / `allow_reassign=` を受け付ける。引数なしのときは default.yaml の `eval_arrivals` / `eval_allow_reassign` を使う
  - default.yaml に新キー `eval_arrivals`, `eval_allow_reassign`, `task_arrival`, `task_density`, `task_p_high`, `task_p_low`, `task_switch_prob`。MMPP の値 (0.10 / 0.01 / 0.01) は学習と同じで、run.py からは変えない
  - 条件名 (`condition`) の末尾に到着の目印が付く: `_bern{p}` (例 `_bern0.05`) / `_mmpp`。再割当ありは従来どおり `_envreassign`
  - `bernoulli` の確率の書き忘れは、回し始める前に `ValueError` で止まる
- **触ったファイル**: `src/config/default.yaml`, `test.py`, `runner.py` (`_condition_id`), `run.py`, `tools/eval_report.py` (条件名の分解に `arrival` / `dynamic` を追加)
- **互換性**: **`fixed` は目印を付けないので、従来の条件名・ログ名と同じ**。ただし test.py を引数なしで回したときの既定は、従来の「fixed × 再割当なし/あり の 2 周」から **default.yaml の `eval_arrivals` × `eval_allow_reassign`** に変わる。到着の数だけ評価時間が増える (1 条件約 16 分の実測 × 組み合わせ数)。同じ到着ならシナリオは条件間で同一 (`episode_seed_base`)。test.py で PPO 割当を学習するとき (`train_task_assigner=True`) は、到着は `task_arrival`、再割当は `allow_reassign_before_pickup` の 1 通りだけ。詳細は [design/multi_seed_eval.md §11](design/multi_seed_eval.md#11-評価時のタスク到着を切り替える-確率固定-3-種--mmpp)

---

### 2026-09-30 PPO 割当の評価で学習済み割当方策を読む / dynamic をモデル名で判定

- **意図**: `task_assigner=ppo` で評価すると、経路方策は TP と組で学習したモデル (`..._safe_base_seedK.th`) を読み、割当方策は何も読まずに乱数初期化の重みで割り当てていた。どちらもエラーにならず、別条件の結果が黙って出ていた。あわせて、動的台数で評価するかを default.yaml ではなく「モデルが動的台数で学習されたか」で決めたい
- **変わること**:
  - 評価のモデル名 (stem) が `tools/collect_runs.py` の install 名と同じ規則になった: `{map}_{N}_{planner}[_{method_tag}][_ppo][_dyn]_{reassign}_seed{K}.th`。`_ppo` は `task_assigner=ppo` のとき、`_dyn` は `use_dynamic_agents=True` のときだけ付く
  - `task_assigner=ppo` の評価 (`training=False`) で、割当方策を `src/task_assign/models/safe/` から経路方策と**同じ名前・同じ `model_seed`** で読む。無ければ `FileNotFoundError` で止まる。`ppo_task_checkpoint_path` を指定したときはそちらが優先
  - `[RESULT]` 行に `task_model=` (読んだ割当方策のファイル名) が出る。aggregate.py はこれを指標でなくメタ情報として扱う
  - test.py が `use_dynamic_agents=True/False` 引数を受け付ける (default.yaml より優先)
  - run.py は各条件で `_dyn` 無し / 有りの両方の名前を探し、モデルがある方だけ `use_dynamic_agents=...` を付けて回す。`model_seed` を数値で固定したときも、ファイルがある方だけ回す
  - dynamic の評価は条件名 (`condition`) に `_dyn`、run.py のログ名に `_dyn` が付く
- **触ったファイル**: `src/all_policy/policy.py` (`model_stem` の引数追加, `model_stem_form_args`, `PATH_MODELS_DIR` / `TASK_MODELS_DIR`), `src/task_assign/task_policy/ppo.py` (`load_model_file`), `runner.py`, `test.py`, `run.py`, `aggregate.py`
- **互換性**: **TP / FIFO かつ静的台数の評価は、読むファイル名・条件名・ログ名とも従来どおり**。PPO 条件と dynamic 条件は、以前のログとは条件名もモデルも違う (以前の PPO 評価の結果は誤ったモデルでの結果なので使わない)。policy.py の定数 `MODELS_DIR` は `PATH_MODELS_DIR` に改名 (外部から import していた場合は `ImportError`)。動的台数の他の設定 (`min_active_agents` 等) は従来どおり default.yaml から取るので、学習時と揃える。詳細は [design/multi_seed_eval.md §9](design/multi_seed_eval.md#9-次の実装-学習済みタスク割当方策-ppo-の評価モード)

---

### 2026-09-27 台数可変時の観測: 非アクティブ機を占有から外す

- **意図**: `use_dynamic_agents=True` のとき、非アクティブ機が観測 (`onehot_fov` の塞がり -1) にだけ占有として残っていた。衝突判定からは除外されているので実際には通れる場所が塞がって見え、さらに非アクティブ機が全員同じステーションに重なることで観測の判定式 (「1 ノードに立てるのは 1 台」前提) が崩れていた
- **変わること**: dynamic 条件で、非アクティブ機がいるときだけ他機から見た塞がり (-1) が減る。ステーションの隣・ステーション上・帰投中の 3 場面で誤って「塞がり」になっていたものが「空き」になる。ステーションに稼働機が本当にいる場合は従来どおり塞がり。観測の次元は変わらない
- **触ったファイル**: `src/main/drp_env/state_repre/wrapper/fov_wrapper.py` (`neighbor_filter_obs`)
- **互換性**: **台数固定 (`use_dynamic_agents=False`) では完全に従来どおり** (8,000 回比較で不一致 0)。dynamic でも全機稼働なら変わらない。既存モデルは読み込めるが、**修正前に学習した dynamic のモデル・run とは観測の意味が違う**ので比較には使わない。GPU 版リポ (LDRP_GPU) には別途同じ変更が必要。詳細は [design/dynamic_agent_count.md](design/dynamic_agent_count.md) の「観測から非アクティブ機を外した」

---

### 2026-09-08 学習 run の収集・監視ツール `tools/` 追加 — PC 版から取り込み

- **意図**: このマシンで走らせている学習の進捗・完遂状態を、集約マシン (Mac) 側から機械的に把握できるようにする。「設定した `t_max` まで完遂したか」「途中でプロセスが死んでいないか」を手作業で確認しなくて済ませる。PC 版コミット `64aaa84` / `8d3e834` の取り込み
- **変わること**:
  - 新ツール `tools/collect_runs.py` (run 収集・状態判定)、`tools/eval_report.py` (評価結果の集計)、`tools/plan.py` (実験計画のパース)、`tools/dashboard/` (ローカル Web UI) を追加。**学習・実行系 (`train.py` / `test.py` / `run.py` / `src/`) には一切触っていない** — sacred が既に吐いている `config.json` / `run.json` / `cout.txt` / `metrics.json` と `results/summary.csv` を読むだけ
  - このマシン (GPU 機) は **「収集される側」** に該当する。手順は [tools/SETUP_export_machine.md](tools/SETUP_export_machine.md) を参照。`python3 tools/collect_runs.py --hosts local --export DIR` で共有フォルダに書き出す運用 (drop モード)
  - 状態判定: `OK` (t_max 完遂) / `SHORT` (COMPLETED だが t_max 未達) / `RUN` (学習中) / `STALL` (heartbeat 停止 = プロセス落ち) / `FAIL`
  - `cout.txt` が 0 byte の環境 (本 GPU 機が該当。`src/epymarl/src/main.py` の `CAPTURE_MODE = "no"` により sacred が stdout を捕捉しないため) では `metrics.json` の末尾 256KB から最後の `t_env` を拾う
  - リモート側の要件は `python3` (標準ライブラリのみ)
  - 接続先 `tools/collect_config.yaml` と実験計画 `tools/plan.md` は `.gitignore` 済み (見本は `.example` として公開)。**このマシンでは `plan.md` を触らない / `--notion` `--publish-models` を使わない** (実験計画は集約マシンが master)
- **触ったファイル**: 新規 `tools/` 一式, `design/run_collector.md` / 改修 `.gitignore` (`tools/.run_cache.jsonl`, `models_inbox/`, `tools/.notion_token`, `tools/collect_config.yaml`, `tools/plan.md` を追加)
- **互換性**: 影響なし (既存コードへの変更なし)

---

### 2026-08-20 階層強化学習: 経路方策と PPO タスク割当器の同時学習 — PC 版から取り込み

- **意図**: これまで経路方策 (epymarl) とタスク割当器 (PPO) を別々に学習していたが、両者を 1 プロセスで同時に学習できるようにする。PC 版コミット `5cc852f Hierarchical Reinforcement Learning` / `9659583 modify` の取り込み
- **変わること**:
  - **新設定キー 2 つ** (`src/epymarl/src/config/default.yaml`):

    | キー | デフォルト | 意味 |
    |---|---|---|
    | `train_task_assigner` | `False` | `True` で PPO タスク割当器を経路方策と同時に学習する |
    | `task_num` | `10` | タスク割当器が扱うタスクスロット数 |

  - **runner の強制切替**: `train_task_assigner=True` かつ `runner != "parallel"` のとき、起動時に `runner="parallel"` へ自動変更される（ログに `[joint] runner='episode' -> change to parallel` が出る）。QMIX 等の episode runner が既定のアルゴでも parallel になる
  - **タスク割当の経路**: env worker が `get_task_state()` を返し、`ParallelRunner` が `PPOAgent.assign_task_from_state()` でタスクを決めて `{"pass": ..., "task": ...}` の dict action として env に送る。報酬は LaRe-Task が学習済みなら `lare_task_proxy_reward`、そうでなければ env 報酬を PPO バッファに積む
  - **モデル保存先の変更 (破壊的)**: 保存先が `results/models/{token}/{t}/` から **`results/models/{token}/{t}/path/`** に変わる。`train_task_assigner=True` のときは併せて **`{t}/task/`** に PPO 割当器が保存される。**この変更は `train_task_assigner` の値に関係なく常に適用される**ので、`.th` を `src/all_policy/models/safe/` へコピーする際のコピー元パスが変わる
  - `test.py` / `run.py` の評価経路は従来どおり（`test.py` は `PPOAgent.assign_task(env)` を使い、こちらの API は維持されている）
- **触ったファイル**: `src/epymarl/src/config/default.yaml`, `src/epymarl/src/envs/__init__.py`, `src/epymarl/src/run.py`, `src/epymarl/src/runners/parallel_runner.py`, `src/main/drp_env/drp_env.py`, `src/task_assign/task_policy/ppo.py`, `train.py`
- **互換性**:
  - `train_task_assigner` のデフォルトは `False` で、この場合タスク割当は従来どおり env 内蔵の TP が行う。学習の数値挙動は変わらない (3agent map_5x4 で smoke 検証済み)
  - **モデル保存パスの変更だけは既定でも効く**（上記）。既存の保存済みモデルの配置は変わらないが、今後の学習で出力される階層が 1 段深くなる
  - `checkpoint_path` からの再開は新旧どちらの形式でも動く。`{t}/path/agent.th` があればそれを、なければ従来どおり `{t}/` を読む

---

### 2026-08-18 動的フリートサイジング (`use_dynamic_agents`) — PC 版から取り込み

- **意図**: 需要に対して配備台数が過剰なとき idle が支配的になる問題に対応するため、稼働エージェント台数を制御できるようにする。PC 版コミット `c8693f5 dynamic fleet sizing` の取り込み
- **現状の実装範囲**: **エピソード開始時に稼働台数を決め、そのエピソード中は台数固定**。エピソード中の増減 (稼働開始・station への帰還) は受け皿だけがコードに入っており、まだ動かない (下記「未接続の部分」参照)
- **変わること**:
  - **新設定キー 5 つ** (`gymma.yaml` の `env_args` / `DrpEnv.__init__`):

    | キー | デフォルト | 意味 |
    |---|---|---|
    | `use_dynamic_agents` | `False` | 動的フリートの ON/OFF。`True` にする場合は `task_flag=True` と station ノードが必須 (`assert`) |
    | `randomize_initial_active` | `False` | `True` でエピソード毎に初期稼働台数を `[min_active_agents, max_active_agents]` からランダム抽選 |
    | `min_active_agents` | `2` (yaml) / `1` (env 引数) | ランダム抽選の下限 |
    | `max_active_agents` | `null` (= 全台) | ランダム抽選の上限 |
    | `initial_active_num` | `null` (= 全台) | `randomize_initial_active=False` のときの初期稼働台数 |

  - **非稼働 (`active[i] == False`) エージェントの扱い**: station ノードに配置され、衝突判定 (`collision_detect` / `get_collision_agents`)・SafeEnv の安全制御・タスク割当 (TP / Random / PPO) のすべてから除外される。選択可能行動もその場待機の 1 つだけになる
  - **未接続の部分 (エピソード中の増減)**: `drp_env.py` には稼働開始 (非稼働エージェントにタスクが割り当てられたら 1 ステップ最大 1 台まで復帰) と帰還 (`task_assign[i] = -2` で最寄り station をゴールにし、到着で非稼働化) の処理が入っているが、**これを駆動する割当器がまだない**。TP / Random / PPO はいずれも非稼働・帰還中のエージェントを skip し、`-2` を返す実装もどこにも存在しないため、現状これらの分岐は通らない
  - **station ノードの定義**: `map/<map名>/node.csv` の 5 列目 `station` が `1` の行。全 13 マップに 1 個ずつ定義済み
  - **設定の入り口**: 学習系 (train.py → epymarl) は `src/epymarl/src/config/envs/gymma.yaml`、評価系 (test.py / run.py) は `src/config/default.yaml`。どちらも同じ 5 キーを持ち、test.py が `gym.make()` へ転送する
- **触ったファイル**: `src/main/drp_env/drp_env.py`, `src/main/drp_env/EE_map.py`, `src/main/drp_env/wrapper/safe_marl.py`, `src/task_assign/task_policy/{ppo,random,tp}.py`, `src/epymarl/src/config/envs/gymma.yaml`, `src/config/default.yaml`, `test.py`, `train.py`
- **互換性**: `use_dynamic_agents` のデフォルトが `False` で、この場合すべての分岐が従来コードと同一パスを通るため既存の学習・評価には影響なし。既存モデル (`.th` / `.pth`) もそのまま使用可能。observation の形状も変わらない

---

### 2026-06-26 MAT-Dec (Multi-Agent Transformer, 分散実行版) アルゴリズム追加

- **意図**: epymarl に Transformer ベースの方策 MAT-Dec を追加し、QMIX/IQL/MAPPO 等に加えて選択可能にする。LDRP_GPU リポの MAT 実装を CPU 開発機向けに移植 (GPU 専用の `multiprocessing spawn` 化は除外し、既定の fork のまま)
- **変わること**:
  - 新アルゴ `--config=mat_dec` で学習可能。専用 `mat_mac` / `mat_learner` / `mlp_mat` agent / `mat_critic` (Transformer Encoder) を新規追加
  - 新設定キー `extra_in_buffer` (デフォルト `[]`) を `default.yaml` に追加。`["log_probs","values"]` 指定時のみ buffer scheme に該当テンソルを格納 (MAT が PPO 更新で使用)。未指定の既存アルゴは従来通り
  - 報酬正規化 `PopArt` を `components/standarize_stream.py` に追加 (MAT の `use_popart` で使用)
  - **MAC API 変更**: `select_actions()` の戻り値を `chosen_actions` → `(chosen_actions, extra_returns)` の 2-tuple に統一 (basic / non_shared / maddpg / mat の全 MAC)。呼び出し元は epymarl の 2 runner のみで、LDRP 評価系 (test.py / all_policy / runner.py) は別経路のため影響なし
- **触ったファイル**: 新規 `controllers/mat_controller.py`, `learners/mat_learner.py`, `modules/agents/mlp_mat_agent.py`, `modules/critics/mat.py`, `config/algs/mat_dec.yaml` / 改修 各 `__init__.py` (4 registry), `config/default.yaml`, `run.py`, `components/episode_buffer.py`, `components/standarize_stream.py`, `runners/parallel_runner.py`, `runners/episode_runner.py`, `controllers/{basic,non_shared,maddpg}_controller.py`
- **互換性**: `extra_in_buffer` 既定 `[]` のため QMIX/IQL/VDN/MAA2C/MAPPO/MADDPG は従来挙動を維持 (smoke 検証済み)。GPU 機の `multiprocessing spawn` 化と CUDA 明示転送は本機 (CPU) では不要なため未移植 — GPU 機へ展開する場合は別途追加が必要

---

### 2026-05-31 LaRe-Path / LaRe-Task の学習スケジュールを MARL4DRP の evaluation_period 方式に整合

- **意図**: 従来 LDRP の `_update()` は `update_freq=3` ごとに「同じ batch を `train_epochs` 回繰り返す」構造で、MARL4DRP の参照実装 (`reward_model_update_freq=128`, `evaluation_episodes=50`) と挙動が一致しなかった (LDRP は overfit 寄り, burst タイミングも違う). 学習動態を MARL4DRP に合わせて再現性を取りたいという要望に対応
- **変わること**:
  - **デフォルト学習スケジュール**: `update_freq=3 → 128`, `train_epochs=1 → 50` (Path / Task 両方)
  - **`_update()` の意味変更**: 1 回呼び出し = 1 batch sample + 1 optimizer step (旧: 1 batch × `train_epochs` epoch). 内部 for ループ撤去
  - **`end_episode()` に評価期間ステートマシン追加**: `update_freq` ep ごとに評価期間を開始 → 連続する `train_epochs` ep の末尾で毎回 `_update()` を 1 回ずつ呼ぶ → カウンタ到達で評価期間終了. MARL4DRP `drp_env.py:1340-1382` の構造と等価
  - **数値挙動**: 各 update は **fresh batch** から行われるので、サンプル多様性が増し、同一 batch 過学習は起きなくなる. 1 トリガあたりの総 step 数は 50 で同じ
  - **autosave throttle**: `_update()` 内で `save_freq_steps` の判定を行うため、評価期間中も 0.5M ステップ単位で間引きされる (= 仕様変更なし)
- **触ったファイル**: `src/lare/path/lare_path_module.py`, `src/lare/task/lare_task_module.py`, `src/main/drp_env/drp_env.py`, `src/config/default.yaml`
- **互換性**:
  - **公開フラグ名・yaml キーは増減なし** (既存の `lare_*_update_freq`, dataclass の `train_epochs` を再解釈)
  - **既存の pretrained 用 .pth は変更なしでロード可能** (decoder/transformer の重み形状は変えていない)
  - **既存の `update_freq=3` を上書き指定していた env_args / yaml は引き続き動く** が、その値は新スキームでは「評価期間トリガの周期 (ep)」を意味する点に注意. 完全に旧挙動を再現したい場合は `train_epochs=1` も合わせて指定する必要あり
  - 設計詳細は [design/lare_path_reward_flow.md §7](design/lare_path_reward_flow.md) を参照

---

### 2026-05-19 モデル保存先を `checkpoints/` と `models/` に分離 + 0.5M ステップごとに保存

- **意図**: ユーザー要望「学習で保存されたモデルと、これから使うために選んだモデルのフォルダを分けたい」。同じディレクトリに大量蓄積されると使いたいモデルを探すのが大変なため。あわせて autosave 頻度を学習頻度 (= update_freq) から切り離し、0.5M 累積ステップごとに 1 回保存するように間引く
- **変わること**:
  - **ディレクトリ構成 (固定)**: `src/lare/{path,task}/checkpoints/` (= autosave 出力先) と `src/lare/{path,task}/models/` (= pretrained / finetuning ロード元) を新設
  - **`.gitignore`**: `checkpoints/` は git 外, `models/` は `**/models/` 除外パターンの例外で **git 公開** 可能に。整理済みモデルを共有できる
  - **`src/lare/{path,task}/models/.gitkeep`**: 空ディレクトリを git に乗せるため追加
  - **autosave 保存先**: 旧 `saved_models/` → 新 `checkpoints/` に切替 (固定パス, yaml 指定不要)
  - **pretrained / finetuning の解決順**: `models/` を 4 番目に追加 (= 絶対 → cwd 相対 → repo 相対 → **models/** → checkpoints/ → 旧 saved_models/). `models/` にコピーしておけばファイル名だけ指定すれば自動解決
  - **保存頻度の throttle**: 新キー `lare_*_save_freq_steps` (デフォ 500000 = 0.5M ステップごと). 学習自体は `update_freq` 通り続行されるが、ファイル書き込みは 0.5M ステップに 1 回に間引かれる
  - **後方互換**: 旧 `lare_*_save_dir: <path>` を指定すれば従来通り load の解決パスに含まれる. autosave 出力先には反映されないので、autosave も従来パスに出したい場合は要 yaml で `lare_*_autosave_path` を絶対パスで指定
- **触ったファイル**: `src/lare/path/lare_path_module.py`, `src/lare/task/lare_task_module.py`, `src/main/drp_env/drp_env.py`, `src/config/default.yaml`, `test.py`, `.gitignore`, `src/lare/{path,task}/models/.gitkeep` (新規), `CLAUDE.md`
- **互換性**: 既存 `lare_*_save_dir` 設定の挙動は load 側でのみ生き残る (autosave は無視される). 既存 `saved_models/` 内のモデルファイルは引き続きロード可能

#### 推奨ワークフロー

```bash
# 1. 学習 (autosave 有効化)
#    → src/lare/path/checkpoints/Safe_QMIX_PATH_..._X.XM_checkpoint.pth に 0.5M ごとに蓄積
python test.py   # use_lare_path: true, lare_path_autosave: true で起動

# 2. 良いモデルを選別
cp src/lare/path/checkpoints/Safe_QMIX_PATH_map_8x5_4agents_2.0M_checkpoint.pth \
   src/lare/path/models/best_v1.pth

# 3. 利用 (pretrained or finetuning)
#    default.yaml で:
#      use_pretrained_lare_path: true
#      pretrained_lare_path_model_name: "best_v1"     # ← ファイル名だけで models/ から自動解決
```

---

### 2026-05-19 `pbs_mode` フラグで SafeEnv ⇔ PBS を排他切替可能に

- **意図**: 直前のコミットで「PBS 用の待機分岐代入を無効化 → SafeEnv 復活」を一方的に選んだが、PBS 使用時にも互換動作が必要になる場面に備えてフラグ化。`path_planner` に応じて自動切替するので追加の手作業は不要
- **変わること**:
  - `src/main/drp_env/drp_env.py`: `__init__` に `pbs_mode=False` パラメータ追加。待機分岐の `self.current_goal_prepare[i] = action_i` を `if self.pbs_mode:` でガード
  - `test.py`: `config.path_planner == "pbs"` のとき自動で `pbs_mode=True` を `gym.make()` に渡す。それ以外は `False` (= 安全制御優先, デフォルト)
  - `CLAUDE.md`: 「SafeEnv と PBS のトレードオフ」セクションを「切替可能」に書き換え。挙動表を追加
- **触ったファイル**: `src/main/drp_env/drp_env.py`, `test.py`, `CLAUDE.md`
- **互換性**:
  - QMIX/IQL/VDN/MAA2C 利用時: 何も変わらず (= 安全制御 ON、デフォルト false)
  - PBS 利用時: 直前のコミットで PBS の挙動が変わってしまった状態から、**PBS 互換動作が戻る**
  - 既存ユーザーが test.py を別の方法で呼んでいる場合、`pbs_mode` を kwargs で明示するか、yaml にフラグを追加して制御することも可能

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
