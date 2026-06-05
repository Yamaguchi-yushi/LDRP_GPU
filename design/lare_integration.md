# 設計書：潜在報酬（LaRe）を LDRP に導入した経路計画・タスク割り当て改善

**作成日:** 2026-04-12  
**対象リポジトリ:**
- 導入先: https://github.com/kaji-ou/LDRP（本リポジトリ）
- 参照実装: https://github.com/Yamaguchi-yushi/Safe-TSL-DBCT
- 潜在報酬の原著: https://github.com/thu-rllab/LaRe（AAAI-2025）

---

## 目次

1. [背景・動機](#1-背景動機)
2. [二つの報酬設計の役割分離（核心）](#2-二つの報酬設計の役割分離核心)
3. [LaRe の概要](#3-lare-の概要)
4. [設計方針](#4-設計方針)
5. [全体アーキテクチャ](#5-全体アーキテクチャ)
6. [System A: LaRe-Path（経路計画用）](#6-system-a-lare-path経路計画用)
   - 6-1. 役割と評価対象
   - 6-2. 潜在因子エンコーダ φ_path(s,a)
   - 6-3. 報酬デコーダ fψ_path
   - 6-4. 時系列クレジット（AREL Transformer）
   - 6-5. epymarl への統合
7. [System B: LaRe-Task（タスク割り当て用）](#7-system-b-lare-taskタスク割り当て用)
   - 7-1. 役割と評価対象
   - 7-2. 潜在因子エンコーダ φ_task(s,a_task)
   - 7-3. 報酬デコーダ fψ_task
   - 7-4. エージェント間クレジット（MARD / Shapley）
   - 7-5. PPO タスク割り当て器への統合
8. [二システムの相互作用](#8-二システムの相互作用)
9. [ファイル構成](#9-ファイル構成)
10. [実装手順](#10-実装手順)
11. [パラメータ一覧](#11-パラメータ一覧)
12. [評価指標・実験設計](#12-評価指標実験設計)
13. [懸念点・リスク](#13-懸念点リスク)

---

## 1. 背景・動機

### 現状の報酬設計

LDRP の環境報酬は 4 要素からなる単一スカラー：

```
r_t = +100（ゴール到達）/ −100（衝突）/ −10（待機）/ −1（移動）
```

この報酬は **経路計画（IQL/QMIX）** と **タスク割り当て（PPO）** の両方に対して共通で使われているが，それぞれの役割に対して次の問題がある：

| 問題 | 経路計画への影響 | タスク割り当てへの影響 |
|------|----------------|----------------------|
| **因子の単純さ** | 移動効率・混雑回避・協調が評価されない | タスクの品質（近さ・待ち時間・均衡）が評価されない |
| **クレジット割り当て問題** | どのステップの行動がゴール到達に寄与したか不明 | どの割り当て決定がタスク完了に貢献したか不明 |
| **報酬の混在** | — | タスク割り当ての PPO が `sum(rew_n)` を使うが，これは経路報酬であり，割り当て品質を正しく反映しない |
| **希薄報酬** | ゴール到達まで待つ | タスク完了まで待つ（さらに疎） |

### 目標

**経路計画用** と **タスク割り当て用** の潜在報酬システムをそれぞれ独立して設計し，互いの学習干渉を排除した上で，両者の精度を同時に向上させる．

---

## 2. 二つの報酬設計の役割分離（核心）

これが本設計の最も重要な点である．経路計画とタスク割り当ては **根本的に異なる決定問題** であり，同一の報酬で学習させることは両者の精度を制限する．

### 比較表

| 項目 | **経路計画（System A）** | **タスク割り当て（System B）** |
|------|----------------------|----------------------------|
| **決定者** | IQL / QMIX（epymarl） | PPO（task_policy/ppo.py） |
| **決定の頻度** | 毎ステップ（高頻度，T 回/エピソード） | エージェントがアイドル時のみ（低頻度，K 回/エピソード，K ≪ T） |
| **決定の内容** | 各エージェントが次に進むノード | 空きエージェントにどのタスクを割り当てるか |
| **評価の視点** | 個エージェントの移動効率・安全性・協調 | システム全体のタスク処理効率・均衡 |
| **クレジット問題の種類** | **時系列クレジット**: どのステップが ゴール到達に寄与したか | **割り当てクレジット**: どの割り当て決定がタスク完了に貢献したか |
| **現状の報酬** | `r_t = goal/collision/wait/move`（ステップ単位） | `sum(rew_n)`（環境報酬の和 → 経路報酬と混在） |
| **学習ターゲット** | エピソード総移動・ゴール報酬の分解 | タスク完了数（エピソード末） |
| **LaRe の適用** | ステップ単位の代理報酬 r̂_path を生成 | 割り当て決定単位の代理報酬 r̂_task を生成 |
| **Transformer の役割** | 時系列の因果的クレジット分解 | タスク完了までの遅延クレジット分解 |

### 問題の構造の違い

```
経路計画の問題構造（密な決定列）:
  t=0: ノード選択 → r_0
  t=1: ノード選択 → r_1
  ...
  t=T: ゴール到達 → r_T = +100
  ↑ どのステップが重要だったか？ ← AREL Transformer で分解

タスク割り当ての問題構造（疎な決定列）:
  k=0: タスクAをエージェント0に割当 → [長いパス後] → タスクA完了（+100）
  k=1: タスクBをエージェント1に割当 → [長いパス後] → タスクB未完了（-）
  k=2: タスクCをエージェント0に割当 → ...
  ↑ どの割り当てが良かったか？ ← MARD + fψ_task で評価
```

### なぜ共通報酬では不十分か

```
現状の PPO タスク割り当ての学習：

  buffer_add_rewards(sum(rew_n), done)
         ↑
  sum(rew_n) = 経路計画の報酬（移動・衝突・ゴール）の合計
                = タスク割り当て品質 + 経路計画品質 が混在
                ↓
  PPO が学習するのは「エージェントが動けば報酬が増える」という信号
  → 割り当て戦略（どのタスクを誰に割当てるか）への学習信号が弱い
```

---

## 3. LaRe の概要

LaRe（Latent Reward: LLM-Empowered Credit Assignment in Episodic Reinforcement Learning，AAAI-2025）

### 数学的定式化

```
p(R | s₁:T, a₁:T) = ∫ [∏ₜ p(rₜ | zᵣ,ₜ) · p(zᵣ,ₜ | sₜ, aₜ)] · p(R | r₁:T) dz dr
```

- **φ(s,a)** : エンコーダ → 潜在因子ベクトル z ∈ ℝᴰ（LLM 生成 Python 関数，または手動実装）
- **fψ** : デコーダ（ニューラルネット）→ 代理報酬 r̂ = fψ(φ(s,a))

**訓練目標：**

```
min_ψ E_τ [ (R(τ) − Σₜ fψ(φ(sₜ, aₜ)))² ]
```

### 理論的優位性

潜在次元 |D| ≪ |S|·|A| のとき，後悔（Regret）上界が改善される：

```
LaRe なし: O(T|S||A| √K log(KT/δ))
LaRe あり: O(T|D|  √K log(KT/δ))
```

### 本設計への適用

本設計では LaRe を **System A / System B に分けて独立適用** する：

| | System A（経路） | System B（タスク） |
|--|-----------------|------------------|
| エンコーダ | φ_path(s_t, a_path_t) | φ_task(s_k, a_task_k) |
| デコーダ | fψ_path: z_path → r̂_path | fψ_task: z_task → r̂_task |
| 訓練ターゲット R | エピソード経路報酬合計 | エピソードタスク完了数 |
| 決定列の長さ | T（ステップ数，長い） | K（割り当て回数，短い） |

---

## 4. 設計方針

1. **二つの独立したシステム**: System A（経路）と System B（タスク）は独立したエンコーダ・デコーダ・バッファを持つ
2. **相互不干渉**: System A の訓練ターゲットに割り当て品質を含めず，System B の訓練ターゲットに経路品質を含めない
3. **既存コードへの影響最小化**: 新規モジュール追加，既存ファイルへの変更は最小限
4. **切り替え可能**: `default.yaml` フラグで各システムを個別に ON/OFF
5. **PBS は対象外**: PBS は報酬を使わないため，System A の対象は IQL/QMIX のみ

---

## 5. 全体アーキテクチャ

```
┌────────────────────────────────────────────────────────────────────────────────┐
│  LDRP + LaRe 全体図                                                             │
│                                                                                │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │  エピソード実行ループ（runner.py）                                          │  │
│  │                                                                          │  │
│  │   obs_t ─────────────────────────────────────────────────────────────┐  │  │
│  │                                                                       │  │  │
│  │   ┌─────────────────────────────────────┐                             │  │  │
│  │   │  System A: LaRe-Path（経路計画用）    │                             │  │  │
│  │   │                                     │                             │  │  │
│  │   │  φ_path(s_t, a_path_t)              │                             │  │  │
│  │   │    ↓ z_path_t ∈ ℝ^D_path           │                             │  │  │
│  │   │  AREL Transformer（因果マスク）      │                             │  │  │
│  │   │    ↓ step-level credit              │                             │  │  │
│  │   │  fψ_path(z_path_t)                  │                             │  │  │
│  │   │    ↓ r̂_path_t（ステップ単位）       │                             │  │  │
│  │   └────────────────┬────────────────────┘                             │  │  │
│  │                    │                                                   │  │  │
│  │                    ▼                                                   │  │  │
│  │   ┌────────────────────────────────────┐                               │  │  │
│  │   │  PolicyManager（IQL / QMIX）        │◄─────────────────────────────┘  │  │
│  │   │  入力: r̂_path_t（代理報酬）          │                                  │  │
│  │   │  出力: a_path_t（移動先ノード）       │                                  │  │
│  │   └────────────────────────────────────┘                                  │  │
│  │                                                                          │  │
│  │   ┌─────────────────────────────────────┐                                  │  │
│  │   │  System B: LaRe-Task（タスク割当用） │  ← アイドル時のみ起動                │  │
│  │   │                                     │                                  │  │
│  │   │  φ_task(s_k, a_task_k)              │                                  │  │
│  │   │    ↓ z_task_k ∈ ℝ^D_task           │                                  │  │
│  │   │  Shapley Attention（MARD）           │                                  │  │
│  │   │    ↓ per-agent credit               │                                  │  │
│  │   │  fψ_task(z_task_k)                  │                                  │  │
│  │   │    ↓ r̂_task_k（割り当て決定単位）    │                                  │  │
│  │   └────────────────┬────────────────────┘                                  │  │
│  │                    │                                                        │  │
│  │                    ▼                                                        │  │
│  │   ┌────────────────────────────────────┐                                    │  │
│  │   │  TaskManager（PPO）                 │                                    │  │
│  │   │  入力: r̂_task_k（代理報酬）          │                                    │  │
│  │   │  出力: a_task_k（タスク割り当て）     │                                    │  │
│  │   └────────────────────────────────────┘                                    │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                │
│  訓練ターゲット（エピソード終了後）                                               │
│    System A: R_path(τ) = Σ_t r_path_t  ← 環境の経路報酬合計                    │
│    System B: R_task(τ) = task_completion ← 完了タスク数                        │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. System A: LaRe-Path（経路計画用）

### 6-1. 役割と評価対象

System A は，**IQL / QMIX がどのノードへ移動するかという毎ステップの決定** を評価する代理報酬を生成する．

- **評価対象**: エージェント i の時刻 t における移動行動 a_path_t
- **訓練ターゲット**: R_path(τ) = エピソード中に得た環境報酬の合計（goal/collision/wait/move）
- **求める代理報酬**: r̂_path_t ← 現在の移動が最終的なゴール到達に寄与しているかを密に評価

### 6-2. 潜在因子エンコーダ φ_path(s,a)

**ファイル:** `src/lare/path/encoder.py`

#### 潜在因子（D_path = 6）

| 因子 | 名前 | 説明 | 計算方法 |
|------|------|------|--------|
| P0 | goal_proximity | 現在のゴールノードへの接近度 | `1 − dist(pos, goal) / max_dist` |
| P1 | path_efficiency | 最短経路との比率 | `shortest_path / actual_steps_so_far` |
| P2 | collision_risk | 近隣エージェントとの距離 | `1 − min_dist(pos, others) / threshold` |
| P3 | detour_factor | ゴール方向への前進率 | `cos_similarity(move_vec, goal_vec)` |
| P4 | wait_indicator | 待機中かどうか | `1 if pos == prev_pos else 0` |
| P5 | congestion_avoidance | 移動先の混雑度 | `1 − #agents_near_next_node / n_agents` |

#### エンコーダ関数の形式

```python
def evaluation_func_path(obs_array, env_info):
    """
    Args:
        obs_array: np.ndarray, shape (batch, n_agents, T, obs_dim)
        env_info: {
            'graph': nx.Graph,
            'current_goal': [node_id_per_agent],
            'agent_positions': [(x,y), ...],
            'prev_positions': [(x,y), ...],
        }
    Returns:
        list of 6 np.ndarray, each shape (batch, n_agents, T, 1)
    """
    # P0: goal_proximity
    # P1: path_efficiency
    # P2: collision_risk
    # P3: detour_factor
    # P4: wait_indicator
    # P5: congestion_avoidance
    return [p0, p1, p2, p3, p4, p5]
```

#### LLM プロンプト（System A 用）

```
タスク: マルチエージェント経路計画（DRP）環境での移動品質を評価する関数を生成してください．

評価の観点（6 因子，各 [0,1] に正規化）:
1. goal_proximity:    現在のゴールノードへの接近度
2. path_efficiency:   最短経路に対する実際の経路の効率性
3. collision_risk:    他エージェントとの衝突リスク（距離ベース）
4. detour_factor:     ゴール方向への前進率（コサイン類似度）
5. wait_indicator:    待機中かどうかのバイナリ指標
6. congestion_avoidance: 移動先ノードの混雑度の逆数

重要: タスクの割り当て品質（どのタスクを割当てたか）は評価しないこと．
純粋に「現在の目標ノードへ向かう移動の質」だけを評価してください．
```

### 6-3. 報酬デコーダ fψ_path

**ファイル:** `src/lare/path/decoder.py`

```python
class PathRewardDecoder(nn.Module):
    """
    z_path ∈ ℝ^D_path → r̂_path ∈ ℝ
    
    訓練: min_ψ E_τ [(R_path(τ) − Σₜ Σᵢ fψ(φ_path(s_ti, a_ti)))²]
    """
    def __init__(self, factor_dim=6, hidden_dim=128, n_layers=3):
        super().__init__()
        layers = [nn.Linear(factor_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers += [nn.Linear(hidden_dim, 1)]
        self.model = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (batch, n_agents, T, D_path)
        → r_hat: (batch, n_agents, T, 1)
        """
        return self.model(z)
```

### 6-4. 時系列クレジット（AREL Transformer）

AREL（Attention-based REward Learning）の Time_Agent_Transformer を使用する．

```
入力:  z_path ∈ ℝ^(B, N, T, D_path)
           ↓
 ┌─────────────────────────────────┐
 │  Time Transformer Block × depth │  ← 因果マスク付き（t' > t を遮断）
 │  Agent Transformer Block × depth│  ← エージェント間の注意（同一時刻）
 └─────────────────────────────────┘
           ↓
  step-level credit ∈ ℝ^(B, N, T, 1)
           ↓
  fψ_path → r̂_path ∈ ℝ^(B, N, T, 1)
```

**因果マスクの重要性（経路計画）:**  
時刻 t の行動のクレジットを計算する際，t+1 以降の情報（まだ起きていないゴール到達）を使ってはならない．これにより，「早い段階での効率的な移動」が正しく評価される．

### 6-5. epymarl への統合

epymarl は `env.step()` の返り値 `reward` を使って学習する．LaRe-Path の代理報酬を注入するために **LaRePathEnv ラッパー** を設ける．

**ファイル:** `src/main/drp_env/wrapper/lare_path_wrapper.py`

```python
class LaRePathEnv(gym.Wrapper):
    """
    DrpEnv.step() の返り値の報酬を LaRe-Path の代理報酬で置き換える
    epymarl から使われる（学習時のみ）
    """
    def __init__(self, env, lare_path_module):
        super().__init__(env)
        self.lare = lare_path_module
        self._trajectory = []
        self._env_reward_sum = 0.0  # 訓練ターゲット用に累積

    def step(self, action):
        prev_obs = self._prev_obs
        obs, reward, done, info = self.env.step(action)
        
        # 環境報酬を訓練ターゲット用に蓄積
        self._env_reward_sum += sum(reward)
        self._trajectory.append((prev_obs, action, reward, obs))
        
        # 代理報酬を計算（デコーダ未学習時は環境報酬をそのまま使う）
        proxy_reward = self.lare.get_proxy_reward(prev_obs, action, self.env)
        
        if done:
            # エピソード終了: 軌跡をバッファへ
            R_path = self._env_reward_sum
            self.lare.add_episode(self._trajectory, R_path)
            self._trajectory = []
            self._env_reward_sum = 0.0

        self._prev_obs = obs
        return obs, proxy_reward if self.lare.is_trained else reward, done, info

    def reset(self):
        obs = self.env.reset()
        self._prev_obs = obs
        self._trajectory = []
        self._env_reward_sum = 0.0
        return obs
```

**epymarl の学習コマンドへの追加（LaRe-Path ON 時）:**

```python
# runner.py / train.py 内で
from src.main.drp_env.wrapper.lare_path_wrapper import LaRePathEnv
from src.lare.path.lare_path_module import LaRePathModule

lare_path = LaRePathModule(args)
env = LaRePathEnv(env, lare_path)
```

---

## 7. System B: LaRe-Task（タスク割り当て用）

### 7-1. 役割と評価対象

System B は，**PPO がどのエージェントにどのタスクを割り当てるかという疎な決定** を評価する代理報酬を生成する．

- **評価対象**: 割り当て決定 a_task_k（K 番目の割り当て）
- **訓練ターゲット**: R_task(τ) = エピソード末のタスク完了数
- **求める代理報酬**: r̂_task_k ← K 番目の割り当て決定が最終的なタスク完了数に寄与しているかを評価

**現状の問題の詳細:**

```python
# 現状（ppo.py の buffer_add_rewards）
self.buffer.add_rewards(sum(rew_n), done)
#                       ↑
# sum(rew_n) = Σᵢ r_path_i_t  ← 経路報酬の和
#
# 問題: タスク割り当ての良し悪しに関わらず，
#       「エージェントが全員動いていれば大きな報酬」
#       → 割り当て戦略の学習が困難
```

**LaRe-Task による解決:**

```python
# 改善後
r̂_task_k = fψ_task(φ_task(s_k, a_task_k))
#
# 訓練ターゲット: task_completion（完了タスク数）
#   → 「良い割り当て」= より多くのタスクが完了する割り当て
#   → 経路報酬とは独立した評価
```

### 7-2. 潜在因子エンコーダ φ_task(s,a_task)

**ファイル:** `src/lare/task/encoder.py`

状態 s_k はタスク割り当て **決定時点** の環境状態（現在のタスクリスト・エージェント位置・割り当て状況），行動 a_task_k は「エージェント q にタスク r を割り当てる」という決定．

#### 潜在因子（D_task = 7）

| 因子 | 名前 | 説明 | 計算方法 |
|------|------|------|--------|
| T0 | pickup_proximity | 割り当て先エージェントのピックアップ地点への近さ | `1 − dist(agent_pos, pickup_node) / max_dist` |
| T1 | task_wait_time | 割り当てタスクの待機時間（長く待っているタスクほど高い） | `wait_steps / time_limit` |
| T2 | load_balance | 割り当て後のエージェント間タスク負荷の均衡度 | `1 − std(task_loads) / max_imbalance` |
| T3 | delivery_distance | ピックアップから配達地点までの経路長（短いほど高い） | `1 − path_len(pickup, dropoff) / max_path_len` |
| T4 | path_congestion | 割り当て先ルート上の混雑度（低いほど高い） | `1 − avg_agents_on_path / n_agents` |
| T5 | urgency_priority | 緊急タスク（待ち時間が長い）の優先度 | `task_wait_time * (1 − pickup_proximity)` |
| T6 | future_task_density | 割り当て後の残タスク密度（今後の割り当てしやすさ） | `1 − #unassigned / task_num` |

#### エンコーダ関数の形式

```python
def evaluation_func_task(assignment_state, env_info):
    """
    タスク割り当て決定の品質を多次元評価する関数
    
    Args:
        assignment_state: dict {
            'assigned_agent': int,        # 割り当て先エージェント ID
            'assigned_task': [pickup, dropoff],  # 割り当てたタスク
            'current_tasklist': [...],    # 割り当て前のタスクリスト
            'assigned_tasks': [...],      # 各エージェントの現タスク状況
            'agent_positions': [(x,y)],  # 各エージェント位置
            'task_wait_steps': [...],    # 各タスクの待機ステップ数
        }
        env_info: {
            'graph': nx.Graph,
            'time_limit': int,
            'n_agents': int,
            'task_num': int,
        }
    Returns:
        list of 7 np.ndarray, each shape (batch, 1)
    """
    # T0: pickup_proximity
    # T1: task_wait_time
    # ...
    return [t0, t1, t2, t3, t4, t5, t6]
```

#### LLM プロンプト（System B 用）

```
タスク: マルチエージェント配送問題（LDRP）でのタスク割り当て品質を評価する関数を生成してください．

文脈:
- 複数のエージェントが動的に発生する配送タスク（ピックアップ地点 → 配達地点）を処理する
- 割り当てはエージェントがタスクを持っていないときのみ可能
- タスクリストには発生済みの未割り当てタスクが蓄積される

評価の観点（7 因子，各 [0,1] に正規化）:
1. pickup_proximity:      割り当て先エージェントがピックアップ地点に近いか
2. task_wait_time:        長く待っているタスクを優先しているか
3. load_balance:          エージェント間の負荷が均衡しているか
4. delivery_distance:     ピックアップ→配達の経路が短いか
5. path_congestion:       割り当てルートが混雑していないか
6. urgency_priority:      緊急度の高いタスクを適切に処理しているか
7. future_task_density:   割り当て後の残タスク管理が適切か

重要: エージェントの移動速度・衝突・個別ノード選択は評価しないこと．
「どのエージェントにどのタスクを割り当てたか」という決定の品質だけを評価してください．
```

### 7-3. 報酬デコーダ fψ_task

**ファイル:** `src/lare/task/decoder.py`

タスク割り当ての決定回数 K はステップ数 T よりはるかに少ない（K ≪ T）ため，デコーダはより小さいネットワークで十分である．

```python
class TaskRewardDecoder(nn.Module):
    """
    z_task ∈ ℝ^D_task → r̂_task ∈ ℝ
    
    訓練: min_ψ E_τ [(R_task(τ) − Σₖ fψ(φ_task(s_k, a_task_k)))²]
    R_task(τ) = task_completion（エピソードのタスク完了数）
    """
    def __init__(self, factor_dim=7, hidden_dim=64, n_layers=2):
        super().__init__()
        layers = [nn.Linear(factor_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers += [nn.Linear(hidden_dim, 1)]
        self.model = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (batch, K, D_task)  ← K は割り当て回数
        → r_hat: (batch, K, 1)
        """
        return self.model(z)
```

### 7-4. エージェント間クレジット（MARD / Shapley 注意機構）

**ファイル:** `src/lare/task/mard.py`

タスク割り当てはエージェント間の協調行動であるため，各割り当て決定がシステム全体のタスク完了にどう貢献したかを Shapley 値で近似する．

```python
class TaskShapelyAttention(nn.Module):
    """
    複数の割り当て決定の相互影響を考慮したクレジット分離
    
    例: エージェント0 にタスクA，エージェント1 にタスクB を割り当てた場合，
        どちらの決定がタスク完了数に多く貢献したかを分離する
    """
    def __init__(self, factor_dim, n_heads, n_agents, sample_num=10):
        ...

    def forward(self, z_task: torch.Tensor) -> torch.Tensor:
        """
        z_task: (batch, K, D_task)  K = 割り当て決定数
        → credits: (batch, K, 1)    各割り当て決定のクレジット
        """
        shapley_credits = []
        for _ in range(self.sample_num):
            coalition_mask = self._sample_coalition(K)
            marginal_credit = self.attention(z_task, mask=coalition_mask)
            shapley_credits.append(marginal_credit)
        return sum(shapley_credits) / self.sample_num
```

### 7-5. PPO タスク割り当て器への統合

**対象ファイル:** `src/task_assign/task_policy/ppo.py`（変更箇所は最小限）

#### 変更点の設計

```python
# ppo.py の PPOAgent クラスに LaReTaskModule への参照を追加
class PPOAgent:
    def __init__(self, args):
        ...
        self.lare_task = None  # 外部から注入（runner.py で設定）
    
    def buffer_add_rewards(self, reward, done):
        """
        変更前: 環境報酬 sum(rew_n) を直接使う
        変更後: LaRe-Task の代理報酬を優先，未学習時は環境報酬を使う
        """
        if self.lare_task is not None and self.lare_task.is_trained:
            # LaRe-Task の代理報酬を使う
            # （この呼び出し元 runner.py で，最新の割り当て状態を渡す）
            lare_reward = self.lare_task.get_latest_proxy_reward()
            self.buffer.add_rewards(lare_reward, done)
        else:
            # フォールバック: 既存の環境報酬
            self.buffer.add_rewards(reward, done)
```

#### runner.py での LaRe-Task の呼び出し

```python
# runner.py の run_episode() 内

# タスク割り当て後，エージェントがアイドルになるたびに
task_assign = self.task_manager.assign_task(self.env)
if any(t != -1 for t in task_assign):
    # 割り当てが発生した
    assignment_state = self._build_assignment_state(task_assign)
    self.lare_task.record_assignment(assignment_state)

# エピソード終了後
if done:
    R_task = info["task_completion"]  # タスク完了数を訓練ターゲットに
    self.lare_task.add_episode(R_task)
```

#### タスク割り当てバッファの訓練ターゲット

```
エピソードバッファ（System B 用）:
  k=0: assignment_state_0 → z_task_0 → r̂_task_0
  k=1: assignment_state_1 → z_task_1 → r̂_task_1
  ...
  k=K: assignment_state_K → z_task_K → r̂_task_K
  
  R_task = task_completion（エピソード末の完了数）
  
  損失: L = (R_task − Σₖ r̂_task_k)²
```

---

## 8. 二システムの相互作用

### 独立性の保証

```
System A（経路計画）             System B（タスク割り当て）
  訓練ターゲット: R_path            訓練ターゲット: R_task
  = Σ_t r_env_t（経路報酬合計）      = task_completion（タスク完了数）
        ↑                                   ↑
      独立（混合しない）               独立（混合しない）
```

### 間接的な相互作用（意図的に許容）

```
                       ┌─────────────────────────────┐
                       │ 「良い経路計画」は            │
                       │  ピックアップ→配達を速くする  │
                       │          ↓                  │
                       │ task_completion が増える     │
                       │          ↓                  │
                       │ System B の R_task が増える  │
                       └─────────────────────────────┘
```

つまり System A（経路）の改善が System B（タスク）の訓練ターゲットを自然に改善する構造になっている．両者を分離して設計しても，この間接的な相乗効果は保持される．

### 情報の流れ

```
環境: env.step(joint_action)
  └─→ obs, r_env, done, info
         │          │         │
         │          │         └─→ info["task_completion"]
         │          │                    ↓
         │          │              LaRe-Task の
         │          │              訓練ターゲット R_task
         │          │
         │          └─→ sum(r_env)  ← LaRe-Path の訓練ターゲット R_path
         │
         └─→ 次のステップの obs
                    ↓
              φ_path(obs, a_path)  ← System A エンコード
              φ_task(obs, a_task)  ← System B エンコード（割り当て時のみ）
```

---

## 9. ファイル構成

### 新規追加ファイル

```
LDRP/
└── src/
    └── lare/                              # 新規ディレクトリ
        ├── __init__.py
        │
        ├── path/                          # System A: 経路計画用
        │   ├── __init__.py
        │   ├── lare_path_module.py        # LaRePathModule（System A の管理クラス）
        │   ├── encoder.py                 # φ_path: 6 次元経路因子エンコーダ
        │   ├── decoder.py                 # fψ_path: 経路報酬デコーダ（MLP）
        │   ├── transformer.py             # AREL Time_Agent_Transformer（経路用）
        │   ├── buffer.py                  # 経路軌跡バッファ（ステップ単位）
        │   └── prompts.py                 # LLM プロンプト（System A 用）
        │
        ├── task/                          # System B: タスク割り当て用
        │   ├── __init__.py
        │   ├── lare_task_module.py        # LaReTaskModule（System B の管理クラス）
        │   ├── encoder.py                 # φ_task: 7 次元タスク因子エンコーダ
        │   ├── decoder.py                 # fψ_task: タスク報酬デコーダ（MLP）
        │   ├── mard.py                    # Shapley Attention（割り当てクレジット）
        │   ├── buffer.py                  # 割り当て決定バッファ（決定単位）
        │   └── prompts.py                 # LLM プロンプト（System B 用）
        │
        └── shared/                        # 共有コンポーネント
            ├── __init__.py
            └── attention.py               # 基本 Attention モジュール
```

### 変更するファイル

```
LDRP/
├── src/config/default.yaml               # LaRe 設定項目を追加
├── runner.py                             # LaRePathModule・LaReTaskModule の初期化・呼び出し
├── src/task_assign/task_policy/ppo.py    # lare_task への参照を追加（最小変更）
└── src/main/drp_env/wrapper/
    └── lare_path_wrapper.py              # 新規: LaRePathEnv（epymarl 用ラッパー）
```

---

## 10. 実装手順

### Phase 1: System B（タスク割り当て）から先に実装

タスク割り当て側のほうが決定頻度が低く（K ≪ T），バッファ管理が単純なため，先に実装・検証する．

**Step 1: LaRe-Task バッファと割り当て状態の定義**
- `src/lare/task/buffer.py`: 割り当て決定単位のバッファを実装
- 割り当て状態 `AssignmentState` の data class を定義
- エピソード末に `task_completion` を R_task として追加する流れを確認

**Step 2: φ_task エンコーダの手動実装**
- まず LLM なしで，手動で 7 因子を実装
- `runner.py` の `_build_assignment_state()` ヘルパーを実装
- 因子の数値が [0,1] に正規化されることをテストで確認

**Step 3: fψ_task デコーダの実装と訓練**
- `src/lare/task/decoder.py` を実装
- `LaReTaskModule.add_episode()` → `_update()` の学習ループを実装
- 合成データ（ランダム割り当て）で MSE 損失が収束するか確認

**Step 4: PPO タスク割り当て器への統合**
- `ppo.py` の `buffer_add_rewards()` を最小限変更
- `runner.py` でタスク割り当て発生時に `record_assignment()` を呼び出す
- `test.py` で `task_flag=True`, `path_planner="pbs"`, `task_assigner="ppo"` で動作確認

### Phase 2: System A（経路計画）を実装

**Step 5: LaRe-Path バッファの実装**
- `src/lare/path/buffer.py`: ステップ単位の軌跡バッファを実装
- メモリ使用量を確認（T=1000, N=5 エージェントの場合）

**Step 6: φ_path エンコーダの手動実装**
- 6 因子を `drp_env.py` から必要な情報（グラフ・位置）を取得して実装
- `LaRePathEnv` ラッパー内で `_extract_path_env_info()` を実装

**Step 7: fψ_path + AREL Transformer の実装**
- `src/lare/path/transformer.py`: Safe-TSL-DBCT の実装を参考に移植
- 因果マスクの正しさをユニットテスト（未来情報の漏れがないか）で確認

**Step 8: epymarl との統合**
- `LaRePathEnv` を epymarl の学習ループに組み込む
- `train.py` で `use_lare_path=true` フラグを追加

### Phase 3: 評価と LLM エンコーダ

**Step 9: 比較実験**
- ベースライン vs LaRe-Task のみ vs LaRe-Path のみ vs 両方

**Step 10: LLM エンコーダの導入（オプション）**
- OpenAI API で φ_path / φ_task の `evaluation_func` を生成
- 手動実装との性能比較

---

## 11. パラメータ一覧

### default.yaml に追加する設定

```yaml
# --- System A: LaRe-Path（経路計画用）---
use_lare_path: false            # 経路計画用 LaRe を有効にするか
lare_path_factor_dim: 6         # φ_path の次元数 D_path
lare_path_hidden_dim: 128       # fψ_path の隠れ層次元数
lare_path_n_layers: 3           # fψ_path の層数
lare_path_heads: 4              # AREL Transformer のアテンションヘッド数
lare_path_depth: 2              # AREL Transformer のブロック深さ
lare_path_buffer_size: 512      # 経路軌跡バッファのエピソード数
lare_path_update_freq: 64       # 更新頻度（エピソード数）
lare_path_lr: 0.0005            # fψ_path の学習率
lare_path_min_buffer: 128       # 学習開始に必要な最小エピソード数

# --- System B: LaRe-Task（タスク割り当て用）---
use_lare_task: false            # タスク割り当て用 LaRe を有効にするか
lare_task_factor_dim: 7         # φ_task の次元数 D_task
lare_task_hidden_dim: 64        # fψ_task の隠れ層次元数（System A より小さくて良い）
lare_task_n_layers: 2           # fψ_task の層数（K ≪ T なのでシンプルで良い）
lare_task_shapley_samples: 10   # Shapley サンプル数
lare_task_buffer_size: 512      # 割り当て決定バッファのエピソード数
lare_task_update_freq: 64       # 更新頻度（エピソード数）
lare_task_lr: 0.0005            # fψ_task の学習率
lare_task_min_buffer: 128       # 学習開始に必要な最小エピソード数
```

---

## 12. 評価指標・実験設計

### 評価指標

| 指標 | 測定方法 | System A との関係 | System B との関係 |
|------|----------|-----------------|-----------------|
| **タスク完了数（主）** | `info["task_completion"]` | 間接的 | 直接的 |
| **衝突率** | `info["collision"]` | 直接的 | 間接的 |
| **平均ステップ数** | `info["step"]` | 直接的 | 間接的 |
| **経路効率** | 実移動距離 / 最短距離 | 直接的 | なし |
| **タスク割り当て待機時間** | 各タスクの待機ステップ平均 | なし | 直接的 |
| **デコーダ損失（A）** | MSE(R_path, Σ r̂_path) | — | — |
| **デコーダ損失（B）** | MSE(R_task, Σ r̂_task) | — | — |

### 比較実験設計

| 条件名 | 経路計画 | タスク割り当て | LaRe-Path | LaRe-Task |
|--------|---------|--------------|-----------|-----------|
| Baseline-PBS-TP | PBS | TP | OFF | OFF |
| Baseline-IQL-TP | IQL | TP | OFF | OFF |
| Baseline-IQL-PPO | IQL | PPO | OFF | OFF |
| **LaRe-Task-Only** | IQL | PPO | OFF | **ON** |
| **LaRe-Path-Only** | IQL | TP | **ON** | OFF |
| **LaRe-Both** | IQL | PPO | **ON** | **ON** |
| LaRe-Both-QMIX | QMIX | PPO | **ON** | **ON** |

---

## 13. 懸念点・リスク

| 懸念点 | 内容 | 対策 |
|--------|------|------|
| **System A と B のバッファサイズの違い** | System A はステップ単位（T=1000）で大きい．System B は決定単位（K≈10〜50）で小さい | System A のバッファは圧縮保持または T を間引く（e.g., 10 step おきにサンプル） |
| **R_task の希薄性** | タスク完了数は小さい整数（0〜10 程度）でバリアンスが高い | 正規化（R_task / max_tasks），またはステップ単位の中間報酬（タスクピックアップ時+0.5 等）を補助訓練ターゲットとして追加 |
| **System B の K の変動** | エピソードによって割り当て回数 K が異なる | パディング + マスク，または動的バッチを使用 |
| **AREL Transformer の計算コスト（System A）** | T×N が大きいと計算量が増大 | シーケンス長を T=100 程度に制限するか，Transformer を学習時のみ使い推論時は fψ のみ使用 |
| **PPO のリセットタイミング** | `ppo.py` の `buffer_reset()` は更新後に呼ばれるが，LaRe 代理報酬との整合が必要 | LaRe-Task の `is_trained` フラグを PPO バッファのリセット前に確認する |
| **drp_env.py の既存変更との干渉** | PBS 用の 200 行目変更が LaRePathEnv に影響する可能性 | ラッパー内で `env.step()` の返り値を変更するだけなので基本的に干渉しない |
| **System A の代理報酬と epymarl の正規化** | epymarl 内部で報酬正規化が入る場合，代理報酬スケールが合わない | epymarl の `normalize_reward` 設定を確認し，代理報酬も同スケールに合わせる |

---

## まとめ

```
LDRP における二つの報酬問題：

【経路計画の問題】            【タスク割り当ての問題】
 決定: 毎ステップ              決定: アイドル時のみ（疎）
 問題: どの移動が効果的か？    問題: どの割り当てが効果的か？
 現状: 単純スカラー報酬          現状: 経路報酬と混在
        ↓                              ↓
 System A (LaRe-Path)         System B (LaRe-Task)
  φ_path: 6 因子（移動品質）    φ_task: 7 因子（割り当て品質）
  fψ_path: ステップ単位報酬     fψ_task: 割り当て単位報酬
  AREL Transformer             MARD Shapley Attention
  訓練ターゲット: R_path        訓練ターゲット: R_task（完了数）
        ↓                              ↓
 IQL/QMIX の学習精度向上       PPO タスク割り当ての学習精度向上
```

**実装の優先順位:** System B（タスク割り当て）→ System A（経路計画）の順で実装する．System B は決定が疎で実装が単純であり，現状の `sum(rew_n)` という明確な問題を修正する効果が見えやすいため．

---

## 14. 今後の拡張 (未実装)

実装済みのコードを書き換えずに，今後追加していく改修を集約する章．既存の章 (1〜13) は設計時の根拠を残すため変更しない．

### 14-1. epymarl 経由での同時学習 (PPO タスク割当 + MARL 経路計画 + LaRe 両系統)

**動機**: 現状 `train.py` (epymarl サブプロセス) で学習されるのは経路計画 MARL のみで，タスク割当は env 内蔵の TP フォールバック (後述 14-2) によりベースラインヒューリスティックで固定される．経路と割当を **共進化** させたい場合に必要．

**現状の制約**:

1. **epymarl の gymma wrapper は path action (`list[int]`) しか env に渡せない**．タスクアクションを epymarl の action space に持ち込めないため，割当決定を学習可能にするには env 内部に PPO を埋め込む必要がある (LaRe-Path / LaRe-Task と同じパターン)．
2. **既存の PPOAgent には複数のバグがある** ([src/task_assign/task_policy/ppo.py](src/task_assign/task_policy/ppo.py))．
   - Bug A: `assign_task(env, current_tasklist, assigned_tasklist)` のシグネチャが TaskManager の呼び出し `assign_task(env)` と不整合
   - Bug B: `update()` の `for _ in range(self.args.epochs):` 内側 `for batch in loader:` の後にインデントされた勾配計算が，バッチごとに走らず epoch ごとに最後の batch だけを使う
   - Bug C: `runner.py` の training 分岐が `self.task_Agent.task_assigner...` を参照しているが，本来 `self.task_manager` の typo (epymarl 経由学習では使われないので残置されている)
   - Bug D: `test.py` で `Runner(config, env, reward_list)` が呼ばれており，`training=True` が渡されない

**実装方針 (案)**:

1. **PPO バグ修正 (A, B)**: epymarl 経路で使う前に必須．Bug C, D は test.py 経由の場合に必要なのでこのフェーズでは保留可．
2. **env への PPO 内蔵**: [drp_env.py](src/main/drp_env/drp_env.py) に `self.ppo_task_module` を追加 (LaRe-Path / LaRe-Task と同じ位置).
   - `use_ppo_task=True` フラグで有効化．
   - `step()` 内で list 形式アクション + `task_flag=True` の場合，TP フォールバックの代わりに PPO で `task_assign` を推論．
   - 状態・行動・log_prob を内部バッファに蓄積．
   - エピソード終了で `process_end_episode()` + `update_ready()` → `update()`.
   - autosave 命名: `Safe_{ALGO}_TASKPPO_{map}_{N}agents_{X.X}M_checkpoint.pth` (LaRe と並列).
3. **LaRe-Task との連携**: 既存の `lare_task_module` が PPO の `assign_task` 決定時に encoder を呼んで proxy 報酬を生成する設計なので，PPO 内蔵時は PPO バッファに記録する報酬を環境報酬合計から **`info["lare_task_proxy_reward"]`** に切り替える．既存の runner.py:108-117 と同じロジックを env step() 内に移植．

**実装規模見積もり**:

| 項目 | 見積もり |
| --- | --- |
| ppo.py の Bug A, B 修正 | 50〜100 行 |
| drp_env.py への PPO 統合 (init / step / end_episode / save) | 200〜300 行 |
| yaml / register 設定追加 | 30〜50 行 |
| 動作検証 (smoke test) | 1 日 |
| 合計 | 2〜3 日 |

**先送りの判断**:

優先度は低い．理由:

- まず経路計画単体での LaRe-Path 効果を切り分けて確認したい．共進化を入れると効果の原因切り分けが難しくなる．
- 元論文 LaRe (AAAI-2025) も「報酬整形器の学習」と「方策学習」を独立に検証している．まずは方針 A (2 段階) で検証して，必要性が見えたら同時学習に踏み込む．

### 14-2. TP フォールバック (実装済み, 2026-05-19)

epymarl 経由で `task_flag=True` 学習を成立させるための env 内蔵タスク割当．`drp_env.step()` が dict 形式アクションを受け取らない (list のみ) 場合，かつ `task_flag=True` のとき，[src/task_assign/task_policy/tp.py](src/task_assign/task_policy/tp.py) と同じ最近隣ヒューリスティックを env メソッドとして実行する．test.py 経由 (dict アクション) の動作は変えない．14-1 で PPO を内蔵するときは，このフォールバックを「PPO 未有効時のデフォルト」として残す．

実装位置: [drp_env.py](src/main/drp_env/drp_env.py) の `_default_task_assign_tp()` メソッドおよび `step()` 冒頭の分岐．

LaRe 機能のファイル対応表
1. 10 因子エンコーダ (factor extraction)
役割	MARL4DRP	LDRP
Path 用 factor 関数 (env state → 10 次元 factor)	drp_env/reward_model/LLMrd/fallback_functions/evaluation_func.py	src/lare/path/encoder.py evaluation_func()
マップ別の特殊 factor 関数 (オプション)	LLMrd/pre_fallback_functions/map_*_*agents.py	(LDRP では未移植)
Task 用 factor 関数	(MARL4DRP は Task LaRe 持たず)	src/lare/task/encoder.py evaluation_func_task()
2. デコーダ (factor → reward)
役割	MARL4DRP	LDRP
Path デコーダ MLP	drp_env/reward_model/LLMrd/factor_reward_model.py (FactorRewardModel)	src/lare/path/decoder.py (PathRewardDecoder)
Task デコーダ MLP	(なし)	src/lare/task/decoder.py
factor + state → reward 統合層	LLMrd/factor_reward_decompose.py (FactorRewardDecomposer)	(LDRP は module 側に直接統合)
3. モジュール / オーケストレータ (バッファ + 学習を回す本体)
役割	MARL4DRP	LDRP
Path 学習ループの司令塔	drp_env/drp_env.py の initialize_lare_system(), perform_episode_update(), evaluation_period 状態機械 (1340-1382 行)	src/lare/path/lare_path_module.py (LaRePathModule クラス)
Task 学習ループの司令塔	(なし)	src/lare/task/lare_task_module.py
重要な違い: MARL4DRP は env (drp_env.py) 内部に直接 LaRe 制御を書いている。LDRP は LaRePathModule という独立クラスに切り出して env からは委譲する設計。

4. エピソードバッファ (経験データの保管)
役割	MARL4DRP	LDRP
エピソードバッファ実装	epymarl/src/utils/replay_memory.py (ReplayMemory_episode)	src/lare/path/buffer.py (PathEpisodeBuffer), src/lare/task/buffer.py
置換戦略	circular buffer (position % capacity)	deque(maxlen=capacity) (FIFO)
memory 分離 (goal/collision/timeup)	goal_memory, collision_memory, timeup_memory を別々に持つ	単一 buffer のみ (LDRP は未移植)
5. 訓練ステップ (loss + gradient)
役割	MARL4DRP	LDRP
train_step 関数 (1 batch の forward + backward + optimizer.step)	epymarl/src/utils/util.py make_train_step() (163-280 行)	src/lare/path/lare_path_module.py:200-254 _update() メソッド
6. Transformer (AREL Time-Agent Attention, オプション)
役割	MARL4DRP	LDRP
時系列・エージェント間の注意機構	drp_env/reward_model/arel/transformers.py	src/lare/path/transformer.py (TimeAgentTransformer)
基本 attention モジュール	drp_env/reward_model/arel/modules.py	src/lare/shared/attention.py
utility (positional encoding 等)	drp_env/reward_model/arel/util.py	(transformer.py 内に統合)
7. MARD (Shapley Attention for assignment, 未移植)
役割	MARL4DRP	LDRP
割当クレジット用 Shapley attention	drp_env/reward_model/mard/mard.py	(未移植。Task LaRe で別アプローチを採用)
MARD の attention モジュール	drp_env/reward_model/mard/modules.py	(なし)
状態正規化	drp_env/reward_model/mard/norm.py	(なし)
8. LLM プロンプト (factor 関数の自動生成)
役割	MARL4DRP	LDRP
GPT に factor 関数を生成させるコード	drp_env/reward_model/LLMrd/factor_chat_with_gpt.py	(未移植。fallback の evaluation_func を直接使用)
プロンプトテンプレート	drp_env/reward_model/LLMrd/prompt_template.py	(なし)
9. env 統合 (step フックポイント)
役割	MARL4DRP	LDRP
env.step() 内 LaRe 呼び出し箇所	drp_env/drp_env.py の step() 内 (1431 行〜)、collision 後の reward 置換 (1601-1619 行)、reset() 内の evaluation_period 管理 (1340-1382 行)	drp_env.py:879-882 で proxy 報酬置換、step 内の _lare_total_step_account += 1
env 設定	drp_env.py コンストラクタの use_lare_reward, use_lare_training, use_pretrained_model, pretrained_model_name	drp_env.py の use_lare_path, use_lare_path_training, use_pretrained_lare_path, pretrained_lare_path_model_name 等
10. ベース報酬モデルの抽象化
役割	MARL4DRP	LDRP
共通基底クラス	drp_env/reward_model/base_reward_model.py	(LDRP では Path/Task で抽象化していない)
module レジストリ	drp_env/reward_model/init.py	(LDRP は分離設計でレジストリ不要)
設計思想の差
軸	MARL4DRP	LDRP
構造	env (drp_env.py) に LaRe ロジックが内蔵	LaRe を別パッケージ (src/lare/) に切り出し、env はフックポイントのみ
Task LaRe	無し (Path 用のみ)	Path / Task の 2 系統 を独立に実装
memory 分離	termination タイプ別に 3 バッファ + weighted sampling	単一 buffer + uniform sampling
LLM 連携	GPT で factor 関数を自動生成可	fallback を直接使用 (LLM 未移植)
MARD (Shapley)	割当に Shapley attention 使用可	未移植 (Task LaRe で別アプローチ)
補足: 未移植の機能
MARL4DRP にあって LDRP に持ってきていないもの:

MARD (Shapley attention) — assignment 用クレジット分解の高度な仕組み
LLM による factor 自動生成 — factor_chat_with_gpt.py, prompt_template.py
マップ別の pre_fallback_functions — map_8x5_3agents 専用の factor 関数等
termination 別 memory 分離 (goal_memory, collision_memory, timeup_memory) + 60/20/20 weighted sampling
use_separete_memory モード — MARL4DRP には memory 分離オプションあり
特に 4 はデコーダの安定学習に効きそうなので、現在の loss 振動が止まらない場合は移植検討の価値ありです。