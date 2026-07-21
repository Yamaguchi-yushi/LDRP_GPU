# 設計書: ALMA (Hierarchical Allocator-Actor) を LDRP に導入する計画

**作成日:** 2026-05-10
**ステータス:** 設計案 (実装はまだ)
**対象リポジトリ:**
- 導入先: [https://github.com/Yamaguchi-yushi/LDRP](https://github.com/Yamaguchi-yushi/LDRP) (本リポ)
- 元論文: ALMA: Hierarchical Learning for Composite Multi-Agent Tasks (NeurIPS 2022, Iqbal et al.)
- 著者実装: [https://github.com/shariqiqbal2810/ALMA](https://github.com/shariqiqbal2810/ALMA)
- 関連: [lare_integration.md](lare_integration.md), [ldrp_extensions.md](ldrp_extensions.md), [../MANUAL.md](../MANUAL.md)

---

## 目次

1. [背景・動機](#1-背景動機)
2. [ALMA の核心アイデア (論文サマリ)](#2-alma-の核心アイデア-論文サマリ)
3. [LDRP との対応関係](#3-ldrp-との対応関係)
4. [設計方針](#4-設計方針)
5. [全体アーキテクチャ](#5-全体アーキテクチャ)
6. [コンポーネント設計](#6-コンポーネント設計)
7. [既存実装との接続](#7-既存実装との接続)
8. [LaRe との関係 (相補性)](#8-lare-との関係-相補性)
9. [ファイル構成 (予定)](#9-ファイル構成-予定)
10. [実装フェーズ + 段階移行戦略](#10-実装フェーズ--段階移行戦略)
11. [パラメータ案 (default.yaml への追加)](#11-パラメータ案-defaultyaml-への追加)
12. [評価設計](#12-評価設計)
13. [リスク・未決事項](#13-リスク未決事項)
14. [軽量代替案 (フル ALMA 採用しない場合)](#14-軽量代替案-フル-alma-採用しない場合)

> **複数タスク保持エージェントへの拡張**は別設計書 [ldrp_extensions.md](ldrp_extensions.md) §3 を参照。

---

## 1. 背景・動機

### LDRP の現状の問題

LDRP は以下の二段階構造を **暗黙に** 持つが、両者が独立に学習・実装されている:

- **タスク割当 (高位)**: 空きエージェントにどの配送タスクを割り当てるか
- **経路計画 (低位)**: 割り当てられたタスクのピックアップ/配達ノードへどう移動するか

現状:

| 段階 | 実装 | 学習信号 |
|---|---|---|
| 割当 | TP (近接ヒューリスティック) / FIFO / PPO | `sum(rew_n)` (経路報酬の和; LaRe-Task で proxy 化可) |
| 経路 | PBS / IQL / QMIX / VDN / MAA2C | 環境スカラー報酬 (LaRe-Path で proxy 化可) |

**両者を独立に学習** しているため、(1) 割当が経路の難易度を考慮しない、(2) 経路が割当の戦略性を活かせない、という相互最適化欠如が起きる。

### ALMA がもたらすもの

ALMA は **「割当方策」と「行動方策」を同時に学習** するアーキテクチャ。本論文の主張:

- 割当方策と行動方策の **同時学習** が単独学習より優位 (Joint training >> 個別 → 結合)
- 割当行動空間 $|I|^{|A|}$ (LDRP なら 9^4=6561 程度) を **Amortized Q-Learning** + **提案分布** で扱える
- サブタスクごとに観測をマスクすることで、低位方策の **状態空間が大幅縮小** + **サブタスク間で重み再利用**

LDRP に適用すると、上記の「相互最適化欠如」を構造的に解消できる可能性が高い。

### ALMA = "経路 MARL + 割当 RL" の一般化 (dual-RL フレーミング)

LDRP の想定するアーキは「**経路 = MARL** + **タスク割当 = もう一つの RL**」という二段 RL 構成。
**現状 LDRP は既にこの構成になっている** が naive 実装にとどまっている。ALMA は同じ二段 RL 構成を **原理的に強化** したものと位置付けられる。

| 観点 | 現状 LDRP の dual-RL | ALMA の dual-RL |
|---|---|---|
| 経路 RL (低位) | IQL / QMIX / VDN / MAA2C (epymarl, MARL) | 同じ MARL バックボーン + **サブタスク条件付け** + **観測マスキング** |
| 割当 RL (高位) | PPO (各 agent 独立に softmax over タスク id) | 全 agent 同時の **組合せ Q-learning** Q(s, b), 提案分布 + AQL |
| 学習方式 | **独立学習 (順次)**: epymarl で経路 → 別途 PPO で割当 | **同時学習 (joint)**: 高位/低位を並行更新 |
| 割当への報酬信号 | `sum(rew_n)` (= 経路報酬の和, 割当品質を反映しにくい) | $\sum_n^{N_t} r_{t+n}$ (時間集約) または LaRe-Task で proxy 化 |
| 観測スコープ | 全 agent + 全状態を見る | 割り当てサブタスクの関連だけ見る |

**ALMA が naive dual-RL に追加する 3 つの貢献**:

1. **同時学習 (Joint training)**: 互いの方策に適応するフィードバックループが回る (論文 §5 Joint Training, Figure 6b)
2. **組合せ Q over 全 agent 割当**: per-agent 独立 softmax より協調戦略 (例「A が近距離取るなら B は遠距離」) を学習可能
3. **サブタスク観測マスキング**: 不要情報を遮断して状態空間激減 (論文 No-mask アブレーションで性能崩壊)

裏返すと、上記 3 つを「採るか採らないか」で **段階的導入** が可能 ([§10](#10-実装フェーズ--段階移行戦略) 参照)。

---

## 2. ALMA の核心アイデア (論文サマリ)

### 2.1 タスクモデル

- 環境は **コンポジット**: 複数の独立サブタスク $\{i \in I\}$ からなる
- 各サブタスクは固有のエンティティ集合 $E_i$ と固有の報酬関数 $r_i$ を持つ
- グローバル報酬は $r = \sum_i r_i$ ベース、または完了ボーナス付き

### 2.2 二段階方策

| 段階 | 記号 | 役割 |
|---|---|---|
| **Allocator (高位)** | $\Pi(b \mid s)$ | 状態 s から割当 $b = \{b^a \in I\}_{a \in A}$ を選ぶ。**$N_t$ ステップに 1 回**だけ更新 |
| **Actor (低位)** | $\pi^a(u^a \mid s_{b_i}, b)$ | 割り当てられたサブタスク $i = b^a$ のローカル状態だけを見て行動 |

### 2.3 大きな割当行動空間への対処

$|I|^{|A|}$ の組合せ爆発があるので、Q-Learning の $\arg\max_b$ を真面目に解けない。

→ **Amortized Q-Learning** [Van de Wiele 2020]:

- 提案分布 $f(b \mid s; \phi)$ から $N_p$ サンプルし、その中で最大 Q を取ったものを近似的な $\arg\max$ とする
- 提案分布の損失:
  $$\mathcal{L}(\phi) = -\log f(b^* \mid s; \phi) - \lambda^{\rm AQL} H(f(\cdot \mid s; \phi))$$
  ここで $b^* = \arg\max_{b \in B_{\rm samp}} Q(s, b)$。エントロピー項で多様性を確保。

### 2.4 提案分布の構造 (ポインターネット風)

自己回帰的に分解:

$$f(b \mid s) = \prod_a f(b^a \mid s, b^{<a})$$

各因子は:

1. エージェント埋め込み $h_a = f^h(s^a)$
2. サブタスク埋め込み $g_i = f^g(s_{E_i})$
3. ロジット $g_i^\top h_a$ を softmax して $b^a \sim f$
4. 選ばれたサブタスクの埋め込みを $g'_{b^a} = g_{b^a} + f^u(g_{b^a}, h_a)$ で更新 (= 後続のエージェントが既割当を考慮)

サブタスク数 $|I|$ が動的に変わってもアーキ的に対応可。

### 2.5 サブタスク独立性の仮定

理想条件下では、サブタスク $i$ の遷移はサブタスク内エージェントだけに依存:

$$Q^{\rm tot}_i(s_{b_i}, u_{b_i}; b)$$

→ **観測マスキング** で各エージェントが自分のサブタスク以外を見ないようにする。これがアブレーション (No mask) で大きく性能落ちることが論文で示されている。

### 2.6 損失まとめ

| 対象 | 損失 |
|---|---|
| Allocator 提案分布 $f(b \mid s; \phi)$ | $-\log f(b^* \mid s) - \lambda H(f)$ |
| Allocator 価値関数 $Q(s, b; \Theta)$ | TD: $\lVert y_t - Q_\Theta(s_t, b_t)\rVert^2,\, y_t = \sum_n^{N_t} r_{t+n} + \gamma Q_{\bar\Theta}(s_{t+N_t}, b^*(s_{t+N_t}))$ |
| Actor サブタスク Q | $\lVert y - Q_i^{\rm tot}(s_{b_i}, u_{b_i}; b)\rVert^2$, 報酬は $r_i^b$ |

---

## 3. LDRP との対応関係

### 3.1 用語マッピング

| ALMA 用語 | LDRP での対応 | 補足 |
|---|---|---|
| Subtask $i \in I$ | 配送タスク `current_tasklist[i] = [pickup, dropoff]` | 動的に増減 |
| Subtask entities $E_i$ | ピックアップノード, 配達ノード | グラフ上の 2 ノード |
| Agent $a$ | エージェント (運搬車) | `env.agent_num` 個 |
| Allocation $b^a$ | エージェント a に割り当てるタスク index (or "idle") | 現状の `task_assign[a]` と同型 |
| High-level controller $\Pi$ | **新設** ALMA Allocator (現 PPO/TP/FIFO を置き換え) | |
| Low-level controller $\pi^a$ | サブタスク条件付き経路計画器 (現 IQL/QMIX を拡張) | |
| Subtask reward $r_i$ | タスク i 起因の報酬 (ピックアップ/配達ボーナス, タスク内移動コスト) | 環境分解が必要 |

### 3.2 LDRP 特有の制約

ALMA は **Multi-agent subtasks (ST-MR-IA)** を主に想定。LDRP は基本的に:

- **1 タスク = 1 エージェント** (ST-SR-IA): 同一タスクに複数エージェントは割当てない
- **1 エージェント = ≤1 タスク**: 既存挙動と同じ

→ 提案分布のサンプリング時に **「同一割当ラウンド内で 2 エージェントに同じタスクを振らない」マスクを追加** する必要がある。論文の自己回帰サンプリング (`f^u` 更新) は既割当タスクの埋め込みを更新するだけで除外はしない。LDRP では明示的にマスクを掛ける。

ここで重要なのは **「割当済み」と「ロック (再割当不可)」を区別する**こと。ピックアップ前のタスクは過去ラウンドで誰かに割当済みでも、次のラウンドで別エージェントに振り直してよい (= 再割当可能)。恒久ロックすると ALMA の毎ラウンド再最適化という強みを殺すため、ロック対象は **ピックアップ済み (配達中) のタスクだけ** に限定する。詳細は [§3.4](#34-ピックアップ前タスクの再割当-reassignment) を参照。

### 3.3 LDRP 特有の追加要素

| 要素 | ALMA 標準 | LDRP 拡張 |
|---|---|---|
| 動的タスク到着 | サブタスク数固定想定 | **毎ステップ新タスクが追加** されうる。提案分布は可変 $\lvert I\rvert$ に対応 |
| Idle 行動 | 全エージェントが何かに割当 | 「タスクなし (-1)」も合法な b^a として許す |
| グラフ構造 | エンティティは座標のみ | グラフ距離 / 衝突制約あり (LDRP-safe wrapper) |

### 3.4 ピックアップ前タスクの再割当 (reassignment)

ALMA Allocator が割当を解き直すとき、**「まだピックアップしていない割当済みタスク」を、より適したエージェントに振り直せる** ようにする。これは追加仕様というより、割当を再最適化する ALMA の素の挙動を LDRP で正しく許可するもの。恒久ロックすると協調的な振り直し (例「近くで空いた別 agent に渡す」) を自分で封じてしまう。

> 割当を解き直すタイミング (割当頻度) は **イベント駆動** で扱う想定 (新タスク到着・ピック完了・エージェント解放などのイベントで Allocator を起動)。本節は「そのイベント時にどこまで振り直してよいか」の範囲を定める。

#### 状態と再割当可否

既存 env の不変条件がこの仕様にそのまま対応する:

- `current_tasklist` = **ピックアップ前のタスクだけ** (ピック到達時に [drp_env.py:943](../src/main/drp_env/drp_env.py#L943) で `pop`)
- ピック後のタスクは `current_tasklist` から除外され、`assigned_tasks[i]` にのみ残る (ドロップで解放)

| タスク状態 | 居場所 | 再割当 |
|---|---|---|
| 未割当 | `current_tasklist`, `assigned_list[idx] == -1` | 可 |
| 割当済み・**ピック前** | `current_tasklist`, `assigned_list[idx] == i` | **可 (本仕様)** |
| **ピック後**・配達中 | `current_tasklist` から除外済み, `assigned_tasks[i]` のみ | **不可 (ロック)** |

→ **「再割当可能 = `current_tasklist` に居るタスク」「ロック = ピック済み」** が追加の状態管理なしで成立する。LDRP には「積荷を途中で降ろす」動作が無いため、ピック後ロックは仕様上の必然。

#### Allocator から見た割当問題

- **エージェント側のプール**: 「空き」+「ピック前タスク保持」エージェント。**ピック後 (配達中) エージェントは割当プールから除外**し、`b^a` を現タスクに固定する。
- **タスク側のプール**: `current_tasklist` 全体 (未割当 + ピック前割当済み)。マスクは「同一ラウンド内で同じタスクを 2 人に振らない」(自己回帰 `f^u` + 明示マスク) のみ。

#### 実装上の必須事項 (既存 env の落とし穴)

現状の割当条件 [drp_env.py:883](../src/main/drp_env/drp_env.py#L883) は

```python
if (self.assigned_tasks[i] == [] or i in self.assigned_list) and task_assign[i] != -1:
```

で、`i in assigned_list` (= 既にピック前タスクを持つ agent) も再割当に入る。**つまり部分的には既に再割当を許している** が、新タスクへ振り替えるとき [drp_env.py:895-897](../src/main/drp_env/drp_env.py#L895-L897) が **旧タスクの `assigned_list` エントリを `-1` に戻していない**。結果、旧タスクが「i に割当済み」のまま宙に浮き、`assigned_list` に i が二重に残る。

→ 本仕様を正式採用するなら、**再割当時に旧タスクを解放する (`assigned_list[old_idx] = -1` に戻す) 処理を必ず入れる**のが第一の修正点。

#### 主リスク: 振動 (thrashing) と緩和策

ピック前タスクを頻繁に振り直すと **A→B→A と往復して移動が無駄になる**。緩和策:

1. **切替コストを Allocator の学習信号に入れる**: 振り替えで生じた無駄走行を報酬に反映しないと churn のコストを学習できない ([§12 割当変動率](#12-評価設計) の指標がこれに対応)
2. **ヒステリシス**: 再割当には現割当より一定マージン以上の Q 改善を要求する
3. イベント駆動側で、振り直しを誘発するイベントの粒度を絞る (例: 微小変化では再割当イベントを発火させない)

#### フラグ化と評価

- `alma_allow_reassign_before_pickup` (bool) で切替可能にし、**デフォルトはロック版 (再割当 OFF)** とする。CLAUDE.md の baseline 不変・段階導入方針に整合。
- 評価では **再割当 ON/OFF** で振動 (割当変動率) と完了数のトレードオフを見る ([§12](#12-評価設計))。

---

## 4. 設計方針

1. **段階的導入**: 既存 PPO/IQL/QMIX/PBS は壊さず、`use_alma: false` でデフォルト動作を完全維持
2. **責務分離**:
   - Allocator は ALMA 流に新規実装 (`src/alma/allocator/`)
   - Actor は **既存の epymarl IQL/QMIX を継承** し、サブタスク条件付け + 観測マスクを注入
3. **観測マスキングは env 側で実装**: ALMA 用の `lare_path_module` 同様に、env.step() がサブタスク条件付き観測 `o^a_{b_i}` を返すフックを追加
4. **サブタスク報酬分解**: 既存の `r_goal/r_coll/r_wait/r_move` を **割当タスクごとに分解**して $r_i^b$ を計算する関数を追加
5. **割当頻度はイベント駆動を正とする**: Allocator の起動は **イベント駆動** (新タスク到着・ピック完了・エージェント解放などのイベントで割当を解き直す。[§3.4](#34-ピックアップ前タスクの再割当-reassignment) と整合)。$N_t$ (`alma_n_t`) は「固定ポーリング周期」ではなく、**イベント発火時に何ステップ分まとめて振り直すかの予算**として解釈する。デフォルト 1 = 現状互換 (実質ステップ単位)、論文同様に N>1 も選べる
6. **LaRe との同時利用**: ALMA の Allocator/Actor の学習信号を LaRe-Task/LaRe-Path に置き換え可能 ([§8](#8-lare-との関係-相補性))
7. **PBS は対象外**: PBS は学習しない探索系なので、ALMA actor 候補から除く (env フラグでチェック)

---

## 5. 全体アーキテクチャ

```
┌─────────────────────────────────────────────────────────────────────────┐
│  LDRP + ALMA (Joint Hierarchical Learning)                              │
│                                                                         │
│  エピソード実行ループ (runner.py)                                          │
│                                                                         │
│   状態 s_t                                                               │
│     │                                                                   │
│     ├─ N_t ステップに 1 回 ───────────────────────────────────┐           │
│     │                                                         │           │
│     │  ┌─────────────────────────────────────────────────┐  │           │
│     │  │  ALMA Allocator (高位)                            │  │           │
│     │  │                                                   │  │           │
│     │  │  Subtask embed g_i = f^g(pickup, dropoff, age)   │  │           │
│     │  │  Agent embed   h_a = f^h(state^a)                │  │           │
│     │  │                                                   │  │           │
│     │  │  Auto-regressive:                                 │  │           │
│     │  │    for a in A:                                    │  │           │
│     │  │      b^a ~ softmax(g·h_a) (with mask)             │  │           │
│     │  │      g_{b^a} += f^u(g_{b^a}, h_a)                │  │           │
│     │  │  Sample N_p alloc, pick b* = argmax_b Q(s,b)     │  │           │
│     │  └─────────────────────────────────────────────────┘  │           │
│     │            │                                           │           │
│     │            ▼ 割当 b = {b^a}_a                          │           │
│     │                                                         │           │
│     ▼                                                         │           │
│   毎ステップ                                                              │
│   ┌─────────────────────────────────────────────────────┐              │
│   │  サブタスク観測マスキング (env)                          │              │
│   │  o^a_{b_i} = mask(s, agent=a, subtask=b^a)            │              │
│   └─────────────────────────────────────────────────────┘              │
│            │                                                            │
│            ▼                                                            │
│   ┌─────────────────────────────────────────────────────┐              │
│   │  ALMA Actor (低位, IQL/QMIX 系の拡張)                    │              │
│   │  u^a = arg max Q^a_i(o^a_{b_i}, ·; b)                │              │
│   └─────────────────────────────────────────────────────┘              │
│            │                                                            │
│            ▼ joint action u = {u^a}                                     │
│   env.step(u) → reward 分解:                                            │
│     r^b_i (per-subtask, local entities only)                            │
│     R = Σ r^b_i + 完了ボーナス (allocator 学習用)                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. コンポーネント設計

### 6.1 Allocator: 提案分布 $f(b|s;\phi)$

**ファイル**: `src/alma/allocator/proposal.py`

```python
class AllocatorProposal(nn.Module):
    """
    Pointer-network style allocation proposal.

    入力:
      agent_states: (B, N_a, D_a)
      subtask_states: (B, N_i, D_i)        # N_i は変動
      subtask_mask:  (B, N_i) 1=available, 0=already assigned/finished
    出力:
      logits per agent over (subtasks + idle)
      auto-regressive sample b: (B, N_a) in {0..N_i, idle}
    """
    def __init__(self, d_agent, d_subtask, d_emb, n_idle_logit=1):
        ...
        self.f_h = MLP(d_agent, d_emb)
        self.f_g = MLP(d_subtask, d_emb)
        self.f_u = MLP(2 * d_emb, d_emb)  # subtask embedding update
        # idle 用に学習可能な専用埋め込みを追加
        self.idle_embedding = nn.Parameter(torch.zeros(d_emb))

    def sample(self, agent_states, subtask_states, subtask_mask, agent_order=None):
        # agent ごとに自己回帰的にサンプリング
        # agent_order: ロールアウト順 (None なら 0..N_a-1)。ランダム順列を渡せる
        # 既割当タスクは subtask_mask で除外 (LDRP の 1-task-per-agent 制約)
        ...
        # log_prob は「実際に使った agent_order での」確率を返す
        return b, log_prob
```

#### 割当順序のランダム化 (agent-0 先手バイアスの除去)

自己回帰の `a = 0,1,…,N_a-1` は **ロールアウト順 (誰から先にサンプルするか) にすぎず、agent identity に縛られない** (各因子は index でなく agent 状態 `h_a = f^h(s^a)` で条件付け)。順序が与える唯一の効果は「先にサンプルした agent の選択を `f^u` 更新で後続が考慮する」= **先手の優先権**で、これは TP の agent-0 総取りバイアスと同種。

→ 順序をランダム化して平均化できる。選択肢:

| 案 | 内容 | 効果 |
| --- | --- | --- |
| A | step ごとに順列をシャッフル | 先手バイアスを平均化 |
| B | **N_p サンプルごとに別の順列** (推奨) | `argmax_b Q` が順序非依存なのでバイアスが洗い流される |
| C | 意味のある順序でソート (緊急度・近接) | バイアス除去でなく事前知識の注入 |

**学習上の追加コストはほぼゼロ**:

- `sample()` は「実際に使った順序での `log_prob`」を返す (上記シグネチャ)
- critic `Q(s, b)` は割当 `b` だけの関数で **順序に依存しない** ([§6.2](#62-allocator-価値関数-qs-b-theta))
- 提案分布損失 `-log f(b*|s) - λH` は sample 時に記録した `log_prob` をそのまま使えば整合

**推奨は案B** (N_p サンプルごとに順列ランダム)。amortized Q-learning と相性が良く、対称 agent の先手バイアスを構造的に消せる。agent が非対称でも条件付けは `h_a` なので安全。再現性が要るなら順列のシードを固定する。

### 6.2 Allocator: 価値関数 $Q(s, b; \Theta)$

**ファイル**: `src/alma/allocator/critic.py`

```python
class AllocationCritic(nn.Module):
    """
    入力: state s + 割当 b (サブタスク埋め込みとエージェント埋め込みの結合)
    出力: スカラー Q(s, b)
    """
    def __init__(self, d_emb):
        ...
        self.attn = SelfAttention(d_emb, ...)
        self.head = nn.Linear(d_emb, 1)

    def forward(self, agent_states, subtask_states, b):
        # b に従って各エージェントを対応サブタスクと結合し、
        # attention で集約してスカラー出力
        ...
```

#### 実装上の制約 (将来の動的エージェント数拡張のため): per-agentトークン方式を維持する

`AllocationCritic` は **各エージェント/サブタスクをトークンとして attention に渡し、集約は attention 側に任せる**設計を維持すること。[modules/critics/mat.py:43](../src/epymarl/src/modules/critics/mat.py#L43) の `MATCritic` のように「他の全エージェントの行動を1本のベクトルに連結してから入力する」方式は避ける。

理由: 連結方式は入力次元に `n_agents` が literal に埋め込まれる (`input_shape += n_actions * self.n_agents`, [mat.py:80](../src/epymarl/src/modules/critics/mat.py#L80))。この場合 `nn.Linear` の重み行列自体が `n_agents` 依存になり、[dynamic_agent_count.md](dynamic_agent_count.md) の N_max + アクティブフラグ方式 (入力次元は固定したまま、非アクティブ分をマスクで無効化する) を適用できなくなる。per-agentトークン方式であれば、重み行列は要素数 (集合のサイズ) に依存しないので、`subtask_mask` と対称な `agent_mask` を追加するだけで動的エージェント数に対応できる。

### 6.3 Allocator モジュール

**ファイル**: `src/alma/allocator/module.py`

`LaRePathModule` 同様、(env, config) で初期化。バッファ・最適化器・更新ループを内包。

```python
class AlmaAllocator:
    def __init__(self, env, cfg):
        self.proposal = AllocatorProposal(...)
        self.critic   = AllocationCritic(...)
        self.target_critic = deepcopy(self.critic)
        self.buffer   = AllocatorBuffer(...)
        ...

    def select_allocation(self, env_state):
        """N_t ステップに 1 回呼ばれる"""
        agent_states, subtask_states, mask = self._extract_state(env_state)
        # サンプル N_p 個
        candidates = [self.proposal.sample(...) for _ in range(self.cfg.n_proposals)]
        Qs = [self.critic(agent_states, subtask_states, b) for b in candidates]
        b_star = candidates[argmax(Qs)]
        return b_star

    def store_transition(self, s_t, b_t, R_seg, s_tNt):
        """N_t ステップ分の累積報酬を allocator バッファに格納"""
        ...

    def update(self):
        # critic loss (TD with b* on target)
        # proposal loss (-log f(b*) - λ H)
        ...
```

### 6.4 Actor (低位): サブタスク条件付き IQL/QMIX

**ファイル**: `src/alma/actor/`

epymarl の IQL/QMIX エージェントを継承し、入力に **サブタスク埋め込み** を結合する形にする。

- 観測 $o^a_{b_i}$ は env が返してくれる (マスキング済み)
- 追加で、自分の現在タスク (pickup/dropoff/idle) を one-hot or 埋め込みベクトルで concat
- Q^a の出力次元は既存と同じ (= n_actions)

epymarl とどう統合するか 2 案:

1. **A. epymarl 内部に組み込み**: `src/epymarl/src/modules/agents/` にサブタスク条件付き agent を追加
2. **B. ラッパー方式**: epymarl はそのまま、env が観測を `(masked_obs, subtask_id)` の concat 形式で返す

A は綺麗だが epymarl への侵入が大きい。B は簡易だが正確な学習信号 (サブタスク Q の独立性) を保ちにくい。**初期実装は B、本格実装は A を検討**。

### 6.5 サブタスク観測マスキング (env)

**変更ファイル**: `src/main/drp_env/drp_env.py`

`env.step()` の戻り値 `obs` に対して、ALMA 有効時のみマスキング後の観測を返す。

```python
def _alma_mask_obs(self, agent_id, b_a, full_obs):
    """
    agent_id が割り当てられているタスク b_a の関連エンティティだけ残し、
    それ以外を 0 / -1 / dummy に置換した観測を返す.
    """
    if b_a == -1:  # idle
        return zeros_like(full_obs)
    pickup, dropoff = current_tasklist[b_a]
    # ノード位置のうち {pickup, dropoff, agent's pos} のみ残す等
    ...
```

obs_repre が `onehot_fov` の場合は既に近傍だけ見えているので、追加マスクは「ゴール = 自分のタスクの pickup/dropoff のみ」で済む。

### 6.6 サブタスク報酬分解 (env)

**変更ファイル**: `src/main/drp_env/drp_env.py`

```python
def _alma_decompose_rewards(self, ri_array, b):
    """
    各エージェントの報酬 ri_array を、彼らが割り当てられているサブタスクごとに集約.
    Returns: dict {subtask_idx: scalar reward, "idle": float}
    """
    per_subtask = defaultdict(float)
    for a in range(self.agent_num):
        if b[a] == -1:
            per_subtask["idle"] += ri_array[a]
        else:
            per_subtask[b[a]] += ri_array[a]
    return per_subtask
```

これを actor の per-subtask Q 学習に使う。

### 6.7 ALMA 全体モジュール

**ファイル**: `src/alma/alma_module.py`

```python
class AlmaModule:
    def __init__(self, env, cfg):
        self.allocator = AlmaAllocator(env, cfg.allocator)
        self.actor     = AlmaActor(env, cfg.actor)
        self.cfg = cfg
        self.t_since_alloc = 0
        self.current_b = None

    def select_joint_action(self, env_state):
        if self.t_since_alloc % self.cfg.N_t == 0 or self.current_b is None:
            self.current_b = self.allocator.select_allocation(env_state)
        u = self.actor.select_actions(env_state, self.current_b)
        self.t_since_alloc += 1
        return u, self.current_b

    def step_callback(self, transition):
        # actor 用: per-subtask 報酬とともに格納
        # allocator 用: N_t ステップ累積報酬
        ...

    def end_episode(self):
        self.actor.end_episode()
        self.allocator.end_episode()  # 必要なら更新トリガ
```

---

## 7. 既存実装との接続

### 7.1 runner.py

```python
# 現状
agents_action = self.path_planner.policy(obs_n, self.env)
task_assign   = self.task_manager.assign_task(self.env)

# ALMA 有効時
if self.use_alma:
    u, b = self.alma.select_joint_action(self.env)
    # u: 移動行動, b: 割当 (現状の task_assign と同型)
    joint_action = {"pass": u, "task": b}
else:
    # 現状通り
    ...
```

### 7.2 src/main/drp_env/drp_env.py

新規パラメータ (LaRe と同パターン):

```python
def __init__(..., use_alma=False, alma_n_t=1, alma_subtask_obs_mask=True,
             use_pretrained_alma=False, pretrained_alma_model_path=None,
             use_finetuning_alma=False, finetuning_alma_model_path=None,
             alma_autosave=False, alma_autosave_path=None, alma_save_dir=None,
             # ALMA hyperparams (下部)
             alma_d_emb=64, alma_n_proposals=10, alma_lambda_aql=0.01, ...):
```

`step()` 内に LaRe フックと同様の場所で:

```python
if self.use_alma:
    # 1) サブタスク観測マスキング (返す obs を差し替え)
    obs = self._alma_mask_obs(...)
    # 2) サブタスク報酬分解 (info に格納)
    info["alma_subtask_rewards"] = self._alma_decompose_rewards(ri_array, b)
```

### 7.3 epymarl との関係

ALMA Actor を epymarl IQL/QMIX として使う場合、現状の epymarl の学習ループはそのまま流用可能。ただし:

- 環境の観測がサブタスク条件付きにマスクされている
- サブタスク id を obs にコンカット (または embedding として渡す)

ALMA Allocator の学習は epymarl の枠外で動かす (= LaRe-Path/Task と同じ場所)。

---

## 8. LaRe との関係 (相補性)

ALMA と LaRe は **直交的に組み合わせ可能**:

| 組合せ | 何が起きるか |
|---|---|
| **ALMA off + LaRe off** | 現状の LDRP (PPO/TP + IQL/QMIX) |
| **ALMA on + LaRe off** | 構造改善 (階層化) のみ。割当・行動を同時最適化 |
| **ALMA off + LaRe on** | 報酬改善のみ (現実装の拡張) |
| **ALMA on + LaRe on** | 構造 + 報酬の二重改善 |

特に注目したい連携:

1. **ALMA Allocator の R を LaRe-Task で置き換え**
   - 現案では Allocator は $\sum_n^{N_t} r_{t+n}$ を学習信号にする
   - これを LaRe-Task の proxy 報酬にすると、N_t 区間の累積より「割当決定の品質」を直接反映できる
2. **ALMA Actor のステップ報酬を LaRe-Path で置き換え**
   - サブタスク内エージェントに対して、LaRe-Path の proxy 報酬を `r^b_i` の代わりに使う
   - 観測マスキング後の状態に対して LaRe-Path エンコーダを評価する必要があるので、エンコーダ側にもマスク対応の差し替えが必要

これらは **後段の拡張** とし、初期実装では ALMA 単独で動くことを優先する。

---

## 9. ファイル構成 (予定)

### 新規ファイル

```
LDRP/
└── src/
    └── alma/                              # 新規
        ├── __init__.py
        ├── alma_module.py                 # AlmaModule (top-level coordinator)
        ├── allocator/
        │   ├── __init__.py
        │   ├── proposal.py                # AllocatorProposal (pointer-net)
        │   ├── critic.py                  # AllocationCritic Q(s,b)
        │   ├── module.py                  # AlmaAllocator (training/inference)
        │   └── buffer.py                  # 高位レベルバッファ (N_t step segments)
        ├── actor/
        │   ├── __init__.py
        │   ├── policy.py                  # サブタスク条件付き IQL/QMIX
        │   ├── module.py                  # AlmaActor
        │   └── buffer.py                  # 低位レベルバッファ
        └── shared/
            ├── __init__.py
            └── attention.py               # 既存の src/lare/shared/attention.py を使い回しても良い
```

### 変更ファイル

```
src/main/drp_env/drp_env.py        # use_alma フラグ追加, step() にマスク+報酬分解フック
src/main/drp_env/__init__.py       # gym register に use_alma 追加
src/config/default.yaml            # ALMA 利用設定 + 内部パラメータ
runner.py                          # alma 有効時のジョイント方策呼び出し分岐
test.py                            # default.yaml の ALMA キー転送
MANUAL.md                          # 章追加 + 更新履歴
```

---

## 10. 実装フェーズ + 段階移行戦略

実装規模が大きいので段階分け。各フェーズで動く状態を保つ。

採用判断 (Step) と実装単位 (Phase) は別概念として整理する:

- **Step** = ALMA をどこまで採用するかの戦略レベル (採用判断ポイント)
- **Phase** = Step を実現するための実装作業単位

### 10.0 段階移行戦略 (Step 1 → 3)

各 Step の終端で性能評価し、効果がなければ次に進まない (= ALMA 不採用) という判断ができる構造にする。

| Step | 説明 | 含むフェーズ | 累積コスト目安 |
|---|---|---|---|
| **Step 1** | LaRe のみ (= 現状の dual-RL の報酬改善, 構造はそのまま) | 完了済み (LaRe-Path/Task 統合) | 0 (実装済み) |
| **Step 2** | ALMA Allocator のみ採用、低位は既存 IQL/QMIX/PBS を流用 | Phase 1-2 (+ 必要なら Phase 5 の保存/ロード) | 〜500-700 行 |
| **Step 3** | ALMA フル (joint training + 観測マスキング + 報酬分解) | Phase 3-4 (+ Phase 6 で LaRe 連携) | 累計 〜1500 行 |

#### Step 1 (実装済み) — 現 dual-RL の報酬改善のみ

- 経路 RL = IQL/QMIX (既存)
- 割当 RL = PPO + LaRe-Task proxy (既存実装)
- 各 RL は独立学習 (現状の sequential 学習を維持)
- **判定**: ベースライン (PBS-TP, IQL-PPO without LaRe) に対して有意な改善があるか

#### Step 2 — Allocator だけ ALMA 化、joint training なし

- 経路 RL = IQL/QMIX (既存, 固定または通常学習)
- 割当 RL = ALMA Allocator (PPO 置き換え)
  - 提案分布 + 組合せ Q (per-agent 独立 softmax より協調戦略を狙う)
  - LaRe-Task proxy を Allocator の学習信号に使うことも可
- **判定**: Step 1 (= LaRe のみ) と比較して、組合せ Q による協調が効くか

#### Step 3 — ALMA フル

- Allocator + Actor を joint training
- サブタスク観測マスキング有効化
- サブタスク報酬分解
- **判定**: Step 2 (= Allocator のみ) と比較して、joint training + マスキングの追加効果があるか

→ 論文の Joint vs 個別比較 (Figure 6b) に対応する切り分け実験。

### 各実装フェーズ

### Phase 0: 設計 (本ドキュメント) ← 今ここ

### Phase 1: 骨組み + dummy allocator

- `src/alma/` ディレクトリ作成
- `AlmaAllocator` をランダム割当で初期化 (既割当マスクのみ動く)
- `AlmaActor` は既存 IQL/QMIX のラッパー (サブタスク条件付けなし)
- `use_alma=true` で動作するが、性能はランダム ≒ FIFO 程度
- **目的**: フックポイント・interface が機能することを確認

### Phase 2: Allocator 学習

- `AllocatorProposal` の pointer-net 実装
- `AllocationCritic` の attention-based Q 実装
- `AllocatorBuffer` (N_t step segments)
- 提案分布 + critic の同時学習 (論文 Eq.3, 4)
- Actor は固定 (Phase 1 のラッパー継続)
- **目的**: 学習可能な Allocator が PPO/TP より良い割当を出すか測る

### Phase 3: Subtask 観測マスキング + 報酬分解

- env.step() にマスキングフック
- `_alma_decompose_rewards` 実装
- Actor の入力にサブタスク条件 (id or embedding) を結合
- 観測空間が縮むので Actor 学習効率が上がるはず
- **目的**: マスキング有 vs 無の比較で論文の "No mask" アブレーションと同じ傾向が出るか

### Phase 4: Joint training

- Allocator と Actor を同時学習
- 高位/低位それぞれのバッファとオプティマイザを並行更新
- **目的**: 論文同様 Joint > 個別 の効果を確認

### Phase 5: 保存/ロード/凍結

- LaRe と同じ 4 モード (off / scratch / pretrained / finetuning)
- 命名: `{Safe_}{ALGO}_ALMA_{map}_{N}agents_{X.X}M_checkpoint.pth`
  - 保存先: `src/alma/saved_models/`
  - Allocator + Actor のセットを 1 ファイルに保存

### Phase 6: LaRe 連携 (オプション)

- ALMA Allocator の累積報酬を LaRe-Task proxy で置換
- ALMA Actor のステップ報酬を LaRe-Path proxy で置換
- 観測マスキング後の状態でも LaRe エンコーダが動くよう調整

### Phase 7: 評価実験

下記 [§12](#12-評価設計) 参照。

---

## 11. パラメータ案 (default.yaml への追加)

`MANUAL.md` の 3 ゾーン構成に従う:

### 上部 (実験設定ゾーン)

```yaml
# --- ALMA (Hierarchical Allocator-Actor) - 利用設定 ---
use_alma: false
use_alma_training: true

use_pretrained_alma: false
pretrained_alma_model_path: null

use_finetuning_alma: false
finetuning_alma_model_path: null

alma_autosave: false
alma_autosave_path: null
```

### 下部 (ハイパーパラメータゾーン)

```yaml
# --- ALMA 内部 ---
alma_n_t: 1                      # イベント発火時の振り直し予算 (1 = 毎ステップ相当, >1 でステップ集約). 起動自体はイベント駆動 (§3.4/§4-5)
alma_d_emb: 64                   # 埋め込み次元
alma_n_proposals: 10             # 提案分布からのサンプル数 N_p
alma_lambda_aql: 0.01            # 提案分布のエントロピー係数 λ
alma_critic_hidden: 128
alma_actor_hidden: 128
alma_buffer_capacity: 1024
alma_min_buffer: 64
alma_update_freq: 16
alma_batch_size: 32
alma_lr_proposal: 0.0005
alma_lr_critic: 0.0005
alma_lr_actor: 0.0005
alma_target_update_tau: 0.01
alma_save_dir: null              # null = src/alma/saved_models/
```

---

## 12. 評価設計

### 比較対象

| 条件 | Allocator | Actor | LaRe |
|---|---|---|---|
| Baseline-PBS-TP | TP | PBS (固定) | OFF |
| Baseline-IQL-PPO | PPO | IQL | OFF |
| LaRe-Both | PPO + LaRe-Task | IQL + LaRe-Path | ON |
| **ALMA-only** | **ALMA** | **ALMA Actor** | OFF |
| ALMA-no-mask | ALMA (マスクなし) | ALMA Actor (マスクなし) | OFF |
| ALMA-no-joint | ALMA (固定 Actor) | IQL pretrained | OFF |
| **ALMA + LaRe** | ALMA + LaRe-Task | ALMA Actor + LaRe-Path | ON |

### 評価指標

- **タスク完了数 (主)**: `info["task_completion"]`
- **衝突率**: `info["collision"]`
- **平均タスク待機時間**: タスク作成 → ピックアップまでのステップ数
- **割当変動率**: N_t ステップ間で割当が変わった回数 (N_t > 1 のとき)
- **Allocator 損失**: 提案分布損失, critic 損失
- **Actor 損失**: per-subtask Q 損失

### 期待される結果

- ALMA-only > Baseline-IQL-PPO: 構造改善で割当戦略が改善
- ALMA-only > LaRe-Both: 構造の効果が報酬の効果を上回る (環境による)
- ALMA + LaRe > ALMA-only: 二重改善で最良
- ALMA-no-mask >> ALMA-only に劣る: 論文の主張通り、マスキング重要

### タスク完了数の改善幅見積 (推測)

LDRP の特性 (1 タスク = 1 エージェント, TP ヒューリスティックが既に強い) を踏まえた現実的な数値:

| 比較 | 予想改善 (タスク完了数) | 確度 |
|---|---|---|
| ALMA-only vs Baseline-PBS-TP | +5〜15% | 中 |
| ALMA-only vs Baseline-IQL-PPO | +10〜25% | 中-高 |
| ALMA-only vs LaRe-Both (Step 1) | +0〜10% | 低-中 |
| ALMA + LaRe vs LaRe-Both (Step 1) | +5〜15% | 中 |

**注意点**:

- 革命的な改善 (>30%) は期待しにくい (LDRP は 1:1 制約で ALMA の強みの一部が活きないため)
- TP ヒューリスティックは既に強いので、ALMA の改善は相対的に控えめになる可能性 (論文 SMAC でも一部 heuristic が善戦)
- マップサイズ・エージェント数が大きいほど ALMA の優位性が出やすい (組合せ Q の効果が顕在化)

各 Step 終端で **Step 1 (LaRe のみ) と比較して有意改善が出るか** を判定基準にする。出なければ次 Step に進まない。

---

## 13. リスク・未決事項

### 高リスク

| 項目 | 内容 | 緩和策 |
|---|---|---|
| **動的タスク数** | 提案分布のサブタスク embedding が長さ可変 | pointer-net 構造なら可変対応可。実装時に `subtask_mask` で扱う |
| **1-task-per-agent 制約** | ALMA は本来複数 agent OK だが LDRP は 1:1 | 自己回帰サンプリング時に既割当タスクをマスク |
| **割当 Q の学習不安定性** | 大きい組合せ空間 + sparse reward で探索難 | エントロピー項 λ を高めに開始, ε-greedy 探索, Np を大きく |
| **観測マスキングと既存 obs_repre の干渉** | onehot_fov / heu_onehot 等で挙動が違う | obs_repre ごとにマスク関数を実装. デフォルトは onehot_fov のみ対応 |
| **epymarl との結合粒度** | Actor を epymarl 内部に入れると侵襲的 | Phase 1-3 はラッパー方式で進め, Phase 4 以降に検討 |

### 中リスク

| 項目 | 内容 | 緩和策 |
|---|---|---|
| **LDRP の "サブタスク独立" の妥当性** | 論文では擬似的に独立と仮定. LDRP は衝突制約があり完全独立ではない | LDRP-safe wrapper では衝突回避が env で処理されるので影響限定 |
| **メモリ・計算コスト** | Allocator バッファ + Actor バッファで倍増 | バッファサイズと update_freq を控えめに |

### 未決事項 (設計レビュー時に決める)

1. **割当頻度 $N_t$ の初期値**: 1 か 5 か
2. **idle の扱い**: 専用の embedding にするか、ダミーのサブタスク embedding にするか
3. **Allocator Critic の状態入力**: 全 agent + 全 subtask 埋め込みを集約するか, グローバル obs を使うか
4. **Pretrained モデル形式**: Allocator と Actor を別ファイルにするか同一にするか (1ファイルが扱いやすそう)
5. **Phase 3 の観測マスキング**: 既存 obs_repre をどこまで尊重するか (例えば fov の半径制限と二重に効く)
6. **ピック前タスクの再割当 ([§3.4](#34-ピックアップ前タスクの再割当-reassignment))**: デフォルト ON/OFF, 切替コストの報酬反映方法, 振り直しを発火させるイベントの粒度
7. **割当順序 ([§6.1](#61-allocator-提案分布-fbsphi))**: 固定 (0..N_a) / step 毎ランダム / サンプル毎ランダム (推奨) / 意味順ソート のどれを既定にするか

---

## 14. 軽量代替案 (フル ALMA 採用しない場合)

フル ALMA (Phase 1-7 完走) は実装規模が大きいので、効果が見えなければ途中で止めて軽量版で済ませる選択肢も用意する。各案は [§10.0 段階移行戦略](#100-段階移行戦略-step-1--3) のどこに該当するかを明示する。

### 14.1 Allocator のみ (= Step 2)

- 経路: 既存 IQL/QMIX/PBS をそのまま使う
- 割当: ALMA Allocator (PPO 置き換え)
- 観測マスキング・報酬分解は実装しない
- joint training も実装しない
- **コスト**: 〜500 行 (Allocator 提案分布 + critic + バッファ)
- **得られるもの**: 組合せ Q による協調戦略のみ

### 14.2 Pointer-net + ヒューリスティックロジット

ALMA の amortized Q-learning を**省略** し、提案分布のロジットを学習するだけ:

- 提案分布の logit = エージェント埋め込みとサブタスク埋め込みの内積 (= ALMA の構造)
- ただしロジットを学習する代わりに **TP の近接スコア** (= ピックアップ距離の逆数) で初期化
- 学習信号として LaRe-Task proxy を使い、徐々にロジットを微調整
- **コスト**: 〜200-300 行
- **得られるもの**: TP の良い性質 (= 近接優先) を保ちつつ、データ駆動の補正

### 14.3 PPO + サブタスク観測マスクだけ追加

ALMA アーキは採用せず、現 PPO に env 側のマスキングだけ追加:

- 各 agent の観測を、自分の割当タスク関連だけに絞る
- PPO の入力次元が縮小 → 学習効率向上を狙う
- joint training は不要 (PPO/IQL が独立学習なのは現状通り)
- **コスト**: 〜100 行 (env.step() のマスク関数のみ)
- **得られるもの**: 観測マスキングの効果単独評価

### 採否の判断フロー

```text
Step 1 (LaRe のみ) で有意改善あり?
  └─ Yes → 14.3 (マスキング追加) を試す → 改善あれば採用、なければ Step 1 で確定
  └─ No  → Step 2 (= 14.1, ALMA Allocator) を試す
            └─ 改善あり → 14.2 (軽量 pointer-net) と比較 → 効果対コストで判断
            └─ 改善なし → ALMA 採用見送り (Step 1 + LaRe で確定)
```

軽量版で十分なら、フル ALMA (Phase 3-4) は**採用見送り** という結論もあり得る。

---

## まとめ

ALMA は LDRP の構造的弱点 (割当と経路の独立学習) を解消する有力な候補。LDRP の想定する **「経路 = MARL + 割当 = もう一つの RL」** という二段 RL 構成を、ALMA は同時学習・組合せ Q・観測マスキングの 3 点で原理的に強化する形で一般化したものと位置付けられる。

実装規模は LaRe より大きい (Allocator 自体が新規アーキテクチャ + サブタスク観測マスク + 報酬分解 + Actor 拡張)。Step 1 (LaRe 統合, 完了済み) → Step 2 (Allocator のみ) → Step 3 (フル) と段階移行し、各 Step 終端で性能を見て次に進むかを判定する戦略にした。軽量版 ([§14](#14-軽量代替案-フル-alma-採用しない場合)) もあるので、効果対コストで適切な落としどころを選べる。

LaRe-Path / LaRe-Task と直交するため、ALMA on/off × LaRe on/off の 4 条件比較が可能。

**次のステップ**: ユーザーがこの設計を承認した後、Step 2 着手の可否を Step 1 (LaRe のみ) の評価結果次第で判断する。Step 2 採用時は Phase 1 (骨組み + dummy allocator) から実装。

---

最終更新: 2026-05-10
