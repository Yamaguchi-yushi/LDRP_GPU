# 設計書: LDRP 拡張 (タスク再配布 / 複数タスク保持)

**作成日:** 2026-05-22 (複数タスク部分) / 2026-06-13 改稿 (タスク再配布を高優先度として追加・再構成)
**ステータス:** 設計案 (実装はまだ)
**対象リポジトリ:** [https://github.com/Yamaguchi-yushi/LDRP](https://github.com/Yamaguchi-yushi/LDRP) (本リポ)
**関連:**
- [alma_integration.md](alma_integration.md) (Allocator 側の拡張点 / 再配布の §3.4 詳細)
- [lare_integration.md](lare_integration.md) (LaRe-Task 因子の再定義点)
- [task_assign_rl.md](task_assign_rl.md)
- [../MANUAL.md](../MANUAL.md)

> **位置付け:** LDRP の **タスク割当 (task assignment) 機能を拡張する 2 方向**を、優先度をつけて整理した設計書。
> 元は ALMA 設計書 §15 の「複数タスク保持エージェント拡張」を独立させた `multi_task_agents.md` だったが、
> 配送ロボット環境では **1 体が複数荷物を同時保持する設定は一般的でない**ため複数タスク拡張は優先度を下げ、
> 代わりに **ピックアップ前のタスク再配布**を高優先度の拡張として前面に据えた (本ファイルへ改名・再構成)。

---

## 目次

1. [拡張方針と優先度](#1-拡張方針と優先度)
2. [【高優先度】タスク再配布 (ピックアップ前再割当)](#2-高優先度タスク再配布-ピックアップ前再割当)
3. [【低優先度】複数タスク保持エージェント](#3-低優先度複数タスク保持エージェント)
4. [推奨ロードマップ](#4-推奨ロードマップ)

---

## 1. 拡張方針と優先度

LDRP の割当を強化する方向は 2 つある。配送ロボットという問題設定に照らして優先度を付ける。

| 優先度 | 拡張 | 内容 | 理由 |
|---|---|---|---|
| **高** | **タスク再配布** | ピックアップ**前**のタスクを別エージェントへ振り直せるようにする | 配送ロボットで自然 (近くで空いた別 agent に渡す等の協調)。env 改修も軽量。ALMA / 既存 PPO/TP のどちらでも効く |
| **低** | **複数タスク保持** | 1 体が複数荷物を同時保持し tour を最適化 (MT-SR) | **配送ロボット環境では複数荷物の同時保持が一般的でない**。問題が VRP/TSP 系 tour 計画に変質し env / Allocator の改修も大きい |

- 両者は独立。再配布は「ownership を動的にする」拡張、複数タスクは「1 agent の保持数を増やす」拡張。
- まず再配布 (§2) を入れ、必要になったら複数タスク (§3) に進む、という段階。

---

## 2. 【高優先度】タスク再配布 (ピックアップ前再割当)

### 2.1 動機

現状 LDRP はタスクを一度エージェントに割り当てると **ピックアップ前でも固定**される。これを、まだピックアップしていない割当済みタスクに限り **より適したエージェントへ振り直せる**ようにする。これは追加仕様というより「割当を再最適化する素の挙動」を正しく許可するもの。恒久ロックすると「近くで空いた別 agent に渡す」といった協調的な振り直しを自分で封じてしまう。

ALMA Allocator を入れる場合の詳細 (Allocator から見た割当問題・提案分布のマスク等) は [alma_integration.md §3.4](alma_integration.md#34-ピックアップ前タスクの再割当-reassignment) を正典とする。本節は **割当器に依存しない env / task_assign レベルの仕様**としてまとめる (PPO/TP でも有効)。

### 2.2 状態とロック

既存 env の不変条件がそのまま対応する:

- `current_tasklist` = **ピックアップ前のタスクだけ** (ピック到達時に `pop` される)
- ピック後のタスクは `current_tasklist` から外れ `assigned_tasks[i]` にのみ残る (ドロップで解放)

| タスク状態 | 居場所 | 再割当 |
|---|---|---|
| 未割当 | `current_tasklist`, `assigned_list[idx] == -1` | 可 |
| 割当済み・**ピック前** | `current_tasklist`, `assigned_list[idx] == i` | **可 (本仕様)** |
| **ピック後**・配達中 | `current_tasklist` から除外済み, `assigned_tasks[i]` のみ | **不可 (ロック)** |

→ 「再割当可能 = `current_tasklist` に居るタスク」「ロック = ピック済み」が**追加の状態管理なしで成立**する。LDRP には「積荷を途中で降ろす」動作が無いため、ピック後ロックは仕様上の必然。

### 2.3 現状の実装と必須修正 (assigned_list 解放漏れ)

現状の割当条件 [drp_env.py:924](../src/main/drp_env/drp_env.py#L924):

```python
if (self.assigned_tasks[i] == [] or i in self.assigned_list) and task_assign[i] != -1:
    ...
    self.assigned_tasks[i] = self.current_tasklist[r]
    self.goal_array[i] = self.assigned_tasks[i][0]   # goal をピックノードへ
    self.assigned_list[task_assign[i]] = i           # 新タスクを i に割当
```

`i in self.assigned_list` (= 既にピック前タスクを持つ agent) も分岐に入るので、**部分的には既に再割当を許している**。しかし新タスクへ振り替えるとき、**旧タスクの `assigned_list` エントリを `-1` に戻していない** ([drp_env.py:936-938](../src/main/drp_env/drp_env.py#L936-L938))。結果、旧タスクが「i に割当済み」のまま宙に浮き、`assigned_list` に i が二重に残る。

→ 本仕様を正式採用するなら、**再割当時に旧タスクを解放する処理を必ず入れる**のが第一の修正点:

```python
# i が既にピック前タスク (old_idx) を持っていて, 別タスク r に振り替える場合:
#   振り替え前に old_idx を解放する.
if i in self.assigned_list:
    old_idx = self.assigned_list.index(i)
    if old_idx != task_assign[i]:
        self.assigned_list[old_idx] = -1   # 旧タスクを未割当に戻す
```

(同一タスクへの再指定 `old_idx == r` は no-op にする。`assigned_list.index(i)` は i が複数残っている既存バグ状態では先頭しか返さない点に注意 — 解放処理を入れて二重残りを根絶するのが前提。)

### 2.4 主リスク: 振動 (thrashing) と緩和策

ピック前タスクを頻繁に振り直すと **A→B→A と往復して移動が無駄になる**。緩和策:

1. **切替コストを学習信号に入れる**: 振り替えで生じた無駄走行を報酬 (または LaRe-Task proxy) に反映しないと churn のコストを学習できない
2. **ヒステリシス**: 再割当には現割当より一定マージン以上の改善 (Q 改善 / 距離短縮) を要求する
3. **イベント粒度を絞る**: 微小変化では再割当イベントを発火させない (イベント駆動の trigger 設計と一体)

### 2.5 フラグ化と評価

- `allow_reassign_before_pickup` (bool) で切替可能にし、**デフォルトは OFF (再割当しない = 現状互換)**。CLAUDE.md の baseline 不変・段階導入方針に整合 (ON 時のみ §2.3 の解放処理が走る)。
  - ALMA 文脈の同等フラグは `alma_allow_reassign_before_pickup` ([alma_integration.md §3.4](alma_integration.md#34-ピックアップ前タスクの再割当-reassignment))。両者を揃えるか一本化するかは実装時に決める。
- 評価では **再割当 ON/OFF** で「割当変動率 (振動)」と「タスク完了数」のトレードオフを見る。

### 2.6 実装規模 / 影響範囲

- **env**: `allow_reassign_before_pickup` 追加 + §2.3 の解放処理 (〜30-50 行)。OFF のとき挙動は完全に現状一致。
- **task_assign**: 既存 PPO/TP/FIFO は OFF なら無変更。ON で「ピック前タスクも候補に含める」割当に拡張。
- **LaRe-Task**: 再割当を good/bad と評価する因子 (insertion 改善 / churn コスト) を足すと学習信号が締まる (任意)。LaRe-Path は経路の質のみ見るので影響なし。
- **ALMA**: 採用する場合は提案分布のマスク等を [alma_integration.md §3.4](alma_integration.md#34-ピックアップ前タスクの再割当-reassignment) に従う。

---

## 3. 【低優先度】複数タスク保持エージェント

> **優先度を下げた理由**: 配送ロボット環境では **1 体が複数荷物を同時に積んで回る設定が一般的でない**。
> 複数タスク化すると問題が単純 dispatch から VRP/TSP 系の **tour 計画問題**に変質し、env / Allocator の
> 改修規模も大きい。研究方向としては有効なので設計は残すが、再配布 (§2) の後に回す。

### 3.1 動機と現状分類

各エージェントが複数タスクを同時保持できる環境への拡張。現状 LDRP は Gerkey-Mararic 分類で **ST-SR-IA** (Single-Task agents)。これを **MT-SR-IA / MT-SR-TA** に拡張する。

| 分類記号 | 意味 | LDRP の現状 / 目標 |
|---|---|---|
| ST-SR-IA | 単タスク・単ロボット・即時割当 | 現状 |
| MT-SR-IA | 多タスク保持・単ロボット担当・即時割当 | 目標 (本節のスコープ) |
| MT-SR-TA | 多タスク保持・時間延長割当 | さらに先 |

### 3.2 環境改修のスコープ

採用方針は **Option B (学習する tour 順)** を推奨。env 改修は moderate (〜400-500 行)、本当の重さは Allocator の tour-aware 化 (〜800 行)。

| 要素 | 変更要否 | 内容 |
|---|---|---|
| `assigned_tasks[i]` | 変更 | `[task]` → 保持タスクの集合 (順序は Allocator が動的に選ぶ) |
| `goal_array[i]` | 変更 | 単一ノード → Allocator が集合から毎イベント「次に向かう先」を選択 |
| `step()` の goal 切替 | 中変更 | 完了時にイベント発火 → Allocator に次行動を問い合わせ. 自動 dequeue はしない |
| 衝突判定 | 無変更 | エージェント位置のみ依存 |
| `task_assign` API | **大変更** | 「次行動 = {保持タスク X を実行 / 新タスク Y をピックアップ / 待機}」へ拡張 |
| 観測 (obs_repre) | 中変更 | 保持タスク集合 + pending 一覧の符号化 (可変長) |
| LaRe-Path | ほぼ無変更 | 経路系エンコーダはタスク数に独立 |
| LaRe-Task | 変更 | 一部因子 (idle_assignment, pickup_proximity 等) の再定義 (§3.5) |
| PBS | 中変更 | 既存 PBS は単一目的地前提. tour 版が必要 |
| ALMA Allocator | **大変更** | 1:1 制約解除 + 出力候補集合を held タスクも含むよう拡張 (auto-regressive pointer-net の自然な拡張) |

### 3.3 設計選択肢 (3 通り, すべてイベント駆動)

毎ステップではなく特定の状態遷移時のみ Allocator を呼ぶ。共通トリガー: **新タスク発生 / タスク配送完了 / エージェントアイドル / (任意) 定期リプラン**。

| 案 | 「何を変えられるか」 | Env 変更 | Allocator 設計 | 上限性能 | 実装規模 |
|---|---|---|---|---|---|
| **A. FIFO キュー** | 追加先 agent のみ (順序固定) | 小 | 既存 (誰のキューに積むか) | 低-中 | 〜200 行 |
| **B. 学習する tour 順** (推奨) | 追加先 + その agent 内の tour 順 | 中 | tour 出力 (pointer-net auto-regressive) | 中-高 | 〜800 行 |
| **C. イベント駆動・全体再配分** | 全 agent 間でのタスク再分配も可能 | 大 | tour 計画と dispatch を統合 (CVRP-RL 系) | 高 | 〜1200 行 |

- **A** は単純すぎて TP 系ヒューリスティック (Cheapest-Insertion) と差別化しにくい (段階移行の中間目標として有用)。
- **B** は ALMA Allocator の自然な拡張 (pointer-net を「複数選ぶ」に変えるだけ)。
- **C** はイベント駆動に絞ればコールが疎になり実装規模も縮む。研究価値も高い。
- なお **C の「全 agent 間再分配」は §2 のピック前再配布の一般化**でもある (§2 = capacity 1 での再配分)。

### 3.4 イベント駆動の利点

1. **計算コスト**: 呼び出しが疎 (1 エピソード数十回) で ALMA Step 2 並みに収まる
2. **既存整合**: 現状 LDRP もイベント駆動 (`task_assign[i] != -1` はアイドル時のみ意味を持つ)
3. **学習信号の質**: 各イベントが意味のある決定点なので Q-learning のサンプル効率が良い

### 3.5 ALMA / LaRe との互換性

- **ALMA**: subtask あたり 1 agent (ST-MR-IA) 仮定。複数タスク化は行動空間 = 順序付き subset 選択になり、提案分布の auto-regressive を「agent ごとに K 個まで sample」に拡張すれば対応可 (論文化価値あり)。
- **LaRe-Task**: 再定義が要る因子 ↓。**LaRe-Path は経路の質だけ見るので影響なし**。

  | 因子 | 単タスク版 | 複タスク版 |
  |---|---|---|
  | idle_assignment | had no task = 1 | `1 - len(queue)/capacity` |
  | pickup_proximity | dist(agent_pos, pickup) | **insertion cost** (tour 挿入時の距離増加) |
  | load_balance | std(loads) | std(キュー長分布) |
  | 新規候補 | — | tour_efficiency, capacity_utilization, pickup_dropoff_ratio |

### 3.6 環境改修以外の本当の課題

| 課題 | 内容 | イベント駆動で緩和? |
|---|---|---|
| 観測の可変長化 | 各 agent の保持数が異なる → 最大値固定か attention 化 | × (本質的に必要) |
| Tour 評価の難しさ | 「良い tour とは」の報酬設計。完了数だけだと長 tour が無視 | × (報酬設計の別問題) |
| PBS との非互換 | PBS は single-target 前提。tour 版を作るか切り分け | × |
| 行動空間爆発 | 容量 K の組合せ × agent 数 | ○ trigger 時の partial allocation のみ決めれば \\(O(\|I\|\cdot K)\\) に圧縮可 |

### 3.7 関連研究 (要点)

- **Kool et al. 2019** (Attention, Learn to Solve Routing): Transformer で VRP を auto-regressive 生成 — tour 生成器の設計パターン
- **Lin et al. 2018** (KDD, Didi): 大規模 fleet 管理、状態圧縮
- **MAPF-TA** (Ma 2017 ほか): LDRP に最も近い。複数タスク保持・優先度・順序最適化 (CBSS 等の古典 + RL ハイブリッド)
- **Capacitated VRP-RL**: `len(assigned_tasks[i]) ≤ capacity` 制約と直接対応
- **ALMA / REFIL** (Iqbal): 可変エンティティの attention MARL。1:1→1:N 拡張は論文射程外 = research contribution の余地

---

## 4. 推奨ロードマップ

```text
Step 1 (完了): 単タスク + LaRe
       ↓
Step 2 (検証中): ALMA Allocator のみ (単タスク強化)
       ↓
Step 3 【高優先度・本書 §2】: ピックアップ前タスク再配布 (capacity 1 のまま ownership 動的化)
       ↓ 必要になったら
Step 4 (低優先度・本書 §3 Option A): 複数タスク化 — FIFO キューで env 改修
       ↓
Step 5 (本書 §3 Option B): 学習 tour — pointer-net 拡張 + LaRe-Task 因子再定義
       ↓
Step 6 (research-grade, 本書 §3 Option C): CVRP-RL 統合 (= 再配布 §2 の一般化)
```

各 Step で **後方互換性を保つ** (再配布 OFF / capacity=1 ⇔ 現状) のがミソ。再配布 (Step 3) は複数タスク (Step 4-6) の前提ではなく独立に入れられる。

---

最終更新: 2026-06-13
