# 設計書: 複数タスク保持エージェントへの拡張

**作成日:** 2026-05-22
**ステータス:** 設計案 (実装はまだ)
**対象リポジトリ:** [https://github.com/Yamaguchi-yushi/LDRP](https://github.com/Yamaguchi-yushi/LDRP) (本リポ)
**関連:**
- [alma_integration.md](alma_integration.md) (Allocator 側の拡張点)
- [lare_integration.md](lare_integration.md) (LaRe-Task 因子の再定義点)
- [../MANUAL.md](../MANUAL.md)

> **位置付け:** 本設計書は ALMA 設計書 ([alma_integration.md](alma_integration.md)) の §15 として書かれていた「複数タスク保持エージェントへの拡張」を独立した設計書として切り出したもの。ALMA 採用判断とは独立に、将来の研究方向として整理する。

---

## 目次

1. [動機と現状分類](#1-動機と現状分類)
2. [環境改修のスコープ](#2-環境改修のスコープ)
3. [設計選択肢 (3 通り)](#3-設計選択肢-3-通り)
4. [関連研究](#4-関連研究)
5. [ALMA / LaRe との互換性](#5-alma--lare-との互換性)
6. [環境改修以外の本当の課題](#6-環境改修以外の本当の課題)
7. [推奨ロードマップ](#7-推奨ロードマップ)
8. [まとめ](#8-まとめ)

---

## 1. 動機と現状分類

将来の方向性として、各エージェントが複数のタスクを同時に保持できる環境への拡張を検討中。
現状 LDRP は Gerkey-Mararic 分類で **ST-SR-IA** (Single-Task agents)。これを **MT-SR-IA / MT-SR-TA** (Multi-Task agents) に拡張すると、問題が単純 dispatch から VRP/TSP 系の **tour 計画問題** に変質する。

| 分類記号 | 意味 | LDRP の現状 / 目標 |
|---|---|---|
| ST-SR-IA | 単タスク・単ロボット・即時割当 | 現状 |
| MT-SR-IA | 多タスク保持・単ロボット担当・即時割当 | 目標 (本設計書のスコープ) |
| MT-SR-TA | 多タスク保持・時間延長割当 | さらに先 |

---

## 2. 環境改修のスコープ

ユーザの当初の懸念は「目的地が複数になり元環境の大改修が必要」だが、実は影響範囲は限定的:

| 要素 | 変更要否 | 内容 |
|---|---|---|
| `assigned_tasks[i]` | 変更 | `[task]` → `[task1, task2, ...]` のキュー化 |
| `goal_array[i]` | 変更 | 単一ノード → キュー先頭タスクの現在ステージ (pickup/dropoff) を都度導出 |
| `step()` の goal 切替 | 小変更 | 完了時に dequeue → 次タスクへ自動遷移 (現 pickup→dropoff 切替ロジックを再利用) |
| 衝突判定 | 無変更 | エージェント位置のみ依存 |
| `task_assign` API | 変更 | 「何 step に何タスクまで追加するか」のセマンティクス再定義 |
| 観測 (obs_repre) | 中変更 | 複数の pending タスクの符号化が必要 |
| `info["task_completion"]` | 無変更 | カウント自体は変わらず |
| LaRe-Path | ほぼ無変更 | 経路系エンコーダはタスク数に独立 |
| LaRe-Task | 変更 | 一部因子 (idle_assignment, pickup_proximity 等) の再定義 |
| PBS | 中変更 | 既存 PBS は単一目的地前提. tour 版が必要 |
| ALMA Allocator | 大変更 | 1:1 制約解除でアーキ刷新 |

→ **元環境への変更は moderate (≒数百行)。本当の重さは Allocator 側の tour 計画化**。

---

## 3. 設計選択肢 (3 通り)

3 案ともすべて **イベント駆動** (= 毎ステップではなく特定の状態遷移時のみ Allocator を呼ぶ) で動かす前提。LDRP のタスクは離散イベント (到着・完了) なので、毎ステップ呼ぶ必要がない。

### 3.1 共通のトリガーイベント

| トリガー | 発生条件 | Allocator が判断する内容 |
|---|---|---|
| **新タスク発生** | `current_tasklist` に新タスクが追加された step | 誰のキューに入れるか / 入れないか |
| **タスク配送完了** | あるエージェントの先頭タスクが完了 (`task_completion += 1`) | (B/C のみ) 残キューの順序 / 他 agent への再配分 |
| **エージェントアイドル** | キューが空になった | (B/C のみ) 未割当タスクから取得するか |
| **(オプション) 定期リプラン** | N step ごと | (C のみ) 全 agent の tour 全面見直し |

### 3.2 3 案の差分 = "Allocator が変えられる範囲"

| 案 | 「何を変えられるか」 | トリガー時の処理 | Env 変更 | Allocator 設計 | 上限性能 | 実装規模 |
|---|---|---|---|---|---|---|
| **A. FIFO キュー** | 追加先 agent のみ (順序固定) | 新タスク発生時に末尾追加 | 小 | 既存 (どのタスクを誰のキューに積むか) | 低-中 | 〜200 行 |
| **B. 学習する tour 順** | 追加先 + その agent 内の tour 順 | 新タスク発生・完了時に**当該 agent のみ** tour 再計画 | 中 | tour 出力 (pointer-net auto-regressive) | 中-高 | 〜800 行 |
| **C. イベント駆動・全体再配分** | 全 agent 間でのタスク再分配も可能 | 新タスク・完了・アイドル時に**全 agent の tour** を再計画 | 大 | tour 計画と dispatch を統合 (CVRP-RL 系) | 高 | 〜1200 行 (毎ステップ判断より軽い) |

### 3.3 推奨は B、ただし C もイベント駆動なら現実的

- **A** は単純すぎて TP 系ヒューリスティック (Cheapest-Insertion) と差別化しにくい
- **B** は **ALMA Allocator の自然な拡張** (pointer-net auto-regressive を「複数選ぶ」に変えるだけ)
- **C** は当初「毎ステップ判断」想定で実装重く見積もったが、**イベント駆動に絞るとコールが疎になり実装規模も縮む**。論文 SMARTS-Routing 等のイベント駆動 dispatch は実用例あり。研究価値も高い

### 3.4 イベント駆動の利点

1. **計算コスト**: Allocator 呼び出しが疎 (1 エピソード数十回程度) になり、ALMA Step 2 と同程度のコストに収まる
2. **既存環境との整合**: 現状 LDRP もイベント駆動 (`task_assign[i] != -1` の判定はアイドル時のみ意味がある) なので、自然な拡張
3. **学習信号の質**: 各イベントが意味のある決定点なので、Q-learning のサンプル効率が良い (毎ステップ判断だと無意味な決定が大半を占める)

---

## 4. 関連研究

複数タスク保持の文脈で参考になる先行研究:

### 4.1 直接参考になる

| 研究 | 内容 | LDRP への教訓 |
|---|---|---|
| **Kool et al. 2019** "Attention, Learn to Solve Routing Problems" (ICLR) | Transformer で VRP/CVRP, tour を auto-regressive 生成 | tour 生成器設計パターン. Allocator を tour-aware にする際の参考 |
| **Lin et al. 2018** "Efficient Large-Scale Fleet Management via MA-DRL" (KDD, Didi) | 配車プラットフォーム, 多数 AGV が複数オーダー保持 | 状態圧縮 (周辺情報のみ), 大規模 agent 数の扱い |
| **Choo et al. 2022** "Simulation-guided Beam Search for NCO" | VRP のビームサーチ + RL | Allocator の探索/活用バランス |
| **MAPF-TA** (Ma et al. 2017, Liu et al. 2019, etc.) | Multi-Agent Path Finding + Task Assignment | LDRP に最も近い問題設定. 複数タスク保持・優先度・順序最適化 |
| **Capacitated VRP-RL** | 容量制約付き VRP の RL | LDRP の `len(assigned_tasks[i]) ≤ capacity` 制約と直接対応 |

### 4.2 部分的に参考になる

| 研究 | 内容 |
|---|---|
| **ALMA** (Iqbal et al. 2022) | 単一タスク前提なので直接は使えない. 1:1 → 1:N 拡張は論文の射程外 (= 拡張余地が research contribution) |
| **REFIL** (Iqbal et al. 2021, ICML) | エンティティ可変の MARL Q. multi-task agent でも attention で扱える可能性 |
| **MAVEN** (Mahajan et al. 2019) | 階層的探索. 複数タスク保持時の探索策に応用可 |

### 4.3 共通する教訓

1. **MAPF-TA 文献から**: 複数タスク保持下では「次にどれを取るか」が **tour 計画問題** になる. MAPF + VRP のハイブリッドで CBSS (Conflict-Based Search with Sets) などの古典手法と RL のハイブリッドが主流
2. **Kool et al. + Didi 論文から**: tour 生成は **pointer-network + attention** が定石. ALMA Allocator の subtask-pointer 構造を **tour 生成** に拡張するのは自然
3. **共通の難所**: 観測の **可変長** (各 agent の保持タスク数が異なる) → attention/transformer 系がほぼ必須

---

## 5. ALMA / LaRe との互換性

### 5.1 ALMA との接続

ALMA は **subtask あたり 1 agent (ST-MR-IA)** を仮定し、行動空間 \\(|I|^|A|\\)。複数タスク化すると:

- 行動空間 = 順序付き subset 選択 → \\(O(|I|! / (|I|-K)!)\\) (K = 容量)
- 提案分布の auto-regressive 構造を **「agent ごとに K 個まで sample」** に拡張すれば対応可
- ALMA §4.1 の多タスク拡張は論文化価値あり (= research contribution として狙える)

### 5.2 LaRe-Task との接続

7-10 因子のうち再定義が必要なもの:

| 因子 | 単タスク版 | 複タスク版 |
|---|---|---|
| idle_assignment | agent had no task = 1 | `1 - len(queue)/capacity` |
| pickup_proximity | dist(agent_pos, pickup) | **insertion cost**: tour に挿入する場合の距離増加 |
| load_balance | std(loads) | std(キュー長分布) |
| queue_drain | unassigned_after / task_num | 同じ概念 |
| 新規追加候補 | — | tour_efficiency, capacity_utilization, pickup_dropoff_ratio |

**LaRe-Path は経路の質だけ見るので影響なし**。

---

## 6. 環境改修以外の本当の課題

| 課題 | 内容 | イベント駆動で緩和されるか |
|---|---|---|
| **観測の可変長化** | 各 agent の保持タスク数が異なる → onehot_fov だと最大値固定にするか attention 化が必要 | × (本質的に必要) |
| **Tour 評価の難しさ** | 「良い tour とは何か」の報酬設計. 完了数だけだと長い tour が無視される | × (報酬設計の問題で別) |
| **PBS との非互換** | PBS は single-target 前提. tour 版 PBS を作るか、PBS は単タスク用と切り分け | × |
| **行動空間爆発** | 容量 K の組合せ × agent 数 | **○** イベント駆動で **trigger 時の partial allocation のみ** 決めれば良いので、毎回の判断空間は \\(O(\|I\|)\\) 〜 \\(O(\|I\| \\cdot K)\\) に圧縮可 |
| **学習データ量** | tour 計画は VRP 級 | **△** trigger が疎なので 1 エピソードあたりの学習サンプル数は減るが、各サンプルの情報量は濃くなる |

---

## 7. 推奨ロードマップ

現行 ALMA 計画 ([alma_integration.md](alma_integration.md)) への接続:

```text
Step 1 (現状, 完了): 単タスク + LaRe
       ↓
Step 2 (検証中): ALMA Allocator のみ (単タスク強化)
       ↓ 効果が出たら
Step 3 (= ALMA 設計書 Phase 3-4): ALMA フル (joint training + 観測マスキング)
       ↓
Step 4 (= 本設計書 Option A): 複数タスク化 — FIFO キューで env 改修
       ↓
Step 5 (= 本設計書 Option B): 学習 tour — pointer-net 拡張 + LaRe-Task 因子再定義
       ↓
Step 6 (research-grade, = 本設計書 Option C): CVRP-RL 統合
```

各 Step で **後方互換性を保つ** (capacity=1 ⇔ 単タスク = 現状) のがミソ。

---

## 8. まとめ

- 環境改修は moderate (数百行). 当初の懸念ほど大きくない
- **本質的な追加複雑性は Allocator 側の tour 計画化**
- 3 案ともすべて **イベント駆動** (新タスク発生 / タスク完了 / アイドル) で動かす. 毎ステップ判断は不要で、現実的な計算コストに収まる
- 3 案の差分は「呼び出し頻度」ではなく「Allocator が何を変えられるか」(キュー追加先のみ / 単 agent 内 tour 順 / 全 agent 間再配分)
- 先行研究を踏まえると **Option B (学習 tour, pointer-net 拡張)** が現実的な落としどころ. **Option C もイベント駆動なら実装規模は当初想定より圧縮可能**
- ALMA の単タスク仮定は構造的拡張 (auto-regressive 提案分布の K 個サンプル化) で解消可
- LaRe-Path は影響なし、LaRe-Task は因子の意味再定義が必要
- **実装の重さの順序** (再評価): 単タスク ALMA Step 2 << Step 4 (Option A) < Step 3 (フル ALMA) < Step 5 (Option B) < Step 6 (Option C, イベント駆動なら従来見積より縮む)

---

最終更新: 2026-05-22
