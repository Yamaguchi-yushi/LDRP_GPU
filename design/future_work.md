# 将来実装メモ (TODO 集約)

軽量な「将来やりたい」「未適用の修正」を集約するファイル。重い独立設計書 (例: [multi_task_agents.md](multi_task_agents.md)) はここから参照のみ。

各項目は **背景 / 現状 / 対策案 / 影響範囲** の節構成で書く。実装に着手したら本ファイルから対応セクションを削除し、必要に応じて [../MANUAL.md](../MANUAL.md) の更新履歴に記録する。

---

## 目次

1. [GPU 環境への移行](#1-gpu-環境への移行)
2. [LaRe-Path 因子の正規化 (3 因子)](#2-lare-path-因子の正規化-3-因子)
3. [wait_count の連続 wait カウント化 (リセット追加)](#3-wait_count-の連続-wait-カウント化-リセット追加)
4. [encoder.py のログ整形 (at_goal の出力位置)](#4-encoderpy-のログ整形-at_goal-の出力位置)

### 重い設計書 (別ファイル)

- [multi_task_agents.md](multi_task_agents.md): 複数タスク保持エージェントへの拡張 (VRP/TSP 系の tour 計画問題)

---

## 1. GPU 環境への移行

### 背景

開発機は CPU のみで、コードは `torch.cuda.is_available()` で分岐済み。実機 GPU で動かすには PyTorch のバージョンアップが必要 (CUDA ビルドの新版)。

### 現状

- 現環境: torch 2.8 + numpy 1.26.4 (Python 3.9)
- CUDA: 開発機では使えない

### 既知の互換性問題

- **torch 2.9+ で [src/epymarl/src/components/episode_buffer.py:152](src/epymarl/src/components/episode_buffer.py#L152) が壊れる**:
  - 該当行 `new_data.transition_data[k] = v[item]` で `item` が list (非タプル)
  - torch 2.8 では DeprecationWarning だけで動作するが、2.9 から `x[seq]` の意味が変わり indexing が破綻する
  - **修正方法**: `v[item]` を `v[tuple(item)]` に変更 (1 行)

### GPU 移行時の他の確認項目 (要検証)

- `torch.cuda.is_available()` 分岐が想定通り True になるか
- LaRe-Path encoder の Dijkstra ループが CPU bound なので、GPU 化しても全体の速度向上が頭打ちになりやすい点
- epymarl の `args.use_cuda` 等の設定との整合 ([src/epymarl/src/config/default.yaml](src/epymarl/src/config/default.yaml))

### 影響範囲

- `requirements.txt` / `setup_env.sh` の torch バージョン指定変更
- `episode_buffer.py:152` の 1 行修正 (torch 2.9+ 移行時のみ)

---

## 2. LaRe-Path 因子の正規化 (3 因子)

### 背景

[CLAUDE.md](../CLAUDE.md) の不変条件「LaRe-Path の 10 因子は概ね [0, 1] に正規化」に対し、3 因子が **未正規化のまま** デコーダ MLP に渡されている。スケール差が大きい因子だけで proxy 報酬が予測される縮退状態に陥り、他因子の情報が学習に乗らないリスクあり。

### 現状の値域 ([encoder.py](../src/lare/path/encoder.py) より)

| # | 因子 | 計算式 | 実際の値域 | 状態 |
|---|---|---|---|---|
| 1 | `prog_goal` | dist_prev - dist_curr | **[-D, +D]** (D=graph_diameter ≈ 200) | ❌ |
| 2 | `in_collision` | 0 or 1 | {0, 1} | ✓ |
| 3 | `others_in_collision` | 0 or 1 | {0, 1} | ✓ |
| 4 | `wait_norm` | wait_count (生) | **[0, time_limit]** (= [0, 500]) | ❌ |
| 5 | `dist_goal_norm` | dist / D | [0, 1] | ✓ |
| 6 | `min_sep_norm` | min_sep / D | [0, 1] | ✓ |
| 7 | `avg_sep_norm` | avg_sep / D | [0, 1] | ✓ |
| 8 | `safety_margin` | min_sep / collision_dist, clip(0, 100) | **[0, 100]** | ❌ |
| 9 | `collision_risk` | 1 if min_sep < coll_dist*2 else 0 | {0, 1} | ✓ |
| 10 | `at_goal` | 1 if dist < eps else 0 | {0, 1} | ✓ |

### 対策案

#### prog_goal (距離変化)

進捗の **符号情報**は actor の学習に効くので、ただ clip するより向きを残したい。

| 案 | 式 | 範囲 | 評価 |
|---|---|---|---|
| A | `prog_goal / D` | [-1, +1] | 情報損失なし。他因子 [0,1] と符号スケールが微妙 |
| B | `(prog_goal / D + 1) / 2` | [0, 1] | 0.5 を「中立」とするシフト。他因子と完全に揃う |
| C | `clip(prog_goal / D, 0, 1)` | [0, 1] | 「進んだ量だけ評価」。MARL4DRP がこれなら踏襲 |

→ **MARL4DRP の参照実装 (`marl4drp-lookup` subagent) で揃え先を確認するのが安全**。

#### wait_norm (連続 wait 回数 - ※項目3も合わせて適用)

| 案 | 分母 | 範囲 | 評価 |
|---|---|---|---|
| A | `time_limit` (500) | [0, 1] | エピソード長依存 |
| B | 固定定数 (例: 20〜50) | [0, 1] clip | 「N step 以上待っている = 完全に詰まっている」のセマンティクス |
| C | `1 - exp(-w / τ)` (τ=5〜10) | [0, 1) | 少数回 wait に強く反応、長期 wait は飽和 |

#### safety_margin

`collision_risk` (#9) が既に「margin < 2 でアラート」を 0/1 で出しているので、本因子の本質は「衝突距離の何倍離れているか」の連続値情報。margin > 1 (衝突距離より遠い) は全部 1 扱いでも情報損失少ない可能性。

| 案 | 式 | 範囲 | 評価 |
|---|---|---|---|
| A | `clip(margin, 0, 1)` | [0, 1] | margin > 1 は全部 1。情報潰れる |
| B | `min(margin / K, 1)` (K=5〜10) | [0, 1] | 中間域の情報を保つ |
| C | `1 - exp(-margin / τ)` | [0, 1) | 近距離で 0、遠距離で漸近的に 1 |

### 影響範囲

- [src/lare/path/encoder.py](../src/lare/path/encoder.py): 3 因子の正規化式変更
- 既存学習済みモデル (`.pth`) は **デコーダ入力スケールが変わるので再学習が必要**

---

## 3. wait_count の連続 wait カウント化 (リセット追加)

### 背景

`wait_norm` は「**連続でどれくらい wait しているか**」を意図した設計だが、現実装は **エピソード開始からの累積カウント** になっている。動いた瞬間にリセットされず、過去の wait 回数を保持し続ける。

### 現状

[src/main/drp_env/drp_env.py:737, 742](../src/main/drp_env/drp_env.py#L737):

```python
for i in range(self.agent_num):
    action_i = joint_action[i]
    if action_i not in self._get_avail_agent_actions(i, self.n_actions)[1]:
        ...
        self.wait_count[i] += 1   # ① unavailable で wait
    elif self.pos[int(action_i)][0]==self.obs[i][0] and ...:
        ...
        self.wait_count[i] += 1   # ② action=現在位置で wait
    else:
        # 動いた分岐 — ★リセットが入っていない★
        self.current_goal_prepare[i] = joint_action[i]
        ...
```

### 対策案

else 分岐 (動いた step) の冒頭で 1 行追加:

```python
else:
    self.wait_count[i] = 0   # 動いたら連続wait カウントを 0 にリセット
    self.current_goal_prepare[i] = joint_action[i]
    ...
```

### 影響範囲

- [src/main/drp_env/drp_env.py:754](../src/main/drp_env/drp_env.py#L754) 周辺の 1 行追加
- `wait_norm` 因子の意味が変わる (累積 → 連続)
  - 既存学習済みモデルとは互換性が崩れる (因子のセマンティクスが変わるため再学習推奨)
- 項目 2 の `wait_norm` 正規化と合わせて適用するのが自然

---

## 4. encoder.py のログ整形 (at_goal の出力位置)

### 背景

[src/lare/path/encoder.py:166-175](../src/lare/path/encoder.py#L166-L175) のデバッグ print で、各 step の出力が:

```
prog_goal: ...
in_collision: ...
...
collision_risk: ...
at_goal: ...      ← 最後
```

の順で出るため、連続する step を流し見すると `at_goal` が前 step の末尾と次 step の `prog_goal` の間に挟まり、**重複表示しているように見えて紛らわしい**。

### 対策案

3 つの方向性:

| 案 | 内容 | 評価 |
|---|---|---|
| A | step ごとに区切り (`print("---")`) を入れる | 最小変更 |
| B | `at_goal` を `prog_goal` の前 (= 各 step の最初) に移動 | 並び順の意味付けを変える |
| C | 全因子を 1 行にまとめる (`print(f"step N: prog={...}, at_goal={...}, ...")`) | 読みやすい |

### 影響範囲

- [src/lare/path/encoder.py](../src/lare/path/encoder.py) の print 文の並び替えのみ
- そもそもこの print 文は **デバッグ用** なので、本格修正前にまず削除/環境変数ガード化する選択肢もある

---

最終更新: 2026-06-01
