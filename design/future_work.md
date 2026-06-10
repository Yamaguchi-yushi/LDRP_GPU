# 将来実装メモ (TODO 集約)

軽量な「将来やりたい」「未適用の修正」を集約するファイル。重い独立設計書 (例: [multi_task_agents.md](multi_task_agents.md)) はここから参照のみ。

各項目は **背景 / 現状 / 対策案 / 影響範囲** の節構成で書く。実装に着手したら本ファイルから対応セクションを削除し、必要に応じて [../MANUAL.md](../MANUAL.md) の更新履歴に記録する。

---

## 目次

1. [GPU 環境への移行](#1-gpu-環境への移行)
2. [LaRe-Path 因子の正規化 (3 因子)](#2-lare-path-因子の正規化-3-因子)
3. [encoder の距離計算が壊れている (エッジ上の partial onehot 破壊)](#3-encoder-の距離計算が壊れている-エッジ上の-partial-onehot-破壊)
4. [LaRe モデルの学習ステップ数と保存名の {X.X}M が一致しない](#4-lare-モデルの学習ステップ数と保存名の-xxm-が一致しない)

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

### GPU 化より先にやるべき最適化: APSP 前計算でテーブル参照化

現状 [encoder.py:75-113](../src/lare/path/encoder.py#L75-L113) の `dijkstra(start, goal)` は `evaluation_func` の内部関数で、**毎 step・各 agent・prev/curr 位置ごと**に隣接リストを組み直してヒープ探索する (距離キャッシュなし)。一方 `graph_diameter` は [lare_path_module.py](../src/lare/path/lare_path_module.py) の init で 1 回だけ計算済み = 「直径は 1 回・ペア距離は毎回」の非対称になっている。

**マップは run 全体で不変** (reset でも変わらない) なので、ペア距離も init で全点対最短距離 (APSP) を前計算してテーブル参照にできる。GPU 化よりこちらの方が距離系因子には効く。

- reset ごとですらなく **module init で N×N の APSP を 1 回**作れば十分。`graph_diameter` と同じパターンで統合でき、**APSP テーブルの最大有限値 = graph_diameter** として両者を 1 回の前計算にまとめられる。
- `functools.lru_cache` は不可: `dijkstra` が `evaluation_func` 内で毎回再定義されるため永続しない。
- 実装方針:
  1. module init で `apsp = compute_apsp(env)` (networkx `all_pairs_dijkstra_path_length` を dict→`np.full((N, N), graph_diameter)` 化、未接続は `graph_diameter`)
  2. `compute_factors` → `evaluation_func` に `apsp` を 1 本通す
  3. 内部の `dijkstra(start, goal)` 呼び出しを `apsp[start, goal]` 参照に置換 (`estimate_partial_distance` の重み補間はテーブル 2 引きに変わるだけで挙動同一)
- **値は完全に不変、速くなるだけ**。小マップ (N≈20〜40) では per-step コストは小さいので、**encoder がボトルネックと実測されてから**着手で十分。

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
| 4 | `wait_norm` | wait_count (連続カウント、リセット済み) | **[0, time_limit]** 理論上限。実態は数十 step 程度に収束 | ❌ |
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

#### wait_norm (連続 wait 回数。リセット動作は既に [drp_env.py:759](../src/main/drp_env/drp_env.py#L759) で適用済み、残るは正規化のみ)

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

## 3. encoder の距離計算が壊れている (エッジ上の partial onehot 破壊)

### 背景

LaRe-Path encoder の **距離関係因子** (`prog_goal`, `dist_goal_norm`, `at_goal` 等) が、エージェントの実際の移動を反映できていない。具体的には:

- agent がエッジ上を動いても、encoder からは **常にノード位置にいる**ように見える
- 結果として `prog_goal=0` が連発、`dist_goal_norm` がノード間距離の **2 値を往復**するだけ
- agent が動いた step でも因子が変化しない → デコーダの proxy 報酬がエッジ上での進捗を学習できない

### 原因

[src/main/drp_env/drp_env.py:961-963](../src/main/drp_env/drp_env.py#L961-L963) の `is_tasklist=True` ブロック末尾で `obs_onehot[i]` を **全クリア + current_start に full 1** に書き直しているため、action 処理 ([drp_env.py:781-784](../src/main/drp_env/drp_env.py#L781-L784)) で書き込んだ **partial onehot** (例: `current_start=0.6, action_i=0.4`) が消えてしまう:

```python
# 現状: drp_env.py:961-963
self.obs_onehot[i] = np.zeros((1, len(list(self.G.nodes()))*2))
self.obs_onehot[i][int(self.current_start[i])] = 1        # ← partial を破壊して full 1 にする
self.obs_onehot[i][int(self.goal_array[i])+len(list(self.G.nodes()))] = 1
```

これにより encoder の `_agent_curr_onehot` (= `obs_onehot[i]` 前半を読む) が常にノード位置の full onehot を返し、`estimate_partial_distance` の 2 要素分岐 ([encoder.py:96-109](../src/lare/path/encoder.py#L96-L109)) が動かない。

#### なぜこの潰しが存在するのか (= 位置潰しに設計意図は無い)

この3行の **本来の目的は「goal 半分の貼り直し」** であって、位置潰しは巻き添えの副作用。

- 直前のループ ([drp_env.py:936-958](../src/main/drp_env/drp_env.py#L936-L958)) で **ピック到達時に `goal_array[i]` がドロップへ書き換わる** ([:941-943](../src/main/drp_env/drp_env.py#L941-L943))。
- obs_onehot の goal 半分は action ループ ([:781/:791](../src/main/drp_env/drp_env.py#L781)) で **古い goal (=ピック) 基準**にセット済みなので、ここで貼り直さないと `calc_obs()` が agent に**間違ったゴール**を見せる。→ goal 半分の更新は **必須**。
- ところが実装が「ゼロ初期化 → ノードに 1 → goal に 1」という **ベクトル丸ごと再構築パターン** (reset やノード到達時と同じ書き方) のため、**ついでに位置半分も current_start ノードに潰している**。

| 部分 | 目的 | 評価 |
| --- | --- | --- |
| goal 半分の貼り直し | `goal_array` 変化への必須対応 | 正当・必要 |
| 位置半分を current_start に潰す | 丸ごと再構築の巻き添え | **設計意図なし (バグ的副作用)** |

- 位置潰しに設計上のメリットは無い。強いて言えば「観測位置がノード単位に離散化される」「コードが1パターンで済む」程度で、エッジ情報を捨てる価値はない。
- この3行は **upstream (LaRe 統合前) に既に存在**。導入コミットは `ea81480` (kaji, 2025-12-11) でメッセージが空、**設計理由は記録されていない**。
- MARL4DRP には**そもそもこの行が無い** (タスク機能が無く goal が切り替わらないので貼り直し自体が不要)。

→ 本来は「goal 半分だけ更新し位置は温存」が正しい実装。それでも[対策案](#対策案)で snapshot 方式 (env のバグは温存し LaRe だけ正しい位置を見る) を採るのは、env を直すと**タスクモードの MARL 観測が「ノード張り付き → エッジ連続」に変わり baseline 不変条件を破る**ため。意味論的正しさ (env 修正) と baseline 維持 (snapshot) のトレードオフ。

### 症状の再現

学習中ログから観測された例 (agent 0 を追跡):

```text
prog_goal: [0, 0, ...]                ← 動いてるのに変化なし
dist_goal_norm: [0.61, 0.52, ...]     ← 0.61 と 0.52 の 2 値を往復のみ
wait_norm: [0, 0, ...]                ← 動いている (= else 分岐に入っている) のは確実
```

`wait_norm=0` (= 動いた step) で `prog_goal=0` (= 距離変化なし) は矛盾。エッジ上の移動を encoder が拾えていない証拠。

### 対策案

⚠️ **env の潰し代入 (961-963) を直接書き換えてはいけない** (前半 position を温存する案は不採用)。理由:

- `obs_onehot` は **MARL / path planner の観測そのもの**。[onehot.py:14](../src/main/drp_env/state_repre/onehot.py#L14) が `return self.env.obs_onehot`、[fov_wrapper.py:19-26](../src/main/drp_env/state_repre/wrapper/fov_wrapper.py#L19-L26) も読む。しかも fov_wrapper は **1ノード (node) と 2ノード (edge) で分岐が異なる** (node 側は `*agent_num` するが edge 側はしない) ため、partial を残すとタスクモードのエッジ上 agent が別分岐に入り **MARL の観測が変わる**。
- この潰し代入は **upstream (LaRe 統合前) にも存在する** (`git show upstream/main:src/main/drp_env/drp_env.py` で確認)。書き換えると [CLAUDE.md](../CLAUDE.md) の不変条件「`use_lare_path=False` のとき LaRe 統合前と完全一致」を破る。

→ 方針: env の潰しは触らず、**潰される直前の加重 onehot を LaRe 専用に snapshot** し、LaRe の **位置読み取りだけ**それを使う。ゴールは潰し後の `obs_onehot` (= 最新ゴール) から読むので正しいまま。物理座標はタスクブロックで動かないため整合的。

修正は 4 箇所:

1. **[drp_env.py](../src/main/drp_env/drp_env.py) reset内** (638-641 の直後): snapshot を初期化

   ```python
   # LaRe-Path: エッジ上で fractional を保つ位置 onehot の snapshot. 初期=ノード onehot.
   self._lare_onehot_curr = copy.deepcopy(self.obs_onehot)
   ```

2. **[drp_env.py](../src/main/drp_env/drp_env.py) step内** (タスクブロック `if self.is_tasklist:` (864) の直前。衝突/非衝突で obs_onehot が確定した後): 潰される前に退避

   ```python
   if self.use_lare_path and self.lare_path_module is not None:
       self._lare_onehot_curr = copy.deepcopy(self.obs_onehot)
   ```

3. **[drp_env.py:565-572](../src/main/drp_env/drp_env.py#L565-L572) `_lare_capture_prev_onehot_pos`**: source を `self._lare_onehot_curr` (無ければ `obs_onehot` にフォールバック) に変更し、prev 側でもエッジ加重を保持。
4. **[encoder.py:246-258](../src/lare/path/encoder.py#L246-L258) `_agent_curr_onehot`**: 位置を `env._lare_onehot_curr` から読む (フォールバック付き)。`_agent_goal_onehot` は潰し後でも最新ゴールを持つ `env.obs_onehot` のまま据え置き。

これで `own_prev` / `own_curr` (自他両方) がエッジ上で 2 ノード加重になり、`estimate_partial_distance` の 2 要素分岐 ([encoder.py:96-109](../src/lare/path/encoder.py#L96-L109)) が機能する。

> 直接書き換え方式 (961-963 で後半=goal だけクリアし前半=位置を温存) は行数こそ少ないが、上記の通り MARL 観測と baseline を変える。env 側で持たせたいなら `use_lare_path` でガードして「LaRe ON のときだけ partial を残す」分岐にする必要があり、結局 snapshot と複雑さは変わらない。

### 期待される効果

- エッジ上の agent は encoder から「ノード A と ノード B の間 (例えば 0.6 : 0.4)」と認識される
- `estimate_partial_distance` の 2 要素分岐が動き、**重み付きで距離が計算される**
- `dist_goal` が連続的に変化 → `prog_goal` が動いた step ごとに非ゼロを返す
- デコーダの proxy 報酬がエッジ上の進捗を学習できる

### 検証

CLAUDE.md の検証スニペットで `time_limit` を伸ばしエッジ上ステップを作り、`dist_goal_norm` が 0/1 階段でなく中間値を取ることを確認する。

### 影響範囲

- `use_lare_path=False` の挙動は完全に不変 (snapshot 取得は LaRe ON 時のみ、env が返す観測は潰しのまま)
- LaRe-Path の `dist_goal_norm` / `prog_goal` / `at_goal` がエッジ上で正しく補間される (学習可能報酬の入力が改善)
- `prog_goal` / `dist_goal_norm` の値分布が変わるので **既存の学習済み LaRe-Path モデルは再学習推奨**
- LaRe-Task は経路の質に依存しないので影響なし
- 項目 2 (因子正規化) と同時に適用すると再学習が 1 回で済む

### 派生課題: タスク切替 step の prog_goal リセット

**背景**: `prog_goal = dist_goal_prev - dist_goal` は両項とも **現在のゴール基準** で計算される ([encoder.py:133-139](../src/lare/path/encoder.py#L133-L139))。ゴールが切り替わった step では `dist_goal_prev` が「**前 step の位置から“新しい”ゴールまでの距離**」になり、前 step に存在しなかった目標に対する差分 = 意味のないクロス目標値になる。

特に害が大きいのが **ピックアップ到達 step**。ドロップ D が来た道方向 (戻る側) にあると:

- 前 step: X→P へ前進 (当時のゴール P へは正しく進捗)
- この step: ゴールが D に切替、`dist_goal_prev = dist(X, D)` 小、`dist_goal = dist(P, D)` 大
- → `prog_goal = 小 − 大 = 大きな負`

ピックアップ成功という good event の瞬間に大きな負の進捗が出て、デコーダに誤信号を与える。新規割当 (idle→pickup) では agent が動いていないので `prog_goal ≈ 0` になりやすく、害は主に「移動しつつゴールが変わる」遷移 (= ピックアップ到達) で出る。

**対策案**: ゴールをまたいだ進捗は **定義不能** なので、その step は中立値に上書きする。

- **`prog_goal` だけ** を中立値にする。`dist_goal_norm` / `at_goal` は「現在状態の量」で新ゴール基準でも正しいのでそのまま残す。
- **中立値は項目 2 の正規化選択に合わせる**: raw / 案A (`prog/D`) / 案C (`clip(prog/D,0,1)`) → **0**。案B (`(prog/D+1)/2`) → **0.5**。
- **検出は env 側が確実**: タスクブロックで `goal_array[i]` が変わるのを env は知っている。`compute_factors` に `goal_changed` マスクを渡し、encoder 側で該当 agent の `prog_goal` を中立値に潰すのがクリーン。
- 実装は **項目 3 本体の snapshot 機構に相乗り**: `_lare_onehot_curr` を退避する箇所で、前 step のゴール (`goal_array` のスナップショット) も 1 本並行保持し、`prev_goal[i] != curr_goal[i]` を判定するだけ。

**影響範囲**:

- `_lare_capture_prev_onehot_pos` 相当にゴールスナップショットを追加 + `compute_factors`/`evaluation_func` に `goal_changed` マスク経路を 1 本追加
- `prog_goal` の分布が変わるため **再学習推奨** (項目 2・項目 3 と同時適用が望ましい)
- `use_lare_path=False` の挙動は不変

---

## 4. LaRe モデルの学習ステップ数と保存名の {X.X}M が一致しない

### 背景

LaRe-Path / LaRe-Task の autosave 保存名 `{...}_{X.X}M_{checkpoint|final}.pth` の `{X.X}M` トークンが、実際に学習したステップ数と一致しない。学習途中のチェックポイントを区別したいのに、名前から本当のステップ数が読み取れない / 別のチェックポイントが同名で上書きされる。

**観測例**: `Safe_QMIX_PATH_map_8x5_2agents_6.7M_checkpoint.pth` は **実際には 5M step (train) で学習したもの**。名前は 6.7M = 約 +1.7M (×1.34) の **系統的な過大カウント**。これは下記「根本原因」で説明済み: **test エピソードの step まで数えている** (6.7M ≒ train 5M + test 約 1.7M)。`.1f` 丸め (10万 step) では桁が合わない。

### 現状

トークンは `_lare_total_step_account` を 100 万で割って **小数第 1 位**で整形している:

- [drp_env.py:250-253](../src/main/drp_env/drp_env.py#L250-L253) `_lare_get_steps_str`:

  ```python
  steps_in_millions = self._lare_total_step_account / 1_000_000
  return f"{steps_in_millions:.1f}M"   # ← 0.1M = 10万 step 単位に丸め
  ```

`_lare_total_step_account` の入り方が経路で違う:

- **test.py 経路** (set_train_step が呼ばれない): [drp_env.py:713-716](../src/main/drp_env/drp_env.py#L713-L716) で env.step() ごとに +1 (LaRe 有効時のみ)。
- **epymarl 経路**: [episode_runner.py:71-72](../src/epymarl/src/runners/episode_runner.py#L71-L72) が `not test_mode` のとき `set_train_step(self.t_env + self.t + 1)` を呼び、以降 auto-increment は停止 ([drp_env.py:598-599](../src/main/drp_env/drp_env.py#L598-L599))。

### 根本原因 (特定済み): set_train_step が DrpEnv に届かず auto-increment が止まらない

`set_train_step` が **実体の DrpEnv まで届いておらず、auto-increment が止まらないため train も test も全 step を数えている**。

- epymarl の `EpisodeRunner.self.env` は **`_GymmaWrapper`** ([envs/\_\_init\_\_.py:78,204](../src/epymarl/src/envs/__init__.py#L78))。実体の DrpEnv は `original_env` (gym.make → TimeLimit の下) に埋まっている。
- `_GymmaWrapper` には **`set_train_step` も `__getattr__` も無い** (step / get_stats 等しか定義なし)。
- よって [episode_runner.py:71](../src/epymarl/src/runners/episode_runner.py#L71) の `hasattr(self.env, "set_train_step")` が **False → set_train_step は一度も呼ばれない**。
- 結果 `_lare_step_externally_set` は **False のまま** ([drp_env.py:598-599](../src/main/drp_env/drp_env.py#L598-L599) に到達しない) → [drp_env.py:713-716](../src/main/drp_env/drp_env.py#L713-L716) の auto-increment が **毎 step 発火**。env は test_mode を知らないので **test エピソードの step も加算**される。

→ 「test 除外」という設計意図 (`if not test_mode`) は、ラッパーが set_train_step を隠しているため **死にコード**になっていた。

**過大率の整合**: 過大率 = `1 + (test 総step / train 総step)`。観測 ×1.34 は、(1) test は 1 評価で `test_nepisode` 本まわす、(2) 学習が進むと訓練エピソードは早期ゴールで短くなり分母が縮む、(3) test はフル長に近い、で説明できる。`6.7M ≒ train 5M + test 約 1.7M`。

**副次的な別問題** (上記とは独立):

- **`.1f` 丸め (10万 step 粒度)**: 1,180,000 も 1,240,000 も `1.2M`。→ 桁ズレとは別に、近接チェックポイントが**同名衝突で上書き**される。
- **env ステップ数 ≠ デコーダ更新回数**: 名前は env ステップであって `update_freq` 間引き後のデコーダ更新回数ではない。「学習回数」を期待するとそもそも別物。

### 対策案

1. **(本命) `_GymmaWrapper` に転送を足す** → freeze が効き test 分が除外される:

   ```python
   def set_train_step(self, t_env):
       if hasattr(self.original_env, "set_train_step"):
           self.original_env.set_train_step(t_env)
   ```

2. 名前の精度を上げる: `f"{steps:.2f}M"` (1万 step 粒度) か生のステップ数 (`..._1234567steps_...`) で衝突回避。
3. 「学習回数」を出したいなら、デコーダの実 update 回数を別カウンタで持ち別トークンにする。

### 影響範囲

- 保存ファイル**名のみ**の問題で、学習・推論の挙動自体は不変。
- 対策1 を入れると、以後の保存名のステップ数が **train のみ** に変わる (これまでの train+test 値と非連続)。既存 `6.7M` 等のファイルとは番号体系が変わる点に注意。
- `.1f` 衝突や命名規則を変える場合は `models/` ロード側の後方互換 ([CLAUDE.md](../CLAUDE.md) の不変条件) に注意。

---

最終更新: 2026-06-09
