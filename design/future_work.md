# 将来実装メモ (TODO 集約)

軽量な「将来やりたい」「未適用の修正」を集約するファイル。重い独立設計書 (例: [ldrp_extensions.md](ldrp_extensions.md)) はここから参照のみ。

各項目は **背景 / 現状 / 対策案 / 影響範囲** の節構成で書く。実装に着手したら本ファイルから対応セクションを削除し、必要に応じて [../MANUAL.md](../MANUAL.md) の更新履歴に記録する。

---

## 目次

1. [GPU 環境への移行](#1-gpu-環境への移行)
2. [LaRe-Path 因子の正規化 (3 因子)](#2-lare-path-因子の正規化-3-因子)
3. [LaRe-Path 距離因子の残課題 (エッジ補間精度・タスク切替時の prog_goal)](#3-lare-path-距離因子の残課題-エッジ補間精度タスク切替時の-prog_goal)
4. [1エピソードあたりの安全制御発動回数を TensorBoard で可視化](#4-1エピソードあたりの安全制御発動回数を-tensorboard-で可視化)
5. [MAT-Dec 学習済みモデルの評価実行 (test.py) 対応](#5-mat-dec-学習済みモデルの評価実行-testpy-対応)

### 重い設計書 (別ファイル)

- [ldrp_extensions.md](ldrp_extensions.md): LDRP 拡張 (高優先度: ピックアップ前タスク再配布 / 低優先度: 複数タスク保持 = VRP/TSP 系 tour 計画)
- [env_maturity.md](env_maturity.md): 環境ソフトウェアとしての成熟度ギャップ (CAMAR/RHCR/LoRR 比較。評価プロトコル・回帰テスト・throughput 指標・大規模輻輳耐性など)
- [dynamic_agent_count.md](dynamic_agent_count.md): タスク割当エージェントによる動的エージェント数制御 (Phase A: 実行時のみ増減 → Phase B: 増減も学習)

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

## 3. LaRe-Path 距離因子の残課題 (エッジ補間精度・タスク切替時の prog_goal)

> **前提 (実装済み)**: エッジ上の位置を partial onehot で `obs_onehot` に温存する仕様を採用済み ([drp_env.py:999-1000](../src/main/drp_env/drp_env.py#L999-L1000))。これにより `estimate_partial_distance` の 2 要素分岐 ([encoder.py:96-113](../src/lare/path/encoder.py#L96-L113)) が機能し、エッジ上の移動が `prog_goal` / `dist_goal_norm` に反映されるようになった。以下はその上に残る精度・定義上の課題。

### 残課題 A: `estimate_partial_distance` がエッジ長 L を無視

2 要素分岐 ([encoder.py:100-109](../src/lare/path/encoder.py#L100-L109)) の距離は `(1-α)·Di + α·Dj` の **線形補間で、エッジ長 L を距離に加算していない**。エッジ中央で最大 `(L²−(Di−Dj)²)/(2L)` (= 等距離端点で 0.5L) 過小評価する。`|Di−Dj|≈L` (ゴール方向に直進) のときは誤差ゼロなので、**長い横向き (迂回) エッジが多いマップでのみ実害**が出る。優先度低 (現状の量子化ノイズに埋もれるレベル)。

**対策案**: 端点経由の min ルーティングに変更 (`graph` の隣接重み = エッジ長 L を利用):

```python
# 正規化重み wi_n, wj_n (= 各ノードへの近さ), L = edge_length(i, j)
#   点→端点の距離はエッジ長 L に比例: d_to_i = wj_n*L, d_to_j = wi_n*L
return min(wj_n * L + Di, wi_n * L + Dj)
```

L が取れない (隣接でない) 異常ケースは従来の線形補間にフォールバック。partial 仕様とは独立な精度改善。

### 残課題 B: タスク切替 step の prog_goal リセット

**背景**: `prog_goal = dist_goal_prev - dist_goal` は両項とも **現在のゴール基準** で計算される ([encoder.py:129-139](../src/lare/path/encoder.py#L129-L139))。ゴールが切り替わった step では `dist_goal_prev` が「**前 step の位置から“新しい”ゴールまでの距離**」になり、前 step に存在しなかった目標に対する差分 = 意味のないクロス目標値になる。

特に害が大きいのが **ピックアップ到達 step**。ドロップ D が来た道方向 (戻る側) にあると:

- 前 step: X→P へ前進 (当時のゴール P へは正しく進捗)
- この step: ゴールが D に切替、`dist_goal_prev = dist(X, D)` 小、`dist_goal = dist(P, D)` 大
- → `prog_goal = 小 − 大 = 大きな負`

ピックアップ成功という good event の瞬間に大きな負の進捗が出て、デコーダに誤信号を与える。新規割当 (idle→pickup) では agent が動いていないので `prog_goal ≈ 0` になりやすく、害は主に「移動しつつゴールが変わる」遷移 (= ピックアップ到達) で出る。

**対策案**: ゴールをまたいだ進捗は **定義不能** なので、その step は中立値に上書きする。

- **`prog_goal` だけ** を中立値にする。`dist_goal_norm` / `at_goal` は「現在状態の量」で新ゴール基準でも正しいのでそのまま残す。
- **中立値は項目 2 の正規化選択に合わせる**: raw / 案A (`prog/D`) / 案C (`clip(prog/D,0,1)`) → **0**。案B (`(prog/D+1)/2`) → **0.5**。
- **検出は env 側が確実**: タスクブロックで `goal_array[i]` が変わるのを env は知っている。`compute_factors` に `goal_changed` マスクを渡し、encoder 側で該当 agent の `prog_goal` を中立値に潰すのがクリーン。
- **実装**: prev onehot を退避する `_lare_capture_prev_onehot_pos` ([drp_env.py:603](../src/main/drp_env/drp_env.py#L603)) で前 step のゴール (`goal_array` のスナップショット) も 1 本並行保持し、`prev_goal[i] != curr_goal[i]` を判定するだけ。

**影響範囲**:

- `_lare_capture_prev_onehot_pos` にゴールスナップショットを追加 + `compute_factors`/`evaluation_func` に `goal_changed` マスク経路を 1 本追加
- `prog_goal` の分布が変わるため **再学習推奨** (項目 2 と同時適用が望ましい)
- `use_lare_path=False` の挙動は不変

---

## 4. 1エピソードあたりの安全制御発動回数を TensorBoard で可視化

### 背景

SafeEnv ([wrapper/safe_marl.py](../src/main/drp_env/wrapper/safe_marl.py)) は待機 agent の衝突を `step()` 内で事前回避しているが、「1エピソードで何回介入が起きたか」を観測する手段がない。安全制御の発動頻度は方策の質の代理指標 (発動が減る = 衝突しにくい方策を学習できている) になり、学習曲線と並べて TensorBoard で追えると有用。

### 現状

- SafeEnv.step() の 2 分岐が action を override する:
  - **act8** (同一目的地, [safe_marl.py:30-35](../src/main/drp_env/wrapper/safe_marl.py#L30-L35)): `joint_action[i]` を `current_start[i]` に戻す (= 待機)
  - **act9** (正面衝突 / head-on swap, [safe_marl.py:39-44](../src/main/drp_env/wrapper/safe_marl.py#L39-L44)): 同上
- どちらも発火回数を数えておらず、info にも出していない
- epymarl 側には **env info の数値キーを runner が自動集計して TensorBoard に出す機構が既にある** (parallel_runner / episode_runner の `cur_stats` 集計 → [run.py](../src/epymarl/src/run.py) の `logger.log_stat`、`use_tensorboard=True` 時)。→ env が info に数値を 1 個足すだけで TB に `<key>_mean` が出る

### 対策案

実装は SafeEnv だけで完結する:

1. **per-episode カウンタを持たせる**: `reset()` を override して `self.safety_intervention_count = 0` に初期化 (act8/act9 を分けるなら 2 つ)。reset 本体は DrpEnv 側なので `super().reset()` を呼ぶ薄いラッパで足りる。
2. **override 箇所でインクリメント** ([safe_marl.py:33](../src/main/drp_env/wrapper/safe_marl.py#L33), [:42](../src/main/drp_env/wrapper/safe_marl.py#L42))。
3. **info に注入**: `super().step()` の後で `info["safety_interventions"] = self.safety_intervention_count` (累積)。任意で `_act8` / `_act9` に分解。
4. これだけで runner が最終 step の info を拾って集計し、**`safety_interventions_mean` (= 1 エピソード平均介入回数) が TB に自動で出る**。追加のログコードは不要。

注意点:

- info の値は**数値**であること (runner の集計は `isinstance(v, (int, float))` フィルタ済み — MAT 移植時に追加)。
- parallel / episode 両 runner とも**最終 step の info だけ**を集計するので、カウンタは「エピソード累積値」を毎 step (少なくとも terminal step) info に入れる。
- `while do` ループ ([safe_marl.py:25](../src/main/drp_env/wrapper/safe_marl.py#L25)) は 1 step 内で複数回 override しうる。**「介入アクション総数」** で数えるか **「介入が起きた step 数」** で数えるか定義を決める (前者は override ごとに +1、後者は step 内で 1 度でも override したら +1)。

### 影響範囲

- [src/main/drp_env/wrapper/safe_marl.py](../src/main/drp_env/wrapper/safe_marl.py) のみ (reset override + step に数行 + info 1 キー)
- 既存挙動は不変 (info にキーが増えるだけ。override ロジックは変更しない)
- 非 Safe env では出ない。`drp_safe-*` マップ限定の指標

---

## 5. MAT-Dec 学習済みモデルの評価実行 (test.py) 対応

### 背景

MAT-Dec ([src/epymarl/src/config/algs/mat_dec.yaml](../src/epymarl/src/config/algs/mat_dec.yaml)) で学習した方策を `test.py` で評価したい。actor の実体 `Decoder` ([mlp_mat_agent.py:100-122](../src/epymarl/src/modules/agents/mlp_mat_agent.py#L100-L122)) は全エージェント重み共有の MLP で `n_agents` に依存する重みを持たないため、**学習時と異なるエージェント数への汎化評価**にも使える (この汎化検証が主目的)。

### 現状

- 推論パイプライン ([src/all_policy/](../src/all_policy/)) は RNNAgent 専用:
  - `PolicyRunner.__init__` ([policy_runner.py:23-26](../src/all_policy/policy_runner.py#L23-L26)) が state_dict に `"fc1.weight"` キーを要求し、MAT-Dec の checkpoint (`decoder.mlp.0.weight` 等) は即 ValueError で落ちる
  - アーキテクチャ自動検出 (`use_rnn` 判定・input_shape 読取) も RNNAgent のキー名前提
- 保存側は QMIX/IQL 系と同じ `basic_controller.save_models()` → `agent.th` (state_dict) なのでファイル形式自体は流用可能
- **`agent.th` には critic (Encoder) の重みも同居する**: `mat_learner.py` の `self.mac.agent.critic = self.critic` ([mat_learner.py:25](../src/epymarl/src/learners/mat_learner.py#L25)) で critic が agent の属性としてアタッチされ、`nn.Module` の属性代入で自動サブモジュール登録されるため。推論時は `decoder.*` だけ使い `critic.*` は無視してよい

### 対策案 (設計確定・実装は未着手)

1. **`src/all_policy/mat_policy_runner.py` を新設**。`Decoder` 相当のクラス (LayerNorm → Linear → GELU ×2 → Linear, `mlp_mat_agent.Decoder` と同一の層構成) をローカル再実装し、`agent.th` の `decoder.` プレフィックス付きキーだけを抜き出して `load_state_dict`。`critic.*` キーは無視。RNNAgent と違い hidden state 管理は不要 (`use_rnn=False` でステートレス)
2. **次元の自動検出**: `decoder.mlp.1.weight` (最初の Linear, shape=`(n_embd, obs_dim)`) と `decoder.mlp.7.weight` (最後の Linear, shape=`(n_actions, n_embd)`) の shape から `obs_dim`/`n_embd`/`n_actions` を復元し、env 側の `input_shape`/`n_actions` と食い違えば ValueError (既存 `PolicyRunner` の自動検出と同じ思想)
3. **クラスの出し分けは `path_planner` の明示分岐**: state_dict のキー自動判別ではなく、`MARLPolicy.policy()` で `self.path_planner in {"mat_dec"}` のときだけ `MatPolicyRunner` を使う (それ以外は既存 `PolicyRunner`)。出力の意味 (Q値 vs 方策logits) がアルゴリズムごとに違うため、キー名だけで自動判別するより明示的な方が安全
4. **行動選択は決定的 argmax**: 出力 (方策logits) を `avail_actions` (既存の index リスト形式のまま、`policy.py` 側の呼び出しインタフェースは変更不要) でマスクし `-1e10` → argmax。既存 `PolicyRunner` の masked-Q argmax と全く同じ書き方に揃える (`logit[a] if a in avail_actions else -1e10` の形)。確率的 sample にはしない (QMIX系評価と挙動を揃え、エピソード間の再現性を優先)
5. **`MARLPolicy.get_model_path`** ([policy.py:42-48](../src/all_policy/policy.py#L42-L48)) の命名規則 `{map}_{N}_{algo}.th` はそのまま `mat_dec` を algo 名として使える (変更不要)
6. **自己回帰 decode は評価時不要**: `discrete_autoregreesive_act` ([mlp_mat_agent.py:61-84](../src/epymarl/src/modules/agents/mlp_mat_agent.py#L61-L84)) は critic (`v_loc`) と絡むが、行動選択自体は per-agent の `decoder(obs)` → argmax で完結する。critic はロード不要

参考: epymarl 自身の eval (train.py 実行中の sacred テストエピソード) は `SoftPoliciesSelector` ([action_selectors.py:67-75](../src/epymarl/src/components/action_selectors.py#L67-L75)) が `test_mode` を見ずに常時サンプリングするため、上記の決定的 argmax とは一致しない。test.py 側の評価はあくまで独自の決定的ポリシーとして扱う。

### 汎化評価時の注意

- `n_embd` 等のハイパーパラメータは checkpoint の shape から復元可能だが、`state_repre_flag='onehot_fov'` の obs 次元はマップサイズ・FOV に依存するため、**汎化はエージェント数方向のみ** (マップをまたぐ汎化は obs 次元が変わり不可)
- `obs_agent_id=True` で学習した checkpoint は入力にエージェント数分の onehot が付くため汎化不可。mat_dec.yaml のデフォルトは `obs_agent_id: False`

### 影響範囲

- `src/all_policy/` に新規ファイル追加 + `policy.py` or `policy_manager.py` に分岐数行
- 既存の RNN 系評価パスは不変

---

最終更新: 2026-07-14
