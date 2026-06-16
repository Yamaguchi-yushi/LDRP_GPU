# 将来実装メモ (TODO 集約)

軽量な「将来やりたい」「未適用の修正」を集約するファイル。重い独立設計書 (例: [ldrp_extensions.md](ldrp_extensions.md)) はここから参照のみ。

各項目は **背景 / 現状 / 対策案 / 影響範囲** の節構成で書く。実装に着手したら本ファイルから対応セクションを削除し、必要に応じて [../MANUAL.md](../MANUAL.md) の更新履歴に記録する。

---

## 目次

1. [GPU 環境への移行](#1-gpu-環境への移行)
2. [LaRe-Path 因子の正規化 (3 因子)](#2-lare-path-因子の正規化-3-因子)
3. [LaRe-Path 距離因子の残課題 (エッジ補間精度・タスク切替時の prog_goal)](#3-lare-path-距離因子の残課題-エッジ補間精度タスク切替時の-prog_goal)
4. [実験パラメータを専用ファイル (exp_config.yaml) に分離](#4-実験パラメータを専用ファイル-exp_configyaml-に分離)

### 重い設計書 (別ファイル)

- [ldrp_extensions.md](ldrp_extensions.md): LDRP 拡張 (高優先度: ピックアップ前タスク再配布 / 低優先度: 複数タスク保持 = VRP/TSP 系 tour 計画)
- [env_maturity.md](env_maturity.md): 環境ソフトウェアとしての成熟度ギャップ (CAMAR/RHCR/LoRR 比較。評価プロトコル・回帰テスト・throughput 指標・大規模輻輳耐性など)

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

## 4. 実験パラメータを専用ファイル (exp_config.yaml) に分離

### 背景

実験ごとに頻繁に変える値 (報酬モード・マップ・エージェント数・学習ステップ数など) が **本体コードに直書き**されているため、`git pull` のたびに衝突する。特に [drp_env.py](../src/main/drp_env/drp_env.py) の `__init__` signature の LaRe パラメータ群 (`use_lare_path` / `use_pretrained_lare_path` / `pretrained_lare_path_model_name` 等) を実験のたびに書き換えており、ここが衝突源。`gymma.yaml` の `t_max`、`train.py` の `env_args.key` も同様。

### 現状

- 実験値と本体ロジックが同じファイルに混在 (drp_env.py signature / gymma.yaml / train.py)
- [CLAUDE.md](../CLAUDE.md) は「実験値は signature を書き換えず env_args (train.py) / yaml (test.py) で上書きする」と定めているが、実運用では signature を直接編集してしまっている = 不変条件と乖離
- 観測された衝突例: drp_env.py の LaRe フラグ、gymma.yaml の `t_max` (50M vs 150M)、train.py の map/agent 数

### 対策案: exp_config.yaml (git 管理外) + 汎用 env_args パススルー

毎回いじる値を 1 ファイルに集約し、train.py が epymarl(sacred) の `with key=value` で上書きする。これにより **gymma.yaml / drp_env.py を一切編集しなくなり、衝突が構造的に消える**。`.claude/settings.local.json` と同じ「ローカル設定は git 外、テンプレはコミット」方式。

- **ファイル**:
  - `exp_config.yaml` (git 管理外, `.gitignore` に追加) ← 各自の実験値
  - `exp_config.example.yaml` (コミット) ← テンプレ
- **スキーマ案** (top-level = sacred 制御, `env_args:` = drp_env 引数を丸ごと上書き):

  ```yaml
  algo: qmix
  t_max: 50050000
  num_runs: 1
  max_processes: 1
  env_args:
    time_limit: 500
    key: "drp_env:drp_safe-5agent_map_8x5-v2"
    state_repre_flag: onehot_fov
    # 普段 drp_env.py で書き換えていた行をここに移すだけ
    use_lare_path: false
    use_lare_path_training: false
    use_pretrained_lare_path: false
    pretrained_lare_path_model_name: "Safe_QMIX_PATH_map_8x5_2agents_6.7M_checkpoint.pth"
    use_lare_task: false
  ```

- **train.py 側**: `env_args:` ブロックを丸ごと `env_args.<k>=<v>` に展開して `with` に渡す (個別キーを列挙しない = 将来の drp_env 引数追加に自動対応)。bool は `True`/`False`、str は要クォートのリテラル化が必要。

  ```python
  def lit(v):
      if isinstance(v, bool): return "True" if v else "False"
      if isinstance(v, str):  return f'"{v}"'
      return str(v)
  parts = [f't_max={cfg["t_max"]}']
  for k, v in cfg.get("env_args", {}).items():
      parts.append(f'env_args.{k}={lit(v)}')
  command = (f'python src/epymarl/src/main.py --config={cfg["algo"]} '
             f'--env-config=gymma -f with ' + ' '.join(parts))   # ← -f 必須 (下記検証結果)
  ```

- **書いたキーだけ上書き、未記載キーは drp_env.py の signature デフォルトが効く** ので、「普段いじる行」だけ移せばよい。

### sacred 動作検証結果 (確定・MAT 調査の副産物)

「`with env_args.use_lare_path=False` が効くか」を実検証した結果:

1. **転送経路 OK**: gymma wrapper は `gym.make(f"{key}", time_limit=time_limit, **kwargs)` ([envs/__init__.py:80](../src/epymarl/src/envs/__init__.py#L80)) で env_args 余剰キーをそのまま `DrpEnv.__init__` に渡す。→ drp_env.py を触らず上書き可能。
2. **未宣言キーはデフォルトで拒否される**: gymma.yaml の `env_args:` に存在しないキー (LaRe 系はすべて未宣言) を `with env_args.use_lare_path=False` で渡すと、sacred が `ConfigAddedError: Added new config entry that is not used anywhere` で停止する。**→ 現状の素の `with` だけでは LaRe パススルーは動かない。**
3. **`-f` / `--force` で解決**: sacred コマンドに `-f` を付けると未宣言キー追加が許可され、env_args 経由で値が `DrpEnv.__init__` に届く (検証で `use_lare_path=False` / `pretrained_lare_path_model_name=...` が反映)。
   - ⚠️ 副作用: `-f` は config-added チェック全般を緩めるため、**env_arg のキー名を typo してもエラーにならず黙って無視される**。対策として展開するキー名は exp_config 側の固定スキーマで管理する。
   - 代替案 (`-f` を使わない): gymma.yaml の `env_args:` に LaRe キーを事前宣言する。ただしデフォルトの第2ソースが増え、不変条件「単一の真実 = drp_env.py signature」と緊張する → **`-f` 方式を推奨**。
4. **`algo` は env_args に置けない**: `--config=<algo>` は sacred の別 named config (`algs/<algo>.yaml`) を指す CLI フラグであり env_arg ではない。さらに LaRe 命名 ([drp_env.py:241-255](../src/main/drp_env/drp_env.py#L241-L255)) は `sys.argv` の `--config=` を読むので**コマンドラインに出ている必要がある**。→ exp_config の top-level `algo` から `--config={algo}` に埋め込む (上記スキーマ通り)。

### 未決事項 (実装時に決める)

- env key を `env_args.key` で直書きするか、`map` / `agent_num` / `safe` から `drp_env:drp_safe-5agent_map_8x5-v2` を自動生成するか
- スコープ: まず train.py のみ。test.py / run.py への同様適用 (同じ exp_config を読ませる) は別途
- sacred 動作確認: `with t_max=...` と `env_args.use_lare_path=False` が実際に効くか、短い `t_max` で 1 回実起動して確認

### 影響範囲

- `train.py` の書き換え + `exp_config.example.yaml` / `.gitignore` の追加 (新規ファイル中心)
- `gymma.yaml` / `drp_env.py` は**編集しなくなる** (= 衝突解消)。デフォルト値の単一の真実は drp_env.py signature のまま維持され、CLAUDE.md の不変条件に整合
- 学習・推論の挙動自体は不変 (パラメータの渡し方が変わるだけ)

---

最終更新: 2026-06-11
