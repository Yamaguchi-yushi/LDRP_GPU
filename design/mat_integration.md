# MAT 統合設計 (Multi-Agent Transformer による台数ゼロショット汎化)

LDRP に Multi-Agent Transformer (MAT) を導入し、**同一マップ上で「少ない台数で学習 →
異なる台数で実行 (ゼロショット汎化)」** を実現するための設計書。

- スコープ: **同一マップ・台数のみ変更**。マップ横断 (n_nodes が変わる) 汎化は対象外。
- 進め方: ステップA (ベースライン汎化評価) → ステップB (MAT 本体)。各ステップ「設計提案 →
  ユーザー確認 → 1 ファイルずつ最小差分で実装」。既存 QMIX 実験は壊さない。

---

## 0. 確定した前提 (調査済み・再調査不要)

| # | 前提 | 根拠 |
|---|---|---|
| 1 | 観測は既に N 非依存。各エージェント = 自位置 one-hot + ゴール one-hot で長さ `n_nodes*2`。他エージェントは連結ではなく同一ベクトル内に -1 マスクで圧縮。`current_tasklist` は policy 観測に含まれない | [drp_env.py:676](../src/main/drp_env/drp_env.py#L676), [onehot_fov.py:12](../src/main/drp_env/state_repre/onehot_fov.py#L12), [fov_wrapper.py:44-48](../src/main/drp_env/state_repre/wrapper/fov_wrapper.py#L44-L48) |
| 2 | 台数汎化を阻む唯一の構造要因は `obs_agent_id=True` の長さ N の one-hot。`th.eye(n_agents)` 連結で入力次元が学習時 N に固定。対象 maa2c/mappo/ippo/ia2c/coma/maddpg (**MAPPO 含む**)。qmix/iql/vdn/qplex は False で無関係 | [basic_controller.py:74-75,84-85](../src/epymarl/src/controllers/basic_controller.py#L74-L85) |
| 3 | Safe Control は agent_id / 観測に非依存。`current_goal`/`current_start`/`joint_action`/ループ変数 `i` のみ参照。優先順位の実体は `range(self.agent_num)` の index 昇順 (環境内部の走査順)。長さ N の one-hot を外しても挙動不変 | [safe_marl.py:27-44](../src/main/drp_env/wrapper/safe_marl.py#L27-L44) |
| 4 | マップ名と agent_num は独立。1〜29 台 × 全マップの直積が登録済み。同一マップ・台数のみ変更の評価は現構成で可能 | [__init__.py:52](../src/main/drp_env/__init__.py#L52), [test.py:32](../test.py#L32), [drp_env.py:652-657](../src/main/drp_env/drp_env.py#L652-L657) |

### エージェント識別の方針 (MAT)

- **長さ N の agent_id one-hot は使わない** (台数汎化を自分で潰すため)。`obs_agent_id=True` は
  epymarl の config 既定値であって MAT の要件ではない。
- 代わりに **N 非依存の位置エンコーディング** (正弦波、または系列内の順序インデックスから生成する
  固定次元の埋め込み) でエージェントを識別。入力次元が N に依存せず、順序で区別できる。
- MAT デコーダのエージェント順序は **`range(N)` の index 昇順に揃える**。これにより
  「系列内の位置 = Safe Control の優先順位」が一致し、位置エンコーディングが優先度情報も兼ねる。
- 均質エージェントなので識別を完全に切っても原理上は動くが、対称性による学習不安定を避けるため
  位置エンコーディングを既定とする。**完全に切る案は比較対象として残す**。

---

## ステップA: MAT 導入前のベースライン汎化評価 (QMIX)

**目的**: 現行 QMIX (`obs_agent_id=False`) で「学習 N → 別 N でゼロショット評価」のベースライン
を取得する。MAT の改善幅を測る対照群。

### A-1. 結論: コード変更ゼロで即実行可能

評価経路は **既に mixer を使わず、per-agent greedy、N 非依存の共有 agent network で動いている**。

- **agent network が N 非依存**: [rnn_agent.py](../src/epymarl/src/modules/agents/rnn_agent.py) の
  `RNNAgent` は `fc1(input_shape→hidden) → rnn → fc2(hidden→n_actions)`。
  qmix.yaml が `obs_agent_id=False`/`obs_last_action=False` なので `input_shape = n_nodes*2`、
  `n_actions = n_nodes` ── どちらもマップ依存で N 非依存。
- **mixer は評価で未使用**: epymarl は agent と mixer を別ファイルに保存
  ([basic_controller.py:55](../src/epymarl/src/controllers/basic_controller.py#L55) `agent.th` /
  [q_learner.py:161](../src/epymarl/src/learners/q_learner.py#L161) `mixer.th`)。
  評価は agent のみロード ([policy_runner.py:21](../src/all_policy/policy_runner.py#L21))。
  `get_action` ([policy_runner.py:25-39](../src/all_policy/policy_runner.py#L25-L39)) は
  エージェント単体 forward → avail マスク → `argmax`。= 「mixing なし per-agent greedy」そのもの。
- **N はループ回数だけに効く**: PolicyRunner は `range(agent_num)` 個の hidden state を生成
  ([policy_runner.py:23](../src/all_policy/policy_runner.py#L23))、MARLPolicy は
  `for agi in range(env.agent_num)` ([policy.py](../src/all_policy/policy.py))。
  network は共有なので N を変えても次元は不変。

### A-2. 唯一の N 結合 = モデルのファイル名

[policy.py](../src/all_policy/policy.py) `MARLPolicy.get_model_path`:
```python
filename = f"{env.map_name}_{env.agent_num}_{self.path_planner}.th"   # 評価時 N を使用
```
N=2 学習モデルは `map_5x4_2_qmix.th`、N=8 評価は `map_5x4_8_qmix.th` を探す。

**実現方法 (2 案)**:
- **(推奨・コード変更ゼロ)** 学習済みファイルを評価 N のファイル名にコピー:
  `cp map_5x4_2_qmix.th map_5x4_8_qmix.th`。agent params は N 非依存なので動く。
- **(クリーン・小差分)** config に「学習時 N (ファイル名専用)」フィールドを追加し
  `get_model_path` でそれを使う。default = agent_num で後方互換。差分は policy.py 1 行 +
  test.py の引数処理のみ。← **ステップA 実装時にユーザーと選択**。

### A-3. 整合性チェック

- 学習 ([train.py:23](../train.py#L23)) も評価 ([test.py:88](../test.py#L88)) も
  `state_repre_flag="onehot_fov"` → 両方 `n_nodes*2`。PolicyRunner は `input_shape=len(obs[0])`
  を実行時取得 ([policy.py](../src/all_policy/policy.py)) なので同一マップで自動一致。
- DummyArgs `use_rnn=False`/hidden_dim=64 は qmix.yaml と一致。
  ⚠️ qmix を `use_rnn=True` で学習すると PolicyRunner 側 (False 固定) と state_dict 不一致。
  現状は両方 False で問題なし。

### A-4. ゼロショット評価手順 (map_5x4 で 2/4/8 台)

1. **学習** (N=2): [train.py:22](../train.py#L22) のキーを `drp_safe-2agent_map_5x4-v2` に書き換え → train.py。
2. **配置**: 保存された `agent.th` を `src/all_policy/models/safe/map_5x4_2_qmix.th` にコピー (CLAUDE.md Step 2)。
3. **別 N で評価**:
   - `cp map_5x4_2_qmix.th map_5x4_4_qmix.th` / `... map_5x4_8_qmix.th`
   - `python test.py map_5x4 4 qmix tp` / `python test.py map_5x4 8 qmix tp`
   - path_planner は MARL 系 (`qmix`)。`pbs` を避け `pbs_mode=False` を保ち SafeEnv の待機保護を効かせる ([test.py:84](../test.py#L84))。
4. **制約・注意**:
   - N ≤ マップのノード数 (reset の `random_start/random_goal`, [drp_env.py:652-657](../src/main/drp_env/drp_env.py#L652-L657))。評価可能台数の上限。
   - SafeEnv は N 非依存に衝突回避を保証 → クラッシュしない。N=2 で学習した協調は N 増で劣化しうる = 測りたい汎化ギャップ = 正当なベースライン。

### A-5. ステップA TODO (実装は確認後)

- [ ] 学習時 N とファイル名を分離する方式の選択 (コピー運用 or get_model_path 小改修)
- [ ] 複数 N をまとめて回す評価スクリプト (run.py 拡張 or 薄いシェル) の要否確認
- [ ] ベースライン結果テーブルの指標確定 (task_completion, steps, 衝突率, ロック率)

---

> **実験パラメータの集約 (LaRe on/off・モデル名等を drp_env.py から外す件) は MAT とは別問題**
> なので [future_work.md §4](future_work.md#4-実験パラメータを専用ファイル-exp_configyaml-に分離) に集約。
> 今回の sacred 検証結果 (未宣言 env_args キーは `ConfigAddedError`、`-f` で抑制可) も同所に記載。

---

## ステップB: MAT 用の入力ビルダと識別方式 (ドラフト・未確定)

> ステップA が固まってから詳細化する。現時点は方向性のメモ。

### B-1. 入力ビルダ (basic_controller の one-hot 連結を迂回)

- 新規 `mat_mac` (MAControllers) を追加し、`_build_inputs` で **`th.eye(n_agents)` 経路を絶対に踏まない**。
- 観測 `(bs, N, n_nodes*2)` をトークン列としてそのままエンコーダへ渡す。`obs_last_action` も既定 off。
- observation_space は変更しない (前提 1)。MAT 側で `(N, n_nodes*2)` のトークン列として扱う。

### B-2. エージェント識別 = N 非依存の位置エンコーディング

- 候補: (a) 正弦波 PE (Transformer 標準), (b) 学習可能だが固定最大長から index で引く埋め込み。
- デコード順を `range(N)` index 昇順に固定 → 系列内位置 = Safe Control 優先順位を一致させる。
- 比較対象として「識別なし (純粋な置換不変)」も残す。

### B-3. mat_mac の構成と既存資産の再利用

- MAPPO の runner / buffer / GAE をどこまで流用できるか調査 (`extra_in_buffer` の log_probs/values 含む)。
- エンコーダ (集中表現) + デコーダ (自己回帰 or 並列) の選択。
- 学習は CTDE、実行は per-agent。実行時は前提 3 より Safe Control が N 非依存に働く。

### B-4. ステップB TODO (ステップA 確定後に詳細化)

- [ ] mat_mac の `_build_inputs` 仕様 (one-hot 迂回の確証)
- [ ] 位置エンコーディング方式の決定 + デコード順との対応
- [ ] MAPPO runner/buffer/GAE の再利用範囲
- [ ] 学習 N → 別 N 実行時のモデルロード経路 (ステップA の命名規約と統一)

---

## 関連設計

- 環境成熟度・汎化階層の位置づけ: [env_maturity.md](env_maturity.md) §2, §5
- 未適用 TODO 集約: [future_work.md](future_work.md)
