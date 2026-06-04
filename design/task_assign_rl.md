# 設計・実装ガイド: タスク割当の強化学習 (Task-Assignment RL)

**作成日:** 2026-05-19 (PPO save/load 設計として)
**拡張日:** 2026-06-02 (タスク割当 RL 全体の設計・実装ガイドに拡張)
**ステータス:** PPO=**実装途中 / 未検証** / DQN=将来案
**関連:** [lare_integration.md](lare_integration.md) (永続化の参照パターン)
**旧名:** 元は「PPO save/load 設計書」(`ppo_persistence.md`)。範囲拡張に伴い `task_assign_rl.md` に改名。

---

## 0. このドキュメントの位置づけ

LDRP の **タスク割当 (task assignment)** を強化学習で行うコンポーネントの設計・実装ガイド。

- **PPO はまだ未完成・未検証**。本ガイドは「現状の ppo.py の到達点」と「完成までに必要な作業」を整理する。
- DQN は **将来追加** する value-based 案。着手時の見取り図として併記する。
- 抽象基底 (`BaseTaskPolicy`) への切り出し等の細かい設計は **後日**。

扱う範囲:

1. 全体アーキテクチャ (TaskManager プラグイン構造)
2. 新アルゴが満たすべき **共通インターフェース契約**
3. **PPO** の現状と残作業 (実装途中)
4. **永続化 (save/load)** 設計
5. **将来実装: DQN** のスケッチ

---

## 1. 全体アーキテクチャ

タスク割当は **プラグイン差し替え** 構造。
[src/task_assign/task_manager.py](../src/task_assign/task_manager.py) が名前で実装をディスパッチ:

| `task_assigner` | 実体 | 学習 | 状態 |
|---|---|---|---|
| `fifo` | `Random` | ✗ | 動作 |
| `tp` | `TP` | ✗ | 動作 |
| `ppo` | `PPOAgent` | ✓ | **実装途中・未検証** (旧版を参考に実装) |

**新 RL アルゴ (DQN 等) を足す = `task_policy/` にクラスを 1 個追加 + task_manager に分岐 1 行**。
runner ループ側は共通インターフェース呼び出しなので、契約を満たせば原則無改修。

### runner ループとの結合点

[runner.py](../runner.py) (抜粋):

```python
task_assign = self.task_manager.assign_task(self.env)          # 割当を取得
joint_action = {"pass": agents_action, "task": task_assign}     # env へ
...
# 報酬投入 (LaRe-Task が trained なら proxy 報酬, でなければ env 報酬の和)
task_reward = float(info["lare_task_proxy_reward"]) if <trained> else float(sum(rew_n))
self.task_manager.task_assigner.buffer_add_rewards(task_reward, done)
...
# エピソード終了 / 更新
self.task_Agent.task_assigner.process_end_episode()
if self.task_Agent.task_assigner.update_ready():
    a_loss, c_loss, e_loss = self.task_Agent.task_assigner.update()
```

> 注: runner は `task_manager` と `task_Agent` を併用している箇所がある。配線の整理も PPO 完成作業の一部。

---

## 2. 共通インターフェース契約

新アルゴ (および完成版 PPO) が満たすべき口:

| メソッド | 役割 | on/off-policy |
|---|---|---|
| `assign_task(env, current_tasklist, assigned_tasklist)` → `list[int]` | 割当ベクトル(長さ=agent数, 値=task idx or -1)を返す。学習時は内部でバッファに行動記録 | 共通 |
| `update()` → `(a_loss, c_loss, e_loss)` | 方策更新。ロスは 3 つ返す慣習 (DQN は a/c の片方を 0 等で埋める) | 共通 |
| `update_ready()` → `bool` | 更新発火判定 (PPO=バッファ満杯, DQN=replay が min_buffer 超え) | 共通 |
| `buffer_add_rewards(reward, done)` | 報酬投入。LaRe-Task proxy もここ経由 | 共通 |
| `buffer_reset()` | バッファ初期化 | 共通 |
| `set_test_mode(bool)` | 推論/学習切替 (PPO=argmax/sample, DQN=ε=0/ε>0) | 共通 |
| `process_end_episode()` | エピソード境界処理 (PPO=return 計算) | **on-policy 固有** |
| `save_model(path)` / `load_model(path)` | 永続化 (§4) | 共通 |

### 行動空間 (全アルゴ共通の前提)

- **フラット離散**: 出力次元 = `task_num × agent_num`。1 つ選ぶと `(agent, task) = divmod(action, task_num)` に分解。
- **3 種のマスク** (`assign_task` 内):
  1. 既にタスクを持つ agent の枠
  2. `current_tasklist` の実数を超えた task 枠
  3. 既に割当済み (`task[0] == -1`) の task 枠
- 1 ステップで複数 agent に割り当てる場合は **agent ループで逐次選択** し、選ぶたびにマスクを更新。

### 状態表現 (`create_state`)

`input_dim = agent_num*node_num*2 + task_num*node_num + agent_num` の 1 次元ベクトル:

| 区間 | 内容 |
|---|---|
| `agent_num*node_num*2` | 各 agent の onehot 観測 (`env.obs_onehot` 連結, start/goal で *2) |
| `task_num*node_num` | 各 task の onehot (start=+1, goal=-1)。空き枠はゼロ |
| `agent_num` | 各 agent が割当済みか (0/1) |

状態表現は **アルゴ非依存** なので、完成後は DQN でもそのまま流用可能。

### 割当・更新のタイミング (想定設計)

**「割当をいつ変えるか」と「いつ学習するか」は別の軸**として分けて設計する。

| 軸 | 想定 | トリガ |
| --- | --- | --- |
| **① 割当の更新** (どの agent にどのタスク) | **イベント駆動** | 新タスク発生 / タスク完了 / エージェントがアイドル になった時のみ |
| **② 強化学習** (policy の gradient 更新) | **複数ステップ後にまとめて学習** | イベントごとに記録した transition が一定量たまってから `update()` |

#### ① 割当の更新 = イベント駆動

- 毎ステップ割当を決める必要はない。**割当に意味がある瞬間 (新タスク到着・完了・アイドル) だけ** `assign_task` が実際に決定する。
- 現 PPO は runner から毎 step 呼ばれるが、内部で **「空き agent かつ 空き task があるときだけ」決定して `buffer.add_actions`** する ([ppo.py](../src/task_assign/task_policy/ppo.py) の `if any(len(task)==0 ...) and len_current_task > 0`)。
  → **実質すでにイベント駆動寄り**。これを明示的なイベント駆動として整理する。
- [multi_task_agents.md](multi_task_agents.md) §3 のイベント駆動方針と一致する。
  ([alma_integration.md](alma_integration.md) の `N_t` 固定周期とは異なり、**周期ではなくイベント**で駆動する。)

#### ② 強化学習 = 複数ステップ後にまとめて学習

- 各イベントで記録した transition (state / action / reward) を **バッファに貯め、複数ステップ分そろってから 1 回 `update()`** する。
- 現 PPO は on-policy バッファが満杯になった時 (`update_ready()`) に更新する形 = **「複数ステップ後にまとめて学習」と整合**している。
- 割当 (①) がイベント駆動で疎になるぶん、1 エピソードあたりの transition 数は減るが、各サンプルは意味のある決定点なので学習効率は良い (multi_task §3.4 と同じ論拠)。

> 当面の PPO 完成 (§3) はこの 2 軸の想定に沿って配線する。N_t 周期版 (ALMA) との統一が必要になったら、その時点で `BaseTaskPolicy` 抽象化と一緒に検討する。

---

## 3. PPO の現状と残作業 (実装途中)

[src/task_assign/task_policy/ppo.py](../src/task_assign/task_policy/ppo.py) は **書きかけ・未検証**。
現状の到達点と、完成までに詰めるべき点を分けて記録する。

### 3.1 現状コードの到達点 (ドラフト)

| 要素 | 現状 |
|---|---|
| ネットワーク | `policy_layer` / `value_layer` を **別々の MLP** (512→256→128) で持つ形まで記述 |
| バッファ | on-policy のスケルトン (`steps/states/actions/log_probs/values` + `rewards/dones`) |
| return | `compute_returns(gamma)` あり。GAE 版 `compute_returns_and_advantages` は **未実装スタブ** |
| 更新 | PPO clip + actor/critic optimizer 分離の骨子あり |
| 行動選択 | `test_mode` で argmax / サンプリング切替まで記述 |
| save/load | **stub (空実装)** |

### 3.1.1 報酬: 元々はグローバル報酬

元々の PPO は **グローバル報酬 (チーム共有報酬)** で学習する設計。
[runner.py:107-117](../runner.py#L107-L117) で 1 step ごとに次のスカラを `buffer_add_rewards` に投入:

```python
if info.get("lare_task_is_trained", False):
    task_reward = float(info.get("lare_task_proxy_reward", 0.0))  # LaRe-Task 有効時のみ差し替え
else:
    task_reward = float(sum(rew_n))                               # ← 元々の報酬 = 全agentの和
```

| 条件 | PPO に渡る報酬 |
|---|---|
| **LaRe-Task OFF (元々)** | `sum(rew_n)` = **全エージェントの env 報酬 (goal/collision/wait/move) の総和**。単一スカラ |
| LaRe-Task が trained | `lare_task_proxy_reward` (分解された proxy 報酬) に差し替え |

含意:

- `rew_n` は agent ごとの報酬リストだが、**総和を取って 1 スカラ**にしている = チーム全体の報酬
- これを `compute_returns` で割引和にし、各 (agent, task) 割当行動に **同じグローバルリターンが帰属**
- → **個々の割当がチーム成果にどれだけ寄与したかの credit assignment は無い** (全行動が同じ報酬を共有)
- この「グローバル報酬だと個々の割当の貢献が分からない」点が **LaRe-Task (報酬分解) 導入の動機**。
  trained になると proxy 報酬に差し替わる

### 3.2 完成までの残作業 (要対応)

- [ ] **学習が回ることの検証** (1 エピソードで loss が出る最小確認 → 学習で性能が上がるか)
- [ ] **`update()` のインデント確認**: `dist = Categorical(...)` 以降が内側 `for batch` の外にあり、
      **最後の minibatch の logits だけで更新**している疑い。要修正の可能性大。
- [ ] **報酬→return の整合**: `compute_returns` の `agent_num` 分のステップ対応ロジックが正しいか検証
- [ ] **advantage**: 現状 `returns - values` の単純版。GAE 採用するか決める
- [ ] **runner 配線整理**: `task_manager` と `task_Agent` の二系統を一本化
- [ ] **save/load 実装** (§4)
- [ ] **マスクの境界条件** (task 0 件 / agent 全員 busy 等) の確認

> まずは「学習が正しく回る」ことを確定させてから save/load (§4) に進むのが順序。

---

## 4. 永続化 (save / load)

> 本ファイルの元々の主題。PPO に限らず **学習する全タスク割当アルゴ共通**の設計とする。
> (PPO 本体の学習が回るようになってから着手する。)

### 4.1 現状の問題

学習結果をテストに引き継げない (`save_model`/`load_model` が stub):

| コンポーネント | save | load | テスト時利用 |
|---|---|---|---|
| Path planner (QMIX/IQL/...) | ✓ (epymarl) | ✓ | 可能 |
| LaRe-Path / LaRe-Task decoder | ✓ | ✓ | 可能 |
| **PPO task assigner** | **✗ stub** | **✗ stub** | 不可能 (そもそも学習も未検証) |

test.py は `training=False` 起動 → PPO は毎回ランダム初期化で推論 → 「訓練→テスト」が繋がらない。

### 4.2 設計方針

LaRe-Path / LaRe-Task の整備パターンをそのまま適用 (対称性維持):

| 要素 | 内容 |
|---|---|
| モード | 4 モード対称 (off / scratch / pretrained / finetuning) |
| ディレクトリ分離 | `checkpoints/` (autosave 出力) と `models/` (整理済み load 元) |
| autosave throttle | 累積環境ステップ単位 (0.5M ごと等), 学習頻度とは独立 |
| ファイル命名 | Safe-TSL-DBCT 流儀 (`{Safe_}<TOKEN>_<...>_X.XM_checkpoint.pth`) |
| 後方互換 | 旧パスがあれば load 解決パスに含める |

### 4.3 ディレクトリ構成 (提案)

```
src/task_assign/
├── checkpoints/      ← autosave 出力 (大量蓄積, .gitignore)
├── models/           ← pretrained / finetuning ロード元 (整理済み, git 公開)
└── task_policy/      ← 既存 (ppo.py 等)
```

`models/` に `.gitkeep` を置いて空 dir を git に乗せる。

### 4.4 ファイル命名

LaRe と衝突しない `PPO` トークン (DQN 追加時は `DQN` 等):

- Scratch: `{Safe_}{ALGO}_PPO_{map}_{N}agents_{X.X}M_checkpoint.pth`
- Finetuning: `FT_{Safe_}{source_base}_{map}_{N}agents_{X.X}M_checkpoint.pth`

`{ALGO}` は **path planner 名** (qmix/iql 等)。例:
`Safe_QMIX_PPO_map_8x5_4agents_2.0M_checkpoint.pth` = SafeEnv + QMIX path planner + PPO task assigner で 2.0M ステップ学習。

### 4.5 設定キー (default.yaml 追加案)

```yaml
# --- task assigner の永続化 (Path planner / LaRe と独立) ---
use_pretrained_ppo: false
pretrained_ppo_model_path: null      # ファイル名だけ指定で src/task_assign/models/ から自動解決
use_finetuning_ppo: false
finetuning_ppo_model_path: null

# 自動保存
ppo_autosave: false
ppo_autosave_path: null              # null なら自動命名 (上記の命名規則)
ppo_save_freq_steps: 500000          # 0.5M step ごとに保存 (LaRe と統一)
ppo_save_dir: null                   # deprecated 用予約. 通常は null
```

> DQN を足すときは `_ppo` を `_dqn` にした並行キー群を追加 (= アルゴごとに独立)。

### 4.6 実装フェーズ

| Phase | 内容 | コスト目安 |
|---|---|---|
| 1 | `save_model`/`load_model` を実装 (`torch.save`/`torch.load`) | 15 行 |
| 2 | runner.py で起動時に pretrained / finetuning に応じて load | 20 行 |
| 3 | runner.py で学習中に throttle 付き autosave | 20 行 |
| 4 | ディレクトリ + `.gitignore` 整備 | 5 行 |
| 5 | default.yaml + test.py に新キー転送 | 10 行 |
| 6 | CLAUDE.md / MANUAL.md 記載 | 適宜 |

合計 ~80 行。

### 4.7 推奨ワークフロー (実装後)

```bash
# 1. 学習 (training=True + ppo_autosave: true)
#    → src/task_assign/checkpoints/Safe_QMIX_PPO_..._X.XM_checkpoint.pth

# 2. 良いモデルを選別
cp src/task_assign/checkpoints/Safe_QMIX_PPO_..._2.0M_checkpoint.pth \
   src/task_assign/models/ppo_best.pth

# 3. テスト (default.yaml: task_assigner: ppo / use_pretrained_ppo: true /
#            pretrained_ppo_model_path: ppo_best)
python test.py
```

### 4.8 test.py での学習済み方策の実行 (経路探索と同形式)

**目的**: 学習済みのタスク割当方策を test.py で実行できるようにする。
**方針**: ロード機構は **経路探索 (MARLPolicy) と完全に同じ形式** にする。
方策を選択するとき「どう学習されたか」は無関係 — `task_assigner` 名と env から **規約でパスを決めてロードするだけ**。

参照する既存実装 ([src/all_policy/policy.py:41-44](../src/all_policy/policy.py#L41-L44)):

```python
def get_model_path(self, env):
    filename = f"{env.map_name}_{env.agent_num}_{self.path_planner}.th"
    return os.path.join(base_dir, "models", "safe", filename)   # 例: map_8x5_4_qmix.th
```

タスク割当も **同じ規約** で揃える:

```python
def get_model_path(self, env):
    filename = f"{env.map_name}_{env.agent_num}_{self.task_assigner}.th"
    return os.path.join(base_dir, "models", "safe", filename)   # 例: map_8x5_4_ppo.th
```

| 段階 | 経路探索 (既存) | タスク割当 (これに揃える) |
|---|---|---|
| 保存先 | `src/all_policy/models/safe/{map}_{N}_{algo}.th` | `src/task_assign/models/safe/{map}_{N}_{task_assigner}.th` |
| 選択方法 | `path_planner` 名 + env から規約解決 | `task_assigner` 名 + env から規約解決 |
| ロード判定 | ファイルが規約位置にあればロード (フラグ不要) | 同左 |
| 推論モード | test 時は学習せずロードのみ | `training=False` / `test_mode=True` で greedy (既に成立) |

**実行経路 (実装後)**:

1. test.py は `task_assigner` 名を受け取り済み (`config.task_assigner` / `sys.argv[4]`)
2. assigner 初期化時に上記 `get_model_path(env)` で規約解決 → ファイルがあれば `load_model`
3. `Runner(..., training=False)` 起動 → 学習スキップ・`test_mode=True` の greedy 推論
4. → **`use_pretrained_*` / `finetuning_*` 等のフラグは不要**。経路探索と同じく「規約位置にファイルがあれば使う」だけ

```bash
# 実行例 (経路探索と同じ呼び出し感覚)
python test.py map_8x5 4 qmix ppo
#   path_planner=qmix → all_policy/models/safe/map_8x5_4_qmix.th
#   task_assigner=ppo → task_assign/models/safe/map_8x5_4_ppo.th
#   両方を規約ロードして実行
```

> **§4.1-4.7 との関係**: 上の経路探索同形式が **実行 (test) の基準**。
> §4.2-4.5 の 4 モード (pretrained/finetuning) や autosave/checkpoints の機構は LaRe 由来の重い案で、
> この規約ベースに寄せるなら **不要 or 簡素化できる**。学習側の保存先も
> `task_assign/models/safe/{map}_{N}_{task_assigner}.th` に統一すれば対称が保てる。
> (§4 をどこまで簡素化するかは別途判断)

---

## 5. 将来実装: DQN

タスク割当を **value-based** でも学習できるようにする案。PPO (policy-based) と比較できると研究上の価値が高い。
**現時点では実装しない** (PPO の完成が先)。着手時の見取り図として記録する。

### 5.1 PPO から流用できるもの (変更不要)

- 状態表現 `create_state` (§2)
- フラット離散行動空間 + 3 種マスク (§2)
- TaskManager 分岐 (`elif name == "dqn":`)
- 永続化の枠組み (§4。トークンを `DQN` に)

### 5.2 PPO から変える必要があるもの

| 項目 | PPO | DQN |
|---|---|---|
| バッファ | on-policy (エピソード単位で消費) | **replay buffer** (off-policy, FIFO で蓄積) |
| ネットワーク | actor + critic 分離 | **Q ネット 1 本** (+ target net) |
| 行動選択 | argmax / `Categorical` sample | **ε-greedy** (`set_test_mode` を ε=0/ε>0 に対応付け) |
| ロス | clip surrogate + value loss | **TD 誤差** (`r + γ max_a' Q_target(s',a') - Q(s,a)`) |
| `update_ready` | バッファ満杯 | replay が `min_buffer` 超え |
| `process_end_episode` | return 計算 | **基本 no-op** (off-policy で不要) |
| マスク適用 | logits を -inf | **無効行動の Q を -inf** にして argmax / target max |

### 5.3 インターフェース上の論点

`process_end_episode` / `compute_returns` は **on-policy 前提**。DQN では空実装になる。
→ アルゴが 2 つ目になる段階で **`BaseTaskPolicy` 抽象基底**に切り出すと冗長さを吸収できる (今は保留)。

### 5.4 クラス骨子 (擬似コード)

```python
class DQNAgent:
    def __init__(self, args):
        self.q      = QNet(input_dim, output_dim).to(device)   # 状態→各(agent,task)のQ
        self.q_targ = copy.deepcopy(self.q)
        self.replay = ReplayBuffer(args.dqn_buffer_size)        # off-policy
        self.eps    = args.dqn_eps_start
        self.test_mode = True

    def assign_task(self, env, current_tasklist, assigned_tasklist):
        # create_state → Q 算出 → マスクで無効行動を -inf
        # test_mode: argmax / else: ε-greedy
        # 学習時は (s, a, mask) を保持して buffer_add_rewards で transition 完成
        ...

    def update(self):
        # replay からサンプル → TD ターゲット → MSE → 定期的に target 同期
        # 返り値は (td_loss, 0.0, 0.0) 等で 3 値慣習に合わせる
        ...

    def update_ready(self):  return len(self.replay) >= self.args.dqn_min_buffer
    def process_end_episode(self): pass        # off-policy なので no-op
    def set_test_mode(self, m): self.test_mode = m
    def save_model(self, path): ...            # §4 と同形式
    def load_model(self, path): ...
```

> 注意: PPO は「行動時に s/a をバッファへ、報酬は後から `buffer_add_rewards`」という分割。
> DQN は **(s,a,r,s',done) の transition 単位**が自然なので、`assign_task` で s,a を一時保持し、
> 次回の `buffer_add_rewards` で前ステップの transition を確定させる繋ぎ込みが要る。

---

## 6. リスク・未決事項

| 項目 | 内容 | 対策 |
|---|---|---|
| PPO がそもそも未検証 | 学習が正しく回るか未確認 | §3.2 のチェックを最優先で潰す |
| PPO `update()` のインデント疑い | 最後の minibatch しか使っていない可能性 (§3.2) | 実装変更前に挙動確認 |
| `state_dict` 互換 | ネット構造変更で旧モデルがロード不能 | チェックポイントに `architecture_version` を埋める |
| optimizer state | finetuning で optimizer も復元したい | `torch.save({"model","optimizer","step"})` に統一 |
| autosave ディスク負荷 | 長時間学習でファイル蓄積 | `save_freq_steps` で間引き + 古いファイル削除は別途 |
| 命名トークン衝突 | LaRe `TASK` と PPO `PPO`、将来 `DQN` の混同 | アルゴごとに別トークン + ドキュメント明記 |
| 共通 IF の on/off-policy 差 | `process_end_episode` 等が on-policy 偏重 | アルゴが増えたら `BaseTaskPolicy` で吸収 (§5.3) |

---

## 7. 実装トリガ / 順序

1. **PPO を完成・検証** (§3.2) ← 最優先。これが終わるまで他は止める
2. **永続化 (§4)**: PPO が学習で性能を出せるようになったら
3. **DQN (§5)**: policy-based と value-based を比較したい具体ニーズが出たとき

それまでは本ガイドを **生きた草稿** として保持。実装時の微調整は許容。

---

## 8. まとめ

タスク割当 RL は TaskManager のプラグイン構造に載っており、**共通インターフェース (§2) を満たすクラスを足すだけ**でアルゴを増やせる設計。
ただし **現行 PPO は未完成・未検証** であり、まず「学習が正しく回る」ことの確定 (§3.2) が最優先。
その後に save/load (§4) を整備し、将来は DQN (§5) を off-policy 化の差分だけで追加できる。抽象基底化はアルゴが 2 つ目に増える段階で検討する。

---

最終更新: 2026-06-02
