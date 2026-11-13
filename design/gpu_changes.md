# GPU 版固有の変更一覧

PC 版 (LDRP) から GPU 版 (LDRP_GPU) に変更を取り込む際、**このファイルに記載のある箇所で衝突が起きたら GPU 版の変更を優先**する。

最終更新: 2026-08-20

---

## 変更ファイル一覧

### 1. エピソードバッファを VRAM→RAM に移行

バッファを常に CPU に置き、学習時だけ GPU に転送することで VRAM を約 140 MiB 削減する。

#### `src/epymarl/src/runners/parallel_runner.py`

| 変更箇所 | PC 版 | GPU 版 |
|---|---|---|
| `setup()` — バッファ生成時のデバイス指定 | `device=self.args.device` | `device="cpu"` |
| `run()` — actions をバッファに書き戻す直前 | `actions.unsqueeze(1)` | `actions.to("cpu").unsqueeze(1)` |

#### `src/epymarl/src/learners/ppo_learner.py`

| 変更箇所 | PC 版 | GPU 版 |
|---|---|---|
| `train()` 冒頭 | なし | `batch = batch.to(self.args.device)` を追加 |

#### `src/epymarl/src/components/episode_buffer.py`

| 変更箇所 | PC 版 | GPU 版 |
|---|---|---|
| `EpisodeBatch.to()` 末尾 | `return` なし (None を返す) | `return self` を追加 |

#### `src/epymarl/src/controllers/basic_controller.py`

| 変更箇所 | PC 版 | GPU 版 |
|---|---|---|
| `forward()` 内、agent 呼び出し前 | なし | `device = self.hidden_states.device` / `agent_inputs = agent_inputs.to(device)` / `avail_actions = avail_actions.to(device)` を追加 |

---

### 2. LaRe デバイス制御 (`lare_device` 引数)

ワーカープロセスの LaRe を CPU に固定して VRAM の多重占有を防ぐ。

#### `src/main/drp_env/drp_env.py`

| 変更箇所 | PC 版 | GPU 版 |
|---|---|---|
| `DrpEnv.__init__` の `use_lare_path` デフォルト | `True` | `False` |
| `DrpEnv.__init__` の `use_pretrained_lare_path` デフォルト | `True` | `False` |
| `DrpEnv.__init__` に新規引数 | なし | `lare_device="auto"` |
| LaRe モジュール初期化呼び出し | `device` 未指定 | `lare_device=self.lare_device` を渡す |

#### `src/lare/path/lare_path_module.py`

| 変更箇所 | PC 版 | GPU 版 |
|---|---|---|
| `LarePathCfg` | `device` フィールドなし | `device: str = "auto"` を追加 |
| `__init__` 内デバイス選択 | `cuda` 無条件 or `is_available()` のみ | `"auto"` → `is_available()` / それ以外 → 指定値 |

#### `src/lare/task/lare_task_module.py`

`lare_path_module.py` と同様の変更（`LareTaskCfg` への `device: str = "auto"` 追加、デバイス選択ロジック変更）。

#### `src/epymarl/src/config/envs/gymma.yaml`

| 変更箇所 | PC 版 | GPU 版 |
|---|---|---|
| `lare_device` キー | なし | `lare_device: "cpu"` を追加 |
| `t_max` | `150050000` | `300050000` (学習ステップ数を倍に延長) |

---

### 3. MAPPO ハイパーパラメータ

#### `src/epymarl/src/config/algs/mappo.yaml`

| キー | PC 版 | GPU 版 | 理由 |
|---|---|---|---|
| `use_rnn` | `False` | `True` | RNN エージェントで時系列情報を活用 |
| `entropy_coef` | `0.01` | `0.05` | 探索を強化 |
| `q_nstep` | `5` | `50` | 長い horizon の価値推定 |

---

---

### 4. PC コミット取り込み (2026-08-15)

PC 版コミット b7c2395 (RNN hidden reset) / f6d5045 (MAT) / 0d77d58 (RNN 修正) / 25fb0f9 (PPO タスク割当大改修) を GPU 版に適用。

#### 適用ファイル (GPU 固有変更なし、PC 版そのまま採用)

| ファイル | 変更内容 |
|---|---|
| `src/all_policy/policy.py` | `reset_hidden()` 追加 |
| `src/all_policy/policy_manager.py` | `reset_hidden()` 追加 |
| `src/all_policy/policy_runner.py` | `reset_hidden()` 追加 |
| `src/epymarl/src/config/algs/mat.yaml` | 新規: MAT アルゴリズム設定 |
| `src/epymarl/src/modules/agents/mat_agent.py` | 新規: MAT エージェント |
| `src/epymarl/src/modules/critics/mat_full.py` | 新規: MAT critic |
| `src/epymarl/src/modules/agents/__init__.py` | MAT 登録追加 |
| `src/epymarl/src/modules/critics/__init__.py` | MAT 登録追加 |
| `src/task_assign/task_manager.py` | ppo1 削除 |
| `src/task_assign/task_policy/ppo.py` | Buffer n_envs 対応 / `update()` → stats dict / モデル保存系追加 |
| `src/task_assign/task_policy/ppo1.py` | 削除 |

#### GPU 版で追加実装した変更

| ファイル | 変更内容 |
|---|---|
| `runner.py` | `both_policy_manager` 参照バグ修正 / TensorBoard 統合 / 学習ループ全面改修 |
| `test.py` | `training` フラグ / `train_task_assigner` バリデーション / seed 統一 / `Runner(training=)` 引数 |
| `src/config/default.yaml` | `train_task_assigner` / `seed` / `eval_seed` 等 PPO 学習制御キー追加 |
| `src/main/drp_env/drp_env.py` | `randomize_task_arrival` / `episode_seed_base` / タスク統計カウンタ追加 |
| `train.py` | `task_p_high=0.10` / `task_p_low=0.01` / `randomize_task_arrival` 等追加 |

---

### 5. PC コミット取り込み (2026-08-18): dynamic fleet sizing

PC 版コミット `c8693f5 dynamic fleet sizing` / `a4ddc6b update` / `681ea84 modify dynamic fleet sizing`
を取り込み。稼働エージェント台数を制御する機能 (`use_dynamic_agents`)。station ノードに待機している
agent を非稼働扱いにし、衝突判定・安全制御・タスク割当から除外する。

**現状は台数固定のみ**: エピソード開始時に稼働台数を決めたら、そのエピソード中は増減しない。
`drp_env.py` には復帰 (非稼働 agent にタスクが割り当てられたら稼働) と帰還 (`task_assign[i] == -2`)
の処理が入っているが、**これを駆動する割当器がない** — TP / Random / PPO はいずれも非稼働・帰還中の
agent を skip し、`-2` を返す実装もどこにも存在しないため、現状これらの分岐は通らない。

#### 適用ファイル (PC 版そのまま採用)

| ファイル | 変更内容 |
|---|---|
| `src/epymarl/src/config/envs/gymma.yaml` | `use_dynamic_agents` / `randomize_initial_active` / `min_active_agents` / `max_active_agents` / `initial_active_num` を追加 |
| `src/config/default.yaml` | 同じ 5 キーを追加 (test.py / run.py 用) |
| `src/main/drp_env/EE_map.py` | `station_nodes` (node.csv 5 列目) 読み込み / `collision_detect(active=)` 追加 |
| `src/main/drp_env/drp_env.py` | dynamic agents 本体 (`active` / `pending_off` / `_nearest_station()` 等) |
| `src/main/drp_env/wrapper/safe_marl.py` | 非稼働 agent を安全制御の対象外に。あわせて `do = True` を「実際に `joint_action` を書き換えたときだけ」立てるよう修正 (681ea84) |
| `src/task_assign/task_policy/{ppo,random,tp}.py` | 非稼働・帰還中 agent にタスクを割り当てない |
| `test.py` | `dynamic_agent_kwargs` を `gym.make()` に渡す |

**`safe_marl.py` の `do = True` 修正について**: 修正前は `joint_action[i] = self.current_start[i]` が
no-op のときも `do = True` が立つため、同一ノードに 2 台が重なって両方その場待機を選ぶと
`while do:` が終わらなかった。動的フリートは非稼働 agent を全員同じ station ノード (全マップ 1 個のみ)
に置くので、そこから 2 台目が稼働開始した瞬間に成立する。GPU 版で先に検出し PC 版に反映済み。

#### GPU 版で修正した点 (PC 版と差分が残る)

| ファイル | PC 版 | GPU 版 | 理由 |
|---|---|---|---|
| `src/main/drp_env/drp_env.py` (station 到着判定) | `str(pos_agenti) == str(self.pos[self.goal_array[i]])` | `[float(v) for v in pos_agenti] == [float(v) for v in self.pos[self.goal_array[i]]]` | numpy 2.x では `str()` による座標比較が禁止 (CLAUDE.md 開発環境)。既存の到着判定と同じ書式に統一 |

#### `train.py` (部分取り込み)

PC 版は実験設定 (`num_runs` / `env_args.key` / `t_max` / `task_arrival`) も同時に書き換えているが、
これらは GPU 版で実行中の設定を壊すため取り込まない。**dynamic fleet 系の env_args 4 キーのみ追加**する。

| 項目 | PC 版 | GPU 版 |
|---|---|---|
| `env_args.use_dynamic_agents` 他 4 キー | 追加 | **同じく追加** (実行ブロック / CPU 版コメントブロックの両方) |
| `num_runs` | `5` → `3` | GPU 版の値を維持 |
| `env_args.key` | `5agent_map_8x5` → `7agent_map_aoba00` | GPU 版の値を維持 |
| `t_max` | `30050000` → `100050000` | GPU 版の値を維持 |
| `env_args.task_arrival` | `bernoulli` → `fixed` | GPU 版の値を維持 |

#### 衝突解決メモ

| ファイル | 衝突内容 | 採用 |
|---|---|---|
| `src/main/drp_env/drp_env.py` | PC 版が `__init__` 引数を `=` 前後に空白を入れる書式へ変更 | PC 版 (書式のみ、挙動差なし) |
| `src/config/default.yaml` | PC 版 681ea84 が PPO 学習制御キー (`train_task_assigner` 等) を末尾に追加。GPU 版は `fd3cccb` で既に上部に追加済み | GPU 版 (重複を避けるため theirs 側ブロックを破棄) |

---

### 6. PC コミット取り込み (2026-08-20): 階層強化学習 (同時学習)

PC 版コミット `5cc852f Hierarchical Reinforcement Learning` / `9659583 modify` を取り込み。epymarl の経路方策学習と
PPO タスク割当器の学習を 1 プロセスで同時に回す (`train_task_assigner=True`)。

#### 適用ファイル (PC 版そのまま採用)

| ファイル | 変更内容 |
|---|---|
| `src/epymarl/src/config/default.yaml` | `train_task_assigner: False` / `task_num: 10` を追加 |
| `src/epymarl/src/envs/__init__.py` | `_GymmaWrapper.step()` が dict action (`{"pass","task"}`) を受理 / `get_task_state()` を追加 |
| `src/epymarl/src/run.py` | `train_task_assigner=True` のとき runner を parallel へ強制切替 / `PPOAgent` を生成し `runner.task_assigner` に設定 / 学習ループに `process_end_episode()` + `update()` / モデル保存先を `path` / `task` サブディレクトリに分離 |
| `src/epymarl/src/runners/parallel_runner.py` | `setup(..., task_assigner=None)` / worker が `task_state` を返す / step 送信を dict action 化 / `buffer_add_rewards(env_idx=)` |
| `src/main/drp_env/drp_env.py` | `get_task_state()` を追加 (current_tasklist / assigned_tasks / obs_onehot / active 等をコピーして返す) |
| `src/task_assign/task_policy/ppo.py` | `assign_task()` を `_assign()` に分離し `assign_task_from_state(task_state, env_idx, step_idx, test_mode)` を追加 |

#### 衝突解決メモ

| ファイル | 衝突内容 | 採用 |
|---|---|---|
| `src/epymarl/src/config/default.yaml` | `save_model_interval` PC 版 `100000 → 5000000` vs GPU 版 `10000000` | **GPU 版 (`10000000`)** |
| `src/epymarl/src/runners/parallel_runner.py` | `setup()` のシグネチャ変更 hunk が GPU 固有の `device="cpu"` をコンテキストに含む | 3-way マージで自動解決。`device="cpu"` と `actions.to("cpu")` は保持されていることを確認済み |

#### `train.py` (部分取り込み)

| 項目 | PC 版 | GPU 版 |
|---|---|---|
| `train_task_assigner=False` | 追加 | **同じく追加** (実行ブロック / CPU 版コメントブロックの両方) |
| `num_runs` | `3` → `5` | GPU 版も `5` のため差分なし |

#### モデル保存パスの変更 (`train_task_assigner` に関係なく常に効く)

save 側が `learner.save_models(os.path.join(save_path, "path"))` に変わったため、出力先が
`{local_results_path}/models/{token}/{t}/` から **`{t}/path/`** に 1 段深くなる。
`train_task_assigner=True` のときは併せて `{t}/task/` に PPO 割当器が保存される。

load 側は PC 版 `9659583 modify` で追従済み — `{t}/path/agent.th` があればそちらを、
なければ従来どおり `{t}/` を見るので、**旧形式のチェックポイントもそのまま読める**。

---

### 7. PC コミット取り込み (2026-08-24): 同時学習・ppo.py 改修

PC 版コミット `b175a7f learning at the same time` / `85cadf7 modify` を取り込み。
PPO タスク割当器を `use_dynamic_agents` に対応させ、phase-2 (非稼働エージェント呼び出し) を追加。

#### 適用ファイル (PC 版そのまま採用)

| ファイル | 変更内容 |
|---|---|
| `src/task_assign/task_policy/ppo.py` | `assign_task_from_state()` に `stride = task_num + 1 if use_dynamic_agents else task_num` を追加し全マスク indexing を `stride` ベースに変更。`can_off` / `mask[-1]` ロジック追加。phase-2 ブロック (非稼働 agent 起動判定) 追加。`if len_current_task <= 0 and not self.use_dynamic_agents: break` に変更 (85cadf7)。`create_state()` に `phase=0` 引数と active agent ベクトル + phase スカラー追加。 |

#### GPU 版で修正した点

特になし (PC 版をそのまま採用)。

---

### 8. PC コミット取り込み (2026-08-26): fleet sizing ppo (c0df9b2)

衝突バグ修正・ステーションノード除外・SafeEnv 強化。

#### 適用ファイル (PC 版そのまま採用)

| ファイル | 変更内容 |
|---|---|
| `src/main/drp_env/EE_map.py` | `create_task()` / `create_tasklist()` に `exclude_nodes` 引数追加 |
| `src/main/drp_env/drp_env.py` | `exclude_station_from_tasks` パラメータ追加 / スポーン時ステーション衝突チェック追加 |
| `src/main/drp_env/wrapper/safe_marl.py` | `_predict_pos()` 追加 / `pending_off` 渋滞防止 / エッジ上衝突予測ブロック追加 |
| `test.py` | `key=val` 引数パース (`model_seed=N`, `use_safe_env=true/false`) / `exclude_station_from_tasks` を `dynamic_agent_kwargs` に追加 / `config.allow_reassign_before_pickup = reassign_flag` をループ内に移動 |
| `src/epymarl/src/run.py` | `_ta.update()` に `use_dynamic_agents` 自動流入を追加 |
| `src/epymarl/src/config/envs/gymma.yaml` | `exclude_station_from_tasks: null` を追加 |
| `src/config/default.yaml` | `exclude_station_from_tasks: null` を追加 |

---

## 2026-09-08 `tools/` (学習 run 収集・監視) 取り込み時の PC との差分

PC 版コミット `64aaa84` / `8d3e834` を取り込んだ際、**意図的に PC と揃えなかった箇所**。
次回 `git merge` / `cherry-pick` で衝突したらここを確認する。

| ファイル | GPU 版の扱い | 理由 |
|---|---|---|
| `.gitignore` | PC が追加した `CLAUDE.md` の行は **入れない**。`tools/` 関連 5 行のみ追加 | GPU 版は `CLAUDE.md` を git 追跡している (PC 版は無視対象)。衝突時は **GPU 側を採用** |
| `design/future_work.md` | **取り込まない** (PC 版の項目 4〜10 の再構成を持ち込まない) | GPU 版は項目 1 に「GPU 環境への移行」を持つため目次が分岐している。監視機能とは無関係の設計項目。衝突時は **GPU 側を採用** |
| `design/run_collector.md` | PC 版をそのまま取り込み (新規ファイル) | 学習・実行系に触らないため差分管理不要 |
| `tools/collect_runs.py` | **GPU 版のみ修正あり** (下記「第 3 フォールバック」)。衝突時は **GPU 側を採用** | PC 機では `cout.txt` が埋まるため顕在化しない、GPU 機固有の欠陥の修正 |
| `tools/` の他ファイル | PC 版をそのまま取り込み (新規ファイルのみ) | 同上 |
| `MANUAL.md` | PC の履歴エントリを GPU 版の日付順に合わせて挿入 (本文は GPU 機の事情に合わせて加筆) | GPU 機は `cout.txt` が 0 byte / このマシンは「収集される側」である旨を明記 |

### このマシン (GPU 機) 固有の注意

- `results/` 直下ではなく `src/epymarl/results/sacred/` に sacred 出力がある (ツールの既定 `DEFAULT_SACRED_SUBDIRS` に含まれるので設定不要)
- マシン名は `cat` (hostname 由来) として記録される
- `tools/plan.md` を触らない / `--notion` `--publish-models` を使わない (実験計画は集約マシンが master)

### `tools/collect_runs.py`: `t_env` の第 3 フォールバック追加 (GPU 版のみ)

**症状**: GPU 機では `cout.txt` が **0 byte**。原因は `src/epymarl/src/main.py` の `SETTINGS['CAPTURE_MODE'] = "no"` (sacred の `no_tee()` は誰も書き込まない `StringIO` を返すため、`--beat-interval` に関係なく常に空)。※ 当初「stdout のシェルリダイレクト」が原因と記録していたが誤り (2026-09-25 訂正。PC 由来の `tools/README.md` / `design/run_collector.md` / `collect_runs.py` のコメントも同じ誤解のまま)
保険の `metrics.json` 末尾 256KB 読みも、このファイルが **pretty-print で 29MB** あるため
末尾が最後の `values` 配列 (`0.0` の羅列) だけで埋まり、`steps` の整数に届かない。
結果 `t_env` が `?` になり、進捗が出ない。

**危険だった点**: 状態判定は `t_last is None` かつ `sacred_status == COMPLETED` の run を
**無検証で `done` (OK) と信じる** 実装 (`scan_run_dir` 下流の状態判定)。
そのため実際には `t_max` 未達の run が `OK` に混ざり得た。

**変更**: `scan_run_dir()` 内、`model = find_model(...)` の直後に以下を追加。

```python
if t_last is None and model and model.get("step") is not None:
    t_last = model["step"]
    t_source = "model"
```

epymarl は `{local_results_path}/models/{unique_token}/{t_env}` に保存するので、
最後のチェックポイント名がそのまま最後の `t_env` になる。既存の `find_model()` が
返す `step` (= 最大チェックポイント番号) を流用するだけで、追加の I/O は無い。
精度は `save_model_interval` の粒度だけ切り捨てられる。

**効果** (294 run で検証):

- `?/150.05M` → `150.02/150.05M (100.0%)` と表示されるようになった
- 進捗不明のまま残るのは 25 件だが**全て `FAIL`** (チェックポイント保存前に落ちた run) で、
  **無検証の `OK` はゼロ**になった
- `t_source` は保存のみで分岐に使われていないため、値 `"model"` の追加による副作用は無い

---

## 2026-09-25 PC コミット取り込み (`42624e3` 〜 `44e5d6d`, 10 件)

### 適用ファイル (PC 版そのまま採用)

`tools/` 一式 (collect_runs.py / dashboard / mini.py / plan.py / plist / README / VISIT_CHECKLIST.md 等), `aggregate.py` (新規), `src/all_policy/policy.py`, `design/run_collector.md`, `design/dynamic_agent_count.md`, `design/make_graph_integration_prompt.md` (新規), `src/lare/path/models/QMIX_PATH_Safe_map_8x5_7agents_20.0M_checkpoint.pth` (新規)

### 衝突解決メモ

| ファイル | 衝突内容 | 採用 |
|---|---|---|
| `.gitignore` | `tools/plan.md` の後ろに PC が `tools/plan.md.bak` を追加 | 両方 (PC の追加行を残す)。`CLAUDE.md` の行は引き続き入れない |
| `src/config/default.yaml` | ① `exclude_station_from_tasks` の説明コメント ② PPO 学習制御キーのブロック | ① PC 側 (コメント追加のみ) ② **GPU 版** (上部に配置済みのため PC 側の重複ブロックを破棄)。PC が追加した `model_seed` / `save_eval_json` / `use_safe_env` / `episode_seed_base: 0` は採用 |
| `run.py` | PC が複数 seed 評価 (`model_stem` / `list_model_seeds` / `aggregate.py` 呼び出し) に全面改修 | **PC 版の構造を採用し、選択リストだけ GPU 版の値に戻した**: `map_8x5` / `agent_num=[7, 10]` / `mappo` / `task_assigner=[fifo, tp]` / `method_tag=[safe, ours]`。GPU の `test.py` は既に `model_seed=` / `use_safe_env=` / 位置引数 `base`・数字を受け付けるので改修不要 |
| `train.py` (`44e5d6d`) | PC の実験条件 (qmix / map_8x5 / t_max / LaRe / `maxpurocesses=2` 等) と dashboard フックが同居 | **部分取り込み**: `_publish_batch()` (`~/.ldrp/batch_<pid>.json` にバッチ進捗を書く) の追加 4 箇所だけ入れ、実験条件は GPU 版のまま |
| `tools/collect_runs.py` | PC が 3 回大改修 (`36a0969` / `d018eff` / `44e5d6d`) | 自動マージで通過。**GPU 版の `t_env` 第 3 フォールバック (`t_source = "model"`) が残っていることを確認済み** (PC 版には無い)。299 run で無検証の `OK` はゼロ |

### スキップ

- `f24d98d new model`: 同一バイナリが GPU 版に `239b4ac` で既にコミット済みのため空コミットになり skip
- `df66cf5 add reward` (ステーション待機中の移動ペナルティ): 報酬まわりは取り込まない方針のため未適用のまま

---

## 2026-10-03 `logging.py`: sacred の `run.info` への二重記録を停止 (GPU 版のみ, `2d3af49`)

**衝突時は GPU 側を採用** (PC 版は旧実装のまま)。

| ファイル | PC 版 | GPU 版 |
|---|---|---|
| `src/epymarl/src/utils/logging.py` `log_stat()` | `self.sacred_info[key].append(value)` 等で `run.info` にも記録 | `run.info` への記録をコメントアウト。`log_scalar` (→ `metrics.json`) のみ |

**理由**: sacred の heartbeat スレッドが `run.info` を丸ごと JSON 化して `info.json` に書く最中に、
メインスレッドの `log_stat()` が同じ辞書・リストを書き換え、segfault していた (同期なし)。
`info.json` が大きい aoba00 / 10agent でほぼ毎回発生。

| run | 条件 | 症状 |
|---|---|---|
| `449919326` (9/30) | mat / aoba00 / 10agent / beat=60 | 最終 heartbeat 09:12:14.465 と segfault 09:12:14 が一致、`info.json` 0 KB |
| `145137824` (10/2) | mappo / aoba00 / 10agent / beat=60 | 同上 (21:42:15 / 21:42:16) |
| `121013450` (10/3) | mappo / aoba00 / 10agent / beat=60 | 同上 (00:45:32 / 00:45:33)。落ちた場所が `_json.cpython-39...so` の中 |

`info.json` はどこからも読まれておらず、中身は `metrics.json` と同じデータの重複なので失う情報は無い。
これにより `train.py` の `--beat-interval=60` (3600 → 60) が安全になる。3600 は過去にこのクラッシュの頻度を
下げるために入れていた値 (理由の記録が無かったため、`train.py` にコメントで残した)。

**未解決**: QMIX (CPU 実行) の segfault 群 (8/11〜9/2) と 9/29 の `libtorch_python.so` 内の segfault は
heartbeat と時刻が一致せず、原因不明。

---

## 2026-10-28 PC コミット取り込み (`808b510` 〜 `b15ec22`, 18 件)

`44e5d6d` を共通の祖先として、各ファイルに PC 側の差分だけを 3-way マージ (`git merge-file`) で適用した
(cherry-pick ではなく、ファイル単位でまとめて取り込み)。

### 適用ファイル (自動マージ / PC 版そのまま)

`.gitignore`, `aggregate.py`, `runner.py`, `src/all_policy/policy.py`, `src/task_assign/task_policy/ppo.py`,
`tools/` 一式 (新規: `tools/dist/`, `export_tools.py`, `lare_factor_check.py`, `lare_factor_smoke.py`,
`lare_models.py`, `md_toc.py`, `setup_launchd.py`),
新規モデル `FT_QMIX_PATH_Safe_map_8x5_5agents_10.0M_Safe_map_aoba00_5agents_10.0M_checkpoint.pth`

### 衝突解決メモ

| ファイル | 衝突内容 | 採用 |
|---|---|---|
| `run.py` | `agent_num` の `10,` の前の空白 | GPU 版 (条件の選択値は GPU 版を維持) |
| `src/config/default.yaml` | PC が `eval_arrivals` / `eval_allow_reassign` / 到着分布の既定値を追加 | PC 側。重複キーなしを確認 |
| `src/epymarl/src/utils/logging.py` | **PC 側も GPU の `info.json` 停止修正を取り込んだ** (コメントの文面だけ違う) | PC 側。TensorBoard の書き方 (`SummaryWriter`) は GPU 版のまま |
| `test.py` | PC が評価ループを「到着分布 × reassign の全組み合わせ」に拡張、`use_dynamic_agents=` / `arrival=` / `allow_reassign=` 引数を追加 | PC 側 (残る差は空白とコメントのみ) |
| `CLAUDE.md` | PC 側で削除 | GPU 版は残す (git 追跡を継続) |

### 取り込みで挙動が変わる点

- **評価 (`test.py` / `run.py`)**: 既定で `eval_arrivals` 4 種 × `eval_allow_reassign` 2 種 = **8 条件**を順に評価する
  (従来は 1 条件)。評価時間が約 8 倍になる。絞るときは `arrival=` / `allow_reassign=` 引数か default.yaml で指定
- **評価モデルの名前規則 (`policy.py` の `model_stem`)**: `task_assigner=ppo` なら `_ppo`、動的なら `_dyn` が
  `reassign_tag` の前に入る。PPO タスク割当器のモデルは `src/task_assign/models/safe/` から自動で読む
  (GPU 機には未作成)
- GPU 固有の差分 (`collect_runs.py` の t_env 第 3 フォールバック、`runner.py` / `ppo.py` / `default.yaml` の
  既存差分) は保持されていることを確認済み

---

## 2026-11-10 PC コミット取り込み (`a4930df` 〜 `16e4a0b`, 6 件)

前回 (2026-10-28) と同じく、`b15ec22` を祖先として PC 側の差分だけをファイル単位で取り込んだ (未コミットの前回取り込み分の上に適用)。
取り込み前の作業ツリーは `git stash` に `backup before PC import a4930df..16e4a0b` として保存 (作業ツリーは変更せず `stash store` のみ)。

### 適用ファイル

| 種別 | ファイル | 方法 |
|---|---|---|
| 新規 | `tools/autorun.py`, `tools/autorun_train.py`, `tools/com.ldrp.autorun.plist`, `tools/eval_lare_factor.py` | PC 版そのまま |
| 新規モデル | `src/lare/path/models/QMIX_PATH_Safe_map_8x5_5agents_10.0M_checkpoint_waitedit.pth`, `..._map_8x5_7agents_10.0M_checkpoint.pth`, `..._map_aoba00_5agents_10.0M_checkpoint_waitedit.pth` | PC 版そのまま (`..._map_aoba00_5agents_10.0M_checkpoint.pth` は `3ab476a` で同一内容を追加済み) |
| GPU 側が祖先と同一 | `tools/dashboard/{app.py, static/app.js, static/style.css, templates/index.html}` | PC 版で置換 |
| 3-way マージ (衝突なし) | `.gitignore`, `tools/collect_runs.py`, `tools/lare_factor_smoke.py`, `tools/lare_models.py` | `git merge-file` |

### GPU 固有差分の保持確認

- `tools/collect_runs.py`: `t_env` 第 3 フォールバック (`t_source = "model"`) は残っている (PC 版との差はこの 8 行のみ)
- `.gitignore`: PC 版にある `CLAUDE.md` 行は入れていない (GPU 版は CLAUDE.md を git 追跡)
- `src/` のコア実装 (env / epymarl / LaRe) は今回の 6 件で変更なし

---

## 2026-11-14 PC コミット取り込み (`835dfcf` 〜 `1ff4397`, 12 件)

`a4930df` を共通の祖先として、各ファイルに PC 側の差分だけを 3-way マージ (`git merge-file`) で適用。
主な中身は **LaRe-Path の評価関数の版切り替え (`lare_path_factor_version`, 版 1〜3)** と tools の更新。

### 衝突解決メモ

| ファイル | 衝突内容 | 採用 |
|---|---|---|
| `src/main/drp_env/drp_env.py` (3 箇所) | 版 2・3 用の記録 (`_lare_events` 等)、info に `off_command` / `off_done` / `spawn` を追加、配達完了時の `_reassign_event = True` | PC 側。GPU 固有 (`lare_device`、LaRe の既定値、`reset_prepick_assignments`) は残っていることを確認 |
| `src/main/drp_env/wrapper/safe_marl.py` (2 箇所) | PC が **動的エージェント用の処理をすべてコメントアウト** (「[動的台数] 2026-11-02 一旦無効化」) し、`_lare_override` を追加 | PC 側 (PC 版と同一)。**`use_dynamic_agents=True` の run では、非アクティブ機も安全制御の対象になり、station の渋滞防止も効かなくなる**。台数固定の run には影響なし |
| `train.py` (3 箇所) | 実験条件 | **GPU 版を維持**。PC の `lare_path_factor_version` の説明コメントだけ追加 |
| `MANUAL.md` | 更新履歴の先頭 | 両方残す (PC の新しい履歴 7 件を GPU 固有の履歴の上に挿入) |
| `tools/collect_runs.py` / `dashboard/app.py` / `lare_models.py` / `autorun.py` | GPU 側に PC の途中の版が入っていた | PC の最新版を採用し、**`collect_runs.py` の t_env 第 3 フォールバックを入れ直した** |

### 取り込みで入った変更 (PC 版の MANUAL に詳細あり)

- `lare_path_factor_version` (既定 1 = 従来どおり)。版 2・3 は学習済みデコーダが無いので scratch でのみ使う
- **LaRe のモデルが無い・読めないと `gym.make` で止まる** (PC 10/16。2026-10-16 に GPU 側で見つけた「黙って LaRe なしになる」問題の修正)
- 新規モデル `FT_QMIX_PATH_Safe_map_aoba00_5agents_10.0M_Safe_map_aoba00_7agents_10.0M_checkpoint.pth`
- `.claude/settings.json` (応答を日本語にする念押しの hook のみ)

### 動作確認 (2026-11-14)

台数固定・動的・LaRe 版 1 (pretrained)・版 2・版 3 (scratch) で reset → 30 step が通ること、存在しないモデル名で
`FileNotFoundError` になることを確認。

---

## PC→GPU 取り込み手順

1. PC 版の変更を `git fetch` / `git cherry-pick` 等で取得
2. 衝突が発生したファイルを確認: `git status`
3. **このファイルに記載のある変更箇所で衝突した場合 → GPU 版 (`<<<<<<< HEAD` 側) を採用**
4. それ以外の衝突は内容を確認して判断
5. `git add` → `git commit`
