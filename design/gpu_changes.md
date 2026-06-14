# GPU 版固有の変更一覧

PC 版 (LDRP) から GPU 版 (LDRP_GPU) に変更を取り込む際、**このファイルに記載のある箇所で衝突が起きたら GPU 版の変更を優先**する。

最終更新: 2026-06-13

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

## PC→GPU 取り込み手順

1. PC 版の変更を `git fetch` / `git cherry-pick` 等で取得
2. 衝突が発生したファイルを確認: `git status`
3. **このファイルに記載のある変更箇所で衝突した場合 → GPU 版 (`<<<<<<< HEAD` 側) を採用**
4. それ以外の衝突は内容を確認して判断
5. `git add` → `git commit`
