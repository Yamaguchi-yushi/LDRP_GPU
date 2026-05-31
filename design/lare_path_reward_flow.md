# LaRe-Path 報酬計算フロー (runtime trace)

`env.step()` が呼ばれてから、エージェントが受け取る報酬 `ri_array` が確定するまでの流れを、コード上の位置 (file:line) と一緒に追ったもの。
学習が想定より短時間で終わる場合の切り分けにも使う。

> 対象: **LaRe-Path (System A)**。
> LaRe-Task (System B) は経路報酬ではなく task 割当用の proxy を `info["lare_task_proxy_reward"]` 経由で PPO に渡すだけで、env reward は書き換えないため、本文書では最後に概略のみ触れる。

---

## 1. 全体フロー (1 ステップで何が起きるか)

`env.step(joint_action)` ([src/main/drp_env/drp_env.py:691](../src/main/drp_env/drp_env.py#L691)) の内部の時系列:

```
step(joint_action)
 ├─ [前処理] LaRe-Path 用に prev_onehot 位置をスナップショット
 │     src/main/drp_env/drp_env.py:709-710
 │
 ├─ [step counter] _lare_total_step_account を +1 (externally_set のときは skip)
 │     drp_env.py:713-716
 │
 ├─ [運動] joint_action に従って各 agent を動かし obs_prepare を作る
 │     drp_env.py:727-799
 │
 ├─ [衝突判定] collision_detect(obs_prepare) → 衝突 pair も同時に算出
 │     drp_env.py:803-806
 │
 ├─ [env reward の生成] ri_array (= 各 agent の env 報酬) を計算
 │     ・collision: r_coll * speed を全 agent に
 │     ・non-collision: self.reward(i) を各 agent ごとに
 │     drp_env.py:819-850
 │
 ├─ [task 処理] (is_tasklist=True のときのみ)
 │     drp_env.py:861-963
 │
 ├─ [LaRe-Path フック] ★ ここで env reward を proxy reward に差し替える可能性 ★
 │     drp_env.py:974-989
 │     ├─ compute_factors(prev_onehot, colliding_pairs)
 │     │     → (n_agents, FACTOR_NUMBER=10) のファクター行列
 │     ├─ record_step(factors, env_reward_sum)
 │     │     → buffer に (ステップごとのファクター, エピソード累積 env_reward) を蓄積
 │     └─ if use_lare_path_training AND module.is_trained:
 │            proxy = decoder(factors)
 │            ri_array = proxy            ← 学習側が受け取る報酬がここで上書き
 │
 ├─ [LaRe-Task フック] info に proxy reward を載せる (env reward は触らない)
 │     drp_env.py:993-995
 │
 └─ [エピソード終端] all(terminated) のとき end_episode() を呼ぶ
       drp_env.py:997-1010
       └─ LaRe-Path.end_episode() で buffer を closeout + (条件成立時) decoder 更新
```

---

## 2. ファクター計算 (encoder: 環境の生情報 → 10 次元ベクトル)

[src/lare/path/encoder.py](../src/lare/path/encoder.py) で定義された `FACTOR_NUMBER = 10` 次元のファクターを各 agent について算出する。

呼び出し:
- `LaRePathModule.compute_factors()` ([src/lare/path/lare_path_module.py:129-145](../src/lare/path/lare_path_module.py#L129-L145))
  - 各 agent ごとに `build_lare_obs_for_agent(env, i, edge_info_cache, graph_diameter, prev_onehot_pos, colliding_pairs)` を呼ぶ
  - 返り値 obs を `evaluation_func(batch_obs)` に通して 10 次元の `factors` (np.float32) を生成

出力 shape: `(n_agents, 10)` — これが 1 ステップ分のファクター。

---

## 3. proxy reward 生成 (decoder: 10 次元 → スカラー報酬)

[src/lare/path/decoder.py](../src/lare/path/decoder.py) の `PathRewardDecoder` (MLP, hidden=64, n_layers=3) が 10 次元 → 1 次元のスカラーを出力する。

呼び出し:
- `LaRePathModule.proxy_rewards(factors)` ([src/lare/path/lare_path_module.py:147-158](../src/lare/path/lare_path_module.py#L147-L158))
  - `is_trained=False` のとき **None を返す** (= env reward が使われ続ける)
  - `is_trained=True` のとき `decoder.eval()` で forward → 各 agent 分のスカラーを返す
- 呼び出し元 ([drp_env.py:984-987](../src/main/drp_env/drp_env.py#L984-L987))
  ```python
  if self.use_lare_path_training and self.lare_path_module.is_trained:
      proxy = self.lare_path_module.proxy_rewards(factors)
      if proxy is not None:
          ri_array = [float(x) for x in proxy]
  ```

ポイント:
- `is_trained` フラグは「**decoder が一度でも学習済み or pretrained weight をロード済み**」を表す。
  - scratch 起動 → 最初の `_update()` が走った瞬間に `True`
  - pretrained ロード → `load_model()` 内で **直ちに True** ([lare_path_module.py:266-267](../src/lare/path/lare_path_module.py#L266-L267))

---

## 4. buffer と decoder 更新タイミング

### 蓄積 (buffer)

`PathEpisodeBuffer` ([src/lare/path/buffer.py](../src/lare/path/buffer.py))

- `add_step(factors, env_reward_sum)` ([buffer.py:27-36](../src/lare/path/buffer.py#L27-L36))
  - その step の `factors` (n_agents × 10) を `_cur_factors` に書き込み
  - `_cur_return += env_reward_sum` (env reward の累積を毎 step 足す)
  - **重要**: ここで保存される `env_reward_sum` は **proxy で書き換える前の生の env reward 合計**
    ([drp_env.py:981-982](../src/main/drp_env/drp_env.py#L981-L982) 参照、`ri_array` を proxy で上書きする **前** に `env_reward_sum` を確定している)
- `end_episode()` ([buffer.py:38-47](../src/lare/path/buffer.py#L38-L47))
  - 完了したエピソードを `(factors, length, return)` の辞書として deque (`capacity=512`) に append

### 学習 (decoder 更新)

`LaRePathModule.end_episode()` ([lare_path_module.py:163-175](../src/lare/path/lare_path_module.py#L163-L175))

```python
def end_episode(self):
    self.buffer.end_episode()
    self.episode_count += 1

    if self.frozen:        # ★ pretrained mode はここで return
        return

    if (len(self.buffer) >= self.cfg.min_buffer
        and self.episode_count % max(1, self.cfg.update_freq) == 0):
        self._update()     # MSE 学習
```

`_update()` ([lare_path_module.py:177-213](../src/lare/path/lare_path_module.py#L177-L213)) は:
1. buffer から `batch_size` 個のエピソードをサンプル
2. 各 step の `decoder(factors)` を時系列方向に sum → predicted return
3. `MSELoss(pred_return, actual_env_return)` で勾配を流す
4. 更新後 `is_trained=True` に固定

---

## 5. 3 モード × 報酬計算の挙動比較

| モード | フラグ設定 | `is_trained` の初期値 | proxy で env reward を上書きするか | decoder の MSE 学習 | end_episode() の処理量 |
|---|---|---|---|---|---|
| **(1) baseline** | `use_lare_path=False` | — (module 自体作らない) | 上書きなし (env reward そのまま) | しない | LaRe 関連処理が全て skip |
| **(2) scratch** | `use_lare_path=True`, `use_pretrained_lare_path=False`, `use_finetuning_lare_path=False` | `False` | 1 回目の `_update()` 完了後から ON | **する** (`update_freq` ごと) | 毎エピソード buffer append + 条件成立時に MSE 1 epoch |
| **(3) pretrained (frozen)** | `use_pretrained_lare_path=True` | **`True` (load 直後から)** | **常に ON** (1 step 目から proxy) | **しない** (`frozen=True` で早期 return) | buffer append のみ。**MSE 更新が走らない** |
| **(4) finetuning** | `use_finetuning_lare_path=True` | **`True` (load 直後から)** | 常に ON | **する** | scratch と同じ (warm start) |

---

## 6. 「学習時間が短くなった」原因の切り分け

### 現在のデフォルト (drp_env.py:28-46)

```python
use_lare_path=True,
use_pretrained_lare_path=True,
pretrained_lare_path_model_name="QMIX_LARE_map_8x5_2agents_1.2M_final.pth",
use_finetuning_lare_path=False,
```

→ **既定では (3) pretrained-frozen モード** で動く。

### この設定で起きること

- `load_model()` の中で `is_trained=True` & `frozen=True` がセット ([lare_path_module.py:266-268](../src/lare/path/lare_path_module.py#L266-L268))
- step ごと:
  - factors 計算は **走る**
  - proxy_rewards が常に呼ばれ、env reward を **毎 step 上書き**する
  - buffer.add_step は **走る**(ただし使われない)
- end_episode で:
  - `if self.frozen: return` ([lare_path_module.py:168-169](../src/lare/path/lare_path_module.py#L168-L169)) によって **decoder の MSE 学習がスキップされる**

### MARL4DRP (参照実装) と比べて時間が短い理由として考えられるもの

| 切り分け対象 | 確認方法 |
|---|---|
| **pretrained-frozen で動いている (= MSE 学習がスキップされている)** | 起動ログに `[LaRe-Path][PRETRAINED] Loaded ...` が出ているか確認 ([drp_env.py:462](../src/main/drp_env/drp_env.py#L462))。出ていれば frozen mode で動いており、decoder 更新分の時間が削減される (これは仕様) |
| **proxy への上書きは起きている**(= LaRe 自体は効いている) | デバッグログで `ri_array` が `compute_factors → decoder(z)` 経由の値になっているか確認。`use_lare_path_training=True` AND `is_trained=True` のときだけ swap 発生 ([drp_env.py:984-987](../src/main/drp_env/drp_env.py#L984-L987)) |
| **buffer に蓄積されている**(= 学習データは溜まっている) | `len(env.unwrapped.lare_path_module.buffer)` を覗く |
| **学習する設計だったか?(scratch / finetuning のつもりだったか?)** | もし「pretrained をロードしつつ追加学習させたい」のなら、`use_pretrained_lare_path=False` + `use_finetuning_lare_path=True` + `finetuning_lare_path_model_name=...` の組み合わせにする必要がある (drp_env.py:373-385 で `frozen` フラグが決まる) |

### 「計算がされていない」のチェック方法

検証スニペット ([CLAUDE.md](../CLAUDE.md) 末尾) のあと:

```python
m = env.unwrapped.lare_path_module
print("is_trained =", m.is_trained)           # pretrained ロード後は True
print("frozen     =", m.frozen)               # pretrained なら True
print("update_count =", m.update_count)       # frozen=True なら増えない
print("len(buffer) =", len(m.buffer))          # 走った episode 数だけ増える
```

期待値:
- pretrained-frozen で運用中なら → `is_trained=True`, `frozen=True`, `update_count` は load 時の値から **増えない**, `len(buffer)` は増える
- finetuning で運用中なら → `is_trained=True`, `frozen=False`, `update_count` が `update_freq` ごとに **増えていく**
- scratch で運用中なら → 最初 `is_trained=False` で env reward のまま走り、`min_buffer` 到達後の最初の `update_freq` で `True` に切り替わる

---

## 7. MARL4DRP (Safe-TSL-DBCT, 参照実装) との比較

「学習時間が従来より短い」の切り分けに直結する。LDRP の LaRe-Path は MARL4DRP (`~/MARL4DRP/`) の実装を出発点にしているが、いくつかの構造的・hyperparam 的な差がある。

### 7.1 構造の対応関係

| 役割 | LDRP | MARL4DRP |
|---|---|---|
| main module (encoder+decoder+buffer 結束) | `LaRePathModule` ([src/lare/path/lare_path_module.py](../src/lare/path/lare_path_module.py)) | `FactorRewardDecomposer` (`drp_env/reward_model/LLMrd/factor_reward_decompose.py:17-59`) |
| decoder MLP | `PathRewardDecoder` ([src/lare/path/decoder.py](../src/lare/path/decoder.py)) | `Factor_Reward_Model` (`drp_env/reward_model/LLMrd/factor_reward_model.py:8-44`) |
| factor 計算 | `evaluation_func` ([src/lare/path/encoder.py](../src/lare/path/encoder.py)) | `evaluation_func` (`drp_env/reward_model/LLMrd/fallback_functions/evaluation_func.py`) |
| buffer | `PathEpisodeBuffer` (deque) | `ReplayMemory_episode` (単一 capacity=1024 / 3-way split capacity=512×3) |
| env からの hook 点 | `step()` の末尾で **1 箇所** ([drp_env.py:974-989](../src/main/drp_env/drp_env.py#L974-L989)) でまとめて `compute_factors → record_step → (条件成立で) ri_array 上書き` | `step()` の **collision/non-collision 各分岐** で agent ごとに `_call_lare_reward_system(i)` を呼ぶ (`drp_env.py:1585-1685`) |

### 7.2 ファクター・decoder

| 項目 | LDRP | MARL4DRP |
|---|---|---|
| ファクター次元 | **10** (`FACTOR_NUMBER=10`) | **10** ✓ 同等 |
| ファクター項目 | `prog_goal / in_collision / others_in_collision / wait_norm / dist_goal_norm / min_sep_norm / avg_sep_norm / safety_margin / collision_risk / at_goal` | 同 ✓ (LDRP の `evaluation_func` は MARL4DRP からポートされており実質同一) |
| decoder | MLP (hidden=64, n_layers=**3**) + 任意で Transformer | MLP (hidden=64, n_layers=**5**) |
| Transformer オプション | あり (`lare_path_use_transformer`) | なし |

### 7.3 学習 hyperparams (★ ここが時間差の主要因 ★)

下表のうち **太字** は LDRP と MARL4DRP で値が異なる箇所。値は LDRP は [src/config/default.yaml](../src/config/default.yaml) を、MARL4DRP は drp_env.py のデフォルトを参照。

| パラメータ | LDRP | MARL4DRP | 効果 |
|---|---|---|---|
| `buffer_capacity` | 1024 | 1024 (単一) / 512×3 (3-way) | 同等 |
| `min_buffer` (= 学習開始しきい値, episodes) | 256 | 256 | 同等 |
| **`update_freq`** (decoder 更新の周期, episodes) | **3** | **128** | LDRP は **約 40 倍頻繁に**更新が走る |
| `batch_size` | 256 | 256 | 同等 |
| **`train_epochs` per update** | **1** | **50** (`evaluation_episodes=50`) | LDRP は 1 update あたりの最適化量が **1/50** |
| 学習 loss | `MSELoss(Σ_t decoder(z_t), env_return_sum)` | 同形 (`MSELoss(Σ_t decoder, episode_return)`) | 構造は同じ |
| 学習目標の元 (return) | `record_step` 時の `env_reward_sum` (proxy で上書きする **前** の env reward 合計) ✓ [drp_env.py:981-982](../src/main/drp_env/drp_env.py#L981-L982) | 同 (リワード合計 `np.sum(rewards, axis=(1,2))`) | 一致 |

**1 episode あたりの decoder 学習計算量 (amortized) の概算**:
- LDRP: `256 sample × 1 epoch / 3 ep` ≈ **85 forward-backward / ep**
- MARL4DRP: `256 sample × 50 epoch / 128 ep` ≈ **100 forward-backward / ep**

amortized では実は近い値だが、**バースト性が全く違う**:
- LDRP は 3 ep ごとに小さく走る (rollout と均す形)
- MARL4DRP は 128 ep ごとに `256×50=12800` step の big chunk が走る (体感の wall time は大きく見える)

### 7.4 モード切替 (pretrained / scratch / finetuning) と挙動の差

| 項目 | LDRP | MARL4DRP |
|---|---|---|
| モードの数 | 3 (scratch / pretrained=frozen / finetuning=trainable warm-start) | 3 (同じ意味) |
| pretrained 時の `is_trained` | load 直後に True ([lare_path_module.py:266-267](../src/lare/path/lare_path_module.py#L266-L267)) | load 直後に True (同等) |
| pretrained 時の **proxy 上書き** | step ごとに env reward を decoder 出力で上書きする ✓ | step ごとに上書きする ✓ |
| pretrained 時の **MSE 学習** | **`frozen=True` で end_episode が早期 return** → MSE 更新を完全スキップ ([lare_path_module.py:168-169](../src/lare/path/lare_path_module.py#L168-L169)) | reset() で `if not use_pretrained_model` チェックして **memory.push をスキップ** する (`drp_env.py:1253`)。train loop は呼ばれるが学習対象データが入らないので実質スキップ |
| 効果の違い | LDRP は train loop 呼び出し自体が走らない (より軽い) | MARL4DRP は train loop の関数呼び出し自体は毎 episode あるが内部で no-op |

### 7.5 ファイル命名規則

| | LDRP | MARL4DRP |
|---|---|---|
| 命名 | `{Safe_}{ALGO}_PATH_{map}_{N}agents_{X.X}M_{checkpoint\|final}.pth` | `{Safe_}{ALGO}_LARE_{map}_{N}agents_{X.X}M_{checkpoint\|final}.pth` |
| 識別子 | **`PATH`** (LaRe-Path / LaRe-Task を区別するため) | **`LARE`** (System を区別する概念がない) |
| 保存先 | `src/lare/path/checkpoints/` (autosave) と `src/lare/path/models/` (load 元) を分離 | `epymarl/src/saved_models/` 単一 |
| finetuning 時の prefix | `FT_<source_base>_...` | `FT_<source_base>_...` ✓ 同 |

### 7.6 「学習時間が短い」原因の確定的な切り分け手順

MARL4DRP との比較で時間が短くなる場合、原因は以下のいずれか(あるいは複合):

1. **pretrained-frozen で動いている**(最有力)
   - LDRP の `gym.make` デフォルトは `use_pretrained_lare_path=True` ([drp_env.py:41](../src/main/drp_env/drp_env.py#L41))
   - これだと `frozen=True` でデコーダ MSE 更新がスキップされる
   - **確認**: 起動ログに `[LaRe-Path][PRETRAINED] Loaded ...` が出る ([drp_env.py:462](../src/main/drp_env/drp_env.py#L462))
   - **判定**: `env.unwrapped.lare_path_module.update_count` が走行中に **増えない** → 完全に frozen
2. **scratch / finetuning で動いているが、`train_epochs=1` のため 1 update が軽い**
   - LDRP は per-update のオプティマイザ実行を 1 epoch しか回さない (MARL4DRP は 50 epoch)
   - 更新頻度 (`update_freq=3` vs MARL4DRP の 128) で部分的に補っているが、wall time の見え方は MARL4DRP の方が「ドサッと長い処理が定期的に発生」する
   - **確認**: `update_count` が `episodes / 3` 程度のペースで増えていれば scratch/finetuning は正常に走っている
3. **MARL4DRP 側で 3-way 分離メモリ (`use_separete_memory=True`) で動いていた場合**、LDRP は単一 deque のみで 3-way 分離なし
   - LDRP の yaml デフォルト `lare_path_buffer_capacity=1024` は MARL4DRP の単一メモリ運用 (`use_separete_memory=False`) に整合
   - もし参照実装側で 3-way 分離を使っていたなら、サンプリング戦略が違うので学習進度に差が出ても説明できる

### 7.7 「同じ条件で動かしたい」場合の補正

MARL4DRP と挙動を揃えて比較したいなら、test.py / train.py 側の env_args で以下を上書きする:

```yaml
# MARL4DRP の単一メモリ + 50 epoch + 128 ep 周期に揃える例
lare_path_update_freq: 128
# LDRP の LaRePathModule は train_epochs を cfg.train_epochs で持っているが
# 現状 yaml 露出していない (dataclass default=1) ので, drp_env 側で
# LaRePathConfig(train_epochs=50, ...) として渡すコード変更が必要 (今は未対応).
```

`train_epochs` だけは yaml 経由で渡せないことに注意。完全一致比較を取りたい場合は LaRePathConfig インスタンス化箇所 ([drp_env.py:391-414](../src/main/drp_env/drp_env.py#L391-L414)) に渡すように修正するか、`update_freq` を上げて 1 update の負荷を擬似的に大きくする。

---

## 8. (参考) LaRe-Task の流れ

経路報酬ではないが構造は対称。

- ファクター: [src/lare/task/encoder.py](../src/lare/task/encoder.py)
- decoder: [src/lare/task/decoder.py](../src/lare/task/decoder.py)
- module: [src/lare/task/lare_task_module.py](../src/lare/task/lare_task_module.py)
- env からの hook 点: [drp_env.py:908-930](../src/main/drp_env/drp_env.py#L908-L930) (task 割当時に `record_step_assignments`) と [drp_env.py:993-995](../src/main/drp_env/drp_env.py#L993-L995) (`info["lare_task_proxy_reward"]` に proxy を載せて PPO に渡す)
- env reward (`ri_array`) は **書き換えない**。task assigner (PPO) が `info[...]` を読みに行く設計。
- 学習タイミング: episode 終端で `lare_task_module.end_episode(self.task_completion)` ([drp_env.py:1006-1008](../src/main/drp_env/drp_env.py#L1006-L1008))
