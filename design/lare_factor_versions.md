# LaRe-Path の評価関数を番号で切り替える (版 1 = DRP の 10 因子 / 版 2・3 = LLM 生成)

- 作成日: 2026-10-15
- 状態: §3〜§4.4 (版 2) は **白 (LDRP) に適用済み (2026-11-14)**。GPU 版 (LDRP_GPU) は未適用
- 2026-10-16: §4.5 (モデルが無い・読めないときは止める) を追加。版の切り替えとは独立に先に入れられる
- 2026-10-16: **§4.5 は白 (LDRP) に適用済み** (本体で §4.5 の確認を通した)。GPU 版 (LDRP_GPU) は未適用
- 2026-11-02: **§8 (版 3 = LLM 評価関数 v2) を追加**。版 2 を入れたあとに入れる。版 2 の上に足した形で、複製で動作確認済み。**未適用**
- 2026-11-14: **§3〜§4.4 (版 2) と §8 (版 3) を白 (LDRP) に適用済み** (ユーザーの指示で Claude が直接編集。§4.2 (e) は §4.5 (a) で済んでいるので入れていない。§4.4・§8.7 の train.py は、既定のコマンドを変えないよう、使い方のコメント行だけ入れた)。本体で §5・§8.8 の確認を通した (期待値と一致)
- 関連: [lare_ldrp_prompt.md](lare_ldrp_prompt.md) (版 2 の評価関数を作った経緯)、[lare_path_reward_fit.md](lare_path_reward_fit.md)

<!-- toc -->

**目次**

- [1. 何をしたいか](#1-何をしたいか)
- [2. 全体の仕組み](#2-全体の仕組み)
- [3. 新しく作るファイル (3 つ)](#3-新しく作るファイル-3-つ)
  - [3.1 `src/lare/path/factors_ldrp_v1.py`: LLM の評価関数をそのまま置く](#31-srclarepathfactors_ldrp_v1py-llm-の評価関数をそのまま置く)
  - [3.2 `src/lare/path/obs_ldrp.py`: 版 2 の観測 (19 区間) を作る](#32-srclarepathobs_ldrppy-版-2-の観測-19-区間-を作る)
  - [3.3 `src/lare/path/factor_versions.py`: 版の登録表](#33-srclarepathfactor_versionspy-版の登録表)
- [4. 既存ファイルの変更](#4-既存ファイルの変更)
  - [4.1 `src/lare/path/lare_path_module.py` (5 か所)](#41-srclarepathlare_path_modulepy-5-か所)
  - [4.2 `src/main/drp_env/drp_env.py` (8 か所、字下げはタブ)](#42-srcmaindrp_envdrp_envpy-8-か所字下げはタブ)
  - [4.3 `src/main/drp_env/wrapper/safe_marl.py` (5 か所、字下げはタブ)](#43-srcmaindrp_envwrappersafe_marlpy-5-か所字下げはタブ)
  - [4.4 train.py](#44-trainpy)
  - [4.5 LaRe のモデルが無い・読めないときは止める (drp_env.py 4 か所、先に単独で入れてよい)](#45-lare-のモデルが無い読めないときは止める-drp_envpy-4-か所先に単独で入れてよい)
- [5. 確認方法](#5-確認方法)
- [6. よくあるつまずき](#6-よくあるつまずき)
- [7. 注意](#7-注意)
- [8. 版 3 (F3 = LLM 評価関数 v2) を足す](#8-版-3-f3--llm-評価関数-v2-を足す)
  - [8.1 何をしたいか](#81-何をしたいか)
  - [8.2 全体の仕組み](#82-全体の仕組み)
  - [8.3 新しく作るファイル (2 つ)](#83-新しく作るファイル-2-つ)
  - [8.4 `src/lare/path/factor_versions.py` (2 か所)](#84-srclarepathfactor_versionspy-2-か所)
  - [8.5 `src/lare/path/lare_path_module.py` (3 か所)](#85-srclarepathlare_path_modulepy-3-か所)
  - [8.6 `src/main/drp_env/drp_env.py` (6 か所、字下げはタブ)](#86-srcmaindrp_envdrp_envpy-6-か所字下げはタブ)
  - [8.7 train.py](#87-trainpy)
  - [8.8 確認方法](#88-確認方法)
  - [8.9 よくあるつまずき](#89-よくあるつまずき)
  - [8.10 注意](#810-注意)

<!-- /toc -->

---

## 1. 何をしたいか

train.py で `env_args.lare_path_factor_version=<番号>` を指定するだけで、LaRe-Path の評価関数 (因子) と、それに合った観測の作り方を切り替える。

| 番号 | 中身 | 観測 |
|---|---|---|
| 1 (既定) | DRP 用の 10 因子 (MARL4DRP の評価関数。**今までと同じ**) | 今の `build_lare_obs_for_agent` |
| 2 | LDRP 用の 10 因子 (LLM が生成。`design/prompts/responses/v1/summarized.json`) | 新しい 19 区間の観測 (`build_ldrp_obs_for_agent`) |
| 3 | LDRP 用の 10 因子 (LLM が生成。`design/prompts/responses/v2/summarized.json`)。[§8](#8-版-3-f3--llm-評価関数-v2-を足す) | 版 2 の観測に要望を足したもの (`build_ldrp_v2_obs_for_agent`) |

- 番号を書かなければ 1 なので、**今までの学習・評価は何も変わらない** (不変条件も守られる)
- 版を足すときは、観測の作り方と評価関数を 1 組にして登録表に足すだけにする

---

## 2. 全体の仕組み

```text
train.py   env_args.lare_path_factor_version=2
   ↓ (epymarl が gym.make に渡す)
drp_env.py DrpEnv(lare_path_factor_version=2)
   ├─ LaRePathConfig(factor_version=2) を作る
   ├─ 版 2 の観測に要る値を記録する (ピックアップ・配達の印、割当時刻、タスクの発生時刻)
   └─ 保存ファイル名に F2 を付ける (QMIX_PATH_F2_Safe_map_8x5_...)
safe_marl.py  SafeEnv が機体ごとに「動きを止めたか」を env に書く (_lare_override)
   ↓
lare_path_module.py  LaRePathModule
   ├─ factor_versions.get_factor_version(2) → {観測の作り方, 評価関数}
   ├─ compute_factors: 観測を作る → 評価関数 → 10 因子
   └─ 保存・読み込み: ファイルに版を書き、違う版のデコーダは読まない
factor_versions.py (新規)  1 → encoder.py の関数 / 2 → obs_ldrp.py + factors_ldrp_v1.py
obs_ldrp.py (新規)         版 2 の 19 区間の観測を作る
factors_ldrp_v1.py (新規)  LLM が書いた evaluation_func をそのまま置く
```

ダッシュボード側 (tools/collect_runs.py) は Claude が対応済み: 版 2 の run は setting が `safe F2`、評価用モデルのタグが `safe_f2` になり、LaRe なし・版 1 の run と別の条件に数えられる。

---

## 3. 新しく作るファイル (3 つ)

### 3.1 `src/lare/path/factors_ldrp_v1.py`: LLM の評価関数をそのまま置く

**何をしたいか:** LLM が書いた関数を 1 文字も変えずに置く (論文で「人の手を加えていない」と言えるように)。手で写すと写し間違いが起きるので、応答の JSON からスクリプトで書き出す。

```bash
cd ~/LDRP
/opt/anaconda3/envs/ldrp/bin/python - <<'EOF'
import json
fn = json.load(open("design/prompts/responses/v1/summarized.json"))["Functions"]
hdr = '''"""LaRe-Path 評価関数 v2 (LDRP 用, LLM が生成).

LLM (Claude Opus 5.5, 思考の強さ: 高) がプロンプト v1 から生成した 3 候補を、同じ LLM が
1 つにまとめたもの。人の手は加えていない (Functions の文字列をそのまま貼った)。
  プロンプト:   design/prompts/lare_ldrp_prompt_v1.txt
  元の応答:     design/prompts/responses/v1/summarized.json
  生成の記録:   design/prompts/responses/v1/README.md
観測の並びは obs_ldrp.build_ldrp_obs_for_agent (19 区間) を前提にしている。
返す因子 (10 個): deliver, pickup, progress, route, override, stall, delay, proximity, blocker, collision
"""

'''
open("src/lare/path/factors_ldrp_v1.py", "w").write(hdr + fn)
print("written")
EOF
```

| 書き方 | 意味 | 注意点 |
|---|---|---|
| `python - <<'EOF' ... EOF` | `EOF` までの行をそのまま python に渡して実行する | `'EOF'` と引用符で囲むので、中の `$` や `\` はシェルに解釈されない |
| `json.load(...)["Functions"]` | JSON の `Functions` の値 (関数のコードの文字列) を取り出す | キーが無いと `KeyError` で止まる |

### 3.2 `src/lare/path/obs_ldrp.py`: 版 2 の観測 (19 区間) を作る

**何をしたいか:** プロンプト v1 に書いた並び (OBSERVATION FORM) のとおりに、機体ごとの観測ベクトルを作る。LLM の評価関数はこの並びを前提に書かれているので、**並びはプロンプトと 1 対 1** にする。

```python
"""LaRe-Path 用の観測 (LDRP 版, 19 区間) を機体ごとに作る.

並びは design/prompts/lare_ldrp_prompt_v1.txt の OBSERVATION FORM と同じ。
評価関数 v2 (factors_ldrp_v1.evaluation_func) はこの並びを前提に書かれている。
並びを変えるときはプロンプトも変え、評価関数を生成し直すこと。

    [own_prev N][own_curr N][dest N][phase 3][delivery N][events 2]
    [steps_since_assign 1][task_age 1][active 1]
    [others_pos (A-1)*2N][others_phase (A-1)*3][others_active (A-1)]
    [demand 2][collision_distance 1][collision_info 2][wait_count 1][safety_override 1]
    [node_coords 2N][meta 6 = diameter, N, A, T, C, t]

env 側で持つ値 (drp_env.py で記録する):
    _lare_events[i]       = [このステップにピックアップした, 配達した]
    _lare_assign_step[i]  = 今のタスクを割り当てられたステップ
    _lare_task_created[i] = 今のタスクが発生したステップ
    _lare_override[i]     = このステップ SafeEnv に動きを止められたか (safe_marl.py が書く)
"""

import numpy as np

from .encoder import get_node_coordinates_flat


def _phase_of(env, i):
    """0 待機中 / 1 ピックアップへ / 2 配達へ."""
    task = env.assigned_tasks[i]
    if not task:
        return 0
    return 2 if env.goal_array[i] == task[1] else 1


def _is_active(env, i):
    if getattr(env, "use_dynamic_agents", False):
        return bool(env.active[i])
    return True


def build_ldrp_obs_for_agent(env, agent_id, graph_diameter,
                             prev_onehot_position, current_colliding_pairs):
    n = env.n_nodes
    A = env.agent_num
    eye3 = np.eye(3)

    def curr_onehot(i):
        oh = np.asarray(env.obs_onehot[i]).flatten()
        return oh[:n].copy() if oh.size >= n else np.zeros(n)

    def prev_onehot(i):
        if prev_onehot_position is not None:
            return np.asarray(prev_onehot_position[i], dtype=float).copy()
        return curr_onehot(i)

    def node_onehot(node):
        v = np.zeros(n)
        if node is not None and 0 <= int(node) < n:
            v[int(node)] = 1.0
        return v

    i = agent_id
    task = env.assigned_tasks[i]
    phase = _phase_of(env, i)
    step = float(env.step_account)

    dest = node_onehot(env.goal_array[i]) if task else np.zeros(n)
    delivery = node_onehot(task[1]) if task else np.zeros(n)
    events = np.asarray(env._lare_events[i], dtype=float)
    if task:
        since = max(0.0, step - float(env._lare_assign_step[i]))
        age = max(0.0, step - float(env._lare_task_created[i]))
    else:
        since, age = 0.0, 0.0
    active = 1.0 if _is_active(env, i) else 0.0

    others_pos, others_phase, others_active = [], [], []
    for j in range(A):
        if j == i:
            continue
        others_pos += [prev_onehot(j), curr_onehot(j)]
        others_phase.append(eye3[_phase_of(env, j)])
        others_active.append(1.0 if _is_active(env, j) else 0.0)

    waiting = float(sum(1 for v in env.assigned_list if v == -1))
    idle = float(sum(1 for j in range(A) if _is_active(env, j) and not env.assigned_tasks[j]))

    self_coll, others_coll = 0.0, 0.0
    for pair in current_colliding_pairs or []:
        if i in pair:
            self_coll = 1.0
        else:
            others_coll = 1.0

    override = getattr(env, "_lare_override", None)
    over = float(override[i]) if override is not None else 0.0

    meta = np.array([float(graph_diameter), float(n), float(A), float(env.time_limit),
                     float(env.task_num), step])

    return np.concatenate([
        prev_onehot(i), curr_onehot(i), dest, eye3[phase], delivery, events,
        [since, age, active],
        np.concatenate(others_pos) if others_pos else np.zeros(0),
        np.concatenate(others_phase) if others_phase else np.zeros(0),
        np.asarray(others_active, dtype=float),
        [waiting, idle, float(env.colli_distan_value), self_coll, others_coll,
         float(env.wait_count[i]), over],
        get_node_coordinates_flat(env), meta,
    ]).astype(np.float64)
```

| 行 | Python としての意味 | ここでの役割 |
|---|---|---|
| `return 2 if env.goal_array[i] == task[1] else 1` | 条件式 (三項演算子) | env はピックアップした瞬間に目的地を配達先へ切り替えるので、目的地が配達先なら「配達へ」 |
| `getattr(env, "use_dynamic_agents", False)` | 属性を名前で取り出し、無ければ False | 台数を変えない設定 (属性はあるが False) でも、古い env でも止まらない |
| `np.eye(3)[phase]` | 単位行列の phase 行目 = one-hot | 段階の one-hot ([待機中, ピックアップへ, 配達へ]) |
| `sum(1 for v in env.assigned_list if v == -1)` | 条件に合う要素を数える (ジェネレータ式) | 未割当で待っているタスクの数 (`assigned_list` はタスクの枠ごとの割当先、-1 は未割当) |
| `getattr(env, "_lare_override", None)` | 無ければ None | SafeEnv を使わない env (drp-…) では誰も書かないので 0 として扱う |
| `np.concatenate([...])` | 配列をつなげて 1 本にする | プロンプトの並び (19 区間) の順につなぐ。順番を変えると評価関数が別の値を読む |

**待機中のとき `dest` を全部 0 にする理由:** env の `goal_array` は待機中も「今いる場所」を指している。そのまま入れると「目的地に着いている」と読めてしまい、プロンプトの説明 (待機中は全部 0) と食い違う。

### 3.3 `src/lare/path/factor_versions.py`: 版の登録表

```python
"""LaRe-Path の評価関数 (因子) の版と、それに対応する観測の作り方.

train.py の env_args.lare_path_factor_version で番号を選ぶ。
  1: DRP 用の 10 因子 (MARL4DRP の評価関数。今までの LaRe-Path)
  2: LDRP 用の 10 因子 (LLM が生成, design/prompts/responses/v1)
版を足すときは、観測の作り方と評価関数を 1 組にしてここに登録する。
学習済みのデコーダは版ごとに別物 (因子の意味が違う) なので、読み込み時に版を照合する。
"""

from .encoder import build_lare_obs_for_agent, evaluation_func as _eval_drp10
from .obs_ldrp import build_ldrp_obs_for_agent
from .factors_ldrp_v1 import evaluation_func as _eval_ldrp_v1

FACTOR_VERSIONS = {
    1: {"name": "drp10", "n_factors": 10,
        "build_obs": build_lare_obs_for_agent, "evaluate": _eval_drp10},
    2: {"name": "ldrp_llm_v1", "n_factors": 10,
        "build_obs": build_ldrp_obs_for_agent, "evaluate": _eval_ldrp_v1},
}


def get_factor_version(version):
    try:
        return FACTOR_VERSIONS[int(version)]
    except (KeyError, TypeError, ValueError):
        raise ValueError("unknown lare_path_factor_version: %r (choose from %s)"
                         % (version, sorted(FACTOR_VERSIONS)))
```

| 書き方 | 意味 | 注意点 |
|---|---|---|
| `from .encoder import evaluation_func as _eval_drp10` | 別の名前を付けて取り込む | 2 つの `evaluation_func` が同じ名前でぶつからないようにする |
| 辞書の値に関数を入れる (`"evaluate": _eval_drp10`) | 関数も値として持てる | 呼ぶときは `spec["evaluate"](obs, apsp)` |
| `except (KeyError, TypeError, ValueError)` | 3 種類の例外をまとめて受ける | 登録に無い番号・数でない値を、分かりやすい 1 つのエラーにする |

---

## 4. 既存ファイルの変更

### 4.1 `src/lare/path/lare_path_module.py` (5 か所)

**(a) 取り込み** — `from .transformer import TimeAgentTransformer` **の次**:

```python
from .decoder import PathRewardDecoder
from .transformer import TimeAgentTransformer
from .factor_versions import get_factor_version
```

**(b) 設定に版を足す** — `class LaRePathConfig:` の `factor_dim: int = FACTOR_NUMBER` **の次**:

```python
class LaRePathConfig:
    factor_dim: int = FACTOR_NUMBER
    # 評価関数 (因子) の版。観測の作り方と評価関数の組を factor_versions.py から選ぶ
    factor_version: int = 1
    decoder_hidden_dim: int = 64
```

**(c) 版から因子の数を決める** — `__init__` の `self.factor_dim = self.cfg.factor_dim` **を置き換える** (~80 行):

```python
        self.apsp, self.graph_diameter = self.compute_apsp(env)

        self.factor_version = int(self.cfg.factor_version)
        self._factor_spec = get_factor_version(self.factor_version)
        self.factor_dim = self._factor_spec["n_factors"]
        self.n_agents = env.agent_num
```

**(d) 版の観測・評価関数を使う** — `compute_factors` (~185 行) の 2 行を置き換える:

```python
        rows = []
        for i in range(self.n_agents):
            obs_row = self._factor_spec["build_obs"](
                self.env, i, self.graph_diameter,
                prev_onehot_position, current_colliding_pairs,
            )
            rows.append(obs_row)
        batch_obs = np.stack(rows, axis=0)
        factor_list = self._factor_spec["evaluate"](batch_obs, self.apsp)
        factor_arr = np.concatenate(factor_list, axis=-1).astype(np.float32)
        return factor_arr
```

**(e) 保存・読み込みで版を照合する** — `save_model` の `"factor_dim": self.factor_dim,` **の次**に 1 行、`load_model` に 2 か所:

```python
            "factor_dim": self.factor_dim,
            "factor_version": self.factor_version,
            "n_agents": self.n_agents,
```

```python
                    f"factor_dim mismatch: model={saved_factor_dim}, current={self.factor_dim}"
                )
            # 版が違うデコーダは因子の意味が違うので使えない (古いファイルは版 1)
            saved_version = int(payload.get("factor_version", 1))
            if saved_version != self.factor_version:
                raise ValueError(
                    f"factor_version mismatch: model={saved_version}, current={self.factor_version}"
                )
            decoder_sd = payload["decoder_state_dict"]
```

```python
        elif "model_state_dict" in payload:
            # -- MARL4DRP format -- (MARL4DRP の評価関数 = 版 1 のデコーダ)
            if self.factor_version != 1:
                raise ValueError(
                    f"factor_version mismatch: MARL4DRP model is version 1, current={self.factor_version}"
                )
            decoder_sd = self._convert_marl4drp_state_dict(payload["model_state_dict"])
```

| 行 | Python としての意味 | ここでの役割 |
|---|---|---|
| `payload.get("factor_version", 1)` | キーが無ければ 1 を返し、止まらない | 今までに保存したファイルには版が書かれていないので、版 1 として扱う。`payload["factor_version"]` だと古いファイルで `KeyError` |
| `raise ValueError(...)` | 例外を出して止める | 因子の数 (10) は同じでも意味が違うので、読めてしまうと気づかずに間違ったデコーダで学習する |

### 4.2 `src/main/drp_env/drp_env.py` (8 か所、字下げはタブ)

| # | 場所 (メソッド / 概算行) | 目印にする既存コード | 内容 |
|---|---|---|---|
| (a) | `__init__` の引数 (~68 行) | `lare_path_save_freq_steps=500_000,` **の次** | 引数 `lare_path_factor_version=1,` |
| (b) | `__init__` (~240 行) | `self.lare_path_save_freq_steps = int(lare_path_save_freq_steps)` **の次** | 属性に保存 |
| (c) | `_lare_build_save_filename` (~359 行) | `return f"{algo}_PATH_{safe_seg}...` **を置き換え** | 版 2 以降は `PATH_F2_` |
| (d) | `_init_lare_path` (~493 行) | `cfg = LaRePathConfig(` の `factor_dim=factor_dim,` **の次** | `factor_version=...` を渡す |
| (e) | `_init_lare_path` の終わり (~526 行) | `except Exception as e:` **の直前** | 設定の誤りは止める |
| (f) | `reset` (~879 行) | `self._agent_task_creation = [None] * self.agent_num` **の次** | 記録用の 4 つのリストを作る |
| (g) | `step` の初め (~992 行) | `# LaRe-Path: snapshot per-agent onehot positions BEFORE movement.` **の直前** | ピックアップ・配達の印を毎ステップ消す |
| (h) | `step` の配達の処理 (~1183 行) | `self.task_completion += 1` **の次** | 配達の印 |
| (i) | `step` の割当の処理 (~1235 行) | `self.assigned_list[task_assign[i]] = i # update assigned_list` **の次** | 割当時刻・発生時刻 |
| (j) | `step` のピックアップの処理 (~1284 行) | `self.goal_array[i] = self.assigned_tasks[i][1]` **の次** | ピックアップの印 |

入れたあとの形:

```python
			lare_path_save_freq_steps=500_000,
			lare_path_factor_version=1,	# LaRe-Path の評価関数の版 (src/lare/path/factor_versions.py)
			lare_task_save_freq_steps=500_000,
```

```python
		self.lare_path_save_freq_steps = int(lare_path_save_freq_steps)
		self.lare_path_factor_version = int(lare_path_factor_version)
		self.lare_task_save_freq_steps = int(lare_task_save_freq_steps)
```

```python
			return f"FT_{source}_{safe_seg}{map_name}_{agents}agents_{steps}_{suffix}.pth"
		# 評価関数の版 1 (今まで) は付けない。2 以降は PATH の後ろに F2 のように付ける
		ver = f"F{self.lare_path_factor_version}_" if self.lare_path_factor_version != 1 else ""
		return f"{algo}_PATH_{ver}{safe_seg}{map_name}_{agents}agents_{steps}_{suffix}.pth"
```

```python
			cfg = LaRePathConfig(
				factor_dim=factor_dim,
				factor_version=self.lare_path_factor_version,
				decoder_hidden_dim=decoder_hidden_dim,
```

```python
					f"factors={factor_dim})"
				)
		except ValueError:
			# 評価関数の版と学習済みデコーダが合わないなど、設定の誤り。
			# 環境の報酬に黙って戻すと LaRe なしで学習していることに気づけないので止める
			# (§4.5 を先に入れた場合は、その except (FileNotFoundError, ValueError) で済むので入れない)
			raise
		except Exception as e:
			print(f"[LaRe-Path] Failed to initialize, falling back to env reward: {e}")
```

```python
		self._agent_task_creation = [None] * self.agent_num
		# LaRe-Path の観測 (版 2 以降) 用の記録. 学習・報酬には使わない
		self._lare_events = [[0, 0] for _ in range(self.agent_num)]	# [ピックアップした, 配達した]
		self._lare_task_key = [None] * self.agent_num			# 今のタスク (割当が変わったかの判定)
		self._lare_assign_step = [0] * self.agent_num			# 今のタスクを割り当てたステップ
		self._lare_task_created = [0] * self.agent_num			# 今のタスクが発生したステップ
```

```python
			task_assign = self._default_task_assign_tp()

		# LaRe-Path の観測用: このステップのピックアップ・配達の印を消す
		self._lare_events = [[0, 0] for _ in range(self.agent_num)]
		# LaRe-Path: snapshot per-agent onehot positions BEFORE movement.
		if self.use_lare_path and self.lare_path_module is not None:
```

```python
							self.assigned_tasks[i] = [] # remove the task from assigned_tasks
							self.task_completion += 1
							self._lare_events[i][1] = 1
							self._lare_task_key[i] = None
							self._reassign_event = True
```

```python
					self.goal_array[i] = self.assigned_tasks[i][0] # update goal to pick node
					self.assigned_list[task_assign[i]] = i # update assigned_list
					# LaRe-Path の観測用: 別のタスクになったときだけ割当時刻を更新する
					# (再割当ありでは毎ステップ同じタスクが割り当て直されるため)
					if self._lare_task_key[i] != tuple(task_r):
						self._lare_task_key[i] = tuple(task_r)
						self._lare_assign_step[i] = self.step_account
						self._lare_task_created[i] = creation_step

					lare_task_decisions.append({
```

```python
							self.start_ori_array[i] = self.goal_array[i]
							self.goal_array[i] = self.assigned_tasks[i][1]
							self._lare_events[i][0] = 1
							try:
```

| 行 | Python としての意味 | ここでの役割 |
|---|---|---|
| `[[0, 0] for _ in range(self.agent_num)]` | 内包表記で、機体ごとに別々の `[0, 0]` を作る | `[[0, 0]] * n` と書くと全機体が同じリストを指し、1 台に印を付けると全員に付く |
| `except ValueError: raise` | ValueError だけは受けてそのまま投げ直す (止める) | 版の番号の誤りなどで、LaRe なしの学習に黙って切り替わるのを防ぐ。それ以外の例外は今までどおり環境の報酬に戻す |
| `tuple(task_r)` | リストを、比べられる不変の組にする | 「前と同じタスクか」を比べる鍵。再割当ありでは、同じ機体に同じタスクが毎ステップ割り当て直される (`reset_prepick_assignments`) ので、変わったときだけ割当時刻を更新する |
| `creation_step` | 割当の処理の少し上で計算済みの変数 | そのタスクが発生したステップ (既存のコードで LaRe-Task 用に計算しているものを使う) |

**`step` の初めで印を消す場所について:** 印はこのステップの中で付き (配達・ピックアップ)、その後の LaRe-Path の因子の計算 (`compute_factors`、~1362 行) で読まれる。次のステップの初めに消せば、因子の計算は「このステップの出来事」だけを見る。

### 4.3 `src/main/drp_env/wrapper/safe_marl.py` (5 か所、字下げはタブ)

**何をしたいか:** 今は止めた回数の合計 (`safety_intervention_count`) しか持っていない。機体ごとに「このステップ止めたか」を持ち、env に書いておく。

| # | 目印にする既存コード | 内容 |
|---|---|---|
| (a) | `step` の `i = 0` **の直前** | `override = [0] * self.agent_num` |
| (b)〜(d) | `self.safety_intervention_count += 1` (3 か所) **それぞれの次** | `override[i] = 1` |
| (e) | `joint_action = {"pass": joint_action, "task": task_assign} if task_assign is not None else joint_action` **の直前** | `self._lare_override = override` |

入れたあとの形 ((a) と (e)、(b)〜(d) は 3 か所とも同じ形):

```python
			self.safety_intervention_count = 0

		# このステップ、機体ごとに動きを止めたか (LaRe-Path の観測 版 2 以降で使う)
		override = [0] * self.agent_num

		i = 0
		do = True
```

```python
							if joint_action[i] != self.current_start[i]:
								self.safety_intervention_count += 1
								override[i] = 1
								joint_action[i] = self.current_start[i]
```

```python
						break
					
		self._lare_override = override
		joint_action = {"pass": joint_action, "task": task_assign} if task_assign is not None else joint_action
		obs, ri_array, self.terminated, info = super().step(joint_action)
```

`self._lare_override` を `super().step(...)` (env の step) **より前**に書くのは、env の step の中で因子を計算するときに読むため。

### 4.4 train.py

`env_args.use_lare_path_training=True` などの LaRe-Path の設定の並びに 1 行足す。

```python
        f'env_args.use_lare_path=True '
        f'env_args.use_lare_path_training=True '
        f'env_args.lare_path_factor_version=2 '   # LaRe-Path の評価関数の版 (1: DRP の 10 因子 / 2: LLM 生成)
        f'env_args.use_pretrained_lare_path=False '
        ...
        f'env_args.use_finetuning_lare_path=False '
```

- **版 2 では、事前学習・fine-tuning のデコーダは使えない** (今あるデコーダは全部版 1)。`use_pretrained_lare_path=False`、`use_finetuning_lare_path=False` にして、スクラッチ (学習しながらデコーダを作る) で回す
  - 両方 False にし忘れると、読み込みで版が合わず「scratch に切り替える」の警告が出てスクラッチで回る (止まらない)。ダッシュボードの setting の表示が「事前学習あり」になって紛らわしいので、False にしておく
  - §4.5 を入れたあとは、版が合わないと起動時に `ValueError` で止まる (警告だけでは進まない)
- 版 1 のときはこの行を書かなくてよい (既定が 1)

### 4.5 LaRe のモデルが無い・読めないときは止める (drp_env.py 4 か所、先に単独で入れてよい)

2026-10-16 に追加。**版の切り替え (§3〜§4.4) とは独立**で、今のコードにそのまま入れられる。先に入れる場合、§4.2 の `except ValueError:` の差し替えは、この節の (a) で済む (同じ場所なので二重に入れない)。

**何をしたいか:** 今は、指定した pretrained / finetuning のモデルが無い・読めないと、**黙って LaRe を切るか scratch に切り替えて学習が進む**。config.json には「LaRe あり (事前学習あり)」と残るので、後から見分けられない。2026-10-15 に GPU で、未配置の `8x5_2 10M→8x5_5 5M` を指定した 4 本が LaRe なしで走った。これを、起動直後に止めるようにする。

止まる仕組みが 2 段ある:

```text
_init_lare_path (初期化)
  try:
    ...
    _load_lare_path_weights(...)      ← (b) ここで読み込む
      ├ ファイルが無い  → FileNotFoundError を投げる (今もそう)
      └ 読めない        → 今は「scratch に切り替え」の print だけ  → (b) で ValueError を投げる
  except (FileNotFoundError, ValueError):   ← (a) 新設。投げ直して止める
      raise
  except Exception as e:                    ← 今まで FileNotFoundError もここで受けて、LaRe を切っていた
      print("... falling back to env reward")
```

| # | 場所 (メソッド / 概算行) | 目印にする既存コード | 内容 |
|---|---|---|---|
| (a) | `_init_lare_path` の末尾 (~522 行) | `except Exception as e:` (次の行が `print(f"[LaRe-Path] Failed to initialize, ...")`) **の直前** | 止める例外を先に受けて投げ直す |
| (b) | `_load_lare_path_weights` の末尾 (~561 行) | `except Exception as e:` **から** `print(f"⚠️ [LaRe-Path][{label}] Load failed ({e}); falling back to scratch.")` **まで** | print をやめて ValueError を投げる |
| (c) | `_init_lare_task` の末尾 (~619 行) | (a) と同じ形の `except Exception as e:` (`[LaRe-Task] Failed to initialize`) **の直前** | (a) と同じ (Task 側) |
| (d) | `_load_lare_task_weights` の末尾 (~648 行) | `except Exception as e:` **から** `print(f"⚠️ [LaRe-Task][{label}] Load failed ...")` **まで** | (b) と同じ (Task 側) |

**(a) 追加するコード**

```python
		except (FileNotFoundError, ValueError):
			# 指定したモデルが無い・読めない・評価関数の版が合わない。
			# 環境の報酬に黙って戻すと use_lare_path=False のまま学習が進み、config.json には
			# 「LaRe あり」と残るので後から見分けられない (2026-10-15 に GPU で 4 本が LaRe なしで走った)
			raise
```

**(a) 入れたあとの形**

```python
			else:
				print(
					f"🌱 [LaRe-Path] Scratch (online train, training={self.use_lare_path_training}, "
					f"factors={factor_dim})"
				)
		except (FileNotFoundError, ValueError):
			# 指定したモデルが無い・読めない・評価関数の版が合わない。
			# 環境の報酬に黙って戻すと use_lare_path=False のまま学習が進み、config.json には
			# 「LaRe あり」と残るので後から見分けられない (2026-10-15 に GPU で 4 本が LaRe なしで走った)
			raise
		except Exception as e:
			print(f"[LaRe-Path] Failed to initialize, falling back to env reward: {e}")
			self.use_lare_path = False
			self.lare_path_module = None
```

**(b) 差し替えるコード** (`except Exception as e:` の行はそのまま、その下の print 1 行を 3 行に)

```python
		except Exception as e:
			# 壊れたファイル・形の違うデコーダなど。scratch に切り替えて続けると、
			# 事前学習ありのつもりで事前学習なしの run になるので止める
			raise ValueError(f"[LaRe-Path][{label}] Load failed: {resolved}: {e}") from e
```

**(b) 入れたあとの形**

```python
		try:
			self.lare_path_module.load_model(resolved, freeze=freeze)
			emoji = "🧊" if freeze else "🔧"
			mode = "frozen (inference only)" if freeze else "trainable (finetuning)"
			print(f"{emoji} [LaRe-Path][{label}] Loaded {resolved} - {mode}")
		except Exception as e:
			# 壊れたファイル・形の違うデコーダなど。scratch に切り替えて続けると、
			# 事前学習ありのつもりで事前学習なしの run になるので止める
			raise ValueError(f"[LaRe-Path][{label}] Load failed: {resolved}: {e}") from e
```

**(c)(d) Task 側:** (a)(b) と同じコードを入れる。読み替えるのは 3 か所: コメントの `use_lare_path` → `use_lare_task`、メッセージの `[LaRe-Path]` → `[LaRe-Task]`。

| 行 | Python としての意味 | ここでの役割 |
|---|---|---|
| `except (FileNotFoundError, ValueError):` | 括弧で並べた例外のどれかなら受ける。`except` は**上から順に**照合され、最初に合ったものだけが動く | `Exception` より**前**に置く。下に置くと、`FileNotFoundError` も `Exception` の一種なので先に `except Exception` が受けてしまい、今と変わらない |
| `raise` (引数なし) | 受けた例外をそのまま投げ直す | `_load_lare_path_weights` の「止める」という本来の意図どおり、`gym.make` ごと止まる |
| `raise ValueError(...) from e` | 新しい例外を投げ、元の例外を原因として付ける (トレースバックに両方出る) | `load_model` が出す例外の種類 (RuntimeError・KeyError・pickle のエラーなど) はばらばらなので、ValueError 1 つにまとめて (a) で止める。`from e` で元の原因も見える |
| (Exception の受け方を残す理由) | — | 想定外の初期化の失敗 (import できない等) は、今までどおり環境の報酬で続ける。止めるのは「指定したものと違う条件で学習が進む」ケースだけ |

**変わること:**

- 指定したモデルが無い・読めない run は、`gym.make` の時点で `FileNotFoundError` / `ValueError` で止まる (学習が始まらない)
- モデルが読める run、scratch、LaRe なしの動きは変わらない (下の確認で同じことを確かめた)
- **破壊的変更**: 今まで警告だけで進んでいた設定が止まるようになる。入れたら MANUAL.md の更新履歴に書く
- 動いている学習には影響しない (起動時にコードを読み込み済み)。**GPU 版 (LDRP_GPU) にも同じコードがある**ので、同じように直す

**確認方法 (複製で確認済み、2026-10-16):** 存在しないモデル名と、壊れたファイルを指定して止まることを確かめる。

```bash
cd ~/LDRP
echo "not a model" > src/lare/path/models/BROKEN.pth
/opt/anaconda3/envs/ldrp/bin/python -c "
import gym, sys; sys.path.append('.'); sys.path.append('./src/main'); import drp_env
base = dict(state_repre_flag='onehot_fov', reward_list={'goal':100,'collision':-100,'wait':-10.,'move':-1},
            time_limit=8, task_flag=True, task_list=None)
for name, kw in [('no LaRe', dict(use_lare_path=False)),
                 ('scratch', dict(use_lare_path=True)),
                 ('pretrained ok', dict(use_lare_path=True, use_pretrained_lare_path=True,
                      pretrained_lare_path_model_name='QMIX_PATH_Safe_map_8x5_2agents_10.0M_checkpoint')),
                 ('missing', dict(use_lare_path=True, use_pretrained_lare_path=True,
                      pretrained_lare_path_model_name='DOES_NOT_EXIST')),
                 ('broken', dict(use_lare_path=True, use_pretrained_lare_path=True,
                      pretrained_lare_path_model_name='BROKEN'))]:
    try:
        env = gym.make('drp_env:drp_safe-3agent_map_5x4-v2', **base, **kw)
        print(name, 'OK  use_lare_path =', env.unwrapped.use_lare_path)
    except Exception as e:
        print(name, 'STOP', type(e).__name__)
" 2>&1 | grep -E 'OK|STOP'
rm src/lare/path/models/BROKEN.pth
```

期待される出力 (修正前は missing と broken も `OK` になり、missing は `use_lare_path = False` になる):

```text
no LaRe OK  use_lare_path = False
scratch OK  use_lare_path = True
pretrained ok OK  use_lare_path = True
missing STOP FileNotFoundError
broken STOP ValueError
```

`BROKEN.pth` は確認用。**最後の `rm` を忘れない** (models/ は git に上げるフォルダ)。

| 症状 | 原因 |
|---|---|
| missing が `OK  use_lare_path = False` のまま | (a) を `except Exception as e:` より**下**に入れた |
| broken が `OK` のまま (「falling back to scratch」が出る) | (b) の print を残したまま raise を足していない / (b) を入れていない |
| `NameError: name 'resolved' is not defined` | (b) を `_load_lare_path_weights` 以外の場所に入れた (`resolved` はこのメソッドの中の変数) |
| `TabError` / `IndentationError` | drp_env.py は字下げがタブ。スペースで入れた |

---

## 5. 確認方法

複製に入れて確認済み (2026-10-15)。入力したあと、同じものを本体で回す。

```bash
cd ~/LDRP
python tools/lare_factor_smoke.py . 2 1    # 版 2、LaRe あり
python tools/lare_factor_smoke.py . 1 1    # 版 1、LaRe あり
python tools/lare_factor_smoke.py . 0 0    # LaRe なし
```

期待される出力 (8x5 の 5 台、200 ステップ × 2 エピソード、機体は最短路で動く):

```text
version 2 lare True reward_sum -2833.2980 completion 3
events pick/deliver [18. 12.]
  deliver   mean  0.006  min   0.00  max   1.00
  pickup    mean  0.009  min   0.00  max   1.00
  progress  mean  0.256  min   0.00  max   1.00
  route     mean -0.139  min  -0.66  max   0.00
  override  mean -0.357  min  -1.00  max   0.00
  stall     mean -0.266  min  -1.00  max   0.00
  delay     mean -0.226  min  -0.70  max   0.00
  prox      mean -0.039  min  -0.66  max  -0.00
  blocker   mean -0.002  min  -0.49  max   0.00
  collision mean  0.000  min   0.00  max   0.00
version 1 lare True reward_sum -2826.5957 completion 3
version 0 lare False reward_sum -35595.0000 completion 3
```

- **版 1 と LaRe なしは、変更前と同じ値**であることを確かめた (変更前の本体でも `-2826.5957` / `-35595.0000`)。今までの学習には影響しない
- 版 2 では、ピックアップ (18 回)・配達 (12 回) の印が付き、目的地への進み (progress) が正、止められた・止まっているがマイナスになっている
- 版 2 の報酬の合計の数値は、デコーダの初期値と学習で決まるので意味は持たない (動いたことの確認だけ)。2026-11-02 に今のコードで確認し直した値 (10-15 の確認では -2819.5551。その後の LaRe モジュールの変更で変わった。因子の値は同じ)
- 版 2 で事前学習のデコーダを読ませると、`factor_version mismatch: model=1, current=2` の警告が出てスクラッチになる (§4.5 を入れたあとは `ValueError` で止まる)

---

## 6. よくあるつまずき

| 症状 | 原因 |
|---|---|
| `ModuleNotFoundError: No module named 'src.lare.path.factors_ldrp_v1'` | §3.1 のスクリプトを `~/LDRP` 以外の場所で実行した (ファイルが別の場所にできた) |
| `ValueError: unknown lare_path_factor_version: 3` | 登録表に無い番号を train.py に書いた |
| `AttributeError: 'DrpEnv' object has no attribute '_lare_events'` | (f) を `reset` に入れていない、または (g) を入れずに `reset` より前に step した |
| 版 2 で override の因子がいつも 0 | (e) の `self._lare_override = override` を `super().step` より後に書いた、または SafeEnv でない env (drp-…) を使っている |
| 全機体に同時に配達の印が付く | (f)・(g) を `[[0, 0]] * self.agent_num` と書いた (内包表記にする) |
| `IndentationError` / `TabError` | drp_env.py・safe_marl.py は字下げがタブ。スペースで入れた |

---

## 7. 注意

- **既存の曖昧さ (今回は直さない)**: LaRe-Path をスクラッチ (版 1) で学習した run は、ダッシュボードの setting も評価用モデルのタグも `safe` で、LaRe なしの run と区別されない。版 2 は `safe F2` / `safe_f2` で区別される
- 版 2 の観測は 1 ステップごとに機体数ぶん作るので、版 1 より少し重い (機体 10 台で、1 ステップ数ミリ秒程度の見込み)。学習時間への影響は、最初の学習の速さで確かめる
- 適用したら、MANUAL.md の更新履歴に「`lare_path_factor_version` の新設」と「LaRe の保存ファイル名に `F2_` が入る」を書く (保存ファイル名の規則の変更なので、CLAUDE.md の不変条件の「保存ファイル命名」にも追記する)

---

## 8. 版 3 (F3 = LLM 評価関数 v2) を足す

2026-11-02 追加。**§3〜§4.4 (版 2) を入れたあとに入れる**。版 3 は版 2 の仕組み (登録表・`_lare_override`・ピックアップ / 配達の印・割当時刻) をそのまま使い、足りない記録だけを足す。

### 8.1 何をしたいか

プロンプト v2 ([lare_ldrp_prompt_v2.txt](prompts/lare_ldrp_prompt_v2.txt)) から LLM が作った評価関数 ([v2/summarized.json](prompts/responses/v2/summarized.json)) を、`lare_path_factor_version=3` で使えるようにする。版 2 との違いは 2 つ。

| 違い | 版 2 | 版 3 |
|---|---|---|
| 観測 | 19 区間、長さ `6N + 2NK + 4K + 21` | 長さ `7N + 4NK + 6K + 26`。版 2 の要望 (RequestedInformation) を入れた ([lare_ldrp_prompt.md §4.8](lare_ldrp_prompt.md)) |
| 評価関数の形 | `evaluation_func(observation, apsp)` | `evaluation_func(observation, apsp, adj)` (`adj` = 辺の長さの N×N 表) |

版 3 で観測に増える値と、どこから取るか:

| 観測の値 | 取り出し元 | 新しく記録するか |
|---|---|---|
| `next_node` / `others_next` (このステップで向かったノード) | env の `step` で、動いた機体の行き先 | **する** (`_lare_next`) |
| `ideal_task_length` (割当時の位置 → ピックアップ → 配達先 の最短の道のり) | 割当の処理 | **する** (`_lare_ideal_len`) |
| `dropped_this_step` (このステップで捨てたタスクの数) | タスクを捨てる処理 | **する** (`_lare_dropped_step`) |
| `others_dest` (他機の目的地) | `goal_array` (待機中は全部 0) | しない |
| `others_override` (他機が止められたか) | 版 2 の `_lare_override` | しない |
| `others_wait` (他機の待機回数) | `wait_count` | しない |
| `delivered_total` / `dropped_total` | `task_completion` / `task_dropped_count` | しない |
| `arrival_rate` (meta の 7 個目) | `task_arrival` などから計算 (env の `info["arrival_density"]` と同じ定義。fixed は 1) | しない |

### 8.2 全体の仕組み

```text
train.py   env_args.lare_path_factor_version=3
   ↓
drp_env.py   版 2 の記録 + 版 3 の記録 (8.6)
   ├─ reset:   _lare_next / _lare_ideal_len / _lare_dropped_step を作る
   ├─ step の初め:      _lare_next と _lare_dropped_step を消す
   ├─ 動いた機体:       _lare_next[i] = 向かったノード
   ├─ タスクを捨てた:   _lare_dropped_step += 1
   └─ 新しいタスクの割当: _lare_ideal_len[i] = _lare_ideal_task_length(i, task)  (新しいメソッド)
lare_path_module.py (8.5)
   ├─ __init__:   版が needs_adj なら self.adj = compute_adj(env)  (新しいメソッド)
   └─ compute_factors: evaluate(obs, apsp, adj) で呼ぶ
factor_versions.py (8.4)   3 → obs_ldrp_v2.build_ldrp_v2_obs_for_agent + factors_ldrp_v2.evaluation_func
obs_ldrp_v2.py (8.3, 新規)  版 3 の観測を作る
factors_ldrp_v2.py (8.3, 新規)  LLM の evaluation_func をそのまま置く
```

safe_marl.py は変えない (版 2 の `_lare_override` を他機の分にも使う)。保存ファイル名の `F3_` も、版 2 で入れた `_lare_build_save_filename` の変更がそのまま付ける。

### 8.3 新しく作るファイル (2 つ)

**(1) `src/lare/path/factors_ldrp_v2.py`** — §3.1 と同じく、応答の JSON からスクリプトで書き出す (人の手を加えない)。

```bash
cd ~/LDRP
/opt/anaconda3/envs/ldrp/bin/python - <<'EOF'
import json
fn = json.load(open("design/prompts/responses/v2/summarized.json"))["Functions"]
hdr = '''"""LaRe-Path 評価関数 v2 (LDRP 用, LLM が生成). 版 3 で使う.

LLM (Claude Opus 5.5, 思考の強さ: 高) がプロンプト v2 から生成した 3 候補を、同じ LLM が
1 つにまとめたもの。人の手は加えていない (Functions の文字列をそのまま貼った)。
  プロンプト:   design/prompts/lare_ldrp_prompt_v2.txt
  元の応答:     design/prompts/responses/v2/summarized.json
  生成の記録:   design/prompts/responses/v2/README.md
観測の並びは obs_ldrp_v2.build_ldrp_v2_obs_for_agent (長さ 7N + 4NK + 6K + 26) を前提にしている。
関数の形は evaluation_func(observation, apsp, adj)。
返す因子 (10 個): progress, remaining, event, delay, stall, safety, proximity, conflict, block, backlog
"""

'''
open("src/lare/path/factors_ldrp_v2.py", "w").write(hdr + fn)
print("written")
EOF
```

**(2) `src/lare/path/obs_ldrp_v2.py`** — 版 3 の観測を作る。並びはプロンプト v2 の OBSERVATION FORM と 1 対 1 (評価関数はこの並びを前提に添字を計算している)。

```python
"""LaRe-Path 用の観測 (LDRP 版 v2, 版 3 で使う) を機体ごとに作る.

並びは design/prompts/lare_ldrp_prompt_v2.txt の OBSERVATION FORM と同じ。
評価関数 (factors_ldrp_v2.evaluation_func) はこの並びを前提に書かれている。
並びを変えるときはプロンプトも変え、評価関数を生成し直すこと。
長さは 7N + 4NK + 6K + 26 (K = A - 1)。

    自分:   [own_prev N][own_curr N][dest N][next_node N][phase 3][delivery N][events 2]
            [steps_since_assign 1][task_age 1][active 1][ideal_task_length 1]
    他の機体: [others_pos 2NK][others_phase 3K][others_active K][others_dest NK]
            [others_next NK][others_override K][others_wait K]
    全体:   [demand 2][collision_distance 1][collision_info 2][wait_count 1][safety_override 1]
            [dropped_this_step 1][delivered_total 1][dropped_total 1]
            [node_coords 2N][meta 7 = diameter, N, A, T, C, t, arrival_rate]

版 2 (obs_ldrp.py) の記録に加えて、env 側で持つ値 (drp_env.py で記録する):
    _lare_next[i]         = このステップで向かったノード (SafeEnv の後)。止まったら None
    _lare_ideal_len[i]    = 今のタスクを割り当てた時点の位置 → ピックアップ → 配達先 の最短の道のり
    _lare_dropped_step    = このステップで捨てたタスクの数
"""

import numpy as np

from .encoder import get_node_coordinates_flat
from .obs_ldrp import _phase_of, _is_active


def _arrival_rate(env):
    """1 ステップあたりの期待到着数 (env の info["arrival_density"] と同じ定義)."""
    arrival = getattr(env, "task_arrival", "fixed")
    if arrival == "mmpp":
        return float((env.task_p_high + env.task_p_low) / 2.0)
    if arrival == "bernoulli":
        return float(env.task_density)
    return 1.0


def build_ldrp_v2_obs_for_agent(env, agent_id, graph_diameter,
                                prev_onehot_position, current_colliding_pairs):
    n = env.n_nodes
    A = env.agent_num
    eye3 = np.eye(3)
    override = getattr(env, "_lare_override", None)

    def curr_onehot(i):
        oh = np.asarray(env.obs_onehot[i]).flatten()
        return oh[:n].copy() if oh.size >= n else np.zeros(n)

    def prev_onehot(i):
        if prev_onehot_position is not None:
            return np.asarray(prev_onehot_position[i], dtype=float).copy()
        return curr_onehot(i)

    def node_onehot(node):
        v = np.zeros(n)
        if node is not None and 0 <= int(node) < n:
            v[int(node)] = 1.0
        return v

    def dest_of(i):
        return node_onehot(env.goal_array[i]) if env.assigned_tasks[i] else np.zeros(n)

    def over_of(i):
        return float(override[i]) if override is not None else 0.0

    i = agent_id
    task = env.assigned_tasks[i]
    phase = _phase_of(env, i)
    step = float(env.step_account)

    delivery = node_onehot(task[1]) if task else np.zeros(n)
    events = np.asarray(env._lare_events[i], dtype=float)
    if task:
        since = max(0.0, step - float(env._lare_assign_step[i]))
        age = max(0.0, step - float(env._lare_task_created[i]))
        ideal = float(env._lare_ideal_len[i])
    else:
        since, age, ideal = 0.0, 0.0, 0.0
    active = 1.0 if _is_active(env, i) else 0.0

    others = [j for j in range(A) if j != i]
    others_pos = []
    for j in others:
        others_pos += [prev_onehot(j), curr_onehot(j)]
    others_phase = [eye3[_phase_of(env, j)] for j in others]
    others_active = [1.0 if _is_active(env, j) else 0.0 for j in others]
    others_dest = [dest_of(j) for j in others]
    others_next = [node_onehot(env._lare_next[j]) for j in others]
    others_over = [over_of(j) for j in others]
    others_wait = [float(env.wait_count[j]) for j in others]

    def cat(parts):
        return np.concatenate(parts) if parts else np.zeros(0)

    waiting = float(sum(1 for v in env.assigned_list if v == -1))
    idle = float(sum(1 for j in range(A) if _is_active(env, j) and not env.assigned_tasks[j]))

    self_coll, others_coll = 0.0, 0.0
    for pair in current_colliding_pairs or []:
        if i in pair:
            self_coll = 1.0
        else:
            others_coll = 1.0

    meta = np.array([float(graph_diameter), float(n), float(A), float(env.time_limit),
                     float(env.task_num), step, _arrival_rate(env)])

    return np.concatenate([
        prev_onehot(i), curr_onehot(i), dest_of(i), node_onehot(env._lare_next[i]),
        eye3[phase], delivery, events, [since, age, active, ideal],
        cat(others_pos), cat(others_phase), np.asarray(others_active, dtype=float),
        cat(others_dest), cat(others_next),
        np.asarray(others_over, dtype=float), np.asarray(others_wait, dtype=float),
        [waiting, idle, float(env.colli_distan_value), self_coll, others_coll,
         float(env.wait_count[i]), over_of(i),
         float(env._lare_dropped_step), float(env.task_completion), float(env.task_dropped_count)],
        get_node_coordinates_flat(env), meta,
    ]).astype(np.float64)
```

| 行 | Python としての意味 | ここでの役割 |
|---|---|---|
| `from .obs_ldrp import _phase_of, _is_active` | 別のファイルの関数を取り込む (名前の先頭の `_` は「内部用」の目印で、取り込みは普通にできる) | 段階 (待機中 / ピックアップへ / 配達へ) と稼働中かの判定は版 2 と同じなので、書き写さずに使う |
| `getattr(env, "task_arrival", "fixed")` | 属性が無ければ `"fixed"` | 到着の設定を持たない古い env でも止まらない。`fixed` は毎ステップ 1 件なので 1 |
| `[j for j in range(A) if j != i]` | 内包表記で、自分以外の機体の番号のリストを作る | 他の機体の部分は 7 種類あり、どれも同じ順番 (k = 0, …, K-1) で並べる必要がある。番号のリストを 1 回作って全部に使う |
| `def cat(parts): ... np.zeros(0)` | 空のリストを `np.concatenate` に渡すと例外で止まるので、長さ 0 の配列を返す | 機体が 1 台 (K = 0) のとき他の機体の部分が空になる |
| `node_onehot(env._lare_next[j])` | `None` なら全部 0 の配列 (`node_onehot` の中で判定) | 止まった機体の `next_node` は全部 0 (プロンプトの定義) |
| `float(env.task_completion)` / `float(env.task_dropped_count)` | — | エピソードの初めからの累計の配達数・捨てた数。env が既に数えている |
| `np.concatenate([...])` の順番 | — | プロンプト v2 の並び (自分 → 他の機体 → 全体 → 座標 → meta) と同じ順。順番を変えると評価関数が別の値を読む |

**待機中の機体の `dest` を全部 0 にする理由**は版 2 と同じ (§3.2 の末尾)。他の機体の `others_dest` も同じ扱いにする (`dest_of`)。

### 8.4 `src/lare/path/factor_versions.py` (2 か所)

**何をしたいか:** 版 3 を登録表に足す。評価関数が `adj` を取ることを、`"needs_adj": True` で登録表に書いておく (モジュール側はこの印を見て `adj` を渡す)。

**(a) 取り込み** — `from .factors_ldrp_v1 import evaluation_func as _eval_ldrp_v1` **の次**に 2 行:

```python
from .obs_ldrp_v2 import build_ldrp_v2_obs_for_agent
from .factors_ldrp_v2 import evaluation_func as _eval_ldrp_v2
```

**(b) 登録** — 版 2 の登録 `        "build_obs": build_ldrp_obs_for_agent, "evaluate": _eval_ldrp_v1},` **の次**に 3 行:

```python
    # 版 3 の評価関数は 3 つ目の引数 adj (辺の長さの N×N 表) を取る (needs_adj)
    3: {"name": "ldrp_llm_v2", "n_factors": 10, "needs_adj": True,
        "build_obs": build_ldrp_v2_obs_for_agent, "evaluate": _eval_ldrp_v2},
```

入れたあとの形:

```python
from .encoder import build_lare_obs_for_agent, evaluation_func as _eval_drp10
from .obs_ldrp import build_ldrp_obs_for_agent
from .factors_ldrp_v1 import evaluation_func as _eval_ldrp_v1
from .obs_ldrp_v2 import build_ldrp_v2_obs_for_agent
from .factors_ldrp_v2 import evaluation_func as _eval_ldrp_v2

FACTOR_VERSIONS = {
    1: {"name": "drp10", "n_factors": 10,
        "build_obs": build_lare_obs_for_agent, "evaluate": _eval_drp10},
    2: {"name": "ldrp_llm_v1", "n_factors": 10,
        "build_obs": build_ldrp_obs_for_agent, "evaluate": _eval_ldrp_v1},
    # 版 3 の評価関数は 3 つ目の引数 adj (辺の長さの N×N 表) を取る (needs_adj)
    3: {"name": "ldrp_llm_v2", "n_factors": 10, "needs_adj": True,
        "build_obs": build_ldrp_v2_obs_for_agent, "evaluate": _eval_ldrp_v2},
}
```

ファイル冒頭の docstring の版の一覧にも `3: LDRP 用の 10 因子 (LLM が生成, design/prompts/responses/v2)` を 1 行足しておく (動きには関係しない)。

### 8.5 `src/lare/path/lare_path_module.py` (3 か所)

**何をしたいか:** 版 3 の評価関数に `adj` を渡す。`adj` は地図ごとに 1 回作れば足りるので、`apsp` と同じく `__init__` で作って持っておく。

| # | 場所 (メソッド / 概算行) | 目印にする既存コード | 内容 |
|---|---|---|---|
| (a) | `__init__` (~85 行) | 版 2 の §4.1 (c) で入れた `self.factor_dim = self._factor_spec["n_factors"]` **の次** | `self.adj` を作る |
| (b) | `compute_factors` (~205 行) | 版 2 の §4.1 (d) で入れた `factor_list = self._factor_spec["evaluate"](batch_obs, self.apsp)` **を置き換え** | `adj` があれば渡す |
| (c) | `compute_apsp` の前 (~143 行) | `    def compute_apsp(self, env):` **の直前** | 新しいメソッド `compute_adj` |

**(a) 追加するコード / 入れたあとの形**

```python
        self.factor_version = int(self.cfg.factor_version)
        self._factor_spec = get_factor_version(self.factor_version)
        self.factor_dim = self._factor_spec["n_factors"]
        # 版 3 以降の評価関数は辺の長さの表 (adj) も使う
        self.adj = self.compute_adj(env) if self._factor_spec.get("needs_adj") else None
        self.n_agents = env.agent_num
```

**(b) 置き換えるコード / 入れたあとの形**

```python
        batch_obs = np.stack(rows, axis=0)
        if self.adj is not None:
            factor_list = self._factor_spec["evaluate"](batch_obs, self.apsp, self.adj)
        else:
            factor_list = self._factor_spec["evaluate"](batch_obs, self.apsp)
        factor_arr = np.concatenate(factor_list, axis=-1).astype(np.float32)
```

**(c) 追加するコード** (`compute_apsp` の直前。前後に空行 1 行)

```python
    def compute_adj(self, env):
        """辺の長さの表. adj[u, v] = 辺 (u, v) の長さ (ノード座標の距離)、辺が無ければ 0.

        compute_apsp と同じく、辺の長さはノード座標のユークリッド距離で測る。
        """
        n_nodes = len(env.G.nodes())
        adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
        for u, v in env.G.edges():
            w = float(np.linalg.norm(np.array(env.pos[u]) - np.array(env.pos[v])))
            adj[u, v] = w
            adj[v, u] = w
        return adj

    def compute_apsp(self, env):
```

| 行 | Python としての意味 | ここでの役割 |
|---|---|---|
| `self._factor_spec.get("needs_adj")` | 辞書の `.get` はキーが無ければ `None` (偽) を返し、止まらない | 版 1・2 の登録には `needs_adj` が無いので `None` → `self.adj = None`。`[]` で引くと版 1・2 で `KeyError` |
| `A if 条件 else B` | 条件式 | 版 1・2 では `adj` を作らない (作っても使わないので) |
| `if self.adj is not None:` | `None` かどうかの判定 | 版 1・2 は今までどおり 2 引数で呼ぶ。版 1・2 の評価関数に 3 つ目を渡すと `TypeError` で止まる |
| `env.G.edges()` | networkx のグラフの辺 (両端のノードの組) を 1 本ずつ返す | 無向グラフなので 1 本の辺は 1 回しか出ない。`adj[u, v]` と `adj[v, u]` の両方に書く |
| `np.linalg.norm(...)` | ベクトルの長さ (ユークリッド距離) | `compute_apsp` と同じ測り方にする。env の `get_path_length` もこの長さと一致する (8x5・aoba00 で差 0 を確認) |

### 8.6 `src/main/drp_env/drp_env.py` (6 か所、字下げはタブ)

| # | 場所 (メソッド / 概算行) | 目印にする既存コード | 内容 |
|---|---|---|---|
| (k) | `reset` (~893 行) | 版 2 の §4.2 (f) で入れた `self._lare_task_created = [0] * self.agent_num` の行 **の次** | 版 3 の記録用の 3 つを作る |
| (l) | `step` の初め (~1006 行) | 版 2 の §4.2 (g) で入れた `self._lare_events = [[0, 0] for _ in range(self.agent_num)]` (`# LaRe-Path: snapshot per-agent onehot positions BEFORE movement.` の直前の行) **の次** | 向かったノード・捨てた数を毎ステップ消す |
| (m) | `step` の移動の処理 (~1062 行) | `self.current_goal_prepare[i] = joint_action[i] #update 行き先ノード when avable action is taken` **の次** | 向かったノード |
| (n) | `step` のタスクの追加 (~1191 行) | `self.task_dropped_count += 1` **の次** | 捨てた数 |
| (o) | `step` の割当の処理 (~1258 行) | 版 2 の §4.2 (i) で入れた `self._lare_task_created[i] = creation_step` **の次** | 理想の道のり |
| (p) | `get_path_length` の次 (~1556 行) | `	def get_near_nodes(self, node_num):` **の直前** | 新しいメソッド `_lare_ideal_task_length` |

**(k) 入れたあとの形**

```python
		self._lare_assign_step = [0] * self.agent_num			# 今のタスクを割り当てたステップ
		self._lare_task_created = [0] * self.agent_num			# 今のタスクが発生したステップ
		# 版 3 の観測用
		self._lare_next = [None] * self.agent_num			# このステップで向かったノード (止まったら None)
		self._lare_ideal_len = [0.0] * self.agent_num		# 割当時の位置 → ピックアップ → 配達先 の最短の道のり
		self._lare_dropped_step = 0					# このステップで捨てたタスクの数
```

**(l) 入れたあとの形**

```python
		# LaRe-Path の観測用: このステップのピックアップ・配達の印を消す
		self._lare_events = [[0, 0] for _ in range(self.agent_num)]
		self._lare_next = [None] * self.agent_num
		self._lare_dropped_step = 0
		# LaRe-Path: snapshot per-agent onehot positions BEFORE movement.
		if self.use_lare_path and self.lare_path_module is not None:
```

**(m) 入れたあとの形**

```python
			else:
				#self.joint_action_old[i] = joint_action[i]
				self.current_goal_prepare[i] = joint_action[i] #update 行き先ノード when avable action is taken
				self._lare_next[i] = int(joint_action[i])	# LaRe-Path の観測用 (版 3)
				obs_i = self.obs[i]
				self.wait_count[i] = 0
```

**(n) 入れたあとの形**

```python
					self.task_arrival_count += 1
				else:
					self.task_dropped_count += 1
					self._lare_dropped_step += 1
```

**(o) 入れたあとの形**

```python
					if self._lare_task_key[i] != tuple(task_r):
						self._lare_task_key[i] = tuple(task_r)
						self._lare_assign_step[i] = self.step_account
						self._lare_task_created[i] = creation_step
						self._lare_ideal_len[i] = self._lare_ideal_task_length(i, task_r)
```

**(p) 追加するコード** (`get_near_nodes` の直前。後ろに空行 1 行)

```python
	def _lare_ideal_task_length(self, i, task):
		"""今の位置 → ピックアップ → 配達先 の最短の道のり (LaRe-Path の観測 版 3 用).

		辺の途中にいるときは、向かっているノード (current_goal) を経由する。
		"""
		pick, drop = int(task[0]), int(task[1])
		if self.current_goal[i] is None:
			to_pick = self.get_path_length(int(self.current_start[i]), pick)
		else:
			g = int(self.current_goal[i])
			here = np.array(self.obs[i][:2], dtype=float)
			to_pick = float(np.linalg.norm(np.array(self.pos[g], dtype=float) - here)) + self.get_path_length(g, pick)
		return float(to_pick + self.get_path_length(pick, drop))

```

| 行 | Python としての意味 | ここでの役割 |
|---|---|---|
| `[None] * self.agent_num` | 同じ値を機体の数だけ並べたリスト | `None` や数は変更できない値なので `*` で作ってよい (版 2 の `[[0, 0]] * n` のような共有の問題は起きない) |
| (l) で毎ステップ消す | — | 止まった機体の `_lare_next` を前のステップの値のまま残さない。捨てた数も「このステップの分」だけにする |
| `self._lare_next[i] = int(joint_action[i])` | — | この分岐は「動ける行動を選んだ」ときだけ通る (止まる・動けない行動は前の 2 つの分岐)。SafeEnv が止めた機体は、ここに来る前に行動が「今いるノード」に書き換えられているので通らない |
| `if self._lare_task_key[i] != tuple(task_r):` の中に入れる | — | 再割当ありでは同じタスクが毎ステップ割り当て直されるので、新しいタスクのときだけ理想の道のりを計算する (割当時刻と同じ扱い) |
| `self.obs[i][:2]` | 先頭の 2 つ (x, y 座標) を取り出すスライス | 割当の処理はこのステップの移動の後なので、移動後の位置 |
| `self.current_goal[i] is None` | — | ノードの上にいる (辺の途中ではない)。辺の途中にいるときは、向かっているノードまでの残りの距離を足す (途中で引き返さない前提) |
| `self.get_path_length(a, b)` | env の既存のメソッド (最短路の長さ) | `a == b` のとき 0 を返す。LaRe の `apsp` と同じ値 |

### 8.7 train.py

train.py の冒頭の変数 `LARE_FACTOR_VERSION` (2026-11-02 に追加。版 1 のときは引数を渡さない) を 3 にする。使う版が決まったら、この仕組みは直す予定。

```python
LARE_FACTOR_VERSION = 3
```

- 版 2 と同じく、**版 3 の学習済みデコーダはまだ無い**。LaRe モデルの学習 (B) はスクラッチ (`use_pretrained_lare_path=False`、`use_finetuning_lare_path=False`) で回す
- LaRe なし (`use_lare_path=False`) の run は版を書いても使われない (ダッシュボードも版 1 として扱う)

### 8.8 確認方法

複製 (版 2 を入れたあとのコード) に入れて確認済み (2026-11-02)。入力したあと、同じものを本体で回す。

```bash
cd ~/LDRP
python tools/lare_factor_smoke.py . 3 1    # 版 3、LaRe あり
python tools/lare_factor_smoke.py . 2 1    # 版 2 (版 3 を足しても変わらないこと)
python tools/lare_factor_smoke.py . 1 1    # 版 1
python tools/lare_factor_smoke.py . 1 0    # LaRe なし
```

期待される出力 (警告の行は除く):

```text
version 3 lare True reward_sum -2793.8430 completion 3
events pick/deliver [18. 12.]
obs length 970 (expected 7N+4NK+6K+26 = 970)
moved agent-steps 612  dropped 4 (env total 4)  ideal_len mean 165.6 max 253.7
  progress  mean  0.255  min   0.00  max   1.00
  remaining mean  0.139  min   0.00  max   0.66
  event     mean  0.010  min   0.00  max   1.00
  delay     mean  0.313  min   0.00  max   1.00
  stall     mean  0.340  min   0.00  max   1.00
  safety    mean  0.357  min   0.00  max   1.00
  prox      mean  0.039  min   0.00  max   0.66
  conflict  mean  0.371  min   0.00  max   1.00
  block     mean  0.085  min   0.00  max   0.75
  backlog   mean  0.178  min   0.00  max   1.00
version 2 lare True reward_sum -2833.2980 completion 3
version 1 lare True reward_sum -2826.5957 completion 3
version 1 lare False reward_sum -35595.0000 completion 3
```

- **版 2・版 1・LaRe なしは版 3 を足す前と同じ値**。版 3 の記録を足しても、今までの学習には影響しない
- `obs length` が式の値と一致する (8x5: N = 40、5 台なので K = 4)
- 版 3 の 10 因子はどれも 0〜1 に収まる (progress だけ −1〜1)。版 1 の因子のように生の距離や回数がそのまま出ることはない
- 報酬の合計 (`reward_sum`) はデコーダの初期値と学習で決まるので、値そのものに意味は無い (同じコードなら毎回同じになることだけ見る)

新しく足した記録が意味どおりかは、次の場面で確かめた (複製、8x5 の 5 台、200 ステップ):

| 確かめたこと | 結果 |
|---|---|
| `_lare_next` は、前のステップの位置から隣のノード (または辺の続き) を指すか | 246 回の移動すべてで成り立つ |
| 割り当てた直後のステップで、版 3 の delay (理想の時間からの超過) は 0 か | 10 回すべて 0 (理想の道のり = 残りの道のり) |
| `_lare_dropped_step` を毎ステップ足すと、env の `task_dropped_count` と一致するか | 41 = 41 |
| meta の `arrival_rate` | bernoulli の p = 0.3 のとき 0.3 |
| `adj` | 対称で、0 でない組の数がグラフの辺の数と一致 (67) |

### 8.9 よくあるつまずき

| 症状 | 原因 |
|---|---|
| `AttributeError: 'DrpEnv' object has no attribute '_lare_next'` | (k) を `reset` に入れていない |
| `AttributeError: ... '_lare_ideal_task_length'` | (p) のメソッドを入れていない、またはクラスの外 (字下げが浅い) に入れた |
| `TypeError: evaluation_func() missing 1 required positional argument: 'adj'` | 8.5 (b) を入れていない (版 3 に 2 引数で渡している) |
| `TypeError: evaluation_func() takes 2 positional arguments but 3 were given` | 8.5 (a) の `if ... else None` を書かずに、版 1・2 でも `adj` を作っている |
| `obs length` が式と合わない | obs_ldrp_v2.py の `np.concatenate` の並びを変えた / 足りない |
| `dropped` がいつも 0 なのに env total が増える | (n) を `self.task_dropped_count += 1` 以外の場所に入れた、または (l) を (n) より後で消している |
| `ideal_len` がいつも 0 | (o) を `if self._lare_task_key[i] != tuple(task_r):` の外 (字下げが浅い) に入れた、または版 2 の (i) を入れていない |
| `TabError` / `IndentationError` | drp_env.py は字下げがタブ。lare_path_module.py・factor_versions.py はスペース 4 個 |

### 8.10 注意

- **版 2 が先**。版 3 は、版 2 で入れる登録表・`_lare_override`・`_lare_events`・`_lare_assign_step`・`_lare_task_created`・`_lare_task_key` を使う
- **`ideal_task_length` は「ノードで止まる」を数えていない**: env はノードに着くとそのステップはそこで止まるので、実際の最短のステップ数は `ideal_task_length / 5` より多い。評価関数の delay はその差も「超過」として数える (v2 の RequestedInformation でも指摘されている)。版 3 では評価関数を変えない (人の手を加えない) ので、このまま使う
- **`arrival_rate` は仮に入れた値** ([lare_ldrp_prompt.md §4.8](lare_ldrp_prompt.md))。v2 の評価関数は読んでいない (meta から読むのは直径・N・A・C だけ) ので、因子には効かない
- 観測が長くなる (8x5 の 5 台で 970、版 2 は 505) が、デコーダに入るのは 10 因子だけなので、学習の重さはほぼ変わらない
- 適用したら、MANUAL.md の更新履歴に「`lare_path_factor_version=3` を追加」を書く
