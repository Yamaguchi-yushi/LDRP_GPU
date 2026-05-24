# 設計書: 研究室配布用 LDRP クリーン版 (LaRe 統合なし)

**作成日:** 2026-05-19
**対象リポジトリ:** [kaji-ou/LDRP](https://github.com/kaji-ou/LDRP) (= LaRe 統合前のオリジナル LDRP)
**配布目的:** 研究室メンバが clone してすぐ動かせる状態にする
**読者:** **kaji-ou/LDRP を clone した後の作業ディレクトリで起動した Claude Code**

---

## 0. このドキュメントの位置付け

このドキュメントは **別の Claude Code セッション** に渡すための指示書です。手順:

1. ユーザが `git clone https://github.com/kaji-ou/LDRP.git` で kaji-ou/LDRP を取得
2. ユーザが本 `distribution_for_lab.md` をその clone ディレクトリ直下に配置
3. ユーザがそのディレクトリで Claude Code を起動
4. ユーザが「distribution_for_lab.md の通り作業して」と指示
5. **その Claude Code が本書を読んで自走** で実装

→ よって本書は **kaji-ou/LDRP のファイル構造を前提** に書く。LaRe / ALMA / 我々の独自拡張は一切触れない。

---

## 1. 配布で実現したいこと

| 項目 | 元の状態 | 配布版で実現したい状態 |
|---|---|---|
| **SafeEnv の保護範囲** | 待機 agent が保護されない (= 衝突多発) | `pbs_mode` フラグで切替可能. **PBS 利用時のみ True**, それ以外はデフォ False で **MARL 系で SafeEnv が機能** |
| **conda 環境セットアップ** | 手動で `conda create` + `pip install` を実行する必要あり | `./setup_env.sh` 1 コマンドで完了 |
| **依存バージョン** | requirements.txt なし → バージョン不整合が起きやすい | `requirements.txt` で動作確認済みバージョンを pin |
| **README** | セットアップ手順の記載不足 | 最初に `./setup_env.sh` を案内 |

---

## 2. 前提

- kaji-ou/LDRP が `git clone` 済みで、作業ディレクトリ直下に `src/`, `runner.py`, `test.py`, `train.py` がある
- conda (Anaconda / Miniconda) がインストール済み
- macOS または Linux を想定 (Windows は WSL2 経由)

---

## 3. 作業手順

以下を順番に実施。各 Step は独立に検証可能。

### Step 1: `src/main/drp_env/drp_env.py` に `pbs_mode` フラグを追加

**目的**: SafeEnv の `if self.current_goal[i] == None:` ガードを待機 agent でも機能させる。
**背景**: 元の drp_env.py には待機分岐で `self.current_goal_prepare[i] = action_i` という代入があり、これが SafeEnv の保護をバイパスする原因になっている。PBS 用の修正だったが、MARL では不要なので **フラグで切替可能** にする。

#### 3.1 `__init__` シグネチャに `pbs_mode=False` を追加

`DrpEnv.__init__` の引数末尾 (= `task_list = None` の次あたり) に追加:

```python
def __init__(self,
        agent_num,
        speed,
        start_ori_array,
        goal_array,
        visu_delay,
        state_repre_flag,
        time_limit,
        collision,
        map_name="map_3x3",
        reward_list={"goal": 100, "collision": -10, "wait": -10, "move": -1},
        task_flag=False,
        task_list=None,
        # 追加: PBS 互換モード. True にすると待機分岐で current_goal を非 None に保つ
        # (PBS の path 計画で必要). False にすると SafeEnv の保護が待機 agent
        # にも効く. デフォルト False = 安全制御優先.
        pbs_mode=False,
        ):
```

その上で `__init__` の本体 (= フィールド初期化部分) のどこかに以下を追加:

```python
self.pbs_mode = bool(pbs_mode)
```

(配置場所目安: `self.agent_num = agent_num` のすぐ次)

#### 3.2 `step()` の待機分岐を条件分岐化

`DrpEnv.step()` 内の以下のブロックを探す:

```python
# if action_i is current start node -> stop
elif self.pos[int(action_i)][0]==self.obs[i][0] and self.pos[int(action_i)][1]==self.obs[i][1]:
    self.obs_prepare.append(self.obs_current_chache[i])
    self.wait_count[i] += 1
    #pbsのため，その場待機でもcurrent_goalをNoneのままでないように変更
    #従来のdrpは以下の行はなし
    self.current_goal_prepare[i] = action_i
```

最後の `self.current_goal_prepare[i] = action_i` 行を **`if self.pbs_mode:` でガード** する形に書き換え:

```python
# if action_i is current start node -> stop
elif self.pos[int(action_i)][0]==self.obs[i][0] and self.pos[int(action_i)][1]==self.obs[i][1]:
    self.obs_prepare.append(self.obs_current_chache[i])
    self.wait_count[i] += 1
    # pbs_mode=True: PBS が他 agent の待機予定を path 計画に反映できるよう
    #                current_goal を非 None (= 待機ノード) に保つ.
    # pbs_mode=False: SafeEnv の `if self.current_goal[i] == None:` ガードが
    #                 待機 agent でも機能し、同一目的地衝突を事前回避できる.
    # デフォルトは False (= 安全制御優先). test.py 側で path_planner=='pbs' を
    # 検出して自動で True に設定.
    if self.pbs_mode:
        self.current_goal_prepare[i] = action_i
```

#### 検証

```bash
python -c "import sys; sys.path.append('src/main'); from drp_env.drp_env import DrpEnv; import inspect; print('pbs_mode' in inspect.signature(DrpEnv.__init__).parameters)"
# → True が出れば OK
```

---

### Step 2: `test.py` で `path_planner` に応じて `pbs_mode` を自動セット

**目的**: ユーザが pbs_mode を意識せず、`path_planner` 指定だけで適切に切り替わる。

`test.py` の `gym.make(env_name, ...)` 呼び出しの直前で `path_planner` を見て `pbs_mode` を決定:

```python
# pbs_mode 自動判定: path_planner が "pbs" のときだけ True にする.
# PBS は待機 agent の予定も path 計画に反映するため current_goal を非 None に
# 保つ必要があるが、それ以外 (QMIX/IQL/VDN/MAA2C) では None のままにして
# SafeEnv の保護を機能させる.
pbs_mode = (getattr(config, "path_planner", "") == "pbs")

env = gym.make(
    env_name,
    state_repre_flag="onehot_fov",
    reward_list=reward_list,
    time_limit=config.time_limit,
    task_flag=True,
    task_list=None,
    pbs_mode=pbs_mode,   # ← 追加
)
```

#### 検証

```bash
python test.py map_8x5 4 qmix tp   # pbs_mode=False になる
python test.py map_8x5 4 pbs tp    # pbs_mode=True になる
```

両方ともクラッシュせず動けば OK。

---

### Step 3: `requirements.txt` をリポジトリ直下に作成

**目的**: 動作確認済みバージョンを固定して再現性を確保。

#### ファイル内容 (新規作成: `<repo_root>/requirements.txt`)

```text
# LDRP の主要依存パッケージ. Python 3.9 を前提.
# 詳細セットアップ手順: ./setup_env.sh または README.md を参照.

# --- LDRP 本体 (test.py 推論用) ---
gym==0.26.2
networkx==3.2.1
numpy==1.26.4
torch==2.8.0
PyYAML==6.0.3
matplotlib==3.8.2

# --- epymarl 経由の MARL 学習 (train.py 用) ---
# sacred:             epymarl の実験管理フレームワーク (main.py で必須)
# tensorboard-logger: epymarl の use_tensorboard=True (デフォルト) で必須
# einops:             PAC 系 learner が eager import するため qmix/iql/vdn でも必須
#                     (learners/__init__.py の REGISTRY で全 learner を eager 登録するため)
# 注意: PAC DCG critic は torch_scatter を要するが, critics/__init__.py の
#       register_pac_critics() 遅延ロードで分離 → qmix/iql/vdn では不要.
#       PAC DCG を実際に使うときだけ `pip install torch_scatter` する.
sacred
tensorboard-logger
einops
# tensorboard-logger 0.1.0 が古い protobuf API を使うため <4 にピン.
# 新しい protobuf (6 系) を入れると `Descriptors cannot be created directly` で死ぬ.
# upstream LDRP の src/epymarl/requirements.txt は `protobuf==3.6.1` で更に厳しくピン.
protobuf<4
# SMAC: epymarl の envs/__init__.py が `from smac.env import MultiAgentEnv` を
# eager import するため必須. SC2 実行ファイルは不要 (基底クラスとしてのみ利用).
# upstream LDRP の src/epymarl/requirements.txt も同じ git URL を指定.
git+https://github.com/oxwhirl/smac.git
```

---

### Step 4: `setup_env.sh` をリポジトリ直下に作成

**目的**: `./setup_env.sh` 1 コマンドで conda env "ldrp" を作成 + 依存インストール + 動作確認まで自動化。

#### ファイル内容 (新規作成: `<repo_root>/setup_env.sh`)

```bash
#!/usr/bin/env bash
#
# LDRP conda env セットアップスクリプト.
#
# 使い方:
#   ./setup_env.sh                    # ldrp env がなければ作成 + 依存をインストール
#   ./setup_env.sh --recreate         # 既存の ldrp env を削除してゼロから作り直す
#   LDRP_ENV_NAME=foo ./setup_env.sh  # env 名を foo にする
#
# 前提:
#   - anaconda / miniconda がインストール済みで `conda` コマンドが使える
#   - リポジトリのルートディレクトリで実行する (= setup_env.sh と同じディレクトリ)
#
set -euo pipefail

ENV_NAME="${LDRP_ENV_NAME:-ldrp}"
PYTHON_VERSION="3.9"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RECREATE=false
for arg in "$@"; do
    case "$arg" in
        --recreate) RECREATE=true ;;
        -h|--help)
            sed -n '2,12p' "${BASH_SOURCE[0]}"
            exit 0
            ;;
        *)
            echo "[setup] 不明な引数: $arg (use --help)" >&2
            exit 2
            ;;
    esac
done

# --- 1) conda の存在確認 -------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
    echo "[setup] ERROR: conda が見つかりません. anaconda / miniconda をインストールしてください." >&2
    exit 1
fi
CONDA_BASE="$(conda info --base)"
ENV_PYTHON="${CONDA_BASE}/envs/${ENV_NAME}/bin/python"

# --- 2) 既存 env のハンドリング ------------------------------------------------
env_exists() {
    conda env list 2>/dev/null | awk '{print $1}' | grep -qx "$1"
}

if env_exists "$ENV_NAME"; then
    if [ "$RECREATE" = "true" ]; then
        echo "[setup] 既存 env '${ENV_NAME}' を削除します..."
        conda env remove -n "$ENV_NAME" -y
    else
        echo "[setup] env '${ENV_NAME}' は既に存在します. (作り直したい場合は --recreate)"
    fi
fi

# --- 3) env 作成 ----------------------------------------------------------------
if ! env_exists "$ENV_NAME"; then
    echo "[setup] conda env '${ENV_NAME}' を作成中 (python ${PYTHON_VERSION})..."
    conda create -n "$ENV_NAME" "python=${PYTHON_VERSION}" -y
fi

# --- 4) pip の更新 + 依存インストール -----------------------------------------
echo "[setup] pip を更新中..."
"$ENV_PYTHON" -m pip install --upgrade pip --quiet

echo "[setup] 依存パッケージをインストール中 (requirements.txt)..."
"$ENV_PYTHON" -m pip install -r "${REPO_ROOT}/requirements.txt"

# --- 5) 編集モードで drp パッケージインストール -----------------------------
echo "[setup] ローカルの drp パッケージを editable モードでインストール中..."
"$ENV_PYTHON" -m pip install -e "${REPO_ROOT}/src/main" --quiet

# --- 6) 動作確認 ----------------------------------------------------------------
echo "[setup] 動作確認中..."
"$ENV_PYTHON" - <<EOF
import gym, numpy, torch, networkx, yaml
print(f"  gym      : {gym.__version__}")
print(f"  numpy    : {numpy.__version__}")
print(f"  torch    : {torch.__version__}")
print(f"  networkx : {networkx.__version__}")
print(f"  PyYAML   : {yaml.__version__}")

# epymarl 学習 (train.py) で必須の追加依存も確認
import sacred, einops
try:
    import tensorboard_logger as _tbl
    tbl_ver = getattr(_tbl, "__version__", "ok")
except Exception as e:
    tbl_ver = f"NG ({e})"
print(f"  sacred              : {sacred.__version__}")
print(f"  einops              : {einops.__version__}")
print(f"  tensorboard-logger  : {tbl_ver}")

import sys
sys.path.append("${REPO_ROOT}")
sys.path.append("${REPO_ROOT}/src/main")
import drp_env
print(f"  drp_env  : {drp_env.__file__}")
print("[setup] verification OK")
EOF

cat <<EOF

[setup] done.

env を activate するには:
  conda activate ${ENV_NAME}

env を activate せずに直接 python を使うには:
  ${ENV_PYTHON} test.py

env を作り直したい場合:
  ./setup_env.sh --recreate

EOF
```

#### 実行権限を付与

```bash
chmod +x setup_env.sh
```

---

### Step 5: README.md にセットアップ手順を追記

**目的**: clone した人がまず読む README に、最低限のセットアップ手順を明示。

`README.md` の冒頭または「## Installation」セクションに以下を追加 (既存内容は破壊しない):

```markdown
## Quick Start (新規セットアップ)

```bash
# 1. リポジトリを clone
git clone https://github.com/kaji-ou/LDRP.git
cd LDRP

# 2. conda env を 1 コマンドでセットアップ
./setup_env.sh

# 3. activate
conda activate ldrp

# 4. テスト実行
python test.py
```

### 前提

- anaconda / miniconda がインストール済み
- macOS または Linux (Windows は WSL2 経由)

### SafeEnv について

`drp_env/wrapper/safe_marl.py` の SafeEnv は MARL 系の path planner (QMIX, IQL, VDN, MAA2C) と組合せた時に **待機 agent も衝突回避保護対象** になります。PBS を使う場合は `test.py` が自動で `pbs_mode=True` に切替えて PBS の path 計画と整合させます。
```

---

### Step 6: 動作確認

すべての手順が完了したら、以下のスモークテストを実行:

```bash
# 1. env 作成
./setup_env.sh

# 2. test.py 単体実行 (QMIX + TP)
conda activate ldrp
python test.py map_8x5 4 qmix tp
# → Average task completion: <数値> が出力されれば OK
# → 衝突 (!!!collision!!! 出力) が前バージョンより減っているはず

# 3. test.py 単体実行 (PBS + TP)
python test.py map_8x5 4 pbs tp
# → クラッシュせず動作すれば OK (= pbs_mode=True が機能)

# 4. epymarl 学習が起動するか軽く確認 (= 早めに止める)
python train.py
# → "Trainer starting..." 等の epymarl ログが出始めれば OK. Ctrl+C で停止
```

---

## 4. 完了チェックリスト

研究室配布前に以下が全て満たされていることを確認:

- [ ] `src/main/drp_env/drp_env.py` に `pbs_mode` パラメータが追加されている
- [ ] `src/main/drp_env/drp_env.py` の待機分岐が `if self.pbs_mode:` でガードされている
- [ ] `test.py` で `pbs_mode = (config.path_planner == "pbs")` が `gym.make` に渡されている
- [ ] `requirements.txt` が repo root に存在し、上記内容と一致している
- [ ] `setup_env.sh` が repo root に存在し、実行権限が付与されている
- [ ] `README.md` に Quick Start セクションが追加されている
- [ ] `./setup_env.sh` がクリーン環境 (= 新規 anaconda) でエラーなく完走する
- [ ] `python test.py map_8x5 4 qmix tp` が動作する
- [ ] `python test.py map_8x5 4 pbs tp` が動作する

---

## 5. 配布作業の最終ステップ (= 受信側 Claude Code への指示ではない)

以下は **ユーザ自身** が手動で行うこと (受信側 Claude Code には委ねない):

1. このディレクトリで `git status` 確認 → 上記 Step 1-5 の変更が全て含まれていることを確認
2. 変更をコミット (例: `git commit -m "feat: pbs_mode flag + setup_env.sh + requirements.txt"`)
3. 研究室メンバ用の新しいリポジトリに push、または zip にして配布
4. README.md の「Quick Start」を実行してもらう

---

## 6. 注意事項

- **LaRe / ALMA / その他の独自拡張は含めない**: 本配布版の目的は「素の LDRP に SafeEnv 修正と環境セットアップだけ追加」。実装の複雑化は研究室メンバの学習コストを上げるので避ける
- **pbs_mode のデフォルトは False**: PBS を使うのは少数派なので、安全制御を有効にする側をデフォルトにする
- **requirements.txt のバージョンピン**: 2026-05 時点の動作確認済みバージョン。新しい anaconda を使うと別バージョンが入ろうとするので **pin したまま** にする
- **設計書 (本書) 自体は配布物に含めない**: 受信側 Claude Code が読んだら役目終了なので、配布リポジトリには含めず削除する

---

## 7. トラブルシューティング

### `./setup_env.sh` が「conda not found」で失敗

→ anaconda / miniconda をインストール後、シェルを開き直すか `source ~/.zshrc` (or `~/.bashrc`) で PATH 更新。

### `pip install` で torch のインストールに長時間かかる

→ M1/M2 Mac で 1-2 分かかるのは正常 (= MPS 対応版を取得中)。

### `python test.py` で `!!!collision!!! with agent X Y` が大量に出る

→ SafeEnv が完全に衝突を防ぐわけではなく、特定の衝突パターン (同一目的地, 正面衝突) のみ事前回避。中継ノード収束等は防げない。これは元 LDRP の仕様。

### `git+https://github.com/oxwhirl/smac.git` のインストールに失敗

→ `git` コマンドがインストールされていない可能性。`brew install git` (macOS) / `apt install git` (Linux) で対処。

---

## 8. このドキュメントを使う Claude Code への最後のメッセージ

このドキュメントの内容を一通り適用し終わったら、ユーザに以下を報告してください:

- 完了したステップ番号 (例: "Step 1-6 を全て適用しました")
- `setup_env.sh` を実行した結果の `verification OK` ログ
- `python test.py map_8x5 4 qmix tp` の出力 (= 衝突数とタスク完了数)

その後、本ドキュメント (`distribution_for_lab.md`) は **配布物には含めない** ので、ユーザに「配布前に削除してください」と伝えてください。
