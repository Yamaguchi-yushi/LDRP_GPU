#!/usr/bin/env bash
#
# LDRP conda env セットアップスクリプト.
#
# 使い方:
#   ./setup_env.sh                    # GPU 版 (CUDA 12.8 / torch+cu128) でインストール
#   ./setup_env.sh --cpu              # CPU 版 (torch CPU wheel) でインストール
#   ./setup_env.sh --recreate         # 既存の ldrp env を削除してゼロから作り直す
#   LDRP_ENV_NAME=foo ./setup_env.sh  # env 名を foo にする
#
# 前提:
#   - anaconda / miniconda がインストール済みで `conda` コマンドが使える
#   - リポジトリのルートディレクトリで実行する (= setup_env.sh と同じディレクトリ)
#   - GPU 版を使う場合: NVIDIA ドライバが CUDA 12.8 対応バージョンであること
#     (Blackwell / RTX 50 系 = 必須, Ada / Ampere = ドライバ 525.60+ 推奨)
#
# Safe-TSL-DBCT (gpu ブランチ) の torch バージョンに揃えてある:
#   torch==2.7.0+cu128 / torchvision==0.22.0+cu128 / torchaudio==2.7.0+cu128
#
set -euo pipefail

ENV_NAME="${LDRP_ENV_NAME:-ldrp}"
PYTHON_VERSION="3.9"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# torch バージョン (Safe-TSL-DBCT gpu branch と一致)
TORCH_VERSION="2.7.0"
TORCHVISION_VERSION="0.22.0"
TORCHAUDIO_VERSION="2.7.0"
CUDA_TAG="cu128"
TORCH_INDEX_URL="https://download.pytorch.org/whl/${CUDA_TAG}"

# gym (Safe-TSL-DBCT gpu freeze と同じ commit を pin). numpy 2.x 対応のため git 版を使う.
GYM_GIT_SPEC="git+https://github.com/openai/gym.git@c755d5c35a25ab118746e2ba885894ff66fb8c43"

RECREATE=false
USE_CPU=false
for arg in "$@"; do
    case "$arg" in
        --recreate) RECREATE=true ;;
        --cpu)      USE_CPU=true ;;
        -h|--help)
            sed -n '2,18p' "${BASH_SOURCE[0]}"
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
    # miniforge の python は pip を含まないので明示インストール
    conda create -n "$ENV_NAME" "python=${PYTHON_VERSION}" pip -y
fi

# --- 4) pip の更新 + 依存インストール -----------------------------------------
# pip が無ければ (古い env を流用したケース等) フォールバックで入れる
if ! "$ENV_PYTHON" -m pip --version >/dev/null 2>&1; then
    echo "[setup] pip が無いので conda 経由でインストール中..."
    conda install -n "$ENV_NAME" pip -y
fi

echo "[setup] pip を更新中..."
"$ENV_PYTHON" -m pip install --upgrade pip --quiet

# 4a) torch を先にインストール (cu128 wheel は通常の PyPI に無いため index-url を切り替える).
if [ "$USE_CPU" = "true" ]; then
    echo "[setup] torch (CPU) をインストール中..."
    "$ENV_PYTHON" -m pip install \
        "torch==${TORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" \
        "torchaudio==${TORCHAUDIO_VERSION}"
else
    echo "[setup] torch (${CUDA_TAG} / CUDA 12.8) をインストール中..."
    "$ENV_PYTHON" -m pip install \
        "torch==${TORCH_VERSION}+${CUDA_TAG}" \
        "torchvision==${TORCHVISION_VERSION}+${CUDA_TAG}" \
        "torchaudio==${TORCHAUDIO_VERSION}+${CUDA_TAG}" \
        --index-url "$TORCH_INDEX_URL"
fi

# 4b) gym を git commit から install (Safe-TSL-DBCT gpu と同じ commit / numpy 2.x 対応).
# 古い gym (c755d5c == 0.21.0) の workaround:
#   - setup.py の extras_require が setuptools 66+ で reject される → setuptools<66 を env に置く
#   - metadata 内の `opencv-python (>=3.)` (trailing dot) が pip 24+ で reject される → pip<24 に落とす
# build isolation を切って env 内の古い setuptools/wheel を使わせ, インストール完了後に pip を最新へ戻す.
echo "[setup] gym build 用に pip/setuptools/wheel をダウングレード中 (古い gym workaround)..."
"$ENV_PYTHON" -m pip install "pip<24" "setuptools<66" "wheel<0.40" --quiet

echo "[setup] gym (git commit pin) をインストール中..."
"$ENV_PYTHON" -m pip install --no-build-isolation "$GYM_GIT_SPEC"

echo "[setup] pip を最新に戻し中..."
"$ENV_PYTHON" -m pip install --upgrade pip --quiet

echo "[setup] 依存パッケージをインストール中 (requirements.txt)..."
"$ENV_PYTHON" -m pip install -r "${REPO_ROOT}/requirements.txt"

# --- 5) ローカル drp パッケージを editable インストール -----------------------
echo "[setup] ローカルの drp パッケージを editable モードでインストール中..."
"$ENV_PYTHON" -m pip install -e "${REPO_ROOT}/src/main" --quiet

# --- 6) 動作確認 ----------------------------------------------------------------
echo "[setup] 動作確認中..."
"$ENV_PYTHON" - <<EOF
import gym, numpy, torch, networkx, yaml
print(f"  gym         : {gym.__version__}")
print(f"  numpy       : {numpy.__version__}")
print(f"  torch       : {torch.__version__}")
print(f"  torch.cuda  : available={torch.cuda.is_available()} "
      f"built_for={torch.version.cuda} "
      f"device_count={torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"  cuda device : {torch.cuda.get_device_name(0)}")
print(f"  networkx    : {networkx.__version__}")
print(f"  PyYAML      : {yaml.__version__}")

# epymarl 学習 (train.py) で必須の追加依存も確認
import sacred, einops
print(f"  sacred      : {sacred.__version__}")
print(f"  einops      : {einops.__version__}")
try:
    import tensorboard as _tb
    tb_ver = _tb.__version__
except Exception as e:
    tb_ver = f"NG ({e})"
print(f"  tensorboard : {tb_ver}")
try:
    import smac
    smac_ver = "ok"
except Exception as e:
    smac_ver = f"NG ({e})"
print(f"  smac        : {smac_ver}")

# epymarl 学習 (train.py) で必須の追加依存も確認
import sacred, einops
try:
    import tensorboard_logger as _tbl
    tbl_ver = getattr(_tbl, "__version__", "ok")
except Exception as e:
    tbl_ver = f"NG ({e})"
try:
    import tensorboard as _tb
    tb_ver = _tb.__version__
except Exception as e:
    tb_ver = f"NG ({e})"
print(f"  sacred              : {sacred.__version__}")
print(f"  einops              : {einops.__version__}")
print(f"  tensorboard-logger  : {tbl_ver}")
print(f"  tensorboard         : {tb_ver}")

import sys
sys.path.append("${REPO_ROOT}")
sys.path.append("${REPO_ROOT}/src/main")
import drp_env
print(f"  drp_env     : {drp_env.__file__}")
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
