"""LaRe-Path の評価関数 (因子) の版と、それに対応する観測の作り方.

train.py の env_args.lare_path_factor_version で番号を選ぶ。
  1: DRP 用の 10 因子 (MARL4DRP の評価関数。今までの LaRe-Path)
  2: LDRP 用の 10 因子 (LLM が生成, design/prompts/responses/v1)
  3: LDRP 用の 10 因子 (LLM が生成, design/prompts/responses/v2)
版を足すときは、観測の作り方と評価関数を 1 組にしてここに登録する。
学習済みのデコーダは版ごとに別物 (因子の意味が違う) なので、読み込み時に版を照合する。
"""

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


def get_factor_version(version):
    try:
        return FACTOR_VERSIONS[int(version)]
    except (KeyError, TypeError, ValueError):
        raise ValueError("unknown lare_path_factor_version: %r (choose from %s)"
                         % (version, sorted(FACTOR_VERSIONS)))
