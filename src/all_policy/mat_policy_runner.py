import torch                       
import numpy as np      
            
from .mlp_mat_agent import Decoder  

class MatPolicyRunner:
    """MAT-Dec (mlp_mat) の agent.th を評価するランナー (分散実行専用)。
    - agent.th には decoder.* と critic.* が同居するが、評価では decoder.* だけ使う。
    - PolicyRunner と同じ get_action(ag_idx, obs, avail_actions) インタフェースに揃える。
    - use_rnn=False でステートレスなので hidden state 管理は不要。
    """

    _DEC_PREFIX = "decoder."  # checkpoint 内で actor の重みに付いている接頭辞

    def __init__(self, model_path, input_shape, n_actions, agent_num):  # モデルパスと env の次元情報を受け取る
        state_dict = torch.load(model_path, map_location="cpu")  # .th を CPU 上に読み込む (重みの辞書)

        # --- decoder.* だけ抽出 (critic.* は捨てる) ---
        dec_sd = {                                          # actor 用の新しい state_dict を作る
            k[len(self._DEC_PREFIX):]: v                    # キーから "decoder." を取り除く (Decoder 単体のキー mlp.* に合わせる)
            for k, v in state_dict.items()                  # 全キーを走査し
            if k.startswith(self._DEC_PREFIX)               # "decoder." で始まるものだけ残す (critic.* は除外)
        }
        if "mlp.1.weight" not in dec_sd or "mlp.7.weight" not in dec_sd:  # 入口/出口 Linear が無ければ
            raise ValueError(                               # MAT-Dec の checkpoint ではない → 早期エラー
                f"[MatPolicyRunner] {model_path}: decoder.mlp.1/7.weight が無い "
                f"(MAT-Dec の checkpoint ではない可能性)"
            )

        # --- shape から次元を復元 ---
        w_in = dec_sd["mlp.1.weight"]    # 最初の Linear の重み。shape = (n_embd, obs_dim)
        w_out = dec_sd["mlp.7.weight"]   # 最後の Linear の重み。shape = (n_actions, n_embd)
        ckpt_obs_dim = w_in.shape[1]     # 列数 = 学習時の obs 次元
        n_embd = w_in.shape[0]           # 行数 = 中間次元
        ckpt_n_actions = w_out.shape[0]  # 行数 = 行動数

        # --- env と食い違えば ValueError ---
        # obs_agent_id=False なので obs_dim はエージェント数に依存しない → 完全一致を要求できる
        if ckpt_obs_dim != input_shape:  # 学習時 obs 次元と実行時 obs 次元が違えば
            raise ValueError(            # マップ/FOV 違い等の設定ミス → エラー
                f"[MatPolicyRunner] obs_dim mismatch: ckpt={ckpt_obs_dim}, "
                f"env input_shape={input_shape}. "
                f"マップ/FOV 設定または obs_agent_id が学習時と異なる可能性。"
            )
        if ckpt_n_actions != n_actions:  # 行動数が違えば (別マップ等)
            raise ValueError(            # エラーで止める
                f"[MatPolicyRunner] n_actions mismatch: ckpt={ckpt_n_actions}, "
                f"env n_actions={n_actions}."
            )

        self.agent_num = agent_num                               # エージェント数を保持 (ログ表示用)
        self.decoder = Decoder(ckpt_obs_dim, ckpt_n_actions, n_embd)  # 復元した 3 次元で Decoder を構築
        self.decoder.load_state_dict(dec_sd)                     # 抽出した重みをロード (キーが mlp.* で一致)
        self.decoder.eval()                                      # 推論モードに固定 (LayerNorm 等を評価用に)
        # ※ RNN 版と違い hidden_states を作らない = ステートレス。ここが最大の差。

        print(                                                   # ロード内容を 1 行で確認表示
            f"[MatPolicyRunner] Loaded {model_path} | obs_dim={ckpt_obs_dim} | "
            f"n_embd={n_embd} | n_actions={ckpt_n_actions} | "
            f"agent_num={agent_num} (重み共有 MLP なのでエージェント数非依存)"
        )

    def get_action(self, ag_idx, obs, avail_actions):  # 1 エージェント分の行動を決める
        # ag_idx はインタフェース互換のため受け取るが、ステートレスなので未使用
        obs_tensor = torch.tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)  # obs を float32 テンソル化し (1, obs) にする
        with torch.no_grad():                                    # 勾配計算を止める (評価用・高速化)
            logit = self.decoder(obs_tensor).squeeze(0).numpy()  # Decoder に通し、バッチ次元を外して numpy 配列の logits に

        # 選べない行動 (avail_actions に無い index) を -1e10 にして絶対に選ばれないようにする
        masked = [logit[a] if a in avail_actions else -1e10 for a in range(len(logit))]
        return int(np.argmax(masked))  # 最大 logit の index = 選ぶ行動 (決定的グリーディ)
