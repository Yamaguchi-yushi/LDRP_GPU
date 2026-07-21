# epymarl 本体 (src/epymarl/src/modules/agents/mlp_mat_agent.py) の Decoder をローカルにコピーしたファイル。
# 学習側の MLPMATAgent には critic 等も入るが、評価では actor = Decoder しか使わないので Decoder だけ写している。
# rnn_agent.py が RNNAgent をローカルコピーしているのと同じ「対」の作り。

import torch.nn as nn  # ニューラルネット層 (Linear/LayerNorm/GELU/Sequential) を使うため

class Decoder(nn.Module):  # MAT-Dec の actor 本体。obs → 行動 logits を出す MLP
    def __init__(self, obs_dim, action_dim, n_embd):  # obs 次元・行動数・中間次元を受け取る
        super(Decoder, self).__init__()  # 親クラス nn.Module の初期化 (パラメータ登録の土台) を必ず呼ぶ

        self.action_dim = action_dim  # 行動数を保持 (推論では未使用だが原本に合わせて残す)
        self.n_embd = n_embd          # 中間次元を保持 (同上)

        self.mlp = nn.Sequential(  # 層を順番に積んだ 1 本の MLP。各層の index が state_dict のキーに対応する
            nn.LayerNorm(obs_dim),                                        # [0] 入力を正規化 (学習を安定させる)
            init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU(),  # [1] 入力→中間の全結合 + [2] GELU 活性化
            nn.LayerNorm(n_embd),                                         # [3] 中間を正規化
            init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(),   # [4] 中間→中間の全結合 + [5] GELU 活性化
            nn.LayerNorm(n_embd),                                         # [6] 中間を正規化
            init_(nn.Linear(n_embd, action_dim))                          # [7] 中間→行動数の全結合 (出力 = logits)
        )  # ※ この並び順を変えると mlp.1 / mlp.7 の index がズレて load_state_dict が壊れるので変更禁止

    def forward(self, obs):  # 前向き計算。obs だけ受け取る (hidden 無し = ステートレス)
        logit = self.mlp(obs)  # MLP に通して各行動のスコア (logits) を得る
        return logit           # logits を返す (呼び出し側で argmax して行動を選ぶ)

def init(module, weight_init, bias_init, gain=1):  # 1 層の重み/バイアスを初期化する汎用ヘルパ
    weight_init(module.weight.data, gain=gain)     # 重みを指定の方法 (直交初期化) で初期化。gain は初期値のスケール倍率
    if module.bias is not None:                    # バイアスを持つ層なら
        bias_init(module.bias.data)                # バイアスを初期化 (定数 0)
    return module                                  # 初期化済みの層を返す (Sequential にそのまま入れる)

def init_(m, gain=0.01, activate=False):  # Linear 用の初期化ラッパ (直交初期化 + バイアス 0)
    if activate:                          # 直後に活性化 (GELU/ReLU) が来る層は
        gain = nn.init.calculate_gain('relu')  # ReLU 系に合わせた gain (≈√2) を使い、活性化で減る信号を補償する
    # 直交初期化 + バイアス定数 0 で層を初期化して返す
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)
    # 注: 評価では load_state_dict が重みを上書きするため、この初期化値は推論結果に影響しない (原本一致のため残す)。
