# CLAUDE.md

このリポジトリで Claude Code が作業するときの前提情報・規約集。
ユーザー向けの全文書は [MANUAL.md](MANUAL.md) を参照する。

---

## プロジェクト概要

LDRP は配送経路問題 (Delivery Routing Problem) のマルチエージェント学習環境。

- **経路計画**: PBS (探索), IQL/QMIX/VDN/MAA2C (epymarl 経由の MARL)
- **タスク割当**: TP (近接優先), FIFO, PPO
- **学習可能報酬**: System A=LaRe-Path, System B=LaRe-Task (本リポで追加)

---

## リファレンス実装

- **LaRe 元論文**: [thu-rllab/LaRe](https://github.com/thu-rllab/LaRe) (AAAI-2025)
- **Safe-TSL-DBCT (LDRP の親戚プロジェクト)**: `/Users/yamaguchiyuushi/MARL4DRP/`
  - LaRe 統合の参照実装。命名規則・ファクター設計はここに揃える
  - 大量の参照が必要な時は **`marl4drp-lookup` subagent** を使うとメインコンテキストを節約できる
- **従来 LDRP (LaRe 統合前)**: `upstream` リモート (`https://github.com/kaji-ou/LDRP.git`)
  - 既に remote 設定済みなので clone 不要
  - 単一ファイル参照: `git show upstream/main:<path>` (例: `git show upstream/main:src/main/drp_env/drp_env.py`)
  - LaRe 統合前との差分: `git fetch upstream && git diff upstream/main -- <path>`
  - 参照頻度が上がってきたら subagent 化を検討

---

## 開発環境 (固定)

- **Python 実行系**: `/opt/anaconda3/envs/ldrp/bin/python` (Python **3.9**)
  - gym 0.26.2 / numpy 1.26.4 / torch 2.8.0 / networkx 3.2.1 / PyYAML 6.0.3
  - **Python 3.10+ 専用構文 (`X | None` 等の PEP 604) は使わない**。`Optional[X]` を使う
  - システム Python (3.13) は numpy 2 系で gym と非互換 → 使わない
- **CUDA**: 開発機では CPU のみ。コードは `torch.cuda.is_available()` で分岐済み

検証は環境フラグを下げて短時間で:
- `lare_*_min_buffer=1, lare_*_update_freq=1` (1 エピソードで学習発火)
- `time_limit=8` 程度

---

## ファイル構成 (ナビゲーションガイド)

```
LDRP/
├── CLAUDE.md                       # 本ファイル
├── MANUAL.md                       # ユーザー向け全文書 + 更新履歴
├── design/                         # 設計書フォルダ
│   ├── lare_integration.md         # LaRe 設計判断の元資料
│   ├── alma_integration.md         # ALMA 統合計画
│   └── multi_task_agents.md        # 複数タスク保持エージェント拡張 (将来)
├── GUIDE.md                        # LDRP 全体の概要
├── runner.py                       # 推論/学習ループ. PPO 報酬入口
├── test.py                         # gym.make + Runner 起動 (LaRe yaml キーを転送)
├── train.py                        # epymarl サブプロセス起動
├── src/
│   ├── config/default.yaml         # 全 yaml キー (LaRe 含む)
│   ├── main/drp_env/
│   │   ├── drp_env.py              # env 本体. step() 中央に LaRe フック
│   │   ├── __init__.py             # gym register
│   │   └── wrapper/safe_marl.py    # SafeEnv (drp_safe-...)
│   ├── all_policy/                 # 経路計画 (PBS, MARL 推論)
│   ├── task_assign/                # タスク割当 (PPO/TP/FIFO)
│   ├── lare/                       # 今回追加した LaRe 本体
│   │   ├── path/                   # System A
│   │   ├── task/                   # System B
│   │   └── shared/                 # SelfAttention 等
│   └── epymarl/                    # MARL 学習フレームワーク
└── .claude/
    ├── agents/marl4drp-lookup.md   # MARL4DRP 参照係 subagent
    └── settings.local.json         # 個人ローカル (gitignored)
```

---

## 不変条件 (Invariants)

これを破らない:

- **`use_lare_path=False` かつ `use_lare_task=False` のとき、LDRP の挙動は LaRe 統合前と完全一致**。新フラグはデフォルトで全部この状態
- LaRe-Path / LaRe-Task は **完全に独立**。同時 ON 可能。バッファ・最適化器・保存先まで全部別
- 4 モード (off / scratch / pretrained / finetuning) は両システムで対称
- 保存ファイル命名: `{Safe_}{ALGO}_{PATH|TASK}_{map}_{N}agents_{X.X}M_{checkpoint|final}.pth`
  - `Safe_` は SafeEnv のときのみ。非Safe では先頭 `_` ごと省略
  - フォルダは `src/lare/path/saved_models/` と `src/lare/task/saved_models/` で分離
- 既存ファイルへの編集は最小限。新機能は新ディレクトリに分離

### SafeEnv と PBS のトレードオフ (重要)

[src/main/drp_env/drp_env.py](src/main/drp_env/drp_env.py) の `step()` 内、「if action_i is current start node -> stop」分岐 (= 待機アクション) に `self.current_goal_prepare[i] = action_i` という 1 行があった。これは元々 **PBS path planner のために** 待機時も `current_goal` を None のままにしない目的で追加されたもの。

ただし、この行があると **SafeEnv (`src/main/drp_env/wrapper/safe_marl.py`) のガード `if self.current_goal[i] == None:` が待機 agent に対して false になり、衝突回避ロジックがバイパスされてしまう**。結果として、待機中の agent と動いている agent の衝突を SafeEnv が防げなくなる。

**現在の判断 (2026-05-19 〜)**: 安全制御を優先するため、当該行は **コメントアウト** されている。よって:

- **QMIX / IQL / VDN / MAA2C 等の MARL path planner を使う限りは安全制御が機能** (これが現状の標準ユースケース)
- **PBS を再度有効化したい場合は、当該行をアンコメントする必要あり** (= 安全制御は失う)
- どちらかしか同時に満たせない既知の制約

履歴: 開発者から「あのコメント行を消すと安全制御が機能する」との情報を受けて修正。以前にこの行起因の SafeEnv 無限ループバグも見つけたが (2026-05-13 修正)、根本原因がこの行であることが分かったため、SafeEnv の bug fix は元に戻し、原因側の行を消す方針に統一。

---

## ユーザー嗜好

- **応答は日本語で簡潔に**。冗長な接頭辞・末尾サマリは付けない
- **ツール呼び出しの `description` (承認プロンプトに表示される説明文) は必ず日本語で書く**。
  - `Bash` の `description`: 例 `"origin に push"`, `"git status を確認"`, `"ldrp 環境で MANUAL.md の構文チェック"`
  - `Agent` の `description`: 何をする subagent 起動か日本語で
  - `TodoWrite` の `content` / `activeForm`: タスクラベルも日本語
  - **理由**: 承認画面で「何をしようとしているか」がユーザーに即座に伝わるようにするため
- **コード/設定を変更したら、応答中で必ず以下をセットで書く**:
  1. **修正の意図** (なぜこの変更が必要か)
  2. **修正によって何がどう変わるか** (挙動・出力・互換性・既存ファイルへの影響)
- **大きな変更を加えたら [MANUAL.md](MANUAL.md) の `更新履歴` セクションに追記する**。
  MANUAL.md は **人間が実装を使うための文書** なので、**ユーザーから見える挙動変更** のみ記録する。
  「大きな変更」とは:
  - 新パッケージ/モジュールの追加 (= 実装が入った時点)
  - 公開フラグ/設定キーの新設・削除・改名
  - 保存ファイル命名規則やディレクトリ構成の変更
  - デフォルト挙動の変更 (特に既存フラグ false 時の挙動)
  - 破壊的変更 (既存モデル/スクリプトとの非互換)

  以下は記録しない:
  - 小さな変更 (リファクタ・コメント・typo)
  - **設計書 (`DESIGN_*.md`) の追加・更新**: 実装前の計画文書は別ファイルとして独立に管理する。実装が完了して **挙動が変わったとき** に MANUAL.md に追記する
  - 内部ツーリング (CLAUDE.md, subagent 等) の追加・更新

  記録フォーマットは `MANUAL.md` 内のテンプレートを使う。
- スコープを広げる前に確認質問する
- 完了報告は1〜2行
- ユーザーの指示なしに先回り判断しない (「冗長を削っておきました」のような勝手な拡張禁止)

---

## 禁止事項

- `git commit --amend` で履歴書き換え (常に新コミット)
- `git push --force` (確認なし)
- `pkill` 等プロセス強制終了 (事前提案・確認)
- `.claude/settings.local.json` をコミットに含める
- `--no-verify` でフックスキップ
- 大きな未承認スコープ変更 (リファクタ含む)

---

## 検証スニペット

LaRe-Path / LaRe-Task が動くか手早く確認するコード:

```python
import gym, numpy as np, sys
sys.path.append('.'); sys.path.append('./src/main')
import drp_env

env = gym.make('drp_env:drp_safe-3agent_map_5x4-v2',
               state_repre_flag='onehot_fov',
               reward_list={'goal':100,'collision':-100,'wait':-10.,'move':-1},
               time_limit=8, task_flag=True, task_list=None,
               use_lare_path=True, use_lare_task=True,
               lare_path_min_buffer=1, lare_path_update_freq=1,
               lare_task_min_buffer=1, lare_task_update_freq=1)
np.random.seed(0)
for ep in range(2):
    env.reset(); t=0; done=False
    while not done and t < 10:
        ta = [-1] * 3
        if env.unwrapped.current_tasklist:
            free = [k for k in range(3) if env.unwrapped.assigned_tasks[k] == []]
            free_t = [j for j, v in enumerate(env.unwrapped.assigned_list) if v == -1]
            if free and free_t:
                ta[np.random.choice(free)] = np.random.choice(free_t)
        a = {'pass': [np.random.randint(env.unwrapped.n_nodes) for _ in range(3)],
             'task': ta}
        ret = env.unwrapped.step(a); done = all(ret[2]); t += 1
    print(env.unwrapped.lare_path_module.is_trained,
          env.unwrapped.lare_task_module.is_trained)
```

期待: ep0 終了で両モジュールの `is_trained` が True に変わる。
