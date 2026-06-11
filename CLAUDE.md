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

## 開発環境 (固定) — GPU 版 (2026-05-15〜)

- **Python 実行系**: `<miniforge>/envs/ldrp/bin/python` (Python **3.9**)
  - 例: linlab GPU マシンでは `/home/linlab/miniforge3/envs/ldrp/bin/python`
  - `./setup_env.sh` を 1 回実行すれば env 作成・依存インストール・editable install まで完了する
- **主要パッケージ** (Safe-TSL-DBCT `gpu` ブランチに揃えてある):
  - torch **2.7.0+cu128** / torchvision 0.22.0+cu128 / torchaudio 2.7.0+cu128
  - numpy **2.0.2** (1.x ではない — 後述のコード互換に注意)
  - gym は **PyPI 0.26.2 ではなく特定 commit の git 版** (`git+https://github.com/openai/gym.git@c755d5c35a...`)。numpy 2.x 対応のため
  - networkx 3.2.1 / PyYAML 6.0.3 / matplotlib 3.9.4
- **CUDA**: 開発機 (linlab GPU マシン) で **CUDA 12.8 利用可**。コードは `torch.cuda.is_available()` で分岐済み
- **CPU フォールバック**: GPU が無い環境では `./setup_env.sh --cpu` で CPU wheel が入る
- **Python 構文制約**: Python 3.10+ 専用構文 (`X | None` 等の PEP 604) は使わない。`Optional[X]` を使う

### numpy 2.x 移行で入った互換パッチ (2026-05-15)

numpy 2.0 で `repr(np.float64(0.0))` が `0.0` → `np.float64(0.0)` に変わり、`str()` を介した座標比較が壊れる。LDRP では下記 3 ファイルを Safe-TSL-DBCT と同じ方針で修正済み:
- `src/main/drp_env/EE_map.py` (`get_avail_action_fun`): `str()` 比較 → `float()` + list 比較
- `src/main/drp_env/state_repre/wrapper/hrs_hot_file.py`: 同様
- `src/main/drp_env/drp_env.py` (`_get_avail_agent_actions`): numpy 2.x は float index を許さないため、`avail_actions` を `int()` 化してから fancy index する

新規コードを書く際は **座標比較に `str()` を使わない**。`float()` 化した list で比較すること。

将来やりたい実装・未適用の修正は [design/future_work.md](design/future_work.md) に集約。

- **実装が完了した項目は future_work.md から削除する** (future_work は「未適用の TODO」集約。実装済みを残すと未着手と完了が混在し TODO として機能しなくなる)。
  - 1 項目に「実装済み部分」と「未着手の残課題」が混在する場合は、**実装済み部分だけ削り残課題は残す** (残課題の理解に必要な最小限の前提だけ 1〜2 行で残してよい)。
  - この削除ルールは **future_work.md 限定**。他の設計書 (`lare_integration.md` 等) は全体設計を記述する文書なので、実装済みでも削除しない。

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
│   ├── multi_task_agents.md        # 複数タスク保持エージェント拡張 (将来)
│   └── future_work.md              # 将来実装メモ (TODO 集約)
├── GUIDE.md                        # LDRP 全体の概要
├── runner.py                       # Runner クラス: 評価ループ本体 (test.py から呼ばれる)
├── run.py                          # 【評価バッチ】 複数条件 (map × agent × path × task) を test.py で並列実行
├── test.py                         # 【評価単発】 学習済み方策モデル (.th) を読み込んで推論
├── train.py                        # 【学習】 epymarl サブプロセス起動 (QMIX/IQL 等の方策学習)
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

## エントリポイントの役割 (混同しがちなので明示)

| スクリプト | 用途 | 入力 | 主な出力 |
|---|---|---|---|
| `train.py` | **学習** (方策パラメータの更新). epymarl サブプロセスで QMIX/IQL 等を学習 | epymarl の config + env_args | `src/epymarl/tmp_results/models/{N}_{map}_safe_{algo}/{step}/{map_name}_{N}_{algo}.th` |
| `test.py` | **評価 (単一条件)**. 学習済み `.th` モデルを読み込んで `test_num` 回エピソード実行 | コマンドライン引数 (`map`, `agent_num`, `path_planner`, `task_assigner`) + `src/all_policy/models/safe/{map}_{N}_{algo}.th` | エピソード集計を標準出力 |
| `run.py` | **評価 (複数条件のバッチ実行)**. map/agent/path/task の組み合わせ分 test.py を subprocess で並列実行 (最大 5 並列) | run.py 内のリスト (`map_name`, `agent_num`, `path_planner`, `task_assigner`) | 各組み合わせのログを `logs/{map}/safe/{map}_{N}_{path}_{task}.txt` に保存 |
| `runner.py` | Runner クラス本体 (= 評価/学習ループの実装). 単独実行はしない | (test.py から渡される args/env) | (test.py 側で集計) |

### 覚え方

- **学習 = `train.py`** (epymarl サブプロセスで方策を学習し `.th` を保存)
- **評価単発 = `test.py`** (コマンドライン引数で 1 条件指定して評価)
- **評価バッチ = `run.py`** (リストで複数条件を回す。ログを `logs/` 配下にまとめる)
- `runner.py` は「クラス定義」(`Runner`). 直接実行はしない

### 学習後 → 評価への流れ

```text
[Step 1] train.py で学習
  ↓ 完了後
[Step 2] src/epymarl/tmp_results/models/.../{map}_{N}_{algo}.th を
         src/all_policy/models/safe/{map}_{N}_{algo}.th に手動コピー
  ↓
[Step 3] 単発: test.py {map} {N} {algo} {task_assigner}
         または
         バッチ: run.py (内部で複数 test.py を起動、logs/ に出力)
```

---

## 不変条件 (Invariants)

これを破らない:

- **`use_lare_path=False` かつ `use_lare_task=False` のとき、LDRP の挙動は LaRe 統合前と完全一致** (baseline 条件は固定)。
  - デフォルト値の単一の真実は [src/main/drp_env/drp_env.py](src/main/drp_env/drp_env.py) の `DrpEnv.__init__` シグネチャ。`drp_safe-*` の register kwargs では LaRe フラグを上書きしない (= signature の値がそのまま `gym.make` のデフォルトになる)
  - 現状の signature デフォルト: `use_lare_path=True` (= LaRe-Path ON), `use_lare_task=False` (= LaRe-Task OFF)。baseline で動かしたいときは env_args (train.py) / yaml (test.py) / `gym.make` kwargs で `use_lare_path=False` を明示する
- LaRe-Path / LaRe-Task は **完全に独立**。同時 ON 可能。バッファ・最適化器・保存先まで全部別
- 4 モード (off / scratch / pretrained / finetuning) は両システムで対称
- 保存ファイル命名: `{Safe_}{ALGO}_{PATH|TASK}_{map}_{N}agents_{X.X}M_{checkpoint|final}.pth`
  - `Safe_` は SafeEnv のときのみ。非Safe では先頭 `_` ごと省略
- ディレクトリ構成 (固定. yaml で変更不要):
  - `src/lare/path/checkpoints/` ← autosave 出力 (大量蓄積, **.gitignore**)
  - `src/lare/path/models/` ← pretrained / finetuning ロード元 (整理済み, **git 公開**)
  - Task 側 (`src/lare/task/`) も同じ構造
  - 旧 `saved_models/` は load の後方互換用に解決パスに残るが新規利用は非推奨
- **autosave 保存頻度**: 累積環境ステップ `lare_*_save_freq_steps` (デフォ 500_000 = 0.5M) ごとに 1 回。学習頻度 (`update_freq`) とは独立
- 良いモデルが見つかったら `cp checkpoints/Safe_..._X.XM_checkpoint.pth models/<好きな名前>.pth` で `models/` にコピー → 次回 `pretrained_lare_*_model_path` でファイル名だけ指定すれば自動解決
- 既存ファイルへの編集は最小限。新機能は新ディレクトリに分離

### SafeEnv と PBS のトレードオフ (= `pbs_mode` フラグで切替可能)

[src/main/drp_env/drp_env.py](src/main/drp_env/drp_env.py) の `step()` 内、「if action_i is current start node -> stop」分岐 (= 待機アクション) に `self.current_goal_prepare[i] = action_i` という代入がある。これは元々 **PBS path planner のために** 待機時も `current_goal` を None のままにしない目的で追加されたもの。

ただし、この代入が常に実行されると **SafeEnv (`src/main/drp_env/wrapper/safe_marl.py`) のガード `if self.current_goal[i] == None:` が待機 agent に対して false になり、衝突回避ロジックがバイパスされてしまう**。結果として、待機中の agent と動いている agent の衝突を SafeEnv が防げなくなる。

**現在の判断 (2026-05-19 〜)**: `pbs_mode` (bool) フラグで切替可能にしている。

| `pbs_mode` | 待機分岐の動作 | 影響 |
|---|---|---|
| **False** (デフォルト) | 代入をスキップ → `current_goal` は None のまま | **SafeEnv が待機 agent も保護** ✓ MARL 系 (QMIX/IQL/VDN/MAA2C) で衝突減 |
| **True** | 代入実行 → `current_goal = 待機ノード id` | **PBS の path 計画が正しく動く** ✓ ただし SafeEnv の保護は失う |

`test.py` は `config.path_planner == "pbs"` のとき自動で `pbs_mode=True` を `gym.make()` に渡す ([test.py](test.py))。MARL 系では `False` (= デフォルト) のまま。

---

## ユーザー嗜好

- **応答は日本語で簡潔に**。冗長な接頭辞・末尾サマリは付けない
- **ツール呼び出しの `description` (承認プロンプトに表示される説明文) は必ず日本語で書く**。
  - `Bash` の `description`: 例 `"origin に push"`, `"git status を確認"`, `"ldrp 環境で MANUAL.md の構文チェック"`
  - `Agent` の `description`: 何をする subagent 起動か日本語で
  - `TodoWrite` の `content` / `activeForm`: タスクラベルも日本語
  - **理由**: 承認画面で「何をしようとしているか」がユーザーに即座に伝わるようにするため
- **コード変更は「Claude が直接ファイルを編集する」のではなく「方針 + 説明 + 完成コードを提示してユーザー自身が入力する」ワークフローを基本とする**。
  - 目的: ユーザーが内容を理解した上でコードを取り込めるようにする (学習・レビューの併走)。
  - 提示する内容: (1) 修正の意図と方針, (2) 挙動・互換性への影響, (3) 差し替える完全な関数 / ブロックを fenced code block で 1 つの応答にまとめる。
  - Edit / Write ツールを直接走らせて良いのは下記のとき:
    - ユーザーが明示的に「直接編集して」「ファイルを書き換えて」と指示したとき
    - 設計書 (`design/`), `CLAUDE.md`, `MANUAL.md` 等のドキュメント類の編集
    - ごく軽微なシンタックス修正 (typo・整形) で `Edit` の方が明らかに楽なとき
  - Bash の grep/cat 等の読み取りや動作確認スクリプトは普通に使ってよい。あくまで「ソースファイルへの書込み」のデフォルトを「ユーザー入力」に倒すルール。
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
