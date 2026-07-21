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

## PC 版→GPU 版への変更取り込み

PC 版 (LDRP) で行った変更を GPU 版 (LDRP_GPU) に取り込む際は、**[design/gpu_changes.md](design/gpu_changes.md) を必ず参照する**。

- GPU 版には VRAM 最適化・LaRe デバイス制御・MAPPO ハイパーパラメータ調整などの固有変更がある
- `git cherry-pick` や `git merge` で衝突が発生した場合、`gpu_changes.md` に記載のある変更箇所は **GPU 版 (`<<<<<<< HEAD` 側) を優先**して採用する
- GPU 版固有の変更をうっかり PC 版で上書きしないよう、衝突解消前に必ず同ファイルで対象ファイルを確認すること
- **GPU 版でコードを変更したら、必ず `gpu_changes.md` に追記する**（変更ファイル・変更前後の値・理由を記載）。PC との衝突管理はこのファイルが唯一の情報源になるため、漏れなく記録する

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

**実行系**: `$LDRP_ENV/bin/python` (Python 3.9, `./setup_env.sh` で自動構築)

**パッケージ** (Safe-TSL-DBCT `gpu` ブランチに揃える): torch 2.7.0+cu128 / numpy 2.0.2 / gym (git版) / CUDA 12.8（GPU 時）

**注意**：numpy 2.x は座標比較で `str()` 禁止 → `float()` + list 比較に変更済み（3ファイル修正）。新規コードも禁止。Python 3.10+ 構文（PEP 604）使用禁止。

**将来の実装**: [design/future_work.md](design/future_work.md) で管理。完了項目は削除（TODO 効力維持のため）。**future_work.md 限定**；他の設計書は実装済みも残す（履歴保存）。

**短時間検証**: `lare_*_min_buffer=1, lare_*_update_freq=1, time_limit=8`

---

## ファイル構成 (ナビゲーションガイド)

```
LDRP/
├── CLAUDE.md                       # 本ファイル
├── MANUAL.md                       # ユーザー向け全文書 + 更新履歴
├── design/                         # 設計書フォルダ
│   ├── lare_integration.md         # LaRe 設計判断の元資料
│   ├── alma_integration.md         # ALMA 統合計画
│   ├── ldrp_extensions.md          # LDRP 拡張 (高:タスク再配布 / 低:複数タスク保持)
│   ├── env_maturity.md             # 環境成熟度ギャップ (CAMAR/RHCR/LoRR 比較)
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

## エントリポイント

| スクリプト | 役割 |
|---|---|
| `train.py` | **学習**：epymarl で `.th` 保存 |
| `test.py` | **評価単発**：引数で 1 条件実行 |
| `run.py` | **評価バッチ**：リスト内の複数条件を並列実行 |
| `runner.py` | クラス定義のみ（単独実行不可） |

**フロー**: `train.py` → `.th` を `src/all_policy/models/safe/` にコピー → `test.py` / `run.py`

---

## 不変条件

- **baseline** (LaRe OFF): `use_lare_path=False, use_lare_task=False` のとき、LaRe 統合前と一致
- **デフォルト値の真実**: [drp_env.py](src/main/drp_env/drp_env.py) `__init__` シグネチャ（現在: Path ON / Task OFF）
- **独立性**: LaRe-Path / LaRe-Task は完全独立。4 モード両方で対称
- **ファイル構成**:
  - `src/lare/{path,task}/checkpoints/` → autosave（大量、`.gitignore`）
  - `src/lare/{path,task}/models/` → 公開モデル（整理済み、git 追跡）
  - 命名: `{Safe_}{ALGO}_{PATH|TASK}_{map}_{N}agents_{X.X}M_{checkpoint|final}.pth`
- **新機能は新ディレクトリに分離** (既存ファイルは最小編集)

### SafeEnv ↔ PBS トレードオフ

**問題**: 待機中の `current_goal` 代入が SafeEnv の衝突判定 (`if current_goal == None`) をバイパス。

**解決**: `pbs_mode` フラグで切替（[drp_env.py](src/main/drp_env/drp_env.py) の `step()` 内）
- `False`（デフォルト）: SafeEnv 保護有 ← MARL 系向け
- `True`: PBS 互換有 ← `test.py` が自動適用（`path_planner=="pbs"` 時）

---

## ユーザー嗜好

- **応答は日本語で簡潔に**。冗長な接頭辞・末尾サマリは付けない
- **ツール承認文は日本語**（`description`）：何をするか即座に伝わるように
## コード変更のワークフロー

ソースコード（src/ 配下）への変更は、**見直しやすさを最優先** にします。以下の判断フローに従う：

### Step 1: 提示（ユーザーが入力）
**対象**：LDRP のコア実装（env / RL / LaRe / epymarl）での機能追加・バグ修正

**理由**：内容理解 + バグ防止。ユーザーがコード差分を確認してから取り込むほうが安全性が高い

**やり方**：修正の意図 → 挙動変更の影響 → 差し替える完全な関数（fenced code block）を 1 つの応答で提示

### Step 2: 直接編集（自動適用）
以下のいずれかに当てはまる場合 **のみ** Edit/Write を使う：

| 対象 | 例 | 根拠 |
|---|---|---|
| **ドキュメント編集** | CLAUDE.md / MANUAL.md / design/*.md | 内容即座に確認可能。ファイル外の動作に影響なし |
| **構成ファイル** | yaml / json / config。内容変更が即コマンド検証可能 | 実行結果で即座に妥当性が判定できる |
| **軽微修正** | typo / インデント / コメント修正。機能に影響なし | 差分が明白で誤解余地がない |
| **ユーザー指示** | 「直接やってください」と明示された場合 | ユーザーの意図を優先 |

**その他（Bash の読み込み等）**：grep/cat/find 等の読み取り、動作確認スクリプトは通常通り使用可。ソースコードへの書込みのデフォルトを「ユーザー入力」に倒すルール。

---

## 変更時の報告

**コード/設定を変更したら、応答中で必ず以下をセットで書く**:
1. **修正の意図** (なぜこの変更が必要か)
2. **修正によって何がどう変わるか** (挙動・出力・互換性・既存ファイルへの影響)

**大きな変更を加えたら [MANUAL.md](MANUAL.md) の `更新履歴` セクションに追記する**。
MANUAL.md は **人間が実装を使うための文書** なので、**ユーザーから見える挙動変更** のみ記録する。

「大きな変更」とは:
- 新パッケージ/モジュールの追加 (= 実装が入った時点)
- 公開フラグ/設定キーの新設・削除・改名
- 保存ファイル命名規則やディレクトリ構成の変更
- デフォルト挙動の変更 (特に既存フラグ false 時の挙動)
- 破壊的変更 (既存モデル/スクリプトとの非互換)

以下は記録しない:
- 小さな変更 (リファクタ・コメント・typo)
- **設計書 (`design_*.md`) の追加・更新**: 実装前の計画文書は別ファイルとして独立に管理する。実装が完了して **挙動が変わったとき** に MANUAL.md に追記する
- 内部ツーリング (CLAUDE.md / subagent 等) の変更

記録フォーマットは `MANUAL.md` 内のテンプレートを使う。

- スコープを広げる前に確認質問する
- 完了報告は1〜2行
- ユーザーの指示なしに先回り判断しない (「冗長を削っておきました」のような勝手な拡張禁止)

---

## リスク操作の代替フロー

以下のような**強い影響を持つ操作**は、常に「状況報告→ユーザー確認→実行」の 2 段階にします：

| やりたいこと | 避ける操作 | 代わりにこれ |
|---|---|---|
| **コミット修正が必要になった** | `git commit --amend` で履歴書き換え | 新しいコミットを作成。修正が必要ならユーザー確認してから `git reset --soft HEAD~1` で提案 |
| **競合ファイルの削除** | `git rm -f` / 権力的 rebase | ファイル内容を確認 → stash で脇へ置く → ユーザーに状況報告 → 意向確認後に削除 |
| **走ってるプロセス停止** | `pkill -9` で強制終了 | プロセス情報をユーザーに伝える → 停止ユーザーの意向確認 → `kill` で柔らかく → どうしても応答しなければ報告 |
| **ディレクトリ削除** | `rm -rf` で即座に削除 | `mv` で別名に変更して脇へ置く → 内容確認後に削除を提案 |
| **git push --force** | `--force-with-lease` なし | 状況説明 → 『push して良い』のユーザー確認を待つ |
| **スキップ系フラグ使用** | `--no-verify` / `--no-gpg-sign` | フックが失敗した理由を特定 → 根本原因を修正 → 改めてコミット。フック回避は最後の手段 |
| **大規模な未承認スコープ変更** | リファクタを勝手に混入 | ユーザーの要求スコープを確認 → 必要な変更のみ → 拡張提案は別途相談 |
| **設定ファイルを追跡対象に** | `.claude/settings.local.json` をコミット | gitignore で除外。ユーザーローカルな設定は共有リポに入れない |

**判断基準**：「その操作を取り消す手段がない / 手作業で復旧が困難」なら確認が必須。

---

## メモリシステム活用

自動メモリ（`/home/linlab/.claude/projects/-home-linlab-yamaguchi-LDRP/memory/`）の判断基準：

| 型 | 保存対象 | タイミング |
|---|---|---|
| `user` | ユーザーの嗜好・スキル・役割 | 初出時 |
| `feedback` | 操作指示の修正・確認 | 判定後 |
| `project` | 期限・施策・制約（絶対日時に変換） | ステータス変化時 |
| `reference` | 外部リソース参照先 | 初出時 |

**保存しない**: ファイル構成・git history・実装パターン・一時的タスク（コード/git から導出可能）

**記録ルール**: future_work.md は完了項目削除 / 設計書は履歴保存 / MANUAL.md は挙動変更のみ

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
