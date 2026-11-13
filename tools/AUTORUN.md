# 実験の自動実行 (tools/autorun.py) — 使い方とセットアップ

「この条件を 2 本学習して」「この条件を評価して」と依頼すると、空いているマシンに順番を最適化して投入する仕組み。
LLM は使わない (投入・順番決め・監視は `autorun.py` が行う)。自然言語で頼みたいときは Claude Code の
subagent `experiment-runner` が窓口になり、依頼を `autorun.py` のコマンドに直す。

- 設計 (なぜこの作りか): [design/auto_runner.md](../design/auto_runner.md)
- 学習の収集・ダッシュボード: [README.md](README.md)

## 目次

- [1. 役割](#1-役割)
- [2. 依頼の出し方 (どのマシンからでも同じ)](#2-依頼の出し方-どのマシンからでも同じ)
- [3. 司令塔 (白) のセットアップ](#3-司令塔-白-のセットアップ)
- [4. 共有フォルダ経由のマシン (黒) のセットアップ](#4-共有フォルダ経由のマシン-黒-のセットアップ)
- [5. 設定のキー](#5-設定のキー)
- [6. 困ったとき](#6-困ったとき)

---

## 1. 役割

```text
                       依頼 (add / cancel / pause)
   黒 など ───────────────────────────────────────┐
   (依頼役)    <drop_root>/_autorun/requests/      │
       ▲                                           ▼
       │ 写し (conds.json など) を読む      ┌──────────────┐  ssh で投入   GPU1 / GPU2
       └──────── <drop_root>/_autorun/ ◄────│ 白 = 司令塔   │─────────────► (学習・評価)
                                            │ autorun.py   │
                                            │ tick (15 分) │  共有フォルダに置く
                                            └──────────────┘─────────────► 黒 (実行役)
                                                                <drop_root>/黒/autorun/inbox/
                                                                   ↓ autorun_worker.py (5 分) が起動
```

| 役割 | マシン | 動かすもの |
|---|---|---|
| **司令塔** | 白 (1 台だけ) | `autorun.py tick` を launchd で 15 分ごと。依頼を見て、どのマシンで何を回すかを決めて投入する |
| **実行役 (ssh)** | GPU1 / GPU2 | 何も常駐させない。白が ssh で起動する |
| **実行役 (共有フォルダ)** | 黒 | `autorun_worker.py` を launchd で 5 分ごと。白が置いたジョブを拾って起動し、空き具合を `status.json` に書く |
| **依頼役** | どのマシンでも | `autorun.py add / status ...`。白以外のマシンでは、依頼は共有フォルダ経由で白に渡り、次の tick で取り込まれる |

- 共有フォルダ = `collect_config.yaml` の `drop_root` (既定は iCloud Drive の `LDRP_runs`)。学習結果の収集 (`collect_runs.py --export`) と同じフォルダを使う
- **学習の優先順位**: 黒 (占有できる) を先に埋め、埋まったときだけ GPU1 / GPU2 に回す。GPU は他の人と共有なので、autorun が GPU 機で同時に動かす学習は全体で `autorun_max_train_total` 本 (既定 2) まで
- **評価**: CPU だけで回す (GPU 機でも `CUDA_VISIBLE_DEVICES=""`)

## 2. 依頼の出し方 (どのマシンからでも同じ)

Claude Code で自然言語で頼む (「aoba00 7 台の 8x5_5→aoba00_5 10M を 2 本学習して」「実験の状況は」) と、
`experiment-runner` が下のコマンドを実行する。手で打つ場合:

```bash
cd <リポジトリ>
python tools/autorun.py find aoba00 7agent "8x5_5 10M aoba00_5 10M" TP   # 計画表の条件を探す (6 文字の id が出る)
python tools/autorun.py add train <id> -n 2 --then-eval                 # 2 本学習し、終わったら評価
python tools/autorun.py add eval <id> [<id> ...]                        # done の run を評価
python tools/autorun.py add train <id> -n 1 --machines 黒               # 回すマシンを指定
python tools/autorun.py add train <id> -n 1 --like <id2> --lare-model QMIX_PATH_....pth
                                    # 実績の無い条件 (新しい LaRe モデル) を、条件 id2 の run の設定で LaRe モデルだけ替えて学習
python tools/autorun.py status                                          # 依頼ごとの進み具合
python tools/autorun.py hosts                                           # マシンごとの空き
python tools/autorun.py cancel q3                                       # 取り消し (動いている run は止めない)
python tools/autorun.py pause | resume                                  # 新しい投入を止める / 再開
python tools/autorun.py pull [黒 M2]                                    # 共有フォルダのマシンに git pull --ff-only させる (5 分以内)
python tools/autorun.py pull --check                                    # その結果 (前 -> 後のコミット、失敗の理由)
    # train.py はマシンごとに書き換えて使うので、pull でも手元の版を残す。上流の新しい版は train.py.upstream に置く
    # (写しは ~/.ldrp/autorun/keep_local/)。対象のファイルは collect_config.yaml の autorun_worker.keep_local で変えられる
```

- `python` は ldrp 環境の Python (白・黒では `/opt/anaconda3/envs/ldrp/bin/python`)
- 学習の seed は指定しない (train.py と同じく実行時に作られる)。パラメータは同じ条件の done の run から取る。
  同じ条件の実績が 1 本も無い条件は投入できない
- 白以外で `add` すると `Sent to the commander (白) through the shared folder` と出る。取り込まれるのは白の次の tick (15 分以内)。
  `status` / `hosts` / `find` は白が tick ごとに書き出す写しを読むので、最大 15 分 + iCloud の同期ぶん古い

## 3. 司令塔 (白) のセットアップ

1. `tools/collect_config.yaml` (git に入らない。雛形は [collect_config.example.yaml](collect_config.example.yaml)) の各 host に `autorun:` を書き、
   **投入してよいマシンだけ `enabled: true`** にする (キーは [§5](#5-設定のキー))
2. 試しに何が投入されるかを見る (何も起動しない):

   ```bash
   /opt/anaconda3/envs/ldrp/bin/python tools/autorun.py hosts
   /opt/anaconda3/envs/ldrp/bin/python tools/autorun.py tick --dry-run
   ```

3. 15 分ごとに回す:

   ```bash
   cp tools/com.ldrp.autorun.plist ~/Library/LaunchAgents/
   launchctl load ~/Library/LaunchAgents/com.ldrp.autorun.plist
   ```

   plist の Python とリポジトリのパスは白のものが書いてある。別のマシンを司令塔にするときは書き換える。

状態は `tools/.autorun/` (queue.json = 依頼、ledger.jsonl = 投入の記録、PAUSE = 一時停止)。

## 4. 共有フォルダ経由のマシン (黒) のセットアップ

**このマシンで作業する Claude Code 向けの手順。** 白から ssh で届かないマシンを、実行役と依頼役の両方にする。
例は黒。別のマシンなら `黒` をそのマシンのラベルに読み替える (白の `collect_config.yaml` の `label:` と完全一致させる)。

### 4.0 前提

- 学習結果の書き出し (`collect_runs.py --export`) が既に動いていること ([SETUP_export_machine.md](SETUP_export_machine.md))。
  同じ共有フォルダを使う
- ldrp 環境の Python がある (`/opt/anaconda3/envs/ldrp/bin/python -c "import gym, torch, yaml"` が通る)
- このリポジトリが最新 (`git pull`)。次のファイルがあること:
  `tools/autorun.py`, `tools/autorun_worker.py`, `tools/autorun_train.py`, `tools/eval_lare_factor.py`,
  `.claude/agents/experiment-runner.md`

### 4.1 このマシンの設定を書く

`tools/collect_config.yaml` (git に入らない。無ければ作る) に次を書く。既にあるファイルなら、この 4 キーを足すだけでよい。

```yaml
machine: 黒                                               # このマシンのラベル (白の label と完全一致)
drop_root: ~/Library/Mobile Documents/com~apple~CloudDocs/LDRP_runs   # 白と同じ共有フォルダ
autorun_commander: 白                                     # 司令塔のラベル。これがあると add などが白に送られる
autorun_worker:
  python: /opt/anaconda3/envs/ldrp/bin/python             # 学習・評価に使う Python
  # train_repo: ~/LDRP                                    # 省略時: このリポジトリ
  # eval_repo: ~/LDRP_eval                                # 省略時: ~/LDRP_eval (評価用のコードの写しを白が置く)
```

### 4.2 1 回だけ手で動かす

```bash
/opt/anaconda3/envs/ldrp/bin/python tools/autorun_worker.py --status-only
ls ~/Library/Mobile\ Documents/com~apple~CloudDocs/LDRP_runs/黒/autorun/
```

期待: `status.json` ができる (空き CPU・メモリ・動いている学習の数・Python とリポジトリの場所)。
`set machine: and drop_root:` と出たら §4.1 の設定が読めていない。

### 4.3 5 分ごとに回す

```bash
/opt/anaconda3/envs/ldrp/bin/python tools/autorun_worker.py --install      # 間隔を変えるなら --interval 秒
launchctl list | grep com.ldrp.autorun-worker                             # 登録されたか
tail ~/.ldrp/autorun/worker.log                                           # 動いた記録
```

- plist は実行した Python とリポジトリのパスで作られる (ユーザー名・場所が白と違っても動く)
- **10 分たっても `launchctl list` の runs が 0 のまま**なら、`--install --calendar` で登録し直す (毎時 0, 5, 10, … 分に起動する
  `StartCalendarInterval` になる)。M2 ではログイン中に登録した `StartInterval` のジョブが一度も発火しなかった (2026-09 と 2026-11 に実測)
- やめるとき: `python tools/autorun_worker.py --uninstall` (起動済みの学習・評価は止まらない)
- 学習中は `caffeinate -i` でスリープを防ぐ。ただし蓋を閉じる・電源を切るとジョブもワーカーも止まる

### 4.4 白から見えたか確かめる

白の `collect_config.yaml` の黒の host に `autorun:` があり、`status.json` が 40 分以内に更新されていれば、
白の `autorun.py hosts` で黒が `reachable=True` になる。黒からは次で確かめられる:

```bash
/opt/anaconda3/envs/ldrp/bin/python tools/autorun.py hosts     # 白が書き出した写し (白の tick 後に出る)
```

### 4.5 依頼役として使う

§2 のコマンド (または Claude Code に自然言語で) をそのまま使う。

### 4.6 更新のしかた

`git pull` するだけ。worker は毎回このリポジトリの `autorun_worker.py` を読むので、登録し直しは要らない
(リポジトリの場所や Python を変えたときだけ `--install` をやり直す)。

## 5. 設定のキー

白の `collect_config.yaml`。各 host の `autorun:` の下に書く。

| キー | 意味 |
|---|---|
| `enabled` | `true` のマシンにだけ投入する |
| `max_train` / `max_eval` | 同時に回す学習・評価の本数の上限 (手で回したものも含めて数える) |
| `train_priority` / `eval_priority` | 優先順位。小さいほど先に埋める (既定 0)。同じ段の中では終わる時刻が早いマシン |
| `shared` | `true` = 他の人と共有のマシン。学習が `autorun_max_train_total` (トップレベル) の上限に数えられる |
| `gpu_mem_per_train_mb` / `max_gpu_util` | GPU の空きメモリが足りない・使用率が高いときは学習を入れない |
| `cpu_algos` | 例 `[mappo]`。このアルゴリズムの学習は GPU を見せずに (`CUDA_VISIBLE_DEVICES=""` を付けて) CPU で回す。GPU の空き・使用率の判定もしない。2026-11-12 から GPU1 / GPU2 で MAPPO に一時的に設定中 |
| `eval_speed` | 評価の速さの比 (1.0 = 1 条件 12 分)。順番の見積もりに使う |
| `python` / `train_repo` / `eval_repo` | ssh のマシンだけ書く。共有フォルダのマシンは worker が `status.json` で知らせる |

## 6. 困ったとき

| 症状 | 原因と対処 |
|---|---|
| 黒が `reachable=False` | `status.json` が 40 分以上古い。黒の worker が動いていない (§4.3 の `launchctl list`・`worker.log`)、黒がスリープ中、または iCloud の同期待ち |
| 黒で `add` したのに `status` に出ない | 白の tick がまだ (15 分おき)、白がスリープ中、または iCloud の同期待ち |
| `no conds.json from the commander yet` | 白がまだ一度も tick していない (白の launchd 未登録、または §4.1 の `drop_root` が白と違う) |
| 依頼がいつまでも投入されない | 空きが無い・`enabled: false`・同じ条件の実績 run が無い・評価する方策モデルがまだ回収されていない。白で `autorun.py status` (`note:` の行) と `tick --dry-run` に理由が出る |
| 黒で学習が始まらない | `~/.ldrp/autorun/worker.log` と `~/.ldrp/autorun/worker_errors.json` を見る。epymarl の設定が見つからないときは `train_repo` が違う |
