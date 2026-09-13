# tools/ — 実験の収集・監視・集計

| ツール | 対象 | 何をするか |
|---|---|---|
| [collect_runs.py](collect_runs.py) | **方策学習** | 全マシンの run を収集し、状態・進捗・終了予定を出す。方策モデルの回収と保管 |
| [eval_report.py](eval_report.py) | **方策評価** | `results/summary.csv` を条件ごとに分解して表示 |
| [plan.py](plan.py) | 実験計画 | `plans/*.md` をパースする。Notion の表をそのまま貼れる |
| [VISIT_CHECKLIST.md](VISIT_CHECKLIST.md) | 運用 | 他マシンに触れる日の作業手順 |
| [dashboard/](dashboard/) | 両方 | ローカル Web アプリ (学習の進捗 / 評価結果 の 2 タブ) |

**収集される側のマシン** (SSH で繋がらない Mac など) のセットアップは
[SETUP_export_machine.md](SETUP_export_machine.md) を参照。そのマシンで
作業する Claude Code 向けの手順書になっている。

---

# collect_runs.py — 学習 run のマシン横断収集

Notion に seed と条件を手で書き写す作業を無くすためのツール。
**日常の確認は [dashboard/](dashboard/) を使う** (Notion 連携は残っているが既定では使わない)。
**実装の説明 (なぜその作りにしたか) は [design/run_collector.md](../design/run_collector.md)**。
各マシンの sacred 出力を読んで「どの条件の seed が、どこまで学習したか」を 1 つの表にする。

- **完遂チェック**: 設定した `t_max` まで到達したか (`OK` / `SHORT`)、途中で死んでいないか (`STALL` / `FAIL`) を判定
- **終了予定時刻**: 実測ペースから残り時間と終了予定時刻を出す (`12h01m -> 09/01 03:40`)
- **学習パラメータの保存**: `config.json` を全文保持し、条件としての同一性を `param_hash` で表す。
  同じ条件のはずの seed でパラメータが割れていたら警告する
- **完了ポップアップ**: run が終わった / 落ちたときに macOS の通知を出す (`--notify`)
- **常駐パネル**: メニューバーやターミナルに貼れる要約を出す (`--format status`)
- **モデル回収**: 完遂した run の最終ステップの方策モデルを各マシンから自動で持ってくる
- **低メモリ**: `config.json` / `run.json` (数 KB) と `cout.txt` の末尾 64KB しか読まない。
  巨大な `metrics.json` (1〜5MB) は「末尾 256KB を読んで最後の t_env を拾う」ときだけ触る。
  全体をメモリに載せることは一度もない
- **リモートに何も置かない**: pull モードではこのスクリプト自身を ssh の stdin で送り込んで実行する。
  リモート側の要件は `python3` (標準ライブラリのみ) だけ

---

## 方策評価の結果を見る (tools/eval_report.py)

`aggregate.py` が出した `results/summary.csv` を読んで、条件を分解して見せる。
**集計そのもの (重複検出 / ddof=1 / seed 数) は再実装しない**。summary.csv を読むだけ。

```bash
# ブラウザで見る (1 枚の HTML。外部依存ゼロ)
python tools/eval_report.py -s results/summary.csv --format html -o eval.html
open eval.html

# 端末で見る
python tools/eval_report.py -s results/summary.csv --map map_8x5 --n 5
```

```text
map      N  env     planner  tag  alloc  task_completion  sd    n  time_sec  sd      n
map_8x5  5  safe    qmix     -    tp     32.78            9.01  5  0.7392    0.0077  5
map_8x5  5  unsafe  qmix     -    tp     1.38             0.91  5  0.0398    0.006   5
map_8x5  5  unsafe  pbs      -    fifo   116.8            -     1  4536.13   -       1
```

### HTML ダッシュボード

- **絞り込み**: map / N / env / planner / tag / alloc、メトリクス選択
- **表**: `mean ± sd`、`n_seeds` (5 未満は黄、1 は赤)、per-seed の実値。列見出しでソート
- **グラフ**: 横軸 N、系列 planner、**per-seed を小さい点で重ねる**。平均は太い点と折れ線
- **log 軸トグル**: `time_sec` は実測 0.0236〜5887.78 秒で **5 桁**開くので、
  線形軸だと MARL 系が潰れて PBS しか見えない

### 条件キーの分解

`runner.py` の `_condition_id()` を逆に解く。

```text
map_aoba00/10agent/safe_mat_dec_dbct_base_tp
  → map=map_aoba00  n=10  env=safe  planner=mat_dec  tag=dbct  reassign=base  alloc=tp
```

**planner 名に `_` が入る** (`mat_dec` / `transf_qmix`) ので、既知の名前リストで
**最長一致**させる。`_` で切ると `mat_dec`+`dbct` と `mat`+`dec_dbct` を区別できない。
新しい planner を足したら `DEFAULT_PLANNERS` にも足す (`--planner-names` でも渡せる)。

古い summary.csv には `reassign` タグが無い (`safe_iql_fifo`) ので、
`base` / `reassign` が現れるかどうかで新旧を判定する。両方読める。

### メトリクスは固定しない

`runner.py` の `[RESULT]` は改版で増減している。手元の summary.csv には
`1agent_goal_account` があるが現在の `runner.py` には無く (`task_completion_per_agent` に改名)、
逆に `n_active_mean` / `busy_ratio` / `deadhead_ratio` などが入っていない。
**summary.csv にある列を動的に読む**ので、回し直して列が増えてもそのまま出る
(`--all-metrics` で全部、`--metrics` で選択)。

> `results.csv` 形式への pivot 書き出しは未実装。追記にするか毎回作り直すかを
> 決めてから入れる。

---

## 3 つのモード

| モード | 実行する場所 | 必要なもの | モデルも集まるか | 使いどころ |
|---|---|---|---|---|
| **pull** | Mac 1 台 | 各ホストへの ssh 鍵 (パスフレーズ無し) | ○ | 既定。Mac だけ設定すれば全ホストを回れる |
| **drop** | 相手マシン + 集約 Mac | 両方から見える共有フォルダ | ○ | **SSH で繋がらないマシン** (別ネットワークの Mac など) |
| **push** | 各マシン | そのマシンに Notion token | × (Notion の行だけ) | 共有フォルダも用意できないマシン |

`run_uid = {machine}:{results|tmp_results}/{algo}/{env_key}/{run_id}` をキーに upsert するので、
**どのモードを混ぜても同じ Notion DB に重複なく集まる**。

---

## セットアップ

### 1. ssh 鍵 (pull モード)

cron から動かすのでパスフレーズ無しで入れる状態にする。

```bash
ssh-copy-id your-gpu-host
ssh -o BatchMode=yes your-gpu-host hostname   # パスワードを聞かれなければ OK
```

### 2. ホスト定義

[collect_config.yaml](collect_config.yaml) の `hosts:` を編集する。`repos:` はそのマシンの
LDRP リポジトリのルート (その下の `src/epymarl/{results,tmp_results}/sacred` を自動で探す)。

疎通確認:

```bash
python tools/collect_runs.py -c tools/collect_config.yaml --format table -v
```

### 3. SSH で繋がらないマシン (drop モード)

**共有フォルダを 1 つ挟むだけ**で、SSH 無しでも run 一覧とモデルの両方が集まる。
フォルダは **iCloud Drive / Dropbox / Google Drive / NFS / USB メモリ、何でもよい**
(置くのはただのファイルなので、同期の仕組みは問わない)。Mac 同士なら iCloud Drive が一番手間がない。

```text
[別の Mac]                            [共有フォルダ]                     [集約する Mac]
 collect_runs.py --export  ───────>   LDRP_runs/MacB/runs.jsonl   ───>   collect_runs.py
   自分の sacred を読む                 LDRP_runs/MacB/models/...          (drop: true で読む)
   完遂モデルをコピー                                                       保管リポジトリへ配置 / 設置
```

**相手のマシン側** (15 分ごとに書き出す):

```bash
python tools/collect_runs.py \
    --export "~/Library/Mobile Documents/com~apple~CloudDocs/LDRP_runs" \
    --machine MacB --repo ~/LDRP
```

定期実行の定義は [com.ldrp.export-runs.plist](com.ldrp.export-runs.plist) を使う
(`USERNAME` と `MacB` を書き換えて `~/Library/LaunchAgents/` へ)。
このコマンドは **外部に一切アクセスしない**。共有フォルダに書くだけ。
進捗ファイル (`runs.jsonl`) を**モデルより先に**書くので、モデルの同期が
詰まっても進捗は最新になる。

**集約する Mac 側** ([collect_config.yaml](collect_config.yaml)):

```yaml
drop_root: ~/Library/Mobile Documents/com~apple~CloudDocs/LDRP_runs

hosts:
  - label: MacB
    drop: true        # drop_root/MacB/ を読む
```

あとは通常どおり実行すれば、SSH のホストと同じ表に並ぶ。

```bash
python tools/collect_runs.py -c tools/collect_config.yaml \
    --cache tools/.run_cache.jsonl \
    --fetch-models ~/models_inbox --publish-models ~/LDRP_models \
    --purge-drop --format status
```

`--purge-drop` は**回収できたモデルだけ**を共有フォルダから消す
(ローカルに同じサイズのファイルがあることを確認してから消す)。iCloud の容量対策。

> - 相手側が書き出す量は「done の run の最終ステップだけ」なので実測 **35 run で 29MB**。
>   iCloud の無料 5GB でも十分収まる
> - `runs.jsonl` が 24 時間以上更新されていないと警告を出す
>   (相手のマシンが止まっているか、フォルダが同期していない)
> - iCloud の「ストレージを最適化」が有効だとファイルが実体を持たない状態になることがある。
>   読み出し時にダウンロードされるので動くが、オフラインだと失敗する

### 4. Notion (無料プランで使える)

1. <https://www.notion.so/my-integrations> で **New integration** を作り、
   `Internal Integration Secret` (`ntn_...` / `secret_...`) をコピー
2. DB を置きたい Notion ページを開き、右上 `...` → **Connections** → 作った integration を追加
3. そのページの URL 末尾 32 桁が `PAGE_ID`
   (`https://www.notion.so/My-Page-**1234abcd...**`)
4. DB を作る (プロパティが自動で全部揃う):

```bash
export NOTION_TOKEN='ntn_xxxxxxxx'
python tools/collect_runs.py --notion-create-db 1234abcd5678...
# -> created database: 87654321-....
```

5. 出た id を `collect_config.yaml` の `notion.database_id` に貼る
6. 疎通を確認する

```bash
python tools/collect_runs.py -c tools/collect_config.yaml --notion-check
# [notion] token ok (integration: ...)
# [notion] database ok: 'LDRP runs' (18 properties)
# [notion] all properties present
```

#### token の置き場所

cron / systemd から環境変数を渡すのは面倒なので、**ファイルからも読む**
(コマンドライン引数では受け取らない。`ps` に出てしまうため)。上から順に探す。

```text
環境変数 NOTION_TOKEN
  → --notion-token-file で指定したファイル
  → tools/.notion_token          (.gitignore 済み)
  → ~/.config/ldrp/notion_token
  → ~/.ldrp/notion_token         (--bootstrap が置く場所)
```

```bash
mkdir -p ~/.config/ldrp && chmod 700 ~/.config/ldrp
printf 'ntn_xxxxxxxx' > ~/.config/ldrp/notion_token
chmod 600 ~/.config/ldrp/notion_token
```

> 既存の Notion DB を使いたい場合は、`database_id` にその DB の id を書き、
> `collect_runs.py` の `NOTION_SCHEMA` に並んだプロパティ名 (`seed` / `machine` /
> `setting` / `algorithm` / `task arrival` / `task assign` / `agents` / `map` /
> `steps (M)` / `t_env` / `progress` / `status` / `lare mode` / `elapsed` /
> `last seen` / `run dir` / `run_uid`) を DB 側に同名・同型で作っておくこと。
> `seed` は title 型、`run_uid` は rich_text 型で、これが重複判定のキーになる。

---

## 使い方

```bash
# 端末で確認するだけ
python tools/collect_runs.py -c tools/collect_config.yaml

# 条件ごとに「どの seed が完遂したか」だけ見る (Notion の表と同じ粒度)
python tools/collect_runs.py -c tools/collect_config.yaml --format summary

# Notion に貼れる Markdown を吐く
python tools/collect_runs.py -c tools/collect_config.yaml --format markdown -o runs.md

# Notion DB に同期 (差分のあるページだけ更新)
export NOTION_TOKEN='ntn_xxxxxxxx'
python tools/collect_runs.py -c tools/collect_config.yaml --notion --format none

# 何が書き込まれるか先に見る
python tools/collect_runs.py -c tools/collect_config.yaml --notion-dry-run --format none

# 異常な run だけ (落ちた / 止まった / t_max に届いていない)
python tools/collect_runs.py -c tools/collect_config.yaml --state stalled,failed,short

# 完遂した run の方策モデルを回収 (先に何が来るか見る)
python tools/collect_runs.py -c tools/collect_config.yaml --format none \
    --fetch-models models_inbox --fetch-dry-run

# 実際に回収
python tools/collect_runs.py -c tools/collect_config.yaml --format none \
    --fetch-models models_inbox

# push モード (そのマシンの分だけを直接 Notion へ)
python tools/collect_runs.py --hosts local --machine GPU1 --repo ~/LDRP \
    --notion --notion-db 87654321-... --format none
```

### 主なオプション

| オプション | 既定 | 意味 |
|---|---|---|
| `--format` | `table` | `table` / `markdown` / `csv` / `jsonl` / `summary` / `none` |
| `--min-steps` | `1e6` | `t_max` がこれ未満の run を隠す (デバッグ run 除け)。`0` で全件 |
| `--since-days` | なし | 直近 N 日に開始した run だけ |
| `--state` | なし | `done,running,stalled,short,failed` から絞る |
| `--stale-minutes` | 90 | RUNNING なのに heartbeat がこれだけ止まっていたら `STALL` |
| `--check` | off | 異常 run が 1 件でもあれば exit 1 (cron の通知条件に使える) |
| `--cache PATH` | なし | 収集結果を JSONL に貯めて次回とマージ。落ちているホストの行が消えなくなる |
| `--tail-bytes` | 65536 | `cout.txt` を末尾何バイト読むか |
| `--export DIR` | — | 自分の run 一覧と完遂モデルを共有フォルダ `DIR` に書き出す (drop モード) |
| `--drop-root` | — | `drop: true` のホストが読む共有フォルダ (config の `drop_root` を上書き) |
| `--notion-check` | — | token と DB とプロパティの過不足を確認して終了 |
| `--notion-token-file` | — | token をこのファイルから読む |
| `--bootstrap LABELS` | — | 各マシンに push 一式を設置 (`all` で全 ssh ホスト) |
| `--bootstrap-install-cron` | off | crontab への登録までやる |
| `--quick` | off | キャッシュ上で実行中の run だけ読み直して通知して終了 (2 分おき用) |
| `--notify` | off | run が完了 / 異常になったらデスクトップ通知を出す (`--cache` を暗黙に使う) |
| `--format status` | — | 常駐パネル向けの要約 (xbar のプラグイン出力と同じ形) |
| `--fetch-models DIR` | off | 完遂 run の最終ステップの方策モデルを `DIR` に回収 |
| `--fetch-state` | `done` | どの状態の run のモデルを取るか |
| `--fetch-what` | `path` | `path` = 経路方策のみ / `all` = タスク割当器 (`task/`) も |
| `--fetch-optimizer` | off | `opt.th` 系も取る (評価には不要なので既定で除外) |
| `--max-fetch-mb` | 2000 | 1 回の実行で取る合計サイズの上限 |
| `--fetch-dry-run` | off | 何を取るかだけ表示して転送しない |
| `--publish-models REPO` | — | 保管用リポジトリの階層へ配置し `manifest.jsonl` に追記 |
| `--publish-commit` | off | 配置後に commit する (変更が無ければ何もしない) |
| `--publish-push` | off | push まで行う |
| `--install-models` | off | 回収したモデルを評価用の名前で `src/all_policy/models/safe/` に設置 |
| `--models-dir` | `src/all_policy/models/safe` | `--install-models` の設置先 |
| `--overwrite-installed` | off | 設置先に同名ファイルがあっても上書きする |

### 状態の意味

| 表示 | 判定 | 意味 |
|---|---|---|
| `OK` | sacred `COMPLETED` かつ `Finished Training` (または t_env が t_max の 99% 以上) | **設定ステップまで完遂** |
| `SHORT` | `COMPLETED` だがログ上 t_max に届いていない | 異常終了。要確認 |
| `RUN` | `RUNNING` かつ heartbeat が新しい | 学習中 (progress 列が進捗) |
| `STALL` | `RUNNING` だが heartbeat が `stale_minutes` 以上止まっている | プロセスが落ちた / マシンが再起動した |
| `FAIL` | sacred `FAILED` / `INTERRUPTED` | 例外か Ctrl-C |

### 学習終了予定時刻 (eta 列)

進行中の run について、**実測ペース**から残り時間と終了予定時刻を出す。

```text
eta = heartbeat + (t_max - t_env) / (t_env / (heartbeat - start_time))
                                     └ ここまでの平均 step/sec
```

- 時刻は `run.json` の `start_time` と `heartbeat` を使う。どちらも sacred が書く UTC
  なので時計系が揃う (`cout.txt` の mtime はリモートの時計なので使わない)
- **`RUN` の run にだけ出す**。停止した run のペースから作った予定時刻は誤解を招くので、
  `OK` / `STALL` / `FAIL` / `SHORT` では空欄になる
- あくまで**開始からの平均ペース**での外挿。マシンの負荷が変われば当然ずれる
  (GPU2 は load average 34 まで上がることがある)

`OK` の根拠は epymarl が `while t_env <= t_max` を抜けた直後に出す `Finished Training`
([src/epymarl/src/run.py](../src/epymarl/src/run.py))。GPU 機のように stdout をシェルで
リダイレクトしていて `cout.txt` が 0 byte のときは、`metrics.json` の末尾から最後の t_env を拾い、
それも取れなければ sacred の `COMPLETED` (= 学習ループを正常に抜けた) を信頼する。

---

## 学習済みモデルの回収

epymarl は `{repo}/results/models/{unique_token}/{t_env}/` にモデルを保存し、
`unique_token = "{algo}_seed{seed}_{env_key}_{起動時刻}"` ([src/epymarl/src/run.py](../src/epymarl/src/run.py))
なので、**seed 込みで sacred の run と 1:1 に紐づく**。これを使って
「完遂した run の最終ステップだけ」を選んで持ってくる。

```text
models_inbox/
├── manifest.jsonl                # 回収したものの記録 (uid / seed / step / files)
├── install_hints.sh              # 評価用ディレクトリへの cp コマンド案 (実行はしない)
├── GPU1/
│   └── qmix_map_8x5_7agents_seed724865803_step30000500/
│       └── path/agent.th
└── 白/
    └── qmix_map_8x5_5agents_seed113162076_step20000500/
        └── path/agent.th
```

- **既に置いてあるものは再取得しない**ので、cron で回し続けても転送は初回だけ
- `opt.th` / `agent_opt.th` / `critic_opt.th` (optimizer state) は評価に要らないので既定で除外。
  実測で done 53 run が **25MB** 程度に収まる (optimizer 込みだと数 GB になる)
- 転送は **ssh 越しの tar ストリーム 1 本**。`unique_token` に空白と `:` が入るため、
  ファイル名は NUL 区切りで `tar --null -T -` に渡す。tar の出力はローカルの tar へ直結するので
  転送内容が Python のメモリに載らない

### 保管用リポジトリへの配置 (`--publish-models`)

モデルは **別の private リポジトリ** に置く。`origin` (Yamaguchi-yushi/LDRP) は PUBLIC で、
学会でコードを公開する前提なので、モデルを同じリポジトリに入れると公開範囲を分けられない。

```bash
# 1 回だけ: private リポジトリを作って clone
gh repo create LDRP_models --private
git clone https://github.com/<user>/LDRP_models.git ~/LDRP_models

# 以降: 回収したモデルを階層に配置する (commit / push はしない)
python tools/collect_runs.py -c tools/collect_config.yaml --format none \
    --fetch-models models_inbox --publish-models ~/LDRP_models
```

```text
LDRP_models/
├── manifest.jsonl        ← インデックス (param_hash 付き)
├── path/                 ← 経路方策
│   └── map_8x5/7agent/qmix_safe/
│         map_8x5_7_qmix_safe_base_seed236379847__30000500_606fa0d5/
│           ├── agent.th
│           └── mixer.th
├── lare_path/            ← LaRe 報酬モデル (名前に全情報があるのでフラット)
└── task/                 ← PPO タスク割当器
```

**階層**: `path/{map}/{N}agent/{planner}_{method_tag}/`。マップ数 5、台数 3〜4、planner 5、
tag 3 程度なので各階層は少数に収まる。

**リーフ名**: `{評価用の名前}__{step}_{param_hash}`

- `__` より前は**評価用のファイル名そのまま**。設置側は `__` で切るだけで戻せる
- `step` があるので同じ seed の複数 checkpoint を並べられる
- `param_hash` があるので、**フォルダを見るだけで 5 seed のパラメータが揃っているか**分かる

```text
$ ls path/map_8x5/5agent/qmix_safe/ | sed 's/map_8x5_5_qmix_safe_base_//'
seed113162076__20000500_464b470b
seed243661819__20000500_464b470b
seed410452773__20000500_464b470b   ← ハッシュが全部同じ = 条件がそろっている
seed421044850__20000500_464b470b
seed659065431__20000500_464b470b
```

`opt.th` は評価に不要でサイズが数倍なので入れない。

#### git で足りるのか (実測)

```text
1 モデルのサイズ : 最小 0.03MB / 中央 0.15MB / 平均 0.71MB / 最大 3.62MB
GitHub の制限    : ファイル 50MB で警告、100MB で push 拒否  → 最大でも 1/28
リポジトリ全体   : 78MB (210 run) → 将来 500 run でも 200〜300MB (推奨 1GB)
```

**足りる。** seed ごとに別ファイルで**更新しない (append-only)** ため履歴も膨らまない。
git が苦しくなるのは `opt.th` (数倍) や中間 checkpoint (実測 12GB) を入れる場合だけで、
どちらも元々入れない設計。

将来 1GB を超えたら **Hugging Face Hub** (private 可・モデル向けで容量が緩い・git-lfs ベース)
へ移すのが移行コストが小さい。`git remote` を差し替えるだけで済む。

#### commit / push まで自動化する

```bash
python tools/collect_runs.py -c tools/collect_config.yaml --format none     --fetch-models models_inbox --publish-models ~/LDRP_models     --publish-commit          # commit まで (手元に留まる)
    # --publish-push          # push まで
```

安全側に倒してある。

| 動作 | 理由 |
|---|---|
| 変更が無ければ commit しない | 定期実行で空コミットが積み上がらない |
| **45MB を超えるファイルがあれば中止** | GitHub が 100MB で push を拒否する。積んでから気づくと履歴から消す羽目になる |
| `--publish-push` は別フラグ | commit だけなら手元で確認してから push できる |
| remote が無ければ push しない | 設定漏れで失敗し続けるのを防ぐ |

既定では commit しない (コマンドを表示するだけ)。定期実行に組み込むなら
`--publish-commit` を、完全自動にするなら `--publish-push` を足す。

> **既存の 22 件は移行できない。** `src/all_policy/models/safe/` にあるモデルは
> どの run から来たか記録が無く、`step` も `param_hash` も復元できない
> (`map_8x5_5_qmix_base.th` のように seed suffix すら無いものもある)。
> `legacy/` に名前のまま置くしかない。今後回収する分から新規則が効く。

### 評価用ディレクトリへの設置 (`--install-models`)

評価側 ([src/all_policy/policy.py](../src/all_policy/policy.py)) が探すのは

```text
src/all_policy/models/safe/{map}_{model_n}_{path_planner}[_{method_tag}]_{reassign_tag}_seed{seed}.th
```

で、中身は epymarl の `path/agent.th` (RNNAgent の state_dict) そのまま。
`--install-models` を付けると、この名前で `src/all_policy/models/safe/` に設置するところまでやる。

```bash
python tools/collect_runs.py -c tools/collect_config.yaml --format none \
    --fetch-models models_inbox --install-models
# agent.th -> map_8x5_5_qmix_safe_base_seed113162076.th
# agent.th -> map_aoba00_7_qmix_dbct_base_seed740597639.th
```

- 同名ファイルが既にあれば**上書きしない** (`--overwrite-installed` で上書き)
- 置き場所は `--models-dir` で変えられる
- `--fetch-dry-run` と併用すると、設置するファイル名だけ確認できる

#### method_tag の決まり方

`method_tag` (`dbct` / `safe`) は学習 config に直接は入っていないが、
**「事前学習した LaRe モデルを使ったか」から決まる**ので機械的に付けられる。
規則は [collect_config.yaml](collect_config.yaml) の `method_tag_by_lare_mode` にある。

| lare mode | 由来 | method_tag |
|---|---|---|
| `pretrained` / `finetuning` | LaRe モデルを読み込んで学習 | `dbct` |
| `scratch` / `scratch(frozen)` / `off` | 事前学習モデル無し | `safe` |
| (非 Safe 環境 `drp-*` で学習) | — | `unsafe` |

別の軸 (タスク割当の同時学習など) で分けたくなったら、この表を書き換える。
`reassign_tag` は学習時に決まる軸ではないので常に `base` にしてある
(reassign 側のモデルが要るときは手でリネームする)。

`--install-models` を使わない場合は、`models_inbox/install_hints.sh` に
同じ内容の `cp` コマンドが書き出されるので、目視してから実行してもよい。

---

## 完了ポップアップと常駐パネル

### 二段構えにする理由

「学習が終わったらすぐ知りたい」と「Notion もモデルも最新にしたい」は、必要な重さが違う。

| | 何を読むか | 所要 | 間隔 | 役割 |
|---|---|---|---|---|
| **quick** (`--quick`) | キャッシュ上で実行中の run **だけ** (数件) | 秒 | 2 分 | 完了ポップアップ |
| **full** (通常) | 全 run (実測 490 件) + モデル索引 | 1〜2 分 | 30 分 | Notion / モデル回収 / パネル |

`--quick` はモデル索引 (`models/` の listdir) も作らないので、2 分おきに回しても負荷にならない。
ssh は `ControlMaster` で接続を使い回すため、2 回目以降は TCP と認証をやり直さない。

```bash
python tools/collect_runs.py -c tools/collect_config.yaml --quick --notify
# [quick] checked 8 running run(s), 1 changed
```

> **完全な「即時」にはできない**。学習側にフックを入れれば可能だが、
> `train.py` / `src/` には手を入れない方針なので、2 分粒度が現実的な下限。
> 17 時間の学習に対しては十分な粒度と考えている。

### 完了ポップアップ (`--notify`)

前回の収集結果 (キャッシュ) と比べて **状態が変わった run だけ** を通知する。

| 遷移 | 通知 |
|---|---|
| → `done` | `Training finished (2)` + Glass 音 |
| → `stalled` / `failed` / `short` | `Runs need attention (1)` + Basso 音 |
| 変化なし | 出さない |

- **初回 (キャッシュが空) は何も出さない**。そうしないと既存の数百件が全部
  「今 done になった」ものとして飛んでくる
- 複数件まとまったときは 1 通にまとめる (先頭 3 件 + `+N more`)
- 前回の状態はキャッシュに `_state` として保存し、そのまま比較する。
  再導出すると「到達できないホストの `running` が時間経過で `stalled` に変わる」等でぶれるため
- macOS は `osascript`、Linux は `notify-send`、どちらも無ければ標準エラーに出す

### 学習パラメータと seed の保存

`config.json` を **whitelist せず全文** 保持する。落とすと再現できない
(`lr` / `gamma` / `batch_size` / `epsilon_*` / `mixer` / `hypernet_*` / `obs_agent_id` /
`exclude_station_from_tasks` など)。1 run 2.5KB、実測 490 run でも 1.2MB なので容量は問題にならない。

全文は `--format jsonl` で書き出せる。これがそのままアーカイブになる。

```bash
python tools/collect_runs.py -c tools/collect_config.yaml --format jsonl -o runs_full.jsonl
```

#### param_hash — 条件としての同一性

seed とパスを除いた config から 8 桁のハッシュを作る。**同じ条件のはずの 5 seed で
ハッシュが割れていたら、どこかの設定が食い違っている**と機械的に分かる。

```text
10agent map_8x5 50M | qmix | safe | bernoulli, mmpp | -   n=5  OK=5
    done seeds: 21003671, 260286646, 453153204, 579160610, 773408812
    [warn] params differ across seeds: 606fa0d5(4)  5a279e5d(1)
```

ハッシュを作るときに 2 つ正規化している。

| 正規化 | 理由 |
|---|---|
| **`None` / `False` は「キーが無い」と同一視** | config のスキーマは版ごとに増えており、古い run にはキー自体が無い。新設フラグの既定は `False` / `None` なので、潰さないと「後から足したキーの有無」だけでハッシュが割れる (実測で 20 条件が誤検出された)。`0` や `""` は意図した設定値なので残す |
| **有効化フラグが off のとき無視されるキーを落とす** | `use_lare_path=False` の run に `use_finetuning_lare_path=True` が残っている事例が実在する。消し忘れで条件が違うと判定されるのを防ぐ |

Notion にも `param hash` 列として出るので、表の上で 5 seed のハッシュが揃っているか目視できる。

### train.py のバッチ検出

`ps` から `train.py` の実行を拾い、マシン別に「実行中 / 予約待ち」を集約する。

```text
▶ 1 +4  next 02:51
---
白       1 run  +4 wait   next 02:51   (train.py x1)
GPU2     8 run                next 14:20
```

sacred は run が始まって初めてディレクトリを作るので、これが無いと
「5 本連続実行の 2 本目」の状態が 1 本にしか見えない。

- 残り本数は `train.py` が `~/.ldrp/batch_<pid>.json` に書き出したときだけ分かる。
  **まだ入れていないので現状は `+3?` のように `?` が付く**
  (入れ方は [design/run_collector.md §12.3](../design/run_collector.md#123-trainpy-の実行予定を枠として見せる))
- プロセスが死んでいる予約ファイルは無視する
- バッチ情報はキャッシュしない (古い予約で枠が埋まったままにならないように)

### 常駐パネル (`--format status`)

```text
▶ 3  next 08/31 02:51  ⚠ 2
---
 77%  qmix     7ag map_aoba00  5M     seed594911358  @白      1h19m → 08/31 02:51
  2%  qmix     7ag map_8x5     30M    seed631380331  @GPU2   32h06m → 09/01 09:38
 19%  mat     10ag map_aoba00  150M   seed383111932  @GPU2   54h15m → 09/02 07:47
---
STALL 1  FAIL 1
done 281 / 286 runs   updated 08/31 01:32
```

終了予定の早い順に並ぶ。1 行目だけでメニューバーに収まる長さにしてある。

ヘッダの記号:

| 表示 | 意味 |
|---|---|
| `▶ 3` | 実行中の本数 |
| `+4` | `train.py` のバッチが予約している残り本数 |
| `⚠ 2` | `STALL` / `FAIL` / `SHORT` の合計 |
| `t_max✗ 3` | `expected_t_max` とズレている run |
| `params✗ 1` | 同じ条件の多数派と `param_hash` が違う run (= 要再実行) |

`params✗` は **`done` と `running` の両方**を見る。17 時間回してから「設定が違った」と
気づくより、走り出してすぐ止めるほうがよいため。
**判断して止めるのは GUI 側の役目**で、`--publish-models` はハッシュでは弾かない
(弾くのは `state != done` のときだけ)。方針は
[design/run_collector.md §12.2](../design/run_collector.md#122-実験管理-gui) にある。

**収集 (遅い) と表示 (速い) を分けるのが肝**。全ホストへの ssh は 1〜2 分かかるので、
パネル側が毎回それを待つと固まる。launchd が定期的に収集してファイルに書き、
パネルはそれを読むだけにする。

```text
launchd (15 分ごと)                    パネル (数秒ごと)
  収集 → 通知 → Notion → モデル回収        ~/.ldrp/status.txt を読むだけ
  → ~/.ldrp/status.txt に書き出し    ────>  (一瞬で終わる)
```

[com.ldrp.collect-runs.plist](com.ldrp.collect-runs.plist) はこの形にしてある
(`--notify --format status --out ~/.ldrp/status.txt`)。

定期実行は 2 本立てになる。

| plist | 間隔 | やること |
|---|---|---|
| [com.ldrp.quick-check.plist](com.ldrp.quick-check.plist) | **2 分** | 実行中の数件だけ確認 → 完了/異常のポップアップ |
| [com.ldrp.collect-runs.plist](com.ldrp.collect-runs.plist) | **30 分** | 全収集 → Notion → モデル回収 → パネル更新 |

```bash
cp tools/com.ldrp.quick-check.plist tools/com.ldrp.collect-runs.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.ldrp.quick-check.plist
launchctl load ~/Library/LaunchAgents/com.ldrp.collect-runs.plist
```

#### 表示方法の選択肢

**A. メニューバー常駐 (xbar)** — 一番「常駐」らしい。`--format status` の出力は
そのまま xbar のプラグイン形式 (1 行目 = メニューバー、`---` 以降 = ドロップダウン)。

```bash
brew install --cask xbar
mkdir -p ~/Library/Application\ Support/xbar/plugins
cat > ~/Library/Application\ Support/xbar/plugins/ldrp.1m.sh <<'EOF'
#!/bin/bash
cat "$HOME/.ldrp/status.txt" 2>/dev/null || echo "LDRP: no data"
EOF
chmod +x ~/Library/Application\ Support/xbar/plugins/ldrp.1m.sh
```

**B. ターミナル常駐** — インストール不要。ウィンドウを 1 つ開いておくだけ。

```bash
while true; do clear; cat ~/.ldrp/status.txt; sleep 30; done
```

**C. メモ.app に書き込む** — 標準アプリのみ。`Notes.app` は AppleScript に対応しているので、
本文を差し替えれば「常駐メモ」になる (スティッキーズは AppleScript 非対応なので不可)。

---

## 定期実行

### macOS (Mac から pull する / 推奨)

**1 時間ごと**に「全ホスト巡回 → Notion 更新 → 完遂モデル回収」を回す定義を
[com.ldrp.collect-runs.plist](com.ldrp.collect-runs.plist) に用意してある。

```bash
cp tools/com.ldrp.collect-runs.plist ~/Library/LaunchAgents/
# NOTION_TOKEN の値を書き換える
vi ~/Library/LaunchAgents/com.ldrp.collect-runs.plist

launchctl load ~/Library/LaunchAgents/com.ldrp.collect-runs.plist
launchctl start com.ldrp.collect-runs      # 初回を即実行して確認
tail -f /tmp/ldrp_collect.log
```

止めるとき:

```bash
launchctl unload ~/Library/LaunchAgents/com.ldrp.collect-runs.plist
```

> `StartInterval` は 3600 秒。Mac がスリープしている間は動かず、復帰後の次の回で実行される。
> ホストに繋がらなかった回は警告を出して他のホストだけ更新する (`--cache` を付けてあるので、
> 繋がらなかったホストの行も Notion から消えない)。

### Ubuntu / GPU 機 (各マシンが自分で Notion に書く / push)

**Mac がスリープしていても各マシンが自分で書き込む**ようにしたい場合はこちら。
`--bootstrap` が、スクリプト本体・token・起動スクリプトをそのマシンの `~/.ldrp/` に置く。
リポジトリを `git pull` させる必要はない (スクリプト自身を ssh で送り込む)。

```bash
# cron 行を表示するだけ (何も登録しない)
python tools/collect_runs.py -c tools/collect_config.yaml --bootstrap GPU1,GPU2

# crontab への登録までやる
python tools/collect_runs.py -c tools/collect_config.yaml --bootstrap GPU1,GPU2 \
    --bootstrap-install-cron
```

置かれるもの:

```text
~/.ldrp/collect_runs.py     このスクリプト本体
~/.ldrp/notion_token        token (chmod 600)
~/.ldrp/push.sh             --hosts local --machine <label> --notion を叩くだけの起動スクリプト
```

`--bootstrap-interval-min` で頻度を変えられる (既定 60 分)。
`--bootstrap all` で config の全 ssh ホストに入れる (drop / local のホストは自動で飛ばす)。

> **共用マシンの crontab を書き換える操作**なので、既定では登録せず行を表示するだけにしてある。
> 内容を確認してから `--bootstrap-install-cron` を付けること。

手で書く場合の cron 行:

```bash
0 * * * * ~/.ldrp/push.sh >> /tmp/ldrp_push.log 2>&1
```

> push モードで Notion に入るのは **run の行だけ**で、モデルは集まらない。
> モデルも欲しいなら pull か drop を使う。

---

## 仕組み

```text
Mac                                    GPU1 / GPU2 / Ubuntu
 collect_runs.py
   |
   |  ssh HOST 'python3 - --scan ...' < collect_runs.py
   |------------------------------------------->  python3 (stdin からスクリプトを実行)
   |                                                 |
   |                                                 |  config.json  (~2.5KB)
   |                                                 |  run.json     (~5KB)
   |                                                 |  cout.txt     (末尾 64KB だけ)
   |                                                 |  metrics.json (末尾 256KB / 必要時のみ)
   |  <-------------------------------------------  JSONL を 1 run 1 行で stdout へ
   |
   |  条件を組み立てて集計 -> table / markdown / Notion upsert
```

drop モードでは ssh の代わりに共有フォルダを挟む。相手のマシンが
`--export` で `runs.jsonl` とモデルを置き、集約側はそれを読むだけ。

```text
[相手のマシン]  --export -->  共有フォルダ/<label>/runs.jsonl
                                        /<label>/models/<run>/path/agent.th
[集約する Mac]  drop: true  <--  同じフォルダを読む
```
