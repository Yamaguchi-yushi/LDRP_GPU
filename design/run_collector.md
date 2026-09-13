# 学習 run 収集ツール (tools/collect_runs.py) の実装

実験結果を Notion に手で書き写す作業をなくすために追加したツールの、**実装の説明**。
使い方の手順は [tools/README.md](../tools/README.md) にある。ここは「なぜその作り方にしたか」を残す。

**このツールは学習・実行系のコードを一切変更していない**。sacred が既に吐いている出力を
読むだけで、`train.py` / `test.py` / `run.py` / `runner.py` / `src/` 配下には触れていない。

---

## 目次

1. [このツールが解決すること](#1-このツールが解決すること)
2. [全体構成](#2-全体構成)
3. [3 つの収集モード](#3-3-つの収集モード)
4. [低メモリで読む工夫](#4-低メモリで読む工夫)
5. [完遂判定 state](#5-完遂判定-state)
6. [実験条件の導出](#6-実験条件の導出)
7. [学習済みモデルの回収](#7-学習済みモデルの回収)
8. [Notion 同期](#8-notion-同期)
9. [ファイルと主な関数](#9-ファイルと主な関数)
10. [設計上の判断と理由](#10-設計上の判断と理由)
11. [未検証の部分と既知の制約](#11-未検証の部分と既知の制約)
12. [将来の実装](#12-将来の実装)
    - [12.1 学習スイープ train_sweep](#121-学習スイープ-train_sweep)
    - [12.2 実験管理 GUI](#122-実験管理-gui)
    - [12.3 train.py の実行予定を枠として見せる](#123-trainpy-の実行予定を枠として見せる)
    - [12.4 評価結果ダッシュボード](#124-評価結果ダッシュボード)
    - [12.5 実験計画の複数ファイル化 (plan_AAMAS.md / plan_XXX.md)](#125-実験計画の複数ファイル化-plan_aamasmd--plan_xxxmd)

---

## 1. このツールが解決すること

| 手作業だったこと | ツールでの扱い |
|---|---|
| seed と条件を Notion に転記する | sacred の `config.json` から自動生成して upsert |
| 設定した `t_max` まで回ったか目視する | `OK` / `SHORT` で自動判定 |
| 落ちた run に気づかない | `STALL` (heartbeat 停止) / `FAIL` で表面化 |
| 複数マシンのログを個別に見る | 1 つの表にまとめる |
| 完遂したモデルを scp する | `--fetch-models` で自動回収、`--install-models` で評価用の名前に設置 |

実測 (2026-08-28 時点、白 / GPU1 / GPU2 の 3 台):

```text
490 runs  FAIL=122  STALL=75  SHORT=1  RUN=11  OK=281
モデル回収: done 210 run 分で 78MB
```

---

## 2. 全体構成

```text
tools/
├── collect_runs.py               # 本体 (単一ファイル。ローカル側とリモート側を兼ねる)
├── collect_config.yaml           # ホスト定義 / method_tag 規則 / Notion 設定
├── com.ldrp.collect-runs.plist   # 集約側の launchd (1 時間ごと)
├── com.ldrp.export-runs.plist    # drop モードの相手側 launchd (1 時間ごと)
└── README.md                     # 使い方
```

処理は 4 段。前段の出力が次段の入力になるだけで、段どうしは疎結合にしてある。

```text
[1] 収集      各マシンの sacred を読んで 1 run = 1 レコード (JSONL) にする
     ↓
[2] 導出      レコードから「条件」と「状態」を組み立てる (derive)
     ↓
[3] 出力      table / markdown / csv / jsonl / summary
     ↓
[4] 反映      Notion upsert / モデル回収 / 評価用ディレクトリへ設置
```

段を分けた理由は、**収集だけをリモートで走らせたいから**。[1] は標準ライブラリだけで動くよう
書いてあり、[2]〜[4] は集約するマシンでしか動かない (PyYAML / Notion API を使う)。
`import yaml` を関数の中に置いてあるのはこのため。

---

## 3. 3 つの収集モード

| モード | 収集経路 | 相手側に要るもの | モデルも集まるか |
|---|---|---|---|
| **pull** | 集約 Mac → ssh → 相手 | `python3` のみ | ○ |
| **drop** | 相手 → 共有フォルダ → 集約 Mac | `python3` + 共有フォルダ | ○ |
| **push** | 相手 → Notion (直接) | `python3` + Notion token | × (行のみ) |

3 つとも `run_uid` をキーに upsert するので、**混ぜても Notion に重複が出ない**。

```text
run_uid = {machine}:{results|tmp_results}/{algo}/{env_key}/{run_id}
          例: GPU1:results/qmix/drp_env:drp_safe-7agent_map_8x5-v2/11
```

### pull: スクリプト自身を ssh の stdin で送る

リモートに何もインストールしないために、`python3 -` (stdin からプログラムを読む) を使う。

```text
ssh HOST 'python3 - --scan --repo ~/LDRP ...' < tools/collect_runs.py
                    ↑ このスクリプト自身が stdin から流れる
```

リモート側は `--scan` モードで動き、1 run 1 行の JSON を stdout に吐く。
`json.dumps(..., ensure_ascii=True)` にしてあるので、リモートのロケールが `C` でも
`UnicodeEncodeError` にならない。

### drop: 共有フォルダを 1 つ挟む

SSH が通らないマシン用。共有フォルダは iCloud Drive / Dropbox / NFS / USB など何でもよい
(置くのはただのファイルなので、同期の仕組みを問わない)。

```text
[相手のマシン]  --export DIR  -->  DIR/<label>/runs.jsonl
                                   DIR/<label>/models/<run>/path/agent.th
[集約する Mac]  drop: true    <--  同じフォルダを読む
```

相手側の `--export` は **Notion にも外部にも一切アクセスしない**。フォルダに書くだけ。
`runs.jsonl` は `.tmp` に書いてから `os.replace` で差し替えるので、
同期中の中途半端なファイルを読むことがない。

### push: 各マシンが直接 Notion に書く

`--bootstrap` でスクリプト・token・`push.sh` を相手に置き、cron 行を出す
(`--bootstrap-install-cron` を付けたときだけ crontab にも登録する)。
git pull を相手にさせなくて済むよう、ここでもスクリプト自身を送り込んでいる。

---

## 4. 低メモリで読む工夫

sacred の run ディレクトリは 4 ファイルあるが、**大きい 2 つは開かない**。

| ファイル | 実測サイズ | 扱い |
|---|---|---|
| `config.json` | ~2.5 KB | 全部読む (必要キーだけ残す) |
| `run.json` | ~5 KB | 全部読む |
| `cout.txt` | 0〜650 KB | **末尾 64KB だけ** (`_tail_text`) |
| `metrics.json` | 1〜5 MB | **末尾 256KB だけ**、しかも `cout.txt` が空のときだけ |
| `info.json` | 2.5 MB | **開かない** |

`_tail_text` は `os.path.getsize` → `f.seek(size - n)` → `f.read()` なので、
ファイルがどれだけ大きくても読むのは n バイト。

転送も溜め込まない。リモートは `emit(rec)` で 1 行ずつ stdout に流し、
集約側は `Popen` の出力を行単位で `json.loads` する。
モデル転送は ssh の tar 出力をローカルの tar の stdin へ **OS レベルで直結**しているので、
転送内容が Python のメモリに載らない。

```python
untar = subprocess.Popen(["tar", "-xf", "-", "-C", tmp_dir], stdin=PIPE)
send  = subprocess.Popen(ssh_cmd, stdin=PIPE, stdout=untar.stdin)   # ← 直結
```

---

## 5. 完遂判定 state

「設定したステップ数まで完遂しているか」を機械判定する部分。

| 表示 | 条件 | 意味 |
|---|---|---|
| `OK` | sacred `COMPLETED` かつ (到達の証拠あり または 証拠が取れない) | 完遂 |
| `SHORT` | `COMPLETED` だが**ログ上**明確に `t_max` 未達 | 異常終了。要確認 |
| `RUN` | `RUNNING` かつ heartbeat が新しい | 学習中 |
| `STALL` | `RUNNING` だが heartbeat が `stale_minutes` 以上停止 | プロセスが落ちた / マシン再起動 |
| `FAIL` | `FAILED` / `INTERRUPTED` | 例外 or Ctrl-C |

### 到達の根拠を 3 段構えにした理由

epymarl は `while runner.t_env <= args.t_max:` を抜けた直後に `Finished Training` を出す
([src/epymarl/src/run.py](../src/epymarl/src/run.py#L291))。これが一番強い証拠だが、
**GPU 機では stdout をシェルでリダイレクトしているため `cout.txt` が 0 バイト**で取れない。
そこで次の順に落とす。

1. `cout.txt` の末尾に `Finished Training` がある → 完遂
2. `cout.txt` / `metrics.json` から拾った最後の `t_env` が `t_max - margin` 以上 → 完遂
3. どちらも取れず sacred が `COMPLETED` → 完遂とみなす
   (sacred の `COMPLETED` は「学習ループを正常に抜けた」= `t_max` 到達を意味する。
   途中で殺された run は `RUNNING` のまま残るか `INTERRUPTED` になる)

`margin = max(t_max * 0.01, 50000)` としたのは、`t_env` のログが `log_interval` 毎にしか
出ないため。最終行が `t_max` をわずかに下回るのは正常
(実測: `t_last=8003659 / t_max=8050000` は完遂した run)。

### metrics.json から t_env を拾う正規表現

```python
ARRAY_INT_RE = re.compile(r"[\[,]\s*(\d{4,})(?=\s*[,\]])")
```

**JSON 配列の要素になっている整数だけ**に当てている。前後を `[` `,` `]` で挟むことで、

- 文字列中の `"2026-08-19T..."` の **年 (2026) を拾ってしまう事故**を防ぐ
  (実際、素朴な `\d{4,}` では全 run の `t_env` が `2026` になった)
- 小数は除外される (json は `1024.0` と必ず小数点付きで書く)

拾った候補のうち `t_max * 1.01` 以下の最大値を `t_env` とみなす。

---

## 6. 実験条件の導出

Notion の列は、すべて `config.json` から機械的に作る (`derive`)。

| Notion 列 | 導出元 | 例 |
|---|---|---|
| `seed` | `config.seed` | `113162076` |
| `machine` | 収集時のホストラベル | `GPU1` |
| `algorithm` | `config.name` | `qmix` |
| `agents` / `map` | `env_args.key` を正規表現で分解 | `7` / `map_8x5` |
| `steps (M)` | `config.t_max` | `30.05` |
| `task arrival` | `randomize_task_arrival` が真なら `bernoulli, mmpp`、偽なら `task_arrival` | `bernoulli, mmpp` |
| `task assign` | `train_task_assigner` | `PPO` / 空 |
| `eta` | 実測ペースからの外挿 (下記) | `09/01 03:40` |
| `lare mode` | LaRe フラグ | `off` / `scratch` / `pretrained` / `finetuning` |
| `setting` | 下記 | `safe` / `8x5_2 10M 8x5_3 5M` |
| `method tag` | `lare mode` から規則で決める | `dbct` / `safe` / `unsafe` |

### eta 列 (学習終了予定時刻)

```text
rate = t_env / (heartbeat - start_time)          ← 開始からの平均 step/sec
eta  = heartbeat + (t_max - t_env) / rate
```

時刻は `run.json` の `start_time` と `heartbeat` を使う。**どちらも sacred が書く UTC
なので時計系が揃う**のが理由で、`cout.txt` / `metrics.json` の mtime はリモートの時計に
依存するため使わない (マシン間で時刻がずれていると予定時刻が壊れる)。

**`running` の run にだけ出す**。停止した run のペースから作った予定時刻は
「もう動いていないのに終わりそうに見える」ので、`done` / `stalled` / `failed` / `short`
では `None` にしてある。

検算 (合成データ): 6 時間で 10.0M step 進んだ 30.05M の run
-> 463 step/sec、残り 12h01m。20.05M / 463 = 12.03h と一致する。

### 完了検知を二段にした理由

「終わったらすぐ知りたい」と「Notion もモデルも最新に」は必要な重さが違うので、
ポーリングを 2 本に分けた。

| | 読むもの | 実測 | 間隔 |
|---|---|---|---|
| `--quick` | キャッシュ上で `running` の run **だけ** (数件) | 秒 | 2 分 |
| 通常 | 全 run (490 件) + モデル索引 | 1〜2 分 | 30 分 |

`--quick` は `--scan-dir` でリモートに「この run ディレクトリだけ読め」と指示する。
`models/` の listdir も行わない。ssh は `ControlMaster` + `ControlPersist=10m` で
マスタ接続を使い回すので、2 回目以降は TCP も認証もやり直さない
(GPU2 は load average 34 まで上がるので、毎回接続し直すと数十秒かかる)。

軽い問い合わせでは `repo` と `model` を取らないため、キャッシュへ書き戻すときに
**古いレコードの `repo` / `model` を引き継ぐ**。これを忘れると、次のモデル回収で
「どのホストのどのパスから取ればいいか」が失われる。

**完全な即時にはできない**。学習側にフック (epymarl の終了時に何か叩く) を入れれば
可能だが、`train.py` / `src/` に手を入れない方針なので 2 分粒度が下限。
17 時間の学習に対しては十分と判断した。

### 完了通知の差分判定

`--notify` は「前回の収集からの状態変化」だけを通知する。前回の state は
**キャッシュに `_state` として保存したものをそのまま使う**。
再導出しないのは、`running` のレコードを時間が経ってから derive し直すと
heartbeat が古くなって `stalled` に化け、到達できないホストの run が
毎回「異常になりました」と通知されてしまうため。

初回 (キャッシュが空) は通知を出さない。出すと既存の数百件が全部
「今 done になった」ものとして飛んでくる。

検証: 前回状態を `done x2 / failed x1 -> running` に書き換えて再実行すると、
`Training finished (2)` と `Runs need attention (1)` の 2 通が飛び、
前回も `done` だったものは再通知されない。

### setting 列は LaRe モデル名から復元する

`pretrained` / `finetuning` のときは、使ったモデル名を Notion の表記に合わせて畳む。

```text
FT_QMIX_PATH_Safe_map_8x5_2agents_10.0M_Safe_map_aoba00_2agents_5.0M_checkpoint.pth
  ↓  LARE_SEG_RE = r"map_(.+?)_(\d+)agents_([\d.]+)M"  を finditer
"8x5_2 10M aoba00_2 5M"
```

`.+?` を非貪欲にしてあるのは、マップ名自体に `_数字` が入る場合
(`map_8x5_2` のような命名) でも `{map}_{N}agents_{steps}M` に正しく割れるようにするため。

### use_lare_path を必ず先に見る

`use_lare_path=False` の run でも `use_finetuning_lare_path=True` が config に残っていることがある
(フラグの消し忘れ)。この場合 LaRe は完全に無効なので、**`use_lare_path` を最初に判定する**。
これを怠ると、baseline の 5 seed が「事前学習あり」と誤表示された (実際に一度そうなった)。

### method_tag の規則

評価側 ([src/all_policy/policy.py](../src/all_policy/policy.py)) が読むファイル名に入る
`method_tag` は学習 config に直接は無いが、**事前学習した LaRe モデルを使ったか**で決まる。
規則は [tools/collect_config.yaml](../tools/collect_config.yaml) の `method_tag_by_lare_mode` に外出しした。

| lare mode | method_tag |
|---|---|
| `pretrained` / `finetuning` | `dbct` |
| `scratch` / `scratch(frozen)` / `off` | `safe` |
| 非 Safe 環境 (`drp-*`) で学習 | `unsafe` |

> YAML では**裸の `off` が `false` になる**ので、設定ファイル側ではクォートしてある。
> コード側でもキーを文字列に正規化して二重に保険をかけている。

---

## 7. 学習済みモデルの回収

epymarl の保存先は `{repo}/results/models/{unique_token}/{t_env}/` で、

```python
unique_token = f"{_config['name']}_seed{_config['seed']}_{map_name}_{datetime.datetime.now()}"
```

([src/epymarl/src/run.py](../src/epymarl/src/run.py#L41))。**seed が入っている**ので、
sacred の run と 1:1 で紐づけられる。`build_model_index` で repo ごとに 1 度だけ
ディレクトリを列挙し、`{algo}_seed{seed}_` の前綴りで引く。

回収するのは **`done` の run の最終ステップのみ**。理由:

- 1 run に checkpoint が 800 個ある run もある (`save_model_interval` 依存)
- `opt.th` / `agent_opt.th` / `critic_opt.th` (optimizer state) は評価に使わない

この 2 つを絞った結果、**done 53 run が 25MB** に収まった (optimizer 込みだと数 GB)。

### 転送に tar を選んだ理由

`unique_token` には**空白と `:` が入る** (`qmix_seed222585621_drp_env:drp_safe-7agent_map_8x5-v2_2026-08-27 01:19:17.599005`)。
`scp` / `rsync` の `host:path` 記法とぶつかるので、ファイル名を NUL 区切りで tar に渡す。

```text
ssh HOST 'tar -cf - -C <repo> --null -T -'   ← ファイル名は stdin から NUL 区切りで
   |
   +--> ローカルの tar -xf - -C <tmp>        ← 直結 (メモリに載らない)
   |
   +--> tmp から dest/{machine}/{algo}_{map}_{N}agents_seed{seed}_step{t_env}/ へ移動
```

ホスト 1 台につき ssh 接続 1 回で済む。既に置いてあるディレクトリは飛ばすので、
cron で回し続けても転送は初回だけになる。

### 保管用リポジトリの構成 (2026-09-08 決定)

モデルの置き場は**用途で 3 つに分かれる**。要件が違うので同じ場所には置けない。

| 種類 | 中身 | 実測 | 失うと | 置き場 |
|---|---|---|---|---|
| 評価用 | 条件ごとに選んだ経路方策 | 2.7MB / 22 件 | **論文の数値が再現不能** | `src/all_policy/models/safe/` (フラット) |
| LaRe 報酬モデル | pretrained / finetuning のロード元 | 364KB / 15 件 | 学習が再現不能 | `src/lare/path/models/` (git 公開済み) |
| 回収アーカイブ | 完遂した全 run の最終モデル | 78MB / 210 件 | 選び直しができない | **`LDRP_models` (private repo)** |
| 中間 checkpoint | 全ステップの保存 | **12GB** / 12480 件 | 途中再開できないだけ | git に入れない |

調査で分かったこと: **`src/all_policy/models/safe/` は `.gitignore` の `**/models/` に
引っかかって git 管理外**になっている (git が追跡しているモデルは 13 件、すべて LaRe 報酬モデル)。
論文の再現に必要な 22 件が版管理されていない状態。

#### なぜ別リポジトリにしたか

`origin` (Yamaguchi-yushi/LDRP) は **PUBLIC** (実測 56MB)。学会投稿でコードを公開する前提なので、

- 同じリポジトリに入れる → 公開範囲を分けられない
- LDRP を private にする → 公開時に切り出す作業が発生し、`upstream` (kaji-ou/LDRP) との関係も整理が要る
- submodule → public から private submodule を参照すると他人が clone できない

ので **別の private リポジトリ**にした。

#### 階層とリーフ名

```text
path/{map}/{N}agent/{planner}_{method_tag}/{評価用の名前}__{step}_{param_hash}/
      agent.th, mixer.th          (opt.th は入れない)
```

`__` より前を評価用のファイル名そのままにしてあるのが要点。設置側 (`--install-models`) は
`__` で切るだけで評価用の名前に戻せるので、**変換表を持たなくてよい**。

**評価用はフラットのまま**にする。`policy.py` の `resolve_model_path()` が
`os.path.join(models_dir, f"{stem}_seed{K}.th")` でサブディレクトリを見ないため、
階層にすると `policy.py` (学習・実行系) を変えることになる。
保管用は階層 / 評価用はフラット、と分けた。

#### 移行できないもの

既存 22 件は**どの run から来たか記録が無い**ので `step` も `param_hash` も復元できない。
`map_8x5_5_qmix_base.th` のように seed suffix すら無いものもある (legacy = seed0 扱い)。
`legacy/` に名前のまま置くしかない。

### 評価用ディレクトリへの設置

`--install-models` で、評価側が探す名前に変えて `src/all_policy/models/safe/` に置く。

```text
models_inbox/白/qmix_map_8x5_5agents_seed113162076_step20000500/path/agent.th
   ↓
src/all_policy/models/safe/map_8x5_5_qmix_safe_base_seed113162076.th
```

中身は epymarl の `path/agent.th` (RNNAgent の `state_dict`) そのままで、
`PolicyRunner` はこれを `torch.load` して読む。既定は off、同名ファイルは上書きしない。
`reassign_tag` は学習時に決まる軸ではないので常に `base` にしてある。

---

## 8. Notion 同期

無料プランで使える Internal Integration を前提にしている。API バージョンは `2022-06-28`
(新しい `2025-09-03` は `database_id` 親が `data_source_id` に変わるため、あえて固定)。

### 書き込み回数を減らす

1. `POST /v1/databases/{id}/query` を**ページングして 1 度だけ**全件取得し、
   `run_uid -> (page_id, プロパティのプレーン値)` の索引を作る
2. 各 run について、新しい値と索引の値を型ごとに比較する
3. **差分があるプロパティだけ**を `PATCH /v1/pages/{page_id}` で送る。差分ゼロなら何もしない

毎回全件 PATCH すると 3 req/s のレート制限に当たるので、この形にしてある。

### プロパティは自前で作れる

`--notion-create-db <PAGE_ID>` で、`NOTION_SCHEMA` に並んだ 18 個のプロパティを持つ DB を
新規作成する。Notion 側で列を手作業で用意する必要がない。
`progress` だけ作成時に `{"format": "percent"}` を指定している。

### token の受け取り方

cron / systemd から環境変数を渡すのは面倒なので、**ファイルも見る**。
コマンドライン引数では受け取らない (`ps` に出てしまうため)。

```text
環境変数 NOTION_TOKEN
  → --notion-token-file で指定したファイル
  → tools/.notion_token          (.gitignore 済み)
  → ~/.config/ldrp/notion_token
  → ~/.ldrp/notion_token         (--bootstrap が置く場所)
```

`--notion-check` で「token が有効か」「DB が見えるか」「プロパティが揃っているか」を
まとめて検証できる。

---

## 9. ファイルと主な関数

`tools/collect_runs.py` は 1 ファイルで完結している (リモートに送り込むため分割できない)。

| 区画 | 主な関数 | 動く場所 |
|---|---|---|
| scanner | `scan` / `scan_run_dir` / `_tail_text` / `_t_env_from_metrics` / `build_model_index` / `find_model` | ローカルとリモート両方 (標準ライブラリのみ) |
| 収集 | `collect` / `collect_local` / `collect_ssh` / `collect_drop` / `export_drop` | 集約側 |
| 導出 | `derive` / `pretty_lare_models` / `eval_model_filename` | 集約側 |
| 出力 | `render_table` / `render_markdown` / `render_csv` / `render_summary` | 集約側 |
| モデル | `fetch_models` / `_fetch_ssh` / `_fetch_drop` / `select_model_files` / `install_models` | 集約側 |
| Notion | `notion_request` / `notion_fetch_index` / `notion_sync` / `notion_create_db` / `notion_check` | 集約側 |
| 配布 | `bootstrap_push` | 集約側 |

---

## 10. 設計上の判断と理由

| 判断 | 理由 |
|---|---|
| 1 ファイルにまとめた | ssh の stdin で丸ごと送り込むため。分割すると相手にファイルを置く必要が出る |
| リモートに何もインストールしない | GPU 機は共用 (`linlab` アカウントに複数人) なので、環境を汚したくない |
| `metrics.json` / `info.json` を開かない | 1 run で最大 5MB。490 run 分を開くと数 GB になる |
| `--min-steps` の既定を 1e6 にした | `t_max=40` のようなデバッグ run が 100 件以上あり、表が読めなくなるため |
| `--cache` で JSONL にマージ保存 | ホストが落ちている回に、その行が表から消えないようにするため |
| 評価用ディレクトリへの設置を既定 off にした | 研究用のモデル置き場に毎時書き込むのは影響が大きい。明示フラグを要求する |
| `--bootstrap` の cron 登録を既定 off にした | 共用マシンの crontab を黙って書き換えない |
| `install_hints.sh` を書き出す | `--install-models` を使わない運用でも、同じ内容を目視してから実行できる |
| 出力文字列を英語にした | CLAUDE.md の規約 (国際会議でのコード公開を想定) |

---

## 11. 未検証の部分と既知の制約

### 未検証

- **Notion への実書き込み**。integration token がまだ無いため、`POST /v1/pages` /
  `PATCH /v1/pages` / `POST /v1/databases` の実レスポンスは確認できていない。
  確認済みなのは「不正な token で正しく 401 が返り、エラーが読める形で出る」ところまで
  (= 認証ヘッダ・API バージョン・エラー処理は動作している)。
  プロパティの組み立てとダイジェスト比較はオフラインで検証済み
- **drop モードの実運用**。同一マシン内で「別 Mac 役」を演じる形で
  export → 収集 → モデル回収 → 設置まで通したが、実際の iCloud 同期越しでは未確認
- **`--bootstrap`**。共用 GPU 機に書き込む操作なので、実行はしていない

### 制約

- **push モードではモデルが集まらない**。Notion の行だけ。
  モデルも欲しいなら pull か drop にする
- **`t_env` が取れない run がある**。開始直後で `cout.txt` も `metrics.json` も
  まだ中身が無い場合、progress が `?` になる (状態判定には影響しない)
- **Mac がスリープ中は pull が動かない**。launchd の `StartInterval` は復帰後の次の回に回る
- **iCloud の「ストレージを最適化」**が有効だと、共有フォルダのファイルが実体を持たない
  状態になることがある。読み出し時にダウンロードされるので動くが、オフラインだと失敗する
- **`method_tag` の規則は 1 軸のみ**。現在は `lare mode` からしか決めていない。
  タスク割当の同時学習など別の軸で分けたくなったら
  `method_tag_by_lare_mode` を増やすだけでは足りず、`derive` に手を入れる必要がある

---

## 12. 将来の実装

まだ作っていないが、このツールの延長線上でやりたいこと。

### 12.1 学習スイープ train_sweep

**現状の非対称**:

| | 条件のスイープ | seed のスイープ | 集計 |
|---|---|---|---|
| 評価 [run.py](../run.py) | ○ `product(map, agent, path, task, method_tag, ...)` | ○ `list_model_seeds(stem)` で自動 | ○ 最後に [aggregate.py](../aggregate.py) を呼ぶ |
| 学習 [train.py](../train.py) | **×** 1 条件を f-string にベタ書き | **×** `num_runs = 1`、seed は未指定 | — |

つまり**学習だけ手動**で、条件を変えるたびに `train.py` を書き換えて起動し直している。
1 run が 5〜17 時間かかるので、「条件 A の 5 seed が終わったら条件 B の 5 seed」を
無人で回せる形にしたい。

**やること**: `tools/train_sweep.py` + `tools/sweep_conditions.yaml` を追加する
(`train.py` は書き換えず、別ファイルとして足す)。

```yaml
common:                          # 全条件に効く
  config: qmix
  time_limit: 500
  env_args:
    state_repre_flag: onehot_fov
    randomize_task_arrival: true
    mmpp_ratio: 0.5
  train_task_assigner: true

conditions:
  - name: aoba00_7agent_ft
    t_max: 5050000
    seeds: [498131655, 546810871, 110248365, 335396418, 534945804]
    env_args:
      key: drp_env:drp_safe-7agent_map_aoba00-v2
      use_lare_path: true
      use_finetuning_lare_path: true
      finetuning_lare_path_model_name: QMIX_PATH_Safe_map_8x5_7agents_20.0M_checkpoint.pth
  - name: aoba00_10agent_ft
    t_max: 5050000
    seeds: [...]
    env_args:
      key: drp_env:drp_safe-10agent_map_aoba00-v2
```

**実装で踏むべき落とし穴** (調査済み):

| # | 注意点 | 根拠 |
|---|---|---|
| 1 | **seed を明示指定する** (`with seed=498131655`)。現状 `train.py` は seed を渡していないので sacred が毎回ランダムに振る = Notion に書いた計画 seed と一致しない | [main.py](../src/epymarl/src/main.py#L35) が `config["seed"]` を読む。sacred の config キーなので `with seed=N` で上書きできる |
| 2 | **`env_args.key` にクォートを付けない**。`shell=False` のリストで渡す | クォート付きで渡すと sacred が引用符ごと文字列にする。実際 `results/sacred/qplex/"drp_env:drp_safe-3agent_map_5x4-v2"` という引用符入りディレクトリが残っている |
| 3 | **stdout をシェルでリダイレクトしない**。ログが欲しければ Python 側で tee 相当を書く | GPU 機の `cout.txt` が 0 バイトになっており、`t_env` の進捗が読めなくなっている ([§5](#5-完遂判定-state))。原因がリダイレクトか `CAPTURE_MODE` かは未確認だが、リダイレクトを避ければどちらでも安全 |
| 4 | **並列度の既定は 1**。GPU 1 枚を複数 run で共有すると遅くなる | `train.py` も `maxpurocesses = 1` |
| 5 | **中断からの再開**。既に `done` / `running` の (algo, env_key, seed, t_max) はスキップする | `collect_runs.py` を import して `scan` + `derive` を再利用すれば、判定ロジックを二重に持たなくて済む |

5 が効くと「スイープを止めて再開しても、終わった条件は飛ばして続きから回る」になる。
`--dry-run` で実行予定のコマンド一覧を先に出せるようにする。

### 12.2 実験管理 GUI

**方針は決定済み**: ローカル Web サーバ (`tools/exp_server.py`) にする。
単なる閲覧画面ではなく、**この Mac から全マシンの学習を起動・予約できる管制画面**にする。

やりたいこと (2026-08-29 に確定):

1. 複数マシンの実行状況を 1 画面で監視する
2. **この PC から全マシンの学習を起動できる**
3. 条件ごとに「学習が終わっている seed の一覧」が Notion と同じ粒度で見える
4. **足りない seed をその場から実行できる**
5. **実行予約**: 「いまメモリが足りないから、空いたら実行」ができる

#### 画面構成

```text
┌ LDRP experiments ────────── 更新 2m前  [今すぐ収集] [キュー一時停止] ┐
│                                                                      │
│ machines                                                             │
│  machine  RAM空き   GPU空き        load   実行中                      │
│  白       12.4 GB   -              1.2    0 / 1                      │
│  GPU1     58.0 GB   14.2 GB /1gpu  3.4    1 / 2                      │
│  GPU2     unreachable                                                │
│                                                                      │
│ conditions                                                           │
│  ┌ 8x5_7agent_30M_safe   7agent map_8x5 30M / qmix / safe / -  5/5 ┐ │
│  │ (724865803)(236379847)(610481256)(524463113)(302911431)         │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│  ┌ aoba00_7agent_5M_ft_ppo  7agent map_aoba00 5M / qmix / 8x5_7   2/5┐│
│  │ (740597639)(594911358 78%)(  ?  )(  ?  )(  ?  )                 │ │
│  │ 594911358 @白 ████████░░ 78%                                    │ │
│  │ [GPU1 ▾] [3 ▾] [キューに追加]   足りない: 3                      │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│ queue                                                                │
│  # 条件                    seed       machine 状態     備考           │
│  4 aoba00_7agent_5M_ft_ppo 812334019  GPU1    running  pid 48120     │
│  5 aoba00_7agent_5M_ft_ppo 220049112  GPU1    waiting  RAM 6.1G < 12G│
└──────────────────────────────────────────────────────────────────────┘
```

seed の色分け: 緑 = done / 青 = 実行中 / 黄 = キュー待ち / 赤 = 失敗・停止 / 灰破線 = 未実行。

#### 実験計画ファイル (`tools/exp_plan.yaml`)

```yaml
common:                     # 全条件に効く既定値。条件側で上書きできる
  algo: qmix
  time_limit: 500
  needs_mem_mb: 8000        # このジョブを起動するのに必要な空き RAM
  env_args:
    state_repre_flag: onehot_fov
    randomize_task_arrival: true
    mmpp_ratio: 0.5

conditions:
  - name: 8x5_7agent_30M_safe
    t_max: 30050000
    seeds: [724865803, 236379847, 610481256, 524463113, 302911431]
    env_args:
      key: drp_env:drp_safe-7agent_map_8x5-v2
      use_lare_path: false

  - name: aoba00_7agent_5M_ft_ppo
    t_max: 5050000
    count: 5                # seeds を書かない分は積むときに採番して固定する
    seeds: [740597639, 594911358]
    train_task_assigner: true
    needs_mem_mb: 12000
    env_args:
      key: drp_env:drp_safe-7agent_map_aoba00-v2
      use_lare_path: true
      use_finetuning_lare_path: true
      finetuning_lare_path_model_name: QMIX_PATH_Safe_map_8x5_7agents_20.0M_checkpoint.pth
      use_dynamic_agents: true
      max_active_agents: 7
```

#### 計画と実績の突き合わせ

**`collect_runs.derive` をそのまま使う**。計画の条件からダミーのレコードを組み立てて
`derive` に通し、実績の run と同じ 6 要素キーで突き合わせる。
こうすると「条件の定義」がツール内で 2 つに割れない。

```python
key = (d["algo"], d["map"], d["agents"], d["t_max"], d["lare_mode"], d["task_assign"])
```

検証済み: 計画側のダミーレコードを `derive` に通すと
`{'algo':'qmix','map':'map_8x5','agents':7,'t_max':30050000,'lare_mode':'finetuning',
'setting':'8x5_2 10M','task_assign':'PPO'}` が得られ、実績側と同じ形になる。

#### ジョブの起動と予約

```text
[待機] --- 空き条件を満たした ---> [実行中] --- pid が消えた ---> [終了]
   ^                                   |
   +-- RAM/GPU/同時実行数が足りない -----+
```

- **起動**: `ssh HOST 'cd <repo> && nohup <python> src/epymarl/src/main.py ... > ~/.ldrp/logs/jobN.log 2>&1 &; echo $!'`
  で pid を受け取る。**nohup で切り離すので GUI を閉じても学習は走り続ける**
- **生死確認**: `kill -0 <pid>`。完遂したかどうかは収集側 (`state == done`) で判定する
- **空き確認**: ホスト上で 1 本の sh を走らせて `mem_avail_mb` / `gpu_free_mb` / `loadavg` を取る
  (Linux は `/proc/meminfo` の `MemAvailable`、macOS は `vm_stat`、GPU は `nvidia-smi`)。
  **共用マシンで他人が使っているメモリも反映される**ので、それも待ち条件になる
- **キューは JSON に永続化**する (`tools/.exp_queue.json`)。GUI を再起動しても予約が消えない

#### API

| メソッド | パス | 用途 |
|---|---|---|
| GET | `/api/state` | 画面が 15 秒ごとに引く全状態 |
| POST | `/api/collect` | 今すぐ全マシンを再スキャン |
| POST | `/api/enqueue` | `{condition, host, n}` で足りない seed を予約 |
| POST | `/api/dequeue` | 待機中ジョブを取り消す |
| POST | `/api/cancel` | 実行中ジョブを止める (確認ダイアログ付き) |
| POST | `/api/pause` | キューの自動起動を止める / 再開する |

#### パラメータ食い違いは GUI で警告して回し直す (2026-09-08 決定)

同じ条件のはずの seed で `param_hash` が割れていることがある。
**これを弾く場所は publish ではなく GUI にする。**

| 場所 | 振る舞い | 理由 |
|---|---|---|
| `--publish-models` | **止めない**。`state != done` だけで弾く | 保管は「回収できたものを残す」役。ここで判断すると、あとから多数派が変わったときに取り返せない |
| GUI | **警告して、その seed を回し直させる** | 対処 (捨てて再実行) ができるのは GUI 側だけ |

具体的な振る舞い:

1. 条件ごとに `done` + `running` の `param_hash` を数え、**多数派を正とする**
2. **充足数は多数派に一致する run だけで数える**。4/5 done でも 1 本が別設定なら実質 4 本なので
   `4/5` ではなく `3/5 (+1 要再実行)` と出す
3. 少数派を「要再実行」として別枠に出し、**ボタン一発でキューに入れ直せる**ようにする
4. `running` も対象に含める。17 時間回してから「設定が違った」と気づくより、
   走り出してすぐ止めるほうがよい

**回し直すときのパラメータは `exp_plan.yaml` ではなく、多数派の run の config を使う。**
`config.json` を全文保存するようにしたので ([§6 の学習パラメータ](#6-実験条件の導出))、
多数派がどんな設定だったかは記録から復元できる。計画ファイルを介すと、
計画を書き換えた後で「当時の多数派」が再現できなくなる。

```text
┌ aoba00_7agent_5M_ft_ppo   7agent map_aoba00 5M / qmix / 8x5_7 20M   3/5 done ┐
│ (740597639)(577768962)(870681097)(  ?  )(  ?  )                              │
│ ⚠ 要再実行 1: seed 594911358  params differ (7095a604 vs 53009eb4)           │
│    env.use_lare_path_training  True -> (多数派は未指定)                       │
│    [多数派の設定で回し直す]  [このまま採用する]                                │
└──────────────────────────────────────────────────────────────────────────────┘
```

「このまま採用する」を残すのは、**少数派のほうが正しい場合があるため**
(条件を意図的に変えた直後は、新しい設定が 1 本しかない)。
押したらその `param_hash` を正として記録し、以降は多数派扱いにする。

実装済みの部分: `odd_param_runs()` が「多数派から外れた `done`/`running` の run」を返す。
`--format status` のヘッダに `params✗ N` として出る。
GUI が要るのは 3 の「回し直す」だけ。

> 2026-09-08 時点では**要再実行 0 件**。`param_hash` が割れている 2 条件は
> いずれも `failed` の run しか含まないため、使える run の中に食い違いは無い。

#### 実装で踏むべき落とし穴

[§12.1](#121-学習スイープ-train_sweep) の 5 点に加えて:

| # | 注意点 | 理由 |
|---|---|---|
| 6 | **バインドは `127.0.0.1` 固定**を既定にする | 学習を起動できる画面なので、LAN に晒さない |
| 7 | **キューの自動起動を止められるようにする** (`--paused` / 画面のボタン) | 予約が意図せず走り出すのを防ぐ。初回は paused で挙動を確かめる |
| 8 | **収集・リソース取得・キュー処理は別スレッド**で回し、例外でスレッドを殺さない | ホストが 1 台落ちただけで画面全体が止まると使い物にならない |
| 9 | **同時実行数の既定は 1** | 共用 GPU 機で勝手に複数本立てない |
| 10 | ログを取るために stdout をファイルへリダイレクトすると `cout.txt` が空になる可能性がある | §12.1 の 3 と同じ。進捗は `metrics.json` から拾えるので実害は小さいが、承知の上でやる |
| 11 | **充足数は多数派 `param_hash` で数える**。単純に `done` を数えない | 別設定の run を 1 本混ぜたまま「5/5 そろった」と表示すると、そのまま論文の表に載る |

#### 段階

1. **監視だけ** (収集 + 条件別 seed 一覧 + 進捗) — キューを持たないので安全
2. **起動** (ボタンから 1 本起動、pid 管理、中止)
3. **予約** (空きメモリ待ち、自動起動、永続化)

1 の時点で「あと何 seed 足りないか」は分かるので、まずそこまで作って使い勝手を見る。

### 12.3 train.py の実行予定を枠として見せる

**ステータス (2026-09-13):** ダッシュボードの `予約待ち` 列は**削除した**。
train.py が予約ファイルを書かないので常に空で、「予約が無い」のか
「仕組みが動いていない」のか区別できず誤解の元になっていたため。
下記を実装したら [app.js](../tools/dashboard/static/app.js) の `t-mach` に列を戻す
(コメントで場所を示してある)。

戻すときに必要な作業は 3 つ:

| # | どこ | 内容 |
|---|---|---|
| 1 | [train.py](../train.py) | `~/.ldrp/batch_<pid>.json` に `{pid, total, started, cmd}` を書く (下記 `_publish_batch`) |
| 2 | [collect_runs.py](../tools/collect_runs.py) `export_drop` | **batch レコードを `runs.jsonl` に含める**。いまは run レコードだけなので、drop 経由 (黒 / M2) では `ps` の検出結果すら白へ届かない |
| 3 | [app.js](../tools/dashboard/static/app.js) | `t-mach` の見出しと行に `予約待ち` を戻す (`b.reserved` / `b.unknown` / `b.batches` は API 側に残してある) |

2 を入れるときは黒・M2 側で `git pull` が要る。1 だけ入れても
**ssh / local のホスト (白・GPU) でしか埋まらない**。

**問題**: sacred は run が**始まって初めて**ディレクトリを作る。
[train.py](../train.py) は `num_runs` 本を連続実行するので、5 本予定の 2 本目を回している時点では
**1 本しか見えない**。マシンが埋まっているのに空いているように見える。

#### 実装済みの部分 (train.py を変更しない範囲)

`ps` から `train.py` のバッチ実行を検出し、マシン別に集約して表示する
(`scan_batches` / `batch_summary` / `render_status`)。

```text
▶ 0 +3?  ⚠ 48
---
白       0 run  +3? wait   (train.py x3)
```

実測でこの Mac の 3 本を検出できている。ただし **`?` が示すとおり残り本数は分からない**。

仕様として決めたこと:

- **プロセスが死んでいる予約ファイルは無視する** (置き土産で枠が埋まったままにならない)
- **バッチ情報はキャッシュしない**。「いまの状態」なので、古い予約が残ると
  終わったバッチの枠が永久に埋まって見える
- 予約を出していないバッチが混ざったら `+3?` のように `?` を付ける

#### 残っている課題: 残り本数が分からない

`num_runs` を知っているのは `train.py` 自身だけ。**ソースを読む方式は使えない**。
根拠: 実際に走っていた `train.py` (pid 44460) は **60 時間前に起動**しており、
そのときのソースの `num_runs` が現在のファイルと同じである保証がない
(確認時点のファイルは `num_runs = 1`)。起動後に編集すれば静かにずれる。

#### 将来の実装: train.py から予約を書き出す

学習の挙動には一切影響しない (ファイルを 1 つ書くだけ)。ファイル冒頭に:

```python
import json
import os

# --- 実行予定を tools/collect_runs.py に知らせる (学習の挙動には影響しない) ---
_SLOT = os.path.expanduser("~/.ldrp/batch_%d.json" % os.getpid())


def _publish_batch(started):
    """残り何本予定かをファイルに書く. 失敗しても学習は続ける."""
    try:
        os.makedirs(os.path.dirname(_SLOT), exist_ok=True)
        with open(_SLOT, "w") as f:
            json.dump({"pid": os.getpid(), "total": num_runs,
                       "started": started, "updated": time.time()}, f)
    except OSError:
        pass
```

| # | 場所 (概算行) | 目印にする既存コード | 内容 |
|---|---|---|---|
| (a) | 冒頭 (~2 行) | `import time` **の次** | 上の `_publish_batch` を追加 |
| (b) | ~13 行 | `maxpurocesses = 1` **の次** | `_publish_batch(0)` |
| (c) | ~90 行 | `proc = subprocess.Popen(command, shell=True)` **の次** | `_publish_batch(i + 1)` (ループ内なのでインデントする) |
| (d) | 末尾 | `print("All runs completed.")` **の前** | `try: os.remove(_SLOT)` / `except OSError: pass` |

収集側は既にこのファイル形式を読むようになっている (`scan_batches` の `source: "file"` 分岐)
ので、**ツール側の追加実装は不要**。入れた瞬間に `?` が取れて残り本数が出る。

```text
▶ 1 +4  next 02:51
---
白       1 run  +4 wait   next 02:51   (train.py x1)
GPU2     8 run                next 14:20
```

検証済み: `~/.ldrp/batch_<pid>.json` に `{"total": 5, "started": 2}` を置くと `+3` と表示される。

### 12.4 評価結果ダッシュボード

§12.2 が「**学習が今どうなっているか**」を見る画面なのに対し、こちらは
「**評価し終わった結果**」を見る画面。データ源も更新頻度もまったく別。

| | §12.2 実験管理 GUI | §12.4 評価結果ダッシュボード |
|---|---|---|
| データ源 | 各マシンの sacred (ssh 巡回) | `results/summary.csv` (ローカルの静的ファイル 1 本) |
| 更新 | 2 分 / 30 分の定期収集 | ファイルの mtime を見るだけ (即時) |
| 見るもの | 実行中 / 完遂 / 残り seed / 終了予定 | 条件ごとの性能 (mean ± std)、条件間比較 |
| 操作 | 学習を起動・予約できる | 読むだけ (+ 論文用の表を書き出す) |

同じローカル Web サーバに **タブとして相乗りさせる**のが素直。評価タブは静的ファイルを
読むだけなので、ssh 巡回の重さに引きずられず即表示できる。

#### 現在の集計パイプラインと、手作業が残っている場所

```text
run.py  ── 条件 × seed を product で展開 → test.py を逐次起動
  ├─→ logs/{map}/{safe|unsafe}/{planner}/{N}agent/..._seed{S}.txt   ([RESULT] 行 1 本)
  └─→ 終了後に aggregate.py --csv results/summary.csv を自動実行

aggregate.py ── logs/**/*.txt から [RESULT] を拾い、condition -> {seed: rec}
  └─→ results/summary.csv   (condition, metric, n_seeds, mean, std, sem, per_seed)

★ ここだけ手作業 ★
  summary.csv ──→ results.csv   論文用の表 (map, allocator, planner, n × tc/et の mean, sd)
```

ダッシュボードの第一の役目は、**この★を GUI のエクスポートに置き換える**こと。
2026-08-31 に 99 行を手で転記したが、条件キーから `(map, allocator, planner, n)` への
対応づけは機械的にできるので、毎回やる作業ではない。

#### データ源は summary.csv にする (logs は直接読まない)

`aggregate.py` が既に

- 同一条件 × 同一 seed の重複検出 (警告して後を採用)
- `ddof=1` の標本標準偏差 (「手元の seed はあり得た学習結果からの標本」という理由付き)
- seed が 1 本しかない条件の `nan` 扱い

を持っているので、**同じロジックをダッシュボード側に二重実装しない**。
`summary.csv` を読むだけにする。

`per_seed` 列にセミコロン区切りで個々の seed の値が入っている
(`13.3;14;45.9;25.9;20.1`) ので、**mean ± std だけでなく分布も描ける**。

#### 見せたいもの

| # | 内容 | 根拠 |
|---|---|---|
| 1 | 条件 × メトリクスの表 (`mean ± std`, `n_seeds` 併記) | いま summary.csv を目で追っている |
| 2 | 絞り込み (map / N / planner / allocator / method_tag / reassign) | 実測 116 条件。全部並べると読めない |
| 3 | 比較グラフ: 横軸 N、系列 planner、縦軸を選択 | 「台数を増やすとどうなるか」が主要な問い |
| 4 | **per-seed を点で重ねる** | `per_seed` があるので描ける。5 本の散らばりが見えると std だけより判断しやすい |
| 5 | **seed 不足の警告** | 実測で `n_seeds=1` が 20 件 (すべて PBS)、`n_seeds=5` が 96 件。混在したまま表に載ると誤読される |
| 6 | 論文用の表 (pivot) を書き出す | ★の置き換え |

#### 実装上の注意

**(a) 時間軸は対数にする。** `time_sec` の実測レンジは
`0.0236 秒` (5x4/5agent/unsafe_iql_tp) 〜 `5887.78 秒` (aoba00/5agent/unsafe_pbs_tp) で **5 桁**開く。
線形軸だと MARL 系が全部ゼロに潰れて PBS しか見えない。
`task_completion` も `0.02` 〜 `183.9` で 4 桁開くので、同様に切り替えられるようにする。

**(b) 条件キーの後付けパースは曖昧。** `_condition_id()` の形式は

```text
{map}/{N}agent/{safe|unsafe}_{planner}[_{method_tag}]_{base|reassign}_{assigner}[_train{N}][_envreassign]
```

だが、**planner 名自体に `_` が入る** (`mat_dec`, `transf_qmix`)。
`safe_mat_dec_dbct_base_tp` を `_` で切っても `mat_dec`+`dbct` と `mat`+`dec_dbct` を区別できない。

| 案 | 内容 | 備考 |
|---|---|---|
| A | 既知の planner 名リストで最長一致させる | ダッシュボード側だけで済む。planner を増やしたらリストも足す |
| B | `runner.py` の `[RESULT]` 行に `map=` `agent_num=` `planner=` `assigner=` を個別フィールドとして足す | 曖昧さが消える。ただし**学習・実行系に手を入れる**ので別途判断が必要 |

**A で始めて、B は余裕があれば**。A のリストは
`src/all_policy/policy.py` の planner 名と揃える。

**(c) メトリクスの列を固定しない。** `runner.py` の `[RESULT]` は改版で増減している。
実測で、手元の summary.csv には `1agent_goal_account` があるが現在の `runner.py` には無く
(`task_completion_per_agent` に改名)、逆に `n_active_mean` / `busy_ratio` / `deadhead_ratio` /
`task_dropped` / `pending_len_avg` などの新しい指標が summary.csv に無い。
**列は summary.csv にあるものを動的に読む**設計にしておかないと、回し直すたびに壊れる。

**(d) グラフの描き方。** 依存を増やしたくないので、
`--format html` と同じく **自前の inline SVG** で描くのが軽い
(棒 / 折れ線 / 点の重ね描き程度なら十分)。
matplotlib で PNG を作ってサーバから返す案もあるが、
拡大・絞り込みのたびに再生成が要るので対話性で劣る。

#### Notion を経由させるか (2026-09-07 検討)

「評価結果も Notion に書いて、ダッシュボードは Notion を読む」構成は**技術的には可能**
(§8 の Notion 同期をそのまま流用できる)。ただし **読み出し元は Notion にしない**方がよい。

| 構成 | 書き込み | ダッシュボードの読み出し | 判定 |
|---|---|---|---|
| A | ローカルのみ | `results/summary.csv` | 速いが Notion に残らない |
| B | Notion のみ | Notion API | **勧めない** (下記) |
| **C** | **ローカル + Notion 両方** | **`results/summary.csv`** | **推奨** |

B を勧めない理由:

| # | 問題 | 具体的な数字 |
|---|---|---|
| 1 | **遅い** | Notion のクエリは 1 リクエスト 100 件まで。実測 116 条件 × 6 メトリクス = **696 行**なら 7 リクエスト。レート制限が平均 3 req/s なので最短でも 2〜3 秒かかる。絞り込みを変えるたびにこれを待つ |
| 2 | **オフラインで見えない** | 解析中にネットワークが切れると使えない。VPN が落ちて GPU に繋がらない事例が実際に起きている |
| 3 | **Notion のグラフでは足りない** | §12.4 で必要とした **対数軸** (`time_sec` は 0.0236〜5887.78 秒で 5 桁開く) と **per-seed の点の重ね描き** が Notion のチャートでは描けない |

C にすると、**書き込みは 1 系統** (同じスクリプトが CSV と Notion の両方へ出す) のまま、
ダッシュボードはローカルファイルを読むので即時。Notion は「研究室で共有する記録」
「スマホから結果を確認する」用途に限定する。

#### Notion 側のデータモデル

`summary.csv` をそのまま 1 行 = 1 条件 × 1 メトリクスで入れると **696 行**になり、
「条件ごとに横並びで見る」のが苦手な形になる。**条件単位に要約して列に展開する**方が実用的。

| 案 | 行数 | 形 | 評価 |
|---|---|---|---|
| 1 | 696 | 1 行 = 条件 × メトリクス (CSV のまま) | 忠実だが読みにくい。upsert に約 4 分 |
| 2 | 116 | 1 行 = 条件、全メトリクス × (mean/std/sem) を列に | 18 列超。§12.4(c) の「列を固定しない」方針と衝突する |
| **3** | **116** | **1 行 = 条件、主要メトリクスだけ列に** | **推奨**。upsert 約 40 秒 |

案 3 の列: `condition` / `map` / `agents` / `planner` / `allocator` / `method_tag` /
`n_seeds` / `task_completion (mean, sd)` / `time_sec (mean, sd)` / `collision_rate` /
`updated`。**論文の表に載せる分だけ**。全メトリクスはローカルの CSV に残す。

**学習 run の DB とは別 DB にする。** 粒度が違う (run 単位 vs 条件単位) ので、
`run_uid` をキーにした既存 DB に混ぜると両方が読みにくくなる。

#### マップ名・台数ごとに表を分ける

いま手で管理している Notion は `## 5agent 20M` のような**見出し + 表**の形になっている。
同じ見え方を作る方法が 2 つあり、性質がかなり違う。

| | P1. DB + グループ化ビュー | P2. ページ本文にテーブルを書く |
|---|---|---|
| 見え方 | map / 台数でグループ分けされた表 | `## {map} {N}agent` の見出し + 表 (現在の手管理と同じ) |
| ツール側の実装 | **`map` / `agents` を独立プロパティにするだけ** | 見出しブロック + `table` / `table_row` ブロックを組み立てる |
| Notion 側の初回作業 | **ビューのグループ化設定を 1 回** (API では設定できない) | 不要 |
| 更新 | 該当ページだけ upsert | **本文を全消し → 再作成** |
| Notion 上での操作 | 並べ替え・絞り込み・別ビュー追加が自由 | できない (静的なテキスト) |
| 列の増減 | プロパティを足すだけ | 表の再生成で対応 |

**P1 を推奨**。ツール側は案 3 の列 (`map` を select、`agents` を number) を持つだけで済み、
あとは Notion の UI で「`map` でグループ化、`agents` で並べ替え」を 1 回設定すれば、
マップごとに区切られた表になる。台数でさらにサブグループ化することもできる。

> **注意: Notion API ではビュー (グループ化・フィルタ・並べ替え) を作れない。**
> API で作れるのはデータベースとページ (行) まで。`--notion-create-db` が作った DB は
> 既定のテーブルビュー 1 つだけなので、**グループ化は Notion の画面で 1 回だけ手で設定する**。
> 一度設定すれば、以降の行の追加・更新は自動で反映される。

P2 は「Notion のページを見た目そのまま自動生成したい」場合の選択肢。
`table` / `table_row` ブロックは API で作れるが、更新のたびに本文を全置換することになり、
Notion 上でメモを書き足しても消える。**手で補足を書き込む余地を残したいなら P1**。

どちらでも、ツールが持つべき情報は同じ ——
**`condition` をパースして `map` / `agents` / `planner` / `allocator` / `method_tag` を
独立したフィールドとして持つこと**。これは §12.4「実装上の注意 (b)」の
条件キーのパースと同じ話で、planner 名に `_` が入る曖昧さもそのまま効く。

#### 段階

1. **表だけ** — summary.csv を読んで絞り込み付きの表を出す。★の pivot 書き出しもここで入る
2. **グラフ** — 横軸 N × 系列 planner、per-seed の点を重ねる。対数軸の切り替え
3. **§12.2 と統合** — 同じサーバのタブにする。「この条件は学習が 3/5 しか終わっていない」と
   「この条件の評価結果」が並ぶので、次に何を回すべきかが 1 画面で分かる
4. **Notion への書き出し** — 上記案 3 の要約を別 DB に upsert (`map` / `agents` を
   独立プロパティにしておき、グループ化は Notion 側で 1 回設定する = P1)。
   ダッシュボードの読み出し元はローカルのまま変えない

3 まで行くと、[§12.2](#122-実験管理-gui) の計画 (`exp_plan.yaml`) と評価結果が同じ条件キーで
突き合わせられるようになる。ただし**学習側の条件キー**
(`algo`/`map`/`agents`/`t_max`/`lare_mode`/`task_assign`) と
**評価側の条件キー** (`_condition_id()`) は別物なので、対応表が必要になる点は注意。

---

### 12.5 実験計画の複数ファイル化 (plan_AAMAS.md / plan_XXX.md)

**現状:** 計画は `tools/plan.md` 1 枚。ダッシュボードはこの 1 ファイルを読み、
該当しない run は「計画外」として表に出さない。

**やりたいこと:** 計画を**投稿先・目的ごとに分ける**。

```text
tools/plans/
├── plan_AAMAS.md      AAMAS 投稿用の 52 条件
├── plan_ablation.md   アブレーション用
└── plan_scratch.md    試運転・探索用
```

AAMAS の条件に一致する run は `plan_AAMAS.md` の表に、別の計画に一致する run は
そちらの表に seed が載る。**1 つの run が複数の計画に一致することもある**
(例: baseline は AAMAS とアブレーションの両方で使う) ので、
**run は計画に対して 1:N** で紐づける。

#### 設計メモ

| 論点 | 方針案 |
|---|---|
| 割り当て | `matches()` を全計画に対して回す。**排他にしない** (1 run が複数表に載ってよい) |
| どこにも一致しない run | いまと同じ「計画外」。実行中セクションには出るが表には書かない |
| 表示 | ダッシュボードに計画のタブ (または select) を足す。`?plan=AAMAS` |
| 進捗の合計 | 計画ごとに `done/want`。「AAMAS は 125/261」のように投稿単位で出す |
| モデルの通し番号 | **計画には紐づけない**。`seed_index` は `eval_stem` 単位で振る (§4 の命名規則)。同じモデルが 2 つの計画に出てきても実体もファイル名も 1 つ |
| 設定 | `collect_config.yaml` に `plans:` (glob) を足す。既定は `tools/plans/*.md`、無ければ従来の `tools/plan.md` にフォールバック |

#### 実装の当たり

- [plan.py](../tools/plan.py) の `parse_plan(path)` は既に 1 ファイル 1 関数なので、
  **複数ファイルを読んで `plan_name` を各条件に足すだけ**で済む
- [app.py](../tools/dashboard/app.py) の `plan_view(rows, shaped)` が
  `(conds, used)` を返す構造なので、`plan_name` ごとにグループを分けて返す形に変える。
  `used` (計画に載った run の uid) は**全計画の和集合**にする —
  片方の計画にしか無い run が「計画外」に二重表示されるのを防ぐため
- 章見出し (`## 5agent 20M`) とマップ行の書式は**変えない**。
  Notion のメモをそのまま貼れる性質を壊さない

#### 先にやっておくとよいこと

seed の通し番号 (§4) を計画ではなく `eval_stem` 単位にしてあるので、
**計画を分割してもモデルのファイル名は動かない**。この不変条件は維持する。
