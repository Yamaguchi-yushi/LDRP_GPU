# 収集される側のマシンのセットアップ (drop モード)

**SSH で繋がらないマシン (黒 / M2) に置く設定と、動かなくなったときの切り分け手順。**

学習の進捗と完遂モデルを共有フォルダ (iCloud Drive) 経由で集約マシン (白) へ渡す。
どのマシンでも手順は同じで、**変えるのは `--machine` の値だけ**。

---

## 目次

| 節 | いつ読むか |
|---|---|
| [このマシンの役割](#このマシンの役割) | 最初に 1 回 |
| [初回セットアップ](#初回セットアップ) | 新しいマシンを追加するとき |
| [**落とし穴: machine 名の書き換え**](#落とし穴-machine-名の書き換え-必ず読む) | **plist をコピーしたら必ず** |
| [launchd の読み方](#launchd-の読み方-load-と-start-は別物) | 定期実行が動かないとき |
| [症状から原因を引く](#症状から原因を引く-診断の決定木) | 白から「N 時間前」と言われたとき |
| [仕様](#仕様) | 何が書き出されるか知りたいとき |
| [過去に起きたこと](#過去に実際に起きたこと) | 同じ症状か照合するとき |

---

## このマシンの役割

**共有フォルダに書き出すだけ。** それ以外は何もしない。

```text
黒 / M2  --export-->  iCloud Drive / LDRP_runs / {machine} /  --読む-->  白
                        runs.jsonl        (進捗)
                        models/...        (完遂した方策モデル)
```

| やること | やらないこと |
|---|---|
| 自分の sacred 出力を読む | 他のマシンへ ssh する |
| 共有フォルダにファイルを書く | Notion に接続する |
| | git に push する |
| | 実験計画 (`tools/plans/`) を作る・触る |
| | モデルを保管リポジトリへ入れる |

**待受ポートは開かない。外部通信もしない。** 書き込むのはローカルの共有フォルダだけ。

---

## 初回セットアップ

### 0. 前提の確認

満たしていないものがあれば先にそこを直す。

```bash
ls -d ~/LDRP                                              # 1. リポジトリ
cd ~/LDRP && git branch --show-current                    # 2. tools/ があるブランチ
python3 -V                                                # 3. python3 (標準ライブラリのみで動く)
ls -d ~/Library/Mobile\ Documents/com~apple~CloudDocs     # 4. iCloud Drive
```

> `--export` は **標準ライブラリだけ**で動く。conda 環境も PyYAML も要らない。
> システムの `python3` でよい。白側の収集スクリプトとは要求が違う。

### 1. 手動で 1 回動かす

**自分のマシン名に読み替えること** (以下は M2 の例)。

```bash
cd ~/LDRP
python3 tools/collect_runs.py \
    --export ~/Library/Mobile\ Documents/com~apple~CloudDocs/LDRP_runs \
    --machine M2 \
    --repo ~/LDRP
```

期待する出力:

```text
[export] 9 run(s), 5 model dir(s) newly copied (0.6 MB) -> .../LDRP_runs/M2
```

書き出されたものを確認する。

```bash
D=~/Library/Mobile\ Documents/com~apple~CloudDocs/LDRP_runs/M2
ls "$D"                      # runs.jsonl と models/ があるはず
wc -l "$D/runs.jsonl"        # 1 run 1 行
du -sh "$D"                  # 数 MB 程度 (モデル回収後は 1MB 未満)
```

### 2. 15 分ごとに回す

```bash
cp ~/LDRP/tools/com.ldrp.export-runs.plist ~/Library/LaunchAgents/

# ★ ここで machine 名を書き換える (次節を必ず読む)
vi ~/Library/LaunchAgents/com.ldrp.export-runs.plist

launchctl load ~/Library/LaunchAgents/com.ldrp.export-runs.plist
launchctl start com.ldrp.export-runs
tail -f /tmp/ldrp_export.log        # 1 回動くのを見届けて Ctrl-C
```

plist は `~` を `/bin/sh` が展開する形にしてあるので、**ユーザー名の書き換えは不要**。
スクリプトは `~/LDRP/tools/collect_runs.py` を直接呼ぶので、**`git pull` すれば更新される**。

止めるとき:

```bash
launchctl unload ~/Library/LaunchAgents/com.ldrp.export-runs.plist
```

### 3. 更新のしかた

```bash
cd ~/LDRP && git pull
```

**`git pull` しただけなら launchd の再読込は要らない。** plist は 15 分ごとに
**新しい python プロセスを起動する**ので、次の実行から新しいコードになる。
再読込が必要なのは **plist 自体が変わったとき**だけ。

```bash
# plist が変わったかどうかは白でこう確認できる
git log --oneline -3 -- tools/*.plist
```

`tools/plans/` と `collect_config.yaml` は `.gitignore` に入っているので、
**pull しても降ってこないし、こちらから push されることもない**。

---

## 落とし穴: machine 名の書き換え (必ず読む)

**リポジトリの `com.ldrp.export-runs.plist` には `--machine 黒` が直書きされている。**

```xml
<string>python3 ~/LDRP/tools/collect_runs.py --export ~/Library/... --machine 黒 --repo ~/LDRP</string>
```

M2 でこれをそのままコピーすると、**M2 の run が「黒」として書き出され、
黒のフォルダを上書きする**。白から見ると黒の run が突然入れ替わり、
M2 は永久に沈黙しているように見える。

```bash
# コピーしたら必ず確認する
grep -o -- '--machine [^ ]*' ~/Library/LaunchAgents/com.ldrp.export-runs.plist
#   → --machine M2  になっているか
```

白側の `collect_config.yaml` の `label:` と**完全一致**していなければならない。
現在の正しい値:

| マシン | `--machine` の値 |
|---|---|
| 黒 | `黒` |
| M2 | `M2` |

混入が起きていないかは白からこう確認できる:

```bash
# 白で。各フォルダの machine ラベルが 1 種類だけならよい
for x in 黒 M2; do
  echo -n "$x: "
  python3 -c "
import json,collections,sys
c=collections.Counter()
for l in open('/Users/\$USER/Library/Mobile Documents/com~apple~CloudDocs/LDRP_runs/$x/runs.jsonl'):
    if l.startswith('{'): c[json.loads(l).get('machine')]+=1
print(dict(c))"
done
```

---

## launchd の読み方 (`load` と `start` は別物)

ここを取り違えると「1 回だけ動いて、その後こない」になる。

| コマンド | 効果 |
|---|---|
| `launchctl load <plist>` | **ジョブを登録する**。`StartInterval`(900秒) が効き始める。`RunAtLoad` があるので**登録直後に 1 回走る** |
| `launchctl start <label>` | **いま 1 回だけ走らせる**。スケジュールには何の影響もない |
| `launchctl unload <plist>` | 登録を解除する。以後 1 回も走らない |

**`start` だけでは定期実行にならない。** 登録されていないジョブに `start` は効かない。

### `launchctl list` の読み方

```bash
launchctl list | grep ldrp
```

```text
-    0    com.ldrp.export-runs
│   │   └─ ラベル
│   └───── 前回の終了コード。0 = 正常、それ以外 = 失敗
└───────── 実行中の PID。"-" は「いまは走っていない」(正常)
```

| 出力 | 意味 |
|---|---|
| 何も出ない | **未登録**。`launchctl load` をしていない → これが原因 |
| `-    0    com.ldrp.export-runs` | 正常。前回成功して、いまは待機中 |
| `-    1    com.ldrp.export-runs` | 前回失敗。`/tmp/ldrp_export.log` を見る |
| `12345    -    com.ldrp.export-runs` | **いま走っている**。長時間この状態なら詰まっている (下記) |

### 前回が終わらないと次は起動しない

launchd は**同じジョブを二重に起動しない**。前回の `collect_runs.py` が
終わらないまま残っていると、以後 15 分ごとの起動がすべて見送られる。

```bash
ps -eo pid,etime,command | grep '[c]ollect_runs'
#   etime が数時間になっていたら詰まっている
```

詰まる場所はほぼ **iCloud へのモデルコピー**。だから `--export` は
**`runs.jsonl` をモデルより先に書く**設計になっている (進捗だけは最新になる)。

---

## 症状から原因を引く (診断の決定木)

白から「`drop data is N h old`」と言われたときは、**上から順に**見る。

### 手順 1: 学習は生きているか (先に確認する)

**export が止まっていても、学習は走り続けていることがある。**
実例として M2 は 9 日間 export が沈黙したが、その間に学習は 0% → 92% まで進んでいた。
**「沈黙 = 死んでいる」と決めつけて消さない。**

```bash
ps aux | grep '[t]rain.py'
```

- **プロセスがある** → 学習は無事。**export だけの問題**なので手順 2 へ
- **プロセスが無い** → 学習も落ちている。`cout.txt` の末尾で原因を見る

```bash
# run_dir は白のダッシュボードで確認できる
tail -30 ~/LDRP/src/epymarl/results/sacred/qmix/'drp_env:drp_safe-10agent_map_aoba00-v2'/1/cout.txt
```

### 手順 2: ジョブは登録されているか

```bash
launchctl list | grep ldrp
```

何も出なければ**未登録**。これが最も多い原因。

```bash
cp ~/LDRP/tools/com.ldrp.export-runs.plist ~/Library/LaunchAgents/
vi ~/Library/LaunchAgents/com.ldrp.export-runs.plist    # ★ machine 名を直す
launchctl load ~/Library/LaunchAgents/com.ldrp.export-runs.plist
```

### 手順 3: 前回の実行が残っていないか

```bash
ps -eo pid,etime,command | grep '[c]ollect_runs'
```

`etime` が長いものがあれば、それが次回以降を塞いでいる。
**消す前に何をしているか確認する** (iCloud のアップロード中なら待てば終わる)。

### 手順 4: ログに失敗が出ていないか

```bash
tail -40 /tmp/ldrp_export.log
```

| ログに出るもの | 意味と対処 |
|---|---|
| `[export] N run(s), M model dir(s) ...` | 正常 |
| `Resource deadlock avoided` (Errno 11) | iCloud のファイルに `shutil.copy2` が失敗した。新しいコードでは回避済み → `git pull` |
| `No such file or directory` (`~/LDRP` 系) | リポジトリのパスが違う。`--repo` を確認 |
| `python3: command not found` | plist は `sh -lc` で起動するのでログインシェルの PATH を使う。`.zprofile` を確認 |

### 手順 5: スリープしていないか

蓋を閉じる運用なら、ここが原因になる。**スリープ中は学習も止まる**ので、
手順 1 で学習も止まっていた場合はこれを疑う。

```bash
pmset -g | grep -E "^ *sleep|disablesleep"
pmset -g log | grep -iE "Sleep|Wake" | tail -20
```

対策:

```bash
caffeinate -dims &        # 学習と一緒に起動しておく
```

### 手順 6: iCloud が同期しているか

```bash
D=~/Library/Mobile\ Documents/com~apple~CloudDocs/LDRP_runs/M2
ls -la "$D"                              # ローカルの mtime
brctl status 2>/dev/null | head -20      # 同期状態
```

- ファイル名の先頭に `.` が付き `.icloud` で終わるものがある → 実体が落ちていない
- 「ストレージを最適化」が有効だと実体が退避される。読むときに自動ダウンロード
  されるので動くが、オフラインだと失敗する。
  システム設定 → Apple ID → iCloud → iCloud Drive で切れる

---

## 仕様

### 書き出されるもの

```text
LDRP_runs/{machine}/
├── runs.jsonl        全 run のレコード (1 run 1 行)
└── models/
    └── {algo}_{map}_{N}agents_seed{S}_step{T}/
        ├── path/agent.th     経路方策 (RNNAgent の state_dict)
        ├── task/agent.th     タスク割当方策 (PPO を使う条件のみ)
        └── .fetched          白が回収し終えた印 (下記)
```

| 項目 | 内容 |
|---|---|
| 対象の run | `~/LDRP/src/epymarl/{results,tmp_results}/sacred` 配下すべて |
| モデルを出す run | **`done` (完遂) のみ**。`failed` / `stalled` は出さない |
| モデルのステップ | **最終 checkpoint のみ**。途中のものは出さない |
| 除外するファイル | `opt.th` / `agent_opt.th` / `critic_opt.th` (optimizer state) と `mixer.th` |
| 再送 | **しない**。すでに共有フォルダにあるものは飛ばす (初回だけ転送) |
| 容量の目安 | モデル 1 件 平均 0.12MB。実測 39 件で 4.8MB |

**書き出す順序は `runs.jsonl` → `models/`。** 逆にすると、モデルの転送が
iCloud で詰まったときに進捗まで巻き添えになる (実測で 3 時間遅れた)。

`mixer.th` は QMIX の学習を再開するときにしか使わず、評価側は読まない。
それでいて `agent.th` の 15 倍あり (平均 1.24MB 対 80KB)、以前は共有フォルダの
93% を占めていたので既定で外した。元のマシンには残っているので、再開したく
なったら `--fetch-mixer` を付けて送り直せる。

`runs.jsonl` には各 run の `config.json` が**全文**入る (86 キー)。
`lr` / `gamma` / `batch_size` / `mixer` / LaRe のフラグなどが白側で見える。
1 run 約 4KB なので、100 run でも 0.5MB 程度。

### `.fetched` マーカーと容量

白が `--purge-drop` 付きで回収すると、**モデルの中身だけを消して
ディレクトリと `.fetched` を残す**。

- ディレクトリが残っているので、こちら側は「もう送った」と判断して**再送しない**
- 実体が消えるので iCloud を圧迫しない (黒は 74 件で 1.0MB まで縮んだ)
- 消す前に「ローカルに同じサイズのファイルがあるか」を白が確認している

### 読み方が軽い理由

巨大なファイルは開かない。

| ファイル | 扱い |
|---|---|
| `config.json` (~2.5KB) | 全部読む |
| `run.json` (~5KB) | 全部読む |
| `cout.txt` (0〜650KB) | **末尾 64KB だけ** |
| `metrics.json` (1〜5MB) | **末尾 256KB だけ**、しかも `cout.txt` が空のときだけ |
| `info.json` (2.5MB) | **開かない** |

2 回目以降は共有フォルダにあるモデルを飛ばすので、ほぼ走査だけで終わる (実測 0.4 秒)。

---

## やってはいけないこと

| してはいけない | 理由 |
|---|---|
| `tools/plans/` を作る / 編集する | 実験計画は白が master。こちらから触ると食い違う。`.gitignore` されているので pull でも降ってこない |
| `--notion` を付ける | Notion への書き込みは白が担当。二重に書くと競合する |
| `--publish-models` / `--publish-push` を付ける | モデルの保管リポジトリへの配置は白が担当 |
| `--machine` の値を他のマシンと同じにする | フォルダを奪い合って上書きする ([落とし穴](#落とし穴-machine-名の書き換え-必ず読む)) |
| 共有フォルダの `.fetched` を消す | 回収済みのモデルを再送してしまう |
| 沈黙している run をいきなり消す | export が止まっているだけで学習は生きていることがある (手順 1) |
| リモートログイン (SSH) を有効にする | この構成では不要。待受ポートを開けないのが drop モードの利点 |

---

## 過去に実際に起きたこと

同じ症状かどうか照合する用。

### M2 が 9 日沈黙したが学習は生きていた (2026-09-15 〜 09-24)

- 最後の export が 09/15 12:35。白からは 8 日以上「情報が届いていません」
- **しかし学習は動いていた。** 09/15 12:29 に起動した 3 本が、復旧時点で 92% まで進んでいた
- 教訓: **`ps aux | grep train.py` を先に見る。** 沈黙は export の故障であって
  学習の死ではない。消していたら 92% を捨てていた

### 黒の進捗が 3 時間遅れた (2026-09 前半)

- `runs.jsonl` (440KB) をモデル (90MB) の**後**に書いていたため、
  iCloud のアップロードが詰まると進捗まで届かなくなっていた
- 対策: 書き出し順序を `runs.jsonl` → `models/` に変更 (実装済み)

### 共有フォルダが mixer.th で溢れた

- `mixer.th` が共有フォルダの 93% を占めていた。評価には一切使わない
- 対策: 既定で除外 (実装済み)。過去分は `find "$D"/*/models -name 'mixer.th' -delete` で消す

---

## 白側で何が起きるか (参考)

書き出したものは、白が 15 分ごとに読み込む。

```text
白: collect_runs.py --cache ... --fetch-models ~/models_inbox
      --publish-models ~/LDRP_models --purge-drop
    ├─ {machine}/runs.jsonl を読んで進捗を反映 (ダッシュボードの conditions 表)
    ├─ {machine}/models/ からモデルを回収
    ├─ 実験計画に載っている条件だけを保管リポジトリへ配置 (計画外は捨てる)
    └─ 回収できたものを共有フォルダから消す (.fetched を残す)
```

こちら側はこの流れに関与しない。**書き出したら終わり。**
