# 収集される側のマシンのセットアップ (drop モード)

**このマシン (黒) で作業する Claude Code 向けの手順書。**

SSH で繋がらないマシンから、学習の進捗と完遂モデルを集約マシン (白) へ渡すための設定。
黒以外のマシンでも、`--machine` の値を変えれば同じ手順で使える。

---

## このマシンの役割

**共有フォルダ (iCloud Drive) に書き出すだけ。** それ以外は何もしない。

```text
黒  --export-->  iCloud Drive / LDRP_runs / 黒 /  --読む-->  白
                   runs.jsonl        (進捗)
                   models/...        (完遂した方策モデル)
```

| やること | やらないこと |
|---|---|
| 自分の sacred 出力を読む | 他のマシンへ ssh する |
| 共有フォルダにファイルを書く | Notion に接続する |
| | git に push する |
| | 実験計画 (`plan.md`) を作る・触る |
| | モデルを保管リポジトリへ入れる |

**待受ポートは開かない。外部通信もしない。** 書き込むのはローカルの共有フォルダだけ。

---

## 前提の確認

作業前に次を確かめる。満たしていないものがあれば、先にそこを直す。

```bash
# 1. リポジトリがあるか (パスが違う場合は以降の ~/LDRP を読み替える)
ls -d ~/LDRP

# 2. tools/ が入っているブランチか
cd ~/LDRP && git branch --show-current
#   tools/ が無ければ:  git fetch origin && git checkout tools/experiment-dashboard

# 3. python3 があるか (標準ライブラリだけで動く。PyYAML も不要)
python3 -V

# 4. iCloud Drive が有効か
ls -d ~/Library/Mobile\ Documents/com~apple~CloudDocs
```

> `--export` は **標準ライブラリだけ**で動く。conda 環境も PyYAML も要らない。
> システムの `python3` でよい。

---

## 手順

### 1. 手動で 1 回動かして確認する

```bash
cd ~/LDRP
python3 tools/collect_runs.py \
    --export ~/Library/Mobile\ Documents/com~apple~CloudDocs/LDRP_runs \
    --machine 黒 \
    --repo ~/LDRP
```

期待する出力:

```text
[export] 123 run(s), 39 model dir(s) newly copied (31.3 MB) -> .../LDRP_runs/黒
```

**`--machine 黒` の値は白側の設定 (`collect_config.yaml` の `label: 黒`) と
完全に一致していなければならない。** ずれると白から見えない。

書き出されたものを確認する。

```bash
D=~/Library/Mobile\ Documents/com~apple~CloudDocs/LDRP_runs/黒
ls "$D"                      # runs.jsonl と models/ があるはず
wc -l "$D/runs.jsonl"        # 1 run 1 行
du -sh "$D"                  # 数十 MB 程度
```

### 2. 1 時間ごとに回す

```bash
cp ~/LDRP/tools/com.ldrp.export-runs.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.ldrp.export-runs.plist
launchctl start com.ldrp.export-runs
tail -f /tmp/ldrp_export.log
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

`plan.md` と `collect_config.yaml` は `.gitignore` に入っているので、
**pull しても降ってこないし、こちらから push されることもない**。

---

## 仕様

### 書き出されるもの

```text
LDRP_runs/黒/
├── runs.jsonl        全 run のレコード (1 run 1 行)
└── models/
    └── {algo}_{map}_{N}agents_seed{S}_step{T}/
        └── path/agent.th      評価に使う方策 (RNNAgent の state_dict)
```

| 項目 | 内容 |
|---|---|
| 対象の run | `~/LDRP/src/epymarl/{results,tmp_results}/sacred` 配下すべて |
| モデルを出す run | **`done` (完遂) のみ**。`failed` / `stalled` は出さない |
| モデルのステップ | **最終 checkpoint のみ**。途中のものは出さない |
| 除外するファイル | `opt.th` / `agent_opt.th` / `critic_opt.th` (optimizer state) と `mixer.th`。どれも評価には使わない |
| 再送 | **しない**。すでに共有フォルダにあるものは飛ばす (初回だけ転送) |
| 容量の目安 | モデル 1 件 平均 0.12MB。実測 39 件で 4.8MB |

`mixer.th` は QMIX の学習を再開するときにしか使わず、評価側は読まない。
それでいて `agent.th` の 15 倍あり (平均 1.24MB 対 80KB)、以前は共有フォルダの
93% を占めていたので既定で外した。元のマシンには残っているので、再開したく
なったら `--fetch-mixer` を付けて送り直せる。

`runs.jsonl` には各 run の `config.json` が**全文**入る (86 キー)。
`lr` / `gamma` / `batch_size` / `mixer` / LaRe のフラグなどが白側で見える。
1 run 約 4KB なので、100 run でも 0.5MB 程度。

### 読み方が軽い理由

巨大なファイルは開かない。

| ファイル | 扱い |
|---|---|
| `config.json` (~2.5KB) | 全部読む |
| `run.json` (~5KB) | 全部読む |
| `cout.txt` (0〜650KB) | **末尾 64KB だけ** |
| `metrics.json` (1〜5MB) | **末尾 256KB だけ**、しかも `cout.txt` が空のときだけ |
| `info.json` (2.5MB) | **開かない** |

---

## やってはいけないこと

| してはいけない | 理由 |
|---|---|
| `tools/plan.md` を作る / 編集する | 実験計画は白が master。こちらから触ると食い違う。`.gitignore` されているので pull でも降ってこない |
| `--notion` を付ける | Notion への書き込みは白が担当。二重に書くと競合する |
| `--publish-models` / `--publish-push` を付ける | モデルの保管リポジトリへの push は白が担当 |
| `--machine` の値を変える | 白の `collect_config.yaml` の `label: 黒` と一致していないと収集されない |
| リモートログイン (SSH) を有効にする | この構成では不要。待受ポートを開けないのが drop モードの利点 |

---

## トラブルシュート

**白から「no runs.jsonl」と言われる**

```bash
ls -la ~/Library/Mobile\ Documents/com~apple~CloudDocs/LDRP_runs/黒/runs.jsonl
```

- ファイルが無い → まだ `--export` を 1 回も実行していない
- ファイルはあるが白に見えない → iCloud の同期待ち。両方のマシンでオンライン確認
- `--machine` の値が `黒` 以外になっていないか確認

**白から「drop data is N h old」と警告が出る**

launchd が動いていない。

```bash
launchctl list | grep ldrp
tail -20 /tmp/ldrp_export.log
```

**モデルが出てこない**

完遂した run が無い可能性がある。状態を確認する。

```bash
cd ~/LDRP
python3 tools/collect_runs.py --hosts local --machine 黒 --repo ~/LDRP --format summary
```

`OK=` が 0 なら、まだ完遂した run が無い (`FAIL` / `STALL` ばかり)。

**iCloud のファイルが雲マークになる**

「ストレージを最適化」が有効だと実体が消える。読むときに自動ダウンロードされるので
動くが、オフラインだと失敗する。気になるなら
システム設定 → Apple ID → iCloud → iCloud Drive で最適化を切る。

---

## 白側で何が起きるか (参考)

黒が書き出したものは、白が 30 分ごとに読み込む。

```text
白: collect_runs.py --cache ... --fetch-models models_inbox
      --publish-models ~/LDRP_models --publish-commit
    ├─ 黒/runs.jsonl を読んで進捗を反映 (ダッシュボードの conditions 表)
    ├─ 黒/models/ からモデルを回収
    └─ 保管リポジトリ (private) へ配置して commit
```

黒はこの流れに関与しない。**書き出したら終わり。**
