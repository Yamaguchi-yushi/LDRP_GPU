# tools — 学習の進捗・モデル・学習曲線を 1 か所に集める

複数のマシン (自分の Mac、研究室の GPU、SSH が通らないノート PC) で回している学習を、
**1 台のマシンに自動で集めて表示する**道具です。手で Notion やスプレッドシートに書き写す作業が要らなくなります。

| できること | 使うもの |
|---|---|
| 全マシンの run の状態・進捗・終了予定を一覧する | `collect_runs.py` / ダッシュボード |
| 実験計画の表 (何を何 seed 回すか) に対して、埋まり具合を見る | ダッシュボードの「conditions」 |
| 完了した run の方策モデルを回収し、評価で読める名前で置く | `collect_runs.py --fetch-models` |
| 学習曲線を CSV に書き出す (make_graph で作図) | `collect_runs.py --export-curves` |
| デスクトップに進捗を常駐表示する (macOS) | `dashboard/mini.py` |

**AI (Claude Code など) にセットアップや作業を頼む場合は、[CLAUDE.md](CLAUDE.md) を読ませてください** (Claude Code は `tools/` で作業するとき自動で読みます)。守ること・手順・調べ方がまとめてあります。

学習や評価のコード (`train.py` / `test.py` / `src/`) には一切手を入れません。sacred が書き出す
`config.json` / `run.json` / `cout.txt` / `metrics.json` を読むだけです。

---

## 目次

1. [はじめに: 公開リポジトリでの注意](#1-はじめに-公開リポジトリでの注意)
2. [準備](#2-準備)
3. [設定ファイル](#3-設定ファイル)
4. [実験計画の表](#4-実験計画の表)
5. [使い方](#5-使い方)
6. [SSH が通らないマシン](#6-ssh-が通らないマシン)
7. [定期実行](#7-定期実行)
8. [評価用モデルの回収](#8-評価用モデルの回収)
9. [学習曲線の書き出し](#9-学習曲線の書き出し)
10. [実行待ちの表示 (任意)](#10-実行待ちの表示-任意)
11. [困ったとき](#11-困ったとき)

---

## 1. はじめに: 公開リポジトリでの注意

**このリポジトリは他の人からも見えます。** 次のものは commit しないでください。
どれも `.gitignore` 済みですが、`git add -f` や別名での保存に注意してください。

| ファイル | 中身 |
|---|---|
| `tools/collect_config.yaml` | 接続先 (ホスト名・ユーザー名) |
| `tools/plans/` | 自分の実験計画 |
| `tools/.run_cache.jsonl` | 収集した run の記録 |

- 接続先の **IP アドレスやユーザー名は `~/.ssh/config` に書き**、設定ファイルには Host 名だけを書く
- README や手順書のメモにも、IP アドレス・ユーザー名・パスワードを書かない
- **GPU は共有です。** このツールは自分のユーザーのプロセスだけを見るので、
  他の人の学習を自分の run と取り違えることはありません
- ダッシュボードは既定で `127.0.0.1` (自分のマシンの中) だけで待ち受けます。
  共有マシンで `--host 0.0.0.0` にすると他の人からも見えるので、しないでください

---

## 2. 準備

必要なもの:

- 集約するマシン: Python 3.8 以上と **PyYAML** (epymarl が使うので LDRP の環境には入っている)
- 学習曲線を書き出すなら、学習したマシンに **tensorboard** (`pip install tensorboard`)
- SSH で集めるマシン: `python3` だけ (標準ライブラリで動く。何もインストールしない)

以降の `python` は、**PyYAML が入った python** を指します (例: `conda activate <環境名>` した後の python)。
入っていない python で動かすと設定が読めず、マシン名がホスト名に化けます。

```bash
python -c "import yaml; print('OK')"      # OK と出れば使える
```

---

## 3. 設定ファイル

```bash
cp tools/collect_config.example.yaml tools/collect_config.yaml
```

`hosts:` に集めたいマシンを書きます。`label` は画面に出る名前で、自由に決めてかまいません。

```yaml
hosts:
  - label: mac          # このマシン
    ssh: local
    repos: [~/LDRP]
  - label: gpu1         # 研究室の GPU
    ssh: lab-gpu1       # ~/.ssh/config の Host 名
    python: python3
    repos: [~/LDRP]
```

SSH の接続先は `~/.ssh/config` に登録します (このファイルは自分のマシンにだけ置かれ、git には入りません)。

```text
Host lab-gpu1
  HostName <GPU のアドレス>
  User <自分のユーザー名>
```

パスフレーズ無しの鍵で入れるようにしておきます (自動収集で毎回聞かれると止まるため)。

```bash
ssh-copy-id lab-gpu1
ssh -o BatchMode=yes lab-gpu1 hostname     # パスワードを聞かれずにホスト名が出れば OK
```

---

## 4. 実験計画の表

「どの条件を何 seed 回すか」を Markdown の表で書いておくと、ダッシュボードが**あと何本回せばよいか**を出します。

```bash
mkdir -p tools/plans
cp tools/plan.example.md tools/plans/plan_main.md
```

書き方は見本の冒頭に書いてあります。要点だけ:

- 章見出し `## 3agent  10M` で台数と t_max を、その上の単独行 (`8x5` など) でマップを決める
- 条件行 (seed と machine が空) に setting / algorithm を書き、下に 5 行続けると 5 seed の枠になる
- **seed は空欄のままでよい。** train.py は起動時に seed をランダムに決めるので、回った run が空いている枠に入る
- 計画に無い run は表に出ません (実行中の一覧には出ます)

---

## 5. 使い方

### 収集して一覧する

```bash
python tools/collect_runs.py -c tools/collect_config.yaml --cache tools/.run_cache.jsonl --format table
```

| 状態 | 意味 |
|---|---|
| `OK` | t_max まで学習し終えた |
| `RUN` | 学習中。Ctrl-Z で一時停止しているものも含む (画面では「⏸ 一時停止中」) |
| `STALL` | 学習中のはずなのに、90 分以上進んでいない (プロセスが落ちた可能性) |
| `FAIL` | 例外で終了した、または Ctrl-C で止めた |
| `SHORT` | 正常終了したが t_max に届いていない |

終了予定は**直近のペース**から出します。一時停止やスリープで止まっていた時間は数えません。

### ダッシュボード

```bash
python tools/dashboard/app.py --collect-on-start
```

ブラウザで <http://127.0.0.1:8765> を開きます。開いている間は 15 分ごとに自動で収集します。

| 画面 | 内容 |
|---|---|
| machines | マシンごとの実行中・完了・モデル回収・情報の新しさ |
| running now | 実行中の run と終了予定。条件名を押すと計画表の該当行へ移動する |
| conditions | 実験計画の表。各条件が何 seed 埋まったか |

### デスクトップの進捗パネル (macOS)

```bash
python tools/dashboard/mini.py &
```

画面の隅に小さなパネルが出ます。クリックでダッシュボードを開き、ドラッグで移動、右クリックでメニューです。
マシンごとに「次に終わる予定」(実行中なら) か「最後に終わった時刻」を出します。

---

## 6. SSH が通らないマシン

自宅のノート PC など SSH で届かないマシンは、**共有フォルダ** (iCloud Drive / Dropbox など) を経由して集めます。

1. 集約するマシンの設定で、そのマシンを `drop: true` にする

   ```yaml
   drop_root: ~/Library/Mobile Documents/com~apple~CloudDocs/LDRP_runs
   hosts:
     - label: laptop
       drop: true
   ```

2. ノート PC 側で、同じ共有フォルダへ書き出す (label を集約側と**完全に同じ**にする)

   ```bash
   python tools/collect_runs.py --export ~/Library/Mobile\ Documents/com~apple~CloudDocs/LDRP_runs \
       --machine laptop --repo ~/LDRP
   ```

進捗と、完了した run の方策モデルだけが共有フォルダに書かれます。回収が済んだモデルは集約側が消すので
(`--purge-drop`)、共有フォルダの容量はほとんど使いません。

---

## 7. 定期実行

### macOS

`setup_launchd.py` が、このスクリプトを動かした python とリポジトリの場所から設定を作って登録します。
**PyYAML が入った python で実行してください。**

```bash
# 集約するマシン
python tools/setup_launchd.py collect quick mini
#   collect: 15 分ごとに全マシンを収集 → 完了の通知 → モデル回収
#   quick  : 2 分ごとに実行中の run だけ確認 (完了をすぐ知る)
#   mini   : ログイン時に進捗パネルを出す

# SSH が通らないマシン
python tools/setup_launchd.py export --machine laptop

# 解除
python tools/setup_launchd.py remove collect
```

`--dry-run` を付けると、登録せずに作る内容だけを表示します。

### Linux (GPU など)

SSH で集めるマシンには何も要りません (集約側がスクリプトを送り込んで実行する)。
Linux を集約するマシンにする場合や、共有フォルダへ書き出す場合は cron を使います。

```bash
crontab -e
# 15 分ごとに共有フォルダへ書き出す例
*/15 * * * * cd ~/LDRP && python tools/collect_runs.py --export <共有フォルダ> --machine <label> --repo ~/LDRP >> /tmp/ldrp_export.log 2>&1
```

---

## 8. 評価用モデルの回収

完了した run の方策モデル (`agent.th`) を各マシンから集め、評価 (`test.py` / `run.py`) がそのまま読める名前で置きます。

```bash
python tools/collect_runs.py -c tools/collect_config.yaml --cache tools/.run_cache.jsonl \
    --fetch-models ~/models_inbox --install-models
```

| 置き場所 | 名前 |
|---|---|
| `src/all_policy/models/safe/` | `{map}_{N}_{algo}.th` (seed 0)、`{map}_{N}_{algo}_seed1.th` 以降 |

- 同じ名前のファイルがあれば**上書きしません** (`--overwrite-installed` で上書き)
- seed の番号は、学習時の seed (9 桁) ではなく 0 から振った通し番号です。対応は `~/models_inbox/manifest.jsonl` に残ります
- 評価に使わないファイル (`opt.th` などの最適化器の状態、QMIX の `mixer.th`) は回収しません

保管用のフォルダ (git で管理するなど) にまとめたいときは `--publish-models ~/LDRP_models` を使います。
計画表の名前ごとにフォルダが分かれます。

---

## 9. 学習曲線の書き出し

TensorBoard の記録から、指定した指標を CSV に書き出します。make_graph で作図するときに使います。
学習したマシンに tensorboard が入っている必要があります。

```bash
python tools/collect_runs.py -c tools/collect_config.yaml --cache tools/.run_cache.jsonl \
    --export-curves ~/curves
```

書き出す指標と手法フォルダは設定ファイルの `curves:` で決めます。
make_graph の画面で作った指定 (`curve_spec.json`) があれば `--curve-spec curve_spec.json` で渡します。

```text
~/curves/
  8x5-v2_3agent/                ← 1 条件 = 1 図
    _meta.json                  ← 色・順番・軸・t_max (make_graph が読む)
    QMIX/run-.../run-...-tag-test_return_mean.csv
```

どんな指標が記録されているかは `--metrics-catalog catalog.json` で一覧できます。

---

## 10. 実行待ちの表示 (任意)

`train.py` で 5 本を順に回すと、sacred は run が**始まってから**記録を作るので、2 本目を回している間は
表に 1 本しか見えません。`train.py` に次の 4 か所を足すと、残りの 3 本を「⏳ 実行待ち」として表に出せます。
学習の挙動には影響しません (予定をファイルに書くだけです)。

**(a) `import time` の次**

```python
import json
import os

_SLOT = os.path.expanduser("~/.ldrp/batch_%d.json" % os.getpid())


def _publish_batch(started, cmd=None):
    """残り何本予定かを書く。失敗しても学習は続ける."""
    try:
        os.makedirs(os.path.dirname(_SLOT), exist_ok=True)
        rec = {"pid": os.getpid(), "total": num_runs,
               "started": started, "updated": time.time()}
        if cmd:
            rec["cmd"] = cmd
        with open(_SLOT, "w") as f:
            json.dump(rec, f)
    except OSError:
        pass
```

**(b) `num_runs = ...` と `running_processes = []` の後**

```python
_publish_batch(0)
```

**(c) ループ内の `subprocess.Popen(command, shell=True)` の次** (インデントに注意)

```python
    _publish_batch(i + 1, command)
```

**(d) 最後の `print("All runs completed.")` の前**

```python
try:
    os.remove(_SLOT)
except OSError:
    pass
```

---

## 11. 困ったとき

| 症状 | 見るところ |
|---|---|
| マシン名がホスト名 (`xxx.local`) になる | PyYAML の無い python で動かしている。[2. 準備](#2-準備) |
| SSH のマシンが出てこない | `ssh -o BatchMode=yes <Host 名> hostname` が通るか。パスワードを聞かれると自動収集は止まる |
| 共有フォルダのマシンが「情報が古い」 | 相手のマシンで `--export` が動いているか (`launchctl list \| grep ldrp`)。スリープしていないか |
| 開始直後の run のステップ数が出ない | 学習ログが書かれるまで待つ。sacred の `--beat-interval` を大きくしていると最大その秒数だけ遅れる |
| 計画表に run が入らない | 表の map・台数・t_max・algorithm が run と一致しているか。t_max は M 単位で比べる |
| 評価で「モデルが見つからない」 | `src/all_policy/models/safe/` の名前が `{map}_{N}_{algo}.th` になっているか |
