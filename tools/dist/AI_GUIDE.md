# tools/ — AI エージェント向けの作業ガイド

このファイルは、`tools/` で作業する AI エージェント (Claude Code など) が最初に読む説明書です。
人間向けの使い方は [README.md](README.md) にあります。**ここには「守ること」と「作業の手順」を書きます。**

利用者はこのガイドを読んだ AI に「tools をセットアップして」「マシンを追加して」のように頼みます。
AI は下の手順に沿って作業し、**判断が要るところ (接続先・実験計画・学習コードの変更) は必ず利用者に聞いてください。**

---

## 0. 最初に必ず守ること

### 0.1 公開リポジトリに機密を書かない

このリポジトリは**他の人からも見えます**。次のものを、git で追跡されるファイル (コード・README・メモ・コミットメッセージ) に**絶対に書かないでください**。

| 書いてはいけないもの | 置く場所 |
|---|---|
| IP アドレス (`10.x` / `172.16〜31.x` / `192.168.x` を含む) | `~/.ssh/config` の `HostName` |
| SSH のユーザー名、`user@host` の形の接続先 | `~/.ssh/config` の `User` |
| パスワード・トークン | 書かない (鍵認証を使う) |
| 自分のホームのパス (`/Users/<名前>` / `/home/<名前>`) | `~` で書く |
| 自分の実験計画・収集結果 | `tools/plans/` / `tools/.run_cache.jsonl` (gitignore 済み) |

`tools/collect_config.yaml` は gitignore 済みですが、**そこにも IP やユーザー名は書かず**、
`~/.ssh/config` の Host 名だけを書くようにしてください (設定を誰かに見せたり、誤って `git add -f` したりしても漏れないように)。

commit の前に、追跡対象に機密が入っていないかを確かめてください。

```bash
git diff --cached | grep -nE '\b(10\.[0-9]+|172\.(1[6-9]|2[0-9]|3[01])|192\.168)\.[0-9]+\.[0-9]+\b|[a-z_][a-z0-9_.-]*@[0-9]+\.[0-9]+|/(Users|home)/[A-Za-z0-9_.-]+' \
  && echo "★ 機密らしきものがある。commit しない" || echo "OK"
```

### 0.2 GPU は共有されている

- **他の人のプロセスを止めない・触らない。** `kill` / `pkill` は利用者に確認してから、自分のプロセスにだけ使う
- 他の人のファイル・学習結果を読まない・消さない。収集するのは設定の `repos:` に書いた自分のリポジトリだけ
- tools は `ps -U <自分の uid>` で**自分のプロセスだけ**を見る。この挙動を変えない
- ダッシュボードは `127.0.0.1` で待ち受ける。共有マシンで `--host 0.0.0.0` にしない (他の人から見える)

### 0.3 学習・評価のコードは勝手に変えない

tools は sacred の出力を**読むだけ**で、学習や評価の挙動に影響しません。次のファイルは、
**利用者が明示的に頼んだときだけ**変更してください。変更するときは、何がどう変わるかを先に説明します。

- `train.py` / `test.py` / `run.py` / `runner.py` / `src/` 以下

### 0.4 その他

- `git push --force` / `git commit --amend` / `--no-verify` は使わない
- 学習結果 (`src/epymarl/results/`) や保管したモデルを消さない
- 動かして確かめてから「できた」と報告する (手順 6 を参照)

---

## 1. 全体の仕組み

```text
各マシンの sacred の出力 (config.json / run.json / cout.txt / metrics.json)
        │
        │  ① local: このマシンのフォルダを読む
        │  ② ssh  : collect_runs.py 自身を送り込んで相手のマシンで走査 (相手には何も置かない)
        │  ③ drop : 相手のマシンが共有フォルダに書き出したものを読む (--export)
        ▼
collect_runs.py  ──  状態の判定 (derive)・計画表との照合・モデル回収・学習曲線の書き出し
        │
        ├─ tools/.run_cache.jsonl        収集結果のキャッシュ
        ├─ dashboard/app.py              ブラウザの画面 (127.0.0.1:8765)
        └─ dashboard/mini.py             デスクトップの進捗パネル (macOS)
```

| ファイル | 役割 |
|---|---|
| `collect_runs.py` | 収集・状態判定・モデル回収・学習曲線の書き出し。単体で CLI としても動く |
| `plan.py` | 実験計画の表 (Markdown) を読み、run と照合する |
| `eval_report.py` | 評価結果 (`results/summary.csv`) の集計。無ければ評価タブは空になる |
| `dashboard/app.py` | 画面と API。集計は collect_runs.py / plan.py を import して使う (再実装しない) |
| `dashboard/mini.py` | 常駐パネル。`/api/mini` を読むだけ |
| `setup_launchd.py` | macOS の定期実行を登録する |
| `collect_config.yaml` | 接続先などの設定。**gitignore 済み。見本は `collect_config.example.yaml`** |
| `plans/*.md` | 実験計画の表。**gitignore 済み。見本は `plan.example.md`** |

### 1.1 run の見分け方

- run の ID: `{label}:{results|tmp_results}/{algo}/{env_key}/{run_id}` (例: `mac:results/qmix/drp_env:drp_safe-5agent_map_8x5-v2/3`)
- 条件の軸: map / 台数 N / t_max / algo / setting (LaRe を使わなければ `safe`) / task arrival / task assign / dynamic
- 状態:

| 状態 | 判定 |
|---|---|
| `done` (OK) | sacred が COMPLETED で、t_max まで届いた |
| `running` (RUN) | sacred が RUNNING で heartbeat が新しい。**Ctrl-Z で一時停止中のものも含む** (`paused=True`) |
| `stalled` (STALL) | sacred が RUNNING なのに heartbeat が `stale_minutes` 以上止まっていて、プロセスも見つからない |
| `failed` (FAIL) | sacred が FAILED / INTERRUPTED (例外か Ctrl-C) |
| `short` (SHORT) | COMPLETED だが t_max に届いていない |

- 終了予定は**直近のペース** (収集ごとに残す heartbeat とステップ数の記録点、直近 6 区間の中央値) から出す

### 1.2 計画表との照合

`plan.matches()` の規則。**ここがずれると「run が表に入らない」になる。**

| 軸 | 規則 |
|---|---|
| map / 台数 | 完全一致 (章見出しの上の単独行 `8x5` → `map_8x5`) |
| t_max | **M 単位に丸めて比較** (`20050000` と `20M` は一致) |
| algorithm | 小文字で一致 (`QMIX` → `qmix`) |
| setting | LaRe を使わない run は `safe`。非 Safe の環境 (`drp-`) は `unsafe` |
| task arrival | 表が空欄なら問わない |
| task assign | 表の `TP` / 空欄 / `-` は run 側の空文字と一致 |
| dynamic | 表が空欄なら問わない |

- 1 条件 = 条件行 1 行 + その下の seed 行。**seed 行の数が枠の数** (5 行 = 5 seed)
- seed 欄が空の枠には、計画に seed が書かれていない run が開始順に入る
- 枠に入るのは `done` と `running` だけ。失敗した run は枠の下に別行で出る

### 1.3 評価用モデルの名前

`collect_config.yaml` の `eval_naming` で決まる。**既定は `template`**。

| eval_naming | 置き場所と名前 | 使う評価コード |
|---|---|---|
| `template` | `src/all_policy/models/safe/{map}_{N}_{algo}.th` (seed 0)、`..._seed1.th` 以降 | このリポジトリの `src/all_policy/policy.py` |
| `ldrp` | `{map}_{N}_{algo}[_{tag}][_{assign}][_dyn]_base_seed{i}.th` | 拡張版 LDRP の評価 |

評価コード (`policy.py`) がどの名前を開くかを確かめてから選んでください。**評価コードの方を変えない。**

---

## 2. セットアップの手順

利用者に「tools をセットアップして」と頼まれたら、この順に進めてください。

### 2.1 python の確認

```bash
python -c "import yaml; print('OK')"
```

`OK` が出ない python では設定が読めません。利用者にどの環境 (conda の環境名など) を使うか聞き、
以降のコマンドはすべてその python で実行してください。

### 2.2 集めるマシンを利用者に聞く

次を聞いてください。**パスワードは聞かない・受け取らない。**

| 聞くこと | 例 |
|---|---|
| マシンの呼び名 (画面に出るラベル) | `mac` / `gpu1` / `laptop` |
| そのマシンへの入り方 | このマシン自身 / SSH で入れる / SSH で入れない |
| SSH で入れる場合: `~/.ssh/config` の Host 名 | `lab-gpu1` |
| LDRP のリポジトリの場所 | `~/LDRP` |

SSH の Host 名がまだ無い場合は、利用者に `~/.ssh/config` へ登録してもらいます (アドレスとユーザー名は利用者が入力する)。
登録後、パスワードなしで入れることを確かめます。

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 lab-gpu1 hostname
```

### 2.3 設定ファイルを作る

```bash
cp tools/collect_config.example.yaml tools/collect_config.yaml
```

`hosts:` を 2.2 の内容で書き換えます。**`ssh:` には Host 名だけを書く。**

### 2.4 実験計画の表を作る

```bash
mkdir -p tools/plans
cp tools/plan.example.md tools/plans/plan_main.md
```

利用者に、回したい条件 (マップ・台数・t_max・アルゴリズム・seed 数) を聞いて表を書きます。
**t_max は train.py の設定と M 単位で一致させる** (1.2 を参照)。

### 2.5 動かして確かめる

手順 6 の確認を行ってから、利用者に結果を報告します。

### 2.6 定期実行 (利用者が望めば)

```bash
python tools/setup_launchd.py collect quick mini      # macOS の集約マシン
python tools/setup_launchd.py export --machine laptop  # SSH で入れないマシン
```

Linux は cron を使います (README の「定期実行」)。

---

## 3. よくある依頼と手順

| 依頼 | 手順 |
|---|---|
| マシンを追加して | 2.2 → `collect_config.yaml` の `hosts:` に追記 → 手順 6 |
| 計画に条件を足して | `tools/plans/*.md` に条件行 + seed 行を足す。1.2 の規則に合わせる |
| run が表に入らない | 表の map・台数・t_max・algorithm と、run の値を比べる (手順 5.2) |
| モデルを評価で使えるようにして | `--fetch-models ~/models_inbox --install-models`。名前は 1.3 |
| 学習曲線を出して | `--export-curves ~/curves`。学習したマシンに tensorboard が要る |
| 実行待ちを表に出して | README の「実行待ちの表示」の 4 か所を `train.py` に足す。**学習コードなので、利用者に確認してから** |
| SSH で入れないマシンから集めて | README の「SSH が通らないマシン」。相手のマシンでの作業は利用者に頼む |
| 一時停止中の run が「停止」と出る | 相手のマシンの tools が古い可能性。`git pull` を利用者に頼む |

---

## 4. コードを直すときの約束

- **既存の挙動を変えない。** 新しい挙動は設定 (`collect_config.yaml`) で切り替え、既定値は今の挙動にする
- 集計や状態判定を dashboard 側に再実装しない。collect_runs.py / plan.py の関数を使う
- リモートで動く部分 (`scan` / `scan_run_dir` とそこから呼ぶ関数) は**標準ライブラリだけ**で書く
  (SSH 先には何もインストールしない前提のため)。Python 3.8 で動く書き方にする
- 画面・ログに出す文字列は英語、コメントは日本語でよい
- 直したら手順 6 の確認を必ず行う

---

## 5. 調べ方

### 5.1 状態を一覧する

```bash
python tools/collect_runs.py -c tools/collect_config.yaml --cache tools/.run_cache.jsonl --format table
python tools/collect_runs.py -c tools/collect_config.yaml --cache tools/.run_cache.jsonl --format summary
```

### 5.2 run が計画表に入らない理由を調べる

```bash
python - <<'EOF'
import sys; sys.path.insert(0, "tools")
import collect_runs as CR, plan as PLAN
conf = CR.load_config("tools/collect_config.yaml")
rows = [CR.derive(r, 90) for r in CR.dedupe(CR.read_cache("tools/.run_cache.jsonl")) if r.get("kind") != "batch"]
conds = PLAN.parse_plans(PLAN.find_plans(None))
for d in rows:
    if d["state"] in ("done", "running") and not any(PLAN.matches(c, d) for c in conds):
        print(d["map"], d["agents"], d["t_max"], d["algo"], d["setting"], repr(d["task_assign"]), d["seed"])
EOF
```

出てきた値と、表の章見出し・条件行を見比べてください。

### 5.3 ダッシュボードの API

| API | 中身 |
|---|---|
| `GET /api/train` | 全 run・計画表・マシンごとの状況 |
| `GET /api/mini` | パネル用の要約 (1KB 未満) |
| `GET /api/status` | 収集中かどうか・最後の収集時刻 |
| `POST /api/collect` | 裏で収集を始める |

---

## 6. 作業後の確認 (必ず行う)

1. 構文

   ```bash
   python -c "import ast; [ast.parse(open(f).read()) for f in ['tools/collect_runs.py','tools/plan.py','tools/dashboard/app.py','tools/dashboard/mini.py']]; print('OK')"
   ```

2. 収集が通る (`[info] <label>: N runs` が全マシン分出る)

   ```bash
   python tools/collect_runs.py -c tools/collect_config.yaml --cache tools/.run_cache.jsonl --format none
   ```

3. ダッシュボードが応答する (別ポートで起動して確かめ、終わったら止める)

   ```bash
   python tools/dashboard/app.py --port 8799 &
   sleep 3; curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8799/api/mini   # 200
   kill %1
   ```

4. commit する前に 0.1 の機密チェックを行う

結果を利用者に報告するときは、実際に出た件数や表示を添えてください。確かめていないことは「確かめていない」と書きます。
