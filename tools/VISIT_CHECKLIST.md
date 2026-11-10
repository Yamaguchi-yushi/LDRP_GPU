# 他マシンにアクセスできる日の作業手順

> **接続先 (IP アドレス・ユーザー名・SSH の接続名) はこのファイルに書かない。** このリポジトリは公開されている。
> 実際の値は `tools/collect_config.yaml` (gitignore 済み) の `hosts:` と `~/.ssh/config` を見る。以下では `<GPU1 の接続先>` のように書く。

**対象: 黒 / M2 / GPU2。GPU1 と白は原則なにもしなくてよい。**

白からは SSH で届かない (黒 / M2) か、ネットワークごと落ちている (GPU2) ため、
現地でしかできない作業をここにまとめる。**上から順に実行すれば終わる。**

---

## 現地に着く前の状況 (2026-09-23 15:00 時点・実機で確認済み)

| マシン | 状態 | 今日やること | 優先 |
|---|---|---|---|
| 黒 | **生きているが遊んでいる** (export 09/23 14:35 / 実行中 0 本) | **学習を投入する** | **高** |
| M2 | **8 日エクスポートが止まっている** (最終 09/15 12:35) | 生死確認 → 再開 | **高** |
| GPU2 | SSH が **タイムアウト** | 電源と結線 | 中 |
| GPU1 | 接続 OK・2 本実行中 | 止まった 1 本だけ確認 | 低 |

**黒と M2 が本命。** 残り **229 seed** に対して、いま動いているのは GPU1 の 2 本だけ。
黒が空いているのが一番もったいない。

---

## 0. 白でやっておくこと (出発前)

このチェックリストが役に立つには、**先に白から push されている**必要がある。
現在 `tools/` に未コミットの変更がある。

```bash
cd ~/LDRP && git status --porcelain tools/
```

いま出るはず (いずれも黒 / M2 側では使わないが、`git pull` を打つ以上は揃えておく):

| ファイル | 内容 |
|---|---|
| `tools/collect_runs.py` | 学習曲線の書き出し (`--export-curves` / `_meta.json` v2) |
| `tools/dashboard/static/app.js` | machines 表の列ずれ修正 / モデル回収の分母を計画内に |
| `tools/collect_config.example.yaml` `tools/README.md` | 上に合わせた更新 |

> `tools/collect_config.yaml` と `tools/plans/` は **commit しない** (IP とユーザ名が入る)。

---

## 1. 黒 (共有フォルダ経由) — **学習を投入する**

収集側は正常に動いている (09/23 14:35 に export 済み、モデルも回収・purge 済みで
共有フォルダは 1.0MB)。**問題は 1 本も学習していないこと。**

### 1-1. 生きているか / 空いているかを確認

```bash
ps aux | grep '[t]rain.py'          # 何も出なければ空いている
launchctl list | grep ldrp          # 真ん中が 0 なら export は正常
tail -5 /tmp/ldrp_export.log
```

### 1-2. 更新を取り込む

```bash
cd ~/LDRP && git pull
```

### 1-3. 学習を投入する

`train.py` の条件を書き換えて起動する (条件はファイル内に直書き)。

```bash
cd ~/LDRP && nohup python train.py > /tmp/train_$(date +%m%d_%H%M).log 2>&1 &
```

**何を回すかは「付録: 残っている実験」を見る。** 残り 229 seed のうち
**約 9 割が `task_assign=PPO`** なので、そこから埋めるのが早い。

> 黒は M1/M2 Mac なので CPU 学習。**台数の多い条件 (10agent 150M) は GPU1 に回し、
> 黒には 8x5 5agent 20M のような軽い条件**を置くほうが全体の回りが良い。

### 1-4. スリープしないことを確認

投入したら、蓋を閉じても止まらないか見ておく。

```bash
pmset -g | grep -E "^ *sleep|disablesleep"
caffeinate -dims &      # 必要ならこれを添えて起動する
```

---

## 2. M2 (共有フォルダ経由) — **8 日沈黙している**

**09/15 12:35 を最後に export が止まっている。** ただし止まる 6 分前
(09/15 12:29) に **3 本を起動している**ので、**まだ走っている可能性がある**。

| seed | 条件 | 最後に見えた進捗 |
|---|---|---|
| 603047445 / 588896263 / 159768905 | `map_aoba00 10agent` qmix + LaRe (dbct) | いずれも 0% |

### 2-1. まず生死を見る (**消す前に確認する**)

```bash
ps aux | grep '[t]rain.py'
uptime                               # 再起動していないか
tail -5 /tmp/ldrp_export.log
launchctl list | grep ldrp
```

- **プロセスが生きていた** → 学習は無事。**止まっていたのは export だけ**なので 2-2 へ
- **プロセスが無い** → 09/15 に落ちている。`cout.txt` の末尾で原因を見る

```bash
tail -30 ~/LDRP/src/epymarl/results/sacred/qmix/'drp_env:drp_safe-10agent_map_aoba00-v2'/1/cout.txt
```

### 2-2. export を復活させる

```bash
cd ~/LDRP && git pull
launchctl unload ~/Library/LaunchAgents/com.ldrp.export-runs.plist 2>/dev/null
launchctl load   ~/Library/LaunchAgents/com.ldrp.export-runs.plist
launchctl start  com.ldrp.export-runs
tail -f /tmp/ldrp_export.log        # 1 回動くのを見届けて Ctrl-C
```

何も出ない / 未登録だった場合:

```bash
cp ~/LDRP/tools/com.ldrp.export-runs.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.ldrp.export-runs.plist
launchctl start com.ldrp.export-runs
```

### 2-3. 止まっていた原因を潰す

8 日も黙っていたので、**スリープが最有力**。

```bash
pmset -g | grep -E "^ *sleep|disablesleep"
pmset -g log | grep -iE "Sleep|Wake" | tail -20
```

蓋を閉じて使うなら `caffeinate -dims` を学習と一緒に起動しておく。

---

## 3. GPU2 — **タイムアウト**

白から SSH が通らない (09/23 時点で `Operation timed out`)。
**IP はこれまでに 2 回変わっている**ので、まず現地で今の IP を確認する。

```bash
# GPU2 の本体で
ip a | grep 'inet '          # 研究室ネットワークのアドレスを控える
sudo systemctl status ssh    # 22 ではなく 2222 で待っているか
nvidia-smi                   # ← 前回 nvidia-smi が無かった。GPU が見えるか要確認
```

IP が変わっていたら**白の `tools/collect_config.yaml`** と `~/.ssh/config` の
GPU2 の接続名 を直す (このファイルは gitignore なので手で書き換える)。

```bash
# 白に戻ってから疎通確認
ssh -o ConnectTimeout=10 <GPU2 の接続先> 'hostname; nvidia-smi --query-gpu=name --format=csv,noheader'
```

復旧したら白側は**何もしなくてよい**。次の収集でモデルまで自動で回収される。

> ラベル注意: **GPU1 と GPU2 の接続先を取り違えやすい** (実際に一度逆に設定していた)。
> 以前は逆に設定されていた (09/13 に修正済み)。

---

## 4. GPU1 — 軽く見るだけ

接続 OK。`aoba00 5agent mat safe` を 2 本 (41% / 25%) 実行中。**触らなくてよい。**

1 点だけ: **seed 291051464** (`map_8x5 7agent 30M` qmix + LaRe) が **75% で stalled**。
計画内の条件なので、放置すると 1 seed 欠ける。

```bash
ssh <GPU1 の接続先>
ps aux | grep '[t]rain.py'      # このプロセスが残っているか
```

生きていれば放置。死んでいれば回し直す (75% 分は捨てになる)。

> 白からの SSH が**ときどき**タイムアウトしている (`/tmp/ldrp_quick.log`)。
> 接続そのものは生きているので、ネットワークが不安定なだけ。急がなくてよい。

---

## 5. 白に戻ってから

```bash
cd ~/LDRP
/opt/anaconda3/envs/ldrp/bin/python tools/collect_runs.py \
    -c tools/collect_config.yaml --cache tools/.run_cache.jsonl \
    --fetch-models ~/models_inbox --publish-models ~/LDRP_models \
    --purge-drop --format status
```

`--purge-drop` は**回収できたモデルだけ**を共有フォルダから消す
(ローカルに同じサイズのファイルがあることを確認してから消す)。

ダッシュボードで確認する。

```bash
/opt/anaconda3/envs/ldrp/bin/python tools/dashboard/app.py --collect-on-start
```

> **`python` ではなく conda の python を使う。** `.venv/bin/python` には PyYAML が
> 無く、設定を読めずにホストラベルが化ける (白の run が二重に溜まる)。

期待する状態:

| 見るところ | 期待 |
|---|---|
| machines の `いつの情報か` | 黒 / M2 が 1 時間以内。赤くない |
| machines の `実行中` | 黒が 0 でなくなっている |
| machines の `モデル回収` | 緑の `n/n` (分母は **計画内の完了**。計画外は数えない) |
| 条件表の 📦 | done の行に付く |

---

## 付録: 残っている実験 (2026-09-23 時点)

**84 条件中 — ✔完了 33 / 途中 10 / 未着手 41。残り 229 seed。**

残りを手法で割ると、**大半が `task_assign=PPO`**:

| algo | setting | task | dynamic | 残り seed |
|---|---|---|---|---|
| mat | safe | PPO | — | 30 |
| mat | safe | PPO | yes | 28 |
| qmix | safe | PPO | — | 25 |
| qmix / mappo / mat | `8x5_2 10M 8x5_3 5M` | PPO | — | 各 15 |
| mat | `8x5_2 10M 8x5_3 5M` | PPO | yes | 15 |
| mappo | safe | PPO | — | 15 |
| mat | `8x5_2 10M aoba00_2 5M` | PPO | —/yes | 各 15 |
| qmix / mappo | `8x5_2 10M aoba00_2 5M` | PPO | — | 各 12 |
| (TP の残り) | — | — | — | 計 14 |

あと 1〜2 seed で埋まる条件 (**ここから潰すと「完了」条件が増える**):

| 条件 | 学習 | あと |
|---|---|---|
| `map_aoba00 5agent 80M` | 4/5 | 1 |
| `map_aoba00 7agent 100M` | 4/5 | 1 |
| `map_aoba00 5agent 80M` (別系列) | 3/5 | 2 |

正確な残りはダッシュボードの条件表 (学習 n/5) が正。

---

## 付録: 各マシンの役割

| | 接続 | 役割 |
|---|---|---|
| 白 | — | 集約。収集・保管・ダッシュボード。**ここだけが plan.md を持つ** |
| 黒 | 共有フォルダ | 学習 + `--export` するだけ |
| M2 | 共有フォルダ | 同上 |
| GPU1 | SSH | 学習。白が SSH で取りに行く |
| GPU2 | SSH | 同上 |

黒 / M2 では **`--notion` / `--publish-models` / `plan.md` に触らない**
(詳細は [SETUP_export_machine.md](SETUP_export_machine.md))。
