# 他マシンにアクセスできる日の作業手順

**対象: 黒 / M2 / GPU2。白 (集約機) では何もしなくてよい。**

白からは SSH で届かない (黒 / M2) か、ネットワークごと落ちている (GPU2) ため、
現地でしかできない作業をここにまとめる。**上から順に実行すれば終わる。**

---

## 現地に着く前の状況 (2026-09-15 10:27 時点)

| マシン | 状態 | 今日やること |
|---|---|---|
| 黒 | **生きている** (最終 export 09/15 09:52) | pull + plist 再読込 + mixer 削除 |
| M2 | **7 日止まっている** (最終 export 09/08 16:05) | 原因確認 + 登録 |
| GPU2 | **Network is unreachable** | 電源と結線 |
| GPU1 (cat) | 接続 OK | 何もしなくてよい |

**M2 と GPU2 が本命。** 黒は動いているので優先度は低い。

---

## 0. 事前 (白でやっておくこと)

このチェックリストが役に立つには、**先に白から push されている**必要がある。

```bash
# 白で
cd ~/LDRP && git log --oneline -1 && git status --porcelain tools/
```

`tools/` に変更が残っていたら、先に commit / push する。

---

## 1. 黒 (SSH なし / 共有フォルダ経由)

### 1-1. なぜ来るのが遅かったか確認する

```bash
launchctl list | grep ldrp
#   真ん中が終了コード。0 なら正常、0 以外なら失敗、"-" は未実行

tail -30 /tmp/ldrp_export.log

# まだ走りっぱなしのジョブが残っていないか (前回が終わらないと次が起動しない)
ps -eo pid,etime,command | grep '[c]ollect_runs'
```

`ps` に古い `collect_runs` が残っていたら、それが原因。**モデルのコピーが
iCloud で詰まって終わらなくなっていた**可能性が高い (1-2 の修正で直る)。

### 1-2. 更新を取り込む

```bash
cd ~/LDRP && git pull
launchctl unload ~/Library/LaunchAgents/com.ldrp.export-runs.plist
launchctl load   ~/Library/LaunchAgents/com.ldrp.export-runs.plist
launchctl start  com.ldrp.export-runs
tail -f /tmp/ldrp_export.log     # 1 回動くのを見届けて Ctrl-C
```

この pull で入る変更は 2 つ。**どちらも黒側で動くコードなので pull が要る。**

| 変更 | 効果 |
|---|---|
| `runs.jsonl` を**モデルより先に**書く | 440KB の進捗が 90MB のモデル転送の後ろに並ばなくなる。実測で進捗が 3 時間遅れていた原因 |
| `mixer.th` を送らない | iCloud への転送が **90MB → 6MB**。`mixer.th` は評価に使わず、`agent.th` の 15 倍ある |

### 1-3. 既に上がっている mixer.th を消す

pull しても**過去に送った分は消えない**。iCloud を 84MB 空ける。

```bash
D=~/Library/Mobile\ Documents/com~apple~CloudDocs/LDRP_runs
du -sh "$D"                                   # 前
find "$D"/*/models -name 'mixer.th' -delete
du -sh "$D"                                   # 後
```

`agent.th` は残るのでディレクトリは空にならず、**再送もされない**
(送信側の重複判定は「ディレクトリが残っているか」で見ている)。

### 1-4. スリープ設定を確認

lid を開けて給電していれば基本は止まらないが、念のため。

```bash
pmset -g | grep -E "^ *sleep|disablesleep"
pmset -g log | grep -iE "Sleep|Wake" | tail -20
```

---

## 2. M2 (SSH なし / 共有フォルダ経由)

**9/8 16:05 を最後に 5 日以上エクスポートが止まっている。** まずそこから。

```bash
launchctl list | grep ldrp
#   何も出なければ未登録 → 下で登録する
tail -30 /tmp/ldrp_export.log
```

### 2-1. 未登録だった場合

```bash
cd ~/LDRP && git pull
cp tools/com.ldrp.export-runs.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.ldrp.export-runs.plist
launchctl start com.ldrp.export-runs
tail -f /tmp/ldrp_export.log
```

### 2-2. 登録済みだった場合

```bash
cd ~/LDRP && git pull
launchctl unload ~/Library/LaunchAgents/com.ldrp.export-runs.plist
launchctl load   ~/Library/LaunchAgents/com.ldrp.export-runs.plist
launchctl start  com.ldrp.export-runs
```

### 2-3. 止まっていた run の確認

9/8 の時点で 1 本が `stalled` (開始 12:55、3 時間進捗なし、heartbeat なし) だった。

```bash
ps aux | grep '[t]rain.py'
tail -30 ~/LDRP/src/epymarl/results/sacred/qmix/drp_env:drp_safe-7agent_map_aoba00-v2/1/cout.txt
```

- プロセスが無ければ落ちている。`cout.txt` の末尾にエラーがあるはず
- 回し直すなら、この run は計画外 (**QMIX + LaRe は計画にある**ので条件表には出る)

---

## 3. GPU2 (172.23.111.219:2222 / SSH)

**9/8 から ping も TCP も通らない。** 電源とネットワークを見る。

```bash
# 現地で本体を確認したあと、白から
ssh -o ConnectTimeout=10 linlabgpu1-linlab 'hostname; uptime; nvidia-smi --query-gpu=name --format=csv,noheader'
```

復旧したら白側は**何もしなくてよい** (設定済み)。`収集` ボタンで
**未回収の 54 モデルが自動で回収される**。

> ラベルは 9/13 に修正済み。**172.23.111.219 = GPU2 / 172.23.111.13 (cat) = GPU1**。
> 以前は逆になっていた。

---

## 4. 白に戻ってから

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
| machines の `モデル回収` | GPU2 が `54/54` になる |
| 条件表の 📦 | done の行に付く |

---

## 付録: 各マシンの役割

| | 接続 | 役割 |
|---|---|---|
| 白 | — | 集約。収集・保管・ダッシュボード。**ここだけが plan.md を持つ** |
| 黒 | 共有フォルダ | 学習 + `--export` するだけ |
| M2 | 共有フォルダ | 同上 |
| GPU1 (172.23.111.13, cat) | SSH | 学習。白が SSH で取りに行く |
| GPU2 (172.23.111.219:2222) | SSH | 同上 |

黒 / M2 では **`--notion` / `--publish-models` / `plan.md` に触らない**
(詳細は [SETUP_export_machine.md](SETUP_export_machine.md))。
