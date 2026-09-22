# make_graph 側への依頼文 (LDRP との連携)

**使い方:** 以下の「依頼文」を `~/make_graph` で作業する Claude Code にそのまま渡す。
LDRP 側の現状と、実機で確認した事実を含めてある。

---

## 依頼文 (ここから)

あなたは `~/make_graph` (学習曲線を描く Flask アプリ) を改修します。
学習側のリポジトリ `~/LDRP` と連携させたいので、以下を読んで設計・実装してください。

### いまの使い方と、困っていること

1. TensorBoard から指標ごとに CSV を手でダウンロード
2. 手法ごとのフォルダに手で振り分け
3. make_graph の GUI に読み込ませて、色・順番・凡例を設定して作図

条件 (マップ × エージェント数 × アルゴリズム × 報酬設計 × タスク割当) が
**84 条件 × 5 seed = 420 run** あるので、1・2 が現実的でなくなりました。

### やりたいこと

**A. 複数条件のフォルダをまとめて渡したら、全条件の学習曲線を一括出力したい**

いまは 1 条件分のフォルダを読ませて 1 セットの図を作る運用ですが、
**条件フォルダを複数含む親フォルダ**を渡したら、条件ごとに図を出してほしい。

```text
curves/                          ← これを渡す
├── map_8x5-v2_5agent/           ← 1 条件 = 1 図のセット
│   ├── _meta.json
│   ├── QMIX/
│   │   └── run-{token}/*.csv
│   └── MAT/
├── map_8x5-v2_7agent/
└── map_aoba00-v2_10agent/
```

**B. 指標の選択を make_graph 側から LDRP 側へ渡したい**

いま LDRP 側は「どの指標を CSV 化するか」を設定ファイルで持っています。
これを **GUI で選んで LDRP に渡す**形にしたい。理由は、
指標が 36 種あり全部出すと 420 run で 1.5 GB になるためです
(1 指標 1 run あたり 27〜170 KB。記録頻度で決まる)。

**C. 色と順番は make_graph 側で決め、それを LDRP に渡して `_meta.json` を作らせたい**

いま `_meta.json` は make_graph が**書き出す**ものですが、
LDRP がフォルダを作る時点で色・順番が入っていれば、
**初回の読み込みから正しい見た目**になります。

### 理想の流れ

```text
make_graph GUI で指定
  ├─ 描きたい指標      (例: test_return_mean, test_task_completion_mean)
  ├─ 手法の順番と色    (例: QMIX=#03AF7A(0), MAPPO=#005AFF(1), MAT=#FF0000(2))
  └─ 条件の絞り込み    (任意)
        ↓  (何らかの形で LDRP へ渡す)
LDRP 側
  ├─ 各マシンの tensorboard event から指定された指標だけ CSV 化
  ├─ 実験計画の表と同じ構成でフォルダを作る
  └─ 渡された色・順番で _meta.json を書く
        ↓
make_graph に親フォルダを渡す → 全条件の画像が一発で出る
```

### LDRP 側で既に確認済みの事実 (実機検証済み)

- **TensorBoard の «Download CSV» は不要。** `tensorboard.backend.event_processing.event_accumulator`
  で event ファイルを直接読め、`Wall time,Step,Value` の CSV を生成できる。
  実際に生成して make_graph で作図まで通した
- 記録されている scalar は 1 run あたり 36 種。`test_` 付きが評価時の値
- epymarl の `unique_token` には `:` が含まれるので `_` に置換が必要
- LDRP 側は条件を `map / agents / t_max / setting / algo / task_assign / dynamic` の
  軸で持っていて、実験計画の表 (Markdown) と突き合わせている

### 相談したいこと

1. **A の一括出力**: 現状のコード (`app.py` / `make_multi_panel_plots.py`) で
   どこを変えるのが素直ですか。GUI とCLI のどちらを主にすべきですか
2. **B・C の受け渡し方法**: どの形が良いですか
   - (a) make_graph が JSON を書き出し、LDRP がそれを読む (ファイル経由・疎結合)
   - (b) make_graph が LDRP のスクリプトを直接呼ぶ
   - (c) LDRP が HTTP で make_graph の API を叩く
   - (d) その他
3. **`_meta.json` の仕様**: LDRP 側が書く場合、いまの
   `{"version":1,"methods":[{"name","color","order"}]}` 以外に必要なフィールドはありますか
4. 条件フォルダ名や手法フォルダ名に**制約**はありますか
   (`extract_map_name` / `extract_agent_count` / `detect_metric` の正規表現に合う必要がある?)

**まず現状のコードを調査して、1〜4 に対する方針を提示してください。**
実装はその後で構いません。

## 依頼文 (ここまで)

---

## LDRP 側で対応が必要になる想定

| 受け渡し方式 | LDRP 側の実装 |
|---|---|
| (a) JSON ファイル経由 | `--export-curves --spec <json>` で仕様を読む。**疎結合で一番安全** |
| (b) 直接呼び出し | make_graph が `collect_runs.py` を subprocess で起動 |
| (c) HTTP | LDRP のダッシュボードに API を足す |

**(a) を推す。** LDRP 側は「指標リスト・手法の色と順番」を受け取るだけでよく、
make_graph を起動していなくても動く。`collect_config.yaml` に既定値を置き、
JSON が渡されたらそちらで上書きする形にできる。

想定する JSON:

```json
{
  "metrics": ["test_return_mean", "test_task_completion_mean", "test_collision_mean"],
  "methods": [
    {"name": "QMIX",  "color": "#03AF7A", "order": 0,
     "match": {"algo": "qmix",  "setting": "safe", "task_assign": ""}},
    {"name": "MAT+LaRe", "color": "#FF0000", "order": 1,
     "match": {"algo": "mat", "setting": "dbct", "task_assign": "PPO"}}
  ],
  "filter": {"maps": ["map_8x5", "map_aoba00"], "agents": [5, 7, 10]}
}
```

`match` の軸は LDRP の実験計画表と同じなので、`plan.matches()` を流用できる。
