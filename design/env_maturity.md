# 設計書: 環境ソフトウェアとしての成熟度ギャップ (CAMAR / RHCR / LoRR との比較)

**作成日:** 2026-06-14
**ステータス:** 調査・方針メモ (実装はまだ)
**対象リポジトリ:** [https://github.com/Yamaguchi-yushi/LDRP](https://github.com/Yamaguchi-yushi/LDRP) (本リポ)
**関連:**
- [future_work.md](future_work.md) (本ファイルはここから参照される重い設計書)
- [lare_integration.md](lare_integration.md) / [alma_integration.md](alma_integration.md) / [ldrp_extensions.md](ldrp_extensions.md)

> **目的:** LDRP を他の代表的 MARL 環境 (CAMAR, RHCR, LoRR) と比較し、「環境ソフトウェアとしての成熟度」の不足点を整理する。
> LDRP は **問題定式化の網羅性 (Lifelong / 衝突 / 割当 / 学習報酬 / グラフ) では強い**一方、
> **評価プロトコル・再現性インフラ・大規模 Lifelong での輻輳耐性**で後れを取る、という外部レビューを受けて作成。
>
> **検証ステータス:** 外部レビューは LDRP の `src/` を直接読めない前提で書かれていたため、本ファイルでは
> **各指摘を実コードで確認し、当てはまるものだけ**を採録した (確認結果は各節「LDRP 実態」に記載)。

---

## 0. コードで確認した LDRP 実態 (2026-06-14 時点)

| 観点 | 確認結果 | 根拠 |
|---|---|---|
| 並列/ベクトル化 | 無し。`test.py` / `run.py` から subprocess 逐次実行 (CPU 中心) | [run.py](../run.py) |
| メトリクス | `task_completion` (件数) と `collision` のみ。throughput(単位時間)/service time/makespan/IQM±CI 無し | [drp_env.py:857](../src/main/drp_env/drp_env.py#L857), [runner.py:201-213](../runner.py#L201) |
| マップ生成 | トポロジは **ファイルから固定読み込み**。`random_start/random_goal` で開始・ゴール位置のみランダム化。グラフの手続き的生成は無し | [EE_map.py:34-83](../src/main/drp_env/EE_map.py#L34) |
| テスト/CI | `.github/workflows` 無し。`drpload_test.py` (ロード確認) のみ。ユニット/回帰テスト無し | リポジトリルート |
| 異質性 | `self.speed` は全 agent 共通スカラ = 均質前提 | [drp_env.py:80,96](../src/main/drp_env/drp_env.py#L80) |
| チェックポイント | `factor_dim` は保存+ロード検証済み。正規化方式/encoder version は未保存 | [lare_path_module.py:264-284](../src/lare/path/lare_path_module.py#L264) |

→ レビュー指摘はおおむね妥当。以下、採録する項目を優先度付きで整理する。

---

## 1. 【高優先度】自動テスト + CI + baseline 不変条件の回帰テスト

### レビュー指摘

CAMAR は `tests/` + `.github/workflows` (CI) を持つ。LDRP は `drpload_test.py` のみでユニット/回帰テスト・CI が無い。

### LDRP 実態と判断 — **採用 (最優先)**

事実。特に [CLAUDE.md](../CLAUDE.md) の最重要不変条件「`use_lare_path=False` かつ `use_lare_task=False` のとき LaRe 統合前と完全一致」を**機械的に検証する手段が手動スニペットしかない**。snapshot 改修・ALMA フック追加・正規化変更のたびに baseline 破壊を検出できる回帰テストが要る。

### 具体案

1. **baseline 回帰テスト**: 固定 seed で `use_lare_*=False` の env を回し、`(obs, reward, done)` 列のハッシュをゴールデン値と比較。upstream (LaRe 統合前) との一致確認にも使える。
2. **smoke テスト**: CLAUDE.md の検証スニペット (両モジュール `is_trained` が True になる) を pytest 化。
3. **CI**: GitHub Actions で上記 + `python -m py_compile` を PR ごとに実行。
   - 開発環境固定 (Python 3.9 / gym 0.26.2 / numpy 1.26.4) なので、CI もこのバージョンに固定する。

---

## 2. 【高優先度】評価プロトコルの明文化 + throughput 系メトリクス + 統計報告

### レビュー指摘

CAMAR は Easy/Medium/Hard の 3 段階プロトコル、IQM + 95% CI、固定 seed、数千評価エピソードで再現性を担保。Lifelong の標準指標は makespan ではなく **throughput / service time**。

### LDRP 実態と判断 — **採用 (高)**

LDRP は throughput / service time を出していない (`task_completion` 件数のみ)。論文の主張 (DBCT の agent-count 横断の再利用性 = 汎化) は**汎化評価軸と直結**するので、ここを固めるとレビュー耐性が大きく上がる。

### 具体案

1. **メトリクス追加** (env の `info` に足すだけで実装は軽い):
   - **throughput** = `task_completion / time_limit` (単位時間あたり完了数)
   - **service time** = タスクごとの (生成 step → ドロップ step)。env は既に `_lare_task_creation_steps` を持つので算出可能。
2. **汎化階層の明文化**: 「学習時と異なる agent 数 / マップ」での評価を Easy/Medium/Hard 相当に固定。DBCT の主張に直結。
3. **統計報告の標準化**: 平均±標準偏差でなく **IQM ± CI95**、seed 数の明示、モデル選択基準 (最終 vs ベスト) の固定。
4. **評価ハーネスの分離**: `test.py` / `run.py` は実行スクリプト。固定プロトコルの評価は別ハーネスに切り出すと再現性が上がる。
5. **原則の明記**: 「proxy 報酬は学習専用、評価指標は常に環境報酬・タスク完了数 (throughput)」を文書化。

---

## 3. 【高優先度】大規模時の輻輳 (congestion) 対策 — PBS の限界

### レビュー指摘

RHCR は Rolling-Horizon Collision Resolution で最大 1,000 体規模を解く。LoRR では congestion / 計画時間制約 / 簡略モデルと現実のギャップが中心課題。LDRP の PBS には輻輳緩和が無い。

### LDRP 実態と判断 — **採用 (高、ただし研究課題)**

README 自身が「PBS は順番に動かすため他 agent が道を封鎖し、衝突しない経路を発見できないことがある」と弱点を認めている。「Lifelong ✓」を主張する以上、大規模で顕在化するこの点は直接比較される。ただし対策は実装が重く、研究課題。

### 具体案 (重い→軽い)

- **スケール検証**: まず「どの agent 数まで衝突解決込みで回るか」の上限を実測・提示する (これは軽く、最初にやる価値が高い)。
- **rolling-horizon 化**: 有限ホライズン内のみ衝突解決する RHCR 流の分解。
- **guidance graph / agent 無効化**: LoRR 上位解の輻輳緩和を割当・経路に取り込む。

---

## 4. 【中優先度】並列ロールアウト + APSP キャッシュ

### レビュー指摘

CAMAR は JAX/GPU で 100K+ SPS、`jax.vmap` で 1000 並列環境、AutoReset ラッパ。LDRP は逐次 CPU。

### LDRP 実態と判断 — **部分採用 (中)**

LDRP は **離散グラフ + 探索 (PBS) が core** なのでフル GPU/JAX 化は設計が変わる (→ §8 で不採用推奨)。現実的な落としどころは:

1. **Python レベルの並列ロールアウト** (`SubprocVecEnv` 相当)。Lifelong は 1 エピソードが長くサンプル効率が課題なので効く。
2. **APSP 事前計算でテーブル参照化** — **既に [future_work.md 項目1](future_work.md#1-gpu-環境への移行) に記載済み**。本ファイルの文脈 (大規模化のボトルネック) でも再掲。
3. **AutoReset ラッパ**: Lifelong/継続型でエピソード境界の自動処理が学習ループを単純化。

---

## 5. 【中優先度】手続き的マップ生成 + 汎化階層

### レビュー指摘

CAMAR は random_grid / labmaze / movingai 等の多様なマップ + `register_map` レジストリ。LDRP は固定マップ中心。

### LDRP 実態と判断 — **採用 (中)**

トポロジは固定ファイル読み込み、start/goal のみランダム化。汎化評価には**グラフ構造のパラメトリック生成** (ノード配置・エッジ密度・障害物) があると強い。§2 の汎化階層と一体。

### 具体案

- ランダムグラフ生成器 (ノード数・エッジ密度・障害物率をパラメータ化)。
- マップ/タスク生成の**レジストリ化** (文字列名で差し替え)。
- (任意) 実配送地点・道路網グラフの取り込み (CAMAR の MovingAI 相当)。

---

## 6. 【中優先度】チェックポイント metadata / versioning + 依存固定

### レビュー指摘

CAMAR は Dockerfile / devcontainer / lockfile / pyproject。チェックポイントに encoder version・正規化方式を持たせるべき。

### LDRP 実態と判断 — **採用 (中、一部既存)**

- **checkpoint metadata**: `factor_dim` は既に保存+検証済み。だが**正規化方式を変えても factor_dim は 10 のままなので検証をすり抜ける** ([future_work.md 項目2 の正規化変更](future_work.md#2-lare-path-因子の正規化-3-因子)時に `factor_version` を同梱すべき)。Lifelong + DBCT で複数設定のモデルを扱うので versioning は要る。
- **依存固定**: `requirements.txt` のみ → lockfile / `pyproject.toml` 化で環境差を抑える。
- **コンテナ化**: GPU サーバ間の再現性確保に直結 (Dockerfile / devcontainer)。

---

## 7. 【低優先度】異種エージェント + 動力学 / 移動コスト抽象化

### レビュー指摘

CAMAR は Holonomic/DiffDrive/Mixed の動力学、agent ごとの半径。

### LDRP 実態と判断 — **据え置き (低、現フェーズ外)**

`self.speed` は全 agent 共通スカラ = 均質前提。速度/容量/優先度の異なる機体混在は ALMA のタスク割当の意味を増すが、**「Lifelong × 割当の検証」という現フェーズではスコープ外**で妥当。エッジコスト/移動コストの差し替え点 (機体差・積載差) を抽象化レイヤとして用意すると将来拡張しやすい、程度に留める。

---

## 8. 標準インターフェース / ベースライン、および「追わない」もの

### 採用 (中-低)

- **PettingZoo/Gymnasium 準拠の薄いラッパ**: 比較実験を同一 API で回しやすくする。
- **古典手法のベースライン化**: PBS 以外の探索・ヒューリスティック割当を同一 API でベースラインとして回せる仕組み (CAMAR は RRT/RRT* を統合)。

### 不採用 (無理に追わない)

- **フル GPU / JAX 化**: LDRP は離散グラフ + 探索 (PBS) が core で、CAMAR の連続空間前提とは設計が別。費用対効果が低い。§4 の「Python 並列 + APSP」で代替する。

---

## 9. 優先度まとめ

| 優先度 | 項目 | 比較対象 | 理由 | 既存項目との関係 |
|---|---|---|---|---|
| **高** | 自動テスト + CI + baseline 回帰テスト (§1) | CAMAR | 実装フェーズの土台。不変条件の機械的検証 | 新規 |
| **高** | 評価プロトコル + throughput/service time + IQM±CI95 (§2) | CAMAR / RHCR | 論文の主張 (DBCT 汎化) と直結 | 新規 |
| **高** | 大規模時の輻輳対策 / スケール検証 (§3) | RHCR / LoRR | Lifelong✓ の信頼性。README 自認の弱点 | 新規 (研究課題) |
| 中 | 並列ロールアウト + APSP キャッシュ (§4) | CAMAR | サンプル効率・実験速度 | APSP は future_work 項目1 |
| 中 | 手続き的マップ生成 + 汎化階層 (§5) | CAMAR | 汎化評価の前提 (§2 と一体) | 新規 |
| 中 | checkpoint metadata/versioning + 依存固定 (§6) | CAMAR | 再現性。一部既存 (factor_dim) | future_work 項目2 と連動 |
| 低 | 異種エージェント・動力学 (§7) | CAMAR | 現フェーズ外でも可 | 新規 |
| 低 | フル GPU/JAX 化 (§8) | CAMAR | 離散+探索 core と設計が別。**不採用** | — |

**論文化を見据えた着手順の推奨**: §1 (回帰テスト) → §2 (評価プロトコル + throughput) → §3 のスケール検証、の 3 つを先に固めると、主張の説得力とレビュー耐性の両面で効く。

---

## 参考 (比較対象)

- **CAMAR**: 連続空間・JAX/GPU の MARL ナビ環境。Table 1 が事実上の評価軸 (連続観測/行動・GPU加速・500+ 体・部分観測・異種・10K+ SPS・手続き生成・評価プロトコル・CI 等)。
- **RHCR** (Li et al.): Rolling-Horizon Collision Resolution。Lifelong MAPF を Windowed MAPF 列に分解、最大 1,000 体規模。
- **LoRR**: League of Robot Runners 競技。計画時間制約・congestion・簡略モデルと現実のギャップが中心課題。

> 出典は外部レビュー (CAMAR コード + RHCR/LoRR 論文・リポジトリ) に基づく。数値・列挙は二次情報を含むため、論文化時は一次資料で再確認すること。

---

最終更新: 2026-06-14
