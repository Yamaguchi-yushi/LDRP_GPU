# 設計書: PPO タスク割当方策の save / load (永続化)

**作成日:** 2026-05-19
**ステータス:** 設計案 (実装はまだ)
**関連:** [lare_integration.md](lare_integration.md) (= 同パターンで整備)

---

## 1. 背景 / 現状の問題

LDRP の **PPO タスク割当器 (`task_assigner: ppo`)** は学習可能だが、学習結果をテスト実行に引き継げない。

[src/task_assign/task_policy/ppo.py](../src/task_assign/task_policy/ppo.py):

```python
class PPO(nn.Module):
    def save_model(self, path):
        pass                      # ← stub: 何もしない

    def load_model(self, path):
        pass                      # ← stub: 何もしない
```

つまり:

- 学習自体は `runner.run()` の `if self.training:` ブロックで進行
- ただし学習終了後に **`task_assigner.save_model(...)` は実装が空** なのでファイル出力なし
- test.py は `training=False` で起動 → PPO は毎回ランダム初期化された状態で推論
- → **「PPO を訓練 → テストで使う」 ワークフローが繋がっていない**

他コンポーネントとの対比:

| コンポーネント | save | load | 永続化先 | テスト時利用 |
|---|---|---|---|---|
| Path planner (QMIX/IQL/...) | ✓ (epymarl) | ✓ | `src/all_policy/models/safe/{map}_{N}_{algo}.th` | **可能** |
| LaRe-Path decoder | ✓ | ✓ | `src/lare/path/{checkpoints,models}/` | **可能** |
| LaRe-Task decoder | ✓ | ✓ | `src/lare/task/{checkpoints,models}/` | ロード自体は可能 (※) |
| **PPO task assigner** | **✗ stub** | **✗ stub** | なし | **不可能** |

(※) LaRe-Task の proxy 報酬は `if self.training:` 内でしか PPO に流れないので、テスト時の利用は学習補助としては機能しない。あくまでロード自体は可能。

## 2. 設計方針

LaRe-Path / LaRe-Task の整備パターン (= 2026-05-19 確定) を **そのまま PPO に適用**。対称性を保ち、メンテナンスコストを下げる。

採用要素:

| 要素 | 内容 |
|---|---|
| **モード** | 4 モード対称 (off / scratch / pretrained / finetuning) |
| **ディレクトリ分離** | `checkpoints/` (autosave 出力) と `models/` (整理済み load 元) |
| **autosave throttle** | 累積環境ステップ単位 (0.5M ごと等), 学習頻度とは独立 |
| **ファイル命名** | Safe-TSL-DBCT 流儀 (`{Safe_}<TOKEN>_<...>_X.XM_checkpoint.pth`) |
| **後方互換** | 旧パスがあれば load 解決パスに含める |

## 3. ディレクトリ構成 (提案)

```
src/task_assign/
├── checkpoints/      ← PPO autosave 出力 (大量蓄積, .gitignore)
├── models/           ← pretrained / finetuning ロード元 (整理済み, git 公開)
└── task_policy/      ← 既存
    └── ppo.py        ← save/load 実装を埋める
```

`.gitkeep` を `models/` に配置して空 dir を git に乗せる。

## 4. ファイル命名

LaRe との衝突を避ける `PPO` トークンを採用:

- Scratch: `{Safe_}{ALGO}_PPO_{map}_{N}agents_{X.X}M_checkpoint.pth`
- Finetuning: `FT_{Safe_}{source_base}_{map}_{N}agents_{X.X}M_checkpoint.pth`

`{ALGO}` は **path planner 名** (qmix/iql 等) を入れる. PPO 単独で使うこともあるが、現運用では path planner と組合せて学習・評価することが多いので、組合せでファイルを識別できる方がレトリーバビリティが高い。

例: `Safe_QMIX_PPO_map_8x5_4agents_2.0M_checkpoint.pth` = SafeEnv + QMIX path planner + PPO task assigner で 2.0M ステップまで学習した PPO モデル。

## 5. 設定キー (default.yaml 追加案)

```yaml
# --- PPO task assigner の永続化 (Path planner / LaRe と独立) ---
use_pretrained_ppo: false
pretrained_ppo_model_path: null      # ファイル名だけ指定で src/task_assign/models/ から自動解決
use_finetuning_ppo: false
finetuning_ppo_model_path: null

# 自動保存
ppo_autosave: false
ppo_autosave_path: null              # null なら自動命名 (上記の命名規則)
ppo_save_freq_steps: 500000          # 0.5M step ごとに保存 (LaRe と統一)
ppo_save_dir: null                   # deprecated 用予約. 通常は null
```

## 6. 実装フェーズ

| Phase | 内容 | コスト目安 |
|---|---|---|
| **1** | `PPO.save_model` / `PPO.load_model` の中身を実装 (`torch.save` / `torch.load`) | 15 行 |
| **2** | runner.py で起動時に pretrained / finetuning モードに応じて load を呼ぶ | 20 行 |
| **3** | runner.py で学習中に throttle 付き autosave | 20 行 |
| **4** | ディレクトリ構成と `.gitignore` 整備 (`checkpoints/` 除外, `models/` 許可) | 5 行 |
| **5** | default.yaml + test.py に新キー転送 | 10 行 |
| **6** | CLAUDE.md と MANUAL.md に記載 | 適宜 |

合計 ~80 行 (= LaRe 統合より軽い. 1 モデル + decoder/encoder 構造不要のため)。

## 7. 推奨ワークフロー (実装後)

```bash
# 1. 学習 (runner を training=True で起動 + ppo_autosave: true)
#    → src/task_assign/checkpoints/Safe_QMIX_PPO_..._X.XM_checkpoint.pth に蓄積

# 2. 良いモデルを選別
cp src/task_assign/checkpoints/Safe_QMIX_PPO_..._2.0M_checkpoint.pth \
   src/task_assign/models/ppo_best.pth

# 3. テスト (test.py で利用)
#    default.yaml:
#      task_assigner: "ppo"
#      use_pretrained_ppo: true
#      pretrained_ppo_model_path: "ppo_best"
python test.py
```

## 8. リスク・未決事項

| 項目 | 内容 | 対策 |
|---|---|---|
| `state_dict` の互換性 | PPO ネットワーク構造を変更すると古いモデルがロード不能 | チェックポイントに `architecture_version` を埋める / load 時にミスマッチ警告 |
| optimizer state の保存 | finetuning では optimizer state も保存・復元したい | `torch.save({"model": ..., "optimizer": ..., "step": ...})` 形式に統一 |
| autosave のディスク負荷 | 学習が長引くと多数のファイル蓄積 | save_freq_steps で間引き + 古いファイル削除ポリシーは別途検討 |
| 命名規則の衝突 | LaRe-Task の `TASK` トークンと PPO の `PPO` トークンは別だが、長期的に混乱しないか | `PPO` を `PPOPOLICY` 等にする選択肢もある (要検討) |
| LaRe-Task 統合との連携 | LaRe-Task が trained で proxy 報酬を流す → PPO もそれで学習 → 両者をセットで save/load したいか | 個別保存で問題なし (= 独立性維持). セット運用が必要になったら別途設計 |

## 9. 実装トリガ

以下のいずれかが発生したら実装を開始する:

- 「PPO を訓練 → テストで使いたい」という具体的なユースケースが発生
- PPO の性能を他手法 (TP, FIFO) と公平に比較したい (= 訓練済み PPO で比較しないと意味が薄い)
- LaRe-Task の効果を測定したい (= LaRe-Task ON/OFF で訓練した PPO を pretrained 比較)

それまでは本設計書を **生きた草稿** として保持。実装時に微調整は許容。

## 10. まとめ

PPO の save/load 未実装は、現状の LDRP で **訓練済み方策をテストに引き継げない最大の穴**。LaRe 整備で確立した「checkpoints/ + models/ + autosave + 4 モード + 0.5M throttle」のパターンをそのまま適用すれば、~80 行の追加で対応可能。実装トリガが来たタイミングで Phase 1-6 を一気に進める想定。

---

最終更新: 2026-05-19
