<!--
  実験計画の表 (研究室配布版の見本)。

    mkdir -p tools/plans && cp tools/plan.example.md tools/plans/plan_main.md

  ファイル名の plan_ の後ろ (この例なら main) が計画の名前になり、
  回収したモデルの保管先フォルダ名にも使われる。tools/plans/ は .gitignore 済み。

  書き方:
    - マップ名を単独行で置くと、以降の章の既定マップになる (例: "8x5" / "aoba00")
    - 章見出し   ## {N}agent  {t_max}M       (例: ## 5agent  20M)
    - 条件行     seed と machine が空の行。setting / algorithm などを書く
    - seed 行    条件行の下に続ける。seed が空欄の行は「未実行の枠」になる
                 (1 条件 5 行 = 5 seed。seed は決めていなければ空欄のままでよい)
    - setting    LaRe を使わないなら safe
    - algorithm  QMIX / IQL / VDN / MAPPO など (大文字小文字は問わない)
    - task arrival / task assign は、このリポジトリでは空欄のままでよい
      (空欄の列は「問わない」扱い)

  seed を書いておく必要はない。train.py は起動時に seed をランダムに決めるので、
  seed 欄は空けておき、回った run が空いている枠に入るのに任せるのが楽。
  実行が完了していない seed は、書いてあっても消してよい。
-->

8x5

## 3agent   10M

| seed | machine | setting | algorithm | task arrival | task assign |
| --- | --- | --- | --- | --- | --- |
|  |  | safe | QMIX |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  | safe | IQL |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
