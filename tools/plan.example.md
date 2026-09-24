<!--
  実験計画。Notion の表をそのまま貼り付けてよい。
  tools/plan.py が読み、ダッシュボードの既定表示 (計画にある条件だけ) に使う。

  書き方:
    - マップ名を単独行で置くと、以降の章の既定マップになる ("
<!--
  以下は書き方の見本。実物 (tools/plan.md) は .gitignore してある。
  実験計画は集約するマシン (白) だけが持つ。他のマシンは --export でデータを
  送るだけで、計画には書き込まない。他マシンで回した seed は収集時に
  「計画外 seed」として検出されるので、それを見てここに追記する。
-->

aoba00

## 5agent   80M

| seed | machine | setting | algorithm | task arrival | task assign | reassign |
| --- | --- | --- | --- | --- | --- | --- |
|  |  | safe | QMIX | bernoulli, mmpp | TP |  |
| 123456789 | GPU1 |  |  |  |  |  |
| 234567890 | GPU1 |  |  |  |  |  |
|  |  |  |  |  |  |  |
|  |  | 8x5_2 10M→aoba00_2 5M | MAPPO | bernoulli, mmpp | PPO |  |
| 345678901 | 黒 |  |  |  |  | F |
|  |  |  |  |  |  |  |

8x5

## 7agent   30M

| seed | machine | setting | algorithm | task arrival | task assign |
| --- | --- | --- | --- | --- | --- |
|  |  | safe | MAT (fixed) | bernoulli, mmpp |  |
| 456789012 | Ubuntu |  |  |  |  |
|  |  |  |  |  |  |
