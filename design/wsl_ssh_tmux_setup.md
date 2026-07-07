# WSL2 での tmux 消失 / SSH 不通の原因と対策

linlab GPU マシン (WSL2 / Ubuntu-24.04, Windows ユーザ `Liala`) で発生していた
「tmux が勝手に終了する」「他ユーザログイン中に SSH できない」問題の調査結果と対策メモ。

---

## 症状と根本原因

WSL2 の distro (Ubuntu-24.04) は **Windows 側の何か (= VSCode Remote-WSL) が起動しないと立ち上がらず**、
distro が落ちると中の tmux も sshd も全部消える。3 つの症状は同じ根に繋がっている。

| 症状 | 原因 |
|---|---|
| SSH サーバが VSCode 起動で立つように見える | `ssh.service` は `disabled` だが `ssh.socket` が `enabled`。VSCode が **distro を起動** → ssh.socket が :22 待受 → 接続で ssh.service が **socket 起動**。VSCode が sshd を持つのではなく、distro 起動のトリガーが VSCode |
| 他 Windows ユーザログイン中に SSH 不可 | WSL2 は **Windows ユーザごとに別 VM**。`Liala` の distro が動いていない文脈では ssh.socket も無く不通 |
| tmux が勝手に終了 | VSCode を閉じる / Windows ユーザ切替 / スリープ / `wsl --shutdown` で **distro (VM) ごと落ちる**と中の tmux も消える。加えて tmux server が login セッションの cgroup (`session-N.scope`) に居るため、linger 無効だとセッション終了でも死ぬ |

**除外できた原因**: OOM (メモリ十分・kern.log に oom-kill 無し)、sshd の同時接続制限 (すべて既定)、PAM maxlogins (無効)。

---

## 対策

### Step 1: linger 有効化 (distro 内・低リスク) ✅ 適用済み

distro が動いている間は、VSCode / SSH を切断しても tmux・学習ジョブが生き続ける。

```bash
loginctl enable-linger linlab
loginctl show-user linlab | grep Linger   # -> Linger=yes
```

- **効果**: SSH/VSCode の切断・再接続では tmux が死ななくなる。
- **限界**: VM ごと落ちるケース (Windows スリープ / ユーザ切替 / `wsl --shutdown` / Windows 再起動) は救えない → Step 2。
- **デメリット**: ログアウト後も背景プロセスが常駐 (自分で管理が必要)。アイドルメモリが数十 MB 増える程度。

### Step 2: distro 常駐 (Windows 側・要判断)

「誰が Windows にログインしていても distro が生存 → SSH 常時可・tmux 生存」にする。
**他ユーザログイン中の SSH 不通もこれで解消**。

#### 1. keepalive タスク登録 (管理者の PowerShell / cmd)

```
schtasks /Create /TN "WSL-KeepAlive-Ubuntu2404" ^
 /TR "C:\Windows\System32\wsl.exe -d Ubuntu-24.04 -u root -e sleep infinity" ^
 /SC ONSTART /RU Liala /RP * /RL LIMITED /F
```

- `/SC ONSTART`: Windows 起動時に実行
- `/RU Liala /RP *`: Liala の資格情報で「ログオン有無に関係なく実行」→ 誰も Windows にログインしていなくても distro 生存
- `wsl ... sleep infinity`: distro を起動し 1 プロセス保持で VM が畳まれないよう固定

すぐ起動 (再起動を待たない場合):
```
schtasks /Run /TN "WSL-KeepAlive-Ubuntu2404"
```

#### 2. 失敗時の自動再起動 (推奨・GUI)

`taskschd.msc` → 該当タスク → プロパティ → 設定タブ →
「タスクが失敗した場合の再起動の間隔」= 1 分、再試行回数 = 3。

#### 3. メモリ肥大の抑制 (推奨)

`C:\Users\Liala\.wslconfig`:
```ini
[wsl2]
memory=24GB
autoMemoryReclaim=gradual
```
反映には一度 `wsl --shutdown` が必要 (タスクが再起動で復帰)。`memory` は学習要求に合わせ調整。

#### 4. セキュリティ (SSH 常時オープンになるため)

`/etc/ssh/sshd_config` で `PasswordAuthentication no` (公開鍵登録済みが前提) →
`sudo systemctl restart ssh`。

#### デメリット

- distro 常駐で **RAM を常に占有** (3 で緩和)
- **sshd 常時オープン** で攻撃面が増える (4 で緩和)
- 共有 PC では他 Windows ユーザの文脈からも distro が動き続ける

#### 動作確認

1. Windows を再起動、または `schtasks /Run`
2. Liala で Windows にログインせず (ロック状態で) 別マシンから SSH → 繋がれば成功
3. `wsl -l -v` で Ubuntu-24.04 が `Running` を維持

---

## 調査に使った確認コマンド (再調査用)

```bash
loginctl list-sessions                       # セッション状態 (closing の有無)
loginctl show-user linlab | grep Linger      # linger 状態
cat /proc/<tmux server pid>/cgroup           # tmux が session-N.scope 配下か
systemctl status ssh ssh.socket              # ssh.service=disabled / ssh.socket=enabled
grep -iE "oom|killed process" /var/log/kern.log   # OOM 除外
```
