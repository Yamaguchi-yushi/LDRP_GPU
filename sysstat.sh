#!/usr/bin/env bash
# sysstat.sh — GPU / CPU / メモリ の現在状況を見やすく表示する.
#
# 使い方:
#   ./sysstat.sh          # 1 回表示
#   ./sysstat.sh -w       # 2 秒ごとに更新 (Ctrl-C で終了)
#   ./sysstat.sh -w 5     # 5 秒ごとに更新
#
# 備考: WSL2 では nvidia-smi が PATH に無いことがあるため,
#       /usr/lib/wsl/lib/nvidia-smi へ自動フォールバックする.

# --- nvidia-smi の場所を解決 ---
NVSMI="$(command -v nvidia-smi 2>/dev/null)"
[ -z "$NVSMI" ] && [ -x /usr/lib/wsl/lib/nvidia-smi ] && NVSMI=/usr/lib/wsl/lib/nvidia-smi

# --- 色 (端末出力のときだけ) ---
if [ -t 1 ]; then
    B=$'\033[1m'; DIM=$'\033[2m'; G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; C=$'\033[36m'; N=$'\033[0m'
else
    B=; DIM=; G=; Y=; R=; C=; N=
fi

# --- tmux ペイン tty → セッション名 のマップ ---
declare -A TMUX_MAP
build_tmux_map() {
    TMUX_MAP=()
    local ptty psess
    while read -r ptty psess; do
        [ -n "$ptty" ] && TMUX_MAP["$ptty"]="$psess"
    done < <(tmux list-panes -a -F '#{pane_tty} #{session_name}' 2>/dev/null)
}
# PID → tmux ラベル ("-" if tty 無し/tmux 外)
pid_tmux() {
    local tty
    tty=$(ps -o tty= -p "$1" 2>/dev/null)
    { [ -z "$tty" ] || [ "$tty" = "?" ]; } && { printf -- '-'; return; }
    printf '%s' "${TMUX_MAP[/dev/$tty]:--}"
}

# 使用率(%)に応じた色: <60 緑 / <90 黄 / それ以上 赤
color_for() {
    local v=${1%.*}
    [ -z "$v" ] && v=0
    if   [ "$v" -ge 90 ] 2>/dev/null; then printf '%s' "$R"
    elif [ "$v" -ge 60 ] 2>/dev/null; then printf '%s' "$Y"
    else printf '%s' "$G"; fi
}

# 横バー: bar <使用率%> [幅]
bar() {
    local pct=${1%.*}; local width=${2:-20} i
    [ -z "$pct" ] && pct=0
    [ "$pct" -lt 0 ]   2>/dev/null && pct=0
    [ "$pct" -gt 100 ] 2>/dev/null && pct=100
    local filled=$(( pct * width / 100 ))
    local s='['
    for ((i=0; i<filled; i++));        do s+='#'; done
    for ((i=filled; i<width; i++));     do s+='-'; done
    printf '%s]' "$s"
}

show() {
    build_tmux_map
    printf '%s\n' "${B}${C}===== システム状況 ($(date '+%Y-%m-%d %H:%M:%S')) =====${N}"

    # ---------- GPU ----------
    printf '%s\n' "${B}■ GPU${N}"
    if [ -n "$NVSMI" ]; then
        "$NVSMI" --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit \
                 --format=csv,noheader,nounits 2>/dev/null | \
        while IFS=, read -r idx name util mused mtot temp pdraw plim; do
            idx=$(echo "$idx"|xargs);  name=$(echo "$name"|xargs)
            util=$(echo "$util"|xargs); mused=$(echo "$mused"|xargs); mtot=$(echo "$mtot"|xargs)
            temp=$(echo "$temp"|xargs); pdraw=$(echo "$pdraw"|xargs); plim=$(echo "$plim"|xargs)
            [ -z "$mtot" ] || [ "$mtot" = "0" ] && mtot=1
            mempct=$(( mused * 100 / mtot ))
            uc=$(color_for "$util"); mc=$(color_for "$mempct")
            printf '  GPU%s  %s\n' "$idx" "$name"
            printf '    使用率 %s%s %3s%%%s  %s温度 %s°C / %s of %s W%s\n' \
                   "$uc" "$(bar "$util")" "$util" "$N" "$DIM" "$temp" "$pdraw" "$plim" "$N"
            printf '    VRAM   %s%s %3s%%%s  %s / %s MiB\n' \
                   "$mc" "$(bar "$mempct")" "$mempct" "$N" "$mused" "$mtot"
        done

        apps="$("$NVSMI" --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null)"
        if [ -n "$apps" ]; then
            printf '    %sプロセス別 VRAM:%s\n' "$DIM" "$N"
            echo "$apps" | while IFS=, read -r pid mem; do
                pid=$(echo "$pid"|xargs); mem=$(echo "$mem"|xargs)
                pname=$(ps -p "$pid" -o comm= 2>/dev/null)
                puser=$(ps -p "$pid" -o user= 2>/dev/null)
                printf '      PID %-7s %7s MiB  %s (%s)  %s[%s]%s\n' \
                       "$pid" "$mem" "${pname:-?}" "${puser:-?}" "$C" "$(pid_tmux "$pid")" "$N"
            done
        else
            printf '    %s(GPU 使用プロセスなし)%s\n' "$DIM" "$N"
        fi
    else
        printf '  %snvidia-smi が見つかりません (GPU 無し / ドライバ未導入)%s\n' "$R" "$N"
    fi

    # ---------- CPU ----------
    printf '%s\n' "${B}■ CPU${N}"
    cores=$(nproc)
    load=$(cut -d' ' -f1-3 /proc/loadavg)
    cpuuse=$(top -bn1 2>/dev/null | awk -F',' '/%Cpu|Cpu\(s\)/{for(i=1;i<=NF;i++) if($i ~ /id/){gsub(/[^0-9.]/,"",$i); printf "%.0f", 100-$i; exit}}')
    [ -z "$cpuuse" ] && cpuuse=0
    cc=$(color_for "$cpuuse")
    printf '    使用率 %s%s %3s%%%s  %sコア %s / load(1,5,15分) %s%s\n' \
           "$cc" "$(bar "$cpuuse")" "$cpuuse" "$N" "$DIM" "$cores" "$load" "$N"
    cpu_top=$(ps -eo pid,user:20,pcpu,rss,comm --sort=-%cpu 2>/dev/null | \
        awk 'NR>1 && $3+0 >= 1.0 {printf "%s %s %.1f %.0f %s\n", $1, $2, $3, $4/1024, $5}' | head -6)
    if [ -n "$cpu_top" ]; then
        printf '    %sプロセス別 CPU:%s\n' "$DIM" "$N"
        while read -r pid user cpu mem comm; do
            printf '      PID %-7s  %5s%%CPU  %6s MiB  %s (%s)  %s[%s]%s\n' \
                   "$pid" "$cpu" "$mem" "$comm" "$user" "$C" "$(pid_tmux "$pid")" "$N"
        done <<< "$cpu_top"
    fi

    # ---------- メモリ ----------
    printf '%s\n' "${B}■ メモリ${N}"
    read -r mtot mused mavail < <(free -m | awk '/^Mem:/{print $2, $3, $7}')
    [ -z "$mtot" ] || [ "$mtot" = "0" ] && mtot=1
    mempct=$(( mused * 100 / mtot ))
    mc=$(color_for "$mempct")
    printf '    使用   %s%s %3s%%%s  %s / %s MiB  (利用可能 %s MiB)\n' \
           "$mc" "$(bar "$mempct")" "$mempct" "$N" "$mused" "$mtot" "$mavail"
    read -r stot sused < <(free -m | awk '/^Swap:/{print $2, $3}')
    if [ "${stot:-0}" -gt 0 ] 2>/dev/null; then
        spct=$(( sused * 100 / stot ))
        printf '    Swap   %s%s %3s%%%s  %s / %s MiB\n' \
               "$(color_for "$spct")" "$(bar "$spct")" "$spct" "$N" "$sused" "$stot"
    fi
    mem_top=$(ps -eo pid,user:20,pcpu,rss,comm --sort=-rss 2>/dev/null | \
        awk 'NR>1 && $4+0 >= 102400 {printf "%s %s %.1f %.0f %s\n", $1, $2, $3, $4/1024, $5}' | head -6)
    if [ -n "$mem_top" ]; then
        printf '    %sプロセス別 RSS:%s\n' "$DIM" "$N"
        while read -r pid user cpu mem comm; do
            printf '      PID %-7s  %6s MiB  %5s%%CPU  %s (%s)  %s[%s]%s\n' \
                   "$pid" "$mem" "$cpu" "$comm" "$user" "$C" "$(pid_tmux "$pid")" "$N"
        done <<< "$mem_top"
    fi
}

# --- watch モード ---
if [ "${1:-}" = "-w" ] || [ "${1:-}" = "--watch" ]; then
    interval="${2:-2}"
    trap 'exit 0' INT
    while true; do clear; show; sleep "$interval"; done
else
    show
fi
