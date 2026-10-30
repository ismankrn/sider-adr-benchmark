#!/usr/bin/env bash
# Tarik hasil eksperimen dari server ke folder code/ lokal.
#   scripts/pull_results.sh            -> sinkronkan results/, figures/, logs/, data/processed/
#   scripts/pull_results.sh status     -> hanya tampilkan progres di server
#   scripts/pull_results.sh summary    -> tarik hasil lalu jalankan 07_summarize.py di lokal
# Override server dengan: SERVER=root@1.2.3.4 scripts/pull_results.sh
set -euo pipefail
SERVER="${SERVER:-sider}"   # ssh alias or user@host of the compute server
case "$SERVER" in vast*) REMOTE_DEFAULT=/workspace/code ;; *) REMOTE_DEFAULT="~/code" ;; esac
REMOTE="${REMOTE:-$REMOTE_DEFAULT}"
LOCAL="$(cd "$(dirname "$0")/.." && pwd)"

status() {
  ssh -n "$SERVER" "cd $REMOTE; echo '--- proses aktif'; ps -o pid,pcpu,etime,args -C python 2>/dev/null | grep scripts/ || echo '(tidak ada)';
    echo '--- baris hasil per CSV'; for f in results/*.csv; do [ -f \$f ] && printf '%-32s %6d baris\n' \$f \$(( \$(wc -l < \$f) - 1 )); done;
    echo '--- log terakhir'; for f in logs/0*_*.log; do echo \"== \$f\"; grep -v -i -E 'warn|conv1d' \$f | tail -2; done; uptime"
}

pull() {
  for d in results figures logs; do
    mkdir -p "$LOCAL/$d"
    n=$(rsync -az --stats "$SERVER:$REMOTE/$d/" "$LOCAL/$d/" 2>/dev/null | grep -i -E "files transferred" | grep -o -E "[0-9]+" | head -1 || true)
    printf '%-14s %s file diperbarui\n' "$d/" "${n:-0 (belum ada di server)}"
  done
  mkdir -p "$LOCAL/data/processed"
  rsync -az "$SERVER:$REMOTE/data/processed/" "$LOCAL/data/processed/" 2>/dev/null || true
  echo "selesai -> $LOCAL/results, figures, logs, data/processed"
}

case "${1:-pull}" in
  status)  status ;;
  pull)    pull ;;
  summary) pull; (cd "$LOCAL" && .venv/bin/python scripts/07_summarize.py | head -40) ;;
  *) echo "pakai: $0 [pull|status|summary]"; exit 1 ;;
esac
