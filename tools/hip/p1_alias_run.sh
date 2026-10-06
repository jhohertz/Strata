#!/usr/bin/env bash
# P1 alias smoke with hang forensics. Runs the arithmetic prompt with the aliasing expert cache and,
# if the engine stops progressing, captures: the GPU busy %, the CPU-side backtraces (gdb), and the last
# trace lines - then kills.  The APU must be rebooted afterwards (kill with possibly in-flight GPU work).
#
#   tools/hip/p1_alias_run.sh [OUTDIR]
set -u
cd "$(dirname "$0")/../.."
OUT="${1:-/tmp/p1-out}"; mkdir -p "$OUT"
IDS=$(python3 -c "import json;print(json.load(open('/tmp/p0-out/smoke_ids.json'))['arithmetic'])")
echo "== $(date -Is) alias arithmetic (STRATA_TRACE=1)"
( env STRATA_IGPU_ALIAS=1 STRATA_TRACE=1 STRATA_HIPBLASLT_TUNING="$PWD/tools/hip/gfx1103-hipblaslt-100401.txt" \
    timeout --signal=KILL 900 "$PWD/build-hip/strata" --pack packs/qwen38-flash-next-q2_0 \
    --native /home/jhohertz/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
    --ple-gguf /home/jhohertz/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf \
    --mmap-experts --expert-profile data/expert-profile.bin --expert-cache 6000 --prefill 512 --spec 4 \
    --spec-min-p 0.5 --max-context 4096 --kv int8 --pool-workers 8 --adapt-every 0 --pcie-frac 0 \
    --vram-reserve-mib 1024 --greedy --max-new 160 --tokens "$IDS" \
    > "$OUT/arit.out" 2> "$OUT/arit.err" ) &
PID=$!
EPID=""   # the real engine: the subshell's child named strata (v1 attached to the timeout wrapper by mistake)
for _ in $(seq 1 120); do EPID=$(pgrep -x strata 2>/dev/null | head -1); [ -n "$EPID" ] && break; sleep 1; done
echo "engine pid=$EPID"
busy_file=/sys/class/drm/card1/device/gpu_busy_percent
last_lines=0; hung_since=0; diagnosed=0
while kill -0 "$PID" 2>/dev/null; do
    sleep 5
    live_lines=$(wc -l < "$OUT/arit.err" 2>/dev/null || echo 0)
    busy=$(cat "$busy_file" 2>/dev/null || echo '?')
    echo "  t=$(date +%H:%M:%S) alive err_lines=$live_lines gpu_busy=${busy}%"
    if [ "$live_lines" != "$last_lines" ]; then last_lines=$live_lines; hung_since=0; continue; fi
    if [ "$hung_since" = "0" ]; then hung_since=$(( $(date +%s) )); continue; fi
    quiet=$(( $(date +%s) - hung_since ))
    if [ "$quiet" -ge 45 ] && [ "$diagnosed" = "0" ]; then
        diagnosed=1
        echo "  HANG FORENSICS (quiet ${quiet}s, gpu_busy=${busy}%)"
        {
            echo "=== $(date -Is) gpu_busy_percent=$(cat $busy_file 2>/dev/null)"
            echo "=== process state (engine $EPID)"; ps -o pid,stat,wchan:40,time -p "$EPID" 2>/dev/null
            for t in $(ls /proc/"$EPID"/task 2>/dev/null); do
                echo "--- thread $t: $(tr -d '\0' < /proc/$EPID/task/$t/comm 2>/dev/null) state=$(cut -d' ' -f3 /proc/$EPID/task/$t/stat 2>/dev/null) wchan=$(cat /proc/$EPID/task/$t/wchan 2>/dev/null)"
            done
            echo "=== last trace lines"; tail -6 "$OUT/arit.err" | cut -c1-160
            echo "=== gdb backtraces (main thread deep, all threads shallow)"
            gdb -p "$EPID" -batch -ex 'set pagination off' \
                -ex 'thread 1 bt 24' \
                -ex 'thread apply all bt 5' 2>/dev/null | grep -E '^(Thread|#[0-9])' | head -120
        } > "$OUT/hang_forensics.txt" 2>&1
        echo "  captured -> $OUT/hang_forensics.txt (the 900 s timeout will kill the engine)"
    fi
done
wait "$PID"; RC=$?
echo "== $(date -Is) exit=$RC"
tail -4 "$OUT/arit.err" | cut -c1-140
