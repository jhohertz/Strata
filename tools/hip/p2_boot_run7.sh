#!/usr/bin/env bash
# igpu-rework P2.7: the STRATA_MMQ_GROUP A/B after a FULL power cycle (hold the power button
# or unplug the PSU - a plain reboot does NOT clear the fault cluster, P2.8) and a long idle.
#
# Hypothesis under test (docs/IGPU.md P2.7): the in-engine per-launch cost (~0.5-3 ms, 3-10x
# the micro's ~50 us) sets the wall, not the kernels - every expert path lands at ~39 tok/s.
# Bigger MMQ groups = fewer product launches (18/section at 16 -> 5/section at 64).
#
# Budget (P2.8): a post-power-cycle boot survives ~4-5 heavy engine runs before the degraded
# slow phase (6.7 s/section -> MES hang) appears.  This script is gate + exactly 3 heavy runs.
# If any run comes in under 20 tok/s, the machine is degrading: stop, do NOT reboot, power
# cycle.  The 16 baseline is known from the same-boot record (32.5 s wall), so 64 and 32
# against that number are still meaningful if the third run is sacrificed.
set -u
cd "$(cd "$(dirname "$0")/../.." && pwd)"
O=/tmp/p2-out
mkdir -p "$O"
export STRATA_IGPU_ALIAS=1 STRATA_GROUP_COPY=1 STRATA_HIPBLASLT_WARMUP=1
export STRATA_HIPBLASLT_TUNING="$PWD/tools/hip/gfx1103-hipblaslt-100401.txt"
export STRATA_HOSTLOOP=1 STRATA_PREFILL_TIMING=1 STRATA_SSD_KEEPALIVE=0

step() { echo "=== P2.7 [$1] $(date '+%H:%M:%S') ===" | tee -a "$O/run.log"; }

need_ids() {
    if [ ! -f "$O/$1" ]; then
        [ -f "$O/smoke_ids.json" ] || python3 tools/hip/gfx1103_smoke.py prep packs/qwen38-flash-next-q2_0 "$O" >&2
        key="${1%.ids}"
        python3 - "$O" "$key" "$1" <<'EOF'
import json, sys
o, key, fname = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(o + "/smoke_ids.json"))[key]
ids = d if isinstance(d, str) else " ".join(map(str, d))
open(o + "/" + fname, "w").write(ids)
EOF
    fi
    cat "$O/$1"
}

# one longfill with thermals; $1 = mmq group, $2 = tag
run_arm() {
    local g="$1" tag="$2"
    step "longfill MMQ_GROUP=$g"
    : > "$O/therm-$tag.log"
    (
        while true; do
            printf '%s ' "$(date '+%H:%M:%S')" >> "$O/therm-$tag.log"
            rocm-smi --showtemp --showpower 2>/dev/null | tr '\n' ' ' | sed 's/  */ /g' >> "$O/therm-$tag.log"
            echo >> "$O/therm-$tag.log"
            sleep 1
        done
    ) & local THERM=$!
    env STRATA_MMQ_GROUP="$g" timeout 900 ./build-hip/strata --pack packs/qwen38-flash-next-q2_0 \
      --native /home/jhohertz/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
      --ple-gguf /home/jhohertz/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf \
      --mmap-experts --expert-profile data/expert-profile.bin --expert-cache 6000 --prefill 512 --spec 4 \
      --spec-min-p 0.5 --max-context 4096 --kv int8 --pool-workers 8 --adapt-every 0 --pcie-frac 0 \
      --vram-reserve-mib 1024 --greedy --max-new 160 --tokens "$IDS" > "$O/lf-mmq$g.out" 2> "$O/lf-mmq$g.err"
    echo "exit=$?" | tee -a "$O/run.log"
    kill "$THERM" 2>/dev/null
    grep -E 'hostloop|strata generate: prefill|prefill timing|error' "$O/lf-mmq$g.err" | tee -a "$O/run.log"
    python3 tools/hip/gfx1103_smoke.py check packs/qwen38-flash-next-q2_0 "$O/lf-mmq$g.out" longfill 2>&1 | tee -a "$O/run.log" | head -3
    local tok
    tok=$(grep -oE 'prefill 1162 tokens in [0-9]+ chunks, [0-9.]+ ms \([0-9.]+ tok/s\)' "$O/lf-mmq$g.err" | grep -oE '\([0-9.]+ tok/s' | grep -oE '[0-9.]+')
    if [ -n "${tok:-}" ] && python3 -c "import sys; sys.exit(0 if float('$tok') >= 20.0 else 1)"; then
        echo "MMQ_GROUP=$g: $tok tok/s - healthy" | tee -a "$O/run.log"
        return 0
    fi
    echo "MMQ_GROUP=$g: ${tok:-NO RESULT} - DEGRADED or faulted; stopping (power cycle, not reboot)" | tee -a "$O/run.log"
    return 1
}

step gate
./build-hip/hip_intrinsics || { echo "gate failed - stop" | tee -a "$O/run.log"; exit 1; }
sleep 60

IDS=$(need_ids longfill.ids)

run_arm 64 g64 || exit 2
sleep 60
run_arm 32 g32 || exit 2
sleep 60
run_arm 16 g16 || exit 2

step summary
grep -hE 'MMQ_GROUP=[0-9]+: ' "$O/run.log" | tail -3 | tee -a "$O/run.log"
echo "P2.7 done" | tee -a "$O/run.log"
