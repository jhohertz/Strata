#!/usr/bin/env bash
# igpu-rework P2.9: the no-reboot-ritual protocol - probe first, burn in second, and stop on the
# machine's own signal instead of praying after an idle.
#
# The old protocol (power cycle, idle hours, run 3 times) assumed a thermal/marginal machine.
# P2.8 killed that: the fault came at 60 C / 65 W, clean thermals, and the micro runs at full
# speed even on the degraded machine - the degradation is the driver's handling of the engine's
# own surface (the 31.64 GiB userptr restore path + the queues), not the DRAM or the package.
#
# This script:
#   1. PROBES the machine state with one longfill (fast -> the boot is usable; slow or faulted ->
#      it stops and records - no guessing, no idling).
#   2. BURNS IN: repeats longfills (60 s apart) and counts the clean runs until the engine's own
#      DEGRADED warning fires (a section > 2 s of host wall after the first five, docs/IGPU.md
#      P2.8), a fault, or MAX_RUNS - whichever first.  The count is the machine's run budget for
#      this configuration.  Baseline (single 31.64 GiB registration): 4-5 per boot (P2.8).
#   3. Captures the restore_userptr_worker journal lines per run (the fault's leading indicator).
#
# Usage:
#   bash tools/hip/p2_boot_run8.sh                     # baseline burn-in (single registration)
#   CHUNK_GIB=8 bash tools/hip/p2_boot_run8.sh         # the chunked-registration mitigation
#   MAX_RUNS=20 CHUNK_GIB=8 bash tools/hip/p2_boot_run8.sh
#
# Success criterion: the chunked boot outlasts the baseline boot (10+ clean runs vs 4-5).  If it
# does, the boot ritual is gone - a machine lasts a day of experiments per power cycle, and the
# power cycle itself becomes rare.  If not, the chunk size or another surface (queue count) is
# the next variable.
set -u
cd "$(cd "$(dirname "$0")/../.." && pwd)"
O=/tmp/p2-out
mkdir -p "$O"
CHUNK_GIB="${CHUNK_GIB:-0}"
MAX_RUNS="${MAX_RUNS:-10}"
TAG="chunk${CHUNK_GIB}"
export STRATA_IGPU_ALIAS=1 STRATA_GROUP_COPY=1 STRATA_HIPBLASLT_WARMUP=1
export STRATA_HIPBLASLT_TUNING="$PWD/tools/hip/gfx1103-hipblaslt-100401.txt"
export STRATA_HOSTLOOP=1 STRATA_PREFILL_TIMING=1 STRATA_SSD_KEEPALIVE=0
if [ "$CHUNK_GIB" != "0" ]; then export STRATA_IGPU_PIN_CHUNK_GIB="$CHUNK_GIB"; fi

step() { echo "=== P2.9 [$TAG $1] $(date '+%H:%M:%S') ===" | tee -a "$O/run8.log"; }

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

# one longfill.  Prints the outcome line: "clean <toks>" | "degraded" | "faulted <exit>".
run_one() {
    local n="$1"
    local t0; t0=$(date '+%F %H:%M:%S')
    timeout 900 ./build-hip/strata --pack packs/qwen38-flash-next-q2_0 \
      --native /home/jhohertz/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
      --ple-gguf /home/jhohertz/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf \
      --mmap-experts --expert-profile data/expert-profile.bin --expert-cache 6000 --prefill 512 --spec 4 \
      --spec-min-p 0.5 --max-context 4096 --kv int8 --pool-workers 8 --adapt-every 0 --pcie-frac 0 \
      --vram-reserve-mib 1024 --greedy --max-new 16 --tokens "$IDS" > "$O/b8-$n.out" 2> "$O/b8-$n.err"
    local rc=$?
    # the fault's leading indicator: the restore worker's pass count in this run's window
    local uw
    uw=$(journalctl --since "$t0" --no-pager 2>/dev/null | grep -c 'restore_userptr_worker' || true)
    if grep -q 'strata prefill hostloop: DEGRADED' "$O/b8-$n.err"; then
        echo "degraded (userptr restore lines since run start: $uw)" | tee -a "$O/run8.log"
        return 2
    fi
    if [ "$rc" -ne 0 ]; then
        echo "faulted exit=$rc (userptr restore lines: $uw)" | tee -a "$O/run8.log"
        return 3
    fi
    local tok
    tok=$(grep -oE '\([0-9.]+ tok/s\)' "$O/b8-$n.err" | head -1 | grep -oE '[0-9.]+')
    if [ -z "${tok:-}" ] || python3 -c "import sys; sys.exit(0 if float('$tok') >= 20.0 else 1)"; then
        echo "clean $tok tok/s (userptr restore lines: $uw)" | tee -a "$O/run8.log"
        return 0
    fi
    echo "degraded $tok tok/s too slow (userptr restore lines: $uw)" | tee -a "$O/run8.log"
    return 2
}

step "gate (micro)"
./build-hip/hip_intrinsics || { echo "gate failed - stop" | tee -a "$O/run8.log"; exit 1; }
sleep 60
IDS=$(need_ids longfill.ids)

step "probe"
run_one probe; pc=$?
if [ "$pc" -ne 0 ]; then
    echo "PROBE NOT CLEAN - the machine is in its slow/degraded phase.  No idling: capture the" | tee -a "$O/run8.log"
    echo "state (docs/IGPU.md P2.9), then a power cycle is the reset.  Journal:" | tee -a "$O/run8.log"
    journalctl --since '10 min ago' --no-pager 2>/dev/null | grep -iE 'amdgpu|MES|hogged|restore' | tail -12 | tee -a "$O/run8.log"
    exit 4
fi

step "burn-in (max $MAX_RUNS runs)"
n=0; clean=0
while [ "$n" -lt "$MAX_RUNS" ]; do
    n=$((n + 1))
    echo "--- burn-in run $n" | tee -a "$O/run8.log"
    run_one "$n"; rc=$?
    if [ "$rc" -eq 0 ]; then clean=$((clean + 1)); sleep 60; continue; fi
    break
done
step "summary"
echo "burn-in $TAG: $clean clean runs (of $n attempted) before: $([ "$rc" -eq 0 ] && echo 'MAX_RUNS reached' || echo 'degradation/fault')" | tee -a "$O/run8.log"
echo "P2.9 done" | tee -a "$O/run8.log"
