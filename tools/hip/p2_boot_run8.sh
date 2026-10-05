#!/usr/bin/env bash
# igpu-rework P2.9: the no-reboot-ritual protocol - probe first, burn in second, and stop on the
# machine's own signal instead of praying after an idle.  (v2: never kills the engine - see below)
#
# The old protocol (power cycle, idle hours, run 3 times) assumed a thermal/marginal machine.
# P2.8 killed that: the fault came at 60 C / 65 W, clean thermals, and the micro runs at full
# speed even on the degraded machine - the degradation is the driver's handling of the engine's
# own surface (the userptr restore path + the KFD queues under sustained load), not the DRAM or
# the package.
#
# v2 lesson (the 2026-10-05 02:3x baseline boot): a faulting run does not exit - it hangs in
# its error path (the "unspecified launch failure" is printed, then the process sits).  v1's
# `timeout 900` therefore SIGTERM'd a hung HIP process 14 minutes after its fault, and the kill
# is what drove the teardown's MES queue-removal hang (the journal's REMOVE_QUEUE lines landed
# exactly at the SIGTERM).  A faulted GPU process must never be killed while it holds queues.
# v2 runs strata in the background and watches its stderr: a clean run exits on its own (~65 s
# warm); a faulted run is RECOGNIZED in seconds, recorded, and left alive - the machine is in
# its bad phase anyway, and the hung process dies with the power cycle that resets the state.
#
# This script:
#   1. PROBES the machine state with one longfill (fast -> the boot is usable; slow or faulted ->
#      it records and stops - no guessing, no idling).
#   2. BURNS IN: repeats longfills (60 s apart) and counts the clean runs until a fault is
#      recognized, the engine's DEGRADED warning fires (a section > 2 s of host wall after the
#      first five, docs/IGPU.md P2.8), or MAX_RUNS.  The count is the boot's run budget for the
#      configuration.  Baseline (single 31.64 GiB registration): 4 engine runs per boot - the
#      probe + 3 clean, the 5th faults (P2.8 04:2x and the 02:3x boot, both).
#   3. Captures the per-run journal forensics: restore_userptr_worker pass lines (the fault's
#      leading indicator) and MES REMOVE_QUEUE lines (the teardown hang).
#
# Usage:
#   bash tools/hip/p2_boot_run8.sh                     # baseline burn-in (single registration)
#   CHUNK_GIB=8 bash tools/hip/p2_boot_run8.sh         # the chunked-registration mitigation
#   MAX_RUNS=20 CHUNK_GIB=8 bash tools/hip/p2_boot_run8.sh
#
# If the script stops after a "faulted" line: DO NOT kill the strata process.  Power cycle.
#
# Success criterion for the A/B: the chunked boot outlasts the baseline (10+ clean runs vs 4).
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

journal_counts() {   # $1 = since timestamp; prints "restore=<n> mes=<n>"
    local s
    s=$(journalctl --since "$1" --no-pager 2>/dev/null | grep -E 'restore_userptr_worker|MES failed to respond' | wc -l)
    local r m
    r=$(journalctl --since "$1" --no-pager 2>/dev/null | grep -c 'restore_userptr_worker' || true)
    m=$(journalctl --since "$1" --no-pager 2>/dev/null | grep -c 'MES failed to respond' || true)
    echo "userptr-restore lines: $r, MES lines: $m"
}

# one longfill, watched, never killed.
# prints "clean <toks>" (0), "degraded ..." (2), or "faulted ..." (3, the process is left alive)
run_one() {
    local n="$1"
    local t0; t0=$(date '+%F %H:%M:%S')
    : > "$O/b8-$n.err"
    ./build-hip/strata --pack packs/qwen38-flash-next-q2_0 \
      --native /home/jhohertz/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
      --ple-gguf /home/jhohertz/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf \
      --mmap-experts --expert-profile data/expert-profile.bin --expert-cache 6000 --prefill 512 --spec 4 \
      --spec-min-p 0.5 --max-context 4096 --kv int8 --pool-workers 8 --adapt-every 0 --pcie-frac 0 \
      --vram-reserve-mib 1024 --greedy --max-new 16 --tokens "$IDS" > "$O/b8-$n.out" 2> "$O/b8-$n.err" &
    local pid=$!
    local waited=0
    while kill -0 "$pid" 2>/dev/null; do
        sleep 3
        waited=$((waited + 3))
        if grep -qE 'unspecified launch failure|error: illegal' "$O/b8-$n.err" 2>/dev/null; then
            echo "faulted at t+${waited}s - process left alive (DO NOT kill it; power cycle to reset)" | tee -a "$O/run8.log"
            journal_counts "$t0" | tee -a "$O/run8.log"
            return 3
        fi
        if grep -q 'strata prefill hostloop: DEGRADED' "$O/b8-$n.err" 2>/dev/null; then
            # the slow phase: let the run run to its end (it faults on its own, ~16 min in),
            # but record the state now - no more clean runs will come this boot
            echo "degraded (DEGRADED warning at t+${waited}s) - waiting for the run to finish on its own" | tee -a "$O/run8.log"
            while kill -0 "$pid" 2>/dev/null; do
                sleep 5
                if grep -qE 'unspecified launch failure|error: illegal' "$O/b8-$n.err" 2>/dev/null; then
                    echo "faulted after the slow phase - process left alive (power cycle)" | tee -a "$O/run8.log"
                    journal_counts "$t0" | tee -a "$O/run8.log"
                    return 3
                fi
            done
            journal_counts "$t0" | tee -a "$O/run8.log"
            return 2
        fi
        if [ "$waited" -gt 1500 ]; then
            echo "still running at t+${waited}s (a slow-phase run; not killing it)" | tee -a "$O/run8.log"
        fi
    done
    local tok
    tok=$(grep -oE '\([0-9.]+ tok/s\)' "$O/b8-$n.err" | head -1 | grep -oE '[0-9.]+')
    journal_counts "$t0" | tee -a "$O/run8.log"
    if [ -n "${tok:-}" ] && python3 -c "import sys; sys.exit(0 if float('$tok') >= 20.0 else 1)"; then
        echo "clean $tok tok/s" | tee -a "$O/run8.log"
        return 0
    fi
    echo "ended without a prefill result at t+${waited}s" | tee -a "$O/run8.log"
    return 2
}

step "gate (micro)"
./build-hip/hip_intrinsics || { echo "gate failed - stop" | tee -a "$O/run8.log"; exit 1; }
sleep 60
IDS=$(need_ids longfill.ids)

step "probe"
run_one probe; pc=$?
if [ "$pc" -ne 0 ]; then
    echo "PROBE NOT CLEAN - the machine is in its slow/degraded phase or faulted.  No idling helps" | tee -a "$O/run8.log"
    echo "(P2.8); the reset is a full power cycle.  Recent journal:" | tee -a "$O/run8.log"
    journalctl --since '10 min ago' --no-pager 2>/dev/null | grep -iE 'amdgpu|MES|hogged|restore' | tail -12 | tee -a "$O/run8.log"
    exit 4
fi

step "burn-in (max $MAX_RUNS runs)"
n=0; clean=0; rc=0
while [ "$n" -lt "$MAX_RUNS" ]; do
    n=$((n + 1))
    echo "--- burn-in run $n" | tee -a "$O/run8.log"
    run_one "$n"; rc=$?
    if [ "$rc" -eq 0 ]; then clean=$((clean + 1)); sleep 60; continue; fi
    break
done
step "summary"
echo "burn-in $TAG: $clean clean runs (of $n attempted) before: $([ "$rc" -eq 0 ] && echo 'MAX_RUNS reached' || echo 'degradation/fault (the faulted process, if any, is still alive - power cycle)')" | tee -a "$O/run8.log"
echo "P2.9 done" | tee -a "$O/run8.log"
