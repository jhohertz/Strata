#!/usr/bin/env bash
source "$(dirname "$0")/gfx1103_env.sh"
# igpu-rework P2.7/P2.9d: the launch-cost A/B - STRATA_MMQ_GROUP 64/32/16, same boot.
#
# Why (docs/IGPU.md P2.7): the wall is ~50,000 launches at their in-engine cost, not any kernel's
# compute (every expert path lands at ~39 tok/s; the f16 A/B has a near-identical phase profile to
# the MMQ path).  Bigger MMQ groups collapse the ~5,200 gemm-side launches (18 groups/section at
# 16 -> 5 at 64).  If the wall is launch count, prefill rises; if it is kernel time, it does not.
#
# Mechanics: v2 from run8 - the engine is NEVER killed.  A fault is recognized from stderr in
# seconds, recorded, and the hung process is left alive (it dies with the power cycle).  The boot
# budget is ~5 loaded sessions (P2.9d): probe + 3 arms fits, arms ordered biggest-lever first so a
# budget overrun still answers the headline question.  60 s cooldowns between sessions.
#
# Usage (after a power cycle; no idle needed - the probe decides):  bash tools/hip/p2_boot_run9.sh
set -u
cd "$(cd "$(dirname "$0")/../.." && pwd)"
O="$OUT"
mkdir -p "$O"
export STRATA_IGPU_ALIAS=1 STRATA_GROUP_COPY=1 STRATA_HIPBLASLT_WARMUP=1
export STRATA_HIPBLASLT_TUNING="$PWD/tools/hip/gfx1103-hipblaslt-100401.txt"
export STRATA_HOSTLOOP=1 STRATA_PREFILL_TIMING=1 STRATA_SSD_KEEPALIVE=0

step() { echo "=== P2.7-AB [$1] $(date '+%H:%M:%S') ===" | tee -a "$O/run9.log"; }

need_ids() {
    if [ ! -f "$O/$1" ]; then
        [ -f "$O/smoke_ids.json" ] || python3 tools/hip/gfx1103_smoke.py prep "$PACK" "$O" >&2
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

# one longfill with $1 = group size; watched, never killed.  Returns 0 clean, 2 degraded, 3 faulted.
run_arm() {
    local g="$1" t0 rc=0
    t0=$(date '+%F %H:%M:%S')
    : > "$O/a9-$g.err"
    env STRATA_MMQ_GROUP="$g" "$BUILD/strata" --pack "$PACK" \
      --native "$SHARD1" \
      --ple-gguf "$SHARD2" \
      --mmap-experts --expert-profile data/expert-profile.bin --expert-cache 6000 --prefill 512 --spec 4 \
      --spec-min-p 0.5 --max-context 4096 --kv int8 --pool-workers 8 --adapt-every 0 --pcie-frac 0 \
      --vram-reserve-mib 1024 --greedy --max-new 16 --tokens "$IDS" > "$O/a9-$g.out" 2> "$O/a9-$g.err" &
    local pid=$! waited=0
    while kill -0 "$pid" 2>/dev/null; do
        sleep 3; waited=$((waited + 3))
        if grep -qE 'unspecified launch failure|error: illegal' "$O/a9-$g.err" 2>/dev/null; then
            echo "MMQ_GROUP=$g: FAULTED at t+${waited}s - process left alive (power cycle; do not kill)" | tee -a "$O/run9.log"
            return 3
        fi
        if grep -q 'strata prefill hostloop: DEGRADED' "$O/a9-$g.err" 2>/dev/null; then
            echo "MMQ_GROUP=$g: DEGRADED at t+${waited}s - letting it finish on its own" | tee -a "$O/run9.log"
        fi
    done
    local tok
    tok=$(grep -oE '\([0-9.]+ tok/s\)' "$O/a9-$g.err" | head -1 | grep -oE '[0-9.]+')
    if [ -z "${tok:-}" ]; then
        echo "MMQ_GROUP=$g: ended without a result at t+${waited}s" | tee -a "$O/run9.log"
        return 2
    fi
    echo "MMQ_GROUP=$g: $tok tok/s" | tee -a "$O/run9.log"
    grep -E 'prefill timing|hostloop' "$O/a9-$g.err" | tee -a "$O/run9.log"
    python3 -c "import sys; sys.exit(0 if float('$tok') >= 20.0 else 1)" || return 2
    return 0
}

step gate
"$BUILD/hip_intrinsics" || { echo "gate failed - stop" | tee -a "$O/run9.log"; exit 1; }
sleep 60
IDS=$(need_ids longfill.ids)

# the budget test: does a 512-token default run at the 16 default look normal on this boot?
step "probe (group default)"
run_arm 16; pc=$?
if [ "$pc" -ne 0 ]; then
    echo "PROBE NOT CLEAN - machine is not live; record and power cycle (no idling helps)" | tee -a "$O/run9.log"
    exit 4
fi

for g in 64 32; do
    sleep 60
    step "arm MMQ_GROUP=$g"
    run_arm "$g" || { echo "stopping the A/B (budget/fault); 64 ran first, 16 baseline is above" | tee -a "$O/run9.log"; exit 2; }
done

step "summary"
grep -hE 'MMQ_GROUP=[0-9]+: [0-9]' "$O/run9.log" | tee -a "$O/run9.log"
echo "P2.7-AB done" | tee -a "$O/run9.log"
