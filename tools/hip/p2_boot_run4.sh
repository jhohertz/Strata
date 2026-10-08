#!/usr/bin/env bash
source "$(dirname "$0")/gfx1103_env.sh"
# igpu-rework P2.5: the host/GPU separation for the expert section, clean boot after the 04:2x flake.
# STRATA_HOSTLOOP=1 (per-section host wall vs GPU span) + STRATA_PREFILL_TIMING=1 (phase split) in one
# run - they compose.  The answer: host ~ wall -> launch-bound (batch launches, drop the per-layer ids
# D2H + full-stream sync); gpu ~ wall with small host -> the kernels are genuinely ~5 ms/expert on
# this APU and the fix is a different expert kernel shape.
set -u
cd "$(cd "$(dirname "$0")/../.." && pwd)"
O="$OUT"
mkdir -p "$O"
export STRATA_IGPU_ALIAS=1 STRATA_GROUP_COPY=1 STRATA_HIPBLASLT_WARMUP=1
export STRATA_HIPBLASLT_TUNING="$PWD/tools/hip/gfx1103-hipblaslt-100401.txt"
export STRATA_HOSTLOOP=1 STRATA_PREFILL_TIMING=1

step() { echo "=== P2.5 [$1] $(date '+%H:%M:%S') ===" | tee -a "$O/run.log"; }

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

step gate
"$BUILD/hip_intrinsics" || { echo "gate failed - stop" | tee -a "$O/run.log"; exit 1; }
sleep 60

step longfill hostloop+timing
IDS=$(need_ids longfill.ids)
timeout 900 "$BUILD/strata" --pack "$PACK" \
  --native "$SHARD1" \
  --ple-gguf "$SHARD2" \
  --mmap-experts --expert-profile data/expert-profile.bin --expert-cache 6000 --prefill 512 --spec 4 \
  --spec-min-p 0.5 --max-context 4096 --kv int8 --pool-workers 8 --adapt-every 0 --pcie-frac 0 \
  --vram-reserve-mib 1024 --greedy --max-new 160 --tokens "$IDS" > "$O/lf-hl.out" 2> "$O/lf-hl.err"
echo "exit=$?" | tee -a "$O/run.log"
grep -E 'hostloop|strata generate: prefill|timing' "$O/lf-hl.err" | tee -a "$O/run.log"
python3 tools/hip/gfx1103_smoke.py check "$PACK" "$O/lf-hl.out" longfill 2>&1 | tee -a "$O/run.log" | head -3
echo "P2.5 done" | tee -a "$O/run.log"
