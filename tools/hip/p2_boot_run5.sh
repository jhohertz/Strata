#!/usr/bin/env bash
# igpu-rework P2.5b: the host/GPU separation WITHOUT the hipBLASLt warmup (the fault has now touched the
# warmup window twice: a 719 in a warmup solution, and gather_rows16 right after the 9/9 warmup).
# Fallback for p2_boot_run4.sh if the warmup window faults again.
# STRATA_HOSTLOOP=1 (per-section host wall vs GPU span) + STRATA_PREFILL_TIMING=1 (phase split) in one
# run - they compose.  The answer: host ~ wall -> launch-bound (batch launches, drop the per-layer ids
# D2H + full-stream sync); gpu ~ wall with small host -> the kernels are genuinely ~5 ms/expert on
# this APU and the fix is a different expert kernel shape.
set -u
cd "$(cd "$(dirname "$0")/../.." && pwd)"
O=/tmp/p2-out
mkdir -p "$O"
export STRATA_IGPU_ALIAS=1 STRATA_GROUP_COPY=1 STRATA_HIPBLASLT_WARMUP=0
export STRATA_HIPBLASLT_TUNING="$PWD/tools/hip/gfx1103-hipblaslt-100401.txt"
export STRATA_HOSTLOOP=1 STRATA_PREFILL_TIMING=1

step() { echo "=== P2.5b [$1] $(date '+%H:%M:%S') ===" | tee -a "$O/run.log"; }

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

step gate
./build-hip/hip_intrinsics || { echo "gate failed - stop" | tee -a "$O/run.log"; exit 1; }
sleep 60

step longfill hostloop+timing
IDS=$(need_ids longfill.ids)
timeout 900 ./build-hip/strata --pack packs/qwen38-flash-next-q2_0 \
  --native /home/jhohertz/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
  --ple-gguf /home/jhohertz/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf \
  --mmap-experts --expert-profile data/expert-profile.bin --expert-cache 6000 --prefill 512 --spec 4 \
  --spec-min-p 0.5 --max-context 4096 --kv int8 --pool-workers 8 --adapt-every 0 --pcie-frac 0 \
  --vram-reserve-mib 1024 --greedy --max-new 160 --tokens "$IDS" > "$O/lf-hl-b.out" 2> "$O/lf-hl-b.err"
echo "exit=$?" | tee -a "$O/run.log"
grep -E 'hostloop|strata generate: prefill|timing' "$O/lf-hl-b.err" | tee -a "$O/run.log"
python3 tools/hip/gfx1103_smoke.py check packs/qwen38-flash-next-q2_0 "$O/lf-hl-b.out" longfill 2>&1 | tee -a "$O/run.log" | head -3
echo "P2.5b done" | tee -a "$O/run.log"
