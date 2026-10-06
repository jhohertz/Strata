#!/usr/bin/env bash
# igpu-rework P2.6: the host/GPU separation with THERMAL LOGGING, after a FULL power cycle
# (hold the power button or unplug the PSU - a plain reboot has not reset the fault cluster)
# and a long idle.  1 Hz rocm-smi temp/power in the background answers the one question the
# software side cannot: does the APU package approach its thermal/power limit in the faulting
# window (the 04:24-18:24 cluster: userptr restore storm + MES queue hang, docs/IGPU.md)?
set -u
cd "$(cd "$(dirname "$0")/../.." && pwd)"
O=/tmp/p2-out
mkdir -p "$O"
export STRATA_IGPU_ALIAS=1 STRATA_GROUP_COPY=1 STRATA_HIPBLASLT_WARMUP=1
export STRATA_HIPBLASLT_TUNING="$PWD/tools/hip/gfx1103-hipblaslt-100401.txt"
export STRATA_HOSTLOOP=1 STRATA_PREFILL_TIMING=1 STRATA_SSD_KEEPALIVE=0

step() { echo "=== P2.6 [$1] $(date '+%H:%M:%S') ===" | tee -a "$O/run.log"; }

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

# 1 Hz thermal log while the engine runs
(
    while true; do
        printf '%s ' "$(date '+%H:%M:%S')" >> "$O/therm.log"
        rocm-smi --showtemp --showpower 2>/dev/null | tr '\n' ' ' | sed 's/  */ /g' >> "$O/therm.log"
        echo >> "$O/therm.log"
        sleep 1
    done
) & THERM=$!
echo "thermal logger pid=$THERM" | tee -a "$O/run.log"

step longfill hostloop+timing+thermals
IDS=$(need_ids longfill.ids)
timeout 900 ./build-hip/strata --pack packs/qwen38-flash-next-q2_0 \
  --native /home/jhohertz/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
  --ple-gguf /home/jhohertz/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf \
  --mmap-experts --expert-profile data/expert-profile.bin --expert-cache 6000 --prefill 512 --spec 4 \
  --spec-min-p 0.5 --max-context 4096 --kv int8 --pool-workers 8 --adapt-every 0 --pcie-frac 0 \
  --vram-reserve-mib 1024 --greedy --max-new 160 --tokens "$IDS" > "$O/lf-hl.out" 2> "$O/lf-hl.err"
echo "exit=$?" | tee -a "$O/run.log"
kill "$THERM" 2>/dev/null
grep -E 'hostloop|strata generate: prefill|prefill timing' "$O/lf-hl.err" | tee -a "$O/run.log"
python3 tools/hip/gfx1103_smoke.py check packs/qwen38-flash-next-q2_0 "$O/lf-hl.out" longfill 2>&1 | tee -a "$O/run.log" | head -3
echo "P2.6 done" | tee -a "$O/run.log"
