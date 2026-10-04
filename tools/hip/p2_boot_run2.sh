#!/usr/bin/env bash
# igpu-rework P2.2: the queued 0.1.29-vs-0.1.36 A/B - STRATA_OLD_IQ_MMVQ=1 (env-only, no rebuild).
# The 0.1.36 characterization (commit 2c110cd) queued this as "faults + prefill regression in one
# lever": the new iq_mmVQ kernel is the suspect for BOTH the 2/9 flake and the -46 % prefill.
# Protocol: gate, then longfill alias with the old mmvq kernel, then arithmetic alias the same.
# Both are checked against the golden tokens.
set -u
cd "$(cd "$(dirname "$0")/../.." && pwd)"
O=/tmp/p2-out
mkdir -p "$O"
export STRATA_IGPU_ALIAS=1 STRATA_GROUP_COPY=1 STRATA_HIPBLASLT_WARMUP=1
export STRATA_HIPBLASLT_TUNING="$PWD/tools/hip/gfx1103-hipblaslt-100401.txt"
export STRATA_OLD_IQ_MMVQ=1

step() { echo "=== P2.2 [$1] $(date '+%H:%M:%S') ===" | tee -a "$O/run.log"; }

step gate
./build-hip/hip_intrinsics || { echo "gate failed - stop" | tee -a "$O/run.log"; exit 1; }
sleep 60

step longfill old-mmvq
IDS=$(cat "$O/lf.ids"); [ -f "$O/lf.ids" ] || { python3 tools/hip/gfx1103_smoke.py prep packs/qwen38-flash-next-q2_0 "$O" >/dev/null 2>&1; IDS=$(cat "$O/lf.ids"); }
timeout 900 ./build-hip/strata --pack packs/qwen38-flash-next-q2_0 \
  --native /home/jhohertz/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
  --ple-gguf /home/jhohertz/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf \
  --mmap-experts --expert-profile data/expert-profile.bin --expert-cache 6000 --prefill 512 --spec 4 \
  --spec-min-p 0.5 --max-context 4096 --kv int8 --pool-workers 8 --adapt-every 0 --pcie-frac 0 \
  --vram-reserve-mib 1024 --greedy --max-new 160 --tokens "$IDS" > "$O/lf-old.out" 2> "$O/lf-old.err"
echo "exit=$?" | tee -a "$O/run.log"
python3 tools/hip/gfx1103_smoke.py check packs/qwen38-flash-next-q2_0 "$O/lf-old.out" longfill 2>&1 | tee -a "$O/run.log" | head -3
sleep 90

step arithmetic old-mmvq
IDS=$(cat "$O/ar.ids")
timeout 900 ./build-hip/strata --pack packs/qwen38-flash-next-q2_0 \
  --native /home/jhohertz/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
  --ple-gguf /home/jhohertz/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf \
  --mmap-experts --expert-profile data/expert-profile.bin --expert-cache 6000 --prefill 512 --spec 4 \
  --spec-min-p 0.5 --max-context 4096 --kv int8 --pool-workers 8 --adapt-every 0 --pcie-frac 0 \
  --vram-reserve-mib 1024 --greedy --max-new 160 --tokens "$IDS" > "$O/ar-old.out" 2> "$O/ar-old.err"
echo "exit=$?" | tee -a "$O/run.log"
python3 tools/hip/gfx1103_smoke.py check packs/qwen38-flash-next-q2_0 "$O/ar-old.out" arithmetic 2>&1 | tee -a "$O/run.log" | head -3
echo "P2.2 done: $(grep -oE '[0-9.]+ prefill tok/s|prefill [0-9.]+|decode [0-9.]+|in [0-9]+ ms' $O/lf-old.err | head -4)" | tee -a "$O/run.log"
