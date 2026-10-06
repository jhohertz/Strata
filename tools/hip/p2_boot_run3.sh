#!/usr/bin/env bash
# igpu-rework P2.3: the 0.1.29 baseline, rebuilt and phase-profiled.
# The -46 % longfill-prefill regression vs 0.1.29 (68.8 -> 37-39 tok/s) is not in the expert gemm
# path (git: byte-identical compute lambda, same llama.cpp pin) and not in iq_mmVQ (P2.2 A/B:
# 39.1 vs 38.0, no change).  This run rebuilds the 0.1.29-era branch (41da073, the commit that
# measured 68.8) in the /home/jhohertz/co/Strata-029 worktree and profiles it with
# STRATA_PREFILL_TIMING=1 - the phase-by-phase diff against P2.1's 0.1.36 alias profile
# (gemm gu 39 %, gather 26 %, gemm dn 13 %, qsa 10 %, gdn 8 %) names the regressed phase.
# Same engine flags as the 0.1.29-era smoke script (copy path, pinned staging, tuned table).
set -u
S=/home/jhohertz/co/Strata-029   # the 41da073 worktree (0.1.29 + gfx1103)
O=/tmp/p2-out-029
mkdir -p "$O"
export STRATA_HIPBLASLT_TUNING="$S/tools/hip/gfx1103-hipblaslt-100401.txt"
export STRATA_PREFILL_TIMING=1

step() { echo "=== P2.3 [$1] $(date '+%H:%M:%S') ===" | tee -a "$O/run.log"; }

# prompts: prep from the 0.1.29 tree (its smoke_ids.json format: a space-joined string per key)
step prep
python3 "$S/tools/hip/gfx1103_smoke.py" prep "$S/packs/qwen38-flash-next-q2_0" "$O" 2>&1 | tee -a "$O/run.log"

run_case() {   # $1 = case name
    local name=$1
    local IDS
    IDS=$(python3 -c "import json;print(json.load(open('$O/smoke_ids.json'))['$name'])")
    step "$name 0.1.29"
    timeout 1500 "$S/build/strata" \
        --pack "$S/packs/qwen38-flash-next-q2_0" \
        --native /home/jhohertz/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
        --ple-gguf /home/jhohertz/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf \
        --mmap-experts --expert-profile "$S/data/expert-profile.bin" --expert-cache 6000 \
        --prefill 512 --spec 4 --spec-min-p 0.5 --max-context 4096 --kv int8 --pool-workers 8 \
        --adapt-every 0 --pcie-frac 0 --vram-reserve-mib 1024 \
        --greedy --max-new 160 --tokens "$IDS" > "$O/$name.out" 2> "$O/$name.err"
    echo "exit=$?" | tee -a "$O/run.log"
    python3 "$S/tools/hip/gfx1103_smoke.py" check "$S/packs/qwen38-flash-next-q2_0" "$O/$name.out" "$name" 2>&1 | tee -a "$O/run.log" | head -3
    sleep 90
}

step gate
"$S/build/hip_intrinsics" || { echo "gate failed - stop" | tee -a "$O/run.log"; exit 1; }
sleep 60

run_case longfill
run_case arithmetic
echo "P2.3 done" | tee -a "$O/run.log"
