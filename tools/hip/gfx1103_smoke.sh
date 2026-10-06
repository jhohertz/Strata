#!/bin/bash
# gfx1103 Phase-B smoke driver (docs/GFX1103.md, §6).  One-shot generate mode,
# four checks, run to completion - the driver NEVER signals the engine mid-run
# (killing a HIP process with in-flight GPU work wedges this APU's GPU; §9.11).
#
# Usage: gfx1103_smoke.sh   (paths overridable via PACK/SHARD1/SHARD2/STRATA/OUT)
set -u
cd /home/jhohertz/co/Strata

PACK=${PACK:-packs/qwen38-flash-next-q2_0}
SHARD1=${SHARD1:-/home/jhohertz/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf}
SHARD2=${SHARD2:-/home/jhohertz/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf}
STRATA=${STRATA:-./build-hip/strata}
OUT=${OUT:-/tmp/strata-smoke}
TUNING=${TUNING:-tools/hip/gfx1103-hipblaslt-100401.txt}
mkdir -p "$OUT"
SUM="$OUT/summary.txt"
: > "$SUM"
say() { echo "$@" | tee -a "$SUM"; }

# ---- GPU health gate: the CMake-built kernel parity test must pass in 60 s --------
# (A wedged GPU hangs or faults even trivial kernels; do NOT start the engine into it
#  - §9.11.)  The gate uses the project's own binary: an ad-hoc `clang++ -x hip` kernel
#  faults with "illegal memory access" on this box while CMake-built kernels run fine
#  (unexplained, §9.13) - a fresh compile would false-alarm on a healthy GPU.
GATE=./build-hip/hip_intrinsics
if [ ! -x "$GATE" ]; then
  say "GPU GATE FAILED: $GATE missing - build first (cmake --build build-hip)."
  exit 2
fi
if ! timeout 60 "$GATE" > "$OUT/gate.log" 2>&1; then
  say "GPU GATE FAILED: hip_intrinsics - the GPU is wedged (§9.11). Reboot; do not proceed."
  tail -3 "$OUT/gate.log" | tee -a "$SUM"
  exit 2
fi
say "GPU gate: OK ($(date +%H:%M:%S))"

# ---- token ids -------------------------------------------------------------------
if [ ! -f "$OUT/smoke_ids.json" ]; then
  python3 tools/hip/gfx1103_smoke.py prep "$PACK" "$OUT" | tee -a "$SUM"
fi

# ---- the four runs -----------------------------------------------------------------
for name in arithmetic python marker longfill; do
  IDS=$(python3 -c "import json;print(json.load(open('$OUT/smoke_ids.json'))['$name'])")
  say "== $name: start $(date +%H:%M:%S) =="
  env STRATA_HIPBLASLT_TUNING="$PWD/$TUNING" \
      timeout 1500 "$STRATA" \
        --pack "$PACK" --native "$SHARD1" --ple-gguf "$SHARD2" \
        --mmap-experts --expert-profile data/expert-profile.bin --expert-cache 6000 \
        --prefill 512 --spec 4 --spec-min-p 0.5 \
        --max-context 4096 --kv int8 --pool-workers 8 \
        --adapt-every 0 --pcie-frac 0 --vram-reserve-mib 1024 \
        --greedy --max-new 160 --tokens "$IDS" \
        > "$OUT/$name.out" 2> "$OUT/$name.err"
  rc=$?
  say "== $name: engine exit=$rc end $(date +%H:%M:%S) =="
  if [ $rc -ne 0 ]; then
    say "  [FAIL] $name: engine exited $rc; stderr tail:"
    tail -3 "$OUT/$name.err" | tee -a "$SUM"
    continue
  fi
  python3 tools/hip/gfx1103_smoke.py check "$PACK" "$OUT/$name.out" "$name" | tee -a "$SUM"
done

say "== smoke pass done $(date +%H:%M:%S): $(grep -c '^\[PASS\]' "$SUM") PASS / $(grep -cE '^\[(PASS|FAIL)\]' "$SUM") =="
grep -qE 'GPU GATE FAILED' "$SUM" && exit 2
[ "$(grep -c '^\[PASS\]' "$SUM")" -ge 4 ]
