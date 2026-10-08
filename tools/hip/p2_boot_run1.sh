#!/usr/bin/env bash
source "$(dirname "$0")/gfx1103_env.sh"
# Boot protocol P2.1: profile the alias prefill's wall time before optimizing it.
# The P1 analysis said ~20.6 s of ~31.3 s per prompt was host-side - but that was measured
# on the 0.1.29 copy path; the alias run mixes the work differently (no H2D copies, no
# staging, no pool misses).  STRATA_PREFILL_TIMING=1 (existing, event-based, dGPU-neutral)
# gives the per-phase GPU timeline + the host breakdown (chunk setup / waiting / after
# chunk / PLE).  Two profiles: the 1162-token prompt (the prefill that matters) and the
# 44-token prompt (the 80 ms/token TTFT case).
set -u
cd "$(dirname "$0")/../.."
mkdir -p "$OUT"
ARGS=(--pack "$PACK" --native "$SHARD1"
      --ple-gguf "$SHARD2"
      --mmap-experts --expert-profile data/expert-profile.bin --expert-cache 6000 --prefill 512 --spec 4
      --spec-min-p 0.5 --max-context 4096 --kv int8 --pool-workers 8 --adapt-every 0 --pcie-frac 0
      --vram-reserve-mib 1024 --greedy --max-new 160)
run_engine() {  # $1 = tag, $2 = case
  local tag=$1 case=$2
  local IDS; IDS=$(python3 -c "import json;print(json.load(open('$OUT/smoke_ids.json'))['$case'])")
  env STRATA_IGPU_ALIAS=1 STRATA_GROUP_COPY=1 STRATA_PREFILL_TIMING=1 \
      STRATA_HIPBLASLT_WARMUP=1 STRATA_HIPBLASLT_TUNING="$PWD/tools/hip/gfx1103-hipblaslt-100401.txt" \
    timeout 900 "$BUILD/strata" "${ARGS[@]}" --tokens "$IDS" > $OUT/$tag.out 2> $OUT/$tag.err
  return $?
}

echo "== 1) gate $(date +%H:%M:%S)"
timeout 60 "$BUILD/hip_intrinsics" 2>&1 | tail -1 | grep -q 'parity OK' || { echo "FAIL: gate"; exit 1; }

echo "== 2) longfill (1162 tokens) + timing  $(date +%H:%M:%S)"
run_engine lf-timing longfill || { echo "FAIL: longfill timed out/faulted"; exit 1; }
grep -E 'strata prefill timing' "$OUT"/lf-timing.err
grep -E 'prefill +[0-9]+ tokens|decode +[0-9]+ tokens' "$OUT"/lf-timing.out | head -2

echo "== 3) arithmetic (44 tokens) + timing  $(date +%H:%M:%S)"
run_engine ar-timing arithmetic || { echo "FAIL: arithmetic timed out/faulted"; exit 1; }
grep -E 'strata prefill timing' "$OUT"/ar-timing.err
grep -E 'prefill +[0-9]+ tokens|decode +[0-9]+ tokens' "$OUT"/ar-timing.out | head -2

echo "== done: the timing lines above are the P2 target list"
