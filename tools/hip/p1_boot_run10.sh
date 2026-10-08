#!/usr/bin/env bash
source "$(dirname "$0")/gfx1103_env.sh"
# Boot protocol 10 (run 28): the micro passes in EVERY variant (legacy, nb stream, nb+6GiB
# pressure); the engine still faults in iq_dequant_gu_f16.  Two new measurements:
#   1) gate
#   2) micro --dequant 32 nb 6: the engine's complement SCALE (a 31.64 GiB registered region,
#      not 8 GiB) + nb stream + 6 GiB device pressure = the engine's full state in the micro.
#      IMA -> the scale is the trigger (bisect 12/24/32 next boot); rc=2 -> cannot test the
#      scale (memlock/alloc); PASS -> continue
#   3) engine alias arithmetic, MMQ OFF, fine checks; the compute dump now prints BEFORE the
#      dequant launch - the actual blob_dev / dq_gu / Xs / ... pointer values at fault time.
#      blob_dev = 0 -> the table is corrupt (how?); blob_dev valid -> the IMA is driver/launch
#      level (the pointer is mangled between the launch descriptor and the GPU)
set -u
cd "$(dirname "$0")/../.."
mkdir -p "$OUT"
ARGS=(--pack "$PACK" --native "$SHARD1"
      --ple-gguf "$SHARD2"
      --mmap-experts --expert-profile data/expert-profile.bin --expert-cache 6000 --prefill 512 --spec 4
      --spec-min-p 0.5 --max-context 4096 --kv int8 --pool-workers 8 --adapt-every 0 --pcie-frac 0
      --vram-reserve-mib 1024 --greedy --max-new 160)
run_micro() {
  local tag=$1; shift
  timeout 600 "$BUILD/igpu_gtt_micro" --dequant "$@" > $OUT/$tag.out 2> $OUT/$tag.err
  local rc=$?
  tail -n 2 $OUT/$tag.out | cut -c1-160; tail -n 2 $OUT/$tag.err | cut -c1-160
  return $rc
}
run_engine() {
  local tag=$1; shift
  local IDS; IDS=$(python3 -c "import json;print(json.load(open('$OUT/smoke_ids.json'))['arithmetic'])")
  env "$@" STRATA_IGPU_ALIAS=1 STRATA_TRACE=1 STRATA_GROUP_COPY=1 STRATA_PREFILL_CHECKS=1 \
      STRATA_PREFILL_MMQ=0 STRATA_HIPBLASLT_WARMUP=1 STRATA_HIPBLASLT_TUNING="$PWD/tools/hip/gfx1103-hipblaslt-100401.txt" \
    timeout 900 "$BUILD/strata" "${ARGS[@]}" --tokens "$IDS" > $OUT/$tag.out 2> $OUT/$tag.err
  return $?
}

echo "== 1) gate $(date +%H:%M:%S)"
timeout 60 "$BUILD/hip_intrinsics" 2>&1 | tail -1 | grep -q 'parity OK' || { echo "FAIL: gate"; exit 1; }

echo "== 2) micro dequant, 32 GiB region + nb + 6 GiB pressure  $(date +%H:%M:%S)"
RC=0
run_micro dq32 32 nb 6 || RC=$?
if [ $RC -eq 1 ]; then
  echo "CONFIRMED at scale: the 32 GiB region reproduces the IMA in the micro - stop and reboot (bisect)"
  exit 1
elif [ $RC -ne 0 ]; then
  echo "cannot test the scale (rc=$RC: allocation/registration) - continuing to the engine anyway"
fi

echo "== 3) engine alias arithmetic, MMQ OFF, dump-before-launch  $(date +%H:%M:%S)"
if run_engine dump; then
  grep -E 'prefill +[0-9]+ tokens|decode +[0-9]+ tokens' "$OUT"/dump.out
  echo "PASS: the engine no longer faults (state-dependent?)"
else
  echo "FAIL: the dump above prints the pointer values at fault time"
  grep -E 'strata trace: compute|prefill: layer' "$OUT"/dump.err | head -4 | cut -c1-300
fi
