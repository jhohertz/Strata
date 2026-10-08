#!/usr/bin/env bash
source "$(dirname "$0")/gfx1103_env.sh"
# Boot protocol 7: the null-base IMA in the expert section (dmesg: GPU VA 0x0-0x6000, 4 KiB stride).
#   1) gate
#   2) engine alias arithmetic with STRATA_PREFILL_MMQ=0 (the non-MMQ subpath: FP16 dequant + gemm,
#      no mmq gemm / no ctx pool) + named section checks
#      PASS -> chain python/marker/longfill (a usable P1 config), then one MMQ-on run LAST for the
#             localization data (expected to fault - it costs the boot, which is over)
#      FAIL -> the checks name the phase; stop and reboot
set -u
cd "$(dirname "$0")/../.."
mkdir -p "$OUT"
ARGS=(--pack "$PACK" --native "$SHARD1"
      --ple-gguf "$SHARD2"
      --mmap-experts --expert-profile data/expert-profile.bin --expert-cache 6000 --prefill 512 --spec 4
      --spec-min-p 0.5 --max-context 4096 --kv int8 --pool-workers 8 --adapt-every 0 --pcie-frac 0
      --vram-reserve-mib 1024 --greedy --max-new 160)
run_engine() {  # $1 = tag, $2.. = extra env
  local tag=$1; shift
  local IDS; IDS=$(python3 -c "import json;print(json.load(open('$OUT/smoke_ids.json'))['arithmetic'])")
  env "$@" STRATA_IGPU_ALIAS=1 STRATA_TRACE=1 STRATA_GROUP_COPY=1 STRATA_PREFILL_CHECKS=1 \
      STRATA_HIPBLASLT_WARMUP=1 STRATA_HIPBLASLT_TUNING="$PWD/tools/hip/gfx1103-hipblaslt-100401.txt" \
    timeout 900 "$BUILD/strata" "${ARGS[@]}" --tokens "$IDS" > $OUT/$tag.out 2> $OUT/$tag.err
}

echo "== 1) gate $(date +%H:%M:%S)"
timeout 60 "$BUILD/hip_intrinsics" 2>&1 | tail -1 | grep -q 'parity OK' || { echo "FAIL: gate"; exit 1; }

echo "== 2) alias arithmetic, MMQ OFF  $(date +%H:%M:%S)"
run_engine mmq0 STRATA_PREFILL_MMQ=0
RC=$?
grep -E 'prefill: layer|VRAM at first' "$OUT"/mmq0.err | head -4
if [ $RC -eq 0 ]; then
  echo "PASS: alias arithmetic with STRATA_PREFILL_MMQ=0"
  grep -E 'decode +160 tokens|prefill +[0-9]+ tokens' "$OUT"/mmq0.out
  echo "== 3) python, marker, longfill, MMQ OFF (same boot)  $(date +%H:%M:%S)"
  for CASE in python marker longfill; do
    IDS=$(python3 -c "import json;print(json.load(open('$OUT/smoke_ids.json'))['$CASE'])")
    env STRATA_PREFILL_MMQ=0 STRATA_IGPU_ALIAS=1 STRATA_GROUP_COPY=1 \
        STRATA_HIPBLASLT_WARMUP=1 STRATA_HIPBLASLT_TUNING="$PWD/tools/hip/gfx1103-hipblaslt-100401.txt" \
      timeout 900 "$BUILD/strata" "${ARGS[@]}" --tokens "$IDS" > $OUT/$CASE.out 2> $OUT/$CASE.err
    RC2=$?
    if [ $RC2 -eq 0 ]; then
      echo "PASS: $CASE"; grep -E 'decode +[0-9]+ tokens|prefill +[0-9]+ tokens' $OUT/$CASE.out | head -2
    else
      echo "FAIL: $CASE (rc=$RC2) - stop and reboot"; tail -3 $OUT/$CASE.err | cut -c1-140
    fi
  done
  echo "== 4) localization: one MMQ-ON run LAST (expected to fault)  $(date +%H:%M:%S)"
  run_engine mmq1
  grep -E 'prefill: layer|VRAM at first' "$OUT"/mmq1.err | head -4
else
  echo "FAIL: alias arithmetic with MMQ OFF (rc=$RC) - the checks above name the phase; stop and reboot"
fi
