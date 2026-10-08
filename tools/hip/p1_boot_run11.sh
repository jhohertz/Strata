#!/usr/bin/env bash
source "$(dirname "$0")/gfx1103_env.sh"
# Boot protocol 11 (run 29): the root cause is fixed - set_aliased_pointers verified the DEVICE
# table and left the HOST mirror (h_ptrs_) at zeros, so device_slot() (the prompt path's lookup)
# returned NULL for every expert: the dequant's NULL-base IMA (dmesg 0x0-0x6000) and the whole
# run 23-28 chain.  This boot runs the REAL P1 path for the first time with a working table:
#   1) gate
#   2) engine alias arithmetic, MMQ ON (the designed path - its gather also reads the complement
#      through device_slot now), named checks active for both subpaths
#      PASS -> chain python, marker, longfill on the same boot (the full P1 validation)
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
  local IDS; IDS=$(python3 -c "import json;print(json.load(open('$OUT/smoke_ids.json'))['$tag'])")
  env "$@" STRATA_IGPU_ALIAS=1 STRATA_TRACE=1 STRATA_GROUP_COPY=1 STRATA_PREFILL_CHECKS=1 \
      STRATA_HIPBLASLT_WARMUP=1 STRATA_HIPBLASLT_TUNING="$PWD/tools/hip/gfx1103-hipblaslt-100401.txt" \
    timeout 900 "$BUILD/strata" "${ARGS[@]}" --tokens "$IDS" > $OUT/$tag.out 2> $OUT/$tag.err
  return $?
}

echo "== 1) gate $(date +%H:%M:%S)"
timeout 60 "$BUILD/hip_intrinsics" 2>&1 | tail -1 | grep -q 'parity OK' || { echo "FAIL: gate"; exit 1; }

echo "== 2) alias arithmetic, MMQ ON (the real P1 path)  $(date +%H:%M:%S)"
if run_engine arithmetic; then
  grep -E 'prefill +[0-9]+ tokens|decode +[0-9]+ tokens' "$OUT"/arithmetic.out
  echo "PASS: alias arithmetic with the fixed table - chain python/marker/longfill"
  for CASE in python marker longfill; do
    echo "== $CASE  $(date +%H:%M:%S)"
    if run_engine $CASE; then
      echo "PASS: $CASE"
      grep -E 'prefill +[0-9]+ tokens|decode +[0-9]+ tokens' $OUT/$CASE.out | head -2
    else
      echo "FAIL: $CASE - the named check says where; stop and reboot"
      grep -E 'prefill: layer' $OUT/$CASE.err | head -3 | cut -c1-200
      break
    fi
  done
else
  echo "FAIL: alias arithmetic - the named check says where; stop and reboot"
  grep -E 'prefill: layer|strata trace: compute' "$OUT"/arithmetic.err | head -4 | cut -c1-300
fi
