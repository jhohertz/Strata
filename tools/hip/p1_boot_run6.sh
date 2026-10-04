#!/usr/bin/env bash
# Boot protocol 6: the IMA in the grp_mapped expert path.
#   1) gate
#   2) micro --alias 8: a kernel read through the cudaHostGetDevicePointer ALIAS of a pinned 8 GiB
#      region (the exact mechanism the engine's copy_i32 kernels use on m.grp_dev)
#   3) engine alias arithmetic with STRATA_GROUP_COPY=1 (the engine's own bypass: no grp buffer,
#      plain cudaMemcpyAsync copies) - if the IMA disappears, the grp_mapped path is the culprit
set -u
cd "$(dirname "$0")/../.."
NAT=/home/jhohertz/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/Q2_0
ARGS=(--pack packs/qwen38-flash-next-q2_0 --native $NAT/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
      --ple-gguf $NAT/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf
      --mmap-experts --expert-profile data/expert-profile.bin --expert-cache 6000 --prefill 512 --spec 4
      --spec-min-p 0.5 --max-context 4096 --kv int8 --pool-workers 8 --adapt-every 0 --pcie-frac 0
      --vram-reserve-mib 1024 --greedy --max-new 160)

echo "== 1) gate $(date +%H:%M:%S)"
timeout 60 ./build-hip/hip_intrinsics 2>&1 | tail -1 | grep -q 'parity OK' || { echo "FAIL: gate"; exit 1; }

echo "== 2) micro --alias 8  $(date +%H:%M:%S)"
timeout 600 ./build-hip/igpu_gtt_micro --alias 8 2>&1 | tail -4
RC=${PIPESTATUS[0]}
echo "(micro rc=$RC; a fault here degrades the APU, but it answers the mechanism question - stop and reboot)"
[ $RC -ne 0 ] && exit 1

echo "== 3) engine alias arithmetic, STRATA_GROUP_COPY=1  $(date +%H:%M:%S)"
IDS=$(python3 -c "import json;print(json.load(open('/tmp/p0-out/smoke_ids.json'))['arithmetic'])")
env STRATA_IGPU_ALIAS=1 STRATA_TRACE=1 STRATA_GROUP_COPY=1 STRATA_HIPBLASLT_WARMUP=1 \
    STRATA_HIPBLASLT_TUNING="$PWD/tools/hip/gfx1103-hipblaslt-100401.txt" \
  timeout 900 ./build-hip/strata "${ARGS[@]}" --tokens "$IDS" > /tmp/p1-out/gc1.out 2> /tmp/p1-out/gc1.err
RC=$?
if [ $RC -eq 0 ]; then
  echo "PASS: alias arithmetic with STRATA_GROUP_COPY=1"
  grep -E 'decode +160 tokens|prefill +[0-9]+ tokens' /tmp/p1-out/gc1.out
  grep -m1 'experts streamed' /tmp/p1-out/gc1.err
  echo "== 4) python, marker, longfill (same boot)  $(date +%H:%M:%S)"
  for CASE in python marker longfill; do
    IDS=$(python3 -c "import json;print(json.load(open('/tmp/p0-out/smoke_ids.json'))['$CASE'])")
    env STRATA_IGPU_ALIAS=1 STRATA_GROUP_COPY=1 STRATA_HIPBLASLT_WARMUP=1 \
        STRATA_HIPBLASLT_TUNING="$PWD/tools/hip/gfx1103-hipblaslt-100401.txt" \
      timeout 900 ./build-hip/strata "${ARGS[@]}" --tokens "$IDS" > /tmp/p1-out/$CASE.out 2> /tmp/p1-out/$CASE.err
    RC2=$?
    if [ $RC2 -eq 0 ]; then
      echo "PASS: $CASE"
      grep -E 'decode +[0-9]+ tokens|prefill +[0-9]+ tokens' /tmp/p1-out/$CASE.out | head -2
    else
      echo "FAIL: $CASE (rc=$RC2) - stop and reboot"; tail -3 /tmp/p1-out/$CASE.err | cut -c1-140
    fi
  done
else
  echo "FAIL: alias arithmetic with GROUP_COPY=1 (rc=$RC) - the IMA is elsewhere; last trace + tail:"
  grep -E 'strata trace' /tmp/p1-out/gc1.err | tail -4
  tail -4 /tmp/p1-out/gc1.err | cut -c1-140
fi
