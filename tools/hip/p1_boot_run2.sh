#!/usr/bin/env bash
# Boot protocol 2: A/B the alias hang against the SAME prompt path without alias.
#
# The alias run forces borrow=nullptr (an alias table has no blob arena to lend), which pushes the
# prompt path onto its "allocates its own buffers" branch - a branch the APU smoke has never taken
# (every run borrowed from a 6000-slot arena).  Copy path + --expert-cache 50 makes k+128 > slots,
# so borrow is null there too: the identical branch, zero alias machinery.
#
#   2 hangs and the alias hung the same way -> the bug is in the own-buffers prompt path (upstream,
#     untested here), not in the alias code; the copy-50 run is now the cheap reproducer.
#   2 passes                                -> the hang is alias-specific; instrument deeper.
set -u
cd "$(dirname "$0")/../.."
NAT=/home/jhohertz/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/Q2_0
ARGS=(--pack packs/qwen38-flash-next-q2_0 --native $NAT/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
      --ple-gguf $NAT/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf
      --mmap-experts --expert-profile data/expert-profile.bin --prefill 512 --spec 4 --spec-min-p 0.5
      --max-context 4096 --kv int8 --pool-workers 8 --adapt-every 0 --pcie-frac 0 --vram-reserve-mib 1024
      --greedy --max-new 160)

echo "== gate $(date +%H:%M:%S)"
timeout 60 ./build-hip/hip_intrinsics 2>&1 | tail -1 | grep -q 'parity OK' || { echo "FAIL: gate"; exit 1; }

IDS=$(python3 -c "import json;print(json.load(open('/tmp/p0-out/smoke_ids.json'))['arithmetic'])")

echo "== 1) COPY path, --expert-cache 50 (borrow=nullptr), arithmetic  $(date +%H:%M:%S)"
env STRATA_IGPU_ALIAS=0 STRATA_TRACE=1 STRATA_HIPBLASLT_TUNING="$PWD/tools/hip/gfx1103-hipblaslt-100401.txt" \
  timeout 600 ./build-hip/strata "${ARGS[@]}" --expert-cache 50 --tokens "$IDS" \
  > /tmp/p1-out/copy50.out 2> /tmp/p1-out/copy50.err
RC=$?
grep -E 'own buffers|borrows' /tmp/p1-out/copy50.err | head -2
if [ $RC -eq 0 ]; then
  echo "PASS: copy50 (own-buffers prompt path works WITHOUT alias) -> the hang is alias-specific"
  grep -E 'decode +160|prefill +4[0-9]' /tmp/p1-out/copy50.out
  echo "== 2) engine alias arithmetic  $(date +%H:%M:%S)"
  env STRATA_IGPU_ALIAS=1 STRATA_TRACE=1 STRATA_HIPBLASLT_TUNING="$PWD/tools/hip/gfx1103-hipblaslt-100401.txt" \
    timeout 600 ./build-hip/strata "${ARGS[@]}" --expert-cache 6000 --tokens "$IDS" \
    > /tmp/p1-out/arit4.out 2> /tmp/p1-out/arit4.err
  RC2=$?
  if [ $RC2 -eq 0 ]; then
    echo "PASS: alias arithmetic"
    grep -E 'decode +160|prefill +4[0-9]' /tmp/p1-out/arit4.out
    grep -m1 'experts streamed' /tmp/p1-out/arit4.err
  else
    echo "FAIL: alias arithmetic (rc=$RC2) while copy50 passed - alias-specific; last lines:"
    tail -4 /tmp/p1-out/arit4.err | cut -c1-140
  fi
else
  echo "FAIL: copy50 (rc=$RC) - the own-buffers prompt path hangs WITHOUT alias: the alias hang is inherited"
  echo "last lines:"; tail -4 /tmp/p1-out/copy50.err | cut -c1-140
fi
