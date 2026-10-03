#!/usr/bin/env bash
# Boot-N protocol for the P1 alias run.  Order: cheap micro A/Bs first (a fault costs the boot but
# answers a narrow question), the engine last (its pass is the P1 verdict and it may chain into the
# rest of the smoke on the same boot - a clean run does not degrade the APU).
#
#   tools/hip/p1_boot_run.sh
set -u
cd "$(dirname "$0")/../.."
M=./build-hip/igpu_gtt_micro
pass() { echo "PASS: $1"; }
fail() { echo "FAIL: $1 (rc=$2) - APU now degraded, stop and reboot"; exit 1; }

echo "== gate $(date +%H:%M:%S)"
timeout 60 ./build-hip/hip_intrinsics 2>&1 | tail -1 | grep -q 'parity OK' || fail "gate" 1

echo "== 1) --reg-mapped 8  (the EXACT flags the first engine run registered with)"
timeout 300 $M --reg-mapped 8 2>&1 | tail -2
[ ${PIPESTATUS[0]} -eq 0 ] && pass "reg-mapped" || fail "reg-mapped 8 (Mapped|Portable register broken on this KFD build?)" 2

echo "== 2) --reg-after-dma 8  (register placed after the process's first DMA, like the engine)"
timeout 300 $M --reg-after-dma 8 2>&1 | tail -2
[ ${PIPESTATUS[0]} -eq 0 ] && pass "reg-after-dma" || fail "reg-after-dma 8 (late register after prior DMA is broken - the engine must register before first GPU traffic or use the H2D touch)" 2

echo "== 3) engine alias arithmetic  $(date +%H:%M:%S)"
IDS=$(python3 -c "import json;print(json.load(open('/tmp/p0-out/smoke_ids.json'))['arithmetic'])")
env STRATA_IGPU_ALIAS=1 STRATA_HIPBLASLT_TUNING="$PWD/tools/hip/gfx1103-hipblaslt-100401.txt" \
  timeout 600 ./build-hip/strata --pack packs/qwen38-flash-next-q2_0 \
  --native /home/jhohertz/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
  --ple-gguf /home/jhohertz/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf \
  --mmap-experts --expert-profile data/expert-profile.bin --expert-cache 6000 --prefill 512 --spec 4 \
  --spec-min-p 0.5 --max-context 4096 --kv int8 --pool-workers 8 --adapt-every 0 --pcie-frac 0 \
  --vram-reserve-mib 1024 --greedy --max-new 160 --tokens "$IDS" \
  > /tmp/p1-out/arit3.out 2> /tmp/p1-out/arit3.err
RC=$?
if [ $RC -eq 0 ]; then
  pass "engine alias arithmetic"
  echo "---- result ----"
  grep -E 'decode +160|prefill +4[0-9]' /tmp/p1-out/arit3.out
  grep -m1 'experts streamed' /tmp/p1-out/arit3.err
  grep -o '396' /tmp/p1-out/arit3.out | head -1
  echo "== 4) engine alias python  (same boot)"
  IDS=$(python3 -c "import json;print(json.load(open('/tmp/p0-out/smoke_ids.json'))['python'])")
  env STRATA_IGPU_ALIAS=1 STRATA_HIPBLASLT_TUNING="$PWD/tools/hip/gfx1103-hipblaslt-100401.txt" \
    timeout 600 ./build-hip/strata --pack packs/qwen38-flash-next-q2_0 \
    --native /home/jhohertz/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
    --ple-gguf /home/jhohertz/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf \
    --mmap-experts --expert-profile data/expert-profile.bin --expert-cache 6000 --prefill 512 --spec 4 \
    --spec-min-p 0.5 --max-context 4096 --kv int8 --pool-workers 8 --adapt-every 0 --pcie-frac 0 \
    --vram-reserve-mib 1024 --greedy --max-new 160 --tokens "$IDS" \
    > /tmp/p1-out/python3.out 2> /tmp/p1-out/python3.err
  RC2=$?
  if [ $RC2 -eq 0 ]; then
    pass "engine alias python"
    grep -E 'decode +160|prefill +5[0-9]' /tmp/p1-out/python3.out
    grep -m1 'sum' /tmp/p1-out/python3.out | head -1
  else
    fail "engine alias python (rc=$RC2)" $RC2
  fi
else
  fail "engine alias arithmetic (rc=$RC)" $RC
fi
