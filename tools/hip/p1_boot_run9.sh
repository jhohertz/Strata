#!/usr/bin/env bash
# Boot protocol 9 (run 27): the IMA is in iq_dequant_gu_f16, and the identical micro workload
# (legacy stream) PASSES.  The only untested engine difference: every complement read so far has
# been on the LEGACY stream - the engine's dequant is the first on a NON-BLOCKING stream.
#   1) gate
#   2) micro --dequant 8          (legacy stream, reconfirm the pass)
#   3) micro --dequant 8 nb       (NON-BLOCKING stream - the hypothesis)
#      FAIL -> confirmed; next boot runs the engine with STRATA_IGPU_BLOCKING_CS=1
#   4) micro --dequant 8 nb 6     (nb stream + 6 GiB of held device memory, the engine's VRAM state)
#   5) engine alias arithmetic, MMQ OFF, STRATA_IGPU_BLOCKING_CS=1 + fine checks
#      PASS -> chain python/marker/longfill (P1 works with the blocking CS)
set -u
cd "$(dirname "$0")/../.."
mkdir -p /tmp/p1-out
NAT=/home/jhohertz/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/Q2_0
ARGS=(--pack packs/qwen38-flash-next-q2_0 --native $NAT/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
      --ple-gguf $NAT/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf
      --mmap-experts --expert-profile data/expert-profile.bin --expert-cache 6000 --prefill 512 --spec 4
      --spec-min-p 0.5 --max-context 4096 --kv int8 --pool-workers 8 --adapt-every 0 --pcie-frac 0
      --vram-reserve-mib 1024 --greedy --max-new 160)
run_micro() {  # $1 = tag, rest = args after --dequant
  local tag=$1; shift
  timeout 600 ./build-hip/igpu_gtt_micro --dequant "$@" > /tmp/p1-out/$tag.out 2> /tmp/p1-out/$tag.err
  local rc=$?
  tail -n 2 /tmp/p1-out/$tag.out | cut -c1-160; tail -n 2 /tmp/p1-out/$tag.err | cut -c1-160
  return $rc
}
run_engine() {  # $1 = tag, $2.. = extra env
  local tag=$1; shift
  local IDS; IDS=$(python3 -c "import json;print(json.load(open('/tmp/p0-out/smoke_ids.json'))['arithmetic'])")
  env "$@" STRATA_IGPU_ALIAS=1 STRATA_TRACE=1 STRATA_GROUP_COPY=1 STRATA_PREFILL_CHECKS=1 \
      STRATA_PREFILL_MMQ=0 STRATA_HIPBLASLT_WARMUP=1 STRATA_HIPBLASLT_TUNING="$PWD/tools/hip/gfx1103-hipblaslt-100401.txt" \
    timeout 900 ./build-hip/strata "${ARGS[@]}" --tokens "$IDS" > /tmp/p1-out/$tag.out 2> /tmp/p1-out/$tag.err
  return $?
}

echo "== 1) gate $(date +%H:%M:%S)"
timeout 60 ./build-hip/hip_intrinsics 2>&1 | tail -1 | grep -q 'parity OK' || { echo "FAIL: gate"; exit 1; }

echo "== 2) micro dequant, legacy stream  $(date +%H:%M:%S)"
run_micro dq-legacy 8 || { echo "UNEXPECTED: the legacy dequant now faults (degraded?); stop and reboot"; exit 1; }

echo "== 3) micro dequant, NON-BLOCKING stream  $(date +%H:%M:%S)"
if run_micro dq-nb 8 nb; then
  echo "PASS: nb stream dequant - the stream hypothesis is dead"
  echo "== 4) micro dequant, nb + 6 GiB pressure  $(date +%H:%M:%S)"
  run_micro dq-nb6 8 nb 6 || { echo "the pressure arm faults - the engine's VRAM state matters; stop and reboot"; exit 1; }
else
  echo "CONFIRMED: the non-blocking stream reproduces the IMA outside the engine - stop and reboot"
  echo "next boot: engine alias arithmetic with STRATA_IGPU_BLOCKING_CS=1"
  exit 1
fi

echo "== 5) engine alias arithmetic, BLOCKING CS  $(date +%H:%M:%S)"
if run_engine bcs STRATA_IGPU_BLOCKING_CS=1; then
  grep -E 'prefill +[0-9]+ tokens|decode +[0-9]+ tokens' /tmp/p1-out/bcs.out
  echo "PASS: alias arithmetic with the blocking compute stream - chain python/marker/longfill"
else
  echo "FAIL: the engine still faults with the blocking CS - the named check says where; stop and reboot"
  grep -E 'prefill: layer|strata trace: compute' /tmp/p1-out/bcs.err | head -4 | cut -c1-200
fi
