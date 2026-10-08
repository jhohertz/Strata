#!/usr/bin/env bash
source "$(dirname "$0")/gfx1103_env.sh"
# Boot protocol 8 (run 26): the expert-section IMA is a NULL-base read (dmesg 0x0-0x6000).
#   1) gate
#   2) micro --dequant 8: the engine's exact layer-0 dequant workload (512 experts x
#      iq_dequant_gu_f16 + iq_dequant_f16, Q2_0, real blob geometry) over a registered
#      8 GiB region after the engine's prefault.  On a CLEAN APU:
#        FAIL -> reproduced without the engine; stop and reboot (bisect the micro)
#        PASS -> the fault needs engine state; continue
#   3) engine alias arithmetic, MMQ OFF (run 25's failing config) + fine-grained checks
#      (dequant gu / dequant d / gu gemm / swiglu / dn gemm) + the first-compute pointer dump
set -u
cd "$(dirname "$0")/../.."
mkdir -p "$OUT"
ARGS=(--pack "$PACK" --native "$SHARD1"
      --ple-gguf "$SHARD2"
      --mmap-experts --expert-profile data/expert-profile.bin --expert-cache 6000 --prefill 512 --spec 4
      --spec-min-p 0.5 --max-context 4096 --kv int8 --pool-workers 8 --adapt-every 0 --pcie-frac 0
      --vram-reserve-mib 1024 --greedy --max-new 160)

echo "== 1) gate $(date +%H:%M:%S)"
timeout 60 "$BUILD/hip_intrinsics" 2>&1 | tail -1 | grep -q 'parity OK' || { echo "FAIL: gate"; exit 1; }

echo "== 2) micro --dequant 8  $(date +%H:%M:%S)"
timeout 600 "$BUILD/igpu_gtt_micro" --dequant 8 > "$OUT/micro-dequant.out" 2> "$OUT/micro-dequant.err"
RC=$?
tail -n 3 "$OUT"/micro-dequant.out | cut -c1-160; tail -n 3 "$OUT"/micro-dequant.err | cut -c1-160
if [ $RC -ne 0 ]; then
  echo "REPRODUCED: the dequant arm faults outside the engine (rc=$RC) - stop and reboot; bisect the micro"
  exit 1
fi

echo "== 3) engine alias arithmetic, MMQ OFF, fine checks  $(date +%H:%M:%S)"
IDS=$(python3 -c "import json;print(json.load(open('$OUT/smoke_ids.json'))['arithmetic'])")
env STRATA_PREFILL_MMQ=0 STRATA_IGPU_ALIAS=1 STRATA_TRACE=1 STRATA_GROUP_COPY=1 STRATA_PREFILL_CHECKS=1 \
    STRATA_HIPBLASLT_WARMUP=1 STRATA_HIPBLASLT_TUNING="$PWD/tools/hip/gfx1103-hipblaslt-100401.txt" \
  timeout 900 "$BUILD/strata" "${ARGS[@]}" --tokens "$IDS" > "$OUT/finer.out" 2> "$OUT/finer.err"
RC=$?
grep -E 'strata trace: compute|prefill: layer|VRAM at first|prefill +[0-9]+ tokens|decode +[0-9]+ tokens' "$OUT"/finer.err "$OUT"/finer.out 2>/dev/null | head -8
if [ $RC -eq 0 ]; then
  echo "PASS: alias arithmetic (MMQ OFF) with the fine checks - chain python/marker/longfill?"
else
  echo "FAIL (rc=$RC): the named check above says which kernel - stop and reboot"
  tail -3 "$OUT"/finer.err | cut -c1-160
fi
