#!/usr/bin/env bash
# Boot protocol 3: the gather-pattern micro first (does the KFD path serve the engine's exact
# access shape on a registered region?), then the traced engine.
set -u
cd "$(dirname "$0")/../.."
NAT=/home/jhohertz/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/Q2_0
echo "== gate $(date +%H:%M:%S)"
timeout 60 ./build-hip/hip_intrinsics 2>&1 | tail -1 | grep -q 'parity OK' || { echo "FAIL: gate"; exit 1; }
echo "== 1) micro --scatter 32 (24,576 CTAs x 1.38 MiB blobs on a registered 32 GiB)  $(date +%H:%M:%S)"
timeout 600 ./build-hip/igpu_gtt_micro --scatter 32 2>&1 | tail -3
RC=${PIPESTATUS[0]}
if [ $RC -ne 0 ]; then
  echo "FAIL: scatter (rc=$RC) - the KFD path cannot serve the gather pattern on a registered region; the alias design must bounce the prompt path (decode-only aliasing). Stop and reboot."
  exit 1
fi
echo "== 2) engine alias arithmetic, traced  $(date +%H:%M:%S)"
IDS=$(python3 -c "import json;print(json.load(open('/tmp/p0-out/smoke_ids.json'))['arithmetic'])")
env STRATA_IGPU_ALIAS=1 STRATA_TRACE=1 STRATA_HIPBLASLT_TUNING="$PWD/tools/hip/gfx1103-hipblaslt-100401.txt" \
  timeout 600 ./build-hip/strata --pack packs/qwen38-flash-next-q2_0 \
  --native $NAT/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
  --ple-gguf $NAT/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf \
  --mmap-experts --expert-profile data/expert-profile.bin --expert-cache 6000 --prefill 512 --spec 4 \
  --spec-min-p 0.5 --max-context 4096 --kv int8 --pool-workers 8 --adapt-every 0 --pcie-frac 0 \
  --vram-reserve-mib 1024 --greedy --max-new 160 --tokens "$IDS" \
  > /tmp/p1-out/arit5.out 2> /tmp/p1-out/arit5.err
RC=$?
if [ $RC -eq 0 ]; then
  echo "PASS: alias arithmetic"
  grep -E 'decode +160|prefill +4[0-9]' /tmp/p1-out/arit5.out
  grep -m1 'experts streamed' /tmp/p1-out/arit5.err
else
  echo "FAIL: alias arithmetic (rc=$RC) - the last trace lines:"
  grep -E 'strata trace' /tmp/p1-out/arit5.err | tail -4
fi
