#!/usr/bin/env bash
source "$(dirname "$0")/gfx1103_env.sh"
# Boot protocol 3: the gather-pattern micro first (does the KFD path serve the engine's exact
# access shape on a registered region?), then the traced engine.
set -u
cd "$(dirname "$0")/../.."
echo "== gate $(date +%H:%M:%S)"
timeout 60 "$BUILD/hip_intrinsics" 2>&1 | tail -1 | grep -q 'parity OK' || { echo "FAIL: gate"; exit 1; }
echo "== 1) micro --scatter 32 (24,576 CTAs x 1.38 MiB blobs on a registered 32 GiB)  $(date +%H:%M:%S)"
timeout 600 "$BUILD/igpu_gtt_micro" --scatter 32 2>&1 | tail -3
RC=${PIPESTATUS[0]}
if [ $RC -ne 0 ]; then
  echo "FAIL: scatter (rc=$RC) - the KFD path cannot serve the gather pattern on a registered region; the alias design must bounce the prompt path (decode-only aliasing). Stop and reboot."
  exit 1
fi
echo "== 2) engine alias arithmetic, traced  $(date +%H:%M:%S)"
IDS=$(python3 -c "import json;print(json.load(open('$OUT/smoke_ids.json'))['arithmetic'])")
env STRATA_IGPU_ALIAS=1 STRATA_TRACE=1 STRATA_HIPBLASLT_TUNING="$PWD/tools/hip/gfx1103-hipblaslt-100401.txt" \
  timeout 600 "$BUILD/strata" --pack "$PACK" \
  --native "$SHARD1" \
  --ple-gguf "$SHARD2" \
  --mmap-experts --expert-profile data/expert-profile.bin --expert-cache 6000 --prefill 512 --spec 4 \
  --spec-min-p 0.5 --max-context 4096 --kv int8 --pool-workers 8 --adapt-every 0 --pcie-frac 0 \
  --vram-reserve-mib 1024 --greedy --max-new 160 --tokens "$IDS" \
  > "$OUT/arit5.out" 2> "$OUT/arit5.err"
RC=$?
if [ $RC -eq 0 ]; then
  echo "PASS: alias arithmetic"
  grep -E 'decode +160|prefill +4[0-9]' "$OUT"/arit5.out
  grep -m1 'experts streamed' "$OUT"/arit5.err
else
  echo "FAIL: alias arithmetic (rc=$RC) - the last trace lines:"
  grep -E 'strata trace' "$OUT"/arit5.err | tail -4
fi
