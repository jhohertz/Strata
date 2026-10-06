#!/bin/bash
# igpu-rework: bring up the OpenAI/Anthropic server on the 8700G/780M box.
#
# Protocol (post-BIOS, docs/IGPU.md P2.14): run unless there is EVIDENCE of a bad driver
# state - not "only on a fresh boot".  The gate is the arithmetic canary plus the journal
# fault lines; the BIOS changeset (P2.11) extended the boot budget to 8-11+ clean
# userptr lifecycles and the canary catches the silent-corruption stage before it matters.
# Steps: boot health check -> arithmetic canary -> start the server (nohup, never killed by
# this script) -> wait for READY -> one API request.
#
# If anything faults: LEAVE THE PROCESS ALIVE (no kill, no timeout on the server), then an
# AC-cut power cycle (unplug/PSU + hold power ~10 s).
set -u
cd /home/jhohertz/co/Strata
MODEL_DIR=/home/jhohertz/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF
export STRATA_IGPU_ALIAS=1 STRATA_GROUP_COPY=1 STRATA_SSD_KEEPALIVE=0
export STRATA_HIPBLASLT_WARMUP=1 STRATA_HIPBLASLT_TUNING="$PWD/tools/hip/gfx1103-hipblaslt-100401.txt"
LOG=strata-igpu-serve.log
PORT=${1:-8080}

echo "=== boot health ==="
uptime
echo "restore/MES-failed/evict-failed lines this boot: $(journalctl -b --no-pager 2>/dev/null | grep -cE 'restore_userptr_worker|MES failed|Failed to evict')"

if [ ! -f /tmp/p2-out/smoke_ids.json ]; then
    mkdir -p /tmp/p2-out
    python3 tools/hip/gfx1103_smoke.py prep packs/qwen38-flash-next-q2_0 /tmp/p2-out || exit 1
fi

echo "=== arithmetic canary (short generate run) ==="
IDS=$(python3 -c "import json; print(json.load(open('/tmp/p2-out/smoke_ids.json'))['arithmetic'])")
./build-hip/strata --pack packs/qwen38-flash-next-q2_0 \
  --native "$MODEL_DIR/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf" \
  --ple-gguf "$MODEL_DIR/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf" \
  --mmap-experts --expert-profile data/expert-profile.bin --expert-cache 6000 --prefill 512 --spec 4 \
  --spec-min-p 0.5 --max-context 4096 --kv int8 --pool-workers 8 --adapt-every 0 --pcie-frac 0 \
  --vram-reserve-mib 1024 --greedy --max-new 160 --tokens "$IDS" > /tmp/p2-out/serve-canary.out 2> /tmp/p2-out/serve-canary.err \
  || { echo "canary engine died"; grep -E 'error|fault' /tmp/p2-out/serve-canary.err | tail -3; exit 1; }
if ! python3 tools/hip/gfx1103_smoke.py check packs/qwen38-flash-next-q2_0 /tmp/p2-out/serve-canary.out arithmetic | head -1 | grep -q PASS; then
    echo "CANARY FAILED - do not start the server"; exit 1
fi
echo "canary PASS"

echo "=== starting the server (nohup; this script never kills it) ==="
: > "$LOG"
nohup python3 serve/server.py --engine strata --config strata-igpu-serve.json --port "$PORT" >> "$LOG" 2>&1 &
SRV=$!
echo "server pid $SRV"

echo "=== waiting for READY (up to 15 min; the first load is the 31.64 GiB complement + MTP draft layer) ==="
for i in $(seq 1 180); do
    sleep 5
    if ! kill -0 "$SRV" 2>/dev/null; then echo "SERVER DIED"; tail -15 "$LOG"; exit 1; fi
    grep -qE 'unspecified launch failure|illegal memory|error: ' "$LOG" && { echo "FAULT in the engine (leaving it alive)"; grep -E 'fault|error' "$LOG" | tail -4; exit 1; }
    if curl -s -m 2 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q qwen38; then echo "READY after ~$((i*5)) s"; break; fi
    [ "$i" -eq 180 ] && { echo "not READY in 15 min"; tail -15 "$LOG"; exit 1; }
done

echo "=== one API request (arithmetic) ==="
curl -s -m 600 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"qwen38-flash-next-q2_0","messages":[{"role":"user","content":"What is 17*23+5? Answer with the number only."}],"max_tokens":160}' \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'][:200])"
echo
echo "=== server is up on 127.0.0.1:$PORT (no API key: this PC only) ==="
echo "log: $LOG"
