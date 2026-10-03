#!/usr/bin/env bash
# Boot protocol 4 (the localizing boot):
#   1) gate
#   2) SKIP: scatter arm (passed twice - the gate is the APU check)
#      32 GiB.  Fault/hang here = the KFD path cannot serve that pattern -> decode-only aliasing.
#   3) the traced engine under gdb (gdb is the parent, so ptrace_scope=1 allows the attach): if it
#      hangs, gdb interrupts it and dumps every thread's backtrace; if it runs to completion, no
#      interrupt is sent and the boot is preserved for a chained smoke.
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

echo "== 2) engine alias arithmetic under gdb, traced  $(date +%H:%M:%S)"
IDS=$(python3 -c "import json;print(json.load(open('/tmp/p0-out/smoke_ids.json'))['arithmetic'])")
cat > /tmp/p1-envwrap.sh << 'W'
#!/usr/bin/env bash
export STRATA_IGPU_ALIAS=1 STRATA_TRACE=1 STRATA_HIPBLASLT_WARMUP=1 STRATA_HIPBLASLT_TUNING="$STRATA_T"
exec /home/jhohertz/co/Strata/build-hip/strata "$@"
W
chmod +x /tmp/p1-envwrap.sh
rm -f /tmp/gdbfifo; mkfifo /tmp/gdbfifo
OUT=/tmp/p1-out/gdb.out
# interactive gdb (NOT -batch: in batch mode a hung `run` never reads the interrupt off stdin).
# timeout is the backstop: it signals the whole process group, so a wedged gdb cannot hold the script.
env STRATA_T="$PWD/tools/hip/gfx1103-hipblaslt-100401.txt" \
  timeout 900 gdb -q -nx -ex 'set debuginfod enabled off' -ex 'set pagination off' -ex 'set confirm off' -ex run \
     --args bash /tmp/p1-envwrap.sh "${ARGS[@]}" --tokens "$IDS" \
  < /tmp/gdbfifo > "$OUT" 2>&1 &
GDBPID=$!
(
  sleep 30
  # GDBPID is the `timeout` process; its child is gdb; gdb's child is the engine (the wrapper execs in place)
  GDBREAL=$(pgrep -P "$GDBPID" | head -1)
  EPID=$(pgrep -P "$GDBREAL" | head -1)
  hung=0
  for _ in $(seq 1 15); do                 # up to ~3 min of hang before we stop it
    kill -0 "$EPID" 2>/dev/null || { echo "engine finished early - no stop" >&2; break; }
    sleep 12
  done
  if kill -0 "$EPID" 2>/dev/null; then
    echo "hung (engine pid $EPID) - SIGSTOP, then gdb backtrace, then kill" >&2
    kill -STOP "$EPID"
    sleep 12
    echo "set pagination off"
    echo "thread apply all bt 12"
    sleep 30
    echo "info threads"
    sleep 8
    echo "kill"
  fi
  sleep 8
) > /tmp/gdbfifo &
FEEDPID=$!
wait "$GDBPID" 2>/dev/null
GDBRC=$?
kill "$FEEDPID" 2>/dev/null; wait "$FEEDPID" 2>/dev/null
echo "(gdb exit: $GDBRC; 124 = the 900 s backstop fired)"
echo "---- engine result lines ----"
grep -E 'decode +160|prefill +4[0-9]|illegal|unspecified|FAIL' "$OUT" | head -4
echo "---- last trace lines ----"
grep -E 'strata trace' "$OUT" | tail -5
if ! grep -qE 'decode +160 tokens' "$OUT"; then
  echo "---- backtrace (if hung) ----"
  awk '/thread apply all bt/,0' "$OUT" | grep -E '^Thread|^#[0-9]+' | head -40
  echo "gdb log: $OUT"
  exit 1
fi
echo "PASS: alias arithmetic"

# A clean run does not degrade the APU: chain the rest of the smoke on this boot.
echo "== 3) python, marker, longfill (same boot)  $(date +%H:%M:%S)"
for CASE in python marker longfill; do
  IDS=$(python3 -c "import json;print(json.load(open('/tmp/p0-out/smoke_ids.json'))['$CASE'])")
  env STRATA_IGPU_ALIAS=1 STRATA_HIPBLASLT_WARMUP=1 STRATA_HIPBLASLT_TUNING="$PWD/tools/hip/gfx1103-hipblaslt-100401.txt" \
    timeout 900 ./build-hip/strata "${ARGS[@]}" --tokens "$IDS" > /tmp/p1-out/$CASE.out 2> /tmp/p1-out/$CASE.err
  RC2=$?
  if [ $RC2 -eq 0 ]; then
    echo "PASS: $CASE"
    grep -E 'decode +[0-9]+ tokens|prefill +[0-9]+ tokens' /tmp/p1-out/$CASE.out | head -2
  else
    echo "FAIL: $CASE (rc=$RC2) - APU may be degraded, stop and reboot"; tail -3 /tmp/p1-out/$CASE.err | cut -c1-140
  fi
done
