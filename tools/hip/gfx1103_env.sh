#!/usr/bin/env bash
# Shared settings for the gfx1103 reference scripts in tools/hip (docs/GFX1103.md).
#
# Nothing here is a fixed path on any particular machine: set the variables in the
# environment before running, or edit the defaults below.  SHARD1 and SHARD2 have no
# default because they name the model files, which differ per install.
#
#   BUILD   the build directory holding the engine            (default: build-hip)
#   PACK    the IQ pack directory                             (default: packs/qwen38-flash-next-q2_0)
#   SHARD1  the model shard the engine reads experts from     (required)
#   SHARD2  the shard holding per_layer_token_embd.weight     (required)
#   OUT     scratch directory for the run's output            (default: /tmp/gfx1103-out)
#   SDK_LIBS optional directory (or colon list) of ROCm runtime libraries to run against -
#           set it when the engine runs on setup's wheels instead of a system ROCm install

: "${BUILD:=build-hip}"
: "${PACK:=packs/qwen38-flash-next-q2_0}"
: "${OUT:=/tmp/gfx1103-out}"
if [ -n "${SDK_LIBS:-}" ]; then export LD_LIBRARY_PATH="$SDK_LIBS:${LD_LIBRARY_PATH:-}"; fi

if [ -z "${SHARD1:-}" ] || [ -z "${SHARD2:-}" ]; then
    echo "set SHARD1 and SHARD2 to the two model shards (docs/GFX1103.md, section 6)" >&2
    exit 1
fi

mkdir -p "$OUT"
