#!/usr/bin/env bash
# Run quantize: int8 and int4 weight quantization, per tensor and per group, and an int8 matmul.
#
#   ./build.sh                 run it
#   VX_HOME=<dir> ./build.sh   use a toolchain other than ~/.vx/current
#
# Needs Vx newer than v0.0.3 (built from main at 39a12e0a or later).

set -euo pipefail

cd "$(dirname "$0")"
VX_HOME="${VX_HOME:-$HOME/.vx/current}"

# The contracts are proved while compiling; a broken one stops the run. The program returns the
# number of failed checks, and prints "ok" when there are none.
out="$("$VX_HOME/bin/vxc" -a run-jit src/main.vx | grep -v '^\[JIT\]' || true)"
echo "$out"
if ! echo "$out" | grep -qx 'ok'; then
  echo "FAIL" >&2
  exit 1
fi
