#!/usr/bin/env bash
# Run lr-schedule: warmup and cosine decay, clipping by global norm, gradient accumulation
# checked with `grad`, and one bad label that clipping survives.
#
#   ./build.sh                 run it (about 1 s)
#   VX_HOME=<dir> ./build.sh   use a toolchain other than ~/.vx/current
#
# Needs Vx newer than v0.0.3 (built from main at 39a12e0a or later). A program that uses `grad`
# runs only under `vxc -a run-jit` (vx-lang/Vx#1594), so there is no binary.

set -euo pipefail

cd "$(dirname "$0")"
VX_HOME="${VX_HOME:-$HOME/.vx/current}"

# The program returns the number of failed checks, and prints "ok" when there are none.
out="$("$VX_HOME/bin/vxc" -a run-jit src/main.vx | grep -v '^\[JIT\]' || true)"
echo "$out"
if ! echo "$out" | grep -qx 'ok'; then
  echo "FAIL" >&2
  exit 1
fi
