#!/usr/bin/env bash
# Run tiny-mlp: check training derivatives against `grad`, then train a small network on XOR.
#
#   ./build.sh                 run it (about 1 s)
#   VX_HOME=<dir> ./build.sh   use a toolchain other than ~/.vx/current
#
# A program that uses `grad` runs only under `vxc -a run-jit` in v0.0.3 (vx-lang/Vx#1594), so
# there is no binary to build. Enzyme, which `grad` needs, comes with the installed toolchain.

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
