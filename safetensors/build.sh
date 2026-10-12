#!/usr/bin/env bash
# Run safetensors: write a checkpoint, read it back, refuse broken files; then read the same
# file with Python's json and struct and compare.
#
#   ./build.sh                 run it
#   VX_HOME=<dir> ./build.sh   use a toolchain other than ~/.vx/current
#
# Needs Vx newer than v0.0.3 (built from main at 39a12e0a or later), and python3.

set -euo pipefail

cd "$(dirname "$0")"
VX_HOME="${VX_HOME:-$HOME/.vx/current}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

# The program returns the number of failed checks, and prints "ok" when there are none.
out="$("$VX_HOME/bin/vxc" -a run-jit src/main.vx -- "$tmp/model.safetensors" | grep -v '^\[JIT\]' || true)"
echo "$out"
if ! echo "$out" | grep -qx 'ok'; then
  echo "FAIL" >&2
  exit 1
fi

# The tensor lines are the ones with "]:".
mine="$(echo "$out" | grep ']:')"
theirs="$(python3 -I check.py "$tmp/model.safetensors" 2>&1 || true)"
if [ "$mine" != "$theirs" ]; then
  echo "FAIL: Python reads the file differently:" >&2
  echo "$theirs" >&2
  exit 1
fi
echo "ok: Python's json and struct read the same tensors"
