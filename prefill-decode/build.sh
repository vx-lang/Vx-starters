#!/usr/bin/env bash
# Run prefill-decode: prefill on a GPU region, ship the KV cache, decode on an NPU region, and
# check the tokens against the host and against Python. Then, if z3 is installed, prove the
# transfers' seam obligations and check that a relaxed transfer is refused.
#
#   ./build.sh                 run it
#   VX_HOME=<dir> ./build.sh   use a toolchain other than ~/.vx/current
#
# Needs Vx newer than v0.0.3 (built from main at 39a12e0a or later) and python3. z3 is optional.

set -euo pipefail

cd "$(dirname "$0")"
VX_HOME="${VX_HOME:-$HOME/.vx/current}"
VXC="$VX_HOME/bin/vxc"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

# 1. Run it. The program returns the number of failed checks, and prints "ok" when there are
#    none.
out="$("$VXC" -a run-jit src/main.vx | grep -v '^\[JIT\]' || true)"
echo "$out"
if ! echo "$out" | grep -qx 'ok'; then
  echo "FAIL" >&2
  exit 1
fi

# 2. The same model in Python.
mine="$(echo "$out" | grep '^GPU + NPU:' | sed 's/^GPU + NPU: *//')"
theirs="$(python3 -I check.py 2>&1 || true)"
if [ "$mine" != "$theirs" ]; then
  echo "FAIL: Python decodes \"$theirs\"" >&2
  exit 1
fi
echo "ok: Python decodes the same tokens"

# 3. The seams, which need z3. Without it, --verify-seams fails every compile (E6024), so the
#    step is skipped rather than failed.
if ! command -v z3 > /dev/null; then
  echo "skipped: seam verification needs z3 on PATH"
  exit 0
fi
"$VXC" --verify-seams -a print-ast src/main.vx > "$tmp/log" 2>&1 || {
  cat "$tmp/log" >&2
  echo "FAIL: the seams of src/main.vx are not proved" >&2
  exit 1
}
grep '^\[seam\]' "$tmp/log" | sed 's/: solver.*//'
# The same program with the host-to-NPU copy relaxed: no synchronizing release, so the decoder
# could read a stale cache. It must be refused.
sed 's/staged.to_device();/staged.to_device_relaxed();/' src/main.vx > "$tmp/relaxed.vx"
grep -q 'to_device_relaxed' "$tmp/relaxed.vx"
if "$VXC" --verify-seams -a print-ast "$tmp/relaxed.vx" > "$tmp/log" 2>&1; then
  echo "FAIL: a relaxed transfer into NPU memory was accepted" >&2
  exit 1
fi
if ! grep -q 'E6004' "$tmp/log"; then
  cat "$tmp/log" >&2
  echo "FAIL: the relaxed transfer was refused, but not with E6004" >&2
  exit 1
fi
grep -o 'E6004.*seam' "$tmp/log"
echo "ok: seams proved, and the relaxed copy refused"
