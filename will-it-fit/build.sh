#!/usr/bin/env bash
# Run will-it-fit: predict by hand whether each program fits each GPU, then ask the compiler.
#
#   ./build.sh                 run it
#   VX_HOME=<dir> ./build.sh   use a toolchain other than ~/.vx/current
#
# Needs Vx newer than v0.0.3 (built from main at 39a12e0a or later).
#
# The machine is a build input: the same program is compiled once per machine file, and the
# verdict can differ. The machine files come with the toolchain, in $VX_HOME/fleet.

set -euo pipefail

cd "$(dirname "$0")"
VX_HOME="${VX_HOME:-$HOME/.vx/current}"
VXC="$VX_HOME/bin/vxc"
FLEET="$VX_HOME/fleet"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

# 1. The arithmetic, at run time. It ends with one line per (program, machine):
#    "<program> <machine> admitted" or "<program> <machine> rejected <code>".
out="$("$VXC" -a run-jit src/sizes.vx | grep -v '^\[JIT\]')"
echo "$out"

# int4 with one more sequence in the batch: the same program with BATCH = 33.
sed 's/admit<32, 32, 8, 128, 8192, 32>()/admit<32, 32, 8, 128, 8192, 33>()/' src/int4.vx > "$tmp/int4-33.vx"
grep -q 'admit<32, 32, 8, 128, 8192, 33>()' "$tmp/int4-33.vx"

# 2. The compiler's verdict for each line. `print-ast` runs every check and stops before code
#    generation (code for fp8 tensors is not written yet); the verdict is read from the
#    diagnostics record.
echo
failed=0
while read -r prog machine verdict code; do
  src="src/$prog.vx"
  if [ "$prog" = "int4-33" ]; then
    src="$tmp/int4-33.vx"
  fi
  # vxc reads stdin, which here is the list being looped over.
  "$VXC" --machine "$FLEET/$machine.vx" --host "$FLEET/host-x86-e5-2666v3.vx" -a print-ast \
    --diagnostics-json "$tmp/verdict.json" "$src" < /dev/null > "$tmp/log" 2>&1 || true
  got="$(grep -o '"verdict": "[a-z]*"' "$tmp/verdict.json" | cut -d'"' -f4)"
  # An admitted program has no codes, and grep then fails; that is not an error here.
  codes="$( (grep -o '"code": "[EW][0-9]*"' "$tmp/verdict.json" || true) | cut -d'"' -f4 | tr '\n' ' ' | sed 's/ $//')"
  printf 'compiler: %-8s on %-9s %s %s\n' "$prog" "$machine" "$got" "$codes"
  if [ "$got" != "$verdict" ] || [ "$codes" != "${code:-}" ]; then
    echo "FAIL: expected $verdict ${code:-}" >&2
    grep -E 'Error|Warning' "$tmp/log" >&2 || true
    failed=$((failed + 1))
  fi
done < <(echo "$out" | grep -E ' (admitted|rejected)')

if [ "$failed" -ne 0 ]; then
  echo "FAIL: $failed verdicts differ" >&2
  exit 1
fi
echo "ok"
