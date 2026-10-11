#!/usr/bin/env bash
# Run bigint's tests.
#
#   ./build.sh                 run the tests
#   VX_HOME=<dir> ./build.sh   use a toolchain other than ~/.vx/current
#
# bigint is imported as `bigint::...`, so the repository root is on the import path, as in
# ../textkit/build.sh. The tests print three numbers; the values below are from Python. Then
# stress.vx checks that division does not leak.

set -euo pipefail

cd "$(dirname "$0")"
VX_HOME="${VX_HOME:-$HOME/.vx/current}"
export VX_STD_PATH="$VX_HOME/stdlib/std:$VX_HOME/stdlib:.."

want="265252859812191058636308480000000
-118181386580595879976868414312001964434038548836769923458287039207
1000000000000000000000000000000000000000000007
ok"
# A failed check makes the program, and so the pipeline, exit non-zero; keep going so the
# output is shown.
got="$("$VX_HOME/bin/vxc" -a run-jit test.vx | grep -v '^\[JIT\]' || true)"
if [ "$got" != "$want" ]; then
  echo "$got" >&2
  echo "FAIL: want" >&2
  echo "$want" >&2
  exit 1
fi

# Memory: build stress.vx as a program and check its peak size. It runs under a 1 GB limit,
# so a leak stops it rather than the machine.
CLANG="$VX_HOME/llvm/bin/clang"
[ -x "$CLANG" ] || CLANG=clang
mkdir -p build
"$VX_HOME/bin/vxc" -O3 -a emit-obj stress.vx -o build/stress.o
"$CLANG" build/stress.o -L"$VX_HOME/lib" -lvx_std_core -Wl,-rpath,"$VX_HOME/lib" -lm \
  -o build/stress
peak_kb="$( (ulimit -v 1000000; /usr/bin/time -f '%M' build/stress > /dev/null) 2>&1 | tail -1)"
if ! [ "$peak_kb" -lt 20000 ] 2>/dev/null; then
  echo "FAIL: stress.vx needed $peak_kb KB, want under 20000" >&2
  exit 1
fi
echo ok
