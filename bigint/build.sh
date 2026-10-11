#!/usr/bin/env bash
# Run bigint's tests.
#
#   ./build.sh                 run the tests
#   VX_HOME=<dir> ./build.sh   use a toolchain other than ~/.vx/current
#
# bigint is imported as `bigint::...`, so the repository root is on the import path, as in
# ../textkit/build.sh. The tests print three numbers; the values below are from Python.

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
echo ok
