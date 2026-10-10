#!/usr/bin/env bash
# Run rational's tests.
#
#   ./build.sh                 run the tests
#   VX_HOME=<dir> ./build.sh   use a toolchain other than ~/.vx/current
#
# rational imports bigint, and both are found from the repository root, as in
# ../textkit/build.sh. The tests print three fractions; the values below are from Python's
# fractions.Fraction.

set -euo pipefail

cd "$(dirname "$0")"
VX_HOME="${VX_HOME:-$HOME/.vx/current}"
export VX_STD_PATH="$VX_HOME/stdlib/std:$VX_HOME/stdlib:.."

want="55835135/15519504
-243/32
-13717421001371742100137174210/109739369
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
