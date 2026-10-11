#!/usr/bin/env bash
# Build cycle-clique into build/cycle-clique, run it, and compare its table with the one the
# paper ships (verification/data/expected-output.txt in the paper's directory).
#
#   ./build.sh                 build, run and compare
#   VX_HOME=<dir> ./build.sh   use a toolchain other than ~/.vx/current
#
# Linking is done here, as in ../wordcount/build.sh. See ../FRICTION.md.

set -euo pipefail

cd "$(dirname "$0")"
VX_HOME="${VX_HOME:-$HOME/.vx/current}"
VXC="$VX_HOME/bin/vxc"
CLANG="$VX_HOME/llvm/bin/clang"
[ -x "$CLANG" ] || CLANG=clang

mkdir -p build
"$VXC" -O3 -a emit-obj src/main.vx -o build/main.o
"$CLANG" build/main.o -L"$VX_HOME/lib" -lvx_std_core -Wl,-rpath,"$VX_HOME/lib" -lm \
  -o build/cycle-clique

# From the paper. The first number after k is the count of patterns not ruled out; the proof
# needs it to be 0.
want="5 0 4 3
6 0 8 6
7 0 21 5
8 0 30 4
9 0 57 6
10 0 71 4
11 0 134 8
12 0 161 3
13 0 305 6
14 0 315 0
15 0 597 4
16 0 468 0
17 0 878 1"
got="$(build/cycle-clique)"
if [ "$got" != "$want" ]; then
  echo "FAIL: cycle-clique printed" >&2
  echo "$got" >&2
  echo "the paper's table is" >&2
  echo "$want" >&2
  exit 1
fi
echo "ok: the same table as the paper"
