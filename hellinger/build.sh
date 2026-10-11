#!/usr/bin/env bash
# Build hellinger into build/hellinger, run it, and check it proves every box, visiting the
# same number of boxes as the paper's C++ checker.
#
#   ./build.sh                 build, run (about 30 s) and compare
#   VX_HOME=<dir> ./build.sh   use a toolchain other than ~/.vx/current
#
# `vxc -O1` and above take minutes on this program (vx-lang/Vx#1589), so this asks vxc for
# unoptimized LLVM, and the bundled clang optimizes it. See ../FRICTION.md.

set -euo pipefail

cd "$(dirname "$0")"
VX_HOME="${VX_HOME:-$HOME/.vx/current}"
VXC="$VX_HOME/bin/vxc"
LLVM="$VX_HOME/llvm/bin"

mkdir -p build
"$VXC" -a emit-llvm src/main.vx > build/main.mlir
"$LLVM/mlir-translate" --mlir-to-llvmir build/main.mlir -o build/main.ll
"$LLVM/clang" -O2 -Wno-override-module -c -x ir build/main.ll -o build/main.o
"$LLVM/clang" build/main.o -L"$VX_HOME/lib" -lvx_std_core -Wl,-rpath,"$VX_HOME/lib" -lm \
  -o build/hellinger

# The C++ checker, with a counter added to its `verify`, visits 798493 boxes.
want="pass (798493 boxes)"
got="$(build/hellinger || true)"
if [ "$got" != "$want" ]; then
  echo "FAIL: hellinger printed: $got" >&2
  echo "want: $want" >&2
  exit 1
fi
echo "ok: $got"
