#!/usr/bin/env bash
# Build wordcount into build/wordcount, then check it against `wc` on sample.txt.
#
#   ./build.sh           build and test
#   VX_HOME=<dir> ./build.sh   use a toolchain other than ~/.vx/current
#
# vxc has no "build an executable" action, so this script compiles to an object file and
# links it against the toolchain's runtime library itself. See ../FRICTION.md.

set -euo pipefail

cd "$(dirname "$0")"
VX_HOME="${VX_HOME:-$HOME/.vx/current}"
VXC="$VX_HOME/bin/vxc"
CLANG="$VX_HOME/llvm/bin/clang"
[ -x "$CLANG" ] || CLANG=clang

mkdir -p build
"$VXC" -a emit-obj src/main.vx -o build/main.o
"$CLANG" build/main.o -L"$VX_HOME/lib" -lvx_std_core -Wl,-rpath,"$VX_HOME/lib" -lm \
  -o build/wordcount

want="$(wc sample.txt | awk '{print $1, $2, $3}')"
got="$(build/wordcount sample.txt)"
if [ "$got" != "$want" ]; then
  echo "FAIL: wordcount says '$got', wc says '$want'" >&2
  exit 1
fi
echo "ok: $got"
