#!/usr/bin/env bash
# Build textstat into build/textstat, then check it against grep and wc on sample.txt.
#
#   ./build.sh                 build and test
#   VX_HOME=<dir> ./build.sh   use a toolchain other than ~/.vx/current
#
# textstat imports textkit, which is found through VX_STD_PATH: see ../textkit/build.sh.
# Linking is done here, as in ../wordcount/build.sh. See ../FRICTION.md.

set -euo pipefail

cd "$(dirname "$0")"
VX_HOME="${VX_HOME:-$HOME/.vx/current}"
VXC="$VX_HOME/bin/vxc"
CLANG="$VX_HOME/llvm/bin/clang"
[ -x "$CLANG" ] || CLANG=clang
export VX_STD_PATH="$VX_HOME/stdlib/std:$VX_HOME/stdlib:.."

mkdir -p build
"$VXC" -a emit-obj src/main.vx -o build/main.o
"$CLANG" build/main.o -L"$VX_HOME/lib" -lvx_std_core -Wl,-rpath,"$VX_HOME/lib" -lm \
  -o build/textstat

want="words $(wc -w < sample.txt)
numbers $(grep -o '[0-9]\+' sample.txt | wc -l)
upper $(grep -o '[A-Z]\+' sample.txt | wc -l)"
got="$(build/textstat sample.txt)"
if [ "$got" != "$want" ]; then
  echo "FAIL: textstat says" >&2
  echo "$got" >&2
  echo "grep and wc say" >&2
  echo "$want" >&2
  exit 1
fi
echo "ok:"
echo "$got"
