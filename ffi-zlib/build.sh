#!/usr/bin/env bash
# Build and run ffi-zlib: Vx calling the system's zlib.
#
#   ./build.sh                 build build/ffi-zlib and run it
#   VX_HOME=<dir> ./build.sh   use a toolchain other than ~/.vx/current
#
# Needs Vx newer than v0.0.3 (built from main at 39a12e0a or later), and zlib (libz.so) where
# the linker finds it.
#
# `vxc -a run-jit` links a fixed list of libraries and cannot be asked for -lz
# (vx-lang/Vx#1652). So this compiles to an object and links it here, naming what
# `vxc -a run-jit` would link (see src/jit.rs in Vx), plus zlib.

set -euo pipefail

cd "$(dirname "$0")"
VX_HOME="${VX_HOME:-$HOME/.vx/current}"
mkdir -p build

"$VX_HOME/bin/vxc" -a emit-obj src/main.vx -o build/main.o
"$VX_HOME/llvm/bin/clang" build/main.o -o build/ffi-zlib \
  -L "$VX_HOME/lib" -lvx_std_core -Wl,-rpath,"$VX_HOME/lib" \
  "$VX_HOME/llvm/lib/libmlir_c_runner_utils.so" "$VX_HOME/llvm/lib/libmlir_runner_utils.so" \
  -Wl,-rpath,"$VX_HOME/llvm/lib" \
  -lz -lm

# The program returns the number of failed checks, and prints "ok" when there are none.
out="$(./build/ffi-zlib || true)"
echo "$out"
if ! echo "$out" | grep -qx 'ok'; then
  echo "FAIL" >&2
  exit 1
fi
