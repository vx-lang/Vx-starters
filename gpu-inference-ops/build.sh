#!/usr/bin/env bash
# Build and run gpu-inference-ops: the kernels of a decode step, checked against host code and
# Python. Then pull the kernels out of the program and assemble them for an A100, and
# build a standalone executable.
#
#   ./build.sh                 run it
#   VX_HOME=<dir> ./build.sh   use a toolchain other than ~/.vx/current
#
# Needs a Vx main from 2026-10-25 or later (machine files with `levels:`), and python3. Assembling the kernels needs ptxas (CUDA);
# without it that step is skipped. No GPU is needed: without one, the kernels run on the host.

set -euo pipefail

cd "$(dirname "$0")"
VX_HOME="${VX_HOME:-$HOME/.vx/current}"
VXC="$VX_HOME/bin/vxc"
MACHINE="$VX_HOME/fleet/a100-40.vx"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDA_HOME
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

# A toolchain built with CUDA links every program, GPU code or not, against the CUDA driver
# library, libcuda.so.1, which only a machine with an NVIDIA driver has (vx-lang/Vx#1737).
# Without a driver, use the toolkit's stub: it lets programs link, and at run time it reports no
# device, so the kernels run on the host.
if [ -e "$VX_HOME/lib/libvx_cuda_dispatch.so" ] && ! ldconfig -p | grep -q 'libcuda\.so\.1 '; then
  mkdir "$tmp/stub"
  ln -s "$CUDA_HOME/lib64/stubs/libcuda.so" "$tmp/stub/libcuda.so.1"
  export LD_LIBRARY_PATH="$tmp/stub${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  echo "no CUDA driver: linking against the toolkit's stub, kernels run on the host"
fi

# 1. Run it. The program returns the number of failed checks, and prints "ok" when there are
#    none.
out="$("$VXC" -a run-jit test.vx --host default --machine "$MACHINE" 2> "$tmp/err" | grep -v '^\[' || true)"
echo "$out"
if ! echo "$out" | grep -qx 'ok'; then
  cat "$tmp/err" >&2
  echo "FAIL" >&2
  exit 1
fi

# 2. The same entries in Python.
echo "$out" | python3 -I check.py

# 3. The kernels: one per GPU region, in one sm_80 module, which NVIDIA's assembler accepts.
"$VXC" -a emit-llvm test.vx --host default --machine "$MACHINE" > "$tmp/main.ll" 2> "$tmp/err" || { cat "$tmp/err" >&2; exit 1; }
python3 -I kernels.py "$tmp/main.ll" "$tmp" 6
PTXAS="$(command -v ptxas || echo "$CUDA_HOME/bin/ptxas")"
if [ -x "$PTXAS" ]; then
  "$PTXAS" -arch sm_80 "$tmp/kernels.ptx" -o "$tmp/kernels.cubin"
  echo "ok: ptxas assembles them for sm_80 (A100)"
else
  echo "skipped: assembling the kernels needs ptxas (CUDA)"
fi

# 4. A standalone executable. vxc writes an object file; linking it needs the libraries that
#    `vxc --run` links by itself (vx-lang/Vx#1499). -rdynamic lets the dispatch library find
#    the host copy of each kernel when there is no GPU.
"$VXC" -a emit-obj test.vx --host default --machine "$MACHINE" -o "$tmp/main.o" > /dev/null 2> "$tmp/err" || { cat "$tmp/err" >&2; exit 1; }
CLANG="$VX_HOME/llvm/bin/clang"
[ -x "$CLANG" ] || CLANG=clang
dispatch="$(ls "$VX_HOME"/lib/*dispatch.so | head -1)"
dispatch="$(basename "$dispatch" .so)"
"$CLANG" -rdynamic "$tmp/main.o" -o "$tmp/gpu-inference-ops" \
  -L"$VX_HOME/lib" -l"${dispatch#lib}" -lvx_std_core -Wl,-rpath,"$VX_HOME/lib" \
  -L"$VX_HOME/llvm/lib" -lmlir_runner_utils -lmlir_c_runner_utils -Wl,-rpath,"$VX_HOME/llvm/lib" \
  ${LD_LIBRARY_PATH:+-Wl,-rpath-link,"$LD_LIBRARY_PATH"} -lffi -lm
exe="$("$tmp/gpu-inference-ops" 2> /dev/null | grep -v '^\[' || true)"
if [ "$exe" != "$out" ]; then
  echo "$exe" >&2
  echo "FAIL: the executable prints something else" >&2
  exit 1
fi
echo "ok: the executable prints the same"
