#!/usr/bin/env bash
# Build and run gpu-convex-hull: the convex hull of many points, with the heavy work on the GPU,
# checked against host code and Python. On a Mac the kernels run on the Apple GPU, and the build
# checks that they ran there.
#
#   ./build.sh                 run it
#   VX_HOME=<dir> ./build.sh   use a toolchain other than ~/.vx/current
#
# Needs a Vx main from 2026-11-23 or later (a488cc3d), and python3. The sort comes from
# ../gpu-scan-sort, which vxc reads as the package `scan_sort`.

set -euo pipefail

cd "$(dirname "$0")"
VX_HOME="${VX_HOME:-$HOME/.vx/current}"
VXC="$VX_HOME/bin/vxc"
mkdir -p build

# On a Mac, fleet/m4-uma.vx places the kernels on the Apple GPU. Elsewhere there is no Metal, so
# the program uses fleet/a100-40.vx, as the other GPU projects do. Without an NVIDIA GPU, the
# kernels then run on their CPU copy.
if [ "$(uname -s)" = Darwin ]; then
  MACHINE="$VX_HOME/fleet/m4-uma.vx"
else
  MACHINE="$VX_HOME/fleet/a100-40.vx"
  # A CUDA toolchain links every program against libcuda.so.1. Without a driver, use the stub
  # of the toolkit, as in ../gpu-scan-sort/build.sh (vx-lang/Vx#1737).
  CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
  if [ -e "$VX_HOME/lib/libvx_cuda_dispatch.so" ] && ! ldconfig -p | grep -q 'libcuda\.so\.1 '; then
    mkdir -p build/stub
    ln -sf "$CUDA_HOME/lib64/stubs/libcuda.so" build/stub/libcuda.so.1
    export LD_LIBRARY_PATH="$PWD/build/stub${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
fi

run() {
  "$VXC" -a run-jit test.vx --package scan_sort=../gpu-scan-sort --host default --machine "$MACHINE" \
    2> "$1" | grep -v '^\[' || true
}

# 1. The results. The program returns the number of failed checks, and prints "ok" when there
#    are none. On a Mac, VX_ENABLE_CPU_FALLBACK=0 stops the program when a kernel would run on
#    the CPU, and VX_DISPATCH_VERBOSE=1 prints a line for each launch.
out="$(VX_DISPATCH_VERBOSE=1 VX_ENABLE_CPU_FALLBACK=0 run build/gpu.log)"
echo "$out" > build/gpu.txt
echo "$out" | grep -v '^hull '
if ! echo "$out" | grep -qx 'ok'; then
  tail -20 build/gpu.log >&2
  echo "FAIL: see build/gpu.log" >&2
  exit 1
fi

if [ "$(uname -s)" = Darwin ]; then
  # 2. The GPU ran each kernel: each launch that the dispatcher takes has a [Vx Metal] line.
  launches="$(grep -c '^\[Vx Dispatcher\] Intercepted kernel dispatch' build/gpu.log || true)"
  on_gpu="$(grep -c '^\[Vx Metal\] vx_npu_kernel_[0-9]*: grid' build/gpu.log || true)"
  if [ "$launches" -eq 0 ] || [ "$launches" != "$on_gpu" ]; then
    echo "FAIL: $launches kernel launches, but $on_gpu of them ran on the Apple GPU" >&2
    exit 1
  fi
  echo "ok: all $launches kernel launches ran on the Apple GPU"

  # 3. The CPU copy of each kernel gives the same output, line for line.
  cpu="$(VX_METAL=0 run build/cpu.log)"
  echo "$cpu" > build/cpu.txt
  if [ "$cpu" != "$out" ]; then
    diff build/gpu.txt build/cpu.txt | head -20 >&2
    echo "FAIL: the CPU copy of the kernels prints something else" >&2
    exit 1
  fi
  echo "ok: the CPU copy of the kernels prints the same"
else
  echo "skipped: no Apple GPU here, so the kernels ran on their CPU copy"
fi

# 4. Python makes the same points, and finds the same hull with its own code.
echo "$out" | python3 -I check.py

echo "drawing: $PWD/build/hull.svg"
