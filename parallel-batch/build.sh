#!/usr/bin/env bash
# Build and run parallel-batch: a batch of transfers run by worker tasks, checked against the
# same batch run in order.
#
#   ./build.sh                 build and run the Vx program and the C program
#   VX_HOME=<dir> ./build.sh   use a toolchain other than ~/.vx/current
#
# Needs Vx built from main at 22a0a84f or later: v0.0.3 has no std::sync::channel and no
# `vxc --package`.

set -euo pipefail

cd "$(dirname "$0")"
VX_HOME="${VX_HOME:-$HOME/.vx/current}"
VXC="$VX_HOME/bin/vxc"
CLANG="$VX_HOME/llvm/bin/clang"
[ -x "$CLANG" ] || CLANG=clang
export VX_STD_PATH="$VX_HOME/stdlib/std:$VX_HOME/stdlib"
LINK=(-L"$VX_HOME/lib" -lvx_std_core -Wl,-rpath,"$VX_HOME/lib" -lpthread -lm)

mkdir -p build

# The Vx program. `--package batch=src` makes `import batch::batch` read src/batch.vx.
"$VXC" -a emit-obj --package batch=src src/main.vx -o build/main.o
"$CLANG" build/main.o "${LINK[@]}" -o build/parallel-batch

# The C program links only the batch code, and the Vx runtime that runs its tasks.
"$VXC" -a emit-obj src/batch.vx -o build/batch.o
"$CLANG" host/host.c build/batch.o "${LINK[@]}" -o build/host

fail=0
check() {
  local name="$1"
  shift
  local out
  out="$("$@" || true)"
  echo "$out"
  if ! echo "$out" | grep -qx 'ok'; then
    echo "FAIL: $name" >&2
    fail=1
  fi
}

# Each task on an OS thread of its own (the default), then many tasks on a few threads.
check "threads" build/parallel-batch
check "shared runtime" env VX_TASK_RUNTIME=shared VX_TASK_WORKERS=2 build/parallel-batch
# One thread, with the seed choosing the next task: the same seed gives the same order.
for seed in 1 2 3; do
  check "seed $seed" env VX_TASK_RUNTIME=shared VX_TASK_SEED="$seed" build/parallel-batch
done
check "from C" build/host
exit "$fail"
