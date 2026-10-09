#!/usr/bin/env bash
# Run textkit's tests.
#
#   ./build.sh                 run the tests
#   VX_HOME=<dir> ./build.sh   use a toolchain other than ~/.vx/current
#
# textkit is imported as `textkit::...`, so the directory that holds textkit/ (the repository
# root) has to be on the import path. VX_STD_PATH replaces the toolchain's stdlib paths, so they
# are listed again. See ../FRICTION.md.

set -euo pipefail

cd "$(dirname "$0")"
VX_HOME="${VX_HOME:-$HOME/.vx/current}"
export VX_STD_PATH="$VX_HOME/stdlib/std:$VX_HOME/stdlib:.."

"$VX_HOME/bin/vxc" -a run-jit test.vx | grep -v '^\[JIT\]'
