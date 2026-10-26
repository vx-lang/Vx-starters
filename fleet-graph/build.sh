#!/usr/bin/env bash
# Build and run fleet-graph: a fleet of GPU domains as a graph, a roofline cost model for serving
# MoE models on it, and a scheduler that places services. Plain host code: no GPU, no CUDA.
#
#   ./build.sh                 run it
#   VX_HOME=<dir> ./build.sh   use a toolchain other than ~/.vx/current
#
# Needs Vx newer than v0.0.3 (built from main at 39a12e0a or later), and python3. The compiler
# checks run `vxc --machine` against the fleet/ machine files of the toolchain.

set -euo pipefail

cd "$(dirname "$0")"
VX_HOME="${VX_HOME:-$HOME/.vx/current}"

# 1. Run it. The program returns the number of failed checks, and prints "ok" when there are none.
out="$("$VX_HOME/bin/vxc" -a run-jit test.vx 2> /dev/null | grep -v '^\[' || true)"
echo "$out"
if ! echo "$out" | grep -qx 'ok'; then
  echo "FAIL" >&2
  exit 1
fi

# 2. The cost model and the placements again, in Python.
echo "$out" | python3 -I check.py

# 3. The compiler against the planner. emit.vx prints what the planner believes (the hardware
#    table, whether each layout fits, the time of each cache transfer); admit.py writes a Vx
#    program for each, compiles it against the machine file of the SKU and compares.
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT
"$VX_HOME/bin/vxc" -a run-jit emit.vx 2> /dev/null | grep -v '^\[' > "$scratch/lines"
python3 -I admit.py "$VX_HOME" "$scratch" < "$scratch/lines"
