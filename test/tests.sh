#!/usr/bin/env bash
# tests.sh – run key test suites and capture logs
# set -euo pipefail

echo "Running derivatives correctness..."
julia --project="." test/test_derivatives.jl > test/test_derivatives.txt 2>&1

echo "Running links correctness..."
julia --project="." test/test_links.jl > test/test_links.txt 2>&1

echo "Running derivative allocation checks..."
julia --project="." test/test_derivative_allocations.jl > test/test_derivative_allocations.txt 2>&1

echo "Done. Logs written to:"
echo " - test/test_derivatives.txt"
echo " - test/test_links.txt"
echo " - test/test_derivative_allocations.txt"
