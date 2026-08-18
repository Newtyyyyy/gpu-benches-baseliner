#!/usr/bin/env bash
# Usage: ./hipify_one.sh <source.cu> <output.hip>
#
# Mechanical CUDA -> HIP conversion of a single benchmark source. Called by
# add_hipified_bench() in cmake/hipify.cmake; not meant to be run by hand.
#
# The reinterpret_cast around hipMalloc*'s first argument is required under
# HIP-on-NVIDIA (nvcc), where only the void** signature exists.

set -euo pipefail

src="$1"
dst="$2"

if ! command -v hipify-perl >/dev/null 2>&1; then
    echo "Error: hipify-perl not found in PATH." >&2
    exit 1
fi

mkdir -p "$(dirname "$dst")"

hipify-perl "$src" 2>/dev/null \
    | sed -e 's/cuda/hip/g' -e 's/CUDA/HIP/g' -e 's/Cuda/Hip/g' -e 's/\.cu/.hip/g' \
          -e 's/\(hipMalloc[A-Za-z]*\)(&\([A-Za-z_][A-Za-z0-9_]*\)/\1(reinterpret_cast<void **>(\&\2)/g' \
    > "$dst"
