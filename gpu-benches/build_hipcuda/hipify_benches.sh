#!/usr/bin/env bash
# Scanne tous les .cu sous */cuda et genere leur conversion mecanique
# (hipify-perl + cuda->hip, CUDA->HIP, Cuda->Hip, .cu->.hip) dans un
# dossier frere hipifiable/. Ne touche jamais a cuda/ ni a hip/.
#
# Post-traitement : enrobe le 1er argument des hipMalloc*(&ptr, ...) d'un
# reinterpret_cast<void**>. CUDA fournit une surcharge templatee de
# cudaMalloc* qui accepte un T**, HIP aussi sur AMD, mais PAS sous
# HIP-sur-NVIDIA (nvcc) ou seule la signature void** existe. Le cast rend la
# sortie compilable partout (AMD et HIP-sur-NVIDIA) et reste entierement
# genere par le script -- aucune retouche manuelle.
#
# Usage: ./hipify_benches.sh [ROOT]

set -euo pipefail

ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

if ! command -v hipify-perl >/dev/null 2>&1; then
    echo "Erreur: hipify-perl introuvable dans le PATH." >&2
    exit 1
fi

find "$ROOT" -type f -path '*/cuda/*.cu' | while read -r src; do
    bench_dir="$(dirname "$(dirname "$src")")"
    out_dir="$bench_dir/hipifiable"
    mkdir -p "$out_dir"

    out_name="$(basename "$src" .cu).hip"
    dst="$out_dir/$out_name"

    echo "hipify: $src -> $dst"
    hipify-perl "$src" 2>/dev/null \
        | sed -e 's/cuda/hip/g' -e 's/CUDA/HIP/g' -e 's/Cuda/Hip/g' -e 's/\.cu/.hip/g' \
              -e 's/\(hipMalloc[A-Za-z]*\)(&\([A-Za-z_][A-Za-z0-9_]*\)/\1(reinterpret_cast<void **>(\&\2)/g' \
        > "$dst"
done
