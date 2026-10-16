#!/usr/bin/env bash
# Build script for native C++20 principal energy kernel.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

CXX="${CXX:-clang++}"
CXXFLAGS="-std=c++20 -O3 -fPIC -Wall -Wextra"

case "$(uname -s)" in
    Darwin)
        OUTPUT="libprincipal_energy.dylib"
        LDFLAGS="-dynamiclib"
        ;;
    Linux)
        OUTPUT="libprincipal_energy.so"
        LDFLAGS="-shared"
        ;;
    *)
        OUTPUT="libprincipal_energy.dll"
        LDFLAGS="-shared"
        ;;
esac

echo "Compiling ${OUTPUT} with ${CXX}..."
"${CXX}" ${CXXFLAGS} ${LDFLAGS} -o "${OUTPUT}" principal_energy.cpp
echo "Build successful: ${SCRIPT_DIR}/${OUTPUT}"
