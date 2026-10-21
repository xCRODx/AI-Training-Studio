#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNTIME_ROOT="${PROJECT_ROOT}/artifacts/retbit-runtime"

echo "==> Configurando CMake para RetBit Core..."
cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release

echo "==> Compilando RetBit Core e CLI..."
cmake --build "${BUILD_DIR}" --config Release -j"$(nproc 2>/dev/null || echo 4)"

echo "==> Executando testes unitários numéricos..."
ctest --test-dir "${BUILD_DIR}" --output-on-failure

BUILT_RUNTIME="${BUILD_DIR}/retbit_cli_build"
if [ ! -f "${BUILT_RUNTIME}" ]; then
    echo "ERRO: Executável RetBit não foi gerado em ${BUILT_RUNTIME}" >&2
    exit 1
fi

mkdir -p "${RUNTIME_ROOT}"
RUNTIME_HASH=$(sha256sum "${BUILT_RUNTIME}" | cut -c1-12)
DEPLOYED_RUNTIME="${RUNTIME_ROOT}/retbit_cli-${RUNTIME_HASH}"

cp "${BUILT_RUNTIME}" "${DEPLOYED_RUNTIME}"
chmod +x "${DEPLOYED_RUNTIME}"
ln -sf "${DEPLOYED_RUNTIME}" "${RUNTIME_ROOT}/retbit_cli"

echo "==> Runtime publicado com sucesso em: ${DEPLOYED_RUNTIME}"
