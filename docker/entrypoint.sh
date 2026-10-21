#!/usr/bin/env bash
set -euo pipefail

cd /workspace

for executable in \
  "${LLAMA_CPP_SERVER}" \
  "${LLAMA_CPP_CLI}" \
  "${BITNET_CPP_SERVER}" \
  "${BITNET_CPP_CLI}" \
  "${RETBIT_CPP_CLI}"; do
  if [[ ! -x "${executable}" ]]; then
    echo "Runtime obrigatório ausente: ${executable}" >&2
    exit 1
  fi
done

mkdir -p checkpoints/base checkpoints_finetuning artifacts/llama artifacts/bitnet data

echo "Workspace compartilhado: /workspace"
echo "llama.cpp: $(${LLAMA_CPP_SERVER} --version 2>&1 | head -n 1)"
echo "bitnet.cpp: $(${BITNET_CPP_SERVER} --version 2>&1 | head -n 1)"

exec "$@"
