"""Architecture selection for the model pipeline.

The architecture is chosen from two boolean flags read from the environment:

- ``USE_BITNET``: use ternary (1.58-bit) quantized linear layers.
- ``USE_RETNET``: use the retention mechanism instead of softmax attention.

Resolution:

- ``USE_BITNET=true`` and ``USE_RETNET=true``  -> ``retbit``  (legacy hybrid, today's default)
- ``USE_BITNET=true``  and ``USE_RETNET=false`` -> ``bitnet``  (ternary transformer + KV cache)
- ``USE_BITNET=false`` and ``USE_RETNET=true``  -> ``retnet``  (retention, float projections)
- ``USE_BITNET=false`` and ``USE_RETNET=false`` -> ``transformer`` (standard attention + KV cache)
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any


class Architecture(str, Enum):
    """Supported model architectures."""

    TRANSFORMER = "transformer"
    BITNET = "bitnet"
    RETNET = "retnet"
    RETBIT = "retbit"


# Human friendly labels used by the web menus.
ARCHITECTURE_LABELS: dict[str, str] = {
    Architecture.TRANSFORMER.value: "Transformer padrão (Llama-style · RoPE · GQA · KV cache)",
    Architecture.BITNET.value: "BitNet 1.58-bit (ternário · RoPE · KV cache)",
    Architecture.RETNET.value: "RetNet (retenção linear · float)",
    Architecture.RETBIT.value: "BitNet + RetNet (camadas ternárias e retenção)",
}

# Default value used when loading legacy checkpoints without an architecture field.
LEGACY_ARCHITECTURE = Architecture.RETBIT


def resolve_architecture(
    use_bitnet: bool | None = None,
    use_retnet: bool | None = None,
) -> Architecture:
    """Resolve the effective architecture from the two environment flags."""
    from src.config import settings

    bitnet = settings.use_bitnet if use_bitnet is None else bool(use_bitnet)
    retnet = settings.use_retnet if use_retnet is None else bool(use_retnet)
    if bitnet and retnet:
        return Architecture.RETBIT
    if bitnet:
        return Architecture.BITNET
    if retnet:
        return Architecture.RETNET
    return Architecture.TRANSFORMER


def is_recurrent(architecture: str | Architecture) -> bool:
    """Retention based architectures use a fixed-size recurrent state."""
    return Architecture(architecture) in (Architecture.RETNET, Architecture.RETBIT)


def is_ternary(architecture: str | Architecture) -> bool:
    """Ternary (BitLinear) based architectures."""
    return Architecture(architecture) in (Architecture.BITNET, Architecture.RETBIT)


def is_attention(architecture: str | Architecture) -> bool:
    """Softmax attention + KV cache architectures."""
    return Architecture(architecture) in (Architecture.TRANSFORMER, Architecture.BITNET)


def architecture_label(architecture: str | Architecture) -> str:
    return ARCHITECTURE_LABELS.get(Architecture(architecture).value, Architecture(architecture).value)


def architecture_from_config(config_path: str | Path) -> Architecture | None:
    """Read the architecture stored inside a checkpoint config.json, if present."""
    path = Path(config_path)
    if not path.is_file():
        return None
    try:
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        value = payload.get("architecture")
        if value is not None:
            return Architecture(value)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return None


def architecture_from_state_dict(state_dict: dict[str, Any]) -> Architecture | None:
    """Infer the architecture from the checkpoint tensor names (legacy fallback).

    Checkpoints antigos (antes das flags) não têm o campo ``architecture`` no
    config.json; os únicos existentes eram o híbrido RetBit (chaves
    ``retention.*``), então essa detecção garante retomada sem quebrar.
    """
    if any(".retention.q_proj.weight" in key for key in state_dict):
        return Architecture.RETBIT
    if any(".attention.q_proj.weight" in key for key in state_dict):
        # Novo checkpoint sem o campo: não dá para distinguir float/ternário só
        # pelas chaves; o .env resolve transformer/bitnet.
        return None
    return None


def architecture_from_checkpoint(
    config: dict[str, Any],
    state_dict: dict[str, Any],
) -> Architecture:
    """Resolve the checkpoint architecture with fallbacks:

    1. Campo ``architecture`` do config.json (checkpoints novos).
    2. Detecção pelas chaves dos tensores (checkpoints legados retbit).
    3. Chaves de attention sem campo explícito -> resolve pelas flags do .env
       (transformer/bitnet são indistinguíveis apenas pelas chaves).
    4. Caso contrário, padrão legado (retbit) — preserva o comportamento original.
    """
    value = config.get("architecture")
    if value is not None:
        try:
            return Architecture(value)
        except ValueError:
            pass
    inferred = architecture_from_state_dict(state_dict)
    if inferred is not None:
        return inferred
    if any(".attention.q_proj.weight" in key for key in state_dict):
        return resolve_architecture()
    return LEGACY_ARCHITECTURE
