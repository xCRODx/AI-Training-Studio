from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

import gguf
import numpy as np
import torch
from safetensors import safe_open

from src.architecture import Architecture, architecture_from_checkpoint
from src.tokenizer import tokenizer, validate_tokenizer_metadata


RETENTION_LINEAR_NAMES = {
    "retention.q_proj": "retention_q",
    "retention.k_proj": "retention_k",
    "retention.v_proj": "retention_v",
    "retention.out_proj": "retention_output",
    "ffn.fc1": "ffn_up",
    "ffn.fc2": "ffn_down",
}

# Nomes no padrão llama.cpp / bitnet.cpp (GGUF tensor map do llama).
ATTENTION_LINEAR_NAMES = {
    "attention.q_proj": "attn_q",
    "attention.k_proj": "attn_k",
    "attention.v_proj": "attn_v",
    "attention.out_proj": "attn_output",
    "ffn.gate_proj": "ffn_gate",
    "ffn.up_proj": "ffn_up",
    "ffn.down_proj": "ffn_down",
}


def _ternary(weight: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    values = weight.float()
    scale = values.abs().mean().clamp(min=1e-5)
    quantized = torch.round(values / scale).clamp(-1, 1).to(torch.int8).reshape(-1).numpy()
    codes = (quantized.astype(np.int16) + 1).astype(np.uint8)
    padding = (-codes.size) % 4
    if padding:
        codes = np.pad(codes, (0, padding), constant_values=1)
    packed = (
        codes[0::4]
        | (codes[1::4] << 2)
        | (codes[2::4] << 4)
        | (codes[3::4] << 6)
    )
    return np.ascontiguousarray(packed.view(np.int8)), np.asarray([float(scale)], dtype=np.float32)


def _ternary_bf16(weight: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Quantiza o peso para {-1, 0, 1} mantendo BF16 + escala separada (bitnet.cpp / llama.cpp)."""
    values = weight.float()
    scale = values.abs().mean().clamp(min=1e-5)
    quantized = torch.round(values / scale).clamp(-1, 1)
    return quantized.to(torch.bfloat16).contiguous(), float(scale.float().item())


def _add_bf16(writer: gguf.GGUFWriter, name: str, tensor: torch.Tensor) -> None:
    contiguous = tensor.to(torch.bfloat16).contiguous()
    raw = np.ascontiguousarray(contiguous.view(torch.uint16).numpy())
    writer.add_tensor(
        name,
        raw,
        raw_shape=tuple(contiguous.shape),
        raw_dtype=gguf.GGMLQuantizationType.BF16,
    )


def _add_f32(writer: gguf.GGUFWriter, name: str, tensor: torch.Tensor) -> None:
    writer.add_tensor(name, np.ascontiguousarray(tensor.float().numpy()))


def _tokenizer_metadata(writer: gguf.GGUFWriter) -> None:
    pieces: list[str] = []
    scores: list[float] = []
    types: list[int] = []
    sentencepiece = getattr(tokenizer, "sp_model", None)
    if sentencepiece is None and getattr(tokenizer, "vocab_file", None):
        import sentencepiece as spm

        sentencepiece = spm.SentencePieceProcessor(model_file=str(tokenizer.vocab_file))
    if sentencepiece is None:
        raise RuntimeError(
            f"Conversão GGUF ainda não suporta o tokenizer {tokenizer.__class__.__name__}."
        )
    special_ids = set(tokenizer.all_special_ids)
    for token_id in range(tokenizer.vocab_size):
        piece = sentencepiece.id_to_piece(token_id)
        score = sentencepiece.get_score(token_id)
        pieces.append(piece)
        scores.append(float(score))
        if token_id == tokenizer.unk_token_id:
            types.append(int(gguf.TokenType.UNKNOWN))
        elif token_id in special_ids:
            types.append(int(gguf.TokenType.CONTROL))
        elif sentencepiece.is_byte(token_id):
            types.append(int(gguf.TokenType.BYTE))
        elif sentencepiece.is_unused(token_id):
            types.append(int(gguf.TokenType.UNUSED))
        elif piece.startswith("<extra_id_"):
            types.append(int(gguf.TokenType.USER_DEFINED))
        else:
            types.append(int(gguf.TokenType.NORMAL))
    tokenizer_family = "llama" if "llama" in tokenizer.__class__.__name__.lower() else "t5"
    writer.add_string("tokenizer.ggml.model", tokenizer_family)
    writer.add_token_list(pieces)
    writer.add_token_scores(scores)
    writer.add_token_types(types)
    if tokenizer.bos_token_id is not None:
        writer.add_bos_token_id(int(tokenizer.bos_token_id))
    if tokenizer.eos_token_id is not None:
        writer.add_eos_token_id(int(tokenizer.eos_token_id))
    if tokenizer.unk_token_id is not None:
        writer.add_unk_token_id(int(tokenizer.unk_token_id))
    if tokenizer.pad_token_id is not None:
        writer.add_pad_token_id(int(tokenizer.pad_token_id))
    writer.add_add_bos_token(bool(getattr(tokenizer, "add_bos_token", False)))
    writer.add_add_eos_token(bool(getattr(tokenizer, "add_eos_token", False)))


def _load_checkpoint(source: Path) -> tuple[dict, Architecture, int, int, int, int, int, float]:
    """Lê config.json e devolve (config, d_model, layers, heads, kv_heads, head_dim, rope_theta)."""
    model_path = source / "model.safetensor"
    config_path = source / "config.json"
    if not model_path.is_file() or not config_path.is_file():
        raise RuntimeError("O checkpoint não contém model.safetensor e config.json válidos.")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    with safe_open(str(model_path), framework="pt", device="cpu") as checkpoint:
        keys = list(checkpoint.keys())
        expected_vocab = int(checkpoint.get_slice("embeddings.weight").get_shape()[0])
    validate_tokenizer_metadata(config, expected_vocab)
    checkpoint_architecture = architecture_from_checkpoint(config, dict.fromkeys(keys))
    d_model = int(config["model_d_model"])
    layers = int(config["model_layers"])
    heads = int(config.get("num_heads", 0) or max(1, d_model // 64))
    head_dim = int(config.get("head_dim", 0) or d_model // heads)
    kv_heads = int(config.get("num_kv_heads", 0) or heads)
    rope_theta = float(config.get("rope_theta", 10000.0))
    return config, checkpoint_architecture, d_model, layers, heads, kv_heads, head_dim, rope_theta


def _context_length(config: dict) -> int:
    context = int(config.get("fine_tuning_context_window", config.get("context_window", 0)) or 0)
    if context <= 1:
        raise RuntimeError("O checkpoint não registra um context_window válido para o GGUF.")
    return context


def _ffn_length(config: dict, d_model: int) -> int:
    explicit = int(config.get("ffn_hidden_size", 0) or 0)
    if explicit > 0:
        return explicit
    multiplier = float(config.get("ffn_multiplier", 4.0))
    multiple_of = int(config.get("ffn_multiple_of", 64))
    raw = int(np.ceil(d_model * multiplier))
    return ((raw + multiple_of - 1) // multiple_of) * multiple_of


def _write_shared_kv(writer: gguf.GGUFWriter, arch: str, config: dict, d_model: int, layers: int) -> None:
    """Metadados padrão estilo llama.cpp (bloco/embeddings/heads/ffn/rope/contexto/vocab)."""
    heads = int(config.get("num_heads", 0) or max(1, d_model // 64))
    head_dim = int(config.get("head_dim", 0) or d_model // heads)
    kv_heads = int(config.get("num_kv_heads", 0) or heads)
    ffn_dim = _ffn_length(config, d_model)
    writer.add_block_count(layers)
    writer.add_embedding_length(d_model)
    writer.add_head_count(heads)
    writer.add_head_count_kv(kv_heads)
    writer.add_feed_forward_length(ffn_dim)
    writer.add_rope_dimension_count(head_dim)
    writer.add_rope_freq_base(float(config.get("rope_theta", 10000.0)))
    writer.add_context_length(_context_length(config))
    writer.add_vocab_size(int(tokenizer.vocab_size))
    writer.add_layer_norm_rms_eps(float(config.get("norm_eps", 1e-6)))


def _convert_retbit(
    writer: gguf.GGUFWriter,
    source: Path,
    config: dict,
    d_model: int,
    layers: int,
    model_name: str,
    progress: Callable[[str], None] | None,
) -> int:
    """RetBit (híbrido RetNet + BitLinear) e RetNet float -> GGUF retbit (runtime C++).

    Os pesos lineares são empacotados no formato ternário 2-bit próprio do
    runtime RetBit, exatamente como antes (compatibilidade preservada).
    """
    heads = int(config.get("num_heads", 0) or max(1, d_model // 64))
    head_dim = int(config.get("head_dim", 0) or d_model // heads)
    ffn_dim = _ffn_length(config, d_model)

    writer.add_name(model_name)
    writer.add_description("Custom RetNet + ternary BitLinear conversational model")
    writer.add_uint32("retbit.block_count", layers)
    writer.add_uint32("retbit.embedding_length", d_model)
    writer.add_uint32("retbit.head_count", heads)
    writer.add_uint32("retbit.head_dim", head_dim)
    writer.add_uint32("retbit.feed_forward_length", ffn_dim)
    writer.add_uint32("retbit.context_length", _context_length(config))
    writer.add_uint32("retbit.vocab_size", int(tokenizer.vocab_size))
    writer.add_float32("retbit.rms_norm_epsilon", float(config.get("norm_eps", 1e-6)))
    writer.add_string("retbit.weight_encoding", "ternary-i2-packed-with-f32-scale")
    writer.add_string("retbit.source", str(source))
    _tokenizer_metadata(writer)

    linear_count = 0
    with safe_open(str(source / "model.safetensor"), framework="pt", device="cpu") as checkpoint:
        _add_bf16(writer, "token_embd.weight", checkpoint.get_tensor("embeddings.weight"))
        _add_bf16(writer, "output_norm.weight", checkpoint.get_tensor("norm_f.weight"))

        for layer in range(layers):
            prefix = f"layers.{layer}"
            _add_bf16(writer, f"blk.{layer}.attn_norm.weight", checkpoint.get_tensor(f"{prefix}.norm1.weight"))
            _add_bf16(writer, f"blk.{layer}.ffn_norm.weight", checkpoint.get_tensor(f"{prefix}.norm2.weight"))
            writer.add_tensor(
                f"blk.{layer}.retention_gamma",
                np.ascontiguousarray(checkpoint.get_tensor(f"{prefix}.retention.gammas").float().numpy()),
            )
            for checkpoint_suffix, gguf_suffix in RETENTION_LINEAR_NAMES.items():
                weight_name = f"{prefix}.{checkpoint_suffix}.weight"
                quantized, scale = _ternary(checkpoint.get_tensor(weight_name))
                writer.add_tensor(f"blk.{layer}.{gguf_suffix}.weight", quantized)
                writer.add_tensor(f"blk.{layer}.{gguf_suffix}.scale", scale)
                linear_count += 1
            if progress:
                progress(f"Bloco {layer + 1}/{layers} quantizado.")

    return linear_count


def _convert_attention_ternary(
    writer: gguf.GGUFWriter,
    source: Path,
    config: dict,
    d_model: int,
    layers: int,
    arch: str,
    model_name: str,
    progress: Callable[[str], None] | None,
) -> int:
    """BitNet do projeto -> GGUF Llama-compatible executável pelo bitnet.cpp.

    O grafo treinado neste projeto é Llama-style (RMSNorm + GQA + SwiGLU), não o
    grafo oficial BitNet b1.58 que exige sub-normalizações adicionais. Exportar
    como ``architecture=bitnet`` faria o runtime oficial carregar outro grafo.
    Por isso materializamos exatamente o peso ternário efetivo em BF16 e usamos
    o contrato Llama, que o fork bitnet.cpp executa sem mudar a matemática.
    """
    writer.add_name(model_name)
    writer.add_description("Llama-compatible transformer with materialized ternary BitNet weights")
    _write_shared_kv(writer, arch, config, d_model, layers)
    writer.add_string("retbit.source_architecture", "bitnet")
    writer.add_string(f"{arch}.weight_encoding", "materialized-ternary-bf16")
    writer.add_string(f"{arch}.source", str(source))
    _tokenizer_metadata(writer)

    linear_count = 0
    with safe_open(str(source / "model.safetensor"), framework="pt", device="cpu") as checkpoint:
        _add_bf16(writer, "token_embd.weight", checkpoint.get_tensor("embeddings.weight"))
        _add_f32(writer, "output_norm.weight", checkpoint.get_tensor("norm_f.weight"))

        for layer in range(layers):
            prefix = f"layers.{layer}"
            _add_f32(writer, f"blk.{layer}.attn_norm.weight", checkpoint.get_tensor(f"{prefix}.norm1.weight"))
            _add_f32(writer, f"blk.{layer}.ffn_norm.weight", checkpoint.get_tensor(f"{prefix}.norm2.weight"))
            for checkpoint_suffix, gguf_suffix in ATTENTION_LINEAR_NAMES.items():
                weight_name = f"{prefix}.{checkpoint_suffix}.weight"
                quantized, scale = _ternary_bf16(checkpoint.get_tensor(weight_name))
                effective = quantized.float().mul_(scale)
                _add_bf16(writer, f"blk.{layer}.{gguf_suffix}.weight", effective)
                linear_count += 1
            if progress:
                progress(f"Bloco {layer + 1}/{layers} quantizado (ternário).")

    return linear_count


def _convert_llama(
    writer: gguf.GGUFWriter,
    source: Path,
    config: dict,
    d_model: int,
    layers: int,
    arch: str,
    model_name: str,
    progress: Callable[[str], None] | None,
) -> int:
    """Transformer padrão -> GGUF arch 'llama' (llama.cpp).

    Pesos BF16, sem quantização (o llama.cpp pode quantizar depois).
    """
    writer.add_name(model_name)
    writer.add_description("Standard Llama-style transformer conversational model")
    _write_shared_kv(writer, arch, config, d_model, layers)
    writer.add_string(f"{arch}.weight_encoding", "bf16")
    writer.add_string(f"{arch}.source", str(source))
    _tokenizer_metadata(writer)

    tensor_count = 0
    with safe_open(str(source / "model.safetensor"), framework="pt", device="cpu") as checkpoint:
        _add_bf16(writer, "token_embd.weight", checkpoint.get_tensor("embeddings.weight"))
        _add_f32(writer, "output_norm.weight", checkpoint.get_tensor("norm_f.weight"))
        tensor_count += 2

        for layer in range(layers):
            prefix = f"layers.{layer}"
            _add_f32(writer, f"blk.{layer}.attn_norm.weight", checkpoint.get_tensor(f"{prefix}.norm1.weight"))
            _add_f32(writer, f"blk.{layer}.ffn_norm.weight", checkpoint.get_tensor(f"{prefix}.norm2.weight"))
            for checkpoint_suffix, gguf_suffix in ATTENTION_LINEAR_NAMES.items():
                weight_name = f"{prefix}.{checkpoint_suffix}.weight"
                _add_bf16(writer, f"blk.{layer}.{gguf_suffix}.weight", checkpoint.get_tensor(weight_name))
                tensor_count += 1
            if progress:
                progress(f"Bloco {layer + 1}/{layers} exportado (BF16).")

    return tensor_count


def convert_checkpoint_to_gguf(
    source_dir: str | Path,
    output_path: str | Path,
    progress: Callable[[str], None] | None = None,
    architecture: str | Architecture | None = None,
) -> dict[str, int | float | str]:
    source = Path(source_dir).resolve()
    config, checkpoint_architecture, d_model, layers, heads, kv_heads, head_dim, rope_theta = _load_checkpoint(source)
    if architecture is not None and Architecture(architecture) is not checkpoint_architecture:
        raise RuntimeError(
            "A arquitetura solicitada para compilação não corresponde ao checkpoint: "
            f"solicitada={Architecture(architecture).value}, checkpoint={checkpoint_architecture.value}."
        )
    arch = checkpoint_architecture

    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")

    if progress:
        progress(f"Lendo checkpoint e preparando metadados GGUF ({arch.value}).")
    gguf_architecture = {
        Architecture.RETBIT: "retbit",
        # O BitNet local preserva o grafo Llama e só troca as projeções por
        # BitLinear; veja _convert_attention_ternary para a justificativa.
        Architecture.BITNET: "llama",
        Architecture.TRANSFORMER: "llama",
    }.get(arch, arch.value)
    writer = gguf.GGUFWriter(temporary, gguf_architecture, use_temp_file=True)

    try:
        if arch is Architecture.RETBIT:
            linear_count = _convert_retbit(
                writer, source, config, d_model, layers, destination.stem, progress
            )
            encoding = "ternary-i2-packed-with-f32-scale"
            target = "retbit-cpp"
            linear_tensors = linear_count
        elif arch is Architecture.RETNET:
            raise RuntimeError(
                "Compilação RetNet float exata ainda não é suportada pelo runtime RetBit ternário. "
                "A conversão foi bloqueada para não alterar silenciosamente os pesos do modelo."
            )
        elif arch is Architecture.BITNET:
            linear_count = _convert_attention_ternary(
                writer, source, config, d_model, layers, "llama", destination.stem, progress
            )
            encoding = "materialized-ternary-bf16"
            target = "bitnet.cpp"
            linear_tensors = linear_count
        elif arch is Architecture.TRANSFORMER:
            tensor_count = _convert_llama(
                writer, source, config, d_model, layers, "llama", destination.stem, progress
            )
            encoding = "bf16"
            target = "llama.cpp"
            linear_tensors = tensor_count
        else:  # pragma: no cover
            raise ValueError(f"Arquitetura de compilação desconhecida: {arch}")

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=False)
        writer.close()
        os.replace(temporary, destination)
    except Exception:
        writer.close()
        if temporary.exists():
            temporary.unlink()
        raise

    size = destination.stat().st_size
    return {
        "path": str(destination),
        "size_bytes": size,
        "size_mb": round(size / 1024**2, 2),
        "layers": layers,
        "linear_tensors": linear_tensors,
        "vocab_size": int(tokenizer.vocab_size),
        "architecture": arch.value,
        "encoding": encoding,
        "target": target,
    }
