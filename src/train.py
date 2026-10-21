import asyncio
import hashlib
import json
import logging
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Generator

import torch
import torch.nn as nn

import bitsandbytes as bnb
from datasets import load_dataset, load_dataset_builder
from safetensors import safe_open
from transformers import PreTrainedTokenizer
from safetensors.torch import save_file, load_file

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

from src.architecture import architecture_from_config, architecture_from_state_dict, resolve_architecture
from src.config import settings
from src.model import CustomLLM
from src.paths import PROJECT_ROOT
from src.tokenizer import tokenizer, tokenizer_metadata, validate_tokenizer_metadata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LIVE_MODEL = None

CHECKPOINT_DIR = str(settings.checkpoint_path)
CHECKPOINT_MODEL = os.path.join(CHECKPOINT_DIR, "model.safetensor")
CHECKPOINT_OPT = os.path.join(CHECKPOINT_DIR, "optimizer.pt")
CHECKPOINT_CONFIG = os.path.join(CHECKPOINT_DIR, "config.json")
DATA_PIPELINE_VERSION = 3
CHECKPOINT_SNAPSHOT_FILES = (
    "model.safetensor",
    "optimizer.pt",
    "scheduler.pt",
    "config.json",
    "training_metrics.csv",
)


class DatasetPositionError(RuntimeError):
    """Raised when continuing could repeat or silently skip the data stream."""


def _advance_accumulation(micro_step: int, accumulation: int) -> tuple[int, bool]:
    next_step = int(micro_step) + 1
    if next_step >= int(accumulation):
        return 0, True
    return next_step, False


def create_persistent_snapshot(tokens_consumed: int) -> Path:
    """Publica apenas os arquivos do checkpoint, nunca a raiz inteira do projeto."""
    checkpoint_root = Path(CHECKPOINT_DIR).resolve()
    snapshot_parent = (
        PROJECT_ROOT / "checkpoints"
        if checkpoint_root == PROJECT_ROOT
        else checkpoint_root.parent
    )
    destination = snapshot_parent / f"checkpoint_{int(tokens_consumed)}"
    if destination.exists():
        return destination

    snapshot_parent.mkdir(parents=True, exist_ok=True)
    partial = snapshot_parent / f".{destination.name}.partial"
    if partial.exists():
        raise RuntimeError(f"Snapshot parcial anterior requer revisão: {partial}")
    partial.mkdir()
    try:
        copied = 0
        for name in CHECKPOINT_SNAPSHOT_FILES:
            source = checkpoint_root / name
            if source.is_file():
                shutil.copy2(source, partial / name)
                copied += 1
        if copied < 3 or not (partial / "model.safetensor").is_file() or not (partial / "config.json").is_file():
            raise RuntimeError("Checkpoint incompleto; snapshot persistente não foi publicado.")
        os.replace(partial, destination)
    except Exception:
        if partial.exists():
            shutil.rmtree(partial)
        raise
    return destination


def _dataset_kwargs(*, split: str | None = None, streaming: bool = False) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "path": settings.dataset_name,
        "streaming": streaming,
    }
    if settings.dataset_config_name:
        kwargs["name"] = settings.dataset_config_name
    if settings.dataset_revision:
        kwargs["revision"] = settings.dataset_revision
    if split:
        kwargs["split"] = split
    if settings.hf_token:
        kwargs["token"] = settings.hf_token
    return kwargs


def pretraining_data_metadata() -> dict[str, object]:
    """Identity of every dataset choice that changes the token stream."""
    return {
        "name": settings.dataset_name,
        "config": settings.dataset_config_name or None,
        "revision": settings.dataset_revision or None,
        "split": settings.dataset_split,
        "text_field": settings.dataset_text_field,
        "quality_field": settings.dataset_quality_field or None,
        "quality_value": settings.dataset_quality_value or None,
        "validation_holdout_fraction": settings.validation_holdout_fraction,
        "validation_batches": settings.validation_batches,
        "train_batch_size": settings.train_batch_size,
        "repeat_dataset": settings.repeat_dataset,
        "documents_use_eos": True,
        "pipeline_version": DATA_PIPELINE_VERSION,
    }


def _validate_pretraining_data_metadata(checkpoint_config: dict[str, Any]) -> None:
    saved = checkpoint_config.get("pretraining_data")
    if not isinstance(saved, dict):
        logger.warning(
            "Checkpoint legado sem identidade completa do dataset; "
            "a retomada não consegue provar que o stream é o mesmo."
        )
        return
    current = pretraining_data_metadata()
    mismatches = [
        f"{key}: checkpoint={saved.get(key)!r}, atual={current.get(key)!r}"
        for key in current
        if saved.get(key, False if key == "repeat_dataset" else None) != current.get(key)
    ]
    if mismatches:
        raise RuntimeError(
            "Dataset/pipeline incompatível com o checkpoint existente. "
            "Use outro CHECKPOINT_DIR para iniciar a nova receita: " + "; ".join(mismatches)
        )


def _training_recipe_metadata() -> dict[str, object]:
    return {
        "train_lr": settings.train_lr,
        "weight_decay": settings.weight_decay,
        "optimizer_beta1": settings.optimizer_beta1,
        "optimizer_beta2": settings.optimizer_beta2,
        "optimizer_eps": settings.optimizer_eps,
        "gradient_accumulation_steps": settings.gradient_accumulation_steps,
        "train_steps": settings.train_steps,
        "warmup_steps": settings.warmup_steps,
    }


def _validate_training_recipe(checkpoint_config: dict[str, Any]) -> None:
    saved = checkpoint_config.get("training_recipe")
    if not isinstance(saved, dict):
        logger.warning("Checkpoint legado sem fingerprint da receita do optimizer/scheduler.")
        return
    current = _training_recipe_metadata()
    mismatches = [
        f"{key}: checkpoint={saved.get(key)!r}, atual={value!r}"
        for key, value in current.items()
        if saved.get(key) != value
    ]
    if mismatches:
        raise RuntimeError(
            "Receita de treino incompatível com o optimizer/scheduler salvo. "
            "Use outro CHECKPOINT_DIR ou restaure o .env original: " + "; ".join(mismatches)
        )


def _amp_dtype():
    """Retorna o dtype de mixed precision conforme a configuração (bf16 default)."""
    precision = (settings.train_precision or "bf16").strip().lower()
    if precision in ("fp16", "float16"):
        return torch.float16
    if precision in ("bf16", "bfloat16"):
        return torch.bfloat16
    if precision in ("fp32", "float32"):
        return torch.float32
    logger.warning(f"train_precision '{precision}' desconhecido; usando bf16.")
    return torch.bfloat16


def _skip_legacy_start(dataset, state: dict):
    """
    Checkpoint antigo (sem posição exata): pula direto para a região do dataset
    correspondente à porcentagem já consumida, evitando reprocessar exemplos que
    o modelo já viu. Usado apenas quando não há state_dict exato para restaurar.
    """
    start_pct = state.get("start_pct", 0.0) or 0.0
    if start_pct <= 0:
        return dataset

    try:
        builder_kwargs = _dataset_kwargs()
        builder_kwargs.pop("streaming", None)
        builder = load_dataset_builder(**builder_kwargs)
        num_examples = builder.info.splits["train"].num_examples
    except Exception as e:
        logger.warning(f"Não foi possível obter o tamanho do dataset para pular: {e}")
        return dataset

    if not num_examples:
        return dataset

    skip = int(num_examples * min(start_pct, 100.0) / 100.0)
    if skip <= 0:
        return dataset

    logger.info(f"Checkpoint legado: pulando ~{skip} exemplos ({start_pct}% do dataset).")
    return dataset.skip(skip)


def _is_validation_document(text: str) -> bool:
    fraction = float(settings.validation_holdout_fraction)
    if fraction <= 0:
        return False
    if fraction >= 1:
        raise ValueError("VALIDATION_HOLDOUT_FRACTION deve ser menor que 1.")
    fingerprint = hashlib.sha256(text.encode("utf-8")).digest()
    value = int.from_bytes(fingerprint[:8], "little")
    return value < int(fraction * (1 << 64))


def _passes_quality_filter(item: dict[str, Any]) -> bool:
    field = settings.dataset_quality_field.strip()
    if not field:
        return True
    return str(item.get(field, "")) == settings.dataset_quality_value


def data_generator(
    tokenizer_obj: PreTrainedTokenizer,
    state: dict | None = None,
    context_window: int | None = None,
) -> Generator[torch.Tensor, None, None]:
    """
    Gerador de dados que carrega o dataset em streaming, tokeniza e divide em chunks
    exatos com tamanho CONTEXT_WINDOW.
    """
    if state is None:
        state = {}
    context_window = int(context_window or settings.context_window)

    logger.info(f"Carregando dataset: {settings.dataset_name}")

    if settings.overfit_mode:
        logger.info("Modo Laboratório (Overfit) ativado.")
        base_tokens = tokenizer_obj.encode(settings.overfit_text, add_special_tokens=True)
        expanded_tokens = []
        while len(expanded_tokens) < context_window + 1:
            expanded_tokens.extend(base_tokens)
        expanded_tokens = expanded_tokens[:context_window + 1]

        while True:
            batch_tensors = [expanded_tokens for _ in range(settings.train_batch_size)]
            yield torch.tensor(batch_tensors, dtype=torch.long)

    dataset = load_dataset(**_dataset_kwargs(split=settings.dataset_split, streaming=True))

    state["dataset"] = dataset

    saved_state = state.get("state_dict")
    start_pct = state.get("start_pct", 0.0) or 0.0
    example_offset = int(state.get("dataset_offset", 0) or 0)
    offset_authoritative = bool(state.get("dataset_offset_authoritative", False))

    # Buffers residuais de uma sessão anterior (retomada exata e sem repetição).
    # Eles permitem continuar exatamente do token seguinte sem reprocessar nada.
    buffer = [int(t) for t in (state.get("token_buffer") or [])]
    batch_buffer = [[int(t) for t in c] for c in (state.get("batch_buffer") or [])]
    validation_buffer = [int(t) for t in (state.get("validation_token_buffer") or [])]
    validation_batch_buffer = [
        [int(t) for t in chunk] for chunk in (state.get("validation_batch_buffer") or [])
    ]
    validation_batches = [
        [[int(t) for t in chunk] for chunk in batch]
        for batch in (state.get("validation_batches") or [])
    ]

    use_offset = example_offset > 0 and offset_authoritative
    if saved_state:
        # CAMINHO RÁPIDO: restaura a posição exata via state_dict do datasets (salta
        # direto para o shard certo, sem re-baixar/reprocessar os shards anteriores).
        # Muito mais rápido que dataset.skip() quando o offset é grande.
        try:
            dataset.load_state_dict(saved_state)
            logger.info("Posição exata do dataset restaurada (state_dict).")
        except Exception as e:
            logger.error(f"Não foi possível restaurar a posição exata do dataset: {e}")
            if use_offset:
                dataset = dataset.skip(example_offset)
                logger.info(f"Retomada exata por offset: pulando {example_offset} exemplos.")
            else:
                raise DatasetPositionError(
                    "A posição exata do dataset não pôde ser restaurada e não existe "
                    "offset absoluto confiável. Retomada interrompida para não repetir dados."
                ) from e
            state["dataset"] = dataset
    elif use_offset:
        # Retomada auto-contida por offset (usado quando não há state_dict exato).
        dataset = dataset.skip(example_offset)
        logger.info(f"Retomada exata por offset: pulando {example_offset} exemplos.")
        state["dataset"] = dataset
    else:
        dataset = _skip_legacy_start(dataset, state)
        state["dataset"] = dataset

    if not saved_state and example_offset == 0 and start_pct <= 0:
        offset_authoritative = True
    state["dataset_offset_authoritative"] = offset_authoritative

    # A contagem de exemplos (dataset_offset) só é confiável quando ela é
    # ABSOLUTA: ou a retomada foi por offset (use_offset), ou o treino começou do
    # zero (sem state_dict). Se retomamos por state_dict, não sabemos o offset
    # absoluto e NUNCA devemos gravar uma contagem relativa — senão a próxima
    # retomada pularia de menos e reprocessaria exemplos.
    example_count = example_offset
    for item in dataset:
        if offset_authoritative:
            example_count += 1
            state["dataset_offset"] = example_count

        if not _passes_quality_filter(item):
            continue

        text = item.get(settings.dataset_text_field, "")
        if not text:
            continue

        tokens = tokenizer_obj.encode(text, add_special_tokens=False)
        if tokenizer_obj.eos_token_id is not None:
            tokens.append(int(tokenizer_obj.eos_token_id))

        # Holdout determinístico: estes documentos nunca entram no buffer de
        # treino. Os primeiros batches ficam persistidos no checkpoint e são
        # reutilizados em round-robin, sem avançar o stream durante avaliação.
        if _is_validation_document(text):
            if len(validation_batches) < settings.validation_batches:
                validation_buffer.extend(tokens)
                while len(validation_buffer) >= context_window + 1:
                    chunk = validation_buffer[: context_window + 1]
                    validation_buffer = validation_buffer[context_window:]
                    validation_batch_buffer.append(chunk)
                    if len(validation_batch_buffer) == settings.train_batch_size:
                        validation_batches.append(validation_batch_buffer)
                        validation_batch_buffer = []
                        if len(validation_batches) >= settings.validation_batches:
                            break
            state["validation_token_buffer"] = validation_buffer
            state["validation_batch_buffer"] = validation_batch_buffer
            state["validation_batches"] = validation_batches
            continue

        buffer.extend(tokens)

        while len(buffer) >= context_window + 1:
            chunk = buffer[: context_window + 1]
            # Sobrepõe o último target como primeiro input da próxima janela;
            # assim nenhuma transição causal é descartada entre chunks.
            buffer = buffer[context_window:]
            batch_buffer.append(chunk)

            if len(batch_buffer) == settings.train_batch_size:
                state["token_buffer"] = buffer
                state["batch_buffer"] = []
                yield torch.tensor(batch_buffer, dtype=torch.long)
                batch_buffer = []

        state["token_buffer"] = buffer
        state["batch_buffer"] = batch_buffer
        state["validation_token_buffer"] = validation_buffer
        state["validation_batch_buffer"] = validation_batch_buffer
        state["validation_batches"] = validation_batches


def _atomic_write_json(path: str, data: dict) -> None:
    """Escreve o JSON de forma atômica para nunca deixar o checkpoint corrompido
    se o processo for morto no meio da escrita."""
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, step: int, tokens_consumed: int, data_state: dict | None = None, scheduler: Any = None) -> None:
    """
    Salva os pesos do modelo, estado do otimizador e metadados no diretório base.
    Sobrescreve os arquivos existentes silenciosamente (com escrita atômica).
    """
    if data_state and int(data_state.get("accumulation_micro_step", 0) or 0) != 0:
        raise RuntimeError(
            "Checkpoint recusado no meio da acumulação de gradientes; aguarde a fronteira segura."
        )
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    state_dict = model.state_dict()
    state_dict.pop("lm_head.weight", None)
    _atomic_safetensors_save(state_dict, CHECKPOINT_MODEL)
    _atomic_torch_save(optimizer.state_dict(), CHECKPOINT_OPT)

    if data_state is None:
        data_state = {}
    dataset_state = _dataset_state(data_state)
    start_pct = data_state.get("start_pct", 0.0) or 0.0
    dataset_offset = int(data_state.get("dataset_offset", 0) or 0)
    token_buffer = [int(t) for t in (data_state.get("token_buffer") or [])]
    batch_buffer = [[int(t) for t in c] for c in (data_state.get("batch_buffer") or [])]
    validation_token_buffer = [int(t) for t in (data_state.get("validation_token_buffer") or [])]
    validation_batch_buffer = [
        [int(t) for t in chunk] for chunk in (data_state.get("validation_batch_buffer") or [])
    ]
    validation_batches = [
        [[int(t) for t in chunk] for chunk in batch]
        for batch in (data_state.get("validation_batches") or [])
    ]

    config_data = {
        "step": step,
        "tokens_consumed": tokens_consumed,
        "dataset_state": dataset_state,
        "dataset_start_pct": start_pct,
        "dataset_offset": dataset_offset,
        "dataset_offset_authoritative": bool(
            data_state.get("dataset_offset_authoritative", False)
        ),
        "token_buffer": token_buffer,
        "batch_buffer": batch_buffer,
        "validation_token_buffer": validation_token_buffer,
        "validation_batch_buffer": validation_batch_buffer,
        "validation_batches": validation_batches,
        "validation_cursor": int(data_state.get("validation_cursor", 0) or 0),
        "data_pipeline_version": DATA_PIPELINE_VERSION,
        "documents_use_eos": True,
        "pretraining_data": pretraining_data_metadata(),
        "training_recipe": _training_recipe_metadata(),
        "model_d_model": int(getattr(model, "d_model", settings.model_d_model)),
        "model_layers": int(getattr(model, "num_layers", settings.model_layers)),
        "model_experts": settings.model_experts,
        "context_window": int(getattr(model, "max_context", settings.context_window)),
        "architecture": model.architecture.value,
        "num_heads": getattr(model, "num_heads", None),
        "head_dim": getattr(model, "head_dim", None),
        "num_kv_heads": getattr(model, "num_kv_heads", None),
        "ffn_hidden_size": getattr(model, "ffn_hidden_size", None),
        "ffn_multiplier": getattr(model, "ffn_multiplier", None),
        "ffn_multiple_of": getattr(model, "ffn_multiple_of", None),
        "norm_eps": getattr(model, "norm_eps", None),
        "init_std": getattr(model, "init_std", None),
        "rope_theta": getattr(model, "rope_theta", None),
        "train_lr": settings.train_lr,
        "optimizer_beta1": settings.optimizer_beta1,
        "optimizer_beta2": settings.optimizer_beta2,
        "optimizer_eps": settings.optimizer_eps,
        "weight_decay": settings.weight_decay,
        "gradient_accumulation_steps": settings.gradient_accumulation_steps,
        "scheduler_step": int(scheduler.state_dict().get("last_epoch", 0)) if scheduler is not None else 0,
        "accumulation_micro_step": int(data_state.get("accumulation_micro_step", 0) or 0),
        "overfit_mode": settings.overfit_mode,
        "overfit_text": settings.overfit_text,
        "tokenizer": tokenizer_metadata(),
    }

    try:
        _atomic_write_json(CHECKPOINT_CONFIG, config_data)
    except TypeError:
        config_data["dataset_state"] = None
        _atomic_write_json(CHECKPOINT_CONFIG, config_data)

    logger.info(f"Checkpoint salvo no step {step} ({tokens_consumed} tokens).")


def _atomic_torch_save(obj, path: str) -> None:
    """Salva tensores de forma atômica (arquivo .tmp + os.replace)."""
    tmp_path = path + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def _atomic_safetensors_save(state_dict, path: str) -> None:
    """Salva pesos em safetensors de forma atômica (arquivo .tmp + os.replace)."""
    tmp_path = path + ".tmp"
    save_file(state_dict, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    """Load a complete checkpoint or fail closed; never silently restart."""
    defaults: dict[str, Any] = {
        "step": 0,
        "tokens_consumed": 0,
        "dataset_state": None,
        "dataset_start_pct": 0.0,
        "tokens_pulled": 0,
        "dataset_offset": 0,
        "dataset_offset_authoritative": None,
        "token_buffer": [],
        "batch_buffer": [],
        "validation_token_buffer": [],
        "validation_batch_buffer": [],
        "validation_batches": [],
        "validation_cursor": 0,
        "scheduler_step": 0,
        "accumulation_micro_step": 0,
    }
    paths = (CHECKPOINT_MODEL, CHECKPOINT_OPT, CHECKPOINT_CONFIG)
    present = [os.path.exists(path) for path in paths]
    if not any(present):
        return defaults
    if not all(present):
        missing_files = [path for path, exists in zip(paths, present, strict=True) if not exists]
        raise RuntimeError(f"Checkpoint parcial; arquivos ausentes: {missing_files}")

    with open(CHECKPOINT_CONFIG, "r", encoding="utf-8") as handle:
        config_data = json.load(handle)

    _validate_pretraining_data_metadata(config_data)
    _validate_training_recipe(config_data)
    model_device = str(next(model.parameters()).device)
    weights = load_file(CHECKPOINT_MODEL, device=model_device)
    expected_vocab = int(weights["embeddings.weight"].shape[0])
    validate_tokenizer_metadata(config_data, expected_vocab)
    missing, unexpected = model.load_state_dict(weights, strict=False)
    invalid_missing = set(missing) - {"lm_head.weight"}
    if invalid_missing or unexpected:
        raise RuntimeError(
            "Pesos incompatíveis com a arquitetura do checkpoint. "
            f"Ausentes={sorted(invalid_missing)}; inesperados={unexpected}."
        )
    optimizer.load_state_dict(torch.load(CHECKPOINT_OPT, map_location=model_device))

    state = dict(defaults)
    state.update(config_data)
    logger.info("Checkpoint validado e carregado. Resumindo do step %s.", state["step"])
    return state


def _dataset_state(data_state: dict):
    dataset = data_state.get("dataset")
    if dataset is None:
        return None
    try:
        return dataset.state_dict()
    except Exception as e:
        logger.warning(f"Não foi possível capturar a posição do dataset: {e}")
        return None


def estimate_dataset_start_pct(target_tokens: int) -> float:
    """
    Estima a porcentagem do dataset já consumida a partir de uma contagem de tokens.
    Usado apenas para checkpoints antigos que salvaram tokens em vez da posição exata.
    """
    if target_tokens <= 0:
        return 0.0

    try:
        builder_kwargs = _dataset_kwargs()
        builder_kwargs.pop("streaming", None)
        builder = load_dataset_builder(**builder_kwargs)
        num_examples = builder.info.splits["train"].num_examples
    except Exception as e:
        logger.warning(f"Não foi possível obter o tamanho do dataset: {e}")
        return 0.0

    if not num_examples:
        return 0.0

    sample_tokens = 0
    sample_items = 0
    try:
        dataset = load_dataset(**_dataset_kwargs(split=settings.dataset_split, streaming=True))
        for item in dataset:
            if not _passes_quality_filter(item):
                continue
            text = item.get(settings.dataset_text_field, "")
            if not text:
                continue
            sample_tokens += len(tokenizer.encode(text, add_special_tokens=False))
            sample_items += 1
            if sample_items >= settings.dataset_estimation_samples:
                break
    except Exception as e:
        logger.warning(f"Erro ao amostrar o dataset para estimativa: {e}")
        return 0.0

    if sample_items == 0 or sample_tokens == 0:
        return 0.0

    avg_tokens_per_item = sample_tokens / sample_items
    total_tokens = num_examples * avg_tokens_per_item
    if total_tokens <= 0:
        return 0.0

    pct = min(target_tokens / total_tokens * 100.0, 99.0)
    return round(pct, 4)


def calculate_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> tuple[float, float, float]:
    """
    Calcula as taxas de acerto Top-1, Top-5 e Top-50 usando uma amostra aleatória
    para não travar a GPU com cálculos estatísticos massivos.
    """
    with torch.no_grad():
        sample_size = min(settings.metrics_accuracy_sample_tokens, logits.size(0))
        indices = torch.randperm(logits.size(0), device=logits.device)[:sample_size]

        sampled_logits = logits[indices]
        sampled_targets = targets[indices].unsqueeze(-1)

        _, top50_indices = sampled_logits.topk(50, dim=-1)

        top1_acc = (top50_indices[:, 0:1] == sampled_targets).float().mean().item()
        top5_acc = (top50_indices[:, :5] == sampled_targets).any(dim=-1).float().mean().item()
        top50_acc = (top50_indices == sampled_targets).any(dim=-1).float().mean().item()

    return top1_acc, top5_acc, top50_acc

def get_ternary_distribution(model: nn.Module) -> tuple[float, float, float]:
    from src.model import BitLinear
    ternary_params = 0
    device = next(model.parameters()).device
    
    neg1_tensor = torch.tensor(0, dtype=torch.long, device=device)
    zero_tensor = torch.tensor(0, dtype=torch.long, device=device)
    pos1_tensor = torch.tensor(0, dtype=torch.long, device=device)
    
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, BitLinear) and module.quantized:
                p = module.weight
                if p.requires_grad:
                    # Aplica a mesma física do Forward Pass para extrair a distribuição real
                    scale = p.abs().mean().clamp(min=1e-5)
                    quantized = torch.round(p / scale).clamp(-1, 1)

                    neg1_tensor += (quantized == -1).sum()
                    zero_tensor += (quantized == 0).sum()
                    pos1_tensor += (quantized == 1).sum()
                    ternary_params += p.numel()
                
    if ternary_params == 0:
        return 0.0, 0.0, 0.0
        
    neg1_count = neg1_tensor.item()
    zero_count = zero_tensor.item()
    pos1_count = pos1_tensor.item()
        
    return (
        (neg1_count / ternary_params) * 100.0,
        (zero_count / ternary_params) * 100.0,
        (pos1_count / ternary_params) * 100.0
    )


def get_optimizer_vram_mb(optimizer: torch.optim.Optimizer) -> float:
    """Calcula o consumo exato de VRAM (em MB) apenas dos estados internos do otimizador."""
    total_bytes = 0
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                total_bytes += v.nelement() * v.element_size()
    return total_bytes / (1024 ** 2)

async def train_loop(queue_metrics: asyncio.Queue, is_training: dict) -> None:
    """
    Loop de treinamento principal assíncrono.
    Recebe um dicionário is_training para controle de parada segura por referência.
    """
    device = torch.device(settings.inference_device if torch.cuda.is_available() else "cpu")
    logger.info(f"Iniciando treinamento no dispositivo: {device}")

    checkpoint_config_data: dict[str, Any] = {}
    if os.path.exists(CHECKPOINT_CONFIG):
        with open(CHECKPOINT_CONFIG, "r", encoding="utf-8") as handle:
            checkpoint_config_data = json.load(handle)

    # Arquitetura e dimensões vêm do checkpoint existente. O .env só cria um
    # modelo novo quando não há checkpoint completo.
    # e só então do .env. Assim o pós/pausa/retomada nunca muda de arquitetura.
    checkpoint_arch = architecture_from_config(CHECKPOINT_CONFIG)
    if checkpoint_arch is None and os.path.exists(CHECKPOINT_MODEL):
        try:
            with safe_open(CHECKPOINT_MODEL, framework="pt", device="cpu") as handle:
                checkpoint_arch = architecture_from_state_dict(dict.fromkeys(handle.keys()))
        except Exception as exc:
            logger.warning(f"Não foi possível detectar a arquitetura do checkpoint: {exc}")
    architecture = checkpoint_arch or resolve_architecture()
    if checkpoint_arch is not None and checkpoint_arch.value != settings.architecture:
        logger.warning(
            f"Checkpoint usa arquitetura '{checkpoint_arch.value}' mas o .env define "
            f"'{settings.architecture}'. Mantendo '{checkpoint_arch.value}' para não quebrar a retomada."
        )

    # Instancia o modelo
    model = CustomLLM(
        vocab_size=tokenizer.vocab_size,
        d_model=int(checkpoint_config_data.get("model_d_model", settings.model_d_model)),
        num_layers=int(checkpoint_config_data.get("model_layers", settings.model_layers)),
        architecture=architecture,
        num_heads=int(checkpoint_config_data.get("num_heads", settings.model_num_heads) or 0),
        head_dim=int(checkpoint_config_data.get("head_dim", settings.model_head_dim) or 0),
        num_kv_heads=int(checkpoint_config_data.get("num_kv_heads", settings.model_kv_heads) or 0),
        rope_theta=float(checkpoint_config_data.get("rope_theta", settings.rope_theta)),
        max_context=int(checkpoint_config_data.get("context_window", settings.context_window)),
        ffn_hidden_size=int(
            checkpoint_config_data.get("ffn_hidden_size", settings.model_ffn_hidden_size) or 0
        ),
        ffn_multiplier=float(
            checkpoint_config_data.get("ffn_multiplier", settings.model_ffn_multiplier)
        ),
        ffn_multiple_of=int(
            checkpoint_config_data.get("ffn_multiple_of", settings.model_ffn_multiple_of)
        ),
        norm_eps=float(checkpoint_config_data.get("norm_eps", settings.model_norm_eps)),
        init_std=float(checkpoint_config_data.get("init_std", settings.model_init_std)),
    ).to(dtype=_amp_dtype() if device.type == "cuda" else torch.float32, device=device)
    global LIVE_MODEL
    LIVE_MODEL = model
    # Gradient Checkpointing: recomputa o forward no backward p/ economizar VRAM.
    # Só afeta o treino (inferência usa prefill/recurrent, que não passam por aqui).
    model.gradient_checkpointing = settings.gradient_checkpointing

    # Otimizador: 8-bit (bitsandbytes) reduz ~75% da VRAM dos estados do otimizador.
    # Fallback para AdamW padrão quando 8-bit indisponível ou desabilitado.
    if torch.cuda.is_available() and settings.use_8bit_optimizer:
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(),
            lr=settings.train_lr,
            betas=(settings.optimizer_beta1, settings.optimizer_beta2),
            eps=settings.optimizer_eps,
            weight_decay=settings.weight_decay
        )
        logger.info("Usando otimizador AdamW 8-bit (bitsandbytes).")
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=settings.train_lr,
            betas=(settings.optimizer_beta1, settings.optimizer_beta2),
            eps=settings.optimizer_eps,
            weight_decay=settings.weight_decay
        )
    pad_token_id = tokenizer.pad_token_id if getattr(tokenizer, 'pad_token_id', None) is not None else 0
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    accumulation = max(1, settings.gradient_accumulation_steps)
    # TRAIN_STEPS e WARMUP_STEPS são contados em MICRO steps (o "Step" exibido no dashboard).
    # O scheduler avança uma vez por OPTIMIZER step (a cada `accumulation` micro steps),
    # então os totais precisam ser convertidos para a unidade do scheduler — senão o LR
    # decai ~`accumulation`x mais devagar do que o esperado.
    scheduler_total_steps = max(1, settings.train_steps // accumulation)
    scheduler_warmup_steps = max(0, min(settings.warmup_steps // accumulation, scheduler_total_steps))

    def _cosine_lr_lambda(current_step: int) -> float:
        if current_step < scheduler_warmup_steps:
            return float(current_step) / float(max(1, scheduler_warmup_steps))
        progress = (current_step - scheduler_warmup_steps) / float(max(1, scheduler_total_steps - scheduler_warmup_steps))
        progress = min(1.0, progress)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_cosine_lr_lambda)
    
    # Carrega estado anterior se houver
    resume = load_checkpoint(model, optimizer)
    checkpoint_exists = (
        os.path.exists(CHECKPOINT_MODEL)
        and os.path.exists(CHECKPOINT_OPT)
        and os.path.exists(CHECKPOINT_CONFIG)
    )
    global_step = int(resume["step"])
    tokens_consumed = int(resume["tokens_consumed"])
    dataset_state = resume.get("dataset_state")
    start_pct = float(resume.get("dataset_start_pct", 0.0) or 0.0)
    legacy_tokens_pulled = int(resume.get("tokens_pulled", 0) or 0)
    dataset_offset = int(resume.get("dataset_offset", 0) or 0)
    saved_offset_authoritative = resume.get("dataset_offset_authoritative")
    dataset_offset_authoritative = (
        bool(saved_offset_authoritative)
        if saved_offset_authoritative is not None
        else dataset_offset > 0
    )
    scheduler_step = int(resume.get("scheduler_step", 0) or 0)
    accumulation_micro_step = int(resume.get("accumulation_micro_step", 0) or 0)
    if accumulation_micro_step != 0:
        raise RuntimeError(
            "Checkpoint registra acumulação parcial, mas gradientes não são persistidos. "
            "Retomada bloqueada para evitar um optimizer step incorreto."
        )

    if dataset_state is None and start_pct <= 0 and legacy_tokens_pulled > 0:
        start_pct = estimate_dataset_start_pct(legacy_tokens_pulled)
        logger.info(f"Checkpoint antigo: retomando dataset a ~{start_pct}% a partir de {legacy_tokens_pulled} tokens.")

    data_state = {
        "dataset": None,
        "state_dict": dataset_state,
        "start_pct": start_pct,
        "dataset_offset": dataset_offset,
        "dataset_offset_authoritative": dataset_offset_authoritative,
        "token_buffer": [int(t) for t in (resume.get("token_buffer") or [])],
        "batch_buffer": [[int(t) for t in chunk] for chunk in (resume.get("batch_buffer") or [])],
        "validation_token_buffer": [
            int(t) for t in (resume.get("validation_token_buffer") or [])
        ],
        "validation_batch_buffer": [
            [int(t) for t in chunk]
            for chunk in (resume.get("validation_batch_buffer") or [])
        ],
        "validation_batches": [
            [[int(t) for t in chunk] for chunk in batch]
            for batch in (resume.get("validation_batches") or [])
        ],
        "validation_cursor": int(resume.get("validation_cursor", 0) or 0),
        "accumulation_micro_step": 0,
    }
    if not checkpoint_exists:
        data_state["dataset_offset_authoritative"] = True
    elif global_step > 0 and dataset_state is None and not dataset_offset_authoritative:
        raise DatasetPositionError(
            "Checkpoint sem posição exata do dataset e sem offset absoluto confiável. "
            "A retomada foi bloqueada para não repetir batches."
        )
    
    # Gatilho de Troca de Resolução (CRÍTICO) - Removido (Não usado no AdamW)

    # O scheduler avança uma vez por OPTIMIZER step (a cada gradient_accumulation_steps
    # microsteps). Na retomada, avançamos até a posição exata salva no checkpoint
    # (scheduler_step). Em checkpoints antigos sem esse campo, usamos como fallback a
    # conversão global_step -> optimizer steps — nunca a contagem de microsteps, senão o
    # LR pula ~16x adiante.
    if scheduler_step > 0:
        for _ in range(min(scheduler_step, scheduler_total_steps)):
            lr_scheduler.step()
    else:
        for _ in range(global_step // accumulation):
            lr_scheduler.step()
    if not checkpoint_exists:
        save_checkpoint(model, optimizer, global_step, tokens_consumed, data_state, lr_scheduler)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    # Gerador de dados
    data_gen = data_generator(tokenizer, state=data_state, context_window=model.max_context)
    
    model.train()
    optimizer.zero_grad()
    
    last_grad_norm = 0.0
    last_weight_norm = 0.0
    last_val_loss = None
    last_val_perplexity = None
    
    start_time = time.time()
    total_tokens_this_step = 0
    was_active = is_training.get("active", False)
    checkpoint_interval = max(1, int(settings.checkpoint_save_steps))
    next_checkpoint_step = ((global_step // checkpoint_interval) + 1) * checkpoint_interval
    
    import csv
    csv_file_path = os.path.join(CHECKPOINT_DIR, "training_metrics.csv")
    csv_headers = [
        "Time_Elapsed_s", "Step", "Tokens_Consumed", "Tokens_Per_Sec", 
        "Total_VRAM_Peak_MB", "Optimizer_VRAM_MB", "GPU_Compute_Util_Pct",
        "Loss", "Val_Loss", "Perplexity", "Val_Perplexity",
        "Acc_Top1", "Acc_Top5", "Acc_Top50", 
        "Firing_Rate", "Grad_Norm", "Logits_Entropy"
    ]
    
    # Cria o arquivo e escreve o cabeçalho se não existir
    if not os.path.exists(csv_file_path):
        os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
        with open(csv_file_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)
    else:
        # Migração: se o CSV antigo não tem a coluna Logits_Entropy, atualiza o
        # cabeçalho para manter o alinhamento das novas linhas com o novo formato.
        with open(csv_file_path, mode='r', newline='', encoding='utf-8') as f:
            first_line = f.readline().strip()
        if "Logits_Entropy" not in first_line:
            csv_headers_line = ",".join(csv_headers)
            with open(csv_file_path, mode='r+', newline='', encoding='utf-8') as f:
                header = f.readline()
                rest = f.read()
                f.seek(0)
                f.write(csv_headers_line + "\n" + rest)
                f.truncate()
    
    # Tempo global absoluto para o CSV
    global_start_time = time.time()
    
    while True:
        current_active = is_training.get("active", False)
        
        # Se acabou de pausar, salva um checkpoint automático
        if was_active and not current_active:
            logger.info("Treinamento pausado. Salvando checkpoint automático...")
            save_checkpoint(model, optimizer, global_step, tokens_consumed, data_state, lr_scheduler)
            is_training["checkpoint_saved"] = True
            if device.type == "cuda":
                torch.cuda.empty_cache()
                
        # Se forçaram o save enquanto estava pausado
        if not current_active and is_training.get("force_save", False):
            logger.info("Salvamento forçado executado durante a pausa.")
            save_checkpoint(model, optimizer, global_step, tokens_consumed, data_state, lr_scheduler)
            is_training["checkpoint_saved"] = True
            if device.type == "cuda":
                torch.cuda.empty_cache()
            is_training["force_save"] = False

        if not current_active and is_training.get("shutdown_requested", False):
            logger.info("Encerramento seguro concluído após persistir o checkpoint.")
            return
            
        was_active = current_active
        
        if not current_active:
            await asyncio.sleep(0.1)
            continue
            
        try:
            batch = await asyncio.to_thread(next, data_gen)
            
            # Entradas e Alvos (Shifted for causal LM)
            x = batch[:, :-1].to(device)
            y = batch[:, 1:].to(device).contiguous().view(-1)
            
            # Forward pass com AMP (Mixed Precision)
            amp_dtype = _amp_dtype()
            with torch.amp.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=device.type == "cuda" and amp_dtype != torch.float32,
            ):
                logits = model(x)
            
                # Loss Calculation
                logits_flat = logits.view(-1, tokenizer.vocab_size)
                ce_loss = criterion(logits_flat, y)
                total_loss = ce_loss
            
                # Accumulation scale
                loss = total_loss / settings.gradient_accumulation_steps

            # Backward
            loss.backward()
            accumulation_micro_step, optimizer_boundary = _advance_accumulation(
                accumulation_micro_step, settings.gradient_accumulation_steps
            )
            data_state["accumulation_micro_step"] = accumulation_micro_step
            
            # Accuracy
            top1, top5, top50 = calculate_accuracy(logits_flat, y)
            
            global_step += 1
            tokens_in_batch = int(y.numel())
            tokens_consumed += tokens_in_batch
            
            optimizer_metrics = {'firing_rate': 0.0, 'accumulator_saturation': 0.0}

            # Optimizer Step
            if optimizer_boundary:
                grad_norm_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), settings.max_grad_norm)
                last_grad_norm = grad_norm_tensor.item() if isinstance(grad_norm_tensor, torch.Tensor) else grad_norm_tensor



                optimizer.step()
                optimizer_metrics = {'firing_rate': 0.0, 'accumulator_saturation': 0.0}
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # ============================================================
            # === HOOK PASSIVO: Learning Rate (leitura do optimizer) ===
            # Custo: zero — leitura direta do param_group, sempre disponível.
            # ============================================================
            current_lr = optimizer.param_groups[0]['lr']
            
            # ============================================================
            # === HOOK PASSIVO: Weight Norms (a cada 10 steps) ===
            # Calcula a magnitude média L2 dos pesos para detectar
            # saturação silenciosa ou explosão de pesos.
            # Custo: mínimo — leitura read-only sem gradient tracking.
            # ============================================================
            if global_step % settings.metrics_log_steps == 0:
                with torch.no_grad():
                    total_norm_sq_tensor = torch.tensor(0.0, device=device)
                    param_count = 0
                    for p in model.parameters():
                        if p.requires_grad:
                            total_norm_sq_tensor += p.data.norm(2) ** 2
                            param_count += 1
                    last_weight_norm = (total_norm_sq_tensor.item() / max(param_count, 1)) ** 0.5
            
            # ============================================================
            # === HOOK PASSIVO: Validation Loss (a cada 50 steps) ===
            # Avalia um batch fixo do holdout persistido, sem avançar o stream
            # de treino. Custo permanece ~1 forward a cada 50 steps.
            # ============================================================
            if global_step % settings.validation_eval_steps == 0:
                validation_batches = data_state.get("validation_batches") or []
                if validation_batches:
                    cursor = int(data_state.get("validation_cursor", 0) or 0)
                    val_input = torch.tensor(
                        validation_batches[cursor % len(validation_batches)], dtype=torch.long
                    ).to(device)
                    data_state["validation_cursor"] = (cursor + 1) % len(validation_batches)
                    val_x = val_input[:, :-1]
                    val_y = val_input[:, 1:].contiguous().view(-1)
                    model.eval()
                    amp_dtype = _amp_dtype()
                    with torch.no_grad(), torch.amp.autocast(
                        device_type=device.type,
                        dtype=amp_dtype,
                        enabled=device.type == "cuda" and amp_dtype != torch.float32,
                    ):
                        val_logits = model(val_x)
                        val_logits_flat = val_logits.view(-1, tokenizer.vocab_size)
                        val_ce = criterion(val_logits_flat, val_y)
                        last_val_loss = val_ce.item()
                        last_val_perplexity = math.exp(min(last_val_loss, 20))  # Clamp para evitar overflow
                    model.train()
            
            total_tokens_this_step += tokens_in_batch

            if global_step % settings.metrics_log_steps == 0:
                # Hardware Metrics
                end_time = time.time()
                step_time = end_time - start_time
                tokens_per_sec = total_tokens_this_step / step_time if step_time > 0 else 0
                batches_per_sec = settings.metrics_log_steps / step_time if step_time > 0 else 0
                
                # CORREÇÃO DA VRAM (Pico do Treino e Isolada do Otimizador)
                vram_max = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0
                opt_vram = get_optimizer_vram_mb(optimizer) if device.type == "cuda" else 0.0
                try:
                    gpu_util = torch.cuda.utilization(device) if device.type == "cuda" else 0
                except:
                    gpu_util = 0
                time_elapsed = end_time - global_start_time
                
                # Previsão em Tempo Real
                predictions = []
                last_seq_logits = logits[-1, -1, :]
                last_target = y.view(batch.size(0), -1)[-1, -1].item()
                
                context_length = min(settings.metrics_preview_tokens, x.size(1))
                context_tokens = x[-1, -context_length:].tolist()
                context_text = tokenizer.decode(context_tokens)
                
                _, top5_pred_indices = last_seq_logits.topk(5, dim=-1)
                top5_probs = torch.softmax(last_seq_logits, dim=-1).gather(0, top5_pred_indices)
                top5_words = [
                    f"{tokenizer.decode([idx.item()])} ({prob.item() * 100:.2f}%)"
                    for idx, prob in zip(top5_pred_indices, top5_probs)
                ]
                # Valores brutos dos logits das top-5 previsões (para exibição no painel)
                top5_logits = [round(float(v), 3) for v in last_seq_logits[top5_pred_indices].tolist()]

                # Entropia média dos logits do batch (amostra para não pesar na GPU).
                # Mostra se as distribuições estão ficando mais confiantes/concentradas.
                with torch.no_grad():
                    sample_n = min(settings.metrics_accuracy_sample_tokens, logits_flat.size(0))
                    sample_idx = torch.randperm(logits_flat.size(0), device=device)[:sample_n]
                    sampled_logits = logits_flat[sample_idx].float()
                    sampled_probs = torch.softmax(sampled_logits, dim=-1)
                    logits_entropy = -(sampled_probs * torch.log(sampled_probs.clamp_min(1e-12))).sum(-1).mean().item()
                
                target_word = tokenizer.decode([last_target])
                
                predictions = {
                    "context": context_text,
                    "target": target_word,
                    "top5_predictions": top5_words,
                    "top5_logits": top5_logits,
                    "logits_entropy": round(logits_entropy, 4)
                }
                
                t_neg, t_zero, t_pos = get_ternary_distribution(model)

                # Gravação no CSV
                with open(csv_file_path, mode='a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        round(time_elapsed, 2),
                        global_step,
                        tokens_consumed,
                        round(tokens_per_sec, 2),
                        round(vram_max, 2),
                        round(opt_vram, 2),
                        gpu_util,
                        round(total_loss.item(), 4),
                        round(last_val_loss, 4) if last_val_loss else "",
                        round(math.exp(min(ce_loss.item(), 20)), 4),
                        round(last_val_perplexity, 4) if last_val_perplexity else "",
                        round(top1, 4),
                        round(top5, 4),
                        round(top50, 4),
                        round(optimizer_metrics.get('firing_rate', 0.0), 4),
                        round(last_grad_norm, 4),
                        round(logits_entropy, 4)
                    ])

                # Metrics Dict
                metrics = {
                    "step": global_step,
                    "loss": total_loss.item(),
                    "ce_loss": ce_loss.item(),
                    "aux_loss": 0.0,
                    "top1_acc": top1,
                    "top5_acc": top5,
                    "top50_acc": top50,
                    "tokens_per_sec": round(tokens_per_sec, 2),
                    "batches_per_sec": round(batches_per_sec, 2),
                    "vram_allocated_mb": round(opt_vram, 2),
                    "vram_max_mb": round(vram_max, 2),
                    "gpu_util_pct": gpu_util,
                    "tokens_consumed": tokens_consumed,
                    "grad_norm": round(last_grad_norm, 4),
                    "predictions": predictions,
                    "group_size": 1,
                    "firing_rate": round(optimizer_metrics.get('firing_rate', 0.0), 4),
                    "accumulator_saturation": round(optimizer_metrics.get('accumulator_saturation', 0.0), 4),
                    "ternary_neg1_pct": round(t_neg, 2),
                    "ternary_zero_pct": round(t_zero, 2),
                    "ternary_pos1_pct": round(t_pos, 2),
                    "train_perplexity": round(math.exp(min(ce_loss.item(), 20)), 4),
                    "learning_rate": current_lr,
                    "weight_norm": round(last_weight_norm, 4),
                    "val_loss": last_val_loss,
                    "val_perplexity": last_val_perplexity,
                    "logits_entropy": round(logits_entropy, 4),
                }
                
                # Envia métricas sem bloquear o loop
                try:
                    queue_metrics.put_nowait(metrics)
                except asyncio.QueueFull:
                    pass
                
                start_time = time.time()
                total_tokens_this_step = 0
            
            # Checkpoint saving
            if global_step >= next_checkpoint_step and accumulation_micro_step == 0:
                save_checkpoint(model, optimizer, global_step, tokens_consumed, data_state, lr_scheduler)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                new_dir = create_persistent_snapshot(tokens_consumed)
                logger.info(f"Cópia persistente salva em {new_dir}")
                while next_checkpoint_step <= global_step:
                    next_checkpoint_step += checkpoint_interval
            
            if is_training.get("force_save", False) and accumulation_micro_step == 0:
                save_checkpoint(model, optimizer, global_step, tokens_consumed, data_state, lr_scheduler)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                new_dir = create_persistent_snapshot(tokens_consumed)
                logger.info(f"Cópia manual persistente salva em {new_dir}")
                is_training["force_save"] = False

            # Uma pausa/saída só se torna efetiva depois de um optimizer step.
            # Assim não há gradientes de microbatches apenas na RAM quando o
            # checkpoint é publicado e o processo pode ser encerrado.
            if (
                is_training.get("pause_requested", False)
                and accumulation_micro_step == 0
            ):
                is_training["pause_requested"] = False
                is_training["active"] = False
             
            # Mantém o event loop respirando em CADA passo
            await asyncio.sleep(0)
            
        except StopIteration:
            if settings.repeat_dataset:
                logger.info("Fim do dataset alcançado. Reiniciando por configuração explícita.")
                data_state = {
                    "dataset": None,
                    "state_dict": None,
                    "start_pct": 0.0,
                    "dataset_offset": 0,
                    "dataset_offset_authoritative": True,
                    "token_buffer": [],
                    "batch_buffer": [],
                    "validation_token_buffer": data_state.get("validation_token_buffer", []),
                    "validation_batch_buffer": data_state.get("validation_batch_buffer", []),
                    "validation_batches": data_state.get("validation_batches", []),
                    "validation_cursor": data_state.get("validation_cursor", 0),
                }
                data_gen = data_generator(tokenizer, state=data_state, context_window=model.max_context)
                continue

            remainder = accumulation_micro_step
            if remainder:
                correction = settings.gradient_accumulation_steps / remainder
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(correction)
                torch.nn.utils.clip_grad_norm_(model.parameters(), settings.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                accumulation_micro_step = 0
                data_state["accumulation_micro_step"] = 0
            logger.info("Fim do dataset alcançado. Salvando checkpoint final sem repetir dados.")
            save_checkpoint(model, optimizer, global_step, tokens_consumed, data_state, lr_scheduler)
            is_training["active"] = False
            is_training["pause_requested"] = False
            is_training["checkpoint_saved"] = True
            return
        except DatasetPositionError:
            is_training["active"] = False
            is_training["pause_requested"] = False
            raise
        except Exception as e:
            import traceback
            logger.error(f"Erro durante o treinamento: {e}")
            logger.error(traceback.format_exc())
            # Salva o progresso atual antes de recriar o gerador, para nunca perder
            # a posição do dataset e nunca reprocessar exemplos já consumidos.
            try:
                recovery_state = _dataset_state(data_state)
                if recovery_state is None and not data_state.get(
                    "dataset_offset_authoritative", False
                ):
                    raise DatasetPositionError(
                        "Falha sem posição restaurável do dataset; recuperação bloqueada "
                        "para não repetir batches."
                    )
                data_state["state_dict"] = recovery_state
                save_checkpoint(model, optimizer, global_step, tokens_consumed, data_state, lr_scheduler)
            except DatasetPositionError:
                is_training["active"] = False
                is_training["pause_requested"] = False
                raise
            except Exception as save_err:
                logger.error(f"Falha ao salvar checkpoint de recuperação: {save_err}")
            if device.type == "cuda":
                torch.cuda.empty_cache()
            # Recria o gerador a partir da posição salva (dataset_offset/buffers),
            # em vez de chamar um gerador já encerrado (o que causaria loop infinito).
            data_state["dataset"] = None
            data_gen = data_generator(tokenizer, state=data_state, context_window=model.max_context)
            await asyncio.sleep(5)
            
    # Salva checkpoint final ao interromper
    logger.info("Interrompendo treinamento e salvando checkpoint final...")
    save_checkpoint(model, optimizer, global_step, tokens_consumed, data_state, lr_scheduler)
    if device.type == "cuda":
        torch.cuda.empty_cache()
