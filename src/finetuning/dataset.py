import hashlib
import json
import logging
import os
import re
import unicodedata
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from src.finetuning.chat_format import ROLE_LABELS, ensure_system_message
from src.finetuning.config import FineTuningSettings

logger = logging.getLogger(__name__)
DATASET_PIPELINE_SCHEMA_VERSION = 5


@dataclass
class PreparedData:
    train: "ConversationTensorDataset"
    validation: "ConversationTensorDataset"
    report: dict[str, Any]


class ConversationTensorDataset(Dataset):
    def __init__(self, tensors: dict[str, torch.Tensor]):
        self.tensors = tensors

    def __len__(self) -> int:
        if "offsets" in self.tensors:
            return max(int(self.tensors["offsets"].numel()) - 1, 0)
        return int(self.tensors["input_ids"].size(0))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if "offsets" in self.tensors:
            start = int(self.tensors["offsets"][index])
            end = int(self.tensors["offsets"][index + 1])
            input_ids = self.tensors["input_ids_flat"][start:end]
            labels = self.tensors["labels_flat"][start:end]
            return {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": torch.ones_like(input_ids),
                "supervised_tokens": self.tensors["supervised_tokens"][index],
            }
        return {key: value[index] for key, value in self.tensors.items()}

    @classmethod
    def from_samples(cls, samples: list[dict[str, torch.Tensor]]) -> "ConversationTensorDataset":
        lengths = torch.tensor([sample["input_ids"].numel() for sample in samples], dtype=torch.long)
        offsets = torch.cat([torch.zeros(1, dtype=torch.long), lengths.cumsum(0)])
        return cls(
            {
                "input_ids_flat": torch.cat([sample["input_ids"] for sample in samples]).contiguous(),
                "labels_flat": torch.cat([sample["labels"] for sample in samples]).contiguous(),
                "offsets": offsets.contiguous(),
                "supervised_tokens": torch.stack(
                    [sample["supervised_tokens"] for sample in samples]
                ).contiguous(),
            }
        )

    @property
    def total_tokens(self) -> int:
        if "input_ids_flat" in self.tensors:
            return int(self.tensors["input_ids_flat"].numel())
        return int(self.tensors["attention_mask"].sum().item())

    @property
    def total_supervised_tokens(self) -> int:
        return int(self.tensors["supervised_tokens"].sum().item())


def collate_conversation_batch(
    samples: list[dict[str, torch.Tensor]], pad_token_id: int
) -> dict[str, torch.Tensor]:
    """Pad only to the longest item in this microbatch, never to model context."""
    max_length = max(int(sample["input_ids"].numel()) for sample in samples)
    batch_size = len(samples)
    input_ids = torch.full((batch_size, max_length), pad_token_id, dtype=torch.long)
    labels = torch.full((batch_size, max_length), -100, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
    supervised_tokens = torch.empty(batch_size, dtype=torch.long)
    for index, sample in enumerate(samples):
        length = int(sample["input_ids"].numel())
        input_ids[index, :length] = sample["input_ids"]
        labels[index, :length] = sample["labels"]
        attention_mask[index, :length] = 1
        supervised_tokens[index] = sample["supervised_tokens"]
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "supervised_tokens": supervised_tokens,
    }


def _clean_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = "".join(char for char in text if char in "\n\t" or unicodedata.category(char)[0] != "C")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _normalize_messages(raw_messages: Any) -> list[dict[str, str]] | None:
    if not isinstance(raw_messages, list) or len(raw_messages) < 2:
        return None

    messages: list[dict[str, str]] = []
    previous_role: str | None = None
    for raw in raw_messages:
        if not isinstance(raw, dict):
            return None
        role = str(raw.get("role", "")).strip().lower()
        content = _clean_text(raw.get("content", ""))
        if role not in {"system", "user", "assistant"} or not content:
            return None
        if role == "system" and messages:
            return None
        if previous_role == role and role != "system":
            return None
        if previous_role == "system" and role != "user":
            return None
        if previous_role == "user" and role != "assistant":
            return None
        if previous_role == "assistant" and role != "user":
            return None
        messages.append({"role": role, "content": content})
        previous_role = role

    if messages[-1]["role"] != "assistant" or not any(item["role"] == "user" for item in messages):
        return None
    return messages


def _conversation_payload(item: dict[str, Any], configured_field: str) -> Any:
    """Read the configured schema while retaining explicit compatibility aliases."""
    candidates = [configured_field, "messages", "conversations"]
    for field in dict.fromkeys(field for field in candidates if field):
        value = item.get(field)
        if value is not None:
            return value
    return None


def _encode_window(
    messages: list[dict[str, str]],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    assistant_only: bool,
) -> dict[str, torch.Tensor] | None:
    input_ids: list[int] = []
    labels: list[int] = []
    eos_id = tokenizer.eos_token_id

    for message in messages:
        role = message["role"]
        role_name = ROLE_LABELS[role]
        prefix_ids = tokenizer.encode(f"{role_name}: ", add_special_tokens=False)
        content_ids = tokenizer.encode(message["content"], add_special_tokens=False)
        newline_ids = tokenizer.encode("\n", add_special_tokens=False)

        input_ids.extend(prefix_ids)
        labels.extend([-100] * len(prefix_ids) if assistant_only else prefix_ids)
        input_ids.extend(content_ids)
        if assistant_only and role != "assistant":
            labels.extend([-100] * len(content_ids))
        else:
            labels.extend(content_ids)

        if role == "assistant" and eos_id is not None:
            input_ids.append(eos_id)
            labels.append(eos_id)
        input_ids.extend(newline_ids)
        labels.extend([-100] * len(newline_ids))

    # Never left-truncate a conversation: doing so can remove the user prompt
    # while retaining the assistant target. Callers split only at complete
    # User -> Assistant boundaries instead.
    if len(input_ids) > max_length:
        return None

    supervised = sum(label != -100 for label in labels)
    if supervised < 2:
        return None

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
        "supervised_tokens": torch.tensor(supervised, dtype=torch.long),
    }


def _conversation_pairs(messages: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[list[dict[str, str]]]]:
    system = [messages[0]] if messages and messages[0]["role"] == "system" else []
    start = 1 if system else 0
    pairs = [messages[index : index + 2] for index in range(start, len(messages), 2)]
    return system, pairs


def _encode_conversation_samples(
    messages: list[dict[str, str]],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    assistant_only: bool,
) -> tuple[list[dict[str, torch.Tensor]], dict[str, int]]:
    """Pack a conversation without ever breaking a User -> Assistant pair."""
    system, pairs = _conversation_pairs(messages)
    stats = {
        "pairs_total": len(pairs),
        "pairs_preserved": 0,
        "pairs_over_context": 0,
        "over_context_conversations": 0,
        "split_conversations": 0,
        "truncated": 0,
        "prompt_preserved_samples": 0,
    }

    complete = _encode_window(messages, tokenizer, max_length, assistant_only)
    if complete is not None:
        stats["pairs_preserved"] = len(pairs)
        stats["prompt_preserved_samples"] = 1
        return [complete], stats

    stats["over_context_conversations"] = 1
    samples: list[dict[str, torch.Tensor]] = []
    current = list(system)
    current_pairs = 0

    def publish_current() -> None:
        nonlocal current, current_pairs
        if current_pairs == 0:
            return
        encoded = _encode_window(current, tokenizer, max_length, assistant_only)
        if encoded is None:
            raise RuntimeError("Uma janela SFT excedeu o contexto após empacotamento por pares.")
        samples.append(encoded)
        stats["pairs_preserved"] += current_pairs
        stats["prompt_preserved_samples"] += 1
        current = list(system)
        current_pairs = 0

    for pair in pairs:
        candidate = current + pair
        if _encode_window(candidate, tokenizer, max_length, assistant_only) is not None:
            current = candidate
            current_pairs += 1
            continue

        publish_current()
        standalone = list(system) + pair
        if _encode_window(standalone, tokenizer, max_length, assistant_only) is None and system:
            standalone = list(pair)
        if _encode_window(standalone, tokenizer, max_length, assistant_only) is None:
            stats["pairs_over_context"] += 1
            continue
        current = standalone
        current_pairs = 1

    publish_current()
    if len(samples) > 1:
        stats["split_conversations"] = 1
    return samples, stats


def _encode_messages(
    messages: list[dict[str, str]],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    assistant_only: bool,
) -> dict[str, torch.Tensor] | None:
    """Compatibility helper for a single test/example; production uses all packed windows."""
    samples, _ = _encode_conversation_samples(messages, tokenizer, max_length, assistant_only)
    return samples[0] if samples else None


def _select(dataset: HFDataset, limit: int, seed: int) -> HFDataset:
    if limit > 0 and len(dataset) > limit:
        # Hugging Face datasets are frequently grouped by source. Selecting the
        # first N rows biases a mixture; a seeded shuffle stays reproducible.
        return dataset.shuffle(seed=seed).select(range(limit))
    return dataset


def _load_split(settings: FineTuningSettings, split: str, *, streaming: bool = False) -> HFDataset:
    kwargs: dict[str, Any] = {
        "path": settings.dataset_name,
        "split": split,
        "cache_dir": str(settings.data_path / "cache"),
        "streaming": streaming,
    }
    if settings.dataset_config:
        kwargs["name"] = settings.dataset_config
    if settings.dataset_revision:
        kwargs["revision"] = settings.dataset_revision
    if settings.hf_token:
        kwargs["token"] = settings.hf_token
    return load_dataset(**kwargs)


def _sample_stream(source: Any, limit: int, seed: int) -> HFDataset:
    """Sample a remote split before it is materialized locally.

    ``smoltalk/all`` is several gigabytes and more than one million rows.  Loading
    it eagerly just to select a small SFT run consumes disk/RAM and can make the
    trainer appear stuck.  Streaming keeps the remote source lazy; only this
    bounded, reproducible sample becomes an in-memory Dataset.
    """
    sampled = source.shuffle(seed=seed, buffer_size=min(max(limit * 2, 1_000), 100_000)).take(limit)
    rows = list(sampled)
    if not rows:
        raise RuntimeError("O dataset SFT não retornou exemplos para a amostra solicitada.")
    return HFDataset.from_list(rows)


def _prepare_split(
    dataset: HFDataset,
    tokenizer: PreTrainedTokenizer,
    settings: FineTuningSettings,
) -> tuple[ConversationTensorDataset, dict[str, int | float]]:
    samples: list[dict[str, torch.Tensor]] = []
    seen: set[str] = set()
    counters: dict[str, int | float] = {
        "source_rows": len(dataset),
        "accepted": 0,
        "accepted_conversations": 0,
        "invalid": 0,
        "duplicates": 0,
        "too_short": 0,
        "pairs_total": 0,
        "pairs_preserved": 0,
        "pairs_over_context": 0,
        "over_context_conversations": 0,
        "split_conversations": 0,
        "truncated": 0,
        "prompt_preserved_samples": 0,
        "system_prompts_injected": 0,
        "assistant_turns": 0,
        "eos_targets": 0,
    }

    for item in dataset:
        messages = _normalize_messages(_conversation_payload(item, settings.messages_field))
        if messages is None:
            counters["invalid"] += 1
            continue
        messages, system_injected = ensure_system_message(messages, settings.system_prompt)
        counters["system_prompts_injected"] += int(system_injected)
        assistant_turns = sum(message["role"] == "assistant" for message in messages)
        counters["assistant_turns"] += assistant_turns
        if tokenizer.eos_token_id is not None:
            counters["eos_targets"] += assistant_turns
        fingerprint = "\u241e".join(f"{message['role']}:{message['content']}" for message in messages)
        if fingerprint in seen:
            counters["duplicates"] += 1
            continue
        seen.add(fingerprint)
        encoded_samples, packing = _encode_conversation_samples(
            messages, tokenizer, settings.context_window, settings.assistant_only_loss
        )
        for key, value in packing.items():
            counters[key] = int(counters[key]) + value
        if not encoded_samples:
            counters["too_short"] += 1
            continue
        samples.extend(encoded_samples)
        counters["accepted"] += len(encoded_samples)
        counters["accepted_conversations"] += 1

    if not samples:
        raise RuntimeError("Nenhuma conversa válida permaneceu após a limpeza do dataset.")
    counters["prompt_preservation_rate"] = round(
        int(counters["prompt_preserved_samples"]) / max(int(counters["accepted"]), 1), 4
    )
    counters["pair_preservation_rate"] = round(
        int(counters["pairs_preserved"]) / max(int(counters["pairs_total"]), 1), 4
    )
    return ConversationTensorDataset.from_samples(samples), counters


def _prepared_cache_paths(
    tokenizer: PreTrainedTokenizer,
    settings: FineTuningSettings,
) -> tuple[Path, Path, Path]:
    identity = json.dumps(
        {
            "dataset": settings.dataset_name,
            "dataset_config": settings.dataset_config or None,
            "revision": settings.dataset_revision,
            "train_split": settings.train_split,
            "validation_split": settings.validation_split,
            "validation_fraction": settings.validation_fraction,
            "messages_field": settings.messages_field,
            "tokenizer": tokenizer.name_or_path,
            "tokenizer_class": tokenizer.__class__.__name__,
            "tokenizer_revision": tokenizer.init_kwargs.get("_commit_hash"),
            "context_window": settings.context_window,
            "assistant_only_loss": settings.assistant_only_loss,
            "max_train_samples": settings.max_train_samples,
            "max_validation_samples": settings.max_validation_samples,
            "system_prompt": settings.system_prompt,
            "schema_version": DATASET_PIPELINE_SCHEMA_VERSION,
        },
        sort_keys=True,
    )
    fingerprint = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    root = settings.data_path / "prepared" / fingerprint
    return root / "train.safetensors", root / "validation.safetensors", root / "cleaning_report.json"


def _load_prepared_cache(
    tokenizer: PreTrainedTokenizer,
    settings: FineTuningSettings,
) -> PreparedData | None:
    train_path, validation_path, report_path = _prepared_cache_paths(tokenizer, settings)
    if not train_path.is_file() or not validation_path.is_file() or not report_path.is_file():
        return None
    train = ConversationTensorDataset(load_file(str(train_path), device="cpu"))
    validation = ConversationTensorDataset(load_file(str(validation_path), device="cpu"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return PreparedData(train=train, validation=validation, report=report)


def _save_prepared_cache(
    prepared: PreparedData,
    tokenizer: PreTrainedTokenizer,
    settings: FineTuningSettings,
) -> None:
    train_path, validation_path, report_path = _prepared_cache_paths(tokenizer, settings)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_tmp = train_path.with_suffix(".safetensors.tmp")
    validation_tmp = validation_path.with_suffix(".safetensors.tmp")
    report_tmp = report_path.with_suffix(".json.tmp")
    save_file(prepared.train.tensors, str(train_tmp))
    save_file(prepared.validation.tensors, str(validation_tmp))
    report_tmp.write_text(json.dumps(prepared.report, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(train_tmp, train_path)
    os.replace(validation_tmp, validation_path)
    os.replace(report_tmp, report_path)


def prepare_conversation_data(
    tokenizer: PreTrainedTokenizer,
    settings: FineTuningSettings,
) -> PreparedData:
    logger.info("Preparando dataset SFT %s", settings.dataset_name)
    cached = _load_prepared_cache(tokenizer, settings)
    if cached is not None:
        logger.info("Dataset SFT pré-tokenizado carregado do cache local.")
        return cached
    # Do not materialize an entire Hub dataset before applying the SFT cap.
    # The holdout is created from the same bounded source to retain the previous
    # deterministic split behavior.
    has_external_validation = bool(
        settings.validation_split and settings.validation_split != settings.train_split
    )
    required_source_rows = (
        settings.max_train_samples
        if has_external_validation
        else ceil(settings.max_train_samples / (1 - settings.validation_fraction))
    )
    source_raw = _sample_stream(
        _load_split(settings, settings.train_split, streaming=True),
        required_source_rows,
        settings.seed,
    )
    external_validation = has_external_validation
    if external_validation:
        try:
            validation_raw = _sample_stream(
                _load_split(settings, settings.validation_split, streaming=True),
                settings.max_validation_samples,
                settings.seed + 1,
            )
            train_raw = source_raw
        except Exception as exc:
            logger.warning(
                "Split de validação '%s' indisponível (%s); criando holdout determinístico de %.2f%%.",
                settings.validation_split,
                exc,
                settings.validation_fraction * 100,
            )
            external_validation = False

    if not external_validation:
        split = source_raw.train_test_split(
            test_size=settings.validation_fraction,
            seed=settings.seed,
            shuffle=True,
        )
        train_raw = split["train"]
        validation_raw = split["test"]

    train_raw = _select(train_raw, settings.max_train_samples, settings.seed)
    validation_raw = _select(
        validation_raw, settings.max_validation_samples, settings.seed + 1
    )

    train, train_report = _prepare_split(train_raw, tokenizer, settings)
    validation, validation_report = _prepare_split(validation_raw, tokenizer, settings)

    total_tokens = train.total_tokens
    supervised_tokens = train.total_supervised_tokens
    report = {
        "dataset_name": settings.dataset_name,
        "dataset_config": settings.dataset_config or None,
        "dataset_revision": settings.dataset_revision or None,
        "train_split": settings.train_split,
        "validation_split": settings.validation_split or "deterministic_holdout",
        "validation_fraction": settings.validation_fraction,
        "messages_field": settings.messages_field,
        "context_window": settings.context_window,
        "train": train_report,
        "validation": validation_report,
        "total_train_tokens": total_tokens,
        "assistant_train_tokens": supervised_tokens,
        "supervised_token_ratio": round(supervised_tokens / max(total_tokens, 1), 4),
        "complete_pairs": int(train_report["pairs_preserved"]),
        "prompt_preservation_rate": float(train_report["prompt_preservation_rate"]),
        "pair_preservation_rate": float(train_report["pair_preservation_rate"]),
        "truncated_samples": int(train_report["truncated"]),
    }
    prepared = PreparedData(train=train, validation=validation, report=report)
    _save_prepared_cache(prepared, tokenizer, settings)
    return prepared
