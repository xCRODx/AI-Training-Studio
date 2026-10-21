import csv
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from src.finetuning.config import FineTuningSettings
from src.finetuning.lora import adapter_state_dict, lora_metadata, merged_state_dict
from src.tokenizer import tokenizer_metadata, validate_tokenizer_metadata


CSV_HEADERS = [
    "timestamp",
    "elapsed_seconds",
    "epoch",
    "micro_step",
    "optimizer_step",
    "examples_seen",
    "assistant_tokens_seen",
    "loss",
    "loss_ema",
    "val_loss",
    "perplexity",
    "val_perplexity",
    "generalization_gap",
    "top1_accuracy",
    "top5_accuracy",
    "learning_rate",
    "gradient_norm",
    "weight_norm",
    "tokens_per_second",
    "sequences_per_second",
    "supervised_token_ratio",
    "vram_allocated_mb",
    "vram_reserved_mb",
    "vram_peak_mb",
    "gpu_utilization_pct",
    "ternary_neg1_pct",
    "ternary_zero_pct",
    "ternary_pos1_pct",
    "progress_pct",
    "eta_seconds",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


class FineTuningCheckpointStore:
    def __init__(self, settings: FineTuningSettings):
        self.settings = settings

    def validate_model_files(self, directory: Path) -> dict[str, Any]:
        model_path = directory / "model.safetensor"
        config_path = directory / "config.json"
        if not model_path.is_file() or model_path.stat().st_size == 0:
            raise RuntimeError(f"Modelo ausente ou vazio em {model_path}")
        if not config_path.is_file() or config_path.stat().st_size == 0:
            raise RuntimeError(f"Configuração ausente ou vazia em {config_path}")

        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)

        tensor_count = 0
        parameter_count = 0
        with safe_open(str(model_path), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            for key in keys:
                tensor_count += 1
                shape = handle.get_slice(key).get_shape()
                parameter_count += math.prod(shape)
            embedding_shape = (
                handle.get_slice("embeddings.weight").get_shape()
                if "embeddings.weight" in keys
                else None
            )

        if embedding_shape is not None:
            validate_tokenizer_metadata(config, int(embedding_shape[0]))

        if tensor_count == 0 or parameter_count == 0:
            raise RuntimeError("O arquivo safetensors não contém pesos válidos.")

        return {
            "model_path": str(model_path),
            "config_path": str(config_path),
            "size_bytes": model_path.stat().st_size,
            "tensor_count": tensor_count,
            "parameter_count": parameter_count,
            "config": config,
        }

    def bootstrap(self) -> bool:
        """Copy the pre-training model/config once, without copying optimizer history."""
        base = self.settings.base_path
        lora_base_ready = self.settings.method != "lora" or self.settings.base_model_path.is_file()
        if (
            (base / "model.safetensor").is_file()
            and (base / "config.json").is_file()
            and lora_base_ready
        ):
            self.validate_model_files(base)
            self.ensure_metrics_file()
            return False
        if self.settings.method == "lora" and base.exists() and any(base.iterdir()):
            raise RuntimeError(
                "O diretório LoRA existe, mas está incompleto. Nenhum arquivo foi sobrescrito; "
                f"inspecione manualmente {base}."
            )

        source = self.settings.source_path
        self.validate_model_files(source)
        base.mkdir(parents=True, exist_ok=True)

        for name in ("model.safetensor", "config.json"):
            src = source / name
            tmp = base / f"{name}.tmp"
            shutil.copy2(src, tmp)
            if _sha256(src) != _sha256(tmp):
                raise RuntimeError(f"A cópia de {name} falhou na verificação SHA-256: {tmp}")
            os.replace(tmp, base / name)

        if self.settings.method == "lora":
            source_model = source / "model.safetensor"
            base_model_tmp = self.settings.base_model_path.with_suffix(".safetensors.tmp")
            shutil.copy2(source_model, base_model_tmp)
            if _sha256(source_model) != _sha256(base_model_tmp):
                raise RuntimeError(
                    f"A cópia do modelo-base LoRA falhou na verificação SHA-256: {base_model_tmp}"
                )
            os.replace(base_model_tmp, self.settings.base_model_path)
            config_path = base / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config.update({"fine_tuning_method": "lora", "fine_tuning_base_model_file": "base_model.safetensors"})
            _atomic_json(config_path, config)

        self.ensure_metrics_file()
        self.validate_model_files(base)
        return True

    def ensure_metrics_file(self) -> None:
        self.settings.base_path.mkdir(parents=True, exist_ok=True)
        if self.settings.metrics_path.exists():
            return
        with self.settings.metrics_path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(CSV_HEADERS)

    def append_metrics(self, metrics: dict[str, Any]) -> None:
        self.ensure_metrics_file()
        row = [metrics.get(header, "") for header in CSV_HEADERS]
        with self.settings.metrics_path.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(row)

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        state: dict[str, Any],
        create_snapshot: bool = False,
    ) -> Path:
        base = self.settings.base_path
        base.mkdir(parents=True, exist_ok=True)

        if self.settings.method == "lora":
            adapter = adapter_state_dict(model)
            if not adapter:
                raise RuntimeError("O checkpoint LoRA não contém parâmetros de adaptador.")
            adapter_tmp = base / "adapter_model.safetensors.tmp"
            save_file(adapter, str(adapter_tmp))
            os.replace(adapter_tmp, self.settings.adapter_path)
            state_dict = merged_state_dict(model)
        else:
            state_dict = {
                key: value.detach().cpu().contiguous()
                for key, value in model.state_dict().items()
                if key != "lm_head.weight"
            }
        model_tmp = base / "model.safetensor.tmp"
        save_file(state_dict, str(model_tmp))
        os.replace(model_tmp, base / "model.safetensor")

        optimizer_tmp = base / "optimizer.pt.tmp"
        torch.save(optimizer.state_dict(), optimizer_tmp)
        os.replace(optimizer_tmp, base / "optimizer.pt")

        scheduler_tmp = base / "scheduler.pt.tmp"
        torch.save(scheduler.state_dict(), scheduler_tmp)
        os.replace(scheduler_tmp, base / "scheduler.pt")

        config_path = base / "config.json"
        existing: dict[str, Any] = {}
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
        existing.update(state)
        existing["fine_tuning_method"] = self.settings.method
        if self.settings.method == "lora":
            existing.update(lora_metadata(model))
            existing.update(
                {
                    "fine_tuning_lora_rank": self.settings.lora_rank,
                    "fine_tuning_lora_alpha": self.settings.lora_alpha,
                    "fine_tuning_lora_dropout": self.settings.lora_dropout,
                    "fine_tuning_lora_target_modules": self.settings.lora_target_modules,
                    "fine_tuning_adapter_file": "adapter_model.safetensors",
                    "fine_tuning_base_model_file": "base_model.safetensors",
                    "fine_tuning_merged_model_file": "model.safetensor",
                }
            )
        existing.update(
            {
                "model_d_model": int(getattr(model, "d_model", existing.get("model_d_model", 0))),
                "model_layers": int(getattr(model, "num_layers", existing.get("model_layers", 0))),
                "context_window": int(getattr(model, "max_context", existing.get("context_window", 0))),
                "architecture": model.architecture.value,
                "num_heads": int(model.num_heads),
                "head_dim": int(model.head_dim),
                "num_kv_heads": int(model.num_kv_heads),
                "ffn_hidden_size": int(model.ffn_hidden_size),
                "ffn_multiplier": float(model.ffn_multiplier),
                "ffn_multiple_of": int(model.ffn_multiple_of),
                "norm_eps": float(model.norm_eps),
                "init_std": float(model.init_std),
                "rope_theta": float(model.rope_theta),
                "tokenizer": tokenizer_metadata(),
            }
        )
        _atomic_json(config_path, existing)

        if create_snapshot:
            step = int(state.get("fine_tuning_optimizer_step", 0))
            prefix = "checkpoint" if self.settings.method == "full" else "checkpoint_lora"
            snapshot = self.settings.root_path / f"{prefix}_{step:08d}"
            if not snapshot.exists():
                staging = self.settings.root_path / f".checkpoint_{step:08d}.tmp"
                if staging.exists():
                    raise RuntimeError(f"Snapshot temporário já existe e requer inspeção manual: {staging}")
                shutil.copytree(base, staging)
                os.replace(staging, snapshot)
            return snapshot
        return base

    def list_snapshots(self) -> list[dict[str, Any]]:
        if not self.settings.root_path.exists():
            return []
        result = []
        pattern = "checkpoint_[0-9]*" if self.settings.method == "full" else "checkpoint_lora_*"
        for path in sorted(self.settings.root_path.glob(pattern), reverse=True):
            if not path.is_dir():
                continue
            config = {}
            config_path = path / "config.json"
            if config_path.exists():
                try:
                    config = json.loads(config_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    config = {}
            result.append(
                {
                    "name": path.name,
                    "optimizer_step": config.get("fine_tuning_optimizer_step", 0),
                    "epoch": config.get("fine_tuning_epoch", 0),
                    "method": config.get("fine_tuning_method", self.settings.method),
                    "size_mb": round(
                        sum(item.stat().st_size for item in path.glob("*") if item.is_file()) / 1024**2,
                        2,
                    ),
                    "modified_at": path.stat().st_mtime,
                }
            )
        return result
