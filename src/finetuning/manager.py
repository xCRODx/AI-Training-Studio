import csv
import json
import logging
import threading
from collections import deque
from pathlib import Path
from typing import Any, Callable

from src.training_lock import TrainingLease

from src.finetuning.backends import get_training_backend
from src.finetuning.checkpoints import FineTuningCheckpointStore
from src.finetuning.config import FineTuningSettings, fine_tuning_settings

logger = logging.getLogger(__name__)


class FineTuningManager:
    ACTIVE_STATES = {"starting", "running", "pausing", "stopping"}

    def __init__(self, settings: FineTuningSettings):
        self.settings = settings
        self.store = FineTuningCheckpointStore(settings)
        self.backend = get_training_backend(settings.engine)
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._pause_event = threading.Event()
        self._stop_event = threading.Event()
        self._checkpoint_event = threading.Event()
        self._resource_guard: Callable[[], bool] = lambda: True
        self._training_lease: TrainingLease | None = None
        self._state = "idle"
        self._message = "Fine-tuning pronto para inicialização."
        self._error: str | None = None
        self._latest_metrics: dict[str, Any] = {}
        self._dataset_report: dict[str, Any] = {}
        self._adapter_report: dict[str, Any] = {}
        self._history: deque[dict[str, Any]] = deque(maxlen=500)
        self._model_cache_key: tuple[float, float, int] | None = None
        self._model_cache: dict[str, Any] | None = None

    def set_resource_guard(self, guard: Callable[[], bool]) -> None:
        self._resource_guard = guard

    def set_training_lease(self, lease: TrainingLease) -> None:
        self._training_lease = lease

    @property
    def pause_requested(self) -> bool:
        return self._pause_event.is_set()

    @property
    def stop_requested(self) -> bool:
        return self._stop_event.is_set()

    def is_active(self) -> bool:
        with self._lock:
            return self._state in self.ACTIVE_STATES

    def _launch(self) -> dict[str, Any]:
        if not self._resource_guard():
            raise RuntimeError("Pare ou pause o pré-treino antes de iniciar o fine-tuning.")
        if self._training_lease is not None:
            self._training_lease.acquire("fine-tuning")
        try:
            self.store.bootstrap()
        except Exception:
            if self._training_lease is not None:
                self._training_lease.release()
            raise
        self._pause_event.clear()
        self._stop_event.clear()
        self._checkpoint_event.clear()
        self._error = None
        self._state = "starting"
        self._message = "Preparando modelo e dataset conversacional..."
        self._thread = self.backend.launch(self._worker)
        return self.status()

    def _configure(self, options: dict[str, Any]) -> None:
        if not options:
            return
        data = self.settings.model_dump()
        data.update({key: value for key, value in options.items() if value is not None})
        updated = FineTuningSettings(_env_file=None, **data)
        profile_changed = updated.base_path != self.settings.base_path
        self.settings = updated
        self.store = FineTuningCheckpointStore(updated)
        self.backend = get_training_backend(updated.engine)
        if profile_changed:
            self._latest_metrics = {}
            self._dataset_report = {}
            self._adapter_report = {}
            self._history.clear()
            self._model_cache_key = None
            self._model_cache = None

    def start(self, **options: Any) -> dict[str, Any]:
        with self._lock:
            if self._state in self.ACTIVE_STATES:
                raise RuntimeError("O fine-tuning já está ativo.")
            if self._state == "paused":
                raise RuntimeError("Retome ou pare o fine-tuning pausado antes de trocar o método.")
            self._configure(options)
            return self._launch()

    def resume(self) -> dict[str, Any]:
        with self._lock:
            if self._state != "paused":
                raise RuntimeError("Não existe um fine-tuning pausado para retomar.")
            return self._launch()

    def request_pause(self) -> dict[str, Any]:
        with self._lock:
            if self._state not in {"starting", "running"}:
                raise RuntimeError("O fine-tuning não está rodando.")
            self._state = "pausing"
            self._message = "Pausa solicitada; salvando no próximo optimizer step."
            self._pause_event.set()
            return self.status()

    def request_stop(self) -> dict[str, Any]:
        with self._lock:
            if self._state == "paused":
                self._state = "stopped"
                self._message = "Fine-tuning parado. O checkpoint pausado foi preservado."
                return self.status()
            if self._state not in {"starting", "running", "pausing"}:
                raise RuntimeError("O fine-tuning não está ativo.")
            self._pause_event.clear()
            self._state = "stopping"
            self._message = "Parada solicitada; salvando checkpoint final."
            self._stop_event.set()
            return self.status()

    def request_checkpoint(self) -> dict[str, Any]:
        with self._lock:
            if self._state != "running":
                raise RuntimeError("Inicie o fine-tuning antes de criar um checkpoint.")
            self._checkpoint_event.set()
            self._message = "Checkpoint solicitado para o próximo optimizer step."
            return self.status()

    def consume_checkpoint_request(self) -> bool:
        if self._checkpoint_event.is_set():
            self._checkpoint_event.clear()
            return True
        return False

    def _worker(self) -> None:
        try:
            from src.finetuning.trainer import LocalFineTuningTrainer

            with self._lock:
                self._state = "running"
                self._message = "Fine-tuning em execução."
            outcome = LocalFineTuningTrainer(self.settings, self.store, self).run()
            with self._lock:
                self._state = outcome
                messages = {
                    "paused": "Fine-tuning pausado com checkpoint salvo.",
                    "stopped": "Fine-tuning parado com checkpoint salvo.",
                    "completed": "Fine-tuning concluído com sucesso.",
                }
                self._message = messages.get(outcome, outcome)
        except Exception as exc:
            logger.exception("Falha no fine-tuning")
            with self._lock:
                self._state = "error"
                self._error = str(exc)
                self._message = "O fine-tuning encontrou um erro."
        finally:
            if self._training_lease is not None:
                self._training_lease.release()
            with self._lock:
                self._thread = None

    def emit_metrics(self, metrics: dict[str, Any]) -> None:
        self.store.append_metrics(metrics)
        with self._lock:
            self._latest_metrics = dict(metrics)
            self._history.append(dict(metrics))

    def set_dataset_report(self, report: dict[str, Any]) -> None:
        with self._lock:
            self._dataset_report = report

    def set_adapter_report(self, report: dict[str, Any]) -> None:
        with self._lock:
            self._adapter_report = dict(report)

    def history(self, limit: int = 300) -> list[dict[str, Any]]:
        with self._lock:
            memory = list(self._history)[-limit:]
        if memory:
            return memory
        path = self.settings.metrics_path
        if not path.exists():
            return []
        with path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))[-limit:]
        numeric_fields = {
            "elapsed_seconds", "epoch", "micro_step", "optimizer_step", "examples_seen",
            "assistant_tokens_seen", "loss", "loss_ema", "val_loss", "perplexity",
            "val_perplexity", "generalization_gap", "top1_accuracy", "top5_accuracy",
            "learning_rate", "gradient_norm", "weight_norm", "tokens_per_second",
            "sequences_per_second", "supervised_token_ratio", "vram_allocated_mb",
            "vram_reserved_mb", "vram_peak_mb", "gpu_utilization_pct", "ternary_neg1_pct",
            "ternary_zero_pct", "ternary_pos1_pct", "progress_pct", "eta_seconds",
        }
        for row in rows:
            for field in numeric_fields:
                value = row.get(field, "")
                if value not in {"", None}:
                    try:
                        row[field] = float(value)
                    except ValueError:
                        pass
        return rows

    def _model_status(self) -> dict[str, Any]:
        base = self.settings.base_path
        ready = (base / "model.safetensor").is_file() and (base / "config.json").is_file()
        result: dict[str, Any] = {"ready": ready, "path": str(base), "valid": False}
        if ready:
            model_path = base / "model.safetensor"
            config_path = base / "config.json"
            cache_key = (model_path.stat().st_mtime, config_path.stat().st_mtime, model_path.stat().st_size)
            if self._model_cache_key == cache_key and self._model_cache is not None:
                return dict(self._model_cache)
            try:
                details = self.store.validate_model_files(base)
                result.update({key: value for key, value in details.items() if key != "config"})
                result["valid"] = True
            except Exception as exc:
                result["error"] = str(exc)
            self._model_cache_key = cache_key
            self._model_cache = dict(result)
        return result

    def _checkpoint_architecture(self) -> dict[str, Any]:
        """Architecture read from the base_finetuning config (same as pre-train).

        Quando ainda não existe base_finetuning, reflete a arquitetura do .env
        (usada no próximo bootstrap) para os menus mostrarem o alvo correto.
        """
        from src.architecture import Architecture, LEGACY_ARCHITECTURE, architecture_label, is_attention

        base_config_path = self.settings.base_path / "config.json"
        source_config_path = self.settings.source_path / "config.json"
        config_path = (
            base_config_path
            if base_config_path.is_file()
            else source_config_path
        )
        config: dict[str, Any] = {}
        if config_path.is_file():
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                config = {}
        value = config.get("architecture")
        if value is None:
            from src.config import settings as pretrain_settings

            arch = Architecture(pretrain_settings.architecture)
            source = "env"
        else:
            try:
                arch = Architecture(value)
            except ValueError:
                arch = LEGACY_ARCHITECTURE
            source = "checkpoint" if config_path == base_config_path else "source_checkpoint"
        return {
            "architecture": arch.value,
            "label": architecture_label(arch),
            "source": source,
            "model_d_model": int(config.get("model_d_model", 0) or 0),
            "num_layers": int(config.get("model_layers", 0) or 0),
            "num_heads": int(config.get("num_heads", 0) or 0),
            "head_dim": int(config.get("head_dim", 0) or 0),
            "num_kv_heads": int(config.get("num_kv_heads", 0) or 0),
            "ffn_hidden_size": int(config.get("ffn_hidden_size", 0) or 0),
            "norm_eps": float(config.get("norm_eps", 0) or 0),
            "context_window": int(config.get("context_window", 0) or 0),
            "kv_cache": {
                "enabled": is_attention(arch),
                "training": False,
                "inference": is_attention(arch),
            },
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            architecture = self._checkpoint_architecture()
            checkpoint_context = int(architecture.get("context_window", 0) or 0)
            requested_context = int(self.settings.context_window)
            effective_context = (
                checkpoint_context
                if requested_context == 0
                else min(requested_context, checkpoint_context)
                if checkpoint_context > 0
                else requested_context
            )
            return {
                "state": self._state,
                "message": self._message,
                "error": self._error,
                "latest_metrics": dict(self._latest_metrics),
                "dataset": dict(self._dataset_report),
                "fine_tuning": dict(self._adapter_report),
                "model": self._model_status(),
                "checkpoints": self.store.list_snapshots(),
                "engine": self.backend.name,
                "backend": self.backend.describe(),
                "configuration": {
                    "method": self.settings.method,
                    "dataset_name": self.settings.dataset_name,
                    "context_window": effective_context,
                    "requested_context_window": requested_context,
                    "context_source": "checkpoint" if requested_context == 0 else "configured_limit",
                    "micro_batch_size": self.settings.micro_batch_size,
                    "global_batch_size": self.settings.global_batch_size,
                    "gradient_accumulation_steps": self.settings.gradient_accumulation_steps,
                    "learning_rate": self.settings.learning_rate,
                    "epochs": self.settings.epochs,
                    "device": self.settings.device,
                    "torch_compile": self.settings.torch_compile,
                    "world_size": self.settings.world_size,
                    "output_path": str(self.settings.base_path),
                    "lora_rank": self.settings.lora_rank,
                    "lora_alpha": self.settings.lora_alpha,
                    "lora_dropout": self.settings.lora_dropout,
                    "lora_target_modules": self.settings.lora_target_modules,
                },
                "architecture": architecture,
            }


finetuning_manager = FineTuningManager(fine_tuning_settings)
