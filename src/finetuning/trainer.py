import json
import hashlib
import logging
import math
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
import bitsandbytes as bnb
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from src.architecture import architecture_from_checkpoint
from src.finetuning.checkpoints import FineTuningCheckpointStore
from src.finetuning.config import FineTuningSettings
from src.finetuning.dataset import (
    DATASET_PIPELINE_SCHEMA_VERSION,
    PreparedData,
    collate_conversation_batch,
    prepare_conversation_data,
)
from src.finetuning.lora import attach_lora, load_adapter, parse_target_modules
from src.model import BitLinear, CustomLLM
from src.tokenizer import tokenizer, validate_tokenizer_metadata

if TYPE_CHECKING:
    from src.finetuning.manager import FineTuningManager

logger = logging.getLogger(__name__)


def resolve_fine_tuning_context(requested_context: int, checkpoint_config: dict[str, Any]) -> int:
    """Keep SFT sequences inside the positional limit learned by the base model."""
    checkpoint_context = int(checkpoint_config.get("context_window", 0) or 0)
    if checkpoint_context <= 1:
        raise RuntimeError(
            "O checkpoint base nao registra um context_window valido; "
            "nao e seguro inferir o limite posicional pelo .env do fine-tuning."
        )
    if requested_context == 0:
        return checkpoint_context
    if requested_context <= 1:
        raise RuntimeError("FINE_TUNING_CONTEXT_WINDOW deve ser maior que 1.")
    return min(int(requested_context), checkpoint_context)


def fine_tuning_recipe_id(
    settings: FineTuningSettings, effective_context: int, train_samples: int
) -> str:
    """Fingerprint every setting that changes data ordering or optimizer scheduling."""
    recipe = {
        "dataset": settings.dataset_name,
        "dataset_config": settings.dataset_config,
        "dataset_revision": settings.dataset_revision,
        "train_split": settings.train_split,
        "validation_split": settings.validation_split,
        "validation_fraction": settings.validation_fraction,
        "messages_field": settings.messages_field,
        "max_train_samples": settings.max_train_samples,
        "max_validation_samples": settings.max_validation_samples,
        "train_samples_after_preparation": int(train_samples),
        "context_window": int(effective_context),
        "system_prompt": settings.system_prompt,
        "assistant_only_loss": settings.assistant_only_loss,
        "dataset_pipeline_schema": DATASET_PIPELINE_SCHEMA_VERSION,
        "seed": settings.seed,
        "micro_batch_size": settings.micro_batch_size,
        "global_batch_size": settings.global_batch_size,
        "gradient_accumulation_steps": settings.gradient_accumulation_steps,
        "learning_rate": settings.learning_rate,
        "weight_decay": settings.weight_decay,
        "optimizer_beta1": settings.optimizer_beta1,
        "optimizer_beta2": settings.optimizer_beta2,
        "optimizer_eps": settings.optimizer_eps,
        "warmup_ratio": settings.warmup_ratio,
        "epochs": settings.epochs,
        "method": settings.method,
        "lora_rank": settings.lora_rank if settings.method == "lora" else None,
        "lora_alpha": settings.lora_alpha if settings.method == "lora" else None,
        "lora_dropout": settings.lora_dropout if settings.method == "lora" else None,
        "lora_target_modules": settings.lora_target_modules if settings.method == "lora" else None,
    }
    encoded = json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _accuracy(
    logits: torch.Tensor, labels: torch.Tensor, sample_tokens: int
) -> tuple[float, float]:
    with torch.no_grad():
        flat_labels = labels.reshape(-1)
        mask = flat_labels != -100
        if not mask.any():
            return 0.0, 0.0
        flat_logits = logits.reshape(-1, logits.size(-1))[mask]
        targets = flat_labels[mask]
        if flat_logits.size(0) > sample_tokens:
            indices = torch.randperm(flat_logits.size(0), device=flat_logits.device)[:sample_tokens]
            flat_logits = flat_logits[indices]
            targets = targets[indices]
        topk = flat_logits.topk(min(5, flat_logits.size(-1)), dim=-1).indices
        top1 = (topk[:, 0] == targets).float().mean().item()
        top5 = (topk == targets.unsqueeze(-1)).any(dim=-1).float().mean().item()
        return top1, top5


def _weight_norm(model: torch.nn.Module) -> float:
    with torch.no_grad():
        norm_sq = 0.0
        count = 0
        for parameter in model.parameters():
            if parameter.requires_grad:
                value = float(parameter.detach().float().norm(2).item())
                norm_sq += value * value
                count += 1
        return math.sqrt(norm_sq / max(count, 1))


def _ternary_distribution(model: torch.nn.Module) -> tuple[float, float, float]:
    neg = zero = pos = total = 0
    with torch.no_grad():
        for module in model.modules():
            if not isinstance(module, BitLinear) or not module.quantized:
                continue
            weight = module.weight.detach()
            scale = weight.abs().mean().clamp(min=1e-5)
            quantized = torch.round(weight / scale).clamp(-1, 1)
            neg += int((quantized == -1).sum().item())
            zero += int((quantized == 0).sum().item())
            pos += int((quantized == 1).sum().item())
            total += quantized.numel()
    if total == 0:
        return 0.0, 0.0, 0.0
    return neg * 100 / total, zero * 100 / total, pos * 100 / total


class LocalFineTuningTrainer:
    def __init__(
        self,
        settings: FineTuningSettings,
        store: FineTuningCheckpointStore,
        manager: "FineTuningManager",
    ):
        self.settings = settings
        self.store = store
        self.manager = manager

    def _load_state(self) -> dict[str, Any]:
        config_path = self.settings.base_path / "config.json"
        if not config_path.exists():
            return {}
        return json.loads(config_path.read_text(encoding="utf-8"))

    def _build_model(self, device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
        config = self._load_state()
        d_model = int(config.get("model_d_model", 0))
        num_layers = int(config.get("model_layers", 0))
        if d_model <= 0 or num_layers <= 0:
            raise RuntimeError("O config.json não contém a arquitetura completa do modelo.")
        weights_path = (
            self.settings.base_model_path
            if self.settings.method == "lora" and self.settings.base_model_path.is_file()
            else self.settings.base_path / "model.safetensor"
        )
        weights = load_file(str(weights_path), device=str(device))
        expected_vocab = int(weights["embeddings.weight"].shape[0])
        validate_tokenizer_metadata(config, expected_vocab)
        for name, tensor in weights.items():
            if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                raise RuntimeError(f"O checkpoint contém valores não finitos no tensor {name}.")
        architecture = architecture_from_checkpoint(config, weights)
        model = CustomLLM(
            vocab_size=expected_vocab,
            d_model=d_model,
            num_layers=num_layers,
            architecture=architecture,
            num_heads=int(config["num_heads"]) if config.get("num_heads") else None,
            head_dim=int(config["head_dim"]) if config.get("head_dim") else None,
            num_kv_heads=int(config.get("num_kv_heads", 0) or 0),
            rope_theta=float(config.get("rope_theta", 10000.0)),
            max_context=int(config.get("context_window", self.settings.context_window)),
            ffn_hidden_size=int(config.get("ffn_hidden_size", 0) or 0),
            ffn_multiplier=float(config.get("ffn_multiplier", 4.0)),
            ffn_multiple_of=int(config.get("ffn_multiple_of", 64)),
            norm_eps=float(config.get("norm_eps", 1e-6)),
            init_std=float(config.get("init_std", 0.02)),
        ).to(device=device, dtype=dtype)
        missing, unexpected = model.load_state_dict(weights, strict=False)
        allowed_missing = {"lm_head.weight"}
        if set(missing) - allowed_missing or unexpected:
            raise RuntimeError(f"Checkpoint incompatível. Ausentes={missing}; inesperados={unexpected}")
        if self.settings.method == "lora":
            report = attach_lora(
                model,
                rank=self.settings.lora_rank,
                alpha=self.settings.lora_alpha,
                dropout=self.settings.lora_dropout,
                target_modules=parse_target_modules(self.settings.lora_target_modules),
            )
            load_adapter(model, self.settings.adapter_path)
            self.manager.set_adapter_report(
                {
                    "method": "lora",
                    "target_modules": list(report.target_modules),
                    "trainable_parameters": report.trainable_parameters,
                    "total_parameters": report.total_parameters,
                    "trainable_percentage": round(report.trainable_percentage, 6),
                }
            )
        else:
            self.manager.set_adapter_report(
                {
                    "method": "full",
                    "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                    "total_parameters": sum(p.numel() for p in model.parameters()),
                    "trainable_percentage": 100.0,
                }
            )
        # Gradient Checkpointing: recomputa o forward no backward p/ economizar VRAM.
        # Só afeta o treino; prefill/recurrent (inferência) não passam por aqui.
        model.gradient_checkpointing = self.settings.activation_checkpointing
        return model

    def _evaluate(
        self,
        model: torch.nn.Module,
        dataset: PreparedData,
        device: torch.device,
        dtype: torch.dtype,
    ) -> float:
        pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
        loader = DataLoader(
            dataset.validation,
            batch_size=self.settings.micro_batch_size,
            shuffle=False,
            collate_fn=lambda samples: collate_conversation_batch(samples, pad_id),
        )
        total_loss = 0.0
        supervised_tokens = 0
        model.eval()
        with torch.no_grad():
            for index, batch in enumerate(loader):
                if self.settings.evaluation_batches > 0 and index >= self.settings.evaluation_batches:
                    break
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                inputs = input_ids[:, :-1]
                targets = labels[:, 1:]
                with torch.amp.autocast(
                    device_type=device.type,
                    dtype=dtype,
                    enabled=device.type == "cuda" and dtype != torch.float32,
                ):
                    logits = model(inputs)
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        targets.reshape(-1),
                        ignore_index=-100,
                        reduction="sum",
                    )
                if not torch.isfinite(loss):
                    raise RuntimeError("A loss de validação tornou-se NaN ou infinita.")
                total_loss += float(loss.item())
                supervised_tokens += int(targets.ne(-100).sum().item())
        model.train()
        return total_loss / max(supervised_tokens, 1)

    def run(self) -> str:
        torch.manual_seed(self.settings.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.settings.seed)

        device_name = self.settings.device if self.settings.device == "cuda" and torch.cuda.is_available() else "cpu"
        device = torch.device(device_name)
        dtype = (
            torch.bfloat16
            if device.type == "cuda" and self.settings.precision == "bf16"
            else torch.float16
            if device.type == "cuda" and self.settings.precision == "fp16"
            else torch.float32
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        saved_state = self._load_state()
        effective_context = resolve_fine_tuning_context(
            self.settings.context_window, saved_state
        )
        if self.settings.context_window == 0:
            logger.info(
                "Contexto SFT automatico: usando os %d tokens registrados no checkpoint.",
                effective_context,
            )
        elif effective_context != self.settings.context_window:
            logger.warning(
                "FINE_TUNING_CONTEXT_WINDOW=%d excede o contexto do checkpoint (%d); "
                "o dataset SFT sera preparado com %d tokens para preservar o mesmo modelo/RoPE.",
                self.settings.context_window,
                int(saved_state["context_window"]),
                effective_context,
            )
        data_settings = self.settings.model_copy(update={"context_window": effective_context})
        data = prepare_conversation_data(tokenizer, data_settings)
        self.manager.set_dataset_report(data.report)
        recipe_id = fine_tuning_recipe_id(data_settings, effective_context, len(data.train))
        previous_steps = int(saved_state.get("fine_tuning_optimizer_step", 0) or 0)
        saved_recipe_id = str(saved_state.get("fine_tuning_recipe_id", ""))
        if previous_steps > 0 and saved_recipe_id and saved_recipe_id != recipe_id:
            raise RuntimeError(
                "A receita de fine-tuning mudou (dataset, split, contexto ou hiperparametros), "
                "mas o diretorio de saida ja contem optimizer/scheduler de outra execucao. "
                "Use um novo FINE_TUNING_ROOT_DIR para nao misturar treinamentos."
            )
        if previous_steps > 0 and not saved_recipe_id:
            legacy_dataset = str(saved_state.get("fine_tuning_dataset", ""))
            legacy_context = int(saved_state.get("fine_tuning_context_window", 0) or 0)
            if legacy_dataset and legacy_dataset != data_settings.dataset_name:
                raise RuntimeError(
                    "O checkpoint SFT legado pertence a outro dataset. "
                    "Use um novo FINE_TUNING_ROOT_DIR."
                )
            if legacy_context and legacy_context != effective_context:
                raise RuntimeError(
                    "O checkpoint SFT legado pertence a outro context_window. "
                    "Use um novo FINE_TUNING_ROOT_DIR."
                )
            logger.warning(
                "Checkpoint SFT legado sem fingerprint completo da receita; "
                "a compatibilidade foi validada pelos campos disponiveis."
            )
        model = self._build_model(device, dtype)
        trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not trainable_parameters:
            raise RuntimeError("O método de fine-tuning não selecionou parâmetros treináveis.")

        if torch.cuda.is_available() and self.settings.use_8bit_optimizer:
            optimizer = bnb.optim.AdamW8bit(
                trainable_parameters,
                lr=self.settings.learning_rate,
                betas=(self.settings.optimizer_beta1, self.settings.optimizer_beta2),
                eps=self.settings.optimizer_eps,
                weight_decay=self.settings.weight_decay,
            )
        else:
            optimizer = torch.optim.AdamW(
                trainable_parameters,
                lr=self.settings.learning_rate,
                betas=(self.settings.optimizer_beta1, self.settings.optimizer_beta2),
                eps=self.settings.optimizer_eps,
                weight_decay=self.settings.weight_decay,
            )

        batches_per_epoch = math.ceil(len(data.train) / self.settings.micro_batch_size)
        optimizer_steps_per_epoch = math.ceil(batches_per_epoch / self.settings.gradient_accumulation_steps)
        total_optimizer_steps = max(1, optimizer_steps_per_epoch * self.settings.epochs)
        warmup_steps = int(total_optimizer_steps * self.settings.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_optimizer_steps)

        optimizer_path = self.settings.base_path / "optimizer.pt"
        scheduler_path = self.settings.base_path / "scheduler.pt"
        if optimizer_path.exists():
            optimizer.load_state_dict(torch.load(optimizer_path, map_location=device, weights_only=True))
        if scheduler_path.exists():
            scheduler.load_state_dict(torch.load(scheduler_path, map_location="cpu", weights_only=True))

        start_epoch = int(saved_state.get("fine_tuning_epoch", 0))
        resume_batch = int(saved_state.get("fine_tuning_batch_in_epoch", 0))
        micro_step = int(saved_state.get("fine_tuning_micro_step", 0))
        optimizer_step = int(saved_state.get("fine_tuning_optimizer_step", 0))
        examples_seen = int(saved_state.get("fine_tuning_examples_seen", 0))
        assistant_tokens_seen = int(saved_state.get("fine_tuning_assistant_tokens_seen", 0))
        if previous_steps > 0 and start_epoch >= self.settings.epochs:
            raise RuntimeError(
                "Este fine-tuning ja concluiu todas as epocas configuradas. "
                "Inicie uma nova receita em outro FINE_TUNING_ROOT_DIR; "
                "reabrir este checkpoint manteria o scheduler em learning rate zero."
            )

        if self.settings.torch_compile:
            model = torch.compile(model, fullgraph=False)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        started_at = time.time()
        interval_started = started_at
        interval_tokens = 0
        interval_sequences = 0
        accumulated_loss = 0.0
        accumulated_micro_steps = 0
        loss_ema: float | None = None
        last_val_loss: float | None = None
        last_grad_norm = 0.0
        last_weight_norm = 0.0
        ternary = (0.0, 0.0, 0.0)

        def checkpoint(epoch_value: int, batch_value: int, snapshot: bool = False) -> None:
            model_to_save = getattr(model, "_orig_mod", model)
            self.store.save(
                model_to_save,
                optimizer,
                scheduler,
                {
                    "fine_tuning_epoch": epoch_value,
                    "fine_tuning_batch_in_epoch": batch_value,
                    "fine_tuning_micro_step": micro_step,
                    "fine_tuning_optimizer_step": optimizer_step,
                    "fine_tuning_examples_seen": examples_seen,
                    "fine_tuning_assistant_tokens_seen": assistant_tokens_seen,
                    "fine_tuning_dataset": self.settings.dataset_name,
                    "fine_tuning_dataset_config": self.settings.dataset_config,
                    "fine_tuning_dataset_revision": self.settings.dataset_revision,
                    "fine_tuning_recipe_id": recipe_id,
                    "fine_tuning_dataset_pipeline_schema": DATASET_PIPELINE_SCHEMA_VERSION,
                    "fine_tuning_learning_rate": self.settings.learning_rate,
                    "fine_tuning_context_window": effective_context,
                    "fine_tuning_method": self.settings.method,
                },
                create_snapshot=snapshot,
            )

        for epoch in range(start_epoch, self.settings.epochs):
            generator = torch.Generator().manual_seed(self.settings.seed + epoch)
            pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
            loader = DataLoader(
                data.train,
                batch_size=self.settings.micro_batch_size,
                shuffle=True,
                generator=generator,
                pin_memory=device.type == "cuda",
                num_workers=0,
                collate_fn=lambda samples: collate_conversation_batch(samples, pad_id),
            )
            for batch_index, batch in enumerate(loader):
                if epoch == start_epoch and batch_index < resume_batch:
                    continue

                input_ids = batch["input_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                inputs = input_ids[:, :-1]
                targets = labels[:, 1:]
                supervised_tokens = int(batch["supervised_tokens"].sum().item())
                actual_tokens = int(batch["attention_mask"].sum().item())

                with torch.amp.autocast(
                    device_type=device.type,
                    dtype=dtype,
                    enabled=device.type == "cuda" and dtype != torch.float32,
                ):
                    logits = model(inputs)
                    raw_loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
                    )
                    loss = raw_loss / self.settings.gradient_accumulation_steps
                if not torch.isfinite(raw_loss):
                    raise RuntimeError("A loss de fine-tuning tornou-se NaN ou infinita.")
                loss.backward()

                micro_step += 1
                examples_seen += inputs.size(0)
                assistant_tokens_seen += supervised_tokens
                interval_tokens += actual_tokens
                interval_sequences += inputs.size(0)
                accumulated_loss += float(raw_loss.item())
                accumulated_micro_steps += 1
                top1, top5 = _accuracy(
                    logits, targets, self.settings.metrics_accuracy_sample_tokens
                )

                is_optimizer_boundary = (batch_index + 1) % self.settings.gradient_accumulation_steps == 0
                is_last_batch = batch_index + 1 == len(loader)
                if is_optimizer_boundary or is_last_batch:
                    grad = torch.nn.utils.clip_grad_norm_(
                        trainable_parameters, self.settings.max_grad_norm, error_if_nonfinite=True
                    )
                    last_grad_norm = float(grad.item() if isinstance(grad, torch.Tensor) else grad)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_step += 1

                    if optimizer_step % self.settings.eval_steps == 0:
                        last_val_loss = self._evaluate(model, data, device, dtype)

                    if optimizer_step % self.settings.checkpoint_steps == 0:
                        checkpoint(epoch, batch_index + 1, snapshot=True)

                    if self.manager.consume_checkpoint_request():
                        checkpoint(epoch, batch_index + 1, snapshot=True)

                    if optimizer_step % self.settings.log_steps == 0:
                        now = time.time()
                        interval = max(now - interval_started, 1e-6)
                        current_loss = accumulated_loss / max(accumulated_micro_steps, 1)
                        loss_ema = current_loss if loss_ema is None else (0.9 * loss_ema + 0.1 * current_loss)
                        if optimizer_step % self.settings.weight_norm_steps == 0:
                            last_weight_norm = _weight_norm(getattr(model, "_orig_mod", model))
                        if optimizer_step % self.settings.ternary_metrics_steps == 0:
                            ternary = _ternary_distribution(getattr(model, "_orig_mod", model))

                        elapsed = now - started_at
                        progress = min(optimizer_step / total_optimizer_steps, 1.0)
                        rate = optimizer_step / max(elapsed, 1e-6)
                        eta = max(total_optimizer_steps - optimizer_step, 0) / max(rate, 1e-6)
                        vram_allocated = vram_reserved = vram_peak = 0.0
                        gpu_utilization = 0
                        if device.type == "cuda":
                            vram_allocated = torch.cuda.memory_allocated(device) / 1024**2
                            vram_reserved = torch.cuda.memory_reserved(device) / 1024**2
                            vram_peak = torch.cuda.max_memory_allocated(device) / 1024**2
                            try:
                                gpu_utilization = int(torch.cuda.utilization(device))
                            except Exception:
                                gpu_utilization = 0

                        metrics = {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "elapsed_seconds": round(elapsed, 2),
                            "epoch": epoch + 1,
                            "micro_step": micro_step,
                            "optimizer_step": optimizer_step,
                            "examples_seen": examples_seen,
                            "assistant_tokens_seen": assistant_tokens_seen,
                            "loss": round(current_loss, 6),
                            "loss_ema": round(loss_ema, 6),
                            "val_loss": round(last_val_loss, 6) if last_val_loss is not None else "",
                            "perplexity": round(math.exp(min(current_loss, 20)), 4),
                            "val_perplexity": round(math.exp(min(last_val_loss, 20)), 4) if last_val_loss is not None else "",
                            "generalization_gap": round(last_val_loss - current_loss, 6) if last_val_loss is not None else "",
                            "top1_accuracy": round(top1, 6),
                            "top5_accuracy": round(top5, 6),
                            "learning_rate": optimizer.param_groups[0]["lr"],
                            "gradient_norm": round(last_grad_norm, 6),
                            "weight_norm": round(last_weight_norm, 6),
                            "tokens_per_second": round(interval_tokens / interval, 2),
                            "sequences_per_second": round(interval_sequences / interval, 2),
                            "supervised_token_ratio": round(supervised_tokens / max(actual_tokens, 1), 6),
                            "vram_allocated_mb": round(vram_allocated, 2),
                            "vram_reserved_mb": round(vram_reserved, 2),
                            "vram_peak_mb": round(vram_peak, 2),
                            "gpu_utilization_pct": gpu_utilization,
                            "ternary_neg1_pct": round(ternary[0], 3),
                            "ternary_zero_pct": round(ternary[1], 3),
                            "ternary_pos1_pct": round(ternary[2], 3),
                            "progress_pct": round(progress * 100, 3),
                            "eta_seconds": round(eta, 2),
                        }
                        self.manager.emit_metrics(metrics)
                        interval_started = now
                        interval_tokens = 0
                        interval_sequences = 0
                        accumulated_loss = 0.0
                        accumulated_micro_steps = 0

                    if self.manager.pause_requested:
                        checkpoint(epoch, batch_index + 1, snapshot=True)
                        return "paused"
                    if self.manager.stop_requested:
                        checkpoint(epoch, batch_index + 1, snapshot=True)
                        return "stopped"

            resume_batch = 0
            checkpoint(epoch + 1, 0, snapshot=True)

        checkpoint(self.settings.epochs, 0, snapshot=True)
        return "completed"
