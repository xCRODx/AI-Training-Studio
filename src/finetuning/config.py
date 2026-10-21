from pathlib import Path
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.config import settings as pretrain_settings
from src.paths import resolve_finetuning_root_dir, resolve_finetuning_source_dir, resolve_project_path, PROJECT_ROOT


class FineTuningSettings(BaseSettings):
    """Fine-tuning settings kept separate from the pre-training namespace."""

    engine: Literal["local", "torchtitan"] = "local"
    method: Literal["full", "lora"] = "full"
    # Vazios seguem o checkpoint efetivo do pré-treino e o diretório SFT padrão.
    source_dir: str = ""
    root_dir: str = ""
    data_dir: str = "data/finetuning_en_smoltalk"

    dataset_name: str = "HuggingFaceTB/smoltalk"
    dataset_config: str = "all"
    dataset_revision: str = ""
    train_split: str = "train"
    # Vazio cria um holdout determinístico a partir do split de treino.
    validation_split: str = ""
    validation_fraction: float = 0.01
    messages_field: str = "messages"
    hf_token: str = ""
    # SFT datasets can be very large.  They are sampled before materialization,
    # keeping the preparation bounded and independent from the pre-training data.
    max_train_samples: int = 20_000
    max_validation_samples: int = 1_000

    # 0 = inherit the exact positional limit from the source checkpoint.
    context_window: int = 0
    micro_batch_size: int = 2
    global_batch_size: int = 32
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    epochs: int = 3
    max_grad_norm: float = 1.0
    optimizer_beta1: float = 0.9
    optimizer_beta2: float = 0.95
    optimizer_eps: float = 1e-6
    seed: int = 42

    log_steps: int = 5
    eval_steps: int = 50
    evaluation_batches: int = 8
    checkpoint_steps: int = 100
    max_checkpoints: int = 5
    metrics_accuracy_sample_tokens: int = 2048
    weight_norm_steps: int = 25
    ternary_metrics_steps: int = 100
    assistant_only_loss: bool = True

    device: str = "cuda"
    chat_device: str = "cpu"
    precision: Literal["bf16", "fp16", "fp32"] = "bf16"
    torch_compile: bool = False
    activation_checkpointing: bool = False
    use_8bit_optimizer: bool = True

    # LoRA is implemented natively for the project BitLinear layers.  Full
    # fine-tuning remains the default and retains its historical paths.
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.05
    lora_target_modules: str = (
        "q_proj,k_proj,v_proj,out_proj,g_proj,gate_proj,up_proj,down_proj,fc1,fc2"
    )

    world_size: int = 1
    data_parallel_size: int = 1
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    context_parallel_size: int = 1

    system_prompt: str = (
        "You are a helpful, concise and friendly assistant. Answer the user's request "
        "directly and stop when the answer is complete."
    )

    model_config = SettingsConfigDict(
        env_file=(".env.local", ".env"),
        env_file_encoding="utf-8",
        env_prefix="FINE_TUNING_",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def source_path(self) -> Path:
        return resolve_finetuning_source_dir(self.source_dir, pretrain_settings.checkpoint_dir)

    @property
    def root_path(self) -> Path:
        return resolve_finetuning_root_dir(self.root_dir)

    @property
    def data_path(self) -> Path:
        return resolve_project_path(self.data_dir, PROJECT_ROOT / "data" / "finetuning")

    @property
    def base_path(self) -> Path:
        name = "base_finetuning" if self.method == "full" else "base_finetuning_lora"
        return self.root_path / name

    @property
    def adapter_path(self) -> Path:
        return self.base_path / "adapter_model.safetensors"

    @property
    def base_model_path(self) -> Path:
        return self.base_path / "base_model.safetensors"

    @property
    def metrics_path(self) -> Path:
        return self.base_path / "training_metrics.csv"

    @model_validator(mode="after")
    def validate_parallelism(self):
        if not 0 < self.validation_fraction < 1:
            raise ValueError("FINE_TUNING_VALIDATION_FRACTION deve estar entre 0 e 1.")
        if self.max_train_samples <= 0 or self.max_validation_samples <= 0:
            raise ValueError(
                "FINE_TUNING_MAX_TRAIN_SAMPLES e FINE_TUNING_MAX_VALIDATION_SAMPLES "
                "devem ser positivos para manter o SFT limitado em memória."
            )
        if not 0 < self.optimizer_beta1 < 1 or not 0 < self.optimizer_beta2 < 1:
            raise ValueError("FINE_TUNING_OPTIMIZER_BETA1/BETA2 devem estar entre 0 e 1.")
        if self.optimizer_eps <= 0:
            raise ValueError("FINE_TUNING_OPTIMIZER_EPS deve ser positivo.")
        if self.lora_rank <= 0:
            raise ValueError("FINE_TUNING_LORA_RANK deve ser positivo.")
        if self.lora_alpha <= 0:
            raise ValueError("FINE_TUNING_LORA_ALPHA deve ser positivo.")
        if not 0 <= self.lora_dropout < 1:
            raise ValueError("FINE_TUNING_LORA_DROPOUT deve estar entre 0 (inclusive) e 1.")
        if not self.lora_target_modules.strip():
            raise ValueError("FINE_TUNING_LORA_TARGET_MODULES não pode ficar vazio.")
        if min(
            self.metrics_accuracy_sample_tokens,
            self.weight_norm_steps,
            self.ternary_metrics_steps,
        ) <= 0:
            raise ValueError("Intervalos/amostras de métricas SFT devem ser positivos.")
        calculated_global_batch = (
            self.micro_batch_size * self.gradient_accumulation_steps * self.data_parallel_size
        )
        if self.global_batch_size != calculated_global_batch:
            raise ValueError(
                "FINE_TUNING_GLOBAL_BATCH_SIZE deve ser igual a MICRO_BATCH_SIZE × "
                "GRADIENT_ACCUMULATION_STEPS × DATA_PARALLEL_SIZE."
            )
        parallel_world = (
            self.data_parallel_size
            * self.tensor_parallel_size
            * self.pipeline_parallel_size
            * self.context_parallel_size
        )
        if self.world_size != parallel_world:
            raise ValueError(
                "FINE_TUNING_WORLD_SIZE deve corresponder ao produto dos graus de paralelismo."
            )
        return self


fine_tuning_settings = FineTuningSettings()
