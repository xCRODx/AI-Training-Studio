from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_d_model: int = 1024
    model_layers: int = 24
    model_experts: int = 1
    # 0 = derive from MODEL_HEAD_DIM; explicit values are validated.
    model_num_heads: int = 0
    model_head_dim: int = 64
    # 0 = derive from multiplier and round to MODEL_FFN_MULTIPLE_OF.
    model_ffn_hidden_size: int = 0
    model_ffn_multiplier: float = 4.0
    model_ffn_multiple_of: int = 64
    model_norm_eps: float = 1e-6
    model_init_std: float = 0.02
    context_window: int = 512
    train_batch_size: int = 2
    train_lr: float = 3e-4
    train_epochs: int = 1
    train_steps: int = 500000
    weight_decay: float = 0.1
    warmup_steps: int = 5000
    checkpoint_save_steps: int = 1000
    max_grad_norm: float = 1.0
    optimizer_beta1: float = 0.9
    optimizer_beta2: float = 0.95
    optimizer_eps: float = 1e-6
    gradient_accumulation_steps: int = 16
    train_precision: str = "bf16"
    gradient_checkpointing: bool = True
    use_8bit_optimizer: bool = True
    inference_device: str = "cuda"
    inference_auto_temperature: float = 0.7
    inference_auto_top_p: float = 0.9
    inference_auto_top_k: int = 50
    inference_auto_repetition_penalty: float = 1.05

    tokenizer_name: str = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    tokenizer_revision: str = ""
    dataset_name: str = "HuggingFaceTB/cosmopedia"
    dataset_config_name: str = ""
    dataset_revision: str = ""
    dataset_split: str = "train"
    dataset_text_field: str = "text"
    # Filtro opcional e declarativo. Vazio desativa o filtro.
    dataset_quality_field: str = ""
    dataset_quality_value: str = ""
    hf_token: str = ""

    # Persistência e validação são configuração da execução, não limites do modelo.
    # Vazio usa sempre checkpoints/base. Veja ``settings.checkpoint_path``.
    checkpoint_dir: str = ""
    validation_holdout_fraction: float = 0.005
    validation_batches: int = 4
    validation_eval_steps: int = 50
    metrics_log_steps: int = 10
    metrics_accuracy_sample_tokens: int = 1000
    metrics_preview_tokens: int = 30
    dataset_estimation_samples: int = 100
    # False garante uma única passagem: ao chegar ao fim, salva e encerra.
    repeat_dataset: bool = False

    # ---------------------------------------------------------------
    # Arquitetura do modelo
    #
    # USE_BITNET e USE_RETNET definem qual arquitetura será usada:
    #   bitnet=true  retnet=true  -> retbit      (híbrido legado: RetNet + BitLinear)
    #   bitnet=true  retnet=false -> bitnet      (transformer ternário + RoPE + KV cache)
    #   bitnet=false retnet=true  -> retnet      (RetNet com projeções float)
    #   bitnet=false retnet=false -> transformer (attention padrão + KV cache tradicional)
    #
    # As flags são FALSAS por padrão: sem nenhuma delas, o transformer
    # tradicional (Llama-style) é usado.
    # ---------------------------------------------------------------
    use_bitnet: bool = False
    use_retnet: bool = False

    # GQA: número de KV heads (0 = automático; deve dividir o nº de Q heads).
    model_kv_heads: int = 0
    # RoPE: frequência base (padrão LLaMA) para transformer/bitnet.
    rope_theta: float = 10000.0

    overfit_mode: bool = False
    overfit_text: str = "O rato roeu a roupa do rei de roma e a rainha riu da roupa roída."

    @property
    def architecture(self) -> str:
        """Architecture resolved from the two boolean flags."""
        if self.use_bitnet and self.use_retnet:
            return "retbit"
        if self.use_bitnet:
            return "bitnet"
        if self.use_retnet:
            return "retnet"
        return "transformer"

    @property
    def architecture_label(self) -> str:
        from src.architecture import architecture_label as _label

        return _label(self.architecture)

    @property
    def checkpoint_path(self):
        from src.paths import resolve_pretraining_checkpoint_dir

        return resolve_pretraining_checkpoint_dir(self.checkpoint_dir)

    @model_validator(mode="after")
    def validate_model_recipe(self):
        if self.model_experts != 1:
            raise ValueError(
                "MODEL_EXPERTS não está implementado: use 1. O painel não fingirá criar MoE."
            )
        if self.model_d_model <= 0 or self.model_layers <= 0 or self.context_window <= 1:
            raise ValueError("Dimensões do modelo e CONTEXT_WINDOW devem ser positivas.")
        if self.model_num_heads > 0:
            if self.model_d_model % self.model_num_heads != 0:
                raise ValueError("MODEL_NUM_HEADS deve dividir MODEL_D_MODEL.")
        elif self.model_head_dim <= 0 or self.model_d_model % self.model_head_dim != 0:
            raise ValueError("MODEL_HEAD_DIM deve ser positivo e dividir MODEL_D_MODEL.")
        if self.model_ffn_hidden_size <= 0 and (
            self.model_ffn_multiplier <= 0 or self.model_ffn_multiple_of <= 0
        ):
            raise ValueError("A configuração automática da FFN deve usar valores positivos.")
        if not 0 <= self.validation_holdout_fraction < 1:
            raise ValueError("VALIDATION_HOLDOUT_FRACTION deve estar no intervalo [0, 1).")
        positive_intervals = {
            "VALIDATION_BATCHES": self.validation_batches,
            "VALIDATION_EVAL_STEPS": self.validation_eval_steps,
            "METRICS_LOG_STEPS": self.metrics_log_steps,
            "METRICS_ACCURACY_SAMPLE_TOKENS": self.metrics_accuracy_sample_tokens,
            "METRICS_PREVIEW_TOKENS": self.metrics_preview_tokens,
            "DATASET_ESTIMATION_SAMPLES": self.dataset_estimation_samples,
        }
        invalid = [name for name, value in positive_intervals.items() if value <= 0]
        if invalid:
            raise ValueError(f"Configurações devem ser > 0: {', '.join(invalid)}")
        if not 0 < self.optimizer_beta1 < 1 or not 0 < self.optimizer_beta2 < 1:
            raise ValueError("OPTIMIZER_BETA1/BETA2 devem estar entre 0 e 1.")
        if self.optimizer_eps <= 0:
            raise ValueError("OPTIMIZER_EPS deve ser positivo.")
        return self

    model_config = SettingsConfigDict(
        env_file=(".env.local", ".env"),
        env_file_encoding="utf-8",
        env_prefix="",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()
