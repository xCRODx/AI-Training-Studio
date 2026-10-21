import json
import threading
from pathlib import Path

import torch
from safetensors.torch import load_file

from src.architecture import architecture_from_checkpoint
from src.model import CustomLLM
from src.tokenizer import tokenizer
from src.tokenizer import validate_tokenizer_metadata


class BaseCheckpointInference:
    """Read-only, lazily loaded inference snapshot for checkpoints/base."""

    def __init__(self, checkpoint_dir: str | Path, context_window: int):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.context_window = context_window
        self.lock = threading.RLock()
        self._models: dict[str, tuple[int, CustomLLM]] = {}

    def load(self, device_name: str) -> CustomLLM:
        with self.lock:
            model_path = self.checkpoint_dir / "model.safetensor"
            config_path = self.checkpoint_dir / "config.json"
            if not model_path.is_file() or not config_path.is_file():
                raise RuntimeError("Nenhum checkpoint base válido foi encontrado para inferência.")

            signature = model_path.stat().st_mtime_ns
            cached = self._models.get(device_name)
            if cached and cached[0] == signature:
                return cached[1]

            config = json.loads(config_path.read_text(encoding="utf-8"))
            weights = load_file(str(model_path), device="cpu")
            expected_vocab = int(weights["embeddings.weight"].shape[0])
            validate_tokenizer_metadata(config, expected_vocab)

            device = torch.device(device_name)
            dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
            architecture = architecture_from_checkpoint(config, weights)
            model = CustomLLM(
                vocab_size=expected_vocab,
                d_model=int(config["model_d_model"]),
                num_layers=int(config["model_layers"]),
                architecture=architecture,
                num_heads=int(config["num_heads"]) if config.get("num_heads") else None,
                head_dim=int(config["head_dim"]) if config.get("head_dim") else None,
                num_kv_heads=int(config.get("num_kv_heads", 0) or 0),
                rope_theta=float(config.get("rope_theta", 10000.0)),
                max_context=int(config.get("context_window", self.context_window)),
                ffn_hidden_size=int(config.get("ffn_hidden_size", 0) or 0),
                ffn_multiplier=float(config.get("ffn_multiplier", 4.0)),
                ffn_multiple_of=int(config.get("ffn_multiple_of", 64)),
                norm_eps=float(config.get("norm_eps", 1e-6)),
                init_std=float(config.get("init_std", 0.02)),
            ).to(device=device, dtype=dtype)
            missing, unexpected = model.load_state_dict(weights, strict=False)
            if set(missing) - {"lm_head.weight"} or unexpected:
                raise RuntimeError(
                    f"Checkpoint base incompatível. Ausentes={missing}; inesperados={unexpected}."
                )
            model.eval()
            model.prepare_for_inference()
            self._models[device_name] = (signature, model)
            return model


def snapshot_live_model(model: CustomLLM, device_name: str) -> CustomLLM:
    """Create a stable inference snapshot without toggling the training model."""
    target_device = torch.device(device_name)
    if next(model.parameters()).is_cuda:
        torch.cuda.synchronize(next(model.parameters()).device)
    target_dtype = torch.bfloat16 if target_device.type == "cuda" else torch.float32
    snapshot = CustomLLM(
        vocab_size=model.embeddings.num_embeddings,
        d_model=model.d_model,
        num_layers=model.num_layers,
        architecture=model.architecture,
        num_heads=model.num_heads,
        head_dim=model.head_dim,
        num_kv_heads=model.num_kv_heads,
        rope_theta=model.rope_theta,
        max_context=model.max_context,
        ffn_hidden_size=model.ffn_hidden_size,
        ffn_multiplier=model.ffn_multiplier,
        ffn_multiple_of=model.ffn_multiple_of,
        norm_eps=model.norm_eps,
        init_std=model.init_std,
    ).to(device=target_device, dtype=target_dtype)
    missing, unexpected = snapshot.load_state_dict(model.state_dict(), strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Falha ao criar snapshot de inferência. Ausentes={missing}; inesperados={unexpected}."
        )
    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
    snapshot.eval()
    snapshot.prepare_for_inference()
    return snapshot
