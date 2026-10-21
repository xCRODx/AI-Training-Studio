"""LoRA adapters for the project-native ``BitLinear`` layers.

The implementation intentionally lives inside the fine-tuning subsystem so the
base model and pre-training code remain untouched.  Adapters are attached with
forward hooks, which preserves the original checkpoint key names.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from src.model import BitLinear


LORA_PREFIX = "lora_adapter."


class LoRAAdapter(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        dropout: float,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = torch.nn.Dropout(float(dropout))
        self.lora_A = torch.nn.Parameter(
            torch.empty(self.rank, in_features, device=device, dtype=dtype)
        )
        self.lora_B = torch.nn.Parameter(
            torch.zeros(out_features, self.rank, device=device, dtype=dtype)
        )
        torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = torch.nn.functional.linear(self.dropout(inputs), self.lora_A)
        return torch.nn.functional.linear(hidden, self.lora_B) * self.scaling

    def delta_weight(self) -> torch.Tensor:
        return torch.matmul(self.lora_B.float(), self.lora_A.float()) * self.scaling


@dataclass(frozen=True)
class LoRAReport:
    target_modules: tuple[str, ...]
    trainable_parameters: int
    total_parameters: int

    @property
    def trainable_percentage(self) -> float:
        return 100.0 * self.trainable_parameters / max(self.total_parameters, 1)


def parse_target_modules(value: str) -> tuple[str, ...]:
    targets = tuple(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))
    if not targets:
        raise ValueError("LoRA requer ao menos um módulo-alvo.")
    return targets


def _matches_target(name: str, targets: tuple[str, ...]) -> bool:
    leaf = name.rsplit(".", 1)[-1]
    return name in targets or leaf in targets or any(name.endswith(f".{target}") for target in targets)


def attach_lora(
    model: torch.nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float,
    target_modules: tuple[str, ...],
) -> LoRAReport:
    """Freeze the base model and attach trainable LoRA branches."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    matched: list[str] = []
    for name, module in tuple(model.named_modules()):
        if not isinstance(module, BitLinear) or not _matches_target(name, target_modules):
            continue
        if hasattr(module, "lora_adapter"):
            raise RuntimeError(f"O módulo {name} já possui um adaptador LoRA.")
        adapter = LoRAAdapter(
            module.in_features,
            module.out_features,
            rank,
            alpha,
            dropout,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
        module.add_module("lora_adapter", adapter)

        def add_adapter_output(
            adapted_module: BitLinear,
            inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> torch.Tensor:
            return output + adapted_module.lora_adapter(inputs[0]).to(output.dtype)

        module._lora_hook_handle = module.register_forward_hook(add_adapter_output)  # type: ignore[attr-defined]
        matched.append(name)

    if not matched:
        available = [name for name, module in model.named_modules() if isinstance(module, BitLinear)]
        raise RuntimeError(
            "Nenhuma camada BitLinear corresponde aos alvos LoRA. "
            f"Alvos={list(target_modules)}; disponíveis={available[:20]}"
        )

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return LoRAReport(tuple(matched), trainable, total)


def adapter_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().contiguous()
        for key, value in model.state_dict().items()
        if f".{LORA_PREFIX}" in key or key.startswith(LORA_PREFIX)
    }


def load_adapter(model: torch.nn.Module, path: Path) -> None:
    if not path.is_file():
        return
    adapter = load_file(str(path), device="cpu")
    current = model.state_dict()
    unknown = set(adapter) - set(current)
    if unknown:
        raise RuntimeError(f"Adaptador LoRA incompatível; tensores inesperados: {sorted(unknown)}")
    with torch.no_grad():
        for key, value in adapter.items():
            current[key].copy_(value.to(device=current[key].device, dtype=current[key].dtype))


def merged_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return a standard model state dict with LoRA deltas merged into weights."""
    state = {
        key: value.detach().cpu().contiguous()
        for key, value in model.state_dict().items()
        if f".{LORA_PREFIX}" not in key
        and not key.startswith(LORA_PREFIX)
        and key != "lm_head.weight"
    }
    for name, module in model.named_modules():
        adapter = getattr(module, "lora_adapter", None)
        if not isinstance(module, BitLinear) or not isinstance(adapter, LoRAAdapter):
            continue
        weight_key = f"{name}.weight" if name else "weight"
        state[weight_key] = (
            module.weight.detach().float().cpu() + adapter.delta_weight().cpu()
        ).to(module.weight.dtype).detach().contiguous()
    return state


def lora_metadata(model: torch.nn.Module) -> dict[str, Any]:
    modules = []
    for name, module in model.named_modules():
        adapter = getattr(module, "lora_adapter", None)
        if isinstance(adapter, LoRAAdapter):
            modules.append(name)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return {
        "fine_tuning_lora_modules": modules,
        "fine_tuning_trainable_parameters": trainable,
        "fine_tuning_total_parameters": total,
        "fine_tuning_trainable_percentage": round(100.0 * trainable / max(total, 1), 6),
    }
