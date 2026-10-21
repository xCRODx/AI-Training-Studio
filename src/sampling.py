from dataclasses import dataclass
from typing import Literal


SamplingMode = Literal["auto", "greedy", "sample"]


@dataclass(frozen=True)
class SamplingParameters:
    """Resolved decoding policy; numeric zero never carries hidden semantics here."""

    mode: Literal["greedy", "sample"]
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float


def resolve_sampling(
    mode: SamplingMode,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    auto_temperature: float,
    auto_top_p: float,
    auto_top_k: int,
    auto_repetition_penalty: float,
) -> SamplingParameters:
    """Make auto, greedy and sampling unambiguous at the API boundary.

    In ``auto`` mode, a positive temperature is treated as an explicit legacy
    sampling request. A zero temperature selects the configured auto policy.
    """
    if mode == "greedy":
        return SamplingParameters("greedy", 0.0, 1.0, 1, 1.0)

    if mode == "auto" and temperature <= 0:
        temperature = auto_temperature
        top_p = auto_top_p
        top_k = auto_top_k
        repetition_penalty = auto_repetition_penalty

    if temperature <= 0:
        raise ValueError("Modo sample exige temperature > 0; use sampling_mode='greedy'.")
    if not 0 < top_p <= 1:
        raise ValueError("top_p deve estar no intervalo (0, 1].")
    if top_k < 0:
        raise ValueError("top_k não pode ser negativo; 0 significa vocabulário completo.")
    if repetition_penalty <= 0:
        raise ValueError("repetition_penalty deve ser positivo.")
    return SamplingParameters(
        "sample",
        float(temperature),
        float(top_p),
        int(top_k),
        float(repetition_penalty),
    )
