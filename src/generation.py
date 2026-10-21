from __future__ import annotations

import torch

from src.model import AnyState, CustomLLM


@torch.inference_mode()
def prefill_generation(
    model: CustomLLM,
    input_ids: torch.Tensor,
    context_window: int,
) -> tuple[torch.Tensor, torch.Tensor, list[AnyState]]:
    """Run the real model prefill on the visible context window."""
    if input_ids.ndim != 2 or input_ids.size(1) == 0:
        raise ValueError("A inferência requer pelo menos um token de prompt.")
    visible_ids = input_ids[:, -int(context_window) :]
    logits, states = model.prefill(visible_ids)
    return visible_ids, logits, states


@torch.inference_mode()
def advance_generation(
    model: CustomLLM,
    visible_ids: torch.Tensor,
    next_token: torch.Tensor,
    states: list[AnyState],
    context_window: int,
) -> tuple[torch.Tensor, torch.Tensor, list[AnyState], bool]:
    """Advance KV-cache/RetNet state and safely roll over a full context.

    Transformer/BitNet use their KV-cache through ``model.recurrent``. RetNet
    and RetBit use the recurrent retention state through the same API. Once the
    trained context window is full, a fresh prefill of the last window avoids
    out-of-range RoPE positions and gives the same semantics as a full sliding
    window forward.
    """
    visible_ids = torch.cat([visible_ids, next_token], dim=-1)
    if visible_ids.size(1) > int(context_window):
        visible_ids = visible_ids[:, -int(context_window) :]
        logits, states = model.prefill(visible_ids)
        return visible_ids, logits, states, True

    logits, states = model.recurrent(next_token, states)
    return visible_ids, logits, states, False
