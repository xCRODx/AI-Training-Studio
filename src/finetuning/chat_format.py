from collections.abc import Iterable, Mapping


ROLE_LABELS = {
    "system": "System",
    "user": "User",
    "assistant": "Assistant",
}


def ensure_system_message(
    messages: Iterable[Mapping[str, str]], system_prompt: str
) -> tuple[list[dict[str, str]], bool]:
    """Return a copy with the inference system prompt when the sample has none."""
    normalized = [
        {"role": str(message.get("role", "")).lower(), "content": str(message.get("content", ""))}
        for message in messages
    ]
    if normalized and normalized[0]["role"] == "system":
        return normalized, False
    prompt = system_prompt.strip()
    if prompt:
        normalized.insert(0, {"role": "system", "content": prompt})
        return normalized, True
    return normalized, False


def format_generation_prompt(messages: Iterable[Mapping[str, str]], system_prompt: str) -> str:
    """Serialize chat exactly as the SFT pipeline does and append the assistant cue."""
    normalized, _ = ensure_system_message(messages, system_prompt)
    parts = []
    for message in normalized:
        role = message["role"]
        if role not in ROLE_LABELS:
            continue
        parts.append(f"{ROLE_LABELS[role]}: {message['content'].strip()}")
    parts.append(f"{ROLE_LABELS['assistant']}:")
    return "\n".join(parts)
