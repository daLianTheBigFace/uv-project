import os


DEFAULT_CHAT_MODEL = "deepseek-v4-flash"


def _is_enabled(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    cleaned = value.strip().lower()
    if cleaned in {"1", "true", "yes", "on", "enabled"}:
        return True
    if cleaned in {"0", "false", "no", "off", "disabled"}:
        return False
    return default


def _pick(*values: str | None) -> str:
    for value in values:
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned:
                return cleaned
    return ""


def resolve_chat_model(explicit_model: str | None = None) -> str:
    """Resolve chat model name with explicit value taking highest priority."""
    return _pick(
        explicit_model,
        os.getenv("DEEPSEEK_CHAT_MODEL"),
        os.getenv("DEEPSEEK_MODEL"),
        DEFAULT_CHAT_MODEL,
    )


def resolve_vision_model(explicit_model: str | None = None) -> str:
    """Resolve vision model name while preserving dedicated env override support."""
    return _pick(
        explicit_model,
        os.getenv("DEEPSEEK_VISION_MODEL"),
        os.getenv("DEEPSEEK_CHAT_MODEL"),
        os.getenv("DEEPSEEK_MODEL"),
        DEFAULT_CHAT_MODEL,
    )


def resolve_reasoning_config() -> dict[str, str]:
    """Resolve DeepSeek thinking mode config.

    By default, thinking mode is disabled to avoid requiring reasoning content
    replay in multi-turn chat clients that only persist plain text messages.
    Set `DEEPSEEK_THINKING_ENABLED=true` to re-enable it.
    """

    thinking_enabled = _is_enabled(os.getenv("DEEPSEEK_THINKING_ENABLED"), default=False)
    return {"type": "enabled" if thinking_enabled else "disabled"}


