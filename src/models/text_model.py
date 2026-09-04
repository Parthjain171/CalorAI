"""Chat model factory for the text (conversation + tool calling) path.

The provider is inferred from the model id, so ``TEXT_MODEL=gpt-4o-mini`` and
``TEXT_MODEL=claude-haiku-4-5`` both work without touching code. Clients are
cached because constructing one sets up an HTTP connection pool — rebuilding it
per turn adds avoidable latency to every message.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from src.utils.config import settings


def provider_for(model_id: str) -> str:
    """Map a model id onto its provider."""
    lowered = model_id.lower()
    if lowered.startswith("claude"):
        return "anthropic"
    if lowered.startswith(("gpt", "o1", "o3", "o4", "chatgpt")):
        return "openai"
    raise ValueError(
        f"Cannot infer a provider for model id {model_id!r}. "
        "Use a claude-* or gpt-* id."
    )


@lru_cache(maxsize=8)
def _build(model_id: str, temperature: float, max_tokens: int) -> BaseChatModel:
    provider = provider_for(model_id)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=30,
            max_retries=2,
        )

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model_id,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=30,
        max_retries=2,
    )


def get_chat_model(
    model_id: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    **_: Any,
) -> BaseChatModel:
    """Return the conversation model (a scripted double when ``CALORAI_MOCK=1``)."""
    if settings.mock:
        from src.models.mock_model import ScriptedChatModel

        return ScriptedChatModel()
    return _build(model_id or settings.text_model, temperature, max_tokens)
