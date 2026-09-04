"""Chat model factory for the text (conversation + tool calling) path.

The provider is inferred from the model id, so ``TEXT_MODEL=gemini-3.5-flash-lite``,
``TEXT_MODEL=gpt-4o-mini`` and ``TEXT_MODEL=claude-haiku-4-5`` all work without
touching code.

Three providers are supported, chosen so the project can run on a genuinely free
key as well as a paid one:

* ``claude-*``  -> Anthropic          (``ANTHROPIC_API_KEY``)
* ``gemini-*``  -> Google AI Studio   (``GOOGLE_API_KEY``) — has a free tier
* everything else -> an OpenAI-compatible endpoint (``OPENAI_API_KEY``, plus
  ``OPENAI_BASE_URL`` to point at Groq, GitHub Models, OpenRouter, Ollama, …)

The OpenAI-compatible branch is the escape hatch: any provider that speaks that
wire format works by setting a base URL, with no code change here.

Clients are cached because constructing one sets up an HTTP connection pool —
rebuilding it per turn adds avoidable latency to every message.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from src.utils.config import settings


def provider_for(model_id: str) -> str:
    """Map a model id onto its provider.

    Unknown ids fall through to the OpenAI-compatible client rather than
    raising, because that is how Groq/OpenRouter/Ollama model names
    (``llama-3.3-70b-versatile``, ``qwen2.5``) are meant to be reached.
    """
    lowered = model_id.lower()
    if lowered.startswith("claude"):
        return "anthropic"
    if lowered.startswith(("gemini", "models/gemini")):
        return "google"
    return "openai"


@lru_cache(maxsize=8)
def _build(model_id: str, temperature: float, max_tokens: int) -> BaseChatModel:
    """Construct (and memoise) a chat client for one model id."""
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

    if provider == "google":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:  # pragma: no cover - dependency guidance
            raise ImportError(
                "Gemini models need the Google integration. Install it with:\n"
                "    pip install langchain-google-genai\n"
                "and set GOOGLE_API_KEY (free key: https://aistudio.google.com/apikey)"
            ) from exc

        # Gemini 3.x models use fixed sampling and warn if temperature is
        # passed; older ones honour it. Only send it where it means something.
        sampling = {} if model_id.lower().startswith("gemini-3") else {"temperature": temperature}
        return ChatGoogleGenerativeAI(
            model=model_id,
            max_output_tokens=max_tokens,
            timeout=30,
            max_retries=2,
            **sampling,
        )

    from langchain_openai import ChatOpenAI

    # base_url lets one client cover every OpenAI-compatible host.
    base_url = os.environ.get("OPENAI_BASE_URL") or None
    # Reasoning models spend hidden tokens thinking before every tool call.
    # For "which tool, with what arguments" that is wasted budget and latency,
    # so default to low effort; override with MODEL_REASONING_EFFORT.
    extra: dict[str, Any] = {}
    if "gpt-oss" in model_id.lower() or model_id.lower().startswith(("o1", "o3", "o4")):
        extra["reasoning_effort"] = os.environ.get("MODEL_REASONING_EFFORT", "low")
    return ChatOpenAI(
        model=model_id,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=30,
        max_retries=2,
        base_url=base_url,
        **extra,
        # Local runtimes (Ollama, LM Studio) ignore the key but the client
        # still requires one to be present.
        api_key=os.environ.get("OPENAI_API_KEY") or ("not-needed" if base_url else None),
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
