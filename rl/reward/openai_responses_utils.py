"""
openai_responses_utils.py

Helpers shared by PMF extractor / scorer / judge code so that the same call
sites work against either the OpenAI Chat Completions endpoint or the newer
OpenAI Responses endpoint. Currently used with the Chat Completions surface
exposed by OpenRouter / OpenAI / local vLLM, but wrapped in a thin abstraction
so future migration to the Responses API requires only local changes here.

Exports used by callers (PMF.extractor / PMF.scorer / PMF.pmf):
    - DEFAULT_OPENAI_MODEL
    - resolve_api_key(api_key, env_names=("OPENAI_API_KEY",))
    - build_client_kwargs(api_key, base_url)
    - build_responses_kwargs(model, messages, temperature, response_format, max_output_tokens)
    - extract_response_text(response)
    - extract_finish_reason(response)
"""

from __future__ import annotations

import os
from typing import Any


DEFAULT_OPENAI_MODEL = (
    os.environ.get("OPENAI_MODEL")
    or "openai/gpt-4o-mini-2024-07-18"
).strip()


def resolve_api_key(
    api_key: str | None = None,
    env_names: tuple[str, ...] = ("OPENAI_API_KEY",),
) -> str:
    if api_key:
        return api_key
    for name in env_names:
        v = os.environ.get(name, "").strip()
        if v:
            return v
    # Accept a blank key for local vLLM servers that do not require auth.
    return ""


def build_client_kwargs(
    *,
    api_key: str = "",
    base_url: str = "",
    timeout: float | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    kwargs["api_key"] = api_key or "dummy-local"
    base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "")).strip()
    if base_url:
        kwargs["base_url"] = base_url
    if timeout is None:
        env_timeout = os.environ.get("OPENAI_TIMEOUT_S", "").strip()
        if env_timeout:
            try:
                timeout = float(env_timeout)
            except ValueError:
                timeout = None
    if timeout is not None:
        kwargs["timeout"] = timeout
    return kwargs


def _coerce_content(content: Any) -> Any:
    if content is None:
        return ""
    if isinstance(content, (list, tuple)):
        return list(content)
    return str(content)


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role") or "user"
        out.append({"role": role, "content": _coerce_content(msg.get("content"))})
    return out


def build_responses_kwargs(
    *,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float | None = None,
    response_format: dict[str, Any] | None = None,
    max_output_tokens: int | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Build kwargs for `client.chat.completions.create(...)` (Chat Completions
    surface). `max_output_tokens` maps to `max_tokens`."""
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": normalize_messages(messages),
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    if response_format is not None:
        kwargs["response_format"] = response_format
    if max_output_tokens is not None:
        kwargs["max_tokens"] = max_output_tokens
    kwargs.update(extra)
    return kwargs


def _safe_get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def extract_response_text(response: Any) -> str:
    """Extract the assistant text from a Chat Completions-style response.

    Tolerant to pydantic objects, dicts, and responses with multiple choices.
    """
    # Responses API (output_text convenience)
    text = _safe_get(response, "output_text")
    if isinstance(text, str) and text:
        return text

    choices = _safe_get(response, "choices") or []
    if not choices:
        return ""
    first = choices[0]
    message = _safe_get(first, "message")
    if message is None:
        # Legacy completions.text
        txt = _safe_get(first, "text")
        return txt or ""
    content = _safe_get(message, "content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Chat content parts (e.g. [{"type": "text", "text": "..."}, ...])
        pieces: list[str] = []
        for part in content:
            t = _safe_get(part, "text")
            if isinstance(t, str):
                pieces.append(t)
        return "".join(pieces)
    return ""


def extract_finish_reason(response: Any) -> str:
    choices = _safe_get(response, "choices") or []
    if not choices:
        return ""
    fr = _safe_get(choices[0], "finish_reason")
    return fr or ""
