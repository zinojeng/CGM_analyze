"""Utility helpers for invoking OpenAI models with resilient fallbacks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, Tuple

from openai import OpenAI


RESPONSES_PREFIXES: Tuple[str, ...] = ("o", "gpt-5", "gpt-4.1", "gpt-4o")
REASONING_PREFIXES: Tuple[str, ...] = ("o", "gpt-5")
DEFAULT_FALLBACK_MODELS: List[str] = ["gpt-5-mini", "gpt-4o-mini"]


def _uses_responses_api(model_name: str) -> bool:
    lower = model_name.lower()
    return any(lower.startswith(prefix) for prefix in RESPONSES_PREFIXES)


def _supports_reasoning_effort(model_name: str) -> bool:
    lower = model_name.lower()
    return any(lower.startswith(prefix) for prefix in REASONING_PREFIXES)


def format_messages_for_responses(messages: Sequence[dict]) -> str:
    parts: List[str] = []
    for message in messages:
        role = message.get("role", "user").upper()
        content = message.get("content", "")
        if isinstance(content, list):
            texts = [item.get("text", "") for item in content if isinstance(item, dict)]
            content = "\n".join(texts)
        parts.append(f"{role}:\n{content}".strip())
    return "\n\n".join(parts)


def _should_try_fallback(error_message: str) -> bool:
    lowered = (error_message or "").lower()
    if "empty_response_text" in lowered:
        return True
    keywords = [
        "model",
        "not",
        "exist",
        "access",
        "permission",
        "model_not_found",
        "unsupported_parameter",
        "rate limit",
    ]
    return all(keyword in lowered for keyword in ["model", "not"]) or any(keyword in lowered for keyword in keywords)


def _extract_text_from_response(response) -> str:
    text = (getattr(response, "output_text", None) or "").strip()
    if text:
        return text

    chunks: List[str] = []
    outputs = getattr(response, "output", None) or []
    for item in outputs:
        contents = getattr(item, "content", None)
        if contents is None and isinstance(item, dict):
            contents = item.get("content")
        if not contents:
            continue
        for content in contents:
            content_type = getattr(content, "type", None)
            if content_type is None and isinstance(content, dict):
                content_type = content.get("type")
            if content_type not in {"output_text", "text"}:
                continue
            text_value = getattr(content, "text", None)
            if text_value is None and isinstance(content, dict):
                text_value = content.get("text")
            if text_value:
                chunks.append(text_value)

    return "\n".join(chunks).strip()


@dataclass
class LLMCallResult:
    text: str
    model_used: str
    failures: List[Tuple[str, str]]


def call_with_fallback(
    api_key: str,
    primary_model: str,
    messages: Sequence[dict],
    *,
    max_tokens: int,
    temperature: float = 0.2,
    fallback_models: Iterable[str] | None = None,
) -> LLMCallResult:
    """Call an OpenAI model and fall back to alternatives when needed."""

    client = OpenAI(api_key=api_key)

    tried_failures: List[Tuple[str, str]] = []
    primary_lower = primary_model.lower()
    models_to_try: List[str] = [primary_model]

    if fallback_models:
        for candidate in fallback_models:
            if candidate and candidate.lower() not in {m.lower() for m in models_to_try}:
                models_to_try.append(candidate)

    for idx, model_name in enumerate(models_to_try):
        try:
            text = _call_single_model(
                client,
                model_name=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return LLMCallResult(text=text, model_used=model_name, failures=tried_failures)
        except Exception as exc:  # pylint: disable=broad-except
            error_message = str(exc)
            tried_failures.append((model_name, error_message))
            is_last_candidate = idx == len(models_to_try) - 1
            if is_last_candidate or not _should_try_fallback(error_message):
                aggregated = \
                    "; ".join(f"{model} failed: {msg}" for model, msg in tried_failures)
                raise RuntimeError(aggregated) from exc

    # Control should never reach here because the loop either returns or raises.
    raise RuntimeError("No models were attempted for the request.")


def request_llm_text(
    api_key: str | None,
    *,
    primary_model: str,
    messages: Sequence[dict],
    max_tokens: int,
    temperature: float = 0.2,
    fallback_models: Iterable[str] | None = None,
    missing_key_error: str | None = None,
    error_formatter: Callable[[str, Exception], str] | None = None,
    fallback_notice_formatter: Callable[[str, LLMCallResult], str] | None = None,
) -> Tuple[str | None, str | None, LLMCallResult | None, str | None]:
    """Wrapper around ``call_with_fallback`` that returns structured success/error feedback."""

    if not api_key:
        return None, missing_key_error or "Missing API key for LLM call.", None, None

    def _default_error_formatter(model_name: str, exc: Exception) -> str:
        return f"LLM call failed ({model_name}): {exc}"

    formatter = error_formatter or _default_error_formatter

    try:
        result = call_with_fallback(
            api_key,
            primary_model=primary_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            fallback_models=fallback_models,
        )
    except RuntimeError as error:
        return None, formatter(primary_model, error), None, None
    except Exception as error:  # pylint: disable=broad-except
        return None, formatter(primary_model, error), None, None

    text = result.text or ""
    notice = None
    if (
        fallback_notice_formatter
        and result.failures
        and result.model_used
        and result.model_used.lower() != primary_model.lower()
    ):
        notice = fallback_notice_formatter(primary_model, result)

    return text, None, result, notice


def _call_single_model(
    client: OpenAI,
    *,
    model_name: str,
    messages: Sequence[dict],
    max_tokens: int,
    temperature: float,
) -> str:
    if _uses_responses_api(model_name):
        kwargs = {
            "model": model_name,
            "input": format_messages_for_responses(messages),
            "max_output_tokens": max_tokens,
        }
        if _supports_reasoning_effort(model_name):
            kwargs["reasoning"] = {"effort": "medium"}

        response = client.responses.create(**kwargs)
        text = _extract_text_from_response(response)
        if not text:
            raise ValueError("empty_response_text")
        return text

    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        n=1,
    )
    if not completion.choices:
        raise ValueError("empty_chat_completion")
    message = completion.choices[0].message
    return (getattr(message, "content", None) or "").strip()
