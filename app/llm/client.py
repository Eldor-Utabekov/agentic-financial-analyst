"""Optional provider-backed client for grounded answer generation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Mapping
from urllib.error import HTTPError
from urllib.error import URLError
from urllib.request import Request
from urllib.request import urlopen

from app.llm.prompts import build_grounded_answer_prompt
from app.llm.prompts import build_grounding_system_prompt

DEFAULT_BASE_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TIMEOUT_SECONDS = 30.0


class LLMClientError(Exception):
    """Raised when provider-backed answer generation fails."""


@dataclass(frozen=True, slots=True)
class SummaryGenerationResult:
    """Result of summary generation with its origin."""

    summary: str
    source: str


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """Minimal environment-backed configuration for an optional LLM provider."""

    api_key: str | None
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    def is_configured(self) -> bool:
        """Return whether provider-backed generation can run."""
        return bool(self.api_key)


def load_llm_config(env: Mapping[str, str] | None = None) -> LLMConfig:
    """Load provider configuration from environment variables."""
    source = os.environ if env is None else env
    api_key = _normalize_optional_value(source.get("OPENAI_API_KEY"))
    base_url = _normalize_optional_value(source.get("OPENAI_BASE_URL")) or DEFAULT_BASE_URL
    model = _normalize_optional_value(source.get("OPENAI_MODEL")) or DEFAULT_MODEL
    timeout_seconds = _parse_timeout_seconds(source.get("OPENAI_TIMEOUT_SECONDS"))

    return LLMConfig(
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout_seconds=timeout_seconds,
    )


def generate_grounded_answer(
    question: str,
    supporting_signals: list[str],
    retrieved_context_texts: list[str],
    fallback_summary: str,
    config: LLMConfig | None = None,
) -> str:
    """Generate a grounded answer, falling back to the deterministic summary when needed."""
    return generate_grounded_answer_result(
        question=question,
        supporting_signals=supporting_signals,
        retrieved_context_texts=retrieved_context_texts,
        fallback_summary=fallback_summary,
        config=config,
    ).summary


def generate_grounded_answer_result(
    question: str,
    supporting_signals: list[str],
    retrieved_context_texts: list[str],
    fallback_summary: str,
    config: LLMConfig | None = None,
) -> SummaryGenerationResult:
    """Generate a grounded answer and record whether provider or fallback was used."""
    resolved_config = load_llm_config() if config is None else config
    if not resolved_config.is_configured():
        return SummaryGenerationResult(summary=fallback_summary, source="deterministic_fallback")

    system_prompt = build_grounding_system_prompt()
    user_prompt = build_grounded_answer_prompt(
        question=question,
        signals=supporting_signals,
        retrieved_context=retrieved_context_texts,
    )

    try:
        generated_answer = _request_chat_completion(
            config=resolved_config,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
    except LLMClientError:
        return SummaryGenerationResult(summary=fallback_summary, source="deterministic_fallback")

    cleaned_answer = generated_answer.strip()
    if not cleaned_answer:
        return SummaryGenerationResult(summary=fallback_summary, source="deterministic_fallback")
    return SummaryGenerationResult(summary=cleaned_answer, source="provider_backed")


def _request_chat_completion(config: LLMConfig, system_prompt: str, user_prompt: str) -> str:
    """Call an OpenAI-compatible chat completions endpoint."""
    if not config.api_key:
        raise LLMClientError("Missing API key for provider-backed answer generation.")

    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
    }
    request = Request(
        url=config.base_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=config.timeout_seconds) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise LLMClientError("Provider-backed answer generation failed.") from exc

    try:
        return str(response_payload["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMClientError("Provider response did not contain a valid message.") from exc


def _normalize_optional_value(value: str | None) -> str | None:
    """Normalize optional string configuration values."""
    if value is None:
        return None
    cleaned_value = value.strip()
    return cleaned_value or None


def _parse_timeout_seconds(value: str | None) -> float:
    """Parse timeout configuration into a positive float value."""
    normalized_value = _normalize_optional_value(value)
    if normalized_value is None:
        return DEFAULT_TIMEOUT_SECONDS

    try:
        timeout_seconds = float(normalized_value)
    except ValueError as exc:
        raise ValueError("OPENAI_TIMEOUT_SECONDS must be a positive number.") from exc

    if timeout_seconds <= 0:
        raise ValueError("OPENAI_TIMEOUT_SECONDS must be a positive number.")

    return timeout_seconds
