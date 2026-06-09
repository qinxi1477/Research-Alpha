from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional

from research_alpha.config import LLMConfig, normalize_provider


class LLMError(RuntimeError):
    """Raised when an LLM request fails."""


@dataclass
class LLMResponse:
    provider: str
    model: str
    text: str
    raw: Dict[str, Any]
    usage: Dict[str, int]
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_cost_usd: float | None = None


class LLMClient:
    RETRYABLE_HTTP_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
    DEFAULT_TIMEOUT_SECONDS = 90
    DEFAULT_MAX_RETRIES = 2
    DEFAULT_RETRY_BASE_DELAY_SECONDS = 0.75
    DEFAULT_MIN_REQUEST_INTERVAL_SECONDS = 0.0

    def __init__(
        self,
        config: LLMConfig,
        *,
        timeout_seconds: float | None = None,
        max_retries: int | None = None,
        retry_base_delay_seconds: float | None = None,
        min_request_interval_seconds: float | None = None,
        max_prompt_tokens: int | None = None,
    ) -> None:
        self.config = config
        self.timeout_seconds = _positive_float(
            timeout_seconds if timeout_seconds is not None else os.environ.get("RA_LLM_TIMEOUT_SECONDS"),
            self.DEFAULT_TIMEOUT_SECONDS,
        )
        self.max_retries = max(
            0,
            int(
                max_retries
                if max_retries is not None
                else _positive_float(os.environ.get("RA_LLM_MAX_RETRIES"), self.DEFAULT_MAX_RETRIES)
            ),
        )
        self.retry_base_delay_seconds = _positive_float(
            retry_base_delay_seconds
            if retry_base_delay_seconds is not None
            else os.environ.get("RA_LLM_RETRY_BASE_DELAY_SECONDS"),
            self.DEFAULT_RETRY_BASE_DELAY_SECONDS,
        )
        self.min_request_interval_seconds = max(
            0.0,
            _positive_float(
                min_request_interval_seconds
                if min_request_interval_seconds is not None
                else os.environ.get("RA_LLM_MIN_REQUEST_INTERVAL_SECONDS"),
                self.DEFAULT_MIN_REQUEST_INTERVAL_SECONDS,
            ),
        )
        self.max_prompt_tokens = max(
            0,
            int(
                max_prompt_tokens
                if max_prompt_tokens is not None
                else _positive_float(os.environ.get("RA_LLM_MAX_PROMPT_TOKENS"), 0)
            ),
        )
        self._last_request_at = 0.0

    def is_configured(self) -> bool:
        return bool(self.config.api_key)

    def chat(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.2,
        *,
        json_mode: bool = False,
        response_schema: Dict[str, Any] | None = None,
        max_tokens: int | None = None,
        max_prompt_tokens: int | None = None,
        timeout_seconds: float | None = None,
    ) -> LLMResponse:
        if not self.is_configured():
            raise LLMError(_missing_api_key_message(self.config.provider))

        messages = self._build_messages(prompt, system_prompt)
        prompt_budget = int(max_prompt_tokens if max_prompt_tokens is not None else self.max_prompt_tokens)
        estimated_prompt_tokens = estimate_tokens(json.dumps(messages, ensure_ascii=False))
        if prompt_budget > 0 and estimated_prompt_tokens > prompt_budget:
            raise LLMError(
                f"LLM prompt budget exceeded: estimated {estimated_prompt_tokens} tokens > budget {prompt_budget}. "
                "Shorten the session memory, reduce paper/context count, or raise RA_LLM_MAX_PROMPT_TOKENS."
            )

        payload = {
            "model": self.config.resolved_model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max(1, int(max_tokens))
        if response_schema:
            payload["response_format"] = {"type": "json_schema", "json_schema": response_schema}
        elif json_mode:
            payload["response_format"] = {"type": "json_object"}
        url = f"{self.config.resolved_base_url}/chat/completions"
        timeout = _positive_float(timeout_seconds, self.timeout_seconds)
        attempts = self.max_retries + 1
        last_content_type = ""
        body = ""
        for attempt in range(1, attempts + 1):
            request = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            self._wait_for_rate_limit()
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    body = response.read().decode("utf-8")
                    headers = getattr(response, "headers", None)
                    last_content_type = headers.get("Content-Type", "") if headers else ""
                    try:
                        raw = json.loads(body)
                    except json.JSONDecodeError as exc:
                        if attempt < attempts and _should_retry_non_json_response(body, last_content_type):
                            self._sleep_before_retry(attempt, exc)
                            continue
                        raise LLMError(_non_json_response_message(self.config.provider, url, last_content_type, body)) from exc
                    break
            except urllib.error.HTTPError as exc:
                detail = _read_http_error_detail(exc)
                if attempt < attempts and exc.code in self.RETRYABLE_HTTP_STATUS:
                    self._sleep_before_retry(attempt, exc)
                    continue
                raise LLMError(_http_error_message(self.config.provider, url, exc.code, detail, attempt)) from exc
            except urllib.error.URLError as exc:
                if attempt < attempts:
                    self._sleep_before_retry(attempt, exc)
                    continue
                raise LLMError(_network_error_message(self.config.provider, url, exc.reason, attempt)) from exc
        text = _extract_text(raw)
        if not text:
            raise LLMError("LLM response did not contain a text answer.")
        usage = _extract_usage(raw)
        estimated_input = usage.get("prompt_tokens") or estimated_prompt_tokens
        estimated_output = usage.get("completion_tokens") or estimate_tokens(text)
        return LLMResponse(
            provider=self.config.provider,
            model=self.config.resolved_model,
            text=text,
            raw=raw,
            usage=usage,
            estimated_input_tokens=estimated_input,
            estimated_output_tokens=estimated_output,
            estimated_cost_usd=estimate_cost_usd(
                self.config.provider,
                self.config.resolved_model,
                prompt_tokens=estimated_input,
                completion_tokens=estimated_output,
            ),
        )

    @staticmethod
    def _build_messages(prompt: str, system_prompt: str) -> list[dict[str, str]]:
        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": prompt.strip()})
        return messages

    def _wait_for_rate_limit(self) -> None:
        if self.min_request_interval_seconds <= 0:
            self._last_request_at = time.monotonic()
            return
        now = time.monotonic()
        elapsed = now - self._last_request_at
        wait = self.min_request_interval_seconds - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request_at = time.monotonic()

    def _sleep_before_retry(self, attempt: int, exc: BaseException) -> None:
        retry_after = _retry_after_seconds(exc)
        if retry_after is None:
            retry_after = self.retry_base_delay_seconds * (2 ** max(0, attempt - 1))
            retry_after += random.uniform(0.0, min(0.25, self.retry_base_delay_seconds))
        time.sleep(max(0.0, retry_after))


def _extract_text(payload: Dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(part for part in parts if part).strip()
    return ""


def _extract_usage(payload: Dict[str, Any]) -> Dict[str, int]:
    raw_usage = payload.get("usage")
    if not isinstance(raw_usage, dict):
        return {}
    usage: Dict[str, int] = {}
    aliases = {
        "prompt_tokens": ("prompt_tokens", "input_tokens"),
        "completion_tokens": ("completion_tokens", "output_tokens"),
        "total_tokens": ("total_tokens",),
    }
    for target, keys in aliases.items():
        for key in keys:
            value = raw_usage.get(key)
            if isinstance(value, (int, float)):
                usage[target] = max(0, int(value))
                break
    if "total_tokens" not in usage and ("prompt_tokens" in usage or "completion_tokens" in usage):
        usage["total_tokens"] = int(usage.get("prompt_tokens", 0)) + int(usage.get("completion_tokens", 0))
    return usage


def estimate_tokens(text: object) -> int:
    value = str(text or "")
    if not value:
        return 0
    cjk = sum(1 for ch in value if "\u4e00" <= ch <= "\u9fff")
    non_cjk = max(0, len(value) - cjk)
    return max(1, cjk + int(non_cjk / 4) + (1 if non_cjk % 4 else 0))


def estimate_cost_usd(provider: str, model: str, *, prompt_tokens: int, completion_tokens: int) -> float | None:
    price = _model_price_per_million_tokens(provider, model)
    if not price:
        return None
    input_price, output_price = price
    cost = (max(0, prompt_tokens) / 1_000_000.0) * input_price
    cost += (max(0, completion_tokens) / 1_000_000.0) * output_price
    return round(cost, 8)


def _model_price_per_million_tokens(provider: str, model: str) -> tuple[float, float] | None:
    normalized_model = str(model or "").lower()
    normalized_provider = normalize_provider(provider)
    if normalized_provider == "deepseek":
        if "deepseek-reasoner" in normalized_model:
            return (0.55, 2.19)
        if "deepseek-chat" in normalized_model:
            return (0.27, 1.10)
    if normalized_provider == "openai":
        if "gpt-4.1-mini" in normalized_model:
            return (0.40, 1.60)
        if "gpt-4.1" in normalized_model:
            return (2.00, 8.00)
        if "gpt-4o-mini" in normalized_model:
            return (0.15, 0.60)
    return None


def provider_summary(config: LLMConfig) -> Dict[str, Optional[str]]:
    return {
        "provider": config.provider,
        "model": config.resolved_model,
        "base_url": config.resolved_base_url,
        "api_key_configured": "yes" if config.api_key else "no",
    }


def _positive_float(value: float | int | str | None, default: float) -> float:
    if value in (None, ""):
        return float(default)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if parsed > 0 else float(default)


def _retry_after_seconds(exc: BaseException) -> float | None:
    headers = getattr(exc, "headers", None)
    if not headers:
        return None
    value = headers.get("Retry-After", "")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _should_retry_non_json_response(body: str, content_type: str) -> bool:
    if not str(body or "").strip():
        return True
    return "application/json" in str(content_type or "").lower()


def _read_http_error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        data = exc.read()
    except Exception:
        data = b""
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return str(data or "")


def _provider_setup_hint(provider: str) -> str:
    normalized = normalize_provider(provider)
    if normalized == "deepseek":
        return "Run `ra ds sk-...` or set `DEEPSEEK_API_KEY`."
    return "Run `ra oa sk-...` or set `OPENAI_API_KEY`."


def _missing_api_key_message(provider: str) -> str:
    normalized = normalize_provider(provider)
    return (
        f"Missing API key for {normalized}. "
        f"{_provider_setup_hint(normalized)} "
        "You can also set `RA_LLM_API_KEY` as a generic fallback."
    )


def _http_error_message(provider: str, url: str, status_code: int, detail: str, attempts: int = 1) -> str:
    normalized = normalize_provider(provider)
    safe_detail = " ".join(str(detail or "").split())
    if len(safe_detail) > 800:
        safe_detail = safe_detail[:797] + "..."
    return (
        f"LLM request to {normalized} failed with HTTP {status_code} at {url} after {attempts} attempt(s): {safe_detail} "
        f"Check that the selected provider and key match. {_provider_setup_hint(normalized)}"
    )


def _network_error_message(provider: str, url: str, reason: object, attempts: int = 1) -> str:
    normalized = normalize_provider(provider)
    return (
        f"LLM request to {normalized} failed before a response came back at {url} after {attempts} attempt(s): {reason}. "
        "This usually means the network, DNS, or proxy is unavailable rather than the prompt being wrong."
    )


def _non_json_response_message(provider: str, url: str, content_type: str, body: str) -> str:
    normalized = normalize_provider(provider)
    snippet = " ".join(body.strip().split())
    if len(snippet) > 240:
        snippet = snippet[:237] + "..."
    content_note = content_type or "unknown content type"
    return (
        f"LLM request to {normalized} returned a non-JSON response at {url} "
        f"({content_note}). Check the provider base URL; OpenAI-compatible endpoints usually end in `/v1`. "
        f"Response preview: {snippet}"
    )
