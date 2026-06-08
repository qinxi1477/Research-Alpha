from __future__ import annotations

import json
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


class LLMClient:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def is_configured(self) -> bool:
        return bool(self.config.api_key)

    def chat(self, prompt: str, system_prompt: str = "", temperature: float = 0.2) -> LLMResponse:
        if not self.is_configured():
            raise LLMError(_missing_api_key_message(self.config.provider))

        payload = {
            "model": self.config.resolved_model,
            "messages": self._build_messages(prompt, system_prompt),
            "temperature": temperature,
        }
        url = f"{self.config.resolved_base_url}/chat/completions"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                body = response.read().decode("utf-8")
                headers = getattr(response, "headers", None)
                content_type = headers.get("Content-Type", "") if headers else ""
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LLMError(_http_error_message(self.config.provider, url, exc.code, detail)) from exc
        except urllib.error.URLError as exc:
            raise LLMError(_network_error_message(self.config.provider, url, exc.reason)) from exc

        try:
            raw = json.loads(body)
        except json.JSONDecodeError as exc:
            raise LLMError(_non_json_response_message(self.config.provider, url, content_type, body)) from exc
        text = _extract_text(raw)
        if not text:
            raise LLMError("LLM response did not contain a text answer.")
        return LLMResponse(
            provider=self.config.provider,
            model=self.config.resolved_model,
            text=text,
            raw=raw,
        )

    @staticmethod
    def _build_messages(prompt: str, system_prompt: str) -> list[dict[str, str]]:
        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": prompt.strip()})
        return messages


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


def provider_summary(config: LLMConfig) -> Dict[str, Optional[str]]:
    return {
        "provider": config.provider,
        "model": config.resolved_model,
        "base_url": config.resolved_base_url,
        "api_key_configured": "yes" if config.api_key else "no",
    }


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


def _http_error_message(provider: str, url: str, status_code: int, detail: str) -> str:
    normalized = normalize_provider(provider)
    return (
        f"LLM request to {normalized} failed with HTTP {status_code} at {url}: {detail} "
        f"Check that the selected provider and key match. {_provider_setup_hint(normalized)}"
    )


def _network_error_message(provider: str, url: str, reason: object) -> str:
    normalized = normalize_provider(provider)
    return (
        f"LLM request to {normalized} failed before a response came back at {url}: {reason}. "
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
