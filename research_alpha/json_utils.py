from __future__ import annotations

import json
import re
from typing import Any


def parse_llm_json(text: str, *, response_name: str = "LLM response") -> Any:
    """Parse JSON from strict or lightly wrapped LLM output."""
    raw = (text or "").strip().lstrip("\ufeff")
    if not raw:
        raise ValueError(
            f"{response_name} was empty; expected strict JSON. "
            "Retry the command, or check whether the selected LLM gateway returned an empty answer."
        )

    candidates = [raw]
    fenced = _extract_fenced_json(raw)
    if fenced and fenced not in candidates:
        candidates.append(fenced)

    decoder = json.JSONDecoder()
    errors: list[str] = []
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            errors.append(str(exc))
        extracted = _extract_first_json_value(candidate, decoder)
        if extracted is not None:
            return extracted

    preview = re.sub(r"\s+", " ", raw)[:240]
    detail = errors[-1] if errors else "no JSON object or array found"
    raise ValueError(
        f"Could not parse {response_name} as JSON ({detail}). "
        f"First characters: {preview!r}"
    )


def _extract_fenced_json(text: str) -> str:
    full = re.fullmatch(r"```(?:json|JSON)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if full:
        return full.group(1).strip()
    block = re.search(r"```(?:json|JSON)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if block:
        return block.group(1).strip()
    return ""


def _extract_first_json_value(text: str, decoder: json.JSONDecoder) -> Any:
    for idx, char in enumerate(text):
        if char not in "{[":
            continue
        try:
            payload, _ = decoder.raw_decode(text[idx:])
            return payload
        except json.JSONDecodeError:
            continue
    return None
