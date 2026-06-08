from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from research_alpha.json_utils import parse_llm_json


CRITIQUE_SYSTEM_PROMPT = """
You are an AI/ML research idea critic.
Your job is not to obey the user's latest instruction blindly.
You should compare the current idea and the user's requested change against high-quality research patterns.
If the user's direction weakens novelty, value, story strength, or defensibility, you should say so clearly.
Return strict JSON.
""".strip()


def build_critique_prompt(
    *,
    current_idea: str,
    user_instruction: str,
    pattern_cards: List[Dict[str, Any]],
    genome_cards: List[Dict[str, Any]],
    session_context: Dict[str, Any],
) -> str:
    payload = {
        "current_idea": current_idea,
        "user_instruction": user_instruction,
        "session_context": session_context,
        "pattern_cards": pattern_cards,
        "genome_cards": genome_cards,
    }
    return (
        "Evaluate the user's requested change to the current research idea.\n"
        "Use the provided session context, pattern evidence, and genome evidence to decide whether to accept, partially accept, or reject the requested change.\n"
        "If the session context includes trend signals or an originating candidate idea, use them explicitly when judging whether the user's change aligns with frontier momentum.\n\n"
        "If the user_instruction or session_context.memory_summary.preferred_language is Chinese/zh, return all natural-language JSON values in Chinese.\n"
        "Use session_context.memory_summary as compressed conversation memory; preserve accepted preferences, avoid rejected preferences, and keep unresolved risks visible.\n"
        "Do not let recent/frontier papers become scoring standards unless the memory/session context marks them as Gold evidence.\n\n"
        "Keep the model creative inside the evidence boundary: you may rename, abstract, synthesize across supplied sources, choose new mechanisms, "
        "invent domain-appropriate constructs, and rewrite the idea, but the revised idea must stay inside the logic space of the supplied "
        "Pattern/Genome/Gold evidence rather than filling a rigid template. "
        "Treat any storyline trace in session_context as an audit trail, not as a paragraph template for the revised answer. "
        "Do not copy a source paper's method as a template; migrate the story logic instead: old belief -> bottleneck -> reframing -> why-now -> evidence design -> failure boundary.\n"
        "supporting_patterns and supporting_genomes must name supplied Pattern/Genome evidence that still supports the revised idea. "
        "If the change is useful but loses the evidence chain, partially accept it and explicitly preserve the original logic chain.\n\n"
        "Return JSON only with this schema:\n"
        "{\n"
        '  "decision": "accept|partial|reject",\n'
        '  "verdict_summary": "...",\n'
        '  "why_user_may_be_wrong": "...",\n'
        '  "supporting_patterns": ["..."],\n'
        '  "supporting_genomes": ["..."],\n'
        '  "trend_alignment": "...",\n'
        '  "preserve": "...",\n'
        '  "revise": "...",\n'
        '  "revised_idea": "...",\n'
        '  "coach_message": "...",\n'
        '  "feasibility_assessment": {\n'
        '    "status": "promising|fragile|unclear",\n'
        '    "summary": "...",\n'
        '    "next_test": "..."\n'
        "  },\n"
        '  "why_not_analysis": {\n'
        '    "status": "serious|moderate|light",\n'
        '    "summary": "...",\n'
        '    "mitigation": "..."\n'
        "  },\n"
        '  "value_judgment": {\n'
        '    "status": "high|medium|low",\n'
        '    "summary": "...",\n'
        '    "proof_gap": "..."\n'
        "  }\n"
        "}\n\n"
        f"Context:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def parse_critique_response(text: str) -> Dict[str, Any]:
    payload = parse_llm_json(text, response_name="Critique response")
    if not isinstance(payload, dict):
        raise ValueError("Critique response must be a JSON object")
    required = [
        "decision",
        "verdict_summary",
        "why_user_may_be_wrong",
        "supporting_patterns",
        "supporting_genomes",
        "trend_alignment",
        "preserve",
        "revise",
        "revised_idea",
        "coach_message",
        "feasibility_assessment",
        "why_not_analysis",
        "value_judgment",
    ]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Critique response missing keys: {', '.join(missing)}")
    decision = str(payload["decision"]).strip().lower()
    if decision not in {"accept", "partial", "reject"}:
        raise ValueError("Critique decision must be one of: accept, partial, reject")
    payload["decision"] = decision
    _validate_dimension(
        payload["feasibility_assessment"],
        field_name="feasibility_assessment",
        statuses={"promising", "fragile", "unclear"},
        extra_key="next_test",
    )
    _validate_dimension(
        payload["why_not_analysis"],
        field_name="why_not_analysis",
        statuses={"serious", "moderate", "light"},
        extra_key="mitigation",
    )
    _validate_dimension(
        payload["value_judgment"],
        field_name="value_judgment",
        statuses={"high", "medium", "low"},
        extra_key="proof_gap",
    )
    _validate_critique_content_quality(payload)
    return payload


def _validate_dimension(
    value: Any,
    *,
    field_name: str,
    statuses: set[str],
    extra_key: str,
) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"Critique field `{field_name}` must be an object")
    required = {"status", "summary", extra_key}
    missing = [key for key in required if key not in value]
    if missing:
        raise ValueError(f"Critique field `{field_name}` missing keys: {', '.join(sorted(missing))}")
    status = str(value["status"]).strip().lower()
    if status not in statuses:
        allowed = ", ".join(sorted(statuses))
        raise ValueError(f"Critique field `{field_name}` status must be one of: {allowed}")
    value["status"] = status
    _validate_substantive_text(value.get("summary", ""), field_name=f"{field_name}.summary", min_signal=5)
    _validate_substantive_text(value.get(extra_key, ""), field_name=f"{field_name}.{extra_key}", min_signal=5)


def _validate_critique_content_quality(payload: Dict[str, Any]) -> None:
    string_fields = [
        "verdict_summary",
        "why_user_may_be_wrong",
        "trend_alignment",
        "preserve",
        "revise",
    ]
    for field in string_fields:
        _validate_substantive_text(payload.get(field, ""), field_name=field, min_signal=6)
    _validate_substantive_text(payload.get("revised_idea", ""), field_name="revised_idea", min_signal=18)


def _validate_substantive_text(value: Any, *, field_name: str, min_signal: int) -> None:
    text = str(value or "").strip()
    if _substantive_signal(text) >= min_signal and not _looks_like_placeholder(text):
        return
    raise ValueError(
        f"Critique response content_quality_failed: `{field_name}` is empty, placeholder-like, or too thin "
        "to safely write as a session turn."
    )


def _substantive_signal(text: str) -> int:
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", text)
    latin_tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", text)
    return len(cjk_chars) + sum(len(token) for token in latin_tokens)


def _looks_like_placeholder(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", str(text or "").strip().lower())
    stripped = normalized.strip(" .。!！?？:：-_*")
    if not stripped:
        return True
    placeholders = {
        "ok",
        "yes",
        "done",
        "n/a",
        "na",
        "none",
        "todo",
        "tbd",
        "same as above",
        "no source needed",
        "looks good",
        "good",
        "fine",
        "已完成",
        "完成",
        "可以",
        "很好",
        "同上",
        "无",
        "暂无",
        "待补充",
        "修改后的 idea",
        "修改后的idea",
    }
    return stripped in placeholders
