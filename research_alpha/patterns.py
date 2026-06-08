from __future__ import annotations

import json
from typing import Any, Dict, List

from research_alpha.json_utils import parse_llm_json


REQUIRED_PATTERN_LOGIC_KEYS = [
    "source_logic",
    "transfer_rule",
    "why_it_generalizes",
    "failure_boundary",
]
PATTERN_EVIDENCE_LEVEL = "logic_line_aggregation"


PATTERN_SYSTEM_PROMPT = """
You are aggregating multiple Idea Genome Cards into one reusable research pattern.
Use only the provided cards.
Use only cards whose grounding audit is valid when such an audit is present.
Do not aggregate keywords. Preserve the reusable argument logic shared by the
papers: source logic, transfer rule, why it generalizes, and failure boundary.
Every canonical example must be one of the provided card titles.
Return strict JSON with concise fields.
""".strip()


def build_pattern_prompt(cards: List[Dict[str, Any]]) -> str:
    items = []
    for idx, card in enumerate(cards, start=1):
        items.append(
            {
                "paper_id": card.get("paper_id"),
                "title": card.get("title"),
                "venue": card.get("venue"),
                "year": card.get("year"),
                "paper_summary": card.get("paper_summary"),
                "problem_reframing": card.get("problem_reframing"),
                "transferable_pattern": card.get("transferable_pattern"),
                "why_now": card.get("why_now"),
                "failure_boundary": card.get("failure_boundary"),
                "logic_line": card.get("logic_line"),
                "grounding_audit": card.get("grounding_audit"),
            }
        )

    return (
        "Aggregate the following genome cards into one Pattern Card.\n\n"
        "Return JSON only with this exact schema:\n"
        "{\n"
        '  "pattern_key": "...",\n'
        '  "pattern_name": "...",\n'
        '  "core_move": "...",\n'
        '  "when_to_use": "...",\n'
        '  "why_it_works": "...",\n'
        '  "failure_modes": "...",\n'
        '  "canonical_examples": ["..."],\n'
        '  "operator_template": "...",\n'
        '  "logic_line_pattern": {"source_logic": "...", "transfer_rule": "...", "why_it_generalizes": "...", "failure_boundary": "..."},\n'
        f'  "evidence_level": "{PATTERN_EVIDENCE_LEVEL}"\n'
        "}\n\n"
        f"Cards:\n{json.dumps(items, ensure_ascii=False, indent=2)}"
    )


def parse_pattern_response(text: str) -> Dict[str, Any]:
    payload = parse_llm_json(text, response_name="Pattern response")
    if not isinstance(payload, dict):
        raise ValueError("Pattern response must be a JSON object")
    required = [
        "pattern_key",
        "pattern_name",
        "core_move",
        "when_to_use",
        "why_it_works",
        "failure_modes",
        "canonical_examples",
        "operator_template",
        "evidence_level",
    ]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Pattern response missing keys: {', '.join(missing)}")
    logic_line_pattern = payload.get("logic_line_pattern")
    if not isinstance(logic_line_pattern, dict):
        logic_line_pattern = {
            "source_logic": payload.get("core_move", ""),
            "transfer_rule": payload.get("operator_template", ""),
            "why_it_generalizes": payload.get("why_it_works", ""),
            "failure_boundary": payload.get("failure_modes", ""),
        }
    missing_logic = [
        key
        for key in REQUIRED_PATTERN_LOGIC_KEYS
        if not str(logic_line_pattern.get(key, "")).strip()
    ]
    if missing_logic:
        raise ValueError(f"Pattern response missing logic_line_pattern keys: {', '.join(missing_logic)}")
    payload["logic_line_pattern"] = {
        key: str(logic_line_pattern.get(key, "")).strip()
        for key in REQUIRED_PATTERN_LOGIC_KEYS
    }
    payload["evidence_level"] = PATTERN_EVIDENCE_LEVEL
    return payload
