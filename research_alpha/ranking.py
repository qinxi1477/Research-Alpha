from __future__ import annotations

import json
from typing import Any, Dict, List

from research_alpha.json_utils import parse_llm_json


RANKING_SYSTEM_PROMPT = """
You are ranking AI/ML research ideas for top-tier conference potential.
Use the provided evidence to compare the ideas directly.
Prefer ideas that combine strong historical grounding, frontier relevance, novelty, value, and evaluability.
Return strict JSON only.
""".strip()


def build_ranking_prompt(
    *,
    query: str,
    ranked_candidates: List[Dict[str, Any]],
) -> str:
    payload = {
        "query": query,
        "ranked_candidates": ranked_candidates,
    }
    return (
        "Re-rank these candidate ideas for which one should be refined first.\n"
        "Treat the heuristic score as a hint, not a rule. You may disagree with it.\n\n"
        "Return JSON only with this schema:\n"
        "{\n"
        '  "ranked_ids": [1, 2, 3],\n'
        '  "summary": "...",\n'
        '  "why_top": "...",\n'
        '  "watchouts": ["...", "..."]\n'
        "}\n\n"
        f"Context:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def parse_ranking_response(text: str) -> Dict[str, Any]:
    payload = parse_llm_json(text, response_name="Ranking response")
    if not isinstance(payload, dict):
        raise ValueError("Ranking response must be a JSON object")
    required = ["ranked_ids", "summary", "why_top", "watchouts"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Ranking response missing keys: {', '.join(missing)}")
    if not isinstance(payload["ranked_ids"], list) or not payload["ranked_ids"]:
        raise ValueError("Ranking response must contain a non-empty `ranked_ids` list")
    payload["ranked_ids"] = [int(item) for item in payload["ranked_ids"]]
    if not isinstance(payload["watchouts"], list):
        raise ValueError("Ranking response `watchouts` must be a list")
    return payload
