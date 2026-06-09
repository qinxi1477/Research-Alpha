from __future__ import annotations

import json
from typing import Any, Dict

from research_alpha.full_text import sanitize_full_text_payload
from research_alpha.json_utils import parse_llm_json


REQUIRED_LOGIC_LINE_KEYS = [
    "old_belief",
    "bottleneck",
    "reframing",
    "why_now",
    "evidence_design",
    "failure_boundary",
]


GENOME_SYSTEM_PROMPT = """
You are extracting a structured Idea Genome Card for a top-tier AI/ML paper.
Use only the provided paper record and evidence sections.
Prefer full-text section snippets when they are supplied; otherwise stay at metadata/abstract level.
Do not claim full-paper certainty unless the prompt includes full_text_sections evidence.
For every logic-line step, cite a short source_text anchor from the provided
title/abstract/metadata/full-text snippets, then state the bounded inference. The logic-line text
may abstract and name the move; it should not merely copy the anchor. If the
provided record does not support a step, write
"INSUFFICIENT_EVIDENCE" for that step instead of inventing a story.
Do not copy abstract keywords as the idea. Extract the paper's logic line:
what the field believed, what bottleneck or hidden assumption the paper exposed,
how it reframed the problem, why the timing made sense, what evidence made the
claim credible, and where the idea should fail.
Return strict JSON with the requested keys and concise values.
""".strip()


def build_genome_prompt(paper: Dict[str, Any]) -> str:
    full_text_context = build_full_text_context(paper)
    evidence_level = "full_text_sections" if full_text_context["has_full_text_sections"] else "abstract_only"
    return f"""
Create an Idea Genome Card from this paper record.

The output is not a keyword summary. It must reconstruct the paper's argument
logic so later idea generation can verify whether a new idea is logically
grounded in this paper's standard.

Evidence policy:
- Use full_text_sections first when available, especially introduction/problem setup, related-work boundary, method reframing, experiments/ablation, limitations, conclusion, and review/rebuttal signal.
- If full_text_sections are missing or thin, you may use title/abstract/metadata, but mark unsupported logic steps as INSUFFICIENT_EVIDENCE.
- The paper's concrete method is not the transferable pattern. Extract the causal story move behind the method.
- Set evidence_level to "{evidence_level}" exactly.

Paper:
- title: {paper.get("title", "")}
- venue: {paper.get("venue", "")}
- year: {paper.get("year", "")}
- award: {paper.get("award", "")}
- citation_count: {paper.get("citation_count", 0)}
- influential_citation_count: {paper.get("influential_citation_count", 0)}
- abstract: {paper.get("abstract", "")}
- full_text_evidence: {json.dumps(full_text_context, ensure_ascii=False, indent=2)}

Return JSON only with this exact schema:
{{
  "paper_summary": "...",
  "pre_publication_belief": "...",
  "bottleneck_or_hidden_assumption": "...",
  "problem_reframing": "...",
  "why_now": "...",
  "evidence_design": "...",
  "story_line": "...",
  "transferable_pattern": "...",
  "failure_boundary": "...",
  "logic_line": {{
    "old_belief": "...",
    "bottleneck": "...",
    "reframing": "...",
    "why_now": "...",
    "evidence_design": "...",
    "failure_boundary": "..."
  }},
  "logic_line_evidence": {{
    "old_belief": {{"source_text": "...", "bounded_inference": "..."}},
    "bottleneck": {{"source_text": "...", "bounded_inference": "..."}},
    "reframing": {{"source_text": "...", "bounded_inference": "..."}},
    "why_now": {{"source_text": "...", "bounded_inference": "..."}},
    "evidence_design": {{"source_text": "...", "bounded_inference": "..."}},
    "failure_boundary": {{"source_text": "...", "bounded_inference": "..."}}
  }},
  "confidence_note": "...",
  "evidence_level": "{evidence_level}"
}}
""".strip()


def build_full_text_context(paper: Dict[str, Any]) -> Dict[str, Any]:
    raw = paper.get("full_text_json", "{}")
    if isinstance(raw, dict):
        payload = raw
    else:
        try:
            payload = json.loads(str(raw or "{}"))
        except (json.JSONDecodeError, TypeError):
            payload = {}
    payload = sanitize_full_text_payload(payload if isinstance(payload, dict) else {})
    sections = payload.get("sections") if isinstance(payload, dict) else {}
    if not isinstance(sections, dict):
        sections = {}
    preferred_order = [
        "introduction",
        "related_work",
        "method",
        "experiments",
        "ablation",
        "limitations",
        "rebuttal_or_review_signal",
        "conclusion",
    ]
    compact_sections = {
        key: compact_text(sections.get(key, ""), limit=1200)
        for key in preferred_order
        if str(sections.get(key, "")).strip()
    }
    return {
        "has_full_text_sections": bool(compact_sections),
        "source_scope": str(payload.get("source_scope", "")).strip() if isinstance(payload, dict) else "",
        "source_url": str(payload.get("source_url", "")).strip() if isinstance(payload, dict) else "",
        "available_sections": list(compact_sections.keys()),
        "sections": compact_sections,
    }


def compact_text(value: Any, *, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def parse_genome_response(text: str) -> Dict[str, Any]:
    payload = parse_llm_json(text, response_name="Genome response")
    if not isinstance(payload, dict):
        raise ValueError("Genome response must be a JSON object")
    required = [
        "paper_summary",
        "pre_publication_belief",
        "bottleneck_or_hidden_assumption",
        "problem_reframing",
        "why_now",
        "evidence_design",
        "story_line",
        "transferable_pattern",
        "failure_boundary",
        "confidence_note",
        "evidence_level",
    ]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Genome response missing keys: {', '.join(missing)}")
    logic_line = payload.get("logic_line")
    if not isinstance(logic_line, dict):
        logic_line = {
            "old_belief": payload.get("pre_publication_belief", ""),
            "bottleneck": payload.get("bottleneck_or_hidden_assumption", ""),
            "reframing": payload.get("problem_reframing", ""),
            "why_now": payload.get("why_now", ""),
            "evidence_design": payload.get("evidence_design", ""),
            "failure_boundary": payload.get("failure_boundary", ""),
        }
    missing_logic = [
        key
        for key in REQUIRED_LOGIC_LINE_KEYS
        if not str(logic_line.get(key, "")).strip()
    ]
    if missing_logic:
        raise ValueError(f"Genome response missing logic_line keys: {', '.join(missing_logic)}")
    payload["logic_line"] = {
        key: str(logic_line.get(key, "")).strip()
        for key in REQUIRED_LOGIC_LINE_KEYS
    }
    evidence = payload.get("logic_line_evidence")
    if isinstance(evidence, dict):
        payload["logic_line_evidence"] = {
            key: {
                "source_text": str((evidence.get(key) or {}).get("source_text", "")).strip()
                if isinstance(evidence.get(key), dict)
                else str(evidence.get(key, "")).strip(),
                "bounded_inference": str((evidence.get(key) or {}).get("bounded_inference", "")).strip()
                if isinstance(evidence.get(key), dict)
                else "",
            }
            for key in REQUIRED_LOGIC_LINE_KEYS
        }
    evidence_level = str(payload.get("evidence_level", "")).strip()
    payload["evidence_level"] = evidence_level if evidence_level in {"abstract_only", "full_text_sections"} else "abstract_only"
    return payload
