from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from research_alpha.json_utils import parse_llm_json


IDEA_SYSTEM_PROMPT = """
You are generating top-tier AI/ML research ideas from historical high-quality paper patterns.
Use the provided pattern cards and genome cards as evidence.
Do not generate generic or incremental ideas.
Do not transplant a source paper's concrete method, model, benchmark, dataset, or application object into the user's domain.
First abstract the source paper's causal storyline, then migrate that storyline to the current frontier.
Every idea must cite exact source_id values that appear in the supplied evidence context.
Return strict JSON only.
""".strip()

VALID_EVIDENCE_SOURCE_TYPES = {"pattern", "genome", "paper"}
VALID_EVIDENCE_USES = {"innovation", "logic", "feasibility", "value", "defensibility"}
REQUIRED_STORYLINE_TRACE_KEYS = [
    "old_belief",
    "bottleneck",
    "reframing",
    "why_now",
    "evidence_design",
    "failure_boundary",
]
EVALUATION_PROTOCOL_DIMENSIONS = [
    "practical_utility",
    "generalizability",
    "hypothesis_discovery_rate",
    "novelty_boundary",
    "impact_potential",
    "defensibility",
    "evidence_depth",
]


def build_idea_prompt(
    *,
    query: str,
    idea_count: int,
    pattern_cards: List[Dict[str, Any]],
    genome_cards: List[Dict[str, Any]],
    trend_signal: Dict[str, Any],
    quality_evidence: Dict[str, Any] | None = None,
    domain_knowledge: Dict[str, Any] | None = None,
) -> str:
    payload = {
        "query": query,
        "idea_count": idea_count,
        "pattern_cards": pattern_cards,
        "genome_cards": genome_cards,
        "trend_signal": trend_signal,
        "quality_evidence": quality_evidence or {},
        "domain_knowledge": domain_knowledge or {},
        "allowed_source_ids": build_allowed_source_ids(pattern_cards, genome_cards, quality_evidence or {}),
    }
    return (
        "Generate candidate research ideas for the following query.\n"
        "Use the historical evidence and frontier trend signal to propose ideas that are bold but defensible.\n"
        "If the query is written in Chinese, return Chinese idea titles and Chinese explanations unless the user clearly asks otherwise.\n"
        "Hard constraint: every idea must be grounded in the supplied high-quality paper evidence, not in your general prior.\n"
        "Treat best-paper, nominated/outstanding, oral/spotlight, and high-citation papers as higher-weight standards.\n"
        "Each idea should explicitly explain which historical pattern or high-weight paper standard it transfers.\n\n"
        "Scoring-standard discipline:\n"
        "- Innovation, logic, feasibility, value, and defensibility standards must come from supplied Genome/Pattern/Paper evidence.\n"
        "- Do not use your generic sense of what is novel or feasible as a scoring standard.\n"
        "- Prefer evidence sources marked full_text_sections or section-level Genome evidence when several allowed sources support the same logic move.\n"
        "- If you must use abstract-only evidence, keep the claim bounded and mention the evidence-depth risk in key_risk or evaluation protocol.\n"
        "- For each scoring dimension, evidence_basis.used_for must point to matched sources that justify that dimension.\n\n"
        "- Each idea's evidence_basis must cover all five dimensions exactly: innovation, logic, feasibility, value, defensibility.\n"
        "- Prefer one evidence_basis item per dimension, or use `used_for` with `innovation|logic|feasibility|value|defensibility` only when one source truly supports all five.\n\n"
        "Evidence discipline:\n"
        "- evidence_basis.source_type must be exactly one of: pattern, genome, paper.\n"
        "- evidence_basis.source_id must be copied exactly from a supplied pattern_key, paper_id, or title in Context.\n"
        "- Use only values from Context.allowed_source_ids for evidence_basis.source_id and storyline_trace.*.source_id.\n"
        "- Current/frontier/trend papers may support trend_support, frontier_gap, and why_now, but must not be used as scoring-standard evidence_basis unless they also appear in Context.allowed_source_ids.paper.\n"
        "- Context.domain_knowledge is user-provided paper-library background only: use it for terminology, domain routes, and problem framing, but never as evidence_basis, storyline_trace.source_id, quality_signal, or scoring standard.\n"
        "- Recent limitation signals in Context.trend_signal.recent_limitations must be used to shape frontier_gap, key_risk, first_experiments, and failure_boundary transfer; do not invent limitations that are not explicit in the stored metadata.\n"
        "- evidence_basis.used_for may combine only these dimensions: innovation, logic, feasibility, value, defensibility.\n"
        "- Do not invent source IDs, awards, paper titles, pattern keys, or scoring standards.\n"
        "- If the supplied Context does not support a strong idea, return fewer ideas rather than filling with prior-only ideas.\n\n"
        "Storyline discipline:\n"
        "- Hard boundary, soft expression: remain inside the supplied evidence's logic space, but do not fill a rigid template.\n"
        "- storyline_trace is an audit trail, not the writing template for the visible idea. Use it to prove the boundary; write the idea itself as a coherent domain-native research concept.\n"
        "- The idea must not be a fixed template or a simple collage of paper keywords.\n"
        "- You may abstract, rename, synthesize across allowed sources, choose a new mechanism, and invent domain-appropriate constructs, as long as the causal story arc remains evidence-grounded.\n"
        "- Do not force every visible field to mirror the six audit steps; the six steps should constrain the logic, not flatten the expression.\n"
        "- Do not transfer surface nouns, method names, model names, benchmark names, dataset names, or application objects from titles/abstracts; transfer the causal argument structure.\n"
        "- The source paper method is evidence for a story move, not the proposed method. If the idea could be described as `use the source paper method in the user domain`, reject it and rethink.\n"
        "- First extract a top-paper story arc: old belief -> bottleneck -> reframing -> why now -> evidence design -> failure boundary.\n"
        "- Then migrate that story arc to the current frontier trend, with exact source IDs for every arc step.\n"
        "- The candidate idea should be the result of that story migration, not a generic recombination of trend terms.\n\n"
        "Anti-copy discipline:\n"
        "- For every storyline_trace step, paper_story_standard must describe the source paper's abstract story role.\n"
        "- transfer_to_current_hotspot must be a domain-specific rewrite for the user's query; it must not repeat paper_story_standard with only nouns swapped.\n"
        "- paper_angle and core_hypothesis must name the new causal tension in the user's domain, not the source paper's concrete technique.\n\n"
        "Recent-limitations discipline:\n"
        "- Treat recent high-quality/frontier paper limitations as strong negative evidence: a good idea should either attack one explicit limitation or explain why it avoids that boundary.\n"
        "- If recent_limitations says full text review is needed, say so as a risk instead of pretending the limitation is known.\n\n"
        "Idea-evaluation discipline:\n"
        "- The project may generate ideas without executing experiments. Therefore each idea must include an idea_evaluation_protocol that evaluates the idea itself.\n"
        "- The protocol should cover practical utility, generalizability, hypothesis discovery rate, novelty boundary, impact potential, defensibility, and evidence depth.\n"
        "- Do not replace the protocol with a full experimental plan; first_experiments may suggest later validation, but idea_evaluation_protocol judges whether the idea is worth pursuing.\n\n"
        "Return JSON only with this exact schema:\n"
        "{\n"
        '  "ideas": [\n'
        "    {\n"
        '      "idea_title": "...",\n'
        '      "core_hypothesis": "...",\n'
        '      "historical_pattern": "...",\n'
        '      "trend_support": "...",\n'
        '      "frontier_gap": "...",\n'
        '      "why_now": "...",\n'
        '      "novelty": "...",\n'
        '      "value": "...",\n'
        '      "key_risk": "...",\n'
        '      "first_experiments": ["..."],\n'
        '      "evaluation_outline": "...",\n'
        '      "idea_evaluation_protocol": {\n'
        '        "purpose": "Evaluate idea quality, not execute the full research project.",\n'
        '        "dimensions": [\n'
        '          {"name": "practical_utility", "question": "...", "evidence_needed": "...", "pass_condition": "..."},\n'
        '          {"name": "generalizability", "question": "...", "evidence_needed": "...", "pass_condition": "..."},\n'
        '          {"name": "hypothesis_discovery_rate", "question": "...", "evidence_needed": "...", "pass_condition": "..."},\n'
        '          {"name": "novelty_boundary", "question": "...", "evidence_needed": "...", "pass_condition": "..."},\n'
        '          {"name": "impact_potential", "question": "...", "evidence_needed": "...", "pass_condition": "..."},\n'
        '          {"name": "defensibility", "question": "...", "evidence_needed": "...", "pass_condition": "..."},\n'
        '          {"name": "evidence_depth", "question": "...", "evidence_needed": "...", "pass_condition": "..."}\n'
        '        ],\n'
        '        "reject_if": ["..."]\n'
        '      },\n'
        '      "paper_angle": "...",\n'
        '      "storyline_trace": {\n'
        '        "old_belief": {"source_id": "...", "paper_story_standard": "...", "transfer_to_current_hotspot": "..."},\n'
        '        "bottleneck": {"source_id": "...", "paper_story_standard": "...", "transfer_to_current_hotspot": "..."},\n'
        '        "reframing": {"source_id": "...", "paper_story_standard": "...", "transfer_to_current_hotspot": "..."},\n'
        '        "why_now": {"source_id": "...", "paper_story_standard": "...", "transfer_to_current_hotspot": "..."},\n'
        '        "evidence_design": {"source_id": "...", "paper_story_standard": "...", "transfer_to_current_hotspot": "..."},\n'
        '        "failure_boundary": {"source_id": "...", "paper_story_standard": "...", "transfer_to_current_hotspot": "..."}\n'
        "      },\n"
        '      "evidence_basis": [\n'
        "        {\n"
        '          "source_type": "pattern|genome|paper",\n'
        '          "source_id": "...",\n'
        '          "quality_signal": "best_paper|oral|high_citation|paper_weight:...",\n'
        '          "borrowed_standard": "...",\n'
        '          "used_for": "innovation"\n'
        "        },\n"
        "        {\n"
        '          "source_type": "pattern|genome|paper",\n'
        '          "source_id": "...",\n'
        '          "quality_signal": "best_paper|oral|high_citation|paper_weight:...",\n'
        '          "borrowed_standard": "...",\n'
        '          "used_for": "logic|feasibility|value|defensibility"\n'
        "        }\n"
        "      ],\n"
        '      "generation_guardrail": "This idea is derived from the provided high-quality evidence by transferring ..."\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"Context:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def build_allowed_source_ids(
    pattern_cards: List[Dict[str, Any]],
    genome_cards: List[Dict[str, Any]],
    quality_evidence: Dict[str, Any],
) -> Dict[str, List[str]]:
    pattern_ids = [
        str(card.get("pattern_key", "")).strip()
        for card in pattern_cards
        if str(card.get("pattern_key", "")).strip()
    ]
    genome_ids = []
    paper_ids = []
    for card in genome_cards:
        title = str(card.get("title", "")).strip()
        paper_id = str(card.get("paper_id", "")).strip()
        if title:
            genome_ids.append(title)
        if paper_id:
            genome_ids.append(paper_id)
    for paper in quality_evidence.get("high_weight_papers", []):
        if not isinstance(paper, dict):
            continue
        title = str(paper.get("title", "")).strip()
        paper_id = str(paper.get("paper_id", "")).strip()
        if title:
            paper_ids.append(title)
        if paper_id:
            paper_ids.append(paper_id)
    return {
        "pattern": dedupe_strings(pattern_ids),
        "genome": dedupe_strings(genome_ids),
        "paper": dedupe_strings(paper_ids),
    }


def dedupe_strings(values: List[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def parse_idea_response(text: str) -> List[Dict[str, Any]]:
    payload = parse_llm_json(text, response_name="Idea response")
    if not isinstance(payload, dict):
        raise ValueError("Idea response must be a JSON object")
    ideas = payload.get("ideas")
    if not isinstance(ideas, list) or not ideas:
        raise ValueError("Idea response must contain a non-empty `ideas` list")

    required = [
        "idea_title",
        "core_hypothesis",
        "historical_pattern",
        "trend_support",
        "frontier_gap",
        "why_now",
        "novelty",
        "value",
        "key_risk",
        "first_experiments",
        "evaluation_outline",
        "paper_angle",
        "storyline_trace",
    ]
    normalized: List[Dict[str, Any]] = []
    seen_titles: set[str] = set()
    for item in ideas:
        if not isinstance(item, dict):
            raise ValueError("Each idea entry must be an object")
        missing = [key for key in required if key not in item]
        if missing:
            raise ValueError(f"Idea entry missing keys: {', '.join(missing)}")
        title_key = str(item.get("idea_title", "")).strip().lower()
        if title_key in seen_titles:
            raise ValueError("Idea response contains duplicate `idea_title` entries")
        seen_titles.add(title_key)
        evidence_basis = item.get("evidence_basis")
        if not isinstance(evidence_basis, list) or not evidence_basis:
            raise ValueError("Idea entry must include a non-empty `evidence_basis` list")
        for basis_item in evidence_basis:
            if not isinstance(basis_item, dict):
                raise ValueError("Each `evidence_basis` entry must be an object")
            basis_missing = [
                key
                for key in ("source_type", "source_id", "quality_signal", "borrowed_standard", "used_for")
                if not str(basis_item.get(key, "")).strip()
            ]
            if basis_missing:
                raise ValueError(f"Evidence basis entry missing keys: {', '.join(basis_missing)}")
            source_type = str(basis_item.get("source_type", "")).strip()
            if source_type not in VALID_EVIDENCE_SOURCE_TYPES:
                raise ValueError(
                    "Evidence basis `source_type` must be one of: "
                    + ", ".join(sorted(VALID_EVIDENCE_SOURCE_TYPES))
                )
            used_for = {
                value.strip()
                for value in str(basis_item.get("used_for", "")).split("|")
                if value.strip()
            }
            invalid_uses = sorted(used_for - VALID_EVIDENCE_USES)
            if invalid_uses:
                raise ValueError(
                    "Evidence basis `used_for` contains unsupported dimensions: "
                    + ", ".join(invalid_uses)
                )
        covered_uses = {
            value.strip()
            for basis_item in evidence_basis
            for value in str(basis_item.get("used_for", "")).split("|")
            if value.strip()
        }
        missing_uses = sorted(VALID_EVIDENCE_USES - covered_uses)
        if missing_uses:
            raise ValueError(
                "Idea entry `evidence_basis.used_for` must cover all scoring dimensions; missing: "
                + ", ".join(missing_uses)
            )
        if not str(item.get("generation_guardrail", "")).strip():
            raise ValueError("Idea entry must include a non-empty `generation_guardrail`")
        _validate_idea_content_quality(item)
        item["idea_evaluation_protocol"] = normalize_idea_evaluation_protocol(item)
        storyline_trace = item.get("storyline_trace")
        if not isinstance(storyline_trace, dict):
            raise ValueError("Idea entry must include a `storyline_trace` object")
        missing_trace = [key for key in REQUIRED_STORYLINE_TRACE_KEYS if key not in storyline_trace]
        if missing_trace:
            raise ValueError(f"Storyline trace missing keys: {', '.join(missing_trace)}")
        for trace_key in REQUIRED_STORYLINE_TRACE_KEYS:
            trace_item = storyline_trace.get(trace_key)
            if not isinstance(trace_item, dict):
                raise ValueError(f"Storyline trace `{trace_key}` must be an object")
            trace_missing = [
                key
                for key in ("source_id", "paper_story_standard", "transfer_to_current_hotspot")
                if not str(trace_item.get(key, "")).strip()
            ]
            if trace_missing:
                raise ValueError(
                    f"Storyline trace `{trace_key}` missing keys: {', '.join(trace_missing)}"
                )
            _validate_substantive_text(
                trace_item.get("paper_story_standard", ""),
                field_name=f"storyline_trace.{trace_key}.paper_story_standard",
                min_signal=12,
            )
            _validate_substantive_text(
                trace_item.get("transfer_to_current_hotspot", ""),
                field_name=f"storyline_trace.{trace_key}.transfer_to_current_hotspot",
                min_signal=12,
            )
        normalized.append(item)
    return normalized


def normalize_idea_evaluation_protocol(item: Dict[str, Any]) -> Dict[str, Any]:
    protocol = item.get("idea_evaluation_protocol")
    if not isinstance(protocol, dict):
        protocol = {}
    raw_dimensions = protocol.get("dimensions", [])
    if not isinstance(raw_dimensions, list):
        raw_dimensions = []
    by_name: Dict[str, Dict[str, str]] = {}
    for entry in raw_dimensions:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", "")).strip()
        if name not in EVALUATION_PROTOCOL_DIMENSIONS:
            continue
        by_name[name] = {
            "name": name,
            "question": str(entry.get("question", "")).strip(),
            "evidence_needed": str(entry.get("evidence_needed", "")).strip(),
            "pass_condition": str(entry.get("pass_condition", "")).strip(),
        }
    dimensions = []
    for name in EVALUATION_PROTOCOL_DIMENSIONS:
        existing = by_name.get(name, {})
        dimensions.append(
            {
                "name": name,
                "question": existing.get("question") or default_protocol_question(name),
                "evidence_needed": existing.get("evidence_needed") or default_protocol_evidence(name, item),
                "pass_condition": existing.get("pass_condition") or default_protocol_pass_condition(name),
            }
        )
    reject_if = protocol.get("reject_if", [])
    if not isinstance(reject_if, list):
        reject_if = []
    cleaned_reject_if = [str(value).strip() for value in reject_if if str(value).strip()]
    if not cleaned_reject_if:
        cleaned_reject_if = [
            "Reject if the closest prior art already covers the core hypothesis or only the nouns changed.",
            "Reject if the idea cannot cite supplied Gold/Genome/Pattern evidence for its logic line.",
            "Reject if evaluation only lists a case study and gives no measurable idea-quality criterion.",
        ]
    return {
        "purpose": str(protocol.get("purpose", "")).strip()
        or "Evaluate idea quality and paper-worthiness before any full experimental project.",
        "dimensions": dimensions,
        "reject_if": cleaned_reject_if,
    }


def default_protocol_question(name: str) -> str:
    return {
        "practical_utility": "Would this idea change what a researcher chooses, trusts, or rejects?",
        "generalizability": "Does the logic travel beyond one narrow dataset, agent, or domain example?",
        "hypothesis_discovery_rate": "Does the process produce more defensible hypotheses per expert-review cycle than a baseline ideation process?",
        "novelty_boundary": "Is the idea separated from the nearest 5-10 prior papers by a real causal or evaluative boundary?",
        "impact_potential": "Would a positive result alter a community assumption, benchmark, or evaluation practice?",
        "defensibility": "Can the idea survive a top-conference reviewer attack on novelty, logic, feasibility, and evidence?",
        "evidence_depth": "Is the logic grounded in full-text or section-level evidence rather than abstract-only guesses?",
    }[name]


def default_protocol_evidence(name: str, item: Dict[str, Any]) -> str:
    if name == "novelty_boundary":
        return "Prior-art gate closest_work, required_differentiation, and novelty field."
    if name == "evidence_depth":
        return "Genome/Pattern provenance, evidence_level, and cited source anchors."
    if name == "hypothesis_discovery_rate":
        return "A comparison of accepted, revised, and rejected ideas across repeated expert-review cycles."
    return str(item.get("evaluation_outline", "")).strip() or "Use the supplied evidence_basis, storyline_trace, and reviewer-loop dossier."


def default_protocol_pass_condition(name: str) -> str:
    return {
        "practical_utility": "At least one concrete decision or evaluation practice changes.",
        "generalizability": "The idea can name two or more settings where the same logic applies without copying methods.",
        "hypothesis_discovery_rate": "The review loop increases accepted non-duplicate ideas or reduces renamed-only failures.",
        "novelty_boundary": "Closest prior art leaves a stated missing part that the idea directly targets.",
        "impact_potential": "The paper angle would be meaningful even if the first implementation is small.",
        "defensibility": "No fatal reviewer attack remains after the review loop.",
        "evidence_depth": "Key logic steps cite section-level or explicitly bounded abstract evidence.",
    }[name]


def _validate_idea_content_quality(item: Dict[str, Any]) -> None:
    thresholds = {
        "idea_title": 8,
        "core_hypothesis": 20,
        "historical_pattern": 12,
        "frontier_gap": 16,
        "why_now": 12,
        "novelty": 16,
        "value": 12,
        "key_risk": 12,
        "evaluation_outline": 16,
        "paper_angle": 16,
        "generation_guardrail": 16,
    }
    for field, min_signal in thresholds.items():
        _validate_substantive_text(item.get(field, ""), field_name=field, min_signal=min_signal)
    experiments = item.get("first_experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("Idea response content_quality_failed: `first_experiments` must contain at least one concrete experiment.")
    for index, experiment in enumerate(experiments[:3], start=1):
        _validate_substantive_text(
            experiment,
            field_name=f"first_experiments[{index}]",
            min_signal=12,
        )


def _validate_substantive_text(value: Any, *, field_name: str, min_signal: int) -> None:
    text = str(value or "").strip()
    if _substantive_signal(text) >= min_signal and not _looks_like_placeholder(text):
        return
    raise ValueError(
        f"Idea response content_quality_failed: `{field_name}` is empty, placeholder-like, or too thin "
        "to safely save as a candidate idea."
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
        "looks good",
        "good",
        "fine",
        "idea",
        "new idea",
        "已完成",
        "完成",
        "可以",
        "很好",
        "同上",
        "无",
        "暂无",
        "待补充",
        "新 idea",
        "新idea",
        "一个新想法",
    }
    return stripped in placeholders
