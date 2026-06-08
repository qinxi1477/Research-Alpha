from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List


MEMORY_VERSION = 1
SCORING_STANDARD_POLICY = (
    "Idea scoring standards must come from Gold Set high-quality papers, Genome Cards, "
    "and Pattern Cards. Frontier/recent papers are allowed only for trend, gap, and "
    "explicit limitation signals unless they are also in the Gold evidence context."
)


def has_chinese_text(value: object) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", stringify(value)))


def load_session_memory(session: Any) -> Dict[str, object]:
    raw = ""
    try:
        raw = str(session["memory_summary_json"]).strip()
    except (KeyError, TypeError, IndexError):
        raw = ""
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def build_session_memory_summary(
    session: Any,
    turns: Iterable[Any],
    session_context: Dict[str, object],
) -> Dict[str, object]:
    turn_items = list(turns)
    turn_payloads = [safe_turn_payload(turn) for turn in turn_items]
    decisions = [str(payload.get("decision", "")).strip().lower() for payload in turn_payloads]
    accepted_preferences: List[str] = []
    rejected_preferences: List[str] = []
    partial_preferences: List[str] = []
    risks: List[str] = []
    next_experiments: List[str] = []
    recent_turns: List[Dict[str, str]] = []

    for turn, payload in zip(turn_items, turn_payloads):
        instruction = row_value(turn, "user_instruction")
        decision = str(payload.get("decision", row_value(turn, "decision"))).strip().lower()
        revised = str(payload.get("revised_idea", row_value(turn, "revised_idea"))).strip()
        if decision == "accept":
            accepted_preferences.append(instruction)
        elif decision == "reject":
            rejected_preferences.append(instruction)
        elif decision == "partial":
            partial_preferences.append(instruction)
        risk = str(payload.get("why_user_may_be_wrong", "")).strip()
        if risk:
            risks.append(risk)
        feasibility = payload.get("feasibility_assessment")
        if isinstance(feasibility, dict):
            next_test = str(feasibility.get("next_test", "")).strip()
            if next_test:
                next_experiments.append(next_test)
        recent_turns.append(
            {
                "turn_index": str(row_value(turn, "turn_index")).strip(),
                "decision": decision or str(row_value(turn, "decision")).strip(),
                "user_instruction": compact_text(instruction, 240),
                "revised_idea": compact_text(revised, 360),
                "verdict_summary": compact_text(payload.get("verdict_summary", ""), 260),
            }
        )

    initial_idea = row_value(session, "initial_idea")
    current_idea = row_value(session, "current_idea")
    preferred_language = infer_preferred_language(session, turn_items, session_context)
    recent_limitations = extract_recent_limitations(session_context)
    review_focus = extract_review_focus(session_context)
    first_experiments = session_context.get("first_experiments", [])
    if isinstance(first_experiments, list):
        next_experiments.extend(str(item).strip() for item in first_experiments if str(item).strip())
    next_experiments.extend(item["next_test"] for item in review_focus if item.get("next_test"))

    return {
        "version": MEMORY_VERSION,
        "preferred_language": preferred_language,
        "turn_count": len(turn_items),
        "decision_tally": {
            "accept": decisions.count("accept"),
            "partial": decisions.count("partial"),
            "reject": decisions.count("reject"),
        },
        "user_direction": compact_text(session_context.get("query") or session_context.get("idea_title") or row_value(session, "name"), 360),
        "initial_idea": compact_text(initial_idea, 600),
        "current_best_idea": compact_text(current_idea, 800),
        "stable_thesis": compact_text(
            session_context.get("core_hypothesis") or current_idea or initial_idea,
            800,
        ),
        "evidence_basis": {
            "candidate_idea_id": session_context.get("candidate_idea_id"),
            "historical_pattern": compact_text(session_context.get("historical_pattern", ""), 500),
            "trend_support": compact_text(session_context.get("trend_support", ""), 500),
            "frontier_gap": compact_text(session_context.get("frontier_gap", ""), 500),
            "prior_art_label": compact_text(session_context.get("prior_art_label", ""), 160),
            "prior_art_summary": compact_text(session_context.get("prior_art_summary", ""), 360),
        },
        "scoring_standard_policy": SCORING_STANDARD_POLICY,
        "recent_limitations_focus": recent_limitations,
        "accepted_user_preferences": compact_list(accepted_preferences, limit=8, chars=240),
        "partially_accepted_preferences": compact_list(partial_preferences, limit=8, chars=240),
        "rejected_user_preferences": compact_list(rejected_preferences, limit=8, chars=240),
        "unresolved_risks": compact_list(
            risks
            or [item.get("risk", "") for item in review_focus]
            or [session_context.get("key_risk", "")],
            limit=8,
            chars=280,
        ),
        "reviewer_focus": review_focus,
        "next_experiments": compact_list(next_experiments, limit=8, chars=280),
        "recent_turns": recent_turns[-5:],
    }


def build_compressed_session_context(
    session: Any,
    turns: Iterable[Any],
    session_context: Dict[str, object],
    *,
    budget_chars: int = 12000,
) -> Dict[str, object]:
    turn_items = list(turns)
    memory = load_session_memory(session)
    if not memory:
        memory = build_session_memory_summary(session, turn_items, session_context)
    compressed = dict(session_context)
    compressed["memory_summary"] = memory
    compressed["context_cache"] = {
        "compression_policy": (
            "Full turn history is compressed into memory_summary. Use recent_turns for local continuity "
            "and session_context fields for original Gold/Pattern evidence grounding."
        ),
        "recent_turns": memory.get("recent_turns", []),
        "source": "idea_sessions.memory_summary_json",
    }
    if "critique_outputs" in compressed:
        compressed["critique_outputs"] = compact_json_like(compressed["critique_outputs"], 1800)
    return trim_context_to_budget(compressed, max(2000, budget_chars))


def render_memory_markdown(memory: Dict[str, object]) -> List[str]:
    if not memory:
        return ["- No compressed memory yet."]
    lines = [
        f"- Preferred language: {memory.get('preferred_language', 'auto')}",
        f"- Turn count: {memory.get('turn_count', 0)}",
        f"- Stable thesis: {memory.get('stable_thesis', '')}",
        f"- Scoring policy: {memory.get('scoring_standard_policy', '')}",
    ]
    evidence = memory.get("evidence_basis", {})
    if isinstance(evidence, dict):
        if str(evidence.get("historical_pattern", "")).strip():
            lines.append(f"- Historical pattern: {evidence.get('historical_pattern')}")
        if str(evidence.get("frontier_gap", "")).strip():
            lines.append(f"- Frontier gap: {evidence.get('frontier_gap')}")
    limitations = memory.get("recent_limitations_focus", [])
    if isinstance(limitations, list) and limitations:
        lines.append("- Recent limitations focus:")
        for item in limitations[:5]:
            if isinstance(item, dict):
                lines.append(f"  - {item.get('title', 'unknown')}: {item.get('limitation', '')}")
    risks = memory.get("unresolved_risks", [])
    if isinstance(risks, list) and risks:
        lines.append("- Unresolved risks:")
        for item in risks[:5]:
            lines.append(f"  - {item}")
    experiments = memory.get("next_experiments", [])
    if isinstance(experiments, list) and experiments:
        lines.append("- Next experiments:")
        for item in experiments[:5]:
            lines.append(f"  - {item}")
    return [line for line in lines if str(line).strip()]


def build_session_memory_status(memory: Dict[str, object]) -> Dict[str, object]:
    common_policy = {
        "memory_version": MEMORY_VERSION,
        "state_policy": "memory_summary_is_compressed_public_state_raw_turns_stay_in_session_history",
        "scoring_standard_policy": SCORING_STANDARD_POLICY,
        "user_library_policy": "User library papers are domain knowledge only and must not become scoring standards or storyline sources.",
    }
    if not memory:
        return {
            **common_policy,
            "preferred_language": "auto",
            "turn_count": 0,
            "stable_thesis": "",
            "recent_limitations_count": 0,
            "unresolved_risk": "",
            "next_experiment": "",
            "cache_policy": "No compressed memory yet.",
        }
    limitations = memory.get("recent_limitations_focus", [])
    risks = memory.get("unresolved_risks", [])
    experiments = memory.get("next_experiments", [])
    try:
        turn_count = int(memory.get("turn_count", 0) or 0)
    except (TypeError, ValueError):
        turn_count = 0
    return {
        **common_policy,
        "preferred_language": str(memory.get("preferred_language", "auto") or "auto"),
        "turn_count": turn_count,
        "stable_thesis": compact_text(memory.get("stable_thesis", ""), 260),
        "recent_limitations_count": len(limitations) if isinstance(limitations, list) else 0,
        "unresolved_risk": compact_text(risks[0], 220) if isinstance(risks, list) and risks else "",
        "next_experiment": compact_text(experiments[0], 220) if isinstance(experiments, list) and experiments else "",
        "cache_policy": "Full turn history is compressed into memory_summary and recent turns are kept for local continuity.",
    }


def infer_preferred_language(session: Any, turns: List[Any], session_context: Dict[str, object]) -> str:
    values: List[object] = [
        row_value(session, "name"),
        row_value(session, "initial_idea"),
        row_value(session, "current_idea"),
        session_context,
    ]
    values.extend(row_value(turn, "user_instruction") for turn in turns[-5:])
    return "zh" if any(has_chinese_text(value) for value in values) else "en"


def extract_recent_limitations(session_context: Dict[str, object]) -> List[Dict[str, str]]:
    candidates: List[Dict[str, str]] = []
    trend_signal = session_context.get("trend_signal")
    if isinstance(trend_signal, dict):
        candidates.extend(limitations_from_payload(trend_signal.get("recent_limitations")))
    candidates.extend(limitations_from_payload(session_context.get("recent_limitations")))
    risk = str(session_context.get("key_risk", "")).strip()
    if risk:
        candidates.append({"title": "originating_candidate", "limitation": compact_text(risk, 260)})
    return dedupe_limitations(candidates)[:8]


def extract_review_focus(session_context: Dict[str, object]) -> List[Dict[str, str]]:
    history = session_context.get("review_history", [])
    if not isinstance(history, list):
        return []
    result: List[Dict[str, str]] = []
    for item in history[-3:]:
        if not isinstance(item, dict):
            continue
        risks: List[str] = []
        for key in ("fatal_flaws", "main_attacks", "missing_logic_links"):
            values = item.get(key, [])
            if isinstance(values, list):
                risks.extend(str(value).strip() for value in values if str(value).strip())
        rethink = item.get("required_rethink", [])
        next_test = ""
        if isinstance(rethink, list):
            next_test = next((str(value).strip() for value in rethink if str(value).strip()), "")
        risk = next((value for value in risks if value), str(item.get("rethink_trigger", "")).strip())
        if risk or next_test:
            result.append(
                {
                    "decision": str(item.get("decision", "")).strip(),
                    "risk": compact_text(risk, 260),
                    "next_test": compact_text(next_test, 260),
                }
            )
    return result[-3:]


def limitations_from_payload(payload: object) -> List[Dict[str, str]]:
    if not isinstance(payload, dict):
        return []
    papers = payload.get("papers", [])
    if not isinstance(papers, list):
        return []
    result: List[Dict[str, str]] = []
    for paper in papers:
        if not isinstance(paper, dict):
            continue
        title = str(paper.get("title", "")).strip()
        limitations = paper.get("explicit_limitations", [])
        if isinstance(limitations, list) and limitations:
            for item in limitations[:2]:
                if isinstance(item, dict):
                    text = str(item.get("text", "")).strip()
                    if text:
                        result.append({"title": title, "limitation": compact_text(text, 260)})
        elif paper.get("needs_full_text_review"):
            result.append(
                {
                    "title": title,
                    "limitation": "No explicit limitation sentence in stored metadata; full-text review is required.",
                }
            )
    return result


def safe_turn_payload(turn: Any) -> Dict[str, object]:
    raw = row_value(turn, "content_json")
    try:
        payload = json.loads(str(raw))
    except (json.JSONDecodeError, TypeError):
        payload = {}
    return payload if isinstance(payload, dict) else {}


def row_value(row: Any, key: str) -> str:
    try:
        value = row[key]
    except (KeyError, TypeError, IndexError):
        try:
            value = dict(row).get(key, "")
        except (TypeError, ValueError):
            value = ""
    return stringify(value)


def stringify(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def redact_memory_text(value: object) -> str:
    text = stringify(value)
    key_name = r"(?:[A-Z0-9_]*(?:API[_-]?KEY|TOKEN|SECRET|PASSWORD))"
    text = re.sub(r"/(?:Users|private|var|tmp)/[^ \n\r\"'<>]+", "[local path]", text)
    text = re.sub(
        rf"(?i)\b({key_name})\s*([=:])\s*([^\s\"'<>]+)",
        lambda match: f"{match.group(1)}{match.group(2)}{' ' if match.group(2) == ':' else ''}[redacted]",
        text,
    )
    text = re.sub(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}", "Bearer [redacted]", text)
    text = re.sub(r"\bsk-[A-Za-z0-9][A-Za-z0-9._-]{6,}\b", "sk-[redacted]", text)
    return text


def compact_text(value: object, limit: int) -> str:
    text = re.sub(r"\s+", " ", redact_memory_text(value)).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "..."


def compact_list(values: Iterable[object], *, limit: int, chars: int) -> List[str]:
    result = []
    seen = set()
    for value in values:
        text = compact_text(value, chars)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def compact_json_like(value: object, limit: int) -> object:
    text = compact_text(value, limit)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def trim_context_to_budget(context: Dict[str, object], budget_chars: int) -> Dict[str, object]:
    text = json.dumps(context, ensure_ascii=False, sort_keys=True)
    if len(text) <= budget_chars:
        return context
    trimmed = dict(context)
    for key in ("critique_outputs", "first_experiments", "prior_art_summary"):
        if key in trimmed and len(json.dumps(trimmed, ensure_ascii=False)) > budget_chars:
            trimmed[key] = compact_text(trimmed[key], 800)
    text = json.dumps(trimmed, ensure_ascii=False, sort_keys=True)
    if len(text) <= budget_chars:
        return trimmed
    existing_cache = trimmed.get("context_cache", {})
    if not isinstance(existing_cache, dict):
        existing_cache = {}
    trimmed["context_cache"] = {
        **dict(existing_cache),
        "truncated": True,
        "budget_chars": budget_chars,
    }
    keep = {
        "source",
        "candidate_idea_id",
        "idea_title",
        "core_hypothesis",
        "historical_pattern",
        "trend_support",
        "frontier_gap",
        "why_now",
        "key_risk",
        "memory_summary",
        "context_cache",
    }
    return {key: value for key, value in trimmed.items() if key in keep}


def dedupe_limitations(items: List[Dict[str, str]]) -> List[Dict[str, str]]:
    result: List[Dict[str, str]] = []
    seen = set()
    for item in items:
        title = str(item.get("title", "")).strip()
        limitation = str(item.get("limitation", "")).strip()
        if not limitation:
            continue
        key = (title.lower(), limitation.lower())
        if key in seen:
            continue
        seen.add(key)
        result.append({"title": title, "limitation": limitation})
    return result
