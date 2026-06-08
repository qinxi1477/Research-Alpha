from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Sequence


TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "about",
    "after",
    "agent",
    "agents",
    "also",
    "among",
    "been",
    "being",
    "between",
    "both",
    "build",
    "could",
    "does",
    "doing",
    "during",
    "each",
    "easy",
    "from",
    "have",
    "idea",
    "into",
    "just",
    "make",
    "many",
    "might",
    "more",
    "most",
    "much",
    "need",
    "only",
    "other",
    "ours",
    "over",
    "paper",
    "papers",
    "same",
    "should",
    "some",
    "such",
    "than",
    "that",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "under",
    "very",
    "want",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "work",
    "would",
}


def normalize_token(token: str) -> str:
    value = token.lower().strip()
    if len(value) > 5 and value.endswith("ing"):
        value = value[:-3]
    elif len(value) > 4 and value.endswith("ied"):
        value = value[:-3] + "y"
    elif len(value) > 4 and value.endswith("ed"):
        value = value[:-2]
    elif len(value) > 4 and value.endswith("ies"):
        value = value[:-3] + "y"
    elif len(value) > 4 and value.endswith("s") and not value.endswith("ss"):
        value = value[:-1]
    return value


def tokenize(text: object) -> List[str]:
    if not text:
        return []
    tokens: List[str] = []
    for raw in TOKEN_RE.findall(str(text).lower()):
        normalized = normalize_token(raw)
        if len(normalized) < 3:
            continue
        if normalized.isdigit():
            continue
        if normalized in STOPWORDS:
            continue
        tokens.append(normalized)
    return tokens


def token_overlap_score(left: Sequence[str], right: Sequence[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    intersection = left_set & right_set
    if not intersection:
        return 0.0
    jaccard = len(intersection) / len(left_set | right_set)
    containment = len(intersection) / min(len(left_set), len(right_set))
    return round((0.45 * jaccard) + (0.55 * containment), 4)


def build_idea_text(payload: Dict[str, Any]) -> str:
    first_experiments = payload.get("first_experiments", [])
    experiments_text = " ".join(str(item).strip() for item in first_experiments) if isinstance(first_experiments, list) else ""
    fields = [
        payload.get("idea_title", ""),
        payload.get("core_hypothesis", ""),
        payload.get("historical_pattern", ""),
        payload.get("trend_support", ""),
        payload.get("frontier_gap", ""),
        payload.get("why_now", ""),
        payload.get("novelty", ""),
        payload.get("value", ""),
        payload.get("key_risk", ""),
        experiments_text,
        payload.get("evaluation_outline", ""),
        payload.get("paper_angle", ""),
    ]
    return " ".join(str(item).strip() for item in fields if str(item).strip())


def classify_overlap(title_score: float, body_score: float, combined_score: float) -> str:
    if combined_score >= 0.66 or (title_score >= 0.75 and body_score >= 0.48):
        return "duplicate"
    if combined_score >= 0.43 or (title_score >= 0.45 and body_score >= 0.32):
        return "near_miss"
    if combined_score >= 0.24:
        return "complementary"
    return "distant"


def top_overlap_terms(left: Iterable[str], right: Iterable[str], limit: int = 5) -> List[str]:
    overlap = sorted(set(left) & set(right))
    return overlap[: max(1, limit)]


def build_differentiation_note(label: str, top_match_title: str, overlap_terms: Sequence[str]) -> str:
    overlap_hint = ", ".join(overlap_terms[:3]) if overlap_terms else "the core framing"
    if label == "duplicate":
        return (
            f"Looks too close to `{top_match_title}` in the current local corpus. "
            f"Differentiate on mechanism, task regime, or evidence target rather than reusing {overlap_hint}."
        )
    if label == "near_miss":
        return (
            f"Close to `{top_match_title}`. Keep an explicit difference in method or evaluation so it does not collapse into {overlap_hint}."
        )
    if label == "complementary":
        return (
            f"Related to `{top_match_title}`, but it can still work if framed as an extension or boundary case around {overlap_hint}."
        )
    return "No strong local prior-art overlap found in the current metadata and abstract corpus."


def analyze_prior_art(
    *,
    idea_payload: Dict[str, Any],
    papers: Sequence[Dict[str, Any]],
    top_k: int = 3,
) -> Dict[str, Any]:
    idea_title_tokens = tokenize(idea_payload.get("idea_title", ""))
    idea_body_tokens = tokenize(build_idea_text(idea_payload))
    matches: List[Dict[str, Any]] = []

    for paper in papers:
        paper_title = str(paper.get("title", "")).strip()
        paper_abstract = str(paper.get("abstract", "")).strip()
        paper_title_tokens = tokenize(paper_title)
        paper_body_tokens = tokenize(f"{paper_title} {paper_abstract}")
        title_score = token_overlap_score(idea_title_tokens, paper_title_tokens)
        body_score = token_overlap_score(idea_body_tokens, paper_body_tokens)
        combined_score = round((0.45 * title_score) + (0.55 * body_score), 4)
        overlap_terms = top_overlap_terms(idea_body_tokens, paper_body_tokens)
        if combined_score <= 0 and not overlap_terms:
            continue
        matches.append(
            {
                "paper_id": int(paper.get("id", 0)),
                "title": paper_title,
                "venue": str(paper.get("venue", "")).strip(),
                "year": int(paper.get("year", 0) or 0),
                "combined_score": round(combined_score, 2),
                "title_score": round(title_score, 2),
                "body_score": round(body_score, 2),
                "overlap_terms": overlap_terms,
                "reason": (
                    "High title/body overlap."
                    if combined_score >= 0.43
                    else "Some overlapping framing or evaluation language."
                ),
            }
        )

    matches.sort(
        key=lambda item: (
            -float(item["combined_score"]),
            -float(item["body_score"]),
            -float(item["title_score"]),
            int(item["paper_id"]),
        )
    )
    top_matches = matches[: max(1, top_k)]
    if not top_matches:
        return {
            "overlap_label": "distant",
            "overlap_score": 0.0,
            "summary": "No strong local prior-art overlap found in the current metadata and abstract corpus.",
            "differentiation_note": "No strong local prior-art overlap found in the current metadata and abstract corpus.",
            "top_matches": [],
            "evidence_scope": "local_metadata_and_abstracts_only",
        }

    best_match = top_matches[0]
    label = classify_overlap(
        float(best_match["title_score"]),
        float(best_match["body_score"]),
        float(best_match["combined_score"]),
    )
    note = build_differentiation_note(label, str(best_match["title"]), list(best_match["overlap_terms"]))
    if label == "duplicate":
        summary = f"Closest local overlap is `{best_match['title']}` and the idea currently looks too close."
    elif label == "near_miss":
        summary = f"Closest local overlap is `{best_match['title']}`; the idea needs a sharper differentiation story."
    elif label == "complementary":
        summary = f"`{best_match['title']}` is related, but the idea still looks plausibly distinct."
    else:
        summary = "No strong local prior-art overlap found, beyond loose topic similarity."

    return {
        "overlap_label": label,
        "overlap_score": round(float(best_match["combined_score"]), 2),
        "summary": summary,
        "differentiation_note": note,
        "top_matches": top_matches,
        "evidence_scope": "local_metadata_and_abstracts_only",
    }
