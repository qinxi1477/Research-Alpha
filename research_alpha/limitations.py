from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta
from typing import Any, Dict, Iterable, List


LIMITATION_PATTERNS = [
    "limitation",
    "limited",
    "challenge",
    "future work",
    "fail",
    "failure",
    "bottleneck",
    "constraint",
    "cannot",
    "unable",
    "difficult",
    "hard to",
    "open problem",
    "robustness",
    "scalability",
    "data availability",
    "compute",
    "局限",
    "限制",
    "挑战",
    "未来工作",
    "失败",
    "不足",
    "瓶颈",
    "无法",
    "难以",
    "鲁棒",
    "可扩展",
    "数据",
    "算力",
]


def build_limitations_payload(paper: Dict[str, Any]) -> Dict[str, object]:
    text = " ".join(
        str(paper.get(field, "")).strip()
        for field in ("title", "abstract")
        if str(paper.get(field, "")).strip()
    )
    sentences = split_sentences(text)
    matches = []
    for sentence in sentences:
        lowered = sentence.lower()
        terms = [term for term in LIMITATION_PATTERNS if term.lower() in lowered or term in sentence]
        if terms:
            matches.append(
                {
                    "text": compact(sentence, 260),
                    "matched_terms": sorted(set(terms)),
                }
            )
    return {
        "source_scope": "metadata_title_abstract",
        "policy": (
            "Only explicit limitation/challenge/failure/future-work language from stored title/abstract is extracted. "
            "If this list is empty, full-paper limitation claims must not be inferred."
        ),
        "explicit_limitations": matches[:8],
        "explicit_count": len(matches[:8]),
        "needs_full_text_review": not bool(matches),
    }


def limitations_json_for_paper(paper: Dict[str, Any]) -> str:
    return json.dumps(build_limitations_payload(paper), ensure_ascii=False, sort_keys=True)


def split_sentences(text: str) -> List[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?。！？])\s+|(?<=[。！？])", cleaned)
    return [part.strip(" ;,，") for part in parts if part.strip(" ;,，")]


def compact(value: object, limit: int = 220) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def parse_publication_date(value: object) -> date | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            parsed = datetime.strptime(raw, fmt)
        except ValueError:
            continue
        return parsed.date()
    return None


def paper_publication_date(paper: Dict[str, Any]) -> date | None:
    parsed = parse_publication_date(paper.get("publication_date"))
    if parsed:
        return parsed
    try:
        year = int(paper.get("year", 0) or 0)
    except (TypeError, ValueError):
        return None
    if year <= 0:
        return None
    return date(year, 1, 1)


def recent_limitations_summary(
    papers: Iterable[Dict[str, Any]],
    *,
    months: int = 6,
    as_of: date | None = None,
    limit: int = 12,
) -> Dict[str, object]:
    today = as_of or date.today()
    cutoff = today - timedelta(days=max(1, months) * 31)
    recent = []
    for paper in papers:
        row = dict(paper)
        published = paper_publication_date(row)
        if not published or published < cutoff or published > today:
            continue
        limitations = row.get("limitations")
        if not isinstance(limitations, dict):
            raw = row.get("limitations_json", "{}")
            try:
                limitations = json.loads(str(raw or "{}"))
            except json.JSONDecodeError:
                limitations = build_limitations_payload(row)
        row["publication_date_resolved"] = published.isoformat()
        row["limitations"] = limitations
        recent.append(row)
    recent.sort(
        key=lambda item: (
            float(item.get("paper_weight", 0) or 0),
            int(item.get("citation_count", 0) or 0),
            item.get("publication_date_resolved", ""),
        ),
        reverse=True,
    )
    selected = []
    for row in recent[: max(1, limit)]:
        limitations = row.get("limitations") if isinstance(row.get("limitations"), dict) else {}
        selected.append(
            {
                "paper_id": int(row.get("id", 0) or 0),
                "title": str(row.get("title", "")).strip(),
                "venue": str(row.get("venue", "")).strip(),
                "year": int(row.get("year", 0) or 0),
                "publication_date": str(row.get("publication_date") or row.get("publication_date_resolved") or "").strip(),
                "source_kind": str(row.get("source_kind", "")).strip(),
                "citation_count": int(row.get("citation_count", 0) or 0),
                "paper_weight": float(row.get("paper_weight", 0) or 0),
                "explicit_limitations": limitations.get("explicit_limitations", []) if isinstance(limitations, dict) else [],
                "needs_full_text_review": bool(limitations.get("needs_full_text_review", True)) if isinstance(limitations, dict) else True,
            }
        )
    return {
        "policy": (
            "Recent limitation signals are extracted only from stored metadata/title/abstract for papers within the configured recent window. "
            "They should guide risk and frontier-gap analysis, not become Gold Set scoring standards."
        ),
        "as_of": today.isoformat(),
        "months": int(months),
        "cutoff_date": cutoff.isoformat(),
        "paper_count": len(recent),
        "papers": selected,
    }
