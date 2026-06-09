from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html import unescape
from typing import Any, Dict, Iterable, List, Optional


class ConnectorError(RuntimeError):
    """Raised when a remote scholarly API call fails."""


@dataclass
class ConnectorConfig:
    api_key: str = ""
    email: str = ""


OPENALEX_VENUE_ALIASES = {
    "iclr": ["International Conference on Learning Representations"],
    "neurips": ["Neural Information Processing Systems", "Advances in Neural Information Processing Systems"],
    "nips": ["Neural Information Processing Systems", "Advances in Neural Information Processing Systems"],
    "icml": ["International Conference on Machine Learning"],
    "acl": ["Annual Meeting of the Association for Computational Linguistics"],
    "emnlp": ["Conference on Empirical Methods in Natural Language Processing"],
    "naacl": ["North American Chapter of the Association for Computational Linguistics"],
    "cvpr": ["Computer Vision and Pattern Recognition"],
    "iccv": ["International Conference on Computer Vision"],
    "eccv": ["European Conference on Computer Vision"],
    "aaai": ["AAAI Conference on Artificial Intelligence"],
    "ijcai": ["International Joint Conference on Artificial Intelligence"],
    "kdd": ["Knowledge Discovery and Data Mining"],
    "sigir": ["International ACM SIGIR Conference on Research and Development in Information Retrieval"],
    "www": ["The Web Conference", "World Wide Web Conference"],
    "chi": ["CHI Conference on Human Factors in Computing Systems", "ACM CHI Conference on Human Factors in Computing Systems"],
    "corl": ["Conference on Robot Learning"],
    "rss": ["Robotics: Science and Systems"],
    "tpami": ["IEEE Transactions on Pattern Analysis and Machine Intelligence"],
    "t-pami": ["IEEE Transactions on Pattern Analysis and Machine Intelligence"],
    "pami": ["IEEE Transactions on Pattern Analysis and Machine Intelligence"],
}

OPENALEX_ALLOWED_JOURNAL_ALIASES = {
    "ieee transactions on pattern analysis and machine intelligence",
}

OPENREVIEW_VENUE_IDS = {
    "iclr": "ICLR.cc/{year}/Conference",
    "neurips": "NeurIPS.cc/{year}/Conference",
    "nips": "NeurIPS.cc/{year}/Conference",
}

ICML_AWARDS_URL = "https://icml.cc/virtual/{year}/awards_detail"
ICML_ORAL_URLS = [
    "https://icml.cc/virtual/{year}/events/oral",
    "https://icml.cc/virtual/{year}/events/orals",
]
NEURIPS_AWARDS_URLS = [
    "https://papers.nips.cc/paper_files/paper/{year}",
    "https://papers.nips.cc/book/advances-in-neural-information-processing-systems-{volume}-{year}",
    "https://blog.neurips.cc/{year}/12/10/announcing-the-neurips-{year}-paper-awards/",
    "https://blog.neurips.cc/{year}/12/11/announcing-the-neurips-{year}-paper-awards/",
]
NEURIPS_ORAL_URLS = [
    "https://neurips.cc/virtual/{year}/events/oral",
    "https://neurips.cc/virtual/{year}/events/orals",
]
ACL_FAMILY_AWARDS_URLS = {
    "acl": [
        "https://{year}.aclweb.org/program/awards/",
        "https://{year}.aclweb.org/program/best_papers/",
    ],
    "emnlp": [
        "https://{year}.emnlp.org/program/best_papers/",
        "https://{year}.emnlp.org/program/awards/",
    ],
    "naacl": [
        "https://{year}.naacl.org/program/awards/",
        "https://{year}.naacl.org/program/best_papers/",
    ],
}
ECCV_AWARDS_URLS = [
    "https://eccv.ecva.net/Conferences/{year}/Awards",
]
CVF_AWARDS_URLS = {
    "cvpr": [
        "https://cvpr.thecvf.com/Conferences/{year}/BestPapersDemos",
        "https://cvpr.thecvf.com/Conferences/{year}/News/Awards",
        "https://cvpr.thecvf.com/Conferences/{year}/Awards",
    ],
    "iccv": [
        "https://iccv.thecvf.com/Conferences/{year}/News/Awards",
        "https://iccv.thecvf.com/Conferences/{year}/Awards",
        "https://iccv.thecvf.com/Conferences/{year}/BestPapersDemos",
    ],
}
CVF_ORAL_URLS = {
    "cvpr": [
        "https://cvpr.thecvf.com/virtual/{year}/events/oral",
        "https://cvpr.thecvf.com/virtual/{year}/events/orals",
    ],
    "iccv": [
        "https://iccv.thecvf.com/virtual/{year}/events/oral",
        "https://iccv.thecvf.com/virtual/{year}/events/orals",
    ],
}
AAAI_AWARDS_URLS = [
    "https://aaai.org/conference/aaai/aaai-{yy}/aaai-{yy}-paper-awards/",
    "https://aaai.org/conference/aaai/aaai-{year}/aaai-{year}-paper-awards/",
]
KDD_AWARDS_URLS = [
    "https://www.kdd.org/awards/view/{year}-sigkdd-best-paper-award-winners",
]


def harvest_openalex(
    *,
    venue: str,
    year: int,
    limit: int,
    query: str = "",
    config: ConnectorConfig | None = None,
) -> List[dict]:
    cfg = config or ConnectorConfig()
    params = {
        "filter": f"publication_year:{int(year)}",
        "per-page": str(max(1, min(limit, 200))),
        "sort": "cited_by_count:desc",
        "select": ",".join(
            [
                "id",
                "ids",
                "display_name",
                "publication_year",
                "publication_date",
                "cited_by_count",
                "primary_location",
                "locations",
                "primary_topic",
                "abstract_inverted_index",
            ]
        ),
    }
    if query.strip():
        params["search"] = query.strip()
    source_ids: List[str] = []
    if venue.strip():
        source_ids = resolve_openalex_source_ids(venue, cfg)
        if source_ids:
            params["filter"] += f",locations.source.id:{'|'.join(openalex_short_source_id(item) for item in source_ids)}"

    data = _get_json("https://api.openalex.org/works", params=params, api_key=cfg.api_key, email=cfg.email)
    records = [
        normalize_openalex_work(item, preferred_source_ids=source_ids)
        for item in data.get("results", [])
        if openalex_work_matches_target_year(item, int(year), preferred_source_ids=source_ids)
    ]
    if venue.strip() and not source_ids:
        return filter_records_by_venue(records, venue)
    return records


def resolve_openalex_source_ids(venue: str, config: ConnectorConfig | None = None) -> List[str]:
    cfg = config or ConnectorConfig()
    query_items = venue_query_candidates(venue)
    for query in query_items:
        data = _get_json(
            "https://api.openalex.org/sources",
            params={
                "search": query,
                "per-page": "10",
                "select": "id,display_name,type",
            },
            api_key=cfg.api_key,
            email=cfg.email,
        )
        ids = ranked_openalex_source_ids(data.get("results", []), query)
        if ids:
            return ids
    return []


def openalex_short_source_id(source_id: str) -> str:
    value = str(source_id).strip()
    if "/" in value:
        return value.rsplit("/", 1)[-1]
    return value


def venue_query_candidates(venue: str) -> List[str]:
    value = venue.strip()
    if not value:
        return []
    aliases = OPENALEX_VENUE_ALIASES.get(value.lower(), [])
    return dedupe_strings([*aliases, value])


def ranked_openalex_source_ids(results: Iterable[Dict[str, Any]], query: str) -> List[str]:
    query_key = query.strip().lower()
    ranked = []
    for item in results:
        source_id = str(item.get("id") or "").strip()
        display_name = str(item.get("display_name") or "").strip()
        source_type = str(item.get("type") or "").strip().lower()
        if not source_id or not display_name:
            continue
        display_key = display_name.lower()
        if query_key and display_key != query_key:
            continue
        score = 0
        if display_key == query_key:
            score += 4
        if source_type == "conference" or (source_type == "journal" and query_key in OPENALEX_ALLOWED_JOURNAL_ALIASES):
            score += 3
        else:
            continue
        ranked.append((score, source_id))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [source_id for score, source_id in ranked if score > 0]


def filter_records_by_venue(records: List[dict], venue: str) -> List[dict]:
    needles = [item.lower() for item in venue_query_candidates(venue)]
    if not needles:
        return records
    filtered = []
    for record in records:
        haystack = str(record.get("venue", "")).lower()
        if any(needle in haystack or haystack in needle for needle in needles):
            filtered.append(record)
    return filtered


def dedupe_strings(values: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        cleaned = str(value).strip()
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result


def harvest_semantic_scholar(
    *,
    query: str,
    limit: int,
    year: int | None = None,
    venue: str = "",
    config: ConnectorConfig | None = None,
) -> List[dict]:
    if not query.strip():
        raise ConnectorError("Semantic Scholar harvest requires a text query.")

    cfg = config or ConnectorConfig()
    params = {
        "query": query.strip(),
        "limit": str(max(1, min(limit, 100))),
        "fields": ",".join(
            [
                "title",
                "abstract",
                "year",
                "venue",
                "citationCount",
                "influentialCitationCount",
                "externalIds",
                "url",
                "publicationDate",
            ]
        ),
    }
    if year is not None:
        params["year"] = str(int(year))
    data = _get_json(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        params=params,
        api_key=cfg.api_key,
    )

    records = []
    for item in data.get("data", []):
        normalized = normalize_semantic_scholar_paper(item)
        if venue.strip() and venue.strip().lower() not in normalized["venue"].lower():
            continue
        records.append(normalized)
    return records


def harvest_openreview(
    *,
    venue: str,
    year: int,
    limit: int,
    query: str = "",
    config: ConnectorConfig | None = None,
) -> List[dict]:
    venue_id = openreview_venue_id(venue, year)
    if not venue_id:
        raise ConnectorError(f"OpenReview harvest is not configured for venue `{venue}`.")
    params = {
        "content.venueid": venue_id,
        "limit": str(max(1, min(limit, 1000))),
    }
    data = _get_json("https://api2.openreview.net/notes", params=params)
    records = [normalize_openreview_note(item, venue_hint=venue, year=year) for item in data.get("notes", [])]
    records = [record for record in records if record.get("title") and record.get("award")]
    if query.strip():
        records = filter_records_by_query_text(records, query, fallback_limit=limit)
    return records[: max(1, int(limit))]


def openreview_venue_id(venue: str, year: int) -> str:
    template = OPENREVIEW_VENUE_IDS.get(str(venue or "").strip().lower())
    return template.format(year=int(year)) if template else ""


def normalize_openreview_note(item: Dict[str, Any], *, venue_hint: str, year: int) -> Dict[str, Any]:
    content = item.get("content") or {}
    venue_text = openreview_content_value(content.get("venue"))
    abstract = openreview_content_value(content.get("abstract"))
    tldr = openreview_content_value(content.get("TLDR"))
    if tldr and tldr not in abstract:
        abstract = f"{tldr}\n\n{abstract}".strip()
    title = openreview_content_value(content.get("title"))
    pdf = openreview_content_value(content.get("pdf"))
    external_ref = str(item.get("id") or "").strip()
    if pdf:
        external_ref = f"https://openreview.net{pdf}" if pdf.startswith("/") else pdf
    return {
        "title": title,
        "abstract": abstract,
        "venue": venue_text or f"{venue_hint.upper()} {int(year)}",
        "year": int(year),
        "publication_date": "",
        "external_ref": external_ref,
        "citation_count": 0,
        "influential_citation_count": 0,
        "award": openreview_award_from_venue(venue_text),
    }


def openreview_content_value(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("value", "")
    if isinstance(value, list):
        return ", ".join(str(item).strip() for item in value if str(item).strip())
    return str(value or "").strip()


def openreview_award_from_venue(venue_text: str) -> str:
    value = str(venue_text or "").strip().lower()
    if "poster" in value:
        return ""
    if "spotlight" in value:
        return "spotlight"
    if openreview_is_outstanding_paper_signal(value):
        return "outstanding_paper"
    if "best paper" in value or ("award" in value and "paper" in value):
        return "best_paper"
    if "oral" in value:
        return "oral"
    return ""


def openreview_is_outstanding_paper_signal(value: str) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False
    if "outstanding" in text and "paper" in text:
        return True
    if "honorable mention" in text or "honourable mention" in text:
        return "paper" in text or "best" in text
    if "runner-up" in text or "runner up" in text:
        return "paper" in text or "best" in text
    if re.search(r"\bbest\s+paper\s+(?:award\s+)?(?:nominee|nomination|nominated|candidate)\b", text):
        return True
    if re.search(r"\bpaper\s+(?:award\s+)?(?:nominee|nomination|nominated)\b", text):
        return True
    return False


def harvest_icml_awards(*, year: int, limit: int, query: str = "") -> List[dict]:
    records: List[dict] = []
    last_error: Exception | None = None
    url = ICML_AWARDS_URL.format(year=int(year))
    try:
        html = _get_text(url)
        records.extend(parse_icml_awards_html(html, year=int(year)))
    except Exception as exc:  # pragma: no cover - depends on remote site availability
        last_error = exc
    for template in ICML_ORAL_URLS:
        oral_url = template.format(year=int(year))
        try:
            html = _get_text(oral_url)
        except Exception as exc:  # pragma: no cover - depends on remote site availability
            last_error = exc
            continue
        records.extend(parse_icml_oral_html(html, year=int(year), source_url=oral_url))
    records = dedupe_icml_award_records(records)
    if last_error and not records:
        raise ConnectorError(f"ICML awards/orals harvest failed for {year}: {last_error}")
    if query.strip():
        records = filter_records_by_query_text(records, query, fallback_limit=limit)
    return records[: max(1, int(limit))]


def harvest_neurips_awards(*, year: int, limit: int, query: str = "") -> List[dict]:
    last_error: Exception | None = None
    fetched_any = False
    collected: List[dict] = []
    for template in NEURIPS_AWARDS_URLS:
        url = template.format(year=int(year), volume=max(1, int(year) - 1987))
        try:
            html = _get_text(url)
        except Exception as exc:  # pragma: no cover - depends on remote site availability
            last_error = exc
            continue
        fetched_any = True
        records = parse_neurips_awards_html(html, year=int(year), source_url=url)
        collected.extend(records)
    for template in NEURIPS_ORAL_URLS:
        url = template.format(year=int(year))
        try:
            html = _get_text(url)
        except Exception as exc:  # pragma: no cover - depends on remote site availability
            last_error = exc
            continue
        fetched_any = True
        collected.extend(parse_neurips_oral_html(html, year=int(year), source_url=url))
    records = dedupe_cvf_award_records(collected)
    if records:
        if query.strip():
            records = filter_records_by_query_text(records, query, fallback_limit=limit)
        return records[: max(1, int(limit))]
    if last_error and not fetched_any:
        raise ConnectorError(f"NeurIPS awards/orals harvest failed for {year}: {last_error}")
    return []


def harvest_acl_awards(*, year: int, limit: int, query: str = "", venue: str = "ACL") -> List[dict]:
    venue_key = str(venue or "ACL").strip().lower()
    templates = ACL_FAMILY_AWARDS_URLS.get(venue_key)
    if not templates:
        raise ConnectorError("ACL-family awards harvest only supports venue `ACL`, `EMNLP`, or `NAACL`.")
    last_error: Exception | None = None
    for template in templates:
        url = template.format(year=int(year))
        try:
            html = _get_text(url)
        except Exception as exc:  # pragma: no cover - depends on remote site availability
            last_error = exc
            continue
        records = parse_acl_awards_html(html, year=int(year), source_url=url, venue=venue_key.upper())
        if records:
            if query.strip():
                records = filter_records_by_query_text(records, query, fallback_limit=limit)
            return records[: max(1, int(limit))]
    if last_error:
        raise ConnectorError(f"{venue_key.upper()} awards harvest failed for {year}: {last_error}")
    return []


def harvest_cvf_awards(*, venue: str, year: int, limit: int, query: str = "") -> List[dict]:
    venue_key = str(venue or "").strip().lower()
    templates = CVF_AWARDS_URLS.get(venue_key)
    if not templates:
        raise ConnectorError("CVF awards harvest only supports venue `CVPR` or `ICCV`.")
    last_error: Exception | None = None
    collected: List[dict] = []
    for template in templates:
        url = template.format(year=int(year))
        try:
            html = _get_text(url)
        except Exception as exc:  # pragma: no cover - depends on remote site availability
            last_error = exc
            continue
        records = parse_cvf_awards_html(html, venue=venue_key.upper(), year=int(year), source_url=url)
        if records:
            collected.extend(records)
    for template in CVF_ORAL_URLS.get(venue_key, []):
        url = template.format(year=int(year))
        try:
            html = _get_text(url)
        except Exception as exc:  # pragma: no cover - depends on remote site availability
            last_error = exc
            continue
        collected.extend(parse_cvf_oral_html(html, venue=venue_key.upper(), year=int(year), source_url=url))
    records = dedupe_cvf_award_records(collected)
    if records:
        if query.strip():
            records = filter_records_by_query_text(records, query, fallback_limit=limit)
        return records[: max(1, int(limit))]
    if last_error and not collected:
        raise ConnectorError(f"{venue_key.upper()} awards harvest failed for {year}: {last_error}")
    return []


def harvest_eccv_awards(*, year: int, limit: int, query: str = "") -> List[dict]:
    last_error: Exception | None = None
    for template in ECCV_AWARDS_URLS:
        url = template.format(year=int(year))
        try:
            html = _get_text(url)
        except Exception as exc:  # pragma: no cover - depends on remote site availability
            last_error = exc
            continue
        records = parse_eccv_awards_html(html, year=int(year), source_url=url)
        if records:
            if query.strip():
                records = filter_records_by_query_text(records, query, fallback_limit=limit)
            return records[: max(1, int(limit))]
    if last_error:
        raise ConnectorError(f"ECCV awards harvest failed for {year}: {last_error}")
    return []


def harvest_aaai_awards(*, year: int, limit: int, query: str = "") -> List[dict]:
    last_error: Exception | None = None
    for template in AAAI_AWARDS_URLS:
        url = template.format(year=int(year), yy=str(int(year))[-2:])
        try:
            html = _get_text(url)
        except Exception as exc:  # pragma: no cover - depends on remote site availability
            last_error = exc
            continue
        records = parse_generic_awards_html(html, venue="AAAI", year=int(year), source_url=url)
        if records:
            if query.strip():
                records = filter_records_by_query_text(records, query, fallback_limit=limit)
            return records[: max(1, int(limit))]
    if last_error:
        raise ConnectorError(f"AAAI awards harvest failed for {year}: {last_error}")
    return []


def harvest_kdd_awards(*, year: int, limit: int, query: str = "") -> List[dict]:
    last_error: Exception | None = None
    for template in KDD_AWARDS_URLS:
        url = template.format(year=int(year))
        try:
            html = _get_text(url)
        except Exception as exc:  # pragma: no cover - depends on remote site availability
            last_error = exc
            continue
        records = parse_generic_awards_html(html, venue="KDD", year=int(year), source_url=url)
        if records:
            if query.strip():
                records = filter_records_by_query_text(records, query, fallback_limit=limit)
            return records[: max(1, int(limit))]
    if last_error:
        raise ConnectorError(f"KDD awards harvest failed for {year}: {last_error}")
    return []


def parse_neurips_awards_html(html: str, *, year: int, source_url: str = "") -> List[dict]:
    records: List[dict] = []
    main_html = cvf_main_content(html)
    records.extend(neurips_section_award_records(main_html, year=year, source_url=source_url))
    records.extend(neurips_badged_paper_records(main_html, year=year, source_url=source_url))
    return dedupe_cvf_award_records(records)


def parse_neurips_oral_html(html: str, *, year: int, source_url: str = "") -> List[dict]:
    records: List[dict] = []
    main_html = cvf_main_content(html)
    for item in cvf_award_items(main_html):
        title = str(item.get("title", "")).strip()
        if not title:
            continue
        href = str(item.get("href", "")).strip()
        context = " ".join([title, href, str(item.get("type", "")), str(item.get("body", "")), source_url])
        if cvf_is_excluded_award_context(context):
            continue
        if not is_official_oral_record(href=href, record_type=str(item.get("type", "")), body=str(item.get("body", "")), source_url=source_url):
            continue
        authors = str(item.get("authors", "")).strip()
        body = str(item.get("body", "")).strip()
        abstract_parts = []
        if authors:
            abstract_parts.append(f"Authors: {authors}")
        if body and body.lower() != title.lower():
            abstract_parts.append(body)
        abstract = "\n\n".join(abstract_parts).strip()
        records.append(
            {
                "title": title,
                "abstract": abstract,
                "venue": f"NeurIPS {int(year)} Oral",
                "year": int(year),
                "publication_date": "",
                "external_ref": urllib.parse.urljoin(source_url, href) if href else source_url,
                "citation_count": 0,
                "influential_citation_count": 0,
                "award": "oral",
            }
        )
    return dedupe_cvf_award_records(records)


def neurips_section_award_records(html: str, *, year: int, source_url: str) -> List[dict]:
    records: List[dict] = []
    for label, section_html in cvf_award_sections(html):
        award = neurips_award_key(label)
        if not award:
            continue
        for item in cvf_award_items(section_html):
            title = str(item.get("title", "")).strip()
            href = str(item.get("href", "")).strip()
            context = " ".join([title, label, href, str(item.get("type", ""))])
            if not title or cvf_is_excluded_award_context(context):
                continue
            authors = str(item.get("authors", "")).strip()
            body = str(item.get("body", "")).strip()
            abstract = f"Authors: {authors}".strip() if authors else ""
            if body and body.lower() != title.lower() and body not in abstract:
                abstract = f"{abstract}\n\n{body}".strip()
            records.append(
                {
                    "title": title,
                    "abstract": abstract,
                    "venue": f"NeurIPS {int(year)} {label}",
                    "year": int(year),
                    "publication_date": "",
                    "external_ref": urllib.parse.urljoin(source_url, href) if href else source_url,
                    "citation_count": 0,
                    "influential_citation_count": 0,
                    "award": award,
                }
            )
    return records


def neurips_badged_paper_records(html: str, *, year: int, source_url: str) -> List[dict]:
    records: List[dict] = []
    for item_html in re.findall(r"<li\b[^>]*>(.*?)</li>", html, flags=re.IGNORECASE | re.DOTALL):
        title, href = title_href_from_html(item_html)
        if not title:
            continue
        context = " ".join([title, href, clean_html_text(item_html)])
        if cvf_is_excluded_award_context(context):
            continue
        award = neurips_award_key(context)
        if not award:
            continue
        authors = authors_from_common_html(item_html)
        records.append(
            {
                "title": title,
                "abstract": f"Authors: {authors}".strip() if authors else "",
                "venue": f"NeurIPS {int(year)} {neurips_award_label_for_key(award)}",
                "year": int(year),
                "publication_date": "",
                "external_ref": urllib.parse.urljoin(source_url, href) if href else source_url,
                "citation_count": 0,
                "influential_citation_count": 0,
                "award": award,
            }
        )
    return records


def neurips_award_key(label: str) -> str:
    value = str(label or "").strip().lower()
    if not value or cvf_is_excluded_award_context(value):
        return ""
    if "test of time" in value or "test-of-time" in value:
        return "test_of_time"
    if "runner" in value and "up" in value and "paper" in value:
        return "outstanding_paper"
    if "honorable mention" in value or "honourable mention" in value:
        return "outstanding_paper"
    if "outstanding" in value and "paper" in value:
        return "outstanding_paper"
    if "best paper" in value or "paper award" in value:
        return "best_paper"
    return ""


def neurips_award_label_for_key(award: str) -> str:
    return {
        "best_paper": "Best Paper",
        "outstanding_paper": "Outstanding Paper",
        "test_of_time": "Test of Time",
    }.get(str(award or "").strip(), "Award")


def parse_generic_awards_html(html: str, *, venue: str, year: int, source_url: str = "") -> List[dict]:
    records: List[dict] = []
    main_html = cvf_main_content(html)
    for label, section_html in cvf_award_sections(main_html):
        award = generic_award_key(label)
        if not award:
            continue
        for item in generic_award_items(section_html):
            title = str(item.get("title", "")).strip()
            href = str(item.get("href", "")).strip()
            context = " ".join([title, label, href, str(item.get("body", ""))])
            if not title or generic_is_excluded_award_context(context):
                continue
            authors = str(item.get("authors", "")).strip()
            body = str(item.get("body", "")).strip()
            abstract = f"Authors: {authors}".strip() if authors else ""
            if body and body.lower() != title.lower() and body not in abstract:
                abstract = f"{abstract}\n\n{body}".strip()
            records.append(
                {
                    "title": title,
                    "abstract": abstract,
                    "venue": f"{str(venue).upper()} {int(year)} {label}",
                    "year": int(year),
                    "publication_date": "",
                    "external_ref": urllib.parse.urljoin(source_url, href) if href else source_url,
                    "citation_count": 0,
                    "influential_citation_count": 0,
                    "award": award,
                }
            )
    return dedupe_cvf_award_records(records)


def generic_award_key(label: str) -> str:
    value = str(label or "").strip().lower()
    if not value or generic_is_excluded_award_context(value):
        return ""
    if "test of time" in value or "test-of-time" in value:
        return "test_of_time"
    if "runner" in value and "up" in value and "paper" in value:
        return "outstanding_paper"
    if "honorable mention" in value or "honourable mention" in value:
        return "outstanding_paper"
    if "outstanding" in value and "paper" in value:
        return "outstanding_paper"
    if "best paper" in value or "best research paper" in value or "paper award" in value:
        return "best_paper"
    return ""


def generic_award_items(section_html: str) -> List[dict]:
    items: List[dict] = []
    for item_html in re.findall(r"<li\b[^>]*>(.*?)</li>", section_html, flags=re.IGNORECASE | re.DOTALL):
        item = generic_item_from_html(item_html)
        if item.get("title"):
            items.append(item)
    if items:
        return items
    for paragraph_html in re.findall(r"<p\b[^>]*>(.*?)</p>", section_html, flags=re.IGNORECASE | re.DOTALL):
        item = generic_item_from_html(paragraph_html)
        if item.get("title"):
            items.append(item)
    if items:
        return items
    for block_html in re.findall(r"<div\b[^>]*>(.*?)</div>", section_html, flags=re.IGNORECASE | re.DOTALL):
        item = generic_item_from_html(block_html)
        if item.get("title"):
            items.append(item)
    return items


def generic_item_from_html(item_html: str) -> dict:
    title, href = title_href_from_html(item_html)
    if not title:
        title = generic_title_from_text(clean_html_text(item_html))
    authors = authors_from_common_html(item_html)
    body = clean_html_text(item_html)
    return {
        "title": title,
        "href": href,
        "authors": authors,
        "body": body,
    }


def generic_title_from_text(text: str) -> str:
    value = re.sub(
        r"^(best|outstanding|runner[- ]?up|honou?rable mention|paper award|winner|award)\b[:\s-]*",
        "",
        str(text or "").strip(),
        flags=re.IGNORECASE,
    )
    for marker in [" Authors:", " by ", " — ", " - "]:
        if marker in value:
            value = value.split(marker, 1)[0]
    if len(value) > 220:
        return ""
    return value.strip(" :;-")


def generic_is_excluded_award_context(value: str) -> bool:
    return cvf_is_excluded_award_context(value)


def parse_eccv_awards_html(html: str, *, year: int, source_url: str = "") -> List[dict]:
    records: List[dict] = []
    main_html = cvf_main_content(html)
    records.extend(cvf_sequence_award_records(main_html, venue="ECCV", year=year, source_url=source_url, page_title=cvf_page_title(html)))
    for label, section_html in cvf_award_sections(main_html):
        award = eccv_award_key(label)
        if not award:
            continue
        for item in cvf_award_items(section_html):
            title = str(item.get("title", "")).strip()
            href = str(item.get("href", "")).strip()
            context = " ".join([title, label, href, str(item.get("type", ""))])
            if not title or cvf_is_excluded_award_context(context):
                continue
            authors = str(item.get("authors", "")).strip()
            body = str(item.get("body", "")).strip()
            abstract = f"Authors: {authors}".strip() if authors else ""
            if body and body.lower() != title.lower() and body not in abstract:
                abstract = f"{abstract}\n\n{body}".strip()
            records.append(
                {
                    "title": title,
                    "abstract": abstract,
                    "venue": f"ECCV {int(year)} {label}",
                    "year": int(year),
                    "publication_date": "",
                    "external_ref": urllib.parse.urljoin(source_url, href) if href else source_url,
                    "citation_count": 0,
                    "influential_citation_count": 0,
                    "award": award,
                }
            )
    return dedupe_cvf_award_records(records)


def eccv_award_key(label: str) -> str:
    value = str(label or "").strip().lower()
    if not value or cvf_is_excluded_award_context(value):
        return ""
    if "honorable mention" in value or "honourable mention" in value:
        return "outstanding_paper"
    if "award candidate" in value and "paper" in value:
        return "outstanding_paper"
    if "best paper" in value:
        return "best_paper"
    return ""


def parse_cvf_awards_html(html: str, *, venue: str, year: int, source_url: str = "") -> List[dict]:
    records: List[dict] = []
    main_html = cvf_main_content(html)
    title_text = cvf_page_title(html)
    records.extend(cvf_sequence_award_records(main_html, venue=venue, year=year, source_url=source_url, page_title=title_text))
    for label, section_html in cvf_award_sections(main_html):
        award = cvf_award_key(label)
        if not award:
            continue
        for item in cvf_award_items(section_html):
            title = str(item.get("title", "")).strip()
            if not title:
                continue
            href = str(item.get("href", "")).strip()
            if not href or not cvf_is_paper_href(href):
                continue
            context = " ".join(
                str(value or "")
                for value in [
                    title,
                    label,
                    href,
                    item.get("type", ""),
                ]
            )
            if cvf_is_excluded_award_context(context):
                continue
            abstract = ""
            authors = str(item.get("authors", "")).strip()
            body = str(item.get("body", "")).strip()
            if authors:
                abstract = f"Authors: {authors}"
            if body and body.lower() != title.lower() and body not in abstract:
                abstract = f"{abstract}\n\n{body}".strip()
            records.append(
                {
                    "title": title,
                    "abstract": abstract,
                    "venue": f"{str(venue).upper()} {int(year)} {label}",
                    "year": int(year),
                    "publication_date": "",
                    "external_ref": urllib.parse.urljoin(source_url, href),
                    "citation_count": 0,
                    "influential_citation_count": 0,
                    "award": award,
                }
            )
    return dedupe_cvf_award_records(records)


def parse_cvf_oral_html(html: str, *, venue: str, year: int, source_url: str = "") -> List[dict]:
    records: List[dict] = []
    main_html = cvf_main_content(html)
    for item in cvf_award_items(main_html):
        title = str(item.get("title", "")).strip()
        if not title:
            continue
        href = str(item.get("href", "")).strip()
        context = " ".join([title, href, str(item.get("type", "")), str(item.get("body", "")), source_url])
        if cvf_is_excluded_award_context(context):
            continue
        if not is_official_oral_record(href=href, record_type=str(item.get("type", "")), body=str(item.get("body", "")), source_url=source_url):
            continue
        authors = str(item.get("authors", "")).strip()
        body = str(item.get("body", "")).strip()
        abstract = f"Authors: {authors}".strip() if authors else ""
        if body and body.lower() != title.lower() and body not in abstract:
            abstract = f"{abstract}\n\n{body}".strip()
        records.append(
            {
                "title": title,
                "abstract": abstract,
                "venue": f"{str(venue).upper()} {int(year)} Oral",
                "year": int(year),
                "publication_date": "",
                "external_ref": urllib.parse.urljoin(source_url, href) if href else source_url,
                "citation_count": 0,
                "influential_citation_count": 0,
                "award": "oral",
            }
        )
    return dedupe_cvf_award_records(records)


def cvf_sequence_award_records(html: str, *, venue: str, year: int, source_url: str, page_title: str) -> List[dict]:
    records: List[dict] = []
    current_award = ""
    current_label = ""
    current_title = ""
    current_authors = ""
    stop_section = False
    scan_html = html
    for paragraph_html in re.findall(r"<p\b[^>]*>(.*?)</p>", scan_html, flags=re.IGNORECASE | re.DOTALL):
        text = clean_html_text(paragraph_html)
        if not text:
            continue
        if re.match(r"^(honorable|honourable) mention papers included\b", text.lower()):
            for link_match in re.finditer(r'<a\b[^>]+href="([^"]+)"[^>]*>(.*?)</a>', paragraph_html, flags=re.IGNORECASE | re.DOTALL):
                href = unescape(str(link_match.group(1)).strip())
                title = clean_html_text(link_match.group(2))
                if title and cvf_is_paper_href(href) and not cvf_is_excluded_award_context(" ".join([title, href])):
                    records.append(
                        {
                            "title": title,
                            "abstract": "",
                            "venue": f"{str(venue).upper()} {int(year)} Best Paper Honorable Mention",
                            "year": int(year),
                            "publication_date": "",
                            "external_ref": urllib.parse.urljoin(source_url, href),
                            "citation_count": 0,
                            "influential_citation_count": 0,
                            "award": "outstanding_paper",
                        }
                    )
            continue
        label = cvf_award_label_from_text(text)
        excluded_label = cvf_excluded_label_from_text(text)
        if label or excluded_label:
            label = label or excluded_label
            current_label = label
            current_award = cvf_award_key(label)
            current_title = ""
            current_authors = ""
            stop_section = current_award == "" or cvf_is_excluded_award_context(label)
            continue
        if stop_section or not current_award:
            continue
        if text.lower().startswith("paper name:"):
            current_title = text.split(":", 1)[1].strip()
            if current_title:
                append_cvf_sequence_record(records, title=current_title, authors="", award=current_award, label=current_label, venue=venue, year=year, source_url=source_url, page_title=page_title)
            continue
        if text.lower().startswith("authors:"):
            current_authors = text.split(":", 1)[1].strip()
            if current_title:
                context = " ".join([current_title, current_label])
                if not cvf_is_excluded_award_context(context):
                    update_cvf_record_authors(records, current_title, current_award, current_authors)
                    if not any(
                        str(record.get("title", "")).strip().lower() == current_title.lower()
                        and str(record.get("award", "")).strip() == current_award
                        for record in records
                    ):
                        records.append(
                            {
                                "title": current_title,
                                "abstract": f"Authors: {current_authors}".strip() if current_authors else "",
                                "venue": f"{str(venue).upper()} {int(year)} {current_label}",
                                "year": int(year),
                                "publication_date": "",
                                "external_ref": source_url,
                                "citation_count": 0,
                                "influential_citation_count": 0,
                                "award": current_award,
                            }
                        )
                current_title = ""
                current_authors = ""
            continue
        strong_match = re.search(r"<strong[^>]*>(.*?)</strong>", paragraph_html, flags=re.IGNORECASE | re.DOTALL)
        if strong_match:
            current_title = clean_html_text(strong_match.group(1))
            append_cvf_sequence_record(records, title=current_title, authors="", award=current_award, label=current_label, venue=venue, year=year, source_url=source_url, page_title=page_title)
    return records


def append_cvf_sequence_record(
    records: List[dict],
    *,
    title: str,
    authors: str,
    award: str,
    label: str,
    venue: str,
    year: int,
    source_url: str,
    page_title: str,
) -> None:
    title = str(title or "").strip()
    if not title:
        return
    context = " ".join([title, label])
    if cvf_is_excluded_award_context(context):
        return
    for record in records:
        if str(record.get("title", "")).strip().lower() == title.lower() and str(record.get("award", "")).strip() == award:
            if authors:
                record["abstract"] = f"Authors: {authors}".strip()
            return
    records.append(
        {
            "title": title,
            "abstract": f"Authors: {authors}".strip() if authors else "",
            "venue": f"{str(venue).upper()} {int(year)} {label}",
            "year": int(year),
            "publication_date": "",
            "external_ref": source_url,
            "citation_count": 0,
            "influential_citation_count": 0,
            "award": award,
        }
    )


def update_cvf_record_authors(records: List[dict], title: str, award: str, authors: str) -> None:
    if not authors:
        return
    title_key = str(title or "").strip().lower()
    for record in reversed(records):
        if str(record.get("title", "")).strip().lower() == title_key and str(record.get("award", "")).strip() == award:
            record["abstract"] = f"Authors: {authors}".strip()
            return


def cvf_award_label_from_text(text: str) -> str:
    value = str(text or "").strip()
    lower = value.lower()
    if len(value) > 120:
        return ""
    if not value.endswith(":") and not re.match(r"^(honorable|honourable) mention papers included\b", lower):
        return ""
    if "paper name:" in lower or "authors:" in lower:
        return ""
    if "best paper" in lower or "honorable mention" in lower or "honourable mention" in lower or "marr prize" in lower:
        return value.strip(" :")
    return ""


def cvf_excluded_label_from_text(text: str) -> str:
    value = str(text or "").strip()
    lower = value.lower()
    if len(value) > 120:
        return ""
    if not value.endswith(":"):
        return ""
    if "best student" in lower or "student paper" in lower or "demo" in lower or "workshop" in lower:
        return value.strip(" :")
    return ""


def cvf_main_content(html: str) -> str:
    match = re.search(r"<main\b[^>]*>(.*?)</main>", html, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1)
    match = re.search(r'<div\b[^>]+class="[^"]*\bcontainer\b[^"]*"[^>]*>(.*)</div>', html, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1) if match else html


def cvf_page_title(html: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
    return clean_html_text(match.group(1)) if match else ""


def cvf_award_sections(html: str) -> List[tuple[str, str]]:
    heading_matches = list(re.finditer(r"<h([1-4])[^>]*>(.*?)</h\1>", html, flags=re.IGNORECASE | re.DOTALL))
    sections: List[tuple[str, str]] = []
    if heading_matches:
        for index, heading in enumerate(heading_matches):
            label = clean_html_text(heading.group(2))
            section_start = heading.end()
            section_end = heading_matches[index + 1].start() if index + 1 < len(heading_matches) else len(html)
            sections.append((label, html[section_start:section_end]))
    for row_match in re.finditer(r"<tr>\s*<td[^>]*>(.*?)</td>\s*<td[^>]*>(.*?)</td>\s*</tr>", html, flags=re.IGNORECASE | re.DOTALL):
        sections.append((clean_html_text(row_match.group(1)), row_match.group(2)))
    return sections


def cvf_award_items(section_html: str) -> List[dict]:
    items: List[dict] = []
    card_matches = list(
        re.finditer(
            r'<div\b[^>]+class="[^"]*\b(?:virtual-card|event-card)\b[^"]*"[^>]*>',
            section_html,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )
    if card_matches:
        for index, card in enumerate(card_matches):
            block_start = card.start()
            block_end = card_matches[index + 1].start() if index + 1 < len(card_matches) else len(section_html)
            block_html = section_html[block_start:block_end]
            item = cvf_item_from_html(block_html)
            if item.get("title"):
                items.append(item)
        return items
    for item_html in re.findall(r"<li\b[^>]*>(.*?)</li>", section_html, flags=re.IGNORECASE | re.DOTALL):
        item = cvf_item_from_html(item_html)
        if item.get("title"):
            items.append(item)
    if items:
        return items
    paragraphs = [
        item
        for item in re.findall(r"<p\b[^>]*>(.*?)</p>", section_html, flags=re.IGNORECASE | re.DOTALL)
        if clean_html_text(item)
    ]
    for paragraph in paragraphs:
        item = cvf_item_from_html(paragraph)
        if item.get("title"):
            items.append(item)
    return items


def cvf_item_from_html(item_html: str) -> dict:
    href = ""
    title = ""
    title, href = title_href_from_html(item_html)
    strong_match = re.search(r"<strong[^>]*>(.*?)</strong>", item_html, flags=re.IGNORECASE | re.DOTALL)
    if strong_match:
        title = clean_html_text(strong_match.group(1))
    if not title:
        title = cvf_title_from_plain_text(clean_html_text(item_html))
    authors = ""
    authors = authors_from_common_html(item_html)
    type_match = re.search(r'<div[^>]+class="[^"]*type_display_name_virtual_card[^"]*"[^>]*>(.*?)</div>', item_html, flags=re.IGNORECASE | re.DOTALL)
    if not type_match:
        type_match = re.search(r'<span[^>]+class="[^"]*event-type-badge[^"]*"[^>]*>(.*?)</span>', item_html, flags=re.IGNORECASE | re.DOTALL)
    abstract = ""
    abstract_match = re.search(r"<summary>\s*Abstract\s*</summary>\s*<div[^>]*>(.*?)</div>", item_html, flags=re.IGNORECASE | re.DOTALL)
    if not abstract_match:
        abstract_match = re.search(r'<div[^>]+class="[^"]*event-abstract[^"]*"[^>]*>(.*?)</div>\s*</div>', item_html, flags=re.IGNORECASE | re.DOTALL)
    if abstract_match:
        abstract = clean_html_text(abstract_match.group(1))
    return {
        "title": title,
        "href": href,
        "authors": authors,
        "type": clean_html_text(type_match.group(1)) if type_match else "",
        "body": abstract,
    }


def title_href_from_html(item_html: str) -> tuple[str, str]:
    link_match = re.search(r'<a\b[^>]+href="([^"]+)"[^>]*>(.*?)</a>', item_html, flags=re.IGNORECASE | re.DOTALL)
    if link_match:
        return clean_html_text(link_match.group(2)), unescape(str(link_match.group(1)).strip())
    strong_match = re.search(r"<strong[^>]*>(.*?)</strong>", item_html, flags=re.IGNORECASE | re.DOTALL)
    if strong_match:
        return clean_html_text(strong_match.group(1)), ""
    return "", ""


def authors_from_common_html(item_html: str) -> str:
    for pattern in (
        r'<div[^>]+class="[^"]*author-str[^"]*"[^>]*>(.*?)</div>',
        r'<div[^>]+class="[^"]*event-speakers[^"]*"[^>]*>(.*?)</div>',
        r'<div[^>]+class="[^"]*authors?[^"]*"[^>]*>(.*?)</div>',
        r"<i[^>]*>(.*?)</i>",
        r"<em[^>]*>(.*?)</em>",
    ):
        match = re.search(pattern, item_html, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return clean_html_text(match.group(1))
    return ""


def cvf_is_paper_href(href: str) -> bool:
    value = str(href or "").strip().lower()
    if not value:
        return False
    if "openaccess.thecvf.com" in value and ("_paper.pdf" in value or "/papers/" in value):
        return True
    if "/virtual/" in value and ("/oral/" in value or "/paper/" in value):
        return True
    return False


def cvf_is_oral_href(href: str) -> bool:
    value = str(href or "").strip().lower()
    return "/virtual/" in value and "/oral/" in value and "/poster/" not in value


def is_official_oral_record(*, href: str, record_type: str = "", body: str = "", source_url: str = "") -> bool:
    href_text = str(href or "").strip().lower()
    if href_text:
        return cvf_is_oral_href(href_text)
    page_text = str(source_url or "").strip().lower()
    if "/oral" not in page_text:
        return False
    context = " ".join([str(record_type or ""), str(body or "")]).strip().lower()
    if cvf_is_excluded_award_context(context):
        return False
    return bool(re.search(r"\boral\b", context))


def cvf_title_from_plain_text(text: str) -> str:
    value = re.sub(r"^(best|honorable|honourable|award|winner|paper|student)\b[:\s-]*", "", str(text or "").strip(), flags=re.IGNORECASE)
    for marker in [" Authors:", " by ", " — ", " - "]:
        if marker in value:
            value = value.split(marker, 1)[0]
    return value.strip(" :;-")


def cvf_award_key(label: str) -> str:
    value = str(label or "").strip().lower()
    if not value or cvf_is_excluded_award_context(value):
        return ""
    if value in {"best papers", "paper awards", "awards"}:
        return ""
    if "longuet" in value or "young researcher" in value or "huang memorial" in value or "pami" in value or "tcpami" in value:
        return ""
    if "honorable mention" in value or "honourable mention" in value:
        return "outstanding_paper"
    if "marr prize" in value or "best paper" in value:
        return "best_paper"
    if "award candidate" in value and "paper" in value:
        return "outstanding_paper"
    return ""


def cvf_is_excluded_award_context(value: str) -> bool:
    raw = str(value or "").strip().lower()
    text = raw.replace("-", "_")
    excluded = [
        "poster",
        "demo",
        "workshop",
        "doctoral",
        "challenge",
        "tutorial",
        "industry",
        "reviewer",
        "area chair",
        "student paper",
        "student_paper",
        "best student",
        "best_student",
        "longuet",
        "young researcher",
        "huang memorial",
        "pami",
        "tcpami",
    ]
    return any(item in text for item in excluded)


def dedupe_cvf_award_records(records: List[dict]) -> List[dict]:
    by_key: Dict[tuple[str, str], dict] = {}
    for record in records:
        key = (
            str(record.get("award", "")).strip().lower(),
            re.sub(r"\s+", " ", str(record.get("title", "")).strip().lower()),
        )
        existing = by_key.get(key)
        if existing is None or cvf_record_preference(record) > cvf_record_preference(existing):
            by_key[key] = record
    return list(by_key.values())


def cvf_record_preference(record: dict) -> int:
    ref = str(record.get("external_ref", "")).lower()
    if "/virtual/" in ref and "/poster/" not in ref:
        return 3
    if ref:
        return 2
    return 1


def parse_acl_awards_html(html: str, *, year: int, source_url: str = "", venue: str = "ACL") -> List[dict]:
    records: List[dict] = []
    venue_label = str(venue or "ACL").strip().upper()
    headings = list(
        re.finditer(
            r"<h([2-4])[^>]*>(.*?)</h\1>",
            html,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )
    heading_stack: List[tuple[int, str]] = []
    for index, heading in enumerate(headings):
        level = int(heading.group(1))
        label = clean_html_text(heading.group(2))
        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()
        context_label = " ".join([item_label for _, item_label in heading_stack] + [label])
        award = acl_award_key(context_label)
        heading_stack.append((level, label))
        if not award:
            continue
        section_start = heading.end()
        section_end = headings[index + 1].start() if index + 1 < len(headings) else len(html)
        section_html = html[section_start:section_end]
        for item_html in re.findall(r"<li\b[^>]*>(.*?)</li>", section_html, flags=re.IGNORECASE | re.DOTALL):
            title = acl_title_from_award_item(item_html)
            if not title:
                continue
            authors = acl_authors_from_award_item(item_html)
            href = acl_href_from_award_item(item_html)
            abstract = f"Authors: {authors}".strip() if authors else ""
            records.append(
                {
                    "title": title,
                    "abstract": abstract,
                    "venue": f"{venue_label} {int(year)} {label}",
                    "year": int(year),
                    "publication_date": "",
                    "external_ref": acl_external_ref(source_url, href),
                    "citation_count": 0,
                    "influential_citation_count": 0,
                    "award": award,
                }
            )
    return dedupe_acl_award_records(records)


def acl_award_key(label: str) -> str:
    value = str(label or "").strip().lower()
    if not value:
        return ""
    excluded_tracks = [
        "poster",
        "demo",
        "student",
        "workshop",
        "industry",
        "sac",
        "area chair",
        "video",
    ]
    if any(track in value for track in excluded_tracks):
        return ""
    if "honorable mention" in value or "honourable mention" in value:
        return "outstanding_paper"
    if "outstanding" in value and "paper" in value:
        return "outstanding_paper"
    if "tacl best paper" in value:
        return "best_paper"
    if "best" in value and "paper" in value:
        return "best_paper"
    return ""


def acl_title_from_award_item(item_html: str) -> str:
    strong_match = re.search(r"<strong[^>]*>(.*?)</strong>", item_html, flags=re.IGNORECASE | re.DOTALL)
    if strong_match:
        return clean_html_text(strong_match.group(1))
    link_match = re.search(r"<a\b[^>]*>(.*?)</a>", item_html, flags=re.IGNORECASE | re.DOTALL)
    if link_match:
        return clean_html_text(link_match.group(1))
    without_authors = re.sub(r"<em\b[^>]*>.*?</em>", "", item_html, flags=re.IGNORECASE | re.DOTALL)
    without_authors = re.split(r"<br\s*/?>", without_authors, maxsplit=1, flags=re.IGNORECASE)[0]
    text = clean_html_text(without_authors)
    if text:
        return text.strip()
    text = clean_html_text(item_html)
    if " by " in text:
        text = text.split(" by ", 1)[0]
    if " Authors:" in text:
        text = text.split(" Authors:", 1)[0]
    return text.strip()


def acl_authors_from_award_item(item_html: str) -> str:
    em_match = re.search(r"<em[^>]*>(.*?)</em>", item_html, flags=re.IGNORECASE | re.DOTALL)
    if em_match:
        return clean_html_text(em_match.group(1))
    text = clean_html_text(item_html)
    if " by " in text:
        return text.split(" by ", 1)[1].strip()
    if " Authors:" in text:
        return text.split(" Authors:", 1)[1].strip()
    return ""


def acl_href_from_award_item(item_html: str) -> str:
    link_match = re.search(r'<a\b[^>]+href="([^"]+)"', item_html, flags=re.IGNORECASE | re.DOTALL)
    return unescape(str(link_match.group(1)).strip()) if link_match else ""


def acl_external_ref(source_url: str, href: str) -> str:
    if not href:
        return source_url
    return urllib.parse.urljoin(source_url or "https://www.aclweb.org/", href)


def dedupe_acl_award_records(records: List[dict]) -> List[dict]:
    by_key: Dict[tuple[str, str], dict] = {}
    for record in records:
        key = (
            str(record.get("award", "")).strip().lower(),
            re.sub(r"\s+", " ", str(record.get("title", "")).strip().lower()),
        )
        if key not in by_key:
            by_key[key] = record
    return list(by_key.values())


def parse_icml_awards_html(html: str, *, year: int) -> List[dict]:
    records: List[dict] = []
    for row_match in re.finditer(r"<tr\b[^>]*>\s*<td\b[^>]*>(.*?)</td>\s*<td\b[^>]*>(.*?)</td>\s*</tr>", html, flags=re.IGNORECASE | re.DOTALL):
        label = clean_html_text(row_match.group(1))
        award = icml_award_key(label)
        if not award:
            continue
        card_html = row_match.group(2)
        title, href = title_href_from_html(card_html)
        if not title:
            continue
        external_ref = f"https://icml.cc{href}" if href.startswith("/") else href
        abstract = ""
        abstract_match = re.search(r"<summary>\s*Abstract\s*</summary>\s*<div[^>]*>(.*?)</div>", card_html, flags=re.IGNORECASE | re.DOTALL)
        if abstract_match:
            abstract = clean_html_text(abstract_match.group(1))
        authors = ""
        authors_match = re.search(r'<div[^>]+class="[^"]*author-str[^"]*"[^>]*>(.*?)</div>', card_html, flags=re.IGNORECASE | re.DOTALL)
        if authors_match:
            authors = clean_html_text(authors_match.group(1))
        if authors:
            abstract = f"Authors: {authors}\n\n{abstract}".strip()
        if title:
            records.append(
                {
                    "title": title,
                    "abstract": abstract,
                    "venue": f"ICML {int(year)} {label}",
                    "year": int(year),
                    "publication_date": "",
                    "external_ref": external_ref,
                    "citation_count": 0,
                    "influential_citation_count": 0,
                    "award": award,
                }
            )
    return dedupe_icml_award_records(records)


def parse_icml_oral_html(html: str, *, year: int, source_url: str = "") -> List[dict]:
    records: List[dict] = []
    main_html = cvf_main_content(html)
    for item in cvf_award_items(main_html):
        title = str(item.get("title", "")).strip()
        if not title:
            continue
        href = str(item.get("href", "")).strip()
        context = " ".join([title, href, str(item.get("type", "")), str(item.get("body", "")), source_url])
        if not is_official_oral_record(href=href, record_type=str(item.get("type", "")), body=str(item.get("body", "")), source_url=source_url):
            continue
        authors = str(item.get("authors", "")).strip()
        body = str(item.get("body", "")).strip()
        abstract = f"Authors: {authors}".strip() if authors else ""
        if body and body.lower() != title.lower() and body not in abstract:
            abstract = f"{abstract}\n\n{body}".strip()
        records.append(
            {
                "title": title,
                "abstract": abstract,
                "venue": f"ICML {int(year)} Oral",
                "year": int(year),
                "publication_date": "",
                "external_ref": urllib.parse.urljoin(source_url, href) if href else source_url,
                "citation_count": 0,
                "influential_citation_count": 0,
                "award": "oral",
            }
        )
    return dedupe_icml_award_records(records)


def icml_award_key(label: str) -> str:
    value = str(label or "").strip().lower()
    if not value:
        return ""
    if "test of time" in value:
        return "test_of_time"
    if "best paper" in value:
        return "best_paper"
    if "outstanding" in value and "paper" in value:
        return "outstanding_paper"
    return ""


def dedupe_icml_award_records(records: List[dict]) -> List[dict]:
    by_key: Dict[tuple[str, str], dict] = {}
    for record in records:
        key = (
            str(record.get("award", "")).strip().lower(),
            re.sub(r"\s+", " ", str(record.get("title", "")).strip().lower()),
        )
        existing = by_key.get(key)
        if existing is None or icml_record_preference(record) > icml_record_preference(existing):
            by_key[key] = record
    return list(by_key.values())


def icml_record_preference(record: dict) -> int:
    ref = str(record.get("external_ref", "")).lower()
    venue = str(record.get("venue", "")).lower()
    if "/oral/" in ref or " oral" in venue:
        return 4
    if "/talk/" in ref or "/test-of-time/" in ref:
        return 3
    if "/poster/" in ref or " poster" in venue:
        return 1
    return 2


def clean_html_text(value: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", str(value or ""), flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    text = text.replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def filter_records_by_query_text(records: List[dict], query: str, fallback_limit: int = 0) -> List[dict]:
    tokens = normalized_query_tokens(query)
    if not tokens:
        return records
    scored = []
    for record in records:
        text = f"{record.get('title', '')} {record.get('abstract', '')}".lower()
        score = sum(1 for token in tokens if token in text)
        if score:
            scored.append((score, record))
    scored.sort(key=lambda item: item[0], reverse=True)
    if scored:
        return [record for _, record in scored]
    return records[: max(1, fallback_limit)] if fallback_limit else []


def normalized_query_tokens(query: str) -> List[str]:
    return [
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", str(query or "").lower())
        if token not in {"the", "and", "for", "with", "from", "paper", "papers", "research", "study"}
    ]


def normalize_openalex_work(item: Dict[str, Any], preferred_source_ids: Iterable[str] = ()) -> Dict[str, Any]:
    source = select_openalex_source(item, preferred_source_ids)
    return {
        "title": item.get("display_name", "").strip(),
        "abstract": decode_openalex_abstract(item.get("abstract_inverted_index") or {}),
        "venue": source.get("display_name", "").strip(),
        "year": int(item.get("publication_year") or 0),
        "publication_date": str(item.get("publication_date") or "").strip(),
        "external_ref": best_openalex_full_text_ref(item),
        "citation_count": int(item.get("cited_by_count") or 0),
        "influential_citation_count": 0,
        "award": "",
    }


def best_openalex_full_text_ref(item: Dict[str, Any]) -> str:
    candidates: List[str] = []
    for location in [item.get("primary_location") or {}, *(item.get("locations") or [])]:
        if not isinstance(location, dict):
            continue
        for key in ("pdf_url", "landing_page_url"):
            value = str(location.get(key) or "").strip()
            if value:
                candidates.append(value)
    ids = item.get("ids") or {}
    if isinstance(ids, dict):
        for key in ("doi", "arxiv", "openalex"):
            value = str(ids.get(key) or "").strip()
            if value:
                candidates.append(value)
    openalex_id = str(item.get("id") or "").strip()
    if openalex_id:
        candidates.append(openalex_id)
    for value in candidates:
        if value.startswith("http://") or value.startswith("https://"):
            return value
    for value in candidates:
        if value:
            return value
    return ""


def select_openalex_source(item: Dict[str, Any], preferred_source_ids: Iterable[str] = ()) -> Dict[str, Any]:
    preferred = {
        openalex_short_source_id(source_id).lower()
        for source_id in preferred_source_ids
        if str(source_id).strip()
    }
    locations = item.get("locations") or []
    if isinstance(locations, list):
        for location in locations:
            if not isinstance(location, dict):
                continue
            source = location.get("source") or {}
            source_id = openalex_short_source_id(str(source.get("id") or "")).lower()
            if preferred and source_id in preferred:
                return source
        for location in locations:
            if not isinstance(location, dict):
                continue
            source = location.get("source") or {}
            if str(source.get("type") or "").strip().lower() == "conference":
                return source
    primary_location = item.get("primary_location") or {}
    return primary_location.get("source") or {}


def openalex_work_matches_target_year(
    item: Dict[str, Any],
    target_year: int,
    preferred_source_ids: Iterable[str] = (),
) -> bool:
    if int(item.get("publication_year") or 0) != int(target_year):
        return False
    evidence_years = openalex_location_year_evidence(item, preferred_source_ids)
    if not evidence_years:
        return True
    return int(target_year) in evidence_years


def openalex_location_year_evidence(
    item: Dict[str, Any],
    preferred_source_ids: Iterable[str] = (),
) -> set[int]:
    preferred = {
        openalex_short_source_id(source_id).lower()
        for source_id in preferred_source_ids
        if str(source_id).strip()
    }
    years: set[int] = set()
    locations = item.get("locations") or []
    if not isinstance(locations, list):
        return years
    for location in locations:
        if not isinstance(location, dict):
            continue
        source = location.get("source") or {}
        source_id = openalex_short_source_id(str(source.get("id") or "")).lower()
        if preferred and source_id not in preferred:
            continue
        for key in ("landing_page_url", "pdf_url"):
            years.update(extract_conference_years_from_url(str(location.get(key) or "")))
    return years


def extract_conference_years_from_url(url: str) -> set[int]:
    value = str(url or "")
    years: set[int] = set()
    for pattern in (
        r"proceedings\.mlr\.press/v\d+/[a-z][a-z-]*(\d{2})[a-z]?/",
        r"proceedings\.mlr\.press/v\d+/[a-z][a-z-]*(\d{2})[a-z]?\.html",
        r"proceedings\.neurips\.cc/paper/(\d{4})/",
        r"papers\.nips\.cc/paper/(\d{4})/",
        r"nips\d{2}-",
    ):
        for match in re.finditer(pattern, value, flags=re.IGNORECASE):
            token = match.group(1) if match.groups() else match.group(0)[4:6]
            years.add(expand_two_digit_year(token))
    return years


def expand_two_digit_year(value: str) -> int:
    number = int(value)
    if number >= 1900:
        return number
    return 2000 + number if number <= 40 else 1900 + number


def normalize_semantic_scholar_paper(item: Dict[str, Any]) -> Dict[str, Any]:
    external_ids = item.get("externalIds") or {}
    external_ref = external_ids.get("DOI") or external_ids.get("ArXiv") or item.get("url") or ""
    return {
        "title": str(item.get("title") or "").strip(),
        "abstract": str(item.get("abstract") or "").strip(),
        "venue": str(item.get("venue") or "").strip(),
        "year": int(item.get("year") or 0),
        "publication_date": str(item.get("publicationDate") or "").strip(),
        "external_ref": str(external_ref).strip(),
        "citation_count": int(item.get("citationCount") or 0),
        "influential_citation_count": int(item.get("influentialCitationCount") or 0),
        "award": "",
    }


def decode_openalex_abstract(index: Dict[str, Iterable[int]]) -> str:
    if not index:
        return ""
    positions: Dict[int, str] = {}
    for token, raw_positions in index.items():
        for pos in raw_positions:
            positions[int(pos)] = token
    return " ".join(positions[pos] for pos in sorted(positions))


def _get_json(url: str, *, params: Dict[str, str], api_key: str = "", email: str = "") -> Dict[str, Any]:
    query = dict(params)
    if api_key and "openalex.org" in url:
        query["api_key"] = api_key
    if email and "openalex.org" in url:
        query["mailto"] = email
    full_url = f"{url}?{urllib.parse.urlencode(query)}"
    headers = {"Accept": "application/json"}
    if api_key and "semanticscholar.org" in url:
        headers["x-api-key"] = api_key
    request = urllib.request.Request(full_url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ConnectorError(f"Remote request failed with HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise ConnectorError(f"Remote request failed: {exc.reason}") from exc
    return json.loads(body)


def _get_text(url: str) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "text/html,application/xhtml+xml",
            "User-Agent": "ResearchAlpha/0.1 (+https://localhost; scholarly metadata harvester)",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ConnectorError(f"Remote request failed with HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise ConnectorError(f"Remote request failed: {exc.reason}") from exc
