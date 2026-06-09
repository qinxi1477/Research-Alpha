from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from io import BytesIO
from html import unescape
from pathlib import Path
from typing import Any, Dict, List


SECTION_ALIASES = {
    "introduction": ["introduction", "1 introduction", "background"],
    "related_work": ["related work", "prior work", "literature review"],
    "method": ["method", "methods", "approach", "model", "algorithm", "framework"],
    "limitations": ["limitation", "limitations", "discussion", "future work", "failure", "shortcoming", "shortcomings"],
    "experiments": ["experiment", "experiments", "experimental setup", "evaluation", "evaluate", "benchmark", "benchmarks", "empirical"],
    "ablation": ["ablation", "ablation study", "analysis"],
    "rebuttal_or_review_signal": ["rebuttal", "review", "reviewer", "oral", "spotlight"],
    "conclusion": ["conclusion", "conclusions"],
}

FULL_TEXT_USER_AGENT = "ResearchAlpha/0.1 (+https://localhost; full-text evidence extractor)"
PDF_STRUCTURAL_TEXT_RE = re.compile(
    r"(?:^|\s)(?:endobj|endstream|xref|trailer|startxref|objstm|flatedecode)(?:\s|$)|"
    r"/(?:Type|ObjStm|Filter|FlateDecode|Length|First|Catalog|Pages|Font)\b|"
    r">>\s*stream\b",
    re.IGNORECASE,
)
PDF_ARTIFACT_TEXT_RE = re.compile(
    r"\b(?:cite|figure|subsection|appendix|table|bibitem)\.[A-Za-z0-9_:-]+|"
    r"\b(?:matplotlib|pdftex|tex live|hyperref|latex|backend)\b|"
    r"\b[A-Za-z]+(?:\d{4})[A-Za-z0-9_:-]*\b",
    re.IGNORECASE,
)


class FullTextError(RuntimeError):
    """Raised when full-text evidence cannot be fetched or parsed."""


def pdf_parser_status() -> Dict[str, object]:
    try:
        import pypdf  # type: ignore
    except Exception:
        return {
            "available": False,
            "parser": "literal_pdf_fallback",
            "quality": "limited",
            "message": "Install project dependencies with uv or pip so pypdf can parse full PDFs.",
        }
    return {
        "available": True,
        "parser": "pypdf",
        "version": str(getattr(pypdf, "__version__", "") or ""),
        "quality": "pdf_text",
    }


def normalize_full_text_url(external_ref: str) -> str:
    value = str(external_ref or "").strip()
    if not value:
        return ""
    if value.startswith("file://"):
        path = urllib.parse.unquote(urllib.parse.urlparse(value).path)
        return str(Path(path).expanduser())
    local_path = Path(value).expanduser()
    if local_path.exists():
        return str(local_path)
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme not in {"http", "https"}:
        return ""
    host = (parsed.hostname or "").lower()
    if host.endswith("arxiv.org"):
        match = re.search(r"/(?:abs|html|pdf)/([^?#]+)", parsed.path)
        if match:
            arxiv_id = re.sub(r"\.pdf$", "", urllib.parse.unquote(match.group(1)), flags=re.IGNORECASE)
            return f"https://arxiv.org/html/{arxiv_id}"
    if host.endswith("openreview.net") and parsed.path.startswith("/pdf"):
        return value
    if value.lower().endswith(".pdf"):
        return value
    return value


def arxiv_pdf_fallback_url(url: str) -> str:
    arxiv_id = arxiv_id_from_full_text_url(url)
    if not arxiv_id:
        return ""
    return f"https://arxiv.org/pdf/{arxiv_id}.pdf"


def arxiv_html_fallback_urls(url: str) -> List[str]:
    arxiv_id = arxiv_id_from_full_text_url(url)
    if not arxiv_id:
        return []
    return [
        f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}",
        f"https://arxiv.org/pdf/{arxiv_id}.pdf",
    ]


def arxiv_id_from_full_text_url(url: str) -> str:
    parsed = urllib.parse.urlparse(str(url or "").strip())
    if parsed.scheme not in {"http", "https"} or not (parsed.hostname or "").lower().endswith("arxiv.org"):
        return ""
    match = re.search(r"/(?:abs|html|pdf)/([^?#]+)", parsed.path)
    if not match:
        return ""
    arxiv_id = urllib.parse.unquote(match.group(1))
    return re.sub(r"\.pdf$", "", arxiv_id, flags=re.IGNORECASE).strip()


def fetch_full_text(url: str, *, timeout: int = 60, max_bytes: int = 5_000_000, follow_links: bool = True) -> tuple[str, str]:
    normalized = normalize_full_text_url(url)
    if not normalized:
        raise FullTextError("No supported full-text URL was available.")
    local_path = Path(normalized).expanduser()
    if local_path.exists():
        raw = local_path.read_bytes()[: max(1, int(max_bytes))]
        return decode_full_text(raw, content_type=local_path.suffix.lower()), str(local_path)
    try:
        raw, content_type, source_url = fetch_url_bytes(
            normalized,
            timeout=max(1, int(timeout)),
            max_bytes=max(1, int(max_bytes)),
        )
    except urllib.error.HTTPError as exc:
        fallbacks = arxiv_html_fallback_urls(normalized)
        if fallbacks:
            return fetch_first_available_full_text(
                fallbacks,
                timeout=max(1, int(timeout)),
                max_bytes=max(1, int(max_bytes)),
                original_error=exc,
            )
        detail = exc.read(1000).decode("utf-8", errors="replace")
        raise FullTextError(f"Full-text request failed with HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise FullTextError(f"Full-text request failed: {exc.reason}") from exc
    text = decode_full_text(raw, content_type=content_type)
    if follow_links and should_follow_html_paper_link(text, content_type=content_type):
        target = discover_paper_link(raw.decode("utf-8", errors="replace"), source_url)
        if target and target != source_url:
            try:
                linked_raw, linked_type, linked_url = fetch_url_bytes(
                    target,
                    timeout=max(1, int(timeout)),
                    max_bytes=max(1, int(max_bytes)),
                )
                linked_text = decode_full_text(linked_raw, content_type=linked_type)
                if linked_text.strip():
                    return linked_text, linked_url
            except (urllib.error.HTTPError, urllib.error.URLError):
                pass
    return text, source_url


def fetch_full_text_with_metadata(
    url: str,
    *,
    title: str = "",
    venue: str = "",
    year: int | str = "",
    timeout: int = 60,
    max_bytes: int = 5_000_000,
) -> tuple[str, str]:
    text, source_url = fetch_full_text(url, timeout=timeout, max_bytes=max_bytes)
    payload = extract_section_evidence(text)
    if is_usable_full_text_payload(payload):
        return text, source_url
    resolved = resolve_full_text_url_from_metadata(
        url,
        title=title,
        venue=venue,
        year=year,
        timeout=timeout,
        max_bytes=max_bytes,
    )
    if not resolved or resolved == source_url or resolved == normalize_full_text_url(url):
        return text, source_url
    resolved_text, resolved_source = fetch_full_text(
        resolved,
        timeout=timeout,
        max_bytes=max_bytes,
        follow_links=False,
    )
    resolved_payload = extract_section_evidence(resolved_text)
    if is_usable_full_text_payload(resolved_payload):
        return resolved_text, resolved_source
    pdf_candidate = discover_pdf_from_paper_page(
        resolved,
        timeout=timeout,
        max_bytes=max_bytes,
    )
    if pdf_candidate and pdf_candidate != resolved_source:
        try:
            pdf_text, pdf_source = fetch_full_text(
                pdf_candidate,
                timeout=timeout,
                max_bytes=max_bytes,
            )
        except FullTextError:
            pdf_text = ""
            pdf_source = ""
        if pdf_text:
            pdf_payload = extract_section_evidence(pdf_text)
            if is_usable_full_text_payload(pdf_payload):
                return pdf_text, pdf_source
    if (
        has_any_section_payload(resolved_payload)
        or len(resolved_text or "") > len(text or "") + 500
        or is_specific_pmlr_paper_page(resolved_source)
    ):
        return resolved_text, resolved_source
    return text, source_url


def is_usable_full_text_payload(payload: Dict[str, object]) -> bool:
    if not isinstance(payload, dict):
        return False
    quality = str(payload.get("quality", "")).strip()
    if quality in {"paper_like_sections", "paper_like_full_text"}:
        return True
    if quality == "metadata_page_sections":
        return False
    available = payload.get("available_sections", [])
    section_count = len(available) if isinstance(available, list) else 0
    return section_count >= 4 and int(payload.get("full_text_chars", 0) or 0) >= 8000


def has_any_section_payload(payload: Dict[str, object]) -> bool:
    if not isinstance(payload, dict):
        return False
    available = payload.get("available_sections", [])
    return bool(available) if isinstance(available, list) else False


def is_specific_pmlr_paper_page(url: str) -> bool:
    parsed = urllib.parse.urlparse(str(url or "").strip())
    return (parsed.hostname or "").lower() == "proceedings.mlr.press" and bool(re.search(r"^/v\d+/.+\.html$", parsed.path.lower()))


def resolve_full_text_url_from_metadata(
    source_url: str,
    *,
    title: str = "",
    venue: str = "",
    year: int | str = "",
    timeout: int = 60,
    max_bytes: int = 5_000_000,
) -> str:
    title_key = canonical_title(title)
    if not title_key:
        return ""
    candidates = metadata_resolution_pages(source_url, venue=venue, year=year)
    for page_url in candidates:
        try:
            raw, content_type, fetched_url = fetch_url_bytes(
                page_url,
                timeout=max(1, int(timeout)),
                max_bytes=max(1, int(max_bytes)),
            )
        except (urllib.error.HTTPError, urllib.error.URLError):
            continue
        if "html" not in str(content_type or "").lower() and b"<html" not in raw[:500].lower():
            continue
        resolved = discover_paper_link_by_title(
            raw.decode("utf-8", errors="replace"),
            fetched_url,
            title=title,
        )
        if resolved:
            return resolved
    return ""


def metadata_resolution_pages(source_url: str, *, venue: str = "", year: int | str = "") -> List[str]:
    pages: List[str] = []
    parsed = urllib.parse.urlparse(str(source_url or "").strip())
    host = (parsed.hostname or "").lower()
    path = parsed.path.lower()
    venue_text = str(venue or "").lower()
    year_text = str(year or "").strip()
    if host.endswith("icml.cc") and "/virtual/2024/" in path:
        pages.append("https://proceedings.mlr.press/v235/")
    if host.endswith("icml.cc") and "/virtual/2025/" in path:
        pages.append("https://proceedings.mlr.press/v267/")
    if host.endswith("icml.cc") and "/virtual/2023/" in path:
        pages.append("https://proceedings.mlr.press/v202/")
    if host.endswith("icml.cc") and "/virtual/2022/" in path:
        pages.append("https://proceedings.mlr.press/v162/")
    if "icml" in venue_text:
        if "2025" in venue_text or year_text == "2025":
            pages.append("https://proceedings.mlr.press/v267/")
        elif "2024" in venue_text or year_text == "2024":
            pages.append("https://proceedings.mlr.press/v235/")
        elif "2023" in venue_text or year_text == "2023":
            pages.append("https://proceedings.mlr.press/v202/")
        elif "2022" in venue_text or year_text == "2022":
            pages.append("https://proceedings.mlr.press/v162/")
    venue_key = venue_text.replace("-", "").replace(" ", "")
    if year_text:
        if "neurips" in venue_key or "nips" in venue_key or host.endswith("neurips.cc"):
            pages.append(f"https://papers.nips.cc/paper_files/paper/{year_text}")
        if "cvpr" in venue_key or host.endswith("cvpr.thecvf.com"):
            pages.append(f"https://openaccess.thecvf.com/CVPR{year_text}?day=all")
        if "iccv" in venue_key or host.endswith("iccv.thecvf.com"):
            pages.append(f"https://openaccess.thecvf.com/ICCV{year_text}?day=all")
    deduped: List[str] = []
    for page in pages:
        if page not in deduped:
            deduped.append(page)
    return deduped


def discover_paper_link_by_title(html: str, base_url: str, *, title: str) -> str:
    title_key = canonical_title(title)
    if not title_key:
        return ""
    for match in re.finditer(r"""(?is)<div\b[^>]*class\s*=\s*["'][^"']*\bpaper\b[^"']*["'][^>]*>(.*?)</div>""", html):
        block = match.group(1)
        block_title = extract_pmlr_block_title(block)
        if canonical_title(block_title) != title_key:
            continue
        link = preferred_link_from_html_block(block, base_url)
        if link:
            return link
    exact_anchor = preferred_exact_title_anchor(html, base_url, title_key=title_key)
    if exact_anchor:
        return exact_anchor
    return ""


def extract_pmlr_block_title(block: str) -> str:
    match = re.search(r"""(?is)<p\b[^>]*class\s*=\s*["'][^"']*\btitle\b[^"']*["'][^>]*>(.*?)</p>""", block)
    if not match:
        return ""
    return normalize_text(re.sub(r"(?s)<[^>]+>", " ", unescape(match.group(1))))


def preferred_link_from_html_block(block: str, base_url: str) -> str:
    scored: List[tuple[int, str]] = []
    for match in re.finditer(r"""(?is)<a\b[^>]*\bhref\s*=\s*["']([^"']+)["'][^>]*>(.*?)</a>""", block):
        href = unescape(match.group(1).strip())
        label = normalize_text(re.sub(r"(?s)<[^>]+>", " ", unescape(match.group(2))))
        absolute = urllib.parse.urljoin(base_url, href)
        score = paper_link_score(absolute, label)
        parsed = urllib.parse.urlparse(absolute)
        if (parsed.hostname or "").lower() == "proceedings.mlr.press" and parsed.path.lower().endswith(".html"):
            score += 8
        if score > 0:
            scored.append((score, absolute))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1] if scored else ""


def preferred_exact_title_anchor(html: str, base_url: str, *, title_key: str) -> str:
    scored: List[tuple[int, str]] = []
    for match in re.finditer(r"""(?is)<a\b[^>]*\bhref\s*=\s*["']([^"']+)["'][^>]*>(.*?)</a>""", html):
        href = unescape(match.group(1).strip())
        label = normalize_text(re.sub(r"(?s)<[^>]+>", " ", unescape(match.group(2))))
        if canonical_title(label) != title_key:
            continue
        absolute = urllib.parse.urljoin(base_url, href)
        score = paper_link_score(absolute, label) + 20
        if score > 0:
            scored.append((score, absolute))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1] if scored else ""


def discover_pdf_from_paper_page(url: str, *, timeout: int = 60, max_bytes: int = 5_000_000) -> str:
    parsed = urllib.parse.urlparse(str(url or "").strip())
    host = (parsed.hostname or "").lower()
    path = parsed.path.lower()
    direct_pdf = direct_pdf_url_for_known_paper_page(str(url or "").strip())
    if direct_pdf:
        return direct_pdf
    supported_html_page = (
        (host == "proceedings.mlr.press" and path.endswith(".html"))
        or (host.endswith("openaccess.thecvf.com") and path.endswith(".html"))
        or (host.endswith("papers.nips.cc") and path.endswith(".html"))
    )
    if not supported_html_page:
        return ""
    try:
        raw, content_type, fetched_url = fetch_url_bytes(
            url,
            timeout=max(1, int(timeout)),
            max_bytes=max(1, int(max_bytes)),
        )
    except (urllib.error.HTTPError, urllib.error.URLError):
        return ""
    if "html" not in str(content_type or "").lower() and b"<html" not in raw[:500].lower():
        return ""
    html = raw.decode("utf-8", errors="replace")
    scored: List[tuple[int, str]] = []
    for match in re.finditer(r"""(?is)<a\b[^>]*\bhref\s*=\s*["']([^"']+)["'][^>]*>(.*?)</a>""", html):
        href = unescape(match.group(1).strip())
        label = normalize_text(re.sub(r"(?s)<[^>]+>", " ", unescape(match.group(2))))
        absolute = urllib.parse.urljoin(fetched_url, href)
        if ".pdf" not in absolute.lower() and "pdf" not in label.lower():
            continue
        score = paper_link_score(absolute, label)
        if score > 0:
            scored.append((score, absolute))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1] if scored else ""


def direct_pdf_url_for_known_paper_page(url: str) -> str:
    parsed = urllib.parse.urlparse(str(url or "").strip())
    host = (parsed.hostname or "").lower()
    path = parsed.path
    if host.endswith("papers.nips.cc") and "-Abstract-" in path and path.endswith(".html"):
        return urllib.parse.urlunparse(
            parsed._replace(path=path.replace("-Abstract-", "-Paper-").replace(".html", ".pdf"), query="", fragment="")
        )
    if host.endswith("openaccess.thecvf.com") and "/html/" in path and path.endswith(".html"):
        return urllib.parse.urlunparse(
            parsed._replace(path=path.replace("/html/", "/papers/").replace(".html", ".pdf"), query="", fragment="")
        )
    return ""


def canonical_title(value: str) -> str:
    text = unescape(str(value or "")).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return normalize_text(text)


def fetch_first_available_full_text(
    urls: List[str],
    *,
    timeout: int,
    max_bytes: int,
    original_error: urllib.error.HTTPError,
) -> tuple[str, str]:
    last_http_error: urllib.error.HTTPError | None = None
    last_url_error: urllib.error.URLError | None = None
    for url in urls:
        try:
            raw, content_type, source_url = fetch_url_bytes(
                url,
                timeout=timeout,
                max_bytes=max_bytes,
            )
        except urllib.error.HTTPError as exc:
            last_http_error = exc
            continue
        except urllib.error.URLError as exc:
            last_url_error = exc
            continue
        text = decode_full_text(raw, content_type=content_type)
        if text.strip():
            return text, source_url
    if last_http_error is not None:
        detail = last_http_error.read(1000).decode("utf-8", errors="replace")
        raise FullTextError(f"Full-text request failed with HTTP {last_http_error.code}: {detail}") from last_http_error
    if last_url_error is not None:
        raise FullTextError(f"Full-text request failed: {last_url_error.reason}") from last_url_error
    detail = original_error.read(1000).decode("utf-8", errors="replace")
    raise FullTextError(f"Full-text request failed with HTTP {original_error.code}: {detail}") from original_error


def should_follow_html_paper_link(text: str, *, content_type: str) -> bool:
    if "pdf" in str(content_type or "").lower():
        return False
    if "proceedings.mlr.press" in text[:1000].lower():
        return False
    return len(text or "") < 6000


def fetch_url_bytes(url: str, *, timeout: int, max_bytes: int) -> tuple[bytes, str, str]:
    last_error: Exception | None = None
    for attempt in range(3):
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "text/html,application/xhtml+xml,application/pdf,text/plain;q=0.8,*/*;q=0.5",
                "User-Agent": FULL_TEXT_USER_AGENT,
            },
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                content_type = response.headers.get("Content-Type", "")
                raw = response.read(max_bytes)
            return raw, content_type, url
        except urllib.error.HTTPError:
            raise
        except urllib.error.URLError as exc:
            last_error = exc
        except OSError as exc:
            last_error = exc
        if attempt < 2:
            time.sleep(0.4 * (2 ** attempt))
    if isinstance(last_error, urllib.error.URLError):
        raise last_error
    raise urllib.error.URLError(last_error or "unknown network error")


def discover_paper_link(html: str, base_url: str) -> str:
    candidates: List[tuple[int, str]] = []
    for match in re.finditer(r"""(?is)<a\b[^>]*\bhref\s*=\s*["']([^"']+)["'][^>]*>(.*?)</a>""", html):
        href = unescape(match.group(1).strip())
        label = normalize_text(re.sub(r"(?s)<[^>]+>", " ", unescape(match.group(2))))
        absolute = urllib.parse.urljoin(base_url, href)
        score = paper_link_score(absolute, label)
        if score > 0:
            candidates.append((score, absolute))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1] if candidates else ""


def paper_link_score(url: str, label: str = "") -> int:
    lowered = f"{url} {label}".lower()
    parsed = urllib.parse.urlparse(url)
    host = (parsed.hostname or "").lower()
    path = parsed.path.lower()
    if host.endswith("icml.cc") and path.endswith("/papers.html"):
        return 0
    score = 0
    if lowered.endswith(".pdf") or "/pdf" in path or "pdf" in label.lower():
        score += 8
    if lowered.endswith(".pdf") or label.strip().lower() in {"pdf", "download pdf"}:
        score += 4
    if "openreview.net/pdf" in lowered:
        score += 10
    if "arxiv.org/pdf" in lowered or "arxiv.org/abs" in lowered:
        score += 9
    if host == "proceedings.mlr.press":
        if re.search(r"^/v\d+/.+\.(html|pdf)$", path):
            score += 6
        else:
            score -= 10
    elif "proceedings.mlr.press" in lowered and ".html" in lowered:
        score += 6
    if host.endswith("openaccess.thecvf.com") and re.search(r"/content/[A-Z]+\d+/(html|papers)/", path):
        score += 8
    if host.endswith("papers.nips.cc") and "/paper_files/paper/" in path:
        score += 8
    if "paper" in lowered or "proceeding" in lowered:
        score += 2
    if host.endswith("slideslive.com") or "youtube" in host or host == "github.com":
        score -= 8
    return max(0, score)


def decode_full_text(raw: bytes, *, content_type: str = "") -> str:
    prefix = raw[:8]
    if b"%PDF" in prefix or "pdf" in content_type.lower():
        return extract_text_from_pdf_bytes(raw)
    text = raw.decode("utf-8", errors="replace")
    if "<html" in text[:500].lower() or "</" in text:
        return html_to_text(text)
    return normalize_text(text)


def html_to_text(html: str) -> str:
    text = re.sub(r"(?is)<(script|style|nav|footer|header|noscript).*?</\1>", " ", html)
    text = re.sub(r"(?i)</?(h[1-6]|p|div|section|article|li|br|blockquote|tr|td|th)[^>]*>", "\n", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return normalize_text(unescape(text))


def extract_text_from_pdf_bytes(raw: bytes) -> str:
    parsed = extract_text_from_pdf_with_pypdf(raw)
    if parsed.strip():
        return parsed
    return extract_text_from_pdf_literal_chunks(raw)


def extract_text_from_pdf_with_pypdf(raw: bytes) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        return ""
    try:
        reader = PdfReader(BytesIO(raw))
        page_texts = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                page_texts.append(text)
    except Exception:
        return ""
    return normalize_text("\n".join(page_texts))


def extract_text_from_pdf_literal_chunks(raw: bytes) -> str:
    chunks: List[str] = []
    # Lightweight fallback: many PDFs contain literal text in parentheses. This
    # is not a replacement for PyMuPDF/GROBID, but it creates a stable hook for
    # tests and for simple publisher PDFs until a heavier parser is configured.
    for match in re.finditer(rb"\(([^()]{8,400})\)", raw):
        chunk = match.group(1).decode("latin-1", errors="ignore")
        chunk = chunk.replace(r"\(", "(").replace(r"\)", ")").replace(r"\n", " ")
        if (
            re.search(r"[A-Za-z]{4,}", chunk)
            and not looks_like_pdf_structural_text(chunk)
            and text_is_readable(chunk, min_alpha_ratio=0.20)
        ):
            chunks.append(chunk)
    text = normalize_text(" ".join(chunks))
    if looks_like_pdf_structural_text(text) or not text_is_readable(text, min_alpha_ratio=0.25):
        return ""
    return text


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def text_is_readable(value: str, *, min_alpha_ratio: float = 0.20) -> bool:
    text = str(value or "")
    if not text:
        return False
    meaningful = [ch for ch in text if not ch.isspace()]
    if not meaningful:
        return False
    printable = [ch for ch in meaningful if ch.isprintable()]
    ascii_alpha = [ch for ch in meaningful if ("a" <= ch.lower() <= "z")]
    ascii_printable = [ch for ch in meaningful if ord(ch) < 128 and ch.isprintable()]
    if len(printable) / len(meaningful) < 0.88:
        return False
    if len(ascii_printable) / len(meaningful) < 0.65:
        return False
    if len(ascii_alpha) / len(meaningful) < float(min_alpha_ratio):
        return False
    return True


def looks_like_pdf_structural_text(value: str) -> bool:
    text = str(value or "")
    if not text:
        return False
    if PDF_STRUCTURAL_TEXT_RE.search(text):
        return True
    structural_tokens = re.findall(r"\b(?:obj|endobj|stream|endstream|FlateDecode|ObjStm|xref|trailer)\b", text, flags=re.IGNORECASE)
    natural_sentence_marks = len(re.findall(r"[.!?]\s+[A-Z]", text))
    if len(structural_tokens) >= 2 and natural_sentence_marks == 0:
        return True
    artifact_tokens = PDF_ARTIFACT_TEXT_RE.findall(text)
    words = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text)
    if len(artifact_tokens) >= 12 and len(artifact_tokens) / max(1, len(words)) > 0.22:
        return True
    return False


def split_sentences(text: str) -> List[str]:
    cleaned = normalize_text(text)
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?。！？])\s+", cleaned)
    return [part.strip() for part in parts if part.strip()]


def extract_section_evidence(text: str, *, max_chars_per_section: int = 1800) -> Dict[str, object]:
    cleaned = normalize_text(text)
    sections = {name: "" for name in SECTION_ALIASES}
    if not cleaned:
        return build_full_text_payload("", sections)
    headings = find_section_headings(cleaned)
    for index, (start, section_name) in enumerate(headings):
        end = headings[index + 1][0] if index + 1 < len(headings) else len(cleaned)
        excerpt = cleaned[start:end].strip()
        if excerpt and not sections.get(section_name):
            sections[section_name] = compact(excerpt, max_chars_per_section)
    if not any(sections.values()):
        sections = extract_keyword_windows(cleaned, max_chars_per_section=max_chars_per_section)
    return build_full_text_payload(cleaned, sections)


def find_section_headings(text: str) -> List[tuple[int, str]]:
    headings: List[tuple[int, str]] = []
    for section_name, aliases in SECTION_ALIASES.items():
        for alias in aliases:
            pattern = rf"(?:^|[.\n]\s*)(?:\d+(?:\.\d+)?\s+)?{re.escape(alias)}\b"
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                headings.append((match.start(), section_name))
    headings.sort(key=lambda item: item[0])
    deduped: List[tuple[int, str]] = []
    for pos, name in headings:
        if deduped and abs(pos - deduped[-1][0]) < 40:
            continue
        deduped.append((pos, name))
    return deduped


def extract_keyword_windows(text: str, *, max_chars_per_section: int) -> Dict[str, str]:
    sections = {name: "" for name in SECTION_ALIASES}
    lowered = text.lower()
    for section_name, aliases in SECTION_ALIASES.items():
        positions = [lowered.find(alias.lower()) for alias in aliases if lowered.find(alias.lower()) >= 0]
        if not positions:
            continue
        start = max(0, min(positions) - 400)
        end = min(len(text), min(positions) + max_chars_per_section)
        sections[section_name] = compact(text[start:end], max_chars_per_section)
    return sections


def build_full_text_payload(text: str, sections: Dict[str, str]) -> Dict[str, object]:
    available = [name for name, value in sections.items() if str(value).strip()]
    quality = infer_full_text_quality(text, available)
    return {
        "source_scope": "full_text_sections",
        "quality": quality,
        "policy": "Extracted section snippets are evidence context for logic-line analysis; they do not by themselves make a paper Gold evidence.",
        "available_sections": available,
        "section_count": len(available),
        "sections": {name: str(value).strip() for name, value in sections.items() if str(value).strip()},
        "full_text_chars": len(text or ""),
        "needs_pdf_parser": False,
    }


def infer_full_text_quality(text: str, available_sections: List[str]) -> str:
    section_set = {str(item).strip() for item in available_sections if str(item).strip()}
    core_sections = {"introduction", "related_work", "method", "experiments", "limitations", "conclusion"}
    has_problem_setup = bool(section_set & {"introduction", "related_work"})
    has_evidence_story = bool(section_set & {"experiments", "limitations", "conclusion"})
    if len(section_set & core_sections) >= 5 and has_problem_setup and has_evidence_story:
        return "paper_like_sections"
    if len(text or "") >= 8000 and len(section_set & core_sections) >= 3 and has_problem_setup:
        return "paper_like_full_text"
    if len(text or "") >= 4000 and len(section_set & core_sections) >= 3 and (has_problem_setup or has_evidence_story):
        return "paper_like_sections"
    if section_set:
        return "metadata_page_sections"
    return "no_sections"


def full_text_json_for_text(text: str) -> str:
    return json.dumps(extract_section_evidence(text), ensure_ascii=False, sort_keys=True)


def sanitize_full_text_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    sections = payload.get("sections")
    if not isinstance(sections, dict):
        sections = {}
    cleaned_sections: Dict[str, str] = {}
    removed_sections: List[str] = []
    for key, value in sections.items():
        name = str(key).strip()
        text = normalize_text(str(value or ""))
        if not name or not text:
            continue
        if looks_like_pdf_structural_text(text) or not text_is_readable(text, min_alpha_ratio=0.18):
            removed_sections.append(name)
            continue
        cleaned_sections[name] = text
    full_text_chars = int(payload.get("full_text_chars", 0) or 0)
    available = [name for name in payload.get("available_sections", []) if str(name).strip()] if isinstance(payload.get("available_sections"), list) else sorted(cleaned_sections)
    available = [str(name).strip() for name in available if str(name).strip() in cleaned_sections]
    quality_text = " ".join(cleaned_sections.values())
    if full_text_chars > len(quality_text):
        quality_text += " " + ("x" * min(full_text_chars - len(quality_text), 20000))
    quality = infer_full_text_quality(quality_text, available)
    sanitized = dict(payload)
    sanitized["sections"] = cleaned_sections
    sanitized["available_sections"] = available
    sanitized["section_count"] = len(available)
    sanitized["quality"] = quality
    if removed_sections:
        sanitized["sanitization"] = {
            "removed_sections": sorted(set(removed_sections)),
            "reason": "unreadable_or_pdf_structural_text",
        }
    return sanitized


def compact(value: object, limit: int) -> str:
    text = normalize_text(str(value or ""))
    if len(text) <= limit:
        return text
    sentences = split_sentences(text)
    kept: List[str] = []
    total = 0
    for sentence in sentences:
        if total + len(sentence) + 1 > limit:
            break
        kept.append(sentence)
        total += len(sentence) + 1
    if kept:
        return " ".join(kept)
    return text[: max(0, limit - 3)].rstrip() + "..."
