from __future__ import annotations

import re
from collections import Counter, defaultdict
from html import escape
from math import log
from typing import Dict, Iterable, List


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "into",
    "is", "it", "its", "of", "on", "or", "that", "the", "their", "this", "to",
    "under", "we", "with", "which", "will", "our", "using", "use", "new", "study",
    "based", "large", "models", "model", "learning", "neural", "language",
}


def analyze_trends(papers: Iterable[dict], frontier_years: int = 5, top_k: int = 10) -> Dict[str, object]:
    rows = []
    for paper in papers:
        row = dict(paper)
        if int(row.get("year", 0) or 0) > 0:
            rows.append(row)
    if not rows:
        return {
            "frontier_years": frontier_years,
            "latest_year": None,
            "paper_count": 0,
            "recent_papers": 0,
            "earlier_papers": 0,
            "top_terms": [],
            "trend_clusters": [],
            "year_counts": {},
        }

    latest_year = max(int(row["year"]) for row in rows)
    recent_cutoff = latest_year - max(1, frontier_years) + 1
    recent_rows = [row for row in rows if int(row["year"]) >= recent_cutoff]
    earlier_rows = [row for row in rows if int(row["year"]) < recent_cutoff]

    recent_terms = Counter()
    earlier_terms = Counter()
    year_counts: Dict[int, int] = defaultdict(int)

    for row in rows:
        year_counts[int(row["year"])] += 1

    for row in recent_rows:
        recent_terms.update(extract_terms(row))
    for row in earlier_rows:
        earlier_terms.update(extract_terms(row))

    scored_terms = []
    recent_denominator = max(1, len(recent_rows))
    earlier_denominator = max(1, len(earlier_rows))
    for term, recent_count in recent_terms.items():
        earlier_count = earlier_terms.get(term, 0)
        growth = (recent_count / recent_denominator) - (earlier_count / earlier_denominator)
        scored_terms.append(
            {
                "term": term,
                "recent_count": recent_count,
                "earlier_count": earlier_count,
                "growth_score": round(growth, 4),
            }
        )

    scored_terms.sort(
        key=lambda item: (
            item["growth_score"],
            item["recent_count"],
            -item["earlier_count"],
            item["term"],
        ),
        reverse=True,
    )

    clusters = build_trend_clusters(
        rows=rows,
        recent_rows=recent_rows,
        earlier_rows=earlier_rows,
        top_terms=scored_terms[: max(1, top_k * 3)],
        limit=max(1, min(top_k, 20)),
    )

    return {
        "frontier_years": frontier_years,
        "latest_year": latest_year,
        "paper_count": len(rows),
        "recent_papers": len(recent_rows),
        "earlier_papers": len(earlier_rows),
        "top_terms": scored_terms[: max(1, top_k)],
        "trend_clusters": clusters,
        "year_counts": {str(year): year_counts[year] for year in sorted(year_counts)},
    }


def extract_terms(paper: dict) -> List[str]:
    text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
    tokens = re.findall(r"[a-z][a-z0-9\-]+", text)
    filtered = [token for token in tokens if token not in STOPWORDS and len(token) >= 4]
    bigrams = [
        f"{filtered[idx]} {filtered[idx + 1]}"
        for idx in range(len(filtered) - 1)
        if filtered[idx] != filtered[idx + 1]
    ]
    return filtered + bigrams


def unique_terms(paper: dict) -> set[str]:
    return set(extract_terms(paper))


def has_quality_signal(paper: dict) -> bool:
    award = str(paper.get("award", "")).strip().lower()
    if award:
        return True
    if float(paper.get("paper_weight", 0) or 0) > 0:
        return True
    return int(paper.get("influential_citation_count", 0) or 0) > 0


def build_trend_clusters(
    *,
    rows: List[dict],
    recent_rows: List[dict],
    earlier_rows: List[dict],
    top_terms: List[Dict[str, object]],
    limit: int,
) -> List[Dict[str, object]]:
    if not top_terms:
        return []

    row_terms = [(row, unique_terms(row)) for row in rows]
    recent_ids = {id(row) for row in recent_rows}
    earlier_ids = {id(row) for row in earlier_rows}
    recent_denominator = max(1, len(recent_rows))
    earlier_denominator = max(1, len(earlier_rows))
    clusters: List[Dict[str, object]] = []
    used_terms: set[str] = set()

    for seed in top_terms:
        seed_term = str(seed.get("term", "")).strip()
        if not seed_term or seed_term in used_terms:
            continue
        matched = [(row, terms) for row, terms in row_terms if seed_term in terms]
        if not matched:
            continue
        recent_matches = [(row, terms) for row, terms in matched if id(row) in recent_ids]
        earlier_matches = [(row, terms) for row, terms in matched if id(row) in earlier_ids]
        if not recent_matches:
            continue

        related_counter = Counter()
        for _, terms in recent_matches:
            related_counter.update(term for term in terms if term != seed_term and term not in used_terms)
        related_terms = [
            term
            for term, _ in related_counter.most_common(4)
            if not term.startswith(seed_term) and not seed_term.startswith(term)
        ]
        cluster_terms = [seed_term, *related_terms[:3]]
        used_terms.update(cluster_terms)

        recent_count = len(recent_matches)
        earlier_count = len(earlier_matches)
        velocity = (recent_count / recent_denominator) - (earlier_count / earlier_denominator)
        saturation = recent_count / recent_denominator
        quality_count = sum(1 for row, _ in recent_matches if has_quality_signal(row))
        award_enrichment = quality_count / max(1, recent_count)
        opportunity_score = round(
            max(0.0, velocity) * 4.0
            + award_enrichment * 1.5
            + log(1 + recent_count)
            - saturation * 0.75,
            3,
        )
        examples = sorted(
            recent_matches,
            key=lambda item: (
                float(item[0].get("paper_weight", 0) or 0),
                int(item[0].get("citation_count", 0) or 0),
                int(item[0].get("year", 0) or 0),
            ),
            reverse=True,
        )[:3]
        clusters.append(
            {
                "cluster_label": seed_term,
                "terms": cluster_terms,
                "recent_count": recent_count,
                "earlier_count": earlier_count,
                "velocity": round(velocity, 4),
                "saturation": round(saturation, 4),
                "award_enrichment": round(award_enrichment, 4),
                "opportunity_score": opportunity_score,
                "signal": classify_cluster_signal(velocity, saturation, award_enrichment),
                "example_papers": [
                    {
                        "title": str(row.get("title", "")).strip(),
                        "venue": str(row.get("venue", "")).strip(),
                        "year": int(row.get("year", 0) or 0),
                    }
                    for row, _ in examples
                ],
            }
        )
        if len(clusters) >= limit:
            break

    clusters.sort(
        key=lambda item: (
            float(item["opportunity_score"]),
            float(item["velocity"]),
            int(item["recent_count"]),
            str(item["cluster_label"]),
        ),
        reverse=True,
    )
    return clusters[:limit]


def classify_cluster_signal(velocity: float, saturation: float, award_enrichment: float) -> str:
    if velocity > 0.2 and saturation < 0.45 and award_enrichment > 0:
        return "high-upside transfer target"
    if velocity > 0.2:
        return "fast-rising frontier"
    if saturation > 0.6:
        return "crowded frontier"
    if award_enrichment > 0.4:
        return "quality-concentrated niche"
    return "watchlist"


def render_trend_report(payload: Dict[str, object]) -> str:
    lines = ["# Trend Report", ""]
    lines.append(f"- latest_year: {payload.get('latest_year')}")
    lines.append(f"- paper_count: {payload.get('paper_count')}")
    if payload.get("available_papers") is not None:
        lines.append(f"- available_papers: {payload.get('available_papers')}")
    if payload.get("trend_source_policy"):
        lines.append(f"- trend_source_policy: {payload.get('trend_source_policy')}")
    lines.append(f"- recent_papers: {payload.get('recent_papers')}")
    lines.append(f"- earlier_papers: {payload.get('earlier_papers')}")
    lines.append("")
    lines.append("## Year Counts")
    for year, count in payload.get("year_counts", {}).items():
        lines.append(f"- {year}: {count}")
    lines.append("")
    lines.append("## Top Terms")
    for item in payload.get("top_terms", []):
        lines.append(
            f"- {item['term']}: recent={item['recent_count']} earlier={item['earlier_count']} growth={item['growth_score']}"
        )
    lines.append("")
    lines.append("## Trend Clusters")
    clusters = payload.get("trend_clusters", [])
    if not clusters:
        lines.append("- No trend clusters detected yet.")
    for item in clusters:
        lines.append(
            f"- {item['cluster_label']}: score={item['opportunity_score']} "
            f"velocity={item['velocity']} saturation={item['saturation']} "
            f"award_enrichment={item['award_enrichment']} signal={item['signal']}"
        )
        terms = ", ".join(str(term) for term in item.get("terms", []))
        if terms:
            lines.append(f"  terms: {terms}")
        examples = item.get("example_papers", [])
        if examples:
            example_text = "; ".join(
                f"{paper.get('year')} {paper.get('venue')} {paper.get('title')}"
                for paper in examples
            )
            lines.append(f"  examples: {example_text}")
    return "\n".join(lines) + "\n"


def render_opportunity_map_html(payload: Dict[str, object]) -> str:
    latest_year = payload.get("latest_year")
    paper_count = int(payload.get("paper_count", 0) or 0)
    recent_papers = int(payload.get("recent_papers", 0) or 0)
    earlier_papers = int(payload.get("earlier_papers", 0) or 0)
    top_terms = payload.get("top_terms", [])
    if not isinstance(top_terms, list):
        top_terms = []
    trend_clusters = payload.get("trend_clusters", [])
    if not isinstance(trend_clusters, list):
        trend_clusters = []
    year_counts = payload.get("year_counts", {})
    if not isinstance(year_counts, dict):
        year_counts = {}

    max_year_count = max([int(count) for count in year_counts.values()], default=1)
    year_rows = []
    for year, count in year_counts.items():
        count_value = int(count)
        width = max(8, int((count_value / max_year_count) * 180)) if max_year_count else 8
        year_rows.append(
            "<div class='year-row'>"
            f"<div class='year-label'>{escape(str(year))}</div>"
            f"<div class='year-bar'><span style='width: {width}px'></span></div>"
            f"<div class='year-count'>{count_value}</div>"
            "</div>"
        )

    term_rows = []
    for item in top_terms:
        recent_count = int(item.get("recent_count", 0) or 0)
        earlier_count = int(item.get("earlier_count", 0) or 0)
        growth_score = float(item.get("growth_score", 0.0) or 0.0)
        opportunity = classify_opportunity(growth_score, recent_count, earlier_count)
        term_rows.append(
            "<tr>"
            f"<td>{escape(str(item.get('term', '')))}</td>"
            f"<td>{recent_count}</td>"
            f"<td>{earlier_count}</td>"
            f"<td>{growth_score:.4f}</td>"
            f"<td>{escape(opportunity)}</td>"
            "</tr>"
        )

    if not term_rows:
        term_rows.append("<tr><td colspan='5'>No frontier terms detected yet.</td></tr>")

    cluster_rows = []
    for item in trend_clusters:
        examples = item.get("example_papers", [])
        if not isinstance(examples, list):
            examples = []
        example_text = "; ".join(
            f"{paper.get('year', '')} {paper.get('venue', '')} {paper.get('title', '')}"
            for paper in examples
            if isinstance(paper, dict)
        )
        terms = ", ".join(str(term) for term in item.get("terms", []) if str(term).strip())
        cluster_rows.append(
            "<tr>"
            f"<td><strong>{escape(str(item.get('cluster_label', '')))}</strong><br><span class='hint'>{escape(terms)}</span></td>"
            f"<td>{int(item.get('recent_count', 0) or 0)}</td>"
            f"<td>{int(item.get('earlier_count', 0) or 0)}</td>"
            f"<td>{float(item.get('velocity', 0.0) or 0.0):.4f}</td>"
            f"<td>{float(item.get('saturation', 0.0) or 0.0):.4f}</td>"
            f"<td>{float(item.get('award_enrichment', 0.0) or 0.0):.4f}</td>"
            f"<td>{float(item.get('opportunity_score', 0.0) or 0.0):.3f}</td>"
            f"<td>{escape(str(item.get('signal', '')))}<br><span class='hint'>{escape(example_text)}</span></td>"
            "</tr>"
        )

    if not cluster_rows:
        cluster_rows.append("<tr><td colspan='8'>No trend clusters detected yet.</td></tr>")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Opportunity Map</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7fb;
      --panel: #ffffff;
      --border: #d8dce8;
      --text: #1f2430;
      --muted: #5b6475;
      --accent: #2160ff;
      --accent-soft: #dfe8ff;
      --good: #1c8c5e;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }}
    main {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 24px;
    }}
    h1, h2 {{ margin: 0 0 12px 0; }}
    p {{ margin: 0; color: var(--muted); }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin: 20px 0 24px 0;
    }}
    .metric, .panel {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
    }}
    .metric {{
      padding: 16px;
      min-height: 96px;
    }}
    .metric-label {{
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
    }}
    .metric-value {{
      margin-top: 8px;
      font-size: 28px;
      font-weight: 600;
    }}
    .panel {{
      padding: 18px;
      margin-bottom: 16px;
    }}
    .year-row {{
      display: grid;
      grid-template-columns: 56px 1fr 32px;
      gap: 12px;
      align-items: center;
      margin: 10px 0;
    }}
    .year-label, .year-count {{
      font-size: 13px;
      color: var(--muted);
    }}
    .year-bar {{
      height: 10px;
      background: #eef1f7;
      border-radius: 999px;
      overflow: hidden;
    }}
    .year-bar span {{
      display: block;
      height: 100%;
      background: linear-gradient(90deg, var(--accent), #6b93ff);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      text-align: left;
      padding: 10px 8px;
      border-bottom: 1px solid #edf0f6;
      vertical-align: top;
    }}
    th {{
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
    }}
    .hint {{
      margin-top: 8px;
      font-size: 13px;
      color: var(--muted);
    }}
    .badge {{
      display: inline-block;
      padding: 3px 8px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent);
      font-size: 12px;
      font-weight: 600;
    }}
    .good {{
      background: #e1f5ec;
      color: var(--good);
    }}
  </style>
</head>
<body>
  <main>
    <h1>Opportunity Map</h1>
    <p>Lightweight frontier view from stored titles, abstracts, and publication years.</p>

    <section class="grid">
      <div class="metric">
        <div class="metric-label">Latest Year</div>
        <div class="metric-value">{escape(str(latest_year))}</div>
      </div>
      <div class="metric">
        <div class="metric-label">Total Papers</div>
        <div class="metric-value">{paper_count}</div>
      </div>
      <div class="metric">
        <div class="metric-label">Recent Window</div>
        <div class="metric-value">{recent_papers}</div>
      </div>
      <div class="metric">
        <div class="metric-label">Earlier Baseline</div>
        <div class="metric-value">{earlier_papers}</div>
      </div>
    </section>

    <section class="panel">
      <h2>Topic Velocity</h2>
      <div class="hint">Paper counts by year help show whether the frontier window is widening or thin.</div>
      {"".join(year_rows) or "<div class='hint'>No yearly counts yet.</div>"}
    </section>

    <section class="panel">
      <h2>Frontier Opportunities</h2>
      <div class="hint">High-growth terms with recent usage are good candidates for pattern transfer and gap hunting.</div>
      <table>
        <thead>
          <tr>
            <th>Term</th>
            <th>Recent</th>
            <th>Earlier</th>
            <th>Growth</th>
            <th>Signal</th>
          </tr>
        </thead>
        <tbody>
          {"".join(term_rows)}
        </tbody>
      </table>
    </section>

    <section class="panel">
      <h2>Trend Clusters</h2>
      <div class="hint">Clusters group related frontier terms and estimate velocity, saturation, quality concentration, and opportunity for pattern transfer.</div>
      <table>
        <thead>
          <tr>
            <th>Cluster</th>
            <th>Recent</th>
            <th>Earlier</th>
            <th>Velocity</th>
            <th>Saturation</th>
            <th>Award</th>
            <th>Score</th>
            <th>Signal</th>
          </tr>
        </thead>
        <tbody>
          {"".join(cluster_rows)}
        </tbody>
      </table>
    </section>
  </main>
</body>
</html>
"""


def classify_opportunity(growth_score: float, recent_count: int, earlier_count: int) -> str:
    if recent_count >= 2 and growth_score > 0.3 and earlier_count == 0:
        return "emerging frontier"
    if growth_score > 0.15:
        return "rising opportunity"
    if recent_count >= earlier_count and growth_score >= 0:
        return "steady signal"
    return "crowded or cooling"
