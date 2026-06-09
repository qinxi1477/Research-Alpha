from __future__ import annotations

import sqlite3
from json import dumps
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    abstract TEXT NOT NULL DEFAULT '',
    venue TEXT NOT NULL,
    year INTEGER NOT NULL,
    source_kind TEXT NOT NULL DEFAULT 'manual',
    external_ref TEXT NOT NULL DEFAULT '',
    award TEXT NOT NULL DEFAULT '',
    citation_count INTEGER NOT NULL DEFAULT 0,
    influential_citation_count INTEGER NOT NULL DEFAULT 0,
    paper_weight REAL NOT NULL DEFAULT 0,
    score_notes TEXT NOT NULL DEFAULT '',
    publication_date TEXT NOT NULL DEFAULT '',
    limitations_json TEXT NOT NULL DEFAULT '{}',
    full_text_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS paper_ids (
    paper_id INTEGER NOT NULL,
    source TEXT NOT NULL,
    external_id TEXT NOT NULL,
    PRIMARY KEY (source, external_id),
    FOREIGN KEY (paper_id) REFERENCES papers(id)
);

CREATE TABLE IF NOT EXISTS quality_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id INTEGER NOT NULL,
    signal_type TEXT NOT NULL,
    signal_value REAL NOT NULL,
    note TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (paper_id) REFERENCES papers(id)
);

CREATE TABLE IF NOT EXISTS idea_cards (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id INTEGER NOT NULL,
    evidence_level TEXT NOT NULL,
    content_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (paper_id) REFERENCES papers(id)
);

CREATE TABLE IF NOT EXISTS pattern_cards (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern_key TEXT NOT NULL UNIQUE,
    content_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS candidate_ideas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    title TEXT NOT NULL,
    content_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS idea_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    initial_idea TEXT NOT NULL,
    current_idea TEXT NOT NULL,
    context_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS idea_session_turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    turn_index INTEGER NOT NULL,
    user_instruction TEXT NOT NULL,
    decision TEXT NOT NULL,
    revised_idea TEXT NOT NULL,
    content_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES idea_sessions(id)
);

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    status TEXT NOT NULL,
    manifest_path TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""

PAPER_COLUMNS = {
    "source_kind": "TEXT NOT NULL DEFAULT 'manual'",
    "external_ref": "TEXT NOT NULL DEFAULT ''",
    "award": "TEXT NOT NULL DEFAULT ''",
    "citation_count": "INTEGER NOT NULL DEFAULT 0",
    "influential_citation_count": "INTEGER NOT NULL DEFAULT 0",
    "paper_weight": "REAL NOT NULL DEFAULT 0",
    "score_notes": "TEXT NOT NULL DEFAULT ''",
    "publication_date": "TEXT NOT NULL DEFAULT ''",
    "limitations_json": "TEXT NOT NULL DEFAULT '{}'",
    "full_text_json": "TEXT NOT NULL DEFAULT '{}'",
}

IDEA_SESSION_COLUMNS = {
    "initial_idea": "TEXT NOT NULL DEFAULT ''",
    "context_json": "TEXT NOT NULL DEFAULT '{}'",
    "memory_summary_json": "TEXT NOT NULL DEFAULT '{}'",
}


def connect(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row
    return connection


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)
        _ensure_paper_columns(conn)
        _ensure_idea_session_columns(conn)


def _ensure_paper_columns(conn: sqlite3.Connection) -> None:
    for column, ddl in PAPER_COLUMNS.items():
        existing = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(papers)").fetchall()
        }
        if column in existing:
            continue
        try:
            conn.execute(f"ALTER TABLE papers ADD COLUMN {column} {ddl}")
        except sqlite3.OperationalError as exc:
            if "duplicate column name" not in str(exc).lower():
                raise


def _ensure_idea_session_columns(conn: sqlite3.Connection) -> None:
    for column, ddl in IDEA_SESSION_COLUMNS.items():
        existing = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(idea_sessions)").fetchall()
        }
        if column in existing:
            continue
        try:
            conn.execute(f"ALTER TABLE idea_sessions ADD COLUMN {column} {ddl}")
        except sqlite3.OperationalError as exc:
            if "duplicate column name" not in str(exc).lower():
                raise
    conn.execute(
        """
        UPDATE idea_sessions
        SET initial_idea = current_idea
        WHERE COALESCE(initial_idea, '') = ''
        """
    )

def add_paper(
    db_path: Path,
    title: str,
    venue: str,
    year: int,
    abstract: str = "",
    source_kind: str = "manual",
    external_ref: str = "",
    award: str = "",
    citation_count: int = 0,
    influential_citation_count: int = 0,
    publication_date: str = "",
    limitations_json: str = "{}",
) -> int:
    with connect(db_path) as conn:
        existing = conn.execute(
            """
            SELECT
                id, abstract, source_kind, external_ref, award, citation_count,
                influential_citation_count, publication_date, limitations_json, full_text_json
            FROM papers
            WHERE lower(title)=lower(?) AND lower(venue)=lower(?) AND year=?
            """,
            (title.strip(), venue.strip(), int(year)),
        ).fetchone()
        if existing:
            conn.execute(
                """
                UPDATE papers
                SET
                    abstract = CASE WHEN COALESCE(abstract, '') = '' AND ? != '' THEN ? ELSE abstract END,
                    source_kind = CASE
                        WHEN COALESCE(source_kind, '') IN ('', 'manual') AND ? != '' THEN ?
                        WHEN COALESCE(source_kind, '') = 'user_library' AND ? LIKE 'gold_%' THEN ?
                        WHEN COALESCE(source_kind, '') LIKE 'frontier_%' AND ? LIKE 'gold_%' THEN ?
                        ELSE source_kind
                    END,
                    external_ref = CASE WHEN COALESCE(external_ref, '') = '' AND ? != '' THEN ? ELSE external_ref END,
                    award = CASE WHEN ? != '' THEN ? ELSE award END,
                    citation_count = MAX(citation_count, ?),
                    influential_citation_count = MAX(influential_citation_count, ?),
                    publication_date = CASE WHEN COALESCE(publication_date, '') = '' AND ? != '' THEN ? ELSE publication_date END,
                    limitations_json = CASE WHEN ? != '{}' THEN ? ELSE limitations_json END
                WHERE id = ?
                """,
                (
                    abstract.strip(),
                    abstract.strip(),
                    source_kind.strip(),
                    source_kind.strip(),
                    source_kind.strip(),
                    source_kind.strip(),
                    source_kind.strip(),
                    source_kind.strip(),
                    external_ref.strip(),
                    external_ref.strip(),
                    award.strip(),
                    award.strip(),
                    int(citation_count),
                    int(influential_citation_count),
                    publication_date.strip(),
                    publication_date.strip(),
                    limitations_json.strip() or "{}",
                    limitations_json.strip() or "{}",
                    int(existing["id"]),
                ),
            )
            return int(existing["id"])
        cursor = conn.execute(
            """
            INSERT INTO papers(
                title, abstract, venue, year, source_kind, external_ref,
                award, citation_count, influential_citation_count, publication_date, limitations_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                title.strip(),
                abstract.strip(),
                venue.strip(),
                int(year),
                source_kind.strip() or "manual",
                external_ref.strip(),
                award.strip(),
                int(citation_count),
                int(influential_citation_count),
                publication_date.strip(),
                limitations_json.strip() or "{}",
            ),
        )
        return int(cursor.lastrowid)


def list_papers(db_path: Path, limit: int = 20) -> List[sqlite3.Row]:
    with connect(db_path) as conn:
        cursor = conn.execute(
            """
            SELECT
                id, title, venue, year, award, citation_count,
                influential_citation_count, paper_weight, publication_date, created_at
            FROM papers
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(limit),),
        )
        return list(cursor.fetchall())


def list_trend_papers(db_path: Path) -> List[sqlite3.Row]:
    with connect(db_path) as conn:
        return list(
            conn.execute(
                """
                SELECT
                    id, title, abstract, venue, year, source_kind, award, citation_count,
                    influential_citation_count, paper_weight, publication_date, limitations_json
                FROM papers
                ORDER BY year DESC, paper_weight DESC, citation_count DESC, id ASC
                """
            ).fetchall()
        )


def list_frontier_papers(db_path: Path, limit: int = 20) -> List[sqlite3.Row]:
    with connect(db_path) as conn:
        return list(
            conn.execute(
                """
                SELECT
                    id, title, abstract, venue, year, source_kind, external_ref,
                    award, citation_count, influential_citation_count, paper_weight,
                    score_notes, publication_date, limitations_json, created_at
                FROM papers
                WHERE (source_kind LIKE 'frontier_%' OR COALESCE(paper_weight, 0) <= 0)
                    AND COALESCE(source_kind, '') != 'user_library'
                ORDER BY
                    CASE WHEN source_kind LIKE 'frontier_%' THEN 0 ELSE 1 END,
                    created_at DESC, year DESC, citation_count DESC, id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        )


def list_user_library_papers(db_path: Path, limit: int = 20) -> List[sqlite3.Row]:
    with connect(db_path) as conn:
        return list(
            conn.execute(
                """
                SELECT
                    id, title, abstract, venue, year, source_kind, external_ref,
                    award, citation_count, influential_citation_count, paper_weight,
                    score_notes, publication_date, limitations_json, full_text_json, created_at
                FROM papers
                WHERE COALESCE(source_kind, '') = 'user_library'
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        )


def upsert_user_library_paper(
    db_path: Path,
    *,
    title: str,
    venue: str,
    year: int,
    abstract: str = "",
    external_ref: str = "",
) -> int:
    with connect(db_path) as conn:
        existing = None
        if external_ref.strip():
            existing = conn.execute(
                """
                SELECT id
                FROM papers
                WHERE COALESCE(source_kind, '') = 'user_library'
                    AND external_ref = ?
                """,
                (external_ref.strip(),),
            ).fetchone()
        if existing:
            conn.execute(
                """
                UPDATE papers
                SET
                    title = CASE WHEN ? != '' THEN ? ELSE title END,
                    abstract = CASE WHEN ? != '' THEN ? ELSE abstract END,
                    venue = CASE WHEN ? != '' THEN ? ELSE venue END,
                    year = ?,
                    award = '',
                    citation_count = 0,
                    influential_citation_count = 0,
                    paper_weight = 0,
                    score_notes = ?
                WHERE id = ?
                """,
                (
                    title.strip(),
                    title.strip(),
                    abstract.strip(),
                    abstract.strip(),
                    venue.strip(),
                    venue.strip(),
                    int(year),
                    dumps({"user_library_domain_knowledge_only": 0.0}, ensure_ascii=True, sort_keys=True),
                    int(existing["id"]),
                ),
            )
            return int(existing["id"])
        cursor = conn.execute(
            """
            INSERT INTO papers(
                title, abstract, venue, year, source_kind, external_ref,
                award, citation_count, influential_citation_count, paper_weight, score_notes
            ) VALUES (?, ?, ?, ?, 'user_library', ?, '', 0, 0, 0, ?)
            """,
            (
                title.strip(),
                abstract.strip(),
                venue.strip(),
                int(year),
                external_ref.strip(),
                dumps({"user_library_domain_knowledge_only": 0.0}, ensure_ascii=True, sort_keys=True),
            ),
        )
        return int(cursor.lastrowid)


def delete_user_library_paper(db_path: Path, paper_id: int) -> bool:
    with connect(db_path) as conn:
        cursor = conn.execute(
            "DELETE FROM papers WHERE id = ? AND COALESCE(source_kind, '') = 'user_library'",
            (int(paper_id),),
        )
        return int(cursor.rowcount or 0) > 0


def table_counts(db_path: Path, tables: Iterable[str]) -> List[tuple[str, int]]:
    counts = []
    with connect(db_path) as conn:
        for table in tables:
            row = conn.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()
            counts.append((table, int(row["count"])))
    return counts


def add_run(db_path: Path, query: str, status: str, manifest_path: Path) -> int:
    with connect(db_path) as conn:
        cursor = conn.execute(
            "INSERT INTO runs(query, status, manifest_path) VALUES (?, ?, ?)",
            (query.strip(), status.strip(), str(manifest_path)),
        )
        return int(cursor.lastrowid)


def list_runs(db_path: Path, limit: int = 20) -> List[sqlite3.Row]:
    with connect(db_path) as conn:
        return list(
            conn.execute(
                """
                SELECT
                    id, query, status, manifest_path, created_at
                FROM runs
                ORDER BY id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        )


def iter_papers(db_path: Path) -> List[sqlite3.Row]:
    with connect(db_path) as conn:
        return list(
            conn.execute(
                """
                SELECT
                    id, title, venue, year, source_kind, award, citation_count,
                    influential_citation_count, paper_weight, publication_date
                FROM papers
                ORDER BY year DESC, venue ASC, title ASC
                """
            ).fetchall()
        )


def get_paper_by_id(db_path: Path, paper_id: int) -> Optional[sqlite3.Row]:
    with connect(db_path) as conn:
        return conn.execute(
            """
            SELECT
                id, title, abstract, venue, year, source_kind, external_ref,
                award, citation_count, influential_citation_count, paper_weight, score_notes,
                publication_date, limitations_json, full_text_json
            FROM papers
            WHERE id = ?
            """,
            (int(paper_id),),
        ).fetchone()


def replace_quality_signals(
    db_path: Path,
    paper_id: int,
    signals: Dict[str, float],
    note: str,
    total_weight: float,
) -> None:
    with connect(db_path) as conn:
        conn.execute("DELETE FROM quality_signals WHERE paper_id = ?", (int(paper_id),))
        for signal_type, value in signals.items():
            conn.execute(
                "INSERT INTO quality_signals(paper_id, signal_type, signal_value, note) VALUES (?, ?, ?, ?)",
                (int(paper_id), signal_type, float(value), note),
            )
        conn.execute(
            "UPDATE papers SET paper_weight = ?, score_notes = ? WHERE id = ?",
            (float(total_weight), note, int(paper_id)),
        )


def reset_user_library_quality(db_path: Path, paper_id: int) -> None:
    with connect(db_path) as conn:
        conn.execute(
            """
            UPDATE papers
            SET award = '', paper_weight = 0, score_notes = ?
            WHERE id = ? AND COALESCE(source_kind, '') = 'user_library'
            """,
            (dumps({"user_library_domain_knowledge_only": 0.0}, ensure_ascii=True, sort_keys=True), int(paper_id)),
        )


def update_paper_quality_metadata(
    db_path: Path,
    paper_id: int,
    *,
    award: str = "",
    source_kind: str = "",
    citation_count: int | None = None,
    influential_citation_count: int | None = None,
) -> None:
    assignments = []
    values: List[object] = []
    if award.strip():
        assignments.append("award = ?")
        values.append(award.strip())
    if source_kind.strip():
        assignments.append("source_kind = ?")
        values.append(source_kind.strip())
    if citation_count is not None:
        assignments.append("citation_count = MAX(citation_count, ?)")
        values.append(int(citation_count))
    if influential_citation_count is not None:
        assignments.append("influential_citation_count = MAX(influential_citation_count, ?)")
        values.append(int(influential_citation_count))
    if not assignments:
        return
    values.append(int(paper_id))
    with connect(db_path) as conn:
        conn.execute(
            f"UPDATE papers SET {', '.join(assignments)} WHERE id = ?",
            tuple(values),
        )


def list_top_papers(db_path: Path, limit: int = 10) -> List[sqlite3.Row]:
    with connect(db_path) as conn:
        return list(
            conn.execute(
                """
                SELECT
                id, title, abstract, venue, year, source_kind, external_ref, award, citation_count,
                influential_citation_count, paper_weight, score_notes,
                publication_date, limitations_json, full_text_json
                FROM papers
                WHERE COALESCE(paper_weight, 0) > 0
                ORDER BY paper_weight DESC, citation_count DESC, id ASC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        )


def count_scored_papers(db_path: Path) -> int:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS count FROM papers WHERE COALESCE(paper_weight, 0) > 0"
        ).fetchone()
        return int(row["count"])


def build_score_note(signals: Dict[str, float]) -> str:
    return dumps(signals, ensure_ascii=True, sort_keys=True)


def upsert_idea_card(
    db_path: Path,
    paper_id: int,
    evidence_level: str,
    content_json: str,
) -> int:
    with connect(db_path) as conn:
        existing = conn.execute(
            "SELECT id FROM idea_cards WHERE paper_id = ?",
            (int(paper_id),),
        ).fetchone()
        if existing:
            conn.execute(
                """
                UPDATE idea_cards
                SET evidence_level = ?, content_json = ?, created_at = CURRENT_TIMESTAMP
                WHERE paper_id = ?
                """,
                (evidence_level.strip(), content_json, int(paper_id)),
            )
            return int(existing["id"])
        cursor = conn.execute(
            "INSERT INTO idea_cards(paper_id, evidence_level, content_json) VALUES (?, ?, ?)",
            (int(paper_id), evidence_level.strip(), content_json),
        )
        return int(cursor.lastrowid)


def list_idea_cards(db_path: Path, limit: int = 20) -> List[sqlite3.Row]:
    with connect(db_path) as conn:
        return list(
            conn.execute(
                """
                SELECT
                    idea_cards.id,
                    idea_cards.paper_id,
                    idea_cards.evidence_level,
                    idea_cards.created_at,
                    papers.title,
                    papers.venue,
                    papers.year,
                    papers.source_kind,
                    papers.paper_weight
                FROM idea_cards
                JOIN papers ON papers.id = idea_cards.paper_id
                ORDER BY idea_cards.created_at DESC, idea_cards.id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        )


def count_idea_cards(db_path: Path) -> int:
    with connect(db_path) as conn:
        row = conn.execute("SELECT COUNT(*) AS count FROM idea_cards").fetchone()
        return int(row["count"])


def list_papers_for_genome_build(
    db_path: Path,
    limit: int = 10,
    only_missing: bool = True,
    high_weight_only: bool = False,
) -> List[sqlite3.Row]:
    where_terms = []
    if only_missing:
        where_terms.append("idea_cards.id IS NULL")
    if high_weight_only:
        where_terms.append("papers.paper_weight > 0")
    where_clause = "WHERE " + " AND ".join(where_terms) if where_terms else ""
    with connect(db_path) as conn:
        return list(
            conn.execute(
                f"""
                SELECT
                    papers.id,
                    papers.title,
                    papers.abstract,
                    papers.venue,
                    papers.year,
                    papers.source_kind,
                    papers.external_ref,
                    papers.award,
                    papers.citation_count,
                    papers.influential_citation_count,
                    papers.paper_weight,
                    papers.score_notes
                    ,papers.publication_date,
                    papers.limitations_json,
                    papers.full_text_json
                FROM papers
                LEFT JOIN idea_cards ON idea_cards.paper_id = papers.id
                {where_clause}
                ORDER BY papers.paper_weight DESC, papers.citation_count DESC, papers.id ASC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        )


def list_genome_card_payloads(db_path: Path, limit: int = 10) -> List[sqlite3.Row]:
    with connect(db_path) as conn:
        return list(
            conn.execute(
                """
                SELECT
                    idea_cards.id,
                    idea_cards.paper_id,
                    idea_cards.evidence_level,
                    idea_cards.content_json,
                    papers.title,
                    papers.venue,
                    papers.year,
                    papers.source_kind,
                    papers.abstract,
                    papers.full_text_json,
                    papers.award,
                    papers.paper_weight
                FROM idea_cards
                JOIN papers ON papers.id = idea_cards.paper_id
                ORDER BY papers.paper_weight DESC, papers.citation_count DESC, idea_cards.id ASC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        )


def list_gold_genome_card_payloads(db_path: Path, limit: int = 10) -> List[sqlite3.Row]:
    with connect(db_path) as conn:
        return list(
            conn.execute(
                """
                SELECT
                    idea_cards.id,
                    idea_cards.paper_id,
                    idea_cards.evidence_level,
                    idea_cards.content_json,
                    papers.title,
                    papers.venue,
                    papers.year,
                    papers.source_kind,
                    papers.abstract,
                    papers.full_text_json,
                    papers.award,
                    papers.paper_weight
                FROM idea_cards
                JOIN papers ON papers.id = idea_cards.paper_id
                WHERE papers.paper_weight > 0
                ORDER BY papers.paper_weight DESC, papers.citation_count DESC, idea_cards.id ASC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        )


def upsert_pattern_card(
    db_path: Path,
    pattern_key: str,
    content_json: str,
) -> int:
    with connect(db_path) as conn:
        existing = conn.execute(
            "SELECT id FROM pattern_cards WHERE pattern_key = ?",
            (pattern_key.strip(),),
        ).fetchone()
        if existing:
            conn.execute(
                """
                UPDATE pattern_cards
                SET content_json = ?, created_at = CURRENT_TIMESTAMP
                WHERE pattern_key = ?
                """,
                (content_json, pattern_key.strip()),
            )
            return int(existing["id"])
        cursor = conn.execute(
            "INSERT INTO pattern_cards(pattern_key, content_json) VALUES (?, ?)",
            (pattern_key.strip(), content_json),
        )
        return int(cursor.lastrowid)


def list_pattern_cards(db_path: Path, limit: int = 20) -> List[sqlite3.Row]:
    with connect(db_path) as conn:
        return list(
            conn.execute(
                """
                SELECT
                    id,
                    pattern_key,
                    content_json,
                    created_at
                FROM pattern_cards
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        )


def count_pattern_cards(db_path: Path) -> int:
    with connect(db_path) as conn:
        row = conn.execute("SELECT COUNT(*) AS count FROM pattern_cards").fetchone()
        return int(row["count"])


def add_candidate_idea(
    db_path: Path,
    query: str,
    title: str,
    content_json: str,
) -> int:
    with connect(db_path) as conn:
        cursor = conn.execute(
            "INSERT INTO candidate_ideas(query, title, content_json) VALUES (?, ?, ?)",
            (query.strip(), title.strip(), content_json),
        )
        return int(cursor.lastrowid)


def get_candidate_idea(db_path: Path, idea_id: int) -> Optional[sqlite3.Row]:
    with connect(db_path) as conn:
        return conn.execute(
            """
            SELECT
                id, query, title, content_json, created_at
            FROM candidate_ideas
            WHERE id = ?
            """,
            (int(idea_id),),
        ).fetchone()


def delete_candidate_idea(db_path: Path, idea_id: int) -> None:
    with connect(db_path) as conn:
        conn.execute("DELETE FROM candidate_ideas WHERE id = ?", (int(idea_id),))


def list_candidate_ideas(db_path: Path, limit: int = 20) -> List[sqlite3.Row]:
    with connect(db_path) as conn:
        return list(
            conn.execute(
                """
                SELECT
                    id, query, title, content_json, created_at
                FROM candidate_ideas
                ORDER BY id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        )


def list_pattern_payloads(db_path: Path, limit: int = 10) -> List[sqlite3.Row]:
    with connect(db_path) as conn:
        return list(
            conn.execute(
                """
                SELECT
                    id,
                    pattern_key,
                    content_json,
                    created_at
                FROM pattern_cards
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        )


def create_idea_session(
    db_path: Path,
    name: str,
    current_idea: str,
    context_json: str = "{}",
) -> int:
    with connect(db_path) as conn:
        cursor = conn.execute(
            "INSERT INTO idea_sessions(name, initial_idea, current_idea, context_json) VALUES (?, ?, ?, ?)",
            (name.strip(), current_idea.strip(), current_idea.strip(), context_json),
        )
        return int(cursor.lastrowid)


def get_idea_session(db_path: Path, session_id: int) -> Optional[sqlite3.Row]:
    with connect(db_path) as conn:
        return conn.execute(
            """
            SELECT
                id, name, initial_idea, current_idea, context_json, memory_summary_json, status, created_at, updated_at
            FROM idea_sessions
            WHERE id = ?
            """,
            (int(session_id),),
        ).fetchone()


def list_idea_sessions(db_path: Path, limit: int = 20) -> List[sqlite3.Row]:
    with connect(db_path) as conn:
        return list(
            conn.execute(
                """
                SELECT
                    id, name, initial_idea, current_idea, context_json, status, created_at, updated_at
                    ,memory_summary_json
                FROM idea_sessions
                ORDER BY updated_at DESC, id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        )


def delete_idea_session(db_path: Path, session_id: int) -> None:
    with connect(db_path) as conn:
        conn.execute("DELETE FROM idea_session_turns WHERE session_id = ?", (int(session_id),))
        conn.execute("DELETE FROM idea_sessions WHERE id = ?", (int(session_id),))


def add_idea_session_turn(
    db_path: Path,
    session_id: int,
    user_instruction: str,
    decision: str,
    revised_idea: str,
    content_json: str,
) -> int:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT COALESCE(MAX(turn_index), 0) AS turn_index FROM idea_session_turns WHERE session_id = ?",
            (int(session_id),),
        ).fetchone()
        next_turn = int(row["turn_index"]) + 1
        cursor = conn.execute(
            """
            INSERT INTO idea_session_turns(
                session_id, turn_index, user_instruction, decision, revised_idea, content_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                int(session_id),
                next_turn,
                user_instruction.strip(),
                decision.strip(),
                revised_idea.strip(),
                content_json,
            ),
        )
        conn.execute(
            """
            UPDATE idea_sessions
            SET current_idea = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (revised_idea.strip(), int(session_id)),
        )
        return int(cursor.lastrowid)


def update_idea_session_context(db_path: Path, session_id: int, context_json: str) -> None:
    with connect(db_path) as conn:
        conn.execute(
            """
            UPDATE idea_sessions
            SET context_json = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (context_json, int(session_id)),
        )


def update_idea_session_memory(db_path: Path, session_id: int, memory_summary_json: str) -> None:
    with connect(db_path) as conn:
        conn.execute(
            """
            UPDATE idea_sessions
            SET memory_summary_json = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (memory_summary_json, int(session_id)),
        )


def update_paper_limitations(db_path: Path, paper_id: int, limitations_json: str) -> None:
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE papers SET limitations_json = ? WHERE id = ?",
            (limitations_json.strip() or "{}", int(paper_id)),
        )


def update_paper_full_text(db_path: Path, paper_id: int, full_text_json: str) -> None:
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE papers SET full_text_json = ? WHERE id = ?",
            (full_text_json.strip() or "{}", int(paper_id)),
        )


def update_user_library_paper_metadata(
    db_path: Path,
    paper_id: int,
    *,
    title: str = "",
    venue: str = "",
    year: int = 0,
    abstract: str = "",
) -> None:
    with connect(db_path) as conn:
        conn.execute(
            """
            UPDATE papers
            SET
                title = CASE WHEN ? != '' THEN ? ELSE title END,
                abstract = CASE WHEN ? != '' THEN ? ELSE abstract END,
                venue = CASE WHEN ? != '' THEN ? ELSE venue END,
                year = CASE WHEN ? > 0 THEN ? ELSE year END,
                award = '',
                citation_count = 0,
                influential_citation_count = 0,
                paper_weight = 0,
                score_notes = ?
            WHERE id = ? AND COALESCE(source_kind, '') = 'user_library'
            """,
            (
                title.strip(),
                title.strip(),
                abstract.strip(),
                abstract.strip(),
                venue.strip(),
                venue.strip(),
                int(year or 0),
                int(year or 0),
                dumps({"user_library_domain_knowledge_only": 0.0}, ensure_ascii=True, sort_keys=True),
                int(paper_id),
            ),
        )


def list_idea_session_turns(db_path: Path, session_id: int) -> List[sqlite3.Row]:
    with connect(db_path) as conn:
        return list(
            conn.execute(
                """
                SELECT
                    id, session_id, turn_index, user_instruction, decision, revised_idea, content_json, created_at
                FROM idea_session_turns
                WHERE session_id = ?
                ORDER BY turn_index ASC
                """,
                (int(session_id),),
            ).fetchall()
        )
