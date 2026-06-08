from __future__ import annotations

import argparse
import csv
import difflib
import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from research_alpha.config import load_config, normalize_provider, provider_api_key_env, upsert_dotenv
from research_alpha.connectors import ConnectorConfig, ConnectorError, harvest_acl_awards, harvest_cvf_awards, harvest_eccv_awards, harvest_icml_awards, harvest_neurips_awards, harvest_openalex, harvest_openreview, harvest_semantic_scholar
from research_alpha.critique import CRITIQUE_SYSTEM_PROMPT, build_critique_prompt, parse_critique_response
from research_alpha.db import (
    add_candidate_idea,
    add_paper,
    add_run,
    add_idea_session_turn,
    build_score_note,
    connect,
    count_idea_cards,
    count_pattern_cards,
    count_scored_papers,
    create_idea_session,
    get_candidate_idea,
    get_paper_by_id,
    get_idea_session,
    init_db,
    iter_papers,
    list_candidate_ideas,
    list_idea_sessions,
    list_idea_session_turns,
    list_gold_genome_card_payloads,
    list_pattern_payloads,
    list_papers_for_genome_build,
    list_papers,
    list_idea_cards,
    list_pattern_cards,
    list_trend_papers,
    list_runs,
    list_top_papers,
    list_user_library_papers,
    replace_quality_signals,
    table_counts,
    update_idea_session_context,
    update_idea_session_memory,
    update_paper_quality_metadata,
    update_paper_limitations,
    upsert_idea_card,
    upsert_pattern_card,
)
from research_alpha.genome import GENOME_SYSTEM_PROMPT, build_genome_prompt, parse_genome_response
from research_alpha.gui import run_gui
from research_alpha.ideas import IDEA_SYSTEM_PROMPT, build_idea_prompt, parse_idea_response
from research_alpha.json_utils import parse_llm_json
from research_alpha.layout import ensure_layout, scaffold_project_files, write_gitkeep
from research_alpha.limitations import build_limitations_payload, limitations_json_for_paper, recent_limitations_summary
from research_alpha.llm import LLMClient, LLMError, provider_summary
from research_alpha.memory import (
    build_compressed_session_context,
    build_session_memory_status,
    build_session_memory_summary,
    compact_list,
    load_session_memory,
    render_memory_markdown,
)
from research_alpha.patterns import PATTERN_SYSTEM_PROMPT, build_pattern_prompt, parse_pattern_response
from research_alpha.prior_art import analyze_prior_art
from research_alpha.ranking import RANKING_SYSTEM_PROMPT, build_ranking_prompt, parse_ranking_response
from research_alpha.trends import analyze_trends, render_opportunity_map_html, render_trend_report


TABLES = ["papers", "paper_ids", "quality_signals", "idea_cards", "pattern_cards", "candidate_ideas", "runs"]
PROVIDER_CHOICES = ["openai", "deepseek", "oa", "o", "ds", "d"]
HELP_CHOICES = {"-h", "--help"}
CLI_CMD = "ra"
FRONTIER_SOURCE_PREFIX = "frontier_"
GOLD_SOURCE_PREFIX = "gold_"
USER_LIBRARY_SOURCE_KIND = "user_library"
REMOTE_FRONTIER_SOURCES = {"openalex", "s2", "semantic_scholar", "arxiv", "openreview", "icml_awards", "neurips_awards", "acl_awards", "cvf_awards", "eccv_awards"}
CORE_GOLD_VENUE_TOKENS = {
    "cvpr",
    "iccv",
    "iclr",
    "neurips",
    "nips",
    "icml",
    "tpami",
    "t-pami",
    "transactions on pattern analysis and machine intelligence",
}


def parse_llm_response_with_repair(
    *,
    client: LLMClient,
    original_prompt: str,
    system_prompt: str,
    original_text: str,
    parser,
    response_name: str,
    validator=None,
) -> object:
    try:
        payload = parser(original_text)
        if validator:
            validator(payload)
        return payload
    except (ValueError, json.JSONDecodeError) as first_error:
        progress_note(f"{response_name} 校验失败，执行一次受约束修复重试")
        repair_prompt = build_json_repair_prompt(
            response_name=response_name,
            original_prompt=original_prompt,
            invalid_text=original_text,
            error=str(first_error),
        )
        repair_response = client.chat(prompt=repair_prompt, system_prompt=system_prompt)
        try:
            payload = parser(repair_response.text)
            if validator:
                validator(payload)
            return payload
        except (ValueError, json.JSONDecodeError) as repair_error:
            raise ValueError(
                f"{response_name} remained invalid after one repair attempt: {repair_error}. "
                f"Original parse error: {first_error}"
            ) from repair_error


def build_json_repair_prompt(
    *,
    response_name: str,
    original_prompt: str,
    invalid_text: str,
    error: str,
) -> str:
    return (
        f"Repair the previous {response_name} so it passes the schema and validation rules from the original task.\n"
        "Return strict JSON only. Do not add markdown, comments, explanations, or new evidence beyond the supplied context.\n"
        "Preserve the same intent, evidence IDs, and user language as much as possible, but add any missing required keys "
        "only when they can be grounded in the original task context. If the error says the answer left the evidence "
        "logic space, repair by citing supplied Pattern/Genome/Gold evidence and by treating user-library/domain-knowledge "
        "papers only as background, never as standards.\n\n"
        f"Validation error:\n{compact_excerpt(error, limit=900)}\n\n"
        f"Invalid response:\n{compact_excerpt(invalid_text, limit=6000)}\n\n"
        f"Original task:\n{compact_excerpt(original_prompt, limit=12000)}"
    )


def build_idea_semantic_retry_prompt(
    *,
    original_prompt: str,
    failure_summary: Dict[str, object],
) -> str:
    return (
        "Retry the idea-generation task once because every previous candidate failed strict product safety gates.\n"
        "Return strict JSON only with the same schema as the original task.\n"
        "This is not permission to loosen the standard: every new idea must still cite supplied Gold/Pattern/Genome "
        "source IDs, cover innovation|logic|feasibility|value|defensibility, and include a complete six-step "
        "storyline_trace.\n\n"
        "Recovery constraints:\n"
        "- First reconstruct the abstract paper storyline, then migrate the logic into the user's domain.\n"
        "- Do not copy source-paper method names, model names, dataset names, benchmark names, task objects, or surface terminology.\n"
        "- Treat user-library/domain-knowledge papers only as terminology and route background; never use them as evidence_basis, "
        "storyline_trace.source_id, quality_signal, Gold evidence, or scoring standard.\n"
        "- If a prior candidate was rejected for hard transfer, choose a different causal tension or rewrite the visible idea so it is "
        "domain-native while preserving the Gold logic.\n"
        "- If a prior candidate used user-library evidence as a standard, replace that standard with an allowed Pattern/Genome/Gold source.\n\n"
        f"Failure summary:\n{json.dumps(sanitize_failure_for_prompt(failure_summary), ensure_ascii=False, indent=2)}\n\n"
        f"Original task and Context:\n{compact_excerpt(original_prompt, limit=14000)}"
    )


def sanitize_failure_for_prompt(value: object) -> object:
    if isinstance(value, dict):
        sanitized: Dict[str, object] = {}
        allowed = {
            "status",
            "query",
            "requested",
            "rejected",
            "primary_reason",
            "reason_codes",
            "next_action",
            "recovery_strategy",
            "blocked_dimensions",
            "retry_prompt_constraints",
            "rejected_ideas",
        }
        for key, item in value.items():
            if key not in allowed:
                continue
            sanitized[str(key)] = sanitize_failure_for_prompt(item)
        return sanitized
    if isinstance(value, list):
        return [sanitize_failure_for_prompt(item) for item in value[:5]]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
CORE_GOLD_REMOTE_SOURCES = {"gold_openreview", "gold_icml_awards", "gold_neurips_awards", "gold_cvf_awards"}
DEFAULT_ASK_SMOKE_PROMPT = "Reply with one short sentence confirming that this LLM connection is working."
DEFAULT_AWARD_WEIGHTS = {
    "best_paper": 5.0,
    "outstanding_paper": 4.0,
    "test_of_time": 4.0,
    "high_citation": 3.0,
    "oral": 2.0,
    "spotlight": 1.5,
}
STRICT_REMOTE_QUALITY_AWARDS = {"best_paper", "outstanding_paper", "oral"}
CORE_GOLD_STANDARD_AWARDS = set(STRICT_REMOTE_QUALITY_AWARDS)
PROMOTABLE_METADATA_AWARDS = set(STRICT_REMOTE_QUALITY_AWARDS)
METADATA_ONLY_GOLD_BUILD_REMOTES = {"openalex"}
CORE_GOLD_QUALITY_POLICY = {
    "core_venues": ["CVPR", "ICCV", "ICLR", "NeurIPS/NIPS", "ICML", "TPAMI"],
    "accepted_quality_signals": ["best_paper", "outstanding_paper/nominee/honorable_mention", "oral"],
    "excluded_from_gold": ["poster", "spotlight", "high_citation_only", "ordinary_accept", "auxiliary_venues"],
}
GOLD_BUILD_SOURCE_PRIORITY = [
    "Official award pages for ICML/NeurIPS/CVF Best and Outstanding papers",
    "Official ICML/NeurIPS/CVF oral event pages and OpenReview metadata for core Oral papers",
    "OpenAlex metadata only as trend/frontier evidence until quality-enrich verifies explicit Best/Nominee/Oral signals",
]
POSTER_RE = re.compile(r"\bposter(?:s|_paper|_presentation)?\b", re.IGNORECASE)
EXTRACTIVE_EVIDENCE_LEVEL = "extractive_abstract_only"
EXTRACTIVE_PATTERN_EVIDENCE_LEVEL = "extractive_abstract_only_aggregation"
TERM_STOPWORDS = {
    "about",
    "across",
    "after",
    "also",
    "approach",
    "based",
    "being",
    "benchmark",
    "benchmarks",
    "between",
    "could",
    "data",
    "from",
    "into",
    "learning",
    "method",
    "model",
    "models",
    "paper",
    "results",
    "show",
    "shows",
    "system",
    "systems",
    "that",
    "their",
    "these",
    "this",
    "through",
    "using",
    "with",
}


def progress_note(message: str) -> None:
    print(f"进度：{message}", flush=True)

EVIDENCE_RUBRIC_DIMENSIONS = [
    {
        "key": "innovation",
        "label": "Innovation",
        "weight": 0.25,
        "candidate_fields": ["novelty", "core_hypothesis", "historical_pattern", "frontier_gap"],
        "pattern_fields": ["core_move", "operator_template", "when_to_use"],
        "genome_fields": ["bottleneck_or_hidden_assumption", "problem_reframing", "transferable_pattern"],
        "paper_fields": ["title", "abstract", "score_notes"],
    },
    {
        "key": "logic",
        "label": "Logic",
        "weight": 0.20,
        "candidate_fields": ["core_hypothesis", "historical_pattern", "trend_support", "why_now", "paper_angle"],
        "pattern_fields": ["core_move", "why_it_works", "operator_template"],
        "genome_fields": ["pre_publication_belief", "why_now", "story_line"],
        "paper_fields": ["title", "abstract"],
    },
    {
        "key": "feasibility",
        "label": "Feasibility",
        "weight": 0.20,
        "candidate_fields": ["first_experiments", "evaluation_outline", "why_now"],
        "pattern_fields": ["when_to_use", "failure_modes"],
        "genome_fields": ["evidence_design", "failure_boundary", "why_now"],
        "paper_fields": ["abstract", "score_notes"],
    },
    {
        "key": "value",
        "label": "Value",
        "weight": 0.20,
        "candidate_fields": ["value", "paper_angle", "trend_support"],
        "pattern_fields": ["why_it_works", "canonical_examples"],
        "genome_fields": ["paper_summary", "story_line", "transferable_pattern"],
        "paper_fields": ["title", "abstract", "score_notes"],
    },
    {
        "key": "defensibility",
        "label": "Defensibility",
        "weight": 0.15,
        "candidate_fields": ["key_risk", "evaluation_outline", "prior_art"],
        "pattern_fields": ["failure_modes", "when_to_use"],
        "genome_fields": ["failure_boundary", "evidence_design", "confidence_note"],
        "paper_fields": ["abstract", "score_notes"],
    },
]


def build_session_context_from_candidate(payload: Dict[str, object], candidate_idea_id: int) -> Dict[str, object]:
    prior_art = payload.get("prior_art", {})
    if not isinstance(prior_art, dict):
        prior_art = {}
    prior_art_gate = payload.get("prior_art_gate", {})
    if not isinstance(prior_art_gate, dict):
        prior_art_gate = {}
    first_experiments = payload.get("first_experiments", [])
    if not isinstance(first_experiments, list):
        first_experiments = []
    critique_outputs = build_candidate_critique_outputs(payload)
    storyline_trace = payload.get("storyline_trace", {})
    if not isinstance(storyline_trace, dict):
        storyline_trace = {}
    return {
        "source": "candidate_idea",
        "candidate_idea_id": int(candidate_idea_id),
        "idea_title": str(payload.get("idea_title", "")).strip(),
        "core_hypothesis": str(payload.get("core_hypothesis", "")).strip(),
        "historical_pattern": str(payload.get("historical_pattern", "")).strip(),
        "trend_support": str(payload.get("trend_support", "")).strip(),
        "frontier_gap": str(payload.get("frontier_gap", "")).strip(),
        "why_now": str(payload.get("why_now", "")).strip(),
        "novelty": str(payload.get("novelty", "")).strip(),
        "value": str(payload.get("value", "")).strip(),
        "key_risk": str(payload.get("key_risk", "")).strip(),
        "evaluation_outline": str(payload.get("evaluation_outline", "")).strip(),
        "first_experiments": [str(item).strip() for item in first_experiments if str(item).strip()],
        "paper_angle": str(payload.get("paper_angle", "")).strip(),
        "storyline_trace": storyline_trace,
        "evidence_basis": normalize_explicit_evidence_basis(payload.get("evidence_basis")),
        "domain_knowledge_context": payload.get("domain_knowledge_context", {})
        if isinstance(payload.get("domain_knowledge_context"), dict)
        else {},
        "generation_evidence_audit": payload.get("generation_evidence_audit", {})
        if isinstance(payload.get("generation_evidence_audit"), dict)
        else {},
        "evidence_grounded_score": payload.get("evidence_grounded_score", {})
        if isinstance(payload.get("evidence_grounded_score"), dict)
        else {},
        "prior_art_label": str(prior_art.get("overlap_label", "")).strip(),
        "prior_art_summary": str(prior_art.get("summary", "")).strip(),
        "differentiation_note": str(prior_art.get("differentiation_note", "")).strip(),
        "prior_art_gate_decision": str(prior_art_gate.get("decision", "")).strip(),
        "prior_art_gate_summary": str(prior_art_gate.get("summary", "")).strip(),
        "why_not_done_yet": [
            str(item).strip()
            for item in prior_art_gate.get("why_not_done_yet", [])
            if str(item).strip()
        ]
        if isinstance(prior_art_gate.get("why_not_done_yet"), list)
        else [],
        "required_differentiation": [
            str(item).strip()
            for item in prior_art_gate.get("required_differentiation", [])
            if str(item).strip()
        ]
        if isinstance(prior_art_gate.get("required_differentiation"), list)
        else [],
        "critique_outputs": critique_outputs,
    }


def refresh_session_memory(root_dir: Path, session_id: int) -> Dict[str, object]:
    config = load_config(root_dir)
    ensure_layout(config)
    init_db(config.db_path)
    session = get_idea_session(config.db_path, session_id)
    if not session:
        raise ValueError(f"Idea session not found: {session_id}")
    turns = list_idea_session_turns(config.db_path, session_id)
    session_context = load_session_context(session)
    memory = build_session_memory_summary(session, turns, session_context)
    update_idea_session_memory(
        config.db_path,
        session_id,
        json.dumps(memory, ensure_ascii=False, indent=2),
    )
    return memory


def load_session_context(session) -> Dict[str, object]:
    raw = str(session["context_json"]).strip() if "context_json" in session.keys() else ""
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def known_command_names(parser: argparse.ArgumentParser) -> set[str]:
    names: set[str] = set()
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            names.update(action.choices.keys())
    return names


def is_provider_choice(token: str) -> bool:
    return token.strip().lower() in PROVIDER_CHOICES


def normalize_provider_shortcut(argv: list[str]) -> list[str]:
    if not argv:
        return argv
    first = argv[0].strip().lower()
    if not is_provider_choice(first):
        return argv
    return ["llm", *argv]


def normalize_llm_shortcut(argv: list[str]) -> list[str]:
    if not argv or argv[0] not in {"llm", "lm"}:
        return argv
    if len(argv) < 2:
        return argv
    second = argv[1].strip()
    if not second or second.startswith("-") or is_provider_choice(second):
        return argv
    return [argv[0], "--api-key", *argv[1:]]


def normalize_default_command(parser: argparse.ArgumentParser, argv: list[str]) -> list[str]:
    if not argv:
        return argv
    first = argv[0]
    if first in HELP_CHOICES:
        return argv
    if first in {"llm", "lm"}:
        return normalize_llm_shortcut(argv)
    if is_provider_choice(first):
        return normalize_provider_shortcut(argv)
    if first in known_command_names(parser):
        return argv
    return ["r", *argv]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ra",
        description="Research Alpha CLI for building and evaluating AI/ML research ideas.",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("init", help="Create the local project layout and SQLite database.")
    subparsers.add_parser("status", help="Show project, database, and LLM configuration status.")

    gui_parser = subparsers.add_parser("gui", help="Start the local Research Alpha GUI.")
    gui_parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    gui_parser.add_argument("--port", default=8765, type=int, help="Port to bind.")
    gui_parser.add_argument("--no-open", action="store_true", help="Do not open a browser automatically.")

    llm_parser = subparsers.add_parser("llm", aliases=["lm"], help="Show or update persisted LLM provider settings.")
    llm_parser.add_argument(
        "provider_name",
        nargs="?",
        help=f"Short provider switch, for example: `{CLI_CMD} llm ds` or `{CLI_CMD} ds`.",
    )
    llm_parser.add_argument(
        "api_key_value",
        nargs="?",
        help=f"Optional short API key form, for example: `{CLI_CMD} llm ds sk-...`, `{CLI_CMD} llm sk-...`, or `{CLI_CMD} ds sk-...`.",
    )
    llm_parser.add_argument("--provider", choices=PROVIDER_CHOICES, help="Persist the default provider.")
    llm_parser.add_argument("--model", help="Persist a default model override.")
    llm_parser.add_argument("--base-url", help="Persist a default base URL override.")
    llm_parser.add_argument("--api-key", help="Persist an API key for the chosen or current provider.")
    llm_parser.add_argument("--clear-model", action="store_true", help="Clear the persisted model override.")
    llm_parser.add_argument("--clear-base-url", action="store_true", help="Clear the persisted base URL override.")
    llm_parser.add_argument("--clear-api-key", action="store_true", help="Clear the persisted API key for the chosen or current provider.")

    ask_parser = subparsers.add_parser("ask", aliases=["a"], help="Send a quick prompt to the configured LLM provider.")
    ask_parser.add_argument(
        "prompt",
        nargs="?",
        help="Optional user prompt. Omit it to run a built-in LLM smoke check.",
    )
    ask_parser.add_argument("--system", default="", help="Optional system prompt.")
    ask_parser.add_argument("--provider", choices=PROVIDER_CHOICES, help="Temporarily override the provider.")
    ask_parser.add_argument("--model", default="", help="Temporarily override the model.")

    add_parser = subparsers.add_parser("add-paper", help="Add one paper record to the local database.")
    add_parser.add_argument("--title", required=True, help="Paper title.")
    add_parser.add_argument("--venue", required=True, help="Venue shorthand or display name.")
    add_parser.add_argument("--year", required=True, type=int, help="Publication year.")
    add_parser.add_argument("--abstract", default="", help="Optional abstract text.")

    papers_parser = subparsers.add_parser("papers", help="List recently added paper records.")
    papers_parser.add_argument("--limit", default=10, type=int, help="Maximum number of rows to print.")

    harvest_parser = subparsers.add_parser(
        "harvest",
        aliases=["grab", "ingest", "h"],
        help="Import local paper metadata from a JSONL, JSON, or CSV file.",
    )
    harvest_parser.add_argument("--file", help="Path to a local seed file.")
    harvest_parser.add_argument("--remote", choices=["openalex", "s2"], help="Remote source to query.")
    harvest_parser.add_argument("--query", default="", help="Optional text query for remote harvest.")
    harvest_parser.add_argument("--venue", default="", help="Venue filter or venue search hint.")
    harvest_parser.add_argument("--year", type=int, help="Publication year filter for remote harvest.")
    harvest_parser.add_argument("--limit", default=20, type=int, help="Maximum number of remote results to import.")
    harvest_parser.add_argument("--source", default="seed", help="Source label stored with the imported papers.")

    gold_parser = subparsers.add_parser(
        "gold-build",
        aliases=["gold"],
        help="Strictly expand excellent-paper evidence from core-venue best/nominee/oral signals.",
    )
    gold_parser.add_argument("--remote", choices=["openalex", "s2", "openreview", "icml_awards", "neurips_awards", "acl_awards", "cvf_awards", "eccv_awards"], default="openreview", help="Remote source to query.")
    gold_parser.add_argument("--query", default="", help="Optional topic query, e.g. agents, diffusion, scientific discovery.")
    gold_parser.add_argument("--venues", default="ICLR,ICML,NeurIPS,CVPR,ICCV,TPAMI", help="Comma-separated venue list.")
    gold_parser.add_argument("--from-year", type=int, default=2014, help="First publication year to harvest.")
    gold_parser.add_argument("--to-year", type=int, default=2026, help="Last publication year to harvest.")
    gold_parser.add_argument("--per-venue-year", type=int, default=8, help="Maximum remote records per venue-year before strict quality filtering.")
    gold_parser.add_argument("--extractive-genomes", action="store_true", help="Also build abstract-only Genome drafts for newly scored excellent papers.")

    subparsers.add_parser("score", aliases=["rank", "s"], help="Compute paper weights from awards and citation signals.")

    top_parser = subparsers.add_parser("top", aliases=["best", "t"], help="Show the highest-weight papers.")
    top_parser.add_argument("--limit", default=10, type=int, help="Maximum number of rows to print.")

    trend_parser = subparsers.add_parser(
        "trends",
        aliases=["tr"],
        help="Build a lightweight frontier trend report from stored papers.",
    )
    trend_parser.add_argument("--years", default=5, type=int, help="Recent-year window for frontier trends.")
    trend_parser.add_argument("--top-k", default=10, type=int, help="How many trend terms to show.")

    limitations_parser = subparsers.add_parser(
        "limitations",
        aliases=["lim"],
        help="Extract and report explicit limitation/challenge signals from recent high-quality/frontier papers.",
    )
    limitations_parser.add_argument("--months", type=int, default=6, help="Recent window in months.")
    limitations_parser.add_argument("--limit", type=int, default=12, help="Maximum papers to include.")

    quality_enrich_parser = subparsers.add_parser(
        "quality-enrich",
        aliases=["qe"],
        help="Use the LLM only as a quality-metadata annotator for best/oral/high-citation signals.",
    )
    quality_enrich_parser.add_argument("--limit", type=int, default=20, help="Maximum papers to annotate.")
    quality_enrich_parser.add_argument("--apply", action="store_true", help="Apply high-confidence quality labels to paper metadata.")
    quality_enrich_parser.add_argument("--provider", choices=PROVIDER_CHOICES, help="Temporarily override the provider.")
    quality_enrich_parser.add_argument("--model", default="", help="Temporarily override the model.")

    evidence_parser = subparsers.add_parser(
        "evidence",
        aliases=["ev"],
        help="Audit whether the high-quality evidence base is ready for strict idea generation.",
    )
    evidence_parser.add_argument("--papers", default=10, type=int, help="How many core Gold papers to include.")

    audit_parser = subparsers.add_parser(
        "audit",
        aliases=["au"],
        help="Run strict provenance and grounding checks over stored evidence, ideas, and the latest run.",
    )
    audit_parser.add_argument("--papers", default=12, type=int, help="How many core Gold papers to include in readiness checks.")
    audit_parser.add_argument("--skip-run", action="store_true", help="Skip latest-run ranking/session consistency checks.")

    cleanup_parser = subparsers.add_parser(
        "cleanup",
        aliases=["clean"],
        help="Dry-run or apply cleanup of non-Gold strict-evidence artifacts from the local database.",
    )
    cleanup_parser.add_argument("--apply", action="store_true", help="Apply cleanup. Without this, only prints a plan.")

    genome_parser = subparsers.add_parser(
        "genome",
        aliases=["g"],
        help="Generate an Idea Genome Card for one stored paper.",
    )
    genome_parser.add_argument("--paper-id", required=True, type=int, help="Paper ID from the local database.")
    genome_parser.add_argument("--provider", choices=PROVIDER_CHOICES, help="Temporarily override the provider.")
    genome_parser.add_argument("--model", default="", help="Temporarily override the model.")
    genome_parser.add_argument(
        "--extractive",
        action="store_true",
        help="Build the card only from stored title/abstract/quality metadata, without an LLM.",
    )

    genome_build_parser = subparsers.add_parser(
        "genome-build",
        aliases=["gb"],
        help="Batch-generate Idea Genome Cards for top-ranked papers.",
    )
    genome_build_parser.add_argument("--limit", default=5, type=int, help="Maximum number of papers to process.")
    genome_build_parser.add_argument("--provider", choices=PROVIDER_CHOICES, help="Temporarily override the provider.")
    genome_build_parser.add_argument("--model", default="", help="Temporarily override the model.")
    genome_build_parser.add_argument(
        "--all",
        action="store_true",
        help="Include papers that already have genome cards.",
    )
    genome_build_parser.add_argument(
        "--extractive",
        action="store_true",
        help="Build cards only from stored title/abstract/quality metadata, without an LLM.",
    )

    genomes_parser = subparsers.add_parser(
        "genomes",
        aliases=["gs"],
        help="List recently generated Idea Genome Cards.",
    )
    genomes_parser.add_argument("--limit", default=10, type=int, help="Maximum number of rows to print.")

    pattern_build_parser = subparsers.add_parser(
        "pattern-build",
        aliases=["pb"],
        help="Aggregate several genome cards into one reusable Pattern Card.",
    )
    pattern_build_parser.add_argument("--limit", default=5, type=int, help="How many genome cards to aggregate.")
    pattern_build_parser.add_argument("--provider", choices=PROVIDER_CHOICES, help="Temporarily override the provider.")
    pattern_build_parser.add_argument("--model", default="", help="Temporarily override the model.")
    pattern_build_parser.add_argument("--pattern-key", default="", help="Optional explicit pattern key.")
    pattern_build_parser.add_argument(
        "--extractive",
        action="store_true",
        help="Build a pattern only from stored Genome Cards, without an LLM.",
    )

    patterns_parser = subparsers.add_parser(
        "patterns",
        aliases=["ps"],
        help="List recently generated Pattern Cards.",
    )
    patterns_parser.add_argument("--limit", default=10, type=int, help="Maximum number of rows to print.")

    ideas_parser = subparsers.add_parser(
        "ideas",
        aliases=["id"],
        help="Generate first-pass candidate idea dossiers from the current evidence base.",
    )
    ideas_parser.add_argument("query_text", nargs="?", help="Short query form, for example: `ra id \"research agents\"`.")
    ideas_parser.add_argument("--query", default="", help="Research direction or question. Defaults to the latest run or idea query.")
    ideas_parser.add_argument("--ideas", default=3, type=int, help="How many candidate ideas to request.")
    ideas_parser.add_argument("--patterns", default=5, type=int, help="How many recent pattern cards to use.")
    ideas_parser.add_argument("--genomes", default=5, type=int, help="How many recent genome cards to use.")
    ideas_parser.add_argument("--provider", choices=PROVIDER_CHOICES, help="Temporarily override the provider.")
    ideas_parser.add_argument("--model", default="", help="Temporarily override the model.")
    ideas_parser.add_argument(
        "--extractive",
        action="store_true",
        help="Generate evidence-bound candidate dossiers without an LLM.",
    )

    ideate_parser = subparsers.add_parser(
        "ideate",
        aliases=["idea"],
        help="User-facing interactive-style idea entry: describe a field/direction, then generate ranked ideas and optionally open a session.",
    )
    ideate_parser.add_argument("direction", nargs="?", help="中文或英文方向描述，例如：多智能体科研自动化中的可靠评测。")
    ideate_parser.add_argument("--direction", dest="direction_option", default="", help="Direction text if not using the positional form.")
    ideate_parser.add_argument("--display-direction", default="", help=argparse.SUPPRESS)
    ideate_parser.add_argument("--frontier-query", default="", help=argparse.SUPPRESS)
    ideate_parser.add_argument("--ideas", default=5, type=int, help="How many ideas to generate.")
    ideate_parser.add_argument("--remote", choices=["openalex", "s2"], default="s2", help="Remote source for fresh frontier papers.")
    ideate_parser.add_argument("--limit", default=20, type=int, help="Maximum frontier papers to harvest before idea generation.")
    ideate_parser.add_argument("--year", type=int, help="Optional frontier year filter.")
    ideate_parser.add_argument("--session", action="store_true", help="Open a refinement session from the top-ranked idea.")
    ideate_parser.add_argument(
        "--review-loop",
        dest="review_loop",
        action="store_true",
        default=True,
        help="When --session is used, run 3-5 reviewer/refinement cycles before printing the final idea.",
    )
    ideate_parser.add_argument(
        "--no-review-loop",
        dest="review_loop",
        action="store_false",
        help="Create the session without automatic reviewer/refinement cycles.",
    )
    ideate_parser.add_argument("--review-rounds", default=4, type=int, help="Automatic reviewer/refinement rounds for ideate --session, clamped to 3-5.")
    ideate_parser.add_argument("--provider", choices=PROVIDER_CHOICES, help="Temporarily override the provider.")
    ideate_parser.add_argument("--model", default="", help="Temporarily override the model.")
    ideate_parser.add_argument("--lang", choices=["auto", "zh", "en"], default="auto", help="Preferred output language.")

    idea_list_parser = subparsers.add_parser(
        "idea-list",
        aliases=["il"],
        help="List recently generated candidate ideas.",
    )
    idea_list_parser.add_argument("--limit", default=10, type=int, help="Maximum number of rows to print.")

    idea_rank_parser = subparsers.add_parser(
        "idea-rank",
        aliases=["ir"],
        help="Rank candidate ideas with a lightweight heuristic over novelty, value, trend fit, and readiness.",
    )
    idea_rank_parser.add_argument("--limit", default=5, type=int, help="Maximum number of ranked ideas to print.")
    idea_rank_parser.add_argument("--query", default="", help="Optional query filter. Defaults to the latest idea-generation query.")
    idea_rank_parser.add_argument("--all", action="store_true", help="Rank across all stored candidate ideas.")
    idea_rank_parser.add_argument("--llm", action="store_true", help="Ask the configured LLM to re-rank the heuristic shortlist.")
    idea_rank_parser.add_argument("--provider", choices=PROVIDER_CHOICES, help="Temporarily override the provider.")
    idea_rank_parser.add_argument("--model", default="", help="Temporarily override the model.")

    idea_session_parser = subparsers.add_parser(
        "idea-session",
        aliases=["ix"],
        help="Create an idea session directly from a candidate idea.",
    )
    idea_session_parser.add_argument("--idea-id", type=int, help="Candidate idea ID.")
    idea_session_parser.add_argument("--latest", action="store_true", help="Use the most recent candidate idea.")
    idea_session_parser.add_argument(
        "--best",
        action="store_true",
        help="Use the top-ranked candidate idea from the latest idea-generation query.",
    )
    idea_session_parser.add_argument("--name", default="", help="Optional session name override.")

    critique_parser = subparsers.add_parser(
        "critique",
        aliases=["c"],
        help="Critique a user-requested idea change using pattern and genome evidence.",
    )
    critique_parser.add_argument("--idea", required=True, help="The current idea statement.")
    critique_parser.add_argument("--user", required=True, help="The user's requested adjustment or instruction.")
    critique_parser.add_argument("--provider", choices=PROVIDER_CHOICES, help="Temporarily override the provider.")
    critique_parser.add_argument("--model", default="", help="Temporarily override the model.")
    critique_parser.add_argument("--patterns", default=5, type=int, help="How many recent pattern cards to use.")
    critique_parser.add_argument("--genomes", default=5, type=int, help="How many recent genome cards to use.")

    reviewer_parser = subparsers.add_parser(
        "reviewer",
        aliases=["rv"],
        help="Manually trigger top-conference reviewer gate for an idea.",
    )
    reviewer_parser.add_argument("--idea-id", type=int, help="Candidate idea ID to review.")
    reviewer_parser.add_argument("--session-id", type=int, help="Idea session ID to review.")
    reviewer_parser.add_argument("--latest", action="store_true", help="Review the latest candidate idea.")
    reviewer_parser.add_argument("--idea", default="", help="Raw idea text if not reviewing a stored candidate.")
    reviewer_parser.add_argument("--provider", choices=PROVIDER_CHOICES, help="Temporarily override the provider.")
    reviewer_parser.add_argument("--model", default="", help="Temporarily override the model.")
    reviewer_parser.add_argument("--patterns", default=5, type=int, help="How many Pattern Cards to include.")
    reviewer_parser.add_argument("--genomes", default=5, type=int, help="How many Genome Cards to include.")

    review_loop_parser = subparsers.add_parser(
        "review-loop",
        aliases=["rl"],
        help="Run 3-5 reviewer/critique refinement cycles before finalizing the current session idea.",
    )
    review_loop_parser.add_argument("--session-id", type=int, help="Idea session ID.")
    review_loop_parser.add_argument("--latest", action="store_true", help="Use the most recent idea session.")
    review_loop_parser.add_argument("--rounds", default=4, type=int, help="Reviewer refinement rounds, clamped to 3-5.")
    review_loop_parser.add_argument("--provider", choices=PROVIDER_CHOICES, help="Temporarily override the provider.")
    review_loop_parser.add_argument("--model", default="", help="Temporarily override the model.")
    review_loop_parser.add_argument("--patterns", default=5, type=int, help="How many Pattern Cards to include.")
    review_loop_parser.add_argument("--genomes", default=5, type=int, help="How many Genome Cards to include.")

    session_parser = subparsers.add_parser(
        "session",
        aliases=["is"],
        help="Create a persistent idea session.",
    )
    session_parser.add_argument("--name", required=True, help="Session name.")
    session_parser.add_argument("--idea", required=True, help="Initial idea statement.")

    sessions_parser = subparsers.add_parser(
        "sessions",
        aliases=["iss"],
        help="List idea sessions.",
    )
    sessions_parser.add_argument("--limit", default=10, type=int, help="Maximum number of rows to print.")

    step_parser = subparsers.add_parser(
        "step",
        aliases=["st"],
        help="Advance one turn in an idea session using critique logic.",
    )
    step_parser.add_argument("--session-id", type=int, help="Idea session ID.")
    step_parser.add_argument("--latest", action="store_true", help="Use the most recent idea session.")
    step_parser.add_argument("user_text", nargs="?", help="Short user instruction form, for example: `ra st \"make it sharper\"`.")
    step_parser.add_argument("--user", default="", help="User instruction for this turn.")
    step_parser.add_argument("--provider", choices=PROVIDER_CHOICES, help="Temporarily override the provider.")
    step_parser.add_argument("--model", default="", help="Temporarily override the model.")
    step_parser.add_argument("--patterns", default=5, type=int, help="How many recent pattern cards to use.")
    step_parser.add_argument("--genomes", default=5, type=int, help="How many recent genome cards to use.")

    session_view_parser = subparsers.add_parser(
        "session-view",
        aliases=["sv"],
        help="Show one idea session and its turns.",
    )
    session_view_parser.add_argument("--session-id", type=int, help="Idea session ID.")
    session_view_parser.add_argument("--latest", action="store_true", help="Use the most recent idea session.")

    session_dossier_parser = subparsers.add_parser(
        "session-dossier",
        aliases=["sd"],
        help="Show the current-best dossier for one idea session.",
    )
    session_dossier_parser.add_argument("--session-id", type=int, help="Idea session ID.")
    session_dossier_parser.add_argument("--latest", action="store_true", help="Use the most recent idea session.")

    chat_parser = subparsers.add_parser(
        "chat",
        aliases=["interactive", "ch"],
        help="Start a multi-turn idea-session loop. Use /review to manually trigger reviewer gate.",
    )
    chat_parser.add_argument("--session-id", type=int, help="Idea session ID.")
    chat_parser.add_argument("--latest", action="store_true", help="Use the most recent idea session.")
    chat_parser.add_argument(
        "--new",
        default="",
        help="Start by running ideate on this direction, open a session, then enter chat.",
    )
    chat_parser.add_argument("--ideas", default=5, type=int, help="How many ideas to request when using --new.")
    chat_parser.add_argument("--remote", choices=["openalex", "s2"], default="s2", help="Frontier source when using --new.")
    chat_parser.add_argument("--limit", default=20, type=int, help="Maximum frontier papers to harvest when using --new.")
    chat_parser.add_argument("--year", type=int, help="Optional frontier year filter when using --new.")
    chat_parser.add_argument("--lang", choices=["auto", "zh", "en"], default="auto", help="Preferred output language when using --new.")
    chat_parser.add_argument("--provider", choices=PROVIDER_CHOICES, help="Temporarily override the provider.")
    chat_parser.add_argument("--model", default="", help="Temporarily override the model.")
    chat_parser.add_argument("--patterns", default=5, type=int, help="How many recent pattern cards to use.")
    chat_parser.add_argument("--genomes", default=5, type=int, help="How many recent genome cards to use.")

    run_parser = subparsers.add_parser(
        "run",
        aliases=["r"],
        help="Run the current minimal idea pipeline for a query.",
    )
    run_parser.add_argument("query_text", nargs="?", help="Short query form, for example: `ra r \"research agents\"`.")
    run_parser.add_argument("--query", default="", help="Research direction or question. Defaults to the latest run or idea query.")
    run_parser.add_argument("--ideas", default=5, type=int, help="How many ideas the eventual pipeline should generate.")
    run_parser.add_argument("--file", help="Optional local seed file to harvest before running.")
    run_parser.add_argument("--remote", choices=["openalex", "s2"], help="Optional remote source to harvest before running.")
    run_parser.add_argument("--venue", default="", help="Venue filter or venue search hint for remote harvest.")
    run_parser.add_argument("--year", type=int, help="Publication year filter for remote harvest.")
    run_parser.add_argument("--limit", default=20, type=int, help="Maximum number of remote or local records to use.")
    run_parser.add_argument("--provider", choices=PROVIDER_CHOICES, help="Temporarily override the provider.")
    run_parser.add_argument("--model", default="", help="Temporarily override the model.")
    run_parser.add_argument("--session", action="store_true", help="If ideas are generated, open a session from the newest one.")
    run_parser.add_argument(
        "--extractive",
        action="store_true",
        help="Run Genome, Pattern, and idea stages without an LLM, using only stored paper evidence.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(normalize_default_command(parser, raw_argv))
    if not args.command:
        parser.print_help()
        return 0

    config = load_config()

    if args.command == "init":
        return cmd_init(config.root_dir)
    if args.command == "status":
        return cmd_status(config.root_dir)
    if args.command == "gui":
        return run_gui(config.root_dir, host=args.host, port=args.port, open_browser=not args.no_open)
    if args.command in {"llm", "lm"}:
        return cmd_llm(
            config.root_dir,
            provider=args.provider or args.provider_name,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key or args.api_key_value,
            clear_model=args.clear_model,
            clear_base_url=args.clear_base_url,
            clear_api_key=args.clear_api_key,
        )
    if args.command in {"ask", "a"}:
        return cmd_ask(config.root_dir, args.prompt, args.system, args.provider, args.model)
    if args.command == "add-paper":
        return cmd_add_paper(config.root_dir, args.title, args.venue, args.year, args.abstract)
    if args.command == "papers":
        return cmd_list_papers(config.root_dir, args.limit)
    if args.command in {"harvest", "grab", "ingest", "h"}:
        return cmd_harvest(
            config.root_dir,
            file_path=args.file,
            source=args.source,
            remote=args.remote,
            query=args.query,
            venue=args.venue,
            year=args.year,
            limit=args.limit,
        )
    if args.command in {"gold-build", "gold"}:
        return cmd_gold_build(
            config.root_dir,
            remote=args.remote,
            query=args.query,
            venues=args.venues,
            from_year=args.from_year,
            to_year=args.to_year,
            per_venue_year=args.per_venue_year,
            extractive_genomes=args.extractive_genomes,
        )
    if args.command in {"score", "rank", "s"}:
        return cmd_score(config.root_dir)
    if args.command in {"top", "best", "t"}:
        return cmd_top(config.root_dir, args.limit)
    if args.command in {"trends", "tr"}:
        return cmd_trends(config.root_dir, args.years, args.top_k)
    if args.command in {"limitations", "lim"}:
        return cmd_limitations(config.root_dir, args.months, args.limit)
    if args.command in {"quality-enrich", "qe"}:
        return cmd_quality_enrich(
            config.root_dir,
            limit=args.limit,
            apply=args.apply,
            provider=args.provider,
            model=args.model,
        )
    if args.command in {"evidence", "ev"}:
        return cmd_evidence(config.root_dir, args.papers)
    if args.command in {"audit", "au"}:
        return cmd_audit(config.root_dir, args.papers, args.skip_run)
    if args.command in {"cleanup", "clean"}:
        return cmd_cleanup(config.root_dir, apply=args.apply)
    if args.command in {"genome", "g"}:
        return cmd_genome(config.root_dir, args.paper_id, args.provider, args.model, args.extractive)
    if args.command in {"genome-build", "gb"}:
        return cmd_genome_build(config.root_dir, args.limit, args.provider, args.model, args.all, args.extractive)
    if args.command in {"genomes", "gs"}:
        return cmd_list_genomes(config.root_dir, args.limit)
    if args.command in {"pattern-build", "pb"}:
        return cmd_pattern_build(config.root_dir, args.limit, args.provider, args.model, args.pattern_key, args.extractive)
    if args.command in {"patterns", "ps"}:
        return cmd_list_patterns(config.root_dir, args.limit)
    if args.command in {"ideas", "id"}:
        return cmd_ideas(
            config.root_dir,
            query=args.query or args.query_text or "",
            idea_count=args.ideas,
            pattern_limit=args.patterns,
            genome_limit=args.genomes,
            provider=args.provider,
            model=args.model,
            extractive=args.extractive,
        )
    if args.command in {"ideate", "idea"}:
        return cmd_ideate(
            config.root_dir,
            direction=args.direction_option or args.direction or "",
            display_direction=args.display_direction,
            frontier_query=args.frontier_query,
            ideas=args.ideas,
            remote=args.remote,
            limit=args.limit,
            year=args.year,
            open_session=args.session,
            auto_review_loop=args.review_loop,
            review_rounds=args.review_rounds,
            provider=args.provider,
            model=args.model,
            lang=args.lang,
        )
    if args.command in {"idea-list", "il"}:
        return cmd_list_candidate_ideas(config.root_dir, args.limit)
    if args.command in {"idea-rank", "ir"}:
        return cmd_rank_candidate_ideas(config.root_dir, args.limit, args.query, args.all, args.llm, args.provider, args.model)
    if args.command in {"idea-session", "ix"}:
        return cmd_session_from_candidate_idea(config.root_dir, args.idea_id, args.latest, args.best, args.name)
    if args.command in {"critique", "c"}:
        return cmd_critique(
            config.root_dir,
            current_idea=args.idea,
            user_instruction=args.user,
            provider=args.provider,
            model=args.model,
            pattern_limit=args.patterns,
            genome_limit=args.genomes,
        )
    if args.command in {"reviewer", "rv"}:
        return cmd_reviewer_gate(
            config.root_dir,
            idea_id=args.idea_id,
            session_id=args.session_id,
            latest=args.latest,
            idea_text=args.idea,
            provider=args.provider,
            model=args.model,
            pattern_limit=args.patterns,
            genome_limit=args.genomes,
        )
    if args.command in {"review-loop", "rl"}:
        return cmd_review_loop(
            config.root_dir,
            session_id=args.session_id,
            latest=args.latest,
            rounds=args.rounds,
            provider=args.provider,
            model=args.model,
            pattern_limit=args.patterns,
            genome_limit=args.genomes,
        )
    if args.command in {"session", "is"}:
        return cmd_session_create(config.root_dir, args.name, args.idea)
    if args.command in {"sessions", "iss"}:
        return cmd_list_sessions(config.root_dir, args.limit)
    if args.command in {"step", "st"}:
        return cmd_session_step(
            config.root_dir,
            session_id=args.session_id,
            latest=args.latest,
            user_instruction=args.user or args.user_text or "",
            provider=args.provider,
            model=args.model,
            pattern_limit=args.patterns,
            genome_limit=args.genomes,
        )
    if args.command in {"session-view", "sv"}:
        return cmd_session_view(config.root_dir, args.session_id, args.latest)
    if args.command in {"session-dossier", "sd"}:
        return cmd_session_dossier(config.root_dir, args.session_id, args.latest)
    if args.command in {"chat", "interactive", "ch"}:
        return cmd_chat(
            config.root_dir,
            session_id=args.session_id,
            latest=args.latest,
            new_direction=args.new,
            ideas=args.ideas,
            remote=args.remote,
            limit=args.limit,
            year=args.year,
            lang=args.lang,
            provider=args.provider,
            model=args.model,
            pattern_limit=args.patterns,
            genome_limit=args.genomes,
        )
    if args.command in {"run", "r"}:
        return cmd_run(
            config.root_dir,
            query=args.query or args.query_text or "",
            ideas=args.ideas,
            file_path=args.file,
            remote=args.remote,
            venue=args.venue,
            year=args.year,
            limit=args.limit,
            provider=args.provider,
            model=args.model,
            open_session=args.session,
            extractive=args.extractive,
        )

    parser.error(f"Unknown command: {args.command}")
    return 2


def cmd_init(root_dir: Path) -> int:
    config = load_config(root_dir)
    ensure_layout(config)
    scaffold_project_files(config)
    write_gitkeep(config.data_dir)
    write_gitkeep(config.output_dir)
    init_db(config.db_path)
    print(f"Initialized Research Alpha in {config.root_dir}")
    print(f"Database: {config.db_path}")
    print(f"Env example: {config.root_dir / '.env.example'}")
    print(f"Demo seeds: {config.root_dir / 'seeds' / 'demo_papers.jsonl'}")
    print("Short commands:")
    print(f"  {CLI_CMD} status")
    print(f"  {CLI_CMD} llm")
    print(f"  {CLI_CMD} ds sk-...")
    print(f"  {CLI_CMD} llm sk-...")
    print(f"  {CLI_CMD} a")
    print(f'  {CLI_CMD} "research agents"')
    print(f"  {CLI_CMD} harvest --file seeds/demo_papers.jsonl")
    print(f"  {CLI_CMD} score")
    print(f"  {CLI_CMD} top")
    print(f"  {CLI_CMD} tr")
    print(f"  {CLI_CMD} genome --paper-id 1")
    print(f"  {CLI_CMD} gb --limit 5")
    print(f"  {CLI_CMD} pb --limit 5")
    print(f'  {CLI_CMD} id "research agents"')
    print(f"  {CLI_CMD} il")
    print(f"  {CLI_CMD} ir")
    print(f"  {CLI_CMD} ix")
    print(f'  {CLI_CMD} c --idea "..." --user "..."')
    print(f'  {CLI_CMD} is --name "..." --idea "..."')
    print(f'  {CLI_CMD} st "..."')
    print(f"  {CLI_CMD} sd")
    print(f"  {CLI_CMD} chat")
    print(f'  {CLI_CMD} r "research agents" --file seeds/demo_papers.jsonl')
    print("Zero-install fallback:")
    print("  ./ra status")
    print("  ./ra a")
    return 0


def cmd_status(root_dir: Path) -> int:
    config = load_config(root_dir)
    ensure_layout(config)
    if not config.db_path.exists():
        init_db(config.db_path)

    print("Project")
    print(f"  root: {config.root_dir}")
    print(f"  db:   {config.db_path}")
    print(f"  data: {config.data_dir}")
    print(f"  out:  {config.output_dir}")

    print("LLM")
    print_llm_summary(config)
    print("Scholarly APIs")
    print(f"  openalex_api_key: {'yes' if config.openalex_api_key else 'no'}")
    print(f"  openalex_email: {'yes' if config.openalex_email else 'no'}")
    print(f"  semantic_scholar_api_key: {'yes' if config.semantic_scholar_api_key else 'no'}")

    print("Tables")
    for table, count in table_counts(config.db_path, TABLES):
        print(f"  {table}: {count}")
    print(f"  scored_papers: {count_scored_papers(config.db_path)}")
    print(f"  genome_cards: {count_idea_cards(config.db_path)}")
    print(f"  pattern_cards: {count_pattern_cards(config.db_path)}")
    return 0


def print_llm_summary(config) -> None:
    summary = provider_summary(config.llm)
    for key in ("provider", "model", "base_url", "api_key_configured"):
        print(f"  {key}: {summary[key]}")


def apply_llm_runtime_overrides(config, provider: str | None, model: str) -> None:
    if provider:
        canonical_provider = normalize_provider(provider)
        config.llm.provider = canonical_provider
        provider_key = os.environ.get(provider_api_key_env(canonical_provider), "").strip()
        generic_key = os.environ.get("RA_LLM_API_KEY", "").strip()
        config.llm.api_key = provider_key or generic_key
    if model:
        config.llm.model = model


def cmd_llm(
    root_dir: Path,
    provider: str | None,
    model: str | None,
    base_url: str | None,
    api_key: str | None,
    clear_model: bool,
    clear_base_url: bool,
    clear_api_key: bool,
) -> int:
    config = load_config(root_dir)
    canonical_provider = None
    if provider:
        candidate_provider = normalize_provider(provider)
        if candidate_provider not in {"openai", "deepseek"}:
            print(f"Unknown provider: {provider}. Use `oa`/`openai` or `ds`/`deepseek`.", file=sys.stderr)
            return 1
        canonical_provider = candidate_provider
    target_provider = canonical_provider or config.llm.provider
    provider_key = provider_api_key_env(target_provider)
    updates: Dict[str, str | None] = {}

    if canonical_provider:
        updates["RA_LLM_PROVIDER"] = canonical_provider
    if model is not None:
        updates["RA_LLM_MODEL"] = model
    if base_url is not None:
        updates["RA_LLM_BASE_URL"] = base_url
    if api_key is not None:
        updates[provider_key] = api_key
    if clear_model:
        updates["RA_LLM_MODEL"] = None
    if clear_base_url:
        updates["RA_LLM_BASE_URL"] = None
    if clear_api_key:
        updates[provider_key] = None

    if updates:
        env_path = upsert_dotenv(Path(root_dir), updates)
        config = load_config(root_dir)
        print(f"Updated LLM settings: {env_path}")
    else:
        env_path = Path(root_dir) / ".env"

    print("LLM")
    print_llm_summary(config)
    print(f"  env_file: {env_path}")
    if updates and api_key is not None:
        print(f"  api_key_target: {provider_key}")
    return 0


def cmd_ask(root_dir: Path, prompt: str | None, system_prompt: str, provider: str | None, model: str) -> int:
    config = load_config(root_dir)
    apply_llm_runtime_overrides(config, provider, model)

    client = LLMClient(config.llm)
    resolved_prompt = prompt.strip() if prompt else ""
    if not resolved_prompt:
        resolved_prompt = DEFAULT_ASK_SMOKE_PROMPT
    try:
        response = client.chat(prompt=resolved_prompt, system_prompt=system_prompt)
    except LLMError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(response.text)
    return 0


def cmd_add_paper(root_dir: Path, title: str, venue: str, year: int, abstract: str) -> int:
    config = load_config(root_dir)
    ensure_layout(config)
    init_db(config.db_path)
    if is_poster_record({"title": title, "venue": venue}):
        print("Skipped poster record; posters are not allowed in the paper library.")
        return 0
    paper_id = add_paper(config.db_path, title=title, venue=venue, year=year, abstract=abstract)
    print(f"Added paper #{paper_id}: {title}")
    return 0


def cmd_list_papers(root_dir: Path, limit: int) -> int:
    config = load_config(root_dir)
    ensure_layout(config)
    init_db(config.db_path)
    rows = list_papers(config.db_path, limit=max(1, limit))
    if not rows:
        print("No papers stored yet.")
        return 0
    for row in rows:
        award = f" award={row['award']}" if row["award"] else ""
        print(
            f"[{row['id']}] {row['year']} {row['venue']} :: {row['title']}"
            f"{award} cites={row['citation_count']} weight={row['paper_weight']:.2f}"
        )
    return 0


def cmd_harvest(
    root_dir: Path,
    file_path: str | None,
    source: str,
    remote: str | None,
    query: str,
    venue: str,
    year: int | None,
    limit: int,
) -> int:
    config = load_config(root_dir)
    ensure_layout(config)
    init_db(config.db_path)

    if remote:
        try:
            records = load_remote_records(
                remote=remote,
                query=query,
                venue=venue,
                year=year,
                limit=limit,
                config=config,
            )
        except ConnectorError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        path_label = f"{remote}:{query or venue or year or 'default'}"
        source = frontier_source_kind(remote)
    else:
        if not file_path:
            print("Use --file for local harvest or --remote for API harvest.", file=sys.stderr)
            return 1
        path = Path(file_path)
        if not path.is_absolute():
            path = config.root_dir / path
        if not path.exists():
            print(f"Seed file not found: {path}", file=sys.stderr)
            return 1
        records = load_seed_records(path)
        path_label = str(path)

    records, skipped_posters = reject_poster_records(records)
    if skipped_posters:
        print(f"Skipped {skipped_posters} poster records; posters are not allowed in the paper library.")

    for record in records:
        limitations_json = str(record.get("limitations_json", "")).strip()
        if not limitations_json:
            limitations_json = limitations_json_for_paper(record)
        add_paper(
            config.db_path,
            title=str(record.get("title", "")).strip(),
            venue=str(record.get("venue", "")).strip(),
            year=int(record.get("year", 0)),
            abstract=str(record.get("abstract", "")).strip(),
            source_kind=source,
            external_ref=str(record.get("external_ref", "")).strip(),
            award=str(record.get("award", "")).strip(),
            citation_count=int(record.get("citation_count", 0) or 0),
            influential_citation_count=int(record.get("influential_citation_count", 0) or 0),
            publication_date=str(record.get("publication_date", "")).strip(),
            limitations_json=limitations_json,
        )

    print(f"Harvested {len(records)} records from {path_label}")
    print(f"Database papers available: {len(iter_papers(config.db_path))}")
    return 0


def parse_csv_values(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def cmd_gold_build(
    root_dir: Path,
    *,
    remote: str,
    query: str,
    venues: str,
    from_year: int,
    to_year: int,
    per_venue_year: int,
    extractive_genomes: bool,
) -> int:
    config = load_config(root_dir)
    ensure_layout(config)
    init_db(config.db_path)
    venue_items = parse_csv_values(venues)
    if not venue_items:
        print("Need at least one venue in --venues.", file=sys.stderr)
        return 1
    if from_year > to_year:
        print("--from-year must be <= --to-year.", file=sys.stderr)
        return 1

    metadata_only_remote = is_metadata_only_gold_build_remote(remote)
    imported = 0
    candidate_imported = 0
    fetched_total = 0
    poster_skipped_total = 0
    openreview_non_excellent_skipped_total = 0
    unqualified_skipped_total = 0
    failures = []
    source_failures = []
    source_empty_results = []
    source_reports = []
    progress_note(f"开始扩充优秀论文库：{remote}，{from_year}-{to_year}，会议 {', '.join(venue_items)}")
    for year in range(int(from_year), int(to_year) + 1):
        for venue in venue_items:
            try:
                progress_note(f"联网检索 {year} {venue} 的 Best / Nominee / Oral 线索")
                records = load_remote_records(
                    remote=remote,
                    query=query,
                    venue=venue,
                    year=year,
                    limit=max(1, per_venue_year),
                    config=config,
                )
            except ConnectorError as exc:
                failures.append(f"{year} {venue}: {exc}")
                source_failures.append(
                    {
                        "remote": remote,
                        "venue": venue,
                        "year": int(year),
                        "error": str(exc),
                        "recommended_action": "重试该来源；如果连续失败，换官方奖项页来源或缩小年份范围。",
                    }
                )
                progress_note(f"{year} {venue} 检索失败，继续下一个来源")
                continue
            fetched_total += len(records)
            progress_note(f"{year} {venue} 抓取 {len(records)} 条，开始硬过滤 Poster")
            fetched_count = len(records)
            if not records:
                source_empty_results.append(
                    {
                        "remote": remote,
                        "venue": venue,
                        "year": int(year),
                        "reason": "source_returned_zero_records",
                        "recommended_action": "检查年份/会议名称；官方奖项页可能尚未发布，或该来源不覆盖该 venue-year。",
                    }
                )
                print(
                    f"{year} {venue}: source returned 0 records from {remote}. "
                    "No paper was written; check year/venue or wait for the official source to publish."
                )
            records, skipped_posters = reject_poster_records(records)
            poster_skipped_total += skipped_posters
            if skipped_posters:
                print(f"{year} {venue}: skipped {skipped_posters} poster records; posters are not allowed in the paper library.")
            skipped_openreview = 0
            if remote == "openreview":
                original_count = len(records)
                records = [
                    record
                    for record in records
                    if normalize_award_key(str(record.get("award", ""))) in STRICT_REMOTE_QUALITY_AWARDS
                ]
                skipped_openreview = original_count - len(records)
                openreview_non_excellent_skipped_total += skipped_openreview
                if skipped_openreview:
                    print(f"{year} {venue}: skipped {skipped_openreview} OpenReview poster/non-excellent records.")
            for record in records:
                limitations_json = str(record.get("limitations_json", "")).strip() or limitations_json_for_paper(record)
                add_paper(
                    config.db_path,
                    title=str(record.get("title", "")).strip(),
                    venue=str(record.get("venue", "")).strip() or venue,
                    year=int(record.get("year", year) or year),
                    abstract=str(record.get("abstract", "")).strip(),
                    source_kind=frontier_source_kind(remote),
                    external_ref=str(record.get("external_ref", "")).strip(),
                    award="",
                    citation_count=int(record.get("citation_count", 0) or 0),
                    influential_citation_count=int(record.get("influential_citation_count", 0) or 0),
                    publication_date=str(record.get("publication_date", "")).strip(),
                    limitations_json=limitations_json,
                )
                candidate_imported += 1
            progress_note(f"{year} {venue} 趋势区写入 {len(records)} 条，开始核验优秀论文信号")
            venue_is_core_gold = is_core_gold_source_paper({"source_kind": gold_source_kind(remote), "venue": venue})
            qualified_records = (
                qualify_remote_quality_records(records, max_items=max(1, per_venue_year))
                if venue_is_core_gold and not metadata_only_remote
                else []
            )
            skipped = max(0, len(records) - len(qualified_records))
            unqualified_skipped_total += skipped
            if metadata_only_remote and records:
                print(
                    f"{year} {venue}: metadata-only source; kept {len(records)} records as frontier evidence. "
                    "Run `ra quality-enrich --apply` after metadata verification to promote explicit Best/Nominee/Oral records."
                )
            elif not venue_is_core_gold and records:
                print(
                    f"{year} {venue}: kept {len(records)} records as auxiliary/frontier evidence; "
                    "this venue is not part of the core Gold standard set."
                )
            elif skipped:
                print(
                    f"{year} {venue}: skipped {skipped} remote records without core best/nominee/oral signal."
                )
            for record in qualified_records:
                limitations_json = str(record.get("limitations_json", "")).strip() or limitations_json_for_paper(record)
                add_paper(
                    config.db_path,
                    title=str(record.get("title", "")).strip(),
                    venue=str(record.get("venue", "")).strip() or venue,
                    year=int(record.get("year", year) or year),
                    abstract=str(record.get("abstract", "")).strip(),
                    source_kind=gold_source_kind(remote),
                    external_ref=str(record.get("external_ref", "")).strip(),
                    award=str(record.get("award", "")).strip(),
                    citation_count=int(record.get("citation_count", 0) or 0),
                    influential_citation_count=int(record.get("influential_citation_count", 0) or 0),
                    publication_date=str(record.get("publication_date", "")).strip(),
                    limitations_json=limitations_json,
                )
                imported += 1
            progress_note(f"{year} {venue} 优秀依据库写入 {len(qualified_records)} 条")
            source_reports.append(
                {
                    "remote": remote,
                    "venue": venue,
                    "year": int(year),
                    "mode": "metadata_only" if metadata_only_remote else ("core_gold" if venue_is_core_gold else "auxiliary_frontier"),
                    "fetched": fetched_count,
                    "empty_result": fetched_count == 0,
                    "poster_skipped": skipped_posters,
                    "frontier_imported": len(records),
                    "excellent_imported": len(qualified_records),
                    "openreview_non_excellent_skipped": skipped_openreview,
                    "unqualified_skipped": skipped,
                }
            )

    progress_note("重新计算优秀论文权重")
    scored = score_papers_in_db(config.db_path) if iter_papers(config.db_path) else 0
    genome_outputs = []
    if extractive_genomes:
        progress_note("为新增优秀论文构建可审计逻辑卡草稿")
        genome_outputs = run_pipeline_genome_build(
            config.root_dir,
            limit=max(1, per_venue_year * len(venue_items)),
            provider=None,
            model="",
            include_existing=False,
            extractive=True,
        )

    current_excellent = count_scored_papers(config.db_path)
    next_actions = build_gold_build_next_actions(
        remote=remote,
        metadata_only_remote=metadata_only_remote,
        candidate_imported=candidate_imported,
        excellent_imported=imported,
        fetched_total=fetched_total,
        failures=len(failures),
        empty_results=len(source_empty_results),
        unqualified_skipped=unqualified_skipped_total,
        openreview_non_excellent_skipped=openreview_non_excellent_skipped_total,
    )
    summary_payload = {
        "remote": remote,
        "venues": venue_items,
        "from_year": int(from_year),
        "to_year": int(to_year),
        "source_mode": "metadata_only" if metadata_only_remote else "strict_gold",
        "metadata_only_remote": metadata_only_remote,
        "fetched": fetched_total,
        "candidate_imported": candidate_imported,
        "excellent_imported": imported,
        "poster_skipped": poster_skipped_total,
        "openreview_non_excellent_skipped": openreview_non_excellent_skipped_total,
        "unqualified_skipped": unqualified_skipped_total,
        "failures": len(failures),
        "scored_papers": scored,
        "current_excellent_papers": current_excellent,
        "extractive_genome_drafts": len(genome_outputs),
        "source_failures": source_failures[:20],
        "source_empty_results": source_empty_results[:20],
        "source_reports": source_reports[:60],
        "quality_policy": CORE_GOLD_QUALITY_POLICY,
        "source_priority": GOLD_BUILD_SOURCE_PRIORITY,
        "next_actions": next_actions,
    }
    print(
        "Gold-build summary: "
        f"fetched={fetched_total}; "
        f"candidate_imported={candidate_imported}; "
        f"excellent_imported={imported}; "
        f"poster_skipped={poster_skipped_total}; "
        f"openreview_non_excellent_skipped={openreview_non_excellent_skipped_total}; "
        f"unqualified_skipped={unqualified_skipped_total}; "
        f"failures={len(failures)}"
    )
    print("Gold-build summary JSON: " + json.dumps(summary_payload, ensure_ascii=False, sort_keys=True))
    print(f"联网趋势论文 imported/updated {candidate_imported} records from {remote}.")
    print(f"优秀论文库 imported/updated {imported} records from {remote}.")
    print(f"Scored papers: {scored}")
    print(f"优秀论文: {current_excellent}")
    if extractive_genomes:
        print(f"Extractive Genome drafts: {len(genome_outputs)}")
    if failures:
        print("Remote failures:")
        for item in failures[:12]:
            print(f"  {item}")
    print("Next:")
    for action in next_actions:
        command = str(action.get("command", "")).strip()
        label = str(action.get("label", "")).strip()
        reason = str(action.get("reason", "")).strip()
        if command:
            print(f"  {command}  # {label or reason}")
        elif label or reason:
            print(f"  {label or reason}")
    print(f"  {CLI_CMD} evidence --papers 20")
    print(f"  {CLI_CMD} gb --limit 20")
    print(f"  {CLI_CMD} pb --limit 10")
    return 0 if imported or not failures else 1


def build_gold_build_next_actions(
    *,
    remote: str,
    metadata_only_remote: bool,
    candidate_imported: int,
    excellent_imported: int,
    fetched_total: int,
    failures: int,
    empty_results: int,
    unqualified_skipped: int,
    openreview_non_excellent_skipped: int,
) -> List[Dict[str, str]]:
    actions: List[Dict[str, str]] = []
    if failures:
        actions.append(
            {
                "label": "有来源失败",
                "reason": "部分 venue-year 没有成功返回；先重试失败源，或缩小年份范围后再跑。",
                "command": "",
            }
        )
    if empty_results:
        actions.append(
            {
                "label": "有空结果来源",
                "reason": "部分 venue-year 请求成功但没有返回任何记录；通常是年份/会议不匹配、官方页面尚未发布，或该来源暂不覆盖。",
                "command": "",
            }
        )
    if metadata_only_remote and candidate_imported:
        actions.append(
            {
                "label": "核验元数据趋势论文",
                "reason": "OpenAlex/TPAMI 这类来源先进入趋势区，必须核验 Best/Nominee/Oral 信号后才可进入核心 Gold。",
                "command": f"{CLI_CMD} quality-enrich --apply --limit 30",
            }
        )
    if excellent_imported:
        actions.append(
            {
                "label": "查看核心优秀依据库",
                "reason": "已有新记录进入生成和评分依据；建议检查逻辑卡覆盖是否足够。",
                "command": f"{CLI_CMD} evidence --papers 20",
            }
        )
    elif candidate_imported:
        actions.append(
            {
                "label": "只写入趋势区",
                "reason": "本次没有明确 Best/Nominee/Oral 证据进入核心 Gold；趋势论文不会参与评分标准。",
                "command": f"{CLI_CMD} evidence --papers 20",
            }
        )
    elif not fetched_total:
        actions.append(
            {
                "label": "没有抓取到记录",
                "reason": "检查网络、年份、会议名称，或改用官方奖项页/OpenReview Oral 来源。",
                "command": "",
            }
        )
    if unqualified_skipped or openreview_non_excellent_skipped:
        actions.append(
            {
                "label": "过滤了非核心优秀记录",
                "reason": "Poster、Spotlight、普通论文和仅高引用记录不会进入核心 Gold；这是预期的严格过滤。",
                "command": "",
            }
        )
    return actions[:5]


def is_metadata_only_gold_build_remote(remote: object) -> bool:
    return str(remote or "").strip().lower() in METADATA_ONLY_GOLD_BUILD_REMOTES


def qualify_remote_quality_records(records: List[Dict[str, object]], *, max_items: int = 5) -> List[Dict[str, object]]:
    """Keep only records with explicit core best/nominee/oral signals."""
    qualified: List[Dict[str, object]] = []
    for record in records:
        if is_poster_record(record):
            continue
        copied = dict(record)
        award_key = normalize_award_key(str(copied.get("award", "")))
        if award_key in STRICT_REMOTE_QUALITY_AWARDS:
            copied["award"] = award_key
            qualified.append(copied)
            continue
    qualified.sort(
        key=lambda item: (
            DEFAULT_AWARD_WEIGHTS.get(normalize_award_key(str(item.get("award", ""))), 0.0),
            int(item.get("citation_count", 0) or 0),
            int(item.get("influential_citation_count", 0) or 0),
        ),
        reverse=True,
    )
    return qualified[: max(1, int(max_items))]


def cmd_score(root_dir: Path) -> int:
    config = load_config(root_dir)
    ensure_layout(config)
    init_db(config.db_path)

    papers = iter_papers(config.db_path)
    if not papers:
        print("No papers to score yet. Run `ra harvest --file ...` first.", file=sys.stderr)
        return 1

    scored = score_papers_in_db(config.db_path)
    cohort_count = len({(str(row["venue"]).lower(), int(row["year"])) for row in papers})
    print(f"Scored {scored} papers across {cohort_count} venue-year cohorts.")
    return 0


def cmd_top(root_dir: Path, limit: int) -> int:
    config = load_config(root_dir)
    ensure_layout(config)
    init_db(config.db_path)
    rows = list_top_papers(config.db_path, limit=max(1, limit))
    if not rows:
        print("No ranked papers yet. Run `ra score` first.")
        return 0
    for row in rows:
        award = row["award"] or "-"
        print(
            f"[{row['id']}] score={row['paper_weight']:.2f} {row['year']} {row['venue']} "
            f"award={award} cites={row['citation_count']} :: {row['title']}"
        )
    return 0


def refresh_limitations_for_all_papers(root_dir: Path) -> int:
    config = load_config(root_dir)
    init_db(config.db_path)
    rows = list_trend_papers(config.db_path)
    updated = 0
    for row in rows:
        payload = build_limitations_payload(dict(row))
        update_paper_limitations(
            config.db_path,
            int(row["id"]),
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
        )
        updated += 1
    return updated


def build_limitations_report(root_dir: Path, *, months: int, limit: int) -> Tuple[Dict[str, object], Path, Path]:
    config = load_config(root_dir)
    ensure_layout(config)
    init_db(config.db_path)
    refresh_limitations_for_all_papers(config.root_dir)
    payload = recent_limitations_summary(
        [dict(row) for row in list_trend_papers(config.db_path)],
        months=max(1, months),
        limit=max(1, limit),
    )
    output_dir = config.output_dir / "limitations"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "recent_limitations.json"
    markdown_path = output_dir / "recent_limitations.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(render_limitations_report_markdown(payload), encoding="utf-8")
    return payload, json_path, markdown_path


def render_limitations_report_markdown(payload: Dict[str, object]) -> str:
    lines = [
        "# Recent Limitations Report",
        "",
        f"- as_of: {payload.get('as_of')}",
        f"- months: {payload.get('months')}",
        f"- cutoff_date: {payload.get('cutoff_date')}",
        f"- paper_count: {payload.get('paper_count')}",
        "",
        "## Policy",
        str(payload.get("policy", "")).strip(),
        "",
        "## Papers",
    ]
    papers = payload.get("papers", [])
    if not isinstance(papers, list) or not papers:
        lines.append("- No recent papers with stored publication dates in this window.")
        return "\n".join(lines).strip() + "\n"
    for paper in papers:
        if not isinstance(paper, dict):
            continue
        lines.append(
            f"- [{paper.get('paper_id')}] {paper.get('publication_date') or paper.get('year')} "
            f"{paper.get('venue')} weight={paper.get('paper_weight')} cites={paper.get('citation_count')}: {paper.get('title')}"
        )
        limitations = paper.get("explicit_limitations", [])
        if isinstance(limitations, list) and limitations:
            for item in limitations[:4]:
                if isinstance(item, dict):
                    lines.append(f"  - limitation: {item.get('text')}")
        elif paper.get("needs_full_text_review"):
            lines.append("  - limitation: No explicit limitation sentence in stored metadata/abstract; full text review needed.")
    return "\n".join(lines).strip() + "\n"


def cmd_limitations(root_dir: Path, months: int, limit: int) -> int:
    payload, json_path, markdown_path = build_limitations_report(root_dir, months=months, limit=limit)
    print(f"Recent limitations: {payload.get('paper_count')} papers")
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")
    return 0


QUALITY_ENRICH_SYSTEM_PROMPT = """
You are a scholarly metadata quality annotator.
You may use your pretrained bibliographic knowledge only to identify paper quality metadata:
best_paper, outstanding_paper, test_of_time, oral, spotlight, high_citation, or none.
Do not propose research ideas, methods, hypotheses, or storylines.
Do not mark Best/Outstanding/Oral as directly applicable from memory alone.
Set needs_verification=false for Best/Outstanding/Oral only when the supplied title/venue/current_award metadata itself contains an explicit award or presentation signal.
If unsure, choose none, mark confidence below 0.75, or set needs_verification=true.
Return strict JSON only.
""".strip()


def build_quality_enrich_prompt(papers: List[Dict[str, object]]) -> str:
    payload = {
        "policy": (
            "Annotate quality metadata only. These labels may affect Gold Set weighting after audit, "
            "but must not directly provide idea content or scoring standards. "
            "Set needs_verification=true when the label should be treated as a candidate note rather than applied metadata. "
            "For Best/Outstanding/Oral, do not rely on pretrained memory alone: the supplied metadata must contain an explicit award/presentation signal."
        ),
        "allowed_labels": ["best_paper", "outstanding_paper", "test_of_time", "oral", "spotlight", "high_citation", "none"],
        "papers": papers,
    }
    return (
        "For each paper, infer whether it is known as a top-quality paper or high-citation paper.\n"
        "Use only bibliographic/venue/award/citation knowledge. Do not invent paper content.\n"
        "For Best/Outstanding/Oral, mark needs_verification=true unless the supplied title, venue, or current_award field explicitly says best, outstanding, nominee, honorable mention, or oral.\n"
        "Return JSON only with this schema:\n"
        "{\n"
        '  "annotations": [\n'
        '    {"paper_id": 1, "label": "best_paper|outstanding_paper|test_of_time|oral|spotlight|high_citation|none", '
        '"confidence": 0.0, "rationale": "...", "needs_verification": true}\n'
        "  ]\n"
        "}\n\n"
        f"Context:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def parse_quality_enrich_response(text: str) -> List[Dict[str, object]]:
    payload = parse_llm_json(text, response_name="Quality enrich response")
    if not isinstance(payload, dict):
        raise ValueError("Quality enrich response must be a JSON object")
    annotations = payload.get("annotations")
    if not isinstance(annotations, list):
        raise ValueError("Quality enrich response must contain `annotations` list")
    allowed = {"best_paper", "outstanding_paper", "test_of_time", "oral", "spotlight", "high_citation", "none"}
    normalized = []
    for item in annotations:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "none")).strip().lower()
        if label not in allowed:
            label = "none"
        try:
            confidence = float(item.get("confidence", 0) or 0)
        except (TypeError, ValueError):
            confidence = 0.0
        normalized.append(
            {
                "paper_id": int(item.get("paper_id", 0) or 0),
                "label": label,
                "confidence": max(0.0, min(1.0, confidence)),
                "rationale": str(item.get("rationale", "")).strip(),
                "needs_verification": bool(item.get("needs_verification", True)),
            }
        )
    return [item for item in normalized if int(item["paper_id"]) > 0]


UNCERTAIN_QUALITY_RE = re.compile(
    r"\b(maybe|may be|possibly|probably|likely|uncertain|unsure|unknown|unverified|needs verification|not verified|guess|seems?)\b",
    re.IGNORECASE,
)


def metadata_annotation_has_explicit_quality_signal(annotation: Dict[str, object], paper: Dict[str, object] | None) -> bool:
    label = normalize_award_key(str(annotation.get("label", "")))
    if label not in PROMOTABLE_METADATA_AWARDS:
        return False
    fields = []
    if paper:
        fields.extend(
            [
                paper.get("title", ""),
                paper.get("venue", ""),
                paper.get("current_award", ""),
                paper.get("source_kind", ""),
            ]
        )
    rationale = str(annotation.get("rationale", "")).strip()
    fields.append(rationale)
    text = " ".join(str(field or "") for field in fields).strip().lower()
    if not text or UNCERTAIN_QUALITY_RE.search(text):
        return False
    if label == "oral":
        return bool(re.search(r"\boral(?:\s+presentation)?\b", text))
    if label == "best_paper":
        return bool(re.search(r"\bbest\s+(?:paper|research\s+paper)\b|\bpaper\s+award\b", text))
    if label == "outstanding_paper":
        return bool(
            re.search(
                r"\boutstanding\s+paper\b|\bbest\s+paper\s+(?:nominee|nomination|nominated|candidate)\b|"
                r"\bhonou?rable\s+mention\b|\brunner[- ]?up\b",
                text,
            )
        )
    return False


def cmd_quality_enrich(root_dir: Path, *, limit: int, apply: bool, provider: str | None, model: str) -> int:
    config = load_config(root_dir)
    ensure_layout(config)
    init_db(config.db_path)
    rows = list_trend_papers(config.db_path)[: max(1, limit)]
    papers = [
        {
            "paper_id": int(row["id"]),
            "title": str(row["title"]).strip(),
            "venue": str(row["venue"]).strip(),
            "year": int(row["year"]),
            "current_award": str(row["award"]).strip(),
            "citation_count": int(row["citation_count"] or 0),
            "influential_citation_count": int(row["influential_citation_count"] or 0),
            "source_kind": str(row["source_kind"]).strip(),
            "paper_weight": float(row["paper_weight"] or 0),
        }
        for row in rows
    ]
    source_by_paper_id = {int(row["id"]): str(row["source_kind"]).strip() for row in rows}
    if not papers:
        print("No papers available for quality enrichment.", file=sys.stderr)
        return 1
    progress_note(f"开始核验 {len(papers)} 篇趋势/待核验论文的优秀论文标签")
    apply_llm_runtime_overrides(config, provider, model)
    client = LLMClient(config.llm)
    progress_note("调用元数据专家，重点判断 Best / Nominee / Oral；Spotlight / 高引用只作辅助发现线索")
    response = client.chat(prompt=build_quality_enrich_prompt(papers), system_prompt=QUALITY_ENRICH_SYSTEM_PROMPT)
    annotations = parse_quality_enrich_response(response.text)
    progress_note(f"收到 {len(annotations)} 条质量标注，开始按置信度和可验证性过滤")
    applied = []
    skipped = []
    if apply:
        for item in annotations:
            label = str(item["label"])
            confidence = float(item["confidence"])
            if label == "none" or confidence < 0.75:
                skipped.append({**item, "skip_reason": "none_or_low_confidence"})
                continue
            if bool(item.get("needs_verification", True)):
                skipped.append({**item, "skip_reason": "needs_verification"})
                continue
            paper_id = int(item["paper_id"])
            source_kind = source_by_paper_id.get(paper_id, "")
            promoted_source = ""
            paper_row = next((paper for paper in papers if int(paper["paper_id"]) == paper_id), None)
            if label in PROMOTABLE_METADATA_AWARDS and not metadata_annotation_has_explicit_quality_signal(item, paper_row):
                skipped.append({**item, "skip_reason": "missing_explicit_quality_metadata"})
                continue
            venue_text = str(paper_row.get("venue", "") if paper_row else "")
            if (
                is_frontier_source_kind(source_kind)
                and label in PROMOTABLE_METADATA_AWARDS
                and is_core_gold_source_paper({"source_kind": gold_source_kind(source_kind[len(FRONTIER_SOURCE_PREFIX):] if source_kind.startswith(FRONTIER_SOURCE_PREFIX) else source_kind), "venue": venue_text})
            ):
                remote_name = source_kind[len(FRONTIER_SOURCE_PREFIX):] if source_kind.startswith(FRONTIER_SOURCE_PREFIX) else source_kind
                promoted_source = gold_source_kind(remote_name)
            update_paper_quality_metadata(
                config.db_path,
                paper_id,
                award=label,
                source_kind=promoted_source,
            )
            applied.append(paper_id)
        progress_note("重新计算优秀论文权重")
        score_papers_in_db(config.db_path)
    output_dir = config.output_dir / "quality"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "quality_enrichment.json"
    payload = {
        "policy": (
            "LLM used only for quality metadata enrichment. Applied labels affect paper weights only; "
            "candidate idea generation still requires explicit Genome/Pattern/Paper evidence."
        ),
        "applied": applied,
        "skipped": skipped,
        "apply_requested": bool(apply),
        "annotations": annotations,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Quality annotations: {len(annotations)}")
    print(f"Applied labels: {len(applied)}")
    print(f"Skipped labels: {len(skipped)}")
    print(f"JSON: {output_path}")
    return 0


def cmd_trends(root_dir: Path, frontier_years: int, top_k: int) -> int:
    config = load_config(root_dir)
    ensure_layout(config)
    init_db(config.db_path)

    rows = list_trend_papers(config.db_path)
    if not rows:
        print("No papers available for trend analysis. Run `ra h --file ...` first.", file=sys.stderr)
        return 1

    payload, report_path, html_path = build_trend_outputs(config.root_dir, frontier_years=max(1, frontier_years), top_k=max(1, top_k))
    print(f"Trend report: {report_path}")
    print(f"Opportunity map: {html_path}")
    print(f"Latest year: {payload.get('latest_year')}")
    print(f"Top terms: {len(payload.get('top_terms', []))}")
    return 0


def cmd_evidence(root_dir: Path, paper_limit: int) -> int:
    config = load_config(root_dir)
    ensure_layout(config)
    init_db(config.db_path)
    payload, json_path, markdown_path = build_evidence_report(config.root_dir, paper_limit=max(1, paper_limit))
    print(f"Evidence status: {payload['status']}")
    print(f"High-weight papers: {payload['counts']['high_weight_papers']}")
    print(f"Genome cards: {payload['counts']['genome_cards']}")
    print(f"Pattern cards: {payload['counts']['pattern_cards']}")
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")
    next_commands = payload.get("next_commands", [])
    if next_commands:
        print("Next:")
        for command in next_commands:
            print(f"  {command}")
    return 0


def cmd_audit(root_dir: Path, paper_limit: int, skip_run: bool) -> int:
    payload, json_path, markdown_path = build_strict_evidence_audit(
        root_dir,
        paper_limit=max(1, paper_limit),
        check_latest_run=not skip_run,
    )
    status = "passed" if payload.get("ok") else "failed"
    print(f"Strict evidence audit: {status}")
    print(f"Checks: {payload.get('check_count', 0)}")
    print(f"Failures: {len(payload.get('failures', []))}")
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")
    if payload.get("failures"):
        print("Failures:")
        for item in payload["failures"][:10]:
            print(f"  - {item.get('code')}: {item.get('message')}")
    return 0 if payload.get("ok") else 1


def cmd_cleanup(root_dir: Path, *, apply: bool) -> int:
    config = load_config(root_dir)
    ensure_layout(config)
    init_db(config.db_path)
    plan = build_cleanup_plan(config.root_dir)
    if apply:
        backup_path = apply_cleanup_plan(config.root_dir, plan)
        print("Cleanup applied.")
        print(f"Backup: {backup_path}")
    else:
        print("Cleanup dry-run. Add --apply to remove these DB rows from strict context.")
    print(f"Non-Gold genome cards: {len(plan['non_gold_genome_cards'])}")
    print(f"Non-Gold strict pattern cards: {len(plan['non_gold_pattern_cards'])}")
    print(f"Candidate ideas outside current Gold context: {len(plan['candidate_ideas_outside_gold_context'])}")
    print(f"Legacy non-core Gold paper records: {len(plan.get('legacy_non_core_gold_papers', []))}")
    if plan.get("legacy_non_core_gold_papers"):
        print("Legacy non-core Gold papers:")
        for row in plan["legacy_non_core_gold_papers"][:8]:
            print(f"  #{row['paper_id']} {row['source_kind']} -> {row['target_source_kind']} :: {row['title']}")
    if plan["non_gold_genome_cards"]:
        print("Genome cards:")
        for row in plan["non_gold_genome_cards"][:8]:
            print(f"  #{row['idea_card_id']} paper #{row['paper_id']} weight={row['paper_weight']} :: {row['title']}")
    if plan["non_gold_pattern_cards"]:
        print("Pattern cards:")
        for row in plan["non_gold_pattern_cards"][:8]:
            print(f"  #{row['pattern_card_id']} {row['pattern_key']} :: {row['reason']}")
    if plan["candidate_ideas_outside_gold_context"]:
        print("Candidate ideas:")
        for row in plan["candidate_ideas_outside_gold_context"][:8]:
            print(f"  #{row['candidate_idea_id']} {row['title']} :: {row['reason']}")
    return 0


def evidence_build_failure_reason(exc: Exception) -> str:
    text = str(exc)
    if "grounding audit failed" in text.lower():
        return "grounding_audit_failed"
    if isinstance(exc, LLMError):
        return "llm_unavailable"
    if isinstance(exc, (json.JSONDecodeError, ValueError)):
        return "invalid_model_response"
    return "evidence_build_failed"


def evidence_build_failure_stage(stage: str, exc: Exception) -> Dict[str, object]:
    return {
        "stage": stage,
        "status": "skipped",
        "reason": evidence_build_failure_reason(exc),
        "detail": str(exc),
    }


def cmd_run(
    root_dir: Path,
    query: str,
    ideas: int,
    file_path: str | None,
    remote: str | None,
    venue: str,
    year: int | None,
    limit: int,
    provider: str | None,
    model: str,
    open_session: bool,
    extractive: bool,
    auto_review_loop: bool = False,
    review_rounds: int = 4,
    frontier_query: str = "",
) -> int:
    config = load_config(root_dir)
    ensure_layout(config)
    init_db(config.db_path)
    try:
        resolved_query, query_source = resolve_run_query(config.root_dir, query)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    apply_llm_runtime_overrides(config, provider, model)
    resolved_frontier_query = frontier_query.strip() or scholarly_search_query(resolved_query)

    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_dir = config.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    stages: List[Dict[str, object]] = []
    progress_note("创建本次研究运行目录")

    if file_path or remote:
        progress_note("开始导入或联网收集论文")
        harvest_result = pipeline_harvest_into_db(
            config.root_dir,
            file_path=file_path,
            remote=remote,
            query=resolved_query,
            venue=venue,
            year=year,
            limit=limit,
        )
        stages.append(harvest_result)

    papers = iter_papers(config.db_path)
    if papers:
        progress_note("重新打分论文库，确认优秀论文权重")
        scored_count = score_papers_in_db(config.db_path)
        stages.append(
            {
                "stage": "score",
                "status": "completed",
                "paper_count": scored_count,
            }
        )
        progress_note("构建近期趋势报告")
        trend_result = run_pipeline_trend_stage(
            config.root_dir,
            frontier_years=5,
            top_k=min(10, max(5, limit)),
        )
        stages.append(trend_result)
    else:
        stages.append(
            {
                "stage": "score",
                "status": "skipped",
                "reason": "no_papers_available",
            }
        )

    if papers and (config.llm.api_key or extractive):
        progress_note("从优秀论文中抽取完整逻辑线 Genome Cards")
        try:
            genome_outputs = run_pipeline_genome_build(
                config.root_dir,
                limit=min(max(2, ideas), max(2, limit)),
                provider=provider,
                model=model,
                include_existing=False,
                extractive=extractive,
            )
        except (LLMError, ValueError, json.JSONDecodeError) as exc:
            genome_outputs = []
            stages.append(evidence_build_failure_stage("genome", exc))
            print(f"Genome build skipped: {exc}", file=sys.stderr)
        else:
            stages.append(
                {
                    "stage": "genome",
                    "status": "completed" if genome_outputs else "skipped",
                    "generated": len(genome_outputs),
                    "outputs": [str(path) for _, path in genome_outputs],
                }
            )
        if not genome_outputs:
            stages.append(
                {
                    "stage": "pattern",
                    "status": "skipped",
                    "reason": "not_enough_grounded_genomes",
                }
            )
            stages.append(
                {
                    "stage": "ideas",
                    "status": "skipped",
                    "reason": "evidence_build_not_ready",
                    "detail": "Genome grounding failed or no strict Genome cards were generated; idea generation was not attempted.",
                }
            )
        else:
            progress_note("聚合优秀论文逻辑线为可迁移 Pattern Card")
            try:
                pattern_result = run_pipeline_pattern_build(
                    config.root_dir,
                    limit=min(max(2, ideas), max(2, limit)),
                    provider=provider,
                    model=model,
                    extractive=extractive,
                )
            except (LLMError, ValueError, json.JSONDecodeError) as exc:
                pattern_result = evidence_build_failure_stage("pattern", exc)
                print(f"Pattern build skipped: {exc}", file=sys.stderr)
            stages.append(pattern_result)
            if pattern_result["status"] == "completed":
                try:
                    progress_note("基于优秀论文逻辑、近期热点和局限性生成 idea 草案")
                    idea_result = run_pipeline_idea_stage(
                        root_dir=config.root_dir,
                        query=resolved_query,
                        frontier_query=resolved_frontier_query,
                        idea_count=ideas,
                        pattern_limit=min(max(2, ideas), max(2, limit)),
                        genome_limit=min(max(2, ideas), max(2, limit)),
                        provider=provider,
                        model=model,
                        extractive=extractive,
                    )
                    stages.append(idea_result)
                except ValueError as exc:
                    repair_result: Dict[str, object] | None = None
                    if not extractive and "Current-hotspot evidence is not ready" in str(exc):
                        repair_result = auto_repair_frontier_evidence(
                            config.root_dir,
                            query=resolved_frontier_query,
                            preferred_remote="openalex",
                            limit=max(12, limit),
                            provider=provider,
                            model=model,
                        )
                        stages.append({"stage": "frontier_repair", **repair_result})
                        if int(repair_result.get("imported", 0) or 0) > 0:
                            try:
                                progress_note("frontier 证据已补充，重试严格 idea 生成")
                                idea_result = run_pipeline_idea_stage(
                                    root_dir=config.root_dir,
                                    query=resolved_query,
                                    frontier_query=resolved_frontier_query,
                                    idea_count=ideas,
                                    pattern_limit=min(max(2, ideas), max(2, limit)),
                                    genome_limit=min(max(2, ideas), max(2, limit)),
                                    provider=provider,
                                    model=model,
                                    extractive=extractive,
                                )
                                stages.append(idea_result)
                                exc = None  # type: ignore[assignment]
                            except ValueError as retry_exc:
                                exc = retry_exc
                    if exc is None:
                        pass
                    else:
                        detail = str(exc)
                        failure_summary = exc.summary if isinstance(exc, IdeaGenerationNotReady) else build_idea_generation_failure_summary(
                            query=resolved_query,
                            requested=ideas,
                            rejected_ideas=[{"title": "idea_generation", "reason": detail}],
                        )
                        if repair_result:
                            detail += (
                                "\nAuto frontier repair: "
                                f"status={repair_result.get('status')}; "
                                f"remote={repair_result.get('remote')}; "
                                f"imported={repair_result.get('imported')}; "
                                "frontier evidence remains separate from Gold scoring standards."
                            )
                        stages.append(
                            {
                                "stage": "ideas",
                                "status": "skipped",
                                "reason": "idea_generation_not_ready",
                                "detail": detail,
                                "failure_summary": failure_summary,
                            }
                        )
    elif papers:
        stages.append(
            {
                "stage": "genome",
                "status": "skipped",
                "reason": "llm_not_configured",
            }
        )
        stages.append(
            {
                "stage": "pattern",
                "status": "skipped",
                "reason": "llm_not_configured",
            }
        )
        stages.append(
            {
                "stage": "ideas",
                "status": "skipped",
                "reason": "llm_not_configured",
            }
        )

    ranking_summary: Dict[str, object] | None = None
    completed_idea_ids = [
        int(item["idea_id"])
        for stage in stages
        if stage.get("stage") == "ideas" and stage.get("status") == "completed"
        for item in stage.get("idea_summaries", [])
        if isinstance(item, dict) and str(item.get("idea_id", "")).strip()
    ]
    if completed_idea_ids:
        try:
            progress_note("对 idea 草案做证据驱动排序")
            ranking_data = build_candidate_idea_ranking(
                root_dir=config.root_dir,
                query=resolved_query,
                include_all=False,
                limit=max(1, ideas),
                idea_ids=completed_idea_ids,
            )
            ranking_summary = {
                "scope": ranking_data["scope"],
                "top_pick_id": int(ranking_data["shown_ideas"][0]["idea_id"]),
                "top_pick_title": str(ranking_data["shown_ideas"][0]["title"]),
                "ranked": len(ranking_data["shown_ideas"]),
                "report_path": str(ranking_data["report_path"]),
            }
            Path(ranking_data["report_path"]).write_text(
                json.dumps(ranking_data["report_payload"], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            stages.append(
                {
                    "stage": "ranking",
                    "status": "completed",
                    "ranked": len(ranking_data["shown_ideas"]),
                    "top_pick_id": ranking_summary["top_pick_id"],
                    "outputs": [str(ranking_data["report_path"])],
                }
            )
        except ValueError:
            pass

    completed = [stage["stage"] for stage in stages if stage["status"] == "completed"]
    skipped = [stage["stage"] for stage in stages if stage["status"] == "skipped"]
    final_status = "idea_candidates_ready" if "ideas" in completed else "partial"
    candidate_dossiers: List[Dict[str, object]] = []
    for stage in stages:
        if stage.get("stage") != "ideas":
            continue
        idea_summaries = stage.get("idea_summaries", [])
        if isinstance(idea_summaries, list):
            candidate_dossiers = [item for item in idea_summaries if isinstance(item, dict)]
            break
    session_summary: Dict[str, object] | None = None
    if open_session and "ideas" in completed:
        progress_note("将最高排序 idea 写入新对话")
        if ranking_summary:
            ranked_row = get_candidate_idea(config.db_path, int(ranking_summary["top_pick_id"]))
            if ranked_row:
                session_summary = create_session_from_candidate_row(
                    config.root_dir,
                    ranked_row,
                    f"{resolved_query[:48].strip() or 'idea-session'}",
                )
        if session_summary is None:
            session_summary = create_session_from_latest_candidate_idea(
                root_dir=config.root_dir,
                name_override=f"{resolved_query[:48].strip() or 'idea-session'}",
            )
        review_result: Dict[str, object] | None = None
        if auto_review_loop:
            review_result = run_auto_review_loop_for_session(
                config.root_dir,
                session_summary,
                rounds=review_rounds,
                provider=provider,
                model=model,
                pattern_limit=min(max(2, ideas), max(2, limit)),
                genome_limit=min(max(2, ideas), max(2, limit)),
            )
            stages.append({"stage": "review_loop", **review_result})
            session_summary["auto_review_loop"] = review_result
            session_summary = refresh_session_summary(config.root_dir, session_summary)
        if review_result is None:
            final_status = "session_ready"
        elif review_result.get("status") == "completed":
            final_status = "session_review_ready"
        else:
            final_status = "session_review_incomplete"

    completed = [stage["stage"] for stage in stages if stage["status"] == "completed"]
    skipped = [stage["stage"] for stage in stages if stage["status"] == "skipped"]
    failed = [stage["stage"] for stage in stages if stage["status"] == "failed"]

    progress_note("写入运行清单和可追溯输出")
    manifest = {
        "run_id": run_id,
        "query": resolved_query,
        "requested_ideas": ideas,
        "status": final_status,
        "completed_stages": completed,
        "skipped_stages": skipped,
        "failed_stages": failed,
        "stages": stages,
        "next_stages": infer_next_stages(stages),
        "next_commands": infer_next_commands(
            stages,
            final_status=final_status,
            has_ranking=ranking_summary is not None,
            has_session=session_summary is not None,
        ),
        "llm_provider": config.llm.provider,
        "llm_model": config.llm.resolved_model,
    }
    if session_summary:
        manifest["auto_session"] = session_summary
    if ranking_summary:
        manifest["ranking"] = ranking_summary
    if candidate_dossiers:
        manifest["candidate_dossiers"] = candidate_dossiers
    run_artifact_paths = copy_run_artifacts(
        run_dir,
        [
            output_path
            for stage in stages
            for key in ("outputs", "dossier_outputs", "markdown_outputs")
            for output_path in (
                stage.get(key, [])
                if isinstance(stage.get(key), list)
                else []
            )
        ],
    )
    if session_summary:
        session_dossier_path = str(session_summary.get("dossier_path", "")).strip()
        if session_dossier_path:
            markdown_path = str(Path(session_dossier_path).with_suffix(".md"))
            run_artifact_paths.extend(copy_run_artifacts(run_dir, [session_dossier_path, markdown_path]))
    if ranking_summary:
        run_artifact_paths.extend(copy_run_artifacts(run_dir, [str(ranking_summary.get("report_path", "")).strip()]))
    if run_artifact_paths:
        manifest["run_artifacts"] = run_artifact_paths
    report_path = run_dir / "run_report.md"
    manifest["report_path"] = str(report_path)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report_path.write_text(render_run_report(manifest), encoding="utf-8")
    run_db_id = add_run(config.db_path, query=resolved_query, status=final_status, manifest_path=manifest_path)

    if query_source != "explicit":
        print(f"Using query ({query_source}): {resolved_query}")
    print(f"Created run artifact: {run_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Report: {report_path}")
    print(f"Run record: {run_db_id}")
    print(f"Pipeline status: {final_status}")
    for stage in stages:
        if stage.get("stage") == "ideas" and stage.get("status") == "skipped" and isinstance(stage.get("failure_summary"), dict):
            print("Idea-generation failure JSON: " + json.dumps(stage["failure_summary"], ensure_ascii=False, sort_keys=True))
    if run_artifact_paths:
        print(f"Artifacts bundled under: {run_dir}")
    for stage in stages:
        line = f"  {stage['stage']}: {stage['status']}"
        if "reason" in stage:
            line += f" ({stage['reason']})"
        elif "generated" in stage:
            line += f" generated={stage['generated']}"
            if "rejected" in stage:
                line += f" rejected={stage['rejected']}"
        elif "paper_count" in stage:
            line += f" papers={stage['paper_count']}"
        elif "ranked" in stage:
            line += f" ideas={stage['ranked']}"
        elif "imported" in stage:
            line += f" imported={stage['imported']}"
        print(line)
    if session_summary:
        print(f"Auto session: #{session_summary['session_id']} {session_summary['name']}")
        if session_summary.get("auto_review_loop"):
            loop = session_summary["auto_review_loop"]
            print(
                "Auto review loop: "
                f"{loop.get('status')} rounds={loop.get('rounds_completed', 0)}/{loop.get('requested_rounds', 0)}"
            )
        print("Idea-session summary JSON: " + json.dumps(session_summary.get("chat_summary", {}), ensure_ascii=False, sort_keys=True))
        print(f"Dossier: {session_summary['dossier_path']}")
        print("Next:")
        print(f'  {CLI_CMD} st --user "..."')
        print(f"  {CLI_CMD} sd")
    elif "ideas" in completed:
        if ranking_summary:
            print(f"Top idea: #{ranking_summary['top_pick_id']} {ranking_summary['top_pick_title']}")
            print(f"Ranking: {ranking_summary['report_path']}")
        print("Next:")
        if ranking_summary:
            print(f"  {CLI_CMD} ix")
        else:
            print(f"  {CLI_CMD} ix")
    return 0


def cmd_genome(root_dir: Path, paper_id: int, provider: str | None, model: str, extractive: bool) -> int:
    config = load_config(root_dir)
    ensure_layout(config)
    init_db(config.db_path)

    paper = get_paper_by_id(config.db_path, paper_id)
    if not paper:
        print(f"Paper not found: {paper_id}", file=sys.stderr)
        return 1

    apply_llm_runtime_overrides(config, provider, model)

    try:
        _, output_path = generate_genome_for_paper(config.root_dir, int(paper["id"]), provider, model, extractive=extractive)
    except (LLMError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    row = get_paper_by_id(config.db_path, paper_id)
    mode = "extractive " if extractive else ""
    print(f"Generated {mode}genome card for paper #{paper_id}")
    print(f"Output: {output_path}")
    return 0


def cmd_list_genomes(root_dir: Path, limit: int) -> int:
    config = load_config(root_dir)
    ensure_layout(config)
    init_db(config.db_path)
    rows = list_idea_cards(config.db_path, limit=max(1, limit))
    if not rows:
        print("No genome cards yet. Run `ra genome --paper-id ...` first.")
        return 0
    for row in rows:
        print(
            f"[{row['id']}] paper={row['paper_id']} {row['year']} {row['venue']} "
            f"evidence={row['evidence_level']} :: {row['title']}"
        )
    return 0


def cmd_pattern_build(
    root_dir: Path,
    limit: int,
    provider: str | None,
    model: str,
    pattern_key: str,
    extractive: bool,
) -> int:
    config = load_config(root_dir)
    ensure_layout(config)
    init_db(config.db_path)

    cards = core_gold_rows(list_gold_genome_card_payloads(config.db_path, limit=max(2, limit)))
    if len(cards) < 2:
        print("Need at least 2 genome cards before building a pattern.", file=sys.stderr)
        return 1

    normalized_cards = []
    for row in cards:
        payload = json.loads(row["content_json"])
        payload["paper_id"] = row["paper_id"]
        payload["title"] = row["title"]
        payload["venue"] = row["venue"]
        payload["year"] = row["year"]
        normalized_cards.append(payload)

    if extractive:
        payload = build_extractive_pattern_payload(normalized_cards, explicit_key=pattern_key)
    else:
        apply_llm_runtime_overrides(config, provider, model)
        client = LLMClient(config.llm)
        prompt = build_pattern_prompt(normalized_cards)
        try:
            response = client.chat(prompt=prompt, system_prompt=PATTERN_SYSTEM_PROMPT)
            payload = parse_pattern_response(response.text)
        except (LLMError, ValueError, json.JSONDecodeError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    if pattern_key:
        payload["pattern_key"] = pattern_key
    attach_pattern_source_provenance(payload, cards)
    if not extractive:
        payload["grounding_audit"] = audit_pattern_grounding(payload, normalized_cards)
        if payload["grounding_audit"]["status"] != "valid":
            print(
                "Pattern grounding audit failed: "
                + json.dumps(payload["grounding_audit"]["failures"][:3], ensure_ascii=False),
                file=sys.stderr,
            )
            return 1

    content_json = json.dumps(payload, ensure_ascii=False, indent=2)
    card_id = upsert_pattern_card(config.db_path, payload["pattern_key"], content_json)

    pattern_dir = config.output_dir / "patterns"
    pattern_dir.mkdir(parents=True, exist_ok=True)
    output_path = pattern_dir / f"{payload['pattern_key']}.json"
    output_path.write_text(content_json, encoding="utf-8")

    mode = "extractive " if extractive else ""
    print(f"Generated {mode}pattern card #{card_id}: {payload['pattern_key']}")
    print(f"Output: {output_path}")
    return 0


def cmd_list_patterns(root_dir: Path, limit: int) -> int:
    config = load_config(root_dir)
    ensure_layout(config)
    init_db(config.db_path)
    rows = list_pattern_cards(config.db_path, limit=max(1, limit))
    if not rows:
        print("No pattern cards yet. Run `ra pb --limit ...` first.")
        return 0
    for row in rows:
        payload = json.loads(row["content_json"])
        print(f"[{row['id']}] {row['pattern_key']} :: {payload.get('pattern_name', '')}")
    return 0


def cmd_ideas(
    root_dir: Path,
    query: str,
    idea_count: int,
    pattern_limit: int,
    genome_limit: int,
    provider: str | None,
    model: str,
    extractive: bool,
) -> int:
    config = load_config(root_dir)
    ensure_layout(config)
    init_db(config.db_path)
    resolved_query, query_source = resolve_idea_query(config.root_dir, query)

    try:
        stage = run_pipeline_idea_stage(
            root_dir=config.root_dir,
            query=resolved_query,
            idea_count=idea_count,
            pattern_limit=pattern_limit,
            genome_limit=genome_limit,
            provider=provider,
            model=model,
            extractive=extractive,
        )
    except (LLMError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    generated = int(stage.get("generated", 0))
    if query_source != "explicit":
        print(f"Using query ({query_source}): {resolved_query}")
    print(f"Generated {generated} candidate ideas.")
    for idx, output_path in enumerate(stage.get("outputs", []), start=1):
        payload = json.loads(Path(output_path).read_text(encoding="utf-8"))
        print(f"  {idx}. {payload['idea_title']}")
    if generated:
        print("Next:")
        print(f"  {CLI_CMD} ir")
        print(f"  {CLI_CMD} ix")
    return 0


def language_prefixed_query(direction: str, lang: str) -> str:
    cleaned = direction.strip()
    if lang == "zh":
        return f"{cleaned}（请用中文输出idea标题、解释、风险和实验计划）"
    if lang == "en":
        return f"{cleaned} (Please return idea titles, explanations, risks, and experiment plans in English.)"
    if re.search(r"[\u4e00-\u9fff]", cleaned):
        return f"{cleaned}（请用中文输出idea标题、解释、风险和实验计划）"
    return cleaned


def has_chinese_text(value: object) -> bool:
    if isinstance(value, dict):
        return any(has_chinese_text(item) for item in value.values())
    if isinstance(value, list):
        return any(has_chinese_text(item) for item in value)
    return bool(re.search(r"[\u4e00-\u9fff]", str(value)))


def scholarly_search_query(direction: str) -> str:
    cleaned = direction.strip()
    if not re.search(r"[\u4e00-\u9fff]", cleaned):
        return cleaned
    lexicon = [
        ("科研", "scientific research"),
        ("研究", "research"),
        ("智能体", "agents"),
        ("多智能体", "multi-agent"),
        ("可靠", "reliable robustness"),
        ("评测", "evaluation benchmark"),
        ("评价", "evaluation"),
        ("基准", "benchmark"),
        ("局限", "limitations"),
        ("失败", "failure modes"),
        ("自动化", "automation"),
        ("科学发现", "scientific discovery"),
        ("工具", "tool use"),
        ("规划", "planning"),
        ("推理", "reasoning"),
    ]
    terms = [english for chinese, english in lexicon if chinese in cleaned]
    if not terms:
        terms = ["AI agents", "evaluation", "benchmark"]
    return " ".join(dedupe_keep_order(" ".join(terms).split()))


def default_frontier_year(year: int | None) -> int:
    return int(year) if year is not None else datetime.utcnow().year


def frontier_year_candidates(year: int | None) -> List[int | None]:
    if year is not None:
        return [int(year)]
    current = datetime.utcnow().year
    return [current, current - 1]


def remote_fallback_order(remote: str) -> List[str]:
    cleaned = remote.strip().lower()
    order = [cleaned] if cleaned else []
    for candidate in ("openalex", "s2"):
        if candidate not in order:
            order.append(candidate)
    return order


def harvest_frontier_with_fallback(
    root_dir: Path,
    *,
    preferred_remote: str,
    query: str,
    year: int | None,
    limit: int,
) -> Dict[str, object]:
    failures = []
    for candidate in remote_fallback_order(preferred_remote):
        year_items = frontier_year_candidates(year) if candidate == "openalex" else [year]
        for candidate_year in year_items:
            try:
                result = pipeline_harvest_into_db(
                    root_dir,
                    file_path=None,
                    remote=candidate,
                    query=query,
                    venue="",
                    year=candidate_year,
                    limit=max(1, limit),
                )
            except ConnectorError as exc:
                failures.append({"remote": candidate, "year": candidate_year, "error": str(exc)})
                continue
            result["remote"] = candidate
            result["query"] = query
            result["year"] = candidate_year
            result["fallback_failures"] = failures
            if int(result.get("imported", 0) or 0) > 0 or candidate != "openalex":
                return result
            failures.append({"remote": candidate, "year": candidate_year, "error": "no matching records imported"})
    return {
        "stage": "harvest",
        "status": "failed",
        "source": preferred_remote,
        "imported": 0,
        "query": query,
        "fallback_failures": failures,
    }


def auto_repair_frontier_evidence(
    root_dir: Path,
    *,
    query: str,
    preferred_remote: str,
    limit: int,
    provider: str | None = None,
    model: str = "",
) -> Dict[str, object]:
    config = load_config(root_dir)
    ensure_layout(config)
    init_db(config.db_path)
    cleaned_query = query.strip()
    if not cleaned_query:
        return {"status": "skipped", "reason": "empty_query"}
    progress_note(f"严格证据门缺少近期对齐证据，自动补充 frontier 趋势线索：{cleaned_query}")
    harvest_result = harvest_frontier_with_fallback(
        config.root_dir,
        preferred_remote=preferred_remote,
        query=cleaned_query,
        year=None,
        limit=max(6, int(limit)),
    )
    progress_note("重新打分并重建趋势/近半年局限性上下文")
    score_papers_in_db(config.db_path)
    build_trend_outputs(config.root_dir, frontier_years=5, top_k=max(10, int(limit)))
    build_limitations_report(config.root_dir, months=6, limit=max(8, int(limit)))
    return {
        "status": "completed" if int(harvest_result.get("imported", 0) or 0) > 0 else "no_new_records",
        "query": cleaned_query,
        "remote": harvest_result.get("remote", harvest_result.get("source", preferred_remote)),
        "imported": int(harvest_result.get("imported", 0) or 0),
        "fallback_failures": harvest_result.get("fallback_failures", []),
    }


def frontier_source_kind(remote: str) -> str:
    cleaned = remote.strip().lower()
    return f"{FRONTIER_SOURCE_PREFIX}{cleaned}" if cleaned else FRONTIER_SOURCE_PREFIX.rstrip("_")


def gold_source_kind(remote: str) -> str:
    cleaned = remote.strip().lower()
    return f"{GOLD_SOURCE_PREFIX}{cleaned}" if cleaned else GOLD_SOURCE_PREFIX.rstrip("_")


def is_frontier_source_kind(source_kind: object) -> bool:
    value = str(source_kind or "").strip().lower()
    return value.startswith(FRONTIER_SOURCE_PREFIX) or value in REMOTE_FRONTIER_SOURCES


def is_user_library_source_kind(source_kind: object) -> bool:
    return str(source_kind or "").strip().lower() == USER_LIBRARY_SOURCE_KIND


def is_poster_record(record: Dict[str, object]) -> bool:
    """Poster papers are never allowed into the evidence or candidate library."""
    fields = (
        record.get("venue", ""),
        record.get("award", ""),
        record.get("source_kind", ""),
        record.get("external_ref", ""),
        record.get("title", ""),
    )
    return any(POSTER_RE.search(str(field or "").replace("-", "_")) for field in fields)


def reject_poster_records(records: List[Dict[str, object]]) -> Tuple[List[Dict[str, object]], int]:
    kept = [record for record in records if not is_poster_record(record)]
    return kept, len(records) - len(kept)


def is_unverified_remote_gold_candidate(paper: dict) -> bool:
    source_kind = str(paper.get("source_kind", "")).strip().lower()
    if not source_kind.startswith(GOLD_SOURCE_PREFIX):
        return False
    award_key = normalize_award_key(str(paper.get("award", "")))
    return not award_key


def is_core_gold_source_paper(paper: dict) -> bool:
    source_kind = str(paper.get("source_kind", "")).strip().lower()
    if is_user_library_source_kind(source_kind):
        return False
    venue = str(paper.get("venue", "")).strip().lower()
    title = str(paper.get("title", "")).strip().lower()
    haystack = f"{venue} {source_kind} {title}"
    if source_kind in CORE_GOLD_REMOTE_SOURCES and any(token in haystack for token in CORE_GOLD_VENUE_TOKENS):
        return True
    if any(token in venue for token in CORE_GOLD_VENUE_TOKENS):
        return True
    return False


def is_core_gold_standard_paper(paper: dict) -> bool:
    """Current Gold standard: core source/venue plus explicit Best/Outstanding/Oral, never Poster."""
    if is_poster_record(paper):
        return False
    if not is_core_gold_source_paper(paper):
        return False
    award_key = normalize_award_key(str(paper.get("award", "")))
    return award_key in CORE_GOLD_STANDARD_AWARDS


def is_trend_source_kind(source_kind: object) -> bool:
    value = str(source_kind or "").strip().lower()
    if is_user_library_source_kind(value):
        return False
    return value.startswith(FRONTIER_SOURCE_PREFIX) or value in REMOTE_FRONTIER_SOURCES


def cmd_ideate(
    root_dir: Path,
    *,
    direction: str,
    display_direction: str = "",
    frontier_query: str = "",
    ideas: int,
    remote: str,
    limit: int,
    year: int | None,
    open_session: bool,
    auto_review_loop: bool,
    review_rounds: int,
    provider: str | None,
    model: str,
    lang: str,
) -> int:
    if not direction.strip():
        print("Need a direction. Example: `ra ideate \"多智能体科研自动化中的可靠评测\" --session`.", file=sys.stderr)
        return 1
    visible_direction = display_direction.strip() or direction.strip()
    query = language_prefixed_query(direction, lang)
    config = load_config(root_dir)
    ensure_layout(config)
    init_db(config.db_path)

    print(f"Direction: {visible_direction}")
    print(f"Query: {query}")
    progress_note("解析用户方向，准备近期热点、局限性和优秀论文证据")
    search_query = frontier_query.strip() or scholarly_search_query(direction)
    if remote:
        if search_query != visible_direction:
            print(f"Frontier search query: {search_query}")
        progress_note(f"开始联网检索近期热点论文：{search_query}")
        harvest_result = harvest_frontier_with_fallback(
            config.root_dir,
            preferred_remote=remote,
            query=search_query,
            year=year,
            limit=max(1, limit),
        )
        print(
            f"Frontier harvest: {harvest_result.get('status')} "
            f"remote={harvest_result.get('remote', harvest_result.get('source', remote))} "
            f"imported={harvest_result.get('imported', 0)}"
        )
        failures = harvest_result.get("fallback_failures", [])
        if isinstance(failures, list):
            for failure in failures[:3]:
                if isinstance(failure, dict):
                    print(f"  fallback note: {failure.get('remote')} failed: {failure.get('error')}")
        if harvest_result.get("status") == "failed":
            print("Remote frontier harvest failed; continuing with existing local evidence.", file=sys.stderr)
    progress_note("重新计算论文质量权重，确保 Poster 和普通趋势论文不进入评分标准")
    score_papers_in_db(config.db_path)
    progress_note("构建近期热点趋势图")
    build_trend_outputs(config.root_dir, frontier_years=5, top_k=max(10, ideas * 3))
    progress_note("抽取近半年优秀论文和趋势论文中的局限性信号")
    build_limitations_report(config.root_dir, months=6, limit=max(8, ideas * 2))
    progress_note("检查优秀论文逻辑卡、模式卡和评分依据是否足够")
    evidence_payload, _, _ = build_evidence_report(config.root_dir, paper_limit=max(12, ideas * 3))
    if evidence_payload.get("status") != "ready_for_strict_generation":
        print(
            "Strict evidence is not ready yet. Build more Gold Set Genome/Pattern evidence first:",
            file=sys.stderr,
        )
        for item in evidence_payload.get("missing_requirements", []):
            print(f"  - {item}", file=sys.stderr)
        print(f"Next: {CLI_CMD} gold-build --query \"{visible_direction}\" --extractive-genomes", file=sys.stderr)
        return 1

    progress_note("证据就绪，进入严格 idea 生成流水线")
    return cmd_run(
        config.root_dir,
        query=query,
        ideas=ideas,
        file_path=None,
        remote=None,
        venue="",
        year=None,
        limit=max(20, limit),
        provider=provider,
        model=model,
        open_session=open_session,
        auto_review_loop=auto_review_loop and open_session,
        review_rounds=review_rounds,
        extractive=False,
        frontier_query=search_query,
    )


def cmd_list_candidate_ideas(root_dir: Path, limit: int) -> int:
    config = load_config(root_dir)
    ensure_layout(config)
    init_db(config.db_path)
    rows = list_candidate_ideas(config.db_path, limit=max(1, limit))
    if not rows:
        print("No candidate ideas yet. Run `ra id --query ...` first.")
        return 0
    for row in rows:
        payload = json.loads(row["content_json"])
        print(f"[{row['id']}] {row['title']} :: {payload.get('paper_angle', '')}")
    return 0


def resolve_query_from_history(root_dir: Path, query: str) -> Tuple[str, str]:
    config = load_config(root_dir)
    ensure_layout(config)
    init_db(config.db_path)

    explicit_query = query.strip()
    if explicit_query:
        return explicit_query, "explicit"

    run_rows = list_runs(config.db_path, limit=1)
    if run_rows:
        latest_run_query = str(run_rows[0]["query"]).strip()
        if latest_run_query:
            return latest_run_query, "latest_run"

    idea_rows = list_candidate_ideas(config.db_path, limit=1)
    if idea_rows:
        latest_idea_query = str(idea_rows[0]["query"]).strip()
        if latest_idea_query:
            return latest_idea_query, "latest_idea"

    return "", "missing"


def resolve_idea_query(root_dir: Path, query: str) -> Tuple[str, str]:
    resolved_query, query_source = resolve_query_from_history(root_dir, query)
    if resolved_query:
        return resolved_query, query_source
    raise ValueError("Need a query. Use `ra id --query \"...\"` first, or create a run with `ra r --query \"...\"`.")


def resolve_run_query(root_dir: Path, query: str) -> Tuple[str, str]:
    resolved_query, query_source = resolve_query_from_history(root_dir, query)
    if resolved_query:
        return resolved_query, query_source
    raise ValueError("Need a query. Use `ra r --query \"...\"` first, or generate ideas with `ra id --query \"...\"`.")


def is_candidate_idea_payload(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    required = {"idea_title", "core_hypothesis", "historical_pattern", "trend_support", "novelty", "value"}
    return required.issubset(payload.keys())


def text_signal_strength(text: object, *, rich_words: int = 8) -> float:
    cleaned = str(text).strip()
    if not cleaned:
        return 0.0
    words = len(cleaned.split())
    if words >= rich_words:
        return 1.0
    return 0.5


def score_candidate_idea_payload(payload: Dict[str, object], known_pattern_keys: set[str]) -> Dict[str, object]:
    pattern_key = str(payload.get("historical_pattern", "")).strip()
    pattern_grounding = 2.5 if pattern_key and pattern_key in known_pattern_keys else (1.25 if pattern_key else 0.0)

    trend_grounding = min(
        2.0,
        text_signal_strength(payload.get("trend_support"), rich_words=6)
        + text_signal_strength(payload.get("frontier_gap"), rich_words=6)
        + (0.5 if str(payload.get("why_now", "")).strip() else 0.0),
    )

    novelty_value = min(
        2.5,
        text_signal_strength(payload.get("novelty"))
        + text_signal_strength(payload.get("value"))
        + (0.5 if str(payload.get("paper_angle", "")).strip() else 0.0),
    )

    experiments = payload.get("first_experiments", [])
    experiment_count = len(experiments) if isinstance(experiments, list) else 0
    execution_readiness = min(
        2.0,
        text_signal_strength(payload.get("evaluation_outline"), rich_words=6)
        + (0.5 if experiment_count >= 1 else 0.0)
        + (0.5 if experiment_count >= 2 else 0.0),
    )

    risk_awareness = text_signal_strength(payload.get("key_risk"), rich_words=6)
    prior_art_payload = payload.get("prior_art", {})
    if not isinstance(prior_art_payload, dict):
        prior_art_payload = {}
    prior_art_label = str(prior_art_payload.get("overlap_label", "")).strip()
    prior_art_distance = {
        "duplicate": -2.5,
        "near_miss": -1.25,
        "complementary": -0.35,
        "distant": 0.5,
    }.get(prior_art_label, 0.0)

    total = round(
        pattern_grounding
        + trend_grounding
        + novelty_value
        + execution_readiness
        + risk_awareness
        + prior_art_distance,
        2,
    )
    return {
        "score": total,
        "components": {
            "pattern_grounding": round(pattern_grounding, 2),
            "trend_grounding": round(trend_grounding, 2),
            "novelty_value": round(novelty_value, 2),
            "execution_readiness": round(execution_readiness, 2),
            "risk_awareness": round(risk_awareness, 2),
            "prior_art_distance": round(prior_art_distance, 2),
        },
    }


def normalize_text(value: object) -> str:
    if isinstance(value, list):
        return " ".join(normalize_text(item) for item in value)
    if isinstance(value, dict):
        return " ".join(normalize_text(item) for item in value.values())
    return str(value or "").strip()


def token_set(value: object) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z][a-z0-9\-]+", normalize_text(value).lower())
        if len(token) >= 4
    }


def compact_excerpt(value: object, limit: int = 180) -> str:
    text = " ".join(normalize_text(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def extract_terms(value: object, limit: int = 8) -> List[str]:
    tokens = [
        token
        for token in re.findall(r"[a-z][a-z0-9\-]+", normalize_text(value).lower())
        if len(token) >= 4 and token not in TERM_STOPWORDS
    ]
    counts: Dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [token for token, _ in ranked[: max(1, limit)]]


def join_terms(terms: List[str], fallback: str) -> str:
    visible = [term for term in terms if term]
    if not visible:
        return fallback
    return ", ".join(visible[:6])


def extractive_quality_note(paper: Dict[str, object]) -> str:
    quality = quality_signal_label(paper)
    weight = float(paper.get("paper_weight", 0) or 0)
    return f"{quality}, paper_weight={weight:.2f}"


def build_extractive_genome_payload(paper: Dict[str, object]) -> Dict[str, object]:
    title = str(paper.get("title", "")).strip()
    abstract = str(paper.get("abstract", "")).strip()
    evidence_text = f"{title}. {abstract}".strip()
    terms = extract_terms(evidence_text)
    term_phrase = join_terms(terms, "the paper's stated problem framing")
    excerpt = compact_excerpt(abstract or title, limit=240)
    quality_note = extractive_quality_note(paper)
    return {
        "paper_summary": (
            f"{title} is stored as high-quality evidence ({quality_note}). "
            f"Extractive abstract signal: {excerpt}"
        ),
        "pre_publication_belief": (
            "Extractive proxy from metadata/abstract only: the field context is represented by recurring terms "
            f"{term_phrase}."
        ),
        "bottleneck_or_hidden_assumption": (
            "Extracted candidate bottleneck from the paper record: progress may depend on how the work reframes "
            f"{term_phrase}."
        ),
        "problem_reframing": (
            "Treat the paper's title/abstract move as a reframing standard, not a free-form model prior: "
            f"{compact_excerpt(title, 160)}."
        ),
        "why_now": (
            f"Publication context: {paper.get('year')} {paper.get('venue')} with {quality_note}. "
            "Use this only as metadata-level why-now evidence."
        ),
        "evidence_design": (
            "Metadata/abstract-only evidence; require future full-paper reading before claiming detailed experimental design."
        ),
        "story_line": (
            "High-quality paper standard: make the abstract-level problem, move, and evidence feel connected and necessary."
        ),
        "transferable_pattern": (
            f"Transfer the paper's extractive move around {term_phrase} to a new frontier gap while preserving evidence traceability."
        ),
        "failure_boundary": (
            "Fails if a proposed idea cannot cite this paper's stored title/abstract/quality signal or needs unstored full-paper claims."
        ),
        "confidence_note": (
            "Built without an LLM from local title, abstract, award, citation, and paper_weight fields only."
        ),
        "evidence_level": EXTRACTIVE_EVIDENCE_LEVEL,
    }


GROUNDING_STOPWORDS = {
    "about",
    "abstract",
    "accepted",
    "around",
    "based",
    "candidate",
    "credible",
    "design",
    "evidence",
    "field",
    "general",
    "idea",
    "logic",
    "metadata",
    "method",
    "paper",
    "papers",
    "problem",
    "progress",
    "quality",
    "record",
    "reframe",
    "reframing",
    "research",
    "source",
    "standard",
    "study",
    "system",
    "task",
    "tasks",
    "test",
    "work",
}


def normalize_grounding_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def grounding_terms(value: object, limit: int = 40) -> List[str]:
    text = normalize_grounding_text(value)
    tokens = re.findall(r"[a-z][a-z0-9_-]{3,}", text)
    terms: List[str] = []
    for token in tokens:
        token = token.strip("_-")
        if len(token) > 4 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 4 and token.endswith("es"):
            token = token[:-2]
        elif len(token) > 4 and token.endswith("s"):
            token = token[:-1]
        if token in GROUNDING_STOPWORDS:
            continue
        if token not in terms:
            terms.append(token)
        if len(terms) >= limit:
            break
    return terms


def source_span_supported(source_text: str, evidence_corpus: str) -> bool:
    source = normalize_grounding_text(source_text)
    corpus = normalize_grounding_text(evidence_corpus)
    if not source or source == "insufficient_abstract_evidence":
        return False
    if len(source) >= 16 and source in corpus:
        return True
    source_terms = grounding_terms(source, limit=20)
    if not source_terms:
        return False
    corpus_terms = set(grounding_terms(corpus, limit=400))
    overlap = sum(1 for term in source_terms if term in corpus_terms)
    required = 2 if len(source_terms) >= 4 else 1
    return overlap >= required


def logic_text_has_evidence_terms(logic_text: str, evidence_text: str) -> bool:
    logic_terms = grounding_terms(logic_text, limit=30)
    if not logic_terms:
        return False
    evidence_terms = set(grounding_terms(evidence_text, limit=400))
    overlap = sum(1 for term in logic_terms if term in evidence_terms)
    required = 2 if len(logic_terms) >= 4 else 1
    return overlap >= required


def logic_text_is_placeholder(logic_text: str) -> bool:
    text = normalize_grounding_text(logic_text)
    if len(text) < 12:
        return True
    return text in {
        "belief",
        "bottleneck",
        "reframing",
        "why now",
        "evidence",
        "story",
        "pattern",
        "boundary",
        "source",
        "rule",
        "generalizes",
    }


def audit_genome_grounding(payload: Dict[str, object], paper: Dict[str, object]) -> Dict[str, object]:
    evidence_corpus = " ".join(
        str(paper.get(field, "")).strip()
        for field in ("title", "venue", "year", "award", "abstract")
        if str(paper.get(field, "")).strip()
    )
    logic_line = payload.get("logic_line") if isinstance(payload.get("logic_line"), dict) else {}
    evidence = payload.get("logic_line_evidence") if isinstance(payload.get("logic_line_evidence"), dict) else {}
    failures: List[Dict[str, object]] = []
    step_reports: Dict[str, object] = {}
    aggregate_logic_text = " ".join(str(logic_line.get(key, "")) for key in ["old_belief", "bottleneck", "reframing", "why_now", "evidence_design", "failure_boundary"]) if isinstance(logic_line, dict) else ""
    aggregate_logic_supported = logic_text_has_evidence_terms(aggregate_logic_text, evidence_corpus)

    for key in ["old_belief", "bottleneck", "reframing", "why_now", "evidence_design", "failure_boundary"]:
        logic_text = str(logic_line.get(key, "")).strip() if isinstance(logic_line, dict) else ""
        evidence_item = evidence.get(key, {}) if isinstance(evidence, dict) else {}
        source_text = (
            str(evidence_item.get("source_text", "")).strip()
            if isinstance(evidence_item, dict)
            else str(evidence_item).strip()
        )
        span_ok = source_span_supported(source_text, evidence_corpus)
        logic_ok = logic_text_has_evidence_terms(logic_text, f"{evidence_corpus} {source_text}")
        step_reports[key] = {
            "source_text": source_text,
            "source_span_supported": span_ok,
            "logic_terms_supported": logic_ok,
            "bounded_inference_allowed": True,
        }
        if not source_text:
            failures.append({"code": "missing_logic_line_evidence", "step": key})
        elif not span_ok:
            failures.append({"code": "unsupported_source_span", "step": key, "source_text": source_text[:160]})
        if logic_text_is_placeholder(logic_text):
            failures.append({"code": "placeholder_logic_step", "step": key, "logic_text": logic_text[:160]})

    if not aggregate_logic_supported:
        failures.append({"code": "logic_line_outside_source_logic_space"})

    confidence_note = str(payload.get("confidence_note", "")).lower()
    if "abstract" not in confidence_note and "metadata" not in confidence_note:
        failures.append({"code": "missing_abstract_only_confidence_note"})

    return {
        "status": "valid" if not failures else "invalid",
        "policy": "logic_line_must_be_grounded_in_title_abstract_metadata",
        "source_paper_id": int(paper.get("id", 0) or 0),
        "checked_steps": step_reports,
        "failures": failures,
    }


def genome_grounding_audit_is_valid(payload: Dict[str, object]) -> bool:
    audit = payload.get("grounding_audit")
    return isinstance(audit, dict) and str(audit.get("status", "")).strip() == "valid"


def audit_pattern_grounding(payload: Dict[str, object], cards: List[Dict[str, object]]) -> Dict[str, object]:
    titles = [str(card.get("title", "")).strip() for card in cards if str(card.get("title", "")).strip()]
    valid_cards = [
        card
        for card in cards
        if not isinstance(card.get("grounding_audit"), dict)
        or str(card.get("grounding_audit", {}).get("status", "")).strip() == "valid"
    ]
    card_logic_texts = []
    for card in valid_cards:
        logic_line = card.get("logic_line")
        if isinstance(logic_line, dict):
            card_logic_texts.append(" ".join(str(value) for value in logic_line.values()))
        card_logic_texts.append(
            " ".join(
                str(card.get(field, "")).strip()
                for field in ("title", "paper_summary", "problem_reframing", "transferable_pattern", "why_now", "failure_boundary")
            )
        )
    source_logic_corpus = " ".join(card_logic_texts)
    source_terms = set(grounding_terms(source_logic_corpus, limit=500))
    failures: List[Dict[str, object]] = []

    if len(valid_cards) < 2:
        failures.append({"code": "not_enough_grounded_source_cards", "valid_card_count": len(valid_cards)})

    examples = payload.get("canonical_examples", [])
    if not isinstance(examples, list) or not examples:
        failures.append({"code": "missing_canonical_examples"})
    else:
        for example in examples:
            example_text = str(example).strip()
            if example_text not in titles:
                failures.append({"code": "canonical_example_not_in_source_cards", "example": example_text})

    logic_line_pattern = payload.get("logic_line_pattern") if isinstance(payload.get("logic_line_pattern"), dict) else {}
    step_reports = {}
    aggregate_pattern_text = " ".join(
        str(logic_line_pattern.get(key, ""))
        for key in ["source_logic", "transfer_rule", "why_it_generalizes", "failure_boundary"]
    ) if isinstance(logic_line_pattern, dict) else ""
    aggregate_terms = grounding_terms(aggregate_pattern_text, limit=80)
    aggregate_overlap = sorted(term for term in aggregate_terms if term in source_terms)
    if len(aggregate_overlap) < min(4, len(aggregate_terms)):
        failures.append({"code": "pattern_outside_source_logic_space", "overlap_terms": aggregate_overlap[:12]})
    for key in ["source_logic", "transfer_rule", "why_it_generalizes", "failure_boundary"]:
        text = str(logic_line_pattern.get(key, "")).strip() if isinstance(logic_line_pattern, dict) else ""
        terms = grounding_terms(text, limit=30)
        overlap = sorted(term for term in terms if term in source_terms)
        step_reports[key] = {"overlap_terms": overlap[:12], "overlap_count": len(overlap)}
        if logic_text_is_placeholder(text):
            failures.append({"code": "placeholder_pattern_step", "step": key, "text": text[:160]})

    return {
        "status": "valid" if not failures else "invalid",
        "policy": "pattern_must_aggregate_supplied_grounded_genome_logic",
        "source_card_count": len(cards),
        "grounded_source_card_count": len(valid_cards),
        "checked_steps": step_reports,
        "failures": failures,
    }


def pattern_grounding_audit_is_valid(payload: Dict[str, object]) -> bool:
    audit = payload.get("grounding_audit")
    return isinstance(audit, dict) and str(audit.get("status", "")).strip() == "valid"


def write_genome_payload(config, paper: Dict[str, object], payload: Dict[str, object], evidence_level: str) -> tuple[int, Path]:
    content_json = json.dumps(payload, ensure_ascii=False, indent=2)
    card_id = upsert_idea_card(
        config.db_path,
        paper_id=int(paper["id"]),
        evidence_level=evidence_level,
        content_json=content_json,
    )

    genome_dir = config.output_dir / "genomes"
    genome_dir.mkdir(parents=True, exist_ok=True)
    output_path = genome_dir / f"paper-{int(paper['id'])}.json"
    output_path.write_text(content_json, encoding="utf-8")
    return card_id, output_path


def core_gold_rows(rows: Iterable[object]) -> List[object]:
    kept = []
    for row in rows:
        item = dict(row)
        if is_core_gold_standard_paper(item) and float(item.get("paper_weight", 0) or 0) > 0:
            kept.append(row)
    return kept


def slugify_key(value: str, fallback: str = "extractive-evidence-transfer") -> str:
    key = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    key = re.sub(r"-+", "-", key)
    return key[:80].strip("-") or fallback


def attach_pattern_source_provenance(payload: Dict[str, object], rows) -> None:
    source_papers = []
    for row in rows:
        row_dict = dict(row)
        source_papers.append(
            {
                "paper_id": int(row_dict["paper_id"]),
                "title": str(row_dict["title"]).strip(),
                "venue": str(row_dict["venue"]).strip(),
                "year": int(row_dict["year"]),
                "source_kind": str(row_dict.get("source_kind", "")).strip(),
                "paper_weight": float(row_dict.get("paper_weight", 0) or 0),
            }
        )
    payload["source_set_policy"] = "gold_set_only"
    payload["source_papers"] = source_papers
    payload["source_paper_ids"] = [item["paper_id"] for item in source_papers]
    payload["source_titles"] = [item["title"] for item in source_papers]


def pattern_source_provenance_is_gold_only(payload: Dict[str, object]) -> bool:
    if str(payload.get("source_set_policy", "")).strip() != "gold_set_only":
        return False
    source_papers = payload.get("source_papers")
    if not isinstance(source_papers, list) or not source_papers:
        return False
    for item in source_papers:
        if not isinstance(item, dict):
            return False
        if float(item.get("paper_weight", 0) or 0) <= 0:
            return False
    return True


def pattern_source_provenance_is_current_gold_only(db_path: Path, payload: Dict[str, object]) -> bool:
    if not pattern_source_provenance_is_gold_only(payload):
        return False
    source_ids = [
        int(item.get("paper_id"))
        for item in payload.get("source_papers") or []
        if isinstance(item, dict) and str(item.get("paper_id", "")).strip()
    ]
    if not source_ids:
        return False
    placeholders = ",".join("?" for _ in source_ids)
    with connect(db_path) as conn:
        rows = conn.execute(
            f"SELECT id, title, venue, year, source_kind, award, paper_weight FROM papers WHERE id IN ({placeholders})",
            tuple(source_ids),
        ).fetchall()
    rows_by_id = {int(row["id"]): dict(row) for row in rows}
    return all(
        is_core_gold_standard_paper(rows_by_id.get(item_id, {}))
        and float(rows_by_id.get(item_id, {}).get("paper_weight", 0) or 0) > 0
        for item_id in source_ids
    )


def row_content_json_payload(row: object) -> Dict[str, object]:
    try:
        payload = json.loads(row["content_json"])
    except (json.JSONDecodeError, KeyError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def is_strict_pattern_payload(payload: Dict[str, object]) -> bool:
    evidence_level = str(payload.get("evidence_level", "")).strip()
    return (
        not evidence_level.startswith("extractive_")
        and has_complete_pattern_logic_line(payload)
    )


def is_generation_ready_pattern_payload(payload: Dict[str, object]) -> bool:
    return is_strict_pattern_payload(payload) and pattern_grounding_audit_is_valid(payload)


def is_strict_genome_row(row: object) -> bool:
    payload = row_content_json_payload(row)
    evidence_level = str(row["evidence_level"]).strip()
    return (
        not evidence_level.startswith("extractive_")
        and has_complete_genome_logic_line(payload)
        and genome_grounding_audit_is_valid(payload)
    )


def list_strict_gold_genome_rows(db_path: Path, limit: int) -> List[object]:
    rows = core_gold_rows(
        list_gold_genome_card_payloads(
            db_path,
            limit=max(1000, count_idea_cards(db_path) + 10, int(limit)),
        )
    )
    return [row for row in rows if is_strict_genome_row(row)][: max(0, int(limit))]


def list_strict_pattern_rows(db_path: Path, limit: int) -> List[object]:
    rows = list_pattern_payloads(
        db_path,
        limit=max(1000, count_pattern_cards(db_path) + 10, int(limit)),
    )
    strict_rows = []
    for row in rows:
        payload = row_content_json_payload(row)
        if not is_generation_ready_pattern_payload(payload):
            continue
        if not pattern_source_provenance_is_current_gold_only(db_path, payload):
            continue
        strict_rows.append(row)
    return strict_rows[: max(0, int(limit))]


def pattern_payload_mentions_titles(payload: Dict[str, object], titles: Iterable[str]) -> List[str]:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True).lower()
    matches = []
    for title in titles:
        normalized = str(title).strip()
        if len(normalized) < 12:
            continue
        if normalized.lower() in text:
            matches.append(normalized)
    return matches


def backfill_pattern_source_provenance(root_dir: Path) -> Dict[str, object]:
    config = load_config(root_dir)
    ensure_layout(config)
    init_db(config.db_path)

    pattern_rows = list_pattern_payloads(config.db_path, limit=max(1000, count_pattern_cards(config.db_path) + 10))
    if not pattern_rows:
        return {"updated": 0, "skipped": 0, "reason": "no_pattern_cards"}

    gold_rows = core_gold_rows(list_gold_genome_card_payloads(config.db_path, limit=max(1000, count_idea_cards(config.db_path) + 10)))
    if not gold_rows:
        return {"updated": 0, "skipped": len(pattern_rows), "reason": "no_gold_genome_cards"}

    non_gold_titles = [
        str(row["title"]).strip()
        for row in list_trend_papers(config.db_path)
        if float(row["paper_weight"] or 0) <= 0 and str(row["title"]).strip()
    ]
    pattern_dir = config.output_dir / "patterns"
    pattern_dir.mkdir(parents=True, exist_ok=True)

    updated = 0
    skipped = 0
    skipped_details = []
    for row in pattern_rows:
        try:
            payload = json.loads(row["content_json"])
        except (json.JSONDecodeError, TypeError):
            skipped += 1
            skipped_details.append({"pattern_key": str(row["pattern_key"]).strip(), "reason": "invalid_json"})
            continue
        if not is_strict_pattern_payload(payload):
            skipped += 1
            skipped_details.append({"pattern_key": str(row["pattern_key"]).strip(), "reason": "not_strict_logic_line_pattern"})
            continue
        if pattern_source_provenance_is_current_gold_only(config.db_path, payload):
            continue
        mentioned_non_gold = pattern_payload_mentions_titles(payload, non_gold_titles)
        if mentioned_non_gold:
            skipped += 1
            skipped_details.append(
                {
                    "pattern_key": str(row["pattern_key"]).strip(),
                    "reason": "mentions_non_gold_paper_title",
                    "titles": mentioned_non_gold[:5],
                }
            )
            continue
        attach_pattern_source_provenance(payload, gold_rows)
        content_json = json.dumps(payload, ensure_ascii=False, indent=2)
        upsert_pattern_card(config.db_path, str(row["pattern_key"]).strip(), content_json)
        (pattern_dir / f"{str(row['pattern_key']).strip()}.json").write_text(content_json, encoding="utf-8")
        updated += 1
    return {
        "updated": updated,
        "skipped": skipped,
        "skipped_details": skipped_details[:20],
        "source_policy": "gold_set_only",
    }


def build_extractive_pattern_payload(cards: List[Dict[str, object]], explicit_key: str = "") -> Dict[str, object]:
    evidence_text = " ".join(
        " ".join(
            str(card.get(field, "")).strip()
            for field in ("title", "problem_reframing", "transferable_pattern", "bottleneck_or_hidden_assumption")
        )
        for card in cards
    )
    terms = extract_terms(evidence_text, limit=10)
    term_phrase = join_terms(terms, "stored core Gold paper moves")
    examples = [
        str(card.get("title", "")).strip()
        for card in cards
        if str(card.get("title", "")).strip()
    ][:5]
    key_seed = explicit_key or "-".join(terms[:3])
    pattern_key = slugify_key(key_seed, fallback="extractive-high-quality-transfer")
    return {
        "pattern_key": pattern_key,
        "pattern_name": "Extractive High-Quality Evidence Transfer",
        "core_move": (
            "Transfer a recurring abstract-level move from stored core Gold papers rather than inventing a generic idea: "
            f"{term_phrase}."
        ),
        "when_to_use": (
            "Use when the query has a frontier gap that can be tied to the same terms and problem framing found in the stored papers."
        ),
        "why_it_works": (
            "The pattern is grounded only in core Gold papers with explicit Best/Outstanding/Oral signals, so the proposal inherits "
            "a documented top-paper standard for problem choice and story logic. Spotlight, high-citation-only, Poster, and auxiliary papers are excluded."
        ),
        "failure_modes": (
            "Weak if the candidate cannot point to exact Genome, Pattern, or paper source IDs, or if it needs details beyond stored abstracts."
        ),
        "canonical_examples": examples,
        "operator_template": (
            f"Pick a frontier gap, cite exact stored evidence around {term_phrase}, transfer the abstract-level move, "
            "then state the first experiment that would test whether the transfer holds."
        ),
        "evidence_level": EXTRACTIVE_PATTERN_EVIDENCE_LEVEL,
        "generation_method": "extractive_from_stored_genome_cards",
    }


def extract_trend_terms(trend_signal: Dict[str, object]) -> List[str]:
    terms: List[str] = []
    raw_terms = trend_signal.get("top_terms", [])
    if isinstance(raw_terms, list):
        for item in raw_terms:
            if isinstance(item, dict):
                term = str(item.get("term", "")).strip()
            else:
                term = str(item).strip()
            if term:
                terms.append(term)
    raw_clusters = trend_signal.get("trend_clusters", [])
    if isinstance(raw_clusters, list):
        for cluster in raw_clusters:
            if not isinstance(cluster, dict):
                continue
            label = str(cluster.get("cluster_label", "")).strip()
            if label:
                terms.append(label)
            cluster_terms = cluster.get("terms", [])
            if isinstance(cluster_terms, list):
                terms.extend(str(term).strip() for term in cluster_terms if str(term).strip())
            examples = cluster.get("example_papers", [])
            if isinstance(examples, list):
                for paper in examples:
                    if isinstance(paper, dict):
                        title = str(paper.get("title", "")).strip()
                        if title:
                            terms.extend(extract_terms(title, limit=5))
    if not terms:
        recent = trend_signal.get("recent_papers", [])
        if isinstance(recent, list):
            terms.extend(extract_terms(" ".join(str(item) for item in recent), limit=5))
    return dedupe_keep_order(terms)[:50]


def query_frontier_tokens(query: str) -> set[str]:
    return {
        normalize_frontier_token(token)
        for token in token_set(query)
        if (token not in TERM_STOPWORDS or token in {"agent", "agents", "benchmark", "benchmarks"})
        and token not in {"research", "paper", "papers", "study", "studies"}
    }


def normalize_frontier_token(token: str) -> str:
    value = token.strip().lower()
    if len(value) > 4 and value.endswith("ies"):
        return value[:-3] + "y"
    if len(value) > 4 and value.endswith("s"):
        return value[:-1]
    return value


def normalized_token_set(value: object) -> set[str]:
    return {normalize_frontier_token(token) for token in token_set(value)}


def validate_query_frontier_evidence(query: str, trend_signal: Dict[str, object], paper_rows: List[Dict[str, object]]) -> Dict[str, object]:
    query_tokens = query_frontier_tokens(query)
    if not query_tokens:
        return {
            "ready": False,
            "query_tokens": [],
            "matched_tokens": [],
            "matched_papers": [],
            "message": "The query is too generic to verify against frontier evidence.",
        }

    trend_terms = set()
    for term in extract_trend_terms(trend_signal):
        trend_terms.update(normalized_token_set(term))
    matched_trend_tokens = set(query_tokens & trend_terms)
    matched_tokens = set(matched_trend_tokens)
    matched_papers = []
    latest_year = trend_signal.get("latest_year")
    cutoff = None
    if isinstance(latest_year, int):
        cutoff = latest_year - 4
    matched_paper_tokens: set[str] = set()
    for paper in paper_rows:
        if cutoff is not None and int(paper.get("year", 0) or 0) < cutoff:
            continue
        paper_tokens = normalized_token_set(f"{paper.get('title', '')} {paper.get('abstract', '')}")
        overlap = query_tokens & paper_tokens
        if overlap:
            matched_paper_tokens.update(overlap)
            matched_tokens.update(overlap)
            matched_papers.append(
                {
                    "title": str(paper.get("title", "")).strip(),
                    "year": int(paper.get("year", 0) or 0),
                    "matched_tokens": sorted(overlap),
                }
            )
    ready = bool(matched_trend_tokens and matched_paper_tokens)
    return {
        "ready": ready,
        "query_tokens": sorted(query_tokens),
        "matched_tokens": sorted(matched_tokens),
        "matched_trend_tokens": sorted(matched_trend_tokens),
        "matched_paper_tokens": sorted(matched_paper_tokens),
        "matched_papers": matched_papers[:5],
        "message": (
            "Frontier evidence is query-aligned."
            if ready
            else "Recent papers and top trend terms are not both aligned with the non-generic query tokens."
        ),
    }


def require_query_frontier_evidence(query: str, trend_signal: Dict[str, object], paper_rows: List[Dict[str, object]]) -> Dict[str, object]:
    audit = validate_query_frontier_evidence(query, trend_signal, paper_rows)
    if audit.get("ready"):
        return audit
    tokens = ", ".join(audit.get("query_tokens", [])) or query
    raise ValueError(
        "Current-hotspot evidence is not ready for this query. "
        f"Need recent stored papers and top trend terms matching: {tokens}. "
        "Harvest query-specific frontier papers first, for example "
        f"`ra h --remote s2 --query \"{query}\" --year 2025 --limit 20`, then rerun trends/evidence."
    )


def make_storyline_idea_title(
    query: str,
    storyline_trace: Dict[str, Dict[str, str]],
    trend_terms: List[str],
    idx: int,
    source_title: str = "",
) -> str:
    reframing_text = storyline_trace.get("reframing", {}).get("transfer_to_current_hotspot", "")
    bottleneck_text = storyline_trace.get("bottleneck", {}).get("transfer_to_current_hotspot", "")
    terms = extract_terms(f"{source_title} {reframing_text} {bottleneck_text} {' '.join(trend_terms)}", limit=6)
    if terms:
        start = idx % max(1, min(len(terms), 4))
        picked = (terms[start:] + terms[:start])[:2]
        concept = " ".join(term.title() for term in picked)
    elif trend_terms:
        concept = str(trend_terms[idx % len(trend_terms)]).title()
    else:
        concept = "Storyline"
    query_title = " ".join(part.capitalize() for part in query.split()[:4]) or "Research Direction"
    return f"{concept} Reframing for {query_title}"


def build_extractive_candidate_ideas(
    *,
    query: str,
    idea_count: int,
    pattern_rows,
    genome_rows,
    paper_rows,
    trend_signal: Dict[str, object],
) -> List[Dict[str, object]]:
    patterns = []
    for row in pattern_rows or []:
        payload = json.loads(row["content_json"])
        payload["pattern_key"] = str(row["pattern_key"]).strip()
        patterns.append(payload)
    genomes = []
    for row in genome_rows or []:
        payload = json.loads(row["content_json"])
        payload["paper_id"] = int(row["paper_id"])
        payload["title"] = str(row["title"]).strip()
        payload["venue"] = str(row["venue"]).strip()
        payload["year"] = int(row["year"])
        genomes.append(payload)
    papers = [dict(row) for row in paper_rows or []]
    if not patterns or not genomes or not papers:
        raise ValueError("Extractive idea generation needs at least one pattern, one genome, and one high-weight paper.")

    trend_terms = extract_trend_terms(trend_signal)
    trend_phrase = ", ".join(trend_terms) if trend_terms else query
    ideas: List[Dict[str, object]] = []
    max_count = min(max(1, idea_count), max(len(genomes), len(papers), 1) * max(len(patterns), 1))
    seen_titles: set[str] = set()
    for idx in range(max_count):
        pattern = patterns[idx % len(patterns)]
        genome = genomes[idx % len(genomes)]
        paper = papers[idx % len(papers)]
        pattern_key = str(pattern.get("pattern_key", "")).strip()
        genome_title = str(genome.get("title", "")).strip()
        paper_title = str(paper.get("title", "")).strip()
        query_phrase = query.strip() or "the target research direction"
        core_move = compact_excerpt(pattern.get("core_move") or pattern.get("operator_template") or pattern_key, 180)
        genome_move = compact_excerpt(genome.get("transferable_pattern") or genome.get("problem_reframing") or genome_title, 180)
        old_belief = compact_excerpt(genome.get("pre_publication_belief") or paper.get("abstract") or paper_title, 180)
        bottleneck = compact_excerpt(
            genome.get("bottleneck_or_hidden_assumption") or pattern.get("when_to_use") or core_move,
            180,
        )
        reframing = compact_excerpt(genome.get("problem_reframing") or core_move, 180)
        why_now_standard = compact_excerpt(genome.get("why_now") or pattern.get("why_it_works") or paper.get("score_notes"), 180)
        evidence_standard = compact_excerpt(genome.get("evidence_design") or "abstract-level evidence design", 180)
        failure_standard = compact_excerpt(genome.get("failure_boundary") or pattern.get("failure_modes") or "unstated boundary", 180)
        hotspot_gap = trend_phrase or query_phrase
        storyline_trace = {
            "old_belief": {
                "source_id": genome_title,
                "paper_story_standard": old_belief,
                "transfer_to_current_hotspot": (
                    f"Start from the current assumption in {query_phrase}: progress is being read through {hotspot_gap} signals."
                ),
            },
            "bottleneck": {
                "source_id": genome_title,
                "paper_story_standard": bottleneck,
                "transfer_to_current_hotspot": (
                    f"Ask what bottleneck this assumption hides in {query_phrase}, rather than adding another capability variant."
                ),
            },
            "reframing": {
                "source_id": pattern_key,
                "paper_story_standard": reframing,
                "transfer_to_current_hotspot": (
                    f"Reframe the project around `{core_move}` and make the frontier gap in {hotspot_gap} measurable."
                ),
            },
            "why_now": {
                "source_id": genome_title,
                "paper_story_standard": why_now_standard,
                "transfer_to_current_hotspot": (
                    f"Use the current trend evidence around {hotspot_gap} as the why-now signal, then validate it with fresh literature."
                ),
            },
            "evidence_design": {
                "source_id": genome_title,
                "paper_story_standard": evidence_standard,
                "transfer_to_current_hotspot": (
                    f"Design a first experiment that would make the reframing falsifiable for {query_phrase}, not just plausible."
                ),
            },
            "failure_boundary": {
                "source_id": pattern_key,
                "paper_story_standard": failure_standard,
                "transfer_to_current_hotspot": (
                    "State where the transfer fails before scaling: weak prior-art separation, no measurable gap, or no two-week signal."
                ),
            },
        }
        idea_title = make_storyline_idea_title(query_phrase, storyline_trace, trend_terms, idx, source_title=paper_title or genome_title)
        if idea_title.lower() in seen_titles:
            source_terms = extract_terms(f"{paper_title} {genome_title}", limit=3)
            suffix = " ".join(term.title() for term in source_terms[:2]) or f"Evidence {idx + 1}"
            idea_title = f"{suffix} Storyline for {' '.join(part.capitalize() for part in query_phrase.split()[:4])}"
        seen_titles.add(idea_title.lower())
        evidence_basis = [
            {
                "source_type": "pattern",
                "source_id": pattern_key,
                "quality_signal": str(pattern.get("evidence_level", EXTRACTIVE_PATTERN_EVIDENCE_LEVEL)).strip(),
                "borrowed_standard": core_move,
                "used_for": "innovation|logic|value",
            },
            {
                "source_type": "genome",
                "source_id": genome_title,
                "quality_signal": str(genome.get("evidence_level", EXTRACTIVE_EVIDENCE_LEVEL)).strip(),
                "borrowed_standard": genome_move,
                "used_for": "logic|feasibility|defensibility",
            },
            {
                "source_type": "paper",
                "source_id": paper_title or str(paper.get("id", "")).strip(),
                "quality_signal": quality_signal_label(paper),
                "borrowed_standard": compact_excerpt(paper.get("score_notes") or paper_title, 180),
                "used_for": "innovation|value|defensibility",
            },
        ]
        ideas.append(
            {
                "idea_title": idea_title,
                "core_hypothesis": (
                    f"If {query_phrase} is currently framed by {hotspot_gap}, then the stronger top-conference move is to "
                    f"migrate the story arc from {genome_title}: expose the hidden bottleneck, reframe it via `{pattern_key}`, "
                    "and test the new framing with a falsifiable first experiment."
                ),
                "historical_pattern": pattern_key,
                "trend_support": (
                    f"Trend terms from the local corpus emphasize {trend_phrase}; this is used only as stored-corpus frontier evidence."
                ),
                "frontier_gap": (
                    f"The local frontier signal around {hotspot_gap} lacks the full story arc extracted from {genome_title}: "
                    "old belief, bottleneck, reframing, why-now, evidence design, and failure boundary."
                ),
                "why_now": (
                    storyline_trace["why_now"]["transfer_to_current_hotspot"]
                ),
                "novelty": (
                    f"Novelty comes from migrating a complete top-paper storyline, not merely combining `{pattern_key}` with {query_phrase}."
                ),
                "value": (
                    "Value should be judged by whether the migrated storyline creates a sharper problem definition, a stronger evidence arc, "
                    "and a more defensible top-conference narrative than the current local frontier."
                ),
                "key_risk": (
                    "This is still abstract-only story migration and may collapse after full prior-art search or full-paper reading."
                ),
                "first_experiments": [
                    f"Read the full text of {paper_title} and {genome_title} to verify the extracted standard.",
                    f"Build a two-week probe for the reframing step: {storyline_trace['reframing']['transfer_to_current_hotspot']}",
                    f"Test the failure boundary explicitly: {storyline_trace['failure_boundary']['transfer_to_current_hotspot']}",
                    "Run remote prior-art search before treating this as a submission-grade idea.",
                ],
                "evaluation_outline": (
                    "Evaluate whether the migrated story changes the problem definition and evidence outcome, not merely whether a new "
                    "method variant scores higher."
                ),
                "paper_angle": (
                    f"The paper angle is a story migration from {paper_title}/{genome_title}: the current field belief is incomplete, "
                    f"the hidden bottleneck can be exposed through `{pattern_key}`, and the resulting claim is testable now."
                ),
                "storyline_trace": storyline_trace,
                "evidence_basis": evidence_basis,
                "generation_guardrail": (
                    "This idea was generated extractively from stored core Gold paper, Genome, Pattern, and trend evidence only; "
                    "no LLM prior was used."
                ),
                "generation_method": "extractive_from_stored_evidence",
            }
        )
    return ideas


def quality_signal_label(source: Dict[str, object]) -> str:
    if str(source.get("source_type", "")).strip() == "pattern":
        logic_line = source.get("logic_line_pattern")
        if isinstance(logic_line, dict) and has_complete_pattern_logic_line(source):
            return "logic_line_aggregation"
    award = str(source.get("award", "")).strip()
    if award:
        return award
    weight = float(source.get("paper_weight", 0) or 0)
    citation_count = int(source.get("citation_count", 0) or 0)
    influential = int(source.get("influential_citation_count", 0) or 0)
    if weight > 0:
        return f"paper_weight:{weight:.2f}"
    if citation_count or influential:
        return f"high_citation:cites={citation_count}, influential={influential}"
    return str(source.get("evidence_level", "")).strip() or "stored_evidence"


def source_identity(source: Dict[str, object]) -> str:
    for key in ("pattern_key", "title", "paper_id", "id"):
        value = str(source.get(key, "")).strip()
        if value:
            return value
    return "unknown_source"


def source_aliases(source: Dict[str, object]) -> set[str]:
    aliases = {
        source_identity(source),
        str(source.get("pattern_key", "")).strip(),
        str(source.get("pattern_name", "")).strip(),
        str(source.get("title", "")).strip(),
        str(source.get("paper_id", "")).strip(),
        str(source.get("id", "")).strip(),
    }
    paper_id = str(source.get("paper_id", "")).strip() or str(source.get("id", "")).strip()
    if paper_id:
        aliases.add(f"paper-{paper_id}")
        aliases.add(f"paper #{paper_id}")
    canonical_examples = source.get("canonical_examples", [])
    if isinstance(canonical_examples, list):
        aliases.update(str(item).strip() for item in canonical_examples if str(item).strip())
    return {alias.lower() for alias in aliases if alias}


def flatten_rubric_sources(
    *,
    pattern_rows=None,
    genome_rows=None,
    paper_rows=None,
) -> List[Dict[str, object]]:
    sources: List[Dict[str, object]] = []
    for row in pattern_rows or []:
        try:
            payload = json.loads(row["content_json"])
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        payload["source_type"] = "pattern"
        payload["pattern_key"] = str(row["pattern_key"]).strip()
        payload.setdefault("quality_signal", "pattern_from_core_gold_genomes")
        sources.append(payload)
    for row in genome_rows or []:
        try:
            payload = json.loads(row["content_json"])
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        payload["source_type"] = "genome"
        payload["paper_id"] = int(row["paper_id"])
        payload["title"] = str(row["title"]).strip()
        payload["venue"] = str(row["venue"]).strip()
        payload["year"] = int(row["year"])
        payload.setdefault("quality_signal", str(row["evidence_level"]).strip())
        sources.append(payload)
    for row in paper_rows or []:
        payload = dict(row)
        payload["source_type"] = "paper"
        sources.append(payload)
    return sources


def build_quality_evidence_summary(root_dir: Path, *, pattern_limit: int, genome_limit: int, paper_limit: int = 12) -> Dict[str, object]:
    config = load_config(root_dir)
    init_db(config.db_path)
    backfill_pattern_source_provenance(config.root_dir)
    top_rows = core_gold_rows(list_top_papers(config.db_path, limit=max(1, paper_limit)))
    genome_rows = list_strict_gold_genome_rows(config.db_path, limit=max(1, genome_limit))
    pattern_rows = list_strict_pattern_rows(config.db_path, limit=max(1, pattern_limit))
    return {
        "scoring_principle": (
            "Ideas must be generated and judged by transfer from stored core Gold evidence only: "
            "core CVPR/ICCV/ICLR/NeurIPS/ICML/TPAMI papers with explicit Best/Outstanding/Oral signals, "
            "their Idea Genome Cards, and Pattern Cards derived from those cards. "
            "Spotlight, high-citation-only, Poster, auxiliary-venue, and ordinary papers are trend context only."
        ),
        "rubric_dimensions": [
            {
                "key": item["key"],
                "label": item["label"],
                "weight": item["weight"],
                "derived_from": {
                    "pattern_fields": item["pattern_fields"],
                    "genome_fields": item["genome_fields"],
                    "paper_fields": item["paper_fields"],
                },
            }
            for item in EVIDENCE_RUBRIC_DIMENSIONS
        ],
        "high_weight_papers": [
            {
                "paper_id": int(row["id"]),
                "title": str(row["title"]).strip(),
                "venue": str(row["venue"]).strip(),
                "year": int(row["year"]),
                "award": str(row["award"]).strip(),
                "citation_count": int(row["citation_count"]),
                "influential_citation_count": int(row["influential_citation_count"]),
                "paper_weight": float(row["paper_weight"]),
                "quality_signal": quality_signal_label(dict(row)),
            }
            for row in top_rows
            if is_core_gold_standard_paper(dict(row)) and float(row["paper_weight"] or 0) > 0
        ],
        "pattern_standards": [
            {
                "pattern_key": str(row["pattern_key"]).strip(),
                "content": json.loads(row["content_json"]),
            }
            for row in pattern_rows
        ],
        "genome_standards": [
            {
                "paper_id": int(row["paper_id"]),
                "title": str(row["title"]).strip(),
                "venue": str(row["venue"]).strip(),
                "year": int(row["year"]),
                "content": json.loads(row["content_json"]),
            }
            for row in genome_rows
        ],
    }


def build_user_domain_knowledge_summary(root_dir: Path, *, limit: int = 12) -> Dict[str, object]:
    config = load_config(root_dir)
    init_db(config.db_path)
    rows = list_user_library_papers(config.db_path, limit=max(1, limit))
    papers = []
    for row in rows:
        papers.append(
            {
                "paper_id": int(row["id"]),
                "title": str(row["title"]).strip(),
                "venue": str(row["venue"]).strip(),
                "year": int(row["year"]),
                "external_ref": str(row["external_ref"]).strip(),
                "route_note": compact_excerpt(str(row["abstract"]).strip(), limit=280),
            }
        )
    return {
        "source_policy": (
            "user_library_domain_knowledge_only: these papers are user-provided background for terminology, "
            "domain routes, and problem framing. They must not be used as Gold evidence, evidence_basis, "
            "storyline_trace source IDs, quality signals, or scoring standards."
        ),
        "allowed_uses": ["terminology", "domain_route_learning", "problem_framing", "application_constraints"],
        "forbidden_uses": ["evidence_basis", "storyline_trace.source_id", "quality_signal", "paper_weight", "scoring_standard"],
        "papers": papers,
    }


def merge_latest_user_domain_knowledge(root_dir: Path, session_context: Dict[str, object], *, limit: int = 12) -> Dict[str, object]:
    merged = dict(session_context)
    domain_knowledge = build_user_domain_knowledge_summary(root_dir, limit=limit)
    if domain_knowledge.get("papers"):
        merged["domain_knowledge_context"] = domain_knowledge
    elif "domain_knowledge_context" in merged:
        merged.pop("domain_knowledge_context", None)
    return merged


def score_dimension_against_sources(
    *,
    candidate: Dict[str, object],
    dimension: Dict[str, object],
    sources: List[Dict[str, object]],
    explicit_dimension_covered: bool = True,
) -> Dict[str, object]:
    candidate_text = " ".join(normalize_text(candidate.get(field)) for field in dimension["candidate_fields"])
    candidate_tokens = token_set(candidate_text)
    matches: List[Dict[str, object]] = []
    for source in sources:
        field_texts = []
        for field in dimension["pattern_fields"] + dimension["genome_fields"] + dimension["paper_fields"]:
            value = source.get(field)
            if value:
                field_texts.append((field, value))
        if not field_texts:
            continue
        source_tokens = token_set(" ".join(normalize_text(value) for _, value in field_texts))
        overlap = candidate_tokens & source_tokens
        if not overlap:
            continue
        best_field, best_value = max(
            field_texts,
            key=lambda item: len(candidate_tokens & token_set(item[1])),
        )
        quality_weight = 1.0 + min(1.5, float(source.get("paper_weight", 0) or 0) / 6.0)
        if str(source.get("award", "")).strip():
            quality_weight += 0.5
        overlap_score = min(1.0, len(overlap) / 5.0)
        matches.append(
            {
                "source_type": str(source.get("source_type", "evidence")),
                "source_id": source_identity(source),
                "quality_signal": quality_signal_label(source),
                "matched_field": best_field,
                "evidence_excerpt": compact_excerpt(best_value),
                "matched_terms": sorted(overlap)[:8],
                "match_strength": round(overlap_score * quality_weight, 3),
            }
        )

    matches.sort(key=lambda item: float(item["match_strength"]), reverse=True)
    top_matches = matches[:3]
    evidence_score = min(10.0, sum(float(item["match_strength"]) for item in top_matches) * 3.2)
    candidate_completeness = min(2.0, max(0.0, len(candidate_tokens) / 16.0))
    evidence_coverage_penalty = 0.0 if explicit_dimension_covered else -2.0
    score = round(min(10.0, max(0.0, evidence_score + candidate_completeness + evidence_coverage_penalty)), 2)
    return {
        "label": dimension["label"],
        "weight": dimension["weight"],
        "score": score,
        "weighted_score": round(score * float(dimension["weight"]), 3),
        "evidence": top_matches,
        "standard_source": "Idea Genome Cards, Pattern Cards, and core Gold papers stored in this project.",
        "explicit_evidence_covered": explicit_dimension_covered,
        "evidence_coverage_penalty": evidence_coverage_penalty,
    }


def build_fallback_evidence_basis(payload: Dict[str, object], sources: List[Dict[str, object]]) -> List[Dict[str, str]]:
    explicit = payload.get("evidence_basis")
    if isinstance(explicit, list) and explicit:
        normalized = []
        for item in explicit:
            if not isinstance(item, dict):
                continue
            normalized.append(
                {
                    "source_type": str(item.get("source_type", "")).strip(),
                    "source_id": str(item.get("source_id", "")).strip(),
                    "quality_signal": str(item.get("quality_signal", "")).strip(),
                    "borrowed_standard": str(item.get("borrowed_standard", "")).strip(),
                    "used_for": str(item.get("used_for", "")).strip(),
                }
            )
        if normalized:
            return normalized
    basis = []
    historical_pattern = str(payload.get("historical_pattern", "")).strip()
    for source in sources:
        source_type = str(source.get("source_type", "")).strip()
        identity = source_identity(source)
        if historical_pattern and historical_pattern not in {identity, str(source.get("pattern_key", "")).strip()}:
            continue
        basis.append(
            {
                "source_type": source_type,
                "source_id": identity,
                "quality_signal": quality_signal_label(source),
                "borrowed_standard": compact_excerpt(
                    source.get("core_move")
                    or source.get("transferable_pattern")
                    or source.get("problem_reframing")
                    or source.get("title")
                ),
                "used_for": "innovation|logic|value",
            }
        )
        if len(basis) >= 3:
            break
    if basis:
        return basis
    for source in sources[:3]:
        basis.append(
            {
                "source_type": str(source.get("source_type", "")).strip(),
                "source_id": source_identity(source),
                "quality_signal": quality_signal_label(source),
                "borrowed_standard": compact_excerpt(
                    source.get("core_move")
                    or source.get("transferable_pattern")
                    or source.get("problem_reframing")
                    or source.get("title")
                ),
                "used_for": "evidence_grounding",
            }
        )
    return basis


def normalize_explicit_evidence_basis(value: object) -> List[Dict[str, str]]:
    if not isinstance(value, list):
        return []
    normalized = []
    for item in value:
        if not isinstance(item, dict):
            continue
        source_id = str(item.get("source_id", "")).strip()
        borrowed_standard = str(item.get("borrowed_standard", "")).strip()
        if not source_id and not borrowed_standard:
            continue
        normalized.append(
                {
                    "source_type": str(item.get("source_type", "")).strip(),
                    "source_id": source_id,
                    "source_title": str(item.get("source_title", "")).strip(),
                    "quality_signal": str(item.get("quality_signal", "")).strip(),
                    "borrowed_standard": borrowed_standard,
                    "used_for": str(item.get("used_for", "")).strip(),
                }
        )
    return normalized


def evidence_basis_used_for_dimensions(items: List[Dict[str, object]]) -> set[str]:
    covered: set[str] = set()
    valid_dimensions = {str(item["key"]) for item in EVIDENCE_RUBRIC_DIMENSIONS}
    for item in items:
        used_for = str(item.get("used_for", "")).strip()
        for value in used_for.split("|"):
            dimension = value.strip()
            if dimension in valid_dimensions:
                covered.add(dimension)
    return covered


def audit_generation_evidence(payload: Dict[str, object], sources: List[Dict[str, object]]) -> Dict[str, object]:
    explicit_basis = normalize_explicit_evidence_basis(payload.get("evidence_basis"))
    source_alias_map: Dict[str, List[Dict[str, object]]] = {}
    for source in sources:
        for alias in source_aliases(source):
            source_alias_map.setdefault(alias, []).append(source)

    matched = []
    unmatched = []
    source_type_mismatches = []
    for item in explicit_basis:
        source_id = str(item.get("source_id", "")).strip().lower()
        candidates = source_alias_map.get(source_id, []) if source_id else []
        declared_type = str(item.get("source_type", "")).strip().lower()
        source = None
        if candidates and declared_type:
            source = next(
                (
                    candidate
                    for candidate in candidates
                    if str(candidate.get("source_type", "")).strip().lower() == declared_type
                ),
                None,
            )
        elif candidates:
            source = candidates[0]
        if source:
            matched.append(
                {
                    **item,
                    "matched_source_type": str(source.get("source_type", "")).strip(),
                    "matched_source_id": source_identity(source),
                    "matched_quality_signal": quality_signal_label(source),
                }
            )
        elif candidates and declared_type:
            source_type_mismatches.append(
                {
                    **item,
                    "matched_source_type": "|".join(
                        sorted(
                            {
                                str(candidate.get("source_type", "")).strip().lower()
                                for candidate in candidates
                                if str(candidate.get("source_type", "")).strip()
                            }
                        )
                    ),
                    "matched_source_id": "|".join(dict.fromkeys(source_identity(candidate) for candidate in candidates)),
                    "matched_quality_signal": "|".join(dict.fromkeys(quality_signal_label(candidate) for candidate in candidates)),
                }
            )
        else:
            unmatched.append(item)

    inferred_basis = build_fallback_evidence_basis(payload, sources)
    if matched:
        status = "explicit_valid"
        penalty = 0.0
    elif explicit_basis:
        status = "explicit_unmatched"
        penalty = -1.5
    elif inferred_basis:
        status = "missing_model_evidence_inferred"
        penalty = -1.0
    else:
        status = "missing_evidence"
        penalty = -2.5
    covered_uses = sorted(evidence_basis_used_for_dimensions(matched))
    required_uses = [str(item["key"]) for item in EVIDENCE_RUBRIC_DIMENSIONS]
    missing_uses = [dimension for dimension in required_uses if dimension not in covered_uses]

    return {
        "status": status,
        "penalty": penalty,
        "explicit_basis_count": len(explicit_basis),
        "matched_basis_count": len(matched),
        "unmatched_basis_count": len(unmatched),
        "source_type_mismatch_count": len(source_type_mismatches),
        "covered_uses": covered_uses,
        "missing_uses": missing_uses,
        "matched_basis": matched,
        "unmatched_basis": unmatched,
        "source_type_mismatches": source_type_mismatches,
        "inferred_basis": inferred_basis,
        "requirement": (
            "Generated ideas must explicitly cite stored core Gold paper, genome, or pattern evidence. "
            "The matched evidence must cover innovation, logic, feasibility, value, and defensibility. "
            "Inferred evidence is kept for debugging but is penalized because it was not supplied by the generator."
        ),
    }


def build_evidence_grounded_score(
    payload: Dict[str, object],
    *,
    pattern_rows=None,
    genome_rows=None,
    paper_rows=None,
) -> Dict[str, object]:
    rubric_paper_rows = [
        row
        for row in paper_rows or []
        if is_core_gold_standard_paper(dict(row)) and float(dict(row).get("paper_weight", 0) or 0) > 0
    ]
    sources = flatten_rubric_sources(
        pattern_rows=pattern_rows,
        genome_rows=genome_rows,
        paper_rows=rubric_paper_rows,
    )
    if not sources:
        return {
            "total": 0.0,
            "dimensions": {},
            "rubric_source": "missing_evidence",
            "evidence_basis": [],
            "generation_evidence_audit": {
                "status": "missing_evidence",
                "penalty": -2.5,
                "requirement": "No stored core Gold evidence was available.",
            },
            "warning": "No stored core Gold paper evidence was available, so evidence-grounded scoring could not be performed.",
        }
    generation_audit = audit_generation_evidence(payload, sources)
    covered_uses = set(generation_audit.get("covered_uses", []))
    dimensions = {
        item["key"]: score_dimension_against_sources(
            candidate=payload,
            dimension=item,
            sources=sources,
            explicit_dimension_covered=str(item["key"]) in covered_uses,
        )
        for item in EVIDENCE_RUBRIC_DIMENSIONS
    }
    total = round(sum(float(item["weighted_score"]) for item in dimensions.values()), 2)
    prior_art = payload.get("prior_art", {})
    prior_penalty = 0.0
    if isinstance(prior_art, dict):
        prior_penalty = {
            "duplicate": -2.0,
            "near_miss": -1.0,
            "complementary": -0.25,
            "distant": 0.25,
        }.get(str(prior_art.get("overlap_label", "")).strip(), 0.0)
    total = round(max(0.0, min(10.0, total + prior_penalty + float(generation_audit["penalty"]))), 2)
    return {
        "total": total,
        "scale": "0-10",
        "prior_art_adjustment": prior_penalty,
        "generation_evidence_adjustment": float(generation_audit["penalty"]),
        "dimensions": dimensions,
        "rubric_source": (
            "Derived strictly from stored core Gold evidence: explicit Best/Outstanding/Oral paper standards, "
            "Idea Genome standards, and Pattern Card standards. Spotlight/high-citation-only papers are excluded."
        ),
        "evidence_basis": generation_audit["matched_basis"],
        "inferred_evidence_basis": generation_audit["inferred_basis"],
        "generation_evidence_audit": generation_audit,
    }


def require_evidence_ready(root_dir: Path, paper_limit: int, *, allow_extractive: bool = False) -> None:
    payload, _, _ = build_evidence_report(root_dir, paper_limit=max(3, paper_limit))
    if payload.get("status") == "ready_for_strict_generation":
        return
    counts = payload.get("counts", {})
    if (
        allow_extractive
        and isinstance(counts, dict)
        and int(counts.get("high_weight_papers", 0) or 0) >= 3
        and int(counts.get("genome_cards", 0) or 0) >= 2
        and int(counts.get("pattern_cards", 0) or 0) >= 1
    ):
        return
    missing = payload.get("missing_requirements", [])
    if isinstance(missing, list) and missing:
        detail = " ".join(str(item).strip() for item in missing if str(item).strip())
    else:
        detail = "Build scored core Gold papers, Idea Genome Cards, and Pattern Cards first."
    raise ValueError(
        "Strict idea generation is not ready: "
        f"{payload.get('status')}. {detail} Run `ra evidence`, then follow its next commands."
    )


def require_strict_evidence_ready(root_dir: Path, paper_limit: int) -> None:
    require_evidence_ready(root_dir, paper_limit, allow_extractive=False)


def require_explicit_valid_generation_evidence(payload: Dict[str, object], score_payload: Dict[str, object]) -> None:
    audit = score_payload.get("generation_evidence_audit", {})
    if not isinstance(audit, dict):
        raise ValueError("Generated idea lacks a generation evidence audit.")
    missing_uses = audit.get("missing_uses", [])
    if not isinstance(missing_uses, list):
        missing_uses = []
    if (
        audit.get("status") == "explicit_valid"
        and int(audit.get("matched_basis_count", 0) or 0) > 0
        and not missing_uses
    ):
        return
    title = str(payload.get("idea_title", "")).strip() or "untitled idea"
    dimension_note = ""
    if missing_uses:
        dimension_note = " Missing explicit evidence dimensions: " + ", ".join(str(item) for item in missing_uses) + "."
    raise ValueError(
        f"Generated idea `{title}` failed strict evidence audit: {audit.get('status')}. "
        "Every accepted idea must explicitly cite stored core Gold paper, genome, or pattern evidence "
        "for innovation, logic, feasibility, value, and defensibility."
        f"{dimension_note}"
    )


def generation_evidence_rejection_reason(payload: Dict[str, object], score_payload: Dict[str, object]) -> str | None:
    user_library_reason = user_library_reference_rejection_reason(payload)
    if user_library_reason:
        return user_library_reason
    try:
        require_explicit_valid_generation_evidence(payload, score_payload)
    except ValueError as exc:
        return str(exc)
    return None


def user_library_reference_rejection_reason(payload: Dict[str, object]) -> str:
    context = payload.get("domain_knowledge_context")
    if not isinstance(context, dict):
        return ""
    aliases = user_library_reference_aliases(context)
    if not aliases:
        return ""

    def matches_user_library(value: object) -> bool:
        cleaned = normalize_support_reference(value)
        return bool(cleaned and cleaned in aliases)

    evidence_basis = payload.get("evidence_basis")
    if isinstance(evidence_basis, list):
        for item in evidence_basis:
            if isinstance(item, dict) and matches_user_library(item.get("source_id", "")):
                return (
                    "candidate_used_user_library_as_scoring_evidence: "
                    "用户论文库只能作为领域知识/路线学习，不能进入 evidence_basis 或 idea 评分标准。"
                )

    trace = payload.get("storyline_trace")
    if isinstance(trace, dict):
        for step in trace.values():
            if isinstance(step, dict) and matches_user_library(step.get("source_id", "")):
                return (
                    "candidate_used_user_library_as_storyline_source: "
                    "用户论文库不能作为顶会故事线 source_id，只能辅助理解用户领域。"
                )
    return ""


def user_library_reference_aliases(context: Dict[str, object]) -> set[str]:
    papers = context.get("papers")
    if not isinstance(papers, list):
        return set()
    aliases: set[str] = set()
    for item in papers:
        if not isinstance(item, dict):
            continue
        for key in ("paper_id", "title", "external_ref"):
            value = normalize_support_reference(item.get(key, ""))
            if value:
                aliases.add(value)
        paper_id = normalize_support_reference(item.get("paper_id", ""))
        if paper_id:
            aliases.add(f"user:{paper_id}")
            aliases.add(f"user_library:{paper_id}")
    return aliases


def normalize_support_reference(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def support_references_match(value: object, aliases: set[str]) -> bool:
    cleaned = normalize_support_reference(value)
    return bool(cleaned and cleaned in aliases)


def build_session_step_evidence_aliases(
    *,
    pattern_rows,
    genome_rows,
    session_context: Dict[str, object],
) -> Tuple[set[str], set[str], set[str]]:
    pattern_aliases: set[str] = set()
    genome_aliases: set[str] = set()
    for row in pattern_rows or []:
        try:
            payload = json.loads(row["content_json"])
        except (json.JSONDecodeError, KeyError, TypeError):
            payload = {}
        payload["source_type"] = "pattern"
        payload["pattern_key"] = str(row["pattern_key"]).strip()
        pattern_aliases.update(source_aliases(payload))
    for row in genome_rows or []:
        try:
            payload = json.loads(row["content_json"])
        except (json.JSONDecodeError, KeyError, TypeError):
            payload = {}
        payload["source_type"] = "genome"
        payload["paper_id"] = int(row["paper_id"])
        payload["title"] = str(row["title"]).strip()
        genome_aliases.update(source_aliases(payload))

    for item in normalize_explicit_evidence_basis(session_context.get("evidence_basis")):
        aliases = {
            normalize_support_reference(item.get("source_id", "")),
            normalize_support_reference(item.get("source_title", "")),
        }
        aliases = {alias for alias in aliases if alias}
        source_type = str(item.get("source_type", "")).strip().lower()
        if source_type == "pattern":
            pattern_aliases.update(aliases)
        elif source_type in {"genome", "paper"}:
            genome_aliases.update(aliases)
        else:
            pattern_aliases.update(aliases)
            genome_aliases.update(aliases)

    story_trace = session_context.get("storyline_trace")
    if isinstance(story_trace, dict):
        for step in story_trace.values():
            if not isinstance(step, dict):
                continue
            source_id = normalize_support_reference(step.get("source_id", ""))
            if source_id:
                pattern_aliases.add(source_id)
                genome_aliases.add(source_id)
    return pattern_aliases, genome_aliases, pattern_aliases | genome_aliases


def critique_text_mentions_user_library_as_standard(payload: Dict[str, object]) -> bool:
    standard_words = [
        "scoring standard",
        "gold evidence",
        "gold standard",
        "evidence_basis",
        "storyline source",
        "评分标准",
        "gold依据",
        "gold 依据",
        "金标准",
        "证据依据",
        "故事线来源",
    ]
    user_words = [
        "user library",
        "user_library",
        "domain knowledge",
        "用户论文库",
        "用户库",
        "领域知识库",
    ]
    text = normalize_support_reference(
        " ".join(
            str(payload.get(key, ""))
            for key in [
                "verdict_summary",
                "why_user_may_be_wrong",
                "trend_alignment",
                "preserve",
                "revise",
                "revised_idea",
                "coach_message",
            ]
        )
    )
    return any(word in text for word in user_words) and any(word in text for word in standard_words)


def validate_session_step_payload(
    payload: Dict[str, object],
    *,
    pattern_rows,
    genome_rows,
    session_context: Dict[str, object],
) -> None:
    pattern_aliases, genome_aliases, any_evidence_aliases = build_session_step_evidence_aliases(
        pattern_rows=pattern_rows,
        genome_rows=genome_rows,
        session_context=session_context,
    )
    user_aliases = user_library_reference_aliases(
        session_context.get("domain_knowledge_context")
        if isinstance(session_context.get("domain_knowledge_context"), dict)
        else {}
    )

    pattern_refs = payload.get("supporting_patterns")
    genome_refs = payload.get("supporting_genomes")
    if not isinstance(pattern_refs, list):
        raise ValueError("Critique response supporting_patterns must be a list")
    if not isinstance(genome_refs, list):
        raise ValueError("Critique response supporting_genomes must be a list")
    refs = [str(item).strip() for item in pattern_refs + genome_refs if str(item).strip()]

    for item in refs:
        if support_references_match(item, user_aliases):
            raise ValueError(
                "session_step_used_user_library_as_standard: "
                "用户论文库只能作为领域知识/路线学习，不能成为多轮修改的 supporting evidence 或评分标准。"
            )
    if critique_text_mentions_user_library_as_standard(payload):
        raise ValueError(
            "session_step_used_user_library_as_standard: "
            "模型把用户论文库/领域知识表述成了评分或故事线标准；本轮拒绝写入。"
        )

    if any_evidence_aliases and not any(support_references_match(item, any_evidence_aliases) for item in refs):
        raise ValueError(
            "session_step_lost_gold_logic_space: "
            "本轮 revised_idea 没有保留任何 supplied Pattern/Genome/Gold 支撑；请在优秀论文逻辑空间内重写，而不是自由发挥。"
        )
    leakage = session_step_surface_leakage(payload, pattern_rows=pattern_rows, genome_rows=genome_rows)
    if leakage:
        raise ValueError(
            "session_step_reused_source_surface_terms: "
            "本轮 revised_idea 复用了源论文/Genome 的表面方法词，像是在硬搬方法而不是迁移故事线。"
            f" source={leakage.get('source_id')} terms={','.join(str(item) for item in leakage.get('matched_surface_tokens', []))}"
        )


def session_step_surface_leakage(
    payload: Dict[str, object],
    *,
    pattern_rows,
    genome_rows,
) -> Dict[str, object] | None:
    audit_payload = {
        "idea_title": payload.get("verdict_summary", ""),
        "core_hypothesis": payload.get("revised_idea", ""),
        "frontier_gap": payload.get("why_user_may_be_wrong", ""),
        "novelty": payload.get("revise", ""),
        "value": payload.get("value_judgment", {}),
        "key_risk": payload.get("why_not_analysis", {}),
        "first_experiments": payload.get("feasibility_assessment", {}),
        "evaluation_outline": payload.get("coach_message", ""),
        "paper_angle": payload.get("preserve", ""),
    }
    sources = flatten_rubric_sources(pattern_rows=pattern_rows, genome_rows=genome_rows)
    for source in sources:
        leakage = source_surface_leakage(audit_payload, source)
        if leakage:
            return leakage
    return None


def build_candidate_critique_outputs(payload: Dict[str, object]) -> Dict[str, Dict[str, str]]:
    next_target = infer_candidate_next_target(payload)
    feasibility = build_feasibility_assessment(payload, next_target)
    why_not = str(payload.get("key_risk", "")).strip() or "No explicit blocker captured yet."
    value = build_value_judgment(payload, "", str(payload.get("idea_title", "")).strip())
    return {
        "feasibility_assessment": {
            "status": infer_feasibility_status(payload),
            "summary": feasibility,
            "next_test": next_target,
        },
        "why_not_analysis": {
            "status": "moderate" if str(payload.get("key_risk", "")).strip() else "light",
            "summary": why_not,
            "mitigation": next_target,
        },
        "value_judgment": {
            "status": infer_value_status(payload),
            "summary": value,
            "proof_gap": next_target,
        },
    }


def infer_candidate_next_target(payload: Dict[str, object]) -> str:
    experiments = payload.get("first_experiments", [])
    if isinstance(experiments, list):
        for item in experiments:
            text = str(item).strip()
            if text:
                return text
    evaluation_outline = str(payload.get("evaluation_outline", "")).strip()
    if evaluation_outline:
        return evaluation_outline
    key_risk = str(payload.get("key_risk", "")).strip()
    if key_risk:
        return key_risk
    return "Turn the idea into the first concrete test."


def infer_feasibility_status(payload: Dict[str, object]) -> str:
    experiments = payload.get("first_experiments", [])
    experiment_count = len(experiments) if isinstance(experiments, list) else 0
    evaluation_outline = str(payload.get("evaluation_outline", "")).strip()
    if experiment_count >= 2 and evaluation_outline:
        return "promising"
    if experiment_count >= 1 or evaluation_outline:
        return "unclear"
    return "fragile"


def infer_value_status(payload: Dict[str, object]) -> str:
    value = str(payload.get("value", "")).strip()
    novelty = str(payload.get("novelty", "")).strip()
    if value and novelty:
        return "high"
    if value or novelty:
        return "medium"
    return "low"


def normalize_dimension_payload(
    value: object,
    *,
    field_name: str,
    extra_key: str,
) -> Dict[str, str]:
    if not isinstance(value, dict):
        return {}
    status = str(value.get("status", "")).strip().lower()
    summary = str(value.get("summary", "")).strip()
    extra_value = str(value.get(extra_key, "")).strip()
    if not (status or summary or extra_value):
        return {}
    return {
        "field_name": field_name,
        "status": status,
        "summary": summary,
        extra_key: extra_value,
    }


def normalize_critique_outputs(value: object) -> Dict[str, Dict[str, str]]:
    if not isinstance(value, dict):
        return {}
    outputs = {
        "feasibility_assessment": normalize_dimension_payload(
            value.get("feasibility_assessment"),
            field_name="feasibility_assessment",
            extra_key="next_test",
        ),
        "why_not_analysis": normalize_dimension_payload(
            value.get("why_not_analysis"),
            field_name="why_not_analysis",
            extra_key="mitigation",
        ),
        "value_judgment": normalize_dimension_payload(
            value.get("value_judgment"),
            field_name="value_judgment",
            extra_key="proof_gap",
        ),
    }
    return {key: payload for key, payload in outputs.items() if payload}


def render_dimension_summary(detail: Dict[str, str], *, extra_key: str) -> str:
    summary = str(detail.get("summary", "")).strip()
    extra_value = str(detail.get(extra_key, "")).strip()
    pieces = []
    if summary:
        pieces.append(summary)
    if extra_value:
        label = {
            "next_test": "Next test",
            "mitigation": "Mitigation",
            "proof_gap": "Proof gap",
        }.get(extra_key, "Next")
        pieces.append(f"{label}: {extra_value}")
    return " ".join(pieces).strip()


def select_candidate_idea_ranking_rows(rows, query: str, include_all: bool):
    parsed = []
    for row in rows:
        try:
            payload = json.loads(row["content_json"])
        except json.JSONDecodeError:
            continue
        if not is_candidate_idea_payload(payload):
            continue
        parsed.append((row, payload))

    if not parsed:
        return [], ""
    if query.strip():
        filtered = [(row, payload) for row, payload in parsed if str(row["query"]).strip() == query.strip()]
        return filtered, query.strip()
    if include_all:
        return parsed, "all"
    latest_query = str(parsed[0][0]["query"]).strip()
    filtered = [(row, payload) for row, payload in parsed if str(row["query"]).strip() == latest_query]
    return filtered, latest_query


def build_candidate_idea_ranking(
    *,
    root_dir: Path,
    query: str,
    include_all: bool,
    limit: int,
    idea_ids: Iterable[int] | None = None,
) -> Dict[str, object]:
    config = load_config(root_dir)
    ensure_layout(config)
    init_db(config.db_path)

    rows = list_candidate_ideas(config.db_path, limit=200)
    cohort, active_query = select_candidate_idea_ranking_rows(rows, query, include_all)
    selected_ids = {int(item) for item in idea_ids or []}
    if selected_ids:
        cohort = [(row, payload) for row, payload in cohort if int(row["id"]) in selected_ids]
        active_query = f"{active_query or query.strip() or 'candidate ideas'} current_run"
    if not cohort:
        raise ValueError("No candidate ideas available for ranking. Run `ra id --query ...` first.")

    pattern_rows = list_strict_pattern_rows(config.db_path, limit=50)
    genome_rows = list_strict_gold_genome_rows(config.db_path, limit=50)
    top_paper_rows = core_gold_rows(list_top_papers(config.db_path, limit=50))
    pattern_keys = {str(row["pattern_key"]).strip() for row in pattern_rows}
    ranked: List[Dict[str, object]] = []
    for row, payload in cohort:
        scoring = score_candidate_idea_payload(payload, pattern_keys)
        evidence_scoring = build_evidence_grounded_score(
            payload,
            pattern_rows=pattern_rows,
            genome_rows=genome_rows,
            paper_rows=top_paper_rows,
        )
        if evidence_scoring.get("rubric_source") != "missing_evidence":
            scoring = {
                "score": float(evidence_scoring["total"]),
                "components": {
                    key: round(float(value.get("score", 0.0)), 2)
                    for key, value in evidence_scoring.get("dimensions", {}).items()
                    if isinstance(value, dict)
                },
            }
        critique_outputs = normalize_critique_outputs(payload.get("critique_outputs"))
        ranked.append(
            {
                "idea_id": int(row["id"]),
                "query": str(row["query"]).strip(),
                "title": str(row["title"]).strip(),
                "paper_angle": str(payload.get("paper_angle", "")).strip(),
                "historical_pattern": str(payload.get("historical_pattern", "")).strip(),
                "trend_support": str(payload.get("trend_support", "")).strip(),
                "prior_art_label": str(payload.get("prior_art", {}).get("overlap_label", "")).strip()
                if isinstance(payload.get("prior_art"), dict)
                else "",
                "prior_art_summary": str(payload.get("prior_art", {}).get("summary", "")).strip()
                if isinstance(payload.get("prior_art"), dict)
                else "",
                "critique_outputs": critique_outputs,
                "feasibility": render_dimension_summary(
                    critique_outputs.get("feasibility_assessment", {}),
                    extra_key="next_test",
                ),
                "why_not_analysis": render_dimension_summary(
                    critique_outputs.get("why_not_analysis", {}),
                    extra_key="mitigation",
                ),
                "value_judgment": render_dimension_summary(
                    critique_outputs.get("value_judgment", {}),
                    extra_key="proof_gap",
                ),
                "score": scoring["score"],
                "components": scoring["components"],
                "evidence_grounded_score": evidence_scoring,
            }
        )
    ranked.sort(key=lambda item: (-float(item["score"]), int(item["idea_id"])))
    shown = ranked[: max(1, limit)]

    idea_dir = config.output_dir / "ideas"
    idea_dir.mkdir(parents=True, exist_ok=True)
    report_path = idea_dir / "idea_ranking.json"
    report_payload: Dict[str, object] = {
        "scope": active_query,
        "ranked_ideas": shown,
        "total_ranked": len(ranked),
    }
    return {
        "scope": active_query,
        "ranked_ideas": ranked,
        "shown_ideas": shown,
        "report_path": report_path,
        "report_payload": report_payload,
    }


def rerank_candidate_ideas_with_llm(
    *,
    root_dir: Path,
    ranked: List[Dict[str, object]],
    query: str,
    provider: str | None,
    model: str,
) -> Dict[str, object]:
    config = load_config(root_dir)
    ensure_layout(config)
    init_db(config.db_path)
    apply_llm_runtime_overrides(config, provider, model)

    client = LLMClient(config.llm)
    shortlist = []
    for item in ranked[:10]:
        shortlist.append(
            {
                "idea_id": item["idea_id"],
                "title": item["title"],
                "historical_pattern": item["historical_pattern"],
                "trend_support": item["trend_support"],
                "paper_angle": item["paper_angle"],
                "heuristic_score": item["score"],
                "heuristic_components": item["components"],
            }
        )
    prompt = build_ranking_prompt(query=query, ranked_candidates=shortlist)
    response = client.chat(prompt=prompt, system_prompt=RANKING_SYSTEM_PROMPT)
    payload = parse_ranking_response(response.text)

    known_ids = [int(item["idea_id"]) for item in ranked]
    ordered_ids: List[int] = []
    for idea_id in payload["ranked_ids"]:
        if idea_id in known_ids and idea_id not in ordered_ids:
            ordered_ids.append(idea_id)
    for idea_id in known_ids:
        if idea_id not in ordered_ids:
            ordered_ids.append(idea_id)

    item_by_id = {int(item["idea_id"]): item for item in ranked}
    reranked = [item_by_id[idea_id] for idea_id in ordered_ids if idea_id in item_by_id]
    return {
        "summary": str(payload["summary"]).strip(),
        "why_top": str(payload["why_top"]).strip(),
        "watchouts": [str(item).strip() for item in payload["watchouts"] if str(item).strip()],
        "ranked_ids": ordered_ids,
        "ranked_ideas": reranked,
    }


def cmd_rank_candidate_ideas(
    root_dir: Path,
    limit: int,
    query: str,
    include_all: bool,
    use_llm: bool,
    provider: str | None,
    model: str,
) -> int:
    try:
        ranking_data = build_candidate_idea_ranking(
            root_dir=root_dir,
            query=query,
            include_all=include_all,
            limit=limit,
        )
    except ValueError as exc:
        print(str(exc))
        return 0
    config = load_config(root_dir)
    active_query = str(ranking_data["scope"])
    ranked = list(ranking_data["ranked_ideas"])
    shown = list(ranking_data["shown_ideas"])
    report_path = ranking_data["report_path"]
    report_payload = dict(ranking_data["report_payload"])

    llm_rerank = None
    if use_llm:
        try:
            llm_rerank = rerank_candidate_ideas_with_llm(
                root_dir=config.root_dir,
                ranked=ranked,
                query=active_query,
                provider=provider,
                model=model,
            )
        except (LLMError, ValueError, json.JSONDecodeError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
        report_payload["llm_rerank"] = {
            "summary": llm_rerank["summary"],
            "why_top": llm_rerank["why_top"],
            "watchouts": llm_rerank["watchouts"],
            "ranked_ids": llm_rerank["ranked_ids"][: max(1, limit)],
        }

    report_path.write_text(
        json.dumps(report_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    scope_label = "all candidate ideas" if active_query == "all" else f"query `{active_query}`"
    print(f"Ranked {len(shown)} candidate ideas from {scope_label}.")
    for idx, item in enumerate(shown, start=1):
        components = item["components"]
        print(f"  {idx}. [{item['idea_id']}] score={item['score']:.2f} :: {item['title']}")
        if "innovation" in components:
            print(
                "     evidence-rubric "
                f"innovation={components.get('innovation', 0.0):.2f} "
                f"logic={components.get('logic', 0.0):.2f} "
                f"feasibility={components.get('feasibility', 0.0):.2f} "
                f"value={components.get('value', 0.0):.2f} "
                f"defensibility={components.get('defensibility', 0.0):.2f}"
            )
        else:
            print(
                "     "
                f"pattern={components['pattern_grounding']:.2f} "
                f"trend={components['trend_grounding']:.2f} "
                f"nv={components['novelty_value']:.2f} "
                f"exec={components['execution_readiness']:.2f} "
                f"risk={components['risk_awareness']:.2f} "
                f"prior={components['prior_art_distance']:.2f}"
            )
        evidence_score = item.get("evidence_grounded_score", {})
        if isinstance(evidence_score, dict) and evidence_score.get("rubric_source") != "missing_evidence":
            print(f"     rubric={evidence_score.get('rubric_source')}")
            audit = evidence_score.get("generation_evidence_audit", {})
            if isinstance(audit, dict):
                print(
                    "     generation_evidence="
                    f"{audit.get('status')} matched={audit.get('matched_basis_count', 0)} "
                    f"penalty={audit.get('penalty', 0.0)}"
                )
        if item["prior_art_label"]:
            print(f"     prior_art={item['prior_art_label']} :: {item['prior_art_summary']}")
    if llm_rerank:
        llm_shown = llm_rerank["ranked_ideas"][: max(1, limit)]
        print("LLM rerank:")
        for idx, item in enumerate(llm_shown, start=1):
            print(f"  {idx}. [{item['idea_id']}] score={item['score']:.2f} :: {item['title']}")
        if llm_rerank["summary"]:
            print(f"LLM summary: {llm_rerank['summary']}")
        if llm_rerank["why_top"]:
            print(f"LLM top pick: {llm_rerank['why_top']}")
        if llm_rerank["watchouts"]:
            print("LLM watchouts:")
            for note in llm_rerank["watchouts"]:
                print(f"  - {note}")
    print(f"Ranking report: {report_path}")
    print("Next:")
    top_pick_id = int((llm_rerank["ranked_ideas"][0]["idea_id"] if llm_rerank else shown[0]["idea_id"]))
    if llm_rerank or query.strip() or include_all:
        print(f"  {CLI_CMD} ix --idea-id {top_pick_id}")
    else:
        print(f"  {CLI_CMD} ix")
    return 0


def create_session_from_candidate_row(root_dir: Path, row, name: str) -> Dict[str, object]:
    config = load_config(root_dir)
    ensure_layout(config)
    init_db(config.db_path)

    payload = json.loads(row["content_json"])
    session_name = name.strip() or str(row["title"]).strip()
    initial_idea = str(payload.get("core_hypothesis", "")).strip() or str(row["title"]).strip()
    session_context = build_session_context_from_candidate(payload, int(row["id"]))
    session_id = create_idea_session(
        config.db_path,
        name=session_name,
        current_idea=initial_idea,
        context_json=json.dumps(session_context, ensure_ascii=False, indent=2),
    )
    refresh_session_memory(config.root_dir, session_id)
    _, dossier_path = refresh_session_dossier(config.root_dir, session_id)
    evidence_score = session_context.get("evidence_grounded_score", {})
    if not isinstance(evidence_score, dict):
        evidence_score = {}
    first_experiments = session_context.get("first_experiments", [])
    if not isinstance(first_experiments, list):
        first_experiments = []
    why_not = session_context.get("why_not_done_yet", [])
    if not isinstance(why_not, list):
        why_not = []
    evidence_chain = build_session_evidence_chain(session_context)
    chat_summary = {
        "title": session_name,
        "current_idea": initial_idea,
        "trend_support": str(session_context.get("trend_support", "")).strip(),
        "evidence_score": evidence_score.get("total", ""),
        "scoring_dimensions": summarize_evidence_score_dimensions(evidence_score),
        "evidence_basis": summarize_session_evidence_basis(session_context),
        "current_method_problem": evidence_chain.get("current_method_problem", ""),
        "storyline_logic": evidence_chain.get("storyline_logic", ""),
        "storyline_steps": summarize_storyline_steps(session_context),
        "recent_limitation_signal": evidence_chain.get("recent_limitation_signal", ""),
        "prior_art_gate_decision": str(session_context.get("prior_art_gate_decision", "")).strip(),
        "prior_art_gate_summary": str(session_context.get("prior_art_gate_summary", "")).strip(),
        "key_risk": str(session_context.get("key_risk", "")).strip(),
        "first_experiment": str(first_experiments[0]).strip() if first_experiments else "",
        "why_not_done_yet": [str(item).strip() for item in why_not if str(item).strip()][:2],
    }
    return {
        "session_id": session_id,
        "candidate_idea_id": int(row["id"]),
        "name": session_name,
        "current_idea": initial_idea,
        "trend_support": session_context.get("trend_support", ""),
        "dossier_path": str(dossier_path),
        "chat_summary": chat_summary,
    }


def refresh_session_summary(root_dir: Path, summary: Dict[str, object]) -> Dict[str, object]:
    config = load_config(root_dir)
    ensure_layout(config)
    init_db(config.db_path)
    session_id = int(summary.get("session_id", 0) or 0)
    session = get_idea_session(config.db_path, session_id) if session_id else None
    if not session:
        return summary
    refreshed = dict(summary)
    current_idea = str(session["current_idea"]).strip()
    refreshed["current_idea"] = current_idea
    refreshed["dossier_path"] = str(config.output_dir / "sessions" / f"session-{session_id}-dossier.json")
    chat_summary = dict(refreshed.get("chat_summary", {}) if isinstance(refreshed.get("chat_summary"), dict) else {})
    chat_summary["current_idea"] = current_idea
    chat_summary["session_id"] = session_id
    if refreshed.get("auto_review_loop"):
        chat_summary["auto_review_loop"] = refreshed["auto_review_loop"]
    refreshed["chat_summary"] = chat_summary
    return refreshed


def run_auto_review_loop_for_session(
    root_dir: Path,
    session_summary: Dict[str, object],
    *,
    rounds: int,
    provider: str | None,
    model: str,
    pattern_limit: int,
    genome_limit: int,
) -> Dict[str, object]:
    session_id = int(session_summary.get("session_id", 0) or 0)
    requested_rounds = max(3, min(5, int(rounds or 4)))
    if session_id <= 0:
        return {
            "status": "skipped",
            "reason": "missing_session_id",
            "session_id": session_id,
            "requested_rounds": requested_rounds,
            "rounds_completed": 0,
        }
    config = load_config(root_dir)
    before_turns = list_idea_session_turns(config.db_path, session_id)
    code = cmd_review_loop(
        root_dir,
        session_id=session_id,
        latest=False,
        rounds=requested_rounds,
        provider=provider,
        model=model,
        pattern_limit=pattern_limit,
        genome_limit=genome_limit,
    )
    session = get_idea_session(config.db_path, session_id)
    final_idea = str(session["current_idea"]).strip() if session else str(session_summary.get("current_idea", "")).strip()
    after_turns = list_idea_session_turns(config.db_path, session_id)
    completed_rounds = min(max(0, len(after_turns) - len(before_turns)), requested_rounds)
    if code != 0:
        failure = {
            "status": "failed",
            "reason": "review_loop_failed",
            "exit_code": int(code),
            "session_id": session_id,
            "requested_rounds": requested_rounds,
            "rounds_completed": completed_rounds,
            "final_idea": final_idea,
        }
        failure.update(review_loop_recovery_fields(session_id=session_id, completed_rounds=completed_rounds))
        return failure
    return {
        "status": "completed",
        "session_id": session_id,
        "requested_rounds": requested_rounds,
        "rounds_completed": completed_rounds,
        "final_idea": final_idea,
    }


def summarize_session_evidence_basis(session_context: Dict[str, object], limit: int = 3) -> List[Dict[str, str]]:
    basis = normalize_explicit_evidence_basis(session_context.get("evidence_basis"))
    result: List[Dict[str, str]] = []
    for item in basis[: max(1, int(limit))]:
        result.append(
            {
                "source_type": str(item.get("source_type", "")).strip(),
                "source_id": str(item.get("source_id", "")).strip(),
                "source_title": compact_excerpt(str(item.get("source_title", "")).strip(), 90),
                "quality_signal": str(item.get("quality_signal", "")).strip(),
                "used_for": str(item.get("used_for", "")).strip(),
                "borrowed_standard": compact_excerpt(str(item.get("borrowed_standard", "")).strip(), 140),
            }
        )
    return [item for item in result if any(str(value).strip() for value in item.values())]


def summarize_storyline_steps(session_context: Dict[str, object], limit: int = 6) -> List[Dict[str, str]]:
    trace = session_context.get("storyline_trace", {})
    if not isinstance(trace, dict):
        return []
    labels = [
        ("old_belief", "旧共识"),
        ("bottleneck", "隐藏瓶颈"),
        ("reframing", "问题重构"),
        ("why_now", "为什么现在"),
        ("evidence_design", "证据设计"),
        ("failure_boundary", "失败边界"),
    ]
    result: List[Dict[str, str]] = []
    for key, label in labels[: max(1, int(limit))]:
        item = trace.get(key)
        if not isinstance(item, dict):
            continue
        transfer = compact_excerpt(str(item.get("transfer_to_current_hotspot", "")).strip(), 180)
        standard = compact_excerpt(str(item.get("paper_story_standard", "")).strip(), 160)
        source = compact_excerpt(str(item.get("source_id", "")).strip(), 120)
        if transfer or standard:
            result.append(
                {
                    "step": key,
                    "label": label,
                    "source_id": source,
                    "paper_standard": standard,
                    "transfer": transfer,
                }
            )
    return result


def summarize_evidence_score_dimensions(evidence_score: Dict[str, object]) -> List[Dict[str, object]]:
    dimensions = evidence_score.get("dimensions", {})
    if not isinstance(dimensions, dict):
        return []
    result: List[Dict[str, object]] = []
    for key in ["innovation", "logic", "feasibility", "value", "defensibility"]:
        item = dimensions.get(key, {})
        if not isinstance(item, dict):
            continue
        result.append(
            {
                "dimension": key,
                "score": item.get("score", 0.0),
                "explicit_evidence_covered": bool(item.get("explicit_evidence_covered", False)),
                "evidence_count": len(item.get("evidence", [])) if isinstance(item.get("evidence", []), list) else 0,
            }
        )
    return result


def create_session_from_latest_candidate_idea(root_dir: Path, name_override: str = "") -> Dict[str, object]:
    config = load_config(root_dir)
    ensure_layout(config)
    init_db(config.db_path)
    rows = list_candidate_ideas(config.db_path, limit=1)
    if not rows:
        raise ValueError("No candidate ideas yet. Run `ra id --query ...` first.")
    return create_session_from_candidate_row(config.root_dir, rows[0], name_override)


def create_session_from_best_candidate_idea(root_dir: Path, name_override: str = "") -> Dict[str, object]:
    ranking_data = build_candidate_idea_ranking(
        root_dir=root_dir,
        query="",
        include_all=False,
        limit=1,
    )
    shown = list(ranking_data["shown_ideas"])
    if not shown:
        raise ValueError("No candidate ideas yet. Run `ra id --query ...` first.")

    config = load_config(root_dir)
    top_pick_id = int(shown[0]["idea_id"])
    row = get_candidate_idea(config.db_path, top_pick_id)
    if not row:
        raise ValueError(f"Candidate idea not found: top-ranked idea {top_pick_id}")

    summary = create_session_from_candidate_row(config.root_dir, row, name_override)
    summary["selection_scope"] = str(ranking_data["scope"]).strip()
    return summary


def cmd_session_from_candidate_idea(root_dir: Path, idea_id: int | None, latest: bool, best: bool, name: str) -> int:
    config = load_config(root_dir)
    ensure_layout(config)
    init_db(config.db_path)

    if best or (idea_id is None and not latest):
        try:
            summary = create_session_from_best_candidate_idea(config.root_dir, name)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(
            f"Created idea session #{summary['session_id']} from top-ranked candidate idea "
            f"#{summary['candidate_idea_id']}"
        )
        if summary.get("selection_scope") and summary["selection_scope"] != "all":
            print(f"Ranking scope: {summary['selection_scope']}")
        print(f"Name: {summary['name']}")
        print(f"Current idea: {summary['current_idea']}")
        if summary.get("trend_support"):
            print(f"Trend support: {summary['trend_support']}")
        print("Idea-session summary JSON: " + json.dumps(summary.get("chat_summary", {}), ensure_ascii=False, sort_keys=True))
        print(f"Dossier: {summary['dossier_path']}")
        return 0
    if latest:
        rows = list_candidate_ideas(config.db_path, limit=1)
        row = rows[0] if rows else None
    else:
        row = get_candidate_idea(config.db_path, idea_id)
    if not row:
        target = "latest candidate idea" if latest else f"candidate idea {idea_id}"
        print(f"Candidate idea not found: {target}", file=sys.stderr)
        return 1

    summary = create_session_from_candidate_row(config.root_dir, row, name)
    print(f"Created idea session #{summary['session_id']} from candidate idea #{row['id']}")
    print(f"Name: {summary['name']}")
    print(f"Current idea: {summary['current_idea']}")
    if summary.get("trend_support"):
        print(f"Trend support: {summary['trend_support']}")
    print("Idea-session summary JSON: " + json.dumps(summary.get("chat_summary", {}), ensure_ascii=False, sort_keys=True))
    print(f"Dossier: {summary['dossier_path']}")
    return 0


def cmd_critique(
    root_dir: Path,
    current_idea: str,
    user_instruction: str,
    provider: str | None,
    model: str,
    pattern_limit: int,
    genome_limit: int,
) -> int:
    config = load_config(root_dir)
    ensure_layout(config)
    init_db(config.db_path)

    pattern_rows = list_strict_pattern_rows(config.db_path, limit=max(1, pattern_limit))
    genome_rows = list_strict_gold_genome_rows(config.db_path, limit=max(1, genome_limit))
    if not pattern_rows and not genome_rows:
        print("Need genome or pattern evidence before critique. Run `ra gb` and `ra pb` first.", file=sys.stderr)
        return 1

    apply_llm_runtime_overrides(config, provider, model)

    patterns = []
    for row in pattern_rows:
        payload = json.loads(row["content_json"])
        payload["pattern_key"] = row["pattern_key"]
        patterns.append(payload)

    genomes = []
    for row in genome_rows:
        payload = json.loads(row["content_json"])
        payload["paper_id"] = row["paper_id"]
        payload["title"] = row["title"]
        payload["venue"] = row["venue"]
        payload["year"] = row["year"]
        genomes.append(payload)

    client = LLMClient(config.llm)
    prompt = build_critique_prompt(
        current_idea=current_idea,
        user_instruction=user_instruction,
        session_context={},
        pattern_cards=patterns,
        genome_cards=genomes,
    )
    try:
        response = client.chat(prompt=prompt, system_prompt=CRITIQUE_SYSTEM_PROMPT)
        payload = parse_llm_response_with_repair(
            client=client,
            original_prompt=prompt,
            system_prompt=CRITIQUE_SYSTEM_PROMPT,
            original_text=response.text,
            parser=parse_critique_response,
            response_name="Critique response",
        )
    except (LLMError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    critique_json = json.dumps(payload, ensure_ascii=False, indent=2)
    critique_dir = config.output_dir / "critiques"
    critique_dir.mkdir(parents=True, exist_ok=True)
    critique_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    output_path = critique_dir / f"{critique_id}.json"
    output_path.write_text(critique_json, encoding="utf-8")
    record_id = add_candidate_idea(
        config.db_path,
        query=current_idea,
        title=payload["decision"],
        content_json=critique_json,
    )

    print(f"Decision: {payload['decision']}")
    print(f"Summary: {payload['verdict_summary']}")
    if str(payload.get("trend_alignment", "")).strip():
        print(f"Trend check: {payload['trend_alignment']}")
    print(f"Revised idea: {payload['revised_idea']}")
    print(f"Saved critique #{record_id}: {output_path}")
    return 0


REVIEWER_GATE_SYSTEM_PROMPT = """
You are a top-tier AI/ML conference review panel with three explicit roles:
1. scoring expert: evaluate innovation, logic, feasibility, value, and defensibility;
2. pattern-logic expert: compare the idea's story arc to supplied Gold/Genome/Pattern evidence;
3. top-conference reviewer: attack novelty, evidence, assumptions, risks, and positioning.
This command is manually triggered only. Do not generate a new idea.
If the idea has serious attack points, set decision to rethink_required.
Return strict JSON only.
""".strip()


PRIOR_ART_GATE_SYSTEM_PROMPT = """
You are a prior-art and novelty investigation expert for top-tier AI/ML conference ideas.
Your job is mandatory screening before any generated idea can be accepted.
Use only the supplied candidate idea and supplied local/frontier paper corpus. Do not invent outside papers.
If similar work appears to already cover the core contribution, set decision to duplicate_rethink.
If no close match appears, explain why this has plausibly not been done yet using evidence from limitations, feasibility barriers, missing evaluation setup, data/tooling gaps, or timing.
Return strict JSON only.
""".strip()


def build_prior_art_gate_prompt(
    *,
    idea_payload: Dict[str, object],
    local_prior_art: Dict[str, object],
    paper_corpus: List[Dict[str, object]],
    query: str,
) -> str:
    preferred_language = "zh" if has_chinese_text(idea_payload) or has_chinese_text(query) else "en"
    language_instruction = (
        "Return every natural-language JSON field in Chinese."
        if preferred_language == "zh"
        else "Return natural-language JSON fields in English."
    )
    compact_corpus = [
        {
            "paper_id": int(paper.get("id", 0) or paper.get("paper_id", 0) or 0),
            "title": str(paper.get("title", "")).strip(),
            "venue": str(paper.get("venue", "")).strip(),
            "year": int(paper.get("year", 0) or 0),
            "paper_weight": float(paper.get("paper_weight", 0) or 0),
            "abstract": compact_excerpt(str(paper.get("abstract", "")).strip(), limit=520),
        }
        for paper in paper_corpus[:40]
    ]
    payload = {
        "policy": (
            "Mandatory prior-art gate. Accepting a duplicate is worse than rejecting a plausible idea. "
            "Judge whether the core contribution has already been done, whether nearby work exists, "
            "and why the idea may not have been done yet. "
            f"{language_instruction}"
        ),
        "preferred_language": preferred_language,
        "query": query,
        "candidate_idea": idea_payload,
        "local_similarity_screen": local_prior_art,
        "paper_corpus": compact_corpus,
    }
    return (
        "Screen this candidate research idea before it is accepted.\n"
        "You must answer three questions: has someone already done this, what is the closest prior work, "
        "and if it is not done, why has it plausibly not been done yet?\n"
        "Use only the supplied corpus and local similarity screen; do not cite unstored papers.\n\n"
        f"{language_instruction}\n\n"
        "Decision policy:\n"
        "- duplicate_rethink: closest work already covers the core hypothesis, method/evaluation, or paper angle.\n"
        "- revise: nearby work exists and the idea needs sharper differentiation, but a plausible novel angle remains.\n"
        "- pass: no close prior-art coverage in the supplied evidence and the why-not-done-yet explanation is concrete.\n\n"
        "Return JSON only with this schema:\n"
        "{\n"
        '  "decision": "pass|revise|duplicate_rethink",\n'
        '  "overlap_label": "distant|complementary|near_miss|duplicate",\n'
        '  "closest_work": [{"paper_id": 0, "title": "...", "overlap_reason": "...", "covered_parts": ["..."], "missing_parts": ["..."]}],\n'
        '  "why_not_done_yet": ["..."],\n'
        '  "required_differentiation": ["..."],\n'
        '  "rethink_prompt": "...",\n'
        '  "summary": "..."\n'
        "}\n\n"
        f"Context:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def parse_prior_art_gate_response(text: str) -> Dict[str, object]:
    payload = parse_llm_json(text, response_name="Prior-art gate response")
    if not isinstance(payload, dict):
        raise ValueError("Prior-art gate response must be a JSON object")
    required = [
        "decision",
        "overlap_label",
        "closest_work",
        "why_not_done_yet",
        "required_differentiation",
        "rethink_prompt",
        "summary",
    ]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Prior-art gate response missing keys: {', '.join(missing)}")
    decision = str(payload["decision"]).strip().lower()
    if decision not in {"pass", "revise", "duplicate_rethink"}:
        raise ValueError("Prior-art gate decision must be pass, revise, or duplicate_rethink")
    overlap_label = str(payload["overlap_label"]).strip().lower()
    if overlap_label not in {"distant", "complementary", "near_miss", "duplicate"}:
        raise ValueError("Prior-art gate overlap_label must be distant, complementary, near_miss, or duplicate")
    payload["decision"] = decision
    payload["overlap_label"] = overlap_label
    if not isinstance(payload["closest_work"], list):
        raise ValueError("Prior-art gate closest_work must be a list")
    if not isinstance(payload["why_not_done_yet"], list):
        raise ValueError("Prior-art gate why_not_done_yet must be a list")
    if not isinstance(payload["required_differentiation"], list):
        raise ValueError("Prior-art gate required_differentiation must be a list")
    why_not_items = [str(item).strip() for item in payload["why_not_done_yet"] if str(item).strip()]
    if decision in {"pass", "revise"} and not why_not_items:
        raise ValueError("Prior-art gate must explain why this has plausibly not been done yet")
    payload["why_not_done_yet"] = why_not_items
    payload["required_differentiation"] = [
        str(item).strip() for item in payload["required_differentiation"] if str(item).strip()
    ]
    return payload


def fallback_prior_art_gate(local_prior_art: Dict[str, object]) -> Dict[str, object]:
    label = str(local_prior_art.get("overlap_label", "")).strip().lower() or "distant"
    top_matches = local_prior_art.get("top_matches", [])
    if not isinstance(top_matches, list):
        top_matches = []
    decision = "duplicate_rethink" if label == "duplicate" else ("revise" if label == "near_miss" else "pass")
    closest_work = [
        {
            "paper_id": int(item.get("paper_id", 0) or 0) if isinstance(item, dict) else 0,
            "title": str(item.get("title", "")).strip() if isinstance(item, dict) else "",
            "overlap_reason": str(item.get("reason", "")).strip() if isinstance(item, dict) else "",
            "covered_parts": list(item.get("overlap_terms", [])) if isinstance(item, dict) and isinstance(item.get("overlap_terms"), list) else [],
            "missing_parts": [],
        }
        for item in top_matches[:3]
    ]
    return {
        "decision": decision,
        "overlap_label": label if label in {"distant", "complementary", "near_miss", "duplicate"} else "distant",
        "closest_work": closest_work,
        "why_not_done_yet": [
            "Local metadata/abstract screen found no exact blocker; full-paper verification is still needed."
            if decision != "duplicate_rethink"
            else "Local metadata/abstract screen found duplicate-like overlap, so the idea must be rethought before acceptance."
        ],
        "required_differentiation": [str(local_prior_art.get("differentiation_note", "")).strip()],
        "rethink_prompt": str(local_prior_art.get("differentiation_note", "")).strip(),
        "summary": str(local_prior_art.get("summary", "")).strip(),
        "fallback": True,
    }


def run_prior_art_gate(
    *,
    config,
    query: str,
    idea_payload: Dict[str, object],
    paper_corpus: List[Dict[str, object]],
    provider: str | None,
    model: str,
) -> Dict[str, object]:
    progress_note("执行本地 prior-art 相似度检查")
    local_prior_art = analyze_prior_art(
        idea_payload=idea_payload,
        papers=paper_corpus,
    )
    if str(local_prior_art.get("overlap_label", "")).strip() == "duplicate":
        progress_note("本地 prior-art 判定高度重复，直接要求重想")
        gate = fallback_prior_art_gate(local_prior_art)
        gate["local_prior_art"] = local_prior_art
        gate["expert_used"] = "local_duplicate_guard"
        return gate

    apply_llm_runtime_overrides(config, provider, model)
    if not str(config.llm.api_key).strip():
        progress_note("没有可用 LLM API，使用本地 prior-art 结论")
        gate = fallback_prior_art_gate(local_prior_art)
        gate["expert_used"] = "local_fallback_no_api_key"
        gate["local_prior_art"] = local_prior_art
        return gate
    client = LLMClient(config.llm)
    try:
        progress_note("调用 prior-art 专家判断是否已有相同工作")
        response = client.chat(
            prompt=build_prior_art_gate_prompt(
                idea_payload=idea_payload,
                local_prior_art=local_prior_art,
                paper_corpus=paper_corpus,
                query=query,
            ),
            system_prompt=PRIOR_ART_GATE_SYSTEM_PROMPT,
        )
        gate = parse_prior_art_gate_response(response.text)
        gate["expert_used"] = "llm_prior_art_expert"
    except Exception as exc:  # Prior-art screening is mandatory; fall back to local evidence if the expert call fails.
        gate = fallback_prior_art_gate(local_prior_art)
        gate["expert_used"] = "local_fallback_after_expert_failure"
        gate["expert_error"] = str(exc)
    gate["local_prior_art"] = local_prior_art
    if gate["overlap_label"] == "duplicate":
        gate["decision"] = "duplicate_rethink"
    if gate["decision"] == "pass" and not [item for item in gate.get("why_not_done_yet", []) if str(item).strip()]:
        gate["decision"] = "revise"
        gate["required_differentiation"] = list(gate.get("required_differentiation", [])) + [
            "Explain concretely why this has not already been done."
        ]
    return gate


def build_reviewer_gate_prompt(
    *,
    idea_payload: Dict[str, object],
    pattern_cards: List[Dict[str, object]],
    genome_cards: List[Dict[str, object]],
    evidence_score: Dict[str, object],
) -> str:
    preferred_language = "zh" if has_chinese_text(idea_payload) else "en"
    language_instruction = (
        "The idea/session is Chinese. Return every natural-language JSON field in Chinese, "
        "including summary, attacks, fatal flaws, questions, rethink_trigger, and required_rethink."
        if preferred_language == "zh"
        else "Return natural-language JSON fields in English."
    )
    payload = {
        "policy": (
            "Use reviewer judgment to audit the idea. The reviewer may score and attack the idea, "
            "but must not introduce new idea content as a replacement. Standards must be tied to supplied evidence. "
            f"{language_instruction}"
        ),
        "preferred_language": preferred_language,
        "idea": idea_payload,
        "evidence_grounded_score": evidence_score,
        "pattern_cards": pattern_cards,
        "genome_cards": genome_cards,
    }
    return (
        "Review this research idea as a top-conference reviewer panel.\n"
        "Act through three named roles: scoring expert, pattern-logic expert, and top-conference reviewer.\n"
        "Return whether it can proceed or must be rethought.\n"
        "If attacking, identify exact weak points and what evidence or reframing is needed before continuing.\n\n"
        f"{language_instruction}\n\n"
        "Return JSON only with this schema:\n"
        "{\n"
        '  "decision": "proceed|revise|rethink_required",\n'
        '  "score_expert": {"innovation": 0, "logic": 0, "feasibility": 0, "value": 0, "defensibility": 0, "overall": 0},\n'
        '  "pattern_logic_expert": {"matched_storyline": "...", "missing_logic_links": ["..."], "gold_standard_alignment": "..."},\n'
        '  "top_conference_reviewer": {"main_attacks": ["..."], "fatal_flaws": ["..."], "questions": ["..."]},\n'
        '  "rethink_trigger": "...",\n'
        '  "required_rethink": ["..."],\n'
        '  "summary": "..."\n'
        "}\n\n"
        f"Context:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def parse_reviewer_gate_response(text: str) -> Dict[str, object]:
    payload = parse_llm_json(text, response_name="Reviewer response")
    if not isinstance(payload, dict):
        raise ValueError("Reviewer response must be a JSON object")
    required = [
        "decision",
        "score_expert",
        "pattern_logic_expert",
        "top_conference_reviewer",
        "rethink_trigger",
        "required_rethink",
        "summary",
    ]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Reviewer response missing keys: {', '.join(missing)}")
    decision = str(payload["decision"]).strip().lower()
    if decision not in {"proceed", "revise", "rethink_required"}:
        raise ValueError("Reviewer decision must be proceed, revise, or rethink_required")
    payload["decision"] = decision
    if not isinstance(payload["required_rethink"], list):
        raise ValueError("Reviewer required_rethink must be a list")
    return payload


def resolve_reviewer_idea(config, idea_id: int | None, latest: bool, idea_text: str) -> tuple[Dict[str, object], str]:
    if idea_text.strip():
        return {"idea_title": "Raw idea", "core_hypothesis": idea_text.strip()}, "raw"
    row = None
    if idea_id is not None:
        row = get_candidate_idea(config.db_path, int(idea_id))
    elif latest or idea_id is None:
        rows = list_candidate_ideas(config.db_path, limit=1)
        row = rows[0] if rows else None
    if not row:
        raise ValueError("No idea found. Use `ra rv --idea-id ...`, `ra rv --latest`, or `ra rv --idea \"...\"`.")
    payload = json.loads(row["content_json"])
    if isinstance(payload, dict):
        payload["candidate_idea_id"] = int(row["id"])
        return payload, f"candidate-{row['id']}"
    raise ValueError("Stored candidate idea payload is invalid.")


def build_session_reviewer_payload(root_dir: Path, session_id: int) -> Dict[str, object]:
    config = load_config(root_dir)
    session = get_idea_session(config.db_path, session_id)
    if not session:
        raise ValueError(f"Idea session not found: {session_id}")
    turns = list_idea_session_turns(config.db_path, session_id)
    context = load_session_context(session)
    candidate_payload: Dict[str, object] = {}
    candidate_id = int(context.get("candidate_idea_id", 0) or 0)
    if candidate_id > 0:
        row = get_candidate_idea(config.db_path, candidate_id)
        if row:
            try:
                loaded = json.loads(row["content_json"])
            except json.JSONDecodeError:
                loaded = {}
            if isinstance(loaded, dict):
                candidate_payload = loaded

    current_idea = str(session["current_idea"]).strip()
    base_title = (
        str(candidate_payload.get("idea_title", "")).strip()
        or str(context.get("idea_title", "")).strip()
        or str(session["name"]).strip()
    )
    payload = dict(candidate_payload)
    payload.update(
        {
            "idea_title": base_title,
            "core_hypothesis": current_idea,
            "session_id": int(session_id),
            "session_name": str(session["name"]).strip(),
            "initial_idea": str(session["initial_idea"]).strip(),
            "current_session_idea": current_idea,
            "turn_count": len(turns),
            "memory_summary": load_session_memory(session),
            "session_context": context,
        }
    )
    if candidate_id > 0:
        payload["candidate_idea_id"] = candidate_id
    if turns:
        payload["recent_turns"] = [
            {
                "turn_index": int(turn["turn_index"]),
                "user_instruction": str(turn["user_instruction"]).strip(),
                "decision": str(turn["decision"]).strip(),
                "revised_idea": str(turn["revised_idea"]).strip(),
            }
            for turn in turns[-5:]
        ]
    return payload


def cmd_reviewer_gate(
    root_dir: Path,
    *,
    idea_id: int | None,
    session_id: int | None,
    latest: bool,
    idea_text: str,
    provider: str | None,
    model: str,
    pattern_limit: int,
    genome_limit: int,
) -> int:
    config = load_config(root_dir)
    ensure_layout(config)
    init_db(config.db_path)
    try:
        if session_id is not None:
            progress_note(f"载入会话 #{int(session_id)} 的当前 idea 和记忆")
            idea_payload, source_label = build_session_reviewer_payload(config.root_dir, int(session_id)), f"session-{int(session_id)}"
        else:
            progress_note("载入待审稿 idea")
            idea_payload, source_label = resolve_reviewer_idea(config, idea_id, latest, idea_text)
    except (ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return cmd_reviewer_gate_for_payload(
        config.root_dir,
        idea_payload=idea_payload,
        source_label=source_label,
        provider=provider,
        model=model,
        pattern_limit=pattern_limit,
        genome_limit=genome_limit,
    )


def cmd_reviewer_gate_for_payload(
    root_dir: Path,
    *,
    idea_payload: Dict[str, object],
    source_label: str,
    provider: str | None,
    model: str,
    pattern_limit: int,
    genome_limit: int,
) -> int:
    code, review = run_reviewer_gate_payload(
        root_dir,
        idea_payload=idea_payload,
        source_label=source_label,
        provider=provider,
        model=model,
        pattern_limit=pattern_limit,
        genome_limit=genome_limit,
    )
    if code == 1:
        return 1
    print(f"Reviewer decision: {review['decision']}")
    print(f"Summary: {review.get('summary', '')}")
    if review["decision"] == "rethink_required":
        print("Rethink required:")
        for item in review.get("required_rethink", [])[:6]:
            print(f"  - {item}")
    print(f"JSON: {review.get('review_artifact_path', '')}")
    return code


def run_reviewer_gate_payload(
    root_dir: Path,
    *,
    idea_payload: Dict[str, object],
    source_label: str,
    provider: str | None,
    model: str,
    pattern_limit: int,
    genome_limit: int,
) -> tuple[int, Dict[str, object]]:
    config = load_config(root_dir)
    ensure_layout(config)
    init_db(config.db_path)
    pattern_rows = list_strict_pattern_rows(config.db_path, limit=max(1, pattern_limit))
    genome_rows = list_strict_gold_genome_rows(config.db_path, limit=max(1, genome_limit))
    top_paper_rows = core_gold_rows(list_top_papers(config.db_path, limit=max(5, pattern_limit + genome_limit)))
    progress_note(f"审稿人载入证据：Pattern {len(pattern_rows)}，Genome {len(genome_rows)}，优秀论文 {len(top_paper_rows)}")
    patterns = []
    for row in pattern_rows:
        payload = json.loads(row["content_json"])
        payload["pattern_key"] = row["pattern_key"]
        patterns.append(payload)
    genomes = []
    for row in genome_rows:
        payload = json.loads(row["content_json"])
        payload["paper_id"] = row["paper_id"]
        payload["title"] = row["title"]
        genomes.append(payload)
    evidence_score = build_evidence_grounded_score(
        idea_payload,
        pattern_rows=pattern_rows,
        genome_rows=genome_rows,
        paper_rows=top_paper_rows,
    )
    progress_note("审稿人先计算证据驱动评分，再攻击创新性、逻辑性、可行性")
    apply_llm_runtime_overrides(config, provider, model)
    client = LLMClient(config.llm)
    try:
        progress_note("调用顶会审稿人 gate，若发现致命攻击点会要求重想")
        reviewer_prompt = build_reviewer_gate_prompt(
            idea_payload=idea_payload,
            pattern_cards=patterns,
            genome_cards=genomes,
            evidence_score=evidence_score,
        )
        response = client.chat(
            prompt=reviewer_prompt,
            system_prompt=REVIEWER_GATE_SYSTEM_PROMPT,
        )
        review = parse_llm_response_with_repair(
            client=client,
            original_prompt=reviewer_prompt,
            system_prompt=REVIEWER_GATE_SYSTEM_PROMPT,
            original_text=response.text,
            parser=parse_reviewer_gate_response,
            response_name="Reviewer response",
        )
    except (LLMError, ValueError, json.JSONDecodeError) as exc:
        summary = build_reviewer_failure_summary(
            source_label=source_label,
            reason=str(exc),
        )
        print(str(exc), file=sys.stderr)
        print("Reviewer failure JSON: " + json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 1, {}
    review["source"] = source_label
    review["evidence_grounded_score"] = evidence_score
    output_dir = config.output_dir / "reviews"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"review-{datetime.utcnow().strftime('%Y%m%dT%H%M%S%fZ')}.json"
    output_path.write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    review["review_artifact_path"] = str(output_path)
    session_id = session_id_from_source_label(source_label)
    if session_id:
        attach_reviewer_history_to_session(config.root_dir, session_id, review, output_path)
    return (2 if review["decision"] == "rethink_required" else 0), review


def review_to_refinement_instruction(review: Dict[str, object], round_index: int, total_rounds: int) -> str:
    reviewer = review.get("top_conference_reviewer", {})
    if not isinstance(reviewer, dict):
        reviewer = {}
    pattern_logic = review.get("pattern_logic_expert", {})
    if not isinstance(pattern_logic, dict):
        pattern_logic = {}
    attacks = compact_list(reviewer.get("main_attacks", []), limit=4, chars=180) if isinstance(reviewer.get("main_attacks", []), list) else []
    flaws = compact_list(reviewer.get("fatal_flaws", []), limit=3, chars=180) if isinstance(reviewer.get("fatal_flaws", []), list) else []
    missing_links = compact_list(pattern_logic.get("missing_logic_links", []), limit=4, chars=180) if isinstance(pattern_logic.get("missing_logic_links", []), list) else []
    required = compact_list(review.get("required_rethink", []), limit=5, chars=180) if isinstance(review.get("required_rethink", []), list) else []
    parts = [
        f"根据第 {round_index}/{total_rounds} 轮顶会审稿意见改写当前 idea。",
        "目标：输出更抗审稿攻击的最终 idea，但必须保留原有 Pattern/Genome/Gold 逻辑空间；不要把用户论文库或趋势论文当评分标准。",
    ]
    if review.get("summary"):
        parts.append("审稿摘要：" + compact_excerpt(str(review.get("summary", "")), limit=280))
    if attacks:
        parts.append("主要攻击：" + "；".join(attacks))
    if flaws:
        parts.append("致命缺陷：" + "；".join(flaws))
    if missing_links:
        parts.append("缺失逻辑链：" + "；".join(missing_links))
    if required:
        parts.append("必须重想：" + "；".join(required))
    if review.get("rethink_trigger"):
        parts.append("触发条件：" + compact_excerpt(str(review.get("rethink_trigger", "")), limit=220))
    parts.append("请给出一版可直接作为最终 idea 的改写，包含核心假设、证据设计、失败边界和最小实验。")
    return "\n".join(parts)


def cmd_review_loop(
    root_dir: Path,
    *,
    session_id: int | None,
    latest: bool,
    rounds: int,
    provider: str | None,
    model: str,
    pattern_limit: int,
    genome_limit: int,
) -> int:
    config = load_config(root_dir)
    ensure_layout(config)
    init_db(config.db_path)
    try:
        resolved_session_id = resolve_session_id(config.root_dir, session_id, latest)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    total_rounds = max(3, min(5, int(rounds or 4)))
    print(f"Review loop session #{resolved_session_id}: {total_rounds} rounds")
    completed_rounds = 0
    review_artifacts: list[str] = []
    for idx in range(1, total_rounds + 1):
        progress_note(f"审稿循环 {idx}/{total_rounds}：调用顶会审稿人")
        review_payload = build_session_reviewer_payload(config.root_dir, resolved_session_id)
        review_code, review = run_reviewer_gate_payload(
            config.root_dir,
            idea_payload=review_payload,
            source_label=f"session-{resolved_session_id}",
            provider=provider,
            model=model,
            pattern_limit=pattern_limit,
            genome_limit=genome_limit,
        )
        if review_code == 1:
            print("Review loop failed before refinement.", file=sys.stderr)
            print(
                "Review-loop failure JSON: "
                + json.dumps(
                    build_review_loop_failure_summary(
                        config.db_path,
                        resolved_session_id,
                        total_rounds,
                        completed_rounds,
                        len(review_artifacts),
                        "reviewer_failed",
                        1,
                    ),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 1
        if review.get("review_artifact_path"):
            review_artifacts.append(str(review.get("review_artifact_path")))
        instruction = review_to_refinement_instruction(review, idx, total_rounds)
        progress_note(f"审稿循环 {idx}/{total_rounds}：按审稿意见改写当前 idea")
        step_code = cmd_session_step(
            config.root_dir,
            session_id=resolved_session_id,
            latest=False,
            user_instruction=instruction,
            provider=provider,
            model=model,
            pattern_limit=pattern_limit,
            genome_limit=genome_limit,
        )
        if step_code != 0:
            print("Review loop stopped because refinement failed.", file=sys.stderr)
            print(
                "Review-loop failure JSON: "
                + json.dumps(
                    build_review_loop_failure_summary(
                        config.db_path,
                        resolved_session_id,
                        total_rounds,
                        completed_rounds,
                        len(review_artifacts),
                        "refinement_failed",
                        step_code,
                    ),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return step_code
        completed_rounds = idx
        if idx >= 3 and str(review.get("decision", "")).strip() == "proceed":
            progress_note("审稿人已给出 proceed，达到最小循环轮数后停止")
            break
    session = get_idea_session(config.db_path, resolved_session_id)
    final_idea = str(session["current_idea"]).strip() if session else ""
    print("Review loop completed.")
    print(
        "Review-loop summary JSON: "
        + json.dumps(
            {
                "session_id": resolved_session_id,
                "rounds_completed": completed_rounds,
                "requested_rounds": total_rounds,
                "review_count": len(review_artifacts),
                "final_idea": final_idea,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def build_review_loop_failure_summary(
    db_path: Path,
    session_id: int,
    requested_rounds: int,
    completed_rounds: int,
    review_count: int,
    reason: str,
    exit_code: int,
) -> Dict[str, object]:
    session = get_idea_session(db_path, int(session_id))
    final_idea = str(session["current_idea"]).strip() if session else ""
    payload = {
        "status": "failed",
        "reason": str(reason or "review_loop_failed"),
        "exit_code": int(exit_code),
        "session_id": int(session_id),
        "rounds_completed": max(0, int(completed_rounds or 0)),
        "requested_rounds": max(0, int(requested_rounds or 0)),
        "review_count": max(0, int(review_count or 0)),
        "final_idea": final_idea,
    }
    payload.update(review_loop_recovery_fields(session_id=int(session_id), completed_rounds=int(completed_rounds or 0)))
    return payload


def review_loop_recovery_fields(*, session_id: int, completed_rounds: int) -> Dict[str, object]:
    partial = int(completed_rounds or 0) > 0
    if partial:
        return {
            "state_policy": "partial_refinements_kept_current_idea_safe_no_raw_review_output",
            "recovery_strategy": "resume_review_loop_from_current_partial_idea",
            "safe_resume_command": f"{CLI_CMD} rl --session-id {int(session_id)}",
            "next_action": "已保留通过前几轮写入的当前 idea；可继续循环审稿，或先用单轮追问修补失败点。",
        }
    return {
        "state_policy": "no_refinement_written_current_idea_unchanged_no_raw_review_output",
        "recovery_strategy": "retry_review_loop_without_state_change",
        "safe_resume_command": f"{CLI_CMD} rl --session-id {int(session_id)}",
        "next_action": "当前 idea 未被审稿循环覆盖；可直接重试循环审稿，若连续失败则先补足 Pattern/Genome/Gold 证据或缩短当前 idea。",
    }


def classify_reviewer_failure(reason: object) -> str:
    text = str(reason or "").lower()
    if "api key" in text or "llm" in text or "provider" in text or "connection" in text or "timeout" in text:
        return "llm_unavailable"
    if "reviewer response" in text or "json" in text or "missing keys" in text or "must be" in text:
        return "invalid_model_response"
    if "no idea found" in text or "idea session not found" in text:
        return "idea_not_found"
    if "evidence" in text or "pattern" in text or "genome" in text:
        return "evidence_not_ready"
    return "reviewer_failed"


def build_reviewer_failure_summary(*, source_label: str, reason: object) -> Dict[str, object]:
    code = classify_reviewer_failure(reason)
    next_actions = {
        "llm_unavailable": "检查模型 API 配置后重试；审稿没有写入任何 review 结果。",
        "invalid_model_response": "模型返回格式不稳定，系统已拒绝保存审稿结果；可直接重试或换模型。",
        "idea_not_found": "先选择有效 idea 或会话，再运行审稿。",
        "evidence_not_ready": "先补核心优秀论文、Genome 和 Pattern 证据，再运行审稿。",
        "reviewer_failed": "审稿没有写入结果；请重试或缩短当前 idea。",
    }
    return {
        "status": "reviewer_failed",
        "reason_code": code,
        "source": str(source_label or "").strip(),
        "reason": compact_excerpt(str(reason or ""), limit=600),
        "state_policy": "no_review_saved_no_session_turn_written",
        "next_action": next_actions.get(code, next_actions["reviewer_failed"]),
    }


def session_id_from_source_label(source_label: str) -> int:
    match = re.match(r"session-(\d+)$", str(source_label or "").strip())
    return int(match.group(1)) if match else 0


def session_artifact_ref(root_dir: Path, output_path: Path) -> str:
    try:
        return str(output_path.resolve().relative_to(root_dir.resolve()))
    except (OSError, ValueError):
        return output_path.name


def compact_reviewer_history_entry(review: Dict[str, object], output_path: Path, root_dir: Path | None = None) -> Dict[str, object]:
    reviewer = review.get("top_conference_reviewer", {})
    if not isinstance(reviewer, dict):
        reviewer = {}
    pattern_logic = review.get("pattern_logic_expert", {})
    if not isinstance(pattern_logic, dict):
        pattern_logic = {}
    score = review.get("evidence_grounded_score", {})
    if not isinstance(score, dict):
        score = {}
    return {
        "decision": str(review.get("decision", "")).strip(),
        "summary": compact_excerpt(str(review.get("summary", "")).strip(), limit=500),
        "rethink_trigger": compact_excerpt(str(review.get("rethink_trigger", "")).strip(), limit=360),
        "required_rethink": compact_list(review.get("required_rethink", []), limit=6, chars=220)
        if isinstance(review.get("required_rethink", []), list)
        else [],
        "fatal_flaws": compact_list(reviewer.get("fatal_flaws", []), limit=5, chars=220)
        if isinstance(reviewer.get("fatal_flaws", []), list)
        else [],
        "main_attacks": compact_list(reviewer.get("main_attacks", []), limit=5, chars=220)
        if isinstance(reviewer.get("main_attacks", []), list)
        else [],
        "missing_logic_links": compact_list(pattern_logic.get("missing_logic_links", []), limit=5, chars=220)
        if isinstance(pattern_logic.get("missing_logic_links", []), list)
        else [],
        "evidence_score": score.get("total", ""),
        "review_artifact": session_artifact_ref(root_dir, output_path) if root_dir else output_path.name,
        "created_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def attach_reviewer_history_to_session(root_dir: Path, session_id: int, review: Dict[str, object], output_path: Path) -> None:
    config = load_config(root_dir)
    session = get_idea_session(config.db_path, int(session_id))
    if not session:
        return
    context = load_session_context(session)
    history = context.get("review_history", [])
    if not isinstance(history, list):
        history = []
    history.append(compact_reviewer_history_entry(review, output_path, config.root_dir))
    context["review_history"] = history[-5:]
    update_idea_session_context(
        config.db_path,
        int(session_id),
        json.dumps(context, ensure_ascii=False, indent=2),
    )
    refresh_session_memory(config.root_dir, int(session_id))
    refresh_session_dossier(config.root_dir, int(session_id))


def cmd_session_create(root_dir: Path, name: str, idea: str) -> int:
    config = load_config(root_dir)
    ensure_layout(config)
    init_db(config.db_path)
    session_id = create_idea_session(config.db_path, name=name, current_idea=idea)
    refresh_session_memory(config.root_dir, session_id)
    _, dossier_path = refresh_session_dossier(config.root_dir, session_id)
    print(f"Created idea session #{session_id}: {name}")
    print(f"Dossier: {dossier_path}")
    return 0


def cmd_list_sessions(root_dir: Path, limit: int) -> int:
    config = load_config(root_dir)
    ensure_layout(config)
    init_db(config.db_path)
    rows = list_idea_sessions(config.db_path, limit=max(1, limit))
    if not rows:
        print("No idea sessions yet. Run `ra is --name ... --idea ...` first.")
        return 0
    for row in rows:
        print(f"[{row['id']}] {row['status']} :: {row['name']} => {row['current_idea']}")
    return 0


def resolve_session_id(root_dir: Path, session_id: int | None, latest: bool) -> int:
    config = load_config(root_dir)
    ensure_layout(config)
    init_db(config.db_path)

    if latest or session_id is None:
        rows = list_idea_sessions(config.db_path, limit=1)
        if not rows:
            raise ValueError("No idea sessions yet. Run `ra ix`, `ra ix --best`, or `ra is --name ... --idea ...` first.")
        return int(rows[0]["id"])
    return int(session_id)


def cmd_session_step(
    root_dir: Path,
    session_id: int | None,
    latest: bool,
    user_instruction: str,
    provider: str | None,
    model: str,
    pattern_limit: int,
    genome_limit: int,
) -> int:
    config = load_config(root_dir)
    ensure_layout(config)
    init_db(config.db_path)
    resolved_instruction = user_instruction.strip()
    if not resolved_instruction:
        print("Need a user instruction. Use `ra st \"...\"` or `ra st --user \"...\"`.", file=sys.stderr)
        return 1
    try:
        resolved_session_id = resolve_session_id(config.root_dir, session_id, latest)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    session = get_idea_session(config.db_path, resolved_session_id)
    if not session:
        print(f"Idea session not found: {resolved_session_id}", file=sys.stderr)
        return 1
    turns = list_idea_session_turns(config.db_path, resolved_session_id)
    progress_note(f"载入会话 #{resolved_session_id}，已有 {len(turns)} 轮历史")
    progress_note("刷新会话记忆并压缩上下文")
    refresh_session_memory(config.root_dir, resolved_session_id)
    session = get_idea_session(config.db_path, resolved_session_id)
    if not session:
        print(f"Idea session not found after memory refresh: {resolved_session_id}", file=sys.stderr)
        return 1
    session_base_context = merge_latest_user_domain_knowledge(
        config.root_dir,
        load_session_context(session),
    )
    if session_base_context.get("domain_knowledge_context"):
        progress_note("载入最新用户论文库领域背景，仅用于术语和路线学习")
    session_context = build_compressed_session_context(
        session,
        turns,
        session_base_context,
    )

    progress_note("调用逻辑专家，根据用户新指令更新 idea")
    try:
        critique_payload = run_critique(
            root_dir=config.root_dir,
            current_idea=session["current_idea"],
            user_instruction=resolved_instruction,
            provider=provider,
            model=model,
            pattern_limit=pattern_limit,
            genome_limit=genome_limit,
            session_context=session_context,
        )
    except (LLMError, ValueError, json.JSONDecodeError) as exc:
        summary = build_session_step_failure_summary(
            session_id=resolved_session_id,
            user_instruction=resolved_instruction,
            reason=str(exc),
        )
        print(str(exc), file=sys.stderr)
        print("Session-step failure JSON: " + json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 1
    content_json = json.dumps(critique_payload, ensure_ascii=False, indent=2)
    progress_note("写入本轮修改、更新记忆和 dossier")
    turn_id = add_idea_session_turn(
        config.db_path,
        session_id=resolved_session_id,
        user_instruction=resolved_instruction,
        decision=critique_payload["decision"],
        revised_idea=critique_payload["revised_idea"],
        content_json=content_json,
    )
    refresh_session_memory(config.root_dir, resolved_session_id)

    session_dir = config.output_dir / "sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    output_path = session_dir / f"session-{resolved_session_id}-turn-{turn_id}.json"
    output_path.write_text(content_json, encoding="utf-8")
    memory = refresh_session_memory(config.root_dir, resolved_session_id)
    dossier, dossier_path = refresh_session_dossier(config.root_dir, resolved_session_id)

    print(f"Turn {turn_id} decision: {critique_payload['decision']}")
    print(f"Revised idea: {critique_payload['revised_idea']}")
    if str(critique_payload.get("trend_alignment", "")).strip():
        print(f"Trend check: {critique_payload['trend_alignment']}")
    print(f"Current best: {dossier['current_best_idea']}")
    print(f"Session memory JSON: {json.dumps(build_session_memory_status(memory), ensure_ascii=False, sort_keys=True)}")
    print(f"Dossier: {dossier_path}")
    print(f"Saved: {output_path}")
    return 0


def classify_session_step_failure(reason: object) -> str:
    text = str(reason or "").lower()
    if "api key" in text or "llm" in text or "provider" in text or "connection" in text or "timeout" in text:
        return "llm_unavailable"
    if (
        "session_step_used_user_library_as_standard" in text
        or "session_step_lost_gold_logic_space" in text
        or "session_step_reused_source_surface_terms" in text
    ):
        return "session_evidence_guard_failed"
    if "content_quality_failed" in text:
        return "invalid_model_response"
    if "critique response" in text or "json" in text or "missing keys" in text or "must contain" in text:
        return "invalid_model_response"
    if "need genome or pattern evidence" in text or "evidence" in text:
        return "evidence_not_ready"
    return "session_step_failed"


def build_session_step_failure_summary(*, session_id: int, user_instruction: str, reason: object) -> Dict[str, object]:
    code = classify_session_step_failure(reason)
    next_actions = {
        "llm_unavailable": "检查模型 API 配置后重试；本轮没有写入对话 turn。",
        "invalid_model_response": "模型返回格式不稳定，系统已拒绝写入本轮；可直接重试或换模型。",
        "evidence_not_ready": "先补 Genome/Pattern 证据，再继续多轮修改；本轮没有写入。",
        "session_evidence_guard_failed": "模型本轮改写越出了优秀论文逻辑空间，或把用户论文库/趋势论文当成了标准；系统已拒绝写入。",
        "session_step_failed": "本轮没有写入对话 turn；请重试或缩短指令。",
    }
    return {
        "status": "session_step_failed",
        "reason_code": code,
        "session_id": int(session_id),
        "instruction_excerpt": compact_excerpt(user_instruction, limit=220),
        "reason": compact_excerpt(str(reason or ""), limit=600),
        "state_policy": "no_turn_written_current_idea_unchanged_memory_safe",
        "next_action": next_actions.get(code, next_actions["session_step_failed"]),
    }


def cmd_session_view(root_dir: Path, session_id: int | None, latest: bool) -> int:
    config = load_config(root_dir)
    ensure_layout(config)
    init_db(config.db_path)
    try:
        resolved_session_id = resolve_session_id(config.root_dir, session_id, latest)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    session = get_idea_session(config.db_path, resolved_session_id)
    if not session:
        print(f"Idea session not found: {resolved_session_id}", file=sys.stderr)
        return 1
    print(f"Session #{session['id']}: {session['name']}")
    print(f"Current idea: {session['current_idea']}")
    memory = load_session_memory(session)
    if memory:
        print(f"Memory: lang={memory.get('preferred_language')} turns={memory.get('turn_count')}")
        if str(memory.get("stable_thesis", "")).strip():
            print(f"Stable thesis: {memory.get('stable_thesis')}")
        print(f"Session memory JSON: {json.dumps(build_session_memory_status(memory), ensure_ascii=False, sort_keys=True)}")
    turns = list_idea_session_turns(config.db_path, resolved_session_id)
    if not turns:
        print("No turns yet.")
        return 0
    for turn in turns:
        print(f"  turn {turn['turn_index']}: {turn['decision']} :: {turn['user_instruction']}")
    return 0


def cmd_session_dossier(root_dir: Path, session_id: int | None, latest: bool) -> int:
    config = load_config(root_dir)
    ensure_layout(config)
    init_db(config.db_path)
    try:
        resolved_session_id = resolve_session_id(config.root_dir, session_id, latest)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    session = get_idea_session(config.db_path, resolved_session_id)
    if not session:
        print(f"Idea session not found: {resolved_session_id}", file=sys.stderr)
        return 1

    dossier, dossier_path = refresh_session_dossier(config.root_dir, resolved_session_id)
    print(f"Session #{dossier['session_id']}: {dossier['name']}")
    print(f"Current best idea: {dossier['current_best_idea']}")
    print(f"Novelty claim: {dossier['novelty_claim']}")
    print(f"Evaluation sketch: {dossier['evaluation_sketch']}")
    print(f"Story hook: {dossier['paper_story_hook']}")
    print(f"Main risk: {dossier['main_risk']}")
    print(f"Trend thesis: {dossier['trend_thesis']}")
    print(f"Prior art: {dossier['prior_art_check']}")
    print(f"Feasibility: {dossier['feasibility']}")
    print(f"Why not: {dossier['why_not_analysis']}")
    print(f"Value: {dossier['value_judgment']}")
    print(f"Target venue: {dossier['target_venue']}")
    print(f"Trajectory: {dossier['trajectory_summary']}")
    print(f"Latest decision: {dossier['latest_decision']}")
    print(f"Keep: {dossier['keep_signal']}")
    print(f"Next: {dossier['next_refinement_target']}")
    print(f"Turns: {dossier['turn_count']}")
    memory = dossier.get("memory_summary", {})
    if isinstance(memory, dict) and memory:
        print(f"Memory: lang={memory.get('preferred_language')} turns={memory.get('turn_count')}")
        print(f"Session memory JSON: {json.dumps(build_session_memory_status(memory), ensure_ascii=False, sort_keys=True)}")
    print(f"Output: {dossier_path}")
    print(f"Markdown: {dossier['markdown_path']}")
    return 0


def cmd_chat(
    root_dir: Path,
    session_id: int | None,
    latest: bool,
    new_direction: str,
    ideas: int,
    remote: str,
    limit: int,
    year: int | None,
    lang: str,
    provider: str | None,
    model: str,
    pattern_limit: int,
    genome_limit: int,
) -> int:
    config = load_config(root_dir)
    ensure_layout(config)
    init_db(config.db_path)
    if new_direction.strip():
        rc = cmd_ideate(
            config.root_dir,
            direction=new_direction,
            ideas=ideas,
            remote=remote,
            limit=limit,
            year=year,
            open_session=True,
            provider=provider,
            model=model,
            lang=lang,
        )
        if rc != 0:
            return rc
        session_id = None
        latest = True
    try:
        resolved_session_id = resolve_session_id(config.root_dir, session_id, latest)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        print(f"Start one directly with: {CLI_CMD} chat --new \"科研智能体可靠评测\" --provider ds", file=sys.stderr)
        return 1

    session = get_idea_session(config.db_path, resolved_session_id)
    if not session:
        print(f"Idea session not found: {resolved_session_id}", file=sys.stderr)
        return 1

    print(f"Research Alpha chat session #{resolved_session_id}: {session['name']}")
    print("Commands: /dossier, /review, /view, /quit")
    print(f"Current idea: {session['current_idea']}")
    while True:
        try:
            user_text = input("ra> ").strip()
        except EOFError:
            print()
            return 0
        if not user_text:
            continue
        command = user_text.lower()
        if command in {"/quit", "/q", "quit", "exit"}:
            return 0
        if command in {"/dossier", "/sd"}:
            rc = cmd_session_dossier(config.root_dir, resolved_session_id, latest=False)
        elif command in {"/view", "/sv"}:
            rc = cmd_session_view(config.root_dir, resolved_session_id, latest=False)
        elif command in {"/review", "/rv"}:
            rc = cmd_reviewer_gate_for_payload(
                config.root_dir,
                idea_payload=build_session_reviewer_payload(config.root_dir, resolved_session_id),
                source_label=f"session-{resolved_session_id}",
                provider=provider,
                model=model,
                pattern_limit=pattern_limit,
                genome_limit=genome_limit,
            )
        else:
            rc = cmd_session_step(
                config.root_dir,
                session_id=resolved_session_id,
                latest=False,
                user_instruction=user_text,
                provider=provider,
                model=model,
                pattern_limit=pattern_limit,
                genome_limit=genome_limit,
            )
        if rc == 2:
            print("Reviewer gate returned rethink_required. Continue refining or type /quit.")
        elif rc != 0:
            print(f"Command returned exit code {rc}. Continue refining or type /quit.", file=sys.stderr)


def cmd_genome_build(
    root_dir: Path,
    limit: int,
    provider: str | None,
    model: str,
    include_existing: bool,
    extractive: bool,
) -> int:
    config = load_config(root_dir)
    ensure_layout(config)
    init_db(config.db_path)

    papers = core_gold_rows(list_papers_for_genome_build(
        config.db_path,
        limit=max(1, limit) * 3,
        only_missing=not include_existing,
        high_weight_only=True,
    ))[: max(1, limit)]
    if not papers:
        print("No candidate papers found for batch genome generation.")
        return 0

    created = 0
    outputs = []
    for paper in papers:
        try:
            _, output_path = generate_genome_for_paper(
                config.root_dir,
                int(paper["id"]),
                provider,
                model,
                extractive=extractive,
            )
        except (LLMError, ValueError, json.JSONDecodeError) as exc:
            print(f"paper #{paper['id']} failed: {exc}", file=sys.stderr)
            continue
        created += 1
        outputs.append((int(paper["id"]), output_path))

    mode = "extractive " if extractive else ""
    print(f"Generated {created} {mode}genome cards.")
    for paper_id, output_path in outputs:
        print(f"  paper #{paper_id}: {output_path}")
    return 0


def load_seed_records(path: Path) -> List[dict]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
        return records
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload
        raise ValueError(f"Expected a JSON list in {path}")
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    raise ValueError(f"Unsupported seed file format: {path.suffix}")


def load_remote_records(
    *,
    remote: str,
    query: str,
    venue: str,
    year: int | None,
    limit: int,
    config,
) -> List[dict]:
    if remote == "openreview":
        if year is None:
            raise ConnectorError("OpenReview harvest currently requires --year.")
        records = harvest_openreview(
            venue=venue,
            year=year,
            limit=limit,
            query=query,
            config=ConnectorConfig(),
        )
        return filter_records_by_query_tokens(records, query, fallback_limit=limit)
    if remote == "icml_awards":
        if year is None:
            raise ConnectorError("ICML awards harvest currently requires --year.")
        if str(venue or "").strip().lower() != "icml":
            raise ConnectorError("ICML awards harvest only supports venue `ICML`.")
        records = harvest_icml_awards(
            year=year,
            limit=limit,
            query="",
        )
        return records[: max(1, int(limit))]
    if remote == "neurips_awards":
        if year is None:
            raise ConnectorError("NeurIPS awards harvest currently requires --year.")
        if str(venue or "").strip().lower() not in {"neurips", "nips"}:
            raise ConnectorError("NeurIPS awards harvest only supports venue `NeurIPS`.")
        records = harvest_neurips_awards(
            year=year,
            limit=limit,
            query="",
        )
        return records[: max(1, int(limit))]
    if remote == "acl_awards":
        if year is None:
            raise ConnectorError("ACL-family awards harvest currently requires --year.")
        if str(venue or "").strip().lower() not in {"acl", "emnlp", "naacl"}:
            raise ConnectorError("ACL-family awards harvest only supports venue `ACL`, `EMNLP`, or `NAACL`.")
        records = harvest_acl_awards(
            year=year,
            limit=limit,
            query="",
            venue=venue,
        )
        return records[: max(1, int(limit))]
    if remote == "cvf_awards":
        if year is None:
            raise ConnectorError("CVF awards harvest currently requires --year.")
        if str(venue or "").strip().lower() not in {"cvpr", "iccv"}:
            raise ConnectorError("CVF awards harvest only supports venue `CVPR` or `ICCV`.")
        records = harvest_cvf_awards(
            venue=venue,
            year=year,
            limit=limit,
            query="",
        )
        return records[: max(1, int(limit))]
    if remote == "eccv_awards":
        if year is None:
            raise ConnectorError("ECCV awards harvest currently requires --year.")
        if str(venue or "").strip().lower() != "eccv":
            raise ConnectorError("ECCV awards harvest only supports venue `ECCV`.")
        records = harvest_eccv_awards(
            year=year,
            limit=limit,
            query="",
        )
        return records[: max(1, int(limit))]
    if remote == "openalex":
        if year is None:
            raise ConnectorError("OpenAlex harvest currently requires --year.")
        records = harvest_openalex(
            venue=venue,
            year=year,
            limit=limit,
            query=query,
            config=ConnectorConfig(
                api_key=config.openalex_api_key,
                email=config.openalex_email,
            ),
        )
        return filter_records_by_query_tokens(records, query, fallback_limit=limit)
    if remote == "s2":
        records = harvest_semantic_scholar(
            query=query,
            year=year,
            venue=venue,
            limit=limit,
            config=ConnectorConfig(api_key=config.semantic_scholar_api_key),
        )
        return filter_records_by_query_tokens(records, query, fallback_limit=limit)
    raise ConnectorError(f"Unsupported remote source: {remote}")


def filter_records_by_query_tokens(records: List[dict], query: str, fallback_limit: int = 0) -> List[dict]:
    query_tokens = query_frontier_tokens(query)
    if not query_tokens:
        return records
    scored = []
    for record in records:
        score = remote_record_query_score(record, query_tokens)
        if score > 0:
            scored.append((score, record))
    scored.sort(key=lambda item: item[0], reverse=True)
    filtered = [record for _, record in scored if remote_record_matches_query(record, query_tokens)]
    if filtered or not fallback_limit:
        return filtered
    return [record for _, record in scored[: max(1, fallback_limit)]]


def remote_record_query_score(record: dict, query_tokens: set[str]) -> float:
    title_tokens = normalized_token_set(record.get("title", ""))
    abstract_tokens = normalized_token_set(record.get("abstract", ""))
    if not title_tokens and not abstract_tokens:
        return 0.0
    title_overlap = query_tokens & title_tokens
    abstract_overlap = query_tokens & abstract_tokens
    intent_tokens = {"agent", "benchmark", "evaluation", "reliable", "robustnes", "failure", "mode"}
    intent_overlap = (title_overlap | abstract_overlap) & intent_tokens
    return len(title_overlap) * 2.0 + len(abstract_overlap) + len(intent_overlap) * 1.5


def remote_record_matches_query(record: dict, query_tokens: set[str]) -> bool:
    title_tokens = normalized_token_set(record.get("title", ""))
    abstract_tokens = normalized_token_set(record.get("abstract", ""))
    text_tokens = title_tokens | abstract_tokens
    if not text_tokens:
        return False
    if len(query_tokens) == 1:
        # A single surviving token such as "agent" is too weak if it appears
        # only incidentally in an abstract. Require title-level intent.
        return bool(query_tokens & title_tokens)
    overlap = query_tokens & text_tokens
    return len(overlap) >= min(2, len(query_tokens)) and remote_record_query_score(record, query_tokens) >= 3.0


def pipeline_harvest_into_db(
    root_dir: Path,
    file_path: str | None,
    remote: str | None,
    query: str,
    venue: str,
    year: int | None,
    limit: int,
    source_kind_override: str = "",
) -> Dict[str, object]:
    config = load_config(root_dir)
    ensure_layout(config)
    init_db(config.db_path)

    if remote:
        records = load_remote_records(
            remote=remote,
            query=query,
            venue=venue,
            year=year,
            limit=limit,
            config=config,
        )
        source = source_kind_override.strip() or frontier_source_kind(remote)
    elif file_path:
        path = Path(file_path)
        if not path.is_absolute():
            path = config.root_dir / path
        records = load_seed_records(path)
        source = source_kind_override.strip() or "seed"
    else:
        return {
            "stage": "harvest",
            "status": "skipped",
            "reason": "no_harvest_input",
        }

    records, skipped_posters = reject_poster_records(records)
    imported = 0
    for record in records[: max(1, limit)]:
        limitations_json = str(record.get("limitations_json", "")).strip()
        if not limitations_json:
            limitations_json = limitations_json_for_paper(record)
        add_paper(
            config.db_path,
            title=str(record.get("title", "")).strip(),
            venue=str(record.get("venue", "")).strip(),
            year=int(record.get("year", 0)),
            abstract=str(record.get("abstract", "")).strip(),
            source_kind=source,
            external_ref=str(record.get("external_ref", "")).strip(),
            award=str(record.get("award", "")).strip(),
            citation_count=int(record.get("citation_count", 0) or 0),
            influential_citation_count=int(record.get("influential_citation_count", 0) or 0),
            publication_date=str(record.get("publication_date", "")).strip(),
            limitations_json=limitations_json,
        )
        imported += 1
    return {
        "stage": "harvest",
        "status": "completed",
        "source": source,
        "imported": imported,
        "skipped_posters": skipped_posters,
    }


def score_papers_in_db(db_path: Path) -> int:
    papers = iter_papers(db_path)
    groups: Dict[tuple[str, int], List[dict]] = {}
    for row in papers:
        paper = dict(row)
        award_key = normalize_award_key(str(paper.get("award", "")))
        source_kind = str(paper.get("source_kind", "")).strip()
        if (
            is_frontier_source_kind(source_kind)
            and award_key in PROMOTABLE_METADATA_AWARDS
            and is_core_gold_source_paper(
                {
                    **paper,
                    "source_kind": gold_source_kind(source_kind[len(FRONTIER_SOURCE_PREFIX):] if source_kind.startswith(FRONTIER_SOURCE_PREFIX) else source_kind),
                }
            )
        ):
            remote_name = source_kind[len(FRONTIER_SOURCE_PREFIX):] if source_kind.startswith(FRONTIER_SOURCE_PREFIX) else source_kind
            promoted_source = gold_source_kind(remote_name)
            update_paper_quality_metadata(
                db_path,
                int(paper["id"]),
                award=award_key,
                source_kind=promoted_source,
            )
            paper["award"] = award_key
            paper["source_kind"] = promoted_source
        groups.setdefault((str(paper["venue"]).lower(), int(paper["year"])), []).append(paper)

    scored = 0
    for cohort in groups.values():
        for paper in cohort:
            signals = compute_signals_for_paper(paper, cohort)
            total = sum(signals.values())
            replace_quality_signals(
                db_path,
                paper_id=int(paper["id"]),
                signals=signals,
                note=build_score_note(signals),
                total_weight=total,
            )
            scored += 1
    return scored


def compute_signals_for_paper(paper: dict, cohort: Iterable[dict]) -> Dict[str, float]:
    if is_user_library_source_kind(paper.get("source_kind", "")):
        return {"user_library_domain_knowledge_only": 0.0}
    if is_frontier_source_kind(paper.get("source_kind", "")):
        return {"frontier_trend_only_not_gold_standard": 0.0}
    award_weights = load_award_weights()
    if is_unverified_remote_gold_candidate(paper):
        return {"unverified_remote_gold_candidate": 0.0}
    if normalize_award_key(str(paper.get("award", ""))) and not is_core_gold_standard_paper(paper):
        return {"non_core_venue_auxiliary_not_gold_standard": 0.0}
    cohort_items = [
        item
        for item in cohort
        if not is_frontier_source_kind(item.get("source_kind", ""))
        and not is_unverified_remote_gold_candidate(item)
        and is_core_gold_standard_paper(item)
    ]
    citation_counts = [int(item.get("citation_count", 0) or 0) for item in cohort_items]
    influential_counts = [int(item.get("influential_citation_count", 0) or 0) for item in cohort_items]

    citation_percentile = percentile_rank(int(paper.get("citation_count", 0) or 0), citation_counts)
    influential_percentile = percentile_rank(
        int(paper.get("influential_citation_count", 0) or 0),
        influential_counts,
    )

    signals: Dict[str, float] = {}
    award_key = normalize_award_key(str(paper.get("award", "")))
    if award_key and award_key in award_weights:
        signals[award_key] = award_weights[award_key]
    if len(cohort_items) >= 3:
        citation_bonus = citation_bonus_for_percentile(citation_percentile)
        if citation_bonus:
            signals["citation_percentile_bonus"] = citation_bonus
        influential_bonus = influential_bonus_for_percentile(influential_percentile)
        if influential_bonus:
            signals["influential_citation_bonus"] = influential_bonus
    elif not signals:
        signals["insufficient_cohort_for_citation_bonus"] = 0.0
    return signals


def load_award_weights() -> Dict[str, float]:
    weights = dict(DEFAULT_AWARD_WEIGHTS)
    path = Path.cwd() / "configs" / "award_signals.yaml"
    if not path.exists():
        return weights
    current_key = ""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped == "signals:":
            continue
        if line.startswith("  ") and stripped.endswith(":") and not stripped.startswith("weight"):
            current_key = stripped[:-1]
            continue
        if current_key and stripped.startswith("weight:"):
            value = stripped.split(":", 1)[1].strip()
            try:
                weights[current_key] = float(value)
            except ValueError:
                pass
    return weights


def normalize_award_key(value: str) -> str:
    key = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "best": "best_paper",
        "best_paper": "best_paper",
        "outstanding": "outstanding_paper",
        "outstanding_paper": "outstanding_paper",
        "nominee": "outstanding_paper",
        "nomination": "outstanding_paper",
        "nominated": "outstanding_paper",
        "best_paper_nominee": "outstanding_paper",
        "best_paper_nomination": "outstanding_paper",
        "best_paper_nominated": "outstanding_paper",
        "honorable_mention": "outstanding_paper",
        "honourable_mention": "outstanding_paper",
        "best_paper_honorable_mention": "outstanding_paper",
        "best_paper_honourable_mention": "outstanding_paper",
        "best_paper_runner_up": "outstanding_paper",
        "runner_up": "outstanding_paper",
        "test_of_time": "test_of_time",
        "high_citation": "high_citation",
        "highly_cited": "high_citation",
        "landmark": "high_citation",
        "oral": "oral",
        "oral_presentation": "oral",
        "conference_oral": "oral",
        "spotlight": "spotlight",
        "spotlight_presentation": "spotlight",
    }
    return aliases.get(key, key)


def percentile_rank(value: int, values: List[int]) -> float:
    if not values:
        return 0.0
    less_or_equal = sum(1 for item in values if item <= value)
    return (less_or_equal / len(values)) * 100.0


def citation_bonus_for_percentile(percentile: float) -> float:
    if percentile >= 99:
        return 3.0
    if percentile >= 95:
        return 2.0
    if percentile >= 90:
        return 1.0
    return 0.0


def influential_bonus_for_percentile(percentile: float) -> float:
    if percentile >= 95:
        return 2.0
    if percentile >= 90:
        return 1.0
    return 0.0


def generate_genome_for_paper(
    root_dir: Path,
    paper_id: int,
    provider: str | None,
    model: str,
    extractive: bool = False,
) -> tuple[int, Path]:
    config = load_config(root_dir)
    ensure_layout(config)
    init_db(config.db_path)

    paper = get_paper_by_id(config.db_path, paper_id)
    if not paper:
        raise ValueError(f"Paper not found: {paper_id}")

    paper_payload = dict(paper)
    if extractive:
        payload = build_extractive_genome_payload(paper_payload)
        return write_genome_payload(config, paper_payload, payload, EXTRACTIVE_EVIDENCE_LEVEL)

    apply_llm_runtime_overrides(config, provider, model)
    client = LLMClient(config.llm)
    prompt = build_genome_prompt(paper_payload)
    progress_note(f"调用论文逻辑专家解析 #{paper_id}，提取 old belief / bottleneck / reframing / evidence")
    response = client.chat(prompt=prompt, system_prompt=GENOME_SYSTEM_PROMPT)
    payload = parse_genome_response(response.text)
    payload["grounding_audit"] = audit_genome_grounding(payload, paper_payload)
    if payload["grounding_audit"]["status"] != "valid":
        raise ValueError(
            "Genome grounding audit failed: "
            + json.dumps(payload["grounding_audit"]["failures"][:3], ensure_ascii=False)
        )

    return write_genome_payload(config, paper_payload, payload, "abstract_only")


def run_pipeline_genome_build(
    root_dir: Path,
    limit: int,
    provider: str | None,
    model: str,
    include_existing: bool,
    extractive: bool = False,
) -> List[Tuple[int, Path]]:
    config = load_config(root_dir)
    ensure_layout(config)
    init_db(config.db_path)

    papers = core_gold_rows(list_papers_for_genome_build(
        config.db_path,
        limit=max(1, limit) * 3,
        only_missing=not include_existing,
        high_weight_only=True,
    ))[: max(1, limit)]
    progress_note(f"准备从 {len(papers)} 篇优秀论文抽取 Genome 逻辑线")
    outputs: List[Tuple[int, Path]] = []
    failures: List[str] = []
    for paper in papers:
        try:
            progress_note(f"抽取论文 #{int(paper['id'])} 的完整逻辑线：{str(paper['title'])[:80]}")
            _, output_path = generate_genome_for_paper(
                config.root_dir,
                int(paper["id"]),
                provider,
                model,
                extractive=extractive,
            )
        except (LLMError, ValueError, json.JSONDecodeError) as exc:
            paper_id = int(paper["id"])
            failures.append(f"paper #{paper_id}: {exc}")
            print(f"Genome generation skipped for paper #{paper_id}: {exc}", file=sys.stderr)
            continue
        outputs.append((int(paper["id"]), output_path))
        progress_note(f"论文 #{int(paper['id'])} Genome 已写入")
    if not outputs and failures:
        raise ValueError(
            "Genome generation failed for every candidate paper. "
            + " | ".join(failures[:3])
        )
    return outputs


def run_pipeline_pattern_build(
    root_dir: Path,
    limit: int,
    provider: str | None,
    model: str,
    extractive: bool = False,
) -> Dict[str, object]:
    config = load_config(root_dir)
    ensure_layout(config)
    init_db(config.db_path)

    cards = core_gold_rows(list_gold_genome_card_payloads(config.db_path, limit=max(2, limit)))
    if len(cards) < 2:
        progress_note("Genome Card 不足 2 张，暂时跳过 Pattern 聚合")
        return {
            "stage": "pattern",
            "status": "skipped",
            "reason": "not_enough_genomes",
        }

    normalized_cards = []
    progress_note(f"聚合 {len(cards)} 张 Gold-only Genome Cards")
    for row in cards:
        payload = json.loads(row["content_json"])
        payload["paper_id"] = row["paper_id"]
        payload["title"] = row["title"]
        payload["venue"] = row["venue"]
        payload["year"] = row["year"]
        normalized_cards.append(payload)

    if extractive:
        payload = build_extractive_pattern_payload(normalized_cards)
    else:
        apply_llm_runtime_overrides(config, provider, model)
        client = LLMClient(config.llm)
        prompt = build_pattern_prompt(normalized_cards)
        progress_note("调用逻辑专家，总结优秀论文可迁移故事线")
        response = client.chat(prompt=prompt, system_prompt=PATTERN_SYSTEM_PROMPT)
        payload = parse_pattern_response(response.text)
    attach_pattern_source_provenance(payload, cards)
    if not extractive:
        payload["grounding_audit"] = audit_pattern_grounding(payload, normalized_cards)
        if payload["grounding_audit"]["status"] != "valid":
            raise ValueError(
                "Pattern grounding audit failed: "
                + json.dumps(payload["grounding_audit"]["failures"][:3], ensure_ascii=False)
            )
    content_json = json.dumps(payload, ensure_ascii=False, indent=2)
    upsert_pattern_card(config.db_path, payload["pattern_key"], content_json)

    pattern_dir = config.output_dir / "patterns"
    pattern_dir.mkdir(parents=True, exist_ok=True)
    output_path = pattern_dir / f"{payload['pattern_key']}.json"
    output_path.write_text(content_json, encoding="utf-8")
    progress_note(f"Pattern Card 已写入：{payload['pattern_key']}")
    return {
        "stage": "pattern",
        "status": "completed",
        "generated": 1,
        "outputs": [str(output_path)],
    }


def build_trend_outputs(root_dir: Path, frontier_years: int, top_k: int) -> Tuple[Dict[str, object], Path, Path]:
    config = load_config(root_dir)
    ensure_layout(config)
    init_db(config.db_path)

    papers = list_trend_papers(config.db_path)
    trend_papers = frontier_rows_for_trends(papers)
    payload = analyze_trends(trend_papers, frontier_years=max(1, frontier_years), top_k=max(1, top_k))
    payload["trend_source_policy"] = (
        "frontier_source_only"
        if len(trend_papers) != len(papers)
        else "all_available_papers_fallback"
    )
    payload["available_papers"] = len(papers)
    trend_dir = config.output_dir / "trends"
    trend_dir.mkdir(parents=True, exist_ok=True)
    report_path = trend_dir / "trend_report.md"
    html_path = trend_dir / "opportunity_map.html"
    report_path.write_text(render_trend_report(payload), encoding="utf-8")
    html_path.write_text(render_opportunity_map_html(payload), encoding="utf-8")
    return payload, report_path, html_path


def frontier_rows_for_trends(rows: List[object]) -> List[object]:
    frontier = [
        row
        for row in rows
        if is_trend_source_kind(row_source_kind(row))
    ]
    return frontier or rows


def row_source_kind(row: object) -> str:
    if isinstance(row, dict):
        return str(row.get("source_kind", "")).strip().lower()
    try:
        return str(dict(row).get("source_kind", "")).strip().lower()
    except (TypeError, ValueError):
        return ""


def has_complete_genome_logic_line(payload: Dict[str, object]) -> bool:
    logic_line = payload.get("logic_line")
    if not isinstance(logic_line, dict):
        return False
    required = ["old_belief", "bottleneck", "reframing", "why_now", "evidence_design", "failure_boundary"]
    return all(str(logic_line.get(key, "")).strip() for key in required)


def has_complete_pattern_logic_line(payload: Dict[str, object]) -> bool:
    logic_line = payload.get("logic_line_pattern")
    if not isinstance(logic_line, dict):
        return False
    required = ["source_logic", "transfer_rule", "why_it_generalizes", "failure_boundary"]
    return all(str(logic_line.get(key, "")).strip() for key in required)


def compute_strict_evidence_readiness(db_path: Path, paper_limit: int) -> Dict[str, object]:
    top_rows = core_gold_rows(list_top_papers(db_path, limit=max(1, paper_limit)))
    genome_rows = core_gold_rows(list_gold_genome_card_payloads(db_path, limit=max(1, paper_limit)))
    pattern_rows = list_pattern_payloads(db_path, limit=max(1, paper_limit))
    high_weight_papers = [
        {
            "paper_id": int(row["id"]),
            "title": str(row["title"]).strip(),
            "venue": str(row["venue"]).strip(),
            "year": int(row["year"]),
            "award": str(row["award"]).strip(),
            "citation_count": int(row["citation_count"]),
            "influential_citation_count": int(row["influential_citation_count"]),
            "paper_weight": float(row["paper_weight"]),
            "quality_signal": quality_signal_label(dict(row)),
        }
        for row in top_rows
        if float(row["paper_weight"]) > 0
    ]
    genome_cards = []
    for row in genome_rows:
        try:
            genome_payload = json.loads(row["content_json"])
        except (json.JSONDecodeError, TypeError):
            genome_payload = {}
        genome_cards.append(
            {
            "card_id": int(row["id"]),
            "paper_id": int(row["paper_id"]),
            "title": str(row["title"]).strip(),
            "venue": str(row["venue"]).strip(),
            "year": int(row["year"]),
            "evidence_level": str(row["evidence_level"]).strip(),
            "logic_line_complete": has_complete_genome_logic_line(genome_payload),
            "grounding_audit_valid": genome_grounding_audit_is_valid(genome_payload),
        }
        )
    strict_genome_cards = [
        card
        for card in genome_cards
        if not str(card.get("evidence_level", "")).startswith("extractive_")
        and bool(card.get("logic_line_complete"))
        and bool(card.get("grounding_audit_valid"))
    ]
    pattern_cards = []
    for row in pattern_rows:
        try:
            pattern_payload = json.loads(row["content_json"])
        except (json.JSONDecodeError, TypeError):
            pattern_payload = {}
        pattern_cards.append(
            {
            "card_id": int(row["id"]),
            "pattern_key": str(row["pattern_key"]).strip(),
            "pattern_name": str(pattern_payload.get("pattern_name", "")).strip(),
            "evidence_level": str(pattern_payload.get("evidence_level", "")).strip(),
            "logic_line_complete": has_complete_pattern_logic_line(pattern_payload),
            "grounding_audit_valid": pattern_grounding_audit_is_valid(pattern_payload),
            "source_set_policy": str(pattern_payload.get("source_set_policy", "")).strip(),
            "gold_only_source_provenance": pattern_source_provenance_is_current_gold_only(db_path, pattern_payload),
            "source_paper_ids": pattern_payload.get("source_paper_ids", []),
        }
        )
    strict_pattern_cards = [
        card
        for card in pattern_cards
        if not str(card.get("evidence_level", "")).startswith("extractive_")
        and bool(card.get("logic_line_complete"))
        and bool(card.get("grounding_audit_valid"))
        and bool(card.get("gold_only_source_provenance"))
    ]
    counts = {
        "high_weight_papers": len(high_weight_papers),
        "genome_cards": len(genome_cards),
        "pattern_cards": len(pattern_cards),
        "strict_genome_cards": len(strict_genome_cards),
        "strict_pattern_cards": len(strict_pattern_cards),
    }
    missing = []
    if counts["high_weight_papers"] < 3:
        missing.append("Add or harvest at least 3 scored core Gold papers with explicit Best/Outstanding/Oral signals.")
    if counts["strict_genome_cards"] < 2:
        missing.append("Generate at least 2 non-extractive Idea Genome Cards with full logic-line fields from core Gold papers.")
    if counts["strict_pattern_cards"] < 1:
        missing.append("Build at least 1 non-extractive Pattern Card from logic-line genome cards.")
    if not missing:
        status = "ready_for_strict_generation"
    elif counts["high_weight_papers"] >= 3:
        status = "partially_ready"
    else:
        status = "not_ready"

    next_commands = []
    if counts["high_weight_papers"] < 3:
        next_commands.append(f"{CLI_CMD} h --file seeds/demo_papers.jsonl")
        next_commands.append(f"{CLI_CMD} score")
    if counts["strict_genome_cards"] < 2 and counts["high_weight_papers"] >= 2:
        next_commands.append(f"{CLI_CMD} gb --limit 5")
    if counts["strict_pattern_cards"] < 1 and counts["strict_genome_cards"] >= 2:
        next_commands.append(f"{CLI_CMD} pb --limit 5")
    if status == "ready_for_strict_generation":
        next_commands.append(f'{CLI_CMD} h --remote s2 --query "research agents" --year 2025 --limit 20')
        next_commands.append(f"{CLI_CMD} tr")
        next_commands.append(f'{CLI_CMD} id "research agents" --ideas 5')
        next_commands.append(f"{CLI_CMD} ir")

    payload: Dict[str, object] = {
        "status": status,
        "counts": counts,
        "missing_requirements": missing,
        "next_commands": next_commands,
        "strict_generation_requirement": (
            "Candidate ideas must cite stored core Gold paper, genome, or pattern evidence. "
            "Genome/Pattern evidence for strict generation must come from non-extractive logic-line cards, not keyword-only extractive drafts. "
            "Pattern Cards must also carry gold_set_only source provenance. "
            "They must also have query-matching frontier evidence in the stored recent corpus. "
            "Missing or unmatched citations are rejected before saving."
        ),
        "high_weight_papers": high_weight_papers,
        "genome_cards": genome_cards,
        "pattern_cards": pattern_cards,
    }
    return payload


def build_evidence_report(root_dir: Path, paper_limit: int) -> Tuple[Dict[str, object], Path, Path]:
    config = load_config(root_dir)
    ensure_layout(config)
    init_db(config.db_path)
    provenance_backfill = backfill_pattern_source_provenance(config.root_dir)
    payload = compute_strict_evidence_readiness(config.db_path, paper_limit)
    payload["pattern_provenance_backfill"] = provenance_backfill
    evidence_dir = config.output_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    json_path = evidence_dir / "evidence_report.json"
    markdown_path = evidence_dir / "evidence_report.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(render_evidence_report_markdown(payload), encoding="utf-8")
    return payload, json_path, markdown_path


def add_audit_failure(failures: List[Dict[str, object]], code: str, message: str, **details: object) -> None:
    payload: Dict[str, object] = {
        "code": code,
        "message": message,
    }
    if details:
        payload["details"] = details
    failures.append(payload)


class IdeaGenerationNotReady(ValueError):
    def __init__(self, message: str, summary: Dict[str, object]):
        super().__init__(message)
        self.summary = summary


def classify_idea_rejection(reason: object) -> str:
    text = str(reason or "").lower()
    if "content_quality_failed" in text:
        return "idea_content_too_thin"
    if "candidate_used_user_library_as_scoring_evidence" in text or "candidate_used_user_library_as_storyline_source" in text:
        return "user_library_used_as_standard"
    if (
        "storyline migration" in text
        or "hard-transfer" in text
        or "transfer_repeats_source_story_standard" in text
        or "visible_idea_reuses_source_surface_terms" in text
    ):
        return "storyline_migration_failed"
    if "prior_art_duplicate_rethink" in text or "duplicate_rethink" in text:
        return "prior_art_too_close"
    if "current-hotspot evidence is not ready" in text:
        return "frontier_evidence_not_ready"
    if "strict evidence audit" in text or "evidence audit" in text or "missing explicit evidence" in text:
        return "gold_evidence_not_ready"
    return "idea_generation_not_ready"


def build_idea_generation_failure_summary(
    *,
    query: str,
    requested: int,
    rejected_ideas: List[Dict[str, str]],
) -> Dict[str, object]:
    codes: List[str] = []
    for item in rejected_ideas:
        code = classify_idea_rejection(item.get("reason", ""))
        if code not in codes:
            codes.append(code)
    primary = codes[0] if codes else "idea_generation_not_ready"
    action_by_code = {
        "idea_content_too_thin": "模型返回了结构合法但没有实质 idea 的空壳内容；系统已拒绝保存。请直接重试或换模型，必要时收窄方向。",
        "storyline_migration_failed": "让生成器重想：先写六步论文故事线，再迁移到当前领域；不要复用源论文方法名、模型名或任务对象。",
        "prior_art_too_close": "换一个更窄的未解矛盾，或补充与现有工作的差异边界后重新生成。",
        "frontier_evidence_not_ready": "补充与当前方向匹配的近期趋势论文，然后重新运行生成。",
        "gold_evidence_not_ready": "补充核心优秀论文、Genome 逻辑卡和 Pattern 卡，让五个评分维度都有显式依据。",
        "user_library_used_as_standard": "保留用户论文库作为领域背景，但重新选择 Gold/Pattern/Genome 里的故事线和评分依据后再生成。",
        "idea_generation_not_ready": "查看被拒原因，补证据或收窄研究方向后重新生成。",
    }
    recovery_by_code = {
        "idea_content_too_thin": {
            "strategy": "retry_with_content_floor",
            "blocked_dimensions": ["substance", "specificity"],
            "retry_prompt_constraints": [
                "每个 idea 必须包含清晰的研究对象、机制假设、首个实验和失败边界。",
                "禁止只给方向词、愿景句或没有实验承诺的空壳标题。",
            ],
        },
        "storyline_migration_failed": {
            "strategy": "storyline_first_retry",
            "blocked_dimensions": ["storyline_migration", "source_surface_leakage"],
            "retry_prompt_constraints": [
                "先抽取优秀论文六步逻辑：旧信念、瓶颈、重构、为什么现在、证据设计、失败边界。",
                "迁移时只复用逻辑角色和论证节奏，不复用源论文方法名、模型名、任务对象或表面术语。",
                "最终 idea 必须显式说明当前领域的新矛盾，而不是把源论文方法替换到新领域。",
            ],
        },
        "prior_art_too_close": {
            "strategy": "novelty_boundary_retry",
            "blocked_dimensions": ["prior_art_boundary"],
            "retry_prompt_constraints": [
                "先写出与最近 prior art 的差异边界，再生成 idea。",
                "如果差异只体现在应用场景替换或指标微调，必须重新选择未解矛盾。",
            ],
        },
        "frontier_evidence_not_ready": {
            "strategy": "frontier_repair_then_retry",
            "blocked_dimensions": ["frontier_evidence"],
            "retry_prompt_constraints": [
                "先补充与当前方向匹配的近期趋势论文。",
                "趋势论文只能支撑问题新近性，不进入 Gold 评分标准。",
            ],
        },
        "gold_evidence_not_ready": {
            "strategy": "gold_logic_repair_then_retry",
            "blocked_dimensions": ["gold_evidence", "genome_cards", "pattern_cards"],
            "retry_prompt_constraints": [
                "先补足核心优秀论文、Genome 逻辑卡和 Pattern 卡。",
                "五个评分维度必须能追溯到 Gold/Pattern/Genome 显式依据。",
            ],
        },
        "user_library_used_as_standard": {
            "strategy": "domain_context_only_retry",
            "blocked_dimensions": ["user_library_boundary"],
            "retry_prompt_constraints": [
                "用户论文库只能作为领域知识和路线学习背景。",
                "不得把用户论文库写入 evidence_basis、storyline_trace.source_id、评分标准或 Gold 来源。",
            ],
        },
        "idea_generation_not_ready": {
            "strategy": "inspect_rejections_then_retry",
            "blocked_dimensions": ["strict_generation_gate"],
            "retry_prompt_constraints": [
                "先读取被拒原因，再补证据、收窄方向或切换模型重试。",
            ],
        },
    }
    recovery = recovery_by_code.get(primary, recovery_by_code["idea_generation_not_ready"])
    return {
        "status": "idea_generation_not_ready",
        "query": query,
        "requested": int(requested),
        "rejected": len(rejected_ideas),
        "primary_reason": primary,
        "reason_codes": codes,
        "next_action": action_by_code.get(primary, action_by_code["idea_generation_not_ready"]),
        "recovery_strategy": recovery["strategy"],
        "blocked_dimensions": recovery["blocked_dimensions"],
        "retry_prompt_constraints": recovery["retry_prompt_constraints"],
        "rejected_ideas": rejected_ideas[:5],
    }


SEMANTIC_RETRYABLE_IDEA_REASONS = {
    "idea_content_too_thin",
    "storyline_migration_failed",
    "user_library_used_as_standard",
}


def idea_generation_failure_is_semantic_retryable(summary: Dict[str, object]) -> bool:
    codes = summary.get("reason_codes")
    if not isinstance(codes, list) or not codes:
        primary = str(summary.get("primary_reason", "")).strip()
        codes = [primary] if primary else []
    return bool(codes) and all(str(code) in SEMANTIC_RETRYABLE_IDEA_REASONS for code in codes)


def required_storyline_trace_complete(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    required = ["old_belief", "bottleneck", "reframing", "why_now", "evidence_design", "failure_boundary"]
    for key in required:
        item = value.get(key)
        if not isinstance(item, dict):
            return False
        for field in ("source_id", "paper_story_standard", "transfer_to_current_hotspot"):
            if not str(item.get(field, "")).strip():
                return False
    return True


STORYLINE_GENERIC_TOKENS = TERM_STOPWORDS | {
    "abstract",
    "assumption",
    "belief",
    "bottleneck",
    "boundary",
    "causal",
    "current",
    "design",
    "domain",
    "evidence",
    "experiment",
    "failure",
    "frontier",
    "hidden",
    "hotspot",
    "paper",
    "problem",
    "reframe",
    "reframing",
    "research",
    "source",
    "standard",
    "story",
    "storyline",
    "target",
    "test",
    "transfer",
}

STORYLINE_VISIBLE_IDEA_FIELDS = [
    "idea_title",
    "core_hypothesis",
    "historical_pattern",
    "frontier_gap",
    "novelty",
    "value",
    "key_risk",
    "first_experiments",
    "evaluation_outline",
    "paper_angle",
]

SOURCE_SURFACE_TOKEN_STOPWORDS = STORYLINE_GENERIC_TOKENS | {
    "agent",
    "agents",
    "analysis",
    "approach",
    "benchmark",
    "benchmarks",
    "case",
    "cases",
    "conference",
    "dataset",
    "datasets",
    "deep",
    "evaluation",
    "experiments",
    "framework",
    "learning",
    "metric",
    "metrics",
    "model",
    "models",
    "neural",
    "paper",
    "propose",
    "proposed",
    "results",
    "tasks",
    "training",
}


def storyline_meaningful_tokens(value: object) -> set[str]:
    return {
        token
        for token in token_set(value)
        if token not in STORYLINE_GENERIC_TOKENS and len(token) >= 4
    }


def storyline_step_looks_copied(paper_story_standard: object, transfer_to_current_hotspot: object) -> bool:
    source = " ".join(normalize_text(paper_story_standard).lower().split())
    transfer = " ".join(normalize_text(transfer_to_current_hotspot).lower().split())
    if not source or not transfer:
        return True
    if source == transfer:
        return True
    similarity = difflib.SequenceMatcher(None, source, transfer).ratio()
    source_tokens = storyline_meaningful_tokens(source)
    transfer_tokens = storyline_meaningful_tokens(transfer)
    if not source_tokens or not transfer_tokens:
        return similarity >= 0.76
    overlap = len(source_tokens & transfer_tokens) / max(1, min(len(source_tokens), len(transfer_tokens)))
    new_tokens = transfer_tokens - source_tokens
    return similarity >= 0.84 or (overlap >= 0.82 and len(new_tokens) < 2)


def source_surface_text(source: Dict[str, object]) -> str:
    fields = [
        "title",
        "abstract",
        "paper_summary",
        "problem_reframing",
        "evidence_design",
        "operator_template",
        "core_move",
        "canonical_examples",
    ]
    logic_line = source.get("logic_line")
    logic_pattern = source.get("logic_line_pattern")
    return " ".join(
        normalize_text(source.get(field))
        for field in fields
        if source.get(field)
    ) + " " + normalize_text(logic_line) + " " + normalize_text(logic_pattern)


def source_surface_tokens(source: Dict[str, object]) -> set[str]:
    aliases = source_aliases(source)
    alias_tokens = {
        token
        for alias in aliases
        for token in token_set(alias)
    }
    return {
        token
        for token in token_set(source_surface_text(source))
        if token not in SOURCE_SURFACE_TOKEN_STOPWORDS
        and token not in alias_tokens
        and len(token) >= 5
    }


def candidate_visible_story_tokens(payload: Dict[str, object]) -> set[str]:
    text = " ".join(normalize_text(payload.get(field)) for field in STORYLINE_VISIBLE_IDEA_FIELDS)
    return {
        token
        for token in token_set(text)
        if token not in STORYLINE_GENERIC_TOKENS
    }


def source_surface_leakage(payload: Dict[str, object], source: Dict[str, object]) -> Dict[str, object] | None:
    if str(source.get("source_type", "")).strip().lower() == "pattern":
        return None
    source_tokens = source_surface_tokens(source)
    if len(source_tokens) < 4:
        return None
    candidate_tokens = candidate_visible_story_tokens(payload)
    leaked = sorted(source_tokens & candidate_tokens)
    if len(leaked) < 4:
        return None
    ratio = len(leaked) / max(1, min(len(source_tokens), len(candidate_tokens)))
    if ratio < 0.38:
        return None
    return {
        "source_id": source_identity(source),
        "matched_surface_tokens": leaked[:10],
        "surface_overlap_ratio": round(ratio, 3),
    }


def audit_storyline_migration(payload: Dict[str, object], sources: List[Dict[str, object]] | None = None) -> Dict[str, object]:
    trace = payload.get("storyline_trace")
    failures: List[Dict[str, str]] = []
    if not required_storyline_trace_complete(trace):
        return {
            "status": "incomplete",
            "valid": False,
            "failures": [{"step": "storyline_trace", "reason": "missing_complete_six_step_trace"}],
            "requirement": "Every accepted idea must migrate a complete six-step paper storyline.",
        }

    source_alias_map = {
        alias
        for source in sources or []
        for alias in source_aliases(source)
    }
    sources_by_alias: Dict[str, List[Dict[str, object]]] = {}
    for source in sources or []:
        for alias in source_aliases(source):
            sources_by_alias.setdefault(alias, []).append(source)
    assert isinstance(trace, dict)
    checked_source_ids: set[str] = set()
    for step, item in trace.items():
        if not isinstance(item, dict):
            failures.append({"step": str(step), "reason": "trace_step_not_object"})
            continue
        source_id = str(item.get("source_id", "")).strip()
        standard = str(item.get("paper_story_standard", "")).strip()
        transfer = str(item.get("transfer_to_current_hotspot", "")).strip()
        if source_alias_map and source_id.lower() not in source_alias_map:
            failures.append({"step": str(step), "reason": "source_id_not_in_gold_story_context", "source_id": source_id})
        else:
            for source in sources_by_alias.get(source_id.lower(), []):
                check_key = f"{str(source.get('source_type', '')).strip()}:{source_identity(source).lower()}"
                if check_key in checked_source_ids:
                    continue
                checked_source_ids.add(check_key)
                leakage = source_surface_leakage(payload, source)
                if leakage:
                    failures.append(
                        {
                            "step": str(step),
                            "reason": "visible_idea_reuses_source_surface_terms",
                            "source_id": str(leakage["source_id"]),
                            "matched_terms": ",".join(str(item) for item in leakage["matched_surface_tokens"]),
                            "surface_overlap_ratio": str(leakage["surface_overlap_ratio"]),
                        }
                    )
                    break
        if storyline_step_looks_copied(standard, transfer):
            failures.append({"step": str(step), "reason": "transfer_repeats_source_story_standard", "source_id": source_id})

    return {
        "status": "storyline_migration_valid" if not failures else "storyline_migration_invalid",
        "valid": not failures,
        "failures": failures,
        "requirement": (
            "Ideas must abstract the source paper's old belief -> bottleneck -> reframing -> why-now -> "
            "evidence-design -> failure-boundary story, then rewrite each step for the user's current domain. "
            "They must not copy the source paper method or reuse the paper story text with only surface nouns swapped."
        ),
    }


def require_storyline_migration_valid(payload: Dict[str, object], sources: List[Dict[str, object]] | None = None) -> None:
    audit = audit_storyline_migration(payload, sources)
    if audit.get("valid"):
        return
    title = str(payload.get("idea_title", "")).strip() or "untitled idea"
    reasons = ", ".join(
        f"{item.get('step')}: {item.get('reason')}"
        for item in audit.get("failures", [])
        if isinstance(item, dict)
    )
    raise ValueError(
        f"Generated idea `{title}` failed storyline migration audit: {reasons or audit.get('status')}. "
        "It must migrate the abstract paper story, not hard-transfer the source method."
    )


def latest_manifest_path(config) -> Path | None:
    runs = list_runs(config.db_path, limit=1)
    if not runs:
        return None
    path = Path(str(runs[0]["manifest_path"]))
    if not path.is_absolute():
        path = config.root_dir / path
    return path if path.exists() else None


def ids_from_idea_output_paths(paths: object) -> List[int]:
    if not isinstance(paths, list):
        return []
    ids = []
    for value in paths:
        match = re.search(r"idea-(\d+)\.json$", str(value))
        if match:
            ids.append(int(match.group(1)))
    return ids


def build_cleanup_plan(root_dir: Path) -> Dict[str, object]:
    config = load_config(root_dir)
    ensure_layout(config)
    init_db(config.db_path)

    pattern_rows_for_score = list_pattern_payloads(config.db_path, limit=max(1000, count_pattern_cards(config.db_path) + 10))
    genome_rows_for_score = core_gold_rows(list_gold_genome_card_payloads(config.db_path, limit=max(1000, count_idea_cards(config.db_path) + 10)))
    paper_rows_for_score = core_gold_rows(list_top_papers(config.db_path, limit=max(1000, len(list_papers(config.db_path, limit=1000)) + 10)))
    rubric_sources = flatten_rubric_sources(
        pattern_rows=pattern_rows_for_score,
        genome_rows=genome_rows_for_score,
        paper_rows=paper_rows_for_score,
    )
    source_aliases_by_id = {
        alias
        for source in rubric_sources
        for alias in source_aliases(source)
    }

    with connect(config.db_path) as conn:
        legacy_non_core_gold_papers = []
        for row in conn.execute(
            """
            SELECT id, title, venue, year, source_kind, award, paper_weight
            FROM papers
            WHERE source_kind LIKE 'gold_%'
            ORDER BY id
            """
        ).fetchall():
            item = dict(row)
            if is_core_gold_standard_paper(item) and float(item.get("paper_weight", 0) or 0) > 0:
                continue
            source_kind = str(item.get("source_kind", "")).strip()
            remote_name = source_kind[len(GOLD_SOURCE_PREFIX):] if source_kind.startswith(GOLD_SOURCE_PREFIX) else source_kind
            legacy_non_core_gold_papers.append(
                {
                    "paper_id": int(item["id"]),
                    "title": str(item["title"]),
                    "venue": str(item["venue"]),
                    "year": int(item["year"]),
                    "source_kind": source_kind,
                    "target_source_kind": frontier_source_kind(remote_name),
                    "award": str(item.get("award", "")),
                    "paper_weight": float(item.get("paper_weight", 0) or 0),
                    "reason": "gold_source_not_core_gold_standard",
                }
            )

        non_gold_genomes = [
            {
                "idea_card_id": int(row["id"]),
                "paper_id": int(row["paper_id"]),
                "title": str(row["title"]),
                "paper_weight": float(row["paper_weight"] or 0),
                "source_kind": str(row["source_kind"]),
                "content_json": str(row["content_json"]),
            }
            for row in conn.execute(
                """
                SELECT
                    idea_cards.id,
                    idea_cards.paper_id,
                    idea_cards.content_json,
                    papers.title,
                    papers.paper_weight,
                    papers.source_kind
                FROM idea_cards
                JOIN papers ON papers.id = idea_cards.paper_id
                WHERE papers.paper_weight <= 0
                ORDER BY idea_cards.id
                """
            ).fetchall()
        ]

        non_gold_patterns = []
        for row in conn.execute("SELECT id, pattern_key, content_json FROM pattern_cards ORDER BY id").fetchall():
            try:
                payload = json.loads(row["content_json"])
            except (json.JSONDecodeError, TypeError):
                continue
            if not is_strict_pattern_payload(payload):
                continue
            if pattern_source_provenance_is_current_gold_only(config.db_path, payload):
                continue
            non_gold_patterns.append(
                {
                    "pattern_card_id": int(row["id"]),
                    "pattern_key": str(row["pattern_key"]),
                    "reason": "strict_pattern_provenance_not_gold_only",
                    "content_json": str(row["content_json"]),
                }
            )

        bad_pattern_keys = {str(row["pattern_key"]).strip().lower() for row in non_gold_patterns}
        bad_genome_ids = {str(row["paper_id"]).strip().lower() for row in non_gold_genomes}
        bad_titles = {str(row["title"]).strip().lower() for row in non_gold_genomes}
        bad_aliases = bad_pattern_keys | bad_genome_ids | bad_titles
        candidate_rows = []
        for row in conn.execute("SELECT id, query, title, content_json FROM candidate_ideas ORDER BY id").fetchall():
            try:
                payload = json.loads(row["content_json"])
            except (json.JSONDecodeError, TypeError):
                continue
            basis = normalize_explicit_evidence_basis(payload.get("evidence_basis"))
            reasons = []
            for item in basis:
                source_id = str(item.get("source_id", "")).strip().lower()
                if source_id and source_id in bad_aliases:
                    reasons.append(f"references_non_gold_source:{source_id}")
                elif source_id and source_id not in source_aliases_by_id:
                    reasons.append(f"source_not_in_current_gold_context:{source_id}")
            if reasons:
                candidate_rows.append(
                    {
                        "candidate_idea_id": int(row["id"]),
                        "query": str(row["query"]),
                        "title": str(row["title"]),
                        "reason": ", ".join(dedupe_cleanup_reasons(reasons)),
                        "content_json": str(row["content_json"]),
                    }
                )

    return {
        "legacy_non_core_gold_papers": legacy_non_core_gold_papers,
        "non_gold_genome_cards": non_gold_genomes,
        "non_gold_pattern_cards": non_gold_patterns,
        "candidate_ideas_outside_gold_context": candidate_rows,
    }


def dedupe_cleanup_reasons(values: Iterable[str]) -> List[str]:
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


def apply_cleanup_plan(root_dir: Path, plan: Dict[str, object]) -> Path:
    config = load_config(root_dir)
    ensure_layout(config)
    init_db(config.db_path)
    cleanup_dir = config.output_dir / "cleanup"
    cleanup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = cleanup_dir / f"cleanup-{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
    backup_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    idea_card_ids = [int(item["idea_card_id"]) for item in plan.get("non_gold_genome_cards", []) if isinstance(item, dict)]
    pattern_ids = [int(item["pattern_card_id"]) for item in plan.get("non_gold_pattern_cards", []) if isinstance(item, dict)]
    candidate_ids = [int(item["candidate_idea_id"]) for item in plan.get("candidate_ideas_outside_gold_context", []) if isinstance(item, dict)]
    legacy_papers = [item for item in plan.get("legacy_non_core_gold_papers", []) if isinstance(item, dict)]

    with connect(config.db_path) as conn:
        for item in legacy_papers:
            target_source_kind = str(item.get("target_source_kind", "")).strip()
            if not target_source_kind:
                continue
            conn.execute(
                """
                UPDATE papers
                SET source_kind = ?, award = '', paper_weight = 0, score_notes = ?
                WHERE id = ?
                """,
                (
                    target_source_kind,
                    build_score_note({"legacy_non_core_gold_downgraded_to_frontier": 0.0}),
                    int(item["paper_id"]),
                ),
            )
        for item_id in idea_card_ids:
            conn.execute("DELETE FROM idea_cards WHERE id = ?", (item_id,))
        for item_id in pattern_ids:
            conn.execute("DELETE FROM pattern_cards WHERE id = ?", (item_id,))
        for item_id in candidate_ids:
            conn.execute("DELETE FROM candidate_ideas WHERE id = ?", (item_id,))
    return backup_path


def build_strict_evidence_audit(
    root_dir: Path,
    *,
    paper_limit: int = 12,
    check_latest_run: bool = True,
) -> Tuple[Dict[str, object], Path, Path]:
    config = load_config(root_dir)
    ensure_layout(config)
    init_db(config.db_path)

    evidence_payload, _, _ = build_evidence_report(config.root_dir, paper_limit=max(1, paper_limit))
    failures: List[Dict[str, object]] = []
    checks: List[Dict[str, object]] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    counts = evidence_payload.get("counts", {})
    ready = evidence_payload.get("status") == "ready_for_strict_generation"
    record("evidence_ready_for_strict_generation", ready, str(evidence_payload.get("status")))
    if not ready:
        add_audit_failure(
            failures,
            "evidence_not_ready",
            "Evidence report is not ready for strict generation.",
            status=evidence_payload.get("status"),
            missing=evidence_payload.get("missing_requirements", []),
        )

    high_weight_count = int(counts.get("high_weight_papers", 0) or 0) if isinstance(counts, dict) else 0
    record("gold_set_has_at_least_three_papers", high_weight_count >= 3, f"high_weight_papers={high_weight_count}")
    if high_weight_count < 3:
        add_audit_failure(failures, "insufficient_gold_set", "Need at least three core Gold Set papers with explicit Best/Outstanding/Oral signals.")

    with connect(config.db_path) as conn:
        legacy_non_core_gold_papers = []
        for row in conn.execute(
            """
            SELECT id, title, venue, year, source_kind, award, paper_weight
            FROM papers
            WHERE source_kind LIKE 'gold_%'
            ORDER BY id
            """
        ).fetchall():
            item = dict(row)
            if is_core_gold_standard_paper(item) and float(item.get("paper_weight", 0) or 0) > 0:
                continue
            legacy_non_core_gold_papers.append(item)
        record(
            "gold_paper_records_are_core_positive_gold_only",
            not legacy_non_core_gold_papers,
            f"violations={len(legacy_non_core_gold_papers)}",
        )
        for item in legacy_non_core_gold_papers:
            add_audit_failure(
                failures,
                "legacy_non_core_gold_paper",
                "Paper is stored under a gold_* source but does not satisfy the current core Gold standard.",
                row=item,
            )

        non_gold_genomes = list(
            conn.execute(
                """
                SELECT idea_cards.id, idea_cards.paper_id, papers.title, papers.paper_weight, papers.source_kind
                FROM idea_cards
                JOIN papers ON papers.id = idea_cards.paper_id
                WHERE papers.paper_weight <= 0
                ORDER BY idea_cards.id
                """
            ).fetchall()
        )
        record("genome_cards_are_gold_only", not non_gold_genomes, f"violations={len(non_gold_genomes)}")
        for row in non_gold_genomes:
            add_audit_failure(
                failures,
                "non_gold_genome_card",
                "Idea Genome Card is attached to a non-Gold paper.",
                row=dict(row),
            )

        pattern_rows = list(conn.execute("SELECT id, pattern_key, content_json FROM pattern_cards ORDER BY id").fetchall())
        strict_patterns = 0
        for row in pattern_rows:
            try:
                payload = json.loads(row["content_json"])
            except (json.JSONDecodeError, TypeError):
                add_audit_failure(
                    failures,
                    "invalid_pattern_json",
                    "Pattern Card content_json is not valid JSON.",
                    id=int(row["id"]),
                    pattern_key=str(row["pattern_key"]),
                )
                continue
            if not is_strict_pattern_payload(payload):
                continue
            strict_patterns += 1
            if not pattern_source_provenance_is_current_gold_only(config.db_path, payload):
                add_audit_failure(
                    failures,
                    "pattern_provenance_not_gold_only",
                    "Strict Pattern Card lacks complete current core Gold provenance.",
                    id=int(row["id"]),
                    pattern_key=str(row["pattern_key"]),
                    source_set_policy=payload.get("source_set_policy"),
                    source_paper_ids=payload.get("source_paper_ids", []),
                )
            for source in payload.get("source_papers") or []:
                title = str(source.get("title", "")).strip()
                if float(source.get("paper_weight", 0) or 0) <= 0:
                    add_audit_failure(
                        failures,
                        "pattern_source_not_gold",
                        "Strict Pattern Card uses a source paper with paper_weight <= 0.",
                        id=int(row["id"]),
                        pattern_key=str(row["pattern_key"]),
                        source=source,
                    )
                if title.lower() == "the rise and potential of large language model based agents: a survey":
                    add_audit_failure(
                        failures,
                        "frontier_survey_used_as_pattern_source",
                        "Frontier survey is being used as a Pattern Card source.",
                        id=int(row["id"]),
                        pattern_key=str(row["pattern_key"]),
                    )
        pattern_failures = [
            item for item in failures
            if str(item.get("code", "")).startswith("pattern_") or item.get("code") == "frontier_survey_used_as_pattern_source"
        ]
        record("strict_pattern_cards_have_gold_provenance", not pattern_failures, f"strict_patterns={strict_patterns}")

        idea_rows = list(conn.execute("SELECT id, query, title, content_json FROM candidate_ideas ORDER BY id").fetchall())

    pattern_rows_for_score = list_pattern_payloads(config.db_path, limit=max(1000, count_pattern_cards(config.db_path) + 10))
    genome_rows_for_score = core_gold_rows(list_gold_genome_card_payloads(config.db_path, limit=max(1000, count_idea_cards(config.db_path) + 10)))
    paper_rows_for_score = core_gold_rows(list_top_papers(config.db_path, limit=max(1000, len(list_papers(config.db_path, limit=1000)) + 10)))
    rubric_sources = flatten_rubric_sources(
        pattern_rows=pattern_rows_for_score,
        genome_rows=genome_rows_for_score,
        paper_rows=paper_rows_for_score,
    )

    for row in idea_rows:
        try:
            payload = json.loads(row["content_json"])
        except (json.JSONDecodeError, TypeError):
            add_audit_failure(
                failures,
                "invalid_candidate_json",
                "Candidate idea content_json is not valid JSON.",
                id=int(row["id"]),
                title=str(row["title"]),
            )
            continue
        score_payload = payload.get("evidence_grounded_score")
        if not isinstance(score_payload, dict):
            score_payload = build_evidence_grounded_score(
                payload,
                pattern_rows=pattern_rows_for_score,
                genome_rows=genome_rows_for_score,
                paper_rows=paper_rows_for_score,
            )
        audit = payload.get("generation_evidence_audit")
        if not isinstance(audit, dict):
            audit = score_payload.get("generation_evidence_audit")
        if not isinstance(audit, dict):
            audit = {}
        if audit.get("status") != "explicit_valid":
            add_audit_failure(
                failures,
                "candidate_generation_audit_not_explicit_valid",
                "Saved candidate idea is not explicitly grounded in matched evidence.",
                id=int(row["id"]),
                title=str(row["title"]),
                status=audit.get("status"),
            )
        if audit.get("missing_uses") not in ([], None):
            add_audit_failure(
                failures,
                "candidate_missing_scoring_dimensions",
                "Saved candidate idea does not explicitly cover all scoring dimensions.",
                id=int(row["id"]),
                title=str(row["title"]),
                missing_uses=audit.get("missing_uses"),
            )
        if int(audit.get("matched_basis_count", 0) or 0) <= 0:
            add_audit_failure(
                failures,
                "candidate_no_matched_basis",
                "Saved candidate idea has no matched evidence basis.",
                id=int(row["id"]),
                title=str(row["title"]),
            )
        if not required_storyline_trace_complete(payload.get("storyline_trace")):
            add_audit_failure(
                failures,
                "candidate_storyline_trace_incomplete",
                "Saved candidate idea lacks a complete six-step storyline trace.",
                id=int(row["id"]),
                title=str(row["title"]),
            )
        story_audit = audit_storyline_migration(payload, rubric_sources)
        if not story_audit.get("valid"):
            add_audit_failure(
                failures,
                "candidate_storyline_migration_invalid",
                "Saved candidate idea appears to hard-transfer a source method instead of migrating the source paper storyline.",
                id=int(row["id"]),
                title=str(row["title"]),
                failures=story_audit.get("failures", []),
            )
        if "large language model based agents: a survey" in json.dumps(score_payload, ensure_ascii=False).lower():
            add_audit_failure(
                failures,
                "frontier_survey_in_evidence_score",
                "Frontier survey appears inside evidence-grounded scoring payload.",
                id=int(row["id"]),
                title=str(row["title"]),
            )
        basis = normalize_explicit_evidence_basis(payload.get("evidence_basis"))
        source_aliases_by_id = {
            alias
            for source in rubric_sources
            for alias in source_aliases(source)
        }
        for item in basis:
            source_id = str(item.get("source_id", "")).strip().lower()
            if source_id and source_id not in source_aliases_by_id:
                add_audit_failure(
                    failures,
                    "candidate_basis_source_not_in_gold_context",
                    "Candidate idea cites an evidence_basis source that is not in the Gold scoring context.",
                    id=int(row["id"]),
                    title=str(row["title"]),
                    source_id=item.get("source_id"),
                )

    record("candidate_ideas_are_explicitly_grounded", not any(
        item.get("code", "").startswith("candidate_") or item.get("code") == "frontier_survey_in_evidence_score"
        for item in failures
    ), f"candidate_ideas={len(idea_rows)}")

    latest_run_summary: Dict[str, object] = {}
    if check_latest_run:
        manifest_path = latest_manifest_path(config)
        if manifest_path is None:
            add_audit_failure(failures, "latest_run_missing", "No latest run manifest is available.")
            record("latest_run_consistency", False, "missing manifest")
        else:
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                add_audit_failure(
                    failures,
                    "latest_run_manifest_invalid",
                    "Latest run manifest could not be parsed.",
                    manifest_path=str(manifest_path),
                    error=str(exc),
                )
                manifest = {}
            stages = manifest.get("stages", [])
            ideas_stage = next(
                (item for item in stages if isinstance(item, dict) and item.get("stage") == "ideas" and item.get("status") == "completed"),
                {},
            )
            generated_ids = ids_from_idea_output_paths(ideas_stage.get("outputs", [])) if isinstance(ideas_stage, dict) else []
            ranking = manifest.get("ranking")
            if not isinstance(ranking, dict):
                ranking = next(
                    (item for item in stages if isinstance(item, dict) and item.get("stage") == "ranking"),
                    {},
                )
            top_pick_id = int(ranking.get("top_pick_id", 0) or 0) if isinstance(ranking, dict) else 0
            auto_session = manifest.get("auto_session")
            session_candidate_id = int(auto_session.get("candidate_idea_id", 0) or 0) if isinstance(auto_session, dict) else 0
            latest_run_summary = {
                "manifest_path": str(manifest_path),
                "status": manifest.get("status"),
                "generated_ids": generated_ids,
                "top_pick_id": top_pick_id,
                "session_candidate_idea_id": session_candidate_id,
            }
            has_completed_ranking = isinstance(ranking, dict) and str(ranking.get("status", "")).strip() == "completed"
            has_ranking_or_session = bool(top_pick_id or session_candidate_id or has_completed_ranking)
            if not has_ranking_or_session:
                record(
                    "latest_run_ranking_and_session_scoped_to_current_ideas",
                    True,
                    json.dumps({**latest_run_summary, "skipped": "latest_run_has_no_ranking_or_session"}, ensure_ascii=False),
                )
            else:
                run_ok = bool(generated_ids) and top_pick_id in generated_ids
                if session_candidate_id:
                    run_ok = run_ok and session_candidate_id == top_pick_id
                record("latest_run_ranking_and_session_scoped_to_current_ideas", run_ok, json.dumps(latest_run_summary, ensure_ascii=False))
            if has_ranking_or_session and not run_ok:
                add_audit_failure(
                    failures,
                    "latest_run_scope_mismatch",
                    "Latest run ranking/session is not scoped to the ideas generated in that run.",
                    **latest_run_summary,
                )

    payload: Dict[str, object] = {
        "ok": not failures,
        "check_count": len(checks),
        "checks": checks,
        "failures": failures,
        "evidence_status": evidence_payload.get("status"),
        "evidence_counts": evidence_payload.get("counts", {}),
        "candidate_idea_count": len(idea_rows),
        "latest_run": latest_run_summary,
        "requirement": (
            "Gold Set papers and their complete logic-line Genome/Pattern Cards define generation and scoring standards. "
            "Frontier papers may support trend/query alignment only and must not enter evidence_basis or evidence_grounded_score."
        ),
    }
    audit_dir = config.output_dir / "evidence"
    audit_dir.mkdir(parents=True, exist_ok=True)
    json_path = audit_dir / "evidence_audit.json"
    markdown_path = audit_dir / "evidence_audit.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(render_strict_evidence_audit_markdown(payload), encoding="utf-8")
    return payload, json_path, markdown_path


def render_strict_evidence_audit_markdown(payload: Dict[str, object]) -> str:
    lines = [
        "# Strict Evidence Audit",
        "",
        f"- ok: {payload.get('ok')}",
        f"- checks: {payload.get('check_count', 0)}",
        f"- failures: {len(payload.get('failures', []))}",
        f"- evidence_status: {payload.get('evidence_status')}",
        f"- candidate_idea_count: {payload.get('candidate_idea_count', 0)}",
        "",
        "## Requirement",
        str(payload.get("requirement", "")).strip(),
        "",
        "## Checks",
    ]
    checks = payload.get("checks", [])
    if isinstance(checks, list) and checks:
        for item in checks:
            if not isinstance(item, dict):
                continue
            marker = "PASS" if item.get("passed") else "FAIL"
            lines.append(f"- {marker} {item.get('name')}: {item.get('detail')}")
    else:
        lines.append("- None.")
    lines.extend(["", "## Failures"])
    failures = payload.get("failures", [])
    if isinstance(failures, list) and failures:
        for item in failures:
            if not isinstance(item, dict):
                continue
            lines.append(f"- {item.get('code')}: {item.get('message')}")
    else:
        lines.append("- None.")
    latest_run = payload.get("latest_run", {})
    if isinstance(latest_run, dict) and latest_run:
        lines.extend(["", "## Latest Run"])
        for key in ("manifest_path", "status", "generated_ids", "top_pick_id", "session_candidate_idea_id"):
            lines.append(f"- {key}: {latest_run.get(key)}")
    return "\n".join(lines).strip() + "\n"


def render_evidence_report_markdown(payload: Dict[str, object]) -> str:
    counts = payload.get("counts", {})
    if not isinstance(counts, dict):
        counts = {}
    lines = [
        "# Evidence Readiness Report",
        "",
        f"- status: {payload.get('status')}",
        f"- high_weight_papers: {counts.get('high_weight_papers', 0)}",
        f"- genome_cards: {counts.get('genome_cards', 0)}",
        f"- pattern_cards: {counts.get('pattern_cards', 0)}",
        f"- strict_genome_cards: {counts.get('strict_genome_cards', 0)}",
        f"- strict_pattern_cards: {counts.get('strict_pattern_cards', 0)}",
        "",
        "## Strict Generation Requirement",
        str(payload.get("strict_generation_requirement", "")).strip(),
        "",
        "## Missing Requirements",
    ]
    missing = payload.get("missing_requirements", [])
    if isinstance(missing, list) and missing:
        lines.extend(f"- {item}" for item in missing)
    else:
        lines.append("- None.")
    lines.extend(["", "## Core Gold Papers"])
    papers = payload.get("high_weight_papers", [])
    if isinstance(papers, list) and papers:
        for paper in papers:
            if not isinstance(paper, dict):
                continue
            lines.append(
                f"- [{paper.get('paper_id')}] score={paper.get('paper_weight')} "
                f"{paper.get('year')} {paper.get('venue')} {paper.get('quality_signal')}: {paper.get('title')}"
            )
    else:
        lines.append("- None.")
    lines.extend(["", "## Genome Cards"])
    genomes = payload.get("genome_cards", [])
    if isinstance(genomes, list) and genomes:
        for card in genomes:
            if not isinstance(card, dict):
                continue
            lines.append(f"- [{card.get('card_id')}] paper={card.get('paper_id')} {card.get('title')}")
    else:
        lines.append("- None.")
    lines.extend(["", "## Pattern Cards"])
    patterns = payload.get("pattern_cards", [])
    if isinstance(patterns, list) and patterns:
        for card in patterns:
            if not isinstance(card, dict):
                continue
            lines.append(f"- [{card.get('card_id')}] {card.get('pattern_key')} :: {card.get('pattern_name')}")
    else:
        lines.append("- None.")
    lines.extend(["", "## Next Commands"])
    next_commands = payload.get("next_commands", [])
    if isinstance(next_commands, list) and next_commands:
        lines.extend(f"- `{command}`" for command in next_commands)
    else:
        lines.append("- None.")
    return "\n".join(lines).strip() + "\n"


def run_pipeline_trend_stage(
    root_dir: Path,
    frontier_years: int,
    top_k: int,
) -> Dict[str, object]:
    config = load_config(root_dir)
    ensure_layout(config)
    init_db(config.db_path)

    papers = list_trend_papers(config.db_path)
    if not papers:
        return {
            "stage": "trends",
            "status": "skipped",
            "reason": "no_papers_available",
        }

    payload, report_path, html_path = build_trend_outputs(
        config.root_dir,
        frontier_years=max(1, frontier_years),
        top_k=max(1, top_k),
    )
    return {
        "stage": "trends",
        "status": "completed",
        "paper_count": int(payload.get("paper_count", 0)),
        "outputs": [str(report_path), str(html_path)],
    }


def run_pipeline_idea_generation(
    *,
    root_dir: Path,
    query: str,
    idea_count: int,
    pattern_limit: int,
    genome_limit: int,
    provider: str | None,
    model: str,
    frontier_query: str = "",
    extractive: bool = False,
    semantic_retry_summary: Dict[str, object] | None = None,
) -> List[Dict[str, object]]:
    config = load_config(root_dir)
    ensure_layout(config)
    init_db(config.db_path)
    if extractive:
        raise ValueError(
            "Extractive mode is evidence-prep only for idea generation. "
            "Run `ra gb --extractive` and `ra pb --extractive` to build auditable drafts, "
            "then use non-extractive `ra gb`, `ra pb`, and `ra id` once the LLM can extract complete paper logic lines."
        )

    pattern_rows = list_strict_pattern_rows(config.db_path, limit=max(1, pattern_limit))
    genome_rows = list_strict_gold_genome_rows(config.db_path, limit=max(1, genome_limit))
    paper_rows = list_trend_papers(config.db_path)
    progress_note(f"载入证据：Pattern {len(pattern_rows)} 张，Genome {len(genome_rows)} 张，论文 {len(paper_rows)} 篇")
    if not pattern_rows and not genome_rows:
        raise ValueError("Need genome or pattern evidence before generating ideas. Run `ra gb` and `ra pb` first.")
    progress_note("执行严格证据就绪检查，确保生成不依赖模型先验")
    require_evidence_ready(
        config.root_dir,
        paper_limit=max(5, pattern_limit + genome_limit),
        allow_extractive=extractive,
    )

    patterns = []
    for row in pattern_rows:
        payload = json.loads(row["content_json"])
        payload["pattern_key"] = row["pattern_key"]
        patterns.append(payload)

    genomes = []
    for row in genome_rows:
        payload = json.loads(row["content_json"])
        payload["paper_id"] = row["paper_id"]
        payload["title"] = row["title"]
        payload["venue"] = row["venue"]
        payload["year"] = row["year"]
        genomes.append(payload)

    progress_note("构建近期热点趋势和近半年局限性上下文")
    trend_payload, _, _ = build_trend_outputs(config.root_dir, frontier_years=5, top_k=max(20, pattern_limit + genome_limit))
    limitation_payload, _, _ = build_limitations_report(config.root_dir, months=6, limit=max(8, pattern_limit + genome_limit))
    trend_signal = {
        "latest_year": trend_payload.get("latest_year"),
        "recent_papers": trend_payload.get("recent_papers"),
        "top_terms": trend_payload.get("top_terms", []),
        "trend_clusters": trend_payload.get("trend_clusters", []),
        "recent_limitations": limitation_payload,
    }
    audit_query = frontier_query.strip() or scholarly_search_query(query)
    progress_note(f"核查用户方向是否有近期论文证据支撑：{audit_query}")
    frontier_audit = require_query_frontier_evidence(audit_query, trend_signal, [dict(row) for row in paper_rows])
    if extractive:
        ideas = build_extractive_candidate_ideas(
            query=query,
            idea_count=max(1, idea_count),
            pattern_rows=pattern_rows,
            genome_rows=genome_rows,
            paper_rows=core_gold_rows(list_top_papers(config.db_path, limit=max(5, pattern_limit + genome_limit))),
            trend_signal=trend_signal,
        )
        for item in ideas:
            item["frontier_evidence_audit"] = frontier_audit
        return ideas

    progress_note("整理优秀论文来源、逻辑线和评分标准摘要")
    quality_evidence = build_quality_evidence_summary(
        config.root_dir,
        pattern_limit=pattern_limit,
        genome_limit=genome_limit,
        paper_limit=max(5, pattern_limit + genome_limit),
    )
    domain_knowledge = build_user_domain_knowledge_summary(
        config.root_dir,
        limit=max(5, min(12, pattern_limit + genome_limit)),
    )
    user_paper_count = len(domain_knowledge.get("papers", [])) if isinstance(domain_knowledge.get("papers"), list) else 0
    if user_paper_count:
        progress_note(f"载入用户论文库领域背景 {user_paper_count} 篇，仅用于术语和路线学习")

    apply_llm_runtime_overrides(config, provider, model)
    client = LLMClient(config.llm)
    prompt = build_idea_prompt(
        query=query,
        idea_count=max(1, idea_count),
        pattern_cards=patterns,
        genome_cards=genomes,
        trend_signal=trend_signal,
        quality_evidence=quality_evidence,
        domain_knowledge=domain_knowledge,
    )
    request_prompt = (
        build_idea_semantic_retry_prompt(original_prompt=prompt, failure_summary=semantic_retry_summary)
        if semantic_retry_summary
        else prompt
    )
    if semantic_retry_summary:
        progress_note("严格审计拒绝了上一批 idea，执行一次受约束语义重试")
    else:
        progress_note("调用 idea 生成专家，要求输出依据链、局限性信号和证据评分")
    response = client.chat(prompt=request_prompt, system_prompt=IDEA_SYSTEM_PROMPT)
    ideas = parse_llm_response_with_repair(
        client=client,
        original_prompt=request_prompt,
        system_prompt=IDEA_SYSTEM_PROMPT,
        original_text=response.text,
        parser=parse_idea_response,
        response_name="Idea response",
    )
    progress_note(f"收到 {len(ideas)} 个 idea 草案，准备进入 prior-art 与证据审计")
    return attach_idea_generation_context(
        ideas,
        frontier_audit=frontier_audit,
        domain_knowledge=domain_knowledge,
        semantic_retry_summary=semantic_retry_summary,
    )


def attach_idea_generation_context(
    ideas: List[Dict[str, object]],
    *,
    frontier_audit: Dict[str, object],
    domain_knowledge: Dict[str, object],
    semantic_retry_summary: Dict[str, object] | None = None,
) -> List[Dict[str, object]]:
    user_paper_count = len(domain_knowledge.get("papers", [])) if isinstance(domain_knowledge.get("papers"), list) else 0
    for item in ideas:
        item["frontier_evidence_audit"] = frontier_audit
        if semantic_retry_summary:
            item["semantic_retry"] = semantic_retry_summary
        if user_paper_count:
            item["domain_knowledge_context"] = {
                "source_policy": domain_knowledge.get("source_policy", ""),
                "paper_count": user_paper_count,
                "forbidden_uses": domain_knowledge.get("forbidden_uses", []),
                "papers": domain_knowledge.get("papers", []),
            }
    return ideas


def run_pipeline_idea_stage(
    *,
    root_dir: Path,
    query: str,
    idea_count: int,
    pattern_limit: int,
    genome_limit: int,
    provider: str | None,
    model: str,
    frontier_query: str = "",
    extractive: bool = False,
) -> Dict[str, object]:
    config = load_config(root_dir)
    ensure_layout(config)
    init_db(config.db_path)

    try:
        ideas = run_pipeline_idea_generation(
            root_dir=config.root_dir,
            query=query,
            frontier_query=frontier_query,
            idea_count=idea_count,
            pattern_limit=pattern_limit,
            genome_limit=genome_limit,
            provider=provider,
            model=model,
            extractive=extractive,
        )
    except (LLMError, ValueError, json.JSONDecodeError) as exc:
        if isinstance(exc, IdeaGenerationNotReady):
            raise
        failure_summary = build_idea_generation_failure_summary(
            query=query,
            requested=idea_count,
            rejected_ideas=[
                {
                    "title": "idea_generation",
                    "reason": str(exc),
                }
            ],
        )
        raise IdeaGenerationNotReady(str(exc), failure_summary) from exc

    paper_rows = list_trend_papers(config.db_path)
    pattern_rows = list_strict_pattern_rows(config.db_path, limit=max(1, pattern_limit))
    genome_rows = list_strict_gold_genome_rows(config.db_path, limit=max(1, genome_limit))
    top_paper_rows = core_gold_rows(list_top_papers(config.db_path, limit=max(5, pattern_limit + genome_limit)))
    rubric_sources = flatten_rubric_sources(
        pattern_rows=pattern_rows,
        genome_rows=genome_rows,
        paper_rows=top_paper_rows,
    )
    paper_corpus = [
        {
            "id": int(row["id"]),
            "title": str(row["title"]).strip(),
            "abstract": str(row["abstract"]).strip(),
            "venue": str(row["venue"]).strip(),
            "year": int(row["year"]),
            "paper_weight": float(row["paper_weight"]),
        }
        for row in paper_rows
    ]

    idea_dir = config.output_dir / "ideas"
    idea_dir.mkdir(parents=True, exist_ok=True)
    output_paths: List[str] = []
    dossier_outputs: List[str] = []
    markdown_outputs: List[str] = []
    idea_summaries: List[Dict[str, object]] = []
    rejected_ideas: List[Dict[str, str]] = []
    semantic_retry_used = False
    while True:
        batch_rejected: List[Dict[str, str]] = []
        batch_generated_before = len(idea_summaries)
        for idx, payload in enumerate(ideas, start=1):
            progress_note(f"审计 idea 草案 {idx}/{len(ideas)}：prior-art、证据链和评分依据")
            prior_art_gate = run_prior_art_gate(
                config=config,
                query=query,
                idea_payload=payload,
                paper_corpus=paper_corpus,
                provider=provider,
                model=model,
            )
            payload["prior_art_gate"] = prior_art_gate
            payload["prior_art"] = prior_art_gate.get("local_prior_art", {})
            if isinstance(payload["prior_art"], dict):
                payload["prior_art"]["expert_decision"] = str(prior_art_gate.get("decision", "")).strip()
                payload["prior_art"]["expert_summary"] = str(prior_art_gate.get("summary", "")).strip()
            payload["evidence_grounded_score"] = build_evidence_grounded_score(
                payload,
                pattern_rows=pattern_rows,
                genome_rows=genome_rows,
                paper_rows=top_paper_rows,
            )
            payload["storyline_migration_audit"] = audit_storyline_migration(payload, rubric_sources)
            prior_decision = str(prior_art_gate.get("decision", "")).strip()
            if prior_decision == "duplicate_rethink":
                progress_note(f"idea 草案 {idx} 与已有工作过近，退回重想")
                batch_rejected.append(
                    {
                        "title": str(payload.get("idea_title", "")).strip() or f"idea-{idx}",
                        "reason": "prior_art_duplicate_rethink: "
                        + str(prior_art_gate.get("summary", "")).strip(),
                    }
                )
                continue
            rejection_reason = generation_evidence_rejection_reason(payload, payload["evidence_grounded_score"])
            if rejection_reason:
                progress_note(f"idea 草案 {idx} 未通过严格证据审计：{rejection_reason[:120]}")
                batch_rejected.append(
                    {
                        "title": str(payload.get("idea_title", "")).strip() or f"idea-{idx}",
                        "reason": rejection_reason,
                    }
                )
                continue
            try:
                require_storyline_migration_valid(payload, rubric_sources)
            except ValueError as exc:
                progress_note(f"idea 草案 {idx} 未通过故事线迁移审计：{str(exc)[:120]}")
                batch_rejected.append(
                    {
                        "title": str(payload.get("idea_title", "")).strip() or f"idea-{idx}",
                        "reason": str(exc),
                    }
                )
                continue
            payload["generation_evidence_audit"] = payload["evidence_grounded_score"].get("generation_evidence_audit", {})
            payload["critique_outputs"] = build_candidate_critique_outputs(payload)
            title = str(payload["idea_title"]).strip()
            content_json = json.dumps(payload, ensure_ascii=False, indent=2)
            record_id = add_candidate_idea(
                config.db_path,
                query=query,
                title=title,
                content_json=content_json,
            )
            output_path = idea_dir / f"idea-{record_id}.json"
            output_path.write_text(content_json, encoding="utf-8")
            output_paths.append(str(output_path))
            progress_note(f"idea 草案 {idx} 通过审计并写入库")
            dossier_payload, dossier_json_path, dossier_markdown_path = write_candidate_dossier(
                config.root_dir,
                idea_id=record_id,
                query=query,
                payload=payload,
            )
            dossier_outputs.append(str(dossier_json_path))
            markdown_outputs.append(str(dossier_markdown_path))
            prior_art_payload = payload.get("prior_art", {})
            if not isinstance(prior_art_payload, dict):
                prior_art_payload = {}
            summary_item = {
                "idea_id": int(record_id),
                "title": title,
                "prior_art_label": str(prior_art_payload.get("overlap_label", "")).strip(),
                "prior_art_gate_decision": str(prior_art_gate.get("decision", "")).strip(),
                "evidence_score": float(payload["evidence_grounded_score"].get("total", 0.0))
                if isinstance(payload.get("evidence_grounded_score"), dict)
                else 0.0,
                "target_venue": str(dossier_payload.get("target_venue", "")).strip(),
                "dossier_json_path": str(dossier_json_path),
                "dossier_markdown_path": str(dossier_markdown_path),
            }
            semantic_retry_payload = payload.get("semantic_retry")
            if isinstance(semantic_retry_payload, dict):
                summary_item["semantic_retry"] = sanitize_failure_for_prompt(semantic_retry_payload)
            idea_summaries.append(summary_item)
        rejected_ideas.extend(batch_rejected)
        semantic_retry_summary = build_idea_generation_failure_summary(
            query=query,
            requested=idea_count,
            rejected_ideas=batch_rejected,
        )
        if (
            idea_summaries
            or semantic_retry_used
            or len(batch_rejected) < len(ideas)
            or not idea_generation_failure_is_semantic_retryable(semantic_retry_summary)
        ):
            break
        progress_note("本批 idea 全部被严格审计拒绝，执行一次语义级兜底重试")
        try:
            ideas = run_pipeline_idea_generation(
                root_dir=config.root_dir,
                query=query,
                frontier_query=frontier_query,
                idea_count=idea_count,
                pattern_limit=pattern_limit,
                genome_limit=genome_limit,
                provider=provider,
                model=model,
                extractive=extractive,
                semantic_retry_summary=semantic_retry_summary,
            )
        except (LLMError, ValueError, json.JSONDecodeError) as exc:
            failure_summary = build_idea_generation_failure_summary(
                query=query,
                requested=idea_count,
                rejected_ideas=batch_rejected + [{"title": "semantic_retry", "reason": str(exc)}],
            )
            raise IdeaGenerationNotReady(str(exc), failure_summary) from exc
        semantic_retry_used = True

    if not idea_summaries:
        failure_summary = build_idea_generation_failure_summary(
            query=query,
            requested=idea_count,
            rejected_ideas=rejected_ideas,
        )
        detail = " No generated ideas passed strict evidence audit."
        if rejected_ideas:
            detail += " " + " ".join(item["reason"] for item in rejected_ideas[:3])
        raise IdeaGenerationNotReady(detail.strip(), failure_summary)

    return {
        "stage": "ideas",
        "status": "completed",
        "generated": len(idea_summaries),
        "rejected": len(rejected_ideas),
        "rejected_ideas": rejected_ideas,
        "outputs": output_paths,
        "dossier_outputs": dossier_outputs,
        "markdown_outputs": markdown_outputs,
        "idea_summaries": idea_summaries,
    }


def infer_next_stages(stages: List[Dict[str, object]]) -> List[str]:
    skipped_reasons = {stage["stage"]: stage.get("reason", "") for stage in stages if stage["status"] == "skipped"}
    completed = {stage["stage"] for stage in stages if stage["status"] == "completed"}
    failed = {stage["stage"] for stage in stages if stage["status"] == "failed"}
    if "review_loop" in failed or skipped_reasons.get("review_loop"):
        return ["review_loop", "session_step", "dossier"]
    if skipped_reasons.get("harvest") == "no_harvest_input":
        return ["harvest", "score", "trends", "genome", "pattern", "ideas"]
    if skipped_reasons.get("score") == "no_papers_available":
        return ["harvest", "score", "trends", "genome", "pattern", "ideas"]
    if skipped_reasons.get("genome") == "llm_not_configured":
        next_steps = ["configure_llm", "genome", "pattern", "ideas"]
        if "trends" not in completed:
            next_steps.insert(0, "trends")
        return next_steps
    if skipped_reasons.get("pattern") == "not_enough_genomes":
        return ["genome", "pattern", "ideas"]
    if skipped_reasons.get("ideas") == "llm_not_configured":
        return ["configure_llm", "ideas", "session", "critique"]
    return ["session", "critique", "dossier"]


def infer_next_commands(
    stages: List[Dict[str, object]],
    *,
    final_status: str,
    has_ranking: bool,
    has_session: bool,
) -> List[str]:
    skipped_reasons = {stage["stage"]: stage.get("reason", "") for stage in stages if stage["status"] == "skipped"}
    completed = {stage["stage"] for stage in stages if stage["status"] == "completed"}
    if skipped_reasons.get("score") == "no_papers_available":
        return [
            f"{CLI_CMD} r --file seeds/demo_papers.jsonl",
            f'{CLI_CMD} r "research agents" --remote openalex --venue ICLR --year 2024 --limit 20',
            f"{CLI_CMD} score",
        ]
    if skipped_reasons.get("genome") == "llm_not_configured":
        return [
            f"{CLI_CMD} ds sk-...",
            f"{CLI_CMD} oa sk-...",
            f"{CLI_CMD} r",
        ]
    if final_status == "session_review_incomplete":
        session_stage = next((stage for stage in stages if stage.get("stage") == "review_loop"), {})
        session_id = ""
        if isinstance(session_stage, dict):
            session_id = str(session_stage.get("session_id", "")).strip()
        session_arg = f" --session-id {session_id}" if session_id else " --latest"
        return [
            f"{CLI_CMD} rl{session_arg}",
            f"{CLI_CMD} sd{session_arg}",
            f'{CLI_CMD} st{session_arg} "..."',
        ]
    if final_status in {"session_ready", "session_review_ready"} or has_session:
        return [
            f'{CLI_CMD} st "..."',
            f"{CLI_CMD} sd",
            f"{CLI_CMD} sv",
        ]
    if "ideas" in completed and has_ranking:
        return [
            f"{CLI_CMD} ix",
            f"{CLI_CMD} r --session",
            f"{CLI_CMD} ir",
        ]
    if "ideas" in completed:
        return [
            f"{CLI_CMD} ix",
            f"{CLI_CMD} il",
            f"{CLI_CMD} ir",
        ]
    if "trends" in completed:
        return [
            f"{CLI_CMD} tr",
            f"{CLI_CMD} gb --limit 5",
            f"{CLI_CMD} pb --limit 5",
        ]
    return [f"{CLI_CMD} status"]


def run_critique(
    *,
    root_dir: Path,
    current_idea: str,
    user_instruction: str,
    provider: str | None,
    model: str,
    pattern_limit: int,
    genome_limit: int,
    session_context: Dict[str, object],
) -> Dict[str, object]:
    config = load_config(root_dir)
    ensure_layout(config)
    init_db(config.db_path)

    pattern_rows = list_strict_pattern_rows(config.db_path, limit=max(1, pattern_limit))
    genome_rows = list_strict_gold_genome_rows(config.db_path, limit=max(1, genome_limit))
    if not pattern_rows and not genome_rows:
        raise ValueError("Need genome or pattern evidence before critique. Run `ra gb` and `ra pb` first.")

    apply_llm_runtime_overrides(config, provider, model)

    patterns = []
    for row in pattern_rows:
        payload = json.loads(row["content_json"])
        payload["pattern_key"] = row["pattern_key"]
        patterns.append(payload)

    genomes = []
    for row in genome_rows:
        payload = json.loads(row["content_json"])
        payload["paper_id"] = row["paper_id"]
        payload["title"] = row["title"]
        payload["venue"] = row["venue"]
        payload["year"] = row["year"]
        genomes.append(payload)

    client = LLMClient(config.llm)
    prompt = build_critique_prompt(
        current_idea=current_idea,
        user_instruction=user_instruction,
        session_context=session_context,
        pattern_cards=patterns,
        genome_cards=genomes,
    )
    response = client.chat(prompt=prompt, system_prompt=CRITIQUE_SYSTEM_PROMPT)
    critique_payload = parse_llm_response_with_repair(
        client=client,
        original_prompt=prompt,
        system_prompt=CRITIQUE_SYSTEM_PROMPT,
        original_text=response.text,
        parser=parse_critique_response,
        response_name="Critique response",
        validator=lambda payload: validate_session_step_payload(
            payload,
            pattern_rows=pattern_rows,
            genome_rows=genome_rows,
            session_context=session_context,
        ),
    )
    return critique_payload


def refresh_session_dossier(root_dir: Path, session_id: int) -> Tuple[Dict[str, object], Path]:
    config = load_config(root_dir)
    ensure_layout(config)
    init_db(config.db_path)

    session = get_idea_session(config.db_path, session_id)
    if not session:
        raise ValueError(f"Idea session not found: {session_id}")
    turns = list_idea_session_turns(config.db_path, session_id)

    dossier = build_session_dossier_payload(session, turns)
    session_dir = config.output_dir / "sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    json_path = session_dir / f"session-{session_id}-dossier.json"
    markdown_path = session_dir / f"session-{session_id}-dossier.md"
    dossier["json_path"] = str(json_path)
    dossier["markdown_path"] = str(markdown_path)
    json_path.write_text(json.dumps(dossier, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(render_session_dossier_markdown(dossier), encoding="utf-8")
    return dossier, json_path


def write_candidate_dossier(
    root_dir: Path,
    *,
    idea_id: int,
    query: str,
    payload: Dict[str, object],
) -> Tuple[Dict[str, object], Path, Path]:
    config = load_config(root_dir)
    ensure_layout(config)
    dossier = build_candidate_dossier_payload(idea_id=idea_id, query=query, payload=payload)
    dossier_dir = config.output_dir / "dossiers"
    dossier_dir.mkdir(parents=True, exist_ok=True)
    json_path = dossier_dir / f"idea-{idea_id}-dossier.json"
    markdown_path = dossier_dir / f"idea-{idea_id}-dossier.md"
    dossier["json_path"] = str(json_path)
    dossier["markdown_path"] = str(markdown_path)
    json_path.write_text(json.dumps(dossier, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(render_candidate_dossier_markdown(dossier), encoding="utf-8")
    return dossier, json_path, markdown_path


def build_candidate_dossier_payload(*, idea_id: int, query: str, payload: Dict[str, object]) -> Dict[str, object]:
    session_context = build_session_context_from_candidate(payload, idea_id)
    critique_outputs = normalize_critique_outputs(payload.get("critique_outputs"))
    if not critique_outputs:
        critique_outputs = build_candidate_critique_outputs(payload)
    next_target = infer_candidate_next_target(payload)
    title = str(payload.get("idea_title", "")).strip()
    core_hypothesis = str(payload.get("core_hypothesis", "")).strip()
    one_line_idea = core_hypothesis or title
    trend_thesis = build_trend_thesis(
        str(session_context.get("trend_support", "")).strip(),
        str(session_context.get("frontier_gap", "")).strip(),
        str(session_context.get("why_now", "")).strip(),
    )
    prior_art_check = build_prior_art_check(session_context)
    feasibility = render_dimension_summary(
        critique_outputs.get("feasibility_assessment", {}),
        extra_key="next_test",
    )
    why_not = render_dimension_summary(
        critique_outputs.get("why_not_analysis", {}),
        extra_key="mitigation",
    )
    value_judgment = render_dimension_summary(
        critique_outputs.get("value_judgment", {}),
        extra_key="proof_gap",
    )
    risk_summary = build_risk_summary(session_context, str(payload.get("key_risk", "")).strip(), next_target)
    experiment_plan = build_experiment_plan(session_context, next_target)
    target_venue = infer_target_venue(title or one_line_idea, session_context)
    historical_pattern = str(payload.get("historical_pattern", "")).strip()
    evidence_grounded_score = payload.get("evidence_grounded_score", {})
    if not isinstance(evidence_grounded_score, dict):
        evidence_grounded_score = {}
    evidence_basis = payload.get("evidence_basis", [])
    if not isinstance(evidence_basis, list):
        evidence_basis = []
    storyline_trace = payload.get("storyline_trace", {})
    if not isinstance(storyline_trace, dict):
        storyline_trace = {}
    generation_evidence_audit = payload.get("generation_evidence_audit", {})
    if not isinstance(generation_evidence_audit, dict):
        generation_evidence_audit = evidence_grounded_score.get("generation_evidence_audit", {})
    if not isinstance(generation_evidence_audit, dict):
        generation_evidence_audit = {}

    return {
        "idea_id": int(idea_id),
        "query": query.strip(),
        "title": title,
        "one_line_idea": one_line_idea,
        "historical_pattern": historical_pattern,
        "trend_thesis": trend_thesis,
        "prior_art_check": prior_art_check,
        "feasibility": feasibility,
        "why_not_analysis": why_not,
        "value_judgment": value_judgment,
        "risk_summary": risk_summary,
        "experiment_plan": experiment_plan,
        "target_venue": target_venue,
        "novelty_claim": str(payload.get("novelty", "")).strip(),
        "paper_story_hook": str(payload.get("paper_angle", "")).strip(),
        "evaluation_sketch": str(payload.get("evaluation_outline", "")).strip(),
        "why_now": str(payload.get("why_now", "")).strip(),
        "frontier_gap": str(payload.get("frontier_gap", "")).strip(),
        "generation_guardrail": str(payload.get("generation_guardrail", "")).strip(),
        "storyline_trace": storyline_trace,
        "evidence_basis": evidence_basis,
        "generation_evidence_audit": generation_evidence_audit,
        "evidence_grounded_score": evidence_grounded_score,
        "critique_outputs": critique_outputs,
        "supporting_patterns": [historical_pattern] if historical_pattern else [],
        "supporting_genomes": [],
        "session_context": session_context,
    }


def build_session_dossier_payload(session, turns) -> Dict[str, object]:
    latest_payload: Dict[str, object] = {}
    session_context = load_session_context(session)
    memory_summary = load_session_memory(session) or build_session_memory_summary(session, turns, session_context)
    context_critique_outputs = normalize_critique_outputs(session_context.get("critique_outputs"))
    accepted = 0
    partial = 0
    rejected = 0
    preserves: List[str] = []
    revises: List[str] = []
    pattern_keys: List[str] = []
    genome_titles: List[str] = []
    verdict_summaries: List[str] = []
    why_notes: List[str] = []
    coach_notes: List[str] = []

    for turn in turns:
        decision = str(turn["decision"]).strip().lower()
        if decision == "accept":
            accepted += 1
        elif decision == "partial":
            partial += 1
        elif decision == "reject":
            rejected += 1
        payload = json.loads(turn["content_json"])
        latest_payload = payload
        verdict = str(payload.get("verdict_summary", "")).strip()
        why_note = str(payload.get("why_user_may_be_wrong", "")).strip()
        coach_note = str(payload.get("coach_message", "")).strip()
        if verdict:
            verdict_summaries.append(verdict)
        if why_note:
            why_notes.append(why_note)
        if coach_note:
            coach_notes.append(coach_note)
        preserve = str(payload.get("preserve", "")).strip()
        revise = str(payload.get("revise", "")).strip()
        if preserve:
            preserves.append(preserve)
        if revise:
            revises.append(revise)
        pattern_keys.extend(str(item).strip() for item in payload.get("supporting_patterns", []) if str(item).strip())
        genome_titles.extend(str(item).strip() for item in payload.get("supporting_genomes", []) if str(item).strip())

    current_best = str(session["current_idea"]).strip()
    initial_idea = str(session["initial_idea"]).strip()
    trajectory_summary = summarize_session_trajectory(turns, accepted, partial, rejected)
    latest_summary = latest_payload.get("verdict_summary", "Initial idea captured; no critique turns yet.")
    keep_signal = preserves[-1] if preserves else initial_idea
    next_target = revises[-1] if revises else "Collect the first critique turn."
    why_not_follow = latest_payload.get("why_user_may_be_wrong", "No critique yet.")
    coach_message = latest_payload.get("coach_message", "Start with a user challenge to pressure-test the idea.")
    trend_support = str(session_context.get("trend_support", "")).strip()
    frontier_gap = str(session_context.get("frontier_gap", "")).strip()
    trend_alignment = str(latest_payload.get("trend_alignment", "")).strip()
    trend_thesis = build_trend_thesis(trend_support, frontier_gap, trend_alignment)
    prior_art_check = build_prior_art_check(session_context)
    evidence_chain = build_session_evidence_chain(session_context)
    critique_outputs = build_session_critique_outputs(
        latest_payload=latest_payload,
        session_context=session_context,
        current_best=current_best,
        latest_summary=latest_summary,
        why_not_follow=why_not_follow,
        next_target=next_target,
        context_critique_outputs=context_critique_outputs,
    )
    feasibility = render_dimension_summary(
        critique_outputs["feasibility_assessment"],
        extra_key="next_test",
    )
    why_not = render_dimension_summary(
        critique_outputs["why_not_analysis"],
        extra_key="mitigation",
    )
    value_judgment = render_dimension_summary(
        critique_outputs["value_judgment"],
        extra_key="proof_gap",
    )
    risk_summary = build_risk_summary(session_context, why_not_follow, next_target)
    experiment_plan = build_experiment_plan(session_context, next_target)
    target_venue = infer_target_venue(current_best, session_context)

    return {
        "session_id": int(session["id"]),
        "name": session["name"],
        "status": session["status"],
        "initial_idea": initial_idea,
        "current_best_idea": current_best,
        "turn_count": len(turns),
        "latest_decision": latest_payload.get("decision", "seed"),
        "trajectory_summary": trajectory_summary,
        "latest_summary": latest_summary,
        "keep_signal": keep_signal,
        "next_refinement_target": next_target,
        "why_not_blindly_follow_user": why_not_follow,
        "coach_message": coach_message,
        "trend_thesis": trend_thesis,
        "novelty_claim": build_novelty_claim(current_best, keep_signal, latest_summary),
        "evaluation_sketch": build_evaluation_sketch(current_best, next_target),
        "paper_story_hook": build_paper_story_hook(current_best, latest_summary, coach_message),
        "main_risk": build_main_risk(why_not_follow, next_target),
        "evidence_chain": evidence_chain,
        "prior_art_check": prior_art_check,
        "critique_outputs": critique_outputs,
        "feasibility": feasibility,
        "why_not_analysis": why_not,
        "value_judgment": value_judgment,
        "risk_summary": risk_summary,
        "experiment_plan": experiment_plan,
        "target_venue": target_venue,
        "decision_tally": {
            "accept": accepted,
            "partial": partial,
            "reject": rejected,
        },
        "critique_memory": {
            "recent_verdicts": verdict_summaries[-3:],
            "recent_pushbacks": why_notes[-3:],
            "recent_coach_messages": coach_notes[-3:],
        },
        "memory_summary": memory_summary,
        "supporting_patterns": dedupe_keep_order(pattern_keys),
        "supporting_genomes": dedupe_keep_order(genome_titles),
        "session_context": session_context,
        "created_at": session["created_at"],
        "updated_at": session["updated_at"],
    }


def summarize_session_trajectory(turns, accepted: int, partial: int, rejected: int) -> str:
    if not turns:
        return "Seed idea only; no critique turns yet."
    latest_turn = turns[-1]
    latest_index = int(latest_turn["turn_index"])
    return (
        f"{len(turns)} turns so far; accept={accepted}, partial={partial}, reject={rejected}. "
        f"Latest turn {latest_index} refined the idea through a {latest_turn['decision']} decision."
    )


def dedupe_keep_order(values: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def build_novelty_claim(current_best: str, keep_signal: str, latest_summary: object) -> str:
    summary_text = str(latest_summary).strip()
    if keep_signal and keep_signal != current_best:
        return f"{current_best} The key non-incremental move to protect is: {keep_signal}"
    if summary_text:
        return f"{current_best} Novelty thesis: {summary_text}"
    return current_best


def build_evaluation_sketch(current_best: str, next_target: str) -> str:
    return (
        f"Test whether the idea really delivers on its claim: {current_best} "
        f"Next evaluation design pressure point: {next_target}"
    )


def build_paper_story_hook(current_best: str, latest_summary: object, coach_message: object) -> str:
    summary_text = str(latest_summary).strip()
    coach_text = str(coach_message).strip()
    if summary_text and coach_text:
        return f"{summary_text} Coach note: {coach_text}"
    if summary_text:
        return summary_text
    return f"The paper story should make this idea feel necessary: {current_best}"


def build_main_risk(why_not_follow: object, next_target: str) -> str:
    reason = str(why_not_follow).strip()
    if reason:
        return f"{reason} The next thing to de-risk is: {next_target}"
    return f"Main unresolved risk: {next_target}"


def build_trend_thesis(trend_support: str, frontier_gap: str, trend_alignment: str) -> str:
    pieces = [piece for piece in [trend_support, frontier_gap, trend_alignment] if piece]
    if pieces:
        return " ".join(pieces)
    return "No explicit trend thesis yet."


def build_session_evidence_chain(session_context: Dict[str, object]) -> Dict[str, object]:
    basis = normalize_explicit_evidence_basis(session_context.get("evidence_basis"))
    source_lines = []
    for item in basis[:5]:
        title = str(item.get("source_title", "")).strip()
        source_type = str(item.get("source_type", "")).strip() or "evidence"
        source_id = str(item.get("source_id", "")).strip()
        borrowed = str(item.get("borrowed_standard", "")).strip()
        used_for = str(item.get("used_for", "")).strip()
        label = title or source_id or source_type
        detail = compact_excerpt(borrowed, 220) if borrowed else "stored core Gold evidence"
        use_note = f" | used_for={used_for}" if used_for else ""
        source_lines.append(f"{label} ({source_type}:{source_id}) -> {detail}{use_note}")

    audit = session_context.get("generation_evidence_audit", {})
    if not isinstance(audit, dict):
        audit = {}
    score = session_context.get("evidence_grounded_score", {})
    if not isinstance(score, dict):
        score = {}

    return {
        "reference_papers": source_lines,
        "storyline_logic": str(session_context.get("paper_angle", "")).strip()
        or str(session_context.get("historical_pattern", "")).strip(),
        "transferred_pattern": str(session_context.get("historical_pattern", "")).strip(),
        "current_method_problem": str(session_context.get("frontier_gap", "")).strip(),
        "recent_limitation_signal": str(session_context.get("trend_support", "")).strip(),
        "why_now": str(session_context.get("why_now", "")).strip(),
        "reviewer_attack_surface": str(session_context.get("key_risk", "")).strip(),
        "audit_status": str(audit.get("status", "")).strip(),
        "matched_basis_count": int(audit.get("matched_basis_count", 0) or 0),
        "evidence_score": score.get("total", ""),
    }


def build_prior_art_check(session_context: Dict[str, object]) -> str:
    gate_decision = str(session_context.get("prior_art_gate_decision", "")).strip()
    gate_summary = str(session_context.get("prior_art_gate_summary", "")).strip()
    why_not_done = session_context.get("why_not_done_yet", [])
    required_diff = session_context.get("required_differentiation", [])
    label = str(session_context.get("prior_art_label", "")).strip()
    summary = str(session_context.get("prior_art_summary", "")).strip()
    note = str(session_context.get("differentiation_note", "")).strip()
    pieces = []
    if gate_decision:
        pieces.append(f"Prior-art expert gate: {gate_decision}.")
    if gate_summary:
        pieces.append(gate_summary)
    if isinstance(why_not_done, list) and why_not_done:
        pieces.append("Why not done yet: " + " ".join(str(item).strip() for item in why_not_done[:3] if str(item).strip()))
    if isinstance(required_diff, list) and required_diff:
        pieces.append("Required differentiation: " + " ".join(str(item).strip() for item in required_diff[:3] if str(item).strip()))
    if label:
        pieces.append(f"Local prior-art screen: {label}.")
    if summary:
        pieces.append(summary)
    if note:
        pieces.append(note)
    return " ".join(pieces) if pieces else "No stored prior-art screen yet."


def build_session_critique_outputs(
    *,
    latest_payload: Dict[str, object],
    session_context: Dict[str, object],
    current_best: str,
    latest_summary: object,
    why_not_follow: object,
    next_target: str,
    context_critique_outputs: Dict[str, Dict[str, str]],
) -> Dict[str, Dict[str, str]]:
    latest_outputs = normalize_critique_outputs(latest_payload)
    if latest_outputs:
        return {
            "feasibility_assessment": latest_outputs.get("feasibility_assessment", {}),
            "why_not_analysis": latest_outputs.get("why_not_analysis", {}),
            "value_judgment": latest_outputs.get("value_judgment", {}),
        }
    if context_critique_outputs:
        return {
            "feasibility_assessment": context_critique_outputs.get("feasibility_assessment", {}),
            "why_not_analysis": context_critique_outputs.get("why_not_analysis", {}),
            "value_judgment": context_critique_outputs.get("value_judgment", {}),
        }
    return {
        "feasibility_assessment": {
            "status": infer_session_feasibility_status(session_context),
            "summary": build_feasibility_assessment(session_context, next_target),
            "next_test": next_target,
        },
        "why_not_analysis": {
            "status": "moderate" if str(why_not_follow).strip() else "light",
            "summary": build_why_not_analysis(why_not_follow, session_context),
            "mitigation": next_target,
        },
        "value_judgment": {
            "status": infer_session_value_status(session_context, latest_summary),
            "summary": build_value_judgment(session_context, latest_summary, current_best),
            "proof_gap": next_target,
        },
    }


def infer_session_feasibility_status(session_context: Dict[str, object]) -> str:
    experiments = session_context.get("first_experiments", [])
    experiment_count = len(experiments) if isinstance(experiments, list) else 0
    evaluation_outline = str(session_context.get("evaluation_outline", "")).strip()
    if experiment_count >= 2 and evaluation_outline:
        return "promising"
    if experiment_count >= 1 or evaluation_outline:
        return "unclear"
    return "fragile"


def infer_session_value_status(session_context: Dict[str, object], latest_summary: object) -> str:
    value = str(session_context.get("value", "")).strip()
    novelty = str(session_context.get("novelty", "")).strip()
    summary = str(latest_summary).strip()
    if value and novelty:
        return "high"
    if value or novelty or summary:
        return "medium"
    return "low"


def build_feasibility_assessment(session_context: Dict[str, object], next_target: str) -> str:
    experiments = session_context.get("first_experiments", [])
    if not isinstance(experiments, list):
        experiments = []
    experiments = [str(item).strip() for item in experiments if str(item).strip()]
    evaluation_outline = str(session_context.get("evaluation_outline", "")).strip()
    pieces = []
    if experiments:
        pieces.append(f"First signal path: {experiments[0]}.")
    if len(experiments) > 1:
        pieces.append(f"Follow-up validation: {experiments[1]}.")
    if evaluation_outline:
        pieces.append(f"Evaluation frame: {evaluation_outline}")
    pieces.append(f"Immediate execution question: {next_target}")
    return " ".join(pieces)


def build_why_not_analysis(why_not_follow: object, session_context: Dict[str, object]) -> str:
    reason = str(why_not_follow).strip()
    stored_risk = str(session_context.get("key_risk", "")).strip()
    why_not_done = session_context.get("why_not_done_yet", [])
    why_not_text = ""
    if isinstance(why_not_done, list):
        why_not_text = " ".join(str(item).strip() for item in why_not_done[:3] if str(item).strip())
    if reason and stored_risk:
        return f"{reason} Existing candidate risk: {stored_risk}" + (f" Why not done yet: {why_not_text}" if why_not_text else "")
    if reason:
        return reason + (f" Why not done yet: {why_not_text}" if why_not_text else "")
    if stored_risk:
        return stored_risk + (f" Why not done yet: {why_not_text}" if why_not_text else "")
    if why_not_text:
        return f"Why not done yet: {why_not_text}"
    return "No explicit blocker captured yet."


def build_value_judgment(session_context: Dict[str, object], latest_summary: object, current_best: str) -> str:
    value = str(session_context.get("value", "")).strip()
    novelty = str(session_context.get("novelty", "")).strip()
    summary = str(latest_summary).strip()
    pieces = []
    if value:
        pieces.append(value)
    if novelty:
        pieces.append(f"Novelty edge: {novelty}")
    if summary:
        pieces.append(f"Current verdict: {summary}")
    if not pieces:
        pieces.append(f"Value case still needs to be made explicit for: {current_best}")
    return " ".join(pieces)


def build_risk_summary(session_context: Dict[str, object], why_not_follow: object, next_target: str) -> str:
    stored_risk = str(session_context.get("key_risk", "")).strip()
    reason = str(why_not_follow).strip()
    why_not_done = session_context.get("why_not_done_yet", [])
    why_not_text = ""
    if isinstance(why_not_done, list):
        why_not_text = " ".join(str(item).strip() for item in why_not_done[:2] if str(item).strip())
    if stored_risk and reason:
        return f"{stored_risk} Critique pressure: {reason} {why_not_text} Next de-risking move: {next_target}".strip()
    if stored_risk:
        return f"{stored_risk} {why_not_text} Next de-risking move: {next_target}".strip()
    if why_not_text:
        return f"{why_not_text} Next de-risking move: {next_target}"
    return build_main_risk(why_not_follow, next_target)


def build_experiment_plan(session_context: Dict[str, object], next_target: str) -> Dict[str, str]:
    experiments = session_context.get("first_experiments", [])
    if not isinstance(experiments, list):
        experiments = []
    experiments = [str(item).strip() for item in experiments if str(item).strip()]
    evaluation_outline = str(session_context.get("evaluation_outline", "")).strip()
    week_2 = experiments[0] if experiments else f"Lock the first concrete test around: {next_target}"
    week_4 = experiments[1] if len(experiments) > 1 else (evaluation_outline or f"Build the first evaluation protocol for: {next_target}")
    week_8 = (
        f"Run a fuller comparison and turn the evidence into a paper story. {evaluation_outline}"
        if evaluation_outline
        else "Run a fuller comparison and turn the evidence into a paper story."
    )
    return {
        "week_2": week_2,
        "week_4": week_4,
        "week_8": week_8,
    }


def infer_target_venue(current_best: str, session_context: Dict[str, object]) -> str:
    text = " ".join(
        [
            current_best,
            str(session_context.get("paper_angle", "")).strip(),
            str(session_context.get("trend_support", "")).strip(),
        ]
    ).lower()
    if any(term in text for term in ["benchmark", "evaluation", "protocol"]):
        return "ICLR"
    if any(term in text for term in ["theory", "mechanism", "analysis", "boundar"]):
        return "NeurIPS"
    if any(term in text for term in ["architecture", "training", "representation", "objective"]):
        return "ICML"
    return "ICLR"


def render_candidate_dossier_markdown(dossier: Dict[str, object]) -> str:
    plan = dossier.get("experiment_plan", {})
    if not isinstance(plan, dict):
        plan = {}
    evidence_score = dossier.get("evidence_grounded_score", {})
    if not isinstance(evidence_score, dict):
        evidence_score = {}
    score_lines = render_evidence_score_lines(evidence_score)
    basis_lines = render_evidence_basis_lines(dossier.get("evidence_basis", []))
    audit_lines = render_generation_evidence_audit_lines(dossier.get("generation_evidence_audit", {}))
    storyline_lines = render_storyline_trace_lines(dossier.get("storyline_trace", {}))

    sections = [
        f"# {str(dossier.get('title', '')).strip() or str(dossier.get('one_line_idea', '')).strip()}",
        "",
        f"- idea_id: {dossier.get('idea_id')}",
        f"- query: {str(dossier.get('query', '')).strip()}",
        "",
        "## One-Line Idea",
        str(dossier.get("one_line_idea", "")).strip(),
        "",
        "## Historical Pattern",
        str(dossier.get("historical_pattern", "")).strip() or "No explicit historical pattern captured.",
        "",
        "## Evidence Basis",
        *(basis_lines or ["No explicit high-quality evidence basis captured."]),
        "",
        "## Generation Evidence Audit",
        *(audit_lines or ["No generation evidence audit available."]),
        "",
        "## Evidence-Grounded Score",
        *(score_lines or ["No evidence-grounded score available yet."]),
        "",
        "## Storyline Migration",
        *(storyline_lines or ["No storyline migration trace available."]),
        "",
        "## Frontier Evidence",
        str(dossier.get("trend_thesis", "")).strip(),
        "",
        "## Prior-Art Check",
        str(dossier.get("prior_art_check", "")).strip(),
        "",
        "## Feasibility",
        str(dossier.get("feasibility", "")).strip(),
        "",
        "## Why-Not Analysis",
        str(dossier.get("why_not_analysis", "")).strip(),
        "",
        "## Value Judgment",
        str(dossier.get("value_judgment", "")).strip(),
        "",
        "## Risk Summary",
        str(dossier.get("risk_summary", "")).strip(),
        "",
        "## Experiment Plan",
        f"- 2 weeks: {str(plan.get('week_2', '')).strip()}",
        f"- 4 weeks: {str(plan.get('week_4', '')).strip()}",
        f"- 8 weeks: {str(plan.get('week_8', '')).strip()}",
        "",
        "## Target Venue",
        str(dossier.get("target_venue", "")).strip(),
        "",
        "## Evaluation Sketch",
        str(dossier.get("evaluation_sketch", "")).strip(),
        "",
        "## Paper Angle",
        str(dossier.get("paper_story_hook", "")).strip(),
    ]
    return "\n".join(sections).strip() + "\n"


def render_storyline_trace_lines(value: object) -> List[str]:
    if not isinstance(value, dict):
        return []
    labels = [
        ("old_belief", "Old belief"),
        ("bottleneck", "Bottleneck"),
        ("reframing", "Reframing"),
        ("why_now", "Why now"),
        ("evidence_design", "Evidence design"),
        ("failure_boundary", "Failure boundary"),
    ]
    lines = []
    for key, label in labels:
        item = value.get(key)
        if not isinstance(item, dict):
            continue
        source = str(item.get("source_id", "")).strip() or "unknown"
        standard = str(item.get("paper_story_standard", "")).strip()
        transfer = str(item.get("transfer_to_current_hotspot", "")).strip()
        if not standard and not transfer:
            continue
        line = f"- {label} [{source}]"
        if standard:
            line += f" paper standard: {standard}"
        if transfer:
            line += f" | transfer: {transfer}"
        lines.append(line)
    return lines


def render_evidence_basis_lines(value: object) -> List[str]:
    if not isinstance(value, list):
        return []
    lines = []
    for item in value:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source_id", "")).strip() or "unknown"
        source_type = str(item.get("source_type", "")).strip() or "evidence"
        quality = str(item.get("quality_signal", "")).strip() or "stored"
        standard = str(item.get("borrowed_standard", "")).strip()
        used_for = str(item.get("used_for", "")).strip()
        line = f"- {source_type}:{source} ({quality})"
        if used_for:
            line += f" -> {used_for}"
        if standard:
            line += f": {standard}"
        lines.append(line)
    return lines


def render_generation_evidence_audit_lines(value: object) -> List[str]:
    if not isinstance(value, dict) or not value:
        return []
    return [
        f"- status: {str(value.get('status', '')).strip()}",
        f"- matched_basis_count: {value.get('matched_basis_count', 0)}",
        f"- unmatched_basis_count: {value.get('unmatched_basis_count', 0)}",
        f"- covered_uses: {', '.join(str(item) for item in value.get('covered_uses', []) or [])}",
        f"- missing_uses: {', '.join(str(item) for item in value.get('missing_uses', []) or [])}",
        f"- score_penalty: {value.get('penalty', 0.0)}",
        f"- requirement: {str(value.get('requirement', '')).strip()}",
    ]


def render_evidence_score_lines(value: Dict[str, object]) -> List[str]:
    if not value:
        return []
    lines = [
        f"- total: {value.get('total', 0.0)}/{value.get('scale', '0-10')}",
        f"- rubric_source: {str(value.get('rubric_source', '')).strip()}",
    ]
    dimensions = value.get("dimensions", {})
    if isinstance(dimensions, dict):
        for key in ["innovation", "logic", "feasibility", "value", "defensibility"]:
            item = dimensions.get(key, {})
            if not isinstance(item, dict):
                continue
            evidence = item.get("evidence", [])
            evidence_note = ""
            if isinstance(evidence, list) and evidence:
                first = evidence[0]
                if isinstance(first, dict):
                    evidence_note = f" via {first.get('source_type')}:{first.get('source_id')}"
            lines.append(
                f"- {key}: {item.get('score', 0.0)}/10 weight={item.get('weight', 0.0)} "
                f"explicit_evidence={item.get('explicit_evidence_covered', False)}{evidence_note}"
            )
    return lines


def render_run_report(manifest: Dict[str, object]) -> str:
    stages = manifest.get("stages", [])
    if not isinstance(stages, list):
        stages = []
    next_stages = manifest.get("next_stages", [])
    if not isinstance(next_stages, list):
        next_stages = []
    next_commands = manifest.get("next_commands", [])
    if not isinstance(next_commands, list):
        next_commands = []
    run_artifacts = manifest.get("run_artifacts", [])
    if not isinstance(run_artifacts, list):
        run_artifacts = []
    candidate_dossiers = manifest.get("candidate_dossiers", [])
    if not isinstance(candidate_dossiers, list):
        candidate_dossiers = []

    lines = [
        f"# Run {str(manifest.get('run_id', '')).strip()}",
        "",
        f"- Query: {str(manifest.get('query', '')).strip()}",
        f"- Status: {str(manifest.get('status', '')).strip()}",
        f"- Requested ideas: {manifest.get('requested_ideas', 0)}",
        f"- LLM provider: {str(manifest.get('llm_provider', '')).strip()}",
        f"- LLM model: {str(manifest.get('llm_model', '')).strip()}",
        "",
        "## Stage Summary",
    ]

    for stage in stages:
        if not isinstance(stage, dict):
            continue
        label = f"- {str(stage.get('stage', '')).strip()}: {str(stage.get('status', '')).strip()}"
        if stage.get("reason"):
            label += f" ({str(stage.get('reason', '')).strip()})"
        elif "generated" in stage:
            label += f" generated={stage.get('generated')}"
            if "rejected" in stage:
                label += f" rejected={stage.get('rejected')}"
        elif "paper_count" in stage:
            label += f" papers={stage.get('paper_count')}"
        elif "ranked" in stage:
            label += f" ideas={stage.get('ranked')}"
        elif "imported" in stage:
            label += f" imported={stage.get('imported')}"
        lines.append(label)

    ranking = manifest.get("ranking", {})
    if isinstance(ranking, dict) and ranking:
        lines.extend(
            [
                "",
                "## Top Pick",
                f"- Idea ID: {ranking.get('top_pick_id')}",
                f"- Title: {str(ranking.get('top_pick_title', '')).strip()}",
                f"- Ranking report: {str(ranking.get('report_path', '')).strip()}",
            ]
        )

    auto_session = manifest.get("auto_session", {})
    if isinstance(auto_session, dict) and auto_session:
        lines.extend(
            [
                "",
                "## Auto Session",
                f"- Session ID: {auto_session.get('session_id')}",
                f"- Name: {str(auto_session.get('name', '')).strip()}",
                f"- Dossier: {str(auto_session.get('dossier_path', '')).strip()}",
            ]
        )

    review_loop_stage = next(
        (stage for stage in stages if isinstance(stage, dict) and stage.get("stage") == "review_loop"),
        {},
    )
    if isinstance(review_loop_stage, dict) and review_loop_stage:
        requested = review_loop_stage.get("requested_rounds", "")
        completed = review_loop_stage.get("rounds_completed", "")
        status = str(review_loop_stage.get("status", "")).strip()
        lines.extend(
            [
                "",
                "## Auto Review Loop",
                f"- Status: {status}",
                f"- Rounds: {completed}/{requested}",
            ]
        )
        if review_loop_stage.get("reason"):
            lines.append(f"- Reason: {str(review_loop_stage.get('reason', '')).strip()}")
        if review_loop_stage.get("session_id"):
            lines.append(f"- Session ID: {review_loop_stage.get('session_id')}")
        if review_loop_stage.get("final_idea"):
            lines.append(f"- Final idea: {str(review_loop_stage.get('final_idea', '')).strip()}")

    if candidate_dossiers:
        lines.extend(["", "## Candidate Dossiers"])
        for item in candidate_dossiers:
            if not isinstance(item, dict):
                continue
            idea_id = item.get("idea_id")
            title = str(item.get("title", "")).strip()
            prior_art = str(item.get("prior_art_label", "")).strip() or "unknown"
            target_venue = str(item.get("target_venue", "")).strip() or "unknown"
            dossier_path = str(item.get("dossier_markdown_path", "")).strip()
            lines.append(
                f"- #{idea_id} {title} | prior-art: {prior_art} | target venue: {target_venue} | dossier: {dossier_path}"
            )

    if run_artifacts:
        lines.extend(["", "## Bundled Artifacts"])
        for path_text in run_artifacts:
            lines.append(f"- {str(path_text).strip()}")

    if next_commands:
        lines.extend(["", "## Next Commands"])
        for command in next_commands:
            lines.append(f"- {str(command).strip()}")

    if next_stages:
        lines.extend(["", "## Next Steps"])
        for stage_name in next_stages:
            lines.append(f"- {str(stage_name).strip()}")

    return "\n".join(lines).strip() + "\n"


def render_session_dossier_markdown(dossier: Dict[str, object]) -> str:
    plan = dossier.get("experiment_plan", {})
    if not isinstance(plan, dict):
        plan = {}
    patterns = dossier.get("supporting_patterns", [])
    if not isinstance(patterns, list):
        patterns = []
    genomes = dossier.get("supporting_genomes", [])
    if not isinstance(genomes, list):
        genomes = []
    memory = dossier.get("memory_summary", {})
    if not isinstance(memory, dict):
        memory = {}
    evidence_chain = dossier.get("evidence_chain", {})
    if not isinstance(evidence_chain, dict):
        evidence_chain = {}

    sections = [
        f"# {dossier['name']}",
        "",
        "## One-Line Idea",
        str(dossier.get("current_best_idea", "")).strip(),
        "",
        "## Evidence Chain",
        *render_session_evidence_chain_lines(evidence_chain),
        "",
        "## Historical Pattern",
        ", ".join(str(item).strip() for item in patterns if str(item).strip()) or "No explicit supporting patterns yet.",
        "",
        "## Frontier Evidence",
        str(dossier.get("trend_thesis", "")).strip(),
        "",
        "## Prior-Art Check",
        str(dossier.get("prior_art_check", "")).strip(),
        "",
        "## Feasibility",
        str(dossier.get("feasibility", "")).strip(),
        "",
        "## Why-Not Analysis",
        str(dossier.get("why_not_analysis", "")).strip(),
        "",
        "## Value Judgment",
        str(dossier.get("value_judgment", "")).strip(),
        "",
        "## Risk Summary",
        str(dossier.get("risk_summary", "")).strip(),
        "",
        "## Experiment Plan",
        f"- 2 weeks: {str(plan.get('week_2', '')).strip()}",
        f"- 4 weeks: {str(plan.get('week_4', '')).strip()}",
        f"- 8 weeks: {str(plan.get('week_8', '')).strip()}",
        "",
        "## Target Venue",
        str(dossier.get("target_venue", "")).strip(),
        "",
        "## Session Memory",
        f"- Latest decision: {str(dossier.get('latest_decision', '')).strip()}",
        f"- Keep signal: {str(dossier.get('keep_signal', '')).strip()}",
        f"- Next refinement target: {str(dossier.get('next_refinement_target', '')).strip()}",
        *render_memory_markdown(memory),
        "",
        "## Supporting Genomes",
        ", ".join(str(item).strip() for item in genomes if str(item).strip()) or "No explicit supporting genomes yet.",
    ]
    return "\n".join(sections).strip() + "\n"


def render_session_evidence_chain_lines(value: Dict[str, object]) -> List[str]:
    lines: List[str] = []
    references = value.get("reference_papers", [])
    if isinstance(references, list) and references:
        lines.append("- Reference basis:")
        lines.extend(f"  - {str(item).strip()}" for item in references if str(item).strip())
    else:
        lines.append("- Reference basis: No explicit high-quality paper evidence captured in this session yet.")

    field_labels = [
        ("storyline_logic", "Extracted storyline logic"),
        ("transferred_pattern", "Transferred pattern"),
        ("current_method_problem", "Current method problem"),
        ("recent_limitation_signal", "Recent limitation signal"),
        ("why_now", "Why now"),
        ("reviewer_attack_surface", "Reviewer attack surface"),
    ]
    for key, label in field_labels:
        text = str(value.get(key, "")).strip()
        if text:
            lines.append(f"- {label}: {text}")

    audit_status = str(value.get("audit_status", "")).strip()
    matched = value.get("matched_basis_count", 0)
    score = value.get("evidence_score", "")
    audit_parts = []
    if audit_status:
        audit_parts.append(f"status={audit_status}")
    if matched:
        audit_parts.append(f"matched_basis={matched}")
    if score != "":
        audit_parts.append(f"score={score}")
    if audit_parts:
        lines.append("- Evidence audit: " + ", ".join(str(item) for item in audit_parts))
    return lines


def copy_run_artifacts(run_dir: Path, artifact_paths: Iterable[str]) -> List[str]:
    copied_paths: List[str] = []
    seen: set[str] = set()
    for raw_path in artifact_paths:
        path_text = str(raw_path).strip()
        if not path_text or path_text in seen:
            continue
        seen.add(path_text)
        src = Path(path_text)
        if not src.exists() or not src.is_file():
            continue
        parent_name = src.parent.name
        if parent_name == "trends":
            dest = run_dir / src.name
        elif parent_name:
            dest = run_dir / parent_name / src.name
        else:
            dest = run_dir / src.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        copied_paths.append(str(dest))
    return copied_paths
