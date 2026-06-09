import io
import json
import re
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from textwrap import dedent
from typing import List, Optional
from unittest.mock import patch

from research_alpha.app import (
    IdeaGenerationNotReady,
    audit_genome_grounding,
    audit_pattern_grounding,
    audit_storyline_migration,
    audit_generation_evidence,
    build_idea_generation_failure_summary,
    build_candidate_idea_ranking,
    build_idea_semantic_retry_prompt,
    build_user_domain_knowledge_summary,
    idea_generation_failure_is_semantic_retryable,
    backfill_pattern_source_provenance,
    build_evidence_report,
    build_evidence_grounded_score,
    compute_strict_evidence_readiness,
    build_quality_enrich_prompt,
    gold_evidence_tier,
    infer_next_commands,
    infer_next_stages,
    parse_prior_art_gate_response,
    build_reviewer_gate_prompt,
    build_strict_evidence_audit,
    main,
    parse_quality_enrich_response,
    parse_reviewer_gate_response,
    render_run_report,
    is_poster_record,
    qualify_remote_quality_records,
    refresh_session_summary,
    run_auto_review_loop_for_session,
    run_pipeline_idea_stage,
    scholarly_search_query,
    user_library_reference_rejection_reason,
    validate_query_frontier_evidence,
)
from research_alpha.connectors import ConnectorError, cvf_award_key, eccv_award_key, harvest_cvf_awards, harvest_icml_awards, harvest_neurips_awards, harvest_openreview, icml_award_key, neurips_award_key, openreview_award_from_venue, parse_acl_awards_html, parse_cvf_awards_html, parse_cvf_oral_html, parse_eccv_awards_html, parse_generic_awards_html, parse_icml_awards_html, parse_icml_oral_html, parse_neurips_awards_html, parse_neurips_oral_html
from research_alpha.db import add_candidate_idea, add_paper, add_run, connect, delete_idea_session, get_candidate_idea, init_db, list_frontier_papers, list_top_papers, list_user_library_papers, upsert_idea_card, upsert_pattern_card, upsert_user_library_paper
from research_alpha.ideas import build_idea_prompt, parse_idea_response
from research_alpha.limitations import limitations_json_for_paper
from research_alpha.memory import build_session_memory_status
from research_alpha.prior_art import analyze_prior_art
from research_alpha.gui import (
    CONSTRAINT_OPTIONS,
    FIELD_OPTIONS,
    GUI_JOB_LIVE_TAIL_CHARS,
    GUI_JOB_MAX_STORED,
    GUI_JOB_RESULT_TAIL_CHARS,
    GUI_CONVERSATION_JOB_KINDS,
    GUI_EVIDENCE_JOB_KINDS,
    GUI_MAX_DIRECTION_CHARS,
    GUI_MAX_GOLD_QUERY_CHARS,
    GUI_MAX_JSON_BODY_BYTES,
    GUI_MAX_SCOPE_CHARS,
    GUI_MAX_VENUE_CHARS,
    GUI_MAX_VENUES,
    GuiState,
    HOTSPOT_OPTIONS,
    VENUE_GROUPS,
    VENUE_PRESETS,
    arxiv_id_from_url,
    build_chat_snapshot,
    build_gui_health_snapshot,
    build_gui_score_message,
    build_gui_snapshot,
    auxiliary_gold_venues,
    core_gold_venues,
    create_gui_http_server,
    current_gui_year,
    default_gold_from_year,
    default_gold_to_year,
    describe_gold_source_plan,
    extract_created_session_id,
    gui_error_response,
    infer_research_brief,
    parse_gui_bool,
    parse_gui_scope,
    metadata_only_gold_venues,
    enrich_user_library_paper,
    opportunistic_user_library_full_text_ingest,
    parse_venue_payload,
    parse_gui_choice,
    parse_gui_int,
    parse_gui_state_session_id,
    parse_gui_text,
    public_job_snapshot,
    redact_local_paths,
    redact_secrets,
    render_index_html,
    resolve_gui_session,
    resolve_gui_state_session_id,
    sanitize_user_facing_text,
    split_gold_venues_by_remote,
    write_http_response,
    year_from_arxiv_id,
)


def strict_evidence_basis(
    source_id: str = "hidden-assumption-reversal",
    used_for: str = "innovation|logic|feasibility|value|defensibility",
) -> list[dict[str, str]]:
    return [
        {
            "source_type": "pattern",
            "source_id": source_id,
            "quality_signal": "best_paper",
            "borrowed_standard": "Reverse an accepted setup.",
            "used_for": used_for,
        }
    ]


def strict_generation_guardrail() -> str:
    return "This idea is derived from stored high-quality evidence by transferring hidden-assumption reversal."


def strict_storyline_trace(source_id: str = "hidden-assumption-reversal") -> dict[str, dict[str, str]]:
    return {
        "old_belief": {
            "source_id": source_id,
            "paper_story_standard": "The source paper identified a dominant assumption before the contribution.",
            "transfer_to_current_hotspot": "Research-agent benchmarks assume narrow success metrics are enough.",
        },
        "bottleneck": {
            "source_id": source_id,
            "paper_story_standard": "The source paper made the hidden bottleneck explicit.",
            "transfer_to_current_hotspot": "Current agent evaluation hides brittle failure modes.",
        },
        "reframing": {
            "source_id": source_id,
            "paper_story_standard": "The source paper reversed the accepted setup.",
            "transfer_to_current_hotspot": "Turn benchmark design from capability scoring into failure discovery.",
        },
        "why_now": {
            "source_id": source_id,
            "paper_story_standard": "The source paper explained why the move became possible at that moment.",
            "transfer_to_current_hotspot": "Agent systems are now strong enough and deployed enough to need stress tests.",
        },
        "evidence_design": {
            "source_id": source_id,
            "paper_story_standard": "The source paper used evidence that made the reframing convincing.",
            "transfer_to_current_hotspot": "Use failure-taxonomy tasks and ranking-change analysis.",
        },
        "failure_boundary": {
            "source_id": source_id,
            "paper_story_standard": "The source paper made clear where the idea should fail.",
            "transfer_to_current_hotspot": "The idea fails if it becomes benchmark churn without mechanism.",
        },
    }


def strict_idea_payload() -> dict:
    return {
        "idea_title": "Stress-Tested Agent Benchmark",
        "core_hypothesis": "Research-agent evaluation should expose hidden assumption failures rather than only report task completion.",
        "historical_pattern": "hidden-assumption-reversal",
        "trend_support": "Recent agentic evaluation work needs stronger stress tests.",
        "frontier_gap": "Current evaluations miss long-horizon failure boundaries.",
        "why_now": "Research agents are becoming plausible enough that evaluation standards matter.",
        "novelty": "Reframes agent evaluation around hidden brittleness and reviewer-visible failure boundaries.",
        "value": "Improves trust in claims about scientific discovery agents.",
        "key_risk": "The protocol may overfit to visible failure modes.",
        "first_experiments": ["Compare ranking changes under ordinary tasks and assumption-boundary audits."],
        "evaluation_outline": "Measure ranking changes and failure-mode coverage.",
        "paper_angle": "Turns benchmark success into an audit of hidden assumptions.",
        "evidence_basis": strict_evidence_basis(),
        "generation_guardrail": strict_generation_guardrail(),
        "storyline_trace": strict_storyline_trace(),
    }


def prior_art_pass_json() -> str:
    return json.dumps(
        {
            "decision": "pass",
            "overlap_label": "complementary",
            "closest_work": [
                {
                    "paper_id": 3,
                    "title": "Outstanding Agent Stress Tests",
                    "overlap_reason": "Shares evaluation language but not the same contribution.",
                    "covered_parts": ["agent", "evaluation"],
                    "missing_parts": ["different novelty boundary and evidence target"],
                    "difference_from_idea": "The candidate targets a different causal boundary and scoring protocol.",
                }
            ],
            "renamed_only_risk": "low",
            "novelty_boundary": "The idea must change the causal evaluation boundary, not only rename stress tests.",
            "why_not_done_yet": ["Stored prior work lacks the candidate's stated novelty boundary and evaluation protocol."],
            "required_differentiation": ["Keep the novelty boundary explicit against the closest stress-test work."],
            "rethink_prompt": "",
            "summary": "Closest work is related but does not cover the full proposed idea.",
        }
    )


def seed_strict_quality_papers(cwd: Path) -> None:
    (cwd / "seeds").mkdir(parents=True, exist_ok=True)
    (cwd / "seeds" / "quality.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "title": f"Genome {chr(65 + idx)}",
                    "venue": "ICLR",
                    "year": 2024,
                    "award": award,
                    "citation_count": 1000 - idx * 100,
                    "influential_citation_count": 100 - idx * 10,
                    "abstract": "Benchmark assumptions hide agent failures and brittle evaluation.",
                }
            )
            for idx, award in enumerate(["best_paper", "oral", "outstanding_paper"])
        )
        + "\n",
        encoding="utf-8",
    )
    main(["h", "--file", "seeds/quality.jsonl"])
    main(["score"])


def grounded_fake_genome_json(source: str = "Benchmark assumptions hide agent failures and brittle evaluation.") -> str:
    theme = source.rstrip(".")
    payload = {
        "paper_summary": f"The abstract frames this paper around {theme}.",
        "pre_publication_belief": f"Before the paper, the field under-weighted {theme}.",
        "bottleneck_or_hidden_assumption": f"The bottleneck is exposed by {theme}.",
        "problem_reframing": f"Reframe the problem through {theme}.",
        "why_now": f"The abstract-only timing signal is bounded to {theme}.",
        "evidence_design": f"Use {theme} as the abstract-grounded evidence frame.",
        "story_line": f"{theme} becomes the paper's abstract-grounded logic line.",
        "transferable_pattern": f"Transfer the logic of {theme} without copying methods.",
        "failure_boundary": f"Fails when the new domain is not connected to {theme}.",
        "logic_line": {
            "old_belief": f"The field under-weighted {theme}.",
            "bottleneck": f"The bottleneck is exposed by {theme}.",
            "reframing": f"Reframe the problem through {theme}.",
            "why_now": f"The timing claim is abstract-only and bounded to {theme}.",
            "evidence_design": f"Use {theme} as the evidence frame.",
            "failure_boundary": f"Fails when the target setting is not tied to {theme}.",
        },
        "logic_line_evidence": {
            key: {
                "source_text": source,
                "bounded_inference": f"Abstract-only inference from {theme}.",
            }
            for key in [
                "old_belief",
                "bottleneck",
                "reframing",
                "why_now",
                "evidence_design",
                "failure_boundary",
            ]
        },
        "confidence_note": "Abstract-only confidence from title, venue, year, award, and abstract metadata.",
        "evidence_level": "abstract_only",
    }
    return json.dumps(payload)


def grounded_fake_pattern_json(examples: Optional[List[str]] = None) -> str:
    example_titles = examples or ["Genome A", "Genome C"]
    payload = {
        "pattern_key": "hidden-assumption-reversal",
        "pattern_name": "Hidden Assumption Reversal",
        "core_move": "Use benchmark assumptions to expose hidden agent failures and brittle evaluation.",
        "when_to_use": "When benchmark assumptions hide agent failures.",
        "why_it_works": "It generalizes because benchmark assumptions and brittle evaluation recur across source cards.",
        "failure_modes": "Fails when brittle evaluation or hidden agent failures are absent.",
        "canonical_examples": example_titles,
        "operator_template": "Extract benchmark assumptions, expose hidden agent failures, then design brittle evaluation tests.",
        "evidence_level": "logic_line_aggregation",
        "logic_line_pattern": {
            "source_logic": "Benchmark assumptions hide agent failures and brittle evaluation.",
            "transfer_rule": "Transfer benchmark assumptions into hidden failure discovery and brittle evaluation.",
            "why_it_generalizes": "Benchmark assumptions, agent failures, and brittle evaluation recur across source cards.",
            "failure_boundary": "Fails when source cards do not support benchmark assumptions or brittle evaluation.",
        },
    }
    return json.dumps(payload)


def grounded_fake_chat_response(prompt: str):
    if "Tool-use research agents expose planning and evaluation bottlenecks" in prompt:
        text = grounded_fake_genome_json("Tool-use research agents expose planning and evaluation bottlenecks.")
    elif "Stress tests reveal hidden assumptions in research agent systems" in prompt:
        text = grounded_fake_genome_json("Stress tests reveal hidden assumptions in research agent systems.")
    elif "Historical logic reveals benchmark assumptions" in prompt:
        text = grounded_fake_genome_json("Historical logic reveals benchmark assumptions, hidden agent failures, and brittle evaluation.")
    elif "A historical high-quality paper" in prompt:
        text = grounded_fake_genome_json("Historical logic reveals benchmark assumptions, hidden agent failures, and brittle evaluation.")
    elif "Benchmark assumptions hide agent failures" in prompt:
        text = grounded_fake_genome_json("Benchmark assumptions hide agent failures.")
    else:
        text = grounded_fake_genome_json()
    return type("Resp", (), {"text": text})()


def seed_strict_generation_evidence(db_path: Path) -> None:
    paper_ids = []
    for idx, (title, award, weight) in enumerate(
        [
            ("Best Agent Benchmark", "best_paper", 5.0),
            ("Oral Agent Tool Use", "oral", 2.0),
            ("Outstanding Agent Stress Tests", "outstanding_paper", 4.0),
        ]
    ):
        paper_id = add_paper(
            db_path,
            title,
            "ICLR",
            2026,
            "Benchmark assumptions hide agent failures and brittle evaluation.",
            source_kind="gold_openreview",
            award=award,
            citation_count=100 - idx * 10,
            influential_citation_count=40 - idx * 5,
        )
        paper_ids.append(paper_id)
        with connect(db_path) as conn:
            conn.execute(
                "UPDATE papers SET paper_weight=?, score_notes=? WHERE id=?",
                (weight, json.dumps({award: weight}), paper_id),
            )
        if idx < 2:
            upsert_idea_card(
                db_path,
                paper_id,
                "strict_logic_line",
                json.dumps(
                    {
                        "paper_summary": "A best paper about hidden benchmark failures.",
                        "problem_reframing": "Turn evaluation into failure discovery.",
                        "transferable_pattern": "Reverse the benchmark assumption.",
                        "evidence_design": "Show hidden failures under stress tests.",
                        "failure_boundary": "Fails if no hidden assumption exists.",
                        "logic_line": {
                            "old_belief": "Benchmarks measure progress.",
                            "bottleneck": "They hide brittle failures.",
                            "reframing": "Evaluate hidden assumptions directly.",
                            "why_now": "Agents are deployed in messier settings.",
                            "evidence_design": "Stress-test tasks expose failures.",
                            "failure_boundary": "Works only when assumptions are hidden.",
                        },
                        "grounding_audit": {"status": "valid"},
                    }
                ),
            )
    upsert_pattern_card(
        db_path,
        "hidden-assumption-reversal",
        json.dumps(
            {
                "pattern_name": "Hidden Assumption Reversal",
                "core_move": "Reverse the benchmark assumption.",
                "why_it_works": "It exposes brittle claims.",
                "operator_template": "Turn hidden assumptions into stress tests.",
                "when_to_use": "When a frontier metric hides failures.",
                "failure_modes": "Fails without measurable stress cases.",
                "logic_line_pattern": {
                    "source_logic": "A top paper turned hidden assumptions into evidence.",
                    "transfer_rule": "Move the hidden-assumption arc to a new frontier gap.",
                    "why_it_generalizes": "Many fields over-trust proxy metrics.",
                    "failure_boundary": "Do not copy the source method.",
                },
                "evidence_level": "strict_pattern",
                "grounding_audit": {"status": "valid"},
                "source_set_policy": "gold_set_only",
                "source_papers": [
                    {
                        "paper_id": paper_ids[0],
                        "title": "Best Agent Benchmark",
                        "venue": "ICLR",
                        "year": 2026,
                        "source_kind": "gold_openreview",
                        "paper_weight": 5.0,
                    }
                ],
                "source_paper_ids": [paper_ids[0]],
            }
        ),
    )


class AppTests(unittest.TestCase):
    def write_gui_test_llm_env(self, cwd: Path) -> None:
        (cwd / ".env").write_text(
            "RA_LLM_PROVIDER=deepseek\nRA_LLM_MODEL=deepseek-chat\nDEEPSEEK_API_KEY=test-gui-key\n",
            encoding="utf-8",
        )

    def gui_post_json(self, server, path, payload):
        raw = payload if isinstance(payload, str) else json.dumps(payload)
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}{path}",
            data=raw.encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8"))

    def test_gui_inline_script_is_javascript_parseable(self) -> None:
        html = render_index_html()
        scripts = re.findall(r"<script>([\s\S]*?)</script>", html)
        self.assertTrue(scripts)
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "gui-inline.js"
            script_path.write_text("\n".join(scripts), encoding="utf-8")
            result = subprocess.run(
                ["node", "--check", str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f'id="fromYear" type="number" value="{default_gold_from_year()}"', html)
        self.assertIn(f'id="toYear" type="number" value="{default_gold_to_year()}"', html)
        self.assertNotIn('id="toYear" type="number" value="2026"', html)

    def test_gui_activity_completion_summary_is_executable_javascript(self) -> None:
        html = render_index_html()
        scripts = re.findall(r"<script>([\s\S]*?)</script>", html)
        self.assertTrue(scripts)
        script = "\n".join(scripts).replace("\n    refresh();", "\n    // refresh disabled in this behavior harness")
        setup = """
const elements = {
  jobCount: {textContent:'', className:'', title:'', onclick:null},
  jobFeed: {textContent:'', className:'', title:'', onclick:null},
  instruction: {value:'', style:{}, scrollHeight:44, addEventListener() {}, focus() {}},
};
global.document = {
  getElementById(id) { return elements[id] || null; },
  addEventListener() {},
};
global.window = {lastData:{}, addEventListener() {}, setTimeout, clearTimeout};
global.setTimeout = setTimeout;
global.clearTimeout = clearTimeout;
global.console = console;
global.refresh = () => {};
"""
        harness = """
function assert(condition, message) { if (!condition) throw new Error(message); }
renderJobs({jobs:[{
  id: 1,
  kind: 'gold-build',
  name: 'gold-build test',
  status: 'completed',
  result: {stdout: 'Gold-build summary: fetched=7; candidate_imported=4; excellent_imported=2; poster_skipped=1; openreview_non_excellent_skipped=0; unqualified_skipped=0; failures=0', stderr: ''},
}], health:{}});
assert(elements.jobFeed.textContent.includes('Gold 2 · 趋势 4 · 失败 0'), elements.jobFeed.textContent);
renderJobs({jobs:[{
  id: 2,
  kind: 'quality-enrich',
  name: 'quality test',
  status: 'completed',
  result: {stdout: 'Applied labels: 3\\nSkipped labels: 5', stderr: ''},
}], health:{}});
assert(elements.jobFeed.textContent.includes('新增标签 3 · 跳过 5'), elements.jobFeed.textContent);
renderJobs({jobs:[{
  id: 4,
  kind: 'quality-enrich',
  name: 'new quality success',
  status: 'completed',
  completed_at: 40,
  result: {stdout: 'Applied labels: 1\\nSkipped labels: 0', stderr: ''},
}, {
  id: 3,
  kind: 'gold-build',
  name: 'old gold failure',
  status: 'failed',
  completed_at: 20,
  error: 'old failure',
}], health:{}});
assert(elements.jobFeed.textContent.includes('最近完成：趋势质量核验'), elements.jobFeed.textContent);
assert(!elements.jobFeed.textContent.includes('最近失败'), elements.jobFeed.textContent);
renderJobs({jobs:[{
  id: 6,
  kind: 'gold-build',
  name: 'ICML success',
  status: 'completed',
  group_id: 'gold-build:draft:test:99',
  completed_at: 60,
  result: {stdout: 'Gold-build summary: fetched=5; candidate_imported=5; excellent_imported=2; poster_skipped=0; openreview_non_excellent_skipped=0; unqualified_skipped=3; failures=0', stderr: ''},
}, {
  id: 7,
  kind: 'gold-build',
  name: 'OpenReview failed',
  status: 'failed',
  group_id: 'gold-build:draft:test:99',
  completed_at: 61,
  error: 'Gold-build summary: fetched=0; candidate_imported=0; excellent_imported=0; poster_skipped=0; openreview_non_excellent_skipped=0; unqualified_skipped=0; failures=1',
}], health:{}});
assert(elements.jobFeed.textContent.includes('部分完成：核心库扩充'), elements.jobFeed.textContent);
assert(elements.jobFeed.textContent.includes('Gold 2 · 趋势 5 · 失败 2'), elements.jobFeed.textContent);
assert(!elements.jobFeed.textContent.includes('最近失败'), elements.jobFeed.textContent);
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "gui-activity-summary.js"
            script_path.write_text(setup + "\n" + script + "\n" + harness, encoding="utf-8")
            result = subprocess.run(
                ["node", str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_remote_quality_build_keeps_only_core_award_records(self) -> None:
        records = [
            {
                "title": "Ordinary Top Venue Paper",
                "venue": "ICLR",
                "year": 2025,
                "citation_count": 80,
                "influential_citation_count": 8,
            },
            {
                "title": "Best Paper Storyline",
                "venue": "ICLR",
                "year": 2025,
                "award": "Best Paper",
                "citation_count": 50,
                "influential_citation_count": 5,
            },
            {
                "title": "Oral Paper Storyline",
                "venue": "ICLR",
                "year": 2025,
                "award": "oral",
                "citation_count": 40,
                "influential_citation_count": 4,
            },
            {
                "title": "High Citation Breakout",
                "venue": "ICLR",
                "year": 2025,
                "citation_count": 10000,
                "influential_citation_count": 900,
            },
            {
                "title": "Ordinary Top Venue Paper 2",
                "venue": "ICLR",
                "year": 2025,
                "citation_count": 60,
                "influential_citation_count": 6,
            },
            {
                "title": "Ordinary Top Venue Paper 3",
                "venue": "ICLR",
                "year": 2025,
                "citation_count": 70,
                "influential_citation_count": 7,
            },
        ]

        qualified = qualify_remote_quality_records(records, max_items=10)
        by_title = {str(item["title"]): item for item in qualified}

        self.assertEqual(set(by_title), {"Best Paper Storyline", "Oral Paper Storyline"})
        self.assertEqual(by_title["Best Paper Storyline"]["award"], "best_paper")
        self.assertEqual(by_title["Oral Paper Storyline"]["award"], "oral")

    def test_remote_quality_build_does_not_import_small_batch_without_quality_signal(self) -> None:
        records = [
            {
                "title": f"Venue Candidate {idx}",
                "venue": "NeurIPS",
                "year": 2025,
                "citation_count": 1000 - idx,
                "influential_citation_count": 100 - idx,
            }
            for idx in range(4)
        ]

        self.assertEqual(qualify_remote_quality_records(records), [])

    def test_poster_records_are_never_quality_records(self) -> None:
        records = [
            {
                "title": "Poster Track Paper",
                "venue": "ICLR 2025 Poster",
                "year": 2025,
                "award": "oral",
                "citation_count": 999,
            },
            {
                "title": "Oral Track Paper",
                "venue": "ICLR 2025 Oral",
                "year": 2025,
                "award": "oral",
                "citation_count": 12,
            },
        ]

        self.assertTrue(is_poster_record(records[0]))
        self.assertEqual(
            [item["title"] for item in qualify_remote_quality_records(records, max_items=10)],
            ["Oral Track Paper"],
        )

    def test_remote_quality_build_accepts_nomination_and_presentation_aliases(self) -> None:
        records = [
            {"title": "Nominee Paper", "venue": "NeurIPS", "year": 2025, "award": "Best Paper Nominee"},
            {"title": "Honorable Paper", "venue": "NeurIPS", "year": 2025, "award": "Honorable Mention"},
            {"title": "Oral Paper", "venue": "NeurIPS", "year": 2025, "award": "Oral Presentation"},
            {"title": "Spotlight Paper", "venue": "NeurIPS", "year": 2025, "award": "Spotlight Presentation"},
        ]

        qualified = qualify_remote_quality_records(records, max_items=10)
        by_title = {str(item["title"]): item for item in qualified}

        self.assertEqual(by_title["Nominee Paper"]["award"], "outstanding_paper")
        self.assertEqual(by_title["Honorable Paper"]["award"], "outstanding_paper")
        self.assertEqual(by_title["Oral Paper"]["award"], "oral")
        self.assertNotIn("Spotlight Paper", by_title)

    def test_openreview_venue_text_maps_to_quality_awards(self) -> None:
        self.assertEqual(openreview_award_from_venue("ICLR 2025 Oral"), "oral")
        self.assertEqual(openreview_award_from_venue("NeurIPS 2024 spotlight"), "spotlight")
        self.assertEqual(openreview_award_from_venue("ICLR 2024 poster"), "")
        self.assertEqual(openreview_award_from_venue("Best Paper Award"), "best_paper")
        self.assertEqual(openreview_award_from_venue("Outstanding Paper Award"), "outstanding_paper")
        self.assertEqual(openreview_award_from_venue("Best Paper Nominee"), "outstanding_paper")
        self.assertEqual(openreview_award_from_venue("Best Paper Award Candidate"), "outstanding_paper")
        self.assertEqual(openreview_award_from_venue("Best Paper Honourable Mention"), "outstanding_paper")
        self.assertEqual(openreview_award_from_venue("ICLR 2025 Submission Candidate"), "")
        self.assertEqual(openreview_award_from_venue("Candidate Paper"), "")
        self.assertEqual(openreview_award_from_venue("Candidate"), "")
        self.assertEqual(openreview_award_from_venue("Paper Nominee"), "outstanding_paper")

    def test_openreview_harvest_filters_out_posters_before_database_import(self) -> None:
        payload = {
            "notes": [
                {
                    "id": "poster",
                    "content": {
                        "title": {"value": "Poster Only Paper"},
                        "abstract": {"value": "Poster abstract."},
                        "venue": {"value": "ICLR 2025 Poster"},
                    },
                },
                {
                    "id": "oral",
                    "content": {
                        "title": {"value": "Oral Paper"},
                        "abstract": {"value": "Oral abstract."},
                        "venue": {"value": "ICLR 2025 Oral"},
                    },
                },
            ]
        }
        with patch("research_alpha.connectors._get_json", return_value=payload):
            records = harvest_openreview(venue="ICLR", year=2025, limit=10)
        self.assertEqual([record["title"] for record in records], ["Oral Paper"])
        self.assertEqual(records[0]["award"], "oral")

    def test_icml_awards_parser_keeps_only_official_award_papers(self) -> None:
        html = """
        <tr>
          <td><div>Outstanding Paper</div></td>
          <td><div class="virtual-card">
            <a class="small-title text-underline-hover" href="/virtual/2025/oral/47251">Train for the Worst</a>
          </div><div class="type_display_name_virtual_card">Oral</div>
          <div class="author-str">A. Author &middot; B. Author</div>
          <details><summary>Abstract</summary><div class="text-start p-4">A strong abstract.</div></details></td>
        </tr>
        <tr>
          <td><div>Poster</div></td>
          <td><div class="virtual-card">
            <a class="small-title text-underline-hover" href="/virtual/2025/poster/1">Ordinary Poster</a>
          </div></td>
        </tr>
        <tr>
          <td><div>Outstanding Paper</div></td>
          <td><div class="virtual-card">
            <a class="small-title text-underline-hover" href="/virtual/2025/poster/2">Train for the Worst</a>
          </div></td>
        </tr>
        """
        records = parse_icml_awards_html(html, year=2025)
        self.assertEqual([record["title"] for record in records], ["Train for the Worst"])
        self.assertEqual(records[0]["award"], "outstanding_paper")
        self.assertIn("/oral/", records[0]["external_ref"])
        self.assertIn("A. Author", records[0]["abstract"])
        self.assertEqual(icml_award_key("Best Paper"), "best_paper")
        self.assertEqual(icml_award_key("Test of Time Honorable Mention"), "test_of_time")

    def test_icml_awards_parser_handles_real_table_attrs_without_small_title_class(self) -> None:
        html = """
        <table>
          <tr class="award-row">
            <td class="label"><div>Outstanding Paper</div></td>
            <td class="paper">
              <div class="virtual-card">
                <a href="/virtual/2025/poster/44169">Score Matching with Missing Data</a>
              </div>
              <div class="type_display_name_virtual_card">Spotlight Poster</div>
              <div class="author-str">Josh Givens &middot; Song Liu</div>
            </td>
          </tr>
          <tr class="award-row">
            <td class="label"><div>Outstanding Paper</div></td>
            <td class="paper">
              <div class="virtual-card">
                <a href="/virtual/2025/oral/45990">Train for the Worst, Plan for the Best</a>
              </div>
              <div class="type_display_name_virtual_card">Oral</div>
            </td>
          </tr>
        </table>
        """
        records = parse_icml_awards_html(html, year=2025)
        by_title = {str(record["title"]): record for record in records}

        self.assertEqual(set(by_title), {"Score Matching with Missing Data", "Train for the Worst, Plan for the Best"})
        self.assertEqual(by_title["Score Matching with Missing Data"]["award"], "outstanding_paper")
        self.assertIn("/poster/44169", by_title["Score Matching with Missing Data"]["external_ref"])
        self.assertIn("Josh Givens", by_title["Score Matching with Missing Data"]["abstract"])
        self.assertIn("/oral/45990", by_title["Train for the Worst, Plan for the Best"]["external_ref"])

    def test_icml_oral_parser_imports_official_orals_without_posters(self) -> None:
        html = """
        <main>
          <div class="virtual-card">
            <a class="small-title" href="/virtual/2025/oral/101">Scaling Agent Evaluation</a>
            <div class="author-str">A. Author &middot; B. Author</div>
          </div>
          <div class="virtual-card">
            <a class="small-title" href="/virtual/2025/poster/202">Poster Should Stay Out</a>
          </div>
        </main>
        """
        records = parse_icml_oral_html(html, year=2025, source_url="https://icml.cc/virtual/2025/events/oral")
        self.assertEqual([record["title"] for record in records], ["Scaling Agent Evaluation"])
        self.assertEqual(records[0]["award"], "oral")
        self.assertIn("/oral/101", records[0]["external_ref"])
        self.assertIn("A. Author", records[0]["abstract"])

    def test_icml_oral_parser_requires_official_oral_signal_when_link_is_missing(self) -> None:
        html = """
        <main>
          <div class="virtual-card">
            <strong>Unlinked Generic Card Should Stay Out</strong>
            <div class="type_display_name_virtual_card">Poster</div>
          </div>
          <div class="virtual-card">
            <strong>Explicit Unlinked Oral Can Stay</strong>
            <div class="type_display_name_virtual_card">Oral</div>
          </div>
        </main>
        """
        records = parse_icml_oral_html(html, year=2025, source_url="https://icml.cc/virtual/2025/events/oral")
        self.assertEqual([record["title"] for record in records], ["Explicit Unlinked Oral Can Stay"])
        self.assertEqual(records[0]["external_ref"], "https://icml.cc/virtual/2025/events/oral")

    def test_acl_awards_parser_keeps_only_official_award_papers(self) -> None:
        html = """
        <h2 id="best-paper-awards">Best Paper Awards</h2>
        <ul>
          <li><strong>Mission: Impossible Language Models</strong><br /><em>Julie Kallini, Christopher Potts</em></li>
          <li><strong>Aya Model</strong><br /><em>Ahmet Ustun, Sara Hooker</em></li>
        </ul>
        <h2 id="outstanding-papers">Outstanding Papers</h2>
        <ul>
          <li><strong>Native Sparse Attention</strong><br /><em>Jingyang Yuan, Wenfeng Liang</em></li>
        </ul>
        <h2 id="regular-papers">Regular Papers</h2>
        <ul>
          <li><strong>Ordinary ACL Paper</strong><br /><em>Someone</em></li>
        </ul>
        <h2 id="poster-awards">Best Poster</h2>
        <ul>
          <li><strong>Poster Should Stay Out</strong><br /><em>Someone</em></li>
        </ul>
        <h2 id="industry-track-awards">Industry Track Awards</h2>
        <h3 id="industry-best-paper">Best Paper</h3>
        <ul>
          <li><strong>Industry Track Paper Should Stay Out</strong><br /><em>Someone</em></li>
        </ul>
        <h2 id="student-research-workshop-paper">Best Student Research Workshop Paper</h2>
        <ul>
          <li><strong>Student Workshop Paper Should Stay Out</strong><br /><em>Someone</em></li>
        </ul>
        """
        records = parse_acl_awards_html(html, year=2025, source_url="https://2025.aclweb.org/program/awards/")
        by_title = {str(record["title"]): record for record in records}

        self.assertEqual(set(by_title), {"Mission: Impossible Language Models", "Aya Model", "Native Sparse Attention"})
        self.assertEqual(by_title["Mission: Impossible Language Models"]["award"], "best_paper")
        self.assertEqual(by_title["Native Sparse Attention"]["award"], "outstanding_paper")
        self.assertIn("Julie Kallini", by_title["Mission: Impossible Language Models"]["abstract"])
        self.assertNotIn("Ordinary ACL Paper", by_title)
        self.assertNotIn("Poster Should Stay Out", by_title)
        self.assertNotIn("Industry Track Paper Should Stay Out", by_title)
        self.assertNotIn("Student Workshop Paper Should Stay Out", by_title)

    def test_acl_family_awards_parser_supports_emnlp_and_keeps_track_awards_out(self) -> None:
        html = """
        <h1>Best Papers</h1>
        <h2 id="best-paper-awards">Best Paper Awards</h2>
        <ul>
          <li><p>Generalization through Memorization<br /><em>Ada Lovelace</em></p></li>
        </ul>
        <h2 id="outstanding-papers">Outstanding Papers</h2>
        <ul>
          <li><a href="/papers/main/2">Tracing Long Context Failures</a><br /><em>Grace Hopper</em></li>
        </ul>
        <h2 id="best-demo-paper-award">Best Demo Paper Award</h2>
        <ul>
          <li><a href="/papers/demo/3">Demo Paper Should Stay Out</a><br /><em>Someone</em></li>
        </ul>
        <h2 id="best-poster-award">Best Poster Award</h2>
        <ul>
          <li><a href="/papers/poster/4">Poster Paper Should Stay Out</a><br /><em>Someone</em></li>
        </ul>
        """
        records = parse_acl_awards_html(html, year=2024, source_url="https://2024.emnlp.org/program/best_papers/", venue="EMNLP")
        by_title = {str(record["title"]): record for record in records}

        self.assertEqual(set(by_title), {"Generalization through Memorization", "Tracing Long Context Failures"})
        self.assertEqual(by_title["Generalization through Memorization"]["venue"], "EMNLP 2024 Best Paper Awards")
        self.assertEqual(by_title["Generalization through Memorization"]["award"], "best_paper")
        self.assertEqual(by_title["Tracing Long Context Failures"]["award"], "outstanding_paper")
        self.assertNotIn("Demo Paper Should Stay Out", by_title)
        self.assertNotIn("Poster Paper Should Stay Out", by_title)

    def test_neurips_awards_parser_keeps_only_official_award_papers(self) -> None:
        html = """
        <main>
          <h2>Best Paper Awards</h2>
          <ul>
            <li><a title="paper title" href="/paper_files/paper/2024/hash/best-Abstract-Conference.html">Geometry of Reasoning at Scale</a><i>A. Author, B. Author</i></li>
          </ul>
          <h2>Outstanding Paper Awards</h2>
          <ul>
            <li><a title="paper title" href="/paper_files/paper/2024/hash/outstanding-Abstract-Conference.html">Robust Tool-Use Evaluation</a><div class="authors">C. Author</div></li>
          </ul>
          <h2>Test of Time Award</h2>
          <ul>
            <li><strong>Foundations of Representation Learning</strong><em>D. Author</em></li>
          </ul>
          <h2>Poster Session</h2>
          <ul>
            <li><a href="/paper_files/paper/2024/hash/poster-Abstract-Conference.html">Poster Should Stay Out</a></li>
          </ul>
          <h2>Best Student Paper Award</h2>
          <ul>
            <li><strong>Student Paper Should Stay Out</strong></li>
          </ul>
          <h2>Workshop Best Paper</h2>
          <ul>
            <li><strong>Workshop Paper Should Stay Out</strong></li>
          </ul>
          <ul>
            <li>
              <a title="paper title" href="/paper_files/paper/2024/hash/badged-Abstract-Conference.html">Badge-Based Award Paper</a>
              <span>Best Paper Award</span>
            </li>
          </ul>
        </main>
        """
        records = parse_neurips_awards_html(html, year=2024, source_url="https://papers.nips.cc/paper_files/paper/2024")
        by_title = {str(record["title"]): record for record in records}

        self.assertEqual(set(by_title), {"Geometry of Reasoning at Scale", "Robust Tool-Use Evaluation", "Foundations of Representation Learning", "Badge-Based Award Paper"})
        self.assertEqual(by_title["Geometry of Reasoning at Scale"]["award"], "best_paper")
        self.assertEqual(by_title["Robust Tool-Use Evaluation"]["award"], "outstanding_paper")
        self.assertEqual(by_title["Foundations of Representation Learning"]["award"], "test_of_time")
        self.assertIn("A. Author", by_title["Geometry of Reasoning at Scale"]["abstract"])
        self.assertNotIn("Poster Should Stay Out", by_title)
        self.assertNotIn("Student Paper Should Stay Out", by_title)
        self.assertNotIn("Workshop Paper Should Stay Out", by_title)
        self.assertEqual(neurips_award_key("Best Paper Award"), "best_paper")
        self.assertEqual(neurips_award_key("Outstanding Paper Awards"), "outstanding_paper")
        self.assertEqual(neurips_award_key("Test-of-Time Award"), "test_of_time")
        self.assertEqual(neurips_award_key("Poster Award"), "")

    def test_neurips_awards_harvest_returns_empty_when_official_page_has_no_awards(self) -> None:
        with patch("research_alpha.connectors._get_text", side_effect=["<main><h1>NeurIPS papers</h1><ul><li><a title=\"paper title\" href=\"/paper\">Ordinary Paper</a></li></ul></main>", ConnectorError("404"), ConnectorError("404"), ConnectorError("404"), ConnectorError("404"), ConnectorError("404")]):
            self.assertEqual(harvest_neurips_awards(year=2024, limit=5), [])

    def test_neurips_oral_parser_imports_official_orals_without_posters(self) -> None:
        html = """
        <main>
          <div class="virtual-card">
            <a class="small-title" href="/virtual/2025/oral/101">Scaling Reliable Reasoning</a>
            <div class="author-str">A. Author &middot; B. Author</div>
          </div>
          <div class="virtual-card">
            <a class="small-title" href="/virtual/2025/poster/202">Poster Should Stay Out</a>
          </div>
        </main>
        """
        records = parse_neurips_oral_html(html, year=2025, source_url="https://neurips.cc/virtual/2025/events/oral")
        self.assertEqual([record["title"] for record in records], ["Scaling Reliable Reasoning"])
        self.assertEqual(records[0]["award"], "oral")
        self.assertIn("/oral/101", records[0]["external_ref"])
        self.assertIn("A. Author", records[0]["abstract"])

    def test_neurips_oral_parser_accepts_event_card_layout(self) -> None:
        html = """
        <main>
          <div class="event-card" data-event-type="Oral">
            <span class="event-type-badge">Oral</span>
            <h3 class="event-title">
              <a href="/virtual/2024/oral/97946">Human Expertise in Algorithmic Prediction</a>
            </h3>
            <div class="event-speakers">Rohan Alur ⋅ Manish Raghavan ⋅ Devavrat Shah</div>
            <div class="event-abstract"><div class="abstract-content">We study prediction with human expertise.</div></div>
          </div>
          <div class="event-card" data-event-type="Poster">
            <span class="event-type-badge">Poster</span>
            <h3 class="event-title">
              <a href="/virtual/2024/poster/111">Poster Should Stay Out</a>
            </h3>
          </div>
        </main>
        """
        records = parse_neurips_oral_html(html, year=2024, source_url="https://neurips.cc/virtual/2024/events/oral")
        by_title = {str(record["title"]): record for record in records}

        self.assertEqual(set(by_title), {"Human Expertise in Algorithmic Prediction"})
        self.assertEqual(by_title["Human Expertise in Algorithmic Prediction"]["award"], "oral")
        self.assertIn("Rohan Alur", by_title["Human Expertise in Algorithmic Prediction"]["abstract"])
        self.assertNotIn("Poster Should Stay Out", by_title)

    def test_neurips_oral_parser_rejects_unlinked_non_oral_cards(self) -> None:
        html = """
        <main>
          <div class="virtual-card">
            <strong>Unlinked Poster Should Stay Out</strong>
            <div class="type_display_name_virtual_card">Poster</div>
          </div>
          <div class="virtual-card">
            <strong>Unlinked Oral Should Stay</strong>
            <div class="type_display_name_virtual_card">Oral</div>
          </div>
        </main>
        """
        records = parse_neurips_oral_html(html, year=2025, source_url="https://neurips.cc/virtual/2025/events/oral")
        self.assertEqual([record["title"] for record in records], ["Unlinked Oral Should Stay"])

    def test_harvest_neurips_awards_merges_awards_and_official_orals(self) -> None:
        award_html = """
        <main>
          <h2>Best Paper Awards</h2>
          <ul><li><a title="paper title" href="/paper_files/paper/2025/hash/best">NeurIPS Best Paper</a></li></ul>
        </main>
        """
        oral_html = """
        <main>
          <div class="virtual-card"><a href="/virtual/2025/oral/2">NeurIPS Oral Paper</a></div>
          <div class="virtual-card"><a href="/virtual/2025/poster/3">NeurIPS Poster Paper</a></div>
        </main>
        """
        with patch("research_alpha.connectors._get_text", side_effect=[award_html, ConnectorError("404"), ConnectorError("404"), ConnectorError("404"), oral_html, ConnectorError("404")]):
            records = harvest_neurips_awards(year=2025, limit=10)
        by_title = {record["title"]: record for record in records}
        self.assertEqual(set(by_title), {"NeurIPS Best Paper", "NeurIPS Oral Paper"})
        self.assertEqual(by_title["NeurIPS Best Paper"]["award"], "best_paper")
        self.assertEqual(by_title["NeurIPS Oral Paper"]["award"], "oral")
        self.assertIn("/oral/2", by_title["NeurIPS Oral Paper"]["external_ref"])
        self.assertNotIn("NeurIPS Poster Paper", by_title)

    def test_cvf_awards_parser_keeps_only_official_award_papers(self) -> None:
        html = """
        <title>CVPR 2025 Best Papers and Best Demos</title>
        <main>
          <h2>Best Papers</h2>
          <p>Best Paper:</p>
          <p>Paper Name: Seeing the World Before It Happens</p>
          <p>Authors: A. Author</p>
          <h2>Honorable Mentions</h2>
          <ul><li><a href="/virtual/2025/oral/42">Robust Visual Reasoning</a><div class="author-str">B. Author &middot; C. Author</div></li></ul>
          <h2>Best Demo Award</h2>
          <p><strong>Demo Should Stay Out</strong></p>
          <h2>Best Student Paper</h2>
          <p><strong>Student Paper Should Stay Out</strong></p>
          <h2>Workshop Awards</h2>
          <h3>Best Paper</h3>
          <p><strong>Workshop Paper Should Stay Out</strong></p>
          <h2>Best Paper Award</h2>
          <div class="virtual-card">
            <a class="small-title" href="/virtual/2025/poster/33887">Poster Link Should Stay Out</a>
          </div>
          <div class="type_display_name_virtual_card">Poster</div>
        </main>
        """
        records = parse_cvf_awards_html(html, venue="CVPR", year=2025, source_url="https://cvpr.thecvf.com/Conferences/2025/BestPapersDemos")
        by_title = {str(record["title"]): record for record in records}

        self.assertEqual(set(by_title), {"Seeing the World Before It Happens", "Robust Visual Reasoning"})
        self.assertEqual(by_title["Seeing the World Before It Happens"]["award"], "best_paper")
        self.assertEqual(by_title["Robust Visual Reasoning"]["award"], "outstanding_paper")
        self.assertIn("/oral/42", by_title["Robust Visual Reasoning"]["external_ref"])
        self.assertEqual(cvf_award_key("Marr Prize Best Paper"), "best_paper")
        self.assertEqual(cvf_award_key("Best Paper Honorable Mention"), "outstanding_paper")
        self.assertEqual(cvf_award_key("Best Demo Award"), "")
        self.assertEqual(cvf_award_key("Best Student Paper"), "")
        self.assertNotIn("Poster Link Should Stay Out", by_title)

    def test_cvf_oral_parser_imports_official_orals_without_posters(self) -> None:
        html = """
        <main>
          <div class="virtual-card">
            <a class="small-title" href="/virtual/2025/oral/101">Scaling Visual Reasoning Tests</a>
            <div class="author-str">A. Author &middot; B. Author</div>
            <details><summary>Abstract</summary><div>Oral paper abstract.</div></details>
          </div>
          <div class="virtual-card">
            <a class="small-title" href="/virtual/2025/poster/202">Poster Should Stay Out</a>
            <div class="type_display_name_virtual_card">Poster</div>
          </div>
        </main>
        """
        records = parse_cvf_oral_html(html, venue="CVPR", year=2025, source_url="https://cvpr.thecvf.com/virtual/2025/events/oral")
        by_title = {str(record["title"]): record for record in records}

        self.assertEqual(set(by_title), {"Scaling Visual Reasoning Tests"})
        self.assertEqual(by_title["Scaling Visual Reasoning Tests"]["award"], "oral")
        self.assertIn("/oral/101", by_title["Scaling Visual Reasoning Tests"]["external_ref"])
        self.assertIn("A. Author", by_title["Scaling Visual Reasoning Tests"]["abstract"])

    def test_cvf_oral_parser_rejects_unlinked_non_oral_cards(self) -> None:
        html = """
        <main>
          <div class="virtual-card">
            <strong>Unlinked Generic CVF Card</strong>
            <div class="type_display_name_virtual_card">Poster</div>
          </div>
          <div class="virtual-card">
            <strong>Unlinked Oral CVF Card</strong>
            <div class="type_display_name_virtual_card">Oral</div>
          </div>
        </main>
        """
        records = parse_cvf_oral_html(html, venue="CVPR", year=2025, source_url="https://cvpr.thecvf.com/virtual/2025/events/oral")
        self.assertEqual([record["title"] for record in records], ["Unlinked Oral CVF Card"])

    def test_harvest_cvf_awards_merges_awards_and_official_orals(self) -> None:
        award_html = """
        <main>
          <h2>Best Paper Award</h2>
          <ul><li><a href="/virtual/2025/oral/1">CVPR Best Paper</a></li></ul>
        </main>
        """
        oral_html = """
        <main>
          <div class="virtual-card"><a href="/virtual/2025/oral/2">CVPR Oral Paper</a></div>
          <div class="virtual-card"><a href="/virtual/2025/poster/3">CVPR Poster Paper</a></div>
        </main>
        """
        with patch("research_alpha.connectors._get_text", side_effect=[award_html, ConnectorError("404"), ConnectorError("404"), oral_html, ConnectorError("404")]):
            records = harvest_cvf_awards(venue="CVPR", year=2025, limit=10)
        by_title = {str(record["title"]): record for record in records}

        self.assertEqual(by_title["CVPR Best Paper"]["award"], "best_paper")
        self.assertEqual(by_title["CVPR Oral Paper"]["award"], "oral")
        self.assertNotIn("CVPR Poster Paper", by_title)

    def test_eccv_awards_parser_keeps_only_official_award_papers(self) -> None:
        html = """
        <title>ECCV 2024 Awards</title>
        <main>
          <h2>Best Paper Award</h2>
          <ul><li><a href="/virtual/2024/oral/11">A Geometry-Aware Foundation Model</a><div class="author-str">A. Author</div></li></ul>
          <h2>Best Paper Honorable Mention</h2>
          <p><strong>Efficient Multimodal Reasoning</strong><br />Authors: B. Author</p>
          <h2>Best Paper Award Candidates</h2>
          <ul><li><a href="/virtual/2024/oral/12">Robust Scene Understanding</a></li></ul>
          <h2>Best Demo Award</h2>
          <p><strong>Demo Should Stay Out</strong></p>
          <h2>Best Student Paper</h2>
          <p><strong>Student Paper Should Stay Out</strong></p>
          <h2>Best Paper Award</h2>
          <div class="virtual-card">
            <a class="small-title" href="/virtual/2024/poster/7">Poster Link Should Stay Out</a>
          </div>
          <div class="type_display_name_virtual_card">Poster</div>
        </main>
        """
        records = parse_eccv_awards_html(html, year=2024, source_url="https://eccv.ecva.net/Conferences/2024/Awards")
        by_title = {str(record["title"]): record for record in records}

        self.assertEqual(set(by_title), {"A Geometry-Aware Foundation Model", "Efficient Multimodal Reasoning", "Robust Scene Understanding"})
        self.assertEqual(by_title["A Geometry-Aware Foundation Model"]["award"], "best_paper")
        self.assertEqual(by_title["Efficient Multimodal Reasoning"]["award"], "outstanding_paper")
        self.assertEqual(by_title["Robust Scene Understanding"]["award"], "outstanding_paper")
        self.assertIn("/oral/11", by_title["A Geometry-Aware Foundation Model"]["external_ref"])
        self.assertEqual(eccv_award_key("Best Paper Award"), "best_paper")
        self.assertEqual(eccv_award_key("Best Paper Honorable Mention"), "outstanding_paper")
        self.assertEqual(eccv_award_key("Best Paper Award Candidates"), "outstanding_paper")
        self.assertEqual(eccv_award_key("Best Demo Award"), "")
        self.assertNotIn("Poster Link Should Stay Out", by_title)

    def test_generic_awards_parser_filters_auxiliary_tracks(self) -> None:
        html = """
        <main>
          <h2>Best Paper Award</h2>
          <ul><li><a href="/paper/1">Core Method Paper</a><em>A. Author</em></li></ul>
          <h2>Outstanding Paper Awards</h2>
          <p><strong>Strong Runner Paper</strong><br />Authors: B. Author</p>
          <h2>Runner-up Best Paper</h2>
          <ul><li><a href="/paper/2">Runner Up Paper</a></li></ul>
          <h2>Best Poster Award</h2>
          <ul><li><a href="/poster/3">Poster Should Stay Out</a></li></ul>
          <h2>Best Student Paper</h2>
          <p><strong>Student Should Stay Out</strong></p>
          <h2>Best Demonstration Award</h2>
          <p><strong>Demo Should Stay Out</strong></p>
        </main>
        """
        records = parse_generic_awards_html(html, venue="AAAI", year=2025, source_url="https://aaai.org/awards")
        by_title = {str(record["title"]): record for record in records}

        self.assertEqual(set(by_title), {"Core Method Paper", "Strong Runner Paper", "Runner Up Paper"})
        self.assertEqual(by_title["Core Method Paper"]["award"], "best_paper")
        self.assertEqual(by_title["Strong Runner Paper"]["award"], "outstanding_paper")
        self.assertEqual(by_title["Runner Up Paper"]["award"], "outstanding_paper")
        self.assertNotIn("Poster Should Stay Out", by_title)
        self.assertNotIn("Student Should Stay Out", by_title)
        self.assertNotIn("Demo Should Stay Out", by_title)

    def test_harvest_file_skips_posters_before_database_import(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                seed_path = cwd / "seed.json"
                seed_path.write_text(
                    json.dumps(
                        [
                            {
                                "title": "Seed Poster Paper",
                                "abstract": "Poster abstract.",
                                "venue": "ICLR 2025 Poster",
                                "year": 2025,
                                "award": "poster",
                            },
                            {
                                "title": "Seed Oral Paper",
                                "abstract": "Oral abstract.",
                                "venue": "ICLR 2025 Oral",
                                "year": 2025,
                                "award": "oral",
                            },
                        ]
                    ),
                    encoding="utf-8",
                )
                output = io.StringIO()
                with redirect_stdout(output):
                    exit_code = main(["harvest", "--file", str(seed_path)])
                self.assertEqual(exit_code, 0)
                self.assertIn("Skipped 1 poster records", output.getvalue())
                with connect(cwd / "data" / "research_alpha.db") as conn:
                    titles = [
                        str(row["title"])
                        for row in conn.execute("select title from papers order by title").fetchall()
                    ]
                self.assertEqual(titles, ["Seed Oral Paper"])
            finally:
                os.chdir(old_cwd)

    def test_gold_build_icml_awards_imports_official_award_papers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                payload = [
                    {
                        "title": "ICML Official Outstanding Paper",
                        "abstract": "An official ICML award paper.",
                        "venue": "ICML 2025 Outstanding Paper",
                        "year": 2025,
                        "external_ref": "https://icml.cc/virtual/2025/oral/1",
                        "award": "outstanding_paper",
                    }
                ]
                with patch("research_alpha.app.harvest_icml_awards", return_value=payload):
                    output = io.StringIO()
                    with redirect_stdout(output):
                        exit_code = main(
                            [
                                "gold-build",
                                "--remote",
                                "icml_awards",
                                "--venues",
                                "ICML",
                                "--from-year",
                                "2025",
                                "--to-year",
                                "2025",
                                "--per-venue-year",
                                "5",
                            ]
                        )
                self.assertEqual(exit_code, 0)
                self.assertIn("优秀论文库 imported/updated 1 records from icml_awards", output.getvalue())
                with connect(cwd / "data" / "research_alpha.db") as conn:
                    row = conn.execute("select source_kind, award, paper_weight from papers where title=?", ("ICML Official Outstanding Paper",)).fetchone()
                self.assertEqual(row["source_kind"], "gold_icml_awards")
                self.assertEqual(row["award"], "outstanding_paper")
                self.assertGreater(float(row["paper_weight"]), 0)
            finally:
                os.chdir(old_cwd)

    def test_harvest_icml_awards_merges_awards_and_official_orals(self) -> None:
        award_html = """
        <tr>
          <td><div>Outstanding Paper</div></td>
          <td><div class="virtual-card"><a href="/virtual/2025/oral/1">ICML Outstanding Paper</a></div></td>
        </tr>
        """
        oral_html = """
        <main>
          <div class="virtual-card"><a href="/virtual/2025/oral/2">ICML Oral Paper</a></div></div>
          <div class="virtual-card"><a href="/virtual/2025/poster/3">ICML Poster Paper</a></div></div>
        </main>
        """
        with patch("research_alpha.connectors._get_text", side_effect=[award_html, oral_html, ConnectorError("404")]):
            records = harvest_icml_awards(year=2025, limit=10)
        by_title = {record["title"]: record for record in records}
        self.assertEqual(by_title["ICML Outstanding Paper"]["award"], "outstanding_paper")
        self.assertEqual(by_title["ICML Oral Paper"]["award"], "oral")
        self.assertNotIn("ICML Poster Paper", by_title)

    def test_gold_build_icml_awards_hard_filters_official_poster_links(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                payload = [
                    {
                        "title": "ICML Outstanding But Poster Link",
                        "abstract": "An official ICML award row whose presentation link is poster.",
                        "venue": "ICML 2025 Outstanding Paper",
                        "year": 2025,
                        "external_ref": "https://icml.cc/virtual/2025/poster/44169",
                        "award": "outstanding_paper",
                    }
                ]
                with patch("research_alpha.app.harvest_icml_awards", return_value=payload):
                    output = io.StringIO()
                    with redirect_stdout(output):
                        exit_code = main(
                            [
                                "gold-build",
                                "--remote",
                                "icml_awards",
                                "--venues",
                                "ICML",
                                "--from-year",
                                "2025",
                                "--to-year",
                                "2025",
                                "--per-venue-year",
                                "5",
                            ]
                        )
                self.assertEqual(exit_code, 0)
                self.assertIn("skipped 1 poster records", output.getvalue())
                self.assertIn("优秀论文库 imported/updated 0 records from icml_awards", output.getvalue())
                with connect(cwd / "data" / "research_alpha.db") as conn:
                    row = conn.execute("select id from papers where title=?", ("ICML Outstanding But Poster Link",)).fetchone()
                self.assertIsNone(row)
            finally:
                os.chdir(old_cwd)

    def test_gold_build_neurips_awards_imports_official_award_papers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                payload = [
                    {
                        "title": "NeurIPS Official Best Paper",
                        "abstract": "An official NeurIPS award paper.",
                        "venue": "NeurIPS 2024 Best Paper",
                        "year": 2024,
                        "external_ref": "https://papers.nips.cc/paper_files/paper/2024",
                        "award": "best_paper",
                    }
                ]
                with patch("research_alpha.app.harvest_neurips_awards", return_value=payload):
                    output = io.StringIO()
                    with redirect_stdout(output):
                        exit_code = main(
                            [
                                "gold-build",
                                "--remote",
                                "neurips_awards",
                                "--venues",
                                "NeurIPS",
                                "--from-year",
                                "2024",
                                "--to-year",
                                "2024",
                                "--per-venue-year",
                                "5",
                            ]
                        )
                self.assertEqual(exit_code, 0)
                self.assertIn("优秀论文库 imported/updated 1 records from neurips_awards", output.getvalue())
                with connect(cwd / "data" / "research_alpha.db") as conn:
                    row = conn.execute("select source_kind, award, paper_weight from papers where title=?", ("NeurIPS Official Best Paper",)).fetchone()
                self.assertEqual(row["source_kind"], "gold_neurips_awards")
                self.assertEqual(row["award"], "best_paper")
                self.assertGreater(float(row["paper_weight"]), 0)
            finally:
                os.chdir(old_cwd)

    def test_gold_build_neurips_awards_imports_official_orals_but_not_posters(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                payload = [
                    {
                        "title": "NeurIPS Official Oral Paper",
                        "abstract": "An official NeurIPS oral paper.",
                        "venue": "NeurIPS 2025 Oral",
                        "year": 2025,
                        "external_ref": "https://neurips.cc/virtual/2025/oral/2",
                        "award": "oral",
                    },
                    {
                        "title": "NeurIPS Poster Should Stay Out",
                        "abstract": "A poster record returned by a flaky source.",
                        "venue": "NeurIPS 2025 Poster",
                        "year": 2025,
                        "external_ref": "https://neurips.cc/virtual/2025/poster/3",
                        "award": "poster",
                    },
                ]
                with patch("research_alpha.app.harvest_neurips_awards", return_value=payload):
                    output = io.StringIO()
                    with redirect_stdout(output):
                        exit_code = main(
                            [
                                "gold-build",
                                "--remote",
                                "neurips_awards",
                                "--venues",
                                "NeurIPS",
                                "--from-year",
                                "2025",
                                "--to-year",
                                "2025",
                                "--per-venue-year",
                                "5",
                            ]
                        )
                self.assertEqual(exit_code, 0)
                text = output.getvalue()
                self.assertIn("skipped 1 poster records", text)
                self.assertIn("优秀论文库 imported/updated 1 records from neurips_awards", text)
                with connect(cwd / "data" / "research_alpha.db") as conn:
                    rows = conn.execute("select title, source_kind, award, paper_weight from papers order by title").fetchall()
                by_title = {row["title"]: row for row in rows}
                self.assertEqual(by_title["NeurIPS Official Oral Paper"]["source_kind"], "gold_neurips_awards")
                self.assertEqual(by_title["NeurIPS Official Oral Paper"]["award"], "oral")
                self.assertGreater(float(by_title["NeurIPS Official Oral Paper"]["paper_weight"]), 0)
                self.assertNotIn("NeurIPS Poster Should Stay Out", by_title)
            finally:
                os.chdir(old_cwd)

    def test_gold_build_official_awards_are_not_query_filtered(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                payload = [
                    {
                        "title": "NeurIPS Official Best Paper Outside Current Topic",
                        "abstract": "An official award paper with no research-agent terms.",
                        "venue": "NeurIPS 2024 Best Paper",
                        "year": 2024,
                        "external_ref": "https://papers.nips.cc/paper_files/paper/2024/outside-topic",
                        "award": "best_paper",
                    }
                ]

                def fake_awards(*, year, limit, query=""):
                    self.assertEqual(query, "")
                    return payload

                with patch("research_alpha.app.harvest_neurips_awards", side_effect=fake_awards):
                    output = io.StringIO()
                    with redirect_stdout(output):
                        exit_code = main(
                            [
                                "gold-build",
                                "--remote",
                                "neurips_awards",
                                "--venues",
                                "NeurIPS",
                                "--from-year",
                                "2024",
                                "--to-year",
                                "2024",
                                "--per-venue-year",
                                "5",
                                "--query",
                                "research agents reliable evaluation",
                            ]
                        )
                self.assertEqual(exit_code, 0)
                self.assertIn("优秀论文库 imported/updated 1 records from neurips_awards", output.getvalue())
                with connect(cwd / "data" / "research_alpha.db") as conn:
                    row = conn.execute(
                        "select source_kind, award, paper_weight from papers where title=?",
                        ("NeurIPS Official Best Paper Outside Current Topic",),
                    ).fetchone()
                self.assertIsNotNone(row)
                self.assertEqual(row["source_kind"], "gold_neurips_awards")
                self.assertEqual(row["award"], "best_paper")
                self.assertGreater(float(row["paper_weight"]), 0.0)
            finally:
                os.chdir(old_cwd)

    def test_gold_build_acl_awards_keeps_official_award_papers_auxiliary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                payload = [
                    {
                        "title": "ACL Best Paper",
                        "abstract": "Authors: A. Author",
                        "venue": "ACL 2025 Best Paper Awards",
                        "year": 2025,
                        "external_ref": "https://2025.aclweb.org/program/awards/",
                        "award": "best_paper",
                    }
                ]
                with patch("research_alpha.app.harvest_acl_awards", return_value=payload):
                    output = io.StringIO()
                    with redirect_stdout(output):
                        exit_code = main(
                            [
                                "gold-build",
                                "--remote",
                                "acl_awards",
                                "--venues",
                                "ACL",
                                "--from-year",
                                "2025",
                                "--to-year",
                                "2025",
                                "--per-venue-year",
                                "5",
                            ]
                        )
                self.assertEqual(exit_code, 0)
                self.assertIn("优秀论文库 imported/updated 0 records from acl_awards", output.getvalue())
                self.assertIn("not part of the core Gold standard set", output.getvalue())
                with connect(cwd / "data" / "research_alpha.db") as conn:
                    row = conn.execute("select source_kind, award, paper_weight from papers where title=?", ("ACL Best Paper",)).fetchone()
                self.assertEqual(row["source_kind"], "frontier_acl_awards")
                self.assertEqual(row["award"], "")
                self.assertEqual(float(row["paper_weight"]), 0)
            finally:
                os.chdir(old_cwd)

    def test_gold_build_cvf_awards_imports_official_award_papers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                payload = [
                    {
                        "title": "CVPR Best Paper",
                        "abstract": "Authors: A. Author",
                        "venue": "CVPR 2025 Best Paper Award",
                        "year": 2025,
                        "external_ref": "https://cvpr.thecvf.com/Conferences/2025/BestPapersDemos",
                        "award": "best_paper",
                    }
                ]
                with patch("research_alpha.app.harvest_cvf_awards", return_value=payload):
                    output = io.StringIO()
                    with redirect_stdout(output):
                        exit_code = main(
                            [
                                "gold-build",
                                "--remote",
                                "cvf_awards",
                                "--venues",
                                "CVPR",
                                "--from-year",
                                "2025",
                                "--to-year",
                                "2025",
                                "--per-venue-year",
                                "5",
                            ]
                        )
                self.assertEqual(exit_code, 0)
                self.assertIn("优秀论文库 imported/updated 1 records from cvf_awards", output.getvalue())
                with connect(cwd / "data" / "research_alpha.db") as conn:
                    row = conn.execute("select source_kind, award, paper_weight from papers where title=?", ("CVPR Best Paper",)).fetchone()
                self.assertEqual(row["source_kind"], "gold_cvf_awards")
                self.assertEqual(row["award"], "best_paper")
                self.assertGreater(float(row["paper_weight"]), 0)
            finally:
                os.chdir(old_cwd)

    def test_gold_build_eccv_awards_keeps_official_award_papers_auxiliary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                payload = [
                    {
                        "title": "ECCV Best Paper",
                        "abstract": "Authors: A. Author",
                        "venue": "ECCV 2024 Best Paper Award",
                        "year": 2024,
                        "external_ref": "https://eccv.ecva.net/Conferences/2024/Awards",
                        "award": "best_paper",
                    }
                ]
                with patch("research_alpha.app.harvest_eccv_awards", return_value=payload):
                    output = io.StringIO()
                    with redirect_stdout(output):
                        exit_code = main(
                            [
                                "gold-build",
                                "--remote",
                                "eccv_awards",
                                "--venues",
                                "ECCV",
                                "--from-year",
                                "2024",
                                "--to-year",
                                "2024",
                                "--per-venue-year",
                                "5",
                            ]
                        )
                self.assertEqual(exit_code, 0)
                self.assertIn("优秀论文库 imported/updated 0 records from eccv_awards", output.getvalue())
                self.assertIn("not part of the core Gold standard set", output.getvalue())
                with connect(cwd / "data" / "research_alpha.db") as conn:
                    row = conn.execute("select source_kind, award, paper_weight from papers where title=?", ("ECCV Best Paper",)).fetchone()
                self.assertEqual(row["source_kind"], "frontier_eccv_awards")
                self.assertEqual(row["award"], "")
                self.assertEqual(float(row["paper_weight"]), 0)
            finally:
                os.chdir(old_cwd)

    def test_gold_build_openreview_imports_only_oral_as_excellent_paper(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                main(["init"])
                payload = [
                    {
                        "title": "OpenReview Oral Paper",
                        "abstract": "An oral paper.",
                        "venue": "ICLR 2025 Oral",
                        "year": 2025,
                        "external_ref": "https://openreview.net/pdf?id=oral",
                        "citation_count": 0,
                        "influential_citation_count": 0,
                        "award": "oral",
                    },
                    {
                        "title": "OpenReview Spotlight Paper",
                        "abstract": "A spotlight paper.",
                        "venue": "ICLR 2025 Spotlight",
                        "year": 2025,
                        "external_ref": "https://openreview.net/pdf?id=spotlight",
                        "citation_count": 0,
                        "influential_citation_count": 0,
                        "award": "spotlight",
                    },
                    {
                        "title": "OpenReview Poster Paper",
                        "abstract": "A poster paper.",
                        "venue": "ICLR 2025 Poster",
                        "year": 2025,
                        "external_ref": "https://openreview.net/pdf?id=poster",
                        "citation_count": 0,
                        "influential_citation_count": 0,
                        "award": "",
                    },
                ]
                with patch("research_alpha.app.load_remote_records", return_value=payload):
                    output = io.StringIO()
                    with redirect_stdout(output):
                        exit_code = main(
                            [
                                "gold-build",
                                "--remote",
                                "openreview",
                                "--venues",
                                "ICLR",
                                "--from-year",
                                "2025",
                                "--to-year",
                                "2025",
                                "--per-venue-year",
                                "10",
                            ]
                        )
                self.assertEqual(exit_code, 0)
                self.assertIn("skipped 1 poster records", output.getvalue())
                self.assertIn("联网趋势论文 imported/updated 1 records from openreview", output.getvalue())
                self.assertIn("优秀论文库 imported/updated 1 records from openreview", output.getvalue())
                with connect(cwd / "data" / "research_alpha.db") as conn:
                    rows = conn.execute(
                        "select title, source_kind, award, paper_weight from papers order by title"
                    ).fetchall()
                by_title = {row["title"]: row for row in rows}
                self.assertEqual(by_title["OpenReview Oral Paper"]["source_kind"], "gold_openreview")
                self.assertEqual(by_title["OpenReview Oral Paper"]["award"], "oral")
                self.assertGreater(float(by_title["OpenReview Oral Paper"]["paper_weight"]), 0)
                self.assertNotIn("OpenReview Spotlight Paper", by_title)
                self.assertNotIn("OpenReview Poster Paper", by_title)
            finally:
                os.chdir(old_cwd)

    def test_gold_build_openreview_rejects_generic_candidate_venue_as_non_excellent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                main(["init"])
                payload = [
                    {
                        "title": "OpenReview Generic Candidate Paper",
                        "abstract": "A generic candidate field, not a best-paper candidate.",
                        "venue": "ICLR 2025 Submission Candidate",
                        "year": 2025,
                        "external_ref": "https://openreview.net/pdf?id=generic-candidate",
                        "citation_count": 0,
                        "influential_citation_count": 0,
                        "award": openreview_award_from_venue("ICLR 2025 Submission Candidate"),
                    },
                    {
                        "title": "OpenReview Best Paper Candidate",
                        "abstract": "A true best-paper candidate.",
                        "venue": "ICLR 2025 Best Paper Candidate",
                        "year": 2025,
                        "external_ref": "https://openreview.net/pdf?id=best-candidate",
                        "citation_count": 0,
                        "influential_citation_count": 0,
                        "award": openreview_award_from_venue("ICLR 2025 Best Paper Candidate"),
                    },
                ]
                with patch("research_alpha.app.load_remote_records", return_value=payload):
                    output = io.StringIO()
                    with redirect_stdout(output):
                        exit_code = main(
                            [
                                "gold-build",
                                "--remote",
                                "openreview",
                                "--venues",
                                "ICLR",
                                "--from-year",
                                "2025",
                                "--to-year",
                                "2025",
                                "--per-venue-year",
                                "10",
                            ]
                        )
                self.assertEqual(exit_code, 0)
                self.assertIn("skipped 1 OpenReview poster/non-excellent records", output.getvalue())
                self.assertIn("优秀论文库 imported/updated 1 records from openreview", output.getvalue())
                with connect(cwd / "data" / "research_alpha.db") as conn:
                    rows = conn.execute(
                        "select title, source_kind, award, paper_weight from papers order by title"
                    ).fetchall()
                by_title = {row["title"]: row for row in rows}
                self.assertNotIn("OpenReview Generic Candidate Paper", by_title)
                self.assertEqual(by_title["OpenReview Best Paper Candidate"]["source_kind"], "gold_openreview")
                self.assertEqual(by_title["OpenReview Best Paper Candidate"]["award"], "outstanding_paper")
                self.assertGreater(float(by_title["OpenReview Best Paper Candidate"]["paper_weight"]), 0)
            finally:
                os.chdir(old_cwd)

    def test_gold_build_openalex_metadata_only_stays_frontier_until_quality_enrich(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                main(["init"])
                payload = [
                    {
                        "title": "TPAMI Metadata Outstanding Candidate",
                        "abstract": "A metadata-only TPAMI record that still needs quality enrichment.",
                        "venue": "IEEE Transactions on Pattern Analysis and Machine Intelligence",
                        "year": 2025,
                        "external_ref": "https://openalex.org/Wtpami",
                        "citation_count": 120,
                        "influential_citation_count": 30,
                        "award": "outstanding_paper",
                    }
                ]
                with patch("research_alpha.app.load_remote_records", return_value=payload):
                    output = io.StringIO()
                    with redirect_stdout(output):
                        exit_code = main(
                            [
                                "gold-build",
                                "--remote",
                                "openalex",
                                "--venues",
                                "TPAMI",
                                "--from-year",
                                "2025",
                                "--to-year",
                                "2025",
                                "--per-venue-year",
                                "5",
                            ]
                        )
                text = output.getvalue()
                self.assertEqual(exit_code, 0)
                self.assertIn("优秀论文库 imported/updated 0 records from openalex", text)
                self.assertIn("Gold-build summary JSON:", text)
                summary_line = next(line for line in text.splitlines() if line.startswith("Gold-build summary JSON: "))
                summary = json.loads(summary_line.removeprefix("Gold-build summary JSON: "))
                self.assertTrue(summary["metadata_only_remote"])
                self.assertEqual(summary["source_mode"], "metadata_only")
                self.assertEqual(summary["candidate_imported"], 1)
                self.assertEqual(summary["excellent_imported"], 0)
                self.assertEqual(summary["current_excellent_papers"], 0)
                self.assertEqual(summary["source_reports"][0]["mode"], "metadata_only")
                self.assertEqual(summary["source_reports"][0]["frontier_imported"], 1)
                self.assertIn("quality_policy", summary)
                self.assertIn("source_priority", summary)
                self.assertIn("CVPR", summary["quality_policy"]["core_venues"])
                self.assertIn("oral", summary["quality_policy"]["accepted_quality_signals"])
                self.assertIn("spotlight", summary["quality_policy"]["excluded_from_gold"])
                self.assertTrue(any("OpenAlex" in item for item in summary["source_priority"]))
                self.assertTrue(any(action["command"].startswith("ra quality-enrich") for action in summary["next_actions"]))
                self.assertIn("metadata-only source; kept", text)
                self.assertIn("quality-enrich", text)
                with connect(cwd / "data" / "research_alpha.db") as conn:
                    row = conn.execute(
                        "select source_kind, award, paper_weight from papers where title=?",
                        ("TPAMI Metadata Outstanding Candidate",),
                    ).fetchone()
                self.assertEqual(row["source_kind"], "frontier_openalex")
                self.assertEqual(row["award"], "")
                self.assertEqual(float(row["paper_weight"]), 0.0)
            finally:
                os.chdir(old_cwd)

    def test_gold_build_uses_per_venue_year_as_quality_cap_for_large_gold_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                payload = [
                    {
                        "title": f"OpenReview Oral Large Set {idx}",
                        "abstract": "An oral paper for expanding a larger excellent-paper set.",
                        "venue": "ICLR 2025 Oral",
                        "year": 2025,
                        "external_ref": f"https://openreview.net/pdf?id=oral-{idx}",
                        "citation_count": 10 + idx,
                        "award": "oral",
                    }
                    for idx in range(8)
                ]
                with patch("research_alpha.app.load_remote_records", return_value=payload):
                    output = io.StringIO()
                    with redirect_stdout(output):
                        exit_code = main(
                            [
                                "gold-build",
                                "--remote",
                                "openreview",
                                "--venues",
                                "ICLR",
                                "--from-year",
                                "2025",
                                "--to-year",
                                "2025",
                                "--per-venue-year",
                                "8",
                            ]
                        )
                self.assertEqual(exit_code, 0)
                text = output.getvalue()
                self.assertIn("Gold-build summary: fetched=8; candidate_imported=8; excellent_imported=8", text)
                self.assertIn("Gold-build summary JSON:", text)
                summary_line = next(line for line in text.splitlines() if line.startswith("Gold-build summary JSON: "))
                summary = json.loads(summary_line.removeprefix("Gold-build summary JSON: "))
                self.assertEqual(summary["remote"], "openreview")
                self.assertEqual(summary["venues"], ["ICLR"])
                self.assertEqual(summary["excellent_imported"], 8)
                self.assertEqual(summary["source_mode"], "strict_gold")
                self.assertEqual(summary["source_reports"][0]["mode"], "core_gold")
                self.assertEqual(summary["source_reports"][0]["excellent_imported"], 8)
                self.assertFalse(summary["metadata_only_remote"])
                self.assertIn("优秀论文库 imported/updated 8 records from openreview", text)
                with connect(cwd / "data" / "research_alpha.db") as conn:
                    count = conn.execute(
                        "select count(*) from papers where source_kind='gold_openreview' and award='oral'"
                    ).fetchone()[0]
                self.assertEqual(count, 8)
            finally:
                os.chdir(old_cwd)

    def test_gold_build_failure_summary_includes_source_failures_and_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                with patch("research_alpha.app.load_remote_records", side_effect=ConnectorError("HTTP 429 Too Many Requests")):
                    output = io.StringIO()
                    with redirect_stdout(output):
                        exit_code = main(
                            [
                                "gold-build",
                                "--remote",
                                "openreview",
                                "--venues",
                                "ICLR",
                                "--from-year",
                                "2025",
                                "--to-year",
                                "2025",
                                "--per-venue-year",
                                "5",
                            ]
                        )
                self.assertEqual(exit_code, 1)
                text = output.getvalue()
                self.assertIn("Gold-build summary JSON:", text)
                summary_line = next(line for line in text.splitlines() if line.startswith("Gold-build summary JSON: "))
                summary = json.loads(summary_line.removeprefix("Gold-build summary JSON: "))
                self.assertEqual(summary["failures"], 1)
                self.assertEqual(summary["source_failures"][0]["venue"], "ICLR")
                self.assertEqual(summary["source_failures"][0]["year"], 2025)
                self.assertIn("HTTP 429", summary["source_failures"][0]["error"])
                self.assertTrue(summary["next_actions"])
                self.assertIn("Remote failures:", text)
            finally:
                os.chdir(old_cwd)

    def test_gold_build_empty_source_results_are_reported_structurally(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                with patch("research_alpha.app.load_remote_records", return_value=[]):
                    output = io.StringIO()
                    with redirect_stdout(output):
                        exit_code = main(
                            [
                                "gold-build",
                                "--remote",
                                "cvf_awards",
                                "--venues",
                                "CVPR",
                                "--from-year",
                                "2026",
                                "--to-year",
                                "2026",
                                "--per-venue-year",
                                "5",
                            ]
                        )
                self.assertEqual(exit_code, 0)
                text = output.getvalue()
                self.assertIn("source returned 0 records from cvf_awards", text)
                self.assertIn("Gold-build summary JSON:", text)
                summary_line = next(line for line in text.splitlines() if line.startswith("Gold-build summary JSON: "))
                summary = json.loads(summary_line.removeprefix("Gold-build summary JSON: "))
                self.assertEqual(summary["fetched"], 0)
                self.assertEqual(summary["excellent_imported"], 0)
                self.assertEqual(summary["candidate_imported"], 0)
                self.assertEqual(summary["source_empty_results"][0]["remote"], "cvf_awards")
                self.assertEqual(summary["source_empty_results"][0]["venue"], "CVPR")
                self.assertEqual(summary["source_empty_results"][0]["reason"], "source_returned_zero_records")
                self.assertTrue(summary["source_reports"][0]["empty_result"])
                self.assertEqual(summary["source_reports"][0]["fetched"], 0)
                self.assertTrue(any(action["label"] == "有空结果来源" for action in summary["next_actions"]))
                with connect(cwd / "data" / "research_alpha.db") as conn:
                    count = conn.execute("select count(*) from papers").fetchone()[0]
                self.assertEqual(count, 0)
            finally:
                os.chdir(old_cwd)

    def test_quality_enriched_frontier_paper_is_promoted_into_excellent_library(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                main(["init"])
                db_path = cwd / "data" / "research_alpha.db"
                paper_id = add_paper(
                    db_path,
                    "Promotable Remote Paper",
                    "ICLR",
                    2026,
                    "A remote candidate with later verified quality metadata.",
                    source_kind="frontier_openalex",
                    citation_count=100,
                )
                from research_alpha.app import gold_source_kind, score_papers_in_db
                from research_alpha.db import update_paper_quality_metadata

                update_paper_quality_metadata(
                    db_path,
                    paper_id,
                    award="oral",
                    source_kind=gold_source_kind("openalex"),
                )
                score_papers_in_db(db_path)
                with connect(db_path) as conn:
                    row = conn.execute(
                        "select source_kind, award, paper_weight from papers where id=?",
                        (paper_id,),
                    ).fetchone()
                self.assertEqual(row["source_kind"], "gold_openalex")
                self.assertEqual(row["award"], "oral")
                self.assertGreater(float(row["paper_weight"]), 0)
            finally:
                os.chdir(old_cwd)

    def test_quality_enriched_tpami_metadata_can_become_core_gold(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                main(["init"])
                db_path = cwd / "data" / "research_alpha.db"
                paper_id = add_paper(
                    db_path,
                    "TPAMI Verified Outstanding Paper",
                    "IEEE Transactions on Pattern Analysis and Machine Intelligence",
                    2026,
                    "A TPAMI paper whose excellent status was verified by metadata audit.",
                    source_kind="frontier_openalex",
                    citation_count=100,
                )
                from research_alpha.app import gold_source_kind, score_papers_in_db
                from research_alpha.db import update_paper_quality_metadata

                update_paper_quality_metadata(
                    db_path,
                    paper_id,
                    award="outstanding_paper",
                    source_kind=gold_source_kind("openalex"),
                )
                score_papers_in_db(db_path)
                with connect(db_path) as conn:
                    row = conn.execute(
                        "select source_kind, award, paper_weight from papers where id=?",
                        (paper_id,),
                    ).fetchone()
                self.assertEqual(row["source_kind"], "gold_openalex")
                self.assertEqual(row["award"], "outstanding_paper")
                self.assertGreater(float(row["paper_weight"]), 0)
            finally:
                os.chdir(old_cwd)

    def test_quality_enriched_high_citation_alone_stays_frontier_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                main(["init"])
                db_path = cwd / "data" / "research_alpha.db"
                paper_id = add_paper(
                    db_path,
                    "LLM High Citation Only Candidate",
                    "ICLR",
                    2026,
                    "A candidate that lacks explicit award or presentation metadata.",
                    source_kind="frontier_openalex",
                    award="high_citation",
                    citation_count=100,
                )
                from research_alpha.app import score_papers_in_db

                score_papers_in_db(db_path)
                with connect(db_path) as conn:
                    row = conn.execute(
                        "select source_kind, award, paper_weight from papers where id=?",
                        (paper_id,),
                    ).fetchone()
                self.assertEqual(row["source_kind"], "frontier_openalex")
                self.assertEqual(row["award"], "high_citation")
                self.assertEqual(float(row["paper_weight"]), 0.0)
            finally:
                os.chdir(old_cwd)

    def test_non_core_venue_award_does_not_become_gold_standard(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                main(["init"])
                db_path = cwd / "data" / "research_alpha.db"
                paper_id = add_paper(
                    db_path,
                    "AAAI Best Paper Should Be Auxiliary",
                    "AAAI",
                    2026,
                    "A non-core award paper should not define the core idea standard.",
                    source_kind="gold_openalex",
                    award="best_paper",
                    citation_count=1000,
                )
                from research_alpha.app import score_papers_in_db

                score_papers_in_db(db_path)
                with connect(db_path) as conn:
                    row = conn.execute(
                        "select source_kind, award, paper_weight, score_notes from papers where id=?",
                        (paper_id,),
                    ).fetchone()
                self.assertEqual(row["source_kind"], "gold_openalex")
                self.assertEqual(row["award"], "best_paper")
                self.assertEqual(float(row["paper_weight"]), 0.0)
                self.assertIn("non_core_venue_auxiliary_not_gold_standard", row["score_notes"])
            finally:
                os.chdir(old_cwd)

    def test_cleanup_downgrades_legacy_non_core_gold_paper_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                db_path = cwd / "data" / "research_alpha.db"
                paper_id = add_paper(
                    db_path,
                    "Legacy ACL Gold Pollution",
                    "ACL",
                    2025,
                    "A legacy record imported before non-core awards were downgraded.",
                    source_kind="gold_acl_awards",
                    award="best_paper",
                    citation_count=500,
                )
                with connect(db_path) as conn:
                    conn.execute("UPDATE papers SET paper_weight=4, score_notes='legacy' WHERE id=?", (paper_id,))

                dry_run = io.StringIO()
                with redirect_stdout(dry_run):
                    dry_exit = main(["cleanup"])
                self.assertEqual(dry_exit, 0)
                self.assertIn("Legacy non-core Gold paper records: 1", dry_run.getvalue())
                self.assertIn("gold_acl_awards -> frontier_acl_awards", dry_run.getvalue())

                apply_out = io.StringIO()
                with redirect_stdout(apply_out):
                    apply_exit = main(["cleanup", "--apply"])
                self.assertEqual(apply_exit, 0)
                self.assertIn("Cleanup applied.", apply_out.getvalue())
                with connect(db_path) as conn:
                    row = conn.execute(
                        "select source_kind, award, paper_weight, score_notes from papers where id=?",
                        (paper_id,),
                    ).fetchone()
                self.assertEqual(row["source_kind"], "frontier_acl_awards")
                self.assertEqual(row["award"], "")
                self.assertEqual(float(row["paper_weight"]), 0.0)
                self.assertIn("legacy_non_core_gold_downgraded_to_frontier", row["score_notes"])
                self.assertTrue(list((cwd / "outputs" / "cleanup").glob("cleanup-*.json")))
            finally:
                os.chdir(old_cwd)

    def test_gui_venue_options_are_structured_and_deduplicated(self) -> None:
        venue_values = [
            venue["value"]
            for group in VENUE_GROUPS
            for venue in group["venues"]
        ]
        self.assertIn("ICLR", venue_values)
        self.assertIn("NeurIPS", venue_values)
        self.assertIn("ICML", venue_values)
        self.assertIn("CVPR", venue_values)
        self.assertIn("ICCV", venue_values)
        self.assertIn("TPAMI", venue_values)
        auxiliary_ranks = {
            venue["value"]: venue["rank"]
            for group in VENUE_GROUPS
            for venue in group["venues"]
            if venue["value"] in {"AAAI", "IJCAI", "KDD", "SIGIR", "WWW", "CHI"}
        }
        self.assertEqual(set(auxiliary_ranks.values()), {"趋势辅助"})
        self.assertTrue(any(all(venue in preset["venues"] for venue in ["ICLR", "NeurIPS", "ICML"]) for preset in VENUE_PRESETS))
        self.assertTrue(any(preset["venues"] == ["ICLR", "NeurIPS", "ICML", "CVPR", "ICCV"] for preset in VENUE_PRESETS))
        self.assertTrue(any(preset["venues"] == ["ICLR", "NeurIPS", "ICML", "CVPR", "ICCV", "TPAMI"] for preset in VENUE_PRESETS))
        self.assertTrue(any(preset["label"] == "核心+跨域趋势" for preset in VENUE_PRESETS))
        self.assertEqual(parse_venue_payload(["ICLR", "NeurIPS", "iclr", ""]), ["ICLR", "NeurIPS"])
        self.assertEqual(parse_venue_payload("ICLR, NeurIPS, ICML"), ["ICLR", "NeurIPS", "ICML"])
        self.assertEqual(parse_venue_payload(["iclr", "nips", "t-pami", "cvpr", "corl"]), ["ICLR", "NeurIPS", "TPAMI", "CVPR", "CoRL"])
        self.assertEqual(parse_venue_payload(["NeurIPS", "nips", "T-PAMI", "tpami", "ICLR"]), ["NeurIPS", "TPAMI", "ICLR"])
        with self.assertRaisesRegex(ValueError, "venues必须是逗号分隔字符串或字符串数组"):
            parse_venue_payload({"venue": "ICLR"})
        with self.assertRaisesRegex(ValueError, "venues数组只能包含字符串"):
            parse_venue_payload(["ICLR", 123])
        with self.assertRaisesRegex(ValueError, f"venues不能超过 {GUI_MAX_VENUES} 个"):
            parse_venue_payload([f"V{idx}" for idx in range(GUI_MAX_VENUES + 1)])
        with self.assertRaisesRegex(ValueError, f"venue名称不能超过 {GUI_MAX_VENUE_CHARS} 个字符"):
            parse_venue_payload(["x" * (GUI_MAX_VENUE_CHARS + 1)])
        self.assertEqual(
            split_gold_venues_by_remote(["ICLR", "ICML", "NeurIPS", "ACL", "EMNLP", "NAACL", "CVPR", "ICCV", "ECCV", "TPAMI"]),
            {"openreview": ["ICLR"], "icml_awards": ["ICML"], "neurips_awards": ["NeurIPS"], "openalex": ["ACL", "EMNLP", "NAACL", "ECCV", "TPAMI"], "cvf_awards": ["CVPR", "ICCV"]},
        )
        self.assertEqual(
            describe_gold_source_plan(split_gold_venues_by_remote(["ICLR", "ICML", "TPAMI", "CVPR"])),
            [
                {"remote": "openreview", "label": "OpenReview Best/Nominee/Oral", "venues": ["ICLR"], "mode": "strict_gold"},
                {"remote": "icml_awards", "label": "ICML 官方奖项页", "venues": ["ICML"], "mode": "strict_gold"},
                {"remote": "openalex", "label": "OpenAlex 元数据趋势", "venues": ["TPAMI"], "mode": "metadata_only"},
                {"remote": "cvf_awards", "label": "CVF 官方奖项页", "venues": ["CVPR"], "mode": "strict_gold"},
            ],
        )
        self.assertEqual(
            metadata_only_gold_venues(split_gold_venues_by_remote(["ICLR", "TPAMI", "CVPR"])),
            ["TPAMI"],
        )
        self.assertEqual(
            core_gold_venues(["ICLR", "ICML", "NeurIPS", "ACL", "EMNLP", "NAACL", "CVPR", "ICCV", "ECCV", "TPAMI"]),
            ["ICLR", "ICML", "NeurIPS", "CVPR", "ICCV", "TPAMI"],
        )
        self.assertEqual(
            core_gold_venues(["nips", "NeurIPS", "tpami", "T-PAMI", "iclr"]),
            ["NeurIPS", "TPAMI", "ICLR"],
        )
        self.assertEqual(
            auxiliary_gold_venues(["ICLR", "KDD", "SIGIR", "WWW", "CHI", "AAAI", "IJCAI"]),
            ["KDD", "SIGIR", "WWW", "CHI", "AAAI", "IJCAI"],
        )
        self.assertEqual(
            split_gold_venues_by_remote(["KDD", "SIGIR", "WWW", "CHI", "CoRL", "RSS", "AAAI", "IJCAI"]),
            {"openalex": ["KDD", "SIGIR", "WWW", "CHI", "CoRL", "RSS", "AAAI", "IJCAI"]},
        )
        self.assertEqual(split_gold_venues_by_remote(core_gold_venues(["KDD", "ICLR", "CHI", "CVPR"])), {"openreview": ["ICLR"], "cvf_awards": ["CVPR"]})
        self.assertEqual(split_gold_venues_by_remote(["NeurIPS", "nips", "ICLR", "iclr"]), {"neurips_awards": ["NeurIPS"], "openreview": ["ICLR"]})

    def test_gui_gold_build_api_returns_source_plan_and_jobs_for_core_venues(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                captured_commands = []

                def fake_streaming(command, root_dir, on_output=None):
                    captured_commands.append(command)
                    if on_output:
                        on_output("stdout", "Gold-build summary JSON: {\"fetched\":0,\"candidate_imported\":0,\"excellent_imported\":0}")
                    return {"exit_code": 0, "stdout": "ok", "stderr": ""}

                with patch("research_alpha.gui.run_app_command_streaming", side_effect=fake_streaming):
                    server = create_gui_http_server(cwd, host="127.0.0.1", port=0)
                    thread = threading.Thread(target=server.serve_forever, daemon=True)
                    thread.start()
                    try:
                        url = f"http://127.0.0.1:{server.server_port}/api/gold-build"
                        payload = json.dumps(
                            {
                                "venues": ["ICLR", "CVPR", "TPAMI", "KDD"],
                                "from_year": 2025,
                                "to_year": 2025,
                                "per_venue_year": 2,
                                "ui_scope": "draft:test",
                            }
                        ).encode("utf-8")
                        request = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
                        with urllib.request.urlopen(request, timeout=5) as response:
                            body = json.loads(response.read().decode("utf-8"))

                        self.assertEqual(response.status, 200)
                        self.assertEqual(body["auxiliary_venues"], ["KDD"])
                        self.assertEqual(body["metadata_only_venues"], ["TPAMI"])
                        self.assertEqual(
                            body["source_plan"],
                            [
                                {"remote": "openreview", "label": "OpenReview Best/Nominee/Oral", "venues": ["ICLR"], "mode": "strict_gold"},
                                {"remote": "cvf_awards", "label": "CVF 官方奖项页", "venues": ["CVPR"], "mode": "strict_gold"},
                                {"remote": "openalex", "label": "OpenAlex 元数据趋势", "venues": ["TPAMI"], "mode": "metadata_only"},
                            ],
                        )
                        self.assertEqual(len(body["jobs"]), 3)
                        self.assertTrue(all(job["ui_scope"] == "draft:test" for job in body["jobs"]))
                        group_ids = {job["group_id"] for job in body["jobs"]}
                        self.assertEqual(len(group_ids), 1)
                        self.assertTrue(next(iter(group_ids)).startswith("gold-build:draft:test:"))

                        for _ in range(50):
                            if len(captured_commands) == 3:
                                break
                            import time

                            time.sleep(0.01)
                        remotes = {command[command.index("--remote") + 1] for command in captured_commands}
                        self.assertEqual(remotes, {"openreview", "cvf_awards", "openalex"})
                    finally:
                        server.shutdown()
                        server.server_close()
                        thread.join(timeout=2)
            finally:
                os.chdir(old_cwd)

    def test_gui_gold_build_defaults_to_recent_completed_year_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                captured_commands = []

                def fake_streaming(command, root_dir, on_output=None):
                    captured_commands.append(command)
                    return {"exit_code": 0, "stdout": "Gold-build summary JSON: {\"fetched\":0,\"candidate_imported\":0,\"excellent_imported\":0}", "stderr": ""}

                with patch("research_alpha.gui.run_app_command_streaming", side_effect=fake_streaming):
                    server = create_gui_http_server(cwd, host="127.0.0.1", port=0)
                    thread = threading.Thread(target=server.serve_forever, daemon=True)
                    thread.start()
                    try:
                        status, body = self.gui_post_json(
                            server,
                            "/api/gold-build",
                            {"venues": ["ICLR"], "per_venue_year": 2, "ui_scope": "draft:default-years"},
                        )
                        self.assertEqual(status, 200)

                        for _ in range(50):
                            if captured_commands:
                                break
                            import time

                            time.sleep(0.01)
                        self.assertEqual(len(captured_commands), 1)
                        command = captured_commands[0]
                        self.assertEqual(command[command.index("--from-year") + 1], str(default_gold_from_year()))
                        self.assertEqual(command[command.index("--to-year") + 1], str(default_gold_to_year()))
                        self.assertNotIn("--no-fulltext", command)
                        self.assertLessEqual(default_gold_to_year(), current_gui_year() - 1)
                        self.assertEqual(body["jobs"][0]["kind"], "gold-build")
                    finally:
                        server.shutdown()
                        server.server_close()
                        thread.join(timeout=2)
            finally:
                os.chdir(old_cwd)

    def test_gui_post_api_rejects_invalid_payloads_with_clear_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                server = create_gui_http_server(cwd, host="127.0.0.1", port=0)
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                try:
                    status, body = self.gui_post_json(server, "/api/ideate", {"direction": ""})
                    self.assertEqual(status, 400)
                    self.assertEqual(body["error"], "direction is required")
                    self.assertFalse(body["ok"])
                    self.assertEqual(body["status"], 400)
                    self.assertEqual(body["code"], "missing_direction")
                    self.assertFalse(body["retryable"])
                    self.assertIn("研究方向", body["next_action"])

                    status, body = self.gui_post_json(server, "/api/ideate", {"direction": "agent eval", "remote": "icml_awards"})
                    self.assertEqual(status, 400)
                    self.assertIn("近期趋势源必须是以下之一", str(body["error"]))
                    self.assertEqual(body["code"], "validation_error")

                    status, body = self.gui_post_json(server, "/api/ideate", {"direction": "agent eval", "provider": "claude"})
                    self.assertEqual(status, 400)
                    self.assertIn("模型必须是以下之一", str(body["error"]))
                    self.assertEqual(body["status"], 400)

                    status, body = self.gui_post_json(server, "/api/ideate", {"direction": "agent eval", "lang": "fr"})
                    self.assertEqual(status, 400)
                    self.assertIn("输出语言必须是以下之一", str(body["error"]))
                    self.assertFalse(body["ok"])

                    status, body = self.gui_post_json(server, "/api/ideate", "{")
                    self.assertEqual(status, 400)
                    self.assertIn("请求 JSON 解析失败", str(body["error"]))
                    self.assertEqual(body["code"], "validation_error")

                    status, body = self.gui_post_json(server, "/api/ideate", {"direction": "x" * (GUI_MAX_DIRECTION_CHARS + 1)})
                    self.assertEqual(status, 400)
                    self.assertIn(f"direction不能超过 {GUI_MAX_DIRECTION_CHARS} 个字符", str(body["error"]))
                    self.assertEqual(body["code"], "validation_error")

                    status, body = self.gui_post_json(server, "/api/gold-build", {"venues": ["ICLR"], "query": "x" * (GUI_MAX_GOLD_QUERY_CHARS + 1)})
                    self.assertEqual(status, 400)
                    self.assertIn(f"Gold query不能超过 {GUI_MAX_GOLD_QUERY_CHARS} 个字符", str(body["error"]))
                    self.assertEqual(body["code"], "validation_error")

                    status, body = self.gui_post_json(server, "/api/gold-build", {"venues": {"venue": "ICLR"}})
                    self.assertEqual(status, 400)
                    self.assertEqual(body["error"], "venues必须是逗号分隔字符串或字符串数组")
                    self.assertEqual(body["code"], "validation_error")

                    status, body = self.gui_post_json(server, "/api/gold-build", {"venues": ["ICLR", 123]})
                    self.assertEqual(status, 400)
                    self.assertEqual(body["error"], "venues数组只能包含字符串")
                    self.assertEqual(body["code"], "validation_error")

                    status, body = self.gui_post_json(server, "/api/cleanup", {"ui_scope": "x" * (GUI_MAX_SCOPE_CHARS + 1)})
                    self.assertEqual(status, 400)
                    self.assertIn(f"ui_scope不能超过 {GUI_MAX_SCOPE_CHARS} 个字符", str(body["error"]))

                    unsafe_scope = "/Users/me/project/" + ("x" * GUI_MAX_SCOPE_CHARS) + " DEEPSEEK_API_KEY=sk-error-secret-123456"
                    status, body = self.gui_post_json(server, "/api/cleanup", {"ui_scope": unsafe_scope})
                    self.assertEqual(status, 400)
                    serialized_error = json.dumps(body, ensure_ascii=False)
                    self.assertNotIn("/Users/me", serialized_error)
                    self.assertNotIn("sk-error-secret", serialized_error)

                    status, body = self.gui_post_json(server, "/api/ideate", {"direction": "x" * GUI_MAX_JSON_BODY_BYTES})
                    self.assertEqual(status, 413)
                    self.assertEqual(body["code"], "request_too_large")
                    self.assertEqual(body["status"], 413)
                    self.assertFalse(body["retryable"])
                    self.assertIn("缩短输入", body["next_action"])

                    status, body = self.gui_post_json(server, "/api/ideate", [])
                    self.assertEqual(status, 400)
                    self.assertEqual(body["error"], "请求体必须是 JSON object")
                    self.assertEqual(body["status"], 400)

                    status, body = self.gui_post_json(server, "/api/gold-build", {"venues": ["KDD"], "ui_scope": "draft:test"})
                    self.assertEqual(status, 400)
                    self.assertEqual(body["code"], "no_core_gold_venues")
                    self.assertFalse(body["retryable"])
                    self.assertIn("选择 ICLR", body["next_action"])
                    self.assertIn("核心优秀论文库只接受", str(body["error"]))
                    self.assertIn("KDD", str(body["error"]))
                    self.assertEqual(body["code"], "no_core_gold_venues")

                    status, body = self.gui_post_json(server, "/api/gold-build", {"venues": ["ICLR"], "from_year": 2026, "to_year": 2025})
                    self.assertEqual(status, 400)
                    self.assertEqual(body["error"], "结束年份不能早于起始年份")
                    self.assertEqual(body["code"], "invalid_year_range")
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=2)
            finally:
                os.chdir(old_cwd)

    def test_gui_ideate_api_dedupes_running_job_for_same_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                self.write_gui_test_llm_env(cwd)
                release = threading.Event()
                created_sessions = []

                def fake_stable_ideate(root_dir, direction, brief_direction, provider, lang, review_rounds=4, on_output=None):
                    created_sessions.append((direction, brief_direction, provider, lang, review_rounds))
                    if on_output:
                        on_output("stdout", "Auto session: #1 research agents")
                    release.wait(2)
                    return {"exit_code": 0, "stdout": "Auto session: #1 research agents", "stderr": ""}

                with patch("research_alpha.gui.run_gui_stable_ideate_session", side_effect=fake_stable_ideate):
                    server = create_gui_http_server(cwd, host="127.0.0.1", port=0)
                    thread = threading.Thread(target=server.serve_forever, daemon=True)
                    thread.start()
                    try:
                        payload = {
                            "direction": "research agents",
                            "session_id": 0,
                            "ui_scope": "draft:dedupe",
                            "remote": "openalex",
                            "ideas": 1,
                            "limit": 4,
                            "provider": "deepseek",
                            "lang": "zh",
                        }
                        status, first = self.gui_post_json(server, "/api/ideate", payload)
                        self.assertEqual(status, 200)
                        self.assertIn("job", first)
                        self.assertNotIn("deduped", first)

                        status, second = self.gui_post_json(server, "/api/ideate", payload)
                        self.assertEqual(status, 200)
                        self.assertTrue(second["deduped"])
                        self.assertEqual(second["job"]["id"], first["job"]["id"])
                        self.assertEqual(second["job"]["status"], "running")

                        for _ in range(50):
                            if created_sessions:
                                break
                            import time

                            time.sleep(0.01)
                        self.assertEqual(len(created_sessions), 1)
                        self.assertEqual(created_sessions[0][0], "research agents")
                        self.assertEqual(created_sessions[0][2], "deepseek")
                    finally:
                        release.set()
                        server.shutdown()
                        server.server_close()
                        thread.join(timeout=2)
            finally:
                os.chdir(old_cwd)

    def test_gui_llm_jobs_fail_fast_when_provider_key_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                captured_commands = []

                def fake_streaming(command, root_dir, on_output=None):
                    captured_commands.append(command)
                    return {"exit_code": 0, "stdout": "should not run", "stderr": ""}

                with patch.dict(os.environ, {"RA_LLM_API_KEY": "", "DEEPSEEK_API_KEY": "", "OPENAI_API_KEY": ""}, clear=False):
                    with patch("research_alpha.gui.run_app_command_streaming", side_effect=fake_streaming):
                        server = create_gui_http_server(cwd, host="127.0.0.1", port=0)
                        thread = threading.Thread(target=server.serve_forever, daemon=True)
                        thread.start()
                        try:
                            status, body = self.gui_post_json(
                                server,
                                "/api/ideate",
                                {
                                    "direction": "research agents",
                                    "ui_scope": "draft:no-key",
                                    "remote": "openalex",
                                    "ideas": 1,
                                    "limit": 4,
                                    "provider": "deepseek",
                                    "lang": "zh",
                                },
                            )
                            self.assertEqual(status, 400)
                            self.assertFalse(body["ok"])
                            self.assertEqual(body["code"], "missing_provider_key")
                            self.assertFalse(body["retryable"])
                            self.assertIn("API key", body["error"])
                            self.assertIn("配置当前模型", body["next_action"])
                            self.assertEqual(captured_commands, [])
                        finally:
                            server.shutdown()
                            server.server_close()
                            thread.join(timeout=2)
            finally:
                os.chdir(old_cwd)

    def test_gui_state_redacts_live_job_paths_from_http_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                self.write_gui_test_llm_env(cwd)
                release = threading.Event()

                def fake_streaming(command, root_dir, on_output=None):
                    if on_output:
                        on_output("stdout", "Manifest: /Users/example/project/outputs/run/manifest.json")
                        on_output("stderr", "Dossier: /private/var/tmp/session.json")
                    release.wait(2)
                    return {"exit_code": 0, "stdout": "Created idea session #1", "stderr": ""}

                with patch("research_alpha.gui.run_app_command_streaming", side_effect=fake_streaming):
                    server = create_gui_http_server(cwd, host="127.0.0.1", port=0)
                    thread = threading.Thread(target=server.serve_forever, daemon=True)
                    thread.start()
                    try:
                        payload = {
                            "direction": "research agents",
                            "session_id": 0,
                            "ui_scope": "draft:redact",
                            "remote": "openalex",
                            "ideas": 1,
                            "limit": 4,
                            "provider": "deepseek",
                            "lang": "zh",
                        }
                        status, body = self.gui_post_json(server, "/api/ideate", payload)
                        self.assertEqual(status, 200)
                        self.assertEqual(body["job"]["status"], "running")

                        for _ in range(50):
                            with urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/api/state?draft=1", timeout=5) as response:
                                snapshot = json.loads(response.read().decode("utf-8"))
                            serialized = json.dumps(snapshot["jobs"], ensure_ascii=False)
                            if snapshot["jobs"]:
                                break
                            import time

                            time.sleep(0.01)
                        self.assertNotIn("/Users/", serialized)
                        self.assertNotIn("/private/", serialized)
                        self.assertNotIn("manifest.json", serialized)
                        self.assertNotIn("[local path]", serialized)
                        self.assertEqual(snapshot["jobs"][0]["live_stdout"], "")
                        self.assertEqual(snapshot["jobs"][0]["live_stderr"], "")
                        self.assertEqual(snapshot["jobs"][0]["last_line"], "")
                    finally:
                        release.set()
                        server.shutdown()
                        server.server_close()
                        thread.join(timeout=2)
            finally:
                os.chdir(old_cwd)

    def test_gui_ideate_api_dedupes_concurrent_requests_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                self.write_gui_test_llm_env(cwd)
                release = threading.Event()
                created_sessions = []
                request_count = 8

                def fake_stable_ideate(root_dir, direction, brief_direction, provider, lang, review_rounds=4, on_output=None):
                    created_sessions.append((direction, provider))
                    if on_output:
                        on_output("stdout", "ideate is running")
                    release.wait(2)
                    return {"exit_code": 0, "stdout": "Auto session: #1 research agents", "stderr": ""}

                with patch("research_alpha.gui.run_gui_stable_ideate_session", side_effect=fake_stable_ideate):
                    server = create_gui_http_server(cwd, host="127.0.0.1", port=0)
                    thread = threading.Thread(target=server.serve_forever, daemon=True)
                    thread.start()
                    try:
                        payload = {
                            "direction": "research agents",
                            "session_id": 0,
                            "ui_scope": "draft:concurrent",
                            "remote": "openalex",
                            "ideas": 1,
                            "limit": 4,
                            "provider": "deepseek",
                            "lang": "zh",
                        }
                        barrier = threading.Barrier(request_count)
                        responses = []
                        response_lock = threading.Lock()

                        def post_once():
                            barrier.wait(2)
                            item = self.gui_post_json(server, "/api/ideate", payload)
                            with response_lock:
                                responses.append(item)

                        workers = [threading.Thread(target=post_once) for _ in range(request_count)]
                        for worker in workers:
                            worker.start()
                        for worker in workers:
                            worker.join(timeout=5)

                        self.assertEqual(len(responses), request_count)
                        self.assertTrue(all(status == 200 for status, _ in responses))
                        job_ids = {body["job"]["id"] for _, body in responses}
                        self.assertEqual(len(job_ids), 1)
                        deduped_count = sum(1 for _, body in responses if body.get("deduped"))
                        self.assertEqual(deduped_count, request_count - 1)

                        for _ in range(50):
                            if created_sessions:
                                break
                            import time

                            time.sleep(0.01)
                        self.assertEqual(created_sessions, [("research agents", "deepseek")])
                    finally:
                        release.set()
                        server.shutdown()
                        server.server_close()
                        thread.join(timeout=2)
            finally:
                os.chdir(old_cwd)

    def test_gui_gold_build_api_dedupes_concurrent_group_requests_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                release = threading.Event()
                captured_commands = []
                command_lock = threading.Lock()
                request_count = 8

                def fake_streaming(command, root_dir, on_output=None):
                    with command_lock:
                        captured_commands.append(command)
                    if on_output:
                        on_output("stdout", "gold build is running")
                    release.wait(2)
                    return {"exit_code": 0, "stdout": "Gold-build summary: fetched=1; candidate_imported=0; excellent_imported=1; poster_skipped=0; openreview_non_excellent_skipped=0; unqualified_skipped=0; failures=0", "stderr": ""}

                with patch("research_alpha.gui.run_app_command_streaming", side_effect=fake_streaming):
                    server = create_gui_http_server(cwd, host="127.0.0.1", port=0)
                    thread = threading.Thread(target=server.serve_forever, daemon=True)
                    thread.start()
                    try:
                        payload = {
                            "venues": ["ICLR", "CVPR", "TPAMI"],
                            "from_year": 2025,
                            "to_year": 2025,
                            "per_venue_year": 2,
                            "ui_scope": "draft:gold-concurrent",
                        }
                        barrier = threading.Barrier(request_count)
                        responses = []
                        response_lock = threading.Lock()

                        def post_once():
                            barrier.wait(2)
                            item = self.gui_post_json(server, "/api/gold-build", payload)
                            with response_lock:
                                responses.append(item)

                        workers = [threading.Thread(target=post_once) for _ in range(request_count)]
                        for worker in workers:
                            worker.start()
                        for worker in workers:
                            worker.join(timeout=5)

                        self.assertEqual(len(responses), request_count)
                        self.assertTrue(all(status == 200 for status, _ in responses))
                        created_bodies = [body for _, body in responses if not body.get("deduped")]
                        deduped_bodies = [body for _, body in responses if body.get("deduped")]
                        self.assertEqual(len(created_bodies), 1)
                        self.assertEqual(len(deduped_bodies), request_count - 1)

                        created_jobs = created_bodies[0]["jobs"]
                        self.assertEqual(len(created_jobs), 3)
                        created_job_ids = {job["id"] for job in created_jobs}
                        self.assertEqual(len(created_job_ids), 3)
                        group_ids = {job["group_id"] for job in created_jobs}
                        self.assertEqual(len(group_ids), 1)
                        self.assertTrue(next(iter(group_ids)).startswith("gold-build:draft:gold-concurrent:"))
                        self.assertTrue(all(body["job"]["id"] in created_job_ids for body in deduped_bodies))
                        self.assertTrue(all(len(body["jobs"]) == 1 for body in deduped_bodies))

                        for _ in range(50):
                            with command_lock:
                                command_count = len(captured_commands)
                            if command_count == 3:
                                break
                            import time

                            time.sleep(0.01)
                        with command_lock:
                            commands = list(captured_commands)
                        self.assertEqual(len(commands), 3)
                        remotes = {command[command.index("--remote") + 1] for command in commands}
                        self.assertEqual(remotes, {"openreview", "cvf_awards", "openalex"})
                    finally:
                        release.set()
                        server.shutdown()
                        server.server_close()
                        thread.join(timeout=2)
            finally:
                os.chdir(old_cwd)

    def test_gui_post_api_returns_compact_public_job_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                huge_stdout = "x" * (GUI_JOB_RESULT_TAIL_CHARS + 200) + "response-tail"
                state = GuiState()
                state.jobs = [
                    {
                        "id": 1,
                        "name": "cleanup evidence dry-run",
                        "kind": "cleanup",
                        "session_id": None,
                        "ui_scope": "draft:compact",
                        "group_id": "",
                        "created_session_id": None,
                        "status": "running",
                        "result": {
                            "command": "ra cleanup --internal",
                            "exit_code": 0,
                            "stdout": huge_stdout,
                            "stderr": "",
                            "debug_path": str(cwd / "outputs" / "debug.json"),
                        },
                        "error": "",
                        "live_stdout": "l" * (GUI_JOB_RESULT_TAIL_CHARS + 300) + "live-tail",
                        "live_stderr": "",
                        "last_line": "live-tail",
                        "created_at": 1.0,
                        "updated_at": 1.0,
                        "last_output_at": 1.0,
                        "completed_at": None,
                    }
                ]
                server = create_gui_http_server(cwd, host="127.0.0.1", port=0, state=state)
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                try:
                    status, body = self.gui_post_json(server, "/api/cleanup", {"ui_scope": "draft:compact"})
                    self.assertEqual(status, 200)
                    self.assertTrue(body["deduped"])
                    self.assertEqual(body["job"]["status"], "running")
                    self.assertEqual(sorted(body["job"]["result"].keys()), ["exit_code", "stderr", "stdout"])
                    self.assertLessEqual(len(body["job"]["result"]["stdout"]), GUI_JOB_RESULT_TAIL_CHARS)
                    self.assertTrue(body["job"]["result"]["stdout"].endswith("response-tail"))
                    self.assertNotIn("command", body["job"]["result"])
                    self.assertNotIn("debug_path", body["job"]["result"])
                    self.assertLessEqual(len(body["job"]["live_stdout"]), GUI_JOB_LIVE_TAIL_CHARS)
                    self.assertTrue(body["job"]["live_stdout"].endswith("live-tail"))
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=2)
            finally:
                os.chdir(old_cwd)

    def test_public_conversation_job_snapshot_hides_intermediate_output(self) -> None:
        job = {
            "id": 1,
            "name": "ideate research agents",
            "kind": "ideate",
            "session_id": None,
            "ui_scope": "draft:conversation",
            "group_id": "",
            "created_session_id": 7,
            "status": "completed",
            "result": {
                "exit_code": 0,
                "stdout": (
                    "进度：中间推理输出\n"
                    'Idea-session summary JSON: {"title":"Agent Memory Idea","current_idea":"Use reviewer memory safely.","evidence_score":8,"storyline_steps":[{"step":"reframing","label":"问题重构","transfer":"Turn memory bugs into evidence-grounded review pressure."}]}\n'
                ),
                "stderr": "debug trace",
            },
            "error": "",
            "live_stdout": "进度：实时输出不应出现在 API\n",
            "live_stderr": "stderr live trace\n",
            "last_line": "stderr live trace",
            "created_at": 1.0,
            "updated_at": 2.0,
            "last_output_at": 1.5,
            "completed_at": 2.0,
            "command": "python -m research_alpha.app --api-key sk-secret",
            "debug_path": "/Users/xiqin/private/debug.log",
            "raw_result": {"stdout": "raw hidden output"},
        }
        public = public_job_snapshot(job)
        self.assertEqual(public["kind"], "ideate")
        self.assertEqual(public["result"], {"exit_code": 0})
        self.assertEqual(public["live_stdout"], "")
        self.assertEqual(public["live_stderr"], "")
        self.assertEqual(public["last_line"], "")
        self.assertEqual(public["result_summary"]["type"], "idea_session_created")
        self.assertEqual(public["result_summary"]["idea_session"]["title"], "Agent Memory Idea")
        self.assertEqual(public["result_summary"]["idea_session"]["current_idea"], "Use reviewer memory safely.")
        self.assertEqual(public["result_summary"]["idea_session"]["storyline_steps"][0]["label"], "问题重构")
        serialized = json.dumps(public, ensure_ascii=False)
        self.assertNotIn("中间推理输出", serialized)
        self.assertNotIn("实时输出", serialized)
        self.assertNotIn("debug trace", serialized)
        self.assertNotIn("command", public)
        self.assertNotIn("debug_path", public)
        self.assertNotIn("raw_result", public)
        self.assertNotIn("sk-secret", serialized)
        self.assertNotIn("/Users/xiqin", serialized)

    def test_public_ideate_summary_exposes_review_loop_readiness_state(self) -> None:
        ready_job = {
            "id": 1,
            "name": "ideate research agents",
            "kind": "ideate",
            "session_id": None,
            "ui_scope": "draft:review-ready",
            "group_id": "",
            "created_session_id": 7,
            "status": "completed",
            "result": {
                "exit_code": 0,
                "stdout": (
                    'Idea-session summary JSON: {"session_id":7,"title":"Agent Idea","current_idea":"Final reviewed idea.",'
                    '"auto_review_loop":{"status":"completed","rounds_completed":3,"requested_rounds":4}}\n'
                ),
                "stderr": "",
            },
            "error": "",
            "live_stdout": "hidden",
            "live_stderr": "",
            "last_line": "hidden",
            "created_at": 1.0,
            "updated_at": 2.0,
            "last_output_at": 1.5,
            "completed_at": 2.0,
        }
        ready = public_job_snapshot(ready_job)
        self.assertEqual(ready["result_summary"]["type"], "idea_session_review_ready")
        self.assertEqual(ready["result_summary"]["idea_session"]["auto_review_loop"]["status"], "completed")

        incomplete_job = dict(ready_job)
        incomplete_job["id"] = 2
        incomplete_job["result"] = {
            "exit_code": 0,
            "stdout": (
                'Idea-session summary JSON: {"session_id":8,"title":"Agent Idea","current_idea":"Unreviewed idea.",'
                '"auto_review_loop":{"status":"failed","reason":"review_loop_failed","rounds_completed":0,"requested_rounds":4}}\n'
            ),
            "stderr": "",
        }
        incomplete = public_job_snapshot(incomplete_job)
        self.assertEqual(incomplete["result_summary"]["type"], "idea_session_review_incomplete")
        self.assertEqual(incomplete["result_summary"]["idea_session"]["auto_review_loop"]["reason"], "review_loop_failed")

    def test_public_review_loop_snapshot_uses_structured_final_summary(self) -> None:
        job = {
            "id": 1,
            "name": "review loop 4 rounds",
            "kind": "review-loop",
            "session_id": 7,
            "ui_scope": "session:7",
            "group_id": "",
            "created_session_id": None,
            "status": "completed",
            "result": {
                "exit_code": 0,
                "stdout": (
                    "审稿循环 1/4：中间审稿意见\n"
                    'Review-loop summary JSON: {"session_id":7,"rounds_completed":3,"requested_rounds":4,"review_count":3,"final_idea":"Final reviewed idea."}\n'
                ),
                "stderr": "debug trace",
            },
            "error": "",
            "live_stdout": "实时输出不应出现",
            "live_stderr": "live debug",
            "last_line": "live debug",
            "created_at": 1.0,
            "updated_at": 2.0,
            "last_output_at": 1.5,
            "completed_at": 2.0,
        }
        public = public_job_snapshot(job)
        self.assertEqual(public["result"], {"exit_code": 0})
        self.assertEqual(public["live_stdout"], "")
        self.assertEqual(public["live_stderr"], "")
        self.assertEqual(public["last_line"], "")
        self.assertEqual(public["result_summary"]["type"], "review_loop_completed")
        self.assertEqual(public["result_summary"]["review_loop"]["rounds_completed"], 3)
        self.assertEqual(public["result_summary"]["review_loop"]["requested_rounds"], 4)
        self.assertEqual(public["result_summary"]["review_loop"]["review_count"], 3)
        self.assertEqual(public["result_summary"]["review_loop"]["final_idea"], "Final reviewed idea.")
        serialized = json.dumps(public, ensure_ascii=False)
        self.assertNotIn("Review-loop summary JSON", serialized)
        self.assertNotIn("中间审稿意见", serialized)
        self.assertNotIn("实时输出", serialized)
        self.assertNotIn("debug trace", serialized)

    def test_public_review_loop_failure_snapshot_hides_streams(self) -> None:
        job = {
            "id": 2,
            "name": "review loop 4 rounds",
            "kind": "review-loop",
            "session_id": 7,
            "ui_scope": "session:7",
            "group_id": "",
            "created_session_id": None,
            "status": "failed",
            "result": {
                "exit_code": 1,
                "stdout": "Review loop failed before refinement.\n",
                "stderr": "Session-step failure JSON: hidden raw failure",
            },
            "error": "hidden raw error",
            "live_stdout": "live hidden",
            "live_stderr": "live hidden",
            "last_line": "live hidden",
            "created_at": 1.0,
            "updated_at": 2.0,
            "last_output_at": 1.5,
            "completed_at": 2.0,
        }
        public = public_job_snapshot(job)
        self.assertEqual(public["result"], {"exit_code": 1})
        self.assertEqual(public["error"], "")
        self.assertEqual(public["result_summary"]["type"], "review_loop_failed")
        self.assertEqual(public["result_summary"]["failure"]["reason_code"], "review_loop_failed")
        serialized = json.dumps(public, ensure_ascii=False)
        self.assertNotIn("Review loop failed before refinement", serialized)
        self.assertNotIn("Session-step failure JSON", serialized)
        self.assertNotIn("hidden raw error", serialized)
        self.assertNotIn("live hidden", serialized)

    def test_public_review_loop_failure_snapshot_uses_structured_failure_summary(self) -> None:
        job = {
            "id": 3,
            "name": "review loop 4 rounds",
            "kind": "review-loop",
            "session_id": 7,
            "ui_scope": "session:7",
            "group_id": "",
            "created_session_id": None,
            "status": "failed",
            "result": {
                "exit_code": 1,
                "stdout": (
                    "审稿循环 1/4：中间审稿意见不应出现\n"
                    'Review-loop failure JSON: {"status":"failed","reason":"refinement_failed","exit_code":1,"session_id":7,"rounds_completed":2,"requested_rounds":4,"review_count":3,"final_idea":"Partially refined idea.","state_policy":"partial_refinements_kept_current_idea_safe_no_raw_review_output","recovery_strategy":"resume_review_loop_from_current_partial_idea","safe_resume_command":"ra rl --session-id 7","next_action":"重试循环审稿。"}\n'
                ),
                "stderr": "debug trace",
            },
            "error": "",
            "live_stdout": "live hidden",
            "live_stderr": "live hidden",
            "last_line": "live hidden",
            "created_at": 1.0,
            "updated_at": 2.0,
            "last_output_at": 1.5,
            "completed_at": 2.0,
        }
        public = public_job_snapshot(job)
        self.assertEqual(public["result"], {"exit_code": 1})
        self.assertEqual(public["result_summary"]["type"], "review_loop_failed")
        self.assertEqual(public["result_summary"]["failure"]["reason"], "refinement_failed")
        self.assertEqual(public["result_summary"]["review_loop"]["rounds_completed"], 2)
        self.assertEqual(public["result_summary"]["review_loop"]["requested_rounds"], 4)
        self.assertEqual(public["result_summary"]["review_loop"]["review_count"], 3)
        self.assertEqual(public["result_summary"]["review_loop"]["final_idea"], "Partially refined idea.")
        self.assertEqual(public["result_summary"]["review_loop"]["state_policy"], "partial_refinements_kept_current_idea_safe_no_raw_review_output")
        self.assertEqual(public["result_summary"]["review_loop"]["recovery_strategy"], "resume_review_loop_from_current_partial_idea")
        self.assertEqual(public["result_summary"]["review_loop"]["safe_resume_command"], "ra rl --session-id 7")
        self.assertEqual(public["result_summary"]["review_loop"]["next_action"], "重试循环审稿。")
        serialized = json.dumps(public, ensure_ascii=False)
        self.assertNotIn("Review-loop failure JSON", serialized)
        self.assertNotIn("中间审稿意见", serialized)
        self.assertNotIn("debug trace", serialized)
        self.assertNotIn("live hidden", serialized)

    def test_public_review_loop_failure_snapshot_redacts_recovery_fields(self) -> None:
        job = {
            "id": 4,
            "name": "review loop 4 rounds",
            "kind": "review-loop",
            "session_id": 8,
            "ui_scope": "session:8",
            "group_id": "",
            "created_session_id": None,
            "status": "failed",
            "result": {
                "exit_code": 1,
                "stdout": (
                    'Review-loop failure JSON: {"status":"failed","reason":"refinement_failed","exit_code":1,'
                    '"session_id":8,"rounds_completed":1,"requested_rounds":4,"review_count":2,'
                    '"final_idea":"Safe idea. Report: /Users/me/project/outputs/reviews/review.json sk-live-secret-123456",'
                    '"state_policy":"partial_refinements_kept_current_idea_safe_no_raw_review_output",'
                    '"recovery_strategy":"resume_review_loop_from_current_partial_idea",'
                    '"safe_resume_command":"ra rl --session-id 8 --notes /Users/me/project/notes.md DEEPSEEK_API_KEY=sk-live-secret-123456",'
                    '"next_action":"Retry after reading /private/var/tmp/debug.txt with Bearer test-deepseek-key-123456."}\n'
                ),
                "stderr": "raw trace /Users/me/project/debug.log sk-raw-secret-123456",
            },
            "error": "",
            "live_stdout": "Report: /Users/me/project/outputs/live.md sk-live-secret-123456",
            "live_stderr": "Bearer test-deepseek-key-123456",
            "last_line": "Database: /Users/me/project/data/research_alpha.db",
            "created_at": 1.0,
            "updated_at": 2.0,
            "last_output_at": 1.5,
            "completed_at": 2.0,
        }
        public = public_job_snapshot(job)
        review_loop = public["result_summary"]["review_loop"]
        self.assertIn("[local path]", review_loop["final_idea"])
        self.assertIn("sk-[redacted]", review_loop["final_idea"])
        self.assertIn("[local path]", review_loop["safe_resume_command"])
        self.assertIn("DEEPSEEK_API_KEY=[redacted]", review_loop["safe_resume_command"])
        self.assertIn("[local path]", review_loop["next_action"])
        self.assertIn("Bearer [redacted]", review_loop["next_action"])
        serialized = json.dumps(public, ensure_ascii=False)
        self.assertNotIn("/Users/me", serialized)
        self.assertNotIn("/private/var/tmp", serialized)
        self.assertNotIn("sk-live-secret", serialized)
        self.assertNotIn("test-deepseek-key", serialized)
        self.assertNotIn("raw trace", serialized)

    def test_run_report_surfaces_auto_review_loop_details(self) -> None:
        report = render_run_report(
            {
                "run_id": "20260608T010101Z",
                "query": "research agents",
                "status": "session_review_ready",
                "requested_ideas": 1,
                "llm_provider": "ds",
                "llm_model": "deepseek-chat",
                "stages": [
                    {
                        "stage": "review_loop",
                        "status": "completed",
                        "session_id": 7,
                        "requested_rounds": 4,
                        "rounds_completed": 3,
                        "final_idea": "Final reviewed idea.",
                    }
                ],
            }
        )
        self.assertIn("## Auto Review Loop", report)
        self.assertIn("- Status: completed", report)
        self.assertIn("- Rounds: 3/4", report)
        self.assertIn("- Session ID: 7", report)
        self.assertIn("- Final idea: Final reviewed idea.", report)

        failed_report = render_run_report(
            {
                "run_id": "20260608T010102Z",
                "query": "research agents",
                "status": "session_review_incomplete",
                "requested_ideas": 1,
                "llm_provider": "ds",
                "llm_model": "deepseek-chat",
                "stages": [
                    {
                        "stage": "review_loop",
                        "status": "failed",
                        "session_id": 8,
                        "requested_rounds": 4,
                        "rounds_completed": 0,
                        "reason": "review_loop_failed",
                    }
                ],
            }
        )
        self.assertIn("- Status: failed", failed_report)
        self.assertIn("- Reason: review_loop_failed", failed_report)

    def test_public_ideate_job_without_saved_session_is_not_false_completed(self) -> None:
        job = {
            "id": 1,
            "name": "ideate research agents",
            "kind": "ideate",
            "session_id": None,
            "ui_scope": "draft:conversation",
            "group_id": "",
            "created_session_id": None,
            "status": "completed",
            "result": {
                "exit_code": 0,
                "stdout": "Strict evidence is not ready yet: missing strict Pattern cards.\n进度：中间输出不应作为回答\n",
                "stderr": "",
            },
            "error": "",
            "live_stdout": "live trace",
            "live_stderr": "",
            "last_line": "live trace",
            "created_at": 1.0,
            "updated_at": 2.0,
            "last_output_at": 1.5,
            "completed_at": 2.0,
        }
        public = public_job_snapshot(job)
        self.assertEqual(public["result"], {"exit_code": 0})
        self.assertEqual(public["result_summary"]["type"], "idea_generation_not_ready")
        self.assertEqual(public["result_summary"]["failure"]["primary_reason"], "gold_evidence_not_ready")
        self.assertEqual(public["result_summary"]["failure"]["accepted"], 0)
        serialized = json.dumps(public, ensure_ascii=False)
        self.assertNotIn("ideate_finished", serialized)
        self.assertNotIn("中间输出不应作为回答", serialized)
        self.assertNotIn("live trace", serialized)

    def test_public_review_job_snapshot_exposes_safe_rethink_summary(self) -> None:
        job = {
            "id": 1,
            "name": "review session",
            "kind": "review",
            "session_id": 3,
            "ui_scope": "session:3",
            "group_id": "",
            "created_session_id": None,
            "status": "completed",
            "result": {
                "exit_code": 2,
                "stdout": "Reviewer decision: rethink_required\nSummary: Evidence design is too weak.\nJSON: /Users/me/project/outputs/reviews/review.json\n",
                "stderr": "raw reviewer trace",
            },
            "error": "",
            "live_stdout": "live trace",
            "live_stderr": "stderr trace",
            "last_line": "stderr trace",
            "created_at": 1.0,
            "updated_at": 2.0,
            "last_output_at": 1.5,
            "completed_at": 2.0,
        }
        public = public_job_snapshot(job)
        self.assertEqual(public["result"], {"exit_code": 2})
        self.assertEqual(public["result_summary"]["type"], "review_completed")
        self.assertEqual(public["result_summary"]["decision"], "rethink_required")
        self.assertEqual(public["result_summary"]["summary"], "Evidence design is too weak.")
        self.assertEqual(public["result_summary"]["review_path"], "[local path]")
        serialized = json.dumps(public, ensure_ascii=False)
        self.assertNotIn("raw reviewer trace", serialized)
        self.assertNotIn("live trace", serialized)
        self.assertNotIn("/Users/me", serialized)

    def test_public_step_failure_snapshot_uses_structured_failure_not_raw_error(self) -> None:
        failure = {
            "status": "session_step_failed",
            "reason_code": "session_evidence_guard_failed",
            "reason": "模型越界",
            "next_action": "重新约束在优秀论文逻辑空间。",
        }
        job = {
            "id": 1,
            "name": "session step",
            "kind": "step",
            "session_id": 3,
            "ui_scope": "session:3",
            "group_id": "",
            "created_session_id": None,
            "status": "failed",
            "result": {
                "exit_code": 1,
                "stdout": "进度：不应泄漏\nSession-step failure JSON: " + json.dumps(failure, ensure_ascii=False) + "\n",
                "stderr": "debug traceback-like text",
            },
            "error": "进度：不应泄漏 debug traceback-like text",
            "live_stdout": "live trace",
            "live_stderr": "stderr trace",
            "last_line": "stderr trace",
            "created_at": 1.0,
            "updated_at": 2.0,
            "last_output_at": 1.5,
            "completed_at": 2.0,
        }
        public = public_job_snapshot(job)
        self.assertEqual(public["error"], "")
        self.assertEqual(public["result_summary"]["type"], "session_step_failed")
        self.assertEqual(public["result_summary"]["failure"]["reason_code"], "session_evidence_guard_failed")
        serialized = json.dumps(public, ensure_ascii=False)
        self.assertNotIn("debug traceback-like text", serialized)
        self.assertNotIn("进度：不应泄漏", serialized)

    def test_public_step_job_without_structured_result_is_not_false_completed(self) -> None:
        job = {
            "id": 1,
            "name": "session step",
            "kind": "step",
            "session_id": 3,
            "ui_scope": "session:3",
            "group_id": "",
            "created_session_id": None,
            "status": "completed",
            "result": {
                "exit_code": 0,
                "stdout": "进度：模型输出了中间文本但没有 Turn decision\n",
                "stderr": "debug trace",
            },
            "error": "",
            "live_stdout": "live trace",
            "live_stderr": "",
            "last_line": "live trace",
            "created_at": 1.0,
            "updated_at": 2.0,
            "last_output_at": 1.5,
            "completed_at": 2.0,
        }
        public = public_job_snapshot(job)
        self.assertEqual(public["result"], {"exit_code": 0})
        self.assertEqual(public["error"], "")
        self.assertEqual(public["result_summary"]["type"], "session_step_failed")
        self.assertEqual(public["result_summary"]["failure"]["reason_code"], "session_step_failed")
        self.assertIn("没有返回可写入会话", public["result_summary"]["failure"]["reason"])
        serialized = json.dumps(public, ensure_ascii=False)
        self.assertNotIn("模型输出了中间文本", serialized)
        self.assertNotIn("debug trace", serialized)
        self.assertNotIn("live trace", serialized)

    def test_public_review_job_without_structured_result_is_not_false_completed(self) -> None:
        job = {
            "id": 1,
            "name": "review session",
            "kind": "review",
            "session_id": 3,
            "ui_scope": "session:3",
            "group_id": "",
            "created_session_id": None,
            "status": "completed",
            "result": {
                "exit_code": 0,
                "stdout": "进度：审稿模型输出了中间文本但没有 Reviewer decision\n",
                "stderr": "debug trace",
            },
            "error": "",
            "live_stdout": "live trace",
            "live_stderr": "",
            "last_line": "live trace",
            "created_at": 1.0,
            "updated_at": 2.0,
            "last_output_at": 1.5,
            "completed_at": 2.0,
        }
        public = public_job_snapshot(job)
        self.assertEqual(public["result"], {"exit_code": 0})
        self.assertEqual(public["error"], "")
        self.assertEqual(public["result_summary"]["type"], "reviewer_failed")
        self.assertEqual(public["result_summary"]["failure"]["reason_code"], "reviewer_failed")
        self.assertIn("没有返回可保存的审稿结论", public["result_summary"]["failure"]["reason"])
        serialized = json.dumps(public, ensure_ascii=False)
        self.assertNotIn("审稿模型输出了中间文本", serialized)
        self.assertNotIn("debug trace", serialized)
        self.assertNotIn("live trace", serialized)

    def test_gui_post_api_parses_string_booleans_without_enabling_dangerous_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                captured_commands = []

                def fake_streaming(command, root_dir, on_output=None):
                    captured_commands.append(command)
                    return {"exit_code": 0, "stdout": "ok", "stderr": ""}

                with patch("research_alpha.gui.run_app_command_streaming", side_effect=fake_streaming):
                    server = create_gui_http_server(cwd, host="127.0.0.1", port=0)
                    thread = threading.Thread(target=server.serve_forever, daemon=True)
                    thread.start()
                    try:
                        status, cleanup_body = self.gui_post_json(server, "/api/cleanup", {"apply": "false", "ui_scope": "draft:bool"})
                        self.assertEqual(status, 200)
                        status, gold_body = self.gui_post_json(
                            server,
                            "/api/gold-build",
                            {
                                "venues": ["ICLR"],
                                "from_year": 2025,
                                "to_year": 2025,
                                "per_venue_year": 2,
                                "extractive_genomes": "false",
                                "ui_scope": "draft:bool-gold",
                            },
                        )
                        self.assertEqual(status, 200)

                        for _ in range(50):
                            if len(captured_commands) >= 2:
                                break
                            import time

                            time.sleep(0.01)
                        self.assertEqual(cleanup_body["job"]["kind"], "cleanup")
                        self.assertEqual(gold_body["job"]["kind"], "gold-build")
                        flat_commands = [" ".join(command) for command in captured_commands]
                        self.assertTrue(any(command.startswith("cleanup") for command in flat_commands))
                        self.assertTrue(any(command.startswith("gold-build") for command in flat_commands))
                        self.assertFalse(any("--apply" in command for command in flat_commands))
                        self.assertFalse(any("--extractive-genomes" in command for command in flat_commands))

                        status, body = self.gui_post_json(server, "/api/cleanup", {"apply": "sometimes", "ui_scope": "draft:bad-bool"})
                        self.assertEqual(status, 400)
                        self.assertFalse(body["ok"])
                        self.assertEqual(body["code"], "validation_error")
                        self.assertIn("是否执行清理必须是布尔值", body["error"])
                    finally:
                        server.shutdown()
                        server.server_close()
                        thread.join(timeout=2)
            finally:
                os.chdir(old_cwd)

    def test_gui_post_api_defaults_blank_ui_scope_before_deduping_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                release = threading.Event()
                captured_commands = []

                def fake_streaming(command, root_dir, on_output=None):
                    captured_commands.append(command)
                    release.wait(2)
                    return {"exit_code": 0, "stdout": "ok", "stderr": ""}

                with patch("research_alpha.gui.run_app_command_streaming", side_effect=fake_streaming):
                    server = create_gui_http_server(cwd, host="127.0.0.1", port=0)
                    thread = threading.Thread(target=server.serve_forever, daemon=True)
                    thread.start()
                    try:
                        status, first = self.gui_post_json(server, "/api/cleanup", {"ui_scope": ""})
                        self.assertEqual(status, 200)
                        self.assertEqual(first["job"]["ui_scope"], "draft")

                        status, second = self.gui_post_json(server, "/api/cleanup", {})
                        self.assertEqual(status, 200)
                        self.assertTrue(second["deduped"])
                        self.assertEqual(second["job"]["id"], first["job"]["id"])
                        self.assertEqual(second["job"]["ui_scope"], "draft")

                        for _ in range(50):
                            if captured_commands:
                                break
                            import time

                            time.sleep(0.01)
                        self.assertEqual(len(captured_commands), 1)
                    finally:
                        release.set()
                        server.shutdown()
                        server.server_close()
                        thread.join(timeout=2)
            finally:
                os.chdir(old_cwd)

    def test_gui_post_api_dedupes_evidence_jobs_across_kinds(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                self.write_gui_test_llm_env(cwd)
                release = threading.Event()
                captured_commands = []

                def fake_streaming(command, root_dir, on_output=None):
                    captured_commands.append(command)
                    if on_output:
                        on_output("stdout", "evidence job is running")
                    release.wait(2)
                    return {"exit_code": 0, "stdout": "ok", "stderr": ""}

                with patch("research_alpha.gui.run_app_command_streaming", side_effect=fake_streaming):
                    server = create_gui_http_server(cwd, host="127.0.0.1", port=0)
                    thread = threading.Thread(target=server.serve_forever, daemon=True)
                    thread.start()
                    try:
                        scope = "draft:evidence-dedupe"
                        status, first = self.gui_post_json(
                            server,
                            "/api/quality-enrich",
                            {"limit": 5, "provider": "deepseek", "ui_scope": scope},
                        )
                        self.assertEqual(status, 200)
                        self.assertEqual(first["job"]["kind"], "quality-enrich")

                        status, cleanup_body = self.gui_post_json(server, "/api/cleanup", {"apply": False, "ui_scope": scope})
                        self.assertEqual(status, 200)
                        self.assertTrue(cleanup_body["deduped"])
                        self.assertEqual(cleanup_body["job"]["id"], first["job"]["id"])
                        self.assertEqual(cleanup_body["job"]["kind"], "quality-enrich")

                        status, gold_body = self.gui_post_json(
                            server,
                            "/api/gold-build",
                            {
                                "venues": ["ICLR"],
                                "from_year": 2025,
                                "to_year": 2025,
                                "per_venue_year": 2,
                                "ui_scope": scope,
                            },
                        )
                        self.assertEqual(status, 200)
                        self.assertTrue(gold_body["deduped"])
                        self.assertEqual(gold_body["job"]["id"], first["job"]["id"])
                        self.assertEqual(gold_body["jobs"][0]["id"], first["job"]["id"])

                        for _ in range(50):
                            if captured_commands:
                                break
                            import time

                            time.sleep(0.01)
                        self.assertEqual(len(captured_commands), 1)
                        self.assertEqual(captured_commands[0][0], "quality-enrich")
                    finally:
                        release.set()
                        server.shutdown()
                        server.server_close()
                        thread.join(timeout=2)
            finally:
                os.chdir(old_cwd)

    def test_gui_renders_choice_driven_research_setup(self) -> None:
        html = render_index_html()
        self.assertIn('id="venuePresets"', html)
        self.assertIn('id="venueTabs"', html)
        self.assertIn('id="chatMessages"', html)
        self.assertIn('id="instruction"', html)
        self.assertIn("sendChat", html)
        self.assertNotIn("#A079FF", html)
        self.assertIn("newChat", html)
        self.assertIn('id="sessionHistory"', html)
        self.assertIn('class="status-cluster"', html)
        self.assertIn('id="healthDot"', html)
        self.assertIn("health.evidence_status === 'building' ? 'building'", html)
        self.assertIn("function evidenceStatusText(status, ready)", html)
        self.assertIn("missing_provider_key: '需要模型'", html)
        self.assertIn("needs_evidence: '需要证据'", html)
        self.assertIn("ready && data.project.api_key_configured", html)
        self.assertIn("main { height:calc(100vh - 56px); grid-template-columns:260px minmax(520px, 1fr) 280px; background:#fbfbfc; }", html)
        self.assertIn(".side-rail { border-left:1px solid var(--line); padding:18px 16px; }", html)
        self.assertIn(".status-cluster::before", html)
        self.assertIn("deleteSession", html)
        self.assertIn('/api/session/delete', html)
        self.assertIn("function requestErrorFromResponse(json={}, status=0, fallback='Request failed')", html)
        self.assertIn("error.status = Number(json.status || status || 0)", html)
        self.assertIn("error.code = json.code || ''", html)
        self.assertIn("error.retryable = Boolean(json.retryable)", html)
        self.assertIn("error.nextAction = json.next_action || ''", html)
        self.assertIn("HTTP ${error.status}", html)
        self.assertIn("throw requestErrorFromResponse(json, res.status)", html)
        self.assertIn("throw requestErrorFromResponse(json, res.status, '状态刷新失败')", html)
        self.assertIn("const error = requestErrorFromResponse(json, res.status, '删除失败')", html)
        self.assertIn("jobs.forEach(mergeJobIntoLastData)", html)
        self.assertIn("await refreshIfCurrent(token, currentScope())", html)
        self.assertIn("showLibrary", html)
        self.assertIn('id="libraryView"', html)
        self.assertIn('id="libraryNav"', html)
        self.assertIn("libraryCollapsed", html)
        self.assertIn("toggleLibraryPanel", html)
        self.assertIn("deleteIdea", html)
        self.assertIn('/api/idea/delete', html)
        self.assertIn("用户论文库", html)
        self.assertIn("addUserLibraryPaper", html)
        self.assertIn('/api/user-library/add', html)
        self.assertIn("deleteUserLibraryPaper", html)
        self.assertIn('/api/user-library/delete', html)
        self.assertIn("openUserLibraryPaper", html)
        self.assertIn("stable_conversation_idea", Path("research_alpha/gui.py").read_text(encoding="utf-8"))
        self.assertIn("不进入 Gold、评分标准或故事线来源", html)
        self.assertIn("event.key === 'Enter'", html)
        self.assertIn("!event.shiftKey", html)
        self.assertIn("wasNearBottom", html)
        self.assertIn("chatRenderKey", html)
        self.assertIn("pendingNewSession", html)
        self.assertIn("pendingNewSessionJobId", html)
        self.assertIn("pendingNewSessionScope", html)
        self.assertIn("pendingIdeateJobsByScope", html)
        self.assertIn("pendingSearchesByScope", html)
        self.assertIn("function sessionScopeForId(id)", html)
        self.assertIn("function latestScopePreview(scope, fallback='')", html)
        self.assertIn("scopeMessages(transientMessagesByScope, scope).concat(scopeMessages(activityMessagesByScope, scope))", html)
        self.assertIn("const preview = latestScopePreview(scope, s.current_idea)", html)
        self.assertIn("progressByScope", html)
        self.assertIn("startProgress(scope, 'ideate'", html)
        self.assertIn("addPendingSearch(scope, direction)", html)
        self.assertIn("renderSessionHistory(window.lastData || {})", html)
        self.assertIn("progressText(scope)", html)
        self.assertIn("function jobScope(job)", html)
        self.assertIn("function isConversationBlockingJob(job)", html)
        self.assertIn("const running = runningJobsForScope(jobs, scope)", html)
        self.assertIn("runningConversationJobsForScope(jobs, scope)", html)
        self.assertIn("liveJobLineForScope(scope)", html)
        self.assertIn("visibleProgressText(scope)", html)
        self.assertIn("function liveJobLineForScope(scope) {\n      return '';\n    }", html)
        self.assertNotIn("实时输出 ·", html)
        self.assertNotIn("displayText(withLine.last_line", html)
        self.assertIn("function jobTsMs(job, key)", html)
        self.assertIn("function runningJobSilenceText(scope)", html)
        self.assertIn("没有新输出，正在等待远端模型或论文源返回", html)
        self.assertIn("function sidebarProgressText(scope)", html)
        self.assertIn("thinking-note", html)
        self.assertIn("selectPendingSearch", html)
        self.assertIn("cancelPendingSearch", html)
        self.assertIn("等待生成正式对话", html)
        self.assertIn("const status = progress ?", html)
        self.assertIn("awaitingJobStartByScope", html)
        self.assertIn("setAwaitingJobStart(scope", html)
        self.assertIn("scopeAwaitingJobStart(scope)", html)
        self.assertIn("scopeIsBusy(scope)", html)
        self.assertIn("setPendingIdeate(scope", html)
        self.assertIn("clearPendingIdeate(scope", html)
        self.assertIn("pendingIdeateForScope(activeScope)", html)
        self.assertIn("reconcilePendingIdeates(jobs, activeScope)", html)
        self.assertIn("runningJobsForScope(jobs, activeScope)", html)
        self.assertIn("正在运行 ${runningJobs.length} 个任务", html)
        self.assertIn("clearPendingNewSession", html)
        self.assertIn("clearAwaitingUiState", html)
        self.assertIn("refreshSeq", html)
        self.assertIn("viewToken", html)
        self.assertIn("function bumpViewToken()", html)
        self.assertIn("function viewStillCurrent(token, scope='')", html)
        self.assertIn("function renderChatIfCurrent(token, scope, chat)", html)
        self.assertIn("async function refreshIfCurrent(token, scope='')", html)
        self.assertIn("function handleUiFailureIfCurrent(token, scope, label, error, options={})", html)
        self.assertIn("const token = currentViewToken()", html)
        self.assertIn("const token = bumpViewToken()", html)
        self.assertIn("if (!viewStillCurrent(token, scope))", html)
        self.assertIn("if (!viewStillCurrent(token, activeScope)) return", html)
        self.assertIn("await refreshIfCurrent(token, scope)", html)
        self.assertIn("handleUiFailureIfCurrent(token, scope", html)
        self.assertIn("draftScopeId", html)
        self.assertIn("item.ui_scope === scope", html)
        self.assertIn("sourceScope=draftScopeId", html)
        self.assertIn("adoptedFromDraft", html)
        self.assertIn("isPendingIdeateJob", html)
        self.assertIn("isCreatedIdeateForScope", html)
        self.assertIn("adoptDraftMessagesToSession", html)
        self.assertIn("setBusy(true)", html)
        self.assertIn("setBusy(hasBlockingRunning || awaitingCurrentScope)", html)
        self.assertIn("awaitingJobStart", html)
        self.assertIn("transientMessagesByScope[draftScopeId] = []", html)
        self.assertNotIn("transientMessagesByScope = {};\n      activityMessagesByScope = {};\n      chatRenderKey = '';", html)
        self.assertIn("sendButton.disabled = isBusy", html)
        self.assertIn("instruction.disabled = isBusy", html)
        self.assertIn("class=\"spinner\"", html)
        self.assertIn("bubble-spinner", html)
        self.assertIn("activityMessagesByScope", html)
        self.assertIn("transientMessagesByScope", html)
        self.assertIn("pushActivityMessage", html)
        self.assertIn("任务已提交：", html)
        self.assertIn("ui_scope", html)
        self.assertIn("?draft=1", html)
        self.assertIn('/api/session/new', html)
        self.assertIn('/api/quality-enrich', html)
        self.assertIn('/api/review-loop', html)
        self.assertIn('id="jobFeed"', html)
        self.assertIn("typing-dots", html)
        self.assertIn("审稿", html)
        self.assertIn("循环", html)
        self.assertIn("循环审稿", html)
        self.assertNotIn("质量审查</button>", html)
        self.assertIn("function reviewLoop()", html)
        self.assertIn('onclick="reviewLoop()"', html)
        self.assertIn("自动经过 3-5 轮审稿专家意见调整", html)
        self.assertIn("自动审稿循环", html)
        self.assertNotIn("并自动进行审稿循环", html)
        self.assertIn("正在生成 idea", html)
        self.assertIn("优秀论文逻辑库", html)
        self.assertIn("证据已就绪", html)
        self.assertIn("const health = data.health || {}", html)
        self.assertIn("health.evidence_ready", html)
        self.assertIn("health.evidence_status === 'building'", html)
        self.assertIn("const statusText = evidenceStatusText(health.evidence_status, ready)", html)
        self.assertIn("evidence_failed: '证据任务失败'", html)
        self.assertIn("证据任务失败", html)
        self.assertIn("latest_failed_evidence_job", html)
        self.assertIn("const action = failedEvidence.kind === 'quality-enrich' ? 'quality-enrich' : 'evidence-build';", html)
        self.assertIn("el.onclick = () => metricAction(action);", html)
        self.assertIn("点击重新核验", html)
        self.assertIn("点击调整检索", html)
        self.assertIn("document.getElementById('strictStatus').onclick = () => metricAction(failedEvidence ? 'evidence-build' : 'excellent');", html)
        self.assertIn("可重试或缩小检索范围", html)
        self.assertIn("missing_requirements", html)
        self.assertIn("证据构建中", html)
        self.assertIn("Run setup", html)
        self.assertIn("Activity", html)
        self.assertIn("history-panel", html)
        self.assertIn("flex:1 1 auto", html)
        self.assertIn("min-height:180px", html)
        self.assertIn(".left-rail { border-right:1px solid var(--line); background:#fff; padding:14px 12px; display:flex; flex-direction:column; gap:12px; overflow:auto; min-height:0; }", html)
        self.assertIn("扩充核心库", html)
        self.assertIn("核验趋势质量", html)
        self.assertIn("Best、Nominee 和 Oral", html)
        self.assertIn("高引用不单独入库", html)
        self.assertIn("趋势论文", html)
        self.assertIn("renderPaperRow", html)
        self.assertIn("paper-reason", html)
        self.assertIn("paper-status", html)
        self.assertIn("library_status", html)
        self.assertIn("library_status_detail", html)
        self.assertIn("statusDetail", html)
        self.assertIn("quality_reason", html)
        self.assertIn("source_label", html)
        self.assertIn("target=\"_blank\"", html)
        self.assertIn("本次联网趋势论文", html)
        self.assertIn("function parseJsonLine(result, label)", html)
        self.assertIn("function parseGroundingAuditFailure", html)
        self.assertIn("groundingAuditJobText", html)
        self.assertIn("没有通过证据锚点审计", html)
        self.assertIn("系统已拦截这次不可靠输出", html)
        self.assertIn("Idea-session summary JSON", html)
        self.assertNotIn("这次没有写入新对话，因为严格生成门没有放行 idea。", html)
        self.assertIn("证据驱动评分：${summary.evidence_score}/10", html)
        self.assertIn("顶会论文逻辑迁移：", html)
        self.assertIn("六步故事线迁移：", html)
        self.assertIn("当前方法问题：", html)
        self.assertIn("近半年局限性/趋势信号：", html)
        self.assertIn("参考依据：", html)
        self.assertIn("评分维度：", html)
        self.assertIn("explicit_evidence_covered", html)
        self.assertIn("Prior-art gate：", html)
        self.assertIn("审稿风险：", html)
        self.assertIn("首个实验：", html)
        self.assertIn("为什么现在可做：", html)
        self.assertIn("function ideateJobText", html)
        self.assertIn("function ideateSummaryText", html)
        self.assertIn("idea_session_review_ready", html)
        self.assertIn("idea_session_review_incomplete", html)
        self.assertIn("自动审稿循环：已完成", html)
        self.assertIn("自动审稿循环没有完成，下面是生成阶段保留的当前 idea。", html)
        self.assertIn("function sessionStepSummaryText", html)
        self.assertIn("function reviewSummaryText", html)
        self.assertIn("job.result_summary || null", html)
        self.assertIn("已生成新的研究 idea，并写入当前对话", html)
        self.assertIn("核心想法", html)
        self.assertIn("近期依据", html)
        self.assertIn("function sessionStepJobText", html)
        self.assertIn("Session-step failure JSON", html)
        self.assertIn("本轮追问没有写入当前会话", html)
        self.assertIn("模型本轮改写越出了优秀论文逻辑空间", html)
        self.assertIn("function reviewJobText", html)
        self.assertIn("Reviewer failure JSON", html)
        self.assertIn("审稿人视角没有写入结果", html)
        self.assertIn("Review-loop failure JSON", html)
        self.assertIn("data.recovery_strategy || failure.recovery_strategy", html)
        self.assertIn("data.safe_resume_command || failure.safe_resume_command", html)
        self.assertIn("安全继续：${safeResume}", html)
        self.assertIn("function cleanupJobText", html)
        self.assertIn("本轮追问已写入当前会话，并刷新了会话记忆和 dossier", html)
        self.assertIn("Session memory JSON", html)
        self.assertIn("记忆状态：", html)
        self.assertIn("稳定主线：", html)
        self.assertIn("未解决风险：", html)
        self.assertIn("下一步实验：", html)
        self.assertIn("审稿人视角已完成：已从 novelty、证据链、逻辑和可行性角度攻击当前 idea", html)
        self.assertIn("旧版非核心 Gold 记录", html)
        self.assertIn("证据上下文体检已完成", html)
        self.assertIn("修改后的 idea", html)
        self.assertIn("攻击摘要", html)
        self.assertIn("近半年局限性信号", html)
        self.assertIn("抗审稿攻击性的证据驱动评分", html)
        self.assertIn('value="8"', html)
        self.assertIn("Paper-grounded assets", html)
        self.assertIn("Start a deep research search", html)
        self.assertIn("fillPrompt", html)
        self.assertIn("quick-prompts", html)
        self.assertIn("<label>近期趋势源</label>", html)
        self.assertIn("扩充优秀论文库的会议和年份在 Evidence library 中设置", html)
        self.assertIn("对话生成设置</div>", html)
        self.assertNotIn("function sourcePlanNote(plan)", html)
        self.assertNotIn("核心优秀论文源路由：", html)
        self.assertNotIn("source_plan", html)
        self.assertNotIn("OpenAlex 近期候选", html)
        self.assertNotIn("Semantic Scholar 近期候选", html)
        self.assertNotIn("辅助候选", html)
        self.assertIn("metric-button", html)
        self.assertIn("function metricAction(action)", html)
        self.assertIn("function renderMetricCard", html)
        self.assertIn("let metricsRenderKey = ''", html)
        self.assertIn("function renderMetrics(metricCards)", html)
        self.assertIn("if (nextKey === metricsRenderKey) return", html)
        self.assertIn("function scrollLibraryTo(id)", html)
        self.assertIn("Corpus status", html)
        self.assertIn("corpusStatusPanel", html)
        self.assertIn("data-tooltip", html)
        self.assertIn("tooltip-anchor", html)
        self.assertIn(".evidence-card.action-panel", html)
        self.assertIn("clickable-row", html)
        self.assertIn('button id="provider" class="status-pill"', html)
        self.assertIn('button id="strictStatus" class="status-pill"', html)
        self.assertIn('onclick="metricAction(\'model-settings\')"', html)
        self.assertIn('id="searchSettings"', html)
        self.assertIn('id="evidenceBuildPanel"', html)
        self.assertIn("function evidenceScope() { return evidenceScopeId; }", html)
        self.assertIn("const evidenceScopeId = 'evidence:global';", html)
        self.assertIn("status-link", html)
        self.assertIn("button.status-pill:hover", html)
        self.assertIn("button.status-mini", html)
        self.assertIn("target.scrollIntoView({block:'start', behavior:'smooth'})", html)
        self.assertIn("核心库", html)
        self.assertIn("逻辑库", html)
        self.assertIn("模式库", html)
        self.assertIn("对话区", html)
        self.assertIn("data-action-target", html)
        self.assertIn("scrollLibraryTo", html)
        self.assertIn("点击查看", html)
        self.assertIn('aria-label="${esc(title)}，打开${esc(target)}"', html)
        self.assertIn('data-action-target="${esc(target)}"', html)
        self.assertIn("metric-button:active", html)
        self.assertIn("action-panel", html)
        self.assertIn("action-panel-link", html)
        self.assertIn("核心库", html)
        self.assertIn("进度", html)
        self.assertIn("empty-action", html)
        self.assertIn("panel-jump", html)
        self.assertIn('aria-label="打开证据库">›</button>', html)
        self.assertNotIn('title="打开证据库">打开证据库</button>', html)
        self.assertIn(".paper-row.clickable-row", html)
        self.assertIn("button.paper-row", html)
        self.assertIn("const helper = hint || '打开对应位置'", html)
        self.assertIn("function cleanUiText(text)", html)
        self.assertIn("function panelNavigate(event, action)", html)
        self.assertIn("function activatePanel(event, action)", html)
        self.assertIn("const candidateWord = '\\u5019\\u9009'", html)
        self.assertIn("const recentCandidateFew = '\\u8fd1\\u671f' + candidateWord + '\\u51e0\\u4e2a'", html)
        self.assertIn(".split(recentCandidateFew).join('近期趋势线索')", html)
        self.assertIn(".split(recentCandidate).join('趋势')", html)
        self.assertIn("section-action", html)
        self.assertIn("target.classList.add('flash')", html)
        self.assertIn("metric-open", html)
        self.assertIn("查看严格论文逻辑卡", html)
        self.assertIn("严格可用故事线", html)
        self.assertIn("只含 Best / Nominee / Oral", html)
        self.assertIn("data-tooltip=\"${esc(helper)} · ${esc(target)}\"", html)
        self.assertIn(".metric-button:hover::after", html)
        self.assertIn("bubble-actions", html)
        self.assertIn("bubble-action", html)
        self.assertIn("function jobActions(job)", html)
        self.assertIn("function jobActivity(job)", html)
        self.assertIn("actions:jobActions(job)", html)
        self.assertIn("function messageAction(action)", html)
        self.assertIn("messageAction('${esc(action.action)}')", html)
        self.assertIn("job.status === 'failed'", html)
        self.assertIn("{label:'调整检索', action:'settings'}", html)
        self.assertIn("if (action === 'settings') metricAction('evidence-build');", html)
        self.assertIn("{label:'重新核验', action:'quality-enrich'}", html)
        self.assertIn("查看核心库", html)
        self.assertIn("查看趋势论文", html)
        self.assertIn("核验质量标签", html)
        self.assertIn("if (action === 'quality-enrich') qualityEnrich();", html)
        job_actions = re.search(r"function jobActions\(job\) \{([\s\S]*?)\n    \}", html)
        self.assertIsNotNone(job_actions)
        job_actions_body = job_actions.group(1)
        self.assertIn("if (job.kind === 'gold-build')", job_actions_body)
        self.assertIn("? [{label:'调整检索', action:'settings'}, {label:'查看趋势论文', action:'frontier'}]", job_actions_body)
        self.assertIn("? [{label:'重新核验', action:'quality-enrich'}, {label:'查看趋势论文', action:'frontier'}]", job_actions_body)

    def test_gui_sidebars_do_not_render_live_output_lines(self) -> None:
        html = render_index_html()
        self.assertIn("function sidebarProgressText(scope)", html)
        self.assertIn("sidebarProgressText(scope) || '任务排队中'", html)
        self.assertIn("const progress = sidebarProgressText(scope);", html)
        self.assertIn("el.textContent = awaitingEvidence ? '证据任务提交中' : (running ? `正在运行 ${runningJobs.length} 个任务` : '空闲');", html)
        self.assertIn("latestFinishedEvidence", html)
        self.assertIn("latestFinishedEvidence.status === 'failed'", html)
        self.assertIn("latestCompletedEvidence", html)
        self.assertIn("function evidenceCompletionSummary(job, jobs=[])", html)
        self.assertIn("Gold ${summary.excellent_imported ?? 0} · 趋势 ${summary.candidate_imported ?? 0} · 失败 ${summary.failures ?? 0}", html)
        self.assertIn("新增标签 ${applied || 0} · 跳过 ${skipped || 0}", html)
        self.assertIn("最近完成：核心库扩充", html)
        self.assertIn("${label}${summary ? `：${summary}` : ''}。点击查看。", html)
        self.assertIn("证据任务已结束；对话区不显示中间输出，结果可在证据库或运行记录查看。", html)
        self.assertIn("function liveJobLineForScope(scope) {\n      return '';\n    }", html)
        self.assertNotIn("实时输出 ·", html)
        self.assertNotIn("displayText(withLine.last_line", html)
        self.assertNotIn("el.textContent = `正在运行 ${runningJobs.length} 个任务：${displayJobName(running)} · ${displayText(running.last_line)}`;", html)

    def test_gui_library_panels_are_collapsible_and_ideas_deletable(self) -> None:
        html = render_index_html()
        self.assertIn("let libraryCollapsed = {}", html)
        self.assertIn("function toggleLibraryPanel(id)", html)
        self.assertIn("class=\"library-toggle\"", html)
        self.assertIn("aria-expanded=\"${collapsed ? 'false' : 'true'}\"", html)
        self.assertIn(".library-panel.collapsed .library-body { display:none; }", html)
        self.assertIn("deleteIdea(${Number(i.id)}, event)", html)
        self.assertIn('/api/idea/delete', html)
        self.assertIn("delete-artifact", html)

    def test_gui_user_library_uses_right_sidebar_jump_and_full_library_panel(self) -> None:
        html = render_index_html()
        self.assertIn('id="sideUserLibrary" class="user-library-tab" role="button" tabindex="0"', html)
        self.assertIn('id="userLibraryPanel" class="library-panel user-library-main"', html)
        self.assertIn(".user-library-main, .evidence-build-main { grid-column:1 / -1; }", html)
        self.assertIn("function renderUserLibraryPanel(userPapers=[])", html)
        self.assertIn("function renderUserLibraryMainPanel(userPapers=[])", html)
        self.assertIn("renderUserLibraryPanel(userPapers);", html)
        self.assertIn("renderUserLibraryMainPanel(userPapers);", html)
        self.assertIn(".side-rail > * { flex:0 0 auto; margin-bottom:0; }", html)
        self.assertIn(".side-rail { display:flex; flex-direction:column; min-height:0; overflow-y:auto; gap:18px; padding-bottom:30px; }", html)
        self.assertIn(".user-library-tab { order:2; min-height:64px;", html)
        self.assertIn(".user-library-tab { order:2; min-height:64px; padding:0; display:flex; align-items:center; justify-content:space-between; gap:10px; margin-top:0;", html)
        self.assertIn(".user-library-tab-label", html)
        self.assertIn(".user-library-tab-arrow", html)
        self.assertNotIn(".user-library-side .paper-list", html)
        self.assertNotIn("side-user-library-head", html)
        self.assertNotIn("user-library-jump", html)
        self.assertNotIn("const previousScrollTop = previousList ? previousList.scrollTop : 0;", html)
        self.assertIn("#searchSettings { order:3; margin-top:0; min-height:112px; }", html)
        self.assertIn(".side-rail > .tool-panel.action-panel { order:1; }", html)
        self.assertIn(".side-rail .tool-panel.action-panel:last-of-type { order:4; margin-top:clamp(10px, 2.4vh, 24px); }", html)
        logic_pos = html.find("优秀论文逻辑库")
        side_library_pos = html.find('id="sideUserLibrary"')
        evidence_build_pos = html.find('id="evidenceBuildPanel"')
        self.assertGreater(side_library_pos, logic_pos)
        self.assertGreater(side_library_pos, evidence_build_pos)
        self.assertIn("扩充优秀论文库", html)
        self.assertIn("会议组合", html)
        self.assertIn("同时生成可审计逻辑卡", html)
        self.assertIn("用户论文库", html)
        self.assertIn("user-library-tab-label", html)
        self.assertIn("只作领域背景和路线学习，不进入 Gold、评分标准或故事线来源。", html)
        self.assertIn("if (action === 'user-library')", html)

    def test_gui_user_library_is_not_appended_to_core_paper_panel(self) -> None:
        html = render_index_html()
        self.assertNotIn("const userLibraryForm", html)
        papers_body_match = re.search(r"const papersBody = ([\s\S]*?);\n      panel\('papers'", html)
        self.assertIsNotNone(papers_body_match)
        papers_body = papers_body_match.group(1)
        self.assertNotIn("userLibrary", papers_body)
        self.assertNotIn("用户论文库", papers_body)

    def test_gui_tooltips_render_in_fixed_portal_not_clipped_panel_pseudo_element(self) -> None:
        html = render_index_html()
        self.assertIn("const tooltipPortal = (() =>", html)
        self.assertIn("document.body.appendChild(node)", html)
        self.assertIn(".tooltip-popover { position:fixed;", html)
        self.assertIn("z-index:2000", html)
        self.assertIn(".tooltip-anchor::after { content:none; display:none; }", html)
        self.assertIn(".side-rail, .tool-panel, .tool-panel .section-title { overflow:visible; }", html)

    def test_gui_clears_stale_refresh_failure_messages_after_recovery(self) -> None:
        html = render_index_html()
        self.assertIn("function clearRefreshFailureMessages(scope='')", html)
        self.assertIn("String(item.text || '').startsWith('状态刷新失败')", html)
        self.assertIn("if (lastStateError) clearRefreshFailureMessages();", html)
        self.assertIn("lastStateError = '';", html)

    def test_gui_completed_ideate_session_uses_chat_result_not_stdout_activity(self) -> None:
        html = render_index_html()
        finalize_match = re.search(r"function finalizeCompletedIdeateJob\(job, activeScope='', options=\{\}\) \{([\s\S]*?)\n    \}", html)
        self.assertIsNotNone(finalize_match)
        finalize_body = finalize_match.group(1)
        self.assertNotIn("if (!createdScope)", finalize_body)
        self.assertNotIn("jobActivity(job)", finalize_body)
        self.assertIn("if (options.activate && createdScope)", finalize_body)
        self.assertIn("if (job.kind === 'ideate' && job.created_session_id)", html)
        self.assertIn("lastJobStatuses.set(job.id, job.status);\n            continue;", html)
        self.assertIn("if (isConversationBlockingJob(job))", html)
        self.assertNotIn("这次没有写入新对话，因为严格生成门没有放行 idea。", html)
        self.assertIn("const text = ideateSummaryText(summary);", html)

    def test_gui_tool_panels_are_quiet_glass_without_corner_badges(self) -> None:
        html = render_index_html()
        self.assertIn(".action-panel::before, .action-panel::after { content:none !important; display:none !important; }", html)
        self.assertIn("background:rgba(255,255,255,.66)", html)
        self.assertIn("backdrop-filter:saturate(155%) blur(16px)", html)
        self.assertIn("-webkit-backdrop-filter:saturate(155%) blur(16px)", html)
        self.assertIn(".tool-panel.accent { border-left:1px solid rgba(17,24,39,.09); box-shadow:0 1px 2px rgba(17,24,39,.018); }", html)
        self.assertIn(".tool-panel .section-title { margin-bottom:20px; align-items:flex-start; gap:14px; }", html)
        self.assertIn(".tool-panel .action-panel-link { display:none; }", html)
        self.assertIn(".tool-panel .tooltip-anchor { width:24px; height:24px; margin-top:1px; border-color:rgba(17,24,39,.12); color:#6b7280;", html)
        self.assertIn(".tool-panel .panel-actions { gap:10px; margin-top:22px; }", html)
        self.assertIn(".tool-panel .panel-actions button { width:100%; min-height:42px; border:1px solid rgba(17,24,39,.10); background:#111827; color:#fff; box-shadow:none; }", html)
        self.assertIn("button.black { background:#111827; color:#fff; box-shadow:var(--shadow-soft); }", html)
        self.assertIn("#sendButton { width:88px; min-width:88px; min-height:40px; padding:0 8px; border-radius:6px; background:#111827; color:#fff;", html)
        self.assertIn(".side-rail details.settings summary::after { content:none; display:none; }", html)
        self.assertIn(".panel-jump { flex:0 0 auto; display:inline-grid; place-items:center; width:28px; height:28px; padding:0;", html)
        self.assertNotIn(".tool-panel.accent { border-left:3px solid var(--accent)", html)
        self.assertNotIn("border-color:#ede8f7", html)
        self.assertNotIn('.action-panel { --action-label:"进入"; }', html)
        self.assertNotIn(".action-panel::after { top:12px; right:14px; }", html)
        self.assertNotIn(".clickable-row::after, .status-link::after, .action-panel::after", html)
        self.assertNotIn(".clickable-row::before, .status-link::before, .action-panel::before", html)
        self.assertNotIn(".action-panel:hover::after", html)
        self.assertNotIn(".action-panel:hover::before", html)
        self.assertNotIn(".tool-panel.accent { border-color:#d9cdfd; background:#fff; box-shadow:inset 3px 0 0 var(--accent)", html)
        self.assertNotIn("background:#5f68d8", html)
        self.assertNotIn("background:#555ec9", html)
        self.assertNotIn(".tool-panel.action-panel::before, .tool-panel.action-panel::after { display:none; }", html)
        final_quiet = html.find(".tool-panel.accent { border-left:1px solid rgba(17,24,39,.09); box-shadow:0 1px 2px rgba(17,24,39,.018); }")
        self.assertNotEqual(final_quiet, -1)
        self.assertNotIn(".tool-panel.accent { box-shadow:inset 4px 0 0 var(--accent)", html)
        final_action_panel_css = html[html.find(".action-panel::before, .action-panel::after { content:none !important") :]
        self.assertNotIn('content:"进入"', final_action_panel_css)
        self.assertNotIn("inset 4px 0 0 var(--accent)", final_action_panel_css)
        self.assertNotIn("border-color:rgba(94,106,210,.24); background:#fbfbff", final_action_panel_css)
        for purple_token in ["#d9cdfd", "#cbbcff", "rgba(94,106,210", "rgba(160,121,255", "#5f35c8"]:
            self.assertNotIn(purple_token, html)
        self.assertNotIn(".tool-panel .tooltip-anchor { width:32px", final_action_panel_css)
        self.assertNotIn("button.black { background:var(--accent);", final_action_panel_css)

    def test_gui_frontend_uses_real_app_contract_not_gemini_stub_contract(self) -> None:
        html = render_index_html()
        self.assertIn("chatMessages", html)
        self.assertIn("sessionHistory", html)
        self.assertIn("goldBuild()", html)
        self.assertIn("qualityEnrich()", html)
        self.assertIn("review()", html)
        self.assertIn("sendChat()", html)
        self.assertIn("/api/ideate", html)
        self.assertIn("/api/step", html)
        self.assertIn("/api/review", html)
        self.assertIn("/api/gold-build", html)
        self.assertNotIn("txPromptInput", html)
        self.assertNotIn("workspaceChatStream", html)
        self.assertNotIn("executeSearchSubmission", html)
        self.assertNotIn("invokeReviewerAttack", html)
        self.assertNotIn("invokeGoldRebuild", html)
        self.assertNotIn("performDomUiSync", html)
        self.assertNotIn("ulHistoryLinks", html)
        self.assertNotIn("tbodyGoldRows", html)
        self.assertNotIn("tbodyFrontierRows", html)
        self.assertIn("api_key_configured", html)
        for internal_field in ["content_json", "context_json", "memory_summary_json", "manifest_path", "api_key:", "apiKey", "db_path", "root_dir"]:
            self.assertNotIn(internal_field, html)

    def test_gui_http_smoke_exposes_page_and_public_state_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                (cwd / ".env").write_text(
                    "RA_LLM_PROVIDER=deepseek\nRA_LLM_MODEL=deepseek-chat\nDEEPSEEK_API_KEY=secret-for-smoke-test\n",
                    encoding="utf-8",
                )
                state = GuiState()
                server = create_gui_http_server(cwd, host="127.0.0.1", port=0, state=state)
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                try:
                    base = f"http://127.0.0.1:{server.server_port}"
                    with patch.dict(
                        os.environ,
                        {
                            "RA_LLM_PROVIDER": "deepseek",
                            "RA_LLM_MODEL": "deepseek-chat",
                            "DEEPSEEK_API_KEY": "secret-for-smoke-test",
                            "OPENAI_API_KEY": "",
                        },
                        clear=False,
                    ):
                        html = urllib.request.urlopen(base + "/", timeout=5).read().decode("utf-8")
                        state_payload = json.loads(
                            urllib.request.urlopen(base + "/api/state?draft=1", timeout=5).read().decode("utf-8")
                        )
                    for marker in [
                        "function sendChat()",
                        "function reviewLoop()",
                        "function renderChat(chat)",
                        "function renderLibrary(data)",
                        "function reviewLoopSummaryText(summary)",
                        "user-library-tab",
                        "/api/ideate",
                        "/api/step",
                        "/api/review-loop",
                        "/api/user-library/add",
                    ]:
                        self.assertIn(marker, html)
                    self.assertEqual(state_payload["project"]["provider"], "deepseek")
                    self.assertTrue(state_payload["project"]["api_key_configured"])
                    self.assertIn("chat", state_payload)
                    self.assertIn("jobs", state_payload)
                    serialized = json.dumps(state_payload, ensure_ascii=False)
                    for forbidden in [
                        "secret-for-smoke-test",
                        "context_json",
                        "content_json",
                        "memory_summary_json",
                        "manifest_path",
                        "db_path",
                        "root_dir",
                        str(cwd),
                    ]:
                        self.assertNotIn(forbidden, serialized)
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=2)
            finally:
                os.chdir(old_cwd)

    def test_gui_clickable_surfaces_use_subtle_hover_not_floating_cards(self) -> None:
        html = render_index_html()
        plastic_patterns = [
            ".metric-button:hover, .metric-button:focus-visible, .clickable-row:hover",
            "button.status-pill:hover, button.status-pill:focus-visible, .nav-button:hover",
            ".session-item:hover",
            "details.settings:hover",
            ".bubble-action:hover",
        ]
        for selector in plastic_patterns:
            start = html.find(selector)
            self.assertNotEqual(start, -1, selector)
            block = html[start : html.find("}", start) + 1]
            self.assertNotIn("translateY(-1px)", block)
            self.assertNotIn("translateY(-2px)", block)
            self.assertNotIn("0 10px 24px", block)
            self.assertNotIn("0 8px 18px", block)
        self.assertIn("box-shadow:inset 0 0 0 1px rgba(17,24,39,.08)", html)
        self.assertIn(".metric-button:hover::before, .metric-button:focus-visible::before { opacity:.72; }", html)
        self.assertIn("if (action === 'review') review();", html)
        self.assertIn("onclick=\"panelNavigate(event, 'evidence-build')\"", html)
        self.assertIn("onkeydown=\"activatePanel(event, 'chat')\"", html)
        self.assertIn("点击去扩充核心优秀论文库", html)
        self.assertIn("handleUiFailure", html)
        self.assertIn("请求失败", html)
        self.assertIn("下一步：${nextAction}", html)
        self.assertIn("只接受 ICLR、NeurIPS/NIPS、ICML、CVPR、ICCV、TPAMI", html)
        self.assertIn("普通论文和元数据来源先进入趋势区", html)
        self.assertNotIn("metadata_only_venues", html)
        self.assertIn("核心优秀论文库只接受 ICLR、NeurIPS/NIPS、ICML、CVPR、ICCV、TPAMI", html)
        self.assertNotIn("个官方优秀论文源正在联网拉取", html)
        self.assertNotIn("alert(", html)
        self.assertNotIn("throw error", html)
        self.assertIn("查看核心优秀论文库", html)
        self.assertIn("回到对话工作区", html)
        self.assertIn("证据任务只显示简短状态，结果进入证据库", html)
        self.assertNotIn("<h2>Jobs</h2>", html)
        self.assertNotIn("Idea Source", html)
        self.assertNotIn("source grounded", html)
        self.assertNotIn("evidence ready", html)
        self.assertNotIn('id="chatTitle"', html)
        self.assertNotIn('id="chatSubtitle"', html)
        self.assertNotIn("<h2>研究工作区</h2>", html)
        self.assertNotIn("New thread", html)
        self.assertNotIn("Evidence assets", html)
        self.assertNotIn('rank": "Gold"', html)
        self.assertNotIn("#101828", html)
        self.assertNotIn('id="fieldOptions"', html)
        self.assertNotIn('id="hotspotOptions"', html)
        self.assertNotIn('id="constraintOptions"', html)
        self.assertNotIn("details class=\"library\"", html)
        self.assertNotIn("updateDirectionDraft", html)
        self.assertNotIn("将发送给 agent 的方向", html)
        self.assertNotIn("强制流程", html)
        self.assertNotIn("Gold Set 构建", html)
        self.assertNotIn("Prior-art 调研专家", html)
        self.assertNotIn("Auto session: #", html)
        self.assertGreaterEqual(len(FIELD_OPTIONS), 5)
        self.assertGreaterEqual(len(HOTSPOT_OPTIONS), 5)
        self.assertTrue(any("近半年" in item["label"] for item in CONSTRAINT_OPTIONS))
        self.assertTrue(any(venue["rank"] == "核心" for group in VENUE_GROUPS for venue in group["venues"]))

    def test_gui_draft_snapshot_keeps_new_chat_empty(self) -> None:
        from research_alpha.gui import build_draft_chat_snapshot

        snapshot = build_draft_chat_snapshot()
        self.assertFalse(snapshot["active"])
        self.assertTrue(snapshot["draft"])
        self.assertIsNone(snapshot["session"])
        self.assertIn("已准备好新对话", snapshot["messages"][0]["text"])

    def test_gui_extracts_created_session_id_from_job_output(self) -> None:
        self.assertEqual(
            extract_created_session_id({"stdout": "Auto session: #42 agent eval\nDossier: outputs/sessions/session-42-dossier.json"}),
            42,
        )
        self.assertEqual(
            extract_created_session_id({"stdout": "Created idea session #17 from candidate idea #3"}),
            17,
        )
        self.assertIsNone(extract_created_session_id({"stdout": "Strict evidence is not ready yet."}))

    def test_gui_ideate_job_completion_does_not_steal_active_session(self) -> None:
        import time

        state = GuiState()
        state.active_session_id = 7
        job = state.add_job(
            "ideate test",
            lambda: {"stdout": "Auto session: #42 agent eval\nDossier: outputs/sessions/session-42-dossier.json"},
            kind="ideate",
            ui_scope="draft",
        )

        for _ in range(50):
            snapshot = state.snapshot()[0]
            if snapshot["status"] == "completed":
                break
            time.sleep(0.01)

        snapshot = state.snapshot()[0]
        self.assertEqual(snapshot["status"], "completed")
        self.assertEqual(snapshot["created_session_id"], 42)
        self.assertEqual(state.active_session_id, 7)
        self.assertEqual(job["ui_scope"], "draft")

    def test_gui_conversation_job_public_snapshot_hides_live_output_lines(self) -> None:
        import time

        state = GuiState()

        def target(job):
            state.append_job_output(job, "stdout", "stage one\n")
            state.append_job_output(job, "stderr", "stage two\n")
            return {"stdout": "done"}

        job = state.add_job("streaming test", target, kind="ideate", ui_scope="draft", pass_job=True)
        for _ in range(50):
            snapshot = state.snapshot()[0]
            if snapshot["status"] == "completed":
                break
            time.sleep(0.01)

        snapshot = state.snapshot()[0]
        self.assertEqual(snapshot["status"], "completed")
        self.assertEqual(snapshot["live_stdout"], "")
        self.assertEqual(snapshot["live_stderr"], "")
        self.assertEqual(snapshot["last_line"], "")
        self.assertIsNotNone(snapshot["last_output_at"])
        self.assertGreaterEqual(float(snapshot["updated_at"]), float(snapshot["created_at"]))
        self.assertGreaterEqual(float(snapshot["last_output_at"]), float(snapshot["created_at"]))
        self.assertGreaterEqual(float(snapshot["completed_at"]), float(snapshot["created_at"]))

    def test_gui_job_public_snapshot_redacts_local_paths(self) -> None:
        job = {
            "id": 1,
            "name": "ideate test /Users/example/project DEEPSEEK_API_KEY=sk-name-secret-123456",
            "kind": "ideate",
            "session_id": None,
            "ui_scope": "draft:/Users/example/project:DEEPSEEK_API_KEY=sk-scope-secret-123456",
            "group_id": "gold-build:/private/var/tmp/project:sk-group-secret-123456",
            "created_session_id": 42,
            "status": "completed",
            "result": {
                "command": "ra ideate --secret-internal",
                "exit_code": 0,
                "stdout": "Manifest: /Users/example/project/outputs/run/manifest.json\nDEEPSEEK_API_KEY=sk-live-secret-123456",
                "stderr": "Authorization: Bearer test-deepseek-key-123456",
                "debug_path": "/Users/example/project/debug.json",
            },
            "error": "Dossier: /private/var/tmp/session.json\nOPENAI_API_KEY: sk-openai-secret-123456",
            "live_stdout": "Report: /Users/example/project/outputs/run/run_report.md\nsk-live-secret-123456\n",
            "live_stderr": "Output: /tmp/secret/run.json\nBearer test-deepseek-key-123456\n",
            "last_line": "Database: /Users/example/project/data/research_alpha.db DEEPSEEK_API_KEY=sk-last-secret-123456",
            "created_at": 1.0,
            "updated_at": 2.0,
            "last_output_at": 1.5,
            "completed_at": 2.0,
        }
        public = public_job_snapshot(job)
        serialized = json.dumps(public, ensure_ascii=False)
        self.assertNotIn("/Users/", serialized)
        self.assertNotIn("/private/", serialized)
        self.assertNotIn("/tmp/secret", serialized)
        self.assertNotIn("research_alpha.db", serialized)
        self.assertNotIn("debug_path", serialized)
        self.assertNotIn("--secret-internal", serialized)
        self.assertNotIn("sk-live-secret", serialized)
        self.assertNotIn("sk-openai-secret", serialized)
        self.assertNotIn("sk-last-secret", serialized)
        self.assertNotIn("sk-name-secret", serialized)
        self.assertNotIn("sk-scope-secret", serialized)
        self.assertNotIn("sk-group-secret", serialized)
        self.assertNotIn("test-deepseek-key", serialized)
        self.assertNotIn("DEEPSEEK_API_KEY=sk-", serialized)
        self.assertNotIn("OPENAI_API_KEY: sk-", serialized)
        self.assertIn("[local path]", public["name"])
        self.assertIn("[redacted]", public["name"])
        self.assertIn("[local path]", public["ui_scope"])
        self.assertIn("[local path]", public["group_id"])
        self.assertEqual(public["result"], {"exit_code": 0})
        self.assertEqual(public["live_stdout"], "")
        self.assertEqual(public["live_stderr"], "")
        self.assertEqual(public["last_line"], "")
        self.assertIn("[local path]", serialized)
        self.assertIn("[redacted]", serialized)
        self.assertEqual(public["created_session_id"], 42)

    def test_gui_local_path_redaction_preserves_non_path_text(self) -> None:
        redacted = redact_local_paths("Created idea session #17\nManifest: /Users/me/run/manifest.json\nok")
        self.assertIn("Created idea session #17", redacted)
        self.assertIn("Manifest: [local path]", redacted)
        self.assertNotIn("/Users/me", redacted)

    def test_gui_secret_redaction_preserves_non_secret_text(self) -> None:
        text = (
            "Created idea session #17\n"
            "DEEPSEEK_API_KEY=sk-test-secret-123456\n"
            "OPENAI_API_KEY: sk-openai-secret-123456\n"
            "Authorization: Bearer test-deepseek-key-123456\n"
            "use command ra ds sk-... for setup"
        )
        redacted = redact_secrets(text)
        self.assertIn("Created idea session #17", redacted)
        self.assertIn("ra ds sk-...", redacted)
        self.assertIn("DEEPSEEK_API_KEY=[redacted]", redacted)
        self.assertIn("OPENAI_API_KEY: [redacted]", redacted)
        self.assertIn("Bearer [redacted]", redacted)
        self.assertNotIn("sk-test-secret", redacted)
        self.assertNotIn("sk-openai-secret", redacted)
        self.assertNotIn("test-deepseek-key", redacted)

    def test_gui_user_facing_text_summarizes_remote_rate_limit_errors(self) -> None:
        raw = (
            'Remote request failed with HTTP 429: {"message":"Too Many Requests",'
            '"error":"Insufficient budget","retryAfter":19960}'
        )
        redacted = sanitize_user_facing_text(raw)
        self.assertIn("远程源限流或预算不足", redacted)
        self.assertNotIn("Too Many Requests", redacted)
        self.assertNotIn("Insufficient budget", redacted)

    def test_gui_job_exception_error_is_public_safe(self) -> None:
        import time

        state = GuiState()

        def target():
            raise RuntimeError(
                "failed with DEEPSEEK_API_KEY=sk-runtime-secret-123456 "
                "Bearer test-runtime-token-123456 at /Users/example/project/secret.py"
            )

        state.add_job("failing exception", target, kind="cleanup", ui_scope="draft")
        for _ in range(50):
            snapshot = state.snapshot()[0]
            if snapshot["status"] == "failed":
                break
            time.sleep(0.01)

        snapshot = state.snapshot()[0]
        serialized = json.dumps(snapshot, ensure_ascii=False)
        self.assertEqual(snapshot["status"], "failed")
        self.assertIn("[redacted]", serialized)
        self.assertIn("[local path]", serialized)
        self.assertNotIn("sk-runtime-secret", serialized)
        self.assertNotIn("test-runtime-token", serialized)
        self.assertNotIn("/Users/example", serialized)

    def test_gui_state_reuses_running_jobs_by_scope_and_kind(self) -> None:
        import threading
        import time

        state = GuiState()
        release = threading.Event()

        def target():
            release.wait(2)
            return {"stdout": "done"}

        job = state.add_job("ideate test", target, kind="ideate", ui_scope="draft:abc")
        for _ in range(50):
            if state.running_job(kind="ideate", ui_scope="draft:abc"):
                break
            time.sleep(0.01)

        existing = state.running_job(kind="ideate", ui_scope="draft:abc")
        self.assertIsNotNone(existing)
        assert existing is not None
        self.assertEqual(existing["id"], job["id"])
        self.assertIsNone(state.running_job(kind="review", ui_scope="draft:abc"))
        self.assertIsNone(state.running_job(kind="ideate", ui_scope="draft:other"))

        release.set()
        for _ in range(50):
            if not state.running_job(kind="ideate", ui_scope="draft:abc"):
                break
            time.sleep(0.01)
        self.assertIsNone(state.running_job(kind="ideate", ui_scope="draft:abc"))

    def test_gui_state_atomically_adds_job_unless_running(self) -> None:
        import threading
        import time

        state = GuiState()
        release = threading.Event()
        target_calls = []
        request_count = 8

        def target():
            target_calls.append("called")
            release.wait(2)
            return {"stdout": "done"}

        barrier = threading.Barrier(request_count)
        results = []
        result_lock = threading.Lock()

        def add_once():
            barrier.wait(2)
            result = state.add_job_unless_running(
                "ideate test",
                target,
                kind="ideate",
                ui_scope="draft:atomic",
                dedupe=lambda job: job.get("kind") == "ideate" and job.get("ui_scope") == "draft:atomic",
            )
            with result_lock:
                results.append(result)

        workers = [threading.Thread(target=add_once) for _ in range(request_count)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=5)

        self.assertEqual(len(results), request_count)
        job_ids = {job["id"] for job, _ in results}
        self.assertEqual(len(job_ids), 1)
        self.assertEqual(sum(1 for _, created in results if created), 1)
        for _ in range(50):
            if target_calls:
                break
            time.sleep(0.01)
        self.assertEqual(len(target_calls), 1)

        release.set()

    def test_gui_state_finds_running_jobs_for_session_before_delete(self) -> None:
        import threading
        import time

        state = GuiState()
        release = threading.Event()

        def target():
            release.wait(2)
            return {"stdout": "done"}

        state.add_job("session step", target, kind="step", session_id=7, ui_scope="session:7")
        state.add_job("draft ideate", target, kind="ideate", ui_scope="draft:other")
        for _ in range(50):
            if state.running_jobs_for_session(7):
                break
            time.sleep(0.01)

        running = state.running_jobs_for_session(7)
        self.assertEqual(len(running), 1)
        self.assertEqual(running[0]["kind"], "step")
        self.assertEqual(running[0]["session_id"], 7)
        self.assertEqual(state.running_jobs_for_session(99), [])

        release.set()

    def test_gui_state_marks_session_delete_in_progress_without_running_jobs(self) -> None:
        import threading
        import time

        state = GuiState()
        release = threading.Event()

        def target():
            release.wait(2)
            return {"stdout": "done"}

        state.add_job("session step", target, kind="step", session_id=7, ui_scope="session:7")
        for _ in range(50):
            if state.running_jobs_for_session(7):
                break
            time.sleep(0.01)

        delete_started, running_jobs = state.begin_session_delete(7)
        self.assertFalse(delete_started)
        self.assertEqual(running_jobs[0]["kind"], "step")
        self.assertFalse(state.is_session_deleting(7))

        release.set()
        for _ in range(50):
            if not state.running_jobs_for_session(7):
                break
            time.sleep(0.01)

        delete_started, running_jobs = state.begin_session_delete(7)
        self.assertTrue(delete_started)
        self.assertEqual(running_jobs, [])
        self.assertTrue(state.is_session_deleting(7))
        duplicate_started, duplicate_running_jobs = state.begin_session_delete(7)
        self.assertFalse(duplicate_started)
        self.assertEqual(duplicate_running_jobs, [])
        self.assertTrue(state.is_session_deleting(7))
        state.finish_session_delete(7)
        self.assertFalse(state.is_session_deleting(7))

    def test_gui_state_finds_running_conversation_job_for_session(self) -> None:
        import threading
        import time

        self.assertEqual(GUI_CONVERSATION_JOB_KINDS, {"ideate", "step", "review", "review-loop"})
        state = GuiState()
        release = threading.Event()

        def target():
            release.wait(2)
            return {"stdout": "done"}

        state.add_job("quality audit", target, kind="quality-enrich", session_id=7, ui_scope="session:7")
        state.add_job("session review", target, kind="review", session_id=7, ui_scope="session:7")
        state.add_job("other session step", target, kind="step", session_id=8, ui_scope="session:8")
        for _ in range(50):
            if state.running_conversation_job_for_session(7):
                break
            time.sleep(0.01)

        running = state.running_conversation_job_for_session(7)
        self.assertIsNotNone(running)
        assert running is not None
        self.assertEqual(running["kind"], "review")
        self.assertEqual(running["session_id"], 7)
        self.assertEqual(state.running_conversation_job_for_session(8)["kind"], "step")
        self.assertIsNone(state.running_conversation_job_for_session(99))

        release.set()

    def test_gui_review_rethink_exit_code_is_completed_job(self) -> None:
        import time

        state = GuiState()
        state.add_job(
            "session review",
            lambda: {
                "exit_code": 2,
                "stdout": "Reviewer decision: rethink_required\nSummary: Reviewer blocks this idea.\nJSON: outputs/reviews/review-x.json\n",
                "stderr": "",
            },
            kind="review",
            session_id=7,
            ui_scope="session:7",
        )
        for _ in range(50):
            snapshot = state.snapshot()[0]
            if snapshot["status"] != "running":
                break
            time.sleep(0.01)
        snapshot = state.snapshot()[0]
        self.assertEqual(snapshot["status"], "completed")
        self.assertEqual(snapshot["result"], {"exit_code": 2})
        self.assertEqual(snapshot["error"], "")

    def test_gui_state_finds_running_evidence_job_for_scope(self) -> None:
        import threading
        import time

        self.assertEqual(GUI_EVIDENCE_JOB_KINDS, {"gold-build", "quality-enrich", "cleanup"})
        state = GuiState()
        release = threading.Event()

        def target():
            release.wait(2)
            return {"stdout": "done"}

        state.add_job("session step", target, kind="step", ui_scope="draft:evidence")
        evidence_job = state.add_job("quality audit", target, kind="quality-enrich", ui_scope="draft:evidence")
        state.add_job("other cleanup", target, kind="cleanup", ui_scope="draft:other")
        for _ in range(50):
            if state.running_evidence_job_for_scope("draft:evidence"):
                break
            time.sleep(0.01)

        running = state.running_evidence_job_for_scope("draft:evidence")
        self.assertIsNotNone(running)
        assert running is not None
        self.assertEqual(running["id"], evidence_job["id"])
        self.assertEqual(running["kind"], "quality-enrich")
        self.assertIsNone(state.running_evidence_job_for_scope("draft:missing"))

        release.set()

    def test_gui_running_job_snapshot_exposes_timestamps_for_silence_feedback(self) -> None:
        import threading
        import time

        state = GuiState()
        release = threading.Event()

        def target():
            release.wait(2)
            return {"stdout": "done"}

        state.add_job("slow model call", target, kind="step", ui_scope="session:1")
        for _ in range(50):
            snapshot = state.snapshot()[0]
            if snapshot["status"] == "running":
                break
            time.sleep(0.01)

        snapshot = state.snapshot()[0]
        self.assertEqual(snapshot["status"], "running")
        self.assertIn("created_at", snapshot)
        self.assertIn("updated_at", snapshot)
        self.assertIn("last_output_at", snapshot)
        self.assertIsNone(snapshot["last_output_at"])
        self.assertGreater(float(snapshot["created_at"]), 0)
        self.assertGreaterEqual(float(snapshot["updated_at"]), float(snapshot["created_at"]))
        release.set()

    def test_gui_job_snapshot_keeps_running_jobs_and_gold_build_groups(self) -> None:
        state = GuiState()
        base_job = {
            "id": 1,
            "name": "old running ideate",
            "kind": "ideate",
            "session_id": None,
            "ui_scope": "draft:old",
            "group_id": "",
            "created_session_id": None,
            "status": "running",
            "result": None,
            "error": "",
            "live_stdout": "",
            "live_stderr": "",
            "last_line": "",
        }
        state.jobs = [dict(base_job)]
        for idx in range(2, 17):
            state.jobs.insert(
                0,
                {
                    **base_job,
                    "id": idx,
                    "name": f"completed {idx}",
                    "kind": "cleanup",
                    "ui_scope": f"draft:{idx}",
                    "status": "completed",
                },
            )
        snapshot = state.snapshot(limit=12)
        self.assertIn(1, {job["id"] for job in snapshot})
        self.assertGreaterEqual(len(snapshot), 12)

        group_id = "gold-build:draft:gold:1"
        state.jobs = [
            {**base_job, "id": 9, "name": "recent grouped source", "kind": "gold-build", "ui_scope": "draft:gold", "group_id": group_id, "status": "completed"},
            {**base_job, "id": 8, "name": "recent cleanup", "kind": "cleanup", "ui_scope": "draft:8", "group_id": "", "status": "completed"},
        ]
        for idx in range(10, 16):
            state.jobs.append({**base_job, "id": idx, "name": f"older cleanup {idx}", "kind": "cleanup", "ui_scope": f"draft:{idx}", "group_id": "", "status": "completed"})
        state.jobs.extend(
            [
                {**base_job, "id": 1, "name": "older grouped source 1", "kind": "gold-build", "ui_scope": "draft:gold", "group_id": group_id, "status": "completed"},
                {**base_job, "id": 2, "name": "older grouped source 2", "kind": "gold-build", "ui_scope": "draft:gold", "group_id": group_id, "status": "completed"},
            ]
        )
        snapshot = state.snapshot(limit=2)
        grouped_ids = {job["id"] for job in snapshot if job.get("group_id") == "gold-build:draft:gold:1"}
        self.assertEqual(grouped_ids, {1, 2, 9})

    def test_gui_job_snapshot_prioritizes_completed_conversation_jobs(self) -> None:
        state = GuiState()
        base_job = {
            "id": 1,
            "name": "completed cleanup",
            "kind": "cleanup",
            "session_id": None,
            "ui_scope": "draft:old",
            "group_id": "",
            "created_session_id": None,
            "status": "completed",
            "result": {"exit_code": 0, "stdout": "done", "stderr": ""},
            "error": "",
            "live_stdout": "",
            "live_stderr": "",
            "last_line": "",
        }
        state.jobs = [
            {
                **base_job,
                "id": 40,
                "name": "completed ideate without session",
                "kind": "ideate",
                "ui_scope": "draft:important",
                "result": {"exit_code": 0, "stdout": "Strict evidence is not ready yet.", "stderr": ""},
            }
        ]
        for idx in range(1, 30):
            state.jobs.insert(
                0,
                {
                    **base_job,
                    "id": idx,
                    "name": f"newer cleanup {idx}",
                    "ui_scope": f"draft:{idx}",
                },
            )

        snapshot = state.snapshot(limit=12)
        ids = [int(job["id"]) for job in snapshot]
        self.assertIn(40, ids)
        ideate = next(job for job in snapshot if int(job["id"]) == 40)
        self.assertEqual(ideate["result_summary"]["type"], "idea_generation_not_ready")
        self.assertEqual(ideate["result_summary"]["failure"]["primary_reason"], "gold_evidence_not_ready")

    def test_gui_job_snapshot_compacts_large_command_outputs(self) -> None:
        import time

        state = GuiState()
        huge_stdout = "x" * (GUI_JOB_RESULT_TAIL_CHARS + 100) + "stdout-tail"
        huge_stderr = "y" * (GUI_JOB_RESULT_TAIL_CHARS + 100) + "stderr-tail"
        state.add_job(
            "large output command",
            lambda: {"exit_code": 0, "stdout": huge_stdout, "stderr": huge_stderr},
            kind="cleanup",
            ui_scope="draft",
        )

        for _ in range(50):
            snapshot = state.snapshot()[0]
            if snapshot["status"] == "completed":
                break
            time.sleep(0.01)

        snapshot = state.snapshot()[0]
        self.assertEqual(snapshot["status"], "completed")
        self.assertLessEqual(len(snapshot["result"]["stdout"]), GUI_JOB_RESULT_TAIL_CHARS)
        self.assertLessEqual(len(snapshot["result"]["stderr"]), GUI_JOB_RESULT_TAIL_CHARS)
        self.assertTrue(snapshot["result"]["stdout"].endswith("stdout-tail"))
        self.assertTrue(snapshot["result"]["stderr"].endswith("stderr-tail"))

    def test_gui_state_prunes_old_completed_jobs_but_keeps_running_jobs(self) -> None:
        state = GuiState()
        base_job = {
            "id": 1,
            "name": "base",
            "kind": "cleanup",
            "session_id": None,
            "ui_scope": "draft",
            "group_id": "",
            "created_session_id": None,
            "status": "completed",
            "result": None,
            "error": "",
            "live_stdout": "",
            "live_stderr": "",
            "last_line": "",
            "created_at": 1.0,
            "updated_at": 1.0,
            "last_output_at": None,
            "completed_at": 1.0,
        }
        state.jobs = [
            {**base_job, "id": idx, "name": f"completed {idx}", "updated_at": float(idx)}
            for idx in range(1, GUI_JOB_MAX_STORED + 12)
        ]
        state.jobs.append({**base_job, "id": 9999, "name": "still running", "status": "running", "updated_at": 0.0})

        state.add_job("new completed", lambda: {"stdout": "done"}, kind="cleanup", ui_scope="draft:new")

        stored_ids = {int(job["id"]) for job in state.jobs}
        self.assertIn(9999, stored_ids)
        self.assertLessEqual(len(state.jobs), GUI_JOB_MAX_STORED)
        self.assertNotIn(1, stored_ids)

    def test_gui_job_marks_nonzero_command_exit_as_failed(self) -> None:
        import time

        state = GuiState()
        state.add_job(
            "failing command",
            lambda: {"exit_code": 1, "stdout": "partial stdout", "stderr": "explicit failure"},
            kind="ideate",
            ui_scope="draft",
        )

        for _ in range(50):
            snapshot = state.snapshot()[0]
            if snapshot["status"] == "failed":
                break
            time.sleep(0.01)

        snapshot = state.snapshot()[0]
        self.assertEqual(snapshot["status"], "failed")
        self.assertEqual(snapshot["result"]["exit_code"], 1)
        self.assertEqual(snapshot["error"], "")
        self.assertEqual(snapshot["result_summary"]["type"], "idea_generation_not_ready")
        self.assertEqual(snapshot["result_summary"]["failure"]["primary_reason"], "idea_generation_not_ready")
        self.assertIn("没有写入新对话", snapshot["result_summary"]["failure"]["reason"])
        serialized = json.dumps(snapshot, ensure_ascii=False)
        self.assertNotIn("explicit failure", serialized)
        self.assertNotIn("partial stdout", serialized)

    def test_gui_http_response_ignores_client_disconnect(self) -> None:
        class DisconnectingWriter:
            def write(self, body):
                raise BrokenPipeError("client closed connection")

        class FakeHandler:
            def __init__(self):
                self.headers = []
                self.wfile = DisconnectingWriter()

            def send_response(self, status):
                self.headers.append(("status", status))

            def send_header(self, key, value):
                self.headers.append((key, value))

            def end_headers(self):
                self.headers.append(("end", "headers"))

        handler = FakeHandler()
        self.assertFalse(
            write_http_response(handler, status=200, content_type="application/json", body=b"{}")
        )
        self.assertIn(("Content-Length", "2"), handler.headers)

    def test_gui_error_response_keeps_legacy_error_with_stable_contract(self) -> None:
        body = gui_error_response("session not found", status=404, code="session_not_found")
        self.assertFalse(body["ok"])
        self.assertEqual(body["error"], "session not found")
        self.assertEqual(body["code"], "session_not_found")
        self.assertEqual(body["status"], 404)
        self.assertFalse(body["retryable"])
        self.assertIn("重新选择", body["next_action"])
        self.assertEqual(gui_error_response("bad")["code"], "validation_error")
        self.assertEqual(gui_error_response("gone", status=404)["code"], "not_found")
        self.assertEqual(gui_error_response("busy", status=409)["code"], "conflict")
        self.assertFalse(gui_error_response("bad")["retryable"])
        self.assertTrue(gui_error_response("busy", status=409)["retryable"])
        self.assertIn("修正输入", gui_error_response("bad")["next_action"])
        self.assertIn("等待当前任务完成", gui_error_response("busy", status=409)["next_action"])
        unsafe = gui_error_response(
            "failed at /Users/me/project with DEEPSEEK_API_KEY=sk-error-secret-123456 "
            "Authorization: Bearer test-error-token-123456",
            status=500,
        )
        serialized = json.dumps(unsafe, ensure_ascii=False)
        self.assertEqual(unsafe["status"], 500)
        self.assertEqual(unsafe["code"], "request_error")
        self.assertTrue(unsafe["retryable"])
        self.assertIn("[local path]", serialized)
        self.assertIn("[redacted]", serialized)
        self.assertNotIn("/Users/me", serialized)
        self.assertNotIn("sk-error-secret", serialized)
        self.assertNotIn("test-error-token", serialized)

    def test_gui_state_project_metadata_does_not_expose_local_paths_or_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                (cwd / ".env").write_text(
                    "RA_LLM_PROVIDER=deepseek\nRA_LLM_MODEL=deepseek-chat\nDEEPSEEK_API_KEY=secret-for-test\n",
                    encoding="utf-8",
                )
                env_override = {
                    "RA_LLM_PROVIDER": "deepseek",
                    "RA_LLM_MODEL": "deepseek-chat",
                    "DEEPSEEK_API_KEY": "secret-for-test",
                    "OPENAI_API_KEY": "",
                }
                with patch.dict(os.environ, env_override, clear=False):
                    snapshot = build_gui_snapshot(cwd, GuiState(), draft_mode=True)
                project = snapshot["project"]
                self.assertEqual(project["provider"], "deepseek")
                self.assertEqual(project["model"], "deepseek-chat")
                self.assertTrue(project["api_key_configured"])
                self.assertNotIn("root", project)
                self.assertNotIn("db", project)
                self.assertNotIn("api_key", project)
                self.assertNotIn("secret-for-test", json.dumps(snapshot, ensure_ascii=False))
            finally:
                os.chdir(old_cwd)

    def test_gui_state_includes_public_health_summary(self) -> None:
        state = GuiState()
        snapshot = build_gui_snapshot(Path.cwd(), state, draft_mode=True)
        health = snapshot["health"]
        self.assertIn(health["evidence_status"], {"missing_provider_key", "needs_evidence", "ready", "building", "evidence_failed"})
        self.assertIsInstance(health["project_configured"], bool)
        self.assertIsInstance(health["evidence_ready"], bool)
        self.assertIn("missing_requirements", health)
        self.assertIn("failed_evidence_jobs", health)
        self.assertIn("latest_failed_evidence_job", health)
        self.assertNotIn("root", health)
        self.assertNotIn("db", health)
        self.assertNotIn("api_key", health)
        self.assertNotIn("secret", json.dumps(health, ensure_ascii=False).lower())

    def test_gui_state_run_cards_do_not_expose_manifest_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                manifest_path = cwd / "outputs" / "secret-run" / "manifest.json"
                manifest_path.parent.mkdir(parents=True, exist_ok=True)
                manifest_path.write_text("{}", encoding="utf-8")
                add_run(cwd / "data" / "research_alpha.db", "secret query", "partial", manifest_path)

                snapshot = build_gui_snapshot(cwd, GuiState(), draft_mode=True)
                self.assertEqual(len(snapshot["runs"]), 1)
                run = snapshot["runs"][0]
                self.assertEqual(run["query"], "secret query")
                self.assertTrue(run["manifest_available"])
                self.assertNotIn("manifest_path", run)
                serialized = json.dumps(snapshot["runs"], ensure_ascii=False)
                self.assertNotIn(str(cwd), serialized)
                self.assertNotIn("secret-run", serialized)
            finally:
                os.chdir(old_cwd)

    def test_gui_state_idea_and_pattern_cards_are_public_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                from research_alpha.db import add_candidate_idea, upsert_pattern_card

                add_candidate_idea(
                    cwd / "data" / "research_alpha.db",
                    query="research agents",
                    title="Novel Agent Evaluation",
                    content_json=json.dumps(
                        {
                            "idea_title": "Novel Agent Evaluation",
                            "internal_prompt_trace": "secret chain that should not be in public state",
                            "prior_art_gate": {
                                "decision": "pass",
                                "summary": "No close prior-art coverage in stored evidence.",
                                "why_not_done_yet": ["Recent systems lacked the required long-horizon traces."],
                            },
                            "evidence_grounded_score": {"total": 7.5},
                        }
                    ),
                )
                upsert_pattern_card(
                    cwd / "data" / "research_alpha.db",
                    "hidden-assumption-reversal",
                    json.dumps({"large_internal_payload": "secret pattern source payload"}),
                )

                snapshot = build_gui_snapshot(cwd, GuiState(), draft_mode=True)
                idea = snapshot["ideas"][0]
                self.assertEqual(idea["title"], "Novel Agent Evaluation")
                self.assertEqual(idea["prior_art_gate_decision"], "pass")
                self.assertEqual(idea["evidence_score"], 7.5)
                self.assertNotIn("content_json", idea)
                self.assertNotIn("internal_prompt_trace", json.dumps(idea, ensure_ascii=False))

                pattern = snapshot["patterns"][0]
                self.assertEqual(pattern["pattern_key"], "hidden-assumption-reversal")
                self.assertNotIn("content_json", pattern)
                serialized = json.dumps(snapshot["patterns"], ensure_ascii=False)
                self.assertNotIn("large_internal_payload", serialized)
                self.assertNotIn("secret pattern source payload", serialized)
            finally:
                os.chdir(old_cwd)

    def test_gui_state_paper_cards_are_public_summaries_without_internal_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                db_path = cwd / "data" / "research_alpha.db"
                paper_id = add_paper(
                    db_path,
                    "Private Scored Paper",
                    "ICLR",
                    2026,
                    "A very long abstract with internal route notes. " * 30,
                    source_kind="gold_openreview",
                    external_ref=str(cwd / "secret" / "paper.pdf"),
                    award="best_paper",
                    citation_count=99,
                )
                with connect(db_path) as conn:
                    conn.execute(
                        "UPDATE papers SET paper_weight=?, score_notes=?, limitations_json=? WHERE id=?",
                        (
                            5.0,
                            json.dumps({"internal_score_trace": "secret scoring details"}),
                            json.dumps({"private_limitations": ["secret limitation trace"]}),
                            paper_id,
                        ),
                    )

                snapshot = build_gui_snapshot(cwd, GuiState(), draft_mode=True)
                paper = next(item for item in snapshot["top_papers"] if item["title"] == "Private Scored Paper")
                self.assertEqual(paper["external_ref"], "")
                self.assertEqual(paper["paper_weight"], 5.0)
                self.assertIn("abstract_excerpt", paper)
                self.assertLessEqual(len(paper["abstract_excerpt"]), 360)
                self.assertNotIn("score_notes", paper)
                self.assertNotIn("limitations_json", paper)
                self.assertNotIn("abstract", paper)
                serialized = json.dumps(snapshot["top_papers"], ensure_ascii=False)
                self.assertNotIn("internal_score_trace", serialized)
                self.assertNotIn("secret limitation trace", serialized)
                self.assertNotIn(str(cwd), serialized)
            finally:
                os.chdir(old_cwd)

    def test_gui_health_summary_tracks_readiness_and_running_jobs(self) -> None:
        not_ready = build_gui_health_snapshot(
            project_configured=True,
            readiness={"high_weight_papers": 1, "genome_cards": 0, "pattern_cards": 0},
            counts={"idea_sessions": 2},
            jobs=[
                {"status": "running", "kind": "gold-build"},
                {"status": "running", "kind": "step"},
                {"status": "completed", "kind": "cleanup"},
            ],
        )
        self.assertFalse(not_ready["evidence_ready"])
        self.assertEqual(not_ready["evidence_status"], "building")
        self.assertEqual(not_ready["running_jobs"], 2)
        self.assertEqual(not_ready["evidence_jobs"], 1)
        self.assertEqual(not_ready["blocking_jobs"], 1)
        self.assertEqual(not_ready["session_count"], 2)
        self.assertEqual(
            {item["key"] for item in not_ready["missing_requirements"]},
            {"high_weight_papers", "genome_cards", "pattern_cards"},
        )

        ready = build_gui_health_snapshot(
            project_configured=True,
            readiness={"high_weight_papers": 3, "genome_cards": 2, "pattern_cards": 1},
            counts={"idea_sessions": 0},
            jobs=[],
        )
        self.assertTrue(ready["evidence_ready"])
        self.assertEqual(ready["evidence_status"], "ready")
        self.assertEqual(ready["missing_requirements"], [])

        missing_key = build_gui_health_snapshot(
            project_configured=False,
            readiness={"high_weight_papers": 3, "genome_cards": 2, "pattern_cards": 1},
            counts={},
            jobs=[],
        )
        self.assertEqual(missing_key["evidence_status"], "missing_provider_key")

        failed = build_gui_health_snapshot(
            project_configured=True,
            readiness={"high_weight_papers": 1, "genome_cards": 0, "pattern_cards": 0},
            counts={},
            jobs=[
                {
                    "id": 12,
                    "status": "failed",
                    "kind": "gold-build",
                    "name": "ICLR source /Users/me/project DEEPSEEK_API_KEY=sk-name-secret-123456",
                    "ui_scope": "draft:/Users/me/project:sk-scope-secret-123456",
                    "error": "failed at /Users/me/project with DEEPSEEK_API_KEY=sk-secret-123456",
                    "completed_at": 20.0,
                },
                {"id": 9, "status": "failed", "kind": "step", "name": "old step", "completed_at": 30.0},
            ],
        )
        self.assertFalse(failed["evidence_ready"])
        self.assertEqual(failed["evidence_status"], "evidence_failed")
        self.assertEqual(failed["failed_evidence_jobs"], 1)
        self.assertEqual(failed["latest_failed_evidence_job"]["id"], 12)
        self.assertEqual(failed["latest_failed_evidence_job"]["kind"], "gold-build")
        self.assertIn("failure_summary", failed["latest_failed_evidence_job"])
        serialized_failure = json.dumps(failed, ensure_ascii=False)
        self.assertIn("[local path]", serialized_failure)
        self.assertIn("[redacted]", serialized_failure)
        self.assertNotIn("/Users/me", serialized_failure)
        self.assertNotIn("sk-secret", serialized_failure)
        self.assertNotIn("sk-name-secret", serialized_failure)
        self.assertNotIn("sk-scope-secret", serialized_failure)

        structured_failed = build_gui_health_snapshot(
            project_configured=True,
            readiness={"high_weight_papers": 1, "genome_cards": 0, "pattern_cards": 0},
            counts={},
            jobs=[
                {
                    "id": 14,
                    "status": "failed",
                    "kind": "gold-build",
                    "name": "ICLR 2026",
                    "result": {
                        "stdout": (
                            'Gold-build summary JSON: {"fetched":0,"candidate_imported":2,"excellent_imported":0,'
                            '"failures":1,'
                            '"source_failures":[{"year":2026,"venue":"ICLR","remote":"openreview","error":"timeout at /Users/me/secret","recommended_action":"retry with smaller year range"}],'
                            '"source_empty_results":[{"year":2025,"venue":"NeurIPS","remote":"awards","recommended_action":"switch official source"}],'
                            '"next_actions":[{"label":"调整检索","reason":"source failed","command":"ra gold-build --api-key sk-test-secret"}]}\n'
                        ),
                        "stderr": "",
                        "exit_code": 1,
                    },
                    "error": "failed",
                    "completed_at": 40.0,
                }
            ],
        )
        summary = structured_failed["latest_failed_evidence_job"]["failure_summary"]
        self.assertEqual(summary["reason_code"], "source_failure")
        self.assertEqual(summary["candidate_imported"], 2)
        self.assertEqual(summary["excellent_imported"], 0)
        self.assertEqual(summary["failures"], 1)
        self.assertEqual(summary["source_failures"][0]["venue"], "ICLR")
        self.assertIn("[local path]", summary["source_failures"][0]["error"])
        self.assertEqual(summary["source_empty_results"][0]["remote"], "awards")
        structured_serialized = json.dumps(structured_failed, ensure_ascii=False)
        self.assertNotIn("/Users/me", structured_serialized)
        self.assertNotIn("sk-test-secret", structured_serialized)
        self.assertIn("[redacted]", structured_serialized)

        recovered_but_not_ready = build_gui_health_snapshot(
            project_configured=True,
            readiness={"high_weight_papers": 1, "genome_cards": 0, "pattern_cards": 0},
            counts={},
            jobs=[
                {"id": 13, "status": "completed", "kind": "quality-enrich", "name": "retry succeeded", "completed_at": 30.0},
                {"id": 12, "status": "failed", "kind": "gold-build", "name": "older failure", "completed_at": 20.0},
            ],
        )
        self.assertEqual(recovered_but_not_ready["evidence_status"], "needs_evidence")
        self.assertEqual(recovered_but_not_ready["failed_evidence_jobs"], 1)
        self.assertIsNone(recovered_but_not_ready["latest_failed_evidence_job"])

    def test_strict_readiness_rejects_raw_counts_without_grounding(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                db_path = cwd / "data" / "research_alpha.db"
                for idx, (title, award, weight) in enumerate(
                    [
                        ("Raw Best Paper", "best_paper", 5.0),
                        ("Raw Oral Paper", "oral", 2.0),
                        ("Raw Outstanding Paper", "outstanding_paper", 4.0),
                    ]
                ):
                    paper_id = add_paper(
                        db_path,
                        title,
                        "ICLR",
                        2026,
                        "Benchmark assumptions hide agent failures and brittle evaluation.",
                        source_kind="gold_openreview",
                        award=award,
                    )
                    with connect(db_path) as conn:
                        conn.execute(
                            "UPDATE papers SET paper_weight=?, score_notes=? WHERE id=?",
                            (weight, json.dumps({award: weight}), paper_id),
                        )
                    upsert_idea_card(
                        db_path,
                        paper_id,
                        "strict_logic_line",
                        json.dumps({"paper_summary": "Looks strict but has no grounded logic line."}),
                    )
                upsert_pattern_card(
                    db_path,
                    "raw-pattern",
                    json.dumps(
                        {
                            "pattern_name": "Raw Pattern",
                            "evidence_level": "logic_line_aggregation",
                            "logic_line_pattern": {
                                "source_logic": "Benchmark assumptions hide agent failures.",
                                "transfer_rule": "Transfer to evaluation.",
                                "why_it_generalizes": "It recurs.",
                                "failure_boundary": "Fails without support.",
                            },
                        },
                    ),
                )

                readiness = compute_strict_evidence_readiness(db_path, paper_limit=12)
                self.assertEqual(readiness["counts"]["high_weight_papers"], 3)
                self.assertEqual(readiness["counts"]["genome_cards"], 3)
                self.assertEqual(readiness["counts"]["pattern_cards"], 1)
                self.assertEqual(readiness["counts"]["strict_genome_cards"], 0)
                self.assertEqual(readiness["counts"]["strict_pattern_cards"], 0)
                self.assertEqual(readiness["status"], "partially_ready")
                self.assertIn("Generate at least 2 non-extractive Idea Genome Cards", readiness["missing_requirements"][0])
            finally:
                os.chdir(old_cwd)

    def test_gui_snapshot_uses_strict_evidence_counts_not_raw_card_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                db_path = cwd / "data" / "research_alpha.db"
                seed_strict_generation_evidence(db_path)
                snapshot = build_gui_snapshot(cwd, GuiState(), draft_mode=True)
                self.assertTrue(snapshot["health"]["evidence_ready"])
                self.assertEqual(snapshot["readiness"]["evidence_status"], snapshot["health"]["evidence_status"])
                self.assertEqual(snapshot["readiness"]["strict_evidence_status"], "ready_for_strict_generation")
                self.assertIn("strict_evidence_status", snapshot["readiness"])
                self.assertEqual(snapshot["readiness"]["genome_cards"], 2)
                self.assertEqual(snapshot["readiness"]["pattern_cards"], 1)

                with connect(db_path) as conn:
                    conn.execute(
                        "UPDATE pattern_cards SET content_json=?",
                        (json.dumps({"pattern_name": "Broken Pattern", "evidence_level": "logic_line_aggregation"}),),
                    )
                broken = build_gui_snapshot(cwd, GuiState(), draft_mode=True)
                self.assertFalse(broken["health"]["evidence_ready"])
                self.assertEqual(broken["readiness"]["evidence_status"], broken["health"]["evidence_status"])
                self.assertIn(broken["readiness"]["evidence_status"], {"missing_provider_key", "needs_evidence"})
                self.assertIn("strict_evidence_status", broken["readiness"])
                self.assertEqual(broken["readiness"]["pattern_cards"], 0)
                self.assertEqual(broken["readiness"]["raw_pattern_cards"], 1)
                self.assertEqual(
                    {item["key"] for item in broken["health"]["missing_requirements"]},
                    {"pattern_cards"},
                )
            finally:
                os.chdir(old_cwd)

    def test_gui_session_api_resolution_rejects_missing_sessions_without_stale_active_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                main(["init"])
                self.write_gui_test_llm_env(cwd)
                from research_alpha.db import create_idea_session

                db_path = cwd / "data" / "research_alpha.db"
                state = GuiState()

                missing_id = resolve_gui_session(db_path, state, 0)
                self.assertFalse(missing_id["ok"])
                self.assertEqual(missing_id["status"], 400)
                self.assertIsNone(state.active_session_id)

                state.active_session_id = 999
                missing_session = resolve_gui_session(db_path, state, 999)
                self.assertFalse(missing_session["ok"])
                self.assertEqual(missing_session["status"], 404)
                self.assertIsNone(state.active_session_id)

                session_id = create_idea_session(db_path, "agent eval", "Initial idea")
                resolved = resolve_gui_session(db_path, state, session_id)
                self.assertTrue(resolved["ok"])
                self.assertEqual(resolved["session_id"], session_id)
                self.assertEqual(state.active_session_id, session_id)

                scoped = resolve_gui_session(db_path, state, session_id, ui_scope=f"session:{session_id}")
                self.assertTrue(scoped["ok"])

                mismatched_scope = resolve_gui_session(db_path, state, session_id, ui_scope="session:999")
                self.assertFalse(mismatched_scope["ok"])
                self.assertEqual(mismatched_scope["status"], 409)
                self.assertIn("session scope mismatch", mismatched_scope["error"])

                delete_started, running_jobs = state.begin_session_delete(session_id)
                self.assertTrue(delete_started)
                self.assertEqual(running_jobs, [])
                deleting = resolve_gui_session(db_path, state, session_id, ui_scope=f"session:{session_id}")
                self.assertFalse(deleting["ok"])
                self.assertEqual(deleting["status"], 409)
                self.assertEqual(deleting["code"], "session_delete_in_progress")
                state.finish_session_delete(session_id)
            finally:
                os.chdir(old_cwd)

    def test_gui_numeric_request_parser_returns_clear_errors(self) -> None:
        self.assertEqual(parse_gui_int("", name="idea 数", default=3, minimum=1, maximum=8), 3)
        self.assertEqual(parse_gui_int(" 6 ", name="idea 数", default=3, minimum=1, maximum=8), 6)

        with self.assertRaisesRegex(ValueError, "idea 数必须是整数"):
            parse_gui_int("many", name="idea 数", default=3, minimum=1, maximum=8)
        with self.assertRaisesRegex(ValueError, "idea 数必须是整数"):
            parse_gui_int(True, name="idea 数", default=3, minimum=1, maximum=8)
        with self.assertRaisesRegex(ValueError, "idea 数不能小于 1"):
            parse_gui_int("0", name="idea 数", default=3, minimum=1, maximum=8)
        with self.assertRaisesRegex(ValueError, "idea 数不能大于 8"):
            parse_gui_int("9", name="idea 数", default=3, minimum=1, maximum=8)

    def test_gui_boolean_request_parser_returns_clear_errors(self) -> None:
        self.assertTrue(parse_gui_bool("", name="是否执行清理", default=True))
        self.assertFalse(parse_gui_bool(None, name="是否执行清理", default=False))
        self.assertTrue(parse_gui_bool(True, name="是否执行清理"))
        self.assertFalse(parse_gui_bool(False, name="是否执行清理"))
        self.assertTrue(parse_gui_bool(" true ", name="是否执行清理"))
        self.assertTrue(parse_gui_bool("1", name="是否执行清理"))
        self.assertTrue(parse_gui_bool("on", name="是否执行清理"))
        self.assertFalse(parse_gui_bool("false", name="是否执行清理"))
        self.assertFalse(parse_gui_bool("0", name="是否执行清理"))
        self.assertFalse(parse_gui_bool("off", name="是否执行清理"))

        with self.assertRaisesRegex(ValueError, "是否执行清理必须是布尔值"):
            parse_gui_bool("sometimes", name="是否执行清理")
        with self.assertRaisesRegex(ValueError, "是否执行清理必须是布尔值"):
            parse_gui_bool(2, name="是否执行清理")

    def test_gui_text_request_parser_returns_clear_errors(self) -> None:
        self.assertEqual(
            parse_gui_text("  research agents  ", name="direction", required=True, max_chars=20),
            "research agents",
        )
        self.assertEqual(parse_gui_text(None, name="Gold query", max_chars=10), "")

        with self.assertRaisesRegex(ValueError, "direction is required"):
            parse_gui_text("", name="direction", required=True, max_chars=20, missing_message="direction is required")
        with self.assertRaisesRegex(ValueError, "instruction不能超过 5 个字符"):
            parse_gui_text("abcdef", name="instruction", required=True, max_chars=5)

    def test_gui_scope_parser_defaults_blank_values(self) -> None:
        self.assertEqual(parse_gui_scope("", default="draft"), "draft")
        self.assertEqual(parse_gui_scope(None, default="draft"), "draft")
        self.assertEqual(parse_gui_scope("  draft:abc  ", default="draft"), "draft:abc")
        self.assertEqual(parse_gui_scope(" session:7\n", default=""), "session:7")
        self.assertEqual(parse_gui_scope("", default=""), "")

        with self.assertRaisesRegex(ValueError, f"ui_scope不能超过 {GUI_MAX_SCOPE_CHARS} 个字符"):
            parse_gui_scope("x" * (GUI_MAX_SCOPE_CHARS + 1), default="draft")

    def test_gui_choice_parser_returns_clear_errors(self) -> None:
        self.assertEqual(parse_gui_choice("", name="模型", default="deepseek", allowed={"deepseek", "openai"}), "deepseek")
        self.assertEqual(parse_gui_choice(" OpenAI ", name="模型", default="deepseek", allowed={"deepseek", "openai"}), "openai")

        with self.assertRaisesRegex(ValueError, "模型必须是以下之一"):
            parse_gui_choice("claude", name="模型", default="deepseek", allowed={"deepseek", "openai"})

    def test_gui_state_session_query_parser_rejects_ambiguous_session_ids(self) -> None:
        self.assertIsNone(parse_gui_state_session_id({}))
        self.assertIsNone(parse_gui_state_session_id({"session_id": [""]}))
        self.assertEqual(parse_gui_state_session_id({"session_id": [" 12 "]}), 12)

        with self.assertRaisesRegex(ValueError, "session_id必须是整数"):
            parse_gui_state_session_id({"session_id": ["abc"]})
        with self.assertRaisesRegex(ValueError, "session_id不能小于 1"):
            parse_gui_state_session_id({"session_id": ["0"]})

    def test_gui_state_without_explicit_session_never_revives_stale_active_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                main(["init"])
                self.write_gui_test_llm_env(cwd)
                from research_alpha.db import create_idea_session

                db_path = cwd / "data" / "research_alpha.db"
                session_id = create_idea_session(db_path, "old session", "Old idea must not revive")
                state = GuiState()
                state.active_session_id = session_id

                self.assertIsNone(
                    resolve_gui_state_session_id(
                        db_path,
                        state,
                        requested_session_id=None,
                        draft_mode=False,
                    )
                )
                self.assertEqual(state.active_session_id, session_id)

                snapshot = build_gui_snapshot(cwd, state, session_id=None, draft_mode=True)
                text = "\n".join(item["text"] for item in snapshot["chat"]["messages"])
                self.assertFalse(snapshot["chat"]["active"])
                self.assertTrue(snapshot["chat"]["draft"])
                self.assertNotIn("Old idea must not revive", text)

                explicit = resolve_gui_state_session_id(
                    db_path,
                    state,
                    requested_session_id=session_id,
                    draft_mode=False,
                )
                self.assertEqual(explicit, session_id)
                self.assertEqual(state.active_session_id, session_id)
            finally:
                os.chdir(old_cwd)

    def test_gui_new_chat_uses_job_created_session_not_latest_fallback(self) -> None:
        html = render_index_html()
        self.assertIn("job.created_session_id", html)
        self.assertIn("fetchStateJson('/api/state?session_id=' + encodeURIComponent(activeSessionId))", html)
        self.assertIn("Number(j.id) === Number(activePending.jobId)", html)
        self.assertIn("Number(job.id) === Number(pending.jobId)", html)
        self.assertIn("function finalizeCompletedIdeateJob(job, activeScope='', options={})", html)
        self.assertIn("function finalizeCompletedIdeateJobs(jobs, activeScope='')", html)
        self.assertIn("finalizedJobKey(job)", html)
        self.assertIn("setAwaitingJobStart(jobScope, false)", html)
        self.assertIn("clearPendingIdeate(jobScope)", html)
        self.assertIn("adoptDraftMessagesToSession(Number(job.created_session_id), jobScope)", html)
        self.assertNotIn("const draftActivity = scopeMessages(activityMessagesByScope, sourceScope)", html)
        self.assertNotIn("if (!createdScope) {\n        const targetScope = jobScope;", html)
        self.assertIn("finalizeCompletedIdeateJob(pendingJob, activeScope, {activate:true})", html)
        self.assertIn("finalizeCompletedIdeateJobs(jobs, activeScope)", html)
        self.assertIn("activate:Boolean(job.created_session_id && job.ui_scope === scope)", html)
        self.assertIn("if (options.activate && createdScope)", html)
        self.assertIn("newChatMode || pendingIdeateForScope(activeScope) ? '?draft=1'", html)
        gui_source = Path("research_alpha/gui.py").read_text(encoding="utf-8")
        self.assertIn("resolve_gui_state_session_id(", gui_source)
        self.assertIn("draft_mode=draft_mode or requested_session_id is None", gui_source)
        self.assertNotIn("None if draft_mode else state.active_session_id", gui_source)
        self.assertIn("pendingIdeateJobsByScope[scope]", html)
        self.assertIn("const belongsToRenderedScope = jobScope === scope || isCreatedIdeateForScope(job, scope)", html)
        self.assertIn("finalizedInThisPass", html)
        self.assertIn("async function fetchStateJson(path)", html)
        self.assertIn("状态刷新失败：", html)
        self.assertIn("function handleMissingActiveSession(data, previousScope)", html)
        self.assertIn("data.chat.missing_session_id", html)
        self.assertIn("clearScopeMessages(previousScope)", html)
        self.assertIn("这条历史对话已经不存在", html)
        self.assertIn("if (handleMissingActiveSession(data, activeScope)) activeScope = currentScope();", html)
        self.assertIn("当前对话没有被切换", html)
        self.assertIn("lastStateError", html)
        self.assertNotIn("jobScope === scope || isPendingIdeateJob(job) || isCreatedIdeateForScope(job, scope)", html)
        self.assertNotIn("data = await (await fetch('/api/state')).json();", html)

    def test_gui_busy_state_is_scoped_to_current_conversation(self) -> None:
        html = render_index_html()
        self.assertIn("function scopeIsBusy(scope)", html)
        self.assertIn("['ideate', 'step', 'review', 'review-loop'].includes", html)
        self.assertIn("Boolean(pendingIdeateForScope(scope))", html)
        self.assertIn("runningConversationJobsForScope(jobs, scope).length > 0", html)
        self.assertIn("function scopeHasRunningEvidenceJob(scope=evidenceScope())", html)
        self.assertIn("function evidenceIsBusy()", html)
        self.assertIn("some(job => !isConversationBlockingJob(job))", html)
        self.assertIn("const awaitingCurrentScope = scopeAwaitingJobStart(scope)", html)
        self.assertIn("let hasBlockingRunning = false", html)
        self.assertIn("if (isConversationBlockingJob(job)) hasBlockingRunning = true", html)
        self.assertIn("setBusy(hasBlockingRunning || awaitingCurrentScope)", html)
        self.assertIn("if (!instruction) return;", html)
        self.assertIn("if (scopeIsBusy(scope))", html)
        self.assertIn("restoreSubmittedInstruction(scope, instruction);", html)
        self.assertNotIn("我已保留这次输入", html)
        self.assertIn("if (evidenceIsBusy()) return;", html)
        ideate_match = re.search(r"async function ideate\(direction\) \{([\s\S]*?)\n    \}", html)
        self.assertIsNotNone(ideate_match)
        self.assertNotIn("已有证据任务在后台运行，本次不重复提交", ideate_match.group(1))
        step_match = re.search(r"async function step\(instruction\) \{([\s\S]*?)\n    \}", html)
        self.assertIsNotNone(step_match)
        self.assertNotIn("中间对话仍可继续使用", step_match.group(1))
        self.assertIn("setAwaitingJobStart(scope, true)", html)
        self.assertIn("setAwaitingJobStart(scope, false)", html)
        self.assertIn("function clearAwaitingUiState(scope=currentScope())", html)
        self.assertIn("const previousScope = currentScope();", html)
        self.assertIn("clearAwaitingUiState(previousScope)", html)
        self.assertIn("saveInstructionDraft(previousScope)", html)
        send_chat_match = re.search(r"async function sendChat\(\) \{([\s\S]*?)\n    \}", html)
        self.assertIsNotNone(send_chat_match)
        send_chat_body = send_chat_match.group(1)
        self.assertNotIn("scopeHasRunningEvidenceJob(scope)", send_chat_body)
        self.assertIn("if (activeSessionId && !newChatMode) await step(text);", send_chat_body)
        self.assertNotIn("if (chatActive && !newChatMode) await step(text);", send_chat_body)
        self.assertIn("else await ideate(text);", send_chat_body)
        gui_source = Path("research_alpha/gui.py").read_text(encoding="utf-8")
        self.assertIn("state.add_job_unless_running(", gui_source)
        self.assertIn('existing.get("kind") == "ideate" and existing.get("ui_scope") == ui_scope', gui_source)
        self.assertIn("state.add_jobs_unless_running(", gui_source)
        self.assertIn("existing.get(\"kind\") in GUI_EVIDENCE_JOB_KINDS and existing.get(\"ui_scope\") == ui_scope", gui_source)
        self.assertIn('"deduped": True', gui_source)
        self.assertIn("function handleDedupedJob(scope, job", html)
        self.assertIn("本次没有重复提交", html)
        self.assertIn("removeTransientThinking(scope)", html)
        self.assertIn("clearPendingIdeate(scope);", html)
        self.assertIn("clearAwaitingUiState(scope);", html)
        self.assertIn("ui_scope:scope, provider:val('providerSelect')", html)
        self.assertIn("run_gui_stable_session_step", gui_source)
        self.assertIn('["reviewer", "--session-id", str(resolved_session_id)', gui_source)
        self.assertIn('["review-loop", "--session-id", str(resolved_session_id), "--rounds", str(rounds), "--provider", provider]', gui_source)
        self.assertIn("run_gui_stable_ideate_session", gui_source)
        self.assertNotIn('["review-loop", "--session-id", str(session_id), "--rounds"', gui_source)
        self.assertIn("resolve_gui_session(config.db_path, state, session_id, ui_scope=ui_scope)", gui_source)
        self.assertIn("session scope mismatch", gui_source)
        self.assertNotIn('["step"] + (["--session-id", str(session_id)] if session_id else ["--latest"])', gui_source)
        self.assertNotIn('+ (["--session-id", str(session_id)] if session_id else ["--latest"])', gui_source)
        self.assertNotIn("if (!instruction || isBusy) return;", html)
        self.assertNotIn("if (isBusy) return;", html)
        self.assertIn("if (activeSessionId && !newChatMode) await step(text);", html)
        self.assertNotIn("if (chatActive && !newChatMode) await step(text);", html)

    def test_gui_input_drafts_are_scoped_and_autosized(self) -> None:
        html = render_index_html()
        self.assertIn("instructionDraftsByScope", html)
        self.assertIn("function saveInstructionDraft(scope=currentScope())", html)
        self.assertIn("function restoreInstructionDraft(scope=currentScope())", html)
        self.assertIn("function clearInstructionDraft(scope=currentScope())", html)
        self.assertIn("function takeInstructionForSubmit(scope=currentScope())", html)
        self.assertIn("function restoreSubmittedInstruction(scope, text)", html)
        self.assertIn("const text = takeInstructionForSubmit(scope);", html)
        self.assertIn("if (activeSessionId && !newChatMode) await step(text);", html)
        self.assertNotIn("if (chatActive && !newChatMode) await step(text);", html)
        self.assertIn("else await ideate(text);", html)
        self.assertIn("restoreText:direction", html)
        self.assertIn("restoreText:instruction", html)
        self.assertIn("if (options.restoreText) restoreSubmittedInstruction(scope, options.restoreText);", html)
        ideate_match = re.search(r"async function ideate\(direction\) \{([\s\S]*?)\n    \}", html)
        self.assertIsNotNone(ideate_match)
        ideate_body = ideate_match.group(1)
        self.assertNotIn("scopeHasRunningEvidenceJob(scope)", ideate_body)
        self.assertNotIn("restoreSubmittedInstruction(scope, direction);\n        renderChatIfCurrent", ideate_body)
        step_match = re.search(r"async function step\(instruction\) \{([\s\S]*?)\n    \}", html)
        self.assertIsNotNone(step_match)
        step_body = step_match.group(1)
        self.assertIn("if (!instruction) return;", step_body)
        self.assertIn("if (scopeIsBusy(scope))", step_body)
        self.assertIn("restoreSubmittedInstruction(scope, instruction);", step_body)
        self.assertNotIn("pushActivityMessage", step_body)
        self.assertNotIn("else { await ideate(text); clearInstructionDraft(scope); }", html)
        self.assertNotIn("async function step() {\n      const instruction = val('instruction').trim();", html)
        self.assertIn("function autosizeInstruction()", html)
        self.assertIn("saveInstructionDraft(currentScope());", html)
        self.assertIn("restoreInstructionDraft(currentScope());", html)
        self.assertIn("restoreInstructionDraft(draftScopeId);", html)
        self.assertIn("clearInstructionDraft(scope);", html)
        self.assertIn("addEventListener('input'", html)
        self.assertIn("Math.min(instruction.scrollHeight, 180)", html)

    def test_gui_gold_build_reports_strict_zero_import_in_chat(self) -> None:
        html = render_index_html()
        self.assertNotIn("正在联网扩充核心优秀论文库", html)
        self.assertIn("imported\\/updated 0 records", html)
        self.assertIn("Gold-build summary:", html)
        self.assertIn("Gold-build summary JSON:", html)
        self.assertIn("function parseGoldBuildSummary(result)", html)
        self.assertIn("JSON.parse(jsonMatch[1])", html)
        self.assertIn("metadata_only_remote", html)
        self.assertIn("function goldBuildDetailLines(summary)", html)
        self.assertIn("function completedGoldBuildGroupForJob(job, jobs)", html)
        self.assertIn("function summarizeGoldBuildGroup(groupJobs)", html)
        self.assertIn("function goldBuildGroupJobText(groupJobs)", html)
        self.assertIn("function goldBuildGroupActions(groupJobs)", html)
        self.assertIn("Number(summary.failures || 0) > 0", html)
        self.assertIn("? [{label:'调整检索', action:'settings'}, {label:'查看趋势论文', action:'frontier'}]", html)
        self.assertIn("if (action === 'settings') metricAction('evidence-build');", html)
        self.assertIn("本轮优秀论文库扩充已完成", html)
        self.assertIn("聚合结果：抓取", html)
        self.assertIn("gold-group:${job.group_id}", html)
        self.assertIn("goldGroupJobs.forEach(item => lastJobStatuses.set(item.id, item.status))", html)
        self.assertIn("source_failures", html)
        self.assertIn("source_empty_results", html)
        self.assertIn("空结果来源：", html)
        self.assertIn("请求成功但没有返回论文", html)
        self.assertIn("source_reports", html)
        self.assertIn("quality_policy", html)
        self.assertIn("source_priority", html)
        self.assertIn("next_actions", html)
        self.assertIn("核心 Gold 标准：", html)
        self.assertIn("来源优先级：", html)
        self.assertIn("失败源：", html)
        self.assertIn("来源明细：", html)
        self.assertIn("建议动作：", html)
        self.assertIn("failure_summary", html)
        self.assertIn("function evidenceFailureHint(job)", html)
        self.assertIn("summary.hint", html)
        self.assertIn("summary.next_step", html)
        self.assertIn("趋势 ${summary.candidate_imported ?? 0} · Gold ${summary.excellent_imported ?? 0} · 失败源 ${summary.failures ?? 0}", html)
        self.assertIn("evidenceFailureHint(failedEvidence)", html)
        self.assertIn("联网扩充结果：抓取", html)
        self.assertIn("Poster 硬过滤", html)
        self.assertIn("这次来源是元数据收集模式", html)
        self.assertIn("严格过滤后没有发现可进入核心优秀论文库的记录", html)
        self.assertIn("legacyRemoteCandidatePaper", html)
        self.assertIn("trendRemotePaper", html)
        self.assertIn("quality-enrich", html)
        self.assertIn("质量核验只更新论文元数据标签", html)

    def test_gui_snapshot_includes_frontier_candidates_separately_from_excellent_papers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                main(["init"])
                db_path = cwd / "data" / "research_alpha.db"
                add_paper(
                    db_path,
                    "Unverified Remote Candidate",
                    "ICLR",
                    2026,
                    "A recent candidate paper.",
                    source_kind="frontier_openalex",
                    citation_count=12,
                )
                add_paper(
                    db_path,
                    "Verified Excellent Paper",
                    "ICLR",
                    2026,
                    "A known excellent paper.",
                    source_kind="seed",
                    award="best_paper",
                    citation_count=500,
                )
                add_paper(
                    db_path,
                    "Legacy Non Core Gold Record",
                    "ACL",
                    2025,
                    "A historical auxiliary award record that should not appear as core Gold.",
                    source_kind="gold_acl_awards",
                    award="best_paper",
                    citation_count=300,
                )
                main(["score"])
                from research_alpha.gui import GuiState, build_gui_snapshot

                snapshot = build_gui_snapshot(cwd, GuiState(), session_id=None)
                self.assertTrue(any(option["value"] == "openalex" for option in snapshot["options"]["remote_options"]))
                self.assertTrue(any(option["value"] == "s2" for option in snapshot["options"]["remote_options"]))
                self.assertFalse(any(option["value"] == "icml_awards" for option in snapshot["options"]["remote_options"]))
                self.assertFalse(any(option["value"] == "neurips_awards" for option in snapshot["options"]["remote_options"]))
                self.assertFalse(any(option["value"] == "cvf_awards" for option in snapshot["options"]["remote_options"]))
                self.assertFalse(any(option["value"] == "acl_awards" for option in snapshot["options"]["remote_options"]))
                self.assertFalse(any(option["value"] == "eccv_awards" for option in snapshot["options"]["remote_options"]))
                self.assertTrue(any(p["title"] == "Unverified Remote Candidate" for p in snapshot["frontier_papers"]))
                self.assertFalse(any(p["title"] == "Unverified Remote Candidate" for p in snapshot["top_papers"]))
                self.assertTrue(any(p["title"] == "Legacy Non Core Gold Record" for p in snapshot["frontier_papers"]))
                self.assertFalse(any(p["title"] == "Legacy Non Core Gold Record" for p in snapshot["top_papers"]))
                self.assertFalse(any(row["title"] == "Legacy Non Core Gold Record" for row in list_top_papers(db_path, limit=20)))
                self.assertTrue(any(row["title"] == "Legacy Non Core Gold Record" for row in list_frontier_papers(db_path, limit=20)))
                self.assertTrue(any(p["title"] == "Verified Excellent Paper" for p in snapshot["top_papers"]))
                excellent = next(p for p in snapshot["top_papers"] if p["title"] == "Verified Excellent Paper")
                frontier = next(p for p in snapshot["frontier_papers"] if p["title"] == "Unverified Remote Candidate")
                auxiliary = next(p for p in snapshot["frontier_papers"] if p["title"] == "Legacy Non Core Gold Record")
                self.assertIn("最佳论文", excellent["quality_reason"])
                self.assertIn("本地导入", excellent["source_label"])
                self.assertEqual(excellent["library_status"], "Gold")
                self.assertIn("参与 idea 生成", excellent["library_status_detail"])
                self.assertIn("仅作为趋势参考", frontier["quality_reason"])
                self.assertIn("OpenAlex 趋势参考", frontier["source_label"])
                self.assertEqual(frontier["library_status"], "待核验趋势")
                self.assertIn("quality-enrich", frontier["library_status_detail"])
                self.assertNotIn("候选", frontier["source_label"])
                self.assertIn("不参与 idea 评分标准", auxiliary["quality_reason"])
                self.assertEqual(auxiliary["library_status"], "辅助/降级")
                self.assertIn("不参与 idea 评分标准", auxiliary["library_status_detail"])
            finally:
                os.chdir(old_cwd)

    def test_user_library_is_domain_knowledge_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                main(["init"])
                db_path = cwd / "data" / "research_alpha.db"
                add_paper(
                    db_path,
                    "User Route Paper",
                    "arXiv",
                    2026,
                    "Useful route knowledge.",
                    source_kind="user_library",
                    award="best_paper",
                    citation_count=10000,
                )
                main(["score"])
                snapshot = build_gui_snapshot(cwd, GuiState(), session_id=None)
                self.assertTrue(any(p["title"] == "User Route Paper" for p in snapshot["user_papers"]))
                self.assertFalse(any(p["title"] == "User Route Paper" for p in snapshot["top_papers"]))
                self.assertFalse(any(p["title"] == "User Route Paper" for p in snapshot["frontier_papers"]))
                user_paper = next(p for p in snapshot["user_papers"] if p["title"] == "User Route Paper")
                self.assertEqual(user_paper["source_label"], "用户自建论文库")
                self.assertEqual(user_paper["library_status"], "用户库")
                self.assertIn("不参与 idea 评分标准", user_paper["library_status_detail"])
                with connect(db_path) as conn:
                    row = conn.execute(
                        "SELECT source_kind, award, paper_weight, score_notes FROM papers WHERE title=?",
                        ("User Route Paper",),
                    ).fetchone()
                self.assertEqual(row["source_kind"], "user_library")
                self.assertEqual(float(row["paper_weight"]), 0.0)
                self.assertIn("user_library_domain_knowledge_only", row["score_notes"])
                self.assertEqual(len(list_user_library_papers(db_path, limit=10)), 1)
                self.assertFalse(any(row["title"] == "User Route Paper" for row in list_top_papers(db_path, limit=20)))
                self.assertFalse(any(row["title"] == "User Route Paper" for row in list_frontier_papers(db_path, limit=20)))
            finally:
                os.chdir(old_cwd)

    def test_user_library_duplicate_title_does_not_downgrade_gold_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                main(["init"])
                db_path = cwd / "data" / "research_alpha.db"
                add_paper(
                    db_path,
                    "Shared Paper",
                    "ICLR",
                    2026,
                    "Official excellent paper.",
                    source_kind="gold_openreview",
                    award="best_paper",
                    citation_count=100,
                )
                main(["score"])
                upsert_user_library_paper(
                    db_path,
                    title="Shared Paper",
                    venue="ICLR",
                    year=2026,
                    abstract="User route note.",
                    external_ref="https://arxiv.org/abs/2604.01234",
                )
                with connect(db_path) as conn:
                    rows = conn.execute(
                        "SELECT title, source_kind, award, paper_weight FROM papers WHERE title=? ORDER BY id",
                        ("Shared Paper",),
                    ).fetchall()
                self.assertEqual(len(rows), 2)
                gold = next(row for row in rows if row["source_kind"] == "gold_openreview")
                user = next(row for row in rows if row["source_kind"] == "user_library")
                self.assertEqual(gold["award"], "best_paper")
                self.assertGreater(float(gold["paper_weight"]), 0.0)
                self.assertEqual(user["award"], "")
                self.assertEqual(float(user["paper_weight"]), 0.0)
            finally:
                os.chdir(old_cwd)

    def test_delete_idea_session_removes_history_and_turns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                main(["init"])
                self.write_gui_test_llm_env(cwd)
                from research_alpha.db import create_idea_session

                db_path = cwd / "data" / "research_alpha.db"
                session_id = create_idea_session(db_path, "agent eval", "Initial idea")
                with connect(db_path) as conn:
                    conn.execute(
                        """
                        INSERT INTO idea_session_turns(
                            session_id, turn_index, user_instruction, decision, revised_idea, content_json
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (session_id, 1, "revise", "partial", "Revised idea", "{}"),
                    )
                delete_idea_session(db_path, session_id)
                snapshot = build_gui_snapshot(cwd, GuiState())
                self.assertEqual(snapshot["sessions"], [])
                with connect(db_path) as conn:
                    turns = conn.execute("SELECT COUNT(*) AS count FROM idea_session_turns").fetchone()
                self.assertEqual(int(turns["count"]), 0)
            finally:
                os.chdir(old_cwd)

    def test_gui_session_delete_api_rejects_running_session_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                self.write_gui_test_llm_env(cwd)
                from research_alpha.db import create_idea_session

                db_path = cwd / "data" / "research_alpha.db"
                session_id = create_idea_session(db_path, "agent eval", "Initial idea")
                state = GuiState()
                release = threading.Event()

                def target():
                    release.wait(2)
                    return {"stdout": "done"}

                state.add_job("session step", target, kind="step", session_id=session_id, ui_scope=f"session:{session_id}")
                server = create_gui_http_server(cwd, host="127.0.0.1", port=0, state=state)
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                try:
                    status, body = self.gui_post_json(server, "/api/session/delete", {"session_id": session_id})
                    self.assertEqual(status, 409)
                    self.assertIn("仍有任务在运行", str(body["error"]))
                    self.assertFalse(body["ok"])
                    self.assertEqual(body["status"], 409)
                    self.assertEqual(body["code"], "conflict")
                    self.assertEqual(body["jobs"][0]["kind"], "step")
                    self.assertTrue(build_chat_snapshot(cwd, session_id=session_id)["active"])

                    release.set()
                    for _ in range(50):
                        if not state.running_jobs_for_session(session_id):
                            break
                        import time

                        time.sleep(0.01)

                    status, body = self.gui_post_json(server, "/api/session/delete", {"session_id": session_id})
                    self.assertEqual(status, 200)
                    self.assertTrue(body["ok"])
                    self.assertFalse(build_chat_snapshot(cwd, session_id=session_id)["active"])
                finally:
                    release.set()
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=2)
            finally:
                os.chdir(old_cwd)

    def test_gui_idea_delete_api_removes_candidate_idea(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                db_path = cwd / "data" / "research_alpha.db"
                idea_id = add_candidate_idea(
                    db_path,
                    query="research agents",
                    title="Delete Me",
                    content_json=json.dumps({"idea_title": "Delete Me"}),
                )
                server = create_gui_http_server(cwd, host="127.0.0.1", port=0)
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                try:
                    status, body = self.gui_post_json(server, "/api/idea/delete", {"idea_id": idea_id})
                    self.assertEqual(status, 200)
                    self.assertTrue(body["ok"])
                    with connect(db_path) as conn:
                        count = conn.execute("SELECT COUNT(*) AS count FROM candidate_ideas WHERE id = ?", (idea_id,)).fetchone()
                    self.assertEqual(int(count["count"]), 0)

                    status, body = self.gui_post_json(server, "/api/idea/delete", {"idea_id": idea_id})
                    self.assertEqual(status, 404)
                    self.assertEqual(body["code"], "idea_not_found")
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=2)
            finally:
                os.chdir(old_cwd)

    def test_gui_user_library_add_api_validates_and_saves_domain_link(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                db_path = cwd / "data" / "research_alpha.db"
                server = create_gui_http_server(cwd, host="127.0.0.1", port=0)
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                try:
                    with patch("research_alpha.gui.urlopen", side_effect=AssertionError("metadata should enrich in background")), patch(
                        "research_alpha.gui.start_user_library_enrichment", return_value={"status": "queued"}
                    ):
                        status, body = self.gui_post_json(
                            server,
                            "/api/user-library/add",
                            {
                                "url": "https://arxiv.org/abs/2601.01234v2",
                                "title": "",
                                "note": "路线学习：领域术语和问题背景。",
                            },
                        )
                    self.assertEqual(status, 200)
                    self.assertTrue(body["ok"])
                    self.assertEqual(body["paper"]["title"], "arXiv:2601.01234")
                    self.assertEqual(body["paper"]["library_status"], "用户库")
                    with connect(db_path) as conn:
                        row = conn.execute(
                            "SELECT title, venue, year, source_kind, external_ref, award, paper_weight FROM papers WHERE id=?",
                            (body["paper_id"],),
                        ).fetchone()
                    self.assertEqual(row["venue"], "arXiv")
                    self.assertEqual(int(row["year"]), 2026)
                    self.assertEqual(row["source_kind"], "user_library")
                    self.assertEqual(row["external_ref"], "https://arxiv.org/abs/2601.01234v2")
                    self.assertEqual(row["award"], "")
                    self.assertEqual(float(row["paper_weight"]), 0.0)

                    status, invalid = self.gui_post_json(server, "/api/user-library/add", {"url": "file:///tmp/paper.pdf"})
                    self.assertEqual(status, 400)
                    self.assertEqual(invalid["code"], "invalid_user_library_link")
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=2)
            finally:
                os.chdir(old_cwd)

    def test_user_library_background_enriches_arxiv_metadata_without_scoring(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            db_path = cwd / "data" / "research_alpha.db"
            init_db(db_path)
            paper_id = upsert_user_library_paper(
                db_path,
                title="arXiv:2601.01234",
                venue="arXiv",
                year=2026,
                abstract="用户关注：多轮记忆机制。",
                external_ref="https://arxiv.org/abs/2601.01234v2",
            )
            atom = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2601.01234v2</id>
    <published>2026-01-08T00:00:00Z</published>
    <title>  Reliable Research Agent Memory  </title>
    <summary>
      We study stable memory for research agents and evidence-grounded idea generation.
    </summary>
  </entry>
</feed>"""

            class FakeResponse:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

                def read(self, limit=-1):
                    return atom

            with patch("research_alpha.gui.urlopen", return_value=FakeResponse()) as mock_urlopen, patch(
                "research_alpha.gui.fetch_full_text_with_metadata", side_effect=RuntimeError("skip full text")
            ):
                status = enrich_user_library_paper(
                    db_path,
                    paper_id,
                    url="https://arxiv.org/abs/2601.01234v2",
                    title="",
                    note="用户关注：多轮记忆机制。",
                    metadata={"title": "arXiv:2601.01234", "venue": "arXiv", "year": 2026, "abstract": "用户关注：多轮记忆机制。"},
                )
            self.assertEqual(status["status"], "failed")
            self.assertEqual(mock_urlopen.call_count, 1)
            with connect(db_path) as conn:
                row = conn.execute(
                    "SELECT title, abstract, venue, year, source_kind, award, paper_weight, score_notes FROM papers WHERE id=?",
                    (paper_id,),
                ).fetchone()
            self.assertEqual(row["title"], "Reliable Research Agent Memory")
            self.assertIn("用户关注：多轮记忆机制。", row["abstract"])
            self.assertIn("stable memory", row["abstract"])
            self.assertEqual(row["venue"], "arXiv")
            self.assertEqual(int(row["year"]), 2026)
            self.assertEqual(row["source_kind"], "user_library")
            self.assertEqual(row["award"], "")
            self.assertEqual(float(row["paper_weight"]), 0.0)
            self.assertIn("user_library_domain_knowledge_only", row["score_notes"])

    def test_gui_user_library_accepts_legacy_arxiv_and_pdf_links_as_domain_only(self) -> None:
        self.assertEqual(arxiv_id_from_url("https://arxiv.org/abs/cs/0501001v2"), "cs/0501001")
        self.assertEqual(arxiv_id_from_url("https://arxiv.org/pdf/cs/0501001.pdf"), "cs/0501001")
        self.assertEqual(arxiv_id_from_url("https://arxiv.org/pdf/2601.01234v3.pdf"), "2601.01234")
        self.assertEqual(year_from_arxiv_id("cs/0501001"), 2005)

        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                db_path = cwd / "data" / "research_alpha.db"
                server = create_gui_http_server(cwd, host="127.0.0.1", port=0)
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                try:
                    with patch("research_alpha.gui.urlopen", side_effect=AssertionError("metadata should enrich in background")), patch(
                        "research_alpha.gui.start_user_library_enrichment", return_value={"status": "queued"}
                    ):
                        status, body = self.gui_post_json(
                            server,
                            "/api/user-library/add",
                            {
                                "url": "https://arxiv.org/pdf/cs/0501001v2.pdf",
                                "title": "",
                                "note": "经典旧 arXiv 路线论文，只作领域知识。",
                            },
                        )
                    self.assertEqual(status, 200)
                    self.assertTrue(body["ok"])
                    self.assertEqual(body["paper"]["title"], "arXiv:cs/0501001")
                    self.assertEqual(body["paper"]["venue"], "arXiv")
                    self.assertEqual(body["paper"]["year"], 2005)
                    self.assertEqual(body["paper"]["library_status"], "用户库")
                    with connect(db_path) as conn:
                        row = conn.execute(
                            "SELECT source_kind, paper_weight, award, score_notes FROM papers WHERE id=?",
                            (body["paper_id"],),
                        ).fetchone()
                    self.assertEqual(row["source_kind"], "user_library")
                    self.assertEqual(float(row["paper_weight"]), 0.0)
                    self.assertEqual(row["award"], "")
                    self.assertIn("user_library_domain_knowledge_only", row["score_notes"])
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=2)
            finally:
                os.chdir(old_cwd)

    def test_gui_user_library_add_api_accepts_doi_and_openreview_as_domain_links(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                db_path = cwd / "data" / "research_alpha.db"
                server = create_gui_http_server(cwd, host="127.0.0.1", port=0)
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                try:
                    with patch("research_alpha.gui.start_user_library_enrichment", return_value={"status": "queued"}):
                        status, doi_body = self.gui_post_json(
                            server,
                            "/api/user-library/add",
                            {
                                "url": "https://doi.org/10.48550/arXiv.2601.01234",
                                "title": "",
                                "note": "路线学习：DOI 论文。",
                            },
                        )
                        self.assertEqual(status, 200)
                        self.assertTrue(doi_body["ok"])
                        self.assertEqual(doi_body["paper"]["title"], "DOI:10.48550/arXiv.2601.01234")

                        status, openreview_body = self.gui_post_json(
                            server,
                            "/api/user-library/add",
                            {
                                "url": "https://openreview.net/forum?id=abc123",
                                "title": "",
                                "note": "路线学习：OpenReview 论文。",
                            },
                        )
                    self.assertEqual(status, 200)
                    self.assertTrue(openreview_body["ok"])
                    self.assertEqual(openreview_body["paper"]["title"], "OpenReview:abc123")

                    status, invalid = self.gui_post_json(
                        server,
                        "/api/user-library/add",
                        {"url": "https://doi.org/10.48550/arXiv.2601.01234?token=secret"},
                    )
                    self.assertEqual(status, 400)
                    self.assertEqual(invalid["code"], "invalid_user_library_link")

                    with connect(db_path) as conn:
                        rows = conn.execute(
                            "SELECT title, abstract, venue, source_kind, award, paper_weight, score_notes FROM papers ORDER BY id"
                        ).fetchall()
                    by_title = {row["title"]: row for row in rows}
                    self.assertEqual(by_title["DOI:10.48550/arXiv.2601.01234"]["venue"], "DOI")
                    self.assertIn("DOI: 10.48550/arXiv.2601.01234", by_title["DOI:10.48550/arXiv.2601.01234"]["abstract"])
                    self.assertEqual(by_title["OpenReview:abc123"]["venue"], "OpenReview")
                    self.assertIn("OpenReview id: abc123", by_title["OpenReview:abc123"]["abstract"])
                    for row in rows:
                        self.assertEqual(row["source_kind"], "user_library")
                        self.assertEqual(row["award"], "")
                    self.assertEqual(float(row["paper_weight"]), 0.0)
                    self.assertIn("user_library_domain_knowledge_only", row["score_notes"])
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=2)
            finally:
                os.chdir(old_cwd)

    def test_gui_user_library_add_queues_full_text_without_blocking_domain_link(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                db_path = cwd / "data" / "research_alpha.db"

                server = create_gui_http_server(cwd, host="127.0.0.1", port=0)
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                try:
                    with patch("research_alpha.gui.urlopen", side_effect=AssertionError("metadata should enrich in background")), patch(
                        "research_alpha.gui.start_user_library_enrichment", return_value={"status": "queued"}
                    ):
                        status, body = self.gui_post_json(
                            server,
                            "/api/user-library/add",
                            {
                                "url": "https://arxiv.org/abs/2601.01234v2",
                                "title": "Reliable User Route Paper",
                                "note": "路线学习：临床交接场景。",
                            },
                        )
                    self.assertEqual(status, 200)
                    self.assertTrue(body["ok"])
                    self.assertEqual(body["full_text"]["status"], "queued")
                    with connect(db_path) as conn:
                        row = conn.execute(
                            "SELECT source_kind, award, paper_weight, score_notes, full_text_json FROM papers WHERE id=?",
                            (body["paper_id"],),
                        ).fetchone()
                    self.assertEqual(row["source_kind"], "user_library")
                    self.assertEqual(row["award"], "")
                    self.assertEqual(float(row["paper_weight"]), 0.0)
                    self.assertIn("user_library_domain_knowledge_only", row["score_notes"])
                    self.assertIn(row["full_text_json"] or "{}", {"", "{}"})
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=2)
            finally:
                os.chdir(old_cwd)

    def test_user_library_full_text_ingest_updates_domain_context_without_scoring(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            db_path = cwd / "data" / "research_alpha.db"
            init_db(db_path)
            paper_id = upsert_user_library_paper(
                db_path,
                title="Reliable User Route Paper",
                venue="arXiv",
                year=2026,
                abstract="Route note.",
                external_ref="https://arxiv.org/abs/2601.01234v2",
            )

            def fake_fetch(url: str, **kwargs):
                self.assertEqual(url, "https://arxiv.org/abs/2601.01234v2")
                self.assertEqual(kwargs["title"], "Reliable User Route Paper")
                return (
                    "Introduction User domain setup explains clinical handoff constraints. "
                    "Related Work Prior systems miss local route evidence. "
                    + ("Method Domain route learning extracts reusable terminology. " * 80)
                    + ("Experiments Domain notes are checked against realistic workflow boundaries. " * 80)
                    + "Limitations Domain notes are context only and do not define scoring standards.",
                    "https://arxiv.org/pdf/2601.01234v2.pdf",
                )

            with patch("research_alpha.gui.fetch_full_text_with_metadata", side_effect=fake_fetch):
                status = opportunistic_user_library_full_text_ingest(
                    db_path,
                    paper_id,
                    url="https://arxiv.org/abs/2601.01234v2",
                    metadata={"title": "Reliable User Route Paper", "venue": "arXiv", "year": 2026},
                )
            self.assertEqual(status["status"], "updated")
            self.assertIn("arxiv.org/pdf/2601.01234v2.pdf", status["source_url"])
            with connect(db_path) as conn:
                row = conn.execute(
                    "SELECT source_kind, award, paper_weight, score_notes, full_text_json FROM papers WHERE id=?",
                    (paper_id,),
                ).fetchone()
            self.assertEqual(row["source_kind"], "user_library")
            self.assertEqual(row["award"], "")
            self.assertEqual(float(row["paper_weight"]), 0.0)
            self.assertIn("user_library_domain_knowledge_only", row["score_notes"])
            self.assertIn("clinical handoff constraints", row["full_text_json"])
            self.assertIn("source_url", row["full_text_json"])

    def test_user_library_full_text_ingest_degrades_without_blocking_domain_link(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            db_path = cwd / "data" / "research_alpha.db"
            init_db(db_path)
            paper_id = upsert_user_library_paper(
                db_path,
                title="Fallback User Domain Paper",
                venue="OpenReview",
                year=2026,
                abstract="Only domain background.",
                external_ref="https://openreview.net/forum?id=abc123",
            )
            with patch("research_alpha.gui.fetch_full_text_with_metadata", side_effect=RuntimeError("network timeout")):
                status = opportunistic_user_library_full_text_ingest(
                    db_path,
                    paper_id,
                    url="https://openreview.net/forum?id=abc123",
                    metadata={"title": "Fallback User Domain Paper", "venue": "OpenReview", "year": 2026},
                )
            self.assertEqual(status["status"], "failed")
            with connect(db_path) as conn:
                row = conn.execute(
                    "SELECT title, source_kind, award, paper_weight, score_notes, full_text_json FROM papers WHERE id=?",
                    (paper_id,),
                ).fetchone()
            self.assertEqual(row["title"], "Fallback User Domain Paper")
            self.assertEqual(row["source_kind"], "user_library")
            self.assertEqual(row["award"], "")
            self.assertEqual(float(row["paper_weight"]), 0.0)
            self.assertIn("user_library_domain_knowledge_only", row["score_notes"])
            self.assertIn(row["full_text_json"] or "{}", {"", "{}"})

    def test_gui_user_library_delete_api_only_removes_user_library_papers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                db_path = cwd / "data" / "research_alpha.db"
                user_id = upsert_user_library_paper(
                    db_path,
                    title="Delete User Paper",
                    venue="arXiv",
                    year=2026,
                    abstract="Route note.",
                    external_ref="https://arxiv.org/abs/2601.01111",
                )
                gold_id = add_paper(
                    db_path,
                    "Keep Gold Paper",
                    "ICLR",
                    2026,
                    "Gold record.",
                    source_kind="gold_openreview",
                    award="best_paper",
                )
                server = create_gui_http_server(cwd, host="127.0.0.1", port=0)
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                try:
                    status, body = self.gui_post_json(server, "/api/user-library/delete", {"paper_id": gold_id})
                    self.assertEqual(status, 404)
                    self.assertEqual(body["code"], "user_library_paper_not_found")

                    status, body = self.gui_post_json(server, "/api/user-library/delete", {"paper_id": user_id})
                    self.assertEqual(status, 200)
                    self.assertTrue(body["ok"])
                    with connect(db_path) as conn:
                        user_count = conn.execute("SELECT COUNT(*) AS count FROM papers WHERE id=?", (user_id,)).fetchone()
                        gold_count = conn.execute("SELECT COUNT(*) AS count FROM papers WHERE id=?", (gold_id,)).fetchone()
                    self.assertEqual(int(user_count["count"]), 0)
                    self.assertEqual(int(gold_count["count"]), 1)
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=2)
            finally:
                os.chdir(old_cwd)

    def test_gui_session_step_api_rejects_delete_in_progress_without_starting_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                from research_alpha.db import create_idea_session

                db_path = cwd / "data" / "research_alpha.db"
                session_id = create_idea_session(db_path, "agent eval", "Initial idea")
                state = GuiState()
                delete_started, running_jobs = state.begin_session_delete(session_id)
                self.assertTrue(delete_started)
                self.assertEqual(running_jobs, [])
                captured_commands = []

                def fake_streaming(command, root_dir, on_output=None):
                    captured_commands.append(command)
                    return {"exit_code": 0, "stdout": "should not run", "stderr": ""}

                with patch("research_alpha.gui.run_app_command_streaming", side_effect=fake_streaming):
                    server = create_gui_http_server(cwd, host="127.0.0.1", port=0, state=state)
                    thread = threading.Thread(target=server.serve_forever, daemon=True)
                    thread.start()
                    try:
                        status, body = self.gui_post_json(
                            server,
                            "/api/step",
                            {
                                "session_id": session_id,
                                "ui_scope": f"session:{session_id}",
                                "instruction": "revise the idea",
                                "provider": "deepseek",
                            },
                        )
                        self.assertEqual(status, 409)
                        self.assertFalse(body["ok"])
                        self.assertEqual(body["code"], "session_delete_in_progress")
                        self.assertEqual(captured_commands, [])
                        self.assertEqual(state.running_jobs_for_session(session_id), [])
                    finally:
                        server.shutdown()
                        server.server_close()
                        thread.join(timeout=2)
                state.finish_session_delete(session_id)
            finally:
                os.chdir(old_cwd)

    def test_gui_session_delete_api_rejects_duplicate_delete_in_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                from research_alpha.db import create_idea_session

                db_path = cwd / "data" / "research_alpha.db"
                session_id = create_idea_session(db_path, "agent eval", "Initial idea")
                state = GuiState()
                delete_started, running_jobs = state.begin_session_delete(session_id)
                self.assertTrue(delete_started)
                self.assertEqual(running_jobs, [])

                server = create_gui_http_server(cwd, host="127.0.0.1", port=0, state=state)
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                try:
                    status, body = self.gui_post_json(server, "/api/session/delete", {"session_id": session_id})
                    self.assertEqual(status, 409)
                    self.assertFalse(body["ok"])
                    self.assertEqual(body["code"], "session_delete_in_progress")
                    self.assertTrue(build_chat_snapshot(cwd, session_id=session_id)["active"])
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=2)
                    state.finish_session_delete(session_id)
            finally:
                os.chdir(old_cwd)

    def test_gui_session_post_api_dedupes_conversation_jobs_across_kinds(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                self.write_gui_test_llm_env(cwd)
                from research_alpha.db import create_idea_session

                db_path = cwd / "data" / "research_alpha.db"
                session_id = create_idea_session(db_path, "agent eval", "Initial idea")
                release = threading.Event()
                captured_steps = []

                def fake_stable_step(root_dir, session_id, instruction, provider, on_output=None):
                    captured_steps.append((session_id, instruction, provider))
                    if on_output:
                        on_output("stdout", "session job is running")
                    release.wait(2)
                    return {"exit_code": 0, "stdout": "Turn 1 decision: revised\nRevised idea: updated", "stderr": ""}

                with patch("research_alpha.gui.run_gui_stable_session_step", side_effect=fake_stable_step):
                    server = create_gui_http_server(cwd, host="127.0.0.1", port=0)
                    thread = threading.Thread(target=server.serve_forever, daemon=True)
                    thread.start()
                    try:
                        step_payload = {
                            "session_id": session_id,
                            "ui_scope": f"session:{session_id}",
                            "instruction": "revise the idea",
                            "provider": "deepseek",
                        }
                        status, first = self.gui_post_json(server, "/api/step", step_payload)
                        self.assertEqual(status, 200)
                        self.assertEqual(first["job"]["kind"], "step")

                        status, second = self.gui_post_json(
                            server,
                            "/api/review",
                            {"session_id": session_id, "ui_scope": f"session:{session_id}", "provider": "deepseek"},
                        )
                        self.assertEqual(status, 200)
                        self.assertTrue(second["deduped"])
                        self.assertEqual(second["job"]["id"], first["job"]["id"])
                        self.assertEqual(second["job"]["kind"], "step")

                        for _ in range(50):
                            if captured_steps:
                                break
                            import time

                            time.sleep(0.01)
                        self.assertEqual(captured_steps, [(session_id, "revise the idea", "deepseek")])
                    finally:
                        release.set()
                        server.shutdown()
                        server.server_close()
                        thread.join(timeout=2)
            finally:
                os.chdir(old_cwd)

    def test_gui_ideate_with_session_scope_routes_to_current_session_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                self.write_gui_test_llm_env(cwd)
                from research_alpha.db import create_idea_session, list_idea_sessions, list_idea_session_turns

                db_path = cwd / "data" / "research_alpha.db"
                session_id = create_idea_session(db_path, "agent eval", "Initial idea")
                captured_steps = []

                def fake_stable_step(root_dir, session_id, instruction, provider, on_output=None):
                    captured_steps.append((session_id, instruction, provider))
                    return {"exit_code": 0, "stdout": "Turn 1 decision: revised\nRevised idea: updated current session", "stderr": ""}

                with patch("research_alpha.gui.run_gui_stable_session_step", side_effect=fake_stable_step):
                    server = create_gui_http_server(cwd, host="127.0.0.1", port=0)
                    thread = threading.Thread(target=server.serve_forever, daemon=True)
                    thread.start()
                    try:
                        status, body = self.gui_post_json(
                            server,
                            "/api/ideate",
                            {
                                "direction": "继续追问当前 idea",
                                "session_id": None,
                                "ui_scope": f"session:{session_id}",
                                "remote": "openalex",
                                "ideas": 3,
                                "limit": 12,
                                "provider": "deepseek",
                                "lang": "zh",
                            },
                        )
                        self.assertEqual(status, 200)
                        self.assertTrue(body["routed_to_session_step"])
                        self.assertEqual(body["job"]["kind"], "step")
                        self.assertEqual(body["job"]["session_id"], session_id)
                        self.assertEqual(body["job"]["ui_scope"], f"session:{session_id}")

                        for _ in range(50):
                            if captured_steps:
                                break
                            import time

                            time.sleep(0.01)
                        self.assertEqual(captured_steps, [(session_id, "继续追问当前 idea", "deepseek")])
                        self.assertEqual(len(list_idea_sessions(db_path, limit=10)), 1)
                        self.assertEqual(len(list_idea_session_turns(db_path, session_id)), 0)
                    finally:
                        server.shutdown()
                        server.server_close()
                        thread.join(timeout=2)
            finally:
                os.chdir(old_cwd)

    def test_gui_review_loop_api_starts_bounded_refinement_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                self.write_gui_test_llm_env(cwd)
                from research_alpha.db import create_idea_session

                db_path = cwd / "data" / "research_alpha.db"
                session_id = create_idea_session(db_path, "agent eval", "Initial idea")
                release = threading.Event()
                captured_commands = []

                def fake_streaming(command, root_dir, on_output=None):
                    captured_commands.append(command)
                    if on_output:
                        on_output("stdout", "Review-loop summary JSON: {\"rounds_completed\":3}")
                    release.wait(2)
                    return {
                        "exit_code": 0,
                        "stdout": 'Review-loop summary JSON: {"rounds_completed":3,"requested_rounds":5,"final_idea":"Final idea"}',
                        "stderr": "",
                    }

                with patch("research_alpha.gui.run_app_command_streaming", side_effect=fake_streaming):
                    server = create_gui_http_server(cwd, host="127.0.0.1", port=0)
                    thread = threading.Thread(target=server.serve_forever, daemon=True)
                    thread.start()
                    try:
                        payload = {
                            "session_id": session_id,
                            "ui_scope": f"session:{session_id}",
                            "rounds": 5,
                            "provider": "deepseek",
                        }
                        status, body = self.gui_post_json(server, "/api/review-loop", payload)
                        self.assertEqual(status, 200)
                        self.assertEqual(body["job"]["kind"], "review-loop")
                        self.assertEqual(body["job"]["session_id"], session_id)
                        self.assertNotIn("Review-loop summary JSON", json.dumps(body, ensure_ascii=False))

                        status, duplicate = self.gui_post_json(server, "/api/review-loop", payload)
                        self.assertEqual(status, 200)
                        self.assertTrue(duplicate["deduped"])
                        self.assertEqual(duplicate["job"]["id"], body["job"]["id"])

                        for _ in range(50):
                            if captured_commands:
                                break
                            import time

                            time.sleep(0.01)
                        self.assertEqual(captured_commands[0], ["review-loop", "--session-id", str(session_id), "--rounds", "5", "--provider", "deepseek"])

                        status, too_low = self.gui_post_json(server, "/api/review-loop", {**payload, "rounds": 2})
                        self.assertEqual(status, 400)
                        self.assertIn("审稿循环轮数不能小于 3", str(too_low["error"]))

                        status, too_high = self.gui_post_json(server, "/api/review-loop", {**payload, "rounds": 6})
                        self.assertEqual(status, 400)
                        self.assertIn("审稿循环轮数不能大于 5", str(too_high["error"]))
                    finally:
                        release.set()
                        server.shutdown()
                        server.server_close()
                        thread.join(timeout=2)
            finally:
                os.chdir(old_cwd)

    def test_gui_session_delete_api_rejects_missing_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                state = GuiState()
                state.active_session_id = 999
                server = create_gui_http_server(cwd, host="127.0.0.1", port=0, state=state)
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                try:
                    status, body = self.gui_post_json(server, "/api/session/delete", {"session_id": 999})
                    self.assertEqual(status, 404)
                    self.assertEqual(body["error"], "session not found")
                    self.assertFalse(body["ok"])
                    self.assertEqual(body["status"], 404)
                    self.assertEqual(body["code"], "session_not_found")
                    self.assertIsNone(state.active_session_id)
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=2)
            finally:
                os.chdir(old_cwd)

    def test_gui_infers_research_brief_from_chat_text(self) -> None:
        brief = infer_research_brief("我想做科研智能体近半年的局限性，重点是可靠评测和失败发现")
        self.assertEqual(brief["user_direction"], "我想做科研智能体近半年的局限性，重点是可靠评测和失败发现")
        self.assertIn("科研智能体", brief["inferred_fields"])
        self.assertIn("可靠评测", brief["inferred_hotspots"])
        self.assertIn("失败发现", brief["inferred_hotspots"])
        self.assertIn("重点利用近半年优秀论文中明确写出的局限性", brief["inferred_constraints"])
        self.assertIn("scientific research agents", brief["frontier_query"])
        self.assertIn("failure modes", brief["frontier_query"])
        self.assertIn("Use these inferred tags only", brief["brief_direction"])

    def test_gui_infers_cv_query_without_agent_default(self) -> None:
        brief = infer_research_brief("给我一个伪装目标检测idea")
        self.assertIn("计算机视觉", brief["inferred_fields"])
        self.assertIn("伪装目标检测", brief["inferred_hotspots"])
        self.assertIn("camouflaged object detection", brief["frontier_query"])
        self.assertIn("concealed object detection", brief["frontier_query"])
        self.assertNotIn("AI agents", brief["frontier_query"])
        self.assertNotIn("evaluation benchmark", brief["frontier_query"])

    def test_prior_art_gate_response_parser(self) -> None:
        payload = parse_prior_art_gate_response(
            json.dumps(
                {
                    "decision": "duplicate_rethink",
                    "overlap_label": "duplicate",
                    "closest_work": [
                        {
                            "paper_id": 4,
                            "title": "Existing Benchmark",
                            "overlap_reason": "same setup",
                            "covered_parts": ["benchmark"],
                            "missing_parts": ["none"],
                            "difference_from_idea": "No real difference.",
                        }
                    ],
                    "renamed_only_risk": "high",
                    "novelty_boundary": "No novelty boundary.",
                    "why_not_done_yet": ["It was already done."],
                    "required_differentiation": ["Change mechanism and evidence target."],
                    "rethink_prompt": "Find a different failure boundary.",
                    "summary": "Too close to existing work.",
                }
            )
        )
        self.assertEqual(payload["decision"], "duplicate_rethink")
        self.assertEqual(payload["overlap_label"], "duplicate")
        self.assertEqual(payload["renamed_only_risk"], "high")
        self.assertEqual(payload["closest_work"][0]["difference_from_idea"], "No real difference.")

        with self.assertRaisesRegex(ValueError, "why this has plausibly not been done yet"):
            parse_prior_art_gate_response(
                json.dumps(
                    {
                        "decision": "pass",
                        "overlap_label": "distant",
                        "closest_work": [],
                        "renamed_only_risk": "low",
                        "novelty_boundary": "No close local match.",
                        "why_not_done_yet": [],
                        "required_differentiation": [],
                        "rethink_prompt": "",
                        "summary": "Looks novel but lacks why-not explanation.",
                    }
                )
            )

    def test_local_prior_art_returns_top_ten_and_novelty_boundary(self) -> None:
        idea = strict_idea_payload()
        papers = [
            {
                "id": index,
                "title": f"Stress-Tested Agent Benchmark {index}",
                "abstract": "Research agent evaluation stress tests hidden benchmark assumption failures.",
                "venue": "ICLR",
                "year": 2026,
            }
            for index in range(1, 13)
        ]
        result = analyze_prior_art(idea_payload=idea, papers=papers)
        self.assertLessEqual(len(result["top_matches"]), 10)
        self.assertIn("novelty_boundary", result)
        self.assertIn(result["renamed_only_risk"], {"low", "medium", "high"})
        self.assertIn("missing_parts", result["top_matches"][0])

    def test_gui_snapshot_exposes_prior_art_gate_details(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                main(["init"])
                payload = {
                    "idea_title": "Novel Agent Evaluation",
                    "prior_art_gate": {
                        "decision": "pass",
                        "summary": "No close prior-art coverage in stored evidence.",
                        "why_not_done_yet": ["Recent systems lacked the required long-horizon traces."],
                    },
                    "evidence_grounded_score": {"total": 7.5},
                }
                from research_alpha.db import add_candidate_idea

                add_candidate_idea(
                    cwd / "data" / "research_alpha.db",
                    query="research agents",
                    title="Novel Agent Evaluation",
                    content_json=json.dumps(payload),
                )
                snapshot = build_gui_snapshot(cwd, GuiState())
                self.assertEqual(snapshot["ideas"][0]["prior_art_gate_decision"], "pass")
                self.assertIn("No close", snapshot["ideas"][0]["prior_art_gate_summary"])
                self.assertEqual(snapshot["ideas"][0]["why_not_done_yet"][0], "Recent systems lacked the required long-horizon traces.")
                self.assertEqual(snapshot["ideas"][0]["evidence_score"], 7.5)
            finally:
                os.chdir(old_cwd)

    def test_gui_chat_snapshot_renders_session_turns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                main(["init"])
                main(["is", "--name", "agent evaluation", "--idea", "Initial idea"])
                with connect(cwd / "data" / "research_alpha.db") as conn:
                    conn.execute(
                        """
                        INSERT INTO idea_session_turns(
                            session_id, turn_index, user_instruction, decision, revised_idea, content_json
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            1,
                            1,
                            "请检查是否重复",
                            "partial",
                            "Revised idea with stronger novelty boundary",
                            json.dumps({"verdict_summary": "Needs sharper prior-art distance."}),
                        ),
                    )
                snapshot = build_chat_snapshot(cwd)
                self.assertTrue(snapshot["active"])
                texts = [item["text"] for item in snapshot["messages"]]
                self.assertIn("Initial idea", texts[0])
                self.assertIn("请检查是否重复", texts[1])
                self.assertIn("Revised idea with stronger novelty boundary", texts[2])
            finally:
                os.chdir(old_cwd)

    def test_gui_chat_snapshot_hides_review_loop_internal_turns_but_shows_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                main(["init"])
                from research_alpha.db import create_idea_session

                context = {
                    "review_history": [
                        {
                            "decision": "rethink_required",
                            "summary": "新颖性边界还不够硬。",
                            "fatal_flaws": ["像把已有方法换到新领域。"],
                            "required_rethink": ["改成先抽取优秀论文逻辑故事线再迁移。"],
                        }
                    ]
                }
                session_id = create_idea_session(
                    cwd / "data" / "research_alpha.db",
                    "agent eval",
                    "Initial idea",
                    context_json=json.dumps(context, ensure_ascii=False),
                )
                with connect(cwd / "data" / "research_alpha.db") as conn:
                    conn.execute(
                        """
                        INSERT INTO idea_session_turns(
                            session_id, turn_index, user_instruction, decision, revised_idea, content_json
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            session_id,
                            1,
                            "根据第 1/4 轮顶会审稿意见改写当前 idea。\n内部长审稿指令",
                            "partial",
                            "Final idea after reviewer loop",
                            json.dumps({"source": "review_loop_refinement", "hidden_in_gui_chat": True}, ensure_ascii=False),
                        ),
                    )
                    conn.execute(
                        "UPDATE idea_sessions SET current_idea=? WHERE id=?",
                        ("Final idea after reviewer loop", session_id),
                    )
                snapshot = build_chat_snapshot(cwd, session_id=session_id)
                texts = [item["text"] for item in snapshot["messages"]]
                joined = "\n".join(texts)
                self.assertIn("Initial idea", texts[0])
                self.assertNotIn("内部长审稿指令", joined)
                self.assertNotIn("根据第 1/4 轮顶会审稿意见", joined)
                self.assertIn("审稿意见：新颖性边界还不够硬。", joined)
                self.assertIn("主要漏洞：像把已有方法换到新领域。", joined)
                self.assertIn("Final idea after reviewer loop", texts[-1])
                self.assertEqual(snapshot["messages"][-1]["meta"], "final idea")
            finally:
                os.chdir(old_cwd)

    def test_gui_chat_snapshot_does_not_repeat_final_idea_after_visible_followup(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                main(["init"])
                from research_alpha.db import create_idea_session

                context = {"review_history": [{"summary": "先收紧 novelty 边界。"}]}
                session_id = create_idea_session(
                    cwd / "data" / "research_alpha.db",
                    "agent eval",
                    "Initial idea",
                    context_json=json.dumps(context, ensure_ascii=False),
                )
                with connect(cwd / "data" / "research_alpha.db") as conn:
                    conn.execute(
                        """
                        INSERT INTO idea_session_turns(
                            session_id, turn_index, user_instruction, decision, revised_idea, content_json
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            session_id,
                            1,
                            "根据第 1/4 轮顶会审稿意见改写当前 idea。",
                            "partial",
                            "Final idea after reviewer loop",
                            json.dumps({"source": "review_loop_refinement", "hidden_in_gui_chat": True}, ensure_ascii=False),
                        ),
                    )
                    conn.execute(
                        """
                        INSERT INTO idea_session_turns(
                            session_id, turn_index, user_instruction, decision, revised_idea, content_json
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            session_id,
                            2,
                            "把这个 idea 再具体一点",
                            "revised",
                            "Visible follow-up revision",
                            json.dumps({"source": "gui_stable_step"}, ensure_ascii=False),
                        ),
                    )
                    conn.execute(
                        "UPDATE idea_sessions SET current_idea=? WHERE id=?",
                        ("Visible follow-up revision", session_id),
                    )
                snapshot = build_chat_snapshot(cwd, session_id=session_id)
                messages = snapshot["messages"]
                texts = [item["text"] for item in messages]
                self.assertIn("把这个 idea 再具体一点", texts)
                self.assertIn("Visible follow-up revision", texts)
                self.assertEqual(texts.count("Visible follow-up revision"), 1)
                self.assertNotEqual(messages[-1]["meta"], "final idea")
            finally:
                os.chdir(old_cwd)

    def test_gui_chat_snapshot_redacts_paths_and_internal_prompt_tags(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                main(["init"])
                from research_alpha.db import add_idea_session_turn, create_idea_session

                session_id = create_idea_session(
                    cwd / "data" / "research_alpha.db",
                    name="agent evaluation /Users/example/name",
                    current_idea=(
                        "Initial idea\n"
                        "Inferred evidence-matching tags: fields=secret\n"
                        "Use these inferred tags only to retrieve and match evidence; do not treat them as a fixed idea template.\n"
                        "Dossier: /Users/example/project/outputs/session.json"
                    ),
                    context_json=json.dumps(
                        {
                            "paper_angle": "See /private/var/tmp/paper.json",
                            "frontier_gap": "Manifest: /Users/example/frontier/manifest.json",
                            "key_risk": "Database: /Users/example/project/data/research_alpha.db",
                            "why_not_done_yet": ["Output: /tmp/secret/why.json"],
                        }
                    ),
                )
                add_idea_session_turn(
                    cwd / "data" / "research_alpha.db",
                    session_id,
                    user_instruction="Please inspect /Users/example/request.txt",
                    decision="partial /private/var/tmp/decision.json",
                    revised_idea="Revised idea\nReport: /Users/example/report.md",
                    content_json=json.dumps(
                        {
                            "verdict_summary": "Manifest: /Users/example/verdict/manifest.json",
                            "coach_message": "Inferred evidence emphasis: hidden internal\nDossier: /private/var/tmp/coach.json",
                            "debug_payload": "should not appear",
                        }
                    ),
                )

                snapshot = build_chat_snapshot(cwd, session_id=session_id)
                serialized = json.dumps(snapshot, ensure_ascii=False)
                self.assertTrue(snapshot["active"])
                self.assertIn("[local path]", serialized)
                self.assertNotIn("/Users/", serialized)
                self.assertNotIn("/private/", serialized)
                self.assertNotIn("/tmp/secret", serialized)
                self.assertNotIn("research_alpha.db", serialized)
                self.assertNotIn("Inferred evidence-matching tags", serialized)
                self.assertNotIn("Inferred evidence emphasis", serialized)
                self.assertNotIn("Use these inferred tags only", serialized)
                self.assertNotIn("debug_payload", serialized)
                self.assertNotIn("should not appear", serialized)
            finally:
                os.chdir(old_cwd)

    def test_gui_chat_snapshot_never_falls_back_when_session_id_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                main(["init"])
                from research_alpha.db import add_idea_session_turn, create_idea_session

                db_path = cwd / "data" / "research_alpha.db"
                first_id = create_idea_session(db_path, "first session", "First seed idea")
                second_id = create_idea_session(db_path, "second session", "Second seed idea")
                add_idea_session_turn(
                    db_path,
                    first_id,
                    user_instruction="只属于第一条会话的追问",
                    decision="continue",
                    revised_idea="Only first revised idea",
                    content_json="{}",
                )
                add_idea_session_turn(
                    db_path,
                    second_id,
                    user_instruction="只属于第二条会话的追问",
                    decision="continue",
                    revised_idea="Only second revised idea",
                    content_json="{}",
                )

                first_snapshot = build_chat_snapshot(cwd, session_id=first_id)
                first_text = "\n".join(item["text"] for item in first_snapshot["messages"])
                self.assertTrue(first_snapshot["active"])
                self.assertEqual(first_snapshot["session"]["id"], first_id)
                self.assertIn("First seed idea", first_text)
                self.assertIn("只属于第一条会话", first_text)
                self.assertNotIn("Second seed idea", first_text)
                self.assertNotIn("只属于第二条会话", first_text)

                second_snapshot = build_chat_snapshot(cwd, session_id=second_id)
                second_text = "\n".join(item["text"] for item in second_snapshot["messages"])
                self.assertTrue(second_snapshot["active"])
                self.assertEqual(second_snapshot["session"]["id"], second_id)
                self.assertIn("Second seed idea", second_text)
                self.assertIn("只属于第二条会话", second_text)
                self.assertNotIn("First seed idea", second_text)
                self.assertNotIn("只属于第一条会话", second_text)

                missing_snapshot = build_chat_snapshot(cwd, session_id=9999)
                missing_text = "\n".join(item["text"] for item in missing_snapshot["messages"])
                self.assertFalse(missing_snapshot["active"])
                self.assertEqual(missing_snapshot["missing_session_id"], 9999)
                self.assertIn("没有找到这条历史对话", missing_text)
                self.assertNotIn("Second seed idea", missing_text)
            finally:
                os.chdir(old_cwd)

    def test_gui_snapshot_draft_mode_does_not_use_latest_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                main(["init"])
                from research_alpha.db import create_idea_session

                create_idea_session(cwd / "data" / "research_alpha.db", "latest session", "Latest idea should stay hidden")
                state = GuiState()
                snapshot = build_gui_snapshot(cwd, state, draft_mode=True)
                text = "\n".join(item["text"] for item in snapshot["chat"]["messages"])
                self.assertFalse(snapshot["chat"]["active"])
                self.assertTrue(snapshot["chat"]["draft"])
                self.assertIn("已准备好新对话", text)
                self.assertNotIn("Latest idea should stay hidden", text)
                self.assertIsNone(state.active_session_id)
            finally:
                os.chdir(old_cwd)


    def test_gui_chat_snapshot_keeps_seed_idea_clean_of_evidence_chain(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                main(["init"])
                from research_alpha.db import create_idea_session

                context = {
                    "paper_angle": "Best Paper A reframed evaluation as failure discovery.",
                    "historical_pattern": "hidden-assumption-reversal",
                    "frontier_gap": "Current agent benchmarks hide long-horizon failures.",
                    "trend_support": "Recent agent papers report brittle tool-use traces.",
                    "novelty": "Measure failure boundaries instead of aggregate success.",
                    "why_now": "Deployed agents now leave enough traces to audit.",
                    "key_risk": "Reviewers may ask whether this is only another benchmark.",
                    "why_not_done_yet": ["Older systems lacked realistic long-horizon traces."],
                    "review_history": [
                        {
                            "decision": "rethink_required",
                            "fatal_flaws": ["No strong evidence design."],
                            "main_attacks": ["Too incremental."],
                            "missing_logic_links": ["No failure boundary."],
                            "required_rethink": ["Rebuild around a stronger bottleneck."],
                            "review_path": str(cwd / "outputs" / "reviews" / "review-secret.json"),
                        }
                    ],
                }
                create_idea_session(
                    cwd / "data" / "research_alpha.db",
                    name="agent eval",
                    current_idea="Initial idea",
                    context_json=json.dumps(context),
                )
                snapshot = build_chat_snapshot(cwd, session_id=1)
                seed = snapshot["messages"][0]["text"]
                self.assertEqual(seed, "Initial idea")
                self.assertNotIn("依据", seed)
                self.assertNotIn("参考优秀论文思路", seed)
                self.assertNotIn("当前方法/趋势的问题", seed)
                self.assertNotIn("审稿防御要点", seed)
                self.assertNotIn("最可能被审稿人攻击的点", seed)
                self.assertNotIn("审稿人 rethink_required 攻击", seed)
                self.assertNotIn("审稿后下一步", seed)
                self.assertNotIn("review-secret", seed)
                self.assertNotIn(str(cwd), seed)
            finally:
                os.chdir(old_cwd)

    def test_gui_score_message_exposes_evidence_grounded_dimensions_in_chinese(self) -> None:
        text = build_gui_score_message(
            {
                "total": 7.4,
                "scale": "0-10",
                "rubric_source": "Derived strictly from stored high-quality evidence.",
                "dimensions": {
                    "innovation": {
                        "score": 8,
                        "explicit_evidence_covered": True,
                        "evidence": [{"source_type": "pattern", "source_id": "hidden-assumption-reversal"}],
                    },
                    "logic": {"score": 7, "explicit_evidence_covered": True, "evidence": []},
                    "feasibility": {"score": 6, "explicit_evidence_covered": False, "evidence": []},
                    "value": {"score": 8, "explicit_evidence_covered": True, "evidence": []},
                    "defensibility": {"score": 6, "explicit_evidence_covered": False, "evidence": []},
                },
            }
        )
        self.assertIn("证据驱动评分", text)
        self.assertIn("总分：7.4/0-10", text)
        self.assertIn("创新性：8/10", text)
        self.assertIn("逻辑性：7/10", text)
        self.assertIn("可行性：6/10", text)
        self.assertIn("价值：8/10", text)
        self.assertIn("抗审稿攻击性：6/10", text)
        self.assertIn("依据：pattern:hidden-assumption-reversal", text)
        self.assertIn("证据不足", text)

    def test_idea_prompt_requires_high_quality_evidence_basis(self) -> None:
        prompt = build_idea_prompt(
            query="research agents",
            idea_count=1,
            pattern_cards=[{"pattern_key": "hidden-assumption-reversal", "core_move": "Reverse the benchmark assumption."}],
            genome_cards=[{"paper_id": 7, "title": "Best Paper A", "problem_reframing": "Turn evaluation into failure discovery."}],
            trend_signal={"top_terms": [{"term": "agent evaluation"}]},
            quality_evidence={
                "high_weight_papers": [
                    {
                        "paper_id": 1,
                        "title": "Best Paper A",
                        "quality_signal": "best_paper",
                        "paper_weight": 8.0,
                    }
                ]
            },
        )
        self.assertIn("Hard constraint", prompt)
        self.assertIn("best-paper", prompt)
        self.assertIn("evidence_basis", prompt)
        self.assertIn("storyline_trace", prompt)
        self.assertIn("old belief -> bottleneck -> reframing", prompt)
        self.assertIn("quality_evidence", prompt)
        self.assertIn("Best Paper A", prompt)
        self.assertIn("allowed_source_ids", prompt)
        self.assertIn("hidden-assumption-reversal", prompt)
        self.assertIn("Do not transfer surface nouns, method names", prompt)
        self.assertIn("use the source paper method in the user domain", prompt)
        self.assertIn("Anti-copy discipline", prompt)
        self.assertIn("storyline_trace is an audit trail", prompt)
        self.assertIn("six steps should constrain the logic, not flatten the expression", prompt)
        self.assertIn("Idea-evaluation discipline", prompt)
        self.assertIn("hypothesis_discovery_rate", prompt)
        self.assertIn("Prefer evidence sources marked full_text_sections", prompt)

    def test_parse_idea_response_backfills_idea_evaluation_protocol(self) -> None:
        parsed = parse_idea_response(json.dumps({"ideas": [strict_idea_payload()]}))
        protocol = parsed[0]["idea_evaluation_protocol"]
        names = [item["name"] for item in protocol["dimensions"]]
        self.assertIn("practical_utility", names)
        self.assertIn("hypothesis_discovery_rate", names)
        self.assertIn("evidence_depth", names)
        self.assertTrue(protocol["reject_if"])

    def test_gold_evidence_tier_layers_source_use(self) -> None:
        self.assertEqual(gold_evidence_tier({"award": "best_paper"})["tier"], "gold_complete_logic")
        self.assertEqual(gold_evidence_tier({"award": "oral"})["tier"], "gold_reviewer_clarity")
        self.assertIn(
            "trend_context_only",
            gold_evidence_tier({"award": "spotlight"})["allowed_standard_use"],
        )

    def test_idea_prompt_includes_user_library_as_non_scoring_domain_knowledge(self) -> None:
        prompt = build_idea_prompt(
            query="medical research agents",
            idea_count=1,
            pattern_cards=[{"pattern_key": "hidden-assumption-reversal", "core_move": "Reverse the benchmark assumption."}],
            genome_cards=[{"paper_id": 7, "title": "Best Paper A", "problem_reframing": "Turn evaluation into failure discovery."}],
            trend_signal={"top_terms": [{"term": "agent evaluation"}]},
            quality_evidence={
                "high_weight_papers": [
                    {
                        "paper_id": 1,
                        "title": "Best Paper A",
                        "quality_signal": "best_paper",
                        "paper_weight": 8.0,
                    }
                ]
            },
            domain_knowledge={
                "source_policy": "user_library_domain_knowledge_only",
                "papers": [
                    {
                        "paper_id": 99,
                        "title": "User Medical Route Paper",
                        "route_note": "Clinical trial route terminology.",
                    }
                ],
            },
        )
        self.assertIn("domain_knowledge", prompt)
        self.assertIn("User Medical Route Paper", prompt)
        self.assertIn("user_library_domain_knowledge_only", prompt)
        self.assertIn("never as evidence_basis", prompt)
        context = json.loads(prompt.split("Context:\n", 1)[1])
        self.assertEqual(context["domain_knowledge"]["papers"][0]["title"], "User Medical Route Paper")
        self.assertNotIn("User Medical Route Paper", context["allowed_source_ids"]["paper"])
        self.assertNotIn("99", context["allowed_source_ids"]["paper"])

    def test_storyline_migration_audit_rejects_copied_source_steps(self) -> None:
        payload = {
            "idea_title": "Copied Method Transfer",
            "storyline_trace": strict_storyline_trace(),
        }
        payload["storyline_trace"]["reframing"]["transfer_to_current_hotspot"] = (
            payload["storyline_trace"]["reframing"]["paper_story_standard"]
        )
        audit = audit_storyline_migration(
            payload,
            [{"source_type": "pattern", "pattern_key": "hidden-assumption-reversal"}],
        )
        self.assertFalse(audit["valid"])
        self.assertEqual(audit["status"], "storyline_migration_invalid")
        self.assertTrue(any(item["reason"] == "transfer_repeats_source_story_standard" for item in audit["failures"]))

    def test_storyline_migration_audit_accepts_domain_specific_rewrite(self) -> None:
        payload = {
            "idea_title": "Stress-Tested Agent Benchmark",
            "storyline_trace": strict_storyline_trace(),
        }
        audit = audit_storyline_migration(
            payload,
            [{"source_type": "pattern", "pattern_key": "hidden-assumption-reversal"}],
        )
        self.assertTrue(audit["valid"])
        self.assertEqual(audit["status"], "storyline_migration_valid")

    def test_storyline_migration_audit_rejects_source_surface_method_leakage(self) -> None:
        payload = {
            "idea_title": "Masked DETR Prototype for Camouflaged Object Detection",
            "core_hypothesis": "Apply masked attention, query denoising, transformer decoder prototypes, and panoptic segmentation masks to camouflaged object detection.",
            "historical_pattern": "hidden-assumption-reversal",
            "frontier_gap": "Camouflaged detection needs better prototypes.",
            "why_now": "Vision systems now need stronger hidden-object evaluation.",
            "novelty": "Uses masked attention and query denoising as the new mechanism.",
            "value": "Could improve panoptic segmentation style masks for hidden objects.",
            "key_risk": "It may be only a method transplant.",
            "first_experiments": ["Train a transformer decoder prototype with masked attention and query denoising."],
            "evaluation_outline": "Compare panoptic mask quality and prototype decoder behavior.",
            "paper_angle": "Masked DETR style query denoising is moved to camouflaged detection.",
            "storyline_trace": strict_storyline_trace("Masked DETR"),
        }
        source = {
            "source_type": "paper",
            "title": "Masked DETR",
            "abstract": (
                "Masked DETR uses masked attention, query denoising, transformer decoder prototypes, "
                "and panoptic segmentation masks for object-centric detection."
            ),
        }
        audit = audit_storyline_migration(payload, [source])
        self.assertFalse(audit["valid"])
        self.assertTrue(any(item["reason"] == "visible_idea_reuses_source_surface_terms" for item in audit["failures"]))

    def test_storyline_migration_audit_allows_abstract_logic_without_source_method_terms(self) -> None:
        payload = {
            "idea_title": "Uncertainty-Boundary Evaluation for Camouflaged Perception",
            "core_hypothesis": "Camouflaged perception should be judged by whether systems expose hidden uncertainty boundaries before localization confidence becomes misleading.",
            "historical_pattern": "hidden-assumption-reversal",
            "frontier_gap": "Current hidden-object evaluations over-trust aggregate accuracy and under-measure ambiguous failure boundaries.",
            "why_now": "Foundation perception models are being used in harder scenes where hidden uncertainty matters.",
            "novelty": "Moves the story arc from confident recognition to explicit boundary discovery without copying the source mechanism.",
            "value": "Could reveal when apparently strong detectors fail under ambiguity.",
            "key_risk": "The benchmark must avoid becoming another leaderboard without a causal failure story.",
            "first_experiments": ["Build ambiguity strata and measure whether confidence shifts before localization errors appear."],
            "evaluation_outline": "Compare model rankings under ordinary accuracy and uncertainty-boundary stress cases.",
            "paper_angle": "Reframe hidden-object detection as boundary discovery rather than better mask fitting.",
            "storyline_trace": strict_storyline_trace("Masked DETR"),
        }
        source = {
            "source_type": "paper",
            "title": "Masked DETR",
            "abstract": (
                "Masked DETR uses masked attention, query denoising, transformer decoder prototypes, "
                "and panoptic segmentation masks for object-centric detection."
            ),
        }
        audit = audit_storyline_migration(payload, [source])
        self.assertTrue(audit["valid"])

    def test_idea_generation_failure_summary_classifies_storyline_hard_transfer(self) -> None:
        summary = build_idea_generation_failure_summary(
            query="camouflaged object detection",
            requested=2,
            rejected_ideas=[
                {
                    "title": "Copied transfer",
                    "reason": "Generated idea `Copied transfer` failed storyline migration audit: reframing: transfer_repeats_source_story_standard. It must migrate the abstract paper story, not hard-transfer the source method.",
                }
            ],
        )
        self.assertEqual(summary["status"], "idea_generation_not_ready")
        self.assertEqual(summary["primary_reason"], "storyline_migration_failed")
        self.assertIn("不要复用源论文方法名", summary["next_action"])
        self.assertEqual(summary["recovery_strategy"], "storyline_first_retry")
        self.assertIn("storyline_migration", summary["blocked_dimensions"])
        self.assertIn("source_surface_leakage", summary["blocked_dimensions"])
        self.assertTrue(any("六步逻辑" in item for item in summary["retry_prompt_constraints"]))
        self.assertTrue(any("不复用源论文方法名" in item for item in summary["retry_prompt_constraints"]))

    def test_idea_generation_failure_summary_classifies_user_library_misuse(self) -> None:
        summary = build_idea_generation_failure_summary(
            query="clinical research agents",
            requested=1,
            rejected_ideas=[
                {
                    "title": "Bad User Evidence Idea",
                    "reason": (
                        "candidate_used_user_library_as_scoring_evidence: "
                        "用户论文库只能作为领域知识/路线学习，不能进入 evidence_basis 或 idea 评分标准。"
                    ),
                }
            ],
        )
        self.assertEqual(summary["status"], "idea_generation_not_ready")
        self.assertEqual(summary["primary_reason"], "user_library_used_as_standard")
        self.assertIn("用户论文库作为领域背景", summary["next_action"])
        self.assertIn("Gold/Pattern/Genome", summary["next_action"])
        self.assertEqual(summary["recovery_strategy"], "domain_context_only_retry")
        self.assertIn("user_library_boundary", summary["blocked_dimensions"])
        self.assertIn("Bad User Evidence Idea", summary["rejected_ideas"][0]["title"])

    def test_idea_generation_semantic_retry_policy_excludes_prior_art_duplicates(self) -> None:
        storyline_summary = build_idea_generation_failure_summary(
            query="research agents",
            requested=1,
            rejected_ideas=[
                {
                    "title": "Copied transfer",
                    "reason": "Generated idea failed storyline migration audit: visible_idea_reuses_source_surface_terms.",
                }
            ],
        )
        duplicate_summary = build_idea_generation_failure_summary(
            query="research agents",
            requested=1,
            rejected_ideas=[
                {
                    "title": "Duplicate",
                    "reason": "prior_art_duplicate_rethink: too close to local prior art.",
                }
            ],
        )
        mixed_summary = build_idea_generation_failure_summary(
            query="research agents",
            requested=2,
            rejected_ideas=[
                {
                    "title": "Copied transfer",
                    "reason": "Generated idea failed storyline migration audit: visible_idea_reuses_source_surface_terms.",
                },
                {
                    "title": "Duplicate",
                    "reason": "prior_art_duplicate_rethink: too close to local prior art.",
                },
            ],
        )
        self.assertTrue(idea_generation_failure_is_semantic_retryable(storyline_summary))
        self.assertFalse(idea_generation_failure_is_semantic_retryable(duplicate_summary))
        self.assertFalse(idea_generation_failure_is_semantic_retryable(mixed_summary))

    def test_user_library_reference_rejection_normalizes_source_alias_spacing(self) -> None:
        payload = {
            "idea_title": "Bad User Evidence Idea",
            "domain_knowledge_context": {
                "papers": [
                    {
                        "paper_id": 99,
                        "title": "User Medical Route Paper",
                        "external_ref": "https://arxiv.org/abs/2601.09999",
                    }
                ]
            },
            "evidence_basis": [
                {
                    "source_type": "paper",
                    "source_id": "User   Medical\nRoute\tPaper",
                    "borrowed_standard": "Use the user paper as a standard.",
                    "used_for": "innovation|logic|feasibility|value|defensibility",
                }
            ],
            "storyline_trace": strict_storyline_trace("hidden-assumption-reversal"),
        }
        reason = user_library_reference_rejection_reason(payload)
        self.assertIn("candidate_used_user_library_as_scoring_evidence", reason)

        payload["evidence_basis"] = strict_evidence_basis()
        payload["storyline_trace"]["reframing"]["source_id"] = "https://arxiv.org/abs/2601.09999"
        reason = user_library_reference_rejection_reason(payload)
        self.assertIn("candidate_used_user_library_as_storyline_source", reason)

    def test_idea_prompt_supports_chinese_and_recent_limitations(self) -> None:
        prompt = build_idea_prompt(
            query="科研智能体可靠评测（请用中文输出idea标题、解释、风险和实验计划）",
            idea_count=1,
            pattern_cards=[{"pattern_key": "hidden-assumption-reversal"}],
            genome_cards=[{"paper_id": 1, "title": "Gold Paper"}],
            trend_signal={
                "top_terms": [{"term": "agent evaluation"}],
                "recent_limitations": {
                    "papers": [
                        {
                            "title": "Recent Agent Paper",
                            "explicit_limitations": [{"text": "The method is limited by sparse long-horizon failures."}],
                        }
                    ]
                },
            },
            quality_evidence={"high_weight_papers": [{"paper_id": 1, "title": "Gold Paper"}]},
        )
        self.assertIn("return Chinese idea titles", prompt)
        self.assertIn("Recent limitation signals", prompt)
        self.assertIn("sparse long-horizon failures", prompt)

    def test_limitations_extracts_only_explicit_metadata_sentences(self) -> None:
        payload_text = limitations_json_for_paper(
            {
                "title": "Agent Evaluation Limits",
                "abstract": (
                    "We introduce a benchmark for research agents. "
                    "However, the current setting is limited by synthetic tasks and cannot measure long-horizon failures."
                ),
                "venue": "ICLR",
                "year": 2026,
                "publication_date": "2026-02-01",
            }
        )
        payload = json.loads(payload_text)
        texts = [item["text"] for item in payload["explicit_limitations"]]
        self.assertEqual(len(texts), 1)
        self.assertIn("limited by synthetic tasks", texts[0])

    def test_chinese_direction_gets_scholarly_search_query(self) -> None:
        query = scholarly_search_query("科研智能体可靠评测")
        self.assertIn("research", query)
        self.assertIn("agents", query)
        self.assertIn("evaluation", query)
        self.assertIn("benchmark", query)

    def test_quality_enrich_prompt_is_metadata_only(self) -> None:
        prompt = build_quality_enrich_prompt(
            [
                {
                    "paper_id": 1,
                    "title": "Famous Paper",
                    "venue": "ICLR",
                    "year": 2024,
                    "citation_count": 1000,
                }
            ]
        )
        self.assertIn("quality metadata only", prompt)
        self.assertIn("Do not invent paper content", prompt)
        self.assertIn("best_paper", prompt)
        annotations = parse_quality_enrich_response(
            json.dumps(
                {
                    "annotations": [
                        {
                            "paper_id": 1,
                            "label": "oral",
                            "confidence": 0.8,
                            "rationale": "Known oral presentation.",
                            "needs_verification": True,
                        }
                    ]
                }
            )
        )
        self.assertEqual(annotations[0]["label"], "oral")

    def test_quality_enrich_apply_skips_labels_that_need_verification(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                main(["init"])
                db_path = cwd / "data" / "research_alpha.db"
                paper_id = add_paper(
                    db_path,
                    "Uncertain Oral Candidate",
                    "ICLR",
                    2026,
                    "A candidate whose presentation metadata is not confirmed.",
                    source_kind="frontier_openalex",
                    citation_count=100,
                )
                fake_response = json.dumps(
                    {
                        "annotations": [
                            {
                                "paper_id": paper_id,
                                "label": "oral",
                                "confidence": 0.95,
                                "rationale": "Maybe an oral.",
                                "needs_verification": True,
                            }
                        ]
                    }
                )
                with patch("research_alpha.llm.LLMClient.chat", return_value=type("Resp", (), {"text": fake_response})()):
                    stdout = io.StringIO()
                    with redirect_stdout(stdout):
                        exit_code = main(["quality-enrich", "--limit", "5", "--apply"])
                self.assertEqual(exit_code, 0)
                self.assertIn("Applied labels: 0", stdout.getvalue())
                self.assertIn("Skipped labels: 1", stdout.getvalue())
                with connect(db_path) as conn:
                    row = conn.execute(
                        "select source_kind, award, paper_weight from papers where id=?",
                        (paper_id,),
                    ).fetchone()
                self.assertEqual(row["source_kind"], "frontier_openalex")
                self.assertEqual(row["award"], "")
                self.assertEqual(float(row["paper_weight"]), 0.0)
                payload = json.loads((cwd / "outputs" / "quality" / "quality_enrichment.json").read_text(encoding="utf-8"))
                self.assertEqual(payload["skipped"][0]["skip_reason"], "needs_verification")
            finally:
                os.chdir(old_cwd)

    def test_quality_enrich_apply_requires_explicit_quality_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                main(["init"])
                db_path = cwd / "data" / "research_alpha.db"
                paper_id = add_paper(
                    db_path,
                    "Memory-Only Oral Claim",
                    "ICLR",
                    2026,
                    "A candidate whose supplied metadata does not mention an oral presentation.",
                    source_kind="frontier_openalex",
                    citation_count=100,
                )
                fake_response = json.dumps(
                    {
                        "annotations": [
                            {
                                "paper_id": paper_id,
                                "label": "oral",
                                "confidence": 0.99,
                                "rationale": "I remember this may be an oral.",
                                "needs_verification": False,
                            }
                        ]
                    }
                )
                with patch("research_alpha.llm.LLMClient.chat", return_value=type("Resp", (), {"text": fake_response})()):
                    stdout = io.StringIO()
                    with redirect_stdout(stdout):
                        exit_code = main(["quality-enrich", "--limit", "5", "--apply"])
                self.assertEqual(exit_code, 0)
                self.assertIn("Applied labels: 0", stdout.getvalue())
                with connect(db_path) as conn:
                    row = conn.execute(
                        "select source_kind, award, paper_weight from papers where id=?",
                        (paper_id,),
                    ).fetchone()
                self.assertEqual(row["source_kind"], "frontier_openalex")
                self.assertEqual(row["award"], "")
                self.assertEqual(float(row["paper_weight"]), 0.0)
                payload = json.loads((cwd / "outputs" / "quality" / "quality_enrichment.json").read_text(encoding="utf-8"))
                self.assertEqual(payload["skipped"][0]["skip_reason"], "missing_explicit_quality_metadata")
            finally:
                os.chdir(old_cwd)

    def test_quality_enrich_apply_accepts_explicit_oral_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                main(["init"])
                db_path = cwd / "data" / "research_alpha.db"
                paper_id = add_paper(
                    db_path,
                    "Explicit Oral Candidate",
                    "ICLR 2026 Oral",
                    2026,
                    "A candidate whose supplied metadata explicitly says oral.",
                    source_kind="frontier_openalex",
                    citation_count=100,
                )
                fake_response = json.dumps(
                    {
                        "annotations": [
                            {
                                "paper_id": paper_id,
                                "label": "oral",
                                "confidence": 0.95,
                                "rationale": "The supplied venue metadata says ICLR 2026 Oral.",
                                "needs_verification": False,
                            }
                        ]
                    }
                )
                with patch("research_alpha.llm.LLMClient.chat", return_value=type("Resp", (), {"text": fake_response})()):
                    stdout = io.StringIO()
                    with redirect_stdout(stdout):
                        exit_code = main(["quality-enrich", "--limit", "5", "--apply"])
                self.assertEqual(exit_code, 0)
                self.assertIn("Applied labels: 1", stdout.getvalue())
                with connect(db_path) as conn:
                    row = conn.execute(
                        "select source_kind, award, paper_weight from papers where id=?",
                        (paper_id,),
                    ).fetchone()
                self.assertEqual(row["source_kind"], "gold_openalex")
                self.assertEqual(row["award"], "oral")
                self.assertGreater(float(row["paper_weight"]), 0.0)
            finally:
                os.chdir(old_cwd)

    def test_reviewer_gate_prompt_and_rethink_parse(self) -> None:
        prompt = build_reviewer_gate_prompt(
            idea_payload={"idea_title": "Stress Benchmark", "core_hypothesis": "Test failures."},
            pattern_cards=[{"pattern_key": "hidden-assumption-reversal"}],
            genome_cards=[{"paper_id": 1, "title": "Gold Paper"}],
            evidence_score={"total": 6.0},
        )
        self.assertIn("scoring expert", prompt)
        self.assertIn("pattern-logic expert", prompt)
        self.assertIn("top-conference reviewer", prompt)
        payload = parse_reviewer_gate_response(
            json.dumps(
                {
                    "decision": "rethink_required",
                    "score_expert": {"innovation": 4, "logic": 3, "feasibility": 6, "value": 5, "defensibility": 2, "overall": 4},
                    "pattern_logic_expert": {
                        "matched_storyline": "Weak match.",
                        "missing_logic_links": ["No failure boundary."],
                        "gold_standard_alignment": "Low.",
                    },
                    "top_conference_reviewer": {
                        "main_attacks": ["Novelty unclear."],
                        "fatal_flaws": ["No evidence design."],
                        "questions": ["What changes?"],
                    },
                    "rethink_trigger": "Fatal evidence-design gap.",
                    "required_rethink": ["Rebuild the evidence design."],
                    "summary": "Needs rethinking.",
                }
            )
        )
        self.assertEqual(payload["decision"], "rethink_required")

    def test_reviewer_gate_prompt_uses_chinese_for_chinese_session(self) -> None:
        prompt = build_reviewer_gate_prompt(
            idea_payload={"idea_title": "科研智能体可靠评测", "core_hypothesis": "用中文审稿。"},
            pattern_cards=[],
            genome_cards=[],
            evidence_score={"total": 4.0},
        )
        self.assertIn("Return every natural-language JSON field in Chinese", prompt)
        self.assertIn('"preferred_language": "zh"', prompt)

    def test_generation_evidence_audit_distinguishes_valid_missing_and_unmatched_basis(self) -> None:
        sources = [
            {
                "source_type": "pattern",
                "pattern_key": "hidden-assumption-reversal",
                "core_move": "Reverse the benchmark assumption.",
                "paper_weight": 8.0,
                "award": "best_paper",
            }
        ]
        valid_payload = {
            "idea_title": "Stress-Tested Agent Benchmark",
            "historical_pattern": "hidden-assumption-reversal",
            "evidence_basis": [
                {
                    "source_type": "pattern",
                    "source_id": "hidden-assumption-reversal",
                    "quality_signal": "best_paper",
                    "borrowed_standard": "Reverse the benchmark assumption.",
                    "used_for": "innovation|logic|feasibility|value|defensibility",
                }
            ],
        }
        valid_audit = audit_generation_evidence(valid_payload, sources)
        self.assertEqual(valid_audit["status"], "explicit_valid")
        self.assertEqual(valid_audit["penalty"], 0.0)
        self.assertEqual(valid_audit["matched_basis_count"], 1)

        type_mismatch_payload = {
            "idea_title": "Mislabeled source type",
            "historical_pattern": "hidden-assumption-reversal",
            "evidence_basis": [
                {
                    "source_type": "paper",
                    "source_id": "hidden-assumption-reversal",
                    "quality_signal": "best_paper",
                    "borrowed_standard": "Reverse the benchmark assumption.",
                    "used_for": "innovation|logic|feasibility|value|defensibility",
                }
            ],
        }
        mismatch_audit = audit_generation_evidence(type_mismatch_payload, sources)
        self.assertEqual(mismatch_audit["status"], "explicit_unmatched")
        self.assertEqual(mismatch_audit["matched_basis_count"], 0)
        self.assertEqual(mismatch_audit["source_type_mismatch_count"], 1)
        self.assertEqual(mismatch_audit["source_type_mismatches"][0]["matched_source_type"], "pattern")

        missing_audit = audit_generation_evidence({"idea_title": "No cited evidence"}, sources)
        self.assertEqual(missing_audit["status"], "missing_model_evidence_inferred")
        self.assertLess(missing_audit["penalty"], 0)

        unmatched_audit = audit_generation_evidence(
            {
                "idea_title": "Hallucinated source",
                "evidence_basis": [{"source_id": "not-in-db", "borrowed_standard": "something"}],
            },
            sources,
        )
        self.assertEqual(unmatched_audit["status"], "explicit_unmatched")
        self.assertEqual(unmatched_audit["matched_basis_count"], 0)

    def test_generation_evidence_audit_accepts_stable_source_aliases(self) -> None:
        sources = [
            {
                "source_type": "genome",
                "paper_id": 7,
                "title": "Best Paper A",
                "problem_reframing": "Turn evaluation into failure discovery.",
                "paper_weight": 5.0,
            },
            {
                "source_type": "pattern",
                "pattern_key": "hidden-assumption-reversal",
                "pattern_name": "Hidden Assumption Reversal",
                "canonical_examples": ["Genome A"],
            },
        ]
        audit = audit_generation_evidence(
            {
                "idea_title": "Alias Test",
                "evidence_basis": [
                    {
                        "source_type": "genome",
                        "source_id": "paper-7",
                        "quality_signal": "best_paper",
                        "borrowed_standard": "Turn evaluation into failure discovery.",
                        "used_for": "innovation|logic|feasibility|value|defensibility",
                    }
                ],
            },
            sources,
        )
        self.assertEqual(audit["status"], "explicit_valid")
        self.assertEqual(audit["matched_basis_count"], 1)
        self.assertIn("evidence_depth", audit)

    def test_generation_evidence_audit_prefers_declared_type_when_alias_collides(self) -> None:
        sources = [
            {
                "source_type": "genome",
                "paper_id": 7,
                "title": "Best Paper A",
                "problem_reframing": "Turn evaluation into failure discovery.",
                "paper_weight": 5.0,
            },
            {
                "source_type": "pattern",
                "pattern_key": "paper-7",
                "core_move": "A pattern whose key collides with a genome alias.",
                "paper_weight": 8.0,
            },
        ]
        audit = audit_generation_evidence(
            {
                "idea_title": "Alias Collision Test",
                "evidence_basis": [
                    {
                        "source_type": "genome",
                        "source_id": "paper-7",
                        "quality_signal": "best_paper",
                        "borrowed_standard": "Turn evaluation into failure discovery.",
                        "used_for": "innovation|logic|feasibility|value|defensibility",
                    }
                ],
            },
            sources,
        )
        self.assertEqual(audit["status"], "explicit_valid")
        self.assertEqual(audit["matched_basis_count"], 1)
        self.assertEqual(audit["matched_basis"][0]["matched_source_type"], "genome")

    def test_idea_parser_rejects_invalid_evidence_enums(self) -> None:
        base_idea = {
            "idea_title": "Stress-Tested Agent Benchmark",
            "core_hypothesis": "Agent benchmarks hide brittleness.",
            "historical_pattern": "hidden-assumption-reversal",
            "trend_support": "Agents need stronger evaluation.",
            "frontier_gap": "Current benchmarks under-measure failure modes.",
            "why_now": "Agent systems are now deployed widely.",
            "novelty": "Reframes benchmark design around hidden brittleness.",
            "value": "Could reset how research agents are evaluated.",
            "key_risk": "Benchmark churn without theory.",
            "first_experiments": ["Build failure-taxonomy tasks"],
            "evaluation_outline": "Measure ranking changes and failure-mode coverage.",
            "paper_angle": "Benchmarks reward the wrong competence.",
            "evidence_basis": strict_evidence_basis(),
            "generation_guardrail": strict_generation_guardrail(),
            "storyline_trace": strict_storyline_trace(),
        }
        bad_source = json.loads(json.dumps(base_idea))
        bad_source["evidence_basis"][0]["source_type"] = "blog"
        with self.assertRaisesRegex(ValueError, "source_type"):
            parse_idea_response(json.dumps({"ideas": [bad_source]}))

        bad_use = json.loads(json.dumps(base_idea))
        bad_use["evidence_basis"][0]["used_for"] = "innovation|vibes"
        with self.assertRaisesRegex(ValueError, "unsupported dimensions"):
            parse_idea_response(json.dumps({"ideas": [bad_use]}))

        missing_dimensions = json.loads(json.dumps(base_idea))
        missing_dimensions["evidence_basis"][0]["used_for"] = "logic"
        with self.assertRaisesRegex(ValueError, "missing: defensibility, feasibility, innovation, value"):
            parse_idea_response(json.dumps({"ideas": [missing_dimensions]}))

        missing_storyline = {key: value for key, value in base_idea.items() if key != "storyline_trace"}
        with self.assertRaisesRegex(ValueError, "storyline_trace"):
            parse_idea_response(json.dumps({"ideas": [missing_storyline]}))

        duplicate = json.loads(json.dumps(base_idea))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_idea_response(json.dumps({"ideas": [base_idea, duplicate]}))

    def test_idea_parser_rejects_thin_placeholder_content(self) -> None:
        thin_idea = {
            "idea_title": "Idea",
            "core_hypothesis": "ok",
            "historical_pattern": "hidden-assumption-reversal",
            "trend_support": "Agent evaluation is rising.",
            "frontier_gap": "ok",
            "why_now": "ok",
            "novelty": "ok",
            "value": "ok",
            "key_risk": "ok",
            "first_experiments": ["ok"],
            "evaluation_outline": "ok",
            "paper_angle": "ok",
            "evidence_basis": strict_evidence_basis(),
            "generation_guardrail": strict_generation_guardrail(),
            "storyline_trace": strict_storyline_trace(),
        }
        with self.assertRaisesRegex(ValueError, "content_quality_failed"):
            parse_idea_response(json.dumps({"ideas": [thin_idea]}))

    def test_evidence_grounded_score_penalizes_missing_model_evidence_basis(self) -> None:
        pattern_rows = [
            {
                "pattern_key": "hidden-assumption-reversal",
                "content_json": json.dumps(
                    {
                        "core_move": "Reverse the benchmark assumption.",
                        "operator_template": "Turn success metrics into failure discovery.",
                        "when_to_use": "When benchmarks hide brittleness.",
                    }
                ),
            }
        ]
        explicit_payload = {
            "idea_title": "Stress-Tested Agent Benchmark",
            "core_hypothesis": "Agent benchmarks hide brittleness.",
            "historical_pattern": "hidden-assumption-reversal",
            "frontier_gap": "Benchmarks hide brittleness.",
            "evidence_basis": [
                {
                    "source_type": "pattern",
                    "source_id": "hidden-assumption-reversal",
                    "quality_signal": "best_paper",
                    "borrowed_standard": "Reverse the benchmark assumption.",
                    "used_for": "innovation|logic|feasibility|value|defensibility",
                }
            ],
        }
        missing_payload = {key: value for key, value in explicit_payload.items() if key != "evidence_basis"}
        explicit_score = build_evidence_grounded_score(explicit_payload, pattern_rows=pattern_rows)
        missing_score = build_evidence_grounded_score(missing_payload, pattern_rows=pattern_rows)
        self.assertEqual(explicit_score["generation_evidence_audit"]["status"], "explicit_valid")
        self.assertEqual(missing_score["generation_evidence_audit"]["status"], "missing_model_evidence_inferred")
        self.assertEqual(explicit_score["generation_evidence_audit"]["missing_uses"], [])
        self.assertIn("logic", missing_score["generation_evidence_audit"]["missing_uses"])
        self.assertTrue(explicit_score["dimensions"]["logic"]["explicit_evidence_covered"])
        self.assertFalse(missing_score["dimensions"]["logic"]["explicit_evidence_covered"])
        self.assertGreater(explicit_score["total"], missing_score["total"])

    def test_evidence_grounded_score_prefers_full_text_depth(self) -> None:
        base_content = {
            "paper_summary": "Agent benchmarks hide brittle failure assumptions.",
            "problem_reframing": "Turn benchmark success into failure discovery.",
            "transferable_pattern": "Hidden assumption reversal for evaluation.",
            "evidence_design": "Compare ranking shifts and failure coverage.",
            "failure_boundary": "Fails when no trace is observable.",
        }
        abstract_genome_rows = [
            {
                "paper_id": 7,
                "title": "Depth Paper",
                "venue": "ICLR",
                "year": 2026,
                "award": "best_paper",
                "paper_weight": 5.0,
                "source_kind": "gold_openreview",
                "evidence_level": "abstract_only",
                "content_json": json.dumps(base_content),
            }
        ]
        full_text_genome_rows = [
            {
                **abstract_genome_rows[0],
                "evidence_level": "full_text_sections",
                "content_json": json.dumps({**base_content, "full_text_sections_used": ["introduction", "experiments"]}),
            }
        ]
        payload = {
            "idea_title": "Stress-Tested Agent Benchmark",
            "core_hypothesis": "Agent benchmarks hide brittle failure assumptions.",
            "historical_pattern": "Hidden assumption reversal for evaluation.",
            "frontier_gap": "Benchmark success misses failure coverage.",
            "evaluation_outline": "Compare ranking shifts and failure coverage.",
            "key_risk": "No trace is observable.",
            "evidence_basis": [
                {
                    "source_type": "genome",
                    "source_id": "paper-7",
                    "quality_signal": "best_paper",
                    "borrowed_standard": "Turn benchmark success into failure discovery.",
                    "used_for": "innovation|logic|feasibility|value|defensibility",
                }
            ],
        }
        abstract_score = build_evidence_grounded_score(payload, genome_rows=abstract_genome_rows)
        full_text_score = build_evidence_grounded_score(payload, genome_rows=full_text_genome_rows)
        self.assertEqual(abstract_score["evidence_depth"]["status"], "abstract_only_grounded")
        self.assertEqual(full_text_score["evidence_depth"]["status"], "full_text_grounded")
        self.assertGreater(full_text_score["total"], abstract_score["total"])
        matched = full_text_score["generation_evidence_audit"]["matched_basis"][0]
        self.assertEqual(matched["matched_evidence_depth"]["status"], "full_text_sections")

    def test_evidence_grounded_score_marks_missing_dimension_coverage(self) -> None:
        pattern_rows = [
            {
                "pattern_key": "hidden-assumption-reversal",
                "content_json": json.dumps(
                    {
                        "core_move": "Reverse the benchmark assumption.",
                        "operator_template": "Turn success metrics into failure discovery.",
                        "when_to_use": "When benchmarks hide brittleness.",
                    }
                ),
            }
        ]
        payload = {
            "idea_title": "Stress-Tested Agent Benchmark",
            "core_hypothesis": "Agent benchmarks hide brittleness.",
            "historical_pattern": "hidden-assumption-reversal",
            "frontier_gap": "Benchmarks hide brittleness.",
            "evidence_basis": strict_evidence_basis(used_for="innovation"),
        }
        score = build_evidence_grounded_score(payload, pattern_rows=pattern_rows)
        self.assertEqual(score["generation_evidence_audit"]["status"], "explicit_valid")
        self.assertIn("logic", score["generation_evidence_audit"]["missing_uses"])
        self.assertTrue(score["dimensions"]["innovation"]["explicit_evidence_covered"])
        self.assertFalse(score["dimensions"]["logic"]["explicit_evidence_covered"])

    def test_frontier_paper_is_not_accepted_as_scoring_standard(self) -> None:
        payload = {
            "idea_title": "Agent Idea",
            "core_hypothesis": "Agents need better adaptation.",
            "historical_pattern": "hidden-assumption-reversal",
            "frontier_gap": "Agents are rising.",
            "evidence_basis": [
                {
                    "source_type": "paper",
                    "source_id": "Frontier Agent Survey",
                    "quality_signal": "high_citation:cites=388",
                    "borrowed_standard": "The frontier survey indicates current agent demand.",
                    "used_for": "feasibility",
                }
            ],
        }
        paper_rows = [
            {
                "id": 1,
                "title": "Historical Best Paper",
                "award": "best_paper",
                "paper_weight": 5.0,
                "citation_count": 1000,
                "influential_citation_count": 100,
            },
            {
                "id": 2,
                "title": "Frontier Agent Survey",
                "award": "",
                "paper_weight": 0.0,
                "citation_count": 388,
                "influential_citation_count": 0,
            },
        ]
        score = build_evidence_grounded_score(payload, paper_rows=paper_rows)
        audit = score["generation_evidence_audit"]
        self.assertEqual(audit["status"], "missing_evidence")
        self.assertNotIn("matched_basis_count", audit)
        self.assertIn("No stored core Gold evidence", audit["requirement"])

    def test_user_library_context_reaches_idea_generation_without_becoming_allowed_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                db_path = cwd / "data" / "research_alpha.db"
                seed_strict_generation_evidence(db_path)
                upsert_user_library_paper(
                    db_path,
                    title="User Medical Route Paper",
                    venue="arXiv",
                    year=2026,
                    abstract="Clinical trial route terminology and failure modes.",
                    external_ref="https://arxiv.org/abs/2601.09999",
                )
                captured_prompt = ""

                def fake_chat(self, prompt, system_prompt=""):
                    nonlocal captured_prompt
                    if "Generate candidate research ideas" in prompt:
                        captured_prompt = prompt
                    return type(
                        "Resp",
                        (),
                        {
                            "text": json.dumps(
                                {
                                    "ideas": [
                                        {
                                            "idea_title": "Clinical Agent Stress Benchmark",
                                            "core_hypothesis": "Clinical research agents need stress tests around trial-route failures.",
                                            "historical_pattern": "hidden-assumption-reversal",
                                            "trend_support": "Recent agent evaluation work points to brittle tool-use workflows.",
                                            "frontier_gap": "Current agent benchmarks under-measure clinical route failure modes.",
                                            "why_now": "Clinical research agents are moving into more realistic protocol tasks.",
                                            "novelty": "Transfers hidden-assumption reversal to clinical route verification.",
                                            "value": "Could reveal clinically important failure boundaries.",
                                            "key_risk": "May look like a domain benchmark unless tied to failure mechanisms.",
                                            "first_experiments": ["Build protocol-route stress tasks", "Measure ranking changes"],
                                            "evaluation_outline": "Compare agents on clinical route stress tests and failure explanations.",
                                            "paper_angle": "Clinical agent progress claims need route-level falsification.",
                                            "evidence_basis": strict_evidence_basis(),
                                            "generation_guardrail": strict_generation_guardrail(),
                                            "storyline_trace": strict_storyline_trace(),
                                        }
                                    ]
                                }
                            )
                        },
                    )()

                with patch("research_alpha.llm.LLMClient.chat", autospec=True, side_effect=fake_chat):
                    result = run_pipeline_idea_stage(
                        root_dir=cwd,
                        query="clinical research agents",
                        idea_count=1,
                        pattern_limit=2,
                        genome_limit=2,
                        provider=None,
                        model="",
                        frontier_query="agent evaluation",
                    )
                context = json.loads(captured_prompt.split("Context:\n", 1)[1])
                self.assertEqual(context["domain_knowledge"]["papers"][0]["title"], "User Medical Route Paper")
                self.assertIn("User Medical Route Paper", captured_prompt)
                self.assertNotIn("User Medical Route Paper", context["allowed_source_ids"]["paper"])
                self.assertEqual(result["generated"], 1)
                idea_id = result["idea_summaries"][0]["idea_id"]
                saved = get_candidate_idea(db_path, idea_id)
                payload = json.loads(saved["content_json"])
                self.assertEqual(payload["domain_knowledge_context"]["paper_count"], 1)
                self.assertEqual(payload["domain_knowledge_context"]["papers"][0]["title"], "User Medical Route Paper")
                self.assertNotEqual(payload["evidence_basis"][0]["source_id"], "User Medical Route Paper")
            finally:
                os.chdir(old_cwd)

    def test_idea_generation_prompt_excludes_non_strict_pattern_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                db_path = cwd / "data" / "research_alpha.db"
                seed_strict_generation_evidence(db_path)
                upsert_pattern_card(
                    db_path,
                    "polluted-user-route-template",
                    json.dumps(
                        {
                            "pattern_name": "Polluted User Route Template",
                            "core_move": "Copy a user-route paper method into the current domain.",
                            "evidence_level": "logic_line_aggregation",
                            "logic_line_pattern": {
                                "source_logic": "A non-Gold user route paper suggests a workflow.",
                                "transfer_rule": "Copy that workflow as the idea.",
                                "why_it_generalizes": "It should generalize by analogy.",
                                "failure_boundary": "Fails if the user paper is not Gold.",
                            },
                            "grounding_audit": {"status": "invalid"},
                            "source_set_policy": "gold_set_only",
                            "source_papers": [
                                {
                                    "paper_id": 999,
                                    "title": "Non Gold Route Paper",
                                    "venue": "arXiv",
                                    "year": 2026,
                                    "source_kind": "user_library",
                                    "paper_weight": 0.0,
                                }
                            ],
                        }
                    ),
                )
                captured_prompt = ""

                def fake_chat(self, prompt, system_prompt=""):
                    nonlocal captured_prompt
                    if "Generate candidate research ideas" in prompt:
                        captured_prompt = prompt
                    return type(
                        "Resp",
                        (),
                        {
                            "text": json.dumps(
                                {
                                    "ideas": [
                                        {
                                            "idea_title": "Clean Strict Evidence Idea",
                                            "core_hypothesis": "Research-agent benchmarks should expose hidden failure assumptions.",
                                            "historical_pattern": "hidden-assumption-reversal",
                                            "trend_support": "Recent agent evaluation work points to brittle tool-use workflows.",
                                            "frontier_gap": "Current benchmarks hide brittle failure modes.",
                                            "why_now": "Agent systems are now deployed enough to need stress tests.",
                                            "novelty": "Transfers the strict hidden-assumption reversal story.",
                                            "value": "Could reveal defensible failure boundaries.",
                                            "key_risk": "May look like benchmark churn.",
                                            "first_experiments": ["Build failure-taxonomy tasks"],
                                            "evaluation_outline": "Compare ranking changes under stress tests.",
                                            "paper_angle": "Agent progress claims need failure-discovery evidence.",
                                            "evidence_basis": strict_evidence_basis(),
                                            "generation_guardrail": strict_generation_guardrail(),
                                            "storyline_trace": strict_storyline_trace(),
                                        }
                                    ]
                                }
                            )
                        },
                    )()

                with patch("research_alpha.llm.LLMClient.chat", autospec=True, side_effect=fake_chat):
                    result = run_pipeline_idea_stage(
                        root_dir=cwd,
                        query="research agents",
                        idea_count=1,
                        pattern_limit=5,
                        genome_limit=5,
                        provider=None,
                        model="",
                        frontier_query="agent evaluation",
                    )

                context = json.loads(captured_prompt.split("Context:\n", 1)[1])
                self.assertIn("hidden-assumption-reversal", context["allowed_source_ids"]["pattern"])
                self.assertNotIn("polluted-user-route-template", context["allowed_source_ids"]["pattern"])
                serialized_patterns = json.dumps(context["pattern_cards"], ensure_ascii=False)
                self.assertNotIn("Polluted User Route Template", serialized_patterns)
                self.assertEqual(result["generated"], 1)
            finally:
                os.chdir(old_cwd)

    def test_user_library_cannot_be_used_as_idea_scoring_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                db_path = cwd / "data" / "research_alpha.db"
                seed_strict_generation_evidence(db_path)
                upsert_user_library_paper(
                    db_path,
                    title="User Medical Route Paper",
                    venue="arXiv",
                    year=2026,
                    abstract="Clinical trial route terminology.",
                    external_ref="https://arxiv.org/abs/2601.09999",
                )

                def fake_chat(self, prompt, system_prompt=""):
                    return type(
                        "Resp",
                        (),
                        {
                            "text": json.dumps(
                                {
                                    "ideas": [
                                        {
                                            "idea_title": "Bad User Evidence Idea",
                                            "core_hypothesis": "Clinical research agents need route checks.",
                                            "historical_pattern": "hidden-assumption-reversal",
                                            "trend_support": "Agent evaluation is rising.",
                                            "frontier_gap": "Benchmarks miss clinical route failures.",
                                            "why_now": "Clinical agents are more capable.",
                                            "novelty": "Uses a user library route paper as evidence.",
                                            "value": "Could help clinical workflows.",
                                            "key_risk": "Evidence boundary is wrong.",
                                            "first_experiments": ["Build route tasks"],
                                            "evaluation_outline": "Compare route failure explanations.",
                                            "paper_angle": "Clinical route failures become benchmark targets.",
                                            "evidence_basis": strict_evidence_basis(source_id="User Medical Route Paper"),
                                            "generation_guardrail": strict_generation_guardrail(),
                                            "storyline_trace": strict_storyline_trace(),
                                        }
                                    ]
                                }
                            )
                        },
                    )()

                with patch("research_alpha.llm.LLMClient.chat", autospec=True, side_effect=fake_chat):
                    with self.assertRaises(IdeaGenerationNotReady) as caught:
                        run_pipeline_idea_stage(
                            root_dir=cwd,
                            query="clinical research agents",
                            idea_count=1,
                            pattern_limit=2,
                            genome_limit=2,
                            provider=None,
                            model="",
                            frontier_query="agent evaluation",
                        )
                self.assertIn("candidate_used_user_library_as_scoring_evidence", str(caught.exception))
                self.assertEqual(caught.exception.summary["primary_reason"], "user_library_used_as_standard")
            finally:
                os.chdir(old_cwd)

    def test_user_library_summary_includes_full_text_hints_without_scoring_weight(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            db_path = cwd / "data" / "research_alpha.db"
            init_db(db_path)
            paper_id = upsert_user_library_paper(
                db_path,
                title="User Full Text Route Paper",
                venue="arXiv",
                year=2026,
                abstract="Abstract-level route note.",
                external_ref="https://arxiv.org/abs/2601.00003",
            )
            with connect(db_path) as conn:
                conn.execute(
                    "UPDATE papers SET full_text_json=? WHERE id=?",
                    (
                        json.dumps(
                            {
                                "source_scope": "full_text_sections",
                                "sections": {
                                    "introduction": "The domain route starts with bedside triage and handoff constraints.",
                                    "method": "The route uses protocol-aware decomposition for local adaptation.",
                                    "limitations": "The route may fail when hospitals use incompatible protocols.",
                                },
                            }
                        ),
                        paper_id,
                    ),
                )
                row = conn.execute(
                    "SELECT source_kind, paper_weight FROM papers WHERE id=?",
                    (paper_id,),
                ).fetchone()
            summary = build_user_domain_knowledge_summary(cwd, limit=5)
            paper = summary["papers"][0]
            self.assertEqual(row["source_kind"], "user_library")
            self.assertEqual(float(row["paper_weight"]), 0.0)
            self.assertIn("full_text_domain_hints", paper)
            self.assertIn("bedside triage", paper["full_text_domain_hints"]["introduction"])
            self.assertIn("scoring_standard", summary["forbidden_uses"])

    def test_idea_generation_rejects_source_method_surface_transfer_before_saving(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                db_path = cwd / "data" / "research_alpha.db"
                seed_strict_generation_evidence(db_path)

                def fake_chat(self, prompt, system_prompt=""):
                    return type(
                        "Resp",
                        (),
                        {
                            "text": json.dumps(
                                {
                                    "ideas": [
                                        {
                                            "idea_title": "Benchmark Assumption Stress Task Transplant",
                                            "core_hypothesis": "Apply benchmark assumptions, brittle evaluation, stress tests, and hidden agent failures directly as the method for research-agent evaluation.",
                                            "historical_pattern": "hidden-assumption-reversal",
                                            "trend_support": "Agent evaluation is rising.",
                                            "frontier_gap": "Benchmarks miss hidden failures.",
                                            "why_now": "Agent systems are now deployed enough.",
                                            "novelty": "Copies benchmark assumptions and brittle evaluation into a new task suite.",
                                            "value": "Could expose hidden agent failures with stress tests.",
                                            "key_risk": "This may be a surface method transplant.",
                                            "first_experiments": ["Reuse benchmark assumptions and brittle evaluation stress tests."],
                                            "evaluation_outline": "Measure hidden agent failures under benchmark assumptions and brittle evaluation.",
                                            "paper_angle": "Best Agent Benchmark style stress tests are moved into research-agent evaluation.",
                                            "evidence_basis": [
                                                {
                                                    "source_type": "genome",
                                                    "source_id": "Best Agent Benchmark",
                                                    "quality_signal": "best_paper",
                                                    "borrowed_standard": "Turn evaluation into hidden failure discovery.",
                                                    "used_for": "innovation|logic|feasibility|value|defensibility",
                                                }
                                            ],
                                            "generation_guardrail": strict_generation_guardrail(),
                                            "storyline_trace": strict_storyline_trace("Best Agent Benchmark"),
                                        }
                                    ]
                                }
                            )
                        },
                    )()

                with patch("research_alpha.llm.LLMClient.chat", autospec=True, side_effect=fake_chat):
                    with self.assertRaises(IdeaGenerationNotReady) as caught:
                        run_pipeline_idea_stage(
                            root_dir=cwd,
                            query="research agents",
                            idea_count=1,
                            pattern_limit=2,
                            genome_limit=2,
                            provider=None,
                            model="",
                            frontier_query="agent evaluation",
                        )
                self.assertIn("visible_idea_reuses_source_surface_terms", str(caught.exception))
                self.assertEqual(caught.exception.summary["primary_reason"], "storyline_migration_failed")
                with connect(db_path) as conn:
                    count = conn.execute("SELECT COUNT(*) AS count FROM candidate_ideas").fetchone()
                self.assertEqual(int(count["count"]), 0)
            finally:
                os.chdir(old_cwd)

    def test_idea_generation_semantic_retry_after_storyline_gate_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                db_path = cwd / "data" / "research_alpha.db"
                seed_strict_generation_evidence(db_path)
                bad_response = json.dumps(
                    {
                        "ideas": [
                            {
                                "idea_title": "Benchmark Assumption Stress Task Transplant",
                                "core_hypothesis": "Apply benchmark assumptions, brittle evaluation, stress tests, and hidden agent failures directly as the method for research-agent evaluation.",
                                "historical_pattern": "hidden-assumption-reversal",
                                "trend_support": "Agent evaluation is rising.",
                                "frontier_gap": "Benchmarks miss hidden failures.",
                                "why_now": "Agent systems are now deployed enough.",
                                "novelty": "Copies benchmark assumptions and brittle evaluation into a new task suite.",
                                "value": "Could expose hidden agent failures with stress tests.",
                                "key_risk": "This may be a surface method transplant.",
                                "first_experiments": ["Reuse benchmark assumptions and brittle evaluation stress tests."],
                                "evaluation_outline": "Measure hidden agent failures under benchmark assumptions and brittle evaluation.",
                                "paper_angle": "Best Agent Benchmark style stress tests are moved into research-agent evaluation.",
                                "evidence_basis": [
                                    {
                                        "source_type": "genome",
                                        "source_id": "Best Agent Benchmark",
                                        "quality_signal": "best_paper",
                                        "borrowed_standard": "Turn evaluation into hidden failure discovery.",
                                        "used_for": "innovation|logic|feasibility|value|defensibility",
                                    }
                                ],
                                "generation_guardrail": strict_generation_guardrail(),
                                "storyline_trace": strict_storyline_trace("Best Agent Benchmark"),
                            }
                        ]
                    }
                )
                recovered_response = json.dumps(
                    {
                        "ideas": [
                            {
                                "idea_title": "Assumption-Boundary Audits for Research Agents",
                                "core_hypothesis": "Research-agent evaluation should expose when success claims depend on hidden assumptions about tool traces, evidence provenance, and recovery behavior.",
                                "historical_pattern": "hidden-assumption-reversal",
                                "trend_support": "Recent agent evaluation work points to brittle tool-use workflows.",
                                "frontier_gap": "Current research-agent benchmarks under-measure assumption boundaries and recovery quality.",
                                "why_now": "Research agents are strong enough to complete tasks, making hidden evaluation assumptions more consequential.",
                                "novelty": "Migrates the hidden-assumption reversal storyline into domain-native audits of evaluation claims rather than copying a source benchmark method.",
                                "value": "Could reveal which agents remain trustworthy when task success depends on fragile evidence routes.",
                                "key_risk": "The audit may become a task list unless each case maps to an explicit assumption boundary.",
                                "first_experiments": ["Build two assumption-boundary audit cases and measure ranking changes plus recovery explanations."],
                                "evaluation_outline": "Compare agent rankings under normal tasks, assumption-boundary audits, and recovery-quality scoring.",
                                "paper_angle": "Research-agent progress claims need assumption-boundary evidence before leaderboard gains are defensible.",
                                "evidence_basis": strict_evidence_basis(),
                                "generation_guardrail": strict_generation_guardrail(),
                                "storyline_trace": strict_storyline_trace(),
                            }
                        ]
                    }
                )
                prompts: list[str] = []

                def fake_chat(self, prompt, system_prompt=""):
                    prompts.append(prompt)
                    if "Retry the idea-generation task once" in prompt:
                        return type("Resp", (), {"text": recovered_response})()
                    if "Generate candidate research ideas" in prompt:
                        return type("Resp", (), {"text": bad_response})()
                    return type("Resp", (), {"text": json.dumps({"decision": "pass", "overlap_label": "distant", "closest_work": [], "why_not_done_yet": ["No prior work in local corpus."], "required_differentiation": [], "rethink_prompt": "", "summary": "Pass."})})()

                with patch("research_alpha.llm.LLMClient.chat", autospec=True, side_effect=fake_chat):
                    result = run_pipeline_idea_stage(
                        root_dir=cwd,
                        query="research agents",
                        idea_count=1,
                        pattern_limit=2,
                        genome_limit=2,
                        provider=None,
                        model="",
                        frontier_query="agent evaluation",
                    )
                retry_prompts = [prompt for prompt in prompts if "Retry the idea-generation task once" in prompt]
                self.assertEqual(len(retry_prompts), 1)
                self.assertIn("storyline_migration_failed", retry_prompts[0])
                self.assertIn("Do not copy source-paper method names", retry_prompts[0])
                self.assertEqual(result["generated"], 1)
                self.assertEqual(result["rejected"], 1)
                self.assertIn("semantic_retry", result["idea_summaries"][0])
                saved = get_candidate_idea(db_path, result["idea_summaries"][0]["idea_id"])
                payload = json.loads(saved["content_json"])
                self.assertEqual(payload["idea_title"], "Assumption-Boundary Audits for Research Agents")
                self.assertIn("semantic_retry", payload)
                with connect(db_path) as conn:
                    count = conn.execute("SELECT COUNT(*) AS count FROM candidate_ideas").fetchone()
                self.assertEqual(int(count["count"]), 1)
            finally:
                os.chdir(old_cwd)

    def test_idea_semantic_retry_prompt_keeps_user_library_as_background(self) -> None:
        prompt = build_idea_semantic_retry_prompt(
            original_prompt="Generate candidate research ideas\nContext: allowed_source_ids",
            failure_summary={
                "status": "idea_generation_not_ready",
                "primary_reason": "user_library_used_as_standard",
                "retry_prompt_constraints": ["不得把用户论文库写入 evidence_basis"],
                "rejected_ideas": [{"title": "Bad", "reason": "secret /Users/me/path should not matter"}],
            },
        )
        self.assertIn("Retry the idea-generation task once", prompt)
        self.assertIn("never use them as evidence_basis", prompt)
        self.assertIn("storyline_trace.source_id", prompt)
        self.assertIn("user_library_used_as_standard", prompt)

    def test_idea_generation_repairs_invalid_json_once_before_failing_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                db_path = cwd / "data" / "research_alpha.db"
                seed_strict_generation_evidence(db_path)
                bad_response = json.dumps({"ideas": [{"idea_title": "Broken schema"}]})
                repaired_response = json.dumps(
                    {
                        "ideas": [
                            {
                                "idea_title": "Repaired Agent Benchmark",
                                "core_hypothesis": "Research-agent benchmarks should expose hidden failure assumptions.",
                                "historical_pattern": "hidden-assumption-reversal",
                                "trend_support": "Recent agent evaluation work points to brittle tool-use workflows.",
                                "frontier_gap": "Current benchmarks hide brittle failure modes.",
                                "why_now": "Agent systems are now deployed enough to need stress tests.",
                                "novelty": "Transfers the strict hidden-assumption reversal story.",
                                "value": "Could reveal defensible failure boundaries.",
                                "key_risk": "May look like benchmark churn.",
                                "first_experiments": ["Build failure-taxonomy tasks"],
                                "evaluation_outline": "Compare ranking changes under stress tests.",
                                "paper_angle": "Agent progress claims need failure-discovery evidence.",
                                "evidence_basis": strict_evidence_basis(),
                                "generation_guardrail": strict_generation_guardrail(),
                                "storyline_trace": strict_storyline_trace(),
                            }
                        ]
                    }
                )
                prompts: list[str] = []

                def fake_chat(self, prompt, system_prompt=""):
                    prompts.append(prompt)
                    if "Repair the previous Idea response" in prompt:
                        return type("Resp", (), {"text": repaired_response})()
                    if "Generate candidate research ideas" in prompt:
                        return type("Resp", (), {"text": bad_response})()
                    return type("Resp", (), {"text": json.dumps({"decision": "pass", "overlap_label": "distant", "closest_work": [], "why_not_done_yet": ["No prior work in local corpus."], "required_differentiation": [], "rethink_prompt": "", "summary": "Pass."})})()

                with patch("research_alpha.llm.LLMClient.chat", autospec=True, side_effect=fake_chat):
                    result = run_pipeline_idea_stage(
                        root_dir=cwd,
                        query="research agents",
                        idea_count=1,
                        pattern_limit=2,
                        genome_limit=2,
                        provider=None,
                        model="",
                        frontier_query="agent evaluation",
                    )
                self.assertEqual(result["generated"], 1)
                repair_prompts = [prompt for prompt in prompts if "Repair the previous Idea response" in prompt]
                self.assertEqual(len(repair_prompts), 1)
                self.assertIn("Validation error", repair_prompts[0])
                saved = get_candidate_idea(db_path, result["idea_summaries"][0]["idea_id"])
                self.assertEqual(json.loads(saved["content_json"])["idea_title"], "Repaired Agent Benchmark")
            finally:
                os.chdir(old_cwd)

    def test_idea_generation_repairs_thin_placeholder_content_before_saving(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                db_path = cwd / "data" / "research_alpha.db"
                seed_strict_generation_evidence(db_path)
                thin_response = json.dumps(
                    {
                        "ideas": [
                            {
                                "idea_title": "Idea",
                                "core_hypothesis": "ok",
                                "historical_pattern": "hidden-assumption-reversal",
                                "trend_support": "Agent evaluation is rising.",
                                "frontier_gap": "ok",
                                "why_now": "ok",
                                "novelty": "ok",
                                "value": "ok",
                                "key_risk": "ok",
                                "first_experiments": ["ok"],
                                "evaluation_outline": "ok",
                                "paper_angle": "ok",
                                "evidence_basis": strict_evidence_basis(),
                                "generation_guardrail": strict_generation_guardrail(),
                                "storyline_trace": strict_storyline_trace(),
                            }
                        ]
                    }
                )
                repaired_response = json.dumps(
                    {
                        "ideas": [
                            {
                                "idea_title": "Assumption-Reversal Agent Stress Tests",
                                "core_hypothesis": "Research-agent progress is overestimated when benchmarks hide failure assumptions instead of testing them.",
                                "historical_pattern": "hidden-assumption-reversal",
                                "trend_support": "Recent agent evaluation work points to brittle tool-use workflows.",
                                "frontier_gap": "Current research-agent benchmarks under-measure hidden failure modes and explanation quality.",
                                "why_now": "Agent systems are strong enough and deployed enough that brittle evaluation now matters.",
                                "novelty": "Transfers hidden-assumption reversal into a benchmark that discovers failure mechanisms.",
                                "value": "Could change which research agents look ready for real scientific workflows.",
                                "key_risk": "The benchmark may become task churn unless each task maps to a failure assumption.",
                                "first_experiments": ["Build two failure-taxonomy tasks and compare agent ranking shifts."],
                                "evaluation_outline": "Measure ranking changes, failure-mode coverage, and explanation agreement under stress tasks.",
                                "paper_angle": "Agent evaluation should explain brittle failures, not only score successful completions.",
                                "evidence_basis": strict_evidence_basis(),
                                "generation_guardrail": strict_generation_guardrail(),
                                "storyline_trace": strict_storyline_trace(),
                            }
                        ]
                    }
                )
                prompts: list[str] = []

                def fake_chat(self, prompt, system_prompt=""):
                    prompts.append(prompt)
                    if "Repair the previous Idea response" in prompt:
                        return type("Resp", (), {"text": repaired_response})()
                    if "Generate candidate research ideas" in prompt:
                        return type("Resp", (), {"text": thin_response})()
                    return type("Resp", (), {"text": json.dumps({"decision": "pass", "overlap_label": "distant", "closest_work": [], "why_not_done_yet": ["No prior work in local corpus."], "required_differentiation": [], "rethink_prompt": "", "summary": "Pass."})})()

                with patch("research_alpha.llm.LLMClient.chat", autospec=True, side_effect=fake_chat):
                    result = run_pipeline_idea_stage(
                        root_dir=cwd,
                        query="research agents",
                        idea_count=1,
                        pattern_limit=2,
                        genome_limit=2,
                        provider=None,
                        model="",
                        frontier_query="agent evaluation",
                    )
                self.assertEqual(result["generated"], 1)
                repair_prompts = [prompt for prompt in prompts if "Repair the previous Idea response" in prompt]
                self.assertEqual(len(repair_prompts), 1)
                self.assertIn("content_quality_failed", repair_prompts[0])
                saved = get_candidate_idea(db_path, result["idea_summaries"][0]["idea_id"])
                self.assertEqual(json.loads(saved["content_json"])["idea_title"], "Assumption-Reversal Agent Stress Tests")
            finally:
                os.chdir(old_cwd)

    def test_idea_generation_rejects_thin_placeholder_content_after_repair_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                db_path = cwd / "data" / "research_alpha.db"
                seed_strict_generation_evidence(db_path)
                thin_response = json.dumps(
                    {
                        "ideas": [
                            {
                                "idea_title": "Idea",
                                "core_hypothesis": "ok",
                                "historical_pattern": "hidden-assumption-reversal",
                                "trend_support": "Agent evaluation is rising.",
                                "frontier_gap": "ok",
                                "why_now": "ok",
                                "novelty": "ok",
                                "value": "ok",
                                "key_risk": "ok",
                                "first_experiments": ["ok"],
                                "evaluation_outline": "ok",
                                "paper_angle": "ok",
                                "evidence_basis": strict_evidence_basis(),
                                "generation_guardrail": strict_generation_guardrail(),
                                "storyline_trace": strict_storyline_trace(),
                            }
                        ]
                    }
                )

                def fake_chat(self, prompt, system_prompt=""):
                    return type("Resp", (), {"text": thin_response})()

                with patch("research_alpha.llm.LLMClient.chat", autospec=True, side_effect=fake_chat):
                    with self.assertRaises(IdeaGenerationNotReady) as caught:
                        run_pipeline_idea_stage(
                            root_dir=cwd,
                            query="research agents",
                            idea_count=1,
                            pattern_limit=2,
                            genome_limit=2,
                            provider=None,
                            model="",
                            frontier_query="agent evaluation",
                        )
                self.assertEqual(caught.exception.summary["primary_reason"], "idea_content_too_thin")
                with connect(db_path) as conn:
                    count = conn.execute("SELECT COUNT(*) AS count FROM candidate_ideas").fetchone()
                self.assertEqual(int(count["count"]), 0)
            finally:
                os.chdir(old_cwd)

    def test_query_frontier_evidence_requires_trend_axis_alignment(self) -> None:
        audit = validate_query_frontier_evidence(
            "research agents",
            {
                "latest_year": 2025,
                "top_terms": [{"term": "cancer"}, {"term": "global prevalence"}],
            },
            [
                {
                    "title": "The rise of language model agents",
                    "abstract": "Research agents use tools and planning.",
                    "year": 2025,
                }
            ],
        )
        self.assertFalse(audit["ready"])
        self.assertEqual(audit["matched_paper_tokens"], ["agent"])
        self.assertEqual(audit["matched_trend_tokens"], [])

    def test_query_frontier_evidence_uses_trend_cluster_examples(self) -> None:
        audit = validate_query_frontier_evidence(
            "research agents",
            {
                "latest_year": 2025,
                "top_terms": [{"term": "survey"}],
                "trend_clusters": [
                    {
                        "cluster_label": "survey",
                        "terms": ["survey", "rise potential", "agents survey"],
                        "example_papers": [
                            {
                                "title": "The Rise and Potential of Large Language Model Based Agents",
                                "year": 2025,
                            }
                        ],
                    }
                ],
            },
            [
                {
                    "title": "The Rise and Potential of Large Language Model Based Agents",
                    "abstract": "",
                    "year": 2025,
                }
            ],
        )
        self.assertTrue(audit["ready"])
        self.assertEqual(audit["matched_paper_tokens"], ["agent"])
        self.assertIn("agent", audit["matched_trend_tokens"])

    def test_init_creates_database_and_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            output = io.StringIO()
            try:
                import os
                os.chdir(cwd)
                with redirect_stdout(output):
                    exit_code = main(["init"])
                self.assertEqual(exit_code, 0)
                self.assertTrue((cwd / "data" / "research_alpha.db").exists())
                self.assertTrue((cwd / ".env.example").exists())
                self.assertTrue((cwd / "ra").exists())
                self.assertTrue((cwd / "seeds" / "demo_papers.jsonl").exists())
                self.assertTrue((cwd / "configs" / "venues.yaml").exists())
                self.assertTrue((cwd / "configs" / "award_signals.yaml").exists())
                self.assertIn("Env example:", output.getvalue())
                self.assertIn("Demo seeds:", output.getvalue())
            finally:
                os.chdir(old_cwd)

    def test_add_and_list_papers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                main(["init"])
                main(["add-paper", "--title", "Test Paper", "--venue", "ICLR", "--year", "2026"])
                output = io.StringIO()
                with redirect_stdout(output):
                    exit_code = main(["papers", "--limit", "5"])
                self.assertEqual(exit_code, 0)
                self.assertIn("Test Paper", output.getvalue())
            finally:
                os.chdir(old_cwd)

    def test_user_library_cli_lists_domain_papers_and_full_text_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                db_path = cwd / "data" / "research_alpha.db"
                paper_id = upsert_user_library_paper(
                    db_path,
                    title="CLI User Route Paper",
                    venue="arXiv",
                    year=2026,
                    abstract="Route note for local debugging.",
                    external_ref="https://arxiv.org/abs/2601.01234",
                )
                with connect(db_path) as conn:
                    conn.execute(
                        "UPDATE papers SET full_text_json=? WHERE id=?",
                        (
                            json.dumps(
                                {
                                    "source_scope": "full_text_sections",
                                    "sections": {
                                        "introduction": "CLI route setup.",
                                        "limitations": "CLI route limitation.",
                                    },
                                }
                            ),
                            paper_id,
                        ),
                    )
                output = io.StringIO()
                with redirect_stdout(output):
                    exit_code = main(["ul", "--limit", "5"])
                self.assertEqual(exit_code, 0)
                text = output.getvalue()
                self.assertIn("User library papers: 1", text)
                self.assertIn("full-text hints: 1", text)
                self.assertIn("CLI User Route Paper", text)
                self.assertIn("full_text_hints=introduction, limitations", text)
                self.assertIn("domain knowledge only", text)

                json_out = io.StringIO()
                with redirect_stdout(json_out):
                    json_exit = main(["user-library", "--json"])
                self.assertEqual(json_exit, 0)
                payload = json.loads(json_out.getvalue())
                self.assertEqual(payload["count"], 1)
                self.assertEqual(payload["full_text_ready_count"], 1)
                self.assertEqual(payload["papers"][0]["paper_weight"], 0.0)
                self.assertEqual(payload["papers"][0]["policy"], "user_library_domain_knowledge_only_never_scoring_standard")
                self.assertIn("limitations", payload["papers"][0]["full_text_hint_sections"])
            finally:
                os.chdir(old_cwd)

    def test_user_library_cli_add_and_delete_use_domain_only_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                db_path = cwd / "data" / "research_alpha.db"
                atom = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2601.01234v2</id>
    <published>2026-01-08T00:00:00Z</published>
    <title>  CLI Reliable User Library  </title>
    <summary>Domain route learning for local CLI workflows.</summary>
  </entry>
</feed>"""

                class FakeResponse:
                    def __enter__(self):
                        return self

                    def __exit__(self, exc_type, exc, tb):
                        return False

                    def read(self, limit=-1):
                        return atom

                add_out = io.StringIO()
                with patch("research_alpha.gui.urlopen", return_value=FakeResponse()):
                    with redirect_stdout(add_out):
                        add_exit = main(
                            [
                                "ul",
                                "add",
                                "https://arxiv.org/abs/2601.01234v2",
                                "--note",
                                "CLI route note.",
                            ]
                        )
                self.assertEqual(add_exit, 0)
                self.assertIn("Added user-library paper", add_out.getvalue())
                with connect(db_path) as conn:
                    row = conn.execute(
                        "SELECT id, title, abstract, venue, source_kind, award, paper_weight, score_notes FROM papers WHERE source_kind='user_library'"
                    ).fetchone()
                self.assertEqual(row["title"], "CLI Reliable User Library")
                self.assertIn("CLI route note.", row["abstract"])
                self.assertEqual(row["venue"], "arXiv")
                self.assertEqual(row["award"], "")
                self.assertEqual(float(row["paper_weight"]), 0.0)
                self.assertIn("user_library_domain_knowledge_only", row["score_notes"])

                json_out = io.StringIO()
                with patch("research_alpha.gui.urlopen", side_effect=OSError("offline")):
                    with redirect_stdout(json_out):
                        json_exit = main(["ul", "add", "--url", "https://openreview.net/forum?id=abc123", "--json"])
                self.assertEqual(json_exit, 0)
                payload = json.loads(json_out.getvalue())
                self.assertTrue(payload["ok"])
                self.assertEqual(payload["paper"]["title"], "OpenReview:abc123")
                self.assertEqual(payload["paper"]["paper_weight"], 0.0)

                delete_out = io.StringIO()
                with redirect_stdout(delete_out):
                    delete_exit = main(["ul", "delete", str(row["id"])])
                self.assertEqual(delete_exit, 0)
                self.assertIn(f"Deleted user-library paper #{row['id']}", delete_out.getvalue())
                with connect(db_path) as conn:
                    deleted_count = conn.execute("SELECT COUNT(*) AS count FROM papers WHERE id=?", (row["id"],)).fetchone()
                self.assertEqual(int(deleted_count["count"]), 0)
            finally:
                os.chdir(old_cwd)

    def test_user_library_cli_rejects_invalid_link_and_will_not_delete_gold(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                db_path = cwd / "data" / "research_alpha.db"
                invalid_err = io.StringIO()
                with redirect_stderr(invalid_err):
                    invalid_exit = main(["ul", "add", "file:///tmp/paper.pdf"])
                self.assertEqual(invalid_exit, 1)
                self.assertIn("http", invalid_err.getvalue())

                gold_id = add_paper(
                    db_path,
                    "Gold Must Stay",
                    "ICLR",
                    2026,
                    abstract="Gold evidence.",
                    source_kind="gold_openreview",
                    award="best_paper",
                )
                delete_err = io.StringIO()
                with redirect_stderr(delete_err):
                    delete_exit = main(["ul", "delete", str(gold_id)])
                self.assertEqual(delete_exit, 1)
                self.assertIn("not found", delete_err.getvalue())
                with connect(db_path) as conn:
                    row = conn.execute("SELECT source_kind FROM papers WHERE id=?", (gold_id,)).fetchone()
                self.assertEqual(row["source_kind"], "gold_openreview")
            finally:
                os.chdir(old_cwd)

    def test_doctor_reports_missing_evidence_without_leaking_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                output = io.StringIO()
                with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-doctor-secret"}, clear=False):
                    with redirect_stdout(output):
                        exit_code = main(["doctor", "--json"])
                self.assertEqual(exit_code, 0)
                payload = json.loads(output.getvalue())
                self.assertEqual(payload["overall_status"], "needs_work")
                self.assertEqual(payload["evidence"]["status"], "not_ready")
                self.assertIn("pdf_parser", payload["runtime"])
                self.assertIn("available", payload["runtime"]["pdf_parser"])
                codes = {item["code"] for item in payload["issues"]}
                self.assertIn("evidence_not_ready", codes)
                self.assertNotIn("sk-", json.dumps(payload, ensure_ascii=False))
            finally:
                os.chdir(old_cwd)

    def test_doctor_summarizes_ready_local_project_and_user_library(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                db_path = cwd / "data" / "research_alpha.db"
                seed_strict_generation_evidence(db_path)
                user_id = upsert_user_library_paper(
                    db_path,
                    title="Doctor User Route",
                    venue="arXiv",
                    year=2026,
                    abstract="Doctor route note.",
                    external_ref="https://arxiv.org/abs/2601.04444",
                )
                with connect(db_path) as conn:
                    conn.execute(
                        "UPDATE papers SET full_text_json=? WHERE id=?",
                        (
                            json.dumps({"sections": {"introduction": "Doctor user route setup."}}),
                            user_id,
                        ),
                    )
                output = io.StringIO()
                with patch.dict(os.environ, {"RA_LLM_PROVIDER": "deepseek", "DEEPSEEK_API_KEY": "sk-doctor-test"}, clear=False):
                    with redirect_stdout(output):
                        exit_code = main(["doc", "--papers", "12"])
                self.assertEqual(exit_code, 0)
                text = output.getvalue()
                self.assertIn("Doctor:", text)
                self.assertIn("api_key=yes", text)
                self.assertIn("Runtime: pdf_parser=", text)
                self.assertIn("status=ready_for_strict_generation", text)
                self.assertIn("paper_like=0", text)
                self.assertIn("User library: papers=1 full_text_hints=1", text)
                self.assertIn('ra id "research agents" --ideas 5', text)
                self.assertNotIn("sk-doctor-test", text)
            finally:
                os.chdir(old_cwd)

    def test_doctor_prioritizes_idea_generation_when_ready_even_with_notices(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                db_path = cwd / "data" / "research_alpha.db"
                seed_strict_generation_evidence(db_path)
                with connect(db_path) as conn:
                    conn.execute(
                        """
                        UPDATE papers
                        SET full_text_json=?
                        WHERE id = (SELECT id FROM papers ORDER BY id LIMIT 1)
                        """,
                        (
                            json.dumps({"status": "ready", "quality": "metadata_page_sections", "sections": {"introduction": "Short event page."}}),
                        ),
                    )
                output = io.StringIO()
                with patch.dict(os.environ, {"RA_LLM_PROVIDER": "deepseek", "DEEPSEEK_API_KEY": "sk-doctor-test"}, clear=False):
                    with redirect_stdout(output):
                        exit_code = main(["doctor", "--json"])
                self.assertEqual(exit_code, 0)
                payload = json.loads(output.getvalue())
                self.assertEqual(payload["overall_status"], "ready")
                self.assertEqual(payload["evidence"]["status"], "ready_for_strict_generation")
                self.assertTrue(any(item["severity"] == "notice" for item in payload["issues"]))
                self.assertEqual(payload["next_commands"][0], 'ra id "research agents" --ideas 5')
            finally:
                os.chdir(old_cwd)

    def test_evidence_report_not_ready_without_high_weight_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                main(["init"])
                output = io.StringIO()
                with redirect_stdout(output):
                    exit_code = main(["ev"])
                self.assertEqual(exit_code, 0)
                self.assertIn("Evidence status: not_ready", output.getvalue())
                payload = json.loads((cwd / "outputs" / "evidence" / "evidence_report.json").read_text(encoding="utf-8"))
                self.assertEqual(payload["status"], "not_ready")
                self.assertIn("Add or harvest at least 3 scored core Gold papers with explicit Best/Outstanding/Oral signals.", payload["missing_requirements"])
            finally:
                os.chdir(old_cwd)

    def test_evidence_report_ready_after_genomes_and_pattern(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                main(["init"])
                awards = ["best_paper", "oral", "outstanding_paper"]
                (cwd / "seeds").mkdir(parents=True, exist_ok=True)
                (cwd / "seeds" / "quality.jsonl").write_text(
                    "\n".join(
                        json.dumps(
                            {
                                "title": f"High Weight Paper {idx}",
                                "venue": "ICLR",
                                "year": 2024,
                                "award": award,
                                "citation_count": 1000 - idx * 100,
                                "influential_citation_count": 100 - idx * 10,
                                "abstract": "Benchmark assumptions hide agent failures.",
                            }
                        )
                        for idx, award in enumerate(awards)
                    )
                    + "\n",
                    encoding="utf-8",
                )
                main(
                    [
                        "h",
                        "--file",
                        "seeds/quality.jsonl",
                    ]
                )
                main(["score"])
                fake_genome = grounded_fake_genome_json("Benchmark assumptions hide agent failures.")
                fake_pattern = grounded_fake_pattern_json(["High Weight Paper 0", "High Weight Paper 2"])
                with patch("research_alpha.llm.LLMClient.chat") as mock_chat:
                    mock_chat.side_effect = lambda prompt, system_prompt=None: (
                        type("Resp", (), {"text": fake_pattern})()
                        if "Aggregate the following genome cards" in prompt
                        else grounded_fake_chat_response(prompt)
                    )
                    main(["gb", "--limit", "2"])
                    main(["pb", "--limit", "2"])
                output = io.StringIO()
                with redirect_stdout(output):
                    exit_code = main(["evidence"])
                self.assertEqual(exit_code, 0)
                self.assertIn("Evidence status: ready_for_strict_generation", output.getvalue())
                payload = json.loads((cwd / "outputs" / "evidence" / "evidence_report.json").read_text(encoding="utf-8"))
                self.assertEqual(payload["status"], "ready_for_strict_generation")
                self.assertEqual(payload["counts"]["high_weight_papers"], 3)
                self.assertEqual(payload["counts"]["genome_cards"], 2)
                self.assertEqual(payload["counts"]["pattern_cards"], 1)
                self.assertIn("Full-text ready papers:", output.getvalue())
                self.assertIn("Stale genome cards:", output.getvalue())
                self.assertIn("Stale pattern cards:", output.getvalue())
                self.assertIn("full_text_status", payload["high_weight_papers"][0])
            finally:
                os.chdir(old_cwd)

    def test_extractive_pipeline_prepares_evidence_but_refuses_idea_generation_without_logic_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                main(["init"])
                seed_strict_quality_papers(cwd)

                gb_out = io.StringIO()
                with redirect_stdout(gb_out):
                    gb_exit = main(["gb", "--limit", "3", "--extractive"])
                self.assertEqual(gb_exit, 0)
                self.assertIn("Generated 3 extractive genome cards.", gb_out.getvalue())

                pb_out = io.StringIO()
                with redirect_stdout(pb_out):
                    pb_exit = main(["pb", "--limit", "3", "--extractive"])
                self.assertEqual(pb_exit, 0)
                self.assertIn("Generated extractive pattern card", pb_out.getvalue())

                ev_out = io.StringIO()
                with redirect_stdout(ev_out):
                    ev_exit = main(["evidence"])
                self.assertEqual(ev_exit, 0)
                self.assertIn("Evidence status: partially_ready", ev_out.getvalue())
                ev_payload = json.loads((cwd / "outputs" / "evidence" / "evidence_report.json").read_text(encoding="utf-8"))
                self.assertEqual(ev_payload["counts"]["strict_genome_cards"], 0)
                self.assertEqual(ev_payload["counts"]["strict_pattern_cards"], 0)

                err = io.StringIO()
                with redirect_stderr(err):
                    id_exit = main(["id", "research agents", "--ideas", "2", "--extractive"])
                self.assertEqual(id_exit, 1)
                self.assertIn("Extractive mode is evidence-prep only", err.getvalue())
                self.assertFalse((cwd / "outputs" / "ideas" / "idea-1.json").exists())
            finally:
                os.chdir(old_cwd)

    def test_run_extractive_prepares_evidence_without_saving_ideas(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                main(["init"])
                (cwd / "seeds").mkdir(parents=True, exist_ok=True)
                (cwd / "seeds" / "mini.jsonl").write_text(
                    dedent(
                        """
                        {"title":"Best Agent Benchmark","venue":"ICLR","year":2024,"award":"best_paper","citation_count":100,"influential_citation_count":40,"abstract":"Benchmark assumptions hide agent failures and brittle evaluation."}
                        {"title":"Oral Agent Planning","venue":"ICLR","year":2024,"award":"oral","citation_count":90,"influential_citation_count":35,"abstract":"Agent planning requires robust tool-use evaluation under failure."}
                        {"title":"Spotlight Agent Stress Tests","venue":"ICLR","year":2024,"award":"spotlight","citation_count":80,"influential_citation_count":30,"abstract":"Stress tests expose hidden assumptions in research agent systems."}
                        """
                    ).strip()
                    + "\n",
                    encoding="utf-8",
                )

                output = io.StringIO()
                with redirect_stdout(output):
                    exit_code = main(
                        [
                            "r",
                            "research agents",
                            "--file",
                            "seeds/mini.jsonl",
                            "--ideas",
                            "2",
                            "--extractive",
                        ]
                    )
                self.assertEqual(exit_code, 0)
                text = output.getvalue()
                self.assertIn("Pipeline status: partial", text)
                self.assertIn("genome: completed", text)
                self.assertIn("pattern: completed", text)
                self.assertIn("ideas: skipped (idea_generation_not_ready)", text)

                manifests = list((cwd / "outputs").glob("*/manifest.json"))
                self.assertEqual(len(manifests), 1)
                manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
                self.assertNotIn("ideas", manifest["completed_stages"])
                self.assertEqual(manifest["status"], "partial")
                self.assertEqual(manifest["requested_ideas"], 2)
                self.assertTrue((manifests[0].parent / "trend_report.md").exists())
                self.assertTrue((manifests[0].parent / "opportunity_map.html").exists())
                self.assertFalse((cwd / "outputs" / "ideas" / "idea_ranking.json").exists())
                self.assertEqual(len(list((cwd / "outputs" / "dossiers").glob("*-dossier.json"))), 0)
            finally:
                os.chdir(old_cwd)

    def test_extractive_idea_generation_rejects_query_without_frontier_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                main(["init"])
                (cwd / "seeds").mkdir(parents=True, exist_ok=True)
                (cwd / "seeds" / "classic.jsonl").write_text(
                    dedent(
                        """
                        {"title":"Vision Language Pretraining","venue":"ICML","year":2024,"award":"best_paper","citation_count":100,"influential_citation_count":40,"abstract":"Contrastive image text pretraining learns transferable visual representations."}
                        {"title":"Efficient Adaptation","venue":"ICLR","year":2024,"award":"oral","citation_count":90,"influential_citation_count":35,"abstract":"Low rank adaptation improves efficient model fine tuning."}
                        {"title":"Latent Generative Models","venue":"ICLR","year":2024,"award":"spotlight","citation_count":80,"influential_citation_count":30,"abstract":"Latent variable models learn generative representations."}
                        """
                    ).strip()
                    + "\n",
                    encoding="utf-8",
                )
                main(["h", "--file", "seeds/classic.jsonl"])
                main(["score"])
                main(["gb", "--limit", "3", "--extractive"])
                main(["pb", "--limit", "3", "--extractive"])

                err = io.StringIO()
                with redirect_stderr(err):
                    exit_code = main(["id", "research agents", "--ideas", "2", "--extractive"])
                self.assertEqual(exit_code, 1)
                self.assertIn("Extractive mode is evidence-prep only", err.getvalue())
                self.assertFalse((cwd / "outputs" / "ideas" / "idea-1.json").exists())
            finally:
                os.chdir(old_cwd)

    def test_run_auto_repairs_frontier_evidence_once_before_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                main(["init"])
                main(["ds", "sk-test"])
                add_paper(
                    cwd / "data" / "research_alpha.db",
                    title="Gold paper",
                    venue="ICLR",
                    year=2024,
                    abstract="A gold evidence paper.",
                    award="best_paper",
                    source_kind="seed",
                )
                calls = {"idea": 0, "repair": 0}

                def fake_idea_stage(**kwargs):
                    calls["idea"] += 1
                    if calls["idea"] == 1:
                        raise ValueError("Current-hotspot evidence is not ready for this query. Need recent stored papers.")
                    return {"stage": "ideas", "status": "completed", "idea_summaries": [{"idea_id": 7, "title": "Recovered idea"}]}

                def fake_repair(*args, **kwargs):
                    calls["repair"] += 1
                    return {"status": "completed", "query": kwargs["query"], "remote": "openalex", "imported": 3}

                with (
                    patch("research_alpha.app.run_pipeline_trend_stage", return_value={"stage": "trends", "status": "completed", "paper_count": 1}),
                    patch("research_alpha.app.run_pipeline_genome_build", return_value=[(1, cwd / "outputs" / "genomes" / "paper-1.json")]),
                    patch("research_alpha.app.run_pipeline_pattern_build", return_value={"stage": "pattern", "status": "completed", "generated": 1}),
                    patch("research_alpha.app.run_pipeline_idea_stage", side_effect=fake_idea_stage),
                    patch("research_alpha.app.auto_repair_frontier_evidence", side_effect=fake_repair),
                    patch("research_alpha.app.build_candidate_idea_ranking", return_value={"scope": "current_run", "shown_ideas": [{"idea_id": 7, "title": "Recovered idea"}], "report_path": str(cwd / "outputs" / "ranking.json"), "report_payload": {"ok": True}}),
                    patch("research_alpha.app.get_candidate_idea", return_value=None),
                    patch("research_alpha.app.create_session_from_latest_candidate_idea", return_value={"session_id": 9, "name": "Recovered idea", "dossier_path": str(cwd / "outputs" / "sessions" / "session-9-dossier.json")}),
                ):
                    output = io.StringIO()
                    with redirect_stdout(output):
                        exit_code = main(["r", "camouflaged object detection", "--ideas", "1", "--session"])
                self.assertEqual(exit_code, 0)
                self.assertEqual(calls, {"idea": 2, "repair": 1})
                text = output.getvalue()
                self.assertIn("frontier_repair: completed", text)
                self.assertIn("Auto session: #9", text)
                manifest = json.loads(next((cwd / "outputs").glob("*/manifest.json")).read_text(encoding="utf-8"))
                self.assertIn("frontier_repair", manifest["completed_stages"])
                self.assertIn("ideas", manifest["completed_stages"])
                self.assertEqual(manifest["status"], "session_ready")
            finally:
                os.chdir(old_cwd)

    def test_ideate_session_defaults_to_auto_review_loop_before_final_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                captured = {}

                def fake_cmd_run(root_dir, **kwargs):
                    captured.update(kwargs)
                    return 0

                with (
                    patch("research_alpha.app.harvest_frontier_with_fallback", return_value={"status": "completed", "remote": "openalex", "imported": 0}),
                    patch("research_alpha.app.score_papers_in_db", return_value=0),
                    patch("research_alpha.app.build_trend_outputs", return_value={}),
                    patch("research_alpha.app.build_limitations_report", return_value={}),
                    patch("research_alpha.app.build_evidence_report", return_value=({"status": "ready_for_strict_generation"}, None, None)),
                    patch("research_alpha.app.cmd_run", side_effect=fake_cmd_run),
                ):
                    exit_code = main(["ideate", "research agents", "--session", "--provider", "ds"])
                self.assertEqual(exit_code, 0)
                self.assertTrue(captured["open_session"])
                self.assertTrue(captured["auto_review_loop"])
                self.assertEqual(captured["review_rounds"], 4)

                captured.clear()
                with (
                    patch("research_alpha.app.harvest_frontier_with_fallback", return_value={"status": "completed", "remote": "openalex", "imported": 0}),
                    patch("research_alpha.app.score_papers_in_db", return_value=0),
                    patch("research_alpha.app.build_trend_outputs", return_value={}),
                    patch("research_alpha.app.build_limitations_report", return_value={}),
                    patch("research_alpha.app.build_evidence_report", return_value=({"status": "ready_for_strict_generation"}, None, None)),
                    patch("research_alpha.app.cmd_run", side_effect=fake_cmd_run),
                ):
                    exit_code = main(["ideate", "research agents", "--session", "--no-review-loop", "--provider", "ds"])
                self.assertEqual(exit_code, 0)
                self.assertFalse(captured["auto_review_loop"])
            finally:
                os.chdir(old_cwd)

    def test_auto_review_loop_refreshes_session_summary_to_final_idea(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                from research_alpha.db import create_idea_session

                db_path = cwd / "data" / "research_alpha.db"
                session_id = create_idea_session(
                    db_path,
                    "session-a",
                    "Initial generated idea before reviewer loop.",
                    context_json=json.dumps({"evidence_basis": strict_evidence_basis()}),
                )
                summary = {
                    "session_id": session_id,
                    "name": "session-a",
                    "current_idea": "Initial generated idea before reviewer loop.",
                    "dossier_path": str(cwd / "outputs" / "sessions" / "session-1-dossier.json"),
                    "chat_summary": {
                        "title": "session-a",
                        "current_idea": "Initial generated idea before reviewer loop.",
                    },
                }

                def fake_review_loop(root_dir, **kwargs):
                    self.assertEqual(kwargs["session_id"], session_id)
                    with connect(db_path) as conn:
                        conn.execute(
                            "UPDATE idea_sessions SET current_idea=? WHERE id=?",
                            ("Final idea after automatic reviewer refinement.", session_id),
                        )
                    return 0

                with patch("research_alpha.app.cmd_review_loop", side_effect=fake_review_loop):
                    loop = run_auto_review_loop_for_session(
                        cwd,
                        summary,
                        rounds=4,
                        provider="ds",
                        model="",
                        pattern_limit=5,
                        genome_limit=5,
                    )
                self.assertEqual(loop["status"], "completed")
                self.assertEqual(loop["requested_rounds"], 4)
                self.assertEqual(loop["final_idea"], "Final idea after automatic reviewer refinement.")
                summary["auto_review_loop"] = loop
                refreshed = refresh_session_summary(cwd, summary)
                self.assertEqual(refreshed["current_idea"], "Final idea after automatic reviewer refinement.")
                self.assertEqual(refreshed["chat_summary"]["current_idea"], "Final idea after automatic reviewer refinement.")
                self.assertEqual(refreshed["chat_summary"]["auto_review_loop"]["status"], "completed")
            finally:
                os.chdir(old_cwd)

    def test_auto_review_loop_failure_is_structured_without_claiming_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                from research_alpha.db import create_idea_session

                db_path = cwd / "data" / "research_alpha.db"
                session_id = create_idea_session(db_path, "session-a", "Idea before failed automatic review.")
                summary = {
                    "session_id": session_id,
                    "name": "session-a",
                    "current_idea": "Idea before failed automatic review.",
                    "chat_summary": {"title": "session-a", "current_idea": "Idea before failed automatic review."},
                }
                with patch("research_alpha.app.cmd_review_loop", return_value=1):
                    loop = run_auto_review_loop_for_session(
                        cwd,
                        summary,
                        rounds=4,
                        provider="ds",
                        model="",
                        pattern_limit=5,
                        genome_limit=5,
                    )
                self.assertEqual(loop["status"], "failed")
                self.assertEqual(loop["reason"], "review_loop_failed")
                self.assertEqual(loop["session_id"], session_id)
                self.assertEqual(loop["rounds_completed"], 0)
                self.assertEqual(loop["final_idea"], "Idea before failed automatic review.")
                self.assertEqual(loop["recovery_strategy"], "retry_review_loop_without_state_change")
                self.assertIn("no_refinement_written", loop["state_policy"])
                self.assertEqual(loop["safe_resume_command"], f"ra rl --session-id {session_id}")
                stages = [{"stage": "review_loop", **loop}]
                self.assertEqual(infer_next_stages(stages), ["review_loop", "session_step", "dossier"])
                self.assertEqual(
                    infer_next_commands(
                        stages,
                        final_status="session_review_incomplete",
                        has_ranking=True,
                        has_session=True,
                    )[0],
                    f"ra rl --session-id {session_id}",
                )
                summary["auto_review_loop"] = loop
                refreshed = refresh_session_summary(cwd, summary)
                self.assertEqual(refreshed["chat_summary"]["auto_review_loop"]["status"], "failed")
                self.assertEqual(refreshed["chat_summary"]["current_idea"], "Idea before failed automatic review.")
            finally:
                os.chdir(old_cwd)

    def test_auto_review_loop_failure_reports_partial_completed_rounds(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                from research_alpha.db import add_idea_session_turn, create_idea_session

                db_path = cwd / "data" / "research_alpha.db"
                session_id = create_idea_session(db_path, "session-a", "Idea before partial failed review.")
                summary = {
                    "session_id": session_id,
                    "name": "session-a",
                    "current_idea": "Idea before partial failed review.",
                    "chat_summary": {"title": "session-a", "current_idea": "Idea before partial failed review."},
                }

                def fake_partial_review_loop(root_dir, **kwargs):
                    self.assertEqual(kwargs["session_id"], session_id)
                    add_idea_session_turn(
                        db_path,
                        session_id,
                        "review round 1",
                        "partial",
                        "Idea after reviewer round one.",
                        json.dumps({"decision": "partial", "revised_idea": "Idea after reviewer round one."}),
                    )
                    add_idea_session_turn(
                        db_path,
                        session_id,
                        "review round 2",
                        "partial",
                        "Idea after reviewer round two.",
                        json.dumps({"decision": "partial", "revised_idea": "Idea after reviewer round two."}),
                    )
                    return 1

                with patch("research_alpha.app.cmd_review_loop", side_effect=fake_partial_review_loop):
                    loop = run_auto_review_loop_for_session(
                        cwd,
                        summary,
                        rounds=4,
                        provider="ds",
                        model="",
                        pattern_limit=5,
                        genome_limit=5,
                    )
                self.assertEqual(loop["status"], "failed")
                self.assertEqual(loop["reason"], "review_loop_failed")
                self.assertEqual(loop["requested_rounds"], 4)
                self.assertEqual(loop["rounds_completed"], 2)
                self.assertEqual(loop["final_idea"], "Idea after reviewer round two.")
                self.assertEqual(loop["recovery_strategy"], "resume_review_loop_from_current_partial_idea")
                self.assertIn("partial_refinements_kept", loop["state_policy"])
                self.assertEqual(loop["safe_resume_command"], f"ra rl --session-id {session_id}")
                summary["auto_review_loop"] = loop
                refreshed = refresh_session_summary(cwd, summary)
                self.assertEqual(refreshed["chat_summary"]["auto_review_loop"]["status"], "failed")
                self.assertEqual(refreshed["chat_summary"]["auto_review_loop"]["rounds_completed"], 2)
                self.assertEqual(refreshed["chat_summary"]["current_idea"], "Idea after reviewer round two.")
            finally:
                os.chdir(old_cwd)

    def test_run_prints_structured_idea_generation_failure_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                from research_alpha.app import IdeaGenerationNotReady

                os.chdir(cwd)
                main(["init"])
                main(["ds", "sk-test"])
                add_paper(
                    cwd / "data" / "research_alpha.db",
                    title="Gold paper",
                    venue="ICLR",
                    year=2024,
                    abstract="A gold evidence paper.",
                    award="best_paper",
                    source_kind="seed",
                )
                summary = build_idea_generation_failure_summary(
                    query="research agents",
                    requested=1,
                    rejected_ideas=[
                        {
                            "title": "Copied transfer",
                            "reason": "failed storyline migration audit: transfer_repeats_source_story_standard; hard-transfer the source method",
                        }
                    ],
                )

                def fake_idea_stage(**kwargs):
                    raise IdeaGenerationNotReady("No generated ideas passed strict evidence audit.", summary)

                with (
                    patch("research_alpha.app.run_pipeline_trend_stage", return_value={"stage": "trends", "status": "completed", "paper_count": 1}),
                    patch("research_alpha.app.run_pipeline_genome_build", return_value=[(1, cwd / "outputs" / "genomes" / "paper-1.json")]),
                    patch("research_alpha.app.run_pipeline_pattern_build", return_value={"stage": "pattern", "status": "completed", "generated": 1}),
                    patch("research_alpha.app.run_pipeline_idea_stage", side_effect=fake_idea_stage),
                ):
                    output = io.StringIO()
                    with redirect_stdout(output):
                        exit_code = main(["r", "research agents", "--ideas", "1"])
                self.assertEqual(exit_code, 0)
                text = output.getvalue()
                self.assertIn("Idea-generation failure JSON:", text)
                failure_line = next(line for line in text.splitlines() if line.startswith("Idea-generation failure JSON: "))
                payload = json.loads(failure_line.split(": ", 1)[1])
                self.assertEqual(payload["primary_reason"], "storyline_migration_failed")
                manifest = json.loads(next((cwd / "outputs").glob("*/manifest.json")).read_text(encoding="utf-8"))
                ideas_stage = next(stage for stage in manifest["stages"] if stage["stage"] == "ideas")
                self.assertEqual(ideas_stage["failure_summary"]["primary_reason"], "storyline_migration_failed")
            finally:
                os.chdir(old_cwd)

    def test_run_creates_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                main(["init"])
                output = io.StringIO()
                with redirect_stdout(output):
                    exit_code = main(["run", "--query", "LLM agents", "--ideas", "3"])
                self.assertEqual(exit_code, 0)
                manifests = list((cwd / "outputs").glob("*/manifest.json"))
                self.assertEqual(len(manifests), 1)
                manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
                self.assertEqual(manifest["requested_ideas"], 3)
                self.assertEqual(manifest["status"], "partial")
                self.assertEqual(manifest["completed_stages"], [])
                self.assertEqual(manifest["skipped_stages"], ["score"])
                self.assertIn("harvest", manifest["next_stages"])
                self.assertEqual(manifest["stages"][0]["stage"], "score")
                self.assertEqual(manifest["stages"][0]["reason"], "no_papers_available")
                self.assertTrue((manifests[0].parent / "run_report.md").exists())
                self.assertEqual(Path(manifest["report_path"]).name, "run_report.md")
                self.assertIn("next_commands", manifest)
                self.assertIn("Run record:", output.getvalue())
                self.assertIn("Report:", output.getvalue())
            finally:
                os.chdir(old_cwd)

    def test_run_accepts_positional_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                main(["init"])
                output = io.StringIO()
                with redirect_stdout(output):
                    exit_code = main(["r", "LLM agents", "--ideas", "3"])
                self.assertEqual(exit_code, 0)
                manifests = list((cwd / "outputs").glob("*/manifest.json"))
                self.assertEqual(len(manifests), 1)
                manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
                self.assertEqual(manifest["query"], "LLM agents")
            finally:
                os.chdir(old_cwd)

    def test_default_entrypoint_runs_query_when_first_arg_is_not_a_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                main(["init"])
                output = io.StringIO()
                with redirect_stdout(output):
                    exit_code = main(["LLM agents", "--ideas", "3"])
                self.assertEqual(exit_code, 0)
                manifests = list((cwd / "outputs").glob("*/manifest.json"))
                self.assertEqual(len(manifests), 1)
                manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
                self.assertEqual(manifest["query"], "LLM agents")
                self.assertIn("Run record:", output.getvalue())
            finally:
                os.chdir(old_cwd)

    def test_default_entrypoint_accepts_run_flags_without_explicit_subcommand(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                main(["init"])
                output = io.StringIO()
                with redirect_stdout(output):
                    exit_code = main(["--query", "LLM agents", "--ideas", "2"])
                self.assertEqual(exit_code, 0)
                manifests = list((cwd / "outputs").glob("*/manifest.json"))
                self.assertEqual(len(manifests), 1)
                manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
                self.assertEqual(manifest["query"], "LLM agents")
                self.assertEqual(manifest["requested_ideas"], 2)
            finally:
                os.chdir(old_cwd)

    def test_run_uses_latest_query_when_omitted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                main(["init"])
                main(["r", "--query", "research agents", "--ideas", "1"])
                output = io.StringIO()
                with redirect_stdout(output):
                    exit_code = main(["r", "--ideas", "1"])
                self.assertEqual(exit_code, 0)
                self.assertIn("Using query (latest_run): research agents", output.getvalue())
                self.assertIn("Run record: 2", output.getvalue())
                manifests = sorted((cwd / "outputs").glob("*/manifest.json"))
                self.assertTrue(manifests)
                latest_manifest = json.loads(manifests[-1].read_text(encoding="utf-8"))
                self.assertEqual(latest_manifest["query"], "research agents")
            finally:
                os.chdir(old_cwd)

    def test_llm_command_persists_provider_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            old_provider = None
            old_model = None
            old_deepseek_key = None
            try:
                import os
                old_provider = os.environ.get("RA_LLM_PROVIDER")
                old_model = os.environ.get("RA_LLM_MODEL")
                old_deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
                os.chdir(cwd)
                main(["init"])
                output = io.StringIO()
                with redirect_stdout(output):
                    exit_code = main(
                        [
                            "llm",
                            "--provider",
                            "ds",
                            "--api-key",
                            "test-deepseek-key",
                            "--model",
                            "deepseek-chat",
                        ]
                    )
                self.assertEqual(exit_code, 0)
                text = output.getvalue()
                self.assertIn("Updated LLM settings:", text)
                self.assertIn("provider: deepseek", text)
                self.assertIn("api_key_configured: yes", text)
                env_text = (cwd / ".env").read_text(encoding="utf-8")
                self.assertIn("RA_LLM_PROVIDER=deepseek", env_text)
                self.assertIn("DEEPSEEK_API_KEY=test-deepseek-key", env_text)
                self.assertIn("RA_LLM_MODEL=deepseek-chat", env_text)

                status_out = io.StringIO()
                with redirect_stdout(status_out):
                    status_exit = main(["llm"])
                self.assertEqual(status_exit, 0)
                self.assertIn("provider: deepseek", status_out.getvalue())
                self.assertIn("api_key_configured: yes", status_out.getvalue())
            finally:
                import os
                if old_provider is not None:
                    os.environ["RA_LLM_PROVIDER"] = old_provider
                else:
                    os.environ.pop("RA_LLM_PROVIDER", None)
                if old_model is not None:
                    os.environ["RA_LLM_MODEL"] = old_model
                else:
                    os.environ.pop("RA_LLM_MODEL", None)
                if old_deepseek_key is not None:
                    os.environ["DEEPSEEK_API_KEY"] = old_deepseek_key
                else:
                    os.environ.pop("DEEPSEEK_API_KEY", None)
                os.chdir(old_cwd)

    def test_llm_command_accepts_positional_provider_switch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            old_provider = None
            try:
                import os
                old_provider = os.environ.get("RA_LLM_PROVIDER")
                os.chdir(cwd)
                main(["init"])
                output = io.StringIO()
                with redirect_stdout(output):
                    exit_code = main(["llm", "oa"])
                self.assertEqual(exit_code, 0)
                text = output.getvalue()
                self.assertIn("Updated LLM settings:", text)
                self.assertIn("provider: openai", text)
                env_text = (cwd / ".env").read_text(encoding="utf-8")
                self.assertIn("RA_LLM_PROVIDER=openai", env_text)
            finally:
                import os
                if old_provider is not None:
                    os.environ["RA_LLM_PROVIDER"] = old_provider
                else:
                    os.environ.pop("RA_LLM_PROVIDER", None)
                os.chdir(old_cwd)

    def test_llm_command_accepts_short_provider_and_positional_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            old_provider = None
            old_openai_key = None
            try:
                import os
                old_provider = os.environ.get("RA_LLM_PROVIDER")
                old_openai_key = os.environ.get("OPENAI_API_KEY")
                os.chdir(cwd)
                main(["init"])
                output = io.StringIO()
                with redirect_stdout(output):
                    exit_code = main(["llm", "oa", "test-openai-key"])
                self.assertEqual(exit_code, 0)
                text = output.getvalue()
                self.assertIn("Updated LLM settings:", text)
                self.assertIn("provider: openai", text)
                self.assertIn("api_key_configured: yes", text)
                env_text = (cwd / ".env").read_text(encoding="utf-8")
                self.assertIn("RA_LLM_PROVIDER=openai", env_text)
                self.assertIn("OPENAI_API_KEY=test-openai-key", env_text)
            finally:
                import os
                if old_provider is not None:
                    os.environ["RA_LLM_PROVIDER"] = old_provider
                else:
                    os.environ.pop("RA_LLM_PROVIDER", None)
                if old_openai_key is not None:
                    os.environ["OPENAI_API_KEY"] = old_openai_key
                else:
                    os.environ.pop("OPENAI_API_KEY", None)
                os.chdir(old_cwd)

    def test_llm_command_accepts_api_key_without_repeating_current_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            old_provider = None
            old_deepseek_key = None
            try:
                import os
                old_provider = os.environ.get("RA_LLM_PROVIDER")
                old_deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
                os.chdir(cwd)
                main(["init"])
                main(["llm", "ds"])
                output = io.StringIO()
                with redirect_stdout(output):
                    exit_code = main(["llm", "test-deepseek-key"])
                self.assertEqual(exit_code, 0)
                text = output.getvalue()
                self.assertIn("provider: deepseek", text)
                self.assertIn("api_key_target: DEEPSEEK_API_KEY", text)
                env_text = (cwd / ".env").read_text(encoding="utf-8")
                self.assertIn("RA_LLM_PROVIDER=deepseek", env_text)
                self.assertIn("DEEPSEEK_API_KEY=test-deepseek-key", env_text)
            finally:
                import os
                if old_provider is not None:
                    os.environ["RA_LLM_PROVIDER"] = old_provider
                else:
                    os.environ.pop("RA_LLM_PROVIDER", None)
                if old_deepseek_key is not None:
                    os.environ["DEEPSEEK_API_KEY"] = old_deepseek_key
                else:
                    os.environ.pop("DEEPSEEK_API_KEY", None)
                os.chdir(old_cwd)

    def test_top_level_provider_shortcut_sets_provider_and_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            old_provider = None
            old_deepseek_key = None
            try:
                import os
                old_provider = os.environ.get("RA_LLM_PROVIDER")
                old_deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
                os.chdir(cwd)
                main(["init"])
                output = io.StringIO()
                with redirect_stdout(output):
                    exit_code = main(["ds", "test-deepseek-key"])
                self.assertEqual(exit_code, 0)
                text = output.getvalue()
                self.assertIn("provider: deepseek", text)
                self.assertIn("api_key_configured: yes", text)
                env_text = (cwd / ".env").read_text(encoding="utf-8")
                self.assertIn("RA_LLM_PROVIDER=deepseek", env_text)
                self.assertIn("DEEPSEEK_API_KEY=test-deepseek-key", env_text)
            finally:
                import os
                if old_provider is not None:
                    os.environ["RA_LLM_PROVIDER"] = old_provider
                else:
                    os.environ.pop("RA_LLM_PROVIDER", None)
                if old_deepseek_key is not None:
                    os.environ["DEEPSEEK_API_KEY"] = old_deepseek_key
                else:
                    os.environ.pop("DEEPSEEK_API_KEY", None)
                os.chdir(old_cwd)

    def test_top_level_full_provider_name_shortcut_sets_provider_and_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            old_provider = None
            old_openai_key = None
            try:
                import os
                old_provider = os.environ.get("RA_LLM_PROVIDER")
                old_openai_key = os.environ.get("OPENAI_API_KEY")
                os.chdir(cwd)
                main(["init"])
                output = io.StringIO()
                with redirect_stdout(output):
                    exit_code = main(["openai", "test-openai-key"])
                self.assertEqual(exit_code, 0)
                text = output.getvalue()
                self.assertIn("provider: openai", text)
                self.assertIn("api_key_configured: yes", text)
                env_text = (cwd / ".env").read_text(encoding="utf-8")
                self.assertIn("RA_LLM_PROVIDER=openai", env_text)
                self.assertIn("OPENAI_API_KEY=test-openai-key", env_text)
            finally:
                import os
                if old_provider is not None:
                    os.environ["RA_LLM_PROVIDER"] = old_provider
                else:
                    os.environ.pop("RA_LLM_PROVIDER", None)
                if old_openai_key is not None:
                    os.environ["OPENAI_API_KEY"] = old_openai_key
                else:
                    os.environ.pop("OPENAI_API_KEY", None)
                os.chdir(old_cwd)

    def test_ask_alias_runs_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                main(["init"])
                output = io.StringIO()
                with patch("research_alpha.llm.LLMClient.chat") as mock_chat:
                    mock_chat.return_value = type("Resp", (), {"text": "hello from alias"})()
                    with redirect_stdout(output):
                        exit_code = main(["a", "test prompt"])
                self.assertEqual(exit_code, 0)
                self.assertEqual(mock_chat.call_args.kwargs["prompt"], "test prompt")
                self.assertIn("hello from alias", output.getvalue())
            finally:
                os.chdir(old_cwd)

    def test_ask_alias_without_prompt_runs_smoke_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                main(["init"])
                output = io.StringIO()
                with patch("research_alpha.llm.LLMClient.chat") as mock_chat:
                    mock_chat.return_value = type("Resp", (), {"text": "connection ok"})()
                    with redirect_stdout(output):
                        exit_code = main(["a"])
                self.assertEqual(exit_code, 0)
                self.assertIn("connection ok", output.getvalue())
                self.assertIn("LLM connection is working", mock_chat.call_args.kwargs["prompt"])
            finally:
                os.chdir(old_cwd)

    def test_repo_root_ra_wrapper_runs_init(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            repo_root = Path(__file__).resolve().parent.parent
            wrapper_path = repo_root / "ra"
            result = subprocess.run(
                [str(wrapper_path), "init"],
                cwd=cwd,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn("Initialized Research Alpha", result.stdout)
            self.assertIn("Zero-install fallback:", result.stdout)
            self.assertTrue((cwd / "data" / "research_alpha.db").exists())

    def test_init_scaffolds_local_ra_wrapper_that_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                exit_code = main(["init"])
                self.assertEqual(exit_code, 0)
                wrapper_path = cwd / "ra"
                self.assertTrue(wrapper_path.exists())
                result = subprocess.run(
                    [str(wrapper_path), "status"],
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, msg=result.stderr)
                self.assertIn("Project", result.stdout)
                self.assertIn("LLM", result.stdout)
            finally:
                os.chdir(old_cwd)

    def test_ask_provider_override_uses_matching_provider_key_and_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            captured: dict[str, object] = {}

            class FakeResponse:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

                def read(self) -> bytes:
                    return b'{"choices":[{"message":{"content":"override ok"}}]}'

            def fake_urlopen(request, timeout=90):
                captured["url"] = request.full_url
                captured["authorization"] = request.headers.get("Authorization")
                captured["payload"] = json.loads(request.data.decode("utf-8"))
                return FakeResponse()

            try:
                import os
                os.chdir(cwd)
                main(["init"])
                (cwd / ".env").write_text(
                    "RA_LLM_PROVIDER=openai\nOPENAI_API_KEY=test-openai-key\nDEEPSEEK_API_KEY=test-deepseek-key\n",
                    encoding="utf-8",
                )
                output = io.StringIO()
                with patch("urllib.request.urlopen", side_effect=fake_urlopen):
                    with redirect_stdout(output):
                        exit_code = main(["a", "test prompt", "--provider", "ds"])
                self.assertEqual(exit_code, 0)
                self.assertIn("override ok", output.getvalue())
                self.assertEqual(captured["url"], "https://api.deepseek.com/v1/chat/completions")
                self.assertEqual(captured["authorization"], "Bearer test-deepseek-key")
                self.assertEqual(captured["payload"]["model"], "deepseek-chat")
            finally:
                os.chdir(old_cwd)

    def test_run_provider_override_uses_matching_provider_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                main(["init"])
                (cwd / ".env").write_text(
                    "RA_LLM_PROVIDER=openai\nDEEPSEEK_API_KEY=test-deepseek-key\n",
                    encoding="utf-8",
                )
                (cwd / "seeds").mkdir(parents=True, exist_ok=True)
                (cwd / "seeds" / "mini.jsonl").write_text(
                    dedent(
                        """
                        {"title": "Agent Paper A", "venue": "ICLR", "year": 2025, "award": "best_paper", "citation_count": 100, "influential_citation_count": 40, "abstract": "Benchmark assumptions hide agent failures and brittle evaluation."}
                        {"title": "Agent Paper B", "venue": "ICLR", "year": 2025, "award": "oral", "citation_count": 90, "influential_citation_count": 35, "abstract": "Tool-use research agents expose planning and evaluation bottlenecks."}
                        {"title": "Agent Paper C", "venue": "ICLR", "year": 2025, "award": "outstanding_paper", "citation_count": 80, "influential_citation_count": 30, "abstract": "Stress tests reveal hidden assumptions in research agent systems."}
                        """
                    ).strip()
                    + "\n",
                    encoding="utf-8",
                )

                call_index = {"value": 0}

                def fake_chat(*args, **kwargs):
                    self_obj = args[0]
                    provider = self_obj.config.provider
                    model = self_obj.config.resolved_model
                    api_key = self_obj.config.api_key
                    if provider != "deepseek" or model != "deepseek-chat" or api_key != "test-deepseek-key":
                        raise AssertionError("provider override did not apply correctly")
                    call_index["value"] += 1
                    prompt = kwargs.get("prompt", "")
                    if "Aggregate the following genome cards" in prompt:
                        return type("Resp", (), {"text": grounded_fake_pattern_json(["Agent Paper A", "Agent Paper C"])})()
                    if "Create an Idea Genome Card" in prompt:
                        return grounded_fake_chat_response(prompt)
                    return type(
                        "Resp",
                        (),
                        {
                            "text": json.dumps(
                                {
                                    "ideas": [
                                        {
                                        "idea_title": "Stress-Tested Agent Benchmark",
                                        "core_hypothesis": "Benchmark agent robustness under stress.",
                                        "historical_pattern": "hidden-assumption-reversal",
                                        "trend_support": "Recent agentic planning work suggests need for stronger evaluation.",
                                        "frontier_gap": "Current benchmarks under-test failure behavior.",
                                        "why_now": "Agents are being deployed into messier settings.",
                                        "novelty": "Turns hidden assumptions into testable stress suites.",
                                        "value": "Could reset how research agents are evaluated.",
                                        "key_risk": "Might look like benchmark churn without theory.",
                                        "evaluation_outline": "Compare current agents on failure-taxonomy tasks.",
                                        "first_experiments": ["Build failure-taxonomy tasks", "Compare to current benchmarks"],
                                        "paper_angle": "A tougher benchmark reframes progress claims.",
                                        "evidence_basis": strict_evidence_basis(),
                                        "generation_guardrail": strict_generation_guardrail(),
                                        "storyline_trace": strict_storyline_trace(),
                                        }
                                    ]
                                }
                            )
                        },
                    )()

                output = io.StringIO()
                with patch("research_alpha.llm.LLMClient.chat", autospec=True, side_effect=fake_chat):
                    with redirect_stdout(output):
                        exit_code = main(
                            ["r", "--query", "research agents", "--file", "seeds/mini.jsonl", "--ideas", "1", "--provider", "ds"]
                        )
                self.assertEqual(exit_code, 0)
                text = output.getvalue()
                self.assertIn("Pipeline status:", text)
                self.assertNotIn("llm_not_configured", text)
                manifests = list((cwd / "outputs").glob("*/manifest.json"))
                manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
                self.assertIn("ideas", manifest["completed_stages"])
            finally:
                os.chdir(old_cwd)

    def test_run_with_seed_file_executes_partial_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                (cwd / "seeds").mkdir(parents=True, exist_ok=True)
                (cwd / "seeds" / "mini.jsonl").write_text(
                    dedent(
                        """
                        {"title":"Best Agent Benchmark","venue":"ICLR","year":2024,"award":"best_paper","citation_count":100,"influential_citation_count":40,"abstract":"Benchmark assumptions hide agent failures and brittle evaluation."}
                        {"title":"Oral Agent Tool Use","venue":"ICLR","year":2024,"award":"oral","citation_count":90,"influential_citation_count":35,"abstract":"Tool-use research agents expose planning and evaluation bottlenecks."}
                        {"title":"Outstanding Agent Stress Tests","venue":"ICLR","year":2024,"award":"outstanding_paper","citation_count":80,"influential_citation_count":30,"abstract":"Stress tests reveal hidden assumptions in research agent systems."}
                        """
                    ).strip()
                    + "\n",
                    encoding="utf-8",
                )
                main(["init"])
                output = io.StringIO()
                with redirect_stdout(output):
                    exit_code = main(["r", "--query", "research agents", "--file", "seeds/mini.jsonl"])
                self.assertEqual(exit_code, 0)
                text = output.getvalue()
                self.assertIn("Pipeline status: partial", text)
                self.assertIn("harvest: completed imported=3", text)
                self.assertIn("score: completed papers=3", text)
                self.assertIn("trends: completed papers=3", text)
                self.assertIn("genome: skipped (llm_not_configured)", text)

                manifests = list((cwd / "outputs").glob("*/manifest.json"))
                self.assertEqual(len(manifests), 1)
                manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
                self.assertEqual(manifest["completed_stages"], ["harvest", "score", "trends"])
                self.assertIn("configure_llm", manifest["next_stages"])
                self.assertTrue((manifests[0].parent / "run_report.md").exists())
                run_report_text = (manifests[0].parent / "run_report.md").read_text(encoding="utf-8")
                self.assertIn("## Stage Summary", run_report_text)
                self.assertIn("genome: skipped (llm_not_configured)", run_report_text)
                self.assertIn("## Next Commands", run_report_text)
                self.assertIn("ra ds sk-...", run_report_text)
                self.assertIn("ra oa sk-...", run_report_text)
                self.assertTrue((cwd / "outputs" / "trends" / "opportunity_map.html").exists())
            finally:
                os.chdir(old_cwd)

    def test_trends_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                (cwd / "seeds").mkdir(parents=True, exist_ok=True)
                (cwd / "seeds" / "mini.jsonl").write_text(
                    dedent(
                        """
                        {"title":"Transformer Agents","venue":"ICLR","year":2022,"award":"oral","citation_count":100,"influential_citation_count":40,"abstract":"Transformer agents for planning and tool use."}
                        {"title":"Agentic Planning","venue":"ICLR","year":2024,"award":"","citation_count":10,"influential_citation_count":5,"abstract":"Agentic planning with tool use and reflection."}
                        """
                    ).strip()
                    + "\n",
                    encoding="utf-8",
                )
                main(["init"])
                main(["h", "--file", "seeds/mini.jsonl"])
                output = io.StringIO()
                with redirect_stdout(output):
                    exit_code = main(["tr", "--years", "3", "--top-k", "5"])
                self.assertEqual(exit_code, 0)
                self.assertIn("Trend report:", output.getvalue())
                self.assertIn("Opportunity map:", output.getvalue())
                self.assertTrue((cwd / "outputs" / "trends" / "trend_report.md").exists())
                self.assertTrue((cwd / "outputs" / "trends" / "opportunity_map.html").exists())
                report_text = (cwd / "outputs" / "trends" / "trend_report.md").read_text(encoding="utf-8")
                html_text = (cwd / "outputs" / "trends" / "opportunity_map.html").read_text(encoding="utf-8")
                self.assertIn("# Trend Report", report_text)
                self.assertIn("Top Terms", report_text)
                self.assertIn("<title>Opportunity Map</title>", html_text)
                self.assertIn("Frontier Opportunities", html_text)
            finally:
                os.chdir(old_cwd)

    def test_trends_prefer_frontier_sources_over_gold_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                main(["init"])
                main(
                    [
                        "add-paper",
                        "--title",
                        "Learning Transferable Visual Models From Natural Language Supervision",
                        "--venue",
                        "ICML",
                        "--year",
                        "2021",
                        "--abstract",
                        "Image text contrastive learning creates transferable visual representations.",
                    ]
                )
                from research_alpha.db import add_paper
                from research_alpha.config import load_config

                config = load_config(cwd)
                add_paper(
                    config.db_path,
                    title="The rise and potential of large language model based agents: a survey",
                    venue="Science China Information Sciences",
                    year=2025,
                    abstract="Large language model based agents are rising for tool use and research automation.",
                    source_kind="openalex",
                    citation_count=388,
                )
                output = io.StringIO()
                with redirect_stdout(output):
                    exit_code = main(["tr", "--years", "5", "--top-k", "10"])
                self.assertEqual(exit_code, 0)
                report_text = (cwd / "outputs" / "trends" / "trend_report.md").read_text(encoding="utf-8")
                self.assertIn("trend_source_policy: frontier_source_only", report_text)
                self.assertIn("large language model based agents", report_text.lower())
                self.assertNotIn("Learning Transferable Visual Models", report_text)
            finally:
                os.chdir(old_cwd)

    def test_ideas_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                main(["init"])
                seed_strict_quality_papers(cwd)

                fake_genome = grounded_fake_genome_json()
                fake_pattern = grounded_fake_pattern_json()
                fake_ideas = json.dumps(
                    {
                        "ideas": [
                            {
                                "idea_title": "Stress-Tested Agent Benchmark",
                                "core_hypothesis": "Agent progress is overestimated by narrow metrics.",
                                "historical_pattern": "hidden-assumption-reversal",
                                "trend_support": "Recent agentic planning and tool-use work suggests demand for stronger evaluation.",
                                "frontier_gap": "Current agent benchmarks under-measure failure modes.",
                                "why_now": "Agent systems are now strong enough for deeper evaluation.",
                                "novelty": "Reframes benchmark design around hidden brittleness.",
                                "value": "Could reset how research agents are evaluated.",
                                "key_risk": "The benchmark may look too evaluative and not mechanism-driven.",
                                "first_experiments": ["Build failure-taxonomy tasks", "Compare to current benchmarks"],
                                "evaluation_outline": "Measure ranking changes and failure-mode coverage.",
                                "paper_angle": "Benchmarks have been rewarding the wrong competence.",
                                "evidence_basis": strict_evidence_basis(),
                                "generation_guardrail": strict_generation_guardrail(),
                                "storyline_trace": strict_storyline_trace()
                            }
                        ]
                    }
                )
                with patch("research_alpha.llm.LLMClient.chat") as mock_chat:
                    mock_chat.side_effect = [
                        type("Resp", (), {"text": fake_genome})(),
                        type("Resp", (), {"text": fake_genome})(),
                        type("Resp", (), {"text": fake_pattern})(),
                        type("Resp", (), {"text": fake_ideas})(),
                        type("Resp", (), {"text": prior_art_pass_json()})(),
                        type("Resp", (), {"text": prior_art_pass_json()})(),
                    ]
                    main(["gb", "--limit", "2"])
                    main(["pb", "--limit", "2"])
                    output = io.StringIO()
                    with redirect_stdout(output):
                        exit_code = main(["id", "--query", "research agents", "--ideas", "1"])
                self.assertEqual(exit_code, 0)
                self.assertIn("Generated 1 candidate ideas.", output.getvalue())
                self.assertIn("Stress-Tested Agent Benchmark", output.getvalue())
                self.assertTrue((cwd / "outputs" / "ideas" / "idea-1.json").exists())
                self.assertTrue((cwd / "outputs" / "dossiers" / "idea-1-dossier.json").exists())
                self.assertTrue((cwd / "outputs" / "dossiers" / "idea-1-dossier.md").exists())
                dossier_payload = json.loads(
                    (cwd / "outputs" / "dossiers" / "idea-1-dossier.json").read_text(encoding="utf-8")
                )
                self.assertEqual(dossier_payload["idea_id"], 1)
                self.assertEqual(dossier_payload["historical_pattern"], "hidden-assumption-reversal")
                self.assertIn("Recent agentic planning", dossier_payload["trend_thesis"])
                self.assertIn("Local prior-art screen", dossier_payload["prior_art_check"])
                self.assertEqual(dossier_payload["target_venue"], "ICLR")
            finally:
                os.chdir(old_cwd)

    def test_ideas_flow_uses_latest_run_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                main(["init"])
                seed_strict_quality_papers(cwd)
                with patch.dict(
                    os.environ,
                    {
                        "RA_LLM_PROVIDER": "openai",
                        "RA_LLM_MODEL": "",
                        "RA_LLM_API_KEY": "",
                        "OPENAI_API_KEY": "",
                        "DEEPSEEK_API_KEY": "",
                    },
                    clear=False,
                ):
                    main(["r", "--query", "research agents", "--ideas", "1"])

                fake_genome = grounded_fake_genome_json()
                fake_pattern = grounded_fake_pattern_json()
                fake_ideas = json.dumps(
                    {
                        "ideas": [
                            {
                                "idea_title": "Stress-Tested Agent Benchmark",
                                "core_hypothesis": "Agent progress is overestimated by narrow metrics.",
                                "historical_pattern": "hidden-assumption-reversal",
                                "trend_support": "Recent agentic planning and tool-use work suggests demand for stronger evaluation.",
                                "frontier_gap": "Current agent benchmarks under-measure failure modes.",
                                "why_now": "Agent systems are now strong enough for deeper evaluation.",
                                "novelty": "Reframes benchmark design around hidden brittleness.",
                                "value": "Could reset how research agents are evaluated.",
                                "key_risk": "The benchmark may look too evaluative and not mechanism-driven.",
                                "first_experiments": ["Build failure-taxonomy tasks", "Compare to current benchmarks"],
                                "evaluation_outline": "Measure ranking changes and failure-mode coverage.",
                                "paper_angle": "Benchmarks have been rewarding the wrong competence.",
                                "evidence_basis": strict_evidence_basis(),
                                "generation_guardrail": strict_generation_guardrail(),
                                "storyline_trace": strict_storyline_trace()
                            }
                        ]
                    }
                )
                with patch("research_alpha.llm.LLMClient.chat") as mock_chat:
                    mock_chat.side_effect = [
                        type("Resp", (), {"text": fake_genome})(),
                        type("Resp", (), {"text": fake_genome})(),
                        type("Resp", (), {"text": fake_pattern})(),
                        type("Resp", (), {"text": fake_ideas})(),
                        type("Resp", (), {"text": prior_art_pass_json()})(),
                    ]
                    main(["gb", "--limit", "2"])
                    main(["pb", "--limit", "2"])
                    output = io.StringIO()
                    with redirect_stdout(output):
                        exit_code = main(["id", "--ideas", "1"])
                self.assertEqual(exit_code, 0)
                self.assertIn("Using query (latest_run): research agents", output.getvalue())
                self.assertIn("Generated 1 candidate ideas.", output.getvalue())
                self.assertTrue((cwd / "outputs" / "ideas" / "idea-1.json").exists())
            finally:
                os.chdir(old_cwd)

    def test_ideas_accept_positional_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                main(["init"])
                seed_strict_quality_papers(cwd)

                fake_genome = grounded_fake_genome_json()
                fake_pattern = grounded_fake_pattern_json()
                fake_ideas = json.dumps(
                    {
                        "ideas": [
                            {
                                "idea_title": "Stress-Tested Agent Benchmark",
                                "core_hypothesis": "Agent progress is overestimated by narrow metrics.",
                                "historical_pattern": "hidden-assumption-reversal",
                                "trend_support": "Recent agentic planning and tool-use work suggests demand for stronger evaluation.",
                                "frontier_gap": "Current agent benchmarks under-measure failure modes.",
                                "why_now": "Agent systems are now strong enough for deeper evaluation.",
                                "novelty": "Reframes benchmark design around hidden brittleness.",
                                "value": "Could reset how research agents are evaluated.",
                                "key_risk": "The benchmark may look too evaluative and not mechanism-driven.",
                                "first_experiments": ["Build failure-taxonomy tasks", "Compare to current benchmarks"],
                                "evaluation_outline": "Measure ranking changes and failure-mode coverage.",
                                "paper_angle": "Benchmarks have been rewarding the wrong competence.",
                                "evidence_basis": strict_evidence_basis(),
                                "generation_guardrail": strict_generation_guardrail(),
                                "storyline_trace": strict_storyline_trace()
                            }
                        ]
                    }
                )
                fake_prior_art_gate = json.dumps(
                    {
                        "decision": "pass",
                        "overlap_label": "distant",
                        "closest_work": [],
                        "why_not_done_yet": ["The stored corpus lacks long-horizon failure traces and prior benchmarks used narrower evidence."],
                        "required_differentiation": ["Keep the mechanism and evaluation target distinct from existing benchmarks."],
                        "rethink_prompt": "",
                        "summary": "No close stored prior-art coverage; novelty depends on long-horizon failure evidence.",
                    }
                )
                with patch.dict(
                    os.environ,
                    {"RA_LLM_PROVIDER": "deepseek", "DEEPSEEK_API_KEY": "sk-test"},
                    clear=False,
                ), patch("research_alpha.llm.LLMClient.chat") as mock_chat:
                    mock_chat.side_effect = [
                        type("Resp", (), {"text": fake_genome})(),
                        type("Resp", (), {"text": fake_genome})(),
                        type("Resp", (), {"text": fake_pattern})(),
                        type("Resp", (), {"text": fake_ideas})(),
                        type("Resp", (), {"text": fake_prior_art_gate})(),
                    ]
                    main(["gb", "--limit", "2"])
                    main(["pb", "--limit", "2"])
                    output = io.StringIO()
                    with redirect_stdout(output):
                        exit_code = main(["id", "research agents", "--ideas", "1"])
                self.assertEqual(exit_code, 0)
                self.assertIn("Generated 1 candidate ideas.", output.getvalue())
                self.assertIn("Stress-Tested Agent Benchmark", output.getvalue())
                saved = json.loads((cwd / "outputs" / "ideas" / "idea-1.json").read_text(encoding="utf-8"))
                self.assertEqual(saved["prior_art_gate"]["decision"], "pass")
                self.assertEqual(saved["prior_art_gate"]["expert_used"], "llm_prior_art_expert")
                self.assertIn("long-horizon failure traces", saved["prior_art_gate"]["why_not_done_yet"][0])
            finally:
                os.chdir(old_cwd)

    def test_ideas_reject_duplicate_prior_art_before_saving(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                main(["init"])
                seed_strict_quality_papers(cwd)
                main(
                    [
                        "add-paper",
                        "--title",
                        "Stress Test Benchmark for Failure Mode Evaluation",
                        "--venue",
                        "ICLR",
                        "--year",
                        "2024",
                        "--abstract",
                        "A benchmark for agent failure mode evaluation with ranking instability and stress testing.",
                    ]
                )

                fake_genome = grounded_fake_genome_json()
                fake_pattern = grounded_fake_pattern_json()
                fake_ideas = json.dumps(
                    {
                        "ideas": [
                            {
                                "idea_title": "Stress Test Benchmark",
                                "core_hypothesis": "Agent progress is overestimated by narrow failure-insensitive metrics.",
                                "historical_pattern": "hidden-assumption-reversal",
                                "trend_support": "Recent tool-use agents create demand for stronger failure evaluation.",
                                "frontier_gap": "Current benchmarks under-measure failure modes and ranking instability.",
                                "why_now": "Agent systems are now strong enough for stress testing.",
                                "novelty": "Reframes benchmark design around failure stress tests.",
                                "value": "Could reset how agent evaluation is interpreted.",
                                "key_risk": "The contribution may look too close to existing evaluation papers.",
                                "first_experiments": ["Build failure stress tasks", "Compare benchmark rankings"],
                                "evaluation_outline": "Measure ranking changes and failure coverage.",
                                "paper_angle": "Benchmarks may be rewarding the wrong competence.",
                                "evidence_basis": strict_evidence_basis(),
                                "generation_guardrail": strict_generation_guardrail(),
                                "storyline_trace": strict_storyline_trace()
                            }
                        ]
                    }
                )
                with patch("research_alpha.llm.LLMClient.chat") as mock_chat:
                    mock_chat.side_effect = [
                        type("Resp", (), {"text": fake_genome})(),
                        type("Resp", (), {"text": fake_genome})(),
                        type("Resp", (), {"text": fake_pattern})(),
                        type("Resp", (), {"text": fake_ideas})(),
                        type("Resp", (), {"text": prior_art_pass_json()})(),
                    ]
                    main(["gb", "--limit", "2"])
                    main(["pb", "--limit", "2"])
                    err = io.StringIO()
                    with redirect_stderr(err):
                        exit_code = main(["id", "--query", "research agents", "--ideas", "1"])
                self.assertEqual(exit_code, 1)
                self.assertIn("prior_art_duplicate_rethink", err.getvalue())
                self.assertFalse((cwd / "outputs" / "ideas" / "idea-1.json").exists())
            finally:
                os.chdir(old_cwd)

    def test_idea_rank_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                main(["init"])
                seed_strict_quality_papers(cwd)

                fake_genome = grounded_fake_genome_json()
                fake_pattern = grounded_fake_pattern_json()
                fake_ideas = json.dumps(
                    {
                        "ideas": [
                            {
                                "idea_title": "Stress-Tested Agent Benchmark",
                                "core_hypothesis": "Agent progress is overestimated by narrow metrics.",
                                "historical_pattern": "hidden-assumption-reversal",
                                "trend_support": "Recent agentic planning and tool-use work suggests demand for stronger evaluation and failure-focused benchmarks.",
                                "frontier_gap": "Current agent benchmarks under-measure failure modes and tool-use brittleness.",
                                "why_now": "Agent systems are now strong enough for deeper evaluation.",
                                "novelty": "Reframes benchmark design around hidden brittleness and ranking instability.",
                                "value": "Could reset how research agents are evaluated and what counts as progress.",
                                "key_risk": "The benchmark may look too evaluative and not mechanism-driven.",
                                "first_experiments": ["Build failure-taxonomy tasks", "Compare to current benchmarks"],
                                "evaluation_outline": "Measure ranking changes, brittleness coverage, and transfer to realistic agent workflows.",
                                "paper_angle": "Benchmarks have been rewarding the wrong competence.",
                                "evidence_basis": strict_evidence_basis(),
                                "generation_guardrail": strict_generation_guardrail(),
                                "storyline_trace": strict_storyline_trace()
                            },
                            {
                                "idea_title": "Small Agent Metric",
                                "core_hypothesis": "A compact diagnostic metric might expose when agent comparisons hide brittle tool-use failures.",
                                "historical_pattern": "unknown-pattern",
                                "trend_support": "Agent evaluation papers increasingly compare tool-use workflows.",
                                "frontier_gap": "Existing metrics still blur different failure mechanisms into one score.",
                                "why_now": "The agent literature now has enough comparable systems to test diagnostic metrics.",
                                "novelty": "Turns a simple metric into a failure-diagnostic lens for agent comparisons.",
                                "value": "Could help researchers notice brittle systems before deployment claims harden.",
                                "key_risk": "The metric may be too narrow to change benchmark conclusions.",
                                "first_experiments": ["Compute the metric on two tool-use benchmarks and compare ranking shifts."],
                                "evaluation_outline": "Compare diagnostic sensitivity, ranking shifts, and failure-mode coverage.",
                                "paper_angle": "A compact metric should reveal hidden failure structure rather than only sort agents.",
                                "evidence_basis": strict_evidence_basis(),
                                "generation_guardrail": strict_generation_guardrail(),
                                "storyline_trace": strict_storyline_trace()
                            }
                        ]
                    }
                )
                with patch("research_alpha.llm.LLMClient.chat") as mock_chat:
                    mock_chat.side_effect = [
                        type("Resp", (), {"text": fake_genome})(),
                        type("Resp", (), {"text": fake_genome})(),
                        type("Resp", (), {"text": fake_pattern})(),
                        type("Resp", (), {"text": fake_ideas})(),
                    ]
                    main(["gb", "--limit", "2"])
                    main(["pb", "--limit", "2"])
                    main(["id", "--query", "research agents", "--ideas", "2"])

                output = io.StringIO()
                with redirect_stdout(output):
                    exit_code = main(["ir"])
                self.assertEqual(exit_code, 0)
                text = output.getvalue()
                self.assertIn("Ranked 2 candidate ideas", text)
                self.assertIn("Stress-Tested Agent Benchmark", text)
                self.assertIn("Small Agent Metric", text)
                self.assertIn("ra ix", text)
                ranking_payload = json.loads((cwd / "outputs" / "ideas" / "idea_ranking.json").read_text(encoding="utf-8"))
                self.assertEqual(ranking_payload["ranked_ideas"][0]["idea_id"], 1)
                self.assertEqual(ranking_payload["ranked_ideas"][0]["title"], "Stress-Tested Agent Benchmark")
                evidence_score = ranking_payload["ranked_ideas"][0]["evidence_grounded_score"]
                self.assertIn("Derived strictly from stored core Gold evidence", evidence_score["rubric_source"])
                self.assertIn("innovation", evidence_score["dimensions"])
                self.assertIn("logic", evidence_score["dimensions"])
                self.assertIn("feasibility", evidence_score["dimensions"])
                self.assertIn("value", evidence_score["dimensions"])
                self.assertIn("defensibility", evidence_score["dimensions"])
                self.assertGreaterEqual(evidence_score["total"], 0)
                self.assertIn("generation_evidence_audit", evidence_score)
                self.assertIn(
                    evidence_score["generation_evidence_audit"]["status"],
                    {"explicit_valid", "missing_model_evidence_inferred"},
                )
            finally:
                os.chdir(old_cwd)

    def test_idea_rank_penalizes_duplicate_like_prior_art(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                main(["init"])
                seed_strict_quality_papers(cwd)
                main(
                    [
                        "add-paper",
                        "--title",
                        "Stress Test Benchmark for Failure Mode Evaluation",
                        "--venue",
                        "ICLR",
                        "--year",
                        "2024",
                        "--abstract",
                        "A benchmark for agent failure mode evaluation with ranking instability and stress testing.",
                    ]
                )

                fake_genome = grounded_fake_genome_json()
                fake_pattern = grounded_fake_pattern_json()
                fake_ideas = json.dumps(
                    {
                        "ideas": [
                            {
                                "idea_title": "Stress Test Benchmark",
                                "core_hypothesis": "Agent progress is overestimated by narrow failure-insensitive metrics.",
                                "historical_pattern": "hidden-assumption-reversal",
                                "trend_support": "Recent tool-use agents create demand for stronger failure evaluation.",
                                "frontier_gap": "Current benchmarks under-measure failure modes and ranking instability.",
                                "why_now": "Agent systems are now strong enough for stress testing.",
                                "novelty": "Reframes benchmark design around failure stress tests.",
                                "value": "Could reset how agent evaluation is interpreted.",
                                "key_risk": "The contribution may look too close to existing evaluation papers.",
                                "first_experiments": ["Build failure stress tasks", "Compare benchmark rankings"],
                                "evaluation_outline": "Measure ranking changes and failure coverage.",
                                "paper_angle": "Benchmarks may be rewarding the wrong competence.",
                                "evidence_basis": strict_evidence_basis(),
                                "generation_guardrail": strict_generation_guardrail(),
                                "storyline_trace": strict_storyline_trace()
                            },
                            {
                                "idea_title": "Mechanistic Trace Stressor",
                                "core_hypothesis": "Mechanistic trace diagnostics may reveal why agent failures cluster under tool pressure.",
                                "historical_pattern": "hidden-assumption-reversal",
                                "trend_support": "Recent tool-use agents create demand for stronger failure evaluation.",
                                "frontier_gap": "Current benchmarks under-measure why systems fail, not just whether they fail.",
                                "why_now": "Agent systems now emit richer traces for explanation-oriented diagnostics.",
                                "novelty": "Connects stress testing to mechanism-level failure explanation instead of another benchmark only.",
                                "value": "Could make evaluation more actionable for researchers building agents.",
                                "key_risk": "Trace analysis may expand scope too far.",
                                "first_experiments": ["Probe tool traces under stress", "Predict benchmark failures from trace signatures"],
                                "evaluation_outline": "Check whether trace signals predict failure modes and ranking shifts.",
                                "paper_angle": "We need explanatory diagnostics, not just harder tests.",
                                "evidence_basis": strict_evidence_basis(),
                                "generation_guardrail": strict_generation_guardrail(),
                                "storyline_trace": strict_storyline_trace()
                            }
                        ]
                    }
                )
                with patch("research_alpha.llm.LLMClient.chat") as mock_chat:
                    mock_chat.side_effect = [
                        type("Resp", (), {"text": fake_genome})(),
                        type("Resp", (), {"text": fake_genome})(),
                        type("Resp", (), {"text": fake_pattern})(),
                        type("Resp", (), {"text": fake_ideas})(),
                    ]
                    main(["gb", "--limit", "2"])
                    main(["pb", "--limit", "2"])
                    main(["id", "--query", "research agents", "--ideas", "2"])

                output = io.StringIO()
                with redirect_stdout(output):
                    exit_code = main(["ir"])
                self.assertEqual(exit_code, 0)
                ranking_payload = json.loads((cwd / "outputs" / "ideas" / "idea_ranking.json").read_text(encoding="utf-8"))
                self.assertEqual(len(ranking_payload["ranked_ideas"]), 1)
                self.assertEqual(ranking_payload["ranked_ideas"][0]["title"], "Mechanistic Trace Stressor")
                self.assertIn(ranking_payload["ranked_ideas"][0]["prior_art_label"], {"distant", "complementary"})
                self.assertNotIn("prior_art=duplicate", output.getvalue())
            finally:
                os.chdir(old_cwd)

    def test_idea_rank_with_llm_rerank(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                main(["init"])
                seed_strict_quality_papers(cwd)

                fake_genome = grounded_fake_genome_json()
                fake_pattern = grounded_fake_pattern_json()
                fake_ideas = json.dumps(
                    {
                        "ideas": [
                            {
                                "idea_title": "Stress-Tested Agent Benchmark",
                                "core_hypothesis": "Agent progress is overestimated by narrow metrics.",
                                "historical_pattern": "hidden-assumption-reversal",
                                "trend_support": "Recent agentic planning and tool-use work suggests demand for stronger evaluation and failure-focused benchmarks.",
                                "frontier_gap": "Current agent benchmarks under-measure failure modes and tool-use brittleness.",
                                "why_now": "Agent systems are now strong enough for deeper evaluation.",
                                "novelty": "Reframes benchmark design around hidden brittleness and ranking instability.",
                                "value": "Could reset how research agents are evaluated and what counts as progress.",
                                "key_risk": "The benchmark may look too evaluative and not mechanism-driven.",
                                "first_experiments": ["Build failure-taxonomy tasks", "Compare to current benchmarks"],
                                "evaluation_outline": "Measure ranking changes, brittleness coverage, and transfer to realistic agent workflows.",
                                "paper_angle": "Benchmarks have been rewarding the wrong competence.",
                                "evidence_basis": strict_evidence_basis(),
                                "generation_guardrail": strict_generation_guardrail(),
                                "storyline_trace": strict_storyline_trace()
                            },
                            {
                                "idea_title": "Mechanistic Agent Stressor",
                                "core_hypothesis": "Mechanistic stress tests may reveal why agent benchmarks fail.",
                                "historical_pattern": "hidden-assumption-reversal",
                                "trend_support": "Recent tool-use agents create pressure for mechanistic diagnostics, not just scores.",
                                "frontier_gap": "Current evaluation rarely explains failure causes.",
                                "why_now": "Agent infrastructure is mature enough to probe internal decisions.",
                                "novelty": "Connects benchmark failures to mechanism-level diagnostics.",
                                "value": "Could make agent evaluation more actionable for builders.",
                                "key_risk": "Mechanistic probes may be too expensive.",
                                "first_experiments": ["Probe tool selection traces", "Link traces to benchmark failures"],
                                "evaluation_outline": "Check whether mechanistic diagnostics predict downstream benchmark failures.",
                                "paper_angle": "We do not just need harder tests, we need explanatory ones.",
                                "evidence_basis": strict_evidence_basis(),
                                "generation_guardrail": strict_generation_guardrail(),
                                "storyline_trace": strict_storyline_trace()
                            }
                        ]
                    }
                )
                with patch("research_alpha.llm.LLMClient.chat") as mock_chat:
                    mock_chat.side_effect = [
                        type("Resp", (), {"text": fake_genome})(),
                        type("Resp", (), {"text": fake_genome})(),
                        type("Resp", (), {"text": fake_pattern})(),
                        type("Resp", (), {"text": fake_ideas})(),
                    ]
                    main(["gb", "--limit", "2"])
                    main(["pb", "--limit", "2"])
                    main(["id", "--query", "research agents", "--ideas", "2"])

                fake_rerank = json.dumps(
                    {
                        "ranked_ids": [2, 1],
                        "summary": "The mechanistic idea is slightly stronger because it combines frontier relevance with a sharper explanatory angle.",
                        "why_top": "Idea 2 should be refined first because it turns evaluation into a mechanism-revealing contribution.",
                        "watchouts": ["Keep the scope manageable.", "Do not lose benchmark comparability."],
                    }
                )
                os.environ["OPENAI_API_KEY"] = "test-openai-key"
                try:
                    output = io.StringIO()
                    with patch("research_alpha.llm.LLMClient.chat") as mock_chat:
                        mock_chat.return_value = type("Resp", (), {"text": fake_rerank})()
                        with redirect_stdout(output):
                            exit_code = main(["ir", "--llm", "--provider", "oa"])
                finally:
                    os.environ.pop("OPENAI_API_KEY", None)
                self.assertEqual(exit_code, 0)
                text = output.getvalue()
                self.assertIn("LLM rerank:", text)
                self.assertIn("Mechanistic Agent Stressor", text)
                self.assertIn("LLM top pick:", text)
                self.assertIn("ra ix --idea-id 2", text)
                ranking_payload = json.loads((cwd / "outputs" / "ideas" / "idea_ranking.json").read_text(encoding="utf-8"))
                self.assertEqual(ranking_payload["llm_rerank"]["ranked_ids"][0], 2)
                self.assertEqual(
                    ranking_payload["ranked_ideas"][0]["critique_outputs"]["feasibility_assessment"]["status"],
                    "promising",
                )
            finally:
                os.chdir(old_cwd)

    def test_candidate_idea_to_session_bridge(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                main(["init"])
                seed_strict_quality_papers(cwd)

                fake_genome = grounded_fake_genome_json()
                fake_pattern = grounded_fake_pattern_json()
                fake_ideas = json.dumps(
                    {
                        "ideas": [
                            {
                                "idea_title": "Stress-Tested Agent Benchmark",
                                "core_hypothesis": "Agent progress is overestimated by narrow metrics.",
                                "historical_pattern": "hidden-assumption-reversal",
                                "trend_support": "Recent agentic planning and tool-use work suggests demand for stronger evaluation.",
                                "frontier_gap": "Current agent benchmarks under-measure failure modes.",
                                "why_now": "Agent systems are now strong enough for deeper evaluation.",
                                "novelty": "Reframes benchmark design around hidden brittleness.",
                                "value": "Could reset how research agents are evaluated.",
                                "key_risk": "The benchmark may look too evaluative and not mechanism-driven.",
                                "first_experiments": ["Build failure-taxonomy tasks", "Compare to current benchmarks"],
                                "evaluation_outline": "Measure ranking changes and failure-mode coverage.",
                                "paper_angle": "Benchmarks have been rewarding the wrong competence.",
                                "evidence_basis": strict_evidence_basis(),
                                "generation_guardrail": strict_generation_guardrail(),
                                "storyline_trace": strict_storyline_trace()
                            }
                        ]
                    }
                )
                with patch("research_alpha.llm.LLMClient.chat") as mock_chat:
                    mock_chat.side_effect = [
                        type("Resp", (), {"text": fake_genome})(),
                        type("Resp", (), {"text": fake_genome})(),
                        type("Resp", (), {"text": fake_pattern})(),
                        type("Resp", (), {"text": fake_ideas})(),
                    ]
                    main(["gb", "--limit", "2"])
                    main(["pb", "--limit", "2"])
                    main(["id", "--query", "research agents", "--ideas", "1"])

                list_out = io.StringIO()
                with redirect_stdout(list_out):
                    list_exit = main(["il"])
                self.assertEqual(list_exit, 0)
                self.assertIn("Stress-Tested Agent Benchmark", list_out.getvalue())

                bridge_out = io.StringIO()
                with redirect_stdout(bridge_out):
                    bridge_exit = main(["ix", "--latest"])
                self.assertEqual(bridge_exit, 0)
                self.assertIn("Created idea session #1 from candidate idea #1", bridge_out.getvalue())
                self.assertIn("Agent progress is overestimated by narrow metrics.", bridge_out.getvalue())
                self.assertTrue((cwd / "outputs" / "sessions" / "session-1-dossier.json").exists())
                self.assertTrue((cwd / "outputs" / "sessions" / "session-1-dossier.md").exists())
                dossier_payload = json.loads(
                    (cwd / "outputs" / "sessions" / "session-1-dossier.json").read_text(encoding="utf-8")
                )
                self.assertIn("Local prior-art screen", dossier_payload["prior_art_check"])
                self.assertIn("Build failure-taxonomy tasks", dossier_payload["feasibility"])
                self.assertIn("Could reset how research agents are evaluated.", dossier_payload["value_judgment"])
                self.assertEqual(dossier_payload["critique_outputs"]["feasibility_assessment"]["status"], "promising")
                self.assertEqual(dossier_payload["critique_outputs"]["value_judgment"]["status"], "high")
                self.assertEqual(dossier_payload["target_venue"], "ICLR")
                markdown_text = (cwd / "outputs" / "sessions" / "session-1-dossier.md").read_text(encoding="utf-8")
                self.assertIn("## Prior-Art Check", markdown_text)
                self.assertIn("## Experiment Plan", markdown_text)
            finally:
                os.chdir(old_cwd)

    def test_best_candidate_idea_to_session_bridge(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                main(["init"])
                seed_strict_quality_papers(cwd)

                fake_genome = grounded_fake_genome_json()
                fake_pattern = grounded_fake_pattern_json()
                fake_ideas = json.dumps(
                    {
                        "ideas": [
                            {
                                "idea_title": "Stress-Tested Agent Benchmark",
                                "core_hypothesis": "Agent progress is overestimated by narrow metrics.",
                                "historical_pattern": "hidden-assumption-reversal",
                                "trend_support": "Recent agentic planning and tool-use work suggests demand for stronger evaluation.",
                                "frontier_gap": "Current agent benchmarks under-measure failure modes.",
                                "why_now": "Agent systems are now strong enough for deeper evaluation.",
                                "novelty": "Reframes benchmark design around hidden brittleness.",
                                "value": "Could reset how research agents are evaluated.",
                                "key_risk": "The benchmark may look too evaluative and not mechanism-driven.",
                                "first_experiments": ["Build failure-taxonomy tasks", "Compare to current benchmarks"],
                                "evaluation_outline": "Measure ranking changes and failure-mode coverage.",
                                "paper_angle": "Benchmarks have been rewarding the wrong competence.",
                                "evidence_basis": strict_evidence_basis(),
                                "generation_guardrail": strict_generation_guardrail(),
                                "storyline_trace": strict_storyline_trace()
                            },
                            {
                                "idea_title": "Small Agent Metric",
                                "core_hypothesis": "A compact diagnostic metric might expose when agent comparisons hide brittle tool-use failures.",
                                "historical_pattern": "unknown-pattern",
                                "trend_support": "Agent evaluation papers increasingly compare tool-use workflows.",
                                "frontier_gap": "Existing metrics still blur different failure mechanisms into one score.",
                                "why_now": "The agent literature now has enough comparable systems to test diagnostic metrics.",
                                "novelty": "Turns a simple metric into a failure-diagnostic lens for agent comparisons.",
                                "value": "Could help researchers notice brittle systems before deployment claims harden.",
                                "key_risk": "The metric may be too narrow to change benchmark conclusions.",
                                "first_experiments": ["Compute the metric on two tool-use benchmarks and compare ranking shifts."],
                                "evaluation_outline": "Compare diagnostic sensitivity, ranking shifts, and failure-mode coverage.",
                                "paper_angle": "A compact metric should reveal hidden failure structure rather than only sort agents.",
                                "evidence_basis": strict_evidence_basis(),
                                "generation_guardrail": strict_generation_guardrail(),
                                "storyline_trace": strict_storyline_trace()
                            }
                        ]
                    }
                )
                with patch("research_alpha.llm.LLMClient.chat") as mock_chat:
                    mock_chat.side_effect = [
                        type("Resp", (), {"text": fake_genome})(),
                        type("Resp", (), {"text": fake_genome})(),
                        type("Resp", (), {"text": fake_pattern})(),
                        type("Resp", (), {"text": fake_ideas})(),
                    ]
                    main(["gb", "--limit", "2"])
                    main(["pb", "--limit", "2"])
                    main(["id", "--query", "research agents", "--ideas", "2"])

                bridge_out = io.StringIO()
                with redirect_stdout(bridge_out):
                    bridge_exit = main(["ix"])
                self.assertEqual(bridge_exit, 0)
                text = bridge_out.getvalue()
                ranking = build_candidate_idea_ranking(root_dir=cwd, query="", include_all=False, limit=1)
                top = ranking["shown_ideas"][0]
                self.assertIn(f"Created idea session #1 from top-ranked candidate idea #{top['idea_id']}", text)
                self.assertIn("Ranking scope: research agents", text)
                self.assertIn(str(top["title"]), text)
                self.assertIn("Idea-session summary JSON:", text)
                summary_line = next(line for line in text.splitlines() if line.startswith("Idea-session summary JSON: "))
                summary = json.loads(summary_line.removeprefix("Idea-session summary JSON: "))
                self.assertEqual(summary["title"], top["title"])
                self.assertTrue(str(summary["current_idea"]).strip())
                self.assertTrue(str(summary["trend_support"]).strip())
                self.assertIsInstance(summary["evidence_score"], (int, float))
                self.assertIsInstance(summary["evidence_basis"], list)
                self.assertGreaterEqual(len(summary["evidence_basis"]), 1)
                self.assertIn("source_type", summary["evidence_basis"][0])
                self.assertIsInstance(summary["scoring_dimensions"], list)
                self.assertTrue(any(item["dimension"] == "innovation" for item in summary["scoring_dimensions"]))
                self.assertIn("current_method_problem", summary)
                self.assertIn("storyline_logic", summary)
                self.assertIsInstance(summary["storyline_steps"], list)
                self.assertEqual(
                    [item["step"] for item in summary["storyline_steps"]],
                    ["old_belief", "bottleneck", "reframing", "why_now", "evidence_design", "failure_boundary"],
                )
                self.assertIn("Research-agent benchmarks assume", summary["storyline_steps"][0]["transfer"])
                self.assertTrue(str(summary["key_risk"]).strip())
                self.assertTrue(str(summary["first_experiment"]).strip())
                self.assertTrue((cwd / "outputs" / "sessions" / "session-1-dossier.json").exists())
            finally:
                os.chdir(old_cwd)

    def test_run_with_llm_completes_idea_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                (cwd / "seeds").mkdir(parents=True, exist_ok=True)
                (cwd / "seeds" / "mini.jsonl").write_text(
                    dedent(
                        """
                        {"title":"Best Agent Benchmark","venue":"ICLR","year":2024,"award":"best_paper","citation_count":100,"influential_citation_count":40,"abstract":"Benchmark assumptions hide agent failures and brittle evaluation."}
                        {"title":"Oral Agent Tool Use","venue":"ICLR","year":2024,"award":"oral","citation_count":90,"influential_citation_count":35,"abstract":"Tool-use research agents expose planning and evaluation bottlenecks."}
                        {"title":"Outstanding Agent Stress Tests","venue":"ICLR","year":2024,"award":"outstanding_paper","citation_count":80,"influential_citation_count":30,"abstract":"Stress tests reveal hidden assumptions in research agent systems."}
                        """
                    ).strip()
                    + "\n",
                    encoding="utf-8",
                )
                main(["init"])
                fake_genome = grounded_fake_genome_json()
                fake_pattern = grounded_fake_pattern_json(["Best Agent Benchmark", "Outstanding Agent Stress Tests"])
                fake_ideas = json.dumps(
                    {
                        "ideas": [
                            {
                                "idea_title": "Stress-Tested Agent Benchmark",
                                "core_hypothesis": "Agent progress is overestimated by narrow metrics.",
                                "historical_pattern": "hidden-assumption-reversal",
                                "trend_support": "Recent agentic planning and tool-use work suggests demand for stronger evaluation.",
                                "frontier_gap": "Current agent benchmarks under-measure failure modes.",
                                "why_now": "Agent systems are now strong enough for deeper evaluation.",
                                "novelty": "Reframes benchmark design around hidden brittleness.",
                                "value": "Could reset how research agents are evaluated.",
                                "key_risk": "The benchmark may look too evaluative and not mechanism-driven.",
                                "first_experiments": ["Build failure-taxonomy tasks", "Compare to current benchmarks"],
                                "evaluation_outline": "Measure ranking changes and failure-mode coverage.",
                                "paper_angle": "Benchmarks have been rewarding the wrong competence.",
                                "evidence_basis": strict_evidence_basis(),
                                "generation_guardrail": strict_generation_guardrail(),
                                "storyline_trace": strict_storyline_trace()
                            }
                        ]
                    }
                )
                with patch("research_alpha.llm.LLMClient.chat") as mock_chat:
                    mock_chat.side_effect = [
                        type("Resp", (), {"text": fake_genome})(),
                        type("Resp", (), {"text": fake_genome})(),
                        type("Resp", (), {"text": fake_pattern})(),
                        type("Resp", (), {"text": fake_ideas})(),
                        type("Resp", (), {"text": prior_art_pass_json()})(),
                    ]
                    os.environ["OPENAI_API_KEY"] = "test-openai-key"
                    try:
                        output = io.StringIO()
                        with redirect_stdout(output):
                            exit_code = main(["r", "--query", "research agents", "--file", "seeds/mini.jsonl", "--ideas", "1"])
                    finally:
                        os.environ.pop("OPENAI_API_KEY", None)
                self.assertEqual(exit_code, 0)
                self.assertIn("ideas: completed generated=1", output.getvalue())
                self.assertIn("ranking: completed ideas=1", output.getvalue())
                self.assertIn("Top idea: #1 Stress-Tested Agent Benchmark", output.getvalue())
                manifests = list((cwd / "outputs").glob("*/manifest.json"))
                self.assertEqual(len(manifests), 1)
                manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
                self.assertEqual(manifest["status"], "idea_candidates_ready")
                self.assertIn("ideas", manifest["completed_stages"])
                self.assertIn("ranking", manifest["completed_stages"])
                self.assertEqual(manifest["ranking"]["top_pick_id"], 1)
                self.assertTrue((cwd / "outputs" / "dossiers" / "idea-1-dossier.json").exists())
                self.assertTrue((cwd / "outputs" / "dossiers" / "idea-1-dossier.md").exists())
                run_dir = manifests[0].parent
                self.assertTrue((run_dir / "trend_report.md").exists())
                self.assertTrue((run_dir / "opportunity_map.html").exists())
                self.assertTrue((run_dir / "ideas" / "idea-1.json").exists())
                self.assertTrue((run_dir / "dossiers" / "idea-1-dossier.json").exists())
                self.assertTrue((run_dir / "dossiers" / "idea-1-dossier.md").exists())
                self.assertTrue((run_dir / "ideas" / "idea_ranking.json").exists())
                self.assertTrue(manifest["run_artifacts"])
                self.assertTrue((run_dir / "run_report.md").exists())
                run_report_text = (run_dir / "run_report.md").read_text(encoding="utf-8")
                self.assertIn("# Run ", run_report_text)
                self.assertIn("## Top Pick", run_report_text)
                self.assertIn("## Candidate Dossiers", run_report_text)
                self.assertIn("## Next Commands", run_report_text)
                self.assertIn("Stress-Tested Agent Benchmark", run_report_text)
                self.assertEqual(len(manifest["candidate_dossiers"]), 1)
                self.assertEqual(manifest["candidate_dossiers"][0]["idea_id"], 1)
                self.assertTrue(manifest["candidate_dossiers"][0]["dossier_markdown_path"].endswith("idea-1-dossier.md"))
            finally:
                os.chdir(old_cwd)

    def test_run_with_genome_grounding_failure_writes_partial_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                (cwd / "seeds").mkdir(parents=True, exist_ok=True)
                (cwd / "seeds" / "mini.jsonl").write_text(
                    dedent(
                        """
                        {"title":"Best Agent Benchmark","venue":"ICLR","year":2024,"award":"best_paper","citation_count":100,"influential_citation_count":40,"abstract":"Benchmark assumptions hide agent failures and brittle evaluation."}
                        {"title":"Outstanding Agent Stress Tests","venue":"ICLR","year":2024,"award":"outstanding_paper","citation_count":80,"influential_citation_count":30,"abstract":"Stress tests reveal hidden assumptions in research agent systems."}
                        """
                    ).strip()
                    + "\n",
                    encoding="utf-8",
                )
                main(["init"])
                bad_genome = json.dumps(
                    {
                        "paper_summary": "summary",
                        "pre_publication_belief": "belief",
                        "bottleneck_or_hidden_assumption": "bottleneck",
                        "problem_reframing": "reframing",
                        "why_now": "why now",
                        "evidence_design": "evidence",
                        "story_line": "story",
                        "transferable_pattern": "pattern",
                        "failure_boundary": "boundary",
                        "confidence_note": "abstract-only confidence",
                        "evidence_level": "abstract_only",
                        "logic_line": {
                            "old_belief": "belief",
                            "bottleneck": "bottleneck",
                            "reframing": "reframing",
                            "why_now": "why now",
                            "evidence_design": "evidence",
                            "failure_boundary": "boundary",
                        },
                    }
                )
                with patch("research_alpha.llm.LLMClient.chat", return_value=type("Resp", (), {"text": bad_genome})()):
                    os.environ["OPENAI_API_KEY"] = "test-openai-key"
                    try:
                        output = io.StringIO()
                        with redirect_stdout(output):
                            exit_code = main(["r", "--query", "research agents", "--file", "seeds/mini.jsonl", "--ideas", "1"])
                    finally:
                        os.environ.pop("OPENAI_API_KEY", None)
                self.assertEqual(exit_code, 0)
                self.assertIn("Pipeline status: partial", output.getvalue())
                self.assertIn("genome: skipped (grounding_audit_failed)", output.getvalue())
                self.assertIn("pattern: skipped (not_enough_grounded_genomes)", output.getvalue())
                manifests = list((cwd / "outputs").glob("*/manifest.json"))
                self.assertEqual(len(manifests), 1)
                manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
                self.assertEqual(manifest["status"], "partial")
                genome_stage = next(item for item in manifest["stages"] if item["stage"] == "genome")
                self.assertEqual(genome_stage["reason"], "grounding_audit_failed")
                self.assertIn("Genome grounding audit failed", genome_stage["detail"])
                self.assertNotIn("ideas", manifest["completed_stages"])
            finally:
                os.chdir(old_cwd)

    def test_run_with_pattern_grounding_failure_writes_partial_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                (cwd / "seeds").mkdir(parents=True, exist_ok=True)
                (cwd / "seeds" / "mini.jsonl").write_text(
                    dedent(
                        """
                        {"title":"Best Agent Benchmark","venue":"ICLR","year":2024,"award":"best_paper","citation_count":100,"influential_citation_count":40,"abstract":"Benchmark assumptions hide agent failures and brittle evaluation."}
                        {"title":"Outstanding Agent Stress Tests","venue":"ICLR","year":2024,"award":"outstanding_paper","citation_count":80,"influential_citation_count":30,"abstract":"Stress tests reveal hidden assumptions in research agent systems."}
                        """
                    ).strip()
                    + "\n",
                    encoding="utf-8",
                )
                main(["init"])
                bad_pattern = grounded_fake_pattern_json(["Unseen Paper"])

                def fake_chat(self, prompt, system_prompt=""):
                    if "Aggregate the following genome cards" in prompt:
                        return type("Resp", (), {"text": bad_pattern})()
                    return grounded_fake_chat_response(prompt)

                with patch("research_alpha.llm.LLMClient.chat", autospec=True, side_effect=fake_chat):
                    os.environ["OPENAI_API_KEY"] = "test-openai-key"
                    try:
                        output = io.StringIO()
                        with redirect_stdout(output):
                            exit_code = main(["r", "--query", "research agents", "--file", "seeds/mini.jsonl", "--ideas", "1"])
                    finally:
                        os.environ.pop("OPENAI_API_KEY", None)
                self.assertEqual(exit_code, 0)
                self.assertIn("Pipeline status: partial", output.getvalue())
                self.assertIn("pattern: skipped (grounding_audit_failed)", output.getvalue())
                manifests = list((cwd / "outputs").glob("*/manifest.json"))
                manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
                pattern_stage = next(item for item in manifest["stages"] if item["stage"] == "pattern")
                self.assertEqual(pattern_stage["reason"], "grounding_audit_failed")
                self.assertIn("Pattern grounding audit failed", pattern_stage["detail"])
                self.assertNotIn("ideas", manifest["completed_stages"])
            finally:
                os.chdir(old_cwd)

    def test_run_with_llm_and_five_ideas_writes_five_dossiers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                (cwd / "seeds").mkdir(parents=True, exist_ok=True)
                (cwd / "seeds" / "mini.jsonl").write_text(
                    dedent(
                        """
                        {"title":"Best Agent Benchmark","venue":"ICLR","year":2024,"award":"best_paper","citation_count":100,"influential_citation_count":40,"abstract":"Benchmark assumptions hide agent failures and brittle evaluation."}
                        {"title":"Oral Agent Tool Use","venue":"ICLR","year":2024,"award":"oral","citation_count":90,"influential_citation_count":35,"abstract":"Tool-use research agents expose planning and evaluation bottlenecks."}
                        {"title":"Outstanding Agent Stress Tests","venue":"ICLR","year":2024,"award":"outstanding_paper","citation_count":80,"influential_citation_count":30,"abstract":"Stress tests reveal hidden assumptions in research agent systems."}
                        """
                    ).strip()
                    + "\n",
                    encoding="utf-8",
                )
                main(["init"])
                fake_genome = grounded_fake_genome_json()
                fake_pattern = grounded_fake_pattern_json(["Best Agent Benchmark", "Outstanding Agent Stress Tests"])
                fake_ideas = json.dumps(
                    {
                        "ideas": [
                            {
                                "idea_title": f"Stress-Tested Agent Benchmark {idx}",
                                "core_hypothesis": f"Agent progress signal {idx} is overestimated by narrow metrics.",
                                "historical_pattern": "hidden-assumption-reversal",
                                "trend_support": "Recent agentic planning and tool-use work suggests demand for stronger evaluation.",
                                "frontier_gap": f"Current agent benchmarks under-measure failure mode family {idx}.",
                                "why_now": "Agent systems are now strong enough for deeper evaluation.",
                                "novelty": "Reframes benchmark design around hidden brittleness.",
                                "value": "Could reset how research agents are evaluated.",
                                "key_risk": "The benchmark may look too evaluative and not mechanism-driven.",
                                "first_experiments": [f"Build failure-taxonomy task set {idx}", f"Compare to current benchmark suite {idx}"],
                                "evaluation_outline": "Measure ranking changes and failure-mode coverage.",
                                "paper_angle": "Benchmarks have been rewarding the wrong competence.",
                                "evidence_basis": strict_evidence_basis(),
                                "generation_guardrail": strict_generation_guardrail(),
                                "storyline_trace": strict_storyline_trace()
                            }
                            for idx in range(1, 6)
                        ]
                    }
                )
                with patch("research_alpha.llm.LLMClient.chat") as mock_chat:
                    mock_chat.side_effect = [
                        type("Resp", (), {"text": fake_genome})(),
                        type("Resp", (), {"text": fake_genome})(),
                        type("Resp", (), {"text": fake_genome})(),
                        type("Resp", (), {"text": fake_pattern})(),
                        type("Resp", (), {"text": fake_ideas})(),
                        *[type("Resp", (), {"text": prior_art_pass_json()})() for _ in range(5)],
                    ]
                    os.environ["OPENAI_API_KEY"] = "test-openai-key"
                    try:
                        output = io.StringIO()
                        with redirect_stdout(output):
                            exit_code = main(["r", "--query", "research agents", "--file", "seeds/mini.jsonl", "--ideas", "5"])
                    finally:
                        os.environ.pop("OPENAI_API_KEY", None)
                self.assertEqual(exit_code, 0)
                self.assertIn("ideas: completed generated=5", output.getvalue())
                manifests = list((cwd / "outputs").glob("*/manifest.json"))
                self.assertEqual(len(manifests), 1)
                manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
                self.assertEqual(manifest["status"], "idea_candidates_ready")
                self.assertEqual(len(manifest["candidate_dossiers"]), 5)
                run_dir = manifests[0].parent
                dossier_jsons = sorted((run_dir / "dossiers").glob("idea-*-dossier.json"))
                dossier_markdowns = sorted((run_dir / "dossiers").glob("idea-*-dossier.md"))
                self.assertEqual(len(dossier_jsons), 5)
                self.assertEqual(len(dossier_markdowns), 5)
                markdown_text = dossier_markdowns[0].read_text(encoding="utf-8")
                self.assertIn("## Frontier Evidence", markdown_text)
                self.assertIn("## Prior-Art Check", markdown_text)
                self.assertIn("## Feasibility", markdown_text)
                self.assertIn("## Why-Not Analysis", markdown_text)
                self.assertIn("## Value Judgment", markdown_text)
                self.assertIn("## Experiment Plan", markdown_text)
                run_report_text = (run_dir / "run_report.md").read_text(encoding="utf-8")
                self.assertIn("## Candidate Dossiers", run_report_text)
                self.assertIn("Stress-Tested Agent Benchmark 1", run_report_text)
                self.assertIn("Stress-Tested Agent Benchmark 5", run_report_text)
            finally:
                os.chdir(old_cwd)

    def test_run_with_session_opens_refinement_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                (cwd / "seeds").mkdir(parents=True, exist_ok=True)
                (cwd / "seeds" / "mini.jsonl").write_text(
                    dedent(
                        """
                        {"title":"Best Agent Benchmark","venue":"ICLR","year":2024,"award":"best_paper","citation_count":100,"influential_citation_count":40,"abstract":"Benchmark assumptions hide agent failures and brittle evaluation."}
                        {"title":"Oral Agent Tool Use","venue":"ICLR","year":2024,"award":"oral","citation_count":90,"influential_citation_count":35,"abstract":"Tool-use research agents expose planning and evaluation bottlenecks."}
                        {"title":"Outstanding Agent Stress Tests","venue":"ICLR","year":2024,"award":"outstanding_paper","citation_count":80,"influential_citation_count":30,"abstract":"Stress tests reveal hidden assumptions in research agent systems."}
                        """
                    ).strip()
                    + "\n",
                    encoding="utf-8",
                )
                main(["init"])
                fake_genome = grounded_fake_genome_json()
                fake_pattern = grounded_fake_pattern_json(["Best Agent Benchmark", "Outstanding Agent Stress Tests"])
                fake_ideas = json.dumps(
                    {
                        "ideas": [
                            {
                                "idea_title": "Stress-Tested Agent Benchmark",
                                "core_hypothesis": "Agent progress is overestimated by narrow metrics.",
                                "historical_pattern": "hidden-assumption-reversal",
                                "trend_support": "Recent agentic planning and tool-use work suggests demand for stronger evaluation.",
                                "frontier_gap": "Current agent benchmarks under-measure failure modes.",
                                "why_now": "Agent systems are now strong enough for deeper evaluation.",
                                "novelty": "Reframes benchmark design around hidden brittleness.",
                                "value": "Could reset how research agents are evaluated.",
                                "key_risk": "The benchmark may look too evaluative and not mechanism-driven.",
                                "first_experiments": ["Build failure-taxonomy tasks", "Compare to current benchmarks"],
                                "evaluation_outline": "Measure ranking changes and failure-mode coverage.",
                                "paper_angle": "Benchmarks have been rewarding the wrong competence.",
                                "evidence_basis": strict_evidence_basis(),
                                "generation_guardrail": strict_generation_guardrail(),
                                "storyline_trace": strict_storyline_trace()
                            }
                        ]
                    }
                )
                with patch("research_alpha.llm.LLMClient.chat") as mock_chat:
                    mock_chat.side_effect = [
                        type("Resp", (), {"text": fake_genome})(),
                        type("Resp", (), {"text": fake_genome})(),
                        type("Resp", (), {"text": fake_pattern})(),
                        type("Resp", (), {"text": fake_ideas})(),
                        type("Resp", (), {"text": prior_art_pass_json()})(),
                    ]
                    os.environ["OPENAI_API_KEY"] = "test-openai-key"
                    try:
                        output = io.StringIO()
                        with redirect_stdout(output):
                            exit_code = main(
                                [
                                    "r",
                                    "--query",
                                    "research agents",
                                    "--file",
                                    "seeds/mini.jsonl",
                                    "--ideas",
                                    "1",
                                    "--session",
                                ]
                            )
                    finally:
                        os.environ.pop("OPENAI_API_KEY", None)
                self.assertEqual(exit_code, 0)
                self.assertIn("Auto session: #1", output.getvalue())
                self.assertIn('ra st --user "..."', output.getvalue())
                self.assertIn("ra sd", output.getvalue())
                manifests = list((cwd / "outputs").glob("*/manifest.json"))
                self.assertEqual(len(manifests), 1)
                manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
                self.assertEqual(manifest["status"], "session_ready")
                self.assertEqual(manifest["auto_session"]["session_id"], 1)
                self.assertEqual(manifest["ranking"]["top_pick_id"], 1)
                self.assertTrue((cwd / "outputs" / "sessions" / "session-1-dossier.json").exists())
            finally:
                os.chdir(old_cwd)

    def test_run_ranking_and_session_are_scoped_to_current_run_ideas(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                from research_alpha.db import add_candidate_idea

                os.chdir(cwd)
                (cwd / "seeds").mkdir(parents=True, exist_ok=True)
                (cwd / "seeds" / "mini.jsonl").write_text(
                    dedent(
                        """
                        {"title":"Best Agent Benchmark","venue":"ICLR","year":2024,"award":"best_paper","citation_count":100,"influential_citation_count":40,"abstract":"Benchmark assumptions hide agent failures and brittle evaluation."}
                        {"title":"Oral Agent Tool Use","venue":"ICLR","year":2024,"award":"oral","citation_count":90,"influential_citation_count":35,"abstract":"Tool-use research agents expose planning and evaluation bottlenecks."}
                        {"title":"Outstanding Agent Stress Tests","venue":"ICLR","year":2024,"award":"outstanding_paper","citation_count":80,"influential_citation_count":30,"abstract":"Stress tests reveal hidden assumptions in research agent systems."}
                        """
                    ).strip()
                    + "\n",
                    encoding="utf-8",
                )
                main(["init"])
                old_payload = {
                    "idea_title": "Older Same-Query Candidate",
                    "core_hypothesis": "An older candidate should not win a current run.",
                    "historical_pattern": "hidden-assumption-reversal",
                    "trend_support": "Recent agentic planning and tool-use work suggests demand for stronger evaluation.",
                    "frontier_gap": "Current agent benchmarks under-measure failure modes.",
                    "why_now": "Agent systems are now strong enough for deeper evaluation.",
                    "novelty": "Reframes benchmark design around hidden brittleness.",
                    "value": "Could reset how research agents are evaluated.",
                    "key_risk": "The benchmark may look too evaluative and not mechanism-driven.",
                    "first_experiments": ["Build failure-taxonomy tasks", "Compare to current benchmarks"],
                    "evaluation_outline": "Measure ranking changes and failure-mode coverage.",
                    "paper_angle": "Benchmarks have been rewarding the wrong competence.",
                    "evidence_basis": strict_evidence_basis(),
                    "generation_guardrail": strict_generation_guardrail(),
                    "storyline_trace": strict_storyline_trace(),
                }
                old_id = add_candidate_idea(
                    cwd / "data" / "research_alpha.db",
                    "research agents",
                    "Older Same-Query Candidate",
                    json.dumps(old_payload),
                )
                fake_genome = grounded_fake_genome_json()
                fake_pattern = grounded_fake_pattern_json(["Best Agent Benchmark", "Outstanding Agent Stress Tests"])
                fake_ideas = json.dumps(
                    {
                        "ideas": [
                            {
                                "idea_title": "Current Run Candidate",
                                "core_hypothesis": "Current run ideas should define the run ranking scope.",
                                "historical_pattern": "hidden-assumption-reversal",
                                "trend_support": "Recent agentic planning and tool-use work suggests demand for stronger evaluation.",
                                "frontier_gap": "Current agent benchmarks under-measure failure modes.",
                                "why_now": "Agent systems are now strong enough for deeper evaluation.",
                                "novelty": "Reframes benchmark design around hidden brittleness.",
                                "value": "Could reset how research agents are evaluated.",
                                "key_risk": "The benchmark may look too evaluative and not mechanism-driven.",
                                "first_experiments": ["Build failure-taxonomy tasks", "Compare to current benchmarks"],
                                "evaluation_outline": "Measure ranking changes and failure-mode coverage.",
                                "paper_angle": "Benchmarks have been rewarding the wrong competence.",
                                "evidence_basis": strict_evidence_basis(),
                                "generation_guardrail": strict_generation_guardrail(),
                                "storyline_trace": strict_storyline_trace(),
                            }
                        ]
                    }
                )
                with patch("research_alpha.llm.LLMClient.chat") as mock_chat:
                    mock_chat.side_effect = [
                        type("Resp", (), {"text": fake_genome})(),
                        type("Resp", (), {"text": fake_genome})(),
                        type("Resp", (), {"text": fake_pattern})(),
                        type("Resp", (), {"text": fake_ideas})(),
                    ]
                    os.environ["OPENAI_API_KEY"] = "test-openai-key"
                    try:
                        output = io.StringIO()
                        with redirect_stdout(output):
                            exit_code = main(
                                [
                                    "r",
                                    "--query",
                                    "research agents",
                                    "--file",
                                    "seeds/mini.jsonl",
                                    "--ideas",
                                    "1",
                                    "--session",
                                ]
                            )
                    finally:
                        os.environ.pop("OPENAI_API_KEY", None)
                self.assertEqual(exit_code, 0)
                manifests = list((cwd / "outputs").glob("*/manifest.json"))
                self.assertEqual(len(manifests), 1)
                manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
                self.assertNotEqual(manifest["ranking"]["top_pick_id"], old_id)
                self.assertEqual(manifest["ranking"]["top_pick_title"], "Current Run Candidate")
                self.assertEqual(manifest["auto_session"]["candidate_idea_id"], manifest["ranking"]["top_pick_id"])
                ranking_payload = json.loads((cwd / "outputs" / "ideas" / "idea_ranking.json").read_text(encoding="utf-8"))
                self.assertEqual(ranking_payload["total_ranked"], 1)
                self.assertEqual(ranking_payload["ranked_ideas"][0]["title"], "Current Run Candidate")
            finally:
                os.chdir(old_cwd)

    def test_harvest_score_and_top_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                (cwd / "seeds").mkdir(parents=True, exist_ok=True)
                (cwd / "seeds" / "mini.jsonl").write_text(
                    dedent(
                        """
                        {"title":"Paper A","venue":"ICLR","year":2024,"award":"best_paper","citation_count":100,"influential_citation_count":40}
                        {"title":"Paper B","venue":"ICLR","year":2024,"award":"oral","citation_count":90,"influential_citation_count":35}
                        {"title":"Paper C","venue":"ICLR","year":2024,"award":"spotlight","citation_count":80,"influential_citation_count":30}
                        """
                    ).strip()
                    + "\n",
                    encoding="utf-8",
                )
                main(["init"])
                harvest_out = io.StringIO()
                with redirect_stdout(harvest_out):
                    harvest_exit = main(["harvest", "--file", "seeds/mini.jsonl"])
                self.assertEqual(harvest_exit, 0)
                self.assertIn("Harvested 3 records", harvest_out.getvalue())

                score_out = io.StringIO()
                with redirect_stdout(score_out):
                    score_exit = main(["score"])
                self.assertEqual(score_exit, 0)
                self.assertIn("Scored 3 papers", score_out.getvalue())

                top_out = io.StringIO()
                with redirect_stdout(top_out):
                    top_exit = main(["top", "--limit", "2"])
                self.assertEqual(top_exit, 0)
                output = top_out.getvalue()
                self.assertIn("Paper A", output)
                self.assertIn("score=", output)
            finally:
                os.chdir(old_cwd)

    def test_gold_seed_imports_only_core_gold_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                seed_path = cwd / "seeds" / "local_gold.jsonl"
                seed_path.write_text(
                    "\n".join(
                        [
                            json.dumps(
                                {
                                    "title": "Local Best Gold",
                                    "venue": "ICLR",
                                    "year": 2026,
                                    "award": "best_paper",
                                    "abstract": "Full logic evidence.",
                                    "external_ref": "https://arxiv.org/abs/2601.00001",
                                }
                            ),
                            json.dumps(
                                {
                                    "title": "Local Oral Gold",
                                    "venue": "NeurIPS",
                                    "year": 2026,
                                    "award": "oral",
                                    "abstract": "Reviewer clarity evidence.",
                                    "external_ref": "https://arxiv.org/abs/2601.00002",
                                }
                            ),
                            json.dumps(
                                {
                                    "title": "Trend Spotlight Only",
                                    "venue": "ICLR",
                                    "year": 2026,
                                    "award": "spotlight",
                                    "abstract": "Trend only.",
                                }
                            ),
                            json.dumps(
                                {
                                    "title": "Poster Should Stay Out",
                                    "venue": "ICLR Poster",
                                    "year": 2026,
                                    "award": "best_paper",
                                    "abstract": "Poster text.",
                                }
                            ),
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )

                output = io.StringIO()
                with redirect_stdout(output):
                    exit_code = main(["gold-seed", "--file", str(seed_path), "--limit", "10"])
                self.assertEqual(exit_code, 0)
                self.assertIn("imported=2", output.getvalue())
                self.assertIn("poster_skipped=1", output.getvalue())
                self.assertIn("non_gold_skipped=1", output.getvalue())
                db_path = cwd / "data" / "research_alpha.db"
                with connect(db_path) as conn:
                    rows = conn.execute(
                        "SELECT title, source_kind, award, paper_weight FROM papers ORDER BY title ASC"
                    ).fetchall()
                self.assertEqual([row["title"] for row in rows], ["Local Best Gold", "Local Oral Gold"])
                self.assertTrue(all(row["source_kind"] == "gold_seed" for row in rows))
                self.assertEqual({row["award"] for row in rows}, {"best_paper", "oral"})
                self.assertTrue(all(float(row["paper_weight"]) > 0 for row in rows))
            finally:
                os.chdir(old_cwd)

    def test_gold_seed_can_fetch_full_text_for_imported_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                seed_path = cwd / "seeds" / "local_gold.jsonl"
                seed_path.write_text(
                    json.dumps(
                        {
                            "title": "Local Full Text Gold",
                            "venue": "ICML",
                            "year": 2026,
                            "award": "best_paper",
                            "abstract": "Short abstract.",
                            "external_ref": "https://example.org/full.html",
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )

                def fake_fetch(url, **kwargs):
                    self.assertEqual(url, "https://example.org/full.html")
                    return (
                        "Introduction Local setup. Related Work Local prior work. "
                        "Methods Local reframing. Experiments Local validation. "
                        "Limitations Local boundary. Conclusion Local close.",
                        url,
                    )

                with patch("research_alpha.app.fetch_full_text_with_metadata", side_effect=fake_fetch):
                    exit_code = main(["gold-seed", "--file", str(seed_path), "--fulltext"])
                self.assertEqual(exit_code, 0)
                with connect(cwd / "data" / "research_alpha.db") as conn:
                    row = conn.execute(
                        "SELECT source_kind, paper_weight, full_text_json FROM papers WHERE title='Local Full Text Gold'"
                    ).fetchone()
                self.assertEqual(row["source_kind"], "gold_seed")
                self.assertGreater(float(row["paper_weight"]), 0.0)
                self.assertIn("Local setup", row["full_text_json"])
                self.assertIn("source_url", row["full_text_json"])
            finally:
                os.chdir(old_cwd)

    def test_single_frontier_paper_does_not_become_gold_set_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                main(["init"])
                main(
                    [
                        "add-paper",
                        "--title",
                        "The rise and potential of large language model based agents: a survey",
                        "--venue",
                        "Science China Information Sciences",
                        "--year",
                        "2025",
                        "--abstract",
                        "A recent survey of large language model based agents.",
                    ]
                )
                main(["score"])
                payload, _, _ = build_evidence_report(cwd, paper_limit=5)
                self.assertEqual(payload["counts"]["high_weight_papers"], 0)
                self.assertIn(
                    "Add or harvest at least 3 scored core Gold papers with explicit Best/Outstanding/Oral signals.",
                    payload["missing_requirements"],
                )
            finally:
                os.chdir(old_cwd)

    def test_frontier_cohort_does_not_become_gold_set_by_citation_percentile(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                main(["init"])
                db_path = cwd / "data" / "research_alpha.db"
                for idx, cites in enumerate([500, 300, 100], start=1):
                    add_paper(
                        db_path,
                        title=f"Recent Frontier Agent Paper {idx}",
                        venue="ICLR",
                        year=2026,
                        abstract="Recent frontier agent paper used only for trend analysis.",
                        source_kind="frontier_s2",
                        citation_count=cites,
                    )
                main(["score"])
                payload, _, _ = build_evidence_report(cwd, paper_limit=5)
                self.assertEqual(payload["counts"]["high_weight_papers"], 0)
                with connect(db_path) as conn:
                    rows = conn.execute(
                        "SELECT paper_weight, score_notes FROM papers ORDER BY id ASC"
                    ).fetchall()
                self.assertTrue(all(float(row["paper_weight"]) == 0.0 for row in rows))
                self.assertTrue(all("frontier_trend_only_not_gold_standard" in row["score_notes"] for row in rows))
            finally:
                os.chdir(old_cwd)

    def test_unverified_remote_gold_candidates_do_not_get_citation_bonus(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                db_path = cwd / "data" / "research_alpha.db"
                for idx, cites in enumerate([500, 300, 100], start=1):
                    add_paper(
                        db_path,
                        title=f"OpenAlex Candidate {idx}",
                        venue="ICLR",
                        year=2024,
                        abstract="Remote venue-year candidate without verified award metadata.",
                        source_kind="gold_openalex",
                        citation_count=cites,
                    )
                main(["score"])
                with connect(db_path) as conn:
                    rows = conn.execute(
                        "SELECT paper_weight, score_notes FROM papers ORDER BY id ASC"
                    ).fetchall()
                self.assertTrue(all(float(row["paper_weight"]) == 0.0 for row in rows))
                self.assertTrue(all("unverified_remote_gold_candidate" in row["score_notes"] for row in rows))
            finally:
                os.chdir(old_cwd)

    def test_batch_genome_build_uses_only_gold_set_papers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                from research_alpha.db import connect
                os.chdir(cwd)
                main(["init"])
                for title, award in [
                    ("Historical Best Paper", "best_paper"),
                    ("Historical Oral Paper", "oral"),
                    ("Historical Spotlight Paper", "spotlight"),
                ]:
                    main(
                        [
                            "add-paper",
                            "--title",
                            title,
                            "--venue",
                            "ICLR",
                            "--year",
                            "2024",
                            "--abstract",
                            "Historical logic reveals benchmark assumptions, hidden agent failures, and brittle evaluation.",
                        ]
                    )
                    with connect(cwd / "data" / "research_alpha.db") as conn:
                        conn.execute("update papers set award=? where title=?", (award, title))
                main(
                    [
                        "add-paper",
                        "--title",
                        "Frontier Agent Survey",
                        "--venue",
                        "Science China Information Sciences",
                        "--year",
                        "2025",
                        "--abstract",
                        "A current frontier survey about agents.",
                    ]
                )
                with connect(cwd / "data" / "research_alpha.db") as conn:
                    conn.execute(
                        "update papers set source_kind='openalex', citation_count=388, award='' where title='Frontier Agent Survey'"
                    )
                main(["score"])
                with connect(cwd / "data" / "research_alpha.db") as conn:
                    frontier = conn.execute(
                        "select paper_weight from papers where title='Frontier Agent Survey'"
                    ).fetchone()
                self.assertEqual(float(frontier["paper_weight"]), 0.0)
                with patch("research_alpha.llm.LLMClient.chat", side_effect=lambda prompt, system_prompt=None: grounded_fake_chat_response(prompt)) as mock_chat:
                    os.environ["OPENAI_API_KEY"] = "test-openai-key"
                    try:
                        from research_alpha.app import run_pipeline_genome_build

                        outputs = run_pipeline_genome_build(
                            cwd,
                            limit=4,
                            provider=None,
                            model="",
                            include_existing=False,
                        )
                    finally:
                        os.environ.pop("OPENAI_API_KEY", None)
                generated_ids = [paper_id for paper_id, _ in outputs]
                self.assertEqual(len(generated_ids), 2)
                self.assertIn(1, generated_ids)
                self.assertIn(2, generated_ids)
                self.assertNotIn(3, generated_ids)
                self.assertNotIn(4, generated_ids)
                self.assertFalse((cwd / "outputs" / "genomes" / "paper-3.json").exists())
                self.assertFalse((cwd / "outputs" / "genomes" / "paper-4.json").exists())
            finally:
                os.chdir(old_cwd)

    def test_pattern_build_records_gold_only_source_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                from research_alpha.db import connect

                os.chdir(cwd)
                main(["init"])
                for title, award in [
                    ("Historical Best Paper", "best_paper"),
                    ("Historical Oral Paper", "oral"),
                    ("Historical Spotlight Paper", "spotlight"),
                ]:
                    main(
                        [
                            "add-paper",
                            "--title",
                            title,
                            "--venue",
                            "ICLR",
                            "--year",
                            "2024",
                            "--abstract",
                            "Historical logic reveals benchmark assumptions, hidden agent failures, and brittle evaluation.",
                        ]
                    )
                    with connect(cwd / "data" / "research_alpha.db") as conn:
                        conn.execute("update papers set award=? where title=?", (award, title))
                main(
                    [
                        "add-paper",
                        "--title",
                        "Frontier Agent Survey",
                        "--venue",
                        "Science China Information Sciences",
                        "--year",
                        "2025",
                        "--abstract",
                        "A current frontier survey about agents.",
                    ]
                )
                with connect(cwd / "data" / "research_alpha.db") as conn:
                    conn.execute(
                        "update papers set source_kind='openalex', citation_count=388, award='' where title='Frontier Agent Survey'"
                    )
                main(["score"])
                fake_pattern_payload = json.loads(grounded_fake_pattern_json(["Historical Best Paper", "Historical Oral Paper"]))
                fake_pattern_payload["pattern_key"] = "historical-transfer"
                fake_pattern_payload["pattern_name"] = "Historical Transfer"
                fake_pattern = json.dumps(fake_pattern_payload)
                with patch("research_alpha.llm.LLMClient.chat") as mock_chat:
                    mock_chat.side_effect = lambda prompt, system_prompt=None: (
                        type("Resp", (), {"text": fake_pattern})()
                        if "Aggregate the following genome cards" in prompt
                        else grounded_fake_chat_response(prompt)
                    )
                    os.environ["OPENAI_API_KEY"] = "test-openai-key"
                    try:
                        from research_alpha.app import run_pipeline_genome_build, run_pipeline_pattern_build

                        run_pipeline_genome_build(
                            cwd,
                            limit=4,
                            provider=None,
                            model="",
                            include_existing=False,
                        )
                        result = run_pipeline_pattern_build(
                            cwd,
                            limit=4,
                            provider=None,
                            model="",
                        )
                    finally:
                        os.environ.pop("OPENAI_API_KEY", None)
                self.assertEqual(result["status"], "completed")
                pattern_payload = json.loads((cwd / "outputs" / "patterns" / "historical-transfer.json").read_text())
                self.assertEqual(pattern_payload["source_set_policy"], "gold_set_only")
                self.assertTrue(pattern_payload["source_papers"])
                self.assertTrue(all(float(item["paper_weight"]) > 0 for item in pattern_payload["source_papers"]))
                self.assertNotIn("Frontier Agent Survey", pattern_payload["source_titles"])
            finally:
                os.chdir(old_cwd)

    def test_evidence_report_backfills_legacy_strict_pattern_source_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                from research_alpha.db import connect

                os.chdir(cwd)
                main(["init"])
                for title, award in [
                    ("Historical Best Paper", "best_paper"),
                    ("Historical Oral Paper", "oral"),
                    ("Historical Spotlight Paper", "spotlight"),
                ]:
                    main(
                        [
                            "add-paper",
                            "--title",
                            title,
                            "--venue",
                            "ICLR",
                            "--year",
                            "2024",
                            "--abstract",
                            "Historical logic reveals benchmark assumptions, hidden agent failures, and brittle evaluation.",
                        ]
                    )
                    with connect(cwd / "data" / "research_alpha.db") as conn:
                        conn.execute("update papers set award=? where title=?", (award, title))
                main(
                    [
                        "add-paper",
                        "--title",
                        "Frontier Agent Survey",
                        "--venue",
                        "Science China Information Sciences",
                        "--year",
                        "2025",
                        "--abstract",
                        "A current frontier survey about agents.",
                    ]
                )
                with connect(cwd / "data" / "research_alpha.db") as conn:
                    conn.execute(
                        "update papers set source_kind='openalex', citation_count=388, award='' where title='Frontier Agent Survey'"
                    )
                main(["score"])
                with patch("research_alpha.llm.LLMClient.chat", side_effect=lambda prompt, system_prompt=None: grounded_fake_chat_response(prompt)) as mock_chat:
                    os.environ["OPENAI_API_KEY"] = "test-openai-key"
                    try:
                        from research_alpha.app import run_pipeline_genome_build

                        run_pipeline_genome_build(
                            cwd,
                            limit=4,
                            provider=None,
                            model="",
                            include_existing=False,
                        )
                    finally:
                        os.environ.pop("OPENAI_API_KEY", None)
                legacy_pattern = {
                    "pattern_key": "legacy-transfer",
                    "pattern_name": "Legacy Transfer",
                    "core_move": "Transfer complete historical logic.",
                    "when_to_use": "When frontier gaps match historical logic.",
                    "why_it_works": "It preserves causal argument structure.",
                    "failure_modes": "Fails if source papers are not high-quality.",
                    "canonical_examples": ["Historical Best Paper", "Historical Oral Paper"],
                    "operator_template": "Move source logic into the frontier gap.",
                    "evidence_level": "logic_line_aggregation",
                    "logic_line_pattern": {
                        "source_logic": "source",
                        "transfer_rule": "rule",
                        "why_it_generalizes": "generalizes",
                        "failure_boundary": "boundary",
                    },
                }
                with connect(cwd / "data" / "research_alpha.db") as conn:
                    conn.execute(
                        "insert into pattern_cards(pattern_key, content_json) values (?, ?)",
                        ("legacy-transfer", json.dumps(legacy_pattern)),
                    )

                result = backfill_pattern_source_provenance(cwd)
                self.assertEqual(result["updated"], 1)
                payload, _, _ = build_evidence_report(cwd, paper_limit=8)
                self.assertEqual(payload["counts"]["strict_pattern_cards"], 0)
                pattern = next(item for item in payload["pattern_cards"] if item["pattern_key"] == "legacy-transfer")
                self.assertTrue(pattern["gold_only_source_provenance"])
                self.assertFalse(pattern["grounding_audit_valid"])
                with connect(cwd / "data" / "research_alpha.db") as conn:
                    stored = conn.execute(
                        "select content_json from pattern_cards where pattern_key='legacy-transfer'"
                    ).fetchone()
                stored_payload = json.loads(stored["content_json"])
                self.assertEqual(stored_payload["source_set_policy"], "gold_set_only")
                self.assertTrue(all(float(item["paper_weight"]) > 0 for item in stored_payload["source_papers"]))
                self.assertNotIn("Frontier Agent Survey", stored_payload["source_titles"])
            finally:
                os.chdir(old_cwd)

    def test_strict_evidence_audit_rejects_frontier_genome_card(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                from research_alpha.db import connect, upsert_idea_card

                os.chdir(cwd)
                main(["init"])
                for title, award in [
                    ("Historical Best Paper", "best_paper"),
                    ("Historical Oral Paper", "oral"),
                    ("Historical Spotlight Paper", "spotlight"),
                ]:
                    main(
                        [
                            "add-paper",
                            "--title",
                            title,
                            "--venue",
                            "ICLR",
                            "--year",
                            "2024",
                            "--abstract",
                            "Historical logic reveals benchmark assumptions, hidden agent failures, and brittle evaluation.",
                        ]
                    )
                    with connect(cwd / "data" / "research_alpha.db") as conn:
                        conn.execute("update papers set award=? where title=?", (award, title))
                main(
                    [
                        "add-paper",
                        "--title",
                        "Frontier Agent Survey",
                        "--venue",
                        "Science China Information Sciences",
                        "--year",
                        "2025",
                        "--abstract",
                        "A current frontier survey about agents.",
                    ]
                )
                with connect(cwd / "data" / "research_alpha.db") as conn:
                    conn.execute(
                        "update papers set source_kind='openalex', citation_count=388, award='' where title='Frontier Agent Survey'"
                    )
                main(["score"])
                fake_genome = grounded_fake_genome_json()
                upsert_idea_card(cwd / "data" / "research_alpha.db", 4, "abstract_only", fake_genome)
                payload, _, _ = build_strict_evidence_audit(cwd, check_latest_run=False)
                self.assertFalse(payload["ok"])
                self.assertIn("non_gold_genome_card", {item["code"] for item in payload["failures"]})
            finally:
                os.chdir(old_cwd)

    def test_strict_evidence_audit_rejects_legacy_non_core_gold_paper_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                db_path = cwd / "data" / "research_alpha.db"
                paper_id = add_paper(
                    db_path,
                    "Legacy ACL Gold Pollution",
                    "ACL",
                    2025,
                    "A legacy record imported before non-core awards were downgraded.",
                    source_kind="gold_acl_awards",
                    award="best_paper",
                    citation_count=500,
                )
                with connect(db_path) as conn:
                    conn.execute("UPDATE papers SET paper_weight=4, score_notes='legacy' WHERE id=?", (paper_id,))

                payload, _, _ = build_strict_evidence_audit(cwd, check_latest_run=False)
                self.assertFalse(payload["ok"])
                failure_codes = {item["code"] for item in payload["failures"]}
                self.assertIn("legacy_non_core_gold_paper", failure_codes)
                self.assertTrue(
                    any(
                        item["name"] == "gold_paper_records_are_core_positive_gold_only"
                        and not item["passed"]
                        for item in payload["checks"]
                    )
                )
            finally:
                os.chdir(old_cwd)

    def test_strict_evidence_audit_rejects_latest_run_scope_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                from research_alpha.db import add_candidate_idea, add_run, connect

                os.chdir(cwd)
                main(["init"])
                for title, award in [
                    ("Historical Best Paper", "best_paper"),
                    ("Historical Oral Paper", "oral"),
                    ("Historical Spotlight Paper", "spotlight"),
                ]:
                    main(
                        [
                            "add-paper",
                            "--title",
                            title,
                            "--venue",
                            "ICLR",
                            "--year",
                            "2024",
                            "--abstract",
                            "Historical logic reveals benchmark assumptions, hidden agent failures, and brittle evaluation.",
                        ]
                    )
                    with connect(cwd / "data" / "research_alpha.db") as conn:
                        conn.execute("update papers set award=? where title=?", (award, title))
                main(["score"])
                fake_genome = grounded_fake_genome_json()
                fake_pattern = grounded_fake_pattern_json(["Historical Best Paper", "Historical Oral Paper"])
                with patch("research_alpha.llm.LLMClient.chat") as mock_chat:
                    mock_chat.side_effect = lambda prompt, system_prompt=None: (
                        type("Resp", (), {"text": fake_pattern})()
                        if "Aggregate the following genome cards" in prompt
                        else grounded_fake_chat_response(prompt)
                    )
                    os.environ["OPENAI_API_KEY"] = "test-openai-key"
                    try:
                        from research_alpha.app import run_pipeline_genome_build, run_pipeline_pattern_build

                        run_pipeline_genome_build(
                            cwd,
                            limit=3,
                            provider=None,
                            model="",
                            include_existing=False,
                        )
                        run_pipeline_pattern_build(cwd, limit=3, provider=None, model="")
                    finally:
                        os.environ.pop("OPENAI_API_KEY", None)
                payload = {
                    "idea_title": "Scoped Idea",
                    "core_hypothesis": "Agent progress is overestimated by narrow metrics.",
                    "historical_pattern": "hidden-assumption-reversal",
                    "trend_support": "Recent agentic planning and tool-use work suggests demand for stronger evaluation.",
                    "frontier_gap": "Current agent benchmarks under-measure failure modes.",
                    "why_now": "Agent systems are now strong enough for deeper evaluation.",
                    "novelty": "Reframes benchmark design around hidden brittleness.",
                    "value": "Could reset how research agents are evaluated.",
                    "key_risk": "Benchmark churn.",
                    "first_experiments": ["Build failure-taxonomy tasks"],
                    "evaluation_outline": "Measure ranking changes.",
                    "paper_angle": "Benchmarks reward the wrong competence.",
                    "evidence_basis": strict_evidence_basis(),
                    "generation_guardrail": strict_generation_guardrail(),
                    "storyline_trace": strict_storyline_trace(),
                }
                payload["evidence_grounded_score"] = build_evidence_grounded_score(
                    payload,
                    pattern_rows=[],
                    paper_rows=[],
                    genome_rows=[],
                )
                payload["generation_evidence_audit"] = {
                    "status": "explicit_valid",
                    "matched_basis_count": 1,
                    "missing_uses": [],
                }
                old_id = add_candidate_idea(
                    cwd / "data" / "research_alpha.db",
                    "research agents",
                    "Old Idea",
                    json.dumps(payload),
                )
                new_id = add_candidate_idea(
                    cwd / "data" / "research_alpha.db",
                    "research agents",
                    "New Idea",
                    json.dumps(payload),
                )
                run_dir = cwd / "outputs" / "fake-run"
                run_dir.mkdir(parents=True)
                (run_dir / f"idea-{new_id}.json").write_text("{}", encoding="utf-8")
                manifest_path = run_dir / "manifest.json"
                manifest_path.write_text(
                    json.dumps(
                        {
                            "status": "session_ready",
                            "stages": [
                                {
                                    "stage": "ideas",
                                    "status": "completed",
                                    "outputs": [str(run_dir / f"idea-{new_id}.json")],
                                },
                                {"stage": "ranking", "status": "completed", "top_pick_id": old_id},
                            ],
                            "ranking": {"top_pick_id": old_id},
                            "auto_session": {"candidate_idea_id": old_id},
                        }
                    ),
                    encoding="utf-8",
                )
                add_run(cwd / "data" / "research_alpha.db", "research agents", "session_ready", manifest_path)
                audit_payload, _, _ = build_strict_evidence_audit(cwd)
                self.assertFalse(audit_payload["ok"])
                failure_codes = {item["code"] for item in audit_payload["failures"]}
                self.assertIn("latest_run_scope_mismatch", failure_codes)
                self.assertTrue(any(item["name"] == "latest_run_ranking_and_session_scoped_to_current_ideas" and not item["passed"] for item in audit_payload["checks"]))
            finally:
                os.chdir(old_cwd)

    def test_strict_evidence_audit_allows_latest_run_without_ranking_or_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                from research_alpha.db import add_run

                os.chdir(cwd)
                main(["init"])
                seed_strict_quality_papers(cwd)
                fake_genome = grounded_fake_genome_json()
                fake_pattern = grounded_fake_pattern_json(["Genome A", "Genome C"])
                with patch("research_alpha.llm.LLMClient.chat") as mock_chat:
                    mock_chat.side_effect = lambda prompt, system_prompt=None: (
                        type("Resp", (), {"text": fake_pattern})()
                        if "Aggregate the following genome cards" in prompt
                        else grounded_fake_chat_response(prompt)
                    )
                    os.environ["OPENAI_API_KEY"] = "test-openai-key"
                    try:
                        from research_alpha.app import run_pipeline_genome_build, run_pipeline_pattern_build

                        run_pipeline_genome_build(cwd, limit=2, provider=None, model="", include_existing=False)
                        run_pipeline_pattern_build(cwd, limit=2, provider=None, model="")
                    finally:
                        os.environ.pop("OPENAI_API_KEY", None)

                run_dir = cwd / "outputs" / "partial-run"
                run_dir.mkdir(parents=True)
                manifest_path = run_dir / "manifest.json"
                manifest_path.write_text(
                    json.dumps(
                        {
                            "status": "partial",
                            "stages": [
                                {"stage": "score", "status": "completed"},
                                {"stage": "trends", "status": "completed"},
                                {
                                    "stage": "ideas",
                                    "status": "skipped",
                                    "reason": "idea_generation_not_ready",
                                },
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                add_run(cwd / "data" / "research_alpha.db", "research agents", "partial", manifest_path)
                audit_payload, _, _ = build_strict_evidence_audit(cwd)
                self.assertTrue(audit_payload["ok"])
                self.assertNotIn("latest_run_scope_mismatch", {item["code"] for item in audit_payload["failures"]})
                self.assertTrue(
                    any(
                        item["name"] == "latest_run_ranking_and_session_scoped_to_current_ideas"
                        and item["passed"]
                        and "latest_run_has_no_ranking_or_session" in item["detail"]
                        for item in audit_payload["checks"]
                    )
                )
            finally:
                os.chdir(old_cwd)

    def test_genome_grounding_audit_allows_bounded_abstraction_with_source_anchors(self) -> None:
        payload = json.loads(grounded_fake_genome_json())
        paper = {
            "id": 7,
            "title": "Genome Paper",
            "venue": "ICLR",
            "year": 2024,
            "award": "best_paper",
            "abstract": "Benchmark assumptions hide agent failures and brittle evaluation.",
        }
        audit = audit_genome_grounding(payload, paper)
        self.assertEqual(audit["status"], "valid")

    def test_genome_grounding_audit_rejects_unanchored_placeholder_story(self) -> None:
        payload = {
            "logic_line": {
                "old_belief": "belief",
                "bottleneck": "bottleneck",
                "reframing": "reframing",
                "why_now": "why now",
                "evidence_design": "evidence",
                "failure_boundary": "boundary",
            },
            "confidence_note": "abstract-only confidence",
            "evidence_level": "abstract_only",
        }
        paper = {
            "id": 7,
            "title": "Genome Paper",
            "venue": "ICLR",
            "year": 2024,
            "award": "best_paper",
            "abstract": "Benchmark assumptions hide agent failures and brittle evaluation.",
        }
        audit = audit_genome_grounding(payload, paper)
        self.assertEqual(audit["status"], "invalid")
        failure_codes = {item["code"] for item in audit["failures"]}
        self.assertIn("missing_logic_line_evidence", failure_codes)
        self.assertIn("placeholder_logic_step", failure_codes)

    def test_pattern_grounding_audit_rejects_examples_outside_source_cards(self) -> None:
        card = json.loads(grounded_fake_genome_json())
        card["title"] = "Genome A"
        card["grounding_audit"] = {"status": "valid"}
        other = json.loads(grounded_fake_genome_json())
        other["title"] = "Genome B"
        other["grounding_audit"] = {"status": "valid"}
        payload = json.loads(grounded_fake_pattern_json(["Unseen Paper"]))
        audit = audit_pattern_grounding(payload, [card, other])
        self.assertEqual(audit["status"], "invalid")
        self.assertIn("canonical_example_not_in_source_cards", {item["code"] for item in audit["failures"]})

    def test_remote_openalex_harvest_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                main(["init"])
                payload = [
                    {
                        "title": "Remote Paper",
                        "abstract": "A remote abstract.",
                        "venue": "ICLR",
                        "year": 2024,
                        "external_ref": "https://openalex.org/W1",
                        "citation_count": 123,
                        "influential_citation_count": 0,
                        "award": "",
                    }
                ]
                with patch("research_alpha.app.load_remote_records", return_value=payload):
                    output = io.StringIO()
                    with redirect_stdout(output):
                        exit_code = main(
                            ["h", "--remote", "openalex", "--venue", "ICLR", "--year", "2024", "--limit", "5"]
                        )
                self.assertEqual(exit_code, 0)
                self.assertIn("Harvested 1 records", output.getvalue())
            finally:
                os.chdir(old_cwd)

    def test_ideate_remote_harvest_falls_back_after_rate_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                main(["init"])
                calls = []

                def fake_pipeline(*args, **kwargs):
                    calls.append(kwargs.get("remote"))
                    if kwargs.get("remote") == "s2":
                        raise ConnectorError("Remote request failed with HTTP 429: Too Many Requests")
                    return {"stage": "harvest", "status": "completed", "source": "frontier_openalex", "imported": 1}

                with patch("research_alpha.app.pipeline_harvest_into_db", side_effect=fake_pipeline):
                    output = io.StringIO()
                    err = io.StringIO()
                    with redirect_stdout(output), redirect_stderr(err):
                        exit_code = main(["ideate", "科研智能体可靠评测", "--remote", "s2", "--limit", "1", "--ideas", "1"])
                self.assertEqual(exit_code, 1)
                self.assertEqual(calls[:2], ["s2", "openalex"])
                self.assertIn("fallback note: s2：远程源限流或预算不足", output.getvalue())
                self.assertNotIn("Too Many Requests", output.getvalue())
                self.assertNotIn("Traceback", err.getvalue())
            finally:
                os.chdir(old_cwd)

    def test_reviewer_command_returns_rethink_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                seed_strict_quality_papers(cwd)
                fake_review = json.dumps(
                    {
                        "decision": "rethink_required",
                        "score_expert": {"innovation": 4, "logic": 3, "feasibility": 6, "value": 5, "defensibility": 2, "overall": 4},
                        "pattern_logic_expert": {
                            "matched_storyline": "Weak match.",
                            "missing_logic_links": ["No Gold-style story arc."],
                            "gold_standard_alignment": "Low.",
                        },
                        "top_conference_reviewer": {
                            "main_attacks": ["Too incremental."],
                            "fatal_flaws": ["No clear evidence design."],
                            "questions": ["Why now?"],
                        },
                        "rethink_trigger": "No defensible top-conference contribution.",
                        "required_rethink": ["Rebuild around a sharper bottleneck."],
                        "summary": "Reviewer blocks this idea.",
                    }
                )
                with patch("research_alpha.llm.LLMClient.chat", return_value=type("Resp", (), {"text": fake_review})()):
                    output = io.StringIO()
                    with redirect_stdout(output):
                        exit_code = main(["rv", "--idea", "A small benchmark idea.", "--provider", "ds"])
                self.assertEqual(exit_code, 2)
                self.assertIn("Reviewer decision: rethink_required", output.getvalue())
                self.assertTrue(list((cwd / "outputs" / "reviews").glob("review-*.json")))
            finally:
                os.chdir(old_cwd)

    def test_reviewer_command_failure_prints_structured_summary_without_saving_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                seed_strict_quality_papers(cwd)
                with patch("research_alpha.llm.LLMClient.chat", return_value=type("Resp", (), {"text": '{"decision":"revise"}'})()):
                    output = io.StringIO()
                    err = io.StringIO()
                    with redirect_stdout(output), redirect_stderr(err):
                        exit_code = main(["rv", "--idea", "A small benchmark idea.", "--provider", "ds"])
                self.assertEqual(exit_code, 1)
                self.assertIn("Reviewer failure JSON:", output.getvalue())
                failure_line = next(line for line in output.getvalue().splitlines() if line.startswith("Reviewer failure JSON: "))
                failure = json.loads(failure_line.split(": ", 1)[1])
                self.assertEqual(failure["status"], "reviewer_failed")
                self.assertEqual(failure["reason_code"], "invalid_model_response")
                self.assertIn("no_review_saved", failure["state_policy"])
                self.assertIn("Reviewer response missing keys", err.getvalue())
                self.assertFalse((cwd / "outputs" / "reviews").exists())
            finally:
                os.chdir(old_cwd)

    def test_reviewer_command_repairs_invalid_json_once_before_saving_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                seed_strict_quality_papers(cwd)
                repaired_review = json.dumps(
                    {
                        "decision": "revise",
                        "score_expert": {"innovation": 6, "logic": 5, "feasibility": 5, "value": 6, "defensibility": 4, "overall": 5},
                        "pattern_logic_expert": {
                            "matched_storyline": "Partially matched.",
                            "missing_logic_links": ["Needs a clearer failure boundary."],
                            "gold_standard_alignment": "Medium.",
                        },
                        "top_conference_reviewer": {
                            "main_attacks": ["Evidence design is thin."],
                            "fatal_flaws": [],
                            "questions": ["What changes in evaluation?"],
                        },
                        "rethink_trigger": "",
                        "required_rethink": ["Tighten the evidence design."],
                        "summary": "Repair produced a valid reviewer result.",
                    }
                )
                prompts: list[str] = []

                def fake_chat(self, prompt, system_prompt=""):
                    prompts.append(prompt)
                    return type("Resp", (), {"text": '{"decision":"revise"}' if len(prompts) == 1 else repaired_review})()

                with patch("research_alpha.llm.LLMClient.chat", autospec=True, side_effect=fake_chat):
                    output = io.StringIO()
                    with redirect_stdout(output):
                        exit_code = main(["rv", "--idea", "A small benchmark idea.", "--provider", "ds"])
                self.assertEqual(exit_code, 0)
                self.assertEqual(len(prompts), 2)
                self.assertIn("Repair the previous Reviewer response", prompts[1])
                self.assertIn("Reviewer decision: revise", output.getvalue())
                self.assertTrue(list((cwd / "outputs" / "reviews").glob("review-*.json")))
            finally:
                os.chdir(old_cwd)

    def test_remote_query_filter_requires_title_match_for_single_core_token(self) -> None:
        from research_alpha.app import filter_records_by_query_tokens

        records = [
            {
                "title": "Convex Optimization Theory",
                "abstract": "Classical optimization problem of agents that converge.",
            },
            {
                "title": "The Rise and Potential of Large Language Model Based Agents",
                "abstract": "",
            },
        ]

        filtered = filter_records_by_query_tokens(records, "research agents")
        self.assertEqual([item["title"] for item in filtered], [records[1]["title"]])

    def test_genome_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                main(["init"])
                main(["add-paper", "--title", "Genome Paper", "--venue", "ICLR", "--year", "2024", "--abstract", "Benchmark assumptions hide agent failures and brittle evaluation."])
                fake_response = grounded_fake_genome_json()
                with patch("research_alpha.llm.LLMClient.chat") as mock_chat:
                    mock_chat.return_value = type(
                        "Resp",
                        (),
                        {"text": fake_response},
                    )()
                    output = io.StringIO()
                    with redirect_stdout(output):
                        exit_code = main(["genome", "--paper-id", "1"])
                self.assertEqual(exit_code, 0)
                self.assertIn("Generated genome card", output.getvalue())
                self.assertTrue((cwd / "outputs" / "genomes" / "paper-1.json").exists())
            finally:
                os.chdir(old_cwd)

    def test_batch_genome_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                main(["init"])
                seed_strict_quality_papers(cwd)
                fake_response = grounded_fake_genome_json()
                with patch("research_alpha.llm.LLMClient.chat") as mock_chat:
                    mock_chat.return_value = type("Resp", (), {"text": fake_response})()
                    output = io.StringIO()
                    with redirect_stdout(output):
                        exit_code = main(["gb", "--limit", "2"])
                self.assertEqual(exit_code, 0)
                self.assertIn("Generated 2 genome cards.", output.getvalue())
            finally:
                os.chdir(old_cwd)

    def test_pattern_build_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                main(["init"])
                seed_strict_quality_papers(cwd)

                fake_genome = grounded_fake_genome_json()
                fake_pattern = grounded_fake_pattern_json(["Genome A", "Genome C"])
                with patch("research_alpha.llm.LLMClient.chat") as mock_chat:
                    mock_chat.side_effect = lambda prompt, system_prompt=None: (
                        type("Resp", (), {"text": fake_pattern})()
                        if "Aggregate the following genome cards" in prompt
                        else grounded_fake_chat_response(prompt)
                    )
                    main(["gb", "--limit", "2"])
                    output = io.StringIO()
                    with redirect_stdout(output):
                        exit_code = main(["pb", "--limit", "2"])
                self.assertEqual(exit_code, 0)
                self.assertIn("Generated pattern card", output.getvalue())
                self.assertTrue((cwd / "outputs" / "patterns" / "hidden-assumption-reversal.json").exists())
            finally:
                os.chdir(old_cwd)

    def test_critique_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                main(["init"])
                seed_strict_quality_papers(cwd)

                fake_genome = grounded_fake_genome_json()
                fake_pattern = grounded_fake_pattern_json()
                fake_critique = json.dumps(
                    {
                        "decision": "reject",
                        "verdict_summary": "The user's change weakens the idea.",
                        "why_user_may_be_wrong": "It removes the strongest novelty axis.",
                        "supporting_patterns": ["hidden-assumption-reversal"],
                        "supporting_genomes": ["Genome A", "Genome B"],
                        "trend_alignment": "The requested simplification moves away from the current evaluation-focused frontier.",
                        "preserve": "Keep the problem reframing.",
                        "revise": "Tighten evaluation rather than narrowing ambition.",
                        "revised_idea": "Keep the original reframing and sharpen the benchmark story.",
                        "coach_message": "Push back on the weaker direction.",
                        "feasibility_assessment": {
                            "status": "fragile",
                            "summary": "The simplified direction would be easier to run but loses the core evidence path.",
                            "next_test": "Stress the benchmark story before shrinking the scope.",
                        },
                        "why_not_analysis": {
                            "status": "serious",
                            "summary": "It removes the strongest novelty axis.",
                            "mitigation": "Keep the reframing and narrow the evaluation protocol instead.",
                        },
                        "value_judgment": {
                            "status": "low",
                            "summary": "The simplified version would be easier to execute but much weaker as a paper.",
                            "proof_gap": "Show that the benchmark still reveals something current work misses.",
                        },
                    }
                )
                with patch("research_alpha.llm.LLMClient.chat") as mock_chat:
                    mock_chat.side_effect = [
                        type("Resp", (), {"text": fake_genome})(),
                        type("Resp", (), {"text": fake_genome})(),
                        type("Resp", (), {"text": fake_pattern})(),
                        type("Resp", (), {"text": fake_critique})(),
                    ]
                    main(["gb", "--limit", "2"])
                    main(["pb", "--limit", "2"])
                    output = io.StringIO()
                    with redirect_stdout(output):
                        exit_code = main(
                            [
                                "c",
                                "--idea",
                                "Use a weaker benchmark to make the paper easier.",
                                "--user",
                                "Let's simplify the idea a lot and aim for a small incremental result.",
                            ]
                        )
                self.assertEqual(exit_code, 0)
                self.assertIn("Decision: reject", output.getvalue())
                self.assertIn("Trend check:", output.getvalue())
                self.assertIn("Revised idea:", output.getvalue())
                critique_dir = cwd / "outputs" / "critiques"
                self.assertTrue(any(critique_dir.glob("*.json")))
            finally:
                os.chdir(old_cwd)

    def test_session_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                main(["init"])
                seed_strict_quality_papers(cwd)

                fake_genome = grounded_fake_genome_json()
                fake_pattern = grounded_fake_pattern_json()
                fake_critique = json.dumps(
                    {
                        "decision": "partial",
                        "verdict_summary": "Part of the user change helps, part weakens the story.",
                        "why_user_may_be_wrong": "It narrows the contribution too much.",
                        "supporting_patterns": ["hidden-assumption-reversal"],
                        "supporting_genomes": ["Genome A", "Genome B"],
                        "trend_alignment": "The idea should stay aligned with the recent push toward stronger agent evaluation.",
                        "preserve": "Keep the core reframing.",
                        "revise": "Adjust the evaluation design only.",
                        "revised_idea": "Keep the bold reframing but simplify the evaluation protocol.",
                        "coach_message": "Do not over-shrink the idea.",
                        "feasibility_assessment": {
                            "status": "promising",
                            "summary": "The lighter protocol is feasible if the core failure analysis stays intact.",
                            "next_test": "Prototype the smallest evaluation slice that still exposes hidden brittleness.",
                        },
                        "why_not_analysis": {
                            "status": "moderate",
                            "summary": "It narrows the contribution too much.",
                            "mitigation": "Shrink the protocol, not the central claim.",
                        },
                        "value_judgment": {
                            "status": "medium",
                            "summary": "The idea keeps value if it still changes how agent evaluation is interpreted.",
                            "proof_gap": "Show the simplified protocol still changes benchmark conclusions.",
                        },
                    }
                )

                with patch("research_alpha.llm.LLMClient.chat") as mock_chat:
                    mock_chat.side_effect = [
                        type("Resp", (), {"text": fake_genome})(),
                        type("Resp", (), {"text": fake_genome})(),
                        type("Resp", (), {"text": fake_pattern})(),
                        type("Resp", (), {"text": fake_critique})(),
                    ]
                    main(["gb", "--limit", "2"])
                    main(["pb", "--limit", "2"])
                    create_out = io.StringIO()
                    with redirect_stdout(create_out):
                        create_exit = main(["is", "--name", "session-a", "--idea", "Build a bold agent benchmark."])
                    self.assertEqual(create_exit, 0)
                    self.assertIn("Created idea session #1", create_out.getvalue())
                    self.assertTrue((cwd / "outputs" / "sessions" / "session-1-dossier.json").exists())

                    step_out = io.StringIO()
                    with redirect_stdout(step_out):
                        step_exit = main(
                            [
                                "st",
                                "--user",
                                "Let us make the idea much smaller and more incremental.",
                            ]
                        )
                    self.assertEqual(step_exit, 0)
                    self.assertIn("Turn 1 decision: partial", step_out.getvalue())
                    self.assertIn("Trend check:", step_out.getvalue())
                    self.assertIn("Current best:", step_out.getvalue())

                    view_out = io.StringIO()
                    with redirect_stdout(view_out):
                        view_exit = main(["sv"])
                    self.assertEqual(view_exit, 0)
                    self.assertIn("Current idea:", view_out.getvalue())
                    self.assertIn("turn 1: partial", view_out.getvalue())

                    dossier_out = io.StringIO()
                    with redirect_stdout(dossier_out):
                        dossier_exit = main(["sd"])
                    self.assertEqual(dossier_exit, 0)
                    dossier_text = dossier_out.getvalue()
                    self.assertIn("Current best idea:", dossier_text)
                    self.assertIn("Novelty claim:", dossier_text)
                    self.assertIn("Evaluation sketch:", dossier_text)
                    self.assertIn("Story hook:", dossier_text)
                    self.assertIn("Main risk:", dossier_text)
                    self.assertIn("Trend thesis:", dossier_text)
                    self.assertIn("Prior art:", dossier_text)
                    self.assertIn("Feasibility:", dossier_text)
                    self.assertIn("Why not:", dossier_text)
                    self.assertIn("Value:", dossier_text)
                    self.assertIn("Target venue:", dossier_text)
                    self.assertIn("Trajectory:", dossier_text)
                    self.assertIn("Latest decision: partial", dossier_text)

                    dossier_payload = json.loads(
                        (cwd / "outputs" / "sessions" / "session-1-dossier.json").read_text(encoding="utf-8")
                    )
                    self.assertEqual(dossier_payload["initial_idea"], "Build a bold agent benchmark.")
                    self.assertEqual(
                        dossier_payload["current_best_idea"],
                        "Keep the bold reframing but simplify the evaluation protocol.",
                    )
                    self.assertEqual(dossier_payload["latest_decision"], "partial")
                    self.assertEqual(dossier_payload["turn_count"], 1)
                    self.assertIn("Keep the bold reframing", dossier_payload["novelty_claim"])
                    self.assertIn("evaluation protocol", dossier_payload["evaluation_sketch"])
                    self.assertIn("Do not over-shrink the idea.", dossier_payload["paper_story_hook"])
                    self.assertIn("narrows the contribution too much", dossier_payload["main_risk"])
                    self.assertIn("recent push toward stronger agent evaluation", dossier_payload["trend_thesis"])
                    self.assertEqual(dossier_payload["prior_art_check"], "No stored prior-art screen yet.")
                    self.assertIn("The lighter protocol is feasible", dossier_payload["feasibility"])
                    self.assertIn("narrows the contribution too much", dossier_payload["why_not_analysis"])
                    self.assertIn("The idea keeps value if it still changes how agent evaluation is interpreted.", dossier_payload["value_judgment"])
                    self.assertEqual(dossier_payload["critique_outputs"]["feasibility_assessment"]["status"], "promising")
                    self.assertEqual(dossier_payload["critique_outputs"]["why_not_analysis"]["status"], "moderate")
                    self.assertEqual(dossier_payload["critique_outputs"]["value_judgment"]["status"], "medium")
                    self.assertEqual(dossier_payload["target_venue"], "ICLR")
                    self.assertIn("Adjust the evaluation design only.", dossier_payload["experiment_plan"]["week_2"])
                    self.assertTrue((cwd / "outputs" / "sessions" / "session-1-dossier.md").exists())
                    markdown_text = (cwd / "outputs" / "sessions" / "session-1-dossier.md").read_text(encoding="utf-8")
                    self.assertIn("## Prior-Art Check", markdown_text)
                    self.assertIn("## Experiment Plan", markdown_text)
                    self.assertEqual(dossier_payload["decision_tally"]["partial"], 1)
                    self.assertEqual(dossier_payload["critique_memory"]["recent_verdicts"][0], "Part of the user change helps, part weakens the story.")
            finally:
                os.chdir(old_cwd)

    def test_session_step_accepts_positional_user(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                main(["init"])
                seed_strict_quality_papers(cwd)

                fake_genome = grounded_fake_genome_json()
                fake_pattern = grounded_fake_pattern_json()
                fake_critique = json.dumps(
                    {
                        "decision": "partial",
                        "verdict_summary": "Part of the user change helps, part weakens the story.",
                        "why_user_may_be_wrong": "It narrows the contribution too much.",
                        "supporting_patterns": ["hidden-assumption-reversal"],
                        "supporting_genomes": ["Genome A", "Genome B"],
                        "trend_alignment": "The idea should stay aligned with the recent push toward stronger agent evaluation.",
                        "preserve": "Keep the core reframing.",
                        "revise": "Adjust the evaluation design only.",
                        "revised_idea": "Keep the bold reframing but simplify the evaluation protocol.",
                        "coach_message": "Do not over-shrink the idea.",
                        "feasibility_assessment": {
                            "status": "promising",
                            "summary": "The lighter protocol is feasible if the core failure analysis stays intact.",
                            "next_test": "Prototype the smallest evaluation slice that still exposes hidden brittleness.",
                        },
                        "why_not_analysis": {
                            "status": "moderate",
                            "summary": "It narrows the contribution too much.",
                            "mitigation": "Shrink the protocol, not the central claim.",
                        },
                        "value_judgment": {
                            "status": "medium",
                            "summary": "The idea keeps value if it still changes how agent evaluation is interpreted.",
                            "proof_gap": "Show the simplified protocol still changes benchmark conclusions.",
                        },
                    }
                )

                with patch("research_alpha.llm.LLMClient.chat") as mock_chat:
                    mock_chat.side_effect = [
                        type("Resp", (), {"text": fake_genome})(),
                        type("Resp", (), {"text": fake_genome})(),
                        type("Resp", (), {"text": fake_pattern})(),
                        type("Resp", (), {"text": fake_critique})(),
                    ]
                    main(["gb", "--limit", "2"])
                    main(["pb", "--limit", "2"])
                    main(["is", "--name", "session-a", "--idea", "构建一个大胆的科研智能体评测基准。"])
                    step_out = io.StringIO()
                    with redirect_stdout(step_out):
                        step_exit = main(["st", "我们把它缩小一点，但不要变成普通增量工作。"])
                self.assertEqual(step_exit, 0)
                self.assertIn("Turn 1 decision: partial", step_out.getvalue())
                self.assertIn("Current best:", step_out.getvalue())
                memory_line = next(
                    line for line in step_out.getvalue().splitlines() if line.startswith("Session memory JSON: ")
                )
                memory_status = json.loads(memory_line.split(": ", 1)[1])
                self.assertEqual(memory_status["preferred_language"], "zh")
                self.assertEqual(memory_status["turn_count"], 1)
                self.assertIn("compressed into memory_summary", memory_status["cache_policy"])
                with connect(cwd / "data" / "research_alpha.db") as conn:
                    row = conn.execute("SELECT memory_summary_json FROM idea_sessions WHERE id = 1").fetchone()
                memory = json.loads(row["memory_summary_json"])
                self.assertEqual(memory["preferred_language"], "zh")
                self.assertEqual(memory["turn_count"], 1)
                self.assertIn("Gold Set", memory["scoring_standard_policy"])
                dossier_text = (cwd / "outputs" / "sessions" / "session-1-dossier.md").read_text(encoding="utf-8")
                self.assertIn("Stable thesis", dossier_text)
                self.assertIn("Scoring policy", dossier_text)
            finally:
                os.chdir(old_cwd)

    def test_session_memory_status_tolerates_corrupt_turn_count(self) -> None:
        status = build_session_memory_status(
            {
                "preferred_language": "zh",
                "turn_count": "not-a-number",
                "stable_thesis": "Keep the core idea stable at /Users/me/project with OPENAI_API_KEY=secret-value.",
                "recent_limitations_focus": "bad-shape",
                "unresolved_risks": ["Risk stays visible with Bearer abcdefghijklmnop."],
                "next_experiments": ["Run a small probe using sk-test-secret-123456."],
            }
        )
        self.assertEqual(status["preferred_language"], "zh")
        self.assertEqual(status["turn_count"], 0)
        self.assertIn("Keep the core idea", status["stable_thesis"])
        self.assertIn("[local path]", status["stable_thesis"])
        self.assertIn("[redacted]", status["stable_thesis"])
        self.assertIn("Risk stays visible with Bearer [redacted]", status["unresolved_risk"])
        self.assertEqual(status["next_experiment"], "Run a small probe using sk-[redacted].")
        serialized_status = json.dumps(status, ensure_ascii=False)
        self.assertNotIn("/Users/me", serialized_status)
        self.assertNotIn("secret-value", serialized_status)
        self.assertNotIn("abcdefghijklmnop", serialized_status)
        self.assertNotIn("sk-test-secret", serialized_status)
        self.assertEqual(status["memory_version"], 1)
        self.assertIn("memory_summary_is_compressed_public_state", status["state_policy"])
        self.assertIn("Gold Set", status["scoring_standard_policy"])
        self.assertIn("domain knowledge only", status["user_library_policy"])

    def test_gui_session_memory_status_exposes_stable_policy_not_raw_chain(self) -> None:
        html = render_index_html()
        self.assertIn("memory.state_policy", html)
        self.assertIn("memory.user_library_policy", html)
        self.assertIn("记忆策略：压缩摘要保持多轮连续性，原始中间过程不进入回答。", html)
        self.assertIn("用户库边界：只作领域背景，不作为评分标准或故事线来源。", html)

    def test_session_step_failure_does_not_write_turn_or_change_current_idea(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                db_path = cwd / "data" / "research_alpha.db"
                seed_strict_generation_evidence(db_path)
                from research_alpha.db import create_idea_session

                session_id = create_idea_session(db_path, "session-a", "Original stable idea")

                def fake_chat(self, prompt, system_prompt=""):
                    return type("Resp", (), {"text": '{"decision":"partial"}'})()

                out = io.StringIO()
                err = io.StringIO()
                with patch("research_alpha.llm.LLMClient.chat", autospec=True, side_effect=fake_chat):
                    with redirect_stdout(out), redirect_stderr(err):
                        exit_code = main(["st", "--session-id", str(session_id), "请缩小一点"])
                self.assertEqual(exit_code, 1)
                self.assertIn("Session-step failure JSON:", out.getvalue())
                failure_line = next(line for line in out.getvalue().splitlines() if line.startswith("Session-step failure JSON: "))
                failure = json.loads(failure_line.split(": ", 1)[1])
                self.assertEqual(failure["status"], "session_step_failed")
                self.assertEqual(failure["reason_code"], "invalid_model_response")
                self.assertIn("no_turn_written", failure["state_policy"])
                with connect(db_path) as conn:
                    turn_count = conn.execute("SELECT COUNT(*) AS count FROM idea_session_turns WHERE session_id=?", (session_id,)).fetchone()
                    session = conn.execute("SELECT current_idea FROM idea_sessions WHERE id=?", (session_id,)).fetchone()
                self.assertEqual(int(turn_count["count"]), 0)
                self.assertEqual(session["current_idea"], "Original stable idea")
                self.assertIn("Critique response missing keys", err.getvalue())
            finally:
                os.chdir(old_cwd)

    def test_session_step_repairs_invalid_json_once_before_writing_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                db_path = cwd / "data" / "research_alpha.db"
                seed_strict_generation_evidence(db_path)
                from research_alpha.db import create_idea_session

                session_id = create_idea_session(
                    db_path,
                    "session-a",
                    "Original stable idea",
                    context_json=json.dumps({"evidence_basis": strict_evidence_basis()}),
                )
                repaired_critique = json.dumps(
                    {
                        "decision": "partial",
                        "verdict_summary": "The change is useful if evidence support is preserved.",
                        "why_user_may_be_wrong": "It could lose the Gold logic chain.",
                        "supporting_patterns": ["hidden-assumption-reversal"],
                        "supporting_genomes": [],
                        "trend_alignment": "Still aligned with agent evaluation.",
                        "preserve": "Keep hidden-assumption reversal.",
                        "revise": "Simplify only the experiment protocol.",
                        "revised_idea": "Original stable idea with a smaller evidence probe.",
                        "coach_message": "Keep the evidence chain visible.",
                        "feasibility_assessment": {"status": "promising", "summary": "Feasible if scoped.", "next_test": "Run a small probe."},
                        "why_not_analysis": {"status": "moderate", "summary": "Could become incremental.", "mitigation": "Keep the reframing."},
                        "value_judgment": {"status": "medium", "summary": "Still useful.", "proof_gap": "Show ranking changes."},
                    }
                )
                prompts: list[str] = []

                def fake_chat(self, prompt, system_prompt=""):
                    prompts.append(prompt)
                    return type("Resp", (), {"text": '{"decision":"partial"}' if len(prompts) == 1 else repaired_critique})()

                out = io.StringIO()
                with patch("research_alpha.llm.LLMClient.chat", autospec=True, side_effect=fake_chat):
                    with redirect_stdout(out):
                        exit_code = main(["st", "--session-id", str(session_id), "请缩小一点"])
                self.assertEqual(exit_code, 0)
                self.assertEqual(len(prompts), 2)
                self.assertIn("Repair the previous Critique response", prompts[1])
                self.assertIn("Turn 1 decision: partial", out.getvalue())
                with connect(db_path) as conn:
                    turn_count = conn.execute("SELECT COUNT(*) AS count FROM idea_session_turns WHERE session_id=?", (session_id,)).fetchone()
                    session = conn.execute("SELECT current_idea FROM idea_sessions WHERE id=?", (session_id,)).fetchone()
                self.assertEqual(int(turn_count["count"]), 1)
                self.assertEqual(session["current_idea"], "Original stable idea with a smaller evidence probe.")
            finally:
                os.chdir(old_cwd)

    def test_session_step_repairs_thin_placeholder_answer_before_writing_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                db_path = cwd / "data" / "research_alpha.db"
                seed_strict_generation_evidence(db_path)
                from research_alpha.db import create_idea_session

                session_id = create_idea_session(
                    db_path,
                    "session-a",
                    "Original stable idea",
                    context_json=json.dumps({"evidence_basis": strict_evidence_basis()}),
                )
                thin_critique = json.dumps(
                    {
                        "decision": "accept",
                        "verdict_summary": "ok",
                        "why_user_may_be_wrong": "ok",
                        "supporting_patterns": ["hidden-assumption-reversal"],
                        "supporting_genomes": [],
                        "trend_alignment": "ok",
                        "preserve": "ok",
                        "revise": "ok",
                        "revised_idea": "已完成",
                        "coach_message": "ok",
                        "feasibility_assessment": {"status": "unclear", "summary": "ok", "next_test": "ok"},
                        "why_not_analysis": {"status": "light", "summary": "ok", "mitigation": "ok"},
                        "value_judgment": {"status": "low", "summary": "ok", "proof_gap": "ok"},
                    }
                )
                repaired_critique = json.dumps(
                    {
                        "decision": "partial",
                        "verdict_summary": "The change is useful only if the evidence boundary remains explicit.",
                        "why_user_may_be_wrong": "A broad rewrite could hide the original Gold logic chain.",
                        "supporting_patterns": ["hidden-assumption-reversal"],
                        "supporting_genomes": [],
                        "trend_alignment": "Still aligned with agent evaluation when trend evidence stays contextual.",
                        "preserve": "Keep hidden-assumption reversal as the causal story standard.",
                        "revise": "Narrow the idea to a failure-assumption probe with explicit evidence hooks.",
                        "revised_idea": "Original stable idea reframed as a hidden-assumption stress probe for research-agent evaluation.",
                        "coach_message": "Keep the pattern support visible.",
                        "feasibility_assessment": {"status": "promising", "summary": "Feasible as a scoped benchmark probe.", "next_test": "Build two stress tasks and compare ranking shifts."},
                        "why_not_analysis": {"status": "moderate", "summary": "It may look incremental without a clear failure taxonomy.", "mitigation": "Tie each task to one hidden assumption."},
                        "value_judgment": {"status": "medium", "summary": "Useful if it changes model selection decisions.", "proof_gap": "Show that current benchmarks miss the failures."},
                    }
                )
                prompts: list[str] = []

                def fake_chat(self, prompt, system_prompt=""):
                    prompts.append(prompt)
                    return type("Resp", (), {"text": thin_critique if len(prompts) == 1 else repaired_critique})()

                out = io.StringIO()
                with patch("research_alpha.llm.LLMClient.chat", autospec=True, side_effect=fake_chat):
                    with redirect_stdout(out):
                        exit_code = main(["st", "--session-id", str(session_id), "给一个更清楚的版本"])
                self.assertEqual(exit_code, 0)
                self.assertEqual(len(prompts), 2)
                self.assertIn("content_quality_failed", prompts[1])
                self.assertIn("Repair the previous Critique response", prompts[1])
                with connect(db_path) as conn:
                    turn_count = conn.execute("SELECT COUNT(*) AS count FROM idea_session_turns WHERE session_id=?", (session_id,)).fetchone()
                    session = conn.execute("SELECT current_idea FROM idea_sessions WHERE id=?", (session_id,)).fetchone()
                self.assertEqual(int(turn_count["count"]), 1)
                self.assertIn("hidden-assumption stress probe", session["current_idea"])
            finally:
                os.chdir(old_cwd)

    def test_session_step_rejects_thin_placeholder_answer_after_repair_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                db_path = cwd / "data" / "research_alpha.db"
                seed_strict_generation_evidence(db_path)
                from research_alpha.db import create_idea_session

                session_id = create_idea_session(
                    db_path,
                    "session-a",
                    "Original stable idea",
                    context_json=json.dumps({"evidence_basis": strict_evidence_basis()}),
                )
                thin_critique = json.dumps(
                    {
                        "decision": "accept",
                        "verdict_summary": "ok",
                        "why_user_may_be_wrong": "ok",
                        "supporting_patterns": ["hidden-assumption-reversal"],
                        "supporting_genomes": [],
                        "trend_alignment": "ok",
                        "preserve": "ok",
                        "revise": "ok",
                        "revised_idea": "已完成",
                        "coach_message": "ok",
                        "feasibility_assessment": {"status": "unclear", "summary": "ok", "next_test": "ok"},
                        "why_not_analysis": {"status": "light", "summary": "ok", "mitigation": "ok"},
                        "value_judgment": {"status": "low", "summary": "ok", "proof_gap": "ok"},
                    }
                )

                out = io.StringIO()
                err = io.StringIO()
                with patch("research_alpha.llm.LLMClient.chat", side_effect=[
                    type("Resp", (), {"text": thin_critique})(),
                    type("Resp", (), {"text": thin_critique})(),
                ]):
                    with redirect_stdout(out), redirect_stderr(err):
                        exit_code = main(["st", "--session-id", str(session_id), "随便完成一下"])
                self.assertEqual(exit_code, 1)
                failure_line = next(line for line in out.getvalue().splitlines() if line.startswith("Session-step failure JSON: "))
                failure = json.loads(failure_line.split(": ", 1)[1])
                self.assertEqual(failure["reason_code"], "invalid_model_response")
                self.assertIn("no_turn_written", failure["state_policy"])
                with connect(db_path) as conn:
                    turn_count = conn.execute("SELECT COUNT(*) AS count FROM idea_session_turns WHERE session_id=?", (session_id,)).fetchone()
                    session = conn.execute("SELECT current_idea FROM idea_sessions WHERE id=?", (session_id,)).fetchone()
                self.assertEqual(int(turn_count["count"]), 0)
                self.assertEqual(session["current_idea"], "Original stable idea")
                self.assertIn("content_quality_failed", err.getvalue())
            finally:
                os.chdir(old_cwd)

    def test_session_step_prompt_refreshes_user_library_as_domain_knowledge_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                db_path = cwd / "data" / "research_alpha.db"
                seed_strict_generation_evidence(db_path)
                from research_alpha.db import create_idea_session

                session_id = create_idea_session(
                    db_path,
                    "session-a",
                    "Original stable idea",
                    context_json=json.dumps({"evidence_basis": strict_evidence_basis()}),
                )
                upsert_user_library_paper(
                    db_path,
                    title="User Route Paper Added Later",
                    venue="arXiv",
                    year=2026,
                    abstract="Late-added route terminology for clinical agent workflows.",
                    external_ref="https://arxiv.org/abs/2601.00002",
                )
                with connect(db_path) as conn:
                    conn.execute(
                        "UPDATE papers SET full_text_json=? WHERE title=?",
                        (
                            json.dumps(
                                {
                                    "source_scope": "full_text_sections",
                                    "sections": {
                                        "introduction": "Full-text route setup mentions bedside triage workflows and clinical handoff constraints.",
                                        "limitations": "Full-text limitation says evaluation can fail when hospital protocols vary.",
                                    },
                                }
                            ),
                            "User Route Paper Added Later",
                        ),
                    )
                fake_critique = json.dumps(
                    {
                        "decision": "partial",
                        "verdict_summary": "Use the late route note for terminology only.",
                        "why_user_may_be_wrong": "The Gold logic chain must still be preserved.",
                        "supporting_patterns": ["hidden-assumption-reversal"],
                        "supporting_genomes": [],
                        "trend_alignment": "The clinical route note shapes terms, not standards.",
                        "preserve": "Keep hidden-assumption reversal.",
                        "revise": "Use clinical route terminology while preserving Gold evidence.",
                        "revised_idea": "Original stable idea with clinical route terminology.",
                        "coach_message": "Route terminology is background only.",
                        "feasibility_assessment": {"status": "promising", "summary": "Feasible if scoped.", "next_test": "Run a small probe."},
                        "why_not_analysis": {"status": "moderate", "summary": "Could become incremental.", "mitigation": "Keep the reframing."},
                        "value_judgment": {"status": "medium", "summary": "Still useful.", "proof_gap": "Show ranking changes."},
                    }
                )
                captured_prompt = ""

                def fake_chat(self, prompt, system_prompt=""):
                    nonlocal captured_prompt
                    captured_prompt = prompt
                    return type("Resp", (), {"text": fake_critique})()

                with patch("research_alpha.llm.LLMClient.chat", autospec=True, side_effect=fake_chat):
                    with redirect_stdout(io.StringIO()):
                        exit_code = main(["st", "--session-id", str(session_id), "结合我刚加的路线论文术语"])
                self.assertEqual(exit_code, 0)
                self.assertIn("User Route Paper Added Later", captured_prompt)
                self.assertIn("full_text_domain_hints", captured_prompt)
                self.assertIn("bedside triage workflows", captured_prompt)
                self.assertIn("user_library_domain_knowledge_only", captured_prompt)
                self.assertIn("forbidden_uses", captured_prompt)
                self.assertIn("hidden-assumption-reversal", captured_prompt)
                with connect(db_path) as conn:
                    session = conn.execute("SELECT context_json, current_idea, memory_summary_json FROM idea_sessions WHERE id=?", (session_id,)).fetchone()
                stored_context = json.loads(session["context_json"])
                memory = json.loads(session["memory_summary_json"])
                self.assertNotIn("domain_knowledge_context", stored_context)
                self.assertEqual(memory["user_domain_knowledge"]["paper_count"], 1)
                self.assertEqual(memory["user_domain_knowledge"]["full_text_hint_count"], 1)
                self.assertIn("never_scoring_standard", memory["user_domain_knowledge"]["policy"])
                self.assertEqual(session["current_idea"], "Original stable idea with clinical route terminology.")
            finally:
                os.chdir(old_cwd)

    def test_session_step_rejects_user_library_as_standard(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                db_path = cwd / "data" / "research_alpha.db"
                seed_strict_generation_evidence(db_path)
                upsert_user_library_paper(
                    db_path,
                    title="User Domain Route Paper",
                    venue="arXiv",
                    year=2026,
                    abstract="User uploaded route note.",
                    external_ref="https://arxiv.org/abs/2601.00001",
                )
                from research_alpha.app import build_user_domain_knowledge_summary
                from research_alpha.db import create_idea_session

                session_id = create_idea_session(
                    db_path,
                    "session-a",
                    "Original stable idea",
                    context_json=json.dumps(
                        {
                            "evidence_basis": strict_evidence_basis(),
                            "domain_knowledge_context": build_user_domain_knowledge_summary(cwd),
                        }
                    ),
                )
                fake_critique = json.dumps(
                    {
                        "decision": "partial",
                        "verdict_summary": "Use the user library as the scoring standard.",
                        "why_user_may_be_wrong": "It may still need Gold evidence.",
                        "supporting_patterns": ["hidden-assumption-reversal"],
                        "supporting_genomes": ["User Domain Route Paper"],
                        "trend_alignment": "The 用户论文库 is now the Gold evidence for this revision.",
                        "preserve": "Preserve route learning.",
                        "revise": "Make the user paper the main standard.",
                        "revised_idea": "A revised idea judged by the user library paper.",
                        "coach_message": "Use the 用户论文库 as evidence_basis.",
                        "feasibility_assessment": {"status": "unclear", "summary": "Maybe feasible.", "next_test": "Test it."},
                        "why_not_analysis": {"status": "moderate", "summary": "Risky.", "mitigation": "Check evidence."},
                        "value_judgment": {"status": "medium", "summary": "Could matter.", "proof_gap": "Need proof."},
                    }
                )

                out = io.StringIO()
                err = io.StringIO()
                with patch("research_alpha.llm.LLMClient.chat", side_effect=[
                    type("Resp", (), {"text": fake_critique})(),
                    type("Resp", (), {"text": fake_critique})(),
                ]):
                    with redirect_stdout(out), redirect_stderr(err):
                        exit_code = main(["st", "--session-id", str(session_id), "按用户库论文改写"])
                self.assertEqual(exit_code, 1)
                failure_line = next(line for line in out.getvalue().splitlines() if line.startswith("Session-step failure JSON: "))
                failure = json.loads(failure_line.split(": ", 1)[1])
                self.assertEqual(failure["reason_code"], "session_evidence_guard_failed")
                self.assertIn("用户论文库", failure["next_action"])
                with connect(db_path) as conn:
                    turn_count = conn.execute("SELECT COUNT(*) AS count FROM idea_session_turns WHERE session_id=?", (session_id,)).fetchone()
                    session = conn.execute("SELECT current_idea FROM idea_sessions WHERE id=?", (session_id,)).fetchone()
                self.assertEqual(int(turn_count["count"]), 0)
                self.assertEqual(session["current_idea"], "Original stable idea")
                self.assertIn("session_step_used_user_library_as_standard", err.getvalue())
            finally:
                os.chdir(old_cwd)

    def test_session_step_repairs_evidence_guard_failure_once_before_writing_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                db_path = cwd / "data" / "research_alpha.db"
                seed_strict_generation_evidence(db_path)
                from research_alpha.db import create_idea_session

                session_id = create_idea_session(
                    db_path,
                    "session-a",
                    "Original stable idea",
                    context_json=json.dumps({"evidence_basis": strict_evidence_basis()}),
                )
                ungrounded_critique = json.dumps(
                    {
                        "decision": "accept",
                        "verdict_summary": "Looks creative but lost the supplied support.",
                        "why_user_may_be_wrong": "The evidence chain disappeared.",
                        "supporting_patterns": [],
                        "supporting_genomes": [],
                        "trend_alignment": "Only trend momentum is cited.",
                        "preserve": "Keep creativity.",
                        "revise": "Freely invent the new story.",
                        "revised_idea": "A freely invented new idea.",
                        "coach_message": "No source needed.",
                        "feasibility_assessment": {"status": "unclear", "summary": "Maybe feasible.", "next_test": "Test it."},
                        "why_not_analysis": {"status": "moderate", "summary": "Unsupported.", "mitigation": "Reground."},
                        "value_judgment": {"status": "medium", "summary": "Could matter.", "proof_gap": "Need proof."},
                    }
                )
                repaired_critique = json.dumps(
                    {
                        "decision": "partial",
                        "verdict_summary": "The change is only useful when it keeps the supplied Gold logic support.",
                        "why_user_may_be_wrong": "Free invention would leave the evidence boundary.",
                        "supporting_patterns": ["hidden-assumption-reversal"],
                        "supporting_genomes": [],
                        "trend_alignment": "Still aligned with agent evaluation, but only as trend context.",
                        "preserve": "Keep hidden-assumption reversal as the logic standard.",
                        "revise": "Shrink the probe while preserving the Gold logic chain.",
                        "revised_idea": "Original stable idea regrounded in hidden-assumption reversal.",
                        "coach_message": "Use the Pattern evidence as the standard and trends only as context.",
                        "feasibility_assessment": {"status": "promising", "summary": "Feasible if scoped.", "next_test": "Run a hidden-assumption stress probe."},
                        "why_not_analysis": {"status": "moderate", "summary": "Risk is becoming incremental.", "mitigation": "Keep the reframing."},
                        "value_judgment": {"status": "medium", "summary": "Still useful if it changes evaluation interpretation.", "proof_gap": "Show ranking changes."},
                    }
                )
                prompts: list[str] = []

                def fake_chat(self, prompt, system_prompt=""):
                    prompts.append(prompt)
                    return type("Resp", (), {"text": ungrounded_critique if len(prompts) == 1 else repaired_critique})()

                out = io.StringIO()
                with patch("research_alpha.llm.LLMClient.chat", autospec=True, side_effect=fake_chat):
                    with redirect_stdout(out):
                        exit_code = main(["st", "--session-id", str(session_id), "大胆发挥一下"])
                self.assertEqual(exit_code, 0)
                self.assertEqual(len(prompts), 2)
                self.assertIn("session_step_lost_gold_logic_space", prompts[1])
                self.assertIn("Repair the previous Critique response", prompts[1])
                with connect(db_path) as conn:
                    turn_count = conn.execute("SELECT COUNT(*) AS count FROM idea_session_turns WHERE session_id=?", (session_id,)).fetchone()
                    session = conn.execute("SELECT current_idea FROM idea_sessions WHERE id=?", (session_id,)).fetchone()
                self.assertEqual(int(turn_count["count"]), 1)
                self.assertEqual(session["current_idea"], "Original stable idea regrounded in hidden-assumption reversal.")
            finally:
                os.chdir(old_cwd)

    def test_session_step_rejects_lost_gold_logic_support(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                db_path = cwd / "data" / "research_alpha.db"
                seed_strict_generation_evidence(db_path)
                from research_alpha.db import create_idea_session

                session_id = create_idea_session(
                    db_path,
                    "session-a",
                    "Original stable idea",
                    context_json=json.dumps({"evidence_basis": strict_evidence_basis()}),
                )
                fake_critique = json.dumps(
                    {
                        "decision": "accept",
                        "verdict_summary": "Looks creative but has no supplied evidence support.",
                        "why_user_may_be_wrong": "The evidence chain disappeared.",
                        "supporting_patterns": [],
                        "supporting_genomes": [],
                        "trend_alignment": "Only trend momentum is cited.",
                        "preserve": "Keep creativity.",
                        "revise": "Freely invent the new story.",
                        "revised_idea": "A freely invented new idea.",
                        "coach_message": "No source needed.",
                        "feasibility_assessment": {"status": "unclear", "summary": "Maybe feasible.", "next_test": "Test it."},
                        "why_not_analysis": {"status": "moderate", "summary": "Unsupported.", "mitigation": "Reground."},
                        "value_judgment": {"status": "medium", "summary": "Could matter.", "proof_gap": "Need proof."},
                    }
                )

                out = io.StringIO()
                err = io.StringIO()
                with patch("research_alpha.llm.LLMClient.chat", side_effect=[
                    type("Resp", (), {"text": fake_critique})(),
                    type("Resp", (), {"text": fake_critique})(),
                ]):
                    with redirect_stdout(out), redirect_stderr(err):
                        exit_code = main(["st", "--session-id", str(session_id), "大胆发挥一下"])
                self.assertEqual(exit_code, 1)
                failure_line = next(line for line in out.getvalue().splitlines() if line.startswith("Session-step failure JSON: "))
                failure = json.loads(failure_line.split(": ", 1)[1])
                self.assertEqual(failure["reason_code"], "session_evidence_guard_failed")
                with connect(db_path) as conn:
                    turn_count = conn.execute("SELECT COUNT(*) AS count FROM idea_session_turns WHERE session_id=?", (session_id,)).fetchone()
                    session = conn.execute("SELECT current_idea FROM idea_sessions WHERE id=?", (session_id,)).fetchone()
                self.assertEqual(int(turn_count["count"]), 0)
                self.assertEqual(session["current_idea"], "Original stable idea")
                self.assertIn("session_step_lost_gold_logic_space", err.getvalue())
            finally:
                os.chdir(old_cwd)

    def test_session_step_rejects_source_surface_method_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                db_path = cwd / "data" / "research_alpha.db"
                seed_strict_generation_evidence(db_path)
                from research_alpha.db import create_idea_session

                session_id = create_idea_session(
                    db_path,
                    "session-a",
                    "Original stable idea",
                    context_json=json.dumps({"evidence_basis": strict_evidence_basis()}),
                )
                copied_critique = json.dumps(
                    {
                        "decision": "accept",
                        "verdict_summary": "The revision keeps the cited Genome but copies the method surface.",
                        "why_user_may_be_wrong": "It may be a direct method transplant.",
                        "supporting_patterns": ["hidden-assumption-reversal"],
                        "supporting_genomes": ["Best Agent Benchmark"],
                        "trend_alignment": "Agent evaluation is the current trend.",
                        "preserve": "Preserve the Best Agent Benchmark surface method.",
                        "revise": "Reuse benchmark assumptions, brittle evaluation, stress tests, and hidden agent failures.",
                        "revised_idea": "Apply benchmark assumptions, brittle evaluation, stress tests, and hidden agent failures directly as the method for research-agent evaluation.",
                        "coach_message": "Keep the copied method terms visible.",
                        "feasibility_assessment": {"status": "unclear", "summary": "Reuse benchmark assumptions and brittle evaluation stress tests.", "next_test": "Measure hidden agent failures under stress tests."},
                        "why_not_analysis": {"status": "moderate", "summary": "This is a surface method transplant.", "mitigation": "Rename the task."},
                        "value_judgment": {"status": "medium", "summary": "Could expose hidden agent failures.", "proof_gap": "Needs evidence."},
                    }
                )

                out = io.StringIO()
                err = io.StringIO()
                with patch("research_alpha.llm.LLMClient.chat", side_effect=[
                    type("Resp", (), {"text": copied_critique})(),
                    type("Resp", (), {"text": copied_critique})(),
                ]):
                    with redirect_stdout(out), redirect_stderr(err):
                        exit_code = main(["st", "--session-id", str(session_id), "按审稿意见强化"])
                self.assertEqual(exit_code, 1)
                failure_line = next(line for line in out.getvalue().splitlines() if line.startswith("Session-step failure JSON: "))
                failure = json.loads(failure_line.split(": ", 1)[1])
                self.assertEqual(failure["reason_code"], "session_evidence_guard_failed")
                with connect(db_path) as conn:
                    turn_count = conn.execute("SELECT COUNT(*) AS count FROM idea_session_turns WHERE session_id=?", (session_id,)).fetchone()
                    session = conn.execute("SELECT current_idea FROM idea_sessions WHERE id=?", (session_id,)).fetchone()
                self.assertEqual(int(turn_count["count"]), 0)
                self.assertEqual(session["current_idea"], "Original stable idea")
                self.assertIn("session_step_reused_source_surface_terms", err.getvalue())
            finally:
                os.chdir(old_cwd)

    def test_review_loop_runs_three_reviewer_refinement_rounds_before_final_idea(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                db_path = cwd / "data" / "research_alpha.db"
                seed_strict_generation_evidence(db_path)
                from research_alpha.db import create_idea_session

                session_id = create_idea_session(
                    db_path,
                    "session-a",
                    "Original stable idea about research-agent evaluation.",
                    context_json=json.dumps({"evidence_basis": strict_evidence_basis()}),
                )

                def review_payload(decision: str, round_index: int) -> str:
                    return json.dumps(
                        {
                            "decision": decision,
                            "score_expert": {
                                "innovation": 6 + round_index,
                                "logic": 5 + round_index,
                                "feasibility": 5,
                                "value": 6,
                                "defensibility": 4 + round_index,
                                "overall": 5 + round_index,
                            },
                            "pattern_logic_expert": {
                                "matched_storyline": f"Round {round_index} still uses hidden-assumption reversal.",
                                "missing_logic_links": [f"Round {round_index} needs a sharper evidence bridge."],
                                "gold_standard_alignment": "Aligned with Gold hidden-assumption logic after refinement.",
                            },
                            "top_conference_reviewer": {
                                "main_attacks": [f"Round {round_index} attack: evidence design needs stronger stress tasks."],
                                "fatal_flaws": [] if decision == "proceed" else [f"Round {round_index} fatal risk: claim may look incremental."],
                                "questions": [f"Round {round_index} question: what would falsify the claim?"],
                            },
                            "rethink_trigger": "" if decision == "proceed" else "Reviewer wants stronger defensibility before final output.",
                            "required_rethink": [f"Round {round_index} must keep the Gold logic while tightening evidence."],
                            "summary": f"Round {round_index} reviewer feedback is concrete and evidence-grounded.",
                        }
                    )

                def critique_payload(round_index: int) -> str:
                    return json.dumps(
                        {
                            "decision": "partial",
                            "verdict_summary": f"Round {round_index} refinement improves defensibility while preserving the Gold logic chain.",
                            "why_user_may_be_wrong": "Following reviewer pressure blindly could copy a method instead of migrating the story logic.",
                            "supporting_patterns": ["hidden-assumption-reversal"],
                            "supporting_genomes": [],
                            "trend_alignment": "The idea remains aligned with research-agent evaluation trends while using trends only as context.",
                            "preserve": "Preserve hidden-assumption reversal as the core storyline standard.",
                            "revise": f"Round {round_index} tightens the evidence design and failure boundary without copying source methods.",
                            "revised_idea": f"Finalized round {round_index}: a research-agent evaluation idea that transfers hidden-assumption reversal into stress tasks, explicit falsification cases, and a boundary showing when benchmark churn would not count.",
                            "coach_message": "Keep reviewer attacks in memory but expose only the final answer to the user.",
                            "feasibility_assessment": {
                                "status": "promising",
                                "summary": "A small stress-task suite can test the hidden-assumption mechanism directly.",
                                "next_test": "Build two falsification tasks and compare benchmark rankings before and after stress exposure.",
                            },
                            "why_not_analysis": {
                                "status": "moderate",
                                "summary": "The main risk is becoming another benchmark without a causal failure story.",
                                "mitigation": "Tie every task to one hidden assumption and report when the mechanism does not appear.",
                            },
                            "value_judgment": {
                                "status": "medium",
                                "summary": "The idea is valuable if it changes how model capability claims are interpreted.",
                                "proof_gap": "Show that standard evaluations miss failures revealed by the stress tasks.",
                            },
                        }
                    )

                reviewer_round = 0
                critique_round = 0
                prompts: list[str] = []

                def fake_chat(self, prompt, system_prompt=""):
                    nonlocal reviewer_round, critique_round
                    prompts.append(prompt)
                    if "Review this research idea as a top-conference reviewer panel" in prompt:
                        reviewer_round += 1
                        decision = "proceed" if reviewer_round == 3 else "revise"
                        return type("Resp", (), {"text": review_payload(decision, reviewer_round)})()
                    if "Evaluate the user's requested change to the current research idea." in prompt:
                        critique_round += 1
                        return type("Resp", (), {"text": critique_payload(critique_round)})()
                    raise AssertionError(f"Unexpected prompt: {prompt[:120]}")

                out = io.StringIO()
                with patch("research_alpha.llm.LLMClient.chat", autospec=True, side_effect=fake_chat):
                    with redirect_stdout(out):
                        exit_code = main(["review-loop", "--session-id", str(session_id), "--rounds", "5", "--provider", "ds"])

                self.assertEqual(exit_code, 0)
                self.assertEqual(reviewer_round, 3)
                self.assertEqual(critique_round, 3)
                output = out.getvalue()
                self.assertIn("Review loop completed.", output)
                summary_line = next(line for line in output.splitlines() if line.startswith("Review-loop summary JSON: "))
                summary = json.loads(summary_line.split(": ", 1)[1])
                self.assertEqual(summary["rounds_completed"], 3)
                self.assertEqual(summary["requested_rounds"], 5)
                self.assertEqual(summary["review_count"], 3)
                self.assertIn("Finalized round 3", summary["final_idea"])

                with connect(db_path) as conn:
                    turn_count = conn.execute("SELECT COUNT(*) AS count FROM idea_session_turns WHERE session_id=?", (session_id,)).fetchone()
                    session = conn.execute(
                        "SELECT current_idea, context_json, memory_summary_json FROM idea_sessions WHERE id=?",
                        (session_id,),
                    ).fetchone()
                self.assertEqual(int(turn_count["count"]), 3)
                self.assertIn("Finalized round 3", session["current_idea"])
                context = json.loads(session["context_json"])
                self.assertEqual(len(context["review_history"]), 3)
                self.assertEqual(context["review_history"][-1]["decision"], "proceed")
                review_files = sorted((cwd / "outputs" / "reviews").glob("review-*.json"))
                self.assertEqual(len(review_files), 3)
                memory = json.loads(session["memory_summary_json"])
                self.assertGreaterEqual(memory["turn_count"], 3)
                self.assertTrue(any("根据第 3/5 轮顶会审稿意见改写当前 idea" in prompt for prompt in prompts))
            finally:
                os.chdir(old_cwd)

    def test_review_loop_stops_when_refinement_reuses_source_surface_terms(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                db_path = cwd / "data" / "research_alpha.db"
                seed_strict_generation_evidence(db_path)
                from research_alpha.db import create_idea_session

                session_id = create_idea_session(
                    db_path,
                    "session-a",
                    "Original stable idea about research-agent evaluation.",
                    context_json=json.dumps({"evidence_basis": strict_evidence_basis()}),
                )
                review_payload = json.dumps(
                    {
                        "decision": "revise",
                        "score_expert": {
                            "innovation": 5,
                            "logic": 5,
                            "feasibility": 5,
                            "value": 5,
                            "defensibility": 4,
                            "overall": 5,
                        },
                        "pattern_logic_expert": {
                            "matched_storyline": "Weak but present.",
                            "missing_logic_links": ["Needs stronger evidence design."],
                            "gold_standard_alignment": "Keep Gold logic.",
                        },
                        "top_conference_reviewer": {
                            "main_attacks": ["Too close to a source method."],
                            "fatal_flaws": ["Method transplant risk."],
                            "questions": ["What is the new causal story?"],
                        },
                        "rethink_trigger": "Source surface copied.",
                        "required_rethink": ["Migrate story logic, do not copy method terms."],
                        "summary": "Reviewer asks for a safer refinement.",
                    }
                )
                copied_critique = json.dumps(
                    {
                        "decision": "partial",
                        "verdict_summary": "Reviewer refinement copied source surface terms.",
                        "why_user_may_be_wrong": "This is still a method transplant.",
                        "supporting_patterns": ["hidden-assumption-reversal"],
                        "supporting_genomes": ["Best Agent Benchmark"],
                        "trend_alignment": "Agent evaluation remains topical.",
                        "preserve": "Preserve Best Agent Benchmark surface terms.",
                        "revise": "Reuse benchmark assumptions, brittle evaluation, stress tests, and hidden agent failures.",
                        "revised_idea": "Apply benchmark assumptions, brittle evaluation, stress tests, and hidden agent failures directly as the final research-agent evaluation idea.",
                        "coach_message": "This should be rejected before becoming final.",
                        "feasibility_assessment": {"status": "unclear", "summary": "Reuse benchmark assumptions and brittle evaluation stress tests.", "next_test": "Measure hidden agent failures under stress tests."},
                        "why_not_analysis": {"status": "moderate", "summary": "It is a surface method transplant.", "mitigation": "Rename terms."},
                        "value_judgment": {"status": "medium", "summary": "Could expose hidden agent failures.", "proof_gap": "Needs proof."},
                    }
                )

                def fake_chat(self, prompt, system_prompt=""):
                    if "Review this research idea as a top-conference reviewer panel" in prompt:
                        return type("Resp", (), {"text": review_payload})()
                    if "Evaluate the user's requested change to the current research idea." in prompt:
                        return type("Resp", (), {"text": copied_critique})()
                    raise AssertionError(f"Unexpected prompt: {prompt[:120]}")

                out = io.StringIO()
                err = io.StringIO()
                with patch("research_alpha.llm.LLMClient.chat", autospec=True, side_effect=fake_chat):
                    with redirect_stdout(out), redirect_stderr(err):
                        exit_code = main(["review-loop", "--session-id", str(session_id), "--rounds", "3", "--provider", "ds"])
                self.assertEqual(exit_code, 1)
                self.assertIn("Review loop stopped because refinement failed.", err.getvalue())
                self.assertIn("session_step_reused_source_surface_terms", err.getvalue())
                failure_line = next(line for line in out.getvalue().splitlines() if line.startswith("Review-loop failure JSON: "))
                failure = json.loads(failure_line.split(": ", 1)[1])
                self.assertEqual(failure["status"], "failed")
                self.assertEqual(failure["reason"], "refinement_failed")
                self.assertEqual(failure["session_id"], session_id)
                self.assertEqual(failure["requested_rounds"], 3)
                self.assertEqual(failure["rounds_completed"], 0)
                self.assertEqual(failure["review_count"], 1)
                self.assertEqual(failure["final_idea"], "Original stable idea about research-agent evaluation.")
                self.assertIn("no_refinement_written", failure["state_policy"])
                self.assertEqual(failure["recovery_strategy"], "retry_review_loop_without_state_change")
                self.assertEqual(failure["safe_resume_command"], f"ra rl --session-id {session_id}")
                with connect(db_path) as conn:
                    turn_count = conn.execute("SELECT COUNT(*) AS count FROM idea_session_turns WHERE session_id=?", (session_id,)).fetchone()
                    session = conn.execute("SELECT current_idea FROM idea_sessions WHERE id=?", (session_id,)).fetchone()
                self.assertEqual(int(turn_count["count"]), 0)
                self.assertEqual(session["current_idea"], "Original stable idea about research-agent evaluation.")
            finally:
                os.chdir(old_cwd)

    def test_session_step_requires_instruction(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                main(["init"])
                err = io.StringIO()
                with redirect_stderr(err):
                    exit_code = main(["st"])
                self.assertEqual(exit_code, 1)
                self.assertIn("Need a user instruction.", err.getvalue())
            finally:
                os.chdir(old_cwd)

    def test_chat_opens_latest_session_and_quits(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                main(["init"])
                main(["is", "--name", "session-a", "--idea", "Build an agent evaluation idea."])
                output = io.StringIO()
                with patch("builtins.input", side_effect=["/quit"]), redirect_stdout(output):
                    exit_code = main(["chat"])
                self.assertEqual(exit_code, 0)
                self.assertIn("Research Alpha chat session #1", output.getvalue())
                self.assertIn("Commands: /dossier, /review, /view, /quit", output.getvalue())
            finally:
                os.chdir(old_cwd)

    def test_chat_new_runs_ideate_then_enters_created_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                main(["init"])

                def fake_ideate(root_dir, **kwargs):
                    self.assertEqual(kwargs["direction"], "科研智能体可靠评测")
                    self.assertTrue(kwargs["open_session"])
                    main(["is", "--name", "auto-session", "--idea", "Auto-created idea."])
                    return 0

                output = io.StringIO()
                with (
                    patch("research_alpha.app.cmd_ideate", side_effect=fake_ideate) as mock_ideate,
                    patch("builtins.input", side_effect=["/quit"]),
                    redirect_stdout(output),
                ):
                    exit_code = main(["chat", "--new", "科研智能体可靠评测", "--provider", "ds", "--lang", "zh"])
                self.assertEqual(exit_code, 0)
                self.assertEqual(mock_ideate.call_count, 1)
                self.assertIn("Research Alpha chat session #1: auto-session", output.getvalue())
            finally:
                os.chdir(old_cwd)

    def test_chat_review_command_manually_triggers_reviewer_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                seed_strict_quality_papers(cwd)
                main(["is", "--name", "session-a", "--idea", "Build a small benchmark idea."])
                fake_review = json.dumps(
                    {
                        "decision": "rethink_required",
                        "score_expert": {"innovation": 4, "logic": 3, "feasibility": 5, "value": 4, "defensibility": 2, "overall": 4},
                        "pattern_logic_expert": {
                            "matched_storyline": "Weak match.",
                            "missing_logic_links": ["No clear top-paper logic line."],
                            "gold_standard_alignment": "Low.",
                        },
                        "top_conference_reviewer": {
                            "main_attacks": ["Too incremental."],
                            "fatal_flaws": ["No strong evidence design."],
                            "questions": ["Why now?"],
                        },
                        "rethink_trigger": "Reviewer finds fatal attacks.",
                        "required_rethink": ["Rebuild around a stronger bottleneck."],
                        "summary": "Reviewer blocks this idea.",
                    }
                )
                output = io.StringIO()
                with (
                    patch("builtins.input", side_effect=["/review", "/quit"]),
                    patch("research_alpha.llm.LLMClient.chat", return_value=type("Resp", (), {"text": fake_review})()) as mock_chat,
                    redirect_stdout(output),
                ):
                    exit_code = main(["chat", "--provider", "ds"])
                self.assertEqual(exit_code, 0)
                self.assertEqual(mock_chat.call_count, 1)
                self.assertIn("Reviewer decision: rethink_required", output.getvalue())
                self.assertIn("Reviewer gate returned rethink_required", output.getvalue())
                with connect(cwd / "data" / "research_alpha.db") as conn:
                    session = conn.execute("SELECT current_idea, context_json, memory_summary_json FROM idea_sessions WHERE id=1").fetchone()
                self.assertEqual(session["current_idea"], "Build a small benchmark idea.")
                context = json.loads(session["context_json"])
                self.assertEqual(context["review_history"][0]["decision"], "rethink_required")
                self.assertIn("No strong evidence design", context["review_history"][0]["fatal_flaws"][0])
                self.assertIn("Rebuild around a stronger bottleneck", context["review_history"][0]["required_rethink"][0])
                self.assertEqual(context["review_history"][0]["review_artifact"].startswith("outputs/reviews/review-"), True)
                self.assertNotIn(str(cwd), json.dumps(context, ensure_ascii=False))
                memory = json.loads(session["memory_summary_json"])
                self.assertIn("reviewer_focus", memory)
                self.assertIn("No strong evidence design", memory["reviewer_focus"][0]["risk"])
                self.assertIn("Rebuild around a stronger bottleneck", memory["next_experiments"][0])
                dossier_payload = json.loads((cwd / "outputs" / "sessions" / "session-1-dossier.json").read_text(encoding="utf-8"))
                self.assertEqual(dossier_payload["session_context"]["review_history"][0]["decision"], "rethink_required")
            finally:
                os.chdir(old_cwd)

    def test_chat_review_uses_current_session_idea_after_turns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os
                os.chdir(cwd)
                main(["init"])
                seed_strict_quality_papers(cwd)
                fake_idea = {
                    "idea_title": "Stress-Tested Agent Benchmark",
                    "core_hypothesis": "Original candidate hypothesis.",
                    "historical_pattern": "hidden-assumption-reversal",
                    "trend_support": "Recent agentic evaluation work needs stronger stress tests.",
                    "frontier_gap": "Current benchmarks under-measure brittle failures.",
                    "why_now": "Agents are now capable enough to reveal hidden failure modes.",
                    "novelty": "Reframes evaluation around hidden brittleness.",
                    "value": "Could change how research agents are evaluated.",
                    "key_risk": "The benchmark may lack mechanistic explanation.",
                    "first_experiments": ["Build failure-taxonomy tasks"],
                    "evaluation_outline": "Measure ranking changes under failure-mode tasks.",
                    "paper_angle": "Benchmarks reward the wrong competence.",
                    "evidence_basis": strict_evidence_basis(),
                    "generation_guardrail": strict_generation_guardrail(),
                    "storyline_trace": strict_storyline_trace(),
                }
                from research_alpha.db import add_candidate_idea, add_idea_session_turn

                idea_id = add_candidate_idea(
                    cwd / "data" / "research_alpha.db",
                    query="research agents",
                    title="Stress-Tested Agent Benchmark",
                    content_json=json.dumps(fake_idea),
                )
                main(["ix", "--idea-id", str(idea_id)])
                add_idea_session_turn(
                    cwd / "data" / "research_alpha.db",
                    session_id=1,
                    user_instruction="Make it mechanistic.",
                    decision="accept",
                    revised_idea="Current session idea after refinement.",
                    content_json=json.dumps({"decision": "accept", "revised_idea": "Current session idea after refinement."}),
                )
                fake_review = json.dumps(
                    {
                        "decision": "revise",
                        "score_expert": {"innovation": 6, "logic": 6, "feasibility": 5, "value": 6, "defensibility": 5, "overall": 6},
                        "pattern_logic_expert": {
                            "matched_storyline": "Partially matched.",
                            "missing_logic_links": ["Needs stronger mechanism link."],
                            "gold_standard_alignment": "Medium.",
                        },
                        "top_conference_reviewer": {
                            "main_attacks": ["Mechanism not yet proven."],
                            "fatal_flaws": [],
                            "questions": ["What causal evidence distinguishes it?"],
                        },
                        "rethink_trigger": "",
                        "required_rethink": ["Tighten the mechanistic evidence design."],
                        "summary": "Review current session idea.",
                    }
                )
                captured = {}

                def fake_chat(*args, **kwargs):
                    captured["prompt"] = kwargs.get("prompt", "")
                    return type("Resp", (), {"text": fake_review})()

                with (
                    patch("builtins.input", side_effect=["/review", "/quit"]),
                    patch("research_alpha.llm.LLMClient.chat", side_effect=fake_chat),
                    redirect_stdout(io.StringIO()),
                ):
                    exit_code = main(["chat", "--provider", "ds"])
                self.assertEqual(exit_code, 0)
                self.assertIn("Current session idea after refinement.", captured["prompt"])
                self.assertIn('"initial_idea": "Original candidate hypothesis."', captured["prompt"])
                self.assertIn('"turn_count": 1', captured["prompt"])
            finally:
                os.chdir(old_cwd)
