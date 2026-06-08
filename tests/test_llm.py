import json
import os
import tempfile
import urllib.error
import unittest
from pathlib import Path
from unittest.mock import patch

from research_alpha.connectors import (
    decode_openalex_abstract,
    extract_conference_years_from_url,
    harvest_openalex,
    normalize_openalex_work,
    normalize_semantic_scholar_paper,
    openalex_work_matches_target_year,
    ranked_openalex_source_ids,
)
from research_alpha.config import LLMConfig, load_config
from research_alpha.genome import parse_genome_response
from research_alpha.llm import LLMClient, LLMError, _extract_text


class LLMTests(unittest.TestCase):
    def test_extract_text_from_string_content(self) -> None:
        payload = {"choices": [{"message": {"content": "hello"}}]}
        self.assertEqual(_extract_text(payload), "hello")

    def test_extract_text_from_list_content(self) -> None:
        payload = {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": "alpha"},
                            {"type": "text", "text": "beta"},
                        ]
                    }
                }
            ]
        }
        self.assertEqual(_extract_text(payload), "alpha\nbeta")

    def test_decode_openalex_abstract(self) -> None:
        index = {"hello": [0], "world": [1]}
        self.assertEqual(decode_openalex_abstract(index), "hello world")

    def test_openalex_venue_alias_uses_source_id_filter(self) -> None:
        calls = []

        def fake_get_json(url, *, params, api_key="", email=""):
            calls.append((url, dict(params)))
            if url.endswith("/sources"):
                return {
                    "results": [
                        {
                            "id": "https://openalex.org/S4306419637",
                            "display_name": "International Conference on Learning Representations",
                            "type": "conference",
                        }
                    ]
                }
            return {
                "results": [
                    {
                        "id": "https://openalex.org/W1",
                        "display_name": "Remote Paper",
                        "publication_year": 2025,
                        "publication_date": "2025-01-01",
                        "cited_by_count": 7,
                        "primary_location": {
                            "source": {"display_name": "International Conference on Learning Representations"}
                        },
                        "abstract_inverted_index": {"agent": [0], "benchmark": [1]},
                    }
                ]
            }

        with patch("research_alpha.connectors._get_json", side_effect=fake_get_json):
            records = harvest_openalex(venue="ICLR", year=2025, limit=3, query="agents")

        self.assertEqual(records[0]["venue"], "International Conference on Learning Representations")
        works_params = calls[-1][1]
        self.assertIn("locations.source.id:S4306419637", works_params["filter"])
        self.assertNotIn("primary_location.source.display_name.search", works_params["filter"])

    def test_openalex_normalize_prefers_matching_conference_location(self) -> None:
        record = normalize_openalex_work(
            {
                "id": "https://openalex.org/W1",
                "display_name": "Remote Paper",
                "publication_year": 2025,
                "publication_date": "2025-01-01",
                "cited_by_count": 7,
                "primary_location": {
                    "source": {
                        "id": "https://openalex.org/S7407053387",
                        "display_name": "TIB Data Manager",
                        "type": "repository",
                    }
                },
                "locations": [
                    {
                        "source": {
                            "id": "https://openalex.org/S7407053387",
                            "display_name": "TIB Data Manager",
                            "type": "repository",
                        }
                    },
                    {
                        "source": {
                            "id": "https://openalex.org/S4306419637",
                            "display_name": "International Conference on Learning Representations",
                            "type": "conference",
                        }
                    },
                ],
                "abstract_inverted_index": {},
            },
            preferred_source_ids=["https://openalex.org/S4306419637"],
        )
        self.assertEqual(record["venue"], "International Conference on Learning Representations")

    def test_openalex_source_ranking_keeps_only_exact_conference(self) -> None:
        results = [
            {
                "id": "https://openalex.org/S4306419644",
                "display_name": "International Conference on Machine Learning",
                "type": "conference",
            },
            {
                "id": "https://openalex.org/S4306419645",
                "display_name": "International Conference on Machine Learning and Applications",
                "type": "conference",
            },
            {
                "id": "https://openalex.org/S4363606243",
                "display_name": "Neural Information Processing Systems",
                "type": "journal",
            },
        ]
        self.assertEqual(
            ranked_openalex_source_ids(results, "International Conference on Machine Learning"),
            ["https://openalex.org/S4306419644"],
        )
        self.assertEqual(ranked_openalex_source_ids(results, "Neural Information Processing Systems"), [])

    def test_openalex_source_ranking_allows_tpami_journal(self) -> None:
        results = [
            {
                "id": "https://openalex.org/S123",
                "display_name": "IEEE Transactions on Pattern Analysis and Machine Intelligence",
                "type": "journal",
            }
        ]
        self.assertEqual(
            ranked_openalex_source_ids(results, "IEEE Transactions on Pattern Analysis and Machine Intelligence"),
            ["https://openalex.org/S123"],
        )

    def test_openalex_url_year_evidence_filters_old_proceedings_locations(self) -> None:
        self.assertEqual(
            extract_conference_years_from_url("http://proceedings.mlr.press/v119/chen20j/chen20j.pdf"),
            {2020},
        )
        self.assertEqual(
            extract_conference_years_from_url("https://proceedings.neurips.cc/paper/2020/file/hash-Paper.pdf"),
            {2020},
        )
        old_work = {
            "publication_year": 2024,
            "locations": [
                {
                    "landing_page_url": "http://proceedings.mlr.press/v119/chen20j/chen20j.pdf",
                    "source": {
                        "id": "https://openalex.org/S4306419644",
                        "display_name": "International Conference on Machine Learning",
                        "type": "conference",
                    },
                }
            ],
        }
        current_work_without_url_year = {
            "publication_year": 2024,
            "locations": [
                {
                    "landing_page_url": "https://openreview.net/pdf?id=abc",
                    "source": {
                        "id": "https://openalex.org/S4306419637",
                        "display_name": "International Conference on Learning Representations",
                        "type": "conference",
                    },
                }
            ],
        }
        self.assertFalse(
            openalex_work_matches_target_year(
                old_work,
                2024,
                preferred_source_ids=["https://openalex.org/S4306419644"],
            )
        )
        self.assertTrue(
            openalex_work_matches_target_year(
                current_work_without_url_year,
                2024,
                preferred_source_ids=["https://openalex.org/S4306419637"],
            )
        )

    def test_normalize_semantic_scholar_paper(self) -> None:
        payload = {
            "title": "Paper",
            "abstract": "Text",
            "year": 2024,
            "venue": "ICLR",
            "citationCount": 12,
            "influentialCitationCount": 3,
            "externalIds": {"DOI": "10.1000/test"},
            "url": "https://example.org",
        }
        record = normalize_semantic_scholar_paper(payload)
        self.assertEqual(record["external_ref"], "10.1000/test")
        self.assertEqual(record["citation_count"], 12)

    def test_parse_genome_response(self) -> None:
        payload = {
            "paper_summary": "summary",
            "pre_publication_belief": "belief",
            "bottleneck_or_hidden_assumption": "bottleneck",
            "problem_reframing": "reframing",
            "why_now": "why now",
            "evidence_design": "evidence",
            "story_line": "story",
            "transferable_pattern": "pattern",
            "failure_boundary": "boundary",
            "confidence_note": "note",
            "evidence_level": "something_else",
        }
        result = parse_genome_response(json.dumps(payload))
        self.assertEqual(result["evidence_level"], "abstract_only")
        self.assertEqual(result["logic_line"]["old_belief"], "belief")
        self.assertEqual(result["logic_line"]["bottleneck"], "bottleneck")
        self.assertEqual(result["logic_line"]["reframing"], "reframing")

    def test_parse_genome_response_accepts_fenced_json(self) -> None:
        payload = {
            "paper_summary": "summary",
            "pre_publication_belief": "belief",
            "bottleneck_or_hidden_assumption": "bottleneck",
            "problem_reframing": "reframing",
            "why_now": "why now",
            "evidence_design": "evidence",
            "story_line": "story",
            "transferable_pattern": "pattern",
            "failure_boundary": "boundary",
            "confidence_note": "note",
            "evidence_level": "something_else",
        }
        result = parse_genome_response(f"```json\n{json.dumps(payload)}\n```")
        self.assertEqual(result["logic_line"]["old_belief"], "belief")

    def test_parse_genome_response_rejects_empty_text_with_clear_message(self) -> None:
        with self.assertRaisesRegex(ValueError, "Genome response was empty"):
            parse_genome_response("")

    def test_load_config_reads_dotenv_and_provider_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            env_path = root / ".env"
            env_path.write_text(
                "RA_LLM_PROVIDER=ds\nDEEPSEEK_API_KEY=test-deepseek-key\n",
                encoding="utf-8",
            )
            old_provider = os.environ.pop("RA_LLM_PROVIDER", None)
            old_deepseek_key = os.environ.pop("DEEPSEEK_API_KEY", None)
            try:
                config = load_config(root)
            finally:
                if old_provider is not None:
                    os.environ["RA_LLM_PROVIDER"] = old_provider
                else:
                    os.environ.pop("RA_LLM_PROVIDER", None)
                if old_deepseek_key is not None:
                    os.environ["DEEPSEEK_API_KEY"] = old_deepseek_key
                else:
                    os.environ.pop("DEEPSEEK_API_KEY", None)
            self.assertEqual(config.llm.provider, "deepseek")
            self.assertEqual(config.llm.api_key, "test-deepseek-key")

    def test_load_config_uses_last_dotenv_value_for_same_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            env_path = root / ".env"
            env_path.write_text(
                "RA_LLM_PROVIDER=openai\nRA_LLM_PROVIDER=ds\nDEEPSEEK_API_KEY=test-deepseek-key\n",
                encoding="utf-8",
            )
            old_provider = os.environ.pop("RA_LLM_PROVIDER", None)
            old_openai_key = os.environ.pop("OPENAI_API_KEY", None)
            old_deepseek_key = os.environ.pop("DEEPSEEK_API_KEY", None)
            try:
                config = load_config(root)
            finally:
                if old_provider is not None:
                    os.environ["RA_LLM_PROVIDER"] = old_provider
                else:
                    os.environ.pop("RA_LLM_PROVIDER", None)
                if old_openai_key is not None:
                    os.environ["OPENAI_API_KEY"] = old_openai_key
                else:
                    os.environ.pop("OPENAI_API_KEY", None)
                if old_deepseek_key is not None:
                    os.environ["DEEPSEEK_API_KEY"] = old_deepseek_key
                else:
                    os.environ.pop("DEEPSEEK_API_KEY", None)
            self.assertEqual(config.llm.provider, "deepseek")
            self.assertEqual(config.llm.api_key, "test-deepseek-key")

    def test_load_config_prefers_provider_specific_key_over_generic_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            env_path = root / ".env"
            env_path.write_text(
                "RA_LLM_PROVIDER=ds\nRA_LLM_API_KEY=generic-key\nDEEPSEEK_API_KEY=deepseek-key\n",
                encoding="utf-8",
            )
            old_provider = os.environ.pop("RA_LLM_PROVIDER", None)
            old_generic_key = os.environ.pop("RA_LLM_API_KEY", None)
            old_deepseek_key = os.environ.pop("DEEPSEEK_API_KEY", None)
            try:
                config = load_config(root)
            finally:
                if old_provider is not None:
                    os.environ["RA_LLM_PROVIDER"] = old_provider
                else:
                    os.environ.pop("RA_LLM_PROVIDER", None)
                if old_generic_key is not None:
                    os.environ["RA_LLM_API_KEY"] = old_generic_key
                else:
                    os.environ.pop("RA_LLM_API_KEY", None)
                if old_deepseek_key is not None:
                    os.environ["DEEPSEEK_API_KEY"] = old_deepseek_key
                else:
                    os.environ.pop("DEEPSEEK_API_KEY", None)
            self.assertEqual(config.llm.provider, "deepseek")
            self.assertEqual(config.llm.api_key, "deepseek-key")

    def test_load_config_prefers_project_dotenv_over_inherited_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            env_path = root / ".env"
            env_path.write_text(
                "RA_LLM_PROVIDER=deepseek\nDEEPSEEK_API_KEY=project-deepseek-key\n",
                encoding="utf-8",
            )
            old_provider = os.environ.get("RA_LLM_PROVIDER")
            old_openai_key = os.environ.get("OPENAI_API_KEY")
            old_deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
            os.environ["RA_LLM_PROVIDER"] = "openai"
            os.environ["OPENAI_API_KEY"] = "ambient-openai-key"
            try:
                config = load_config(root)
            finally:
                if old_provider is not None:
                    os.environ["RA_LLM_PROVIDER"] = old_provider
                else:
                    os.environ.pop("RA_LLM_PROVIDER", None)
                if old_openai_key is not None:
                    os.environ["OPENAI_API_KEY"] = old_openai_key
                else:
                    os.environ.pop("OPENAI_API_KEY", None)
                if old_deepseek_key is not None:
                    os.environ["DEEPSEEK_API_KEY"] = old_deepseek_key
                else:
                    os.environ.pop("DEEPSEEK_API_KEY", None)
            self.assertEqual(config.llm.provider, "deepseek")
            self.assertEqual(config.llm.api_key, "project-deepseek-key")

    def test_missing_api_key_message_points_to_provider_shortcut(self) -> None:
        client = LLMClient(LLMConfig(provider="deepseek", api_key=""))
        with self.assertRaises(LLMError) as excinfo:
            client.chat("test prompt")
        message = str(excinfo.exception)
        self.assertIn("Missing API key for deepseek", message)
        self.assertIn("ra ds sk-...", message)
        self.assertIn("DEEPSEEK_API_KEY", message)

    def test_network_error_message_names_provider_and_connectivity(self) -> None:
        client = LLMClient(LLMConfig(provider="openai", api_key="test-openai-key"))
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("[Errno 8] nodename nor servname provided")):
            with self.assertRaises(LLMError) as excinfo:
                client.chat("test prompt")
        message = str(excinfo.exception)
        self.assertIn("LLM request to openai failed before a response came back", message)
        self.assertIn("https://api.openai.com/v1/chat/completions", message)
        self.assertIn("network, DNS, or proxy", message)

    def test_non_json_response_message_points_to_base_url(self) -> None:
        class FakeResponse:
            headers = {"Content-Type": "text/html; charset=utf-8"}

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self) -> bytes:
                return b"<!doctype html><title>Provider UI</title>"

        client = LLMClient(LLMConfig(provider="openai", api_key="test-openai-key", base_url="https://gateway.example"))
        with patch("urllib.request.urlopen", return_value=FakeResponse()):
            with self.assertRaises(LLMError) as excinfo:
                client.chat("test prompt")
        message = str(excinfo.exception)
        self.assertIn("returned a non-JSON response", message)
        self.assertIn("https://gateway.example/chat/completions", message)
        self.assertIn("usually end in `/v1`", message)
