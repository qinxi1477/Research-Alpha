import io
import json
import tempfile
import unittest
import urllib.error
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from research_alpha.app import (
    EXTRACTIVE_FULL_TEXT_EVIDENCE_LEVEL,
    build_extractive_genome_payload,
    fetch_full_text_for_paper_rows,
    build_evidence_report,
    main,
    run_pipeline_genome_build,
)
from research_alpha.db import add_paper, connect, init_db, upsert_user_library_paper
from research_alpha.genome import build_genome_prompt
from research_alpha.full_text import arxiv_pdf_fallback_url, direct_pdf_url_for_known_paper_page, discover_paper_link, extract_section_evidence, extract_text_from_pdf_bytes, extract_text_from_pdf_literal_chunks, fetch_full_text, full_text_json_for_text, infer_full_text_quality, normalize_full_text_url, sanitize_full_text_payload
from research_alpha.full_text import discover_pdf_from_paper_page, fetch_full_text_with_metadata, resolve_full_text_url_from_metadata


class FullTextTests(unittest.TestCase):
    def test_normalize_arxiv_abs_url_prefers_html(self) -> None:
        self.assertEqual(
            normalize_full_text_url("https://arxiv.org/abs/2601.01234v2"),
            "https://arxiv.org/html/2601.01234v2",
        )
        self.assertEqual(
            normalize_full_text_url("https://arxiv.org/pdf/2601.01234v2.pdf"),
            "https://arxiv.org/html/2601.01234v2",
        )

    def test_fetch_full_text_accepts_local_html_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "paper.html"
            path.write_text("<html><body><h1>Introduction</h1><p>Local evidence text.</p></body></html>", encoding="utf-8")
            text, source = fetch_full_text(str(path))
        self.assertIn("Local evidence text", text)
        self.assertEqual(source, str(path))

    def test_fetch_full_text_falls_back_from_arxiv_html_to_pdf(self) -> None:
        def fake_fetch(url, *, timeout, max_bytes):
            if url == "https://arxiv.org/html/2601.01234v2":
                raise urllib.error.HTTPError(url, 404, "missing html", {}, None)
            if url == "https://ar5iv.labs.arxiv.org/html/2601.01234v2":
                raise urllib.error.HTTPError(url, 404, "missing ar5iv", {}, None)
            self.assertEqual(url, "https://arxiv.org/pdf/2601.01234v2.pdf")
            return b"%PDF-1.4\n(Introduction PDF fallback evidence.)\n%%EOF", "application/pdf", url

        with patch("research_alpha.full_text.fetch_url_bytes", side_effect=fake_fetch):
            text, source = fetch_full_text("https://arxiv.org/abs/2601.01234v2")
        self.assertIn("PDF fallback evidence", text)
        self.assertEqual(source, "https://arxiv.org/pdf/2601.01234v2.pdf")

    def test_fetch_full_text_prefers_ar5iv_before_pdf_fallback(self) -> None:
        seen = []

        def fake_fetch(url, *, timeout, max_bytes):
            seen.append(url)
            if url == "https://arxiv.org/html/2601.01234v2":
                raise urllib.error.HTTPError(url, 404, "missing html", {}, None)
            self.assertEqual(url, "https://ar5iv.labs.arxiv.org/html/2601.01234v2")
            return (
                b"<html><body><h2>Introduction</h2><p>ar5iv full paper setup.</p><h2>Experiments</h2><p>ar5iv validation.</p></body></html>",
                "text/html",
                url,
            )

        with patch("research_alpha.full_text.fetch_url_bytes", side_effect=fake_fetch):
            text, source = fetch_full_text("https://arxiv.org/abs/2601.01234v2")
        self.assertIn("ar5iv full paper setup", text)
        self.assertEqual(source, "https://ar5iv.labs.arxiv.org/html/2601.01234v2")
        self.assertNotIn("https://arxiv.org/pdf/2601.01234v2.pdf", seen)

    def test_fetch_full_text_follows_short_html_paper_link(self) -> None:
        def fake_fetch(url, *, timeout, max_bytes):
            if url == "https://conf.example/paper/1":
                return (
                    b'<html><body><a href="https://openreview.net/pdf?id=abc">PDF</a><p>Abstract only.</p></body></html>',
                    "text/html",
                    url,
                )
            self.assertEqual(url, "https://openreview.net/pdf?id=abc")
            return b"%PDF-1.4\n(Introduction Linked PDF evidence text.)\n%%EOF", "application/pdf", url

        with patch("research_alpha.full_text.fetch_url_bytes", side_effect=fake_fetch):
            text, source = fetch_full_text("https://conf.example/paper/1")
        self.assertIn("Linked PDF evidence text", text)
        self.assertEqual(source, "https://openreview.net/pdf?id=abc")

    def test_discover_paper_link_ignores_generic_proceedings_homepage(self) -> None:
        html = '<a href="https://proceedings.mlr.press">ICML Proceedings at PMLR</a>'
        self.assertEqual(discover_paper_link(html, "https://icml.cc/virtual/2024/oral/1"), "")

    def test_resolve_icml_virtual_page_to_pmlr_pdf_by_title(self) -> None:
        pmlr_html = """
        <html><body>
        <div class="paper">
          <p class="title">Other Paper</p>
          <p class="links">[<a href="https://raw.githubusercontent.com/mlresearch/v235/main/assets/other/other.pdf">Download PDF</a>]</p>
        </div>
        <div class="paper">
          <p class="title">VideoPoet: A Large Language Model for Zero-Shot Video Generation</p>
          <p class="links">
            [<a href="https://proceedings.mlr.press/v235/kondratyuk24a.html">abs</a>]
            [<a href="https://raw.githubusercontent.com/mlresearch/v235/main/assets/kondratyuk24a/kondratyuk24a.pdf">Download PDF</a>]
          </p>
        </div>
        </body></html>
        """

        def fake_fetch(url, *, timeout, max_bytes):
            self.assertEqual(url, "https://proceedings.mlr.press/v235/")
            return pmlr_html.encode("utf-8"), "text/html", url

        with patch("research_alpha.full_text.fetch_url_bytes", side_effect=fake_fetch):
            resolved = resolve_full_text_url_from_metadata(
                "https://icml.cc/virtual/2024/oral/35537",
                title="VideoPoet: A Large Language Model for Zero-Shot Video Generation",
                venue="ICML 2024 Best Paper",
                year=2024,
            )
        self.assertEqual(resolved, "https://proceedings.mlr.press/v235/kondratyuk24a.html")

    def test_resolve_icml_2025_virtual_page_to_pmlr_volume(self) -> None:
        pmlr_html = """
        <html><body>
        <div class="paper">
          <p class="title">Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions</p>
          <p class="links">[<a href="https://proceedings.mlr.press/v267/chen25abc.html">abs</a>]</p>
        </div>
        </body></html>
        """

        def fake_fetch(url, *, timeout, max_bytes):
            self.assertEqual(url, "https://proceedings.mlr.press/v267/")
            return pmlr_html.encode("utf-8"), "text/html", url

        with patch("research_alpha.full_text.fetch_url_bytes", side_effect=fake_fetch):
            resolved = resolve_full_text_url_from_metadata(
                "https://icml.cc/virtual/2025/oral/47251",
                title="Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions",
                venue="ICML 2025 Outstanding Paper",
                year=2025,
            )
        self.assertEqual(resolved, "https://proceedings.mlr.press/v267/chen25abc.html")

    def test_resolve_neurips_virtual_page_to_proceedings_by_title(self) -> None:
        neurips_html = """
        <html><body>
        <ul class="paper-list">
          <li><div class="paper-content"><a title="paper title" href="/paper_files/paper/2024/hash/abc-Abstract-Conference.html">Learning diffusion at lightspeed</a></div></li>
        </ul>
        </body></html>
        """

        def fake_fetch(url, *, timeout, max_bytes):
            self.assertEqual(url, "https://papers.nips.cc/paper_files/paper/2024")
            return neurips_html.encode("utf-8"), "text/html", url

        with patch("research_alpha.full_text.fetch_url_bytes", side_effect=fake_fetch):
            resolved = resolve_full_text_url_from_metadata(
                "https://neurips.cc/virtual/2024/oral/97944",
                title="Learning diffusion at lightspeed",
                venue="NeurIPS 2024 Oral",
                year=2024,
            )
        self.assertEqual(resolved, "https://papers.nips.cc/paper_files/paper/2024/hash/abc-Abstract-Conference.html")

    def test_resolve_cvf_event_page_to_openaccess_by_title(self) -> None:
        cvf_html = """
        <html><body>
        <dt class="ptitle"><br><a href="/content/CVPR2025/html/Wang_VGGT_Visual_Geometry_Grounded_Transformer_CVPR_2025_paper.html">VGGT: Visual Geometry Grounded Transformer</a></dt>
        [<a href="/content/CVPR2025/papers/Wang_VGGT_Visual_Geometry_Grounded_Transformer_CVPR_2025_paper.pdf">pdf</a>]
        </body></html>
        """

        def fake_fetch(url, *, timeout, max_bytes):
            self.assertEqual(url, "https://openaccess.thecvf.com/CVPR2025?day=all")
            return cvf_html.encode("utf-8"), "text/html", url

        with patch("research_alpha.full_text.fetch_url_bytes", side_effect=fake_fetch):
            resolved = resolve_full_text_url_from_metadata(
                "https://cvpr.thecvf.com/Conferences/2025/BestPapersDemos",
                title="VGGT: Visual Geometry Grounded Transformer",
                venue="CVPR 2025 Best Paper",
                year=2025,
            )
        self.assertEqual(
            resolved,
            "https://openaccess.thecvf.com/content/CVPR2025/html/Wang_VGGT_Visual_Geometry_Grounded_Transformer_CVPR_2025_paper.html",
        )

    def test_fetch_full_text_with_metadata_falls_back_from_icml_shell_to_pmlr_pdf(self) -> None:
        pmlr_html_url = "https://proceedings.mlr.press/v235/kondratyuk24a.html"
        pmlr_pdf = "https://raw.githubusercontent.com/mlresearch/v235/main/assets/kondratyuk24a/kondratyuk24a.pdf"
        pmlr_html = f"""
        <html><body>
        <div class="paper">
          <p class="title">VideoPoet: A Large Language Model for Zero-Shot Video Generation</p>
          <p class="links">
            [<a href="{pmlr_html_url}">abs</a>]
            [<a href="{pmlr_pdf}">Download PDF</a>]
          </p>
        </div>
        </body></html>
        """
        pdf_text = (
            "Introduction This paper studies video generation with language-model structure and explains the problem setup. "
            "Related Work Prior video generation systems require task-specific components and narrow assumptions. "
            "Method We recast multimodal generation as token prediction over video and audio representations. "
            "Experiments We evaluate zero-shot generation, editing, and multimodal controllability across benchmarks. "
            "Limitations The method still depends on tokenizer quality and large-scale data availability. "
            "Conclusion The logic shows how a language-model interface unifies several generation tasks."
        )

        def fake_fetch(url, *, timeout, max_bytes):
            if url == "https://icml.cc/virtual/2024/oral/35537":
                return b'<html><body><a href="https://icml.cc/virtual/2024/papers.html">Papers</a></body></html>', "text/html", url
            if url == "https://proceedings.mlr.press/v235/":
                return pmlr_html.encode("utf-8"), "text/html", url
            if url == pmlr_html_url:
                return b"<html><body><p>Abstract only. Method brief. Experiments brief.</p><a href=\"https://raw.githubusercontent.com/mlresearch/v235/main/assets/kondratyuk24a/kondratyuk24a.pdf\">Download PDF</a></body></html>", "text/html", url
            self.assertEqual(url, pmlr_pdf)
            chunks = "".join(f"({sentence.strip() + '.'})" for sentence in pdf_text.split(".") if sentence.strip())
            return f"%PDF-1.4\n{chunks}\n%%EOF".encode("utf-8"), "application/pdf", url

        with patch("research_alpha.full_text.fetch_url_bytes", side_effect=fake_fetch):
            text, source = fetch_full_text_with_metadata(
                "https://icml.cc/virtual/2024/oral/35537",
                title="VideoPoet: A Large Language Model for Zero-Shot Video Generation",
                venue="ICML 2024 Best Paper",
                year=2024,
            )
        self.assertIn("unifies several generation tasks", text)
        self.assertEqual(source, pmlr_pdf)

    def test_fetch_full_text_with_metadata_keeps_pmlr_html_when_pdf_is_too_thin(self) -> None:
        pmlr_html_url = "https://proceedings.mlr.press/v235/kondratyuk24a.html"
        pmlr_pdf = "https://raw.githubusercontent.com/mlresearch/v235/main/assets/kondratyuk24a/kondratyuk24a.pdf"
        pmlr_index = f"""
        <html><body><div class="paper">
          <p class="title">VideoPoet: A Large Language Model for Zero-Shot Video Generation</p>
          <p class="links">[<a href="{pmlr_html_url}">abs</a>][<a href="{pmlr_pdf}">Download PDF</a>]</p>
        </div></body></html>
        """
        pmlr_page = f"""
        <html><body>
          <p>Abstract only.</p>
          <p>Method brief but readable. Experiments brief. Review signal.</p>
          <a href="{pmlr_pdf}">Download PDF</a>
        </body></html>
        """

        def fake_fetch(url, *, timeout, max_bytes):
            if url == "https://icml.cc/virtual/2024/oral/35537":
                return b"<html><body>Oral shell</body></html>", "text/html", url
            if url == "https://proceedings.mlr.press/v235/":
                return pmlr_index.encode("utf-8"), "text/html", url
            if url == pmlr_html_url:
                return pmlr_page.encode("utf-8"), "text/html", url
            self.assertEqual(url, pmlr_pdf)
            return b"%PDF-1.4\n(./figure/no_text.pdf)\n%%EOF", "application/pdf", url

        with patch("research_alpha.full_text.fetch_url_bytes", side_effect=fake_fetch):
            text, source = fetch_full_text_with_metadata(
                "https://icml.cc/virtual/2024/oral/35537",
                title="VideoPoet: A Large Language Model for Zero-Shot Video Generation",
                venue="ICML 2024 Best Paper",
                year=2024,
            )
        self.assertIn("Method brief", text)
        self.assertEqual(source, pmlr_html_url)

    def test_discover_pdf_from_paper_page_finds_pmlr_download_link(self) -> None:
        html = '<html><body><a href="https://raw.githubusercontent.com/mlresearch/v235/main/assets/x/x.pdf">Download PDF</a></body></html>'

        def fake_fetch(url, *, timeout, max_bytes):
            self.assertEqual(url, "https://proceedings.mlr.press/v235/x.html")
            return html.encode("utf-8"), "text/html", url

        with patch("research_alpha.full_text.fetch_url_bytes", side_effect=fake_fetch):
            self.assertEqual(
                discover_pdf_from_paper_page("https://proceedings.mlr.press/v235/x.html"),
                "https://raw.githubusercontent.com/mlresearch/v235/main/assets/x/x.pdf",
            )

    def test_direct_pdf_url_for_known_paper_pages(self) -> None:
        self.assertEqual(
            direct_pdf_url_for_known_paper_page(
                "https://papers.nips.cc/paper_files/paper/2024/hash/abc-Abstract-Conference.html"
            ),
            "https://papers.nips.cc/paper_files/paper/2024/hash/abc-Paper-Conference.pdf",
        )
        self.assertEqual(
            direct_pdf_url_for_known_paper_page(
                "https://openaccess.thecvf.com/content/CVPR2025/html/Wang_VGGT_Visual_Geometry_Grounded_Transformer_CVPR_2025_paper.html"
            ),
            "https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_VGGT_Visual_Geometry_Grounded_Transformer_CVPR_2025_paper.pdf",
        )

    def test_arxiv_pdf_fallback_url_only_applies_to_arxiv_html(self) -> None:
        self.assertEqual(
            arxiv_pdf_fallback_url("https://arxiv.org/html/2601.01234v2"),
            "https://arxiv.org/pdf/2601.01234v2.pdf",
        )
        self.assertEqual(arxiv_pdf_fallback_url("https://openreview.net/pdf?id=x"), "")

    def test_extract_section_evidence_from_headings(self) -> None:
        text = """
        Introduction This paper studies hidden failures in research agents and explains the old benchmark belief.
        Experiments We construct stress tasks, compare ranking shifts, and report ablations.
        Limitations The study is limited by small model coverage and future work should test more domains.
        Conclusion The proposed route changes how progress claims are interpreted.
        """
        payload = extract_section_evidence(text)
        self.assertIn("introduction", payload["available_sections"])
        self.assertIn("experiments", payload["available_sections"])
        self.assertIn("limitations", payload["available_sections"])
        self.assertIn("conclusion", payload["available_sections"])
        self.assertIn("hidden failures", payload["sections"]["introduction"])
        self.assertIn("small model coverage", payload["sections"]["limitations"])

    def test_genome_prompt_prefers_full_text_sections(self) -> None:
        full_text_json = full_text_json_for_text(
            "Introduction Full paper problem setup. "
            "Related Work Prior boundary. "
            "Methods Core reframing. "
            "Experiments Evidence narrative. "
            "Limitations Failure boundary."
        )
        prompt = build_genome_prompt(
            {
                "title": "Full Evidence Paper",
                "venue": "ICML",
                "year": 2026,
                "award": "best_paper",
                "abstract": "Short abstract.",
                "full_text_json": full_text_json,
            }
        )
        self.assertIn('"has_full_text_sections": true', prompt)
        self.assertIn('"evidence_level": "full_text_sections"', prompt)
        self.assertIn("Full paper problem setup", prompt)

    def test_genome_prompt_does_not_use_stale_noisy_full_text_sections(self) -> None:
        noisy_json = json.dumps(
            {
                "source_scope": "full_text_sections",
                "available_sections": ["method"],
                "sections": {
                    "method": "Method /Type /ObjStm /Filter /FlateDecode /Length 3536 >> stream x9c334T endstream endobj"
                },
            }
        )
        prompt = build_genome_prompt(
            {
                "title": "Noisy Cached Paper",
                "venue": "ICLR",
                "year": 2026,
                "award": "best_paper",
                "abstract": "Usable abstract.",
                "full_text_json": noisy_json,
            }
        )
        self.assertIn('"has_full_text_sections": false', prompt)
        self.assertIn('"evidence_level": "abstract_only"', prompt)
        self.assertNotIn("FlateDecode", prompt)

    def test_full_text_json_for_text_is_serializable(self) -> None:
        payload = json.loads(full_text_json_for_text("Introduction Text. Conclusion Done."))
        self.assertEqual(payload["source_scope"], "full_text_sections")
        self.assertGreaterEqual(payload["section_count"], 1)
        self.assertIn("quality", payload)

    def test_sanitize_full_text_payload_removes_stale_pdf_object_sections(self) -> None:
        payload = {
            "source_scope": "full_text_sections",
            "full_text_chars": 50000,
            "available_sections": ["method", "experiments"],
            "sections": {
                "method": "Method /Type /ObjStm /Filter /FlateDecode /Length 3536 >> stream x9c334T endstream endobj",
                "experiments": "Experiments compare robust local evidence against clear baselines.",
            },
        }
        sanitized = sanitize_full_text_payload(payload)
        self.assertEqual(sanitized["available_sections"], ["experiments"])
        self.assertNotIn("method", sanitized["sections"])
        self.assertEqual(sanitized["sanitization"]["reason"], "unreadable_or_pdf_structural_text")

    def test_sanitize_full_text_payload_removes_pdf_artifact_sections(self) -> None:
        artifact_text = (
            "Experiments cite.alpha2020 cite.beta2019 figure.10 subsection.3.1 appendix.A "
            "matplotlib pdf backend TeX Live pdfTeX hyperref cite.gamma2021 cite.delta2018 "
            "figure.21 cite.epsilon2017 cite.zeta2016 cite.eta2015 cite.theta2014"
        )
        payload = {
            "source_scope": "full_text_sections",
            "full_text_chars": 90000,
            "available_sections": ["experiments"],
            "sections": {"experiments": artifact_text},
        }
        sanitized = sanitize_full_text_payload(payload)
        self.assertEqual(sanitized["available_sections"], [])
        self.assertEqual(sanitized["sections"], {})

    def test_full_text_quality_distinguishes_metadata_page_sections(self) -> None:
        self.assertEqual(
            infer_full_text_quality("Method brief event page. Experiments short.", ["method", "experiments"]),
            "metadata_page_sections",
        )
        long_text = "Introduction problem setup. " + ("Detailed evidence sentence. " * 260)
        self.assertEqual(
            infer_full_text_quality(long_text, ["introduction", "method", "experiments", "limitations"]),
            "paper_like_sections",
        )

    def test_pdf_text_extraction_uses_pypdf_when_available(self) -> None:
        try:
            from pypdf import PdfWriter
            from pypdf.generic import DecodedStreamObject, NameObject
        except Exception as exc:  # pragma: no cover - dependency guard
            self.skipTest(f"pypdf unavailable: {exc}")
        writer = PdfWriter()
        page = writer.add_blank_page(width=300, height=120)
        stream = DecodedStreamObject()
        stream.set_data(b"BT /F1 12 Tf 30 80 Td (Introduction PDF evidence text) Tj ET")
        page[NameObject("/Contents")] = stream
        buffer = tempfile.SpooledTemporaryFile()
        writer.write(buffer)
        buffer.seek(0)
        text = extract_text_from_pdf_bytes(buffer.read())
        self.assertIn("Introduction PDF evidence text", text)

    def test_pdf_literal_fallback_rejects_binary_noise(self) -> None:
        noisy_pdf = b"%PDF-1.5\n" + b"(\x80\x81\x82\xff\x00\x01abcdefghi)\n" * 30 + b"%%EOF"
        self.assertEqual(extract_text_from_pdf_literal_chunks(noisy_pdf), "")

    def test_pdf_literal_fallback_rejects_pdf_object_stream_text(self) -> None:
        structural_pdf = (
            b"%PDF-1.5\n"
            b"(Method /Type /ObjStm /Filter /FlateDecode /Length 3536 >> stream x9c334T endstream endobj)\n"
            b"(Experiments /Type /ObjStm /Filter /FlateDecode /Length 64 >> stream x9c334T endstream endobj)\n"
            b"%%EOF"
        )
        self.assertEqual(extract_text_from_pdf_literal_chunks(structural_pdf), "")

    def test_cmd_fulltext_updates_paper_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                db_path = cwd / "data" / "research_alpha.db"
                init_db(db_path)
                paper_id = add_paper(
                    db_path,
                    "Full Text Paper",
                    "ICML",
                    2026,
                    abstract="",
                    external_ref="https://example.org/paper.html",
                    award="best_paper",
                )

                def fake_fetch(url, **kwargs):
                    self.assertEqual(url, "https://example.org/paper.html")
                    return (
                        "Introduction Full text logic sets up the problem and prior assumption. "
                        "Related Work Earlier systems optimize local benchmarks without route-level evidence. "
                        "Method The paper reframes the task around structured evidence and decision boundaries. "
                        "Experiments Stress tasks and ablations validate the route. "
                        "Limitations Needs more domains. Conclusion The story is clear.",
                        url,
                    )

                with patch("research_alpha.app.fetch_full_text_with_metadata", side_effect=fake_fetch):
                    exit_code = main(["fulltext", "--paper-id", str(paper_id)])
                self.assertEqual(exit_code, 0)
                with connect(db_path) as conn:
                    row = conn.execute("SELECT full_text_json FROM papers WHERE id=?", (paper_id,)).fetchone()
                payload = json.loads(row["full_text_json"])
                self.assertIn("introduction", payload["available_sections"])
                self.assertIn("source_url", payload)
            finally:
                os.chdir(old_cwd)

    def test_cmd_fulltext_existing_paper_like_payload_is_successful_noop(self) -> None:
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
                    "Already Full Text Paper",
                    "ICML",
                    2026,
                    abstract="",
                    external_ref="https://example.org/already.html",
                    source_kind="gold_openreview",
                    award="best_paper",
                )
                payload = full_text_json_for_text(
                    "Introduction Existing full text. Related Work Previous systems. "
                    + ("Method Strong evidence route. " * 100)
                    + "Experiments Robust validation. "
                    + ("Evaluation confirms stable behavior. " * 100)
                    + "Limitations Some boundary. Conclusion Done."
                )
                with connect(db_path) as conn:
                    conn.execute("UPDATE papers SET full_text_json=? WHERE id=?", (payload, paper_id))
                with patch("research_alpha.app.fetch_full_text_with_metadata", side_effect=AssertionError("should not refetch")):
                    self.assertEqual(main(["fulltext", "--paper-id", str(paper_id)]), 0)
            finally:
                os.chdir(old_cwd)

    def test_cmd_fulltext_can_process_user_library_without_scoring_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                db_path = cwd / "data" / "research_alpha.db"
                init_db(db_path)
                gold_id = add_paper(
                    db_path,
                    "Gold Full Text",
                    "ICLR",
                    2026,
                    abstract="Gold abstract.",
                    source_kind="gold_openreview",
                    external_ref="https://example.org/gold.html",
                    award="best_paper",
                )
                user_id = upsert_user_library_paper(
                    db_path,
                    title="User Library Full Text",
                    venue="arXiv",
                    year=2026,
                    abstract="User abstract.",
                    external_ref="https://arxiv.org/abs/2601.09999",
                )
                main(["score"])

                def fake_fetch(url, **kwargs):
                    self.assertEqual(url, "https://arxiv.org/abs/2601.09999")
                    return (
                        "Introduction User domain setup describes the local route and terminology. "
                        "Related Work User library papers provide domain background only. "
                        "Method Domain mechanisms are summarized for memory, not scoring. "
                        "Experiments User examples clarify evaluation context. "
                        "Limitations User boundary remains outside Gold evidence. "
                        "Conclusion The user library informs the field context.",
                        url,
                    )

                with patch("research_alpha.app.fetch_full_text_with_metadata", side_effect=fake_fetch):
                    exit_code = main(["fulltext", "--user-library", "--limit", "5"])
                self.assertEqual(exit_code, 0)
                with connect(db_path) as conn:
                    gold = conn.execute(
                        "SELECT full_text_json FROM papers WHERE id=?",
                        (gold_id,),
                    ).fetchone()
                    user = conn.execute(
                        "SELECT source_kind, paper_weight, full_text_json FROM papers WHERE id=?",
                        (user_id,),
                    ).fetchone()
                self.assertEqual(gold["full_text_json"], "{}")
                self.assertEqual(user["source_kind"], "user_library")
                self.assertEqual(float(user["paper_weight"]), 0.0)
                self.assertIn("User domain setup", user["full_text_json"])
            finally:
                os.chdir(old_cwd)

    def test_fetch_full_text_for_rows_degrades_without_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            db_path = cwd / "data" / "research_alpha.db"
            init_db(db_path)
            ok_id = add_paper(
                db_path,
                "Full Text OK",
                "ICLR",
                2026,
                external_ref="https://example.org/ok.html",
                award="best_paper",
            )
            missing_id = add_paper(
                db_path,
                "Full Text Missing Ref",
                "ICLR",
                2026,
                award="best_paper",
            )
            with connect(db_path) as conn:
                rows = conn.execute("SELECT * FROM papers ORDER BY id ASC").fetchall()

            def fake_fetch(url, **kwargs):
                return (
                    "Introduction Good local evidence frames the problem. "
                    "Related Work Local prior work defines the boundary. "
                    "Method The paper changes the evidence route. "
                    "Experiments Validation checks the central claim. "
                    "Limitations Boundary is explicit. Conclusion Done.",
                    url,
                )

            with patch("research_alpha.app.fetch_full_text_with_metadata", side_effect=fake_fetch):
                summary = fetch_full_text_for_paper_rows(db_path, rows)
            self.assertEqual(summary["updated"], 1)
            self.assertEqual(summary["skipped"], 1)
            with connect(db_path) as conn:
                ok_row = conn.execute("SELECT full_text_json FROM papers WHERE id=?", (ok_id,)).fetchone()
                missing_row = conn.execute("SELECT full_text_json FROM papers WHERE id=?", (missing_id,)).fetchone()
            self.assertIn("Good local evidence", ok_row["full_text_json"])
            self.assertEqual(missing_row["full_text_json"], "{}")

    def test_genome_build_can_fetch_full_text_before_extractive_card(self) -> None:
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
                    "Auto Full Text Gold",
                    "ICLR",
                    2026,
                    abstract="Short abstract.",
                    source_kind="gold_openreview",
                    external_ref="https://example.org/auto.html",
                    award="best_paper",
                )
                main(["score"])

                def fake_fetch(url, **kwargs):
                    return (
                        "Introduction Full evidence setup. "
                        "Related Work Earlier tools lack evidence grounding. "
                        "Methods Construct validation reframing. "
                        + ("Evidence grounding protocol aligns paper sections with logic moves. " * 80)
                        + "Experiments Ranking shifts. "
                        + ("Evaluation checks whether generated logic survives reviewer attack. " * 80)
                        + "Limitations Black-box agents. "
                        "Conclusion The route is usable.",
                        url,
                    )

                with patch("research_alpha.app.fetch_full_text_with_metadata", side_effect=fake_fetch):
                    outputs = run_pipeline_genome_build(
                        cwd,
                        limit=1,
                        provider=None,
                        model="",
                        include_existing=False,
                        extractive=True,
                        fetch_full_text_before=True,
                    )
                self.assertEqual(outputs[0][0], paper_id)
                payload = json.loads((cwd / "outputs" / "genomes" / f"paper-{paper_id}.json").read_text(encoding="utf-8"))
                self.assertEqual(payload["evidence_level"], EXTRACTIVE_FULL_TEXT_EVIDENCE_LEVEL)
                self.assertIn("experiments", payload["full_text_sections_used"])
            finally:
                os.chdir(old_cwd)

    def test_genome_build_refresh_stale_rebuilds_shallow_cards_after_full_text_arrives(self) -> None:
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
                    "Stale Genome Gold",
                    "ICLR",
                    2026,
                    abstract="Short abstract before full text.",
                    source_kind="gold_openreview",
                    award="best_paper",
                )
                main(["score"])
                self.assertEqual(main(["gb", "--limit", "1", "--extractive"]), 0)
                with connect(db_path) as conn:
                    row = conn.execute("SELECT evidence_level FROM idea_cards WHERE paper_id=?", (paper_id,)).fetchone()
                self.assertEqual(row["evidence_level"], "extractive_abstract_only")

                full_text_json = full_text_json_for_text(
                    "Introduction Full setup. Related Work Prior boundary. "
                    + ("Methods define the reframing mechanism. " * 80)
                    + "Experiments validate the reframing. "
                    + ("Validation compares clear baselines. " * 80)
                    + "Limitations name failure conditions. Conclusion done."
                )
                with connect(db_path) as conn:
                    conn.execute("UPDATE papers SET full_text_json=? WHERE id=?", (full_text_json, paper_id))
                ev_payload, _, _ = build_evidence_report(cwd, paper_limit=5)
                self.assertEqual(ev_payload["counts"]["stale_genome_cards"], 1)
                self.assertTrue(ev_payload["genome_cards"][0]["needs_full_text_refresh"])

                self.assertEqual(main(["gb", "--refresh-stale", "--extractive", "--limit", "1"]), 0)
                with connect(db_path) as conn:
                    row = conn.execute("SELECT evidence_level, content_json FROM idea_cards WHERE paper_id=?", (paper_id,)).fetchone()
                self.assertEqual(row["evidence_level"], EXTRACTIVE_FULL_TEXT_EVIDENCE_LEVEL)
                payload = json.loads(row["content_json"])
                self.assertIn("experiments", payload["full_text_sections_used"])
                refreshed, _, _ = build_evidence_report(cwd, paper_limit=5)
                self.assertEqual(refreshed["counts"]["stale_genome_cards"], 0)
            finally:
                os.chdir(old_cwd)

    def test_extractive_stale_refresh_does_not_downgrade_strict_genome_card(self) -> None:
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
                    "Strict Stale Gold",
                    "ICLR",
                    2026,
                    abstract="Short abstract.",
                    source_kind="gold_openreview",
                    award="best_paper",
                )
                main(["score"])
                with connect(db_path) as conn:
                    conn.execute(
                        "INSERT INTO idea_cards(paper_id, evidence_level, content_json) VALUES (?, ?, ?)",
                        (
                            paper_id,
                            "full_text_sections",
                            json.dumps({"paper_summary": "strict card should not be downgraded"}),
                        ),
                    )
                shallow_full_text = full_text_json_for_text(
                    "Introduction Short setup. Method Short method. Experiments Short validation."
                )
                with connect(db_path) as conn:
                    conn.execute("UPDATE papers SET full_text_json=? WHERE id=?", (shallow_full_text, paper_id))

                self.assertEqual(main(["gb", "--refresh-stale", "--extractive", "--limit", "1"]), 0)
                with connect(db_path) as conn:
                    row = conn.execute("SELECT evidence_level, content_json FROM idea_cards WHERE paper_id=?", (paper_id,)).fetchone()
                self.assertEqual(row["evidence_level"], "full_text_sections")
                self.assertIn("strict card should not be downgraded", row["content_json"])
            finally:
                os.chdir(old_cwd)

    def test_extractive_stale_refresh_downgrades_unsupported_full_text_claim(self) -> None:
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
                    "Unsupported Full Text Claim Gold",
                    "CVPR",
                    2025,
                    abstract="Abstract evidence remains usable for metadata-level logic.",
                    source_kind="gold_cvf_awards",
                    award="outstanding_paper",
                )
                main(["score"])
                shallow_full_text = full_text_json_for_text("Method short metadata page only.")
                with connect(db_path) as conn:
                    conn.execute("UPDATE papers SET full_text_json=? WHERE id=?", (shallow_full_text, paper_id))
                    conn.execute(
                        "INSERT INTO idea_cards(paper_id, evidence_level, content_json) VALUES (?, ?, ?)",
                        (
                            paper_id,
                            EXTRACTIVE_FULL_TEXT_EVIDENCE_LEVEL,
                            json.dumps({"paper_summary": "stale full text claim", "evidence_level": EXTRACTIVE_FULL_TEXT_EVIDENCE_LEVEL}),
                        ),
                    )

                ev_payload, _, _ = build_evidence_report(cwd, paper_limit=5)
                self.assertEqual(ev_payload["counts"]["stale_genome_cards"], 1)
                self.assertEqual(main(["gb", "--refresh-stale", "--extractive", "--limit", "1"]), 0)
                with connect(db_path) as conn:
                    row = conn.execute("SELECT evidence_level, content_json FROM idea_cards WHERE paper_id=?", (paper_id,)).fetchone()
                self.assertEqual(row["evidence_level"], "extractive_abstract_only")
                payload = json.loads(row["content_json"])
                self.assertEqual(payload["evidence_level"], "extractive_abstract_only")
                self.assertEqual(payload["full_text_sections_used"], [])
            finally:
                os.chdir(old_cwd)

    def test_genome_build_refresh_stale_dry_run_does_not_write_or_call_llm(self) -> None:
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
                    "Dry Run Stale Gold",
                    "ICLR",
                    2026,
                    abstract="Short abstract.",
                    source_kind="gold_openreview",
                    award="best_paper",
                )
                main(["score"])
                with connect(db_path) as conn:
                    conn.execute(
                        "INSERT INTO idea_cards(paper_id, evidence_level, content_json) VALUES (?, ?, ?)",
                        (paper_id, "extractive_abstract_only", json.dumps({"paper_summary": "old"})),
                    )
                    conn.execute(
                        "UPDATE papers SET full_text_json=? WHERE id=?",
                        (
                            full_text_json_for_text(
                                "Introduction Deep setup. Related Work Strong boundary. "
                                + ("Methods define the route. Experiments validate it. " * 120)
                                + "Limitations name failures. Conclusion closes."
                            ),
                            paper_id,
                        ),
                    )
                output = io.StringIO()
                with patch("research_alpha.app.generate_genome_for_paper", side_effect=AssertionError("LLM should not be called")):
                    with redirect_stdout(output):
                        self.assertEqual(main(["gb", "--refresh-stale", "--dry-run", "--limit", "1"]), 0)
                text = output.getvalue()
                self.assertIn("Genome build dry-run", text)
                self.assertIn("LLM calls: yes", text)
                self.assertIn(f"paper #{paper_id}", text)
                with connect(db_path) as conn:
                    row = conn.execute("SELECT evidence_level, content_json FROM idea_cards WHERE paper_id=?", (paper_id,)).fetchone()
                self.assertEqual(row["evidence_level"], "extractive_abstract_only")
                self.assertIn("old", row["content_json"])
            finally:
                os.chdir(old_cwd)

    def test_pattern_report_marks_dependency_on_stale_genome_card(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                db_path = cwd / "data" / "research_alpha.db"
                paper_ids = []
                for idx in range(2):
                    paper_ids.append(
                        add_paper(
                            db_path,
                            f"Pattern Stale Source {idx}",
                            "ICLR",
                            2026,
                            abstract="Abstract evidence.",
                            source_kind="gold_openreview",
                            award="best_paper" if idx == 0 else "oral",
                        )
                    )
                main(["score"])
                # No trusted full text exists, so this full-text claim is stale.
                with connect(db_path) as conn:
                    conn.execute(
                        "INSERT INTO idea_cards(paper_id, evidence_level, content_json) VALUES (?, ?, ?)",
                        (
                            paper_ids[0],
                            EXTRACTIVE_FULL_TEXT_EVIDENCE_LEVEL,
                            json.dumps({"paper_summary": "stale", "logic_line": {"old_belief": "x", "bottleneck": "x", "reframing": "x", "why_now": "x", "evidence_design": "x", "failure_boundary": "x"}}),
                        ),
                    )
                    conn.execute(
                        "INSERT INTO idea_cards(paper_id, evidence_level, content_json) VALUES (?, ?, ?)",
                        (
                            paper_ids[1],
                            "extractive_abstract_only",
                            json.dumps({"paper_summary": "fresh enough"}),
                        ),
                    )
                    conn.execute(
                        "INSERT INTO pattern_cards(pattern_key, content_json) VALUES (?, ?)",
                        (
                            "stale-dependency",
                            json.dumps(
                                {
                                    "pattern_key": "stale-dependency",
                                    "pattern_name": "Stale Dependency",
                                    "evidence_level": "extractive_abstract_only_aggregation",
                                    "source_set_policy": "gold_set_only",
                                    "source_paper_ids": paper_ids,
                                    "source_papers": [
                                        {"paper_id": paper_ids[0], "paper_weight": 5.0},
                                        {"paper_id": paper_ids[1], "paper_weight": 2.0},
                                    ],
                                }
                            ),
                        ),
                    )
                payload, _, _ = build_evidence_report(cwd, paper_limit=5)
                self.assertEqual(payload["counts"]["stale_genome_cards"], 1)
                self.assertEqual(payload["counts"]["stale_pattern_cards"], 1)
                pattern = next(item for item in payload["pattern_cards"] if item["pattern_key"] == "stale-dependency")
                self.assertTrue(pattern["needs_genome_refresh"])

                self.assertEqual(main(["gb", "--refresh-stale", "--extractive", "--limit", "5"]), 0)
                refreshed, _, _ = build_evidence_report(cwd, paper_limit=5)
                self.assertEqual(refreshed["counts"]["stale_genome_cards"], 0)
                self.assertEqual(refreshed["counts"]["stale_pattern_cards"], 0)
            finally:
                os.chdir(old_cwd)

    def test_evidence_report_exposes_full_text_depth(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                db_path = cwd / "data" / "research_alpha.db"
                full_text_json = full_text_json_for_text(
                    "Introduction Deep evidence. Related Work Strong boundary. "
                    + ("Methods and Experiments give strong validation. " * 180)
                    + "Limitations name the failure boundary."
                )
                for idx, award in enumerate(["best_paper", "oral", "outstanding_paper"], start=1):
                    add_paper(
                        db_path,
                        f"Gold Evidence Depth {idx}",
                        "ICLR",
                        2026,
                        abstract="Abstract evidence.",
                        source_kind="gold_openreview",
                        award=award,
                    )
                with connect(db_path) as conn:
                    conn.execute("UPDATE papers SET full_text_json=? WHERE title=?", (full_text_json, "Gold Evidence Depth 1"))
                main(["score"])
                payload, _, markdown_path = build_evidence_report(cwd, paper_limit=5)
                self.assertEqual(payload["counts"]["full_text_ready_papers"], 1)
                self.assertEqual(payload["counts"]["paper_like_full_text_papers"], 1)
                self.assertEqual(payload["counts"]["metadata_page_section_papers"], 0)
                self.assertEqual(payload["counts"]["abstract_only_papers"], 2)
                first = next(item for item in payload["high_weight_papers"] if item["title"] == "Gold Evidence Depth 1")
                self.assertEqual(first["full_text_status"]["status"], "full_text_sections")
                self.assertIn("paper_like_full_text", first["full_text_status"]["quality"])
                self.assertIn("full_text=full_text_sections:paper_like_full_text", markdown_path.read_text(encoding="utf-8"))
            finally:
                os.chdir(old_cwd)

    def test_evidence_report_sanitizes_stale_noisy_full_text_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(cwd)
                main(["init"])
                db_path = cwd / "data" / "research_alpha.db"
                noisy_payload = json.dumps(
                    {
                        "source_scope": "full_text_sections",
                        "quality": "paper_like_full_text",
                        "full_text_chars": 50000,
                        "available_sections": ["method", "experiments", "limitations"],
                        "sections": {
                            "method": "/Type /ObjStm /Filter /FlateDecode /Length 3536 >> stream x9c endstream endobj",
                            "experiments": "/Type /ObjStm /Filter /FlateDecode /Length 64 >> stream x9c endstream endobj",
                            "limitations": "/Type /ObjStm /Filter /FlateDecode /Length 88 >> stream x9c endstream endobj",
                        },
                    }
                )
                add_paper(
                    db_path,
                    "Noisy Cached Gold",
                    "ICLR",
                    2026,
                    abstract="Abstract evidence remains.",
                    source_kind="gold_openreview",
                    award="best_paper",
                )
                with connect(db_path) as conn:
                    conn.execute("UPDATE papers SET full_text_json=? WHERE title='Noisy Cached Gold'", (noisy_payload,))
                main(["score"])
                payload, _, _ = build_evidence_report(cwd, paper_limit=5)
                paper = next(item for item in payload["high_weight_papers"] if item["title"] == "Noisy Cached Gold")
                self.assertEqual(paper["full_text_status"]["status"], "abstract_or_metadata_only")
                self.assertEqual(paper["full_text_status"]["quality"], "no_sections")
                self.assertEqual(paper["full_text_status"]["available_sections"], [])
            finally:
                os.chdir(old_cwd)

    def test_extractive_genome_uses_full_text_sections_when_available(self) -> None:
        full_text_json = full_text_json_for_text(
            "Introduction Hidden setup. Methods Reframe the construct. "
            + ("Full-text evidence maps the paper's setup into a grounded logic line. " * 80)
            + "Experiments Validate ranking shifts. "
            + ("Evaluation checks ranking shifts and reviewer-facing failure boundaries. " * 80)
            + "Limitations Black-box agents fail."
        )
        payload = build_extractive_genome_payload(
            {
                "title": "Full Text Gold",
                "venue": "ICLR",
                "year": 2026,
                "award": "best_paper",
                "abstract": "Short abstract.",
                "paper_weight": 5,
                "full_text_json": full_text_json,
            }
        )
        self.assertEqual(payload["evidence_level"], EXTRACTIVE_FULL_TEXT_EVIDENCE_LEVEL)
        self.assertIn("full-text section evidence", payload["paper_summary"])
        self.assertIn("experiments", payload["full_text_sections_used"])


if __name__ == "__main__":
    unittest.main()
