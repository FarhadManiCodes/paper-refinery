"""Tests for the CLI: the chunks-manifest writer and the main orchestration wiring."""

import json
import logging

from click.testing import CliRunner

from paper_refinery import cli
from paper_refinery.chunker import Chunk
from paper_refinery.cli import _relativize_image_links, write_chunks
from paper_refinery.config import RefineryConfig
from paper_refinery.parse import ParseResult


def test_relativize_image_links_rewrites_absolute_paths(tmp_path):
    md = f"![FIGURE_CROP 1:0]({tmp_path}/figures/page_1_fig_0.png)\n\ntext"
    out = _relativize_image_links(md, tmp_path)
    assert "](figures/page_1_fig_0.png)" in out
    assert str(tmp_path) not in out


def test_relativize_image_links_leaves_relative_paths_alone():
    md = "![FIGURE 4.1](figures/fig_4.1.png)"
    assert _relativize_image_links(md, "/anything") == md


def test_relativize_image_links_handles_multiple_links(tmp_path):
    md = f"![a]({tmp_path}/figures/a.png) text ![b]({tmp_path}/other/b.png)"
    out = _relativize_image_links(md, tmp_path)
    assert "](figures/a.png)" in out
    assert "](other/b.png)" in out


def test_write_chunks_roundtrip(tmp_path):
    chunks = [Chunk("hello", 0, 1, 1, 0, "-"), Chunk("world", 1, 1, 2, 250, "SENT")]
    out = tmp_path / "p.chunks.json"
    write_chunks(chunks, "mydoc", "/x/p.pdf", out)

    data = json.loads(out.read_text())
    assert data["docname"] == "mydoc"
    assert data["source_pdf"] == "/x/p.pdf"
    assert data["parser"] == "paper-refinery"
    assert len(data["chunks"]) == 2
    assert data["chunks"][0]["text"] == "hello"
    assert data["chunks"][1]["overlap_mode"] == "SENT"
    assert data["chunks"][1]["page_end"] == 2


def test_main_wires_stages_and_writes_json(tmp_path, monkeypatch):
    pdf = tmp_path / "p.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    seen = {}
    monkeypatch.setattr(cli, "load_config", lambda: RefineryConfig())
    monkeypatch.setattr(
        cli, "parse_pdf_cached", lambda p, d, c, force=False: ParseResult(markdown="MD")
    )
    monkeypatch.setattr(cli, "make_client", lambda cfg: object())

    def fake_enrich(parsed, cfg, describe):
        seen["md"] = parsed.markdown  # parse output reached enrich
        return "ENRICHED"

    monkeypatch.setattr(cli, "enrich_markdown", fake_enrich)
    monkeypatch.setattr(cli, "chunk_markdown", lambda md, cfg: [Chunk(md, 0, 1, 1)])

    result = CliRunner().invoke(cli.main, [str(pdf)])
    assert result.exit_code == 0, result.output

    data = json.loads(pdf.with_suffix(".chunks.json").read_text())
    assert seen["md"] == "MD"
    assert data["chunks"][0]["text"] == "ENRICHED"  # enrich output reached chunk -> json
    assert data["docname"] == "p"
    # the enriched markdown is kept as a work-dir artifact before chunking
    work_dir = pdf.with_suffix(".refinery")
    assert (work_dir / "refinery.md").read_text() == "ENRICHED"
    # no references from parse_pdf -> no references.md, no citations stage
    assert not (work_dir / "references.md").exists()
    assert not pdf.with_suffix(".citations.json").exists()
    assert "references ->" not in result.output


def test_main_defaults_work_dir_next_to_pdf(tmp_path, monkeypatch):
    pdf = tmp_path / "p.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    seen = {}

    def fake_parse_pdf(p, d, c, force=False):
        seen["image_dir"] = d
        return ParseResult(markdown="MD")

    monkeypatch.setattr(cli, "load_config", lambda: RefineryConfig())
    monkeypatch.setattr(cli, "parse_pdf_cached", fake_parse_pdf)
    monkeypatch.setattr(cli, "make_client", lambda cfg: object())
    monkeypatch.setattr(cli, "enrich_markdown", lambda parsed, cfg, describe: parsed.markdown)
    monkeypatch.setattr(cli, "chunk_markdown", lambda md, cfg: [Chunk(md, 0, 1, 1)])

    result = CliRunner().invoke(cli.main, [str(pdf)])
    assert result.exit_code == 0, result.output
    # figure crops land inside the per-paper work dir -- two PDFs in one folder must
    # never share (and overwrite) a common figures/
    assert seen["image_dir"] == pdf.with_suffix(".refinery")
    assert seen["image_dir"].is_dir()  # created before parse_pdf needs it


def test_main_needs_no_gemini_client_for_figureless_paper(tmp_path, monkeypatch):
    # a paper with zero figure crops and zero references must not require
    # GOOGLE_API_KEY at all
    pdf = tmp_path / "p.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    def boom(*a, **kw):
        raise AssertionError("no Gemini-dependent stage may run without figures/references")

    monkeypatch.setattr(cli, "load_config", lambda: RefineryConfig())
    monkeypatch.setattr(
        cli, "parse_pdf_cached", lambda p, d, c, force=False: ParseResult(markdown="MD")
    )
    monkeypatch.setattr(cli, "make_client", boom)
    monkeypatch.setattr(cli, "extract_references", boom)
    monkeypatch.setattr(cli, "enrich_markdown", lambda parsed, cfg, describe: parsed.markdown)
    monkeypatch.setattr(cli, "chunk_markdown", lambda md, cfg: [Chunk(md, 0, 1, 1)])

    result = CliRunner().invoke(cli.main, [str(pdf)])
    assert result.exit_code == 0, result.output


def test_main_writes_references_markdown_and_keeps_it_out_of_chunking(tmp_path, monkeypatch):
    pdf = tmp_path / "p.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    refs_md = "<page_number>3</page_number>\n\n[1] Smith, J. (2020)."
    monkeypatch.setattr(cli, "load_config", lambda: RefineryConfig())
    monkeypatch.setattr(
        cli,
        "parse_pdf_cached",
        lambda p, d, c, force=False: ParseResult(markdown="MD", references_markdown=refs_md),
    )
    monkeypatch.setattr(cli, "make_client", lambda cfg: object())
    monkeypatch.setattr(cli, "enrich_markdown", lambda parsed, cfg, describe: parsed.markdown)

    seen_chunk_input = {}

    def fake_chunk(md, cfg):
        seen_chunk_input["md"] = md
        return [Chunk(md, 0, 1, 1)]

    monkeypatch.setattr(cli, "chunk_markdown", fake_chunk)

    result = CliRunner().invoke(cli.main, [str(pdf)])
    assert result.exit_code == 0, result.output

    refs_path = pdf.with_suffix(".refinery") / "references.md"
    assert refs_path.exists()
    assert refs_path.read_text() == refs_md
    assert f"references -> {refs_path}" in result.output
    # references never reach the chunker
    assert "Smith" not in seen_chunk_input["md"]


_REFS = [{"page": 3, "number": "1", "text": "[1] Smith, J. (2020). A title. J. 1:1-2."}]


def test_main_runs_citation_stack_and_writes_citations_json(tmp_path, monkeypatch):
    pdf = tmp_path / "p.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    parsed = ParseResult(
        markdown="Body cites Smith (2019) here.",
        references=_REFS,
        references_markdown="[1] Smith...",
    )
    monkeypatch.setattr(cli, "load_config", lambda: RefineryConfig())
    monkeypatch.setattr(cli, "parse_pdf_cached", lambda p, d, c, force=False: parsed)
    monkeypatch.setattr(cli, "make_client", lambda cfg: object())
    monkeypatch.setattr(cli, "enrich_markdown", lambda parsed, cfg, describe: parsed.markdown)
    monkeypatch.setattr(cli, "chunk_markdown", lambda md, cfg: [Chunk(md, 0, 1, 1)])

    extracted = [{"title": "A title", "year": 2019, "authors": [{"family": "Smith"}]}]
    # resolution legitimately moved the year forward (published version) -- the body
    # still prints 2019, so linking must run on the EXTRACTED view to find the marker
    resolved = [
        {**extracted[0], "year": 2020, "verified": True, "match": "crossref", "doi": "10.1/x"}
    ]
    monkeypatch.setattr(cli, "extract_references", lambda texts, cfg: extracted)
    monkeypatch.setattr(cli, "resolve_references", lambda ext, refs, cfg, source=None: resolved)
    monkeypatch.setattr(cli, "format_resolution_report", lambda ext, res: "REPORT")

    result = CliRunner().invoke(cli.main, [str(pdf)])
    assert result.exit_code == 0, result.output

    data = json.loads(pdf.with_suffix(".citations.json").read_text())
    assert data["docname"] == "p"
    assert data["references"][0]["verified"] and data["references"][0]["doi"] == "10.1/x"
    assert data["references"][0]["year"] == 2020  # JSON carries the resolved metadata...
    # ...while linking matched the printed form (would find nothing against year 2020)
    assert data["linking"]["style"] == "author-year"
    assert data["linking"]["markers"] == [{"text": "Smith (2019)", "refs": [0]}]
    assert data["linking"]["uncited"] == []
    # the human diff report lands in the work dir
    assert (pdf.with_suffix(".refinery") / "resolution_report.txt").read_text() == "REPORT\n"
    assert "citations: 1/1 verified" in result.output


def test_main_rewrites_markers_to_verified_citekeys_before_chunking(tmp_path, monkeypatch):
    # stage 3b: the printed marker becomes a papis-style citekey built from the
    # RESOLVED (verified) surname/year, not layer-1's guess -- and the rewrite must
    # land in both the reviewable refinery.md and the chunks that ship to papis-ask
    pdf = tmp_path / "p.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    parsed = ParseResult(
        markdown="Body cites Smith (2019) here.",
        references=_REFS,
        references_markdown="[1] Smith...",
    )
    monkeypatch.setattr(cli, "load_config", lambda: RefineryConfig())
    monkeypatch.setattr(cli, "parse_pdf_cached", lambda p, d, c, force=False: parsed)
    monkeypatch.setattr(cli, "make_client", lambda cfg: object())
    monkeypatch.setattr(cli, "enrich_markdown", lambda parsed, cfg, describe: parsed.markdown)

    seen_chunk_input = {}

    def fake_chunk(md, cfg):
        seen_chunk_input["md"] = md
        return [Chunk(md, 0, 1, 1)]

    monkeypatch.setattr(cli, "chunk_markdown", fake_chunk)

    extracted = [{"title": "A title", "year": 2019, "authors": [{"family": "Smith"}]}]
    resolved = [
        {**extracted[0], "year": 2020, "verified": True, "match": "crossref", "doi": "10.1/x"}
    ]
    monkeypatch.setattr(cli, "extract_references", lambda texts, cfg: extracted)
    monkeypatch.setattr(cli, "resolve_references", lambda ext, refs, cfg, source=None: resolved)
    monkeypatch.setattr(cli, "format_resolution_report", lambda ext, res: "REPORT")

    result = CliRunner().invoke(cli.main, [str(pdf)])
    assert result.exit_code == 0, result.output

    work_dir = pdf.with_suffix(".refinery")
    assert (work_dir / "refinery.md").read_text() == "Body cites [smith_2020] here."
    assert seen_chunk_input["md"] == "Body cites [smith_2020] here."


def test_main_leaves_marker_unrewritten_when_citekey_is_missing(tmp_path, monkeypatch):
    # an unresolved entry (no verified year) yields no citekey -- the marker is left
    # exactly as printed rather than guessed at
    pdf = tmp_path / "p.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    parsed = ParseResult(
        markdown="Body cites Smith (2019) here.",
        references=_REFS,
        references_markdown="[1] Smith...",
    )
    monkeypatch.setattr(cli, "load_config", lambda: RefineryConfig())
    monkeypatch.setattr(cli, "parse_pdf_cached", lambda p, d, c, force=False: parsed)
    monkeypatch.setattr(cli, "make_client", lambda cfg: object())
    monkeypatch.setattr(cli, "enrich_markdown", lambda parsed, cfg, describe: parsed.markdown)
    monkeypatch.setattr(cli, "chunk_markdown", lambda md, cfg: [Chunk(md, 0, 1, 1)])

    extracted = [{"title": "A title", "year": 2019, "authors": [{"family": "Smith"}]}]
    resolved = [{**extracted[0], "year": None, "verified": False}]
    monkeypatch.setattr(cli, "extract_references", lambda texts, cfg: extracted)
    monkeypatch.setattr(cli, "resolve_references", lambda ext, refs, cfg, source=None: resolved)
    monkeypatch.setattr(cli, "format_resolution_report", lambda ext, res: "REPORT")

    result = CliRunner().invoke(cli.main, [str(pdf)])
    assert result.exit_code == 0, result.output

    work_dir = pdf.with_suffix(".refinery")
    assert (work_dir / "refinery.md").read_text() == "Body cites Smith (2019) here."


def test_main_citation_failure_degrades_to_warning(tmp_path, monkeypatch, caplog):
    # the chunks manifest is the primary product: a citation-stage crash (missing
    # API key, providers down) must warn loudly but never fail the run
    pdf = tmp_path / "p.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    parsed = ParseResult(markdown="MD", references=_REFS, references_markdown="[1] Smith...")
    monkeypatch.setattr(cli, "load_config", lambda: RefineryConfig())
    monkeypatch.setattr(cli, "parse_pdf_cached", lambda p, d, c, force=False: parsed)
    monkeypatch.setattr(cli, "make_client", lambda cfg: object())
    monkeypatch.setattr(cli, "enrich_markdown", lambda parsed, cfg, describe: parsed.markdown)
    monkeypatch.setattr(cli, "chunk_markdown", lambda md, cfg: [Chunk(md, 0, 1, 1)])

    def boom(texts, cfg):
        raise RuntimeError("GOOGLE_API_KEY not set")

    monkeypatch.setattr(cli, "extract_references", boom)

    with caplog.at_level(logging.WARNING):
        result = CliRunner().invoke(cli.main, [str(pdf)])
    assert result.exit_code == 0, result.output
    assert pdf.with_suffix(".chunks.json").exists()
    assert not pdf.with_suffix(".citations.json").exists()
    assert "citation stage failed" in caplog.text


def test_force_parse_flag_controls_the_checkpoint_bypass(tmp_path, monkeypatch):
    pdf = tmp_path / "p.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    seen = {}

    def fake_parse(p, d, c, force=False):
        seen["force"] = force
        return ParseResult(markdown="MD")

    monkeypatch.setattr(cli, "load_config", lambda: RefineryConfig())
    monkeypatch.setattr(cli, "parse_pdf_cached", fake_parse)
    monkeypatch.setattr(cli, "make_client", lambda cfg: object())
    monkeypatch.setattr(cli, "enrich_markdown", lambda parsed, cfg, describe: parsed.markdown)
    monkeypatch.setattr(cli, "chunk_markdown", lambda md, cfg: [Chunk(md, 0, 1, 1)])

    CliRunner().invoke(cli.main, [str(pdf)])
    assert seen["force"] is False  # default reuses the parse checkpoint
    CliRunner().invoke(cli.main, [str(pdf), "--force-parse"])
    assert seen["force"] is True  # flag forces a re-OCR


def test_from_chunk_rechunks_refinery_md_without_running_upstream(tmp_path, monkeypatch):
    pdf = tmp_path / "p.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    work_dir = pdf.with_suffix(".refinery")
    work_dir.mkdir()
    (work_dir / "refinery.md").write_text("Enriched body [smith_2020].")

    def boom(*a, **k):
        raise AssertionError("--from chunk must not run any upstream stage")

    monkeypatch.setattr(cli, "load_config", lambda: RefineryConfig())
    monkeypatch.setattr(cli, "parse_pdf_cached", boom)
    monkeypatch.setattr(cli, "enrich_markdown", boom)
    monkeypatch.setattr(cli, "extract_references", boom)

    seen = {}

    def fake_chunk(md, cfg):
        seen["md"] = md
        return [Chunk(md, 0, 1, 1)]

    monkeypatch.setattr(cli, "chunk_markdown", fake_chunk)

    result = CliRunner().invoke(cli.main, [str(pdf), "--from", "chunk"])
    assert result.exit_code == 0, result.output
    assert seen["md"] == "Enriched body [smith_2020]."  # chunked straight from refinery.md
    data = json.loads(pdf.with_suffix(".chunks.json").read_text())
    assert data["chunks"][0]["text"] == "Enriched body [smith_2020]."
    assert data["docname"] == "p"
    assert "re-chunked" in result.output


def test_from_chunk_errors_when_refinery_md_missing(tmp_path, monkeypatch):
    pdf = tmp_path / "p.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    monkeypatch.setattr(cli, "load_config", lambda: RefineryConfig())

    result = CliRunner().invoke(cli.main, [str(pdf), "--from", "chunk"])
    assert result.exit_code != 0
    assert "refinery.md" in result.output


def test_refine_public_api_returns_result_and_writes_artifacts(tmp_path, monkeypatch):
    # the in-process entry point papis-ask will call: runs the pipeline, returns the
    # chunks + artifact paths, and writes the same files as the CLI
    pdf = tmp_path / "p.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    seen = {}

    def fake_parse(p, d, c, force=False):
        seen["force"] = force
        return ParseResult(markdown="MD")

    monkeypatch.setattr(cli, "parse_pdf_cached", fake_parse)
    monkeypatch.setattr(cli, "enrich_markdown", lambda parsed, cfg, describe: "ENRICHED")
    monkeypatch.setattr(cli, "chunk_markdown", lambda md, cfg: [Chunk(md, 0, 1, 1)])

    result = cli.refine(pdf, RefineryConfig())

    assert [c.text for c in result.chunks] == ["ENRICHED"]  # chunks returned in memory
    assert result.chunks_path == pdf.with_suffix(".chunks.json")  # defaults next to the PDF
    assert result.citations_path == pdf.with_suffix(".citations.json")
    assert result.work_dir == pdf.with_suffix(".refinery")
    data = json.loads(result.chunks_path.read_text())  # and the hand-off JSON is written
    assert data["chunks"][0]["text"] == "ENRICHED"
    assert data["docname"] == "p"
    assert seen["force"] is False

    cli.refine(pdf, RefineryConfig(), force_parse=True)
    assert seen["force"] is True  # force_parse threads through to the checkpoint


def test_refine_is_exported_at_package_top_level():
    import paper_refinery

    assert paper_refinery.refine is cli.refine
    assert paper_refinery.RefineResult is cli.RefineResult
