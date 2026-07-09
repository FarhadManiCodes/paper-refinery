"""Tests for the CLI: the chunks-manifest writer and the main orchestration wiring."""

import json
import logging
import threading

import pytest
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


def test_relativize_image_links_rewrites_cwd_relative_crop_paths():
    # the real bug: invoked with a RELATIVE pdf path (`refinery samples/x.pdf`), the work dir
    # and crop paths are relative-to-CWD -- the old absolute-only check skipped them, leaving
    # repo-root-relative links that break when refinery.md is opened from its own directory.
    # The link must end up relative to the .md's OWN location (figures/ sits beside it).
    md = "![FIGURE 1](samples/x.refinery/figures/fig_1.png)"
    assert _relativize_image_links(md, "samples/x.refinery") == "![FIGURE 1](figures/fig_1.png)"


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
    assert data["schema_version"] == cli.MANIFEST_SCHEMA_VERSION  # versioned cross-tool seam
    assert len(data["chunks"]) == 2
    assert data["chunks"][0]["text"] == "hello"
    assert data["chunks"][1]["overlap_mode"] == "SENT"
    assert data["chunks"][1]["page_end"] == 2


def test_main_many_refines_all_pdfs_and_reports(tmp_path, monkeypatch):
    pdfs = [tmp_path / "a.pdf", tmp_path / "b.pdf"]
    for p in pdfs:
        p.write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setattr(cli, "load_config", lambda: RefineryConfig())

    seen = {}

    def fake_refine_many(pdf_list, cfg, *, force_parse, workers, ocr_workers, sources):
        seen["call"] = ([p.name for p in pdf_list], force_parse, workers, ocr_workers, sources)
        for p in pdf_list:
            yield cli.RefineResult(
                [Chunk("x", 0, 1, 1)],
                p.with_suffix(".chunks.json"),
                p.with_suffix(".citations.json"),
                p.with_suffix(".refinery"),
            )

    monkeypatch.setattr(cli, "refine_many", fake_refine_many)

    result = CliRunner().invoke(cli.main_many, [str(pdfs[0]), str(pdfs[1]), "--ocr-workers", "1"])
    assert result.exit_code == 0, result.output
    assert "refined 2/2 papers" in result.output
    assert "a.chunks.json: 1 chunks" in result.output
    # CLI flags reach refine_many; no --meta-map -> sources is None
    assert seen["call"] == (["a.pdf", "b.pdf"], False, 4, 1, None)


def test_main_many_meta_map_feeds_sources_by_path(tmp_path, monkeypatch):
    # --meta-map aligns each PDF to its SourceMeta bundle by path; a PDF absent from the map
    # gets None (OCR-title fallback). Path-keyed, so order/count drift can't misattribute.
    pdfs = [tmp_path / "a.pdf", tmp_path / "b.pdf"]
    for p in pdfs:
        p.write_bytes(b"%PDF-1.4 fake")
    meta = tmp_path / "meta.json"
    bundle = {"doi": "10.1/a", "title": "A", "year": 2020}
    meta.write_text(json.dumps({str(pdfs[0].resolve()): bundle}))
    monkeypatch.setattr(cli, "load_config", lambda: RefineryConfig())

    captured = {}

    def fake_refine_many(pdf_list, cfg, *, force_parse, workers, ocr_workers, sources):
        captured["sources"] = sources
        for p in pdf_list:
            yield cli.RefineResult([Chunk("x", 0, 1, 1)], p.with_suffix(".chunks.json"), p, p)

    monkeypatch.setattr(cli, "refine_many", fake_refine_many)

    result = CliRunner().invoke(
        cli.main_many, [str(pdfs[0]), str(pdfs[1]), "--meta-map", str(meta)]
    )
    assert result.exit_code == 0, result.output
    assert captured["sources"] == [{"doi": "10.1/a", "title": "A", "year": 2020}, None]


def test_main_many_requires_at_least_one_pdf():
    result = CliRunner().invoke(cli.main_many, [])
    assert result.exit_code != 0  # nargs=-1 required -> click usage error


def test_main_many_from_chunk_rechunks_each_saved_md_without_refining(tmp_path, monkeypatch):
    # batch --from chunk re-chunks each paper's saved refinery.md, never touching OCR/network
    pdfs = [tmp_path / "a.pdf", tmp_path / "b.pdf"]
    for p in pdfs:
        p.write_bytes(b"%PDF-1.4 fake")
        wd = p.with_suffix(".refinery")
        wd.mkdir()
        (wd / "refinery.md").write_text(f"# {p.stem}\n\nbody")
    monkeypatch.setattr(cli, "load_config", lambda: RefineryConfig())

    def boom(*a, **k):
        raise AssertionError("refine_many must not run for --from chunk")

    monkeypatch.setattr(cli, "refine_many", boom)
    monkeypatch.setattr(cli, "chunk_markdown", lambda md, cfg: [Chunk(md, 0, 1, 1)])

    result = CliRunner().invoke(cli.main_many, [str(pdfs[0]), str(pdfs[1]), "--from", "chunk"])
    assert result.exit_code == 0, result.output
    assert "re-chunked 2/2 papers" in result.output
    assert json.loads(pdfs[0].with_suffix(".chunks.json").read_text())["parser"] == "paper-refinery"


def test_main_many_from_chunk_skips_paper_missing_refinery_md(tmp_path, monkeypatch):
    # a paper without a saved refinery.md is skipped (warned), not fatal to the batch
    pdfs = [tmp_path / "a.pdf", tmp_path / "b.pdf"]
    for p in pdfs:
        p.write_bytes(b"%PDF-1.4 fake")
    wd = pdfs[0].with_suffix(".refinery")  # only a.pdf has a saved md
    wd.mkdir()
    (wd / "refinery.md").write_text("# a\n\nbody")
    monkeypatch.setattr(cli, "load_config", lambda: RefineryConfig())
    monkeypatch.setattr(cli, "chunk_markdown", lambda md, cfg: [Chunk(md, 0, 1, 1)])

    result = CliRunner().invoke(cli.main_many, [str(pdfs[0]), str(pdfs[1]), "--from", "chunk"])
    assert result.exit_code == 0, result.output
    assert "re-chunked 1/2 papers" in result.output


def test_main_many_exits_nonzero_when_nothing_refined(tmp_path, monkeypatch):
    # every paper failed -> refine_many yields nothing -> exit non-zero, so a
    # `refinery-batch ... && papis ask index` chain stops instead of indexing an empty result
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setattr(cli, "load_config", lambda: RefineryConfig())
    monkeypatch.setattr(cli, "refine_many", lambda *a, **k: iter(()))  # nothing succeeds

    result = CliRunner().invoke(cli.main_many, [str(pdf)])
    assert result.exit_code == 1
    assert "refined 0/1 papers" in result.output


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
    assert data["parser"] == "paper-refinery"  # same versioned envelope as chunks.json
    assert data["schema_version"] == cli.MANIFEST_SCHEMA_VERSION
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
    assert paper_refinery.refine_many is cli.refine_many
    assert paper_refinery.RefineResult is cli.RefineResult


def _maas_parse(pdf, wd, pc, force=False):
    """Default-mode (maas) mock for the parse leg: ``_refine`` calls ``parse_pdf_cached``."""
    return ParseResult(markdown=pdf.stem)


def test_refine_many_maas_streams_each_result_with_its_doi(tmp_path, monkeypatch):
    # default mode is maas: each paper runs its whole pipeline on the pool; the parse leg
    # is parse_pdf_cached (which spawns its own cloud backend on a miss), not _parse_for_batch
    pdfs = [tmp_path / "a.pdf", tmp_path / "b.pdf"]
    for p in pdfs:
        p.write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setattr(cli, "load_config", lambda: RefineryConfig())
    monkeypatch.setattr(cli, "parse_pdf_cached", _maas_parse)

    seen = {}

    def fake_refine_parsed(parsed, pdf, out, cit, wd, cfg, source=None):
        seen[pdf.stem] = source
        return [Chunk(parsed.markdown, 0, 1, 1)], []

    monkeypatch.setattr(cli, "_refine_parsed", fake_refine_parsed)

    results = list(cli.refine_many(pdfs, dois=["10.1/a", None]))

    assert {r.chunks_path.name for r in results} == {"a.chunks.json", "b.chunks.json"}
    # the doi shorthand reaches each paper's pipeline folded into its source bundle
    assert seen == {"a": {"doi": "10.1/a"}, "b": {}}


def test_refine_many_routes_full_source_bundle_per_paper(tmp_path, monkeypatch):
    # sources (the rich SourceMeta bundles) reach each paper's pipeline, aligned to pdfs
    pdfs = [tmp_path / "a.pdf", tmp_path / "b.pdf"]
    for p in pdfs:
        p.write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setattr(cli, "load_config", lambda: RefineryConfig())
    monkeypatch.setattr(cli, "parse_pdf_cached", _maas_parse)
    seen = {}

    def fake_refine_parsed(parsed, pdf, out, cit, wd, cfg, source=None):
        seen[pdf.stem] = source
        return [Chunk(parsed.markdown, 0, 1, 1)], []

    monkeypatch.setattr(cli, "_refine_parsed", fake_refine_parsed)
    bundles = [{"doi": "10.1/a", "title": "A", "year": 2020}, None]
    list(cli.refine_many(pdfs, sources=bundles))
    assert seen == {"a": {"doi": "10.1/a", "title": "A", "year": 2020}, "b": {}}


def test_refine_many_maas_parses_papers_concurrently(tmp_path, monkeypatch):
    # the point of the cloud refactor: OCR has no shared server, so papers parse in
    # PARALLEL, not one-at-a-time. Both parse calls must be in flight at once -- the
    # barrier only trips when the second thread arrives, so if OCR were serialized the
    # first thread waits forever, the barrier times out, and the batch yields nothing.
    pdfs = [tmp_path / "a.pdf", tmp_path / "b.pdf"]
    for p in pdfs:
        p.write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setattr(cli, "load_config", lambda: RefineryConfig())

    both_parsing = threading.Barrier(2, timeout=5)

    def fake_parse(pdf, wd, pc, force=False):
        both_parsing.wait()  # requires the other paper's OCR to be concurrent
        return ParseResult(markdown=pdf.stem)

    monkeypatch.setattr(cli, "parse_pdf_cached", fake_parse)
    monkeypatch.setattr(
        cli,
        "_refine_parsed",
        lambda parsed, pdf, out, cit, wd, cfg, doi=None: ([Chunk(parsed.markdown, 0, 1, 1)], []),
    )

    results = list(cli.refine_many(pdfs, workers=2))
    assert {r.chunks_path.name for r in results} == {"a.chunks.json", "b.chunks.json"}


def test_refine_many_maas_caps_ocr_concurrency(tmp_path, monkeypatch):
    # ocr_workers gates only the OCR stage: with ocr_workers=1, at most ONE paper is inside
    # parse at a time even though workers=3 lets all three run their network tails at once.
    import time

    pdfs = [tmp_path / f"{c}.pdf" for c in "abc"]
    for p in pdfs:
        p.write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setattr(cli, "load_config", lambda: RefineryConfig())

    lock = threading.Lock()
    state = {"cur": 0, "peak": 0}

    def fake_parse(pdf, wd, pc, force=False):
        with lock:
            state["cur"] += 1
            state["peak"] = max(state["peak"], state["cur"])
        time.sleep(0.05)  # hold the OCR slot so a concurrency violation would be observed
        with lock:
            state["cur"] -= 1
        return ParseResult(markdown=pdf.stem)

    monkeypatch.setattr(cli, "parse_pdf_cached", fake_parse)
    monkeypatch.setattr(
        cli, "_refine_parsed", lambda *a, **k: ([Chunk("MD", 0, 1, 1)], [])
    )

    results = list(cli.refine_many(pdfs, workers=3, ocr_workers=1))
    assert len(results) == 3
    assert state["peak"] == 1  # the OCR gate never let two parses overlap


def test_refine_many_maas_skips_paper_whose_pipeline_fails(tmp_path, monkeypatch, caplog):
    pdfs = [tmp_path / "a.pdf", tmp_path / "b.pdf"]
    for p in pdfs:
        p.write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setattr(cli, "load_config", lambda: RefineryConfig())

    def fake_parse(pdf, wd, pc, force=False):
        if pdf.stem == "a":
            raise RuntimeError("corrupt pdf")
        return ParseResult(markdown="MD")

    monkeypatch.setattr(cli, "parse_pdf_cached", fake_parse)
    monkeypatch.setattr(
        cli,
        "_refine_parsed",
        lambda parsed, pdf, out, cit, wd, cfg, doi=None: ([Chunk("MD", 0, 1, 1)], []),
    )

    with caplog.at_level(logging.WARNING):
        results = list(cli.refine_many(pdfs))

    # a failed paper is skipped; the batch still yields the rest
    assert [r.chunks_path.name for r in results] == ["b.chunks.json"]
    assert "refine failed for a.pdf" in caplog.text


def _selfhosted_config():
    cfg = RefineryConfig()
    cfg.parse.mode = "selfhosted"  # opt into the serial-OCR path (one local llama-server)
    return cfg


def test_refine_many_selfhosted_overlaps_next_ocr_with_prior_network_stage(tmp_path, monkeypatch):
    # selfhosted keeps the serial-OCR design: paper a's network stage must run WHILE paper
    # b is still OCR-ing -- not serialized behind it. a's _refine_parsed blocks until b's
    # parse has started; if OCR were serialized behind the network stage, b never parses,
    # a's wait times out, and the assertion below fails.
    pdfs = [tmp_path / "a.pdf", tmp_path / "b.pdf"]
    for p in pdfs:
        p.write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setattr(cli, "load_config", _selfhosted_config)

    b_parse_started = threading.Event()

    def fake_parse(pdf, wd, cfg, backend, force):
        if pdf.stem == "b":
            b_parse_started.set()
        return ParseResult(markdown=pdf.stem)

    def fake_refine_parsed(parsed, pdf, out, cit, wd, cfg, doi=None):
        if pdf.stem == "a":
            assert b_parse_started.wait(timeout=5), "b's OCR did not overlap a's network stage"
        return [Chunk(parsed.markdown, 0, 1, 1)], []

    monkeypatch.setattr(cli, "_parse_for_batch", fake_parse)
    monkeypatch.setattr(cli, "_refine_parsed", fake_refine_parsed)

    results = list(cli.refine_many(pdfs))
    assert {r.chunks_path.name for r in results} == {"a.chunks.json", "b.chunks.json"}


def test_refine_many_selfhosted_does_not_spawn_ocr_backend_when_all_cached(tmp_path, monkeypatch):
    # a selfhosted batch where every parse is a checkpoint hit must never pay the server
    # spawn + model load -- the lazy backend stays unspawned
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setattr(cli, "load_config", _selfhosted_config)
    monkeypatch.setattr(cli, "load_checkpoint", lambda wd, p, cfg: ParseResult(markdown="MD"))

    def boom(cfg=None):
        raise AssertionError("ocr_backend must not spawn for an all-cached batch")

    monkeypatch.setattr(cli, "ocr_backend", boom)
    monkeypatch.setattr(
        cli,
        "_refine_parsed",
        lambda parsed, pdf, out, cit, wd, cfg, doi=None: ([Chunk("MD", 0, 1, 1)], []),
    )

    results = list(cli.refine_many([pdf]))
    assert len(results) == 1


def test_refine_many_rejects_mismatched_dois_eagerly(tmp_path):
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    # raises at the call, before any iteration -- not deferred to the first next()
    with pytest.raises(ValueError):
        cli.refine_many([pdf], RefineryConfig(), dois=[])
