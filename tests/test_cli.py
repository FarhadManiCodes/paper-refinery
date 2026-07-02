"""Tests for the CLI: the chunks-manifest writer and the main orchestration wiring."""

import json

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
    md = (
        f"![a]({tmp_path}/figures/a.png) text ![b]({tmp_path}/other/b.png)"
    )
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
    monkeypatch.setattr(cli, "parse_pdf", lambda p, d, c: ParseResult(markdown="MD"))
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
    # the enriched markdown is kept as an artifact before chunking
    assert pdf.with_suffix(".refinery.md").read_text() == "ENRICHED"
    # no references from parse_pdf in this test -> no sidecar written
    assert not pdf.with_suffix(".references.json").exists()
    assert "references ->" not in result.output


def test_main_defaults_image_dir_next_to_md_out(tmp_path, monkeypatch):
    pdf = tmp_path / "p.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    seen = {}

    def fake_parse_pdf(p, d, c):
        seen["image_dir"] = d
        return ParseResult(markdown="MD")

    monkeypatch.setattr(cli, "load_config", lambda: RefineryConfig())
    monkeypatch.setattr(cli, "parse_pdf", fake_parse_pdf)
    monkeypatch.setattr(cli, "make_client", lambda cfg: object())
    monkeypatch.setattr(cli, "enrich_markdown", lambda parsed, cfg, describe: parsed.markdown)
    monkeypatch.setattr(cli, "chunk_markdown", lambda md, cfg: [Chunk(md, 0, 1, 1)])

    result = CliRunner().invoke(cli.main, [str(pdf)])
    assert result.exit_code == 0, result.output
    # no --image-dir given -> figures persist next to the .refinery.md, not a temp dir
    assert seen["image_dir"] == pdf.with_suffix(".refinery.md").parent


def test_main_writes_references_sidecar_and_keeps_it_out_of_chunking(tmp_path, monkeypatch):
    pdf = tmp_path / "p.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    refs = [{"page": 3, "text": "[1] Smith, J. (2020)."}]
    monkeypatch.setattr(cli, "load_config", lambda: RefineryConfig())
    monkeypatch.setattr(
        cli, "parse_pdf", lambda p, d, c: ParseResult(markdown="MD", references=refs)
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

    refs_path = pdf.with_suffix(".references.json")
    assert refs_path.exists()
    data = json.loads(refs_path.read_text())
    assert data["references"] == refs
    assert data["source_pdf"] == str(pdf)
    assert f"references -> {refs_path}" in result.output
    # references never reach the chunker
    assert "Smith" not in seen_chunk_input["md"]
