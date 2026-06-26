"""Tests for the CLI: the chunks-manifest writer and the main orchestration wiring."""

import json

from click.testing import CliRunner

from paper_refinery import cli
from paper_refinery.chunker import Chunk
from paper_refinery.cli import write_chunks
from paper_refinery.parse import ParseResult


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
