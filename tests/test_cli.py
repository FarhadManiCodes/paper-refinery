"""Tests for the chunks-manifest writer."""

import json

from paper_refinery.chunker import Chunk
from paper_refinery.cli import write_chunks


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
