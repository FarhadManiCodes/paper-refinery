"""Tests for the LlamaParse wrapper (dataclasses now; parse_pdf when implemented)."""

from pathlib import Path

import pytest

from paper_refinery.parse import Figure, ParseResult


def test_dataclasses_construct():
    fig = Figure(image_path=Path("a.png"), page=3, caption="Fig 1")
    res = ParseResult(markdown="# x", figures=[fig])
    assert res.markdown == "# x"
    assert res.figures[0].page == 3 and res.figures[0].caption == "Fig 1"


def test_parse_result_defaults_to_no_figures():
    assert ParseResult(markdown="x").figures == []


@pytest.mark.skip(reason="parse_pdf needs LlamaParse (network); add with a recorded fixture")
def test_parse_pdf_inserts_authoritative_page_markers():
    """When implemented: assert every page boundary yields one <page_number> marker."""
