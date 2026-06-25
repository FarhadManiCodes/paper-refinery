"""Tests for the LlamaParse wrapper.

The pure helpers (markdown assembly, figure collection) are tested directly; the
network-bound ``parse_pdf`` is exercised separately with a recorded fixture (skipped).
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from paper_refinery.parse import Figure, ParseResult, _build_markdown, _collect_figures


def test_dataclasses_construct():
    fig = Figure(image_path=Path("a.png"), page=3, caption="Fig 1")
    res = ParseResult(markdown="# x", figures=[fig])
    assert res.markdown == "# x"
    assert res.figures[0].page == 3 and res.figures[0].caption == "Fig 1"


def test_parse_result_defaults_to_no_figures():
    assert ParseResult(markdown="x").figures == []


def test_build_markdown_inserts_one_marker_per_page():
    pages = [SimpleNamespace(page=1, md="alpha"), SimpleNamespace(page=2, md="beta")]
    md = _build_markdown(pages)
    assert md.count("<page_number>") == 2
    assert "<page_number>1</page_number>" in md and "<page_number>2</page_number>" in md
    assert "alpha" in md and "beta" in md


def test_build_markdown_replaces_stale_inline_tags():
    pages = [SimpleNamespace(page=1, md="x <page_number>99</page_number> y")]
    md = _build_markdown(pages)
    assert "99" not in md  # stale inline tag stripped
    assert md.count("<page_number>") == 1
    assert "<page_number>1</page_number>" in md


def test_collect_figures_skips_screenshots_and_reads_page():
    paths = ["/o/chart_p6_0.png", "/o/img_p8_1.png", "/o/page_3.jpg", "/o/page_10.png"]
    figs = _collect_figures(paths)
    assert [f.page for f in figs] == [6, 8]  # page screenshots dropped
    assert all(f.image_path.name.startswith(("chart", "img")) for f in figs)


def test_collect_figures_handles_unparseable_name():
    figs = _collect_figures(["/o/weird.png"])
    assert len(figs) == 1 and figs[0].page is None


@pytest.mark.skip(reason="parse_pdf needs LlamaParse (network); add with a recorded fixture")
def test_parse_pdf_inserts_authoritative_page_markers():
    """When implemented: assert every page boundary yields one <page_number> marker."""
