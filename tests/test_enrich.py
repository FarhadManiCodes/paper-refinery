"""Tests for placeholder-anchored, page-batched figure enrichment (injected describer)."""

from pathlib import Path

from paper_refinery.config import FigureConfig
from paper_refinery.enrich import (
    _caption_after,
    _find_mentions,
    _find_placeholders,
    enrich_markdown,
)
from paper_refinery.parse import ParseResult

MD = (
    "<page_number>1</page_number>\n\n## Intro\n\nWe reference Figure 1 here.\n\n"
    "<page_number>2</page_number>\n\n"
    "![Alt fig one.](src)\n\nFIGURE 1. A comparison of methods A and B.\n\n"
    "<page_number>3</page_number>\n\n"
    "![Alt fig two.](src)\n\nFIGURE 2. Second figure caption.\n\n"
)
RENDERS = {2: Path("page_2.jpg"), 3: Path("page_3.jpg")}


def test_find_placeholders_with_pages():
    phs = _find_placeholders(MD)
    assert len(phs) == 2
    assert phs[0].alt.startswith("Alt fig one") and phs[0].page == 2
    assert phs[1].page == 3


def test_caption_after_placeholder():
    cap = _caption_after(MD, _find_placeholders(MD)[0].end)
    assert cap.number == "1" and "comparison of methods A and B" in cap.text


def test_find_mentions_excludes_caption():
    mentions = _find_mentions(MD, "1")
    assert any("We reference Figure 1" in m for m in mentions)
    assert not any(m.startswith("FIGURE 1.") for m in mentions)


def test_enrich_one_call_per_page_and_splices_after_caption():
    calls = []

    def fake(render, requests, cfg):
        calls.append((render.name, [n for n, _ in requests]))
        return {num: f"DESC-{num}" for num, _ in requests}

    out = enrich_markdown(ParseResult(MD, RENDERS), describe=fake)
    assert ("page_2.jpg", ["1"]) in calls and ("page_3.jpg", ["2"]) in calls
    i1, i2 = out.index("FIGURE 1."), out.index("FIGURE 2.")
    assert i1 < out.index("DESC-1") < i2
    assert i2 < out.index("DESC-2")


def test_enrich_batches_multiple_figures_on_one_page():
    md = (
        "<page_number>5</page_number>\n\n"
        "![a](src)\n\nFIGURE 3. First.\n\n"
        "![b](src)\n\nFIGURE 4. Second.\n\n"
    )
    calls = []

    def fake(render, requests, cfg):
        calls.append([n for n, _ in requests])
        return {num: f"D{num}" for num, _ in requests}

    out = enrich_markdown(ParseResult(md, {5: Path("page_5.jpg")}), describe=fake)
    assert len(calls) == 1  # ONE call for the page
    assert calls[0] == ["3", "4"]  # both figures in the single request
    assert "FIGURE 3. First.\n\n> **Figure description (auto):** D3" in out
    assert "FIGURE 4. Second.\n\n> **Figure description (auto):** D4" in out


def test_enrich_context_is_caption_only_by_default():
    captured = {}

    def fake(render, requests, cfg):
        captured["reqs"] = dict(requests)
        return {num: "D" for num, _ in requests}

    enrich_markdown(ParseResult(MD, {2: Path("page_2.jpg")}), describe=fake)
    assert "comparison of methods A and B" in captured["reqs"]["1"]
    assert "We reference Figure 1" not in captured["reqs"]["1"]


def test_enrich_includes_references_when_enabled():
    captured = {}

    def fake(render, requests, cfg):
        captured["reqs"] = dict(requests)
        return {num: "D" for num, _ in requests}

    enrich_markdown(
        ParseResult(MD, {2: Path("page_2.jpg")}),
        cfg=FigureConfig(include_references=True),
        describe=fake,
    )
    assert "comparison of methods A and B" in captured["reqs"]["1"]
    assert "We reference Figure 1" in captured["reqs"]["1"]


def test_enrich_falls_back_to_alt_when_no_render():
    out = enrich_markdown(ParseResult(MD, page_renders={}), describe=lambda r, q, cfg: {})
    assert "Figure description (auto):** Alt fig one" in out
    assert "Figure description (auto):** Alt fig two" in out


def test_enrich_falls_back_to_alt_when_figure_omitted():
    # Gemini returns nothing for the page -> alt-text fallback for each figure
    out = enrich_markdown(ParseResult(MD, RENDERS), describe=lambda r, q, cfg: {})
    assert "Figure description (auto):** Alt fig one" in out
