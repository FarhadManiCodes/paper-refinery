"""Tests for placeholder-anchored figure enrichment (pure logic, injected describer)."""

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
    "![Alt describing fig one: methods A and B.](image_url_placeholder)\n\n"
    "FIGURE 1. A comparison of methods A and B.\n\n"
    "<page_number>3</page_number>\n\n"
    "![Alt for fig two.](image_url_placeholder)\n\n"
    "FIGURE 2. Second figure caption.\n\n"
)
RENDERS = {2: Path("page_2.jpg"), 3: Path("page_3.jpg")}


def test_find_placeholders_with_pages():
    phs = _find_placeholders(MD)
    assert len(phs) == 2
    assert phs[0].alt.startswith("Alt describing fig one")
    assert phs[0].page == 2 and phs[1].page == 3


def test_caption_after_placeholder():
    phs = _find_placeholders(MD)
    cap = _caption_after(MD, phs[0].end)
    assert cap.number == "1"
    assert "comparison of methods A and B" in cap.text


def test_find_mentions_excludes_caption():
    mentions = _find_mentions(MD, "1")
    assert any("We reference Figure 1" in m for m in mentions)
    assert not any(m.startswith("FIGURE 1.") for m in mentions)


def test_enrich_sends_each_figures_page_render_and_splices_after_caption():
    calls = {}

    def fake(render, context, cfg):
        calls[render.name] = context
        return f"DESC-{render.name}"

    out = enrich_markdown(ParseResult(MD, RENDERS), describe=fake)
    # each figure described from its own page render, grounded on its caption
    assert calls["page_2.jpg"].startswith("FIGURE 1.")
    assert calls["page_3.jpg"].startswith("FIGURE 2.")
    i1, i2 = out.index("FIGURE 1."), out.index("FIGURE 2.")
    assert i1 < out.index("DESC-page_2.jpg") < i2  # fig 1 description under FIGURE 1
    assert i2 < out.index("DESC-page_3.jpg")  # fig 2 description under FIGURE 2


def test_enrich_context_is_caption_only_by_default():
    captured = {}

    def fake(render, context, cfg):
        captured["c"] = context
        return "D"

    enrich_markdown(ParseResult(MD, {2: Path("page_2.jpg")}), describe=fake)
    assert "comparison of methods A and B" in captured["c"]  # the caption
    assert "We reference Figure 1" not in captured["c"]  # mentions excluded by default


def test_enrich_includes_references_when_enabled():
    captured = {}

    def fake(render, context, cfg):
        captured["c"] = context
        return "D"

    enrich_markdown(
        ParseResult(MD, {2: Path("page_2.jpg")}),
        cfg=FigureConfig(include_references=True),
        describe=fake,
    )
    assert "comparison of methods A and B" in captured["c"]  # caption
    assert "We reference Figure 1" in captured["c"]  # mentions now included


def test_enrich_falls_back_to_alt_text_when_no_render():
    # no page render for the figure's page -> describe never reached -> alt-text used
    out = enrich_markdown(ParseResult(MD, page_renders={}), describe=lambda r, c, cfg: "X")
    assert "Figure description (auto):** Alt describing fig one" in out
    assert "Figure description (auto):** X" not in out


def test_enrich_falls_back_when_gemini_returns_empty():
    out = enrich_markdown(ParseResult(MD, RENDERS), describe=lambda r, c, cfg: "")
    assert "Figure description (auto):** Alt describing fig one" in out
    assert "Figure description (auto):** Alt for fig two" in out
