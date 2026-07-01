"""Tests for the local GLM-OCR parser.

The pure helpers (region dispatch, markdown assembly, table conversion, formula-number
merging, the llama-server health check) are tested directly against hand-built
glmocr-shaped region dicts; the network/process-bound ``parse_pdf`` itself is exercised
separately with a live llama-server + GLM-OCR (skipped by default).
"""

from __future__ import annotations

import subprocess

import pytest

from paper_refinery.config import ParseConfig
from paper_refinery.parse import (
    ParseResult,
    _build_markdown,
    _dispatch_region,
    _dotted_overrides,
    _html_table_to_markdown,
    _llama_server,
    _merge_formula_numbers,
    _save_figure_crop,
)


def test_parse_result_defaults_are_empty():
    r = ParseResult(markdown="x")
    assert r.figure_crops == {} and r.references == []


# ---------------------------------------------------------------------------
# _dispatch_region
# ---------------------------------------------------------------------------


def _region(label: str, content: str = "", **extra) -> dict:
    return {"native_label": label, "label": label, "content": content, "index": 0, **extra}


@pytest.mark.parametrize(
    "label",
    ["header", "footer", "number", "footnote", "aside_text", "reference", "footer_image", "header_image"],
)
def test_dispatch_abandons_boilerplate(label):
    kind, text = _dispatch_region(_region(label, "should be dropped"), ParseConfig())
    assert kind == "abandon"
    assert text == ""


def test_dispatch_doc_title_becomes_h1():
    kind, text = _dispatch_region(_region("doc_title", "My Paper"), ParseConfig())
    assert kind == "body" and text == "# My Paper"


def test_dispatch_paragraph_title_becomes_h2():
    kind, text = _dispatch_region(_region("paragraph_title", "Methods"), ParseConfig())
    assert kind == "body" and text == "## Methods"


def test_dispatch_title_strips_ocrs_own_heading_marker():
    # GLM-OCR sometimes emits its own "##" inside a title region's content; our own
    # prefix must not double up with it (regression: "## ## Section Title").
    kind, text = _dispatch_region(_region("paragraph_title", "## B. A subsection"), ParseConfig())
    assert text == "## B. A subsection"

    kind, text = _dispatch_region(_region("doc_title", "# My Paper"), ParseConfig())
    assert text == "# My Paper"


def test_dispatch_figure_title_is_plain_body_text():
    # the "FIGURE N. ..." caption line -- must land in body markdown as-is, since
    # enrich.py's caption regex scans the body text for it
    kind, text = _dispatch_region(_region("figure_title", "FIGURE 4.1. A comparison."), ParseConfig())
    assert kind == "body" and text == "FIGURE 4.1. A comparison."


def test_dispatch_algorithm_is_fenced_code_block():
    kind, text = _dispatch_region(_region("algorithm", "for i in range(n):\n    do(i)"), ParseConfig())
    assert kind == "body"
    assert text == "```\nfor i in range(n):\n    do(i)\n```"


def test_dispatch_reference_content_is_routed_separately():
    kind, text = _dispatch_region(_region("reference_content", "Smith et al. 2020."), ParseConfig())
    assert kind == "reference"
    assert text == "Smith et al. 2020."


def test_dispatch_chart_and_image_are_figure_kind():
    for label in ("chart", "image"):
        kind, text = _dispatch_region(_region(label, ""), ParseConfig())
        assert kind == "figure"


def test_dispatch_table_converts_html_to_markdown_by_default():
    html = "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>"
    kind, text = _dispatch_region(_region("table", html), ParseConfig())
    assert kind == "body"
    assert "| A | B |" in text
    assert "| 1 | 2 |" in text


def test_dispatch_table_keeps_html_when_configured():
    html = "<table><tr><td>x</td></tr></table>"
    kind, text = _dispatch_region(_region("table", html), ParseConfig(table_format="html"))
    assert kind == "body" and text == html


def test_dispatch_formula_wraps_in_dollars():
    kind, text = _dispatch_region(_region("display_formula", "E = mc^2"), ParseConfig())
    assert kind == "formula"
    assert text == "$$\nE = mc^2\n$$"


def test_dispatch_formula_strips_existing_fences():
    kind, text = _dispatch_region(_region("inline_formula", "$$ x^2 $$"), ParseConfig())
    assert text == "$$\nx^2\n$$"


def test_dispatch_formula_number_is_own_kind():
    kind, text = _dispatch_region(_region("formula_number", "(1)"), ParseConfig())
    assert kind == "formula_number" and text == "(1)"


def test_dispatch_unknown_label_kept_as_body_text():
    kind, text = _dispatch_region(_region("some_new_label", "hello"), ParseConfig())
    assert kind == "body" and text == "hello"


# ---------------------------------------------------------------------------
# _html_table_to_markdown
# ---------------------------------------------------------------------------


def test_html_table_simple_roundtrip():
    html = "<table><tr><th>Name</th><th>Value</th></tr><tr><td>a</td><td>1</td></tr></table>"
    md = _html_table_to_markdown(html)
    lines = md.splitlines()
    assert lines[0] == "| Name | Value |"
    assert lines[1] == "| --- | --- |"
    assert lines[2] == "| a | 1 |"


def test_html_table_duplicates_colspan_value():
    html = "<table><tr><td colspan='2'>Header</td></tr><tr><td>a</td><td>b</td></tr></table>"
    md = _html_table_to_markdown(html, strategy="duplicate")
    assert md.splitlines()[0] == "| Header | Header |"


def test_html_table_duplicates_rowspan_value():
    html = (
        "<table>"
        "<tr><td rowspan='2'>Method</td><td>1</td></tr>"
        "<tr><td>2</td></tr>"
        "</table>"
    )
    md = _html_table_to_markdown(html, strategy="duplicate")
    lines = md.splitlines()
    assert lines[0] == "| Method | 1 |"
    assert lines[2] == "| Method | 2 |"


def test_html_table_tolerates_malformed_span_attribute():
    # GLM-OCR's own model-generated HTML, not hand-authored -- a garbled rowspan
    # shouldn't crash table conversion (and by extension, the whole page/PDF)
    html = "<table><tr><td rowspan='not-a-number'>x</td><td>y</td></tr></table>"
    md = _html_table_to_markdown(html)
    assert md.splitlines()[0] == "| x | y |"


def test_html_table_missing_table_tag_falls_back_to_stripped_text():
    assert _html_table_to_markdown("just some text") == "just some text"


# ---------------------------------------------------------------------------
# _merge_formula_numbers
# ---------------------------------------------------------------------------


def test_merge_formula_then_number():
    triples = [("formula", "$$\nE=mc^2\n$$", {}), ("formula_number", "(1)", {})]
    merged = _merge_formula_numbers(triples)
    assert len(merged) == 1
    assert merged[0][0] == "body"
    assert merged[0][1] == "$$\nE=mc^2 \\tag{1}\n$$"


def test_merge_number_then_formula():
    triples = [("formula_number", "(2)", {}), ("formula", "$$\nx^2\n$$", {})]
    merged = _merge_formula_numbers(triples)
    assert len(merged) == 1
    assert merged[0][1] == "$$\nx^2 \\tag{2}\n$$"


def test_unmerged_formula_number_becomes_parenthetical_body():
    triples = [("formula_number", "(3)", {})]
    merged = _merge_formula_numbers(triples)
    assert merged == [("body", "(3)", {})]


def test_unmerged_formula_passes_through():
    triples = [("formula", "$$\nx\n$$", {}), ("body", "next paragraph", {})]
    merged = _merge_formula_numbers(triples)
    assert merged[0] == ("body", "$$\nx\n$$", {})
    assert merged[1] == ("body", "next paragraph", {})


# ---------------------------------------------------------------------------
# _build_markdown (integration of dispatch + merge + figure/reference routing)
# ---------------------------------------------------------------------------


def test_build_markdown_drops_boilerplate_and_keeps_body(tmp_path):
    pages = [
        [
            _region("header", "Running Head", index=0),
            _region("doc_title", "A Paper", index=1),
            _region("text", "Hello world.", index=2),
            _region("footer", "1", index=3),
        ]
    ]
    md, crops, refs = _build_markdown(pages, {}, tmp_path, ParseConfig())
    assert "Running Head" not in md
    assert md.count("<page_number>") == 1 and "<page_number>1</page_number>" in md
    assert "# A Paper" in md
    assert "Hello world." in md
    assert crops == {} and refs == []


def test_build_markdown_routes_references_out_of_body(tmp_path):
    pages = [
        [
            _region("text", "Body text.", index=0),
            _region("reference_content", "[1] Smith, J. (2020).", index=1),
        ]
    ]
    md, crops, refs = _build_markdown(pages, {}, tmp_path, ParseConfig())
    assert "Smith, J." not in md
    assert refs == [{"page": 1, "text": "[1] Smith, J. (2020)."}]


def test_build_markdown_saves_figure_crop_and_inserts_placeholder(tmp_path):
    from PIL import Image

    img = Image.new("RGB", (10, 10), "white")
    image_files = {"cropped_page0_idx0.jpg": img}
    pages = [[_region("chart", "", index=0, image_path="imgs/cropped_page0_idx0.jpg")]]

    md, crops, refs = _build_markdown(pages, image_files, tmp_path, ParseConfig())

    assert 1 in crops and len(crops[1]) == 1
    assert crops[1][0].exists()
    assert "FIGURE_CROP 1:0" in md


def test_build_markdown_multiple_pages_have_distinct_markers(tmp_path):
    pages = [[_region("text", "page one", index=0)], [_region("text", "page two", index=0)]]
    md, _, _ = _build_markdown(pages, {}, tmp_path, ParseConfig())
    assert "<page_number>1</page_number>" in md and "<page_number>2</page_number>" in md
    assert md.index("<page_number>1</page_number>") < md.index("page one")
    assert md.index("<page_number>2</page_number>") < md.index("page two")


# ---------------------------------------------------------------------------
# _save_figure_crop
# ---------------------------------------------------------------------------


def test_save_figure_crop_warns_and_returns_none_when_missing(tmp_path):
    region = {"image_path": None}
    with pytest.warns(UserWarning):
        result = _save_figure_crop(region, {}, set(), tmp_path, page=1, idx=0)
    assert result is None


# ---------------------------------------------------------------------------
# _dotted_overrides
# ---------------------------------------------------------------------------


def test_dotted_overrides_widens_only_figure_class_ids():
    dotted = _dotted_overrides(ParseConfig(figure_crop_margin=1.1))
    ratios = dotted["pipeline.layout.layout_unclip_ratio"]
    assert ratios == {3: (1.1, 1.1), 14: (1.1, 1.1)}


def test_dotted_overrides_lets_explicit_override_win():
    cfg = ParseConfig(
        figure_crop_margin=1.1,
        glmocr_config_overrides={"pipeline.layout.layout_unclip_ratio": 1.0},
    )
    dotted = _dotted_overrides(cfg)
    assert dotted["pipeline.layout.layout_unclip_ratio"] == 1.0


def test_dotted_overrides_keeps_unrelated_user_overrides():
    cfg = ParseConfig(glmocr_config_overrides={"pipeline.max_workers": 1})
    dotted = _dotted_overrides(cfg)
    assert dotted["pipeline.max_workers"] == 1
    assert "pipeline.layout.layout_unclip_ratio" in dotted


# ---------------------------------------------------------------------------
# _llama_server
# ---------------------------------------------------------------------------


def test_llama_server_requires_model_path():
    with pytest.raises(RuntimeError, match="model_path"):
        with _llama_server(ParseConfig(mmproj_path="/x/mmproj.gguf")):
            pass


def test_llama_server_requires_mmproj_path():
    with pytest.raises(RuntimeError, match="mmproj_path"):
        with _llama_server(ParseConfig(model_path="/x/model.gguf")):
            pass


def test_llama_server_raises_cleanly_on_early_exit(monkeypatch):
    class FakeProc:
        def __init__(self, *a, **k):
            self.returncode = 1

        def poll(self):
            return self.returncode

        def terminate(self):
            pass

        def wait(self, timeout=None):
            pass

        def kill(self):
            pass

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: FakeProc())
    cfg = ParseConfig(model_path="/x/model.gguf", mmproj_path="/x/mmproj.gguf", startup_timeout_s=5)
    with pytest.raises(RuntimeError, match="exited early"):
        with _llama_server(cfg):
            pass


@pytest.mark.skip(reason="parse_pdf needs a running llama-server + GLM-OCR weights")
def test_parse_pdf_builds_markers_figures_and_references():
    """When run against a real backend: assert one <page_number> per page, figure crops
    saved under image_dir/figures, references routed to ParseResult.references and never
    appearing in .markdown, and boilerplate (headers/footers/page numbers) absent."""
