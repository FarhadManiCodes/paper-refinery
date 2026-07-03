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
    _drop_trailing_boilerplate,
    _html_table_to_markdown,
    _llama_server,
    _merge_formula_numbers,
    _merge_reference_numbers,
    _render_references_markdown,
    _save_figure_crop,
    _sort_references_by_number,
)


def test_parse_result_defaults_are_empty():
    r = ParseResult(markdown="x")
    assert r.figure_crops == {} and r.references == [] and r.references_markdown == ""


# ---------------------------------------------------------------------------
# _dispatch_region
# ---------------------------------------------------------------------------


def _region(label: str, content: str = "", **extra) -> dict:
    return {"native_label": label, "label": label, "content": content, "index": 0, **extra}


@pytest.mark.parametrize(
    "label",
    ["header", "footer", "number", "footnote", "aside_text", "footer_image", "header_image"],
)
def test_dispatch_abandons_boilerplate(label):
    kind, text = _dispatch_region(_region(label, "should be dropped"), ParseConfig())
    assert kind == "abandon"
    assert text == ""


def test_dispatch_reference_number_is_own_kind():
    # the bracket/number marker before a bibliography entry -- kept (not abandoned) so
    # it can be paired back with its reference_content sibling by _merge_reference_numbers
    kind, text = _dispatch_region(_region("reference", "[12]"), ParseConfig())
    assert kind == "reference_number" and text == "[12]"


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
# _merge_reference_numbers
# ---------------------------------------------------------------------------


def test_merge_reference_number_then_content():
    triples = [("reference_number", "[12]", {}), ("reference", "Smith, J. (2020).", {})]
    merged = _merge_reference_numbers(triples)
    assert merged == [("reference", "Smith, J. (2020).", {"number": "12"})]


def test_merge_content_then_reference_number():
    triples = [("reference", "Smith, J. (2020).", {}), ("reference_number", "23.", {})]
    merged = _merge_reference_numbers(triples)
    assert merged == [("reference", "Smith, J. (2020).", {"number": "23"})]


def test_unmerged_reference_keeps_number_none():
    triples = [("reference", "Smith, J. (2020).", {})]
    merged = _merge_reference_numbers(triples)
    assert len(merged) == 1
    assert merged[0][:2] == ("reference", "Smith, J. (2020).")
    assert merged[0][2].get("number") is None  # unpaired: region passes through as-is


def test_unmerged_reference_number_is_dropped():
    triples = [("reference_number", "[12]", {}), ("body", "next paragraph", {})]
    merged = _merge_reference_numbers(triples)
    assert merged == [("body", "next paragraph", {})]


# ---------------------------------------------------------------------------
# _render_references_markdown
# ---------------------------------------------------------------------------


def test_render_references_markdown_groups_by_page_with_number_prefix():
    refs = [
        {"page": 1, "number": "1", "text": "Smith, J. (2020)."},
        {"page": 1, "number": "2", "text": "Jones, A. (2019)."},
        {"page": 2, "number": None, "text": "Lee, K. (2018)."},
    ]
    md = _render_references_markdown(refs)
    assert "<page_number>1</page_number>" in md
    assert "<page_number>2</page_number>" in md
    assert "[1] Smith, J. (2020)." in md
    assert "[2] Jones, A. (2019)." in md
    assert "Lee, K. (2018)." in md and "[None]" not in md
    assert md.index("<page_number>1</page_number>") < md.index("[1]")
    assert md.index("<page_number>2</page_number>") < md.index("Lee, K.")


def test_render_references_markdown_empty_list():
    assert _render_references_markdown([]) == ""


# ---------------------------------------------------------------------------
# _drop_trailing_boilerplate
# ---------------------------------------------------------------------------


def _ref(text: str, page: int = 1, number: str | None = None) -> dict:
    return {"page": page, "number": number, "text": text}


def test_drop_trailing_boilerplate_removes_single_trailing_entry():
    refs = [
        _ref("Smith, J. (2020). Real paper."),
        _ref(
            "Conflict of Interest: The authors declare that the research was conducted "
            "in the absence of any commercial or financial relationships."
        ),
    ]
    cleaned = _drop_trailing_boilerplate(refs)
    assert cleaned == [refs[0]]


def test_drop_trailing_boilerplate_removes_multiple_trailing_entries():
    refs = [
        _ref("Smith, J. (2020). Real paper."),
        _ref("Conflict of Interest: none declared."),
        _ref("Copyright © 2021 Author. This is an open-access article distributed..."),
    ]
    cleaned = _drop_trailing_boilerplate(refs)
    assert cleaned == [refs[0]]


def test_drop_trailing_boilerplate_stops_at_first_non_matching_entry():
    # a real reference sits between two boilerplate-like entries -- only the
    # *trailing* run should be dropped, not a match buried earlier in the list
    refs = [
        _ref("Smith, J. (2020). Real paper."),
        _ref("Jones, A. On Copyright Law and Academic Publishing. (2019)."),
        _ref("Conflict of Interest: none declared."),
    ]
    cleaned = _drop_trailing_boilerplate(refs)
    assert cleaned == refs[:2]


def test_drop_trailing_boilerplate_caps_how_many_it_checks():
    # more than _MAX_TRAILING_BOILERPLATE_CHECK consecutive "boilerplate-looking"
    # entries -- only the last few are dropped, not the whole list
    refs = [_ref(f"Copyright © 2021. Entry {i}.") for i in range(5)]
    cleaned = _drop_trailing_boilerplate(refs)
    assert len(cleaned) == 2


def test_drop_trailing_boilerplate_leaves_clean_list_untouched():
    refs = [_ref("Smith, J. (2020)."), _ref("Jones, A. (2019).")]
    assert _drop_trailing_boilerplate(refs) == refs


def test_drop_trailing_boilerplate_handles_empty_list():
    assert _drop_trailing_boilerplate([]) == []


# ---------------------------------------------------------------------------
# _sort_references_by_number
# ---------------------------------------------------------------------------


def test_sort_references_by_number_fixes_scrambled_order():
    # e.g. kalman-1960.pdf's two-column bibliography: region index order doesn't
    # match reading order, but every entry has a clean numeric marker
    refs = [_ref("Wiener", number="2"), _ref("Zadeh", number="1"), _ref("Bode", number="3")]
    sorted_refs = _sort_references_by_number(refs)
    assert [r["number"] for r in sorted_refs] == ["1", "2", "3"]


def test_sort_references_by_number_leaves_unnumbered_style_untouched():
    # author-year style (e.g. fmech-07-655266.pdf): no numbers at all
    refs = [_ref("Bianchini"), _ref("Boness"), _ref("Burberi")]
    assert _sort_references_by_number(refs) == refs


def test_sort_references_by_number_leaves_partial_numbering_untouched():
    # one entry missing its number (e.g. an unpaired reference_content) -- don't
    # guess at a partial sort, leave detected order as-is
    refs = [_ref("Wiener", number="2"), _ref("Zadeh", number=None), _ref("Bode", number="3")]
    assert _sort_references_by_number(refs) == refs


def test_sort_references_by_number_leaves_non_numeric_marker_untouched():
    # a garbled/non-integer marker on at least one entry -- bail out entirely
    refs = [_ref("Wiener", number="2"), _ref("Zadeh", number="1a")]
    assert _sort_references_by_number(refs) == refs


def test_sort_references_by_number_handles_empty_list():
    assert _sort_references_by_number([]) == []


def test_sort_references_by_number_falls_back_to_leading_number_in_text():
    # the real kalman-1960.pdf bug: PP-DocLayout-V3 never produces a separate
    # reference_number region for this paper at all -- the marker is just the leading
    # digits of the OCR'd text blob itself, so `number` is None on every entry
    refs = [
        _ref("2 L. A. Zadeh and J. R. Ragazzini, An Extension of..."),
        _ref("1 N. Wiener, The Extrapolation, Interpolation..."),
        _ref("10 R. C. Davis, On the Theory of Prediction..."),
        _ref("3 H. W. Bode and C. E. Shannon, A Simplified..."),
    ]
    sorted_refs = _sort_references_by_number(refs)
    assert [r["text"][:2].strip() for r in sorted_refs] == ["1", "2", "3", "10"]


def test_sort_references_by_number_falls_back_to_bracketed_number_in_text():
    refs = [_ref("[2] Second entry"), _ref("[1] First entry")]
    sorted_refs = _sort_references_by_number(refs)
    assert [r["text"] for r in sorted_refs] == ["[1] First entry", "[2] Second entry"]


def test_sort_references_by_number_mixed_region_and_text_number_sources():
    # one entry has a real paired `number`, another only has it embedded in the text --
    # both should still contribute a usable sort key
    refs = [_ref("Second entry", number="2"), _ref("1 First entry")]
    sorted_refs = _sort_references_by_number(refs)
    assert sorted_refs[0]["text"] == "1 First entry"


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
            _region("reference_content", "Smith, J. (2020).", index=1),
        ]
    ]
    md, crops, refs = _build_markdown(pages, {}, tmp_path, ParseConfig())
    assert "Smith, J." not in md
    assert refs == [{"page": 1, "number": None, "text": "Smith, J. (2020)."}]


def test_build_markdown_pairs_reference_number_with_content(tmp_path):
    pages = [
        [
            _region("reference", "[1]", index=0),
            _region("reference_content", "Smith, J. (2020).", index=1),
        ]
    ]
    md, crops, refs = _build_markdown(pages, {}, tmp_path, ParseConfig())
    assert refs == [{"page": 1, "number": "1", "text": "Smith, J. (2020)."}]


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
