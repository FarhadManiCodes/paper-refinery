"""Tests for the local GLM-OCR parser.

The pure helpers (region dispatch, markdown assembly, table conversion, reference-number
merging) are tested directly against hand-built glmocr-shaped region dicts; ``parse_pdf``
itself is exercised offline against a fake backend (server lifecycle tests live in
test_backend.py), plus one live test with a real llama-server (skipped by default).
"""

from __future__ import annotations

import logging

import pytest

from paper_refinery.backend import OcrBackend
from paper_refinery.config import ParseConfig
from paper_refinery.parse import (
    ParseResult,
    _build_markdown,
    _dispatch_region,
    _merge_reference_numbers,
    _save_figure_crop,
    parse_pdf,
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
    kind, text = _dispatch_region(_region(label, "should be dropped"))
    assert kind == "abandon"
    assert text == ""


def test_dispatch_reference_number_is_own_kind():
    # the bracket/number marker before a bibliography entry -- kept (not abandoned) so
    # it can be paired back with its reference_content sibling by _merge_reference_numbers
    kind, text = _dispatch_region(_region("reference", "[12]"))
    assert kind == "reference_number" and text == "[12]"


def test_dispatch_doc_title_becomes_h1():
    kind, text = _dispatch_region(_region("doc_title", "My Paper"))
    assert kind == "body" and text == "# My Paper"


def test_dispatch_paragraph_title_becomes_h2():
    kind, text = _dispatch_region(_region("paragraph_title", "Methods"))
    assert kind == "body" and text == "## Methods"


def test_dispatch_title_strips_ocrs_own_heading_marker():
    # GLM-OCR sometimes emits its own "##" inside a title region's content; our own
    # prefix must not double up with it (regression: "## ## Section Title").
    kind, text = _dispatch_region(_region("paragraph_title", "## B. A subsection"))
    assert text == "## B. A subsection"

    kind, text = _dispatch_region(_region("doc_title", "# My Paper"))
    assert text == "# My Paper"


def test_dispatch_figure_title_is_caption_kind():
    # the "FIGURE N. ..." caption line -- routed as its own kind so _build_markdown
    # both keeps it in the body markdown AND records it as a known caption
    kind, text = _dispatch_region(_region("figure_title", "FIGURE 4.1. A comparison."))
    assert kind == "caption" and text == "FIGURE 4.1. A comparison."


def test_dispatch_algorithm_is_fenced_code_block():
    kind, text = _dispatch_region(_region("algorithm", "for i in range(n):\n    do(i)"))
    assert kind == "body"
    assert text == "```\nfor i in range(n):\n    do(i)\n```"


def test_dispatch_reference_content_is_routed_separately():
    kind, text = _dispatch_region(_region("reference_content", "Smith et al. 2020."))
    assert kind == "reference"
    assert text == "Smith et al. 2020."


def test_dispatch_chart_and_image_are_figure_kind():
    for label in ("chart", "image"):
        kind, text = _dispatch_region(_region(label, ""))
        assert kind == "figure"


def test_dispatch_table_converts_html_to_markdown():
    html = "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>"
    kind, text = _dispatch_region(_region("table", html))
    assert kind == "body"
    assert "| A | B |" in text
    assert "| 1 | 2 |" in text


def test_dispatch_formula_wraps_in_dollars():
    kind, text = _dispatch_region(_region("display_formula", "E = mc^2"))
    assert kind == "body"
    assert text == "$$\nE = mc^2\n$$"


def test_dispatch_formula_strips_existing_fences():
    kind, text = _dispatch_region(_region("inline_formula", "$$ x^2 $$"))
    assert text == "$$\nx^2\n$$"


def test_dispatch_formula_number_becomes_parenthetical_body():
    # glmocr merges these into the formula upstream by default; a standalone one (merge
    # disabled via overrides) degrades to its own "(N)" paragraph rather than being dropped
    kind, text = _dispatch_region(_region("formula_number", "(1)"))
    assert kind == "body" and text == "(1)"


def test_dispatch_empty_formula_number_yields_empty_body():
    kind, text = _dispatch_region(_region("formula_number", ""))
    assert kind == "body" and text == ""


def test_dispatch_unknown_label_kept_as_body_text():
    kind, text = _dispatch_region(_region("some_new_label", "hello"))
    assert kind == "body" and text == "hello"


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


def test_build_markdown_reclaims_mislabeled_first_reference(tmp_path):
    pages = [
        [
            _region("text", "ACKNOWLEDGMENTS. Thanks everyone.", index=0),
            _region("text", "1. Jordan MI (2015) Machine learning. Science.", index=1),
            _region("reference_content", "2. Bongard J (2007) Automated.", index=2),
        ]
    ]
    md, _, _, refs = _build_markdown(pages, {}, tmp_path)
    assert "Jordan" not in md  # moved out of the body...
    assert [r["text"] for r in refs] == [  # ...into the references, in order
        "1. Jordan MI (2015) Machine learning. Science.",
        "2. Bongard J (2007) Automated.",
    ]


def test_build_markdown_warns_on_numbered_gap(tmp_path, caplog):
    pages = [
        [
            _region("reference_content", "1. First.", index=0),
            _region("reference_content", "3. Third.", index=1),
        ]
    ]
    with caplog.at_level(logging.WARNING):
        _build_markdown(pages, {}, tmp_path)
    assert "missing entr(ies): [2]" in caplog.text


def test_build_markdown_no_gap_warning_when_contiguous(tmp_path, caplog):
    pages = [
        [
            _region("reference_content", "1. First.", index=0),
            _region("reference_content", "2. Second.", index=1),
        ]
    ]
    with caplog.at_level(logging.WARNING):
        _build_markdown(pages, {}, tmp_path)
    assert "missing entr" not in caplog.text


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
    md, crops, _caps, refs = _build_markdown(pages, {}, tmp_path)
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
    md, crops, _caps, refs = _build_markdown(pages, {}, tmp_path)
    assert "Smith, J." not in md
    assert refs == [{"page": 1, "number": None, "text": "Smith, J. (2020)."}]


def test_build_markdown_pairs_reference_number_with_content(tmp_path):
    pages = [
        [
            _region("reference", "[1]", index=0),
            _region("reference_content", "Smith, J. (2020).", index=1),
        ]
    ]
    md, crops, _caps, refs = _build_markdown(pages, {}, tmp_path)
    assert refs == [{"page": 1, "number": "1", "text": "Smith, J. (2020)."}]


def test_build_markdown_saves_figure_crop_and_inserts_placeholder(tmp_path):
    from PIL import Image

    img = Image.new("RGB", (10, 10), "white")
    image_files = {"cropped_page0_idx0.jpg": img}
    pages = [[_region("chart", "", index=0, image_path="imgs/cropped_page0_idx0.jpg")]]

    md, crops, _caps, refs = _build_markdown(pages, image_files, tmp_path)

    assert 1 in crops and len(crops[1]) == 1
    assert crops[1][0].path.exists()
    assert "FIGURE_CROP 1:0" in md


def test_build_markdown_multiple_pages_have_distinct_markers(tmp_path):
    pages = [[_region("text", "page one", index=0)], [_region("text", "page two", index=0)]]
    md, _, _, _ = _build_markdown(pages, {}, tmp_path)
    assert "<page_number>1</page_number>" in md and "<page_number>2</page_number>" in md
    assert md.index("<page_number>1</page_number>") < md.index("page one")
    assert md.index("<page_number>2</page_number>") < md.index("page two")


def test_build_markdown_collects_captions_and_keeps_them_in_body(tmp_path):
    pages = [
        [
            _region("text", "Body text.", index=0),
            _region("figure_title", "FIGURE 2. A comparison of things.", index=1),
        ]
    ]
    md, _, caps, _ = _build_markdown(pages, {}, tmp_path)
    assert "FIGURE 2. A comparison of things." in md  # still body markdown...
    assert [c.text for c in caps[1]] == ["FIGURE 2. A comparison of things."]  # ...and known
    assert caps[1][0].bbox is None  # fixture region carries no bbox_2d


def test_build_markdown_all_reference_page_emits_no_empty_part(tmp_path):
    # a page whose regions are all references (or boilerplate) contributes only its
    # marker -- no stray empty string producing a triple blank line
    pages = [
        [_region("text", "body", index=0)],
        [_region("reference_content", "Smith, J. (2020).", index=0)],
    ]
    md, _, _, refs = _build_markdown(pages, {}, tmp_path)
    assert "\n\n\n" not in md
    assert md.endswith("<page_number>2</page_number>")
    assert len(refs) == 1


# ---------------------------------------------------------------------------
# _save_figure_crop
# ---------------------------------------------------------------------------


def test_save_figure_crop_warns_and_returns_none_when_missing(tmp_path, caplog):
    region = {"image_path": None}
    with caplog.at_level(logging.WARNING):
        result = _save_figure_crop(region, {}, set(), tmp_path, page=1, idx=0)
    assert result is None
    assert "no cropped image" in caplog.text


# ---------------------------------------------------------------------------
# parse_pdf against a fake backend (offline end-to-end; lifecycle in test_backend.py)
# ---------------------------------------------------------------------------


class _FakeResult:
    def __init__(self, json_result, image_files=None):
        self.json_result = json_result
        self.image_files = image_files or {}


class _FakeParser:
    def __init__(self, result):
        self._result = result

    def parse(self, path):
        return self._result


class _FakeServer:
    def __init__(self):
        self.killed = False

    def kill(self):
        self.killed = True


def test_parse_pdf_with_backend_builds_full_result(tmp_path):
    pages = [
        [
            _region("doc_title", "A Paper", index=0),
            _region("text", "Body text.", index=1),
            _region("reference_content", "1. Smith J (2020) Things.", index=2),
        ]
    ]
    backend = OcrBackend(parser=_FakeParser(_FakeResult(pages)), server=_FakeServer())
    result = parse_pdf(tmp_path / "x.pdf", tmp_path, ParseConfig(), backend=backend)
    assert "# A Paper" in result.markdown
    assert "<page_number>1</page_number>" in result.markdown
    assert "Smith" not in result.markdown
    assert len(result.references) == 1
    assert "1. Smith J (2020) Things." in result.references_markdown


def test_parse_pdf_clears_stale_crops_but_not_foreign_files(tmp_path):
    # crops are derived artifacts: leftovers from a previous run (raw or enrich-renamed
    # naming) would accumulate in the reused work dir and break check-by-filename
    figures_dir = tmp_path / "figures"
    figures_dir.mkdir()
    (figures_dir / "page_2_fig_0.png").write_bytes(b"stale raw")
    (figures_dir / "fig_4.1.png").write_bytes(b"stale renamed")
    (figures_dir / "notes.txt").write_bytes(b"not ours")

    backend = OcrBackend(parser=_FakeParser(_FakeResult([[]])), server=_FakeServer())
    parse_pdf(tmp_path / "x.pdf", tmp_path, ParseConfig(), backend=backend)

    assert not (figures_dir / "page_2_fig_0.png").exists()
    assert not (figures_dir / "fig_4.1.png").exists()
    assert (figures_dir / "notes.txt").exists()  # only our own patterns are touched


def test_parse_pdf_watchdog_kills_server_and_raises(tmp_path):
    import threading

    released = threading.Event()

    class HangingParser:
        def parse(self, path):
            released.wait(timeout=10)  # hangs until the watchdog kills the "server"
            raise ConnectionError("server gone")

    class Server(_FakeServer):
        def kill(self):
            super().kill()
            released.set()

    backend = OcrBackend(parser=HangingParser(), server=Server())
    cfg = ParseConfig(parse_timeout_s=0.05)
    with pytest.raises(RuntimeError, match="exceeded ParseConfig.parse_timeout_s"):
        parse_pdf(tmp_path / "x.pdf", tmp_path, cfg, backend=backend)
    assert backend.server.killed


@pytest.mark.skip(reason="parse_pdf needs a running llama-server + GLM-OCR weights")
def test_parse_pdf_builds_markers_figures_and_references():
    """When run against a real backend: assert one <page_number> per page, figure crops
    saved under image_dir/figures, references routed to ParseResult.references and never
    appearing in .markdown, and boilerplate (headers/footers/page numbers) absent."""
