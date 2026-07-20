"""Tests for pdf_split.py: splitting a too-long PDF for OCR and gluing the resulting
per-part ParseResults back into one, so the rest of the pipeline never has to know a
split happened."""

import fitz
import pytest

from paper_refinery.parse import CaptionRegion, CropRegion, ParseResult
from paper_refinery.pdf_split import merge_parse_results, page_count, split_pdf


def _make_pdf(path, n_pages):
    doc = fitz.open()
    for _ in range(n_pages):
        doc.new_page()
    doc.save(path)
    doc.close()


def test_page_count_reads_real_page_count(tmp_path):
    pdf = tmp_path / "p.pdf"
    _make_pdf(pdf, 7)
    assert page_count(pdf) == 7


def test_split_pdf_returns_original_path_when_already_under_threshold(tmp_path):
    pdf = tmp_path / "p.pdf"
    _make_pdf(pdf, 40)
    out_dir = tmp_path / "parts"
    parts = split_pdf(pdf, out_dir, max_pages=100)
    assert parts == [(pdf, 40)]
    assert not out_dir.exists()  # no copy made for the common case


def test_split_pdf_exact_multiple(tmp_path):
    pdf = tmp_path / "p.pdf"
    _make_pdf(pdf, 200)
    parts = split_pdf(pdf, tmp_path / "parts", max_pages=100)
    assert [n for _, n in parts] == [100, 100]
    assert [p.name for p, _ in parts] == ["part_0.pdf", "part_1.pdf"]
    for part_path, n in parts:
        assert page_count(part_path) == n


def test_split_pdf_remainder(tmp_path):
    pdf = tmp_path / "p.pdf"
    _make_pdf(pdf, 250)
    parts = split_pdf(pdf, tmp_path / "parts", max_pages=100)
    assert [n for _, n in parts] == [100, 100, 50]
    for part_path, n in parts:
        assert page_count(part_path) == n


def test_split_pdf_single_page_over_threshold_by_one(tmp_path):
    pdf = tmp_path / "p.pdf"
    _make_pdf(pdf, 101)
    parts = split_pdf(pdf, tmp_path / "parts", max_pages=100)
    assert [n for _, n in parts] == [100, 1]


def test_split_pdf_is_byte_deterministic(tmp_path):
    # MuPDF stamps a fresh random trailer /ID on every save unless no_new_id=True, so the
    # same source pages otherwise hash differently each run -- which (before the fix) made
    # parse_cache's checkpoint never hit for any split (>100-page) book. See split_pdf.
    import hashlib

    pdf = tmp_path / "p.pdf"
    _make_pdf(pdf, 150)

    def part_hashes(out_dir):
        parts = split_pdf(pdf, out_dir, max_pages=100)
        return [hashlib.sha256(p.read_bytes()).hexdigest() for p, _ in parts]

    assert part_hashes(tmp_path / "run_a") == part_hashes(tmp_path / "run_b")


def _crop(tmp_path, name):
    path = tmp_path / name
    path.write_bytes(b"fake-png")
    return path


def test_merge_parse_results_offsets_page_markers(tmp_path):
    part_a = ParseResult(markdown="<page_number>1</page_number>\n\nfoo")
    part_b = ParseResult(markdown="<page_number>1</page_number>\n\nbar")
    merged = merge_parse_results([part_a, part_b], [100, 50], tmp_path / "figures")
    assert "<page_number>1</page_number>" in merged.markdown
    assert "<page_number>101</page_number>" in merged.markdown
    assert "foo" in merged.markdown and "bar" in merged.markdown


def test_merge_parse_results_renames_crops_without_collision(tmp_path):
    crops_a = tmp_path / "crops_a"
    crops_b = tmp_path / "crops_b"
    crops_a.mkdir()
    crops_b.mkdir()
    # both parts independently restart crop numbering at their own page 1
    crop_a = _crop(crops_a, "page_1_fig_0.png")
    crop_b = _crop(crops_b, "page_1_fig_0.png")
    part_a = ParseResult(
        markdown=f"m\n\n![FIGURE_CROP 1:0]({crop_a})",
        figure_crops={1: [CropRegion(crop_a, (0, 0, 1, 1))]},
    )
    part_b = ParseResult(
        markdown=f"m\n\n![FIGURE_CROP 1:0]({crop_b})",
        figure_crops={1: [CropRegion(crop_b, (0, 0, 1, 1))]},
    )
    figures_dir = tmp_path / "figures"
    merged = merge_parse_results([part_a, part_b], [100, 50], figures_dir)

    assert set(merged.figure_crops) == {1, 101}
    a_path = merged.figure_crops[1][0].path
    b_path = merged.figure_crops[101][0].path
    assert a_path.name == "page_1_fig_0.png"
    assert b_path.name == "page_101_fig_0.png"
    assert a_path.parent == figures_dir and b_path.parent == figures_dir
    assert a_path.exists() and b_path.exists()
    assert a_path != b_path  # no collision

    # the invariant enrich.py's _rename_crops depends on: for every merged crop, the
    # EXACT placeholder text it will search-and-replace against must actually be
    # present in the merged markdown, pointing at the NEW (post-move) path and the
    # NEW (global) page number -- not the stale pre-merge one.
    assert f"![FIGURE_CROP 1:0]({a_path})" in merged.markdown
    assert f"![FIGURE_CROP 101:0]({b_path})" in merged.markdown
    assert str(crop_a) not in merged.markdown
    assert str(crop_b) not in merged.markdown


def test_merge_parse_results_offsets_reference_pages(tmp_path):
    part_a = ParseResult(markdown="m")
    part_b = ParseResult(markdown="m", references=[{"page": 3, "number": "1", "text": "Some Ref"}])
    merged = merge_parse_results([part_a, part_b], [100, 50], tmp_path / "figures")
    assert merged.references == [{"page": 103, "number": "1", "text": "Some Ref"}]


def test_merge_parse_results_offsets_caption_pages(tmp_path):
    part_a = ParseResult(markdown="m")
    part_b = ParseResult(markdown="m", figure_captions={2: [CaptionRegion("FIGURE 1.", None)]})
    merged = merge_parse_results([part_a, part_b], [100, 50], tmp_path / "figures")
    assert list(merged.figure_captions) == [102]
    assert merged.figure_captions[102][0].text == "FIGURE 1."


def test_merge_parse_results_concatenates_references_markdown(tmp_path):
    part_a = ParseResult(markdown="m", references_markdown="")  # e.g. no bibliography in part 1
    part_b = ParseResult(markdown="m", references_markdown="[1] Some Ref")
    merged = merge_parse_results([part_a, part_b], [100, 50], tmp_path / "figures")
    assert merged.references_markdown == "[1] Some Ref"


@pytest.mark.parametrize(
    ("crops", "expected_page"),
    [({}, None)],  # a part contributing zero crops must not appear in the merged dict at all
)
def test_merge_parse_results_skips_empty_crop_pages(tmp_path, crops, expected_page):
    part = ParseResult(markdown="m", figure_crops=crops)
    merged = merge_parse_results([part], [10], tmp_path / "figures")
    assert merged.figure_crops == {}
