"""Tests for geometric reading-order repair (``reading_order.py``), against hand-built
glmocr-shaped region dicts with real-world bbox layouts."""

from __future__ import annotations

from paper_refinery.reading_order import reading_order


def _box_region(index: int, bbox: list[int], text: str = "") -> dict:
    return {"native_label": "text", "content": text, "index": index, "bbox_2d": bbox}


def test_reading_order_fixes_in_column_inversion():
    # the real kalman page-8 bug: index order put "Theorem 4" (y=436) before the
    # "Fig. 4" caption (y=381) it follows in the same (right) column. The left-column
    # region establishes the true page width, as on any real two-column page.
    regions = [
        _box_region(0, [100, 68, 500, 900], "left column"),
        _box_region(1, [512, 234, 901, 382], "figure"),
        _box_region(2, [512, 436, 908, 461], "Theorem 4..."),
        _box_region(3, [571, 381, 849, 393], "Fig. 4 caption"),
        _box_region(4, [512, 398, 908, 435], "Comparing equations..."),
    ]
    out = reading_order(regions)
    assert [r["index"] for r in out] == [0, 1, 3, 4, 2]


def test_reading_order_keeps_correct_pages_unchanged():
    regions = [
        _box_region(0, [100, 68, 500, 92]),
        _box_region(1, [100, 94, 500, 242]),
        _box_region(2, [101, 242, 498, 317]),
    ]
    assert [r["index"] for r in reading_order(regions)] == [0, 1, 2]


def test_reading_order_never_interleaves_columns():
    # macro order (left column fully, then right) is glmocr's job and must survive,
    # even though the right column starts higher on the page than the left one ends
    regions = [
        _box_region(0, [100, 68, 500, 400], "left top"),
        _box_region(1, [100, 410, 500, 800], "left bottom"),
        _box_region(2, [512, 68, 908, 400], "right top"),
        _box_region(3, [512, 410, 908, 800], "right bottom"),
    ]
    assert [r["index"] for r in reading_order(regions)] == [0, 1, 2, 3]


def test_reading_order_wide_region_breaks_runs():
    # a page-wide region (title, spanning table) separates runs: text above and below
    # it is never re-sorted across it
    regions = [
        _box_region(0, [100, 300, 500, 400], "left col after title"),
        _box_region(1, [100, 68, 908, 120], "PAGE-WIDE TITLE"),
        _box_region(2, [100, 410, 500, 500], "more left col"),
    ]
    # the wide region is its own run; the sort must not pull idx=1 above idx=0's run
    out = reading_order(regions)
    assert [r["index"] for r in out] == [0, 1, 2]


def test_reading_order_same_line_regions_keep_index_order():
    # confirmed live: one formula split into two side-by-side regions 1px apart --
    # exact-y sorting would swap what glmocr ordered correctly
    regions = [
        _box_region(0, [514, 408, 767, 423], "Pr[x(t_{n+1}) <="),
        _box_region(1, [565, 407, 857, 471], "xi_1) <= xi_{n+..."),
    ]
    assert [r["index"] for r in reading_order(regions)] == [0, 1]


def test_reading_order_missing_bbox_falls_back_to_index_order():
    regions = [
        {"native_label": "text", "content": "b", "index": 1},
        {"native_label": "text", "content": "a", "index": 0},
    ]
    assert [r["index"] for r in reading_order(regions)] == [0, 1]
