"""Geometric reading-order repair for glmocr's per-page region lists.

PP-DocLayout-V3's ``index`` field gets the macro order right (which column when) but is
frequently wrong *within* a column. ``reading_order`` re-sorts each contiguous run of
same-column regions by their top edge, leaving everything else in ``index`` order -- the
known root cause of body/figure ordering quirks, isolated here from parse.py's
region-classification logic. ``region_bbox`` is the shared ``bbox_2d`` accessor (parse.py
reuses it to attach bboxes to its crop/caption regions).
"""

from __future__ import annotations

from functools import cmp_to_key

_WIDE_FRACTION = 0.6  # of page width: a region this wide spans columns (title, wide table)
_COLUMN_OVERLAP = 0.5  # of the narrower region's width: x-overlap needed to share a column
_Y_TOL_FRACTION = 0.01  # of page width: y-difference treated as "same line" (~half a line)


type _Box = tuple[float, float, float, float]  # (x1, y1, x2, y2) in page pixels


def region_bbox(region: dict) -> _Box | None:
    """The region's ``bbox_2d`` as an (x1, y1, x2, y2) tuple; None unless well-formed."""
    box = region.get("bbox_2d")
    if isinstance(box, (list, tuple)) and len(box) == 4:
        x1, y1, x2, y2 = (float(v) for v in box)
        return (x1, y1, x2, y2)
    return None


def reading_order(regions: list[dict]) -> list[dict]:
    """Regions in reading order: glmocr's own ``index`` order, with local in-column
    inversions repaired by geometry.

    PP-DocLayout-V3's ``index`` gets the macro order right (which column when), but is
    frequently wrong *within* a column -- measured live on kalman-1960.pdf: 11 genuine
    inversions across 12 pages, e.g. "Theorem 4" emitted two regions before the "Fig. 4"
    caption it follows on the page, and bibliography entries swapped pairwise. Within one
    column, top-to-bottom *is* reading order, so each contiguous run of same-column
    regions (x-overlap > ``_COLUMN_OVERLAP`` of the narrower one) is re-sorted by its
    top edge. Deliberately conservative everywhere else:

    - regions wider than ``_WIDE_FRACTION`` of the page (titles, column-spanning
      tables/figures) break runs, so left/right-column text can never interleave;
    - y-differences within ``_Y_TOL_FRACTION`` of page width count as the same line
      (confirmed live: one formula split into two side-by-side regions 1px apart --
      exact-y sorting would swap what glmocr ordered correctly);
    - any region missing a well-formed ``bbox_2d`` -> plain index order, unchanged.
    """
    regs = sorted(regions, key=lambda r: r.get("index", 0))
    boxes = [region_bbox(r) for r in regs]
    if not boxes or any(b is None for b in boxes):
        return regs
    boxes = [b for b in boxes if b is not None]  # all well-formed past the guard above
    page_w = (max(b[2] for b in boxes) - min(b[0] for b in boxes)) or 1
    y_tol = _Y_TOL_FRACTION * page_w

    def is_wide(box: _Box) -> bool:
        return (box[2] - box[0]) > _WIDE_FRACTION * page_w

    def same_column(a: _Box, b: _Box) -> bool:
        overlap = min(a[2], b[2]) - max(a[0], b[0])
        return overlap > _COLUMN_OVERLAP * min(a[2] - a[0], b[2] - b[0])

    def by_top(a: dict, b: dict) -> int:
        # same line (within tolerance) -> 0, so the stable sort keeps glmocr's order.
        # The tolerance makes this comparator non-transitive by design; validated live
        # and preferred over line-bucketing -- see the module docstring.
        dy = a["bbox_2d"][1] - b["bbox_2d"][1]
        if abs(dy) <= y_tol:
            return 0
        return -1 if dy < 0 else 1

    ordered: list[dict] = []
    run: list[dict] = []
    for region in regs:
        box = region["bbox_2d"]
        if (
            run
            and not is_wide(box)
            and not is_wide(run[-1]["bbox_2d"])
            and same_column(run[-1]["bbox_2d"], box)
        ):
            run.append(region)
        else:
            ordered.extend(sorted(run, key=cmp_to_key(by_top)))
            run = [region]
    ordered.extend(sorted(run, key=cmp_to_key(by_top)))
    return ordered
