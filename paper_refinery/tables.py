"""Convert GLM-OCR's HTML tables into markdown pipe tables.

GLM-OCR emits a ``table`` region's content as model-generated HTML (``<table>``/``<tr>``/
``<td>``), not markdown. ``parse.py`` routes those regions here to get a markdown pipe
table back for the body. This is a self-contained ``str -> str`` conversion -- the only
place ``beautifulsoup4`` is used -- kept apart from parse.py's region-classification logic.
"""

from __future__ import annotations

from bs4 import BeautifulSoup, Tag


def _int_attr(cell: Tag, name: str, default: int = 1) -> int:
    """A tag's ``rowspan``/``colspan`` attribute as an int, tolerating a malformed value.

    The source HTML is GLM-OCR's own model-generated output, not hand-authored markup --
    a non-numeric span value is a real (if rare) failure mode, not a "can't happen" input,
    and one bad cell shouldn't crash the whole page's table conversion. (bs4 types an
    attribute as ``str | list | None``; a span is never multi-valued, so a list -- like an
    absent attribute -- falls back to the default.)
    """
    value = cell.get(name)
    if not isinstance(value, str):
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _place_row_cells(tr: Tag, active: dict[int, tuple[int, str]]) -> tuple[list[str], set[int]]:
    """Place the cells that ORIGINATE in this ``<tr>`` into a fresh row.

    Skips columns still occupied by an earlier row's rowspan (tracked in ``active``),
    duplicates a colspan cell across the columns it covers, and registers any rowspan so
    later rows carry it. Returns the row and the set of columns it filled directly.
    """
    row: list[str] = []
    col = 0
    placed_this_row: set[int] = set()
    for cell in tr.find_all(["td", "th"]):
        while active.get(col, (0, ""))[0] > 0:
            col += 1
        text = cell.get_text(" ", strip=True)
        rowspan = _int_attr(cell, "rowspan")
        colspan = _int_attr(cell, "colspan")
        for i in range(colspan):
            c = col + i
            while len(row) <= c:
                row.append("")
            row[c] = text
            placed_this_row.add(c)
            if rowspan > 1:
                active[c] = (rowspan - 1, text)
        col += colspan
    return row, placed_this_row


def _fill_carried_spans(
    row: list[str], active: dict[int, tuple[int, str]], placed_this_row: set[int]
) -> None:
    """Fill columns carried into this row by an earlier row's rowspan (in place).

    Only columns NOT already placed by a cell originating in this row (those were handled
    in ``_place_row_cells``); each carried span's remaining-row counter is decremented.
    """
    max_col = max([len(row)] + [c + 1 for c in active])
    for c in range(max_col):
        if c in placed_this_row:
            continue
        remaining, text = active.get(c, (0, ""))
        if remaining > 0:
            while len(row) <= c:
                row.append("")
            if not row[c]:
                row[c] = text
            active[c] = (remaining - 1, text)


def _build_grid(table: Tag) -> list[list[str]]:
    """Expand an HTML ``<table>`` into a rectangular-ish grid of cell texts, resolving
    row/colspans by duplicating a spanning cell's value into every position it covers."""
    grid: list[list[str]] = []
    active: dict[int, tuple[int, str]] = {}  # col -> (rows_remaining, text)
    for tr in table.find_all("tr"):
        row, placed_this_row = _place_row_cells(tr, active)
        _fill_carried_spans(row, active, placed_this_row)
        grid.append(row)
    return grid


def _render_pipe_table(grid: list[list[str]]) -> str:
    """Render a (ragged) grid as a markdown pipe table: pad rows to a common width, escape
    ``|``/newlines, and emit header + separator + body rows."""
    width = max(len(r) for r in grid)
    grid = [r + [""] * (width - len(r)) for r in grid]

    def esc(s: str) -> str:
        return s.replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(esc(c) for c in grid[0]) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines.extend("| " + " | ".join(esc(c) for c in r) + " |" for r in grid[1:])
    return "\n".join(lines)


def html_table_to_markdown(html: str) -> str:
    """Convert a GLM-OCR HTML table to a markdown pipe table.

    Markdown has no merged-cell concept; a rowspan/colspan cell's value is duplicated
    into every grid position it visually spans rather than silently dropped. This can
    repeat a spanning header across columns/rows it covers -- an accepted, documented
    fidelity tradeoff, not a bug.
    """
    soup = BeautifulSoup(html or "", "html.parser")
    table = soup.find("table")
    if table is None:
        return (html or "").strip()
    grid = _build_grid(table)
    if not grid:
        return ""
    return _render_pipe_table(grid)
