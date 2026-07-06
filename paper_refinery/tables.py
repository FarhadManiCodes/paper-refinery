"""Convert GLM-OCR's HTML tables into markdown pipe tables.

GLM-OCR emits a ``table`` region's content as model-generated HTML (``<table>``/``<tr>``/
``<td>``), not markdown. ``parse.py`` routes those regions here to get a markdown pipe
table back for the body. This is a self-contained ``str -> str`` conversion -- the only
place ``beautifulsoup4`` is used -- kept apart from parse.py's region-classification logic.
"""

from __future__ import annotations

from bs4 import BeautifulSoup


def _int_attr(cell, name: str, default: int = 1) -> int:
    """A tag's ``rowspan``/``colspan`` attribute as an int, tolerating a malformed value.

    The source HTML is GLM-OCR's own model-generated output, not hand-authored markup --
    a non-numeric span value is a real (if rare) failure mode, not a "can't happen" input,
    and one bad cell shouldn't crash the whole page's table conversion.
    """
    try:
        return int(cell.get(name, default) or default)
    except (TypeError, ValueError):
        return default


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

    grid: list[list[str]] = []
    active: dict[int, tuple[int, str]] = {}  # col -> (rows_remaining, text)

    for tr in table.find_all("tr"):
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

        # fill in columns carried over by an earlier row's rowspan (not one that
        # originated in this row -- that was already placed above)
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

        grid.append(row)

    if not grid:
        return ""

    width = max(len(r) for r in grid)
    grid = [r + [""] * (width - len(r)) for r in grid]

    def esc(s: str) -> str:
        return s.replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(esc(c) for c in grid[0]) + " |"]
    lines.append("| " + " | ".join(["---"] * width) + " |")
    for r in grid[1:]:
        lines.append("| " + " | ".join(esc(c) for c in r) + " |")
    return "\n".join(lines)
