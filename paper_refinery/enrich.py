"""Splice figure descriptions into the markdown, anchored on inline image placeholders.

LlamaParse inlines a ``![alt](src)`` placeholder at each real figure's position, with
the figure's caption (``FIGURE N.M ...``) on the following line. We use those
placeholders as the figure list -- they are exactly the real figures (no table/text
crops, no page-matching, no ties). For each:

  1. read the caption on the next line (splice point + grounding context)
  2. pick the extracted image file for that page (preferring img_* over chart_*)
  3. describe it with Gemini (grounded on the caption); if there is no usable image or
     Gemini flags a non-figure, fall back to LlamaParse's own alt-text
  4. insert the description right after the caption
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import FigureConfig
from .figures import describe_figure as _default_describe
from .parse import Figure, ParseResult

_PAGE = re.compile(r"<page_number>\s*(\d+)\s*</page_number>")
_PLACEHOLDER = re.compile(r"!\[(?P<alt>[^\]]*)\]\([^)]*\)")
_CAPTION_LINE = re.compile(
    r"(?m)^[ \t]*\**[ \t]*(FIGURE|Figure)[ \t]+(\d+(?:\.\d+)?)\b[.:]?[ \t]*(.*)$"
)

Describer = Callable[[Path, str | None, FigureConfig], str]


@dataclass
class _Placeholder:
    alt: str
    end: int  # offset just past the ![...](...) placeholder
    page: int | None


@dataclass
class _Caption:
    number: str
    text: str
    line_end: int  # offset just past the caption line (where a description is spliced)


def _page_at(markdown: str, offset: int) -> int | None:
    page = None
    for m in _PAGE.finditer(markdown):
        if m.start() <= offset:
            page = int(m.group(1))
        else:
            break
    return page


def _find_placeholders(markdown: str) -> list[_Placeholder]:
    return [
        _Placeholder(m.group("alt").strip(), m.end(), _page_at(markdown, m.start()))
        for m in _PLACEHOLDER.finditer(markdown)
    ]


def _caption_after(markdown: str, pos: int, window: int = 600) -> _Caption | None:
    """The first FIGURE caption line within `window` chars after `pos`."""
    m = _CAPTION_LINE.search(markdown, pos, pos + window)
    if not m:
        return None
    text = (m.group(3) or "").strip().strip("*").strip()  # drop leaked markdown bold
    return _Caption(m.group(2), text, m.end())


def _pick_image(
    page: int | None, figures: list[Figure], used: set[int] | None = None
) -> Figure | None:
    """The nearest unused image file to the figure's page; prefer img_* over chart_*."""
    if page is None:
        return None
    used = used or set()
    cands = [
        f for f in figures
        if id(f) not in used and f.page is not None and abs(f.page - page) <= 1
    ]
    # figures float above their captions, so a real img_* one page away beats a
    # chart_* crop on the exact page: prefer img_* first, then page proximity.
    cands.sort(key=lambda f: (0 if f.image_path.name.startswith("img") else 1, abs(f.page - page)))
    return cands[0] if cands else None


def _find_mentions(markdown: str, number: str) -> list[str]:
    """Body paragraphs that reference 'Figure N' (future cross-referencing)."""
    ref = re.compile(rf"\bFigure[ \t]+{re.escape(number)}\b")
    cap = re.compile(rf"^[ \t]*\**[ \t]*(?:FIGURE|Figure)[ \t]+{re.escape(number)}\b")
    seen: set[str] = set()
    out: list[str] = []
    for para in markdown.split("\n\n"):
        p = para.strip()
        if p and ref.search(p) and not cap.match(p) and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def enrich_markdown(
    parsed: ParseResult,
    cfg: FigureConfig | None = None,
    describe: Describer | None = None,
) -> str:
    """Return the markdown with figure descriptions spliced next to their captions."""
    cfg = cfg or FigureConfig()
    describe = describe or _default_describe
    md = parsed.markdown

    def context_for(caption: _Caption) -> str:
        head = f"FIGURE {caption.number}. {caption.text}".strip()
        if not cfg.include_references:
            return head
        return "\n\n".join([head, *_find_mentions(md, caption.number)])

    insertions: list[tuple[int, str]] = []
    used: set[int] = set()
    for ph in _find_placeholders(md):
        caption = _caption_after(md, ph.end)
        image = _pick_image(ph.page, parsed.figures, used)
        if image is not None:
            used.add(id(image))
        desc = describe(image.image_path, context_for(caption) if caption else ph.alt, cfg) if image else ""
        if not desc:  # no usable image, or Gemini judged a non-figure -> LlamaParse alt-text
            desc = ph.alt
        if not desc:
            continue
        anchor = caption.line_end if caption else ph.end
        insertions.append((anchor, f"\n\n> **Figure description (auto):** {desc}"))

    # apply back-to-front so earlier offsets stay valid
    for offset, text in sorted(insertions, key=lambda it: it[0], reverse=True):
        md = md[:offset] + text + md[offset:]
    return md
