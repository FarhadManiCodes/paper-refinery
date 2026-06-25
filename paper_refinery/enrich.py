"""Splice figure descriptions into the markdown, anchored on inline image placeholders.

LlamaParse inlines a ``![alt](src)`` placeholder at each real figure's position, with
the figure's caption (``FIGURE N.M ...``) on the following line. For each placeholder we
send Gemini the **full-page render** for that figure's page (the caption tells it which
figure to describe) and splice the description after the caption.

This deliberately avoids LlamaParse's per-figure image crops, which are unreliable (they
miss some figures and mis-classify text as charts). The page render always contains the
figure, so this works uniformly -- including figures LlamaParse failed to crop.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import FigureConfig
from .figures import describe_figure as _default_describe
from .parse import ParseResult

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


def _find_mentions(markdown: str, number: str) -> list[str]:
    """Body paragraphs that reference 'Figure N' (used only when include_references)."""
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
    mentions: dict[str, list[str]] = {}

    def context_for(caption: _Caption) -> str:
        head = f"FIGURE {caption.number}. {caption.text}".strip()
        if not cfg.include_references:
            return head
        mentions.setdefault(caption.number, _find_mentions(md, caption.number))
        return "\n\n".join([head, *mentions[caption.number]])

    insertions: list[tuple[int, str]] = []
    for ph in _find_placeholders(md):
        caption = _caption_after(md, ph.end)
        render = parsed.page_renders.get(ph.page) if ph.page is not None else None
        context = context_for(caption) if caption else ph.alt
        desc = describe(render, context, cfg) if render else ""
        if not desc:  # no page render, or Gemini found no matching figure
            desc = ph.alt  # fall back to LlamaParse's own alt-text
        if not desc:
            continue
        anchor = caption.line_end if caption else ph.end
        insertions.append((anchor, f"\n\n> **Figure description (auto):** {desc}"))

    # apply back-to-front so earlier offsets stay valid
    for offset, text in sorted(insertions, key=lambda it: it[0], reverse=True):
        md = md[:offset] + text + md[offset:]
    return md
