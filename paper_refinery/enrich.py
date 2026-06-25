"""Splice figure descriptions into the markdown at the correct location.

For each extracted image:
  1. find the nearest ``FIGURE N.M`` caption (by page) -> gives splice point + context
  2. gather the in-text ``Figure N.M`` references -> context grounding
  3. ask Gemini to describe it (non-figures, e.g. tables/text crops, return "" -> dropped)
  4. insert the description right after the caption

This also filters LlamaParse's noisy chart crops: a table/text image either matches no
figure caption or Gemini flags it as a non-figure, so it is never spliced in.
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
_CAPTION_LINE = re.compile(
    r"(?m)^[ \t]*\**[ \t]*(FIGURE|Figure)[ \t]+(\d+(?:\.\d+)?)\b[.:]?[ \t]*(.*)$"
)

Describer = Callable[[Path, str | None, FigureConfig], str]


@dataclass
class _Caption:
    number: str
    text: str
    line_end: int  # offset just past the caption line (where a description is spliced)
    page: int | None


def _page_at(markdown: str, offset: int) -> int | None:
    page = None
    for m in _PAGE.finditer(markdown):
        if m.start() <= offset:
            page = int(m.group(1))
        else:
            break
    return page


def _find_captions(markdown: str) -> list[_Caption]:
    """One caption per figure number; the uppercase FIGURE line wins if both exist."""
    caps: dict[str, _Caption] = {}
    for m in _CAPTION_LINE.finditer(markdown):
        kind, number = m.group(1), m.group(2)
        text = (m.group(3) or "").strip().strip("*").strip()  # drop leaked markdown bold
        if caps.get(number) is None or kind == "FIGURE":
            caps[number] = _Caption(number, text, m.end(), _page_at(markdown, m.start()))
    return list(caps.values())


def _find_mentions(markdown: str, number: str) -> list[str]:
    """Paragraphs that reference 'Figure N' in the body (excluding the caption line)."""
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


def _nearest_caption(page: int | None, captions: list[_Caption]) -> _Caption | None:
    """Nearest caption within +-1 page, used to ground the Gemini description."""
    if page is None:
        return None
    cands = [c for c in captions if c.page is not None and abs(c.page - page) <= 1]
    return min(cands, key=lambda c: abs(c.page - page)) if cands else None


def _assign_caption(
    page: int | None, captions: list[_Caption], used: set[int]
) -> _Caption | None:
    """Nearest *unused* caption (any distance), used to place the description. Greedy
    assignment in page order resolves ties and multiple-figures-per-page correctly."""
    if page is None:
        return None
    cands = [c for c in captions if id(c) not in used and c.page is not None]
    return min(cands, key=lambda c: abs(c.page - page)) if cands else None


def _build_context(caption: _Caption, mentions: list[str]) -> str:
    head = f"FIGURE {caption.number}. {caption.text}".strip()
    return "\n\n".join(dict.fromkeys(p for p in (head, *mentions) if p))


def enrich_markdown(
    parsed: ParseResult,
    cfg: FigureConfig | None = None,
    describe: Describer | None = None,
) -> str:
    """Return the markdown with figure descriptions spliced next to their captions."""
    cfg = cfg or FigureConfig()
    describe = describe or _default_describe
    md = parsed.markdown
    captions = _find_captions(md)
    mentions: dict[str, list[str]] = {}

    def context_for(caption: _Caption | None) -> str | None:
        """Caption-only by default; in-text references added when cfg.include_references."""
        if caption is None:
            return None
        if not cfg.include_references:
            return f"FIGURE {caption.number}. {caption.text}".strip()
        mentions.setdefault(caption.number, _find_mentions(md, caption.number))
        return _build_context(caption, mentions[caption.number])

    # 1) filter: describe each candidate with its nearest-caption context; keep the real
    #    figures -- non-figures (tables/text crops) come back empty and are dropped.
    survivors: list[tuple[Figure, _Caption | None, str]] = []  # (figure, ground, desc)
    for fig in parsed.figures:
        ground = _nearest_caption(fig.page, captions)
        desc = describe(fig.image_path, context_for(ground), cfg)
        if desc:
            survivors.append((fig, ground, desc))

    # 2) place: greedily assign each survivor (in page order) to its nearest unused
    #    caption, so ties / multiple-figures-per-page resolve correctly. If the assigned
    #    caption differs from the grounding one (a tie), re-describe with the right one.
    survivors.sort(key=lambda s: s[0].page if s[0].page is not None else 1 << 30)
    used: set[int] = set()
    insertions: list[tuple[int, str]] = []
    for fig, ground, desc in survivors:
        caption = _assign_caption(fig.page, captions, used)
        if caption is not None:
            used.add(id(caption))
            if caption is not ground:
                desc = describe(fig.image_path, context_for(caption), cfg) or desc
        anchor = caption.line_end if caption else len(md)
        insertions.append((anchor, f"\n\n> **Figure description (auto):** {desc}"))

    # apply back-to-front so earlier offsets stay valid
    for offset, text in sorted(insertions, key=lambda it: it[0], reverse=True):
        md = md[:offset] + text + md[offset:]
    return md
