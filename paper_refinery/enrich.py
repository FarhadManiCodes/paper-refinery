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
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import FigureConfig
from .figures import describe_page_figures as _default_describe
from .markers import PAGE_MARKER_RE
from .parse import ParseResult

_PLACEHOLDER = re.compile(r"!\[(?P<alt>[^\]]*)\]\([^)]*\)")
_CAPTION_LINE = re.compile(
    r"(?m)^[ \t]*\**[ \t]*(FIGURE|Figure)[ \t]+(\d+(?:\.\d+)?)\b[.:]?[ \t]*(.*)$"
)

# (page_render, [(figure_number, caption)]) -> {figure_number: description}
PageDescriber = Callable[[Path, list[tuple[str, str]], FigureConfig], dict[str, str]]


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
    for m in PAGE_MARKER_RE.finditer(markdown):
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
    describe: PageDescriber | None = None,
    max_workers: int = 4,
) -> str:
    """Return the markdown with figure descriptions spliced next to their captions.

    Figures are grouped by page and described one page at a time (a single Gemini call
    per page). The per-page calls are independent, so they run concurrently. Splicing is
    deterministic regardless of completion order.
    """
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

    # group each figure (placeholder + its caption) by the page it sits on
    by_page: dict[int | None, list[tuple[_Placeholder, _Caption | None]]] = defaultdict(list)
    for ph in _find_placeholders(md):
        by_page[ph.page].append((ph, _caption_after(md, ph.end)))

    # one describe task per page that has a render and at least one caption
    tasks: dict[int, tuple[Path, list[tuple[str, str]]]] = {}
    for page, items in by_page.items():
        render = parsed.page_renders.get(page) if page is not None else None
        requests = [(cap.number, context_for(cap)) for _, cap in items if cap]
        if render is not None and requests:
            tasks[page] = (render, requests)

    # run the page calls concurrently; one page failing falls back to alt-text, not the doc
    results_by_page: dict[int, dict[str, str]] = {}
    if tasks:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(describe, render, requests, cfg): page
                for page, (render, requests) in tasks.items()
            }
            for future, page in futures.items():
                try:
                    results_by_page[page] = future.result()
                except Exception:
                    results_by_page[page] = {}

    # splice deterministically (document order; offsets applied back-to-front)
    insertions: list[tuple[int, str]] = []
    for page, items in by_page.items():
        results = results_by_page.get(page, {})
        for ph, caption in items:
            desc = results.get(caption.number, "") if caption else ""
            if not desc:  # no render, Gemini missed it, or the page call failed
                desc = ph.alt  # fall back to LlamaParse's own alt-text
            if not desc:
                continue
            anchor = caption.line_end if caption else ph.end
            insertions.append((anchor, f"\n\n> **Figure description (auto):** {desc}"))

    for offset, text in sorted(insertions, key=lambda it: it[0], reverse=True):
        md = md[:offset] + text + md[offset:]
    return md
