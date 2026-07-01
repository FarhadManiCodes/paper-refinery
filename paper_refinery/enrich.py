"""Splice figure descriptions into the markdown, anchored on FIGURE captions.

Each figure has a ``FIGURE N.M ...`` caption in the text. The caption is the reliable
anchor: figure/chart placeholders can shift between parser runs, but the caption (number +
page + position) is stable. We find one caption per figure number, group them by the page
they sit on, describe each page's figures with Gemini in a single call (given that page's
crop files), and splice the description right after the caption.
"""

from __future__ import annotations

import re
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import FigureConfig
from .figures import describe_page_figures as _default_describe
from .markers import PAGE_MARKER_RE
from .parse import ParseResult

_CAPTION_LINE = re.compile(
    r"(?m)^[ \t]*\**[ \t]*(FIGURE|Figure)[ \t]+(\d+(?:\.\d+)?)\b[.:]?[ \t]*(.*)$"
)

# (crops, [(figure_number, caption)]) -> {figure_number: description}
PageDescriber = Callable[[list[Path], list[tuple[str, str]], FigureConfig], dict[str, str]]


@dataclass
class _Caption:
    number: str
    text: str
    line_end: int  # offset just past the caption line (where a description is spliced)
    page: int | None
    upper: bool  # True if the line was "FIGURE" (the canonical caption form)


def _page_at(markdown: str, offset: int) -> int | None:
    page = None
    for m in PAGE_MARKER_RE.finditer(markdown):
        if m.start() <= offset:
            page = int(m.group(1))
        else:
            break
    return page


def _find_captions(markdown: str) -> list[_Caption]:
    """One caption per figure number; the uppercase FIGURE line wins if both forms exist."""
    caps: dict[str, _Caption] = {}
    for m in _CAPTION_LINE.finditer(markdown):
        kind, number = m.group(1), m.group(2)
        text = (m.group(3) or "").strip().strip("*").strip()  # drop leaked markdown bold
        upper = kind == "FIGURE"
        cur = caps.get(number)
        if cur is None or (upper and not cur.upper):
            caps[number] = _Caption(number, text, m.end(), _page_at(markdown, m.start()), upper)
    return list(caps.values())


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

    Captions are grouped by page and described one page at a time (a single Gemini call
    per page). The per-page calls are independent and run concurrently. A page that fails
    (after retries) leaves its figures undescribed; splicing is deterministic.
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

    by_page: dict[int | None, list[_Caption]] = defaultdict(list)
    for caption in _find_captions(md):
        by_page[caption.page].append(caption)

    # one describe task per page that has figure/chart crops
    tasks: dict[int, tuple[list[Path], list[tuple[str, str]]]] = {}
    for page, captions in by_page.items():
        crops = parsed.figure_crops.get(page, []) if page is not None else []
        if crops:
            tasks[page] = (crops, [(c.number, context_for(c)) for c in captions])

    results_by_page: dict[int, dict[str, str]] = {}
    if tasks:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(describe, crops, requests, cfg): page
                for page, (crops, requests) in tasks.items()
            }
            for future, page in futures.items():
                try:
                    results_by_page[page] = future.result()
                except Exception as exc:  # surface it; leave this page undescribed
                    warnings.warn(f"figure description failed on page {page}: {exc!r}")
                    results_by_page[page] = {}

    insertions: list[tuple[int, str]] = []
    for page, captions in by_page.items():
        results = results_by_page.get(page, {})
        for caption in captions:
            desc = results.get(caption.number, "")
            if not desc:
                continue
            insertions.append(
                (caption.line_end, f"\n\n> **Figure description (auto):** {desc}")
            )

    # apply back-to-front so earlier offsets stay valid
    for offset, text in sorted(insertions, key=lambda it: it[0], reverse=True):
        md = md[:offset] + text + md[offset:]
    return md
