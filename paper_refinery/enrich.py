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
    r"(?m)^[ \t]*\**[ \t]*(FIGURE|Figure|FIG|Fig)\.?[ \t]*(\d+(?:\.\d+)?)\b[.:]?[ \t]*(.*)$"
)

# (crops, [(figure_number, caption)]) -> {figure_number: description}
PageDescriber = Callable[[list[Path], list[tuple[str, str]], FigureConfig], dict[str, str]]


@dataclass
class _Caption:
    number: str
    text: str
    line_end: int  # offset just past the caption line (where a description is spliced)
    page: int | None
    upper: bool  # True if the line's "FIGURE"/"FIG" was ALL-CAPS (the canonical caption form)


def _page_at(markdown: str, offset: int) -> int | None:
    page = None
    for m in PAGE_MARKER_RE.finditer(markdown):
        if m.start() <= offset:
            page = int(m.group(1))
        else:
            break
    return page


def _find_captions(markdown: str) -> list[_Caption]:
    """One caption per figure number; an ALL-CAPS "FIGURE"/"FIG" line wins over a
    lower-case in-text mention if both forms exist."""
    caps: dict[str, _Caption] = {}
    for m in _CAPTION_LINE.finditer(markdown):
        kind, number = m.group(1), m.group(2)
        text = (m.group(3) or "").strip().strip("*").strip()  # drop leaked markdown bold
        upper = kind.isupper()
        cur = caps.get(number)
        if cur is None or (upper and not cur.upper):
            caps[number] = _Caption(number, text, m.end(), _page_at(markdown, m.start()), upper)
    return list(caps.values())


def _rename_crops_to_captions(
    md: str, page: int, crops: list[Path], captions: list[_Caption]
) -> tuple[str, list[Path]]:
    """Pair this page's crops with its captions by reading order (position-based, not
    bbox matching) and rename each crop file to its figure number, e.g.
    ``page_2_fig_0.png`` -> ``fig_4.1.png``. Lets cropping/placement be checked by
    filename alone, before any Gemini call. A crop beyond the caption count (count
    mismatch) keeps its original name -- nothing to pair it with.
    """
    renamed: list[Path] = []
    for i, crop in enumerate(crops):
        if i >= len(captions):
            renamed.append(crop)
            continue
        number = captions[i].number
        safe = re.sub(r"[^\w.-]", "_", number)
        new_path = crop.with_name(f"fig_{safe}{crop.suffix}")
        if new_path != crop and crop.exists():
            crop.replace(new_path)
            md = md.replace(
                f"![FIGURE_CROP {page}:{i}]({crop})",
                f"![FIGURE {number}]({new_path})",
            )
            renamed.append(new_path)
        else:
            renamed.append(crop)
    return md, renamed


def _group_by_page(md: str) -> dict[int | None, list[_Caption]]:
    grouped: dict[int | None, list[_Caption]] = defaultdict(list)
    for caption in _find_captions(md):
        grouped[caption.page].append(caption)
    return grouped


def _find_mentions(markdown: str, number: str) -> list[str]:
    """Body paragraphs that reference 'Figure N'/'Fig. N' (used only when include_references)."""
    ref = re.compile(rf"\b(?:FIGURE|Figure|FIG|Fig)\.?[ \t]*{re.escape(number)}\b")
    cap = re.compile(rf"^[ \t]*\**[ \t]*(?:FIGURE|Figure|FIG|Fig)\.?[ \t]*{re.escape(number)}\b")
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
) -> str:
    """Return the markdown with figure descriptions spliced next to their captions.

    Captions are grouped by page and described one page at a time (a single Gemini call
    per page). The per-page calls are independent and run concurrently. A page that fails
    (after retries) leaves its figures undescribed; splicing is deterministic.
    """
    cfg = cfg or FigureConfig()
    describe = describe or _default_describe
    md = parsed.markdown

    def context_for(caption: _Caption) -> str:
        head = f"FIGURE {caption.number}. {caption.text}".strip()
        if not cfg.include_references:
            return head
        return "\n\n".join([head, *_find_mentions(md, caption.number)])

    # first pass: only to learn page/reading-order for crop-to-caption pairing below.
    # Renaming can change md's length (placeholder text differs from the new image
    # line), which would invalidate any caption.line_end offsets computed against the
    # pre-rename text -- so captions are re-found from scratch afterward, once md is final.
    first_pass = _group_by_page(md)

    # rename each page's crops to their matched figure number, before any Gemini call --
    # lets cropping/placement be checked by filename alone
    figure_crops = dict(parsed.figure_crops)
    for page, captions in first_pass.items():
        if page is None:
            continue
        crops = figure_crops.get(page, [])
        if crops:
            md, figure_crops[page] = _rename_crops_to_captions(md, page, crops, captions)

    by_page = _group_by_page(md)

    # one describe task per page that has figure/chart crops
    tasks: dict[int, tuple[list[Path], list[tuple[str, str]]]] = {}
    for page, captions in by_page.items():
        crops = figure_crops.get(page, []) if page is not None else []
        if crops:
            tasks[page] = (crops, [(c.number, context_for(c)) for c in captions])

    results_by_page: dict[int, dict[str, str]] = {}
    if tasks:
        with ThreadPoolExecutor(max_workers=cfg.max_workers) as pool:
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
