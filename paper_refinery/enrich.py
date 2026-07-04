"""Splice figure descriptions into the markdown, anchored on FIGURE captions.

Each figure has a ``FIGURE N.M ...`` caption in the text. The caption is the reliable
anchor: figure/chart placeholders can shift between parser runs, but the caption (number +
page + position) is stable. We find one caption per figure number, pair each caption with
its crop(s) *geometrically* (nearest caption with x-overlap, using the bboxes parse.py
carries on ``CropRegion``/``CaptionRegion``), describe each figure in its own Gemini call
(all panels of a multi-panel figure together), and splice the description right after the
caption. Positional pairing survives only as the fallback for papers where the layout
model produced no usable geometry.

When the parse provides layout-model-detected captions (``ParseResult.figure_captions``,
from PP-DocLayout-V3's figure_title regions), only body lines matching one of those are
accepted as anchors -- a body paragraph starting "Figure 4 shows..." can no longer steal
a caption's anchor. The caption regex still does the number extraction either way, and
remains the sole mechanism for papers where the layout model detected no captions at all.

Per-figure context (title, abstract, neighboring paragraphs) is assembled here and handed
to figures.py, which constrains its use to dictionary-only lookups -- see the prompt in
``FigureConfig``.
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
from .figures import describe_figure as _default_describe
from .markers import PAGE_MARKER_RE
from .parse import CaptionRegion, CropRegion, ParseResult

_CAPTION_LINE = re.compile(
    r"(?m)^[ \t]*\**[ \t]*(FIGURE|Figure|FIG|Fig)\.?[ \t]*(\d+(?:\.\d+)?)\b[.:]?[ \t]*(.*)$"
)

# (figure's crops, figure number, caption text, context, cfg) -> description dict or None
FigureDescriber = Callable[[list[Path], str, str, dict, FigureConfig], "dict | None"]


@dataclass
class _Caption:
    number: str
    text: str  # caption text after the number
    line: str  # the whole matched caption line (for matching against layout regions)
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


def _normalize_caption(text: str) -> str:
    return " ".join(text.split()).strip("* ").strip()


def _matches_known(line: str, known: set[str]) -> bool:
    """True if this matched line is (the start of) a layout-model-detected caption.

    Prefix matching in both directions, not just equality: a multi-line caption region
    lands in the markdown with its first line matching the regex, so the line is a
    prefix of the known caption text (never the other way around for a genuine match;
    an in-text mention is a full sentence that prefixes no real caption).
    """
    line = _normalize_caption(line)
    return any(k.startswith(line) or line.startswith(k) for k in known)


def _known_numbers(known: set[str]) -> set[str]:
    """Figure numbers that have a layout-model-detected caption (parsed off the known
    caption texts with the same regex)."""
    numbers: set[str] = set()
    for text in known:
        m = _CAPTION_LINE.match(text)
        if m:
            numbers.add(m.group(2))
    return numbers


def _find_captions(markdown: str, known: set[str] | None = None) -> list[_Caption]:
    """One caption per figure number; an ALL-CAPS "FIGURE"/"FIG" line wins over a
    lower-case in-text mention if both forms exist. With ``known`` (normalized
    figure_title texts from the parse), non-matching lines are rejected -- but only
    for numbers the layout model actually produced a caption region for. A number
    with NO known caption anywhere falls back to pure regex acceptance: the layout
    model sometimes misses one caption entirely (live: brunton's "Fig. 3." on page 5
    was plain text to PP-DocLayout-V3, and the strict filter orphaned all six of that
    figure's panel crops). Line-anchored matching still keeps mid-paragraph mentions
    out, and the anti-stealing guarantee is unchanged wherever a region exists."""
    caps: dict[str, _Caption] = {}
    covered = _known_numbers(known) if known else set()
    for m in _CAPTION_LINE.finditer(markdown):
        kind, number = m.group(1), m.group(2)
        if known and number in covered and not _matches_known(m.group(0), known):
            continue
        text = (m.group(3) or "").strip().strip("*").strip()  # drop leaked markdown bold
        upper = kind.isupper()
        cur = caps.get(number)
        if cur is None or (upper and not cur.upper):
            caps[number] = _Caption(
                number, text, m.group(0), m.end(), _page_at(markdown, m.start()), upper
            )
    return list(caps.values())


def _group_by_page(md: str, known: set[str] | None = None) -> dict[int | None, list[_Caption]]:
    grouped: dict[int | None, list[_Caption]] = defaultdict(list)
    for caption in _find_captions(md, known):
        grouped[caption.page].append(caption)
    return grouped


def _caption_bbox(
    caption: _Caption, regions: list[CaptionRegion]
) -> tuple[float, float, float, float] | None:
    """The layout-model bbox for a markdown-found caption, matched by text prefix
    (same rule as ``_matches_known``); None when no region matches or it has no box."""
    line = _normalize_caption(caption.line)
    for region in regions:
        text = _normalize_caption(region.text)
        if text.startswith(line) or line.startswith(text):
            return region.bbox
    return None


def _pair_crops(
    crops: list[CropRegion],
    caption_bboxes: list[tuple[float, float, float, float] | None],
) -> dict[int, list[int]]:
    """Assign each crop to a caption geometrically: caption index -> [crop indices].

    A crop belongs to the caption nearest by vertical edge gap among captions with any
    x-overlap (captions sit directly above/below their figure; horizontal-center
    distance breaks ties). Captions naturally collect several crops -- that IS the
    multi-panel case. A crop overlapping no caption horizontally (a banner, a logo)
    stays unassigned: no rename, no Gemini call. When any bbox is missing, geometry
    can't be trusted: a single-caption page assigns ALL crops to that caption (the
    multi-panel assumption -- live: brunton's regex-recovered "Fig. 3." has no region
    bbox, and its six panel crops have nothing else on the page to belong to);
    multi-caption pages fall back to positional pairing (crop i <-> caption i).
    """
    if not crops or not caption_bboxes:
        return {}
    if any(c.bbox is None for c in crops) or any(b is None for b in caption_bboxes):
        if len(caption_bboxes) == 1:
            return {0: list(range(len(crops)))}
        return {i: [i] for i in range(min(len(crops), len(caption_bboxes)))}
    assignment: dict[int, list[int]] = defaultdict(list)
    for ci, crop in enumerate(crops):
        best, best_key = None, None
        for ki, box in enumerate(caption_bboxes):
            overlap = min(crop.bbox[2], box[2]) - max(crop.bbox[0], box[0])
            if overlap <= 0:
                continue
            gap = max(box[1] - crop.bbox[3], crop.bbox[1] - box[3], 0.0)
            centers = abs((crop.bbox[0] + crop.bbox[2]) - (box[0] + box[2])) / 2
            if best_key is None or (gap, centers) < best_key:
                best, best_key = ki, (gap, centers)
        if best is not None:
            assignment[best].append(ci)
    return dict(assignment)


def _rename_crops(
    md: str,
    page: int,
    crops: list[CropRegion],
    assignment: dict[int, list[int]],
    captions: list[_Caption],
) -> tuple[str, dict[int, Path]]:
    """Rename each assigned crop to its figure number (``fig_4.1.png``; panels of a
    multi-crop figure get ``fig_4.1_1.png``, ``fig_4.1_2.png``, ...) and rewrite the
    markdown placeholder. Lets cropping/pairing be checked by filename alone, before
    any Gemini call. Unassigned crops keep their original name and placeholder.
    """
    renamed: dict[int, Path] = {}
    for ki, crop_indices in assignment.items():
        number = captions[ki].number
        safe = re.sub(r"[^\w.-]", "_", number)
        multi = len(crop_indices) > 1
        for panel, ci in enumerate(sorted(crop_indices), start=1):
            crop = crops[ci].path
            suffix = f"_{panel}" if multi else ""
            new_path = crop.with_name(f"fig_{safe}{suffix}{crop.suffix}")
            if new_path != crop and crop.exists():
                crop.replace(new_path)
                md = md.replace(
                    f"![FIGURE_CROP {page}:{ci}]({crop})",
                    f"![FIGURE {number}]({new_path})",
                )
                renamed[ci] = new_path
            else:
                renamed[ci] = crop
    return md, renamed


_SKIP_PREFIXES = ("#", "!", "$$", ">", "|")  # headings, images, math, quotes, tables


def _is_prose(paragraph: str) -> bool:
    p = paragraph.strip()
    return bool(p) and not (
        PAGE_MARKER_RE.fullmatch(p) or p.startswith(_SKIP_PREFIXES) or _CAPTION_LINE.match(p)
    )


def _paper_header(md: str) -> dict:
    """Best-effort ``{title, abstract}`` from the front matter: the first H1 line and
    the first long prose paragraph (abstracts are long; author lists and affiliations
    are short). Empty strings when absent -- figures.py omits empty context fields."""
    title, abstract = "", ""
    for paragraph in md.split("\n\n"):
        p = paragraph.strip()
        if not title and p.startswith("# "):
            title = _normalize_caption(p.lstrip("# "))
            continue
        if not _is_prose(p):
            continue
        if len(p) >= 300:
            abstract = " ".join(p.split())[:2000]
            break
    return {"title": title, "abstract": abstract}


def _neighbor_context(md: str, caption: _Caption, n: int) -> dict:
    """``{before, after}``: up to ``n`` prose paragraphs on each side of the caption
    line -- page markers, images, math blocks, tables, and other captions are skipped,
    so the context is the running text the figure actually sits in."""
    paragraphs: list[tuple[int, str]] = []
    pos = 0
    for chunk in md.split("\n\n"):
        paragraphs.append((pos, chunk))
        pos += len(chunk) + 2
    idx = next(
        (
            i
            for i, (start, chunk) in enumerate(paragraphs)
            if start < caption.line_end <= start + len(chunk) + 1
        ),
        None,
    )
    if idx is None:
        return {"before": "", "after": ""}
    before = [c for _, c in paragraphs[:idx] if _is_prose(c)][-n:] if n else []
    after = [c for _, c in paragraphs[idx + 1 :] if _is_prose(c)][:n] if n else []
    return {
        "before": "\n\n".join(" ".join(c.split()) for c in before),
        "after": "\n\n".join(" ".join(c.split()) for c in after),
    }


def enrich_markdown(
    parsed: ParseResult,
    cfg: FigureConfig | None = None,
    describe: FigureDescriber | None = None,
) -> str:
    """Return the markdown with figure descriptions spliced next to their captions.

    One describe call per figure (its panels together), run concurrently across
    figures. A figure that fails (after retries) is left undescribed with a warning;
    splicing is deterministic.
    """
    cfg = cfg or FigureConfig()
    describe = describe or _default_describe
    md = parsed.markdown

    # layout-model-detected captions, when the parse provided them (empty set -> None:
    # fall back to pure regex detection for papers with no figure_title regions)
    known = {
        _normalize_caption(r.text) for regions in parsed.figure_captions.values() for r in regions
    } or None

    # first pass: only to learn page grouping for crop-to-caption pairing below.
    # Renaming can change md's length (placeholder text differs from the new image
    # line), which would invalidate any caption.line_end offsets computed against the
    # pre-rename text -- so captions are re-found from scratch afterward, once md is final.
    figure_crops: dict[tuple[int, str], list[Path]] = {}  # (page, number) -> figure's crops
    for page, captions in _group_by_page(md, known).items():
        if page is None:
            continue
        crops = parsed.figure_crops.get(page, [])
        if not crops:
            continue
        caption_regions = parsed.figure_captions.get(page, [])
        bboxes = [_caption_bbox(c, caption_regions) for c in captions]
        assignment = _pair_crops(crops, bboxes)
        md, renamed = _rename_crops(md, page, crops, assignment, captions)
        for ki, crop_indices in assignment.items():
            figure_crops[(page, captions[ki].number)] = [renamed[ci] for ci in sorted(crop_indices)]

    by_page = _group_by_page(md, known)
    header = _paper_header(md)

    # one describe task per figure that has crop(s)
    tasks: dict[tuple[int | None, str], tuple[list[Path], _Caption, dict]] = {}
    for page, captions in by_page.items():
        for caption in captions:
            crops = figure_crops.get((page, caption.number))
            if crops:
                context = {**header, **_neighbor_context(md, caption, cfg.context_paragraphs)}
                tasks[(page, caption.number)] = (crops, caption, context)

    results: dict[tuple[int | None, str], dict | None] = {}
    if tasks:
        with ThreadPoolExecutor(max_workers=cfg.max_workers) as pool:
            futures = {
                pool.submit(describe, crops, caption.number, caption.text, context, cfg): key
                for key, (crops, caption, context) in tasks.items()
            }
            for future, key in futures.items():
                try:
                    results[key] = future.result()
                except Exception as exc:  # surface it; leave this figure undescribed
                    warnings.warn(f"figure description failed for FIGURE {key[1]}: {exc!r}")
                    results[key] = None

    insertions: list[tuple[int, str]] = []
    for key, (_, caption, _) in tasks.items():
        desc = results.get(key)
        if not desc or not desc.get("description"):
            continue
        figure_type = (desc.get("figure_type") or "").replace("_", " ").strip()
        label = (
            f"Figure description (auto, {figure_type})"
            if figure_type
            else ("Figure description (auto)")
        )
        insertions.append((caption.line_end, f"\n\n> **{label}:** {desc['description']}"))

    # apply back-to-front so earlier offsets stay valid
    for offset, text in sorted(insertions, key=lambda it: it[0], reverse=True):
        md = md[:offset] + text + md[offset:]
    return md
