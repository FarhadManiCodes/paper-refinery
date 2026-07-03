"""Local GLM-OCR backend: llama-server (inference) + the official ``glmocr`` SDK
(PP-DocLayout-V3 layout detection + per-region OCR dispatch + result formatting).

GLM-OCR is a region-level recognizer, not a page-to-markdown model: it OCRs a crop given
a task prompt ("Text Recognition:", "Table Recognition:", "Formula Recognition:") and has
no layout awareness of its own. The ``glmocr`` SDK supplies the layout pass (PP-DocLayout-V3),
crops each region, dispatches per-region OCR calls to our own ``llama-server`` (an
OpenAI-compatible ``/v1/chat/completions`` server, same protocol the SDK expects from
vLLM/SGLang/Ollama), and hands back per-page regions plus already-cropped images for
figure/chart regions.

We rebuild markdown from ``result.json_result`` ourselves (rather than using glmocr's own
``markdown_result``) so we can: drop boilerplate regions outright, route reference content
out of the body into its own raw markdown, convert HTML tables to markdown, and turn
figure/chart regions into placeholders + saved crop files (never interpreted here -- see
``figures.py`` for that, now working from crops instead of full-page renders).

This module produces two markdown outputs (``ParseResult.markdown`` and
``.references_markdown``) and nothing more: GLM-OCR's own native output format is
markdown (glmocr's own ``markdown_result`` field, though we don't use it directly --
see above), and that applies identically to bibliography text, which gets the exact
same generic "Text Recognition:" OCR treatment as any other paragraph. Structuring the
bibliography into anything beyond plain OCR'd text -- authors/title/venue/DOI, in-text
citation-marker linking -- is deliberately a separate, later concern, not this module's
job; the one thing this module *does* do for references is pair each entry with its
detected number (``_merge_reference_numbers``, plain region adjacency), since that's
reassembling what the layout model already segmented, not interpreting it.

Page boundaries are our own ``<page_number>N</page_number>`` markers (``markers.py``),
independent of any OCR-detected page-number region (which is discarded as boilerplate).
"""

from __future__ import annotations

import json
import re
import warnings
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from functools import cmp_to_key
from pathlib import Path

from bs4 import BeautifulSoup

from .backend import OcrBackend, ocr_backend
from .config import ParseConfig, load_config
from .markers import page_marker

# PP-DocLayout-V3 label taxonomy (glmocr's config.yaml `label_task_mapping` / `id2label`).
_ABANDON_LABELS = {
    "header",
    "footer",
    "number",  # printed page number -- we inject our own <page_number> markers instead
    "footnote",
    "aside_text",
    "footer_image",
    "header_image",
}
_DOC_TITLE_LABEL = "doc_title"
_PARAGRAPH_TITLE_LABEL = "paragraph_title"
_FIGURE_TITLE_LABEL = "figure_title"  # a "FIGURE N. ..." caption line -- plain body text,
#   named explicitly rather than relying on the unknown-label fallback, since enrich.py's
#   caption regex depends on this text actually landing in the body markdown
_REFERENCE_NUMBER_LABEL = "reference"  # the bracket/number marker before a bibliography
#   entry -- its own region, sibling to "reference_content" (same split as
#   formula/formula_number); paired back together by _merge_reference_numbers below
#   rather than discarded, so the raw bibliography keeps its original numbering
_REFERENCE_LABEL = "reference_content"
_TABLE_LABEL = "table"
_ALGORITHM_LABEL = "algorithm"  # pseudocode block: fenced so markdown preserves its structure
_FORMULA_LABELS = {"display_formula", "inline_formula"}
_FORMULA_NUMBER_LABEL = "formula_number"
_FIGURE_LABELS = {"chart", "image"}  # glmocr's "skip" task: cropped, never OCR'd

_HEADING_PREFIX_RE = re.compile(r"^#+\s*")


def _strip_heading_prefix(content: str) -> str:
    """Strip a leading ``#``/``##`` (or ``-``/``* ``) the OCR itself may have emitted.

    Confirmed against a live glmocr response: title-region content sometimes already
    starts with its own markdown heading marker, which would otherwise double up with
    the one we prepend (producing ``"## ## Section Title"``).
    """
    content = content.strip()
    if content.startswith(("- ", "* ")):
        content = content[2:].lstrip()
    return _HEADING_PREFIX_RE.sub("", content)


@dataclass
class ParseResult:
    # markdown with one authoritative <page_number>N</page_number> per page boundary,
    # boilerplate/reference regions removed, tables as markdown, formulas as LaTeX
    markdown: str
    figure_crops: dict[int, list[Path]] = field(default_factory=dict)  # page -> crop paths
    # figure/table caption texts as detected by the layout model (figure_title regions),
    # page -> texts in reading order. The same text also stays in `markdown` as plain
    # body content; this field exists so enrich.py can anchor descriptions on known
    # captions instead of regex-guessing them from body text (where an in-text
    # "Figure 4 shows..." paragraph could otherwise steal the anchor).
    figure_captions: dict[int, list[str]] = field(default_factory=dict)
    # raw bibliography, routed out of `markdown` entirely -- structuring/linking these
    # is a separate, later concern (not this module's job)
    references: list[dict] = field(default_factory=list)  # [{"page", "number", "text"}, ...]
    references_markdown: str = ""  # _render_references_markdown(references); see parse_pdf


def _wrap_formula(content: str) -> str:
    text = (content or "").strip()
    for fence in ("$$", "\\[", "\\("):
        if text.startswith(fence):
            text = text[len(fence) :].strip()
    for fence in ("$$", "\\]", "\\)"):
        if text.endswith(fence):
            text = text[: -len(fence)].strip()
    return f"$$\n{text}\n$$"


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


def _html_table_to_markdown(html: str) -> str:
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


def _dispatch_region(region: dict) -> tuple[str, str]:
    """Classify one glmocr region and format its content.

    Returns ``(kind, text)``; kind is one of "abandon", "body", "caption", "reference",
    "reference_number", "figure". Prefers ``native_label``
    (PP-DocLayout-V3's fine-grained class, e.g. "paragraph_title"/"display_formula"/
    "reference_content") over ``label`` (glmocr's coarse text/table/formula/skip bucket,
    e.g. "text"/"formula"), which can't distinguish reference_content from body text, or
    chart from image. Confirmed against a live glmocr response: both keys are present on
    every region dict.
    """
    label = region.get("native_label") or region.get("label") or ""
    content = (region.get("content") or "").strip()

    if label in _ABANDON_LABELS:
        return "abandon", ""
    if label == _DOC_TITLE_LABEL:
        return "body", f"# {_strip_heading_prefix(content)}"
    if label == _PARAGRAPH_TITLE_LABEL:
        return "body", f"## {_strip_heading_prefix(content)}"
    if label == _FIGURE_TITLE_LABEL:
        # kept in the body markdown like plain text, but ALSO surfaced as a known
        # caption (ParseResult.figure_captions) so enrich.py can anchor on layout-model
        # ground truth instead of guessing captions from body text by regex alone
        return "caption", content
    if label == _REFERENCE_NUMBER_LABEL:
        return "reference_number", content
    if label == _REFERENCE_LABEL:
        return "reference", content
    if label == _TABLE_LABEL:
        return "body", _html_table_to_markdown(content)
    if label == _ALGORITHM_LABEL:
        return "body", f"```\n{content}\n```"
    if label in _FORMULA_LABELS:
        return "body", _wrap_formula(content)
    if label == _FORMULA_NUMBER_LABEL:
        # standalone equation-number region: glmocr folds these into the formula's own
        # content upstream by default (enable_merge_formula_numbers), so under default
        # settings this never fires. If a glmocr_config_overrides caller disables that
        # merge, the number degrades to its own "(N)" paragraph rather than being
        # dropped -- occasional loss of \tag{} fidelity is acceptable here.
        return "body", f"({content.strip('() ')})" if content else ""
    if label in _FIGURE_LABELS:
        return "figure", ""
    return "body", content  # unknown label: keep as text rather than silently drop it


def _merge_reference_numbers(
    triples: list[tuple[str, str, dict]],
) -> list[tuple[str, str, dict]]:
    """Fold an adjacent reference_number/reference pair (either order) into one
    "reference" triple, carrying the number in ``region["number"]``.

    PP-DocLayout-V3 labels a bibliography entry's leading "[12]"/"23." marker as its own
    region (native_label "reference"), separate from "reference_content" -- the same
    kind of split it makes for formula/formula_number (which glmocr itself merges
    upstream). An
    unpaired reference_content still becomes an entry with no number attached (never
    dropped over a missing number -- the entry's text is what matters most); an unpaired
    reference_number is dropped (nothing to attach it to).
    """
    merged: list[tuple[str, str, dict]] = []
    i = 0
    while i < len(triples):
        kind, text, region = triples[i]
        nxt = triples[i + 1] if i + 1 < len(triples) else None
        if kind == "reference_number" and nxt and nxt[0] == "reference":
            number = text.strip().strip("[]().").strip()
            merged.append(("reference", nxt[1], {**nxt[2], "number": number}))
            i += 2
        elif kind == "reference" and nxt and nxt[0] == "reference_number":
            number = nxt[1].strip().strip("[]().").strip()
            merged.append(("reference", text, {**region, "number": number}))
            i += 2
        elif kind == "reference_number":
            i += 1  # unpaired number marker: nothing to attach it to
        else:
            merged.append((kind, text, region))
            i += 1
    return merged


# a bibliography-entry-looking start: "[12] " or "12. " (bracket/dot required -- a bare
# "2 " would match too much ordinary body text to be safe as a reclaim signal)
_MISLABELED_REFERENCE_RE = re.compile(r"^\s*\[?\d{1,3}[\].]\s")


def _reclaim_mislabeled_references(
    triples: list[tuple[str, str, dict]],
) -> list[tuple[str, str, dict]]:
    """Reroute body regions that are clearly bibliography entries back into the references.

    PP-DocLayout-V3 sometimes mislabels a bibliography's first entr(ies) as plain text --
    confirmed live on brunton-2016.pdf, where "1. Jordan MI, ..." was the last *body*
    paragraph while reference_content detection only began at entry 3. Recovery rule,
    deliberately narrow: on a page that already has detected reference regions, a body
    region is reclaimed iff it starts with a bibliography-entry marker ("[1] " / "1. ",
    see ``_MISLABELED_REFERENCE_RE``) *and* is directly adjacent to a reference region --
    iterated to a fixed point, so a contiguous run of mislabeled entries chains onto the
    reference run one by one. Numbered body text anywhere else on the page is never
    touched.
    """
    if not any(kind == "reference" for kind, _, _ in triples):
        return triples
    out = list(triples)
    changed = True
    while changed:
        changed = False
        for i, (kind, text, region) in enumerate(out):
            if kind != "body" or not _MISLABELED_REFERENCE_RE.match(text):
                continue
            prev_is_ref = i > 0 and out[i - 1][0] == "reference"
            next_is_ref = i + 1 < len(out) and out[i + 1][0] == "reference"
            if prev_is_ref or next_is_ref:
                out[i] = ("reference", text, region)
                changed = True
    return out


def _warn_reference_gaps(references: list[dict]) -> None:
    """Warn (never fix or drop) when a numbered bibliography has holes.

    A gap means the layout model produced no region at all for an entry (confirmed live:
    brunton-2016's entry 2 simply has no region) -- nothing downstream can recover text
    that was never OCR'd, but a silent loss is worse than a loud one. Only fires when
    every entry has a clean numeric key; unnumbered (author-year) styles say nothing.
    """
    keys = [_reference_sort_key(ref) for ref in references]
    if not keys or any(key is None for key in keys):
        return
    expected = set(range(1, max(keys) + 1))
    missing = sorted(expected - set(keys))
    if missing:
        warnings.warn(
            f"numbered bibliography has {len(missing)} missing entr(ies): {missing} -- "
            "the layout model likely produced no region for them (unrecoverable here)"
        )


def _save_figure_crop(
    region: dict,
    image_files: dict,
    used_images: set[str],
    figures_dir: Path,
    page: int,
    idx: int,
) -> Path | None:
    """Save the region's already-cropped image (from glmocr's ``image_files``) to disk.

    Confirmed against a live glmocr response: a chart/image region carries
    ``image_path: "imgs/{filename}"`` where ``filename`` is exactly a key in
    ``image_files`` (e.g. ``"cropped_page0_idx0.jpg"``, 0-indexed by input-list position).
    The page-number-substring fallback below is defense-in-depth only, for a glmocr
    version where that key is absent.
    """
    image_path = region.get("image_path")
    filename = Path(image_path).name if image_path else None
    img = image_files.get(filename) if filename else None

    if img is None:
        for candidate_name, candidate_img in image_files.items():
            if candidate_name in used_images:
                continue
            if f"page{page - 1}_" in candidate_name or f"page{page}_" in candidate_name:
                img, filename = candidate_img, candidate_name
                break

    if img is None or filename is None:
        warnings.warn(f"no cropped image found for a figure/chart region on page {page}")
        return None

    used_images.add(filename)
    crop_path = figures_dir / f"page_{page}_fig_{idx}.png"
    img.save(crop_path)
    return crop_path


# Back-matter (Conflict of Interest disclosure, copyright/licensing notice) that
# PP-DocLayout-V3 can misclassify as a reference_content region -- confirmed live on a
# Frontiers journal paper, where this text sits immediately after the real bibliography
# in reading order, a plausible source of the misclassification.
_REFERENCE_BOILERPLATE_SIGNALS = (
    "conflict of interest",
    "copyright ©",
    "creative commons",
    "open-access article distributed",
)


def _is_reference_boilerplate(text: str) -> bool:
    """True if ``text`` looks like misclassified back-matter rather than an actual
    bibliography entry."""
    lowered = text.lower()
    return any(signal in lowered for signal in _REFERENCE_BOILERPLATE_SIGNALS)


_MAX_TRAILING_BOILERPLATE_CHECK = 3


def _drop_trailing_boilerplate(references: list[dict]) -> list[dict]:
    """Drop a trailing run of misclassified back-matter entries from the bibliography.

    Checks only the last few entries (up to ``_MAX_TRAILING_BOILERPLATE_CHECK``), and
    only ever removes a *contiguous run starting from the very end* -- stops at the
    first entry that doesn't match, so a genuine reference is never dropped just for
    being near the end of the list (e.g. one whose own title happens to mention
    "copyright"). Sometimes the boilerplate itself splits across more than one entry
    (e.g. "Conflict of Interest: ..." and "Copyright © ..." as two separate regions),
    which is why this checks more than just the single last entry.
    """
    cleaned = list(references)
    checked = 0
    while (
        cleaned
        and checked < _MAX_TRAILING_BOILERPLATE_CHECK
        and _is_reference_boilerplate(cleaned[-1]["text"])
    ):
        cleaned.pop()
        checked += 1
    return cleaned


_LEADING_REFERENCE_NUMBER_RE = re.compile(r"^\s*\[?(\d+)[\]. ]?\s")


def _reference_sort_key(ref: dict) -> int | None:
    """Best-effort integer ordering key for one reference.

    Prefers the region-paired ``number`` field (see ``_merge_reference_numbers``), but
    that pairing is the *uncommon* case in practice -- confirmed live on kalman-1960.pdf,
    where PP-DocLayout-V3 never produces a separate reference_number region at all; the
    marker is just the leading digits of the OCR'd text blob itself (e.g. "2 L. A.
    Zadeh..."). Falls back to parsing that leading number directly off the text before
    giving up. Returns ``None`` when neither source yields a clean integer.
    """
    raw = ref.get("number")
    if raw is not None:
        try:
            return int(raw)
        except ValueError:
            return None
    match = _LEADING_REFERENCE_NUMBER_RE.match(ref["text"])
    return int(match.group(1)) if match else None


def _sort_references_by_number(references: list[dict]) -> list[dict]:
    """Re-sort references by their best-effort numeric marker, when doing so is
    unambiguous.

    PP-DocLayout-V3's own region ``index`` (used to order regions before merging, see
    ``_build_markdown``) isn't always reading order for a multi-column bibliography --
    confirmed live on kalman-1960.pdf, where two side-by-side columns produced entries
    out of numeric order (2, 1, 3, 4, 5, 7, 6, ...). Since a numbered bibliography always
    prints in ascending order, re-sorting by the parsed number is a safe, unambiguous
    fix -- but only when *every* entry yields a clean integer key (see
    ``_reference_sort_key``) *and* the keys form one contiguous run (a numbered
    bibliography is always 1..n): a duplicate key (two entries garbled to the same
    number) or an outlier (an unnumbered entry whose text happens to start with a year,
    e.g. "2019 IEEE Conference on...") means the keys can't be trusted at all. A style
    with no numbers (author-year, e.g. fmech-07-655266) or a partial/garbled parse is
    left in detected order rather than guessing at a partial sort.
    """
    if not references:
        return references
    keys = [_reference_sort_key(ref) for ref in references]
    if any(key is None for key in keys):
        return references
    if sorted(keys) != list(range(min(keys), min(keys) + len(keys))):
        return references  # duplicates or gaps -> don't trust the keys
    return [ref for _, ref in sorted(zip(keys, references), key=lambda pair: pair[0])]


_WIDE_FRACTION = 0.6  # of page width: a region this wide spans columns (title, wide table)
_COLUMN_OVERLAP = 0.5  # of the narrower region's width: x-overlap needed to share a column
_Y_TOL_FRACTION = 0.01  # of page width: y-difference treated as "same line" (~half a line)


def _reading_order(regions: list[dict]) -> list[dict]:
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
    boxes = [r.get("bbox_2d") for r in regs]
    if not boxes or any(b is None or len(b) != 4 for b in boxes):
        return regs
    page_w = (max(b[2] for b in boxes) - min(b[0] for b in boxes)) or 1
    y_tol = _Y_TOL_FRACTION * page_w

    def is_wide(b) -> bool:
        return (b[2] - b[0]) > _WIDE_FRACTION * page_w

    def same_column(a, b) -> bool:
        overlap = min(a[2], b[2]) - max(a[0], b[0])
        return overlap > _COLUMN_OVERLAP * min(a[2] - a[0], b[2] - b[0])

    def by_top(a: dict, b: dict) -> int:
        dy = a["bbox_2d"][1] - b["bbox_2d"][1]
        if abs(dy) <= y_tol:
            return 0  # same line -> stable sort keeps glmocr's order
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


def _build_markdown(
    pages_regions: list[list[dict]],
    image_files: dict,
    figures_dir: Path,
    cfg: ParseConfig,
) -> tuple[str, dict[int, list[Path]], dict[int, list[str]], list[dict]]:
    """Assemble page-marked markdown, figure crops, known captions, and a raw
    references list from glmocr's per-page region lists."""
    parts: list[str] = []
    figure_crops: dict[int, list[Path]] = {}
    figure_captions: dict[int, list[str]] = {}
    references: list[dict] = []
    used_images: set[str] = set()

    for page_idx, regions in enumerate(pages_regions):
        page = page_idx + 1  # glmocr's page_idx is 0-based by input order; confirmed live
        sorted_regions = _reading_order(regions)
        triples = [(*_dispatch_region(r), r) for r in sorted_regions]
        triples = [t for t in triples if t[0] != "abandon"]
        triples = _merge_reference_numbers(triples)
        triples = _reclaim_mislabeled_references(triples)

        body_parts: list[str] = []
        for kind, text, region in triples:
            if kind == "body":
                if text.strip():
                    body_parts.append(text)
            elif kind == "caption":
                if text.strip():
                    body_parts.append(text)  # stays in the body markdown as-is...
                    figure_captions.setdefault(page, []).append(text)  # ...and is known
            elif kind == "reference":
                if text.strip():
                    references.append({"page": page, "number": region.get("number"), "text": text})
            elif kind == "figure":
                idx = len(figure_crops.get(page, []))
                crop_path = _save_figure_crop(region, image_files, used_images, figures_dir, page, idx)
                if crop_path is None:
                    continue
                figure_crops.setdefault(page, []).append(crop_path)
                body_parts.append(f"![FIGURE_CROP {page}:{idx}]({crop_path})")

        parts.append(page_marker(page))
        if body_parts:  # a page can be all references/boilerplate -- no empty part then
            parts.append("\n\n".join(body_parts))

    references = _drop_trailing_boilerplate(references)
    references = _sort_references_by_number(references)
    _warn_reference_gaps(references)

    return "\n\n".join(parts), figure_crops, figure_captions, references


def _render_references_markdown(references: list[dict]) -> str:
    """Plain markdown rendering of the raw bibliography: one entry per line, grouped
    under each page's own ``<page_number>`` marker, in reading order.

    Zero interpretation -- this is GLM-OCR's own OCR'd text, reassembled only using the
    number/content region pairing already done in ``_merge_reference_numbers``. No
    schema, no external tool: structuring the bibliography into anything more is a
    separate, later concern, kept out of the OCR/parse layer entirely.
    """
    if not references:
        return ""
    by_page: dict[int, list[dict]] = {}
    for ref in references:
        by_page.setdefault(ref["page"], []).append(ref)

    parts: list[str] = []
    for page in sorted(by_page):
        lines = [
            f"[{ref['number']}] {ref['text']}" if ref["number"] else ref["text"]
            for ref in by_page[page]
        ]
        parts.append(page_marker(page))
        parts.append("\n\n".join(lines))
    return "\n\n".join(parts)


def _parse_with_watchdog(backend: OcrBackend, pdf_path: Path, cfg: ParseConfig):
    """Run one PDF's parse under ``cfg.parse_timeout_s``.

    ``parser.parse`` has no timeout of its own; a wedged llama-server hangs it forever
    (observed live: 36+ minutes on a paper that normally takes ~10). The parse runs in a
    worker thread; on timeout the server process is killed -- that's what actually breaks
    glmocr's in-flight HTTP calls loose -- and a clear error is raised. With a shared
    (batch) backend this also ends the backend for any remaining PDFs: acceptable, since
    a wedged server can't be trusted for the next paper either.
    """
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        future = pool.submit(backend.parser.parse, str(pdf_path))
        try:
            return future.result(timeout=cfg.parse_timeout_s)
        except FuturesTimeout:
            backend.server.kill()
            raise RuntimeError(
                f"parsing {pdf_path.name} exceeded ParseConfig.parse_timeout_s "
                f"({cfg.parse_timeout_s:.0f}s) -- llama-server killed; the run was "
                "almost certainly wedged, not just slow"
            ) from None
    finally:
        # don't block on the worker: after a server kill it unwinds on its own error
        pool.shutdown(wait=False, cancel_futures=True)


def parse_pdf(
    pdf_path: Path,
    image_dir: Path,
    cfg: ParseConfig | None = None,
    backend: OcrBackend | None = None,
) -> ParseResult:
    """Parse a PDF into page-marked markdown (with figure placeholders + LaTeX + markdown
    tables) via a local llama-server serving GLM-OCR, orchestrated by the glmocr SDK.

    Pass ``backend`` (see ``backend.ocr_backend``) to reuse one running server across
    many PDFs -- server spawn + model load is the dominant fixed cost of a parse. When
    omitted, a backend is spawned and torn down for just this call (the CLI's case).
    """
    cfg = cfg or load_config().parse
    pdf_path, image_dir = Path(pdf_path), Path(image_dir)
    figures_dir = image_dir / cfg.figures_dir_name
    figures_dir.mkdir(parents=True, exist_ok=True)

    if backend is None:
        with ocr_backend(cfg) as own:
            result = _parse_with_watchdog(own, pdf_path, cfg)
    else:
        result = _parse_with_watchdog(backend, pdf_path, cfg)

    json_result = result.json_result
    if isinstance(json_result, str):
        json_result = json.loads(json_result)

    markdown, figure_crops, figure_captions, references = _build_markdown(
        json_result, result.image_files, figures_dir, cfg
    )
    return ParseResult(
        markdown=markdown,
        figure_crops=figure_crops,
        figure_captions=figure_captions,
        references=references,
        references_markdown=_render_references_markdown(references),
    )
