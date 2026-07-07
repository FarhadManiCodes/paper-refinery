"""Transform a PDF into a ``ParseResult``: page-marked body markdown (tables as markdown,
formulas as LaTeX), figure/chart crops with their detected captions, and a raw references
list. This is the pipeline's parse stage; the llama-server + ``glmocr`` SDK *lifecycle* it
runs on lives in ``backend.py``.

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
reassembling what the layout model already segmented, not interpreting it. All
repair of the collected reference list (mislabel reclaim, split merging, text-layer
recovery, ordering) lives in ``references.py``.

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
from pathlib import Path
from typing import cast

from .backend import OcrBackend, ocr_backend
from .config import ParseConfig, load_config
from .markers import page_marker
from .reading_order import reading_order, region_bbox
from .references import (
    RawReference,
    reclaim_mislabeled_references,
    render_references_markdown,
    repair_references,
)
from .tables import html_table_to_markdown

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


@dataclass(frozen=True)
class CropRegion:
    """One saved figure/chart crop with its layout-detected bbox ([x1, y1, x2, y2] in
    page pixels; None when the layout model gave no well-formed box)."""

    path: Path
    bbox: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class CaptionRegion:
    """One figure_title region's text + bbox, same coordinate space as CropRegion.

    The bbox is what lets enrich.py pair captions with crops *geometrically* (nearest
    caption overlapping on either axis) instead of by position-in-list -- positional
    pairing mislabels the moment a multi-panel figure splits into more crops than captions.
    """

    text: str
    bbox: tuple[float, float, float, float] | None = None


@dataclass
class ParseResult:
    # markdown with one authoritative <page_number>N</page_number> per page boundary,
    # boilerplate/reference regions removed, tables as markdown, formulas as LaTeX
    markdown: str
    figure_crops: dict[int, list[CropRegion]] = field(default_factory=dict)  # page -> crops
    # figure/table captions as detected by the layout model (figure_title regions),
    # page -> regions in reading order. The same text also stays in `markdown` as plain
    # body content; this field exists so enrich.py can anchor descriptions on known
    # captions instead of regex-guessing them from body text (where an in-text
    # "Figure 4 shows..." paragraph could otherwise steal the anchor), and its bboxes
    # drive geometric crop-to-caption pairing.
    figure_captions: dict[int, list[CaptionRegion]] = field(default_factory=dict)
    # raw bibliography, routed out of `markdown` entirely -- structuring/linking these
    # is a separate, later concern (not this module's job)
    references: list[RawReference] = field(default_factory=list)
    references_markdown: str = ""  # references.render_references_markdown; see parse_pdf


def _wrap_formula(content: str) -> str:
    text = (content or "").strip()
    for fence in ("$$", "\\[", "\\("):
        if text.startswith(fence):
            text = text[len(fence) :].strip()
    for fence in ("$$", "\\]", "\\)"):
        if text.endswith(fence):
            text = text[: -len(fence)].strip()
    return f"$$\n{text}\n$$"


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
        return "body", html_table_to_markdown(content)
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
    upstream). An unpaired reference_content still becomes an entry with no number
    attached (never dropped over a missing number -- the entry's text is what matters
    most); an unpaired reference_number is dropped (nothing to attach it to).
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


def _build_markdown(
    pages_regions: list[list[dict]],
    image_files: dict,
    figures_dir: Path,
    pdf_path: Path | None = None,
) -> tuple[str, dict[int, list[CropRegion]], dict[int, list[CaptionRegion]], list[RawReference]]:
    """Assemble page-marked markdown, figure crops, known captions, and a raw
    references list from glmocr's per-page region lists.

    ``pdf_path`` enables text-layer recovery of skipped bibliography entries
    (``_recover_missing_references``); None (tests, callers without the file) skips it."""
    parts: list[str] = []
    figure_crops: dict[int, list[CropRegion]] = {}
    figure_captions: dict[int, list[CaptionRegion]] = {}
    references: list[RawReference] = []
    used_images: set[str] = set()

    for page_idx, regions in enumerate(pages_regions):
        page = page_idx + 1  # glmocr's page_idx is 0-based by input order; confirmed live
        sorted_regions = reading_order(regions)
        triples = [(*_dispatch_region(r), r) for r in sorted_regions]
        triples = [t for t in triples if t[0] != "abandon"]
        triples = _merge_reference_numbers(triples)
        triples = reclaim_mislabeled_references(triples)

        body_parts: list[str] = []
        for kind, text, region in triples:
            if kind == "body":
                if text.strip():
                    body_parts.append(text)
            elif kind == "caption":
                if text.strip():
                    body_parts.append(text)  # stays in the body markdown as-is...
                    figure_captions.setdefault(page, []).append(  # ...and is known
                        CaptionRegion(text, region_bbox(region))
                    )
            elif kind == "reference":
                if text.strip():
                    references.append({"page": page, "number": region.get("number"), "text": text})
            elif kind == "figure":
                idx = len(figure_crops.get(page, []))
                crop_path = _save_figure_crop(
                    region, image_files, used_images, figures_dir, page, idx
                )
                if crop_path is None:
                    continue
                figure_crops.setdefault(page, []).append(CropRegion(crop_path, region_bbox(region)))
                body_parts.append(f"![FIGURE_CROP {page}:{idx}]({crop_path})")

        parts.append(page_marker(page))
        if body_parts:  # a page can be all references/boilerplate -- no empty part then
            parts.append("\n\n".join(body_parts))

    references = repair_references(references, pdf_path)

    return "\n\n".join(parts), figure_crops, figure_captions, references


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


def clear_stale_crops(figures_dir: Path) -> None:
    """Delete crops left in ``figures_dir`` by a previous run (under either the raw
    ``page_*_fig_*`` or the enrich-renamed ``fig_*`` naming).

    Crops are wholly derived artifacts; leftovers would otherwise accumulate and break
    checking crop pairing by filename. Only our own patterns are touched. Shared by
    ``parse_pdf`` (before a fresh OCR) and ``parse_cache`` (before restoring a snapshot's
    raw crops on a cache hit), so the two never drift.
    """
    for stale in (*figures_dir.glob("page_*_fig_*"), *figures_dir.glob("fig_*")):
        stale.unlink()


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
    clear_stale_crops(figures_dir)

    if backend is None:
        with ocr_backend(cfg) as own:
            result = _parse_with_watchdog(own, pdf_path, cfg)
    else:
        result = _parse_with_watchdog(backend, pdf_path, cfg)

    json_result = result.json_result
    if isinstance(json_result, str):
        json_result = json.loads(json_result)

    markdown, figure_crops, figure_captions, references = _build_markdown(
        # json_result / image_files come untyped from the glmocr result object; the shapes
        # are the SDK contract confirmed live (see _dispatch_region / _save_figure_crop)
        cast("list[list[dict]]", json_result),
        result.image_files or {},
        figures_dir,
        pdf_path=pdf_path,
    )
    return ParseResult(
        markdown=markdown,
        figure_crops=figure_crops,
        figure_captions=figure_captions,
        references=references,
        references_markdown=render_references_markdown(references),
    )
