"""Split a too-long PDF into OCR-sized parts, then glue the per-part ``ParseResult``s
back into one -- so the rest of the pipeline (enrich, citations, chunking) never needs
to know a split happened.

Cloud OCR (z.ai's maas endpoint) hard-caps a single request at 100 pages (HTTP 400
above it); nothing else in the pipeline has any page-count limit. A page is already
GLM-OCR's atomic unit (one region set per page, dispatched independently), so splitting
at page boundaries loses nothing a single big parse wouldn't already risk at its own
per-page boundaries -- no overlap between parts is needed.

Each part gets its own parse checkpoint (a distinct ``work_dir``), not just its own
temp file: re-running an unchanged book skips OCR for every part, exactly like a normal
single-parse document.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .markers import PAGE_MARKER_RE, page_marker
from .parse import CaptionRegion, CropRegion, ParseResult
from .references import RawReference

logger = logging.getLogger(__name__)


def page_count(pdf_path: Path) -> int:
    import fitz  # PyMuPDF; already present transitively via glmocr

    with fitz.open(pdf_path) as doc:
        return doc.page_count


def split_pdf(pdf_path: Path, out_dir: Path, max_pages: int) -> list[tuple[Path, int]]:
    """Split ``pdf_path`` into ``<=max_pages``-page parts under ``out_dir``
    (``part_0.pdf``, ``part_1.pdf``, ...), in document order.

    Returns ``[(part_path, page_count), ...]``. When the PDF is already within
    ``max_pages``, returns ``[(pdf_path, page_count)]`` unchanged -- no copy, no new file.

    Each part is written with ``no_new_id=True`` so splitting the same source pages twice
    produces byte-identical output. Without it, MuPDF stamps a fresh random ``/ID`` into
    the trailer on every save -- confirmed live: the exact same source pages hashed
    differently each run. parse_cache keys its checkpoint on the *original* PDF's hash +
    part index (not the part file's own bytes) so correctness never depended on this, but
    a reproducible split is worth having anyway (diffable parts, defense in depth).
    """
    import fitz  # PyMuPDF; already present transitively via glmocr

    with fitz.open(pdf_path) as doc:
        n_pages = doc.page_count
        if n_pages <= max_pages:
            return [(pdf_path, n_pages)]

        out_dir.mkdir(parents=True, exist_ok=True)
        parts: list[tuple[Path, int]] = []
        for i, start in enumerate(range(0, n_pages, max_pages)):
            end = min(start + max_pages, n_pages) - 1  # fitz page ranges are inclusive
            part_path = out_dir / f"part_{i}.pdf"
            with fitz.open() as part:
                part.insert_pdf(doc, from_page=start, to_page=end)
                part.save(part_path, no_new_id=True)
            parts.append((part_path, end - start + 1))
        return parts


def _offset_page_markers(markdown: str, offset: int) -> str:
    if offset == 0:
        return markdown
    return PAGE_MARKER_RE.sub(lambda m: page_marker(int(m.group(1)) + offset), markdown)


def merge_parse_results(
    results: list[ParseResult],
    pages_per_part: list[int],
    figures_dir: Path,
) -> ParseResult:
    """Glue per-part ``ParseResult``s (each with its own local, 1-based page numbers)
    into one ``ParseResult`` with globally-offset page numbers.

    Each part independently restarts crop numbering at ``page_1_fig_0.png``, so crop
    files are moved into the shared ``figures_dir`` and renamed with their global page
    number as they're merged in, to avoid collisions.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    markdown_parts: list[str] = []
    references_markdown_parts: list[str] = []
    figure_crops: dict[int, list[CropRegion]] = {}
    figure_captions: dict[int, list[CaptionRegion]] = {}
    references: list[RawReference] = []

    offset = 0
    for result, n_pages in zip(results, pages_per_part, strict=True):
        part_markdown = _offset_page_markers(result.markdown, offset)
        if result.references_markdown:
            references_markdown_parts.append(result.references_markdown)

        for local_page, crops in result.figure_crops.items():
            global_page = local_page + offset
            moved: list[CropRegion] = []
            for idx, crop in enumerate(crops):
                new_path = figures_dir / f"page_{global_page}_fig_{idx}.png"
                if crop.path.exists():
                    crop.path.replace(new_path)
                # parse.py embeds the crop path straight into a "![FIGURE_CROP
                # page:idx](path)" placeholder -- moving the file (above) doesn't
                # retarget that text, so enrich.py's later rename pass (which matches
                # this exact placeholder against the CURRENT figure_crops path) would
                # silently find no match and leave the stale pre-merge link in the
                # final markdown. Rewrite it here, in lockstep with the move.
                part_markdown = part_markdown.replace(
                    f"![FIGURE_CROP {local_page}:{idx}]({crop.path})",
                    f"![FIGURE_CROP {global_page}:{idx}]({new_path})",
                )
                moved.append(CropRegion(new_path, crop.bbox))
            figure_crops[global_page] = moved

        markdown_parts.append(part_markdown)

        for local_page, captions in result.figure_captions.items():
            figure_captions[local_page + offset] = captions

        references.extend({**ref, "page": ref["page"] + offset} for ref in result.references)

        offset += n_pages

    return ParseResult(
        markdown="\n\n".join(markdown_parts),
        figure_crops=figure_crops,
        figure_captions=figure_captions,
        references=references,
        references_markdown="\n\n".join(references_markdown_parts),
    )
