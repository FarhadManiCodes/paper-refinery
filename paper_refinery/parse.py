"""LlamaParse: PDF -> clean markdown + extracted figure images + page markers.

Uses agentic parse mode (best equations/tables). Specialized chart-to-table parsing
is deliberately OFF (it fabricates precise numbers from plots); figure understanding
is done separately in ``figures.py``.

Page boundaries are marked authoritatively from LlamaParse's per-page output as
``<page_number>N</page_number>`` (any inline tags LlamaParse emitted are replaced),
so the chunker can resolve each chunk's page range.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import ParseConfig

# figure files are named like chart_p6_0.png / img_p8_1.png; page screenshots page_N.jpg
_PAGE_FROM_NAME = re.compile(r"_p(\d+)_")
_SCREENSHOT = re.compile(r"^page_\d+\.(?:jpe?g|png)$", re.IGNORECASE)
_EXISTING_PAGE_TAG = re.compile(r"<page_number>\s*\d+\s*</page_number>")


@dataclass
class Figure:
    """An extracted figure image plus where it came from."""

    image_path: Path
    page: int | None = None
    caption: str | None = None


@dataclass
class ParseResult:
    # markdown with one authoritative <page_number>N</page_number> per page boundary
    markdown: str
    figures: list[Figure] = field(default_factory=list)


def _build_markdown(pages) -> str:
    """Concatenate per-page markdown, prefixing each page with an authoritative marker."""
    parts: list[str] = []
    for page in pages:
        body = _EXISTING_PAGE_TAG.sub("", page.md or "")
        parts.append(f"<page_number>{page.page}</page_number>")
        parts.append(body)
    return "\n\n".join(parts)


def _collect_figures(image_paths: list[str]) -> list[Figure]:
    """Turn saved image paths into Figures, skipping full-page screenshots and
    reading the page number from the filename (chart_pN_* / img_pN_*)."""
    figures: list[Figure] = []
    for raw in image_paths:
        path = Path(raw)
        if _SCREENSHOT.match(path.name):
            continue
        m = _PAGE_FROM_NAME.search(path.name)
        figures.append(Figure(image_path=path, page=int(m.group(1)) if m else None))
    return figures


def parse_pdf(
    pdf_path: Path, image_dir: Path, cfg: ParseConfig | None = None
) -> ParseResult:
    """Parse a PDF into page-marked markdown plus extracted figures via LlamaParse."""
    from llama_cloud_services import LlamaParse

    cfg = cfg or ParseConfig()
    pdf_path, image_dir = Path(pdf_path), Path(image_dir)
    api_key = os.environ.get(cfg.api_key_env)
    if not api_key:
        raise RuntimeError(f"{cfg.api_key_env} is not set")
    image_dir.mkdir(parents=True, exist_ok=True)

    parser = LlamaParse(
        api_key=api_key,
        result_type="markdown",
        parse_mode=cfg.parse_mode,
        save_images=cfg.save_images,
        extract_charts=cfg.extract_charts,
        inline_images_in_markdown=cfg.inline_images,
    )
    result = parser.parse(str(pdf_path))

    markdown = _build_markdown(result.pages)
    image_paths = result.save_all_images(str(image_dir)) if cfg.save_images else []
    return ParseResult(markdown=markdown, figures=_collect_figures(image_paths))
