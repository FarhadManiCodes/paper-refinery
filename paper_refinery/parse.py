"""LlamaParse: PDF -> clean markdown + extracted figure images + page markers.

Uses agentic parse mode (best equations/tables). Specialized chart-to-table parsing
is deliberately OFF (it fabricates precise numbers from plots); figure understanding
is done separately in ``figures.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .config import ParseConfig


@dataclass
class Figure:
    """An extracted figure image plus where it came from."""

    image_path: Path
    page: int | None = None
    caption: str | None = None


@dataclass
class ParseResult:
    # markdown with page boundaries marked as <page_number>N</page_number>
    markdown: str
    figures: list[Figure] = field(default_factory=list)


def parse_pdf(
    pdf_path: Path, image_dir: Path, cfg: ParseConfig | None = None
) -> ParseResult:
    """Parse a PDF into page-marked markdown plus extracted figures.

    TODO: implement with LlamaParse:
      - parser = LlamaParse(parse_mode=cfg.parse_mode, save_images=cfg.save_images,
                            inline_images_in_markdown=cfg.inline_images, ...)
      - res = parser.parse(pdf_path)
      - markdown = concat of res.pages' md, inserting <page_number>N</page_number>
        at each page boundary (authoritative, from per-page data)
      - figures = res.save_all_images(image_dir) paired with page info from
        res.get_image_nodes(); drop full-page screenshots (bbox == 0,0)
    """
    raise NotImplementedError("parse_pdf: LlamaParse integration not implemented yet")
