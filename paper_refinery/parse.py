"""LlamaParse: PDF -> clean markdown (with inline figure placeholders + page markers)
plus a full-page render per page.

We deliberately do NOT use LlamaParse's per-figure image crops: that extraction is
unreliable (it misses some figures entirely and mis-classifies text blocks as charts).
Instead, enrichment works from the inline ``![alt](src)`` placeholders (a reliable
figure list, each with its caption on the next line) and the full-page render for each
figure's page, which always contains the figure.

Page boundaries are marked authoritatively from LlamaParse's per-page output as
``<page_number>N</page_number>`` (any inline tags LlamaParse emitted are replaced).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import ParseConfig

_PAGE_RENDER = re.compile(r"^page_(\d+)\.(?:jpe?g|png)$", re.IGNORECASE)
_EXISTING_PAGE_TAG = re.compile(r"<page_number>\s*\d+\s*</page_number>")


@dataclass
class ParseResult:
    # markdown with one authoritative <page_number>N</page_number> per page boundary
    # and inline ![alt](src) placeholders at each figure
    markdown: str
    page_renders: dict[int, Path] = field(default_factory=dict)  # page number -> page_N.jpg


def _build_markdown(pages) -> str:
    """Concatenate per-page markdown, prefixing each page with an authoritative marker."""
    parts: list[str] = []
    for page in pages:
        body = _EXISTING_PAGE_TAG.sub("", page.md or "")
        parts.append(f"<page_number>{page.page}</page_number>")
        parts.append(body)
    return "\n\n".join(parts)


def _page_renders(image_paths: list[str]) -> dict[int, Path]:
    """Map page number -> full-page render (page_N.jpg), ignoring figure crops."""
    renders: dict[int, Path] = {}
    for raw in image_paths:
        path = Path(raw)
        m = _PAGE_RENDER.match(path.name)
        if m:
            renders[int(m.group(1))] = path
    return renders


def parse_pdf(
    pdf_path: Path, image_dir: Path, cfg: ParseConfig | None = None
) -> ParseResult:
    """Parse a PDF into page-marked markdown (with figure placeholders) plus a full-page
    render per page, via LlamaParse."""
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
        inline_images_in_markdown=cfg.inline_images,
        take_screenshot=cfg.take_screenshot,
        disable_image_extraction=cfg.disable_image_extraction,
    )
    result = parser.parse(str(pdf_path))

    markdown = _build_markdown(result.pages)
    image_paths = result.save_all_images(str(image_dir))
    return ParseResult(markdown=markdown, page_renders=_page_renders(image_paths))
