"""The page-boundary marker, shared across the pipeline.

`parse.py` writes one ``<page_number>N</page_number>`` per page boundary; `enrich.py`
and `chunker.py` read them. Keeping the format in one place makes it a single contract.
"""

from __future__ import annotations

import re

# matches a marker and captures the page number; safe for .sub() and .finditer()
PAGE_MARKER_RE = re.compile(r"<page_number>\s*(\d+)\s*</page_number>")


def page_marker(page: int) -> str:
    return f"<page_number>{page}</page_number>"
