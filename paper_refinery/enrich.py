"""Splice figure descriptions into the markdown at the correct location.

Anchors on each figure's caption ("FIGURE 4.3 ...") matched by page, inserting the
Gemini description right after the caption. Falls back to inline image placeholders or
page position when no caption anchor is found.
"""

from __future__ import annotations

from pathlib import Path

from .parse import ParseResult


def enrich_markdown(parsed: ParseResult, descriptions: dict[Path, str]) -> str:
    """Return markdown with each figure's description spliced next to its caption.

    TODO: implement caption-anchored insertion:
      - for each figure, find its "FIGURE N.M" caption near the figure's page
      - insert "> Figure description (auto): <text>" right after the caption
      - fall back to the inline ![caption](path) placeholder, else page position
    """
    raise NotImplementedError("enrich_markdown: caption-anchored splicing not implemented yet")
