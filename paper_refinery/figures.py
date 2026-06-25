"""Figure understanding via Gemini.

Given a full-page render and a figure's caption, describe the captioned figure for
retrieval (what is compared, axes/variables, qualitative trends). The caption tells
Gemini which figure on the page to describe. Never invents precise numeric values read
off plotted curves -- those belong to the paper's own tables.
"""

from __future__ import annotations

import os
from pathlib import Path

from .config import FigureConfig

_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def _mime_type(path: Path) -> str:
    return _MIME.get(path.suffix.lower(), "image/png")


def _finalize(text: str | None, cfg: FigureConfig) -> str:
    """Strip whitespace; return "" when Gemini flagged the image as a non-figure."""
    text = (text or "").strip()
    return "" if text.upper().startswith(cfg.skip_marker.upper()) else text


def _prompt(context: str | None, cfg: FigureConfig) -> str:
    """The instruction sent to Gemini. ``context`` is the caption (plus optional in-text
    references) that identifies which figure on the page to describe."""
    if context:
        return (
            f"{cfg.prompt}\n\nCaption of the figure to describe (use it to locate the "
            f"figure on the page and to ground your description):\n{context.strip()}"
        )
    return cfg.prompt


def describe_figure(
    page_render: Path, context: str | None = None, cfg: FigureConfig | None = None
) -> str:
    """Describe the captioned figure on a full-page render.

    ``page_render`` is the full-page image (page_N.jpg); ``context`` is the figure's
    caption (plus optional in-text references) telling Gemini which figure to describe.
    Returns "" if Gemini reports no matching figure on the page.
    """
    from google import genai
    from google.genai import types

    cfg = cfg or FigureConfig()
    page_render = Path(page_render)
    api_key = os.environ.get(cfg.api_key_env)
    if not api_key:
        raise RuntimeError(f"{cfg.api_key_env} is not set")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=cfg.model,
        contents=[
            types.Part.from_bytes(
                data=page_render.read_bytes(), mime_type=_mime_type(page_render)
            ),
            _prompt(context, cfg),
        ],
    )
    return _finalize(response.text, cfg)
