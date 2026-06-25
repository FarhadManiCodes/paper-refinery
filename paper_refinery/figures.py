"""Figure understanding via Gemini.

Describes each extracted figure for retrieval (what is compared, axes/variables, and
qualitative trends). Never invents precise numeric values read off plotted curves —
those belong to the paper's own tables.
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


def _prompt(context: str | None, cfg: FigureConfig) -> str:
    """The instruction sent to Gemini, optionally grounded by the figure's caption
    and how it is referenced in the paper's text."""
    if context:
        return (
            f"{cfg.prompt}\n\nUse this context from the paper to ground your "
            f"description (the figure's caption and how it is referenced in the "
            f"text):\n{context.strip()}"
        )
    return cfg.prompt


def describe_figure(
    image_path: Path, context: str | None = None, cfg: FigureConfig | None = None
) -> str:
    """Return a short, retrieval-oriented description of a figure image.

    ``context`` should be the figure's caption plus any in-text references; it grounds
    the description in the paper's own framing.
    """
    from google import genai
    from google.genai import types

    cfg = cfg or FigureConfig()
    image_path = Path(image_path)
    api_key = os.environ.get(cfg.api_key_env)
    if not api_key:
        raise RuntimeError(f"{cfg.api_key_env} is not set")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=cfg.model,
        contents=[
            types.Part.from_bytes(
                data=image_path.read_bytes(), mime_type=_mime_type(image_path)
            ),
            _prompt(context, cfg),
        ],
    )
    return (response.text or "").strip()
