"""Figure understanding via Gemini.

Given a page's figure/chart crops (from the local GLM-OCR parser -- see parse.py) and the
captions of the figure(s) on that page, describe every figure in a single call. One call
per page (not per figure) keeps multi-figure pages cheap; each crop is downscaled first to
save image tokens. Never invents precise numeric values read off plotted curves -- those
belong to the paper's own tables.
"""

from __future__ import annotations

import io
import json
import os
import re
import time
from pathlib import Path

from .config import FigureConfig

_JSON = re.compile(r"\{.*\}", re.DOTALL)


def _generate(client, model, contents, attempts: int = 4, base_delay: float = 4.0):
    """generate_content with exponential backoff (handles transient rate limits)."""
    for i in range(attempts):
        try:
            return client.models.generate_content(model=model, contents=contents)
        except Exception:
            if i == attempts - 1:
                raise
            time.sleep(base_delay * (2**i))


def _render_bytes(page_render: Path, max_px: int) -> bytes:
    """JPEG bytes of the page render, downscaled so its long side <= max_px."""
    from PIL import Image

    img = Image.open(page_render).convert("RGB")
    if max(img.size) > max_px:
        img.thumbnail((max_px, max_px))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _prompt(figures: list[tuple[str, str]], cfg: FigureConfig) -> str:
    """cfg.prompt followed by '- <number>: <caption>' for each figure on the page."""
    listing = "\n".join(f"- {number}: {caption}" for number, caption in figures)
    return f"{cfg.prompt}\n\nFigures on this page:\n{listing}"


def _parse(text: str | None, cfg: FigureConfig) -> dict[str, str]:
    """Parse Gemini's JSON {number: description}; tolerate fences / surrounding prose."""
    m = _JSON.search(text or "")
    if not m:
        return {}
    try:
        data = json.loads(m.group(0))
    except (ValueError, TypeError):
        return {}
    out: dict[str, str] = {}
    for number, desc in (data or {}).items():
        desc = str(desc).strip()
        if desc and not desc.upper().startswith(cfg.skip_marker.upper()):
            out[str(number).strip()] = desc
    return out


def make_client(cfg: FigureConfig | None = None):
    """Create a Gemini client. Build one and reuse it across pages (see cli.py)."""
    from google import genai

    cfg = cfg or FigureConfig()
    api_key = os.environ.get(cfg.api_key_env)
    if not api_key:
        raise RuntimeError(f"{cfg.api_key_env} is not set")
    return genai.Client(api_key=api_key)


def describe_page_figures(
    crops: list[Path],
    figures: list[tuple[str, str]],
    cfg: FigureConfig | None = None,
    client=None,
) -> dict[str, str]:
    """Describe every figure on a page in one Gemini call.

    ``crops`` are the page's figure/chart crop files (from the local GLM-OCR parser), in
    reading order; ``figures`` is a list of ``(number, caption)`` for the figures on this
    page. All crops are sent as separate image parts alongside one caption listing --
    Gemini reconciles which crop matches which caption itself, same as it already does for
    a busy full page, so an exact crop-to-caption count match isn't required. Returns
    ``{number: description}``; figures Gemini does not find are omitted. Pass ``client``
    to reuse one Gemini client across pages.
    """
    from google.genai import types

    cfg = cfg or FigureConfig()
    if not figures or not crops:
        return {}
    if client is None:
        client = make_client(cfg)
    parts = [
        types.Part.from_bytes(
            data=_render_bytes(Path(crop), cfg.max_image_px),
            mime_type="image/jpeg",
        )
        for crop in crops
    ]
    response = _generate(client, cfg.model, [*parts, _prompt(figures, cfg)])
    return _parse(response.text, cfg)
