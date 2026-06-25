"""Figure understanding via Gemini.

Describes each extracted figure for retrieval (what is compared, axes/variables, and
qualitative trends). Never invents precise numeric values read off plotted curves —
those belong to the paper's own tables.
"""

from __future__ import annotations

from pathlib import Path

from .config import FigureConfig


def describe_figure(
    image_path: Path, caption: str | None = None, cfg: FigureConfig | None = None
) -> str:
    """Return a short, retrieval-oriented description of a figure image.

    TODO: implement with google-genai:
      - client = genai.Client(api_key=os.environ[cfg.api_key_env])
      - upload the image, prompt with cfg.prompt (+ caption for context)
      - return the response text (trends/comparisons only, no fabricated numbers)
    """
    raise NotImplementedError("describe_figure: Gemini integration not implemented yet")
