"""Tests for Gemini figure descriptions.

Pure helpers (mime detection, prompt building) are tested directly; the network call
is exercised live only (skipped here).
"""

from pathlib import Path

import pytest

from paper_refinery.config import FigureConfig
from paper_refinery.figures import _mime_type, _prompt


def test_mime_type_from_extension():
    assert _mime_type(Path("a.png")) == "image/png"
    assert _mime_type(Path("a.JPG")) == "image/jpeg"
    assert _mime_type(Path("a.jpeg")) == "image/jpeg"
    assert _mime_type(Path("a.unknown")) == "image/png"  # safe default


def test_prompt_is_base_instructions_without_context():
    cfg = FigureConfig()
    assert _prompt(None, cfg) == cfg.prompt
    assert "Do NOT" in _prompt(None, cfg)  # forbids fabricated numbers


def test_prompt_grounds_on_context_when_present():
    cfg = FigureConfig()
    context = "FIGURE 4.3 Error evolution\nFigure 4.3 shows the evolution of errors."
    p = _prompt(context, cfg)
    assert context in p
    assert cfg.prompt in p


@pytest.mark.skip(reason="describe_figure needs Gemini (network); run live, not in CI")
def test_describe_figure_returns_trend_text_without_fabricated_numbers():
    """When run live: describe a real plot image and assert non-empty text."""
