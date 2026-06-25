"""Tests for Gemini figure descriptions (to be filled with a mocked client)."""

import pytest


@pytest.mark.skip(reason="describe_figure needs Gemini (network); add with a mock genai client")
def test_describe_figure_returns_trend_text_without_fabricated_numbers():
    """When implemented: mock the client, assert the prompt forbids reading exact
    values off curves and that the returned description is non-empty."""
