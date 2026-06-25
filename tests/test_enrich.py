"""Tests for caption-anchored figure-description splicing (to be implemented)."""

import pytest


@pytest.mark.skip(reason="enrich_markdown not implemented; add a caption-anchor fixture test")
def test_description_inserted_after_matching_caption():
    """When implemented: given markdown with 'FIGURE 4.3 ...' and a description for the
    page-9 figure, assert the description lands right after that caption."""
