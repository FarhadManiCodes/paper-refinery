"""Sanity checks on default configuration."""

from paper_refinery.config import ChunkConfig, FigureConfig, ParseConfig, RefineryConfig


def test_chunk_defaults_are_consistent():
    c = ChunkConfig()
    assert c.overlap_lo < c.overlap_ideal < c.overlap_hi
    assert c.min_chars < c.max_chars


def test_refinery_config_composes_subconfigs():
    r = RefineryConfig()
    assert isinstance(r.chunk, ChunkConfig)
    assert isinstance(r.parse, ParseConfig)
    assert isinstance(r.figure, FigureConfig)


def test_chart_to_table_is_off_by_default():
    # specialized chart parsing fabricates numbers; must stay off (design decision)
    assert "specialized_chart" not in ParseConfig().__dict__  # not even exposed as on


def test_chart_images_are_extracted_by_default():
    # we DO want chart images (so figures.py can describe the method-comparison plots)
    assert ParseConfig().extract_charts is True


def test_figure_prompt_forbids_fabricated_numbers():
    assert "Do NOT" in FigureConfig().prompt
