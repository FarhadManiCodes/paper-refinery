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


def test_parse_config_has_no_dual_backend_flag():
    # single local backend (llama-server + glmocr); no LlamaParse/cloud fallback flag
    keys = ParseConfig().__dict__.keys()
    assert "api_key_env" not in keys
    assert "parse_mode" not in keys


def test_parse_config_local_backend_defaults():
    c = ParseConfig()
    assert c.table_format == "markdown"
    assert c.merged_cell_strategy == "duplicate"
    assert c.model_path == ""
    assert c.mmproj_path == ""
    assert c.glmocr_config_overrides == {}


def test_figure_prompt_forbids_fabricated_numbers():
    assert "Do NOT" in FigureConfig().prompt
