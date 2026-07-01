"""Sanity checks on default configuration."""

import pytest

from paper_refinery.config import (
    ChunkConfig,
    FigureConfig,
    ParseConfig,
    RefineryConfig,
    load_config,
)


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
    assert c.figure_crop_margin == 1.1
    assert c.glmocr_config_overrides == {}


def test_figure_prompt_forbids_fabricated_numbers():
    assert "Do NOT" in FigureConfig().prompt


def test_figure_config_retry_defaults():
    c = FigureConfig()
    assert c.retry_attempts == 4
    assert c.retry_base_delay == 4.0


def test_load_config_returns_defaults_when_file_missing(tmp_path):
    cfg = load_config(tmp_path / "does-not-exist.toml")
    assert cfg == RefineryConfig()


def test_load_config_overlays_parse_section(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[parse]\nmodel_path = "/models/glm-ocr.gguf"\nn_gpu_layers = 20\n')
    cfg = load_config(path)
    assert cfg.parse.model_path == "/models/glm-ocr.gguf"
    assert cfg.parse.n_gpu_layers == 20
    assert cfg.parse.mmproj_path == ""  # untouched fields keep their code default


def test_load_config_rejects_unknown_section(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("[bogus]\nx = 1\n")
    with pytest.raises(ValueError, match="unknown config section"):
        load_config(path)


def test_load_config_rejects_unknown_key(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("[parse]\nnot_a_real_field = 1\n")
    with pytest.raises(ValueError, match="unknown key"):
        load_config(path)
