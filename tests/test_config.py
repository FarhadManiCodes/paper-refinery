"""Sanity checks on default configuration."""

import pytest

from paper_refinery.config import (
    ChunkConfig,
    CitationConfig,
    FigureConfig,
    ParseConfig,
    RefineryConfig,
    _default_config_path,
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
    assert isinstance(r.citation, CitationConfig)


def test_citation_config_defaults():
    c = CitationConfig()
    assert c.model == "gemini-3.1-flash-lite"
    assert c.api_key_env == "GOOGLE_API_KEY"
    assert c.retry_attempts == 4
    assert c.retry_base_delay == 4.0


def test_parse_config_defaults_to_selfhosted_with_maas_option():
    # dual backend: local llama-server + glmocr (default), or Zhipu cloud OCR (maas)
    c = ParseConfig()
    assert c.mode == "selfhosted"  # local is the default; maas is opt-in
    assert c.api_key_env == "ZHIPU_API_KEY"
    assert c.maas_api_url.endswith("/layout_parsing")
    assert c.maas_model == "glm-ocr"


def test_parse_config_local_backend_defaults():
    c = ParseConfig()
    assert c.model_path == ""
    assert c.mmproj_path == ""
    assert c.figure_crop_margin == 1.1
    assert c.glmocr_config_overrides == {}


def test_figure_prompt_constrains_context_and_numbers():
    prompt = FigureConfig().prompt
    assert "ONLY as a" in prompt  # reference text is a dictionary, never summarized
    assert "Do not report precise numeric values" in prompt


def test_figure_config_retry_defaults():
    c = FigureConfig()
    assert c.retry_attempts == 4
    assert c.retry_base_delay == 4.0
    assert c.max_workers == 4


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


def test_load_config_overlays_citation_section(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[citation]\nmodel = "gemini-3.1-flash"\nretry_attempts = 2\n')
    cfg = load_config(path)
    assert cfg.citation.model == "gemini-3.1-flash"
    assert cfg.citation.retry_attempts == 2


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


def test_load_config_rejects_wrong_type(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[parse]\nport = "8080"\n')  # quoted -> string, but port is an int
    with pytest.raises(ValueError, match="expects int"):
        load_config(path)


def test_load_config_coerces_int_to_float(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("[parse]\nfigure_crop_margin = 2\n")  # TOML int -> float field
    cfg = load_config(path)
    assert cfg.parse.figure_crop_margin == 2.0
    assert isinstance(cfg.parse.figure_crop_margin, float)


def test_load_config_coerces_list_to_tuple_for_server_args(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[parse]\nextra_server_args = ["--flash-attn", "off"]\n')
    cfg = load_config(path)
    assert cfg.parse.extra_server_args == ("--flash-attn", "off")


def test_default_config_path_honors_xdg_config_home(monkeypatch, tmp_path):
    monkeypatch.delenv("PAPER_REFINERY_CONFIG", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert _default_config_path() == tmp_path / "paper-refinery" / "config.toml"


def test_default_config_path_env_override_wins(monkeypatch, tmp_path):
    override = tmp_path / "custom.toml"
    monkeypatch.setenv("PAPER_REFINERY_CONFIG", str(override))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "ignored"))
    assert _default_config_path() == override
