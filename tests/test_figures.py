"""Tests for Gemini figure descriptions (pure helpers; the network call is live-only)."""

import io

from paper_refinery.config import FigureConfig
from paper_refinery.figures import _parse, _prompt, _render_bytes


def test_prompt_lists_each_figure_with_its_caption():
    cfg = FigureConfig()
    p = _prompt([("4.1", "Gray-Scott evolution"), ("4.2", "Helmholtz coefficients")], cfg)
    assert cfg.prompt in p
    assert "- 4.1: Gray-Scott evolution" in p
    assert "- 4.2: Helmholtz coefficients" in p


def test_parse_reads_number_to_description_json():
    out = _parse('{"4.1": "desc one", "4.2": "desc two"}', FigureConfig())
    assert out == {"4.1": "desc one", "4.2": "desc two"}


def test_parse_tolerates_fences_and_prose():
    out = _parse('Sure:\n```json\n{"4.1": "d"}\n```', FigureConfig())
    assert out == {"4.1": "d"}


def test_parse_drops_skip_marker_and_handles_garbage():
    cfg = FigureConfig()
    assert _parse('{"4.1": "NOT_A_FIGURE", "4.2": "real"}', cfg) == {"4.2": "real"}
    assert _parse("no json here", cfg) == {}
    assert _parse(None, cfg) == {}


def test_render_bytes_downscales_long_side(tmp_path):
    from PIL import Image

    src = tmp_path / "page.png"
    Image.new("RGB", (2000, 2600), "white").save(src)
    w, h = Image.open(io.BytesIO(_render_bytes(src, max_px=1024))).size
    assert 1020 <= max(w, h) <= 1024  # long side at the cap
    assert h > w  # portrait aspect preserved


def test_render_bytes_does_not_upscale_small_images(tmp_path):
    from PIL import Image

    src = tmp_path / "small.png"
    Image.new("RGB", (500, 400), "white").save(src)
    assert Image.open(io.BytesIO(_render_bytes(src, max_px=1024))).size == (500, 400)


def test_describe_page_figures_uses_injected_client(tmp_path):
    # an injected client means no make_client / no API key needed
    from PIL import Image

    from paper_refinery.figures import describe_page_figures

    img = tmp_path / "page.png"
    Image.new("RGB", (80, 100), "white").save(img)

    class FakeClient:
        class models:
            @staticmethod
            def generate_content(model, contents):
                return type("R", (), {"text": '{"4.1": "a description"}'})()

    out = describe_page_figures(img, [("4.1", "caption")], client=FakeClient())
    assert out == {"4.1": "a description"}
