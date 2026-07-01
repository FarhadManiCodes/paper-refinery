"""Tests for Gemini figure descriptions (pure helpers; the network call is live-only)."""

import io

from paper_refinery.config import FigureConfig
from paper_refinery.figures import _crop_bytes, _parse, _prompt


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


def test_parse_ignores_stray_brace_in_leading_prose():
    # a greedy \{.*\} regex would span from the example's brace all the way to the
    # real answer's closing brace, producing malformed JSON and silently losing the
    # answer -- the real JSON here is the last one, closest '{' to the final '}'
    text = 'The format looks like {"example": "ignore this"}. Here:\n{"4.1": "real desc"}'
    assert _parse(text, FigureConfig()) == {"4.1": "real desc"}


def test_crop_bytes_downscales_long_side(tmp_path):
    from PIL import Image

    src = tmp_path / "crop.png"
    Image.new("RGB", (2000, 2600), "white").save(src)
    w, h = Image.open(io.BytesIO(_crop_bytes(src, max_px=1024))).size
    assert 1020 <= max(w, h) <= 1024  # long side at the cap
    assert h > w  # portrait aspect preserved


def test_crop_bytes_does_not_upscale_small_images(tmp_path):
    from PIL import Image

    src = tmp_path / "small.png"
    Image.new("RGB", (500, 400), "white").save(src)
    assert Image.open(io.BytesIO(_crop_bytes(src, max_px=1024))).size == (500, 400)


def test_describe_page_figures_uses_injected_client(tmp_path):
    # an injected client means no make_client / no API key needed
    from PIL import Image

    from paper_refinery.figures import describe_page_figures

    img = tmp_path / "crop.png"
    Image.new("RGB", (80, 100), "white").save(img)

    class FakeClient:
        class models:
            @staticmethod
            def generate_content(model, contents):
                return type("R", (), {"text": '{"4.1": "a description"}'})()

    out = describe_page_figures([img], [("4.1", "caption")], client=FakeClient())
    assert out == {"4.1": "a description"}


def test_describe_page_figures_sends_one_image_part_per_crop(tmp_path):
    from PIL import Image

    from paper_refinery.figures import describe_page_figures

    crop1, crop2 = tmp_path / "c1.png", tmp_path / "c2.png"
    Image.new("RGB", (40, 40), "white").save(crop1)
    Image.new("RGB", (40, 40), "white").save(crop2)

    calls = []

    class FakeClient:
        class models:
            @staticmethod
            def generate_content(model, contents):
                calls.append(contents)
                return type("R", (), {"text": '{"1": "d1", "2": "d2"}'})()

    out = describe_page_figures(
        [crop1, crop2], [("1", "cap one"), ("2", "cap two")], client=FakeClient()
    )
    assert out == {"1": "d1", "2": "d2"}
    # two image parts + one trailing text prompt
    assert len(calls[0]) == 3


def test_describe_page_figures_returns_empty_without_crops():
    from paper_refinery.figures import describe_page_figures

    assert describe_page_figures([], [("1", "cap")], client=object()) == {}
