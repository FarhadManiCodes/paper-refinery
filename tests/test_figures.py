"""Tests for per-figure Gemini description (pure/mocked; the real call is live-only)."""

import io

from paper_refinery.config import FigureConfig
from paper_refinery.figures import FigureDescription, _crop_bytes, build_prompt, describe_figure


def _cfg(tmp_path=None, **kw) -> FigureConfig:
    # never let a unit test touch the real on-disk cache
    kw.setdefault("figure_cache_dir", str(tmp_path / "cache") if tmp_path else "")
    return FigureConfig(**kw)


def _png(tmp_path, name="crop.png", size=(40, 30), color="red"):
    from PIL import Image

    path = tmp_path / name
    Image.new("RGB", size, color).save(path)
    return path


def _client(parsed, calls=None):
    class FakeClient:
        class models:
            @staticmethod
            def generate_content(model, contents, config):
                if calls is not None:
                    calls.append((model, contents, config))
                return type("R", (), {"parsed": parsed})()

    return FakeClient()


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------


def test_build_prompt_layers_instructions_taxonomy_and_reference_text():
    cfg = _cfg()
    context = {
        "title": "Churning Losses in Gearboxes",
        "abstract": "We study churning losses.",
        "before": "The preceding paragraph.",
        "after": "The succeeding paragraph.",
    }
    p = build_prompt("4", "Power loss vs speed.", context, cfg)
    assert p.startswith(cfg.prompt)
    assert "- convergence_plot: attend to" in p  # taxonomy embedded
    assert "- non_figure:" in p
    assert "Caption: FIGURE 4. Power loss vs speed." in p
    assert "Title: Churning Losses in Gearboxes" in p
    assert "Preceding paragraphs: The preceding paragraph." in p
    assert "Succeeding paragraphs: The succeeding paragraph." in p


def test_build_prompt_omits_empty_context_fields():
    p = build_prompt("2", "A caption.", {"title": "", "abstract": "  "}, _cfg())
    assert "Title:" not in p
    assert "Abstract:" not in p
    assert "Caption: FIGURE 2. A caption." in p


# ---------------------------------------------------------------------------
# _crop_bytes
# ---------------------------------------------------------------------------


def test_crop_bytes_downscales_long_side(tmp_path):
    from PIL import Image

    out = _crop_bytes(_png(tmp_path, size=(400, 200)), max_px=100)
    img = Image.open(io.BytesIO(out))
    assert max(img.size) <= 100


def test_crop_bytes_does_not_upscale_small_images(tmp_path):
    from PIL import Image

    out = _crop_bytes(_png(tmp_path, size=(40, 30)), max_px=100)
    img = Image.open(io.BytesIO(out))
    assert img.size == (40, 30)


# ---------------------------------------------------------------------------
# describe_figure
# ---------------------------------------------------------------------------


def test_describe_figure_returns_type_and_description(tmp_path):
    crop = _png(tmp_path)
    parsed = FigureDescription(figure_type="line_plot", description="Two curves compared.")
    calls = []
    out = describe_figure(
        [crop], "4", "A caption.", {}, _cfg(tmp_path), client=_client(parsed, calls)
    )
    assert out == {"figure_type": "line_plot", "description": "Two curves compared."}
    # one image part per crop, prompt as the trailing text part
    (model, contents, config) = calls[0]
    assert len(contents) == 2
    assert isinstance(contents[-1], str) and contents[-1].startswith(_cfg().prompt[:20])
    assert config.response_schema is FigureDescription


def test_describe_figure_sends_all_panels_in_one_call(tmp_path):
    crops = [_png(tmp_path, "a.png", color="red"), _png(tmp_path, "b.png", color="blue")]
    parsed = FigureDescription(figure_type="multi_panel_composite", description="Panels.")
    calls = []
    describe_figure(crops, "3", "Cap.", {}, _cfg(tmp_path), client=_client(parsed, calls))
    assert len(calls[0][1]) == 3  # two image parts + one prompt


def test_describe_figure_non_figure_returns_none(tmp_path):
    parsed = FigureDescription(figure_type="non_figure", description="")
    out = describe_figure(
        [_png(tmp_path)], "1", "Banner.", {}, _cfg(tmp_path), client=_client(parsed)
    )
    assert out is None


def test_describe_figure_empty_description_returns_none(tmp_path):
    parsed = FigureDescription(figure_type="line_plot", description="   ")
    out = describe_figure([_png(tmp_path)], "1", "Cap.", {}, _cfg(tmp_path), client=_client(parsed))
    assert out is None


def test_describe_figure_no_crops_returns_none():
    def boom():
        raise AssertionError("no client should be built")

    assert describe_figure([], "1", "Cap.", {}, _cfg(), client=boom) is None


# ---------------------------------------------------------------------------
# the disk cache
# ---------------------------------------------------------------------------


def test_describe_figure_cache_hit_skips_the_call(tmp_path):
    crop = _png(tmp_path)
    cfg = _cfg(tmp_path)
    parsed = FigureDescription(figure_type="mesh_discretization", description="A mesh.")
    calls = []
    first = describe_figure([crop], "2", "Cap.", {}, cfg, client=_client(parsed, calls))

    class ExplodingClient:
        class models:
            @staticmethod
            def generate_content(model, contents, config):
                raise AssertionError("cache hit must not call Gemini")

    second = describe_figure([crop], "2", "Cap.", {}, cfg, client=ExplodingClient())
    assert first == second and len(calls) == 1


def test_describe_figure_cache_key_includes_context(tmp_path):
    # different context -> different prompt -> a fresh call, not a stale cache hit
    crop = _png(tmp_path)
    cfg = _cfg(tmp_path)
    calls = []
    parsed = FigureDescription(figure_type="line_plot", description="D.")
    describe_figure([crop], "2", "Cap.", {"title": "A"}, cfg, client=_client(parsed, calls))
    describe_figure([crop], "2", "Cap.", {"title": "B"}, cfg, client=_client(parsed, calls))
    assert len(calls) == 2


def test_describe_figure_caches_non_figure_verdict(tmp_path):
    crop = _png(tmp_path)
    cfg = _cfg(tmp_path)
    parsed = FigureDescription(figure_type="non_figure", description="")
    assert describe_figure([crop], "1", "Cap.", {}, cfg, client=_client(parsed)) is None

    class ExplodingClient:
        class models:
            @staticmethod
            def generate_content(model, contents, config):
                raise AssertionError("a cached non_figure verdict must stay free")

    assert describe_figure([crop], "1", "Cap.", {}, cfg, client=ExplodingClient()) is None


def test_describe_figure_schema_failure_is_not_cached(tmp_path):
    crop = _png(tmp_path)
    cfg = _cfg(tmp_path)
    assert describe_figure([crop], "1", "Cap.", {}, cfg, client=_client(None)) is None
    assert not any((tmp_path / "cache").glob("*.json"))  # next run gets to retry


def test_describe_figure_rename_does_not_invalidate_cache(tmp_path):
    # enrich renames crops to fig_N.png AFTER pairing -- the cache keys on bytes
    crop = _png(tmp_path, "page_2_fig_0.png")
    cfg = _cfg(tmp_path)
    parsed = FigureDescription(figure_type="line_plot", description="D.")
    calls = []
    describe_figure([crop], "2", "Cap.", {}, cfg, client=_client(parsed, calls))
    renamed = crop.with_name("fig_2.png")
    crop.replace(renamed)
    describe_figure([renamed], "2", "Cap.", {}, cfg, client=_client(parsed, calls))
    assert len(calls) == 1


def test_describe_figure_empty_cache_dir_disables_caching(tmp_path):
    crop = _png(tmp_path)
    cfg = _cfg(figure_cache_dir="")
    parsed = FigureDescription(figure_type="line_plot", description="D.")
    calls = []
    describe_figure([crop], "2", "Cap.", {}, cfg, client=_client(parsed, calls))
    describe_figure([crop], "2", "Cap.", {}, cfg, client=_client(parsed, calls))
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# orphan combining-mark cleanup (kalman mojibake)
# ---------------------------------------------------------------------------


def test_strip_orphan_combining_drops_glyphs_with_no_base():
    from paper_refinery.figures import _strip_orphan_combining

    # the live kalman case: a stray combining mark where "Phi" belongs
    assert (
        _strip_orphan_combining("through a block labeled ̓(t + 1; t) back")
        == "through a block labeled (t + 1; t) back"
    )
    # start-of-string orphan
    assert _strip_orphan_combining("̀abc") == "abc"


def test_strip_orphan_combining_keeps_genuine_accents():
    from paper_refinery.figures import _strip_orphan_combining

    decomposed = "Hénon map"  # e + combining acute: a real accent, attached
    assert _strip_orphan_combining(decomposed) == decomposed


def test_describe_figure_cleans_description(tmp_path):
    parsed = FigureDescription(figure_type="block_diagram", description="gain ̓(t) loop")
    out = describe_figure([_png(tmp_path)], "1", "Cap.", {}, _cfg(tmp_path), client=_client(parsed))
    assert out["description"] == "gain (t) loop"
