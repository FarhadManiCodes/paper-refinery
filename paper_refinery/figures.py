"""Figure understanding via Gemini: one call per figure.

Each call carries one figure's crop(s) -- a multi-panel figure split into several
crops by the layout model is still ONE call (enrich.py groups panels geometrically) --
plus a prompt built from three layers:

1. critical instructions (``FigureConfig.prompt``): strict visual-only description;
   the paper text may be used ONLY as a dictionary for acronyms/variables/axis labels,
   never summarized, and no conclusion is stated unless visibly plotted;
2. a figure-type taxonomy (``_TAXONOMY``): the model first identifies which type the
   figure is, then follows that type's attention checklist -- a convergence plot gets
   asked about axis scales and slopes, a mesh about element type and refinement. One
   embedded taxonomy beats a separate classify-then-describe call: no second model,
   no misclassification marching the describer down the wrong prompt, and the chosen
   type comes back as metadata;
3. reference text: paper title, abstract, the figure's own caption, and its
   neighboring paragraphs (assembled by enrich.py).

Output is schema-enforced JSON ({figure_type, description}), and every response is
cached on disk keyed by crop bytes + prompt, so re-running a paper re-bills nothing.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path

from pydantic import BaseModel, Field

from .config import FigureConfig
from .retry import call_with_backoff

# (type id, what to pay attention to) -- shown to the model verbatim. Types cover the
# figures that actually appear in computational engineering / CS / math papers.
_TAXONOMY: tuple[tuple[str, str], ...] = (
    ("line_plot", "curves compared, axis variables and scales, crossings, visible trends"),
    (
        "convergence_plot",
        "log/linear axes, error vs resolution/iterations, slopes or annotated convergence orders",
    ),
    ("scatter_plot", "variables, clustering, correlation direction, outliers, any fitted line"),
    ("bar_chart", "categories compared, which bars dominate, orderings"),
    (
        "contour_field_plot",
        "field shown, colorbar variable and direction, domain geometry, high/low regions",
    ),
    ("vector_flow_plot", "streamlines/arrows, flow direction, recirculation zones, boundaries"),
    ("mesh_discretization", "element type (tri/quad/tet/hex), refinement regions, boundaries"),
    ("geometry_schematic", "components, labeled dimensions, boundary conditions, coordinate axes"),
    ("experimental_setup", "apparatus components, instrumentation, physical arrangement"),
    ("block_diagram", "blocks/stages, direction of connections, inputs/outputs, feedback loops"),
    ("nn_architecture", "layers/modules, connectivity, input/output labels or shapes"),
    ("phase_portrait", "state variables on the axes, fixed points/attractors, trajectory shape"),
    ("time_series", "signals plotted, transient vs steady behavior, events or discontinuities"),
    ("spectrum_plot", "frequency axis and scale, dominant peaks and their order, decay behavior"),
    ("heatmap_matrix", "what rows/columns index, diagonal/off-diagonal structure, extremes"),
    ("graph_tree", "what nodes and edges represent, topology, highlighted paths"),
    ("algorithm_listing", "the visible steps/structure only; invent no semantics beyond the text"),
    ("table_image", "row/column headers and the quantities tabulated"),
    (
        "multi_panel_composite",
        "each panel's type and content briefly, and what varies across the panels",
    ),
    ("non_figure", "a logo, banner, or decorative strip -- return an empty description"),
)


class FigureDescription(BaseModel):
    """Schema-enforced response shape; replaces prose-tolerant JSON fishing."""

    figure_type: str = Field(description="One type id from the FIGURE TYPES list, verbatim.")
    description: str = Field(
        description="The visual-only description (2-5 sentences); empty for non_figure."
    )


def _crop_bytes(crop: Path, max_px: int) -> bytes:
    """JPEG bytes of the figure crop, downscaled so its long side <= max_px."""
    from PIL import Image

    img = Image.open(crop).convert("RGB")
    if max(img.size) > max_px:
        img.thumbnail((max_px, max_px))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def build_prompt(number: str, caption: str, context: dict, cfg: FigureConfig) -> str:
    """Assemble the full per-figure prompt: instructions + taxonomy + reference text.

    ``context`` keys (all optional; enrich.py fills them): ``title``, ``abstract``,
    ``before``, ``after`` -- absent/empty entries are omitted rather than sent blank.
    Public (not underscored) because the disk cache keys on this exact string: the
    prompt IS part of the cache identity, and tests pin its structure.
    """
    types_block = "\n".join(f"- {tid}: attend to {hint}" for tid, hint in _TAXONOMY)
    ref_lines = [f"Caption: FIGURE {number}. {caption}".rstrip(". ") + "."]
    for label, key in (
        ("Title", "title"),
        ("Abstract", "abstract"),
        ("Preceding paragraphs", "before"),
        ("Succeeding paragraphs", "after"),
    ):
        value = (context.get(key) or "").strip()
        if value:
            ref_lines.append(f"{label}: {value}")
    return (
        f"{cfg.prompt}\n\n"
        f"FIGURE TYPES -- identify which one fits, then follow its checklist:\n{types_block}\n\n"
        "REFERENCE TEXT:\n" + "\n\n".join(ref_lines)
    )


def _cache_path(crops: list[Path], prompt: str, cfg: FigureConfig) -> Path | None:
    """Cache key = sha256 of the crop BYTES + the exact prompt -- renaming a crop file
    (enrich.py renames to fig_N.png) or re-running an unchanged paper never re-bills;
    any change to the pixels, the context, or the prompt text naturally re-describes."""
    if not cfg.figure_cache_dir:
        return None
    digest = hashlib.sha256()
    for crop in crops:
        digest.update(Path(crop).read_bytes())
    digest.update(prompt.encode())
    return Path(cfg.figure_cache_dir).expanduser() / f"{digest.hexdigest()}.json"


def make_client(cfg: FigureConfig | None = None):
    """Create a Gemini client. Build one and reuse it across figures (see cli.py).

    Assumes API keys are already loaded into the environment -- see
    ``config.load_config``'s ``_load_secrets`` call.
    """
    from google import genai

    cfg = cfg or FigureConfig()
    api_key = os.environ.get(cfg.api_key_env)
    if not api_key:
        raise RuntimeError(f"{cfg.api_key_env} is not set")
    return genai.Client(api_key=api_key)


def describe_figure(
    crops: list[Path],
    number: str,
    caption: str,
    context: dict | None = None,
    cfg: FigureConfig | None = None,
    client=None,
) -> dict | None:
    """Describe one figure (all its panel crops together) in one Gemini call.

    Returns ``{"figure_type": ..., "description": ...}``, or None when the model says
    non_figure / returns nothing usable. A definitive non_figure verdict IS cached
    (as ``{}``) -- a banner stays skipped for free on re-runs; transport/schema
    failures are never cached and surface as exceptions for the caller to handle.
    """
    from google.genai import types

    cfg = cfg or FigureConfig()
    if not crops:
        return None
    prompt = build_prompt(number, caption, context or {}, cfg)
    cache = _cache_path(crops, prompt, cfg)
    if cache and cache.exists():
        try:
            data = json.loads(cache.read_text())
        except ValueError:
            data = None  # corrupt entry: fall through to a real call
        if isinstance(data, dict):
            return data or None  # {} is a cached non_figure verdict

    if client is None:
        client = make_client(cfg)
    parts = [
        types.Part.from_bytes(data=_crop_bytes(c, cfg.max_image_px), mime_type="image/jpeg")
        for c in crops
    ]
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=FigureDescription,
    )
    response = call_with_backoff(
        lambda: client.models.generate_content(
            model=cfg.model, contents=[*parts, prompt], config=config
        ),
        cfg.retry_attempts,
        cfg.retry_base_delay,
    )
    parsed = response.parsed
    if parsed is None:
        return None  # schema coercion failed -- not cached, next run retries
    result: dict | None = {
        "figure_type": parsed.figure_type.strip(),
        "description": parsed.description.strip(),
    }
    if result["figure_type"] == "non_figure" or not result["description"]:
        result = None
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(result or {}))
    return result
