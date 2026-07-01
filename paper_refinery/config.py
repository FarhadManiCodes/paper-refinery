"""Tunable configuration for the refinery pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ChunkConfig:
    """Section-aware chunking with a guaranteed soft-window overlap."""

    max_chars: int = 6000  # sub-split sections larger than this
    min_chars: int = 1000  # merge sections smaller than this into a neighbour
    overlap_lo: int = 300  # soft overlap window: minimum size
    overlap_hi: int = 700  # soft overlap window: maximum size
    overlap_ideal: int = 500  # preferred overlap size within the window


@dataclass
class ParseConfig:
    """Local GLM-OCR backend: llama-server (inference) + glmocr SDK (layout + orchestration)."""

    llama_server_bin: str = "llama-server"  # resolved via PATH unless overridden
    model_path: str = ""  # GLM-OCR GGUF weights; empty -> RuntimeError, no guessed path
    mmproj_path: str = ""  # GLM-OCR GGUF vision projector; empty -> RuntimeError
    host: str = "127.0.0.1"
    port: int = 8080
    n_gpu_layers: int = 99  # -ngl: offload all layers (assumes a GPU is available)
    extra_server_args: tuple[str, ...] = ("--flash-attn", "off", "-fit", "off")
    #   required for GLM-OCR as of ggml-org/llama.cpp discussion #19721; re-check on upgrade
    startup_timeout_s: float = 120.0  # health-check polling budget (model load can be slow)
    layout_device: str | None = None  # None = glmocr auto-selects CUDA/CPU for PP-DocLayout-V3
    table_format: str = "markdown"  # glmocr emits HTML tables; we convert to markdown
    merged_cell_strategy: str = "duplicate"  # rowspan/colspan fallback: no lossless markdown equivalent
    figures_dir_name: str = "figures"  # subdir of image_dir where figure/chart crops are saved
    references_suffix: str = ".references.json"  # sidecar: "{pdf.stem}{references_suffix}"
    glmocr_config_overrides: dict = field(default_factory=dict)
    #   dotted-path escape hatch into glmocr's own config (e.g. {"pipeline.max_workers": 1}
    #   to cut region-OCR concurrency on constrained hardware); forwarded as GlmOcr(_dotted=...)


@dataclass
class FigureConfig:
    """Gemini figure-description options."""

    model: str = "gemini-3-flash-preview"  # more accurate figure reading than 2.5-flash
    api_key_env: str = "GOOGLE_API_KEY"
    skip_marker: str = "NOT_A_FIGURE"  # Gemini omits / flags figures not on the page
    include_references: bool = False  # also feed in-text "Figure N" mentions as context
    # (off = caption-only context; cross-referencing is a future improvement)
    max_image_px: int = 1024  # downscale each crop's long side before sending (saves tokens)
    # One call per page: describe every listed figure, return JSON {number: description}.
    prompt: str = (
        "The attached image(s) are cropped figure/chart regions from a scientific "
        "paper page. Together they contain the figure(s) listed below by caption. "
        "For EACH listed figure, write a 2-4 "
        "sentence description for search and retrieval: what is compared, the "
        "variables/axes, and the qualitative trends or conclusions. Do NOT report "
        "precise numeric values read off plotted curves — give ranges or directions "
        "only. Respond with ONLY a JSON object mapping each figure number (as a "
        'string) to its description, e.g. {"4.1": "...", "4.2": "..."}. If a listed '
        "figure is not actually present on the page, omit it from the JSON."
    )


@dataclass
class RefineryConfig:
    """Top-level config bundling each stage."""

    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    parse: ParseConfig = field(default_factory=ParseConfig)
    figure: FigureConfig = field(default_factory=FigureConfig)
