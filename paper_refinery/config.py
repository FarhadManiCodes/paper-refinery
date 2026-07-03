"""Tunable configuration for the refinery pipeline."""

from __future__ import annotations

import os
import stat
import tomllib
import warnings
from dataclasses import dataclass, field
from pathlib import Path


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
    # GLM-OCR GGUF weights / vision projector -- not secrets, just local file paths, so
    # they're read from ~/.config/paper-refinery/config.toml (see `load_config` below)
    # rather than an env var. Still empty -> RuntimeError (no guessed path). The
    # `refinery` CLI's --model-path/--mmproj-path flags take precedence over the file.
    model_path: str = ""
    mmproj_path: str = ""
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
    figure_crop_margin: float = 1.1
    #   multiplier applied to each detected figure/chart box's width and height before
    #   cropping (1.1 = 10% larger on each axis) -- glmocr's own default is 1.0, i.e. no
    #   margin, which can clip axis labels/legends/edges sitting right at the detected
    #   boundary. Applied only to the "chart"/"image" layout classes (see
    #   _FIGURE_CLASS_IDS in parse.py); text/table/formula crops keep native detection
    #   precision, since a looser box there would just add OCR noise.
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
    retry_attempts: int = 4  # generate_content attempts before giving up
    retry_base_delay: float = 4.0  # seconds; doubles each retry (4, 8, 16, ...)
    max_workers: int = 4  # concurrent per-page describe_page_figures calls in enrich.py
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
class CitationConfig:
    """Bibliography extraction: one Gemini call turns each raw OCR'd reference string
    into rough structured fields (title, authors, year, venue, ...). A labeling guess,
    not ground truth -- verifying it against real bibliographic data is a separate,
    not-yet-designed concern, deliberately not part of this config."""

    model: str = "gemini-3.1-flash-lite"
    api_key_env: str = "GOOGLE_API_KEY"
    retry_attempts: int = 4  # generate_content attempts before giving up
    retry_base_delay: float = 4.0  # seconds; doubles each retry


@dataclass
class RefineryConfig:
    """Top-level config bundling each stage."""

    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    parse: ParseConfig = field(default_factory=ParseConfig)
    figure: FigureConfig = field(default_factory=FigureConfig)
    citation: CitationConfig = field(default_factory=CitationConfig)


DEFAULT_CONFIG_PATH = Path.home() / ".config" / "paper-refinery" / "config.toml"
DEFAULT_SECRETS_DIR = Path.home() / ".config" / "paper-refinery" / "secrets"
#   One small file per service (e.g. google.env holding GOOGLE_API_KEY, hf.env holding
#   HF_TOKEN) rather than one combined file -- keeps each credential's blast radius
#   minimal. Deliberately its own directory, never shared with any other tool's secrets
#   (e.g. papis-ask's own env can set OPENAI_BASE_URL/OPENAI_API_KEY for its local
#   embedding server -- sourcing that into this process would silently redirect
#   glmocr's OpenAI-compatible client away from our own local llama-server).
#   Overridable via $PAPER_REFINERY_SECRETS_DIR. User-managed: paper-refinery only ever
#   reads files here (via python-dotenv, which no-ops if the directory is absent/empty),
#   never creates or writes any of them.


def _load_secrets(dir_path: Path | None = None) -> None:
    """Load every ``*.env`` file in the secrets directory into the environment, if any
    exist.

    Delegates entirely to ``python-dotenv`` for each file -- this code never opens a
    file itself, never logs or inspects its contents, and never overwrites a variable
    already set some other way (``load_dotenv``'s default, ``override=False``). The one
    thing it does check is each file's permission bits, since that's determinable
    without ever reading what's inside: a warning (not a failure) for a file readable by
    more than its owner.
    """
    from dotenv import load_dotenv

    dir_path = dir_path or Path(
        os.environ.get("PAPER_REFINERY_SECRETS_DIR", DEFAULT_SECRETS_DIR)
    )
    if not dir_path.is_dir():
        return
    for env_file in sorted(dir_path.glob("*.env")):
        mode = stat.S_IMODE(env_file.stat().st_mode)
        if mode & (stat.S_IRWXG | stat.S_IRWXO):
            warnings.warn(
                f"{env_file} is readable by more than its owner (mode {oct(mode)}) -- "
                f"consider `chmod 600 {env_file}`"
            )
        load_dotenv(env_file)


def load_config(path: Path | None = None) -> RefineryConfig:
    """Build a RefineryConfig, first loading API keys (see ``_load_secrets``), then
    overlaying values from a TOML file (XDG-style user config).

    Secrets are loaded here, early, rather than lazily at each Gemini call site: the
    secrets directory is dedicated to this project alone (no risk of an unrelated
    tool's env vars, like papis-ask's OPENAI_BASE_URL, leaking in), and HF_TOKEN needs
    to already be set before parse_pdf() runs, not just before a later Gemini call, to
    matter for the HF Hub version-check glmocr/transformers make.

    The TOML file is optional -- every field already has a code default -- and only
    needs to set what differs from that default, e.g. local, machine-specific GLM-OCR
    model paths::

        [parse]
        model_path = "/home/you/.cache/paper-refinery/models/GLM-OCR-f16.gguf"
        mmproj_path = "/home/you/.cache/paper-refinery/models/mmproj-GLM-OCR-Q8_0.gguf"

    Each top-level TOML table maps to a ``RefineryConfig`` sub-config by name (``parse``,
    ``figure``, ``chunk``, ``citation``); each key in it must match a dataclass field on
    that sub-config.
    """
    _load_secrets()
    cfg = RefineryConfig()
    path = path or DEFAULT_CONFIG_PATH
    if not path.exists():
        return cfg
    with path.open("rb") as f:
        data = tomllib.load(f)
    for section, values in data.items():
        sub = getattr(cfg, section, None)
        if sub is None:
            raise ValueError(f"{path}: unknown config section [{section}]")
        for key, value in values.items():
            if not hasattr(sub, key):
                raise ValueError(f"{path}: unknown key '{key}' in [{section}]")
            setattr(sub, key, value)
    return cfg
