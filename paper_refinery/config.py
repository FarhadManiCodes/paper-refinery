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
    parse_timeout_s: float = 1800.0
    #   watchdog budget for one PDF's whole parse (layout + all OCR calls). A wedged
    #   llama-server otherwise hangs parse_pdf forever -- observed live: one run sat
    #   36+ minutes on a 12-page paper that normally takes ~10. On timeout the server
    #   is killed and parse_pdf raises. Generous by design; raise it for huge PDFs.
    layout_device: str | None = None  # None = glmocr auto-selects CUDA/CPU for PP-DocLayout-V3
    figures_dir_name: str = "figures"  # subdir of image_dir where figure/chart crops are saved
    figure_crop_margin: float = 1.1
    #   multiplier applied to each detected figure/chart box's width and height before
    #   cropping (1.1 = 10% larger on each axis) -- glmocr's own default is 1.0, i.e. no
    #   margin, which can clip axis labels/legends/edges sitting right at the detected
    #   boundary. Applied only to the "chart"/"image" layout classes (see
    #   _FIGURE_CLASS_IDS in backend.py); text/table/formula crops keep native detection
    #   precision, since a looser box there would just add OCR noise.
    glmocr_config_overrides: dict = field(default_factory=dict)
    #   dotted-path escape hatch into glmocr's own config (e.g. {"pipeline.max_workers": 1}
    #   to cut region-OCR concurrency on constrained hardware); forwarded as GlmOcr(_dotted=...)


@dataclass
class FigureConfig:
    """Gemini figure-description options."""

    model: str = "gemini-3-flash-preview"  # more accurate figure reading than 2.5-flash
    api_key_env: str = "GOOGLE_API_KEY"
    max_image_px: int = 1024  # downscale each crop's long side before sending (saves tokens)
    retry_attempts: int = 4  # generate_content attempts before giving up
    retry_base_delay: float = 4.0  # seconds; doubles each retry (4, 8, 16, ...)
    max_workers: int = 4  # concurrent per-figure describe_figure calls in enrich.py
    context_paragraphs: int = 2  # body paragraphs on each side of the caption sent as context
    figure_cache_dir: str = "~/.cache/paper-refinery/figure-cache"
    #   every schema-valid description (including a definitive non-figure verdict) is
    #   cached here, keyed by crop bytes + assembled prompt -- re-running an unchanged
    #   paper makes zero Gemini calls. Sibling of the api-cache; safe to delete anytime.
    # Critical instructions for the per-figure call; figures.py appends the figure-type
    # taxonomy and the REFERENCE TEXT block (title/abstract/caption/neighbor paragraphs).
    prompt: str = (
        "You are analyzing ONE figure cropped from a scientific paper (multiple "
        "attached images are panels of that SAME figure). Provide a strict, "
        "visual-only description for search and retrieval: what is plotted or "
        "depicted, the variables/axes/labels, and the qualitative trends or "
        "relationships visible in the image itself.\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "- The REFERENCE TEXT below is context from the paper. Use it ONLY as a "
        "dictionary to resolve acronyms, variable names, and axis labels that appear "
        "in the image.\n"
        "- Do not summarize the reference text.\n"
        "- Do not state conclusions, physics, or behaviors from the text unless they "
        "are visibly plotted in the image.\n"
        "- Do not report precise numeric values read off plotted curves -- give "
        "ranges or directions only.\n"
        "- Write mathematical symbols in plain readable text: Greek letters by name "
        "(Phi, Sigma, alpha), sub/superscripts inline (x(t+1), P_LG0). Never emit "
        "isolated combining or diacritical glyphs.\n"
        "- First identify the figure type from the FIGURE TYPES list below, then pay "
        "attention to the aspects that type calls out. Write 2-5 sentences."
    )


@dataclass
class CitationConfig:
    """Citation pipeline: extraction (one Gemini call turns each raw OCR'd reference
    string into rough structured fields) and resolution (verifying/completing those
    fields against Semantic Scholar -> CrossRef -> OpenAlex; see citation_resolution.py).
    One config class for both, mirroring how FigureConfig is shared by figures/enrich."""

    # -- extraction (citation_extraction.py) --
    model: str = "gemini-3.1-flash-lite"
    api_key_env: str = "GOOGLE_API_KEY"
    retry_attempts: int = 4  # attempts before giving up (Gemini and resolver HTTP alike)
    retry_base_delay: float = 4.0  # seconds; doubles each retry

    # -- resolution (citation_resolution.py) --
    s2_api_base: str = "https://api.semanticscholar.org/graph/v1"
    s2_api_key_env: str = "S2_API_KEY"  # optional; sent as x-api-key when set
    s2_min_interval_s: float = 1.1
    #   unauthenticated S2 hard-rate-limits bursts (~1 req/s; confirmed live: an
    #   immediate second call 429s) -- calls are globally throttled to this interval
    crossref_api_base: str = "https://api.crossref.org"
    openalex_api_base: str = "https://api.openalex.org"
    mailto: str = ""
    #   contact email for CrossRef/OpenAlex "polite pool" (better rate limits & support);
    #   optional but recommended -- set it in config.toml, not here
    request_timeout_s: float = 30.0
    max_workers: int = 4  # concurrent per-reference resolutions
    api_retry_attempts: int = 2
    api_retry_base_delay: float = 2.0
    #   resolver HTTP gets a smaller budget than Gemini's retry_attempts/retry_base_delay:
    #   fallback providers exist, so hammering a saturated keyless endpoint (S2's public
    #   search pool 429s persistently under load -- confirmed live) buys nothing and
    #   isn't "mindful" use of a shared resource
    title_similarity_threshold: float = 0.90  # difflib ratio a title-search hit must clear
    title_similarity_relaxed: float = 0.75
    #   second acceptance tier (user: 0.90 alone is too strict for OCR-garbled titles):
    #   a hit in [relaxed, threshold) is accepted only with stronger corroboration --
    #   exact year match AND first-author surname match (diacritic-folded)
    year_tolerance: int = 1
    #   |extracted year - provider year| allowed (confirmed live: S2 reports the arXiv
    #   preprint year for brunton-2016, one year before the published version)
    api_cache_dir: str = "~/.cache/paper-refinery/api-cache"
    #   every successful provider response is cached here (key: sha256 of the URL) so
    #   iterating on matching logic never re-hits the keyless APIs; failures are never
    #   cached. Sibling of the models cache; safe to delete anytime.


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

    dir_path = dir_path or Path(os.environ.get("PAPER_REFINERY_SECRETS_DIR", DEFAULT_SECRETS_DIR))
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
