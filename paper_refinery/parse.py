"""Local GLM-OCR backend: llama-server (inference) + the official ``glmocr`` SDK
(PP-DocLayout-V3 layout detection + per-region OCR dispatch + result formatting).

GLM-OCR is a region-level recognizer, not a page-to-markdown model: it OCRs a crop given
a task prompt ("Text Recognition:", "Table Recognition:", "Formula Recognition:") and has
no layout awareness of its own. The ``glmocr`` SDK supplies the layout pass (PP-DocLayout-V3),
crops each region, dispatches per-region OCR calls to our own ``llama-server`` (an
OpenAI-compatible ``/v1/chat/completions`` server, same protocol the SDK expects from
vLLM/SGLang/Ollama), and hands back per-page regions plus already-cropped images for
figure/chart regions.

We rebuild markdown from ``result.json_result`` ourselves (rather than using glmocr's own
``markdown_result``) so we can: drop boilerplate regions outright, route reference content
out of the body into its own raw markdown, convert HTML tables to markdown, and turn
figure/chart regions into placeholders + saved crop files (never interpreted here -- see
``figures.py`` for that, now working from crops instead of full-page renders).

This module produces two markdown outputs (``ParseResult.markdown`` and
``.references_markdown``) and nothing more: GLM-OCR's own native output format is
markdown (glmocr's own ``markdown_result`` field, though we don't use it directly --
see above), and that applies identically to bibliography text, which gets the exact
same generic "Text Recognition:" OCR treatment as any other paragraph. Structuring the
bibliography into anything beyond plain OCR'd text -- authors/title/venue/DOI, in-text
citation-marker linking -- is deliberately a separate, later concern, not this module's
job; the one thing this module *does* do for references is pair each entry with its
detected number (the same region-adjacency technique ``_merge_formula_numbers`` already
uses for formula/formula_number), since that's reassembling what the layout model
already segmented, not interpreting it.

Page boundaries are our own ``<page_number>N</page_number>`` markers (``markers.py``),
independent of any OCR-detected page-number region (which is discarded as boilerplate).
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup

from .config import ParseConfig, load_config
from .markers import page_marker

# PP-DocLayout-V3 label taxonomy (glmocr's config.yaml `label_task_mapping` / `id2label`).
_ABANDON_LABELS = {
    "header",
    "footer",
    "number",  # printed page number -- we inject our own <page_number> markers instead
    "footnote",
    "aside_text",
    "footer_image",
    "header_image",
}
_DOC_TITLE_LABEL = "doc_title"
_PARAGRAPH_TITLE_LABEL = "paragraph_title"
_FIGURE_TITLE_LABEL = "figure_title"  # a "FIGURE N. ..." caption line -- plain body text,
#   named explicitly rather than relying on the unknown-label fallback, since enrich.py's
#   caption regex depends on this text actually landing in the body markdown
_REFERENCE_NUMBER_LABEL = "reference"  # the bracket/number marker before a bibliography
#   entry -- its own region, sibling to "reference_content" (same split as
#   formula/formula_number); paired back together by _merge_reference_numbers below
#   rather than discarded, so the raw bibliography keeps its original numbering
_REFERENCE_LABEL = "reference_content"
_TABLE_LABEL = "table"
_ALGORITHM_LABEL = "algorithm"  # pseudocode block: fenced so markdown preserves its structure
_FORMULA_LABELS = {"display_formula", "inline_formula"}
_FORMULA_NUMBER_LABEL = "formula_number"
_FIGURE_LABELS = {"chart", "image"}  # glmocr's "skip" task: cropped, never OCR'd
_FIGURE_CLASS_IDS = (3, 14)  # "chart", "image" in glmocr's id2label (config.yaml); re-check on upgrade

_HEADING_PREFIX_RE = re.compile(r"^#+\s*")


def _strip_heading_prefix(content: str) -> str:
    """Strip a leading ``#``/``##`` (or ``-``/``* ``) the OCR itself may have emitted.

    Confirmed against a live glmocr response: title-region content sometimes already
    starts with its own markdown heading marker, which would otherwise double up with
    the one we prepend (producing ``"## ## Section Title"``).
    """
    content = content.strip()
    if content.startswith(("- ", "* ")):
        content = content[2:].lstrip()
    return _HEADING_PREFIX_RE.sub("", content)


@dataclass
class ParseResult:
    # markdown with one authoritative <page_number>N</page_number> per page boundary,
    # boilerplate/reference regions removed, tables as markdown, formulas as LaTeX
    markdown: str
    figure_crops: dict[int, list[Path]] = field(default_factory=dict)  # page -> crop paths
    # raw bibliography, routed out of `markdown` entirely -- structuring/linking these
    # is a separate, later concern (not this module's job)
    references: list[dict] = field(default_factory=list)  # [{"page", "number", "text"}, ...]
    references_markdown: str = ""  # _render_references_markdown(references); see parse_pdf


@contextmanager
def _llama_server(cfg: ParseConfig):
    """Spawn llama-server serving GLM-OCR, wait for it to become healthy, tear it down."""
    if not cfg.model_path:
        raise RuntimeError("ParseConfig.model_path is not set (GLM-OCR GGUF weights)")
    if not cfg.mmproj_path:
        raise RuntimeError("ParseConfig.mmproj_path is not set (GLM-OCR GGUF vision projector)")

    cmd = [
        cfg.llama_server_bin,
        "-m",
        cfg.model_path,
        "--mmproj",
        cfg.mmproj_path,
        "--host",
        cfg.host,
        "--port",
        str(cfg.port),
        "-ngl",
        str(cfg.n_gpu_layers),
        *cfg.extra_server_args,
    ]
    fd, log_path_str = tempfile.mkstemp(prefix="llama-server-", suffix=".log")
    log_path = Path(log_path_str)
    try:
        with open(fd, "w") as log_file:
            proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
            try:
                _wait_for_health(proc, cfg, log_path)
                yield proc
            finally:
                # sole place that owns process teardown, for every exit path (including
                # a health-check timeout/early-exit raised from _wait_for_health)
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)
    finally:
        log_path.unlink(missing_ok=True)


def _read_log(log_path: Path) -> str:
    return log_path.read_text(errors="replace")


def _wait_for_health(proc: subprocess.Popen, cfg: ParseConfig, log_path: Path) -> None:
    url = f"http://{cfg.host}:{cfg.port}/health"
    deadline = time.monotonic() + cfg.startup_timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"llama-server exited early (code {proc.returncode}); log:\n{_read_log(log_path)}"
            )
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(1.0)
    raise RuntimeError(
        f"llama-server did not become healthy within {cfg.startup_timeout_s}s; log:\n"
        f"{_read_log(log_path)}"
    )


def _wrap_formula(content: str) -> str:
    text = (content or "").strip()
    for fence in ("$$", "\\[", "\\("):
        if text.startswith(fence):
            text = text[len(fence) :].strip()
    for fence in ("$$", "\\]", "\\)"):
        if text.endswith(fence):
            text = text[: -len(fence)].strip()
    return f"$$\n{text}\n$$"


def _merge_formula_number(formula_md: str, number: str) -> str:
    """Fold an adjacent formula_number region into the formula via ``\\tag{}``."""
    number = number.strip().strip("()")
    if formula_md.endswith("\n$$"):
        return formula_md[: -len("\n$$")] + f" \\tag{{{number}}}\n$$"
    return formula_md


def _int_attr(cell, name: str, default: int = 1) -> int:
    """A tag's ``rowspan``/``colspan`` attribute as an int, tolerating a malformed value.

    The source HTML is GLM-OCR's own model-generated output, not hand-authored markup --
    a non-numeric span value is a real (if rare) failure mode, not a "can't happen" input,
    and one bad cell shouldn't crash the whole page's table conversion.
    """
    try:
        return int(cell.get(name, default) or default)
    except (TypeError, ValueError):
        return default


def _html_table_to_markdown(html: str, strategy: str = "duplicate") -> str:
    """Convert a GLM-OCR HTML table to a markdown pipe table.

    Markdown has no merged-cell concept; a rowspan/colspan cell's value is duplicated
    into every grid position it visually spans (``strategy="duplicate"``) rather than
    silently dropped. This can repeat a spanning header across columns/rows it covers --
    an accepted, documented fidelity tradeoff, not a bug.
    """
    soup = BeautifulSoup(html or "", "html.parser")
    table = soup.find("table")
    if table is None:
        return (html or "").strip()

    grid: list[list[str]] = []
    active: dict[int, tuple[int, str]] = {}  # col -> (rows_remaining, text)

    for tr in table.find_all("tr"):
        row: list[str] = []
        col = 0
        placed_this_row: set[int] = set()
        for cell in tr.find_all(["td", "th"]):
            while active.get(col, (0, ""))[0] > 0:
                col += 1
            text = cell.get_text(" ", strip=True)
            rowspan = _int_attr(cell, "rowspan")
            colspan = _int_attr(cell, "colspan")
            for i in range(colspan):
                c = col + i
                while len(row) <= c:
                    row.append("")
                row[c] = text if strategy == "duplicate" or i == 0 else ""
                placed_this_row.add(c)
                if rowspan > 1:
                    active[c] = (rowspan - 1, text)
            col += colspan

        # fill in columns carried over by an earlier row's rowspan (not one that
        # originated in this row -- that was already placed above)
        max_col = max([len(row)] + [c + 1 for c in active])
        for c in range(max_col):
            if c in placed_this_row:
                continue
            remaining, text = active.get(c, (0, ""))
            if remaining > 0:
                while len(row) <= c:
                    row.append("")
                if not row[c] and strategy == "duplicate":
                    row[c] = text
                active[c] = (remaining - 1, text)

        grid.append(row)

    if not grid:
        return ""

    width = max(len(r) for r in grid)
    grid = [r + [""] * (width - len(r)) for r in grid]

    def esc(s: str) -> str:
        return s.replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(esc(c) for c in grid[0]) + " |"]
    lines.append("| " + " | ".join(["---"] * width) + " |")
    for r in grid[1:]:
        lines.append("| " + " | ".join(esc(c) for c in r) + " |")
    return "\n".join(lines)


def _dispatch_region(region: dict, cfg: ParseConfig) -> tuple[str, str]:
    """Classify one glmocr region and format its content.

    Returns ``(kind, text)``; kind is one of "abandon", "body", "reference",
    "reference_number", "figure", "formula", "formula_number". Prefers ``native_label``
    (PP-DocLayout-V3's fine-grained class, e.g. "paragraph_title"/"display_formula"/
    "reference_content") over ``label`` (glmocr's coarse text/table/formula/skip bucket,
    e.g. "text"/"formula"), which can't distinguish reference_content from body text, or
    chart from image. Confirmed against a live glmocr response: both keys are present on
    every region dict.
    """
    label = region.get("native_label") or region.get("label") or ""
    content = (region.get("content") or "").strip()

    if label in _ABANDON_LABELS:
        return "abandon", ""
    if label == _DOC_TITLE_LABEL:
        return "body", f"# {_strip_heading_prefix(content)}"
    if label == _PARAGRAPH_TITLE_LABEL:
        return "body", f"## {_strip_heading_prefix(content)}"
    if label == _FIGURE_TITLE_LABEL:
        return "body", content
    if label == _REFERENCE_NUMBER_LABEL:
        return "reference_number", content
    if label == _REFERENCE_LABEL:
        return "reference", content
    if label == _TABLE_LABEL:
        if cfg.table_format == "markdown":
            return "body", _html_table_to_markdown(content, cfg.merged_cell_strategy)
        return "body", content
    if label == _ALGORITHM_LABEL:
        return "body", f"```\n{content}\n```"
    if label in _FORMULA_LABELS:
        return "formula", _wrap_formula(content)
    if label == _FORMULA_NUMBER_LABEL:
        return "formula_number", content
    if label in _FIGURE_LABELS:
        return "figure", ""
    return "body", content  # unknown label: keep as text rather than silently drop it


def _merge_formula_numbers(
    triples: list[tuple[str, str, dict]],
) -> list[tuple[str, str, dict]]:
    """Fold adjacent formula/formula_number pairs (either order) into one "body" item.

    glmocr merges these upstream by default (``enable_merge_formula_numbers``), so under
    default settings a standalone ``formula_number`` sibling never reaches us here. This
    exists for the one case that isn't dead code: ``ParseConfig.glmocr_config_overrides``
    lets a caller turn that default off, at which point this is the only thing that still
    merges the number in. Idempotent either way: a formula whose content already contains
    ``\\tag{...}`` passes through unchanged.
    """
    merged: list[tuple[str, str, dict]] = []
    i = 0
    while i < len(triples):
        kind, text, region = triples[i]
        nxt = triples[i + 1] if i + 1 < len(triples) else None
        if kind == "formula" and nxt and nxt[0] == "formula_number":
            merged.append(("body", _merge_formula_number(text, nxt[1]), region))
            i += 2
        elif kind == "formula_number" and nxt and nxt[0] == "formula":
            merged.append(("body", _merge_formula_number(nxt[1], text), nxt[2]))
            i += 2
        elif kind == "formula":
            merged.append(("body", text, region))
            i += 1
        elif kind == "formula_number":
            merged.append(("body", f"({text.strip('() ')})" if text else "", region))
            i += 1
        else:
            merged.append((kind, text, region))
            i += 1
    return merged


def _merge_reference_numbers(
    triples: list[tuple[str, str, dict]],
) -> list[tuple[str, str, dict]]:
    """Fold an adjacent reference_number/reference pair (either order) into one
    "reference" triple, carrying the number in ``region["number"]``.

    PP-DocLayout-V3 labels a bibliography entry's leading "[12]"/"23." marker as its own
    region (native_label "reference"), separate from "reference_content" -- the same
    split _merge_formula_numbers already resolves for formula/formula_number. An
    unpaired reference_content still becomes an entry with no number attached (never
    dropped over a missing number -- the entry's text is what matters most); an unpaired
    reference_number is dropped (nothing to attach it to).
    """
    merged: list[tuple[str, str, dict]] = []
    i = 0
    while i < len(triples):
        kind, text, region = triples[i]
        nxt = triples[i + 1] if i + 1 < len(triples) else None
        if kind == "reference_number" and nxt and nxt[0] == "reference":
            number = text.strip().strip("[]().").strip()
            merged.append(("reference", nxt[1], {**nxt[2], "number": number}))
            i += 2
        elif kind == "reference" and nxt and nxt[0] == "reference_number":
            number = nxt[1].strip().strip("[]().").strip()
            merged.append(("reference", text, {**region, "number": number}))
            i += 2
        elif kind == "reference_number":
            i += 1  # unpaired number marker: nothing to attach it to
        else:
            merged.append((kind, text, region))
            i += 1
    return merged


def _save_figure_crop(
    region: dict,
    image_files: dict,
    used_images: set[str],
    figures_dir: Path,
    page: int,
    idx: int,
) -> Path | None:
    """Save the region's already-cropped image (from glmocr's ``image_files``) to disk.

    Confirmed against a live glmocr response: a chart/image region carries
    ``image_path: "imgs/{filename}"`` where ``filename`` is exactly a key in
    ``image_files`` (e.g. ``"cropped_page0_idx0.jpg"``, 0-indexed by input-list position).
    The page-number-substring fallback below is defense-in-depth only, for a glmocr
    version where that key is absent.
    """
    image_path = region.get("image_path")
    filename = Path(image_path).name if image_path else None
    img = image_files.get(filename) if filename else None

    if img is None:
        for candidate_name, candidate_img in image_files.items():
            if candidate_name in used_images:
                continue
            if f"page{page - 1}_" in candidate_name or f"page{page}_" in candidate_name:
                img, filename = candidate_img, candidate_name
                break

    if img is None or filename is None:
        warnings.warn(f"no cropped image found for a figure/chart region on page {page}")
        return None

    used_images.add(filename)
    crop_path = figures_dir / f"page_{page}_fig_{idx}.png"
    img.save(crop_path)
    return crop_path


def _build_markdown(
    pages_regions: list[list[dict]],
    image_files: dict,
    figures_dir: Path,
    cfg: ParseConfig,
) -> tuple[str, dict[int, list[Path]], list[dict]]:
    """Assemble page-marked markdown, figure crops, and a raw references list from
    glmocr's per-page region lists."""
    parts: list[str] = []
    figure_crops: dict[int, list[Path]] = {}
    references: list[dict] = []
    used_images: set[str] = set()

    for page_idx, regions in enumerate(pages_regions):
        page = page_idx + 1  # glmocr's page_idx is 0-based by input order; confirmed live
        sorted_regions = sorted(regions, key=lambda r: r.get("index", 0))
        triples = [(*_dispatch_region(r, cfg), r) for r in sorted_regions]
        triples = [t for t in triples if t[0] != "abandon"]
        triples = _merge_formula_numbers(triples)
        triples = _merge_reference_numbers(triples)

        body_parts: list[str] = []
        for kind, text, region in triples:
            if kind == "body":
                if text.strip():
                    body_parts.append(text)
            elif kind == "reference":
                if text.strip():
                    references.append({"page": page, "number": region.get("number"), "text": text})
            elif kind == "figure":
                idx = len(figure_crops.get(page, []))
                crop_path = _save_figure_crop(region, image_files, used_images, figures_dir, page, idx)
                if crop_path is None:
                    continue
                figure_crops.setdefault(page, []).append(crop_path)
                body_parts.append(f"![FIGURE_CROP {page}:{idx}]({crop_path})")

        parts.append(page_marker(page))
        parts.append("\n\n".join(body_parts))

    return "\n\n".join(parts), figure_crops, references


def _render_references_markdown(references: list[dict]) -> str:
    """Plain markdown rendering of the raw bibliography: one entry per line, grouped
    under each page's own ``<page_number>`` marker, in reading order.

    Zero interpretation -- this is GLM-OCR's own OCR'd text, reassembled only using the
    number/content region pairing already done in ``_merge_reference_numbers``. No
    schema, no external tool: structuring the bibliography into anything more is a
    separate, later concern, kept out of the OCR/parse layer entirely.
    """
    if not references:
        return ""
    by_page: dict[int, list[dict]] = {}
    for ref in references:
        by_page.setdefault(ref["page"], []).append(ref)

    parts: list[str] = []
    for page in sorted(by_page):
        lines = [
            f"[{ref['number']}] {ref['text']}" if ref["number"] else ref["text"]
            for ref in by_page[page]
        ]
        parts.append(page_marker(page))
        parts.append("\n\n".join(lines))
    return "\n\n".join(parts)


def _dotted_overrides(cfg: ParseConfig) -> dict:
    """Build glmocr's ``_dotted`` config-override dict for this run.

    Widens the layout-detection box only for figure/chart classes (``cfg.figure_crop_margin``,
    per-class via ``pipeline.layout.layout_unclip_ratio``) before layering on
    ``cfg.glmocr_config_overrides`` -- an explicit user override always wins over our
    computed default for the same dotted path.
    """
    margin = (cfg.figure_crop_margin, cfg.figure_crop_margin)
    dotted = {
        "pipeline.layout.layout_unclip_ratio": {cid: margin for cid in _FIGURE_CLASS_IDS},
    }
    dotted.update(cfg.glmocr_config_overrides)
    return dotted


def parse_pdf(pdf_path: Path, image_dir: Path, cfg: ParseConfig | None = None) -> ParseResult:
    """Parse a PDF into page-marked markdown (with figure placeholders + LaTeX + markdown
    tables) via a local llama-server serving GLM-OCR, orchestrated by the glmocr SDK."""
    from glmocr import GlmOcr

    cfg = cfg or load_config().parse
    pdf_path, image_dir = Path(pdf_path), Path(image_dir)
    figures_dir = image_dir / cfg.figures_dir_name
    figures_dir.mkdir(parents=True, exist_ok=True)

    with _llama_server(cfg):
        with GlmOcr(
            mode="selfhosted",
            ocr_api_host=cfg.host,
            ocr_api_port=cfg.port,
            layout_device=cfg.layout_device,
            _dotted=_dotted_overrides(cfg),
        ) as parser:
            result = parser.parse(str(pdf_path))

    json_result = result.json_result
    if isinstance(json_result, str):
        json_result = json.loads(json_result)

    markdown, figure_crops, references = _build_markdown(
        json_result, result.image_files, figures_dir, cfg
    )
    return ParseResult(
        markdown=markdown,
        figure_crops=figure_crops,
        references=references,
        references_markdown=_render_references_markdown(references),
    )
