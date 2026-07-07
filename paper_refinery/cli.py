"""refinery CLI -- parse -> {figures+enrich || citations} -> chunk -> write manifests.

`refinery paper.pdf` produces two finals next to the PDF -- `paper.chunks.json` (the
hand-off papis-ask ingests via paper-qa's ``Docs.aadd_texts``) and
`paper.citations.json` (the verified/enriched bibliography + in-text linking) -- and
one work directory `paper.refinery/` holding everything reviewable or intermediate:
`refinery.md`, `references.md`, `resolution_report.txt`, `figures/`. The work
directory is self-contained (markdown image links are relative to it) and per-paper,
so two PDFs in one folder no longer share -- and overwrite -- a common `figures/`.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import click

from .chunker import Chunk, chunk_markdown
from .citation_extraction import extract_references
from .citation_linking import link_citations, make_citekey, rewrite_markers
from .citation_resolution import format_resolution_report, resolve_references
from .config import RefineryConfig, load_config
from .enrich import enrich_markdown
from .figures import describe_figure, make_client
from .parse import ParseResult
from .parse_cache import parse_pdf_cached

logger = logging.getLogger(__name__)

_IMAGE_LINK_RE = re.compile(r"(!\[[^\]]*\]\()([^)\s]+)(\))")


def _relativize_image_links(md: str, base: Path) -> str:
    """Rewrite absolute image-link paths in markdown to be relative to ``base``.

    The .md file and its figure crops are meant to travel together (as siblings under
    the work directory) -- an absolute path breaks the moment either is moved, renamed,
    or shared with someone else.
    """

    def _rel(m: re.Match) -> str:
        path = Path(m.group(2))
        if not path.is_absolute():
            return m.group(0)
        return f"{m.group(1)}{os.path.relpath(path, start=base)}{m.group(3)}"

    return _IMAGE_LINK_RE.sub(_rel, md)


def _default_outputs(
    pdf: Path, out: Path | None, citations_out: Path | None, work_dir: Path | None
) -> tuple[Path, Path, Path]:
    """Fill in the default artifact locations (next to the PDF) for any left as None."""
    return (
        out or pdf.with_suffix(".chunks.json"),
        citations_out or pdf.with_suffix(".citations.json"),
        work_dir or pdf.with_suffix(".refinery"),
    )


def write_chunks(chunks: list[Chunk], docname: str, source_pdf: str, out_path: Path) -> None:
    """Serialize chunks to the hand-off JSON that papis-ask ingests via aadd_texts."""
    payload = {
        "source_pdf": source_pdf,
        "docname": docname,
        "parser": "paper-refinery",
        "chunks": [c.to_dict() for c in chunks],
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _run_citations(
    parsed: ParseResult, cfg: RefineryConfig, work_dir: Path
) -> tuple[list[dict], list[dict]]:
    """Citation stack on one paper's parse output: extraction (one Gemini call) ->
    resolution (verify/enrich against CrossRef/S2/OpenAlex, disk-cached).

    Needs only ``parsed`` -- nothing from the figure/enrich side -- which is what lets
    ``_refine`` run it concurrently with figure describing. The human-readable
    resolution diff report lands in the work directory. Returns ``(extracted,
    resolved)``, positionally aligned -- ``_refine`` links markers against
    ``extracted`` (below) and builds citekeys from ``resolved``.
    """
    extracted = extract_references([r["text"] for r in parsed.references], cfg.citation)
    resolved = resolve_references(extracted, parsed.references, cfg.citation)
    report = format_resolution_report(extracted, resolved)
    (work_dir / "resolution_report.txt").write_text(report + "\n")
    return extracted, resolved


def _rechunk(pdf: Path, out: Path, work_dir: Path, cfg: RefineryConfig) -> list[str]:
    """`--from chunk`: re-chunk the saved enriched refinery.md, skipping parse/enrich/
    citations entirely. For iterating on chunk policy without paying the (expensive)
    upstream stages -- refinery.md is the exact input the full run feeds to the chunker,
    so this reproduces the same chunks. The citations manifest is left untouched."""
    md_path = work_dir / "refinery.md"
    if not md_path.exists():
        raise click.ClickException(
            f"--from chunk needs {md_path} from a previous full run, but it's missing. "
            f"Run `refinery {pdf.name}` once first."
        )
    chunks = chunk_markdown(md_path.read_text(), cfg.chunk)
    write_chunks(chunks, pdf.stem, str(pdf), out)
    return [f"re-chunked {len(chunks)} chunks (from {md_path.name}) -> {out}"]


def _refine(
    pdf: Path,
    out: Path,
    citations_out: Path,
    work_dir: Path,
    cfg: RefineryConfig,
    force_parse: bool = False,
) -> tuple[list[Chunk], list[str]]:
    """Run the pipeline; returns the chunks plus human-readable summary lines to echo.

    The citation stack runs in a worker thread concurrently with figure describing --
    the two sides are independent (citations need the references + body markdown,
    figures need the crops) and both are network-bound. A citation-stage failure
    (missing GOOGLE_API_KEY, providers down) degrades to a loud warning: the chunks
    manifest is the primary product and must still be written.

    ``force_parse`` bypasses the parse checkpoint (re-runs OCR); otherwise a matching
    checkpoint is reused (see parse_cache.parse_pdf_cached).
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    parsed = parse_pdf_cached(pdf, work_dir, cfg.parse, force=force_parse)
    summary: list[str] = []

    if parsed.references_markdown:
        refs_path = work_dir / "references.md"
        refs_path.write_text(parsed.references_markdown)
        summary.append(f"references -> {refs_path}")

    with ThreadPoolExecutor(max_workers=1) as pool:
        citations_future = (
            pool.submit(_run_citations, parsed, cfg, work_dir) if parsed.references else None
        )

        # one Gemini client, reused across pages (the page calls run concurrently in
        # enrich). Only built when there's actually a figure to describe -- a paper
        # with no figures and no references must not require GOOGLE_API_KEY at all.
        describe = None
        if parsed.figure_crops:
            client = make_client(cfg.figure)
            describe = functools.partial(describe_figure, client=client)
        enriched = enrich_markdown(parsed, cfg.figure, describe=describe)
        enriched = _relativize_image_links(enriched, work_dir)

        if citations_future is not None:
            try:
                extracted, resolved = citations_future.result()
            except Exception as exc:
                logger.warning(
                    "citation stage failed (%s) -- %s not written", exc, citations_out.name
                )
            else:
                # Detection matches the PRINTED form (extracted), never the resolved
                # one: resolution legitimately moves years/authors off what the paper
                # actually prints (confirmed live on fmech, where linking against
                # resolved entries lost 7 of 36 author-year markers). Citekeys, by
                # contrast, are built only from verified (resolved) surnames/years --
                # an OCR-corrupted surname must not propagate into every citekey in
                # the body -- and rewrite_markers only touches a marker when EVERY
                # reference it points at has one; everything else stays as printed.
                # Linking runs on the fully enriched markdown (figure descriptions
                # already spliced in) so marker offsets land in the same text that
                # gets chunked; Gemini's dictionary-only figure descriptions never
                # interpret paper content, so they can't introduce citation-shaped text.
                link = link_citations(enriched, extracted)
                citekeys = [make_citekey(r) for r in resolved]
                enriched = rewrite_markers(enriched, link.markers, citekeys)
                payload = {
                    "source_pdf": str(pdf),
                    "docname": pdf.stem,
                    "references": resolved,
                    "linking": {
                        "style": link.style,
                        "markers": [{"text": m.text, "refs": m.ref_indices} for m in link.markers],
                        "uncited": link.uncited,
                        "ambiguous": link.ambiguous,
                    },
                }
                citations_out.write_text(json.dumps(payload, indent=1, ensure_ascii=False))
                verified = sum(1 for r in resolved if r.get("verified"))
                summary.append(f"citations: {verified}/{len(resolved)} verified -> {citations_out}")

        # keep the enriched markdown (citekeys rewritten, if the citation stage ran)
        # as a reviewable artifact, before chunking
        md_out = work_dir / "refinery.md"
        md_out.write_text(enriched)
        summary.append(f"enriched markdown -> {md_out}")

    chunks = chunk_markdown(enriched, cfg.chunk)
    write_chunks(chunks, pdf.stem, str(pdf), out)
    summary.append(f"wrote {len(chunks)} chunks -> {out}")
    return chunks, summary


@dataclass
class RefineResult:
    """What ``refine()`` produced: the chunks in memory plus the on-disk artifacts."""

    chunks: list[Chunk]
    chunks_path: Path  # <pdf>.chunks.json -- the papis-ask hand-off
    citations_path: Path  # <pdf>.citations.json -- absent if the paper had no references
    work_dir: Path  # <pdf>.refinery/ -- refinery.md, references.md, figures/, parse_cache/


def refine(
    pdf: Path,
    cfg: RefineryConfig | None = None,
    *,
    out: Path | None = None,
    citations_out: Path | None = None,
    work_dir: Path | None = None,
    force_parse: bool = False,
) -> RefineResult:
    """Run the full pipeline on one PDF and return its chunks + artifact paths.

    The single stable entry point for in-process callers (e.g. papis-ask): parse ->
    figure-enrich -> citation-verify -> chunk, reusing the parse checkpoint so a repeated
    call on an unchanged PDF skips OCR (pass ``force_parse=True`` to re-OCR). Writes
    ``<pdf>.chunks.json`` / ``.citations.json`` and the ``<pdf>.refinery/`` work directory,
    exactly like the ``refinery`` CLI. Output locations default next to the PDF; override
    any of them explicitly. ``cfg`` defaults to ``load_config()``.

    Returns refinery's own types/paths only -- no paper-qa objects cross this boundary; the
    consumer owns converting chunks into whatever its indexer wants (it may read
    ``chunks_path`` or use the returned ``chunks`` directly).
    """
    pdf = Path(pdf)
    cfg = cfg or load_config()
    out, citations_out, work_dir = _default_outputs(pdf, out, citations_out, work_dir)
    chunks, _summary = _refine(pdf, out, citations_out, work_dir, cfg, force_parse=force_parse)
    return RefineResult(
        chunks=chunks, chunks_path=out, citations_path=citations_out, work_dir=work_dir
    )


def _setup_logging() -> None:
    """Route paper-refinery's own logs to stderr for a CLI run.

    Configures only the package logger (not the root), so third-party INFO chatter from
    glmocr/urllib stays quiet, and guards against duplicate handlers when the CLI is
    invoked repeatedly in one process (tests). In-process callers of ``refine()`` set up
    their own logging; WARNINGs still reach them via logging's last-resort handler.
    """
    pkg_logger = logging.getLogger("paper_refinery")
    if not pkg_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        pkg_logger.addHandler(handler)
    pkg_logger.setLevel(logging.INFO)


@click.command()
@click.argument("pdf", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--out",
    type=click.Path(path_type=Path),
    default=None,
    help="Output chunks JSON (default: <pdf>.chunks.json next to the PDF).",
)
@click.option(
    "--citations-out",
    type=click.Path(path_type=Path),
    default=None,
    help="Output citations JSON (default: <pdf>.citations.json next to the PDF).",
)
@click.option(
    "--work-dir",
    type=click.Path(path_type=Path),
    default=None,
    help=(
        "Directory for reviewable/intermediate artifacts: refinery.md, references.md, "
        "resolution_report.txt, figures/ (default: <pdf-stem>.refinery/ next to the PDF)."
    ),
)
@click.option(
    "--model-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="GLM-OCR GGUF weights (default: 'parse.model_path' in config.toml).",
)
@click.option(
    "--mmproj-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="GLM-OCR GGUF vision projector (default: 'parse.mmproj_path' in config.toml).",
)
@click.option(
    "--force-parse",
    is_flag=True,
    default=False,
    help="Re-run OCR, bypassing the parse checkpoint (<pdf>.refinery/parse_cache/).",
)
@click.option(
    "--from",
    "from_stage",
    type=click.Choice(["chunk"]),
    default=None,
    help="Resume from a stage, reusing earlier artifacts. 'chunk' re-chunks the saved "
    "refinery.md only (instant; for chunk-policy tuning).",
)
def main(
    pdf: Path,
    out: Path | None,
    citations_out: Path | None,
    work_dir: Path | None,
    model_path: Path | None,
    mmproj_path: Path | None,
    force_parse: bool,
    from_stage: str | None,
) -> None:
    """Parse, figure-enrich, citation-verify, and chunk PDF for papis-ask."""
    _setup_logging()
    cfg = load_config()
    if model_path is not None:
        cfg.parse.model_path = str(model_path)
    if mmproj_path is not None:
        cfg.parse.mmproj_path = str(mmproj_path)
    out, citations_out, work_dir = _default_outputs(pdf, out, citations_out, work_dir)

    if from_stage == "chunk":
        summary = _rechunk(pdf, out, work_dir, cfg)
    else:
        _chunks, summary = _refine(pdf, out, citations_out, work_dir, cfg, force_parse=force_parse)
    click.echo("\n".join(summary))


if __name__ == "__main__":
    main()
