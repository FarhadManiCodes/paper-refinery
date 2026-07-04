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
import os
import re
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import click

from .chunker import Chunk, chunk_markdown
from .citation_extraction import extract_references
from .citation_linking import link_citations
from .citation_resolution import format_resolution_report, resolve_references
from .config import RefineryConfig, load_config
from .enrich import enrich_markdown
from .figures import describe_page_figures, make_client
from .parse import ParseResult, parse_pdf

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


def write_chunks(chunks: list[Chunk], docname: str, source_pdf: str, out_path: Path) -> None:
    """Serialize chunks to the hand-off JSON that papis-ask ingests via aadd_texts."""
    payload = {
        "source_pdf": source_pdf,
        "docname": docname,
        "parser": "paper-refinery",
        "chunks": [c.to_dict() for c in chunks],
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _run_citations(parsed: ParseResult, cfg: RefineryConfig, work_dir: Path) -> dict:
    """Citation stack on one paper's parse output: extraction (one Gemini call) ->
    resolution (verify/enrich against CrossRef/S2/OpenAlex, disk-cached) -> linking
    (deterministic in-text marker detection).

    Needs only ``parsed`` -- nothing from the figure/enrich side -- which is what lets
    ``_refine`` run it concurrently with figure describing. The human-readable
    resolution diff report lands in the work directory; the returned payload is the
    ``.citations.json`` body. Marker offsets are deliberately NOT serialized: they
    index the raw parse markdown, not the enriched ``refinery.md`` a reader would
    open, so only the marker text and its target reference indices are kept.

    Linking runs against the layer-1 EXTRACTED entries, not the resolved ones:
    detection must match what the paper *prints*, and resolution legitimately moves
    fields away from the printed form (a verified year corrected forward, a
    provider's author spelling) -- confirmed live on fmech, where linking against
    resolved entries lost 7 of 36 author-year markers. The marker indices apply to
    the resolved list identically (the two are positionally aligned); verified
    surnames/years still serve the future citekey rewrite, which reads ``resolved``.
    """
    extracted = extract_references([r["text"] for r in parsed.references], cfg.citation)
    resolved = resolve_references(extracted, parsed.references, cfg.citation)
    report = format_resolution_report(extracted, resolved)
    (work_dir / "resolution_report.txt").write_text(report + "\n")

    link = link_citations(parsed.markdown, extracted)
    return {
        "references": resolved,
        "linking": {
            "style": link.style,
            "markers": [{"text": m.text, "refs": m.ref_indices} for m in link.markers],
            "uncited": link.uncited,
            "ambiguous": link.ambiguous,
        },
    }


def _refine(
    pdf: Path, out: Path, citations_out: Path, work_dir: Path, cfg: RefineryConfig
) -> list[str]:
    """Run the pipeline; returns human-readable summary lines for the CLI to echo.

    The citation stack runs in a worker thread concurrently with figure describing --
    the two sides are independent (citations need the references + body markdown,
    figures need the crops) and both are network-bound. A citation-stage failure
    (missing GOOGLE_API_KEY, providers down) degrades to a loud warning: the chunks
    manifest is the primary product and must still be written.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    parsed = parse_pdf(pdf, work_dir, cfg.parse)
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
            describe = functools.partial(describe_page_figures, client=client)
        enriched = enrich_markdown(parsed, cfg.figure, describe=describe)
        enriched = _relativize_image_links(enriched, work_dir)

        # keep the enriched markdown as a reviewable artifact, before chunking
        md_out = work_dir / "refinery.md"
        md_out.write_text(enriched)
        summary.append(f"enriched markdown -> {md_out}")

        if citations_future is not None:
            try:
                payload = citations_future.result()
            except Exception as exc:
                warnings.warn(f"citation stage failed ({exc}) -- {citations_out.name} not written")
            else:
                payload = {"source_pdf": str(pdf), "docname": pdf.stem, **payload}
                citations_out.write_text(json.dumps(payload, indent=1, ensure_ascii=False))
                verified = sum(1 for r in payload["references"] if r.get("verified"))
                summary.append(
                    f"citations: {verified}/{len(payload['references'])} verified "
                    f"-> {citations_out}"
                )

    chunks = chunk_markdown(enriched, cfg.chunk)
    write_chunks(chunks, pdf.stem, str(pdf), out)
    summary.append(f"wrote {len(chunks)} chunks -> {out}")
    return summary


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
def main(
    pdf: Path,
    out: Path | None,
    citations_out: Path | None,
    work_dir: Path | None,
    model_path: Path | None,
    mmproj_path: Path | None,
) -> None:
    """Parse, figure-enrich, citation-verify, and chunk PDF for papis-ask."""
    cfg = load_config()
    if model_path is not None:
        cfg.parse.model_path = str(model_path)
    if mmproj_path is not None:
        cfg.parse.mmproj_path = str(mmproj_path)
    out = out or pdf.with_suffix(".chunks.json")
    citations_out = citations_out or pdf.with_suffix(".citations.json")
    work_dir = work_dir or pdf.with_suffix(".refinery")

    summary = _refine(pdf, out, citations_out, work_dir, cfg)
    click.echo("\n".join(summary))


if __name__ == "__main__":
    main()
