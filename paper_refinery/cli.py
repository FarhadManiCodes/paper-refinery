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

import contextlib
import functools
import json
import logging
import os
import queue
import re
import threading
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import click

from .backend import OcrBackend, ocr_backend
from .chunker import Chunk, chunk_markdown
from .citation_extraction import extract_references
from .citation_linking import link_citations, make_citekey, rewrite_markers
from .citation_resolution import SourcePaper, format_resolution_report, resolve_references
from .config import ParseConfig, RefineryConfig, load_config
from .enrich import enrich_markdown
from .figures import describe_figure, make_client
from .parse import ParseResult
from .parse_cache import load_checkpoint, parse_pdf_cached

logger = logging.getLogger(__name__)

_IMAGE_LINK_RE = re.compile(r"(!\[[^\]]*\]\()([^)\s]+)(\))")
_H1_RE = re.compile(r"(?m)^#\s+(.+)$")


def _source_title(markdown: str) -> str | None:
    """The paper's own title from the first H1 (the OCR'd doc-title) -- used to look the
    SOURCE paper up in S2 for the reference fast-path. None when there's no H1."""
    m = _H1_RE.search(markdown)
    return " ".join(m.group(1).split()) if m else None


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
    parsed: ParseResult, cfg: RefineryConfig, work_dir: Path, source_doi: str | None = None
) -> tuple[list[dict], list[dict]]:
    """Citation stack on one paper's parse output: extraction (one Gemini call) ->
    resolution (verify/enrich against CrossRef/S2/OpenAlex, disk-cached).

    Needs only ``parsed`` -- nothing from the figure/enrich side -- which is what lets
    ``_refine`` run it concurrently with figure describing. The human-readable
    resolution diff report lands in the work directory. Returns ``(extracted,
    resolved)``, positionally aligned -- ``_refine`` links markers against
    ``extracted`` (below) and builds citekeys from ``resolved``.

    The source paper (``source_doi`` if a caller has it, else the OCR'd title) drives the
    S2 bulk-references fast-path in ``resolve_references``; unidentified/unmatched entries
    fall back to the per-entry provider search.
    """
    extracted = extract_references([r["text"] for r in parsed.references], cfg.citation)
    source = SourcePaper(doi=source_doi, title=_source_title(parsed.markdown))
    resolved = resolve_references(extracted, parsed.references, cfg.citation, source=source)
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
    source_doi: str | None = None,
) -> tuple[list[Chunk], list[str]]:
    """Run the pipeline; returns the chunks plus human-readable summary lines to echo.

    ``force_parse`` bypasses the parse checkpoint (re-runs OCR); otherwise a matching
    checkpoint is reused (see parse_cache.parse_pdf_cached). The OCR pass is the one
    stage that can't overlap across papers (single GPU/llama-server); the rest lives in
    ``_refine_parsed``, which ``refine_many`` runs concurrently across papers while the
    next paper OCRs.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    parsed = parse_pdf_cached(pdf, work_dir, cfg.parse, force=force_parse)
    return _refine_parsed(parsed, pdf, out, citations_out, work_dir, cfg, source_doi)


def _refine_parsed(
    parsed: ParseResult,
    pdf: Path,
    out: Path,
    citations_out: Path,
    work_dir: Path,
    cfg: RefineryConfig,
    source_doi: str | None = None,
) -> tuple[list[Chunk], list[str]]:
    """The post-parse pipeline for one already-parsed paper: {figures+enrich ||
    citations} -> chunk -> write manifests. Returns the chunks plus summary lines.

    Split from ``_refine`` so it can run on a network-worker pool concurrently across
    papers (nothing here touches the OCR backend). The citation stack runs in a worker
    thread concurrently with figure describing -- the two sides are independent
    (citations need the references + body markdown, figures need the crops) and both are
    network-bound. A citation-stage failure (missing GOOGLE_API_KEY, providers down)
    degrades to a loud warning: the chunks manifest is the primary product and must
    still be written.
    """
    summary: list[str] = []

    if parsed.references_markdown:
        refs_path = work_dir / "references.md"
        refs_path.write_text(parsed.references_markdown)
        summary.append(f"references -> {refs_path}")

    with ThreadPoolExecutor(max_workers=1) as pool:
        citations_future = (
            pool.submit(_run_citations, parsed, cfg, work_dir, source_doi)
            if parsed.references
            else None
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


@dataclass(slots=True)
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
    doi: str | None = None,
) -> RefineResult:
    """Run the full pipeline on one PDF and return its chunks + artifact paths.

    The single stable entry point for in-process callers (e.g. papis-ask): parse ->
    figure-enrich -> citation-verify -> chunk, reusing the parse checkpoint so a repeated
    call on an unchanged PDF skips OCR (pass ``force_parse=True`` to re-OCR). Writes
    ``<pdf>.chunks.json`` / ``.citations.json`` and the ``<pdf>.refinery/`` work directory,
    exactly like the ``refinery`` CLI. Output locations default next to the PDF; override
    any of them explicitly. ``cfg`` defaults to ``load_config()``.

    Pass the source paper's ``doi`` (papis has it in info.yaml) to enable the S2
    bulk-references fast-path -- one call fetches the paper's whole reference list instead
    of a per-reference provider search. Without it, the OCR'd title is used opportunistically.

    Returns refinery's own types/paths only -- no paper-qa objects cross this boundary; the
    consumer owns converting chunks into whatever its indexer wants (it may read
    ``chunks_path`` or use the returned ``chunks`` directly).
    """
    pdf = Path(pdf)
    cfg = cfg or load_config()
    out, citations_out, work_dir = _default_outputs(pdf, out, citations_out, work_dir)
    chunks, _summary = _refine(
        pdf, out, citations_out, work_dir, cfg, force_parse=force_parse, source_doi=doi
    )
    return RefineResult(
        chunks=chunks, chunks_path=out, citations_path=citations_out, work_dir=work_dir
    )


def refine_many(
    pdfs: Sequence[Path],
    cfg: RefineryConfig | None = None,
    *,
    dois: Sequence[str | None] | None = None,
    force_parse: bool = False,
    workers: int = 4,
    ocr_workers: int = 2,
) -> Iterator[RefineResult]:
    """Refine many PDFs concurrently, yielding each ``RefineResult`` as it finishes.

    In the default ``maas`` (cloud) mode every paper runs its *whole* pipeline
    (parse -> {figures || citations} -> chunk) on a pool of ``workers``, and results stream
    out in completion order (NOT input order), the instant each paper is ready. A caller
    can index each paper as it lands rather than blocking on the batch
    (``list(refine_many(...))`` if you do want them all). A single PDF gains nothing from
    the pool -- use ``refine`` there.

    OCR is gated separately from the rest by ``ocr_workers`` (default 2): the cloud OCR
    endpoint rate-limits concurrent requests (z.ai returns 429 above ~2-3 at once, and the
    glmocr SDK reports an exhausted 429 as an *empty* parse -- guarded in ``parse_pdf``), so
    only ``ocr_workers`` papers may be OCR-ing at any moment. The other network stages
    (Gemini figure calls, citation providers) hit different hosts with their own limits, so
    up to ``workers`` papers run those concurrently -- a paper does its figures/citations
    while the OCR slot it freed is taken by the next. Keep ``ocr_workers`` at/below your
    z.ai tier's concurrency (see ``z.ai/manage-apikey/rate-limits``); raise ``workers`` to
    overlap more network tails.

    In ``selfhosted`` mode OCR is bound to one local llama-server (one GPU), so it can't
    parallelize; ``refine_many`` falls back to a serial-OCR path where OCR runs one paper
    at a time on a shared backend while each finished parse's network stages overlap the
    next paper's OCR (``ocr_workers`` is not used there -- see ``_stream_serial_ocr``).

    Outputs default next to each PDF, exactly like ``refine``. Pass ``dois`` (aligned to
    ``pdfs``) to feed each paper's DOI into the S2 bulk-references fast-path. A paper that
    fails (corrupt PDF, provider outage, an OCR call the cloud never fulfilled) is logged
    and skipped rather than failing the batch.
    """
    pdfs = [Path(p) for p in pdfs]
    if dois is not None and len(dois) != len(pdfs):
        raise ValueError(f"dois has {len(dois)} entries but there are {len(pdfs)} pdfs")
    cfg = cfg or load_config()
    doi_list: list[str | None] = list(dois) if dois is not None else [None] * len(pdfs)
    # a real function (not a bare generator) so the arg validation above raises eagerly,
    # at the call, rather than being deferred to the first ``next()``.
    if cfg.parse.mode == "maas":
        return _stream_parallel(pdfs, doi_list, cfg, force_parse, workers, ocr_workers)
    return _stream_serial_ocr(pdfs, doi_list, cfg, force_parse, workers)


def _refine_one(
    pdf: Path,
    doi: str | None,
    cfg: RefineryConfig,
    force_parse: bool,
    ocr_gate: threading.Semaphore,
) -> RefineResult:
    """Run the full pipeline on one PDF (default output locations) -> ``RefineResult``.

    The unit of work the ``maas`` pool runs concurrently. Only the parse (OCR) is held
    under ``ocr_gate`` -- its own cloud backend is spawned inside ``parse_pdf`` on a
    checkpoint miss -- so at most ``ocr_workers`` papers hit the rate-limited OCR endpoint
    at once; the network-bound {figures || citations} -> chunk tail runs outside the gate,
    freeing the OCR slot for the next paper immediately. Nothing is shared between papers.
    """
    out, citations_out, work_dir = _default_outputs(pdf, None, None, None)
    work_dir.mkdir(parents=True, exist_ok=True)
    logger.info("%s: refine start", pdf.name)
    with ocr_gate:
        parsed = parse_pdf_cached(pdf, work_dir, cfg.parse, force=force_parse)
    chunks, _summary = _refine_parsed(parsed, pdf, out, citations_out, work_dir, cfg, doi)
    logger.info("%s: done (%d chunks) -- streaming result", pdf.name, len(chunks))
    return RefineResult(chunks, out, citations_out, work_dir)


def _stream_parallel(
    pdfs: list[Path],
    dois: list[str | None],
    cfg: RefineryConfig,
    force_parse: bool,
    workers: int,
    ocr_workers: int,
) -> Iterator[RefineResult]:
    """maas engine: the full pipeline per paper on a flat pool, yielded as each completes.

    ``workers`` papers run end to end via ``_refine_one``, streamed via ``as_completed``;
    an ``ocr_workers``-permit semaphore caps how many are inside the rate-limited OCR call
    at once (the rest overlap on their network stages). A paper that raises anywhere in its
    pipeline -- including the empty-parse guard when the cloud never fulfilled its OCR -- is
    logged and skipped; the rest keep flowing.

    Not a ``with`` block on purpose: the plain context manager shuts the pool down with
    ``wait=True`` and no cancellation, so a caller that ``break``\\ s out of the stream (or
    a downstream error closing the generator) would kick off *every* still-queued paper
    just to throw the results away. The explicit ``cancel_futures=True`` teardown instead
    drops papers that haven't started; only the <=``workers`` already in flight finish
    (a running parse can't be interrupted), which we still wait for so no thread keeps
    writing to disk after the caller has moved on.
    """
    ocr_gate = threading.Semaphore(ocr_workers)
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="refine")
    try:
        futures = {
            pool.submit(_refine_one, pdf, doi, cfg, force_parse, ocr_gate): pdf
            for pdf, doi in zip(pdfs, dois, strict=True)
        }
        for fut in as_completed(futures):
            pdf = futures[fut]
            try:
                yield fut.result()
            except Exception as exc:
                logger.warning("refine failed for %s (%s) -- skipping", pdf.name, exc)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


# --- selfhosted (local single-GPU) serial-OCR path -------------------------------------
# Kept intact for ParseConfig.mode="selfhosted": one llama-server means OCR must run one
# paper at a time. The maas path above is the default; this is the fallback for .[local],
# and is what a cherry-pick back to a local-first setup would build on.


class _LazyBackend:
    """Spawns the shared OCR backend on first request and reuses it thereafter.

    A whole batch of already-parsed papers (every parse a checkpoint hit) never touches
    OCR, so the expensive server spawn + model load must not happen just because a batch
    was started -- only the first genuine cache miss pays for it. ``close`` tears down
    the server if it was ever started.
    """

    def __init__(self, cfg: ParseConfig) -> None:
        self._cfg = cfg
        self._stack = contextlib.ExitStack()
        self._backend: OcrBackend | None = None

    def get(self) -> OcrBackend:
        if self._backend is None:
            self._backend = self._stack.enter_context(ocr_backend(self._cfg))
        return self._backend

    def close(self) -> None:
        self._stack.close()


def _parse_for_batch(
    pdf: Path, work_dir: Path, cfg: RefineryConfig, backend: _LazyBackend, force_parse: bool
) -> ParseResult:
    """Parse one paper for the selfhosted batch, spawning the shared server only on a miss.

    Checks the checkpoint first (unless ``force_parse``) so a hit skips OCR entirely --
    the backend stays unspawned. Only a real miss calls ``backend.get()``, keeping OCR
    the single serial bottleneck without ever spinning up a server the batch doesn't need.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    if not force_parse:
        cached = load_checkpoint(work_dir, pdf, cfg.parse)
        if cached is not None:
            logger.info("%s: parse checkpoint hit (no OCR)", pdf.name)
            return cached
    logger.info("%s: OCR start (serial -- the batch's one shared-GPU stage)", pdf.name)
    return parse_pdf_cached(pdf, work_dir, cfg.parse, backend=backend.get(), force=force_parse)


def _stream_serial_ocr(
    pdfs: list[Path],
    dois: list[str | None],
    cfg: RefineryConfig,
    force_parse: bool,
    network_workers: int,
) -> Iterator[RefineResult]:
    """selfhosted engine: a serial-OCR producer thread feeding a network-worker pool,
    yielding results in completion order.

    OCR runs on one background thread (the serial GPU queue); each parse is submitted to
    ``net_pool`` the instant it completes, and a done-callback drops the finished
    ``RefineResult`` on a queue the consumer drains. So OCR(paper N+1) overlaps
    network(paper N), and the consumer sees each paper the moment its own pipeline ends.
    Every paper contributes exactly one queue item (a result, or ``None`` when it was
    skipped), so draining exactly ``len(pdfs)`` items terminates without a sentinel.
    """
    done: queue.Queue[RefineResult | None] = queue.Queue()
    backend = _LazyBackend(cfg.parse)

    def _on_network_done(
        pdf: Path, out: Path, citations_out: Path, work_dir: Path
    ) -> Callable[[Future], None]:
        def _cb(fut: Future) -> None:
            try:
                chunks, _summary = fut.result()
            except Exception as exc:
                logger.warning("refine failed for %s (%s) -- skipping", pdf.name, exc)
                done.put(None)
            else:
                logger.info(
                    "%s: network stages done (%d chunks) -- streaming result", pdf.name, len(chunks)
                )
                done.put(RefineResult(chunks, out, citations_out, work_dir))

        return _cb

    def _produce(net_pool: ThreadPoolExecutor) -> None:
        try:
            for n, (pdf, doi) in enumerate(zip(pdfs, dois, strict=True), start=1):
                out, citations_out, work_dir = _default_outputs(pdf, None, None, None)
                try:
                    parsed = _parse_for_batch(pdf, work_dir, cfg, backend, force_parse)
                except Exception as exc:
                    logger.warning("parse failed for %s (%s) -- skipping", pdf.name, exc)
                    done.put(None)
                    continue
                # OCR (the serial leg) for this paper is finished; its network stages now
                # run on the pool WHILE this thread moves straight on to the next paper's
                # OCR. Seeing an earlier paper's "network stages done" line appear between
                # here and the next paper's completion is the overlap, made visible.
                logger.info(
                    "%s: parse done (%d/%d) -> handed to network pool; OCR queue free for "
                    "the next paper",
                    pdf.name,
                    n,
                    len(pdfs),
                )
                fut = net_pool.submit(
                    _refine_parsed, parsed, pdf, out, citations_out, work_dir, cfg, doi
                )
                fut.add_done_callback(_on_network_done(pdf, out, citations_out, work_dir))
        finally:
            # OCR is fully issued; free the GPU while the network jobs still drain.
            backend.close()

    with ThreadPoolExecutor(max_workers=network_workers, thread_name_prefix="refine-net") as pool:
        producer = threading.Thread(target=_produce, args=(pool,), name="refine-ocr")
        producer.start()
        try:
            for _ in range(len(pdfs)):
                item = done.get()
                if item is not None:
                    yield item
        finally:
            producer.join()


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
    "--doi",
    default=None,
    help="Source paper DOI -- enables the S2 bulk-references fast-path (one call instead "
    "of a per-reference search). Without it the OCR'd title is tried opportunistically.",
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
    doi: str | None,
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
        _chunks, summary = _refine(
            pdf, out, citations_out, work_dir, cfg, force_parse=force_parse, source_doi=doi
        )
    click.echo("\n".join(summary))


if __name__ == "__main__":
    main()
