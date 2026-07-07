"""Parse-stage checkpoint: persist a ``ParseResult`` (plus its raw figure crops) so a
re-run skips the expensive OCR pass -- by far the slowest stage (~10 min; model load is
the dominant fixed cost), and the only one with no cache until now (figures and citations
already cache their Gemini/HTTP calls).

The checkpoint lives inside the work directory::

    <pdf>.refinery/
      parse_cache/
        parse.json     manifest (pdf sha256 + parse-config signature + version) + result
        crops/         pristine copy of the RAW page_*_fig_* crops
      figures/         the WORKING crops enrich mutates (fig_* after a run)

Why the crops are snapshotted rather than just referenced: ``enrich._rename_crops`` moves
the raw ``page_N_fig_i.png`` crops in place (``Path.replace``) to ``fig_*.png``, so after
one full run the raw crops are gone. The checkpoint keeps an immutable copy; on a hit it
restores them into ``figures/`` so enrich re-runs exactly as on a cold run. Reuse parse
<=> reuse crops, atomically.

Invalidation is by manifest: the PDF's content hash and the subset of ``ParseConfig`` that
changes OCR *output* (model paths, crop margin, glmocr overrides, server args) -- plus a
``CHECKPOINT_VERSION`` bumped when parse logic itself changes materially. Perf/transport
fields (ports, gpu layers, timeouts, device) deliberately do not invalidate.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from .backend import OcrBackend
from .config import ParseConfig
from .disk_cache import write_json
from .parse import (
    CaptionRegion,
    CropRegion,
    ParseResult,
    clear_stale_crops,
    parse_pdf,
)

CHECKPOINT_VERSION = 1  # bump when parse logic changes in a way that alters output
_CHECKPOINT_DIRNAME = "parse_cache"
_CHECKPOINT_FILE = "parse.json"
_CROPS_SUBDIR = "crops"


def _pdf_sha256(pdf: Path) -> str:
    h = hashlib.sha256()
    with pdf.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _parse_signature(cfg: ParseConfig) -> dict:
    """The ParseConfig subset that changes OCR OUTPUT (not just performance/transport)."""
    return {
        "model_path": cfg.model_path,
        "mmproj_path": cfg.mmproj_path,
        "figure_crop_margin": cfg.figure_crop_margin,
        "glmocr_config_overrides": cfg.glmocr_config_overrides,
        "extra_server_args": list(cfg.extra_server_args),
    }


def _manifest(pdf: Path, cfg: ParseConfig) -> dict:
    return {
        "version": CHECKPOINT_VERSION,
        "pdf_sha256": _pdf_sha256(pdf),
        "parse_signature": _parse_signature(cfg),
    }


def _bbox_to_list(bbox: tuple | None) -> list | None:
    return list(bbox) if bbox is not None else None


def _bbox_to_tuple(bbox: list | None) -> tuple | None:
    return tuple(bbox) if bbox is not None else None


def _serialize(result: ParseResult) -> dict:
    # `orig` is the crop's path exactly as embedded in `markdown` (parse_pdf writes both
    # from the same Path), so `_deserialize` can rewrite the markdown link to wherever the
    # crop is restored -- keeping markdown and figure_crops consistent even if the work
    # directory moved between runs.
    return {
        "markdown": result.markdown,
        "references": result.references,
        "references_markdown": result.references_markdown,
        "figure_captions": {
            str(page): [{"text": c.text, "bbox": _bbox_to_list(c.bbox)} for c in caps]
            for page, caps in result.figure_captions.items()
        },
        "figure_crops": {
            str(page): [
                {"orig": str(cr.path), "name": cr.path.name, "bbox": _bbox_to_list(cr.bbox)}
                for cr in crops
            ]
            for page, crops in result.figure_crops.items()
        },
    }


def _deserialize(data: dict, figures_dir: Path) -> ParseResult:
    markdown = data["markdown"]
    figure_crops: dict[int, list[CropRegion]] = {}
    for page_s, crops in data["figure_crops"].items():
        restored: list[CropRegion] = []
        for cr in crops:
            new_path = figures_dir / cr["name"]
            markdown = markdown.replace(cr["orig"], str(new_path))
            restored.append(CropRegion(new_path, _bbox_to_tuple(cr["bbox"])))
        figure_crops[int(page_s)] = restored
    figure_captions = {
        int(page_s): [CaptionRegion(c["text"], _bbox_to_tuple(c["bbox"])) for c in caps]
        for page_s, caps in data["figure_captions"].items()
    }
    return ParseResult(
        markdown=markdown,
        figure_crops=figure_crops,
        figure_captions=figure_captions,
        references=data["references"],
        references_markdown=data["references_markdown"],
    )


def save_checkpoint(work_dir: Path, pdf: Path, cfg: ParseConfig, result: ParseResult) -> None:
    """Snapshot a fresh parse result: write parse.json and copy the RAW crops (as produced
    by parse_pdf, before enrich renames them) into the checkpoint. Call immediately after
    parse_pdf, before enrich."""
    ckpt = work_dir / _CHECKPOINT_DIRNAME
    crops_dir = ckpt / _CROPS_SUBDIR
    crops_dir.mkdir(parents=True, exist_ok=True)
    for stale in crops_dir.glob("*"):  # a re-parse may produce different crops
        stale.unlink()
    for crops in result.figure_crops.values():
        for cr in crops:
            if cr.path.exists():
                shutil.copy2(cr.path, crops_dir / cr.path.name)
    payload = {"manifest": _manifest(pdf, cfg), "result": _serialize(result)}
    write_json(ckpt / _CHECKPOINT_FILE, payload)  # atomic (temp + os.replace), written last


def load_checkpoint(work_dir: Path, pdf: Path, cfg: ParseConfig) -> ParseResult | None:
    """The cached ParseResult if a valid checkpoint matches the current pdf+config, with its
    raw crops restored into figures/ (replacing any enrich-renamed leftovers). None on a
    miss (absent / manifest mismatch / corrupt)."""
    ckpt = work_dir / _CHECKPOINT_DIRNAME
    try:
        payload = json.loads((ckpt / _CHECKPOINT_FILE).read_text())
    except (OSError, ValueError):
        return None
    if payload.get("manifest") != _manifest(pdf, cfg):
        return None
    figures_dir = work_dir / cfg.figures_dir_name
    figures_dir.mkdir(parents=True, exist_ok=True)
    clear_stale_crops(figures_dir)
    for crop_file in (ckpt / _CROPS_SUBDIR).glob("*"):
        shutil.copy2(crop_file, figures_dir / crop_file.name)
    return _deserialize(payload["result"], figures_dir)


def parse_pdf_cached(
    pdf_path: Path,
    work_dir: Path,
    cfg: ParseConfig,
    backend: OcrBackend | None = None,
    force: bool = False,
) -> ParseResult:
    """``parse_pdf`` with a persistent checkpoint. On a hit (matching pdf+config and not
    ``force``) reuse the saved result and restore its raw crops WITHOUT running OCR; on a
    miss run parse_pdf and save a fresh checkpoint. Either way ``figures/`` ends up holding
    the raw crops, ready for enrich."""
    pdf_path, work_dir = Path(pdf_path), Path(work_dir)
    if not force:
        cached = load_checkpoint(work_dir, pdf_path, cfg)
        if cached is not None:
            return cached
    result = parse_pdf(pdf_path, work_dir, cfg, backend)
    save_checkpoint(work_dir, pdf_path, cfg, result)
    return result
