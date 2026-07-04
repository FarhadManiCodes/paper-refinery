# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A standalone preprocessor for **papis-ask**: it turns a PDF into section-aware, overlapping
text chunks (with figure descriptions and page numbers) that papis-ask ingests via
paper-qa's `Docs.aadd_texts` — replacing pypdf's blind char-window chunking. The end product
is `<pdf>.chunks.json`; secondary artifacts are `<pdf>.refinery.md` (reviewable enriched
markdown) and `<pdf>.references.md` (raw bibliography).

## Commands

Tooling lives in `.venv/` (no `uv.lock` — this is a plain venv). Prefix commands with the
venv or activate it first.

```bash
.venv/bin/pytest                              # full test suite
.venv/bin/pytest tests/test_parse.py          # one file
.venv/bin/pytest tests/test_parse.py::test_sort_references_by_number_fixes_scrambled_order  # one test
.venv/bin/ruff check .                         # lint (line-length 100)
.venv/bin/ruff format .                        # format
.venv/bin/refinery path/to/paper.pdf           # run the full pipeline
```

The test suite is fully offline: Gemini and the OCR backend are always mocked/injected
(`describe`/`client` params, hand-built region fixtures). No test needs network, a GPU, or
`llama-server`.

## Running the pipeline live requires a local OCR backend

Parsing is **local**, not cloud. `parse_pdf` spawns and tears down a `llama-server`
(llama.cpp) serving GLM-OCR GGUF weights for the duration of each run, driven by the
`glmocr` SDK. To run `refinery` end-to-end you need: `llama-server` on PATH, the two GGUF
files (model + vision projector), and their paths set in `~/.config/paper-refinery/config.toml`
under `[parse]` (`model_path`, `mmproj_path`). See README.md for full backend setup. `glmocr`
also downloads PP-DocLayout-V3 from HF Hub on first use (needs `HF_TOKEN` and one-time
network egress). **Always test pipeline changes live against real PDFs** — unit tests alone
have missed real OCR-ordering bugs.

## Architecture

Pipeline stages, each a single-purpose module; `cli.py::_refine` owns all wiring, the
modules own logic (never call each other except `enrich`→`parse` for its result type
and `parse`→`references` for bibliography repair):

```
backend.py     llama-server + GlmOcr lifecycle; ocr_backend() is reusable across PDFs
parse.py       PDF -> ParseResult (body markdown + figure crops + raw references)
references.py  bibliography repair, used only by parse.py: reclaim mislabeled entries,
               merge page-break splits, drop copyright tails, recover layout-skipped
               entries from the PDF text layer, contiguity-guarded number sort --
               every rule exists because a live paper broke the raw list
figures.py     describe ONE figure per Gemini call (all its panel crops together):
               embedded figure-type taxonomy (classify + type-specific attention in the
               same pass), context used dictionary-only, schema output, disk cache keyed
               on crop bytes + prompt (~/.cache/paper-refinery/figure-cache)
enrich.py      pair crops to captions geometrically (bboxes from parse; positional
               fallback), rename crops to fig_N.png, assemble per-figure context
               (title/abstract/neighbor paragraphs), splice descriptions after captions
chunker.py     section-aware split + guaranteed soft-overlap + page ranges -> list[Chunk]
cli.py         orchestrate the above; write .chunks.json / .refinery.md / .references.md
```

`citation_extraction.py` (layer 1: one Gemini call → rough structured reference fields) is
built but **not yet wired into `cli.py`**. A layer-2/3 `citation_resolution.py` (API
verification against Semantic Scholar / CrossRef / OpenAlex) is planned but unimplemented —
see the plan file referenced in memory.

### Key cross-cutting contracts

- **Page markers** (`markers.py`): `parse.py` emits exactly one `<page_number>N</page_number>`
  per page boundary; `enrich.py` and `chunker.py` read them; `chunker.py` strips them from
  final chunk text after resolving each chunk's page range. This is the single source of
  page numbers — OCR-detected page-number regions are discarded as boilerplate. Change the
  format only in `markers.py`.
- **GLM-OCR is region-level, not page-to-markdown.** `parse.py` rebuilds markdown itself from
  `result.json_result` (not glmocr's `markdown_result`) so it can drop boilerplate, route
  references out of the body, convert HTML tables → markdown, and turn figure/chart regions
  into placeholders + saved crops. Regions are sorted by PP-DocLayout-V3's `index` field,
  which is **not always true reading order** — this is the known root cause of body/figure
  ordering quirks. Numbered bibliographies are re-sorted by their printed number
  (`_sort_references_by_number`) precisely because `index` can't be trusted there.
- **Figure anchoring is caption-based, not placeholder-based.** Placeholders shift between
  parser runs; the `FIGURE N.M` caption is the stable anchor. When the layout model detected
  caption regions (`ParseResult.figure_captions`), enrich accepts only body lines matching
  them — the caption regex alone is the fallback. Crops pair with captions *geometrically*
  (nearest caption with x-overlap, via the bboxes on `CropRegion`/`CaptionRegion`; positional
  pairing only when geometry is missing): a multi-panel figure's crops all land on one
  caption (one Gemini call, names `fig_4.1_1.png`…), and a crop overlapping no caption (a
  banner) is never renamed or described. Renaming happens before any Gemini call, so
  cropping/pairing can be sanity-checked by filename alone.
- **Image links are relativized last.** `parse.py` emits absolute crop paths; `cli._refine`
  rewrites them relative to the markdown's own directory (`_relativize_image_links`) at the
  very end, since only `_refine` knows both the crop dir and the final `.md` path. Scratchpad
  scripts that call `parse_pdf` directly bypass this and will show absolute paths — expected.

### Config & secrets

- `config.py`: dataclass-per-stage (`ParseConfig`, `FigureConfig`, `ChunkConfig`,
  `CitationConfig`) bundled in `RefineryConfig`. `load_config()` overlays
  `~/.config/paper-refinery/config.toml` — each TOML table maps to a sub-config by name, each
  key must match a dataclass field (unknown section/key raises). Every field has a code
  default; the TOML file is optional and only sets machine-specific values (model paths,
  `n_gpu_layers`, `glmocr_config_overrides`).
- Secrets load from `~/.config/paper-refinery/secrets/` — **one `*.env` file per service**
  (`google.env`, `hf.env`), loaded via python-dotenv inside `load_config()` early (before
  `parse_pdf`, since `HF_TOKEN` must be set for the HF Hub check). This directory is
  deliberately project-owned and never shared with other tools' secrets files — sourcing an
  unrelated tool's env (e.g. papis-ask's `OPENAI_BASE_URL`) would silently redirect glmocr's
  OpenAI-compatible client away from our own `llama-server`. paper-refinery only ever *reads*
  these files; never create, write, or read their contents.
