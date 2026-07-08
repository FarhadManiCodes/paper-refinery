# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A standalone preprocessor for **papis-ask**: it turns a PDF into section-aware, overlapping
text chunks (with figure descriptions, page numbers, and standardized `[surname_year]`
in-text citekeys) that papis-ask ingests via paper-qa's `Docs.aadd_texts` — replacing
pypdf's blind char-window chunking. Two finals land next to the PDF: `<pdf>.chunks.json`
(the hand-off) and `<pdf>.citations.json` (verified/enriched bibliography + in-text
linking map). Everything reviewable or intermediate goes in `<pdf>.refinery/`:
`refinery.md` (enriched markdown, citekeys already rewritten — the last human-readable
form before chunking), `references.md` (raw bibliography), `resolution_report.txt`
(per-reference verification diff), `figures/`.

## Commands

Tooling lives in `.venv/` (no `uv.lock` — this is a plain venv, Python 3.14). The default
install is cloud-only and torch-free: `uv pip install -e ".[dev]"`. Only the optional LOCAL
OCR backend (`.[local]` extra) pulls torch, and only there does the CPU-only-torch-first
two-step matter — see README.md's "Development setup" / "Local (selfhosted) OCR backend".

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

## Running the pipeline live: two OCR backends (`ParseConfig.mode`)

**`mode="maas"` (default): cloud.** Layout+OCR run on Zhipu's GLM-OCR API
(`api.z.ai/api/paas/v4/layout_parsing`, model `glm-ocr`) via the glmocr SDK's maas client —
no GPU, no GGUF, no llama-server. Needs `ZHIPU_API_KEY` (secrets `zai.env`). The SDK's maas
`json_result` DROPS `native_label`; `parse._normalize_maas_regions` restores it from the raw
`result._maas_response` and strips the cloud's `<div>` content wrappers (see
[[feature_maas_cloud_ocr]]).

**`mode="selfhosted"`: local.** `parse_pdf` spawns/tears down a `llama-server` (llama.cpp)
serving GLM-OCR GGUF weights, driven by the glmocr SDK. Needs the `.[local]` install extra
(torch + PP-DocLayout-V3, downloaded from HF Hub on first use with `HF_TOKEN`), `llama-server`
on PATH, the two GGUF files, and their paths in `config.toml` `[parse]`
(`model_path`/`mmproj_path`). `OcrBackend.server` is None in maas mode; the watchdog guards
the kill.

**Always test pipeline changes live against real PDFs** — unit tests alone have missed real
OCR-ordering bugs. See [[feedback_test_live_ocr]].

## Architecture

Pipeline stages, each a single-purpose module; `cli.py` owns all wiring (`_refine` parses,
then `_refine_parsed` runs the post-parse stages), the modules own logic (never call each
other except `enrich`→`parse` for its result type, `parse`→`references` for bibliography
repair, `citation_resolution`→`citation_providers` for the actual provider HTTP calls, and
`parse_cache`→`parse`/`backend` for checkpointed parsing):

```
backend.py               llama-server + GlmOcr lifecycle; ocr_backend() is reusable across PDFs
parse.py                 PDF -> ParseResult (body markdown + figure crops + raw references)
references.py            bibliography repair, used only by parse.py: reclaim mislabeled
                          entries, merge page-break splits, drop copyright tails, recover
                          layout-skipped entries from the PDF text layer, contiguity-guarded
                          number sort -- every rule exists because a live paper broke the
                          raw list
figures.py               describe ONE figure per Gemini call (all its panel crops together):
                          embedded figure-type taxonomy (classify + type-specific attention
                          in the same pass), context used dictionary-only, schema output,
                          disk cache keyed on crop bytes + prompt
                          (~/.cache/paper-refinery/figure-cache)
enrich.py                pair crops to captions geometrically (bboxes from parse; overlap on
                          either axis so side captions work, positional fallback when bboxes
                          are missing), rename crops to fig_N.png, assemble per-figure context
                          (title/abstract/neighbor paragraphs), splice descriptions after
                          captions
citation_extraction.py   layer 1: one Gemini call -> rough structured reference fields
                          (citation_key/title/authors/year/...); sanitizes Gemini-fabricated
                          numeric citation_keys for markerless bibliographies
citation_providers.py    HTTP layer for layer 2/3: talks to CrossRef/S2/OpenAlex (one GET
                          each, disk-cached at ~/.cache/paper-refinery/api-cache) and
                          normalizes their very different response shapes into one common
                          candidate shape -- knows how to ask, not what to trust
citation_resolution.py   layer 2/3: verify/enrich each reference against CrossRef -> S2 ->
                          OpenAlex (DOI-first shortcut, title-similarity + year-tolerance
                          acceptance); a verified year never overwrites *below* the printed
                          one (providers can merge preprint+published and report the
                          earlier year). FAST-PATH: when the SOURCE paper is identified (its
                          DOI or OCR'd title -> SourcePaper), fetch its whole bibliography in
                          one S2 bulk /references call (filled from OpenAlex referenced_works
                          when S2 is publisher-elided), match each printed ref locally, and
                          fall back to the per-entry provider search only for what doesn't
                          match -- decides what to trust from citation_providers.py's answers
citation_linking.py      layer 4: deterministic (no LLM) in-text marker detection against
                          the layer-1 EXTRACTED (printed-form) entries, plus the
                          resolution-verified-only `[surname_year]` citekey rewrite
chunker.py                section-aware split + guaranteed soft-overlap + page ranges ->
                          list[Chunk]
parse_cache.py           parse checkpoint: persist ParseResult + its raw crops to
                          <pdf>.refinery/parse_cache/ keyed on pdf-hash + parse-config
                          signature, so a re-run skips the (~10-min) OCR pass; parse_pdf_cached()
                          wraps parse_pdf, restoring crops on a hit
cli.py                    orchestrate the above (citations run concurrently with figure
                          enrich); write .chunks.json / .citations.json / paper.refinery/.
                          Public API: refine() (one PDF) and refine_many() (streaming batch,
                          yields in completion order -- branches on ParseConfig.mode: maas
                          runs each paper's whole pipeline on a flat pool of `workers`, with
                          OCR gated separately to `ocr_workers`=2 (z.ai rate-limits ~2-3
                          concurrent OCR calls; the network tails run at the higher `workers`);
                          selfhosted keeps the serial-OCR-on-one-shared-backend path);
                          CLI flags --force-parse / --from chunk / --doi. parse_pdf raises on a
                          zero-region OCR result (maas SDK reports an exhausted 429 as empty)
                          so a throttled paper is skipped, never written as a 0-chunk manifest.
```

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
  (nearest caption overlapping on EITHER axis — a caption beside its panels, not just above
  them, still pairs — via the bboxes on `CropRegion`/`CaptionRegion`, plus single-caption-page
  cluster growth for diagonal panels; positional pairing only when geometry is missing): a
  multi-panel figure's crops all land on one caption (one Gemini call, names `fig_4.1_1.png`…),
  and a crop overlapping no caption on either axis (a banner) is never renamed or described.
  Renaming happens before any Gemini call, so cropping/pairing can be sanity-checked by
  filename alone.
- **Image links are relativized last.** `parse.py` emits absolute crop paths; `cli._refine`
  rewrites them relative to the markdown's own directory (`_relativize_image_links`) at the
  very end, since only `_refine` knows both the crop dir and the final `.md` path. Scratchpad
  scripts that call `parse_pdf` directly bypass this and will show absolute paths — expected.
- **Citation detection matches the printed form; citekeys come only from verified data.**
  `citation_linking.link_citations` always runs against layer-1 EXTRACTED entries, never
  resolved ones — resolution legitimately moves years/authors off what the paper prints
  (confirmed live: linking against resolved entries lost 7/36 fmech markers). Detection runs
  on the fully ENRICHED markdown (after figure descriptions are spliced in), so marker
  offsets match the text that gets chunked — figure descriptions never introduce
  citation-shaped text by construction (dictionary-only prompt). `rewrite_markers` then
  builds `[surname_year]` citekeys from resolution-VERIFIED entries only, and only rewrites
  a marker when every reference it points at has one; everything else is left exactly as
  printed. Citations and figure-enrich still run concurrently in `cli._refine` — only this
  final rewrite pass is serialized after both join.

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
