# AGENTS.md

Repository guidance for coding agents working on paper-refinery. Keep this file as the
single repository instruction file; update it when implementation contracts change.

## What this is

A standalone preprocessor for **papis-ask**: it turns a PDF into section-aware, overlapping
text chunks (with figure descriptions, page numbers, and standardized `[surname_year]`
in-text citekeys) that papis-ask ingests via paper-qa's `Docs.aadd_texts` — replacing
pypdf's blind char-window chunking. A full run writes `<stem>.chunks.json` and sibling
`<stem>.md` (a math-collapsed review copy with one-based physical PDF page markers), plus
`<stem>.citations.json` when references are found. `<stem>.refinery/refinery.md` is the
uncollapsed input for `--from chunk`, which does not refresh the review copy;
`--from review` writes only the review copy from it (no config/keys/network; `cli._review_only`). The work
directory also holds `references.md`, `resolution_report.txt` and `figures/`.

## Commands

Tooling lives in `.venv/` (plain venv, Python 3.14). The existing local `uv.lock` is ignored
by Git; the development workflow uses this environment directly. The default
install is cloud-only and torch-free: `uv pip install -e ".[dev]"`. Only the optional LOCAL
OCR backend (`.[local]` extra) pulls torch, and only there does the CPU-only-torch-first
two-step matter — see README.md's "Development setup" / "Local (selfhosted) OCR backend".

`.envrc` selects this project-local environment. Call tools by their `.venv/bin/` paths;
do not create another environment or use `uv run` to introduce a lockfile. The package
supports Python >=3.12; keep syntax compatible with that floor. `mathunicode` is an editable
sibling dependency at `../mathunicode` (see `[tool.uv.sources]` in `pyproject.toml`).

```bash
.venv/bin/pytest                              # full test suite
.venv/bin/pytest tests/test_parse.py          # one file
.venv/bin/pytest tests/test_references.py::test_sort_references_by_number_fixes_scrambled_order  # one test
.venv/bin/ruff check .                         # lint (line-length 100)
.venv/bin/ruff format --check .                # check formatting without changing files
.venv/bin/ruff format .                        # format
.venv/bin/refinery path/to/paper.pdf           # run the full pipeline
```

The test suite is fully offline: Gemini and the OCR backend are always mocked/injected
(`describe`/`client` params, hand-built region fixtures). No test needs network, a GPU, or
`llama-server`. Backend port tests do need permission to open local sockets. The autouse
fixture in `tests/conftest.py` isolates HOME and clears config/cache overrides so tests do
not read real secrets or caches. Preserve that isolation and use synthetic fixtures.

Source is in `paper_refinery/`, tests in `tests/test_<module>.py`, and model evaluations in
`scripts/eval_models.py`. Use four-space indentation, Ruff's import sorting and formatting,
and descriptive snake_case names. Ruff targets Python 3.12 with a 100-character line limit
and McCabe complexity <=10; decompose complex functions rather than suppressing the check.
`pyproject.toml` also defines the BasedPyright standard gate for `paper_refinery`, with
cross-module private imports reported as errors. Keep public contracts explicit.
Recent commit subjects use `feat:`, `fix:`, and `chore:` with concise imperative descriptions.

## Running the pipeline live: two OCR backends (`ParseConfig.mode`)

**`mode="maas"` (default): cloud.** Layout+OCR run on Zhipu's GLM-OCR API
(`api.z.ai/api/paas/v4/layout_parsing`, model `glm-ocr`) via the glmocr SDK's maas client —
no GPU, no GGUF, no llama-server. Needs `ZHIPU_API_KEY` (secrets `zai.env`). The SDK's maas
`json_result` DROPS `native_label`; `parse._normalize_maas_regions` restores it from the raw
`result._maas_response` and strips the cloud's `<div>` content wrappers.

**`mode="selfhosted"`: local.** `parse_pdf` spawns/tears down a `llama-server` (llama.cpp)
serving GLM-OCR GGUF weights, driven by the glmocr SDK. Needs the `.[local]` install extra
(torch + PP-DocLayout-V3, downloaded from HF Hub on first use with `HF_TOKEN`), `llama-server`
on PATH, the two GGUF files, and their paths in `config.toml` `[parse]`
(`model_path`/`mmproj_path`). `OcrBackend.server` is None in maas mode; the watchdog guards
the kill.

**Always test pipeline changes live against real PDFs** — unit tests alone have missed real
OCR-ordering bugs. Use an isolated work directory and the smallest relevant sample; inspect
page order, captions, references, and output links as appropriate. Reuse checkpoints unless
the change specifically needs fresh OCR. Documentation-only changes need documentation
checks rather than paid OCR/API runs. Read the local papers-and-PDFs skill's matching
reference before expensive conversion or library operations.

## Architecture

Pipeline stages, each a single-purpose module; `cli.py` owns all wiring (`_refine` parses,
then `_refine_parsed` runs the post-parse stages), and the modules own logic. Keep orchestration
in `cli.py`; module dependencies serve shared types, stage-specific helpers, provider HTTP
calls, and checkpointing rather than duplicating pipeline orchestration.

```
backend.py               llama-server + GlmOcr lifecycle; ocr_backend() is reusable across PDFs
parse.py                 PDF -> ParseResult (body markdown + figure crops + raw references)
reading_order.py         conservative geometric repair within same-column runs; preserve
                          macro column order and fall back to index order without bboxes
pdf_split.py             split long PDFs into parts (default <=100 pages), then merge results
                          with page markers, crops, captions, and references offset correctly
references.py            bibliography repair used by parse.py: reclaim mislabeled
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
citation_extraction.py   layer 1: batched Gemini calls -> rough structured reference fields
                          (citation_key/title/authors/year/...); sanitizes Gemini-fabricated
                          numeric citation_keys for markerless bibliographies
citation_providers.py    HTTP layer for layer 2/3: talks to CrossRef/S2/OpenAlex (one GET
                          each, disk-cached at ~/.cache/paper-refinery/api-cache) and
                          normalizes their very different response shapes into one common
                          candidate shape -- knows how to ask, not what to trust
citation_resolution.py   layer 2/3: verify/enrich references in CitationConfig.title_search_order
                          (default CrossRef -> S2 -> OpenAlex; DOI-first shortcut,
                          title-similarity + year-tolerance
                          acceptance); a verified year never overwrites *below* the printed
                          one (providers can merge preprint+published and report the
                          earlier year). FAST-PATH: when the SOURCE paper is identified (its
                          DOI or OCR'd title -> SourcePaper), fetch its whole bibliography in
                          bulk reference-list calls to list-capable providers (S2/OpenAlex)
                          in configured order until coverage is sufficient, match locally, and
                          fall back to the per-entry provider search only for what doesn't
                          match -- decides what to trust from citation_providers.py's answers.
                          METADATA CHANNEL: SourceMeta (external-input sibling of SourcePaper)
                          carries what a caller already knows (doi/title/year/authors +
                          references); source_from_meta -> SourcePaper (strengthens source ID;
                          _source_confident is two-tier so year/author never veto a strong
                          title). Caller-supplied `references` (papis citations, any of
                          refinery's or CrossRef-reference shape) are normalized into a local
                          bulk pool matched FIRST (match="papis"); provider list fetches are
                          skipped entirely when they cover every printed ref. to_papis_citations() is
                          the inverse (resolved refs -> papis/CrossRef citations: shape).
citation_linking.py      layer 4: deterministic (no LLM) in-text marker detection against
                          the layer-1 EXTRACTED (printed-form) entries, plus the
                          resolution-verified-only `[surname_year]` citekey rewrite
chunker.py                section-aware split + guaranteed soft-overlap + page ranges ->
                          list[Chunk]
parse_cache.py           parse checkpoint: persist ParseResult + its raw crops to
                          <stem>.refinery/parse_cache/ keyed on pdf-hash + parse-config
                          signature, so a re-run skips the (~10-min) OCR pass; parse_pdf_cached()
                          wraps parse_pdf, restoring crops on a hit
markers.py               single page-marker format and regex
tables.py / fences.py    HTML-table conversion and OCR code-fence repair
text_utils.py            shared text/name normalization and leading-number parsing
disk_cache.py / retry.py shared atomic JSON caching and retry/backoff helpers
typeset.py                markdown -> typeset PDF via pandoc + xelatex (system binaries, not
                          pip deps): a clean, reflowed reading copy with a TOC and inline
                          images -- no figure-description or citation stages. render_pdf()
                          drives xelatex directly (twice, for TOC page numbers), not via
                          pandoc's own --pdf-engine path, since that treats any nonzero engine
                          exit code as total failure and discards a PDF that nonstopmode's
                          error recovery actually completed -- judged instead by whether a
                          PDF came out. Repairs OCR'd input before compiling: strips
                          <page_number> markers (bypassing chunker.py, which normally does),
                          blanks a FIGURE_CROP/FIGURE-numbered image's alt text (else pandoc
                          auto-captions/floats it, duplicating the real caption already in the
                          body), trims the padding space inside $ .../$$ ...$$ (pandoc's
                          dollar-math needs none), merges an OCR-malformed stacked
                          superscript/subscript (TeX's one truly fatal, non-recoverable math
                          error class), and demotes an immediately-repeated heading (OCR
                          re-tagging a book's running header as its own heading) -- demotion
                          is positional (by occurrence index among all headings, from
                          _find_headings), never text-matched, since a title like "EXERCISES"
                          can legitimately repeat many times as real section starts.
                          TypesetConfig.clean_toc_with_llm (opt-in, off by default) adds two
                          sequential batched Gemini calls (reusing CitationConfig's
                          model/key/retry settings), not one: classify_real_headings
                          first (heading text + what follows it only) catches a printed
                          table-of-contents line mistagged as a heading, inconsistently, so no
                          positional rule catches all of it; classify_chapter_level then runs
                          only on confirmed-real headings (adding "preceded by" context) to tell
                          a chapter/part heading from a section/subsection one, so LaTeX
                          page-breaks correctly (\chapter vs \section) and the TOC nests
                          properly. Kept as two calls, not one combined pass -- confirmed live
                          that giving "preceded by" context during the real/not-real judgment
                          backfires: a dense run of consecutive table-of-contents lines are each
                          "preceded by" something that itself looks like a chapter title (the
                          correct pattern for a genuine heading), tricking the model into
                          treating the whole polluted run as real chapters. A final deterministic
                          pass, _promote_orphaned_chapter_subsections, catches what neither
                          classification call can: a chapter whose bare "CHAPTER N" marker was
                          never OCR-tagged as its own heading at all AND doesn't follow the
                          previous chapter's "EXERCISES" (confirmed live on Weinstock's
                          "Calculus of Variations" -- chapters 2 and 9 both), by promoting a
                          "section"-level heading to "chapter" when it's immediately followed by
                          a subsection numbered "N-1." (chapter numbering always resets there).
                          Two adjacent "chapter"-level headings (a bare "CHAPTER N" marker still
                          OCR'd as its own region, immediately followed by its title) are merged
                          into one combined H1 -- otherwise LaTeX would page-break twice per
                          chapter. Judgments are matched back to their candidate by an explicit
                          `index` field the model itself echoes, not by response-list position --
                          confirmed live the model's output count occasionally doesn't exactly
                          match the input (150 judgments for 147 candidates) -- and fail open
                          per-candidate on anything unmatched, so a few stray indices cost only
                          those candidates, not the whole pass. A font/size/line-spacing default
                          (Noto Serif, 11pt, 1.15x) is tuned for reading rather than LaTeX
                          defaults; TypesetConfig.main_font falls back to xelatex's own default
                          (via fc-match) when not installed, rather than hard-failing the compile
                          over a missing system font.
cli.py                    orchestrate the above (citations run concurrently with figure
                          enrich); write .chunks.json / .citations.json / paper.refinery/.
                          Public API: refine() (one PDF) and refine_many() (streaming batch,
                          yields in completion order -- branches on ParseConfig.mode: maas
                          runs each paper's whole pipeline on a flat pool of `workers`, with
                          OCR gated separately to `ocr_workers`=2 (z.ai rate-limits ~2-3
                          concurrent OCR calls; the network tails run at the higher `workers`);
                          selfhosted keeps the serial-OCR-on-one-shared-backend path).
                          refine()/refine_many() take a SourceMeta `source`/`sources` bundle
                          (see citation_resolution) -- `doi=`/`dois=` are shorthands folded in.
                          Four console scripts: `refinery` (one PDF; main()), `refinery-batch`
                          (many PDFs; main_many() -> refine_many; --workers/--ocr-workers/
                          --force-parse/--from chunk|review/--meta-map FILE={pdf_path: SourceMeta}),
                          `refinery-export-citations` (main_export_citations() ->
                          to_papis_citations -> papis citations: YAML), and `refinery-typeset`
                          (main_typeset() -> _typeset() -> typeset.render_pdf; PDF input is
                          parsed with checkpoint reuse and an absolute work_dir -- parse.py
                          bakes its work_dir Path straight into each crop's markdown link, so
                          a relative one would resolve against the wrong base once render_pdf
                          runs pandoc/xelatex with cwd=<parsed.md's directory>; markdown input
                          is typeset as-is, e.g. a prior run's refinery.md, already correctly
                          relativized by _relativize_image_links). parse_pdf raises on a
                          zero-region OCR result (maas SDK reports an exhausted 429 as empty) so
                          a throttled paper is skipped, never written as a 0-chunk manifest.
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
  into placeholders + saved crops. `reading_order.py` starts from PP-DocLayout-V3's `index`
  order, then repairs within-column inversions geometrically without interleaving columns;
  missing geometry retains index order. Numbered bibliographies are re-sorted by printed number
  (`_sort_references_by_number`) precisely because `index` can't be trusted there.
- **Figure anchoring is caption-based, not placeholder-based.** Placeholders shift between
  parser runs; the `FIGURE N.M` caption is the stable anchor. When the layout model detected
  caption regions (`ParseResult.figure_captions`), enrich accepts only body lines matching
  them — the caption regex alone is the fallback. Crops pair with captions *geometrically*
  (nearest caption overlapping on EITHER axis — a caption beside its panels, not just above
  them, still pairs — via the bboxes on `CropRegion`/`CaptionRegion`, plus single-caption-page
  cluster growth for diagonal panels; positional pairing only when geometry is missing): a
  multi-panel figure's crops all land on one caption (one Gemini call, names `fig_4.1_1.png`…),
  and a crop overlapping no caption on either axis (a banner) is left alone by default.
  `--describe-uncaptioned` opts into separate descriptions for unmatched images.
  Renaming happens before any Gemini call, so cropping/pairing can be sanity-checked by
  filename alone.
- **Image links are relativized after enrichment.** `parse.py` emits crop paths based on
  work_dir; `cli._refine_parsed` rewrites them relative to the markdown's own directory
  (`_relativize_image_links`), since orchestration knows both the crop dir and final `.md`
  path. Scripts that call `parse_pdf` directly bypass this and retain the original paths.
- **Citation detection matches the printed form; citekeys come only from verified data.**
  `citation_linking.link_citations` always runs against layer-1 EXTRACTED entries, never
  resolved ones — resolution legitimately moves years/authors off what the paper prints
  (confirmed live: linking against resolved entries lost 7/36 fmech markers). Detection runs
  on the fully ENRICHED markdown (after figure descriptions are spliced in), so marker
  offsets match the text that gets chunked — figure descriptions never introduce
  citation-shaped text by construction (dictionary-only prompt). `rewrite_markers` then
  builds `[surname_year]` citekeys from resolution-VERIFIED entries only, and only rewrites
  a marker when every reference it points at has one; everything else is left exactly as
  printed. Citations and figure-enrich run concurrently in `cli._refine_parsed` — only this
  final rewrite pass is serialized after both join.

### Config & secrets

- `config.py`: dataclass-per-stage (`ParseConfig`, `FigureConfig`, `ChunkConfig`,
  `CitationConfig`, `TypesetConfig`) bundled in `RefineryConfig`. `load_config()` overlays
  `~/.config/paper-refinery/config.toml` — each TOML table maps to a sub-config by name, each
  key must match a dataclass field (unknown section/key raises). Every field has a code
  default; the TOML file is optional and only sets machine-specific values (model paths,
  `n_gpu_layers`, `glmocr_config_overrides`) and deliberate stage overrides.
  `XDG_CONFIG_HOME` changes the config base; `PAPER_REFINERY_CONFIG` and
  `PAPER_REFINERY_SECRETS_DIR` override the individual locations.
- Secrets load from `~/.config/paper-refinery/secrets/` — **one `*.env` file per service**
  (`google.env`, `hf.env`, `zai.env`, `openalex.env`, `contact.env`, optionally `s2.env`),
  loaded via python-dotenv inside `load_config()` early (before
  `parse_pdf`, since `HF_TOKEN` must be set for the HF Hub check). This directory is
  deliberately project-owned and never shared with other tools' secrets files — sourcing an
  unrelated tool's env (e.g. papis-ask's `OPENAI_BASE_URL`) would silently redirect glmocr's
  OpenAI-compatible client away from our own `llama-server`. paper-refinery only ever *reads*
  these files; never create, write, or read their contents.

## Cached outputs and hand edits

- Full runs write uncollapsed `refinery.md` as the exact chunking input; keep math collapsing
  confined to the sibling review copy. `--from chunk` only re-chunks; `--from review` only
  writes the review copy and bypasses config, keys, OCR, and network. Both work in batch mode.
- Full runs checksum `refinery.md` in `refinery.md.sha256`. A changed file is backed up and
  rejected before OCR/API calls unless `--overwrite-edits` is explicitly selected; legacy
  files without checksums are backed up before replacement. Preserve this guard and never
  use a full run merely to refresh a review copy of an imported or hand-corrected book.
- OCR checkpoints use PDF hash plus parse configuration. Long books also cache split parts
  under `parts/part_N/`, keyed on the original PDF hash. Check existing parts before OCR-ing
  again; `--force-parse` deliberately bypasses cached parsing.
- Default chunks target 6000 maximum / 1000 minimum characters with a guaranteed soft
  overlap window of 300–700 characters (ideal 500). Page ranges describe the chunk body
  before overlap is prepended. Back matter is excluded by default; references belong in
  the citation manifest. Preserve manifest schema versioning and plain-data API boundaries.
- Do not commit sample PDFs, generated manifests, figure crops, evaluation reports, private
  logs, caches, or credentials. Use ignored `samples/`, `scratch/`, `out/`, or `/tmp` for
  local validation and scratch work.

## Local entry points and installation

- `refinery` performs OCR, Gemini figure descriptions, and citation lookup. For a general
  PDF that only needs OCR/markdown or a clean reading PDF, use `refinery-typeset`; PDF input
  writes `<stem>.refinery/parsed.md` and uses only the OCR backend unless `--clean-toc` is
  requested. Markdown input skips OCR. Typesetting needs system `pandoc` and `xelatex`.
- For immediate inspection without OCR or network, use `mutool info` / `mutool draw`.
  This machine's PDF Read rendering path lacks Poppler; render a selected one-based page
  range with `mutool draw -q -r 80 -o /tmp/refinery-page-%d.png file.pdf 1-2`, then inspect
  the images. Text extraction only reads the embedded text layer; scans need OCR.
- `papis` owns library management and indexing. `pask index` may refine PDFs and index more
  broadly than its refine query; `--no-refine` and `--raw` embed raw pypdf text rather than
  merely skipping OCR. Consult the papers-and-PDFs skill's `references/ingest.md` before
  changing library indexes. Its `references/convert.md` covers conversion and re-chunking.
- `refinery` on PATH is a non-editable uv tool snapshot, separate from this checkout.
  Use `.venv/bin/refinery` to exercise source changes. Compare installed/source packages
  before assuming a change is installed; reinstall only when no refinery process is running:
  `uv tool install --force --from ~/projects/paper-refinery paper-refinery`.
  Never install development dependencies into the uv tool environment.

## Model changes

Pin exact model names, never `-latest` aliases. Before changing a model, evaluate candidates
on the real library with `.venv/bin/python scripts/eval_models.py check|extraction|figures`
(see README.md for arguments). `check` lists API models without paid generation;
`extraction` and `figures` make paid Gemini calls. Judge figure descriptions against the
images, not just their text. The figure cache is keyed on crop bytes and prompt, excluding
the model, so a model change alone does not regenerate existing descriptions. Changes to
papis-ask's embedding model require a full re-embed and are outside ordinary refinery edits.
