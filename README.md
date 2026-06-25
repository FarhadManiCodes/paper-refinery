# paper-refinery

Parse, figure-enrich, and chunk papers into RAG-ready chunks for **papis-ask**.

It is a standalone preprocessor: it turns a PDF into a set of clean, section-aware,
overlapping text chunks (with figure descriptions and page numbers) that papis-ask
ingests directly — bypassing pypdf's blind char-window chunking.

## Pipeline

```
PDF
 └─ parse    LlamaParse  -> clean markdown (LaTeX intact) + figure images + page markers
 └─ figures  Gemini      -> describe each figure (what's compared, trends; never fabricated numbers)
 └─ enrich               -> splice figure descriptions next to their captions
 └─ chunker              -> section-aware split + guaranteed soft-overlap + page numbers
 └─ paper.chunks.json    -> consumed by papis-ask via aadd_texts
```

## Why

pypdf mangles equations and emits glyph garbage for figures; paper-qa then
char-chunks the result blindly. paper-refinery produces faithful, semantically
chunked text instead. See the design notes for the validated chunking policy.

## Status

Early scaffold. The **chunker** is implemented and tested; `parse` / `figures` /
`enrich` are stubs being filled in.

## Usage (planned)

```bash
refinery path/to/paper.pdf            # -> path/to/paper.chunks.json
```
