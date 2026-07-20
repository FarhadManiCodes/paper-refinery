"""Markdown -> typeset PDF via pandoc + xelatex: a clean, re-typeset reading copy with a
table of contents and inline images, for a PDF that's otherwise been just OCR'd (no figure
description, no citation resolution -- see cli.py's ``_typeset``/``main_typeset``).

OCR'd math and headings need a few targeted repairs before they survive a LaTeX compile;
see each helper below for why. ``render_pdf`` drives xelatex directly rather than through
pandoc's own ``--pdf-engine`` path -- see its docstring.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from .citation_extraction import make_client
from .config import CitationConfig, TypesetConfig
from .markers import PAGE_MARKER_RE
from .retry import call_with_backoff

if TYPE_CHECKING:
    from google.genai import Client

logger = logging.getLogger(__name__)

# refinery's OCR convention pads math delimiters with a space ("$ x $", "$$ ... $$") for
# markdown-viewer readability, but pandoc's dollar-math parser only recognizes math when no
# whitespace touches the $/$$ -- otherwise it's left as literal text, which then breaks the
# LaTeX compile on the un-consumed tokens (e.g. a bare \prime outside math mode).
_MATH_SPAN_RE = re.compile(r"\$\$(.*?)\$\$|\$(.*?)\$", re.S)

# OCR occasionally emits a stacked superscript/subscript (e.g. "y'^2" as "y ^ {\prime} ^
# {2}") instead of nesting it -- valid math notation, but TeX's "double superscript"/"double
# subscript" is a hard, non-recoverable error even in nonstopmode (unlike "Missing $
# inserted", which TeX patches around on its own and keeps compiling). Merging the two
# scripts into one is visually close enough -- the same fidelity bar already accepted for
# occasional equation-numbering slips -- and keeps one malformed equation from aborting the
# whole document's compile.
_DOUBLE_SCRIPT_RE = re.compile(r"([\^_])\s*\{([^{}]*)\}\s*\1\s*\{([^{}]*)\}")

_HEADING_RE = re.compile(r"^(#{1,2})\s+(.*)$")

# parse.py names a figure placeholder's alt text "FIGURE_CROP page:idx"; enrich.py later
# renames it to "FIGURE number" once crops are paired to captions -- either way it's an
# internal id, not a caption. pandoc turns any standalone "![alt](path)" into a floating,
# auto-numbered LaTeX figure captioned with that alt text; the real OCR'd caption is
# already body text right next to the placeholder (see enrich.py's caption-based figure
# anchoring), so a non-blank alt text would duplicate it and let the float drift away from
# that paragraph. Matches both forms -- always refinery's own placeholder, never
# user-authored text.
_FIGURE_CROP_ALT_RE = re.compile(r"!\[FIGURE(?:_CROP)?[^\]]*\]")

# parse.py renders a footnote region as its own clearly-labeled paragraph ("> **Footnote:**
# ..."), right where the region falls in reading order -- deliberately not attempting to
# re-anchor it at its in-text marker (OCR gives no reliable way to do that). That's the
# right shape for a RAG chunk (unambiguous even as plain text) but reads as just another
# quoted paragraph in a *typeset* PDF, not an actual footnote. Converted here, PDF-only, to
# pandoc's footnote syntax (a `[^N]` reference + `[^N]: content` definition), which LaTeX
# renders as a real page-bottom footnote -- no manual positioning needed, LaTeX places it on
# whatever page the reference lands on. The reference is attached to the end of the
# *preceding* paragraph (the best available anchor, still not the true in-text marker).
_FOOTNOTE_BLOCKQUOTE_RE = re.compile(r"\A> \*\*Footnote:\*\* (.+)\Z", re.S)

_LUA_FILTER_TEMPLATE = """
function Image(img)
  if img.attributes.width == nil then
    img.attributes.width = "{width}"
  end
  return img
end
"""


def _font_available(name: str) -> bool:
    """Whether fontconfig actually has ``name`` installed, not just a fallback match --
    ``fc-match`` always returns *some* font, so a mismatched family name in its output
    means the requested one wasn't found. Used to fall back gracefully rather than let
    xelatex hard-fail the whole compile over a missing default font
    (``TypesetConfig.main_font``) that isn't installed on a given machine.
    """
    try:
        result = subprocess.run(
            ["fc-match", name], capture_output=True, text=True, timeout=5, check=False
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return True  # can't check -- don't block on it, let xelatex itself decide
    return name.lower() in result.stdout.lower()


def _fix_double_scripts(text: str) -> str:
    while True:
        new_text = _DOUBLE_SCRIPT_RE.sub(r"\1{\2 \3}", text)
        if new_text == text:
            return text
        text = new_text


def _normalize_math_dollars(md: str) -> str:
    def repl(m: re.Match) -> str:
        if m.group(1) is not None:
            return f"$${_fix_double_scripts(m.group(1)).strip()}$$"
        return f"${_fix_double_scripts(m.group(2)).strip()}$"

    return _MATH_SPAN_RE.sub(repl, md)


def _normalize_heading_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().rstrip(".").upper()


def _find_headings(md: str) -> list[tuple[int, str]]:
    """(line index, heading text) for every ``#``/``##`` line, in document order -- the
    shared indexing ``_apply_heading_levels`` and the optional LLM classification passes
    (``classify_real_headings``/``classify_chapter_level``) rely on staying aligned: a
    judgment's position in its input list must mean the same heading occurrence as a
    ``heading_levels`` key."""
    lines = md.split("\n")
    found: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        m = _HEADING_RE.match(line)
        if m:
            found.append((i, m.group(2)))
    return found


def heading_candidates(md: str, max_context_chars: int = 220) -> list[tuple[str, str, str]]:
    """(preceding heading's text, heading text, following body text) for every heading, in
    the same order/indexing as ``_find_headings`` -- the classification input for
    ``classify_real_headings``/``classify_chapter_level``. What a heading is followed by
    (substantial prose for a genuine one; another heading-like line, or a dense listing of
    subsection titles/page numbers, for a mistagged table-of-contents line) is the signal
    real/not-real classification judges on. What precedes it (a chapter title directly
    follows either a bare "CHAPTER N" marker or the previous chapter's "EXERCISES") is a
    separate signal chapter/section-level classification judges on -- useful since the OCR
    layout model sometimes never tags the bare "CHAPTER N" marker itself as its own heading
    at all, confirmed live on more than one chapter of a real book, so a chapter's *title*
    is the only heading candidate that will ever exist for it. Kept as one combined
    candidate tuple, but deliberately fed to the two classification passes at different
    times (see ``_llm_heading_levels``) -- giving the "preceded by" signal during the
    real/not-real judgment confirmed live to backfire: a dense run of consecutive
    table-of-contents lines are each "preceded by" something that itself looks like a
    chapter title, which is exactly the (correct, for a genuine heading) pattern that
    signal is meant to catch, so revealing it up front tricks the model into treating the
    whole polluted run as real chapters."""
    lines = md.split("\n")
    headings = _find_headings(md)
    candidates: list[tuple[str, str, str]] = []
    for pos, (line_i, text) in enumerate(headings):
        preceding = headings[pos - 1][1] if pos > 0 else ""
        start = line_i + 1
        end = headings[pos + 1][0] if pos + 1 < len(headings) else len(lines)
        context = " ".join(line.strip() for line in lines[start:end] if line.strip())
        candidates.append((preceding, text, context[:max_context_chars]))
    return candidates


_HEADING_LEVELS = frozenset({"chapter", "section", "not_heading"})


class _RealHeadingJudgment(BaseModel):
    index: int = Field(
        description="The 1-based candidate number this judgment is for, exactly as given "
        "in the numbered input list below -- required so judgments can be matched back to "
        "their candidate even if the model's output count doesn't exactly match the input."
    )
    is_real_heading: bool = Field(
        description="True if this is a genuine heading marking the actual start of that "
        "content. False if it's a line copied from the document's own printed table of "
        "contents, index, or a similar front/back-matter listing that got mistagged as a "
        "heading because it visually resembles one -- typically because it's followed "
        "immediately by another heading-like line, or by a dense listing of subsection "
        "titles/page numbers, rather than substantial prose content. This can happen even "
        "when the line's own text is IDENTICAL to a real chapter's title, since a table "
        "of contents quotes titles verbatim -- judge from what follows the candidate, "
        "never from how chapter-like its own text looks."
    )


_REAL_HEADING_PROMPT = (
    "A scanned document was OCR'd and its layout model tagged some lines as section "
    "headings. Most are genuine, but occasionally a line from the document's own printed "
    "table of contents (or a similar listing) gets mistagged as a heading because it "
    "visually resembles one -- see the schema for exactly what distinguishes the two.\n\n"
    "For each numbered candidate below, given its heading text and the text immediately "
    "following it in reading order, judge whether it's a genuine heading or a mistagged "
    "listing line.\n\n"
    "Return one JSON object per candidate, each carrying that candidate's own number as "
    "`index`, matching the schema exactly."
)


def classify_real_headings(
    candidates: list[tuple[str, str]], cfg: CitationConfig, client: Client | None = None
) -> list[bool]:
    """Judge each (heading text, following context) candidate as a genuine heading or a
    mistagged table-of-contents/listing line, in one batched, schema-enforced Gemini call.
    Matched back to its candidate by the judgment's own declared ``index`` (not
    response-list position) -- confirmed live: the model's output count occasionally
    doesn't exactly match the input count (e.g. 150 judgments for 147 candidates), and
    matching positionally would misattribute every judgment after the first discrepancy.
    Fails open per-candidate (defaults to real) on anything unmatched, so a handful of
    stray/duplicate indices only cost those few candidates their classification, not the
    whole pass -- this only ever demotes a heading it's confident about; never silently
    drops a real one."""
    from google.genai import types

    if not candidates:
        return []
    client = client or make_client(cfg)

    listing = "\n\n".join(
        f"{i + 1}. HEADING: {text}\n   FOLLOWED BY: {context}"
        for i, (text, context) in enumerate(candidates)
    )
    prompt = f"{_REAL_HEADING_PROMPT}\n\n{listing}"
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=list[_RealHeadingJudgment],
        temperature=0.0,
        thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL),
    )

    response = call_with_backoff(
        lambda: client.models.generate_content(model=cfg.model, contents=prompt, config=config),
        cfg.retry_attempts,
        cfg.retry_base_delay,
    )

    parsed = response.parsed
    rows = parsed if isinstance(parsed, list) else []
    by_index = {r.index: r.is_real_heading for r in rows if isinstance(r, _RealHeadingJudgment)}
    if len(by_index) != len(candidates):
        logger.warning(
            "classify_real_headings: got %d usable judgments for %d candidates; unmatched "
            "ones keep their heading",
            len(by_index),
            len(candidates),
        )
    return [by_index.get(i + 1, True) for i in range(len(candidates))]


class _ChapterLevelJudgment(BaseModel):
    index: int = Field(
        description="The 1-based candidate number this judgment is for, exactly as given "
        "in the numbered input list below -- required so judgments can be matched back to "
        "their candidate even if the model's output count doesn't exactly match the input."
    )
    level: str = Field(
        description='"chapter" for a top-level chapter/part heading (even if it lacks an '
        'explicit "CHAPTER N"-style marker: a chapter\'s own title always directly '
        "follows either a bare chapter-number marker or the previous chapter's "
        '"EXERCISES", and is usually followed by subsection numbering that restarts at '
        '1, e.g. "9-1."). "section" for a genuine subsection heading within a chapter '
        '(e.g. numbered "N-M. Title", or "EXERCISES").'
    )


_CHAPTER_LEVEL_PROMPT = (
    "Every candidate below is already confirmed to be a genuine heading (not a "
    "table-of-contents line or similar) in a scanned, OCR'd document. For each, given its "
    "heading text, the heading immediately preceding it, and the text immediately "
    "following it in reading order, classify it as a top-level chapter/part heading or a "
    "section/subsection heading within a chapter -- see the schema for exactly what each "
    "value means.\n\n"
    "Return one JSON object per candidate, each carrying that candidate's own number as "
    "`index`, matching the schema exactly."
)


def classify_chapter_level(
    candidates: list[tuple[str, str, str]], cfg: CitationConfig, client: Client | None = None
) -> list[str]:
    """Classify each already-confirmed-real (preceding heading, heading text, following
    context) candidate as "chapter" or "section", in one batched, schema-enforced Gemini
    call. Matched back to its candidate by declared ``index``, same rationale as
    ``classify_real_headings``. Fails open per-candidate (defaults to "section", today's
    baseline level) on anything unmatched or an unrecognized value."""
    from google.genai import types

    if not candidates:
        return []
    client = client or make_client(cfg)

    listing = "\n\n".join(
        f"{i + 1}. HEADING: {text}\n   PRECEDED BY: {preceding or '(nothing before it)'}\n"
        f"   FOLLOWED BY: {context}"
        for i, (preceding, text, context) in enumerate(candidates)
    )
    prompt = f"{_CHAPTER_LEVEL_PROMPT}\n\n{listing}"
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=list[_ChapterLevelJudgment],
        temperature=0.0,
        thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL),
    )

    response = call_with_backoff(
        lambda: client.models.generate_content(model=cfg.model, contents=prompt, config=config),
        cfg.retry_attempts,
        cfg.retry_base_delay,
    )

    parsed = response.parsed
    rows = parsed if isinstance(parsed, list) else []
    by_index = {
        r.index: r.level
        for r in rows
        if isinstance(r, _ChapterLevelJudgment) and r.level in ("chapter", "section")
    }
    if len(by_index) != len(candidates):
        logger.warning(
            "classify_chapter_level: got %d usable judgments for %d candidates; unmatched "
            'ones default to "section"',
            len(by_index),
            len(candidates),
        )
    return [by_index.get(i + 1, "section") for i in range(len(candidates))]


def _resolve_heading_levels(headings: list[tuple[int, str]], levels: dict[int, str]) -> list[str]:
    """Per-heading-occurrence final level, folding in the mechanical duplicate rule: a
    heading that repeats the immediately preceding *kept* heading verbatim is always
    "not_heading" regardless of its classified level -- typesetting convention repeats a
    running section title atop the next page, which the OCR layout model then tags as its
    own heading (observed: "PREFACE" tagged on both its start page and the very next
    page). Only collapses a heading against the *last kept* heading, so two genuinely
    different sections named the same thing far apart in the book are untouched. Anything
    unclassified (or an unrecognized value) defaults to "section", today's flat baseline.
    """
    resolved: list[str] = []
    last_kept_norm: str | None = None
    for pos, (_line_i, text) in enumerate(headings):
        norm = _normalize_heading_text(text)
        if norm == last_kept_norm:
            resolved.append("not_heading")
            continue
        level = levels.get(pos, "section")
        if level not in _HEADING_LEVELS:
            level = "section"
        resolved.append(level)
        if level != "not_heading":
            last_kept_norm = norm
    return resolved


def _heading_line_replacements(
    headings: list[tuple[int, str]], resolved: list[str]
) -> dict[int, str | None]:
    """line index -> replacement text (or None to drop the line entirely), per each
    heading's resolved level: "chapter" -> H1, "section" -> H2, "not_heading" -> demoted to
    plain text. Two adjacent "chapter" headings (a bare "CHAPTER N" marker immediately
    followed by its title, still two separate OCR regions -- or, when the marker itself was
    never tagged as a heading at all, just the title alone) are merged into one combined H1
    line, with the merged-away line(s) mapped to None: otherwise LaTeX would page-break
    twice per chapter, once for each."""
    line_action: dict[int, str | None] = {}
    pos = 0
    n = len(headings)
    while pos < n:
        line_i, text = headings[pos]
        level = resolved[pos]
        if level == "chapter":
            combined = text
            while pos + 1 < n and resolved[pos + 1] == "chapter":
                next_line_i, next_text = headings[pos + 1]
                combined = f"{combined}: {next_text}"
                line_action[next_line_i] = None
                pos += 1
            line_action[line_i] = f"# {combined}"
        elif level == "section":
            line_action[line_i] = f"## {text}"
        else:  # not_heading
            line_action[line_i] = text  # demote: keep the text, drop the heading markup
        pos += 1
    return line_action


def _apply_heading_levels(md: str, levels: dict[int, str] | None = None) -> str:
    """Rewrite each heading per its classified level -- see ``_resolve_heading_levels``
    (the mechanical-duplicate + default-level rule) and ``_heading_line_replacements``
    (H1/H2/demote + adjacent-chapter merging) for what each step does and why."""
    headings = _find_headings(md)
    resolved = _resolve_heading_levels(headings, levels or {})
    line_action = _heading_line_replacements(headings, resolved)

    out_lines: list[str] = []
    for i, line in enumerate(md.split("\n")):
        if i not in line_action:
            out_lines.append(line)
            continue
        replacement = line_action[i]
        if replacement is not None:  # None -> merged away, drop the line
            out_lines.append(replacement)
    return "\n".join(out_lines)


_FIRST_SUBSECTION_RE = re.compile(r"^\d+-1\.")  # a chapter's own first subsection, e.g. "9-1."


def _promote_orphaned_chapter_subsections(
    heading_texts: list[str], resolved: list[str]
) -> list[str]:
    """Promote a "section"-level heading to "chapter" when it's immediately followed (among
    kept headings) by a chapter's first subsection ("N-1. Title") -- subsection numbering
    always resets at a chapter boundary, so an "N-1." heading whose immediately preceding
    kept heading isn't already "chapter"-level means that preceding heading IS chapter N's
    own title, just missing (or never OCR-tagged with) an explicit "CHAPTER N" marker.
    Deterministic, not model judgment: confirmed live (Weinstock "Calculus of Variations")
    the classification pass alone doesn't reliably catch a chapter with no marker AND no
    preceding "EXERCISES" to anchor on, but chapter numbering resetting is domain knowledge
    we already know for certain. Never promotes "EXERCISES" itself (the one heading that's
    always real and always precedes a new chapter, including a normal, correctly-marked
    one) -- promoting it would misfire on every ordinary chapter transition."""
    resolved = list(resolved)
    last_kept_pos: int | None = None
    for pos, text in enumerate(heading_texts):
        if resolved[pos] == "not_heading":
            continue
        if (
            _FIRST_SUBSECTION_RE.match(text.strip())
            and last_kept_pos is not None
            and resolved[last_kept_pos] == "section"
            and _normalize_heading_text(heading_texts[last_kept_pos]) != "EXERCISES"
        ):
            resolved[last_kept_pos] = "chapter"
        last_kept_pos = pos
    return resolved


def _llm_heading_levels(
    md_text: str, cfg: TypesetConfig, citation_cfg: CitationConfig | None
) -> dict[int, str]:
    """Heading positions (see ``_find_headings``) mapped to the optional LLM
    classification's judged level, or an empty dict when the pass is off or fails -- a
    failure here degrades to a warning, same as the main pipeline's citation stage, since
    ``_apply_heading_levels`` treats a missing entry as "section" (today's baseline), never
    corrupting body content.

    Two sequential calls, not one combined pass: ``classify_real_headings`` first (given
    only each heading's own text + what follows it, no preceding-heading context), then
    ``classify_chapter_level`` only for headings it confirmed real (given the fuller
    preceding+following context). Confirmed live combining these into one call backfires --
    see ``heading_candidates``'s docstring for why revealing "preceded by" during the
    real/not-real judgment tricks the model into treating a whole polluted run of
    table-of-contents lines as real chapters. A final deterministic pass
    (``_promote_orphaned_chapter_subsections``) catches what classification alone still
    misses: a chapter with neither a marker nor a preceding "EXERCISES" to anchor on."""
    if not cfg.clean_toc_with_llm:
        return {}
    try:
        citation_cfg = citation_cfg or CitationConfig()
        client = make_client(citation_cfg)
        candidates = heading_candidates(md_text)
        is_real = classify_real_headings(
            [(text, context) for (_preceding, text, context) in candidates],
            citation_cfg,
            client=client,
        )

        levels: dict[int, str] = {
            pos: "not_heading" for pos, real in enumerate(is_real) if not real
        }
        real_positions = [pos for pos, real in enumerate(is_real) if real]
        if real_positions:
            chapter_levels = classify_chapter_level(
                [candidates[pos] for pos in real_positions], citation_cfg, client=client
            )
            levels.update(zip(real_positions, chapter_levels, strict=True))

        heading_texts = [text for (_preceding, text, _following) in candidates]
        resolved = _promote_orphaned_chapter_subsections(
            heading_texts, [levels[pos] for pos in range(len(candidates))]
        )
        return dict(enumerate(resolved))
    except Exception as exc:
        logger.warning("LLM TOC cleanup failed (%s) -- keeping the flat heading structure", exc)
        return {}


def _convert_footnotes_to_pandoc_notes(md: str) -> str:
    """Rewrite each "> **Footnote:** ..." paragraph (parse.py's RAG/chunking-friendly
    marking -- see its own docstring) into pandoc's footnote syntax, so it typesets as a
    real page-bottom footnote instead of just another quoted paragraph. See
    ``_FOOTNOTE_BLOCKQUOTE_RE``'s comment for why the reference lands on the *preceding*
    paragraph. A footnote with nothing before it (the very first block in the document --
    vanishingly unlikely in practice) is left as its own unreferenced quoted paragraph:
    pandoc drops a footnote *definition* that nothing ever references."""
    blocks = md.split("\n\n")
    out_blocks: list[str] = []
    n = 0
    for block in blocks:
        m = _FOOTNOTE_BLOCKQUOTE_RE.match(block.strip())
        if m and out_blocks:
            n += 1
            out_blocks[-1] = f"{out_blocks[-1]}[^{n}]"
            out_blocks.append(f"[^{n}]: {m.group(1)}")
        else:
            out_blocks.append(block)
    return "\n\n".join(out_blocks)


def _prepare_pandoc_input(md: str, heading_levels: dict[int, str] | None = None) -> str:
    # chunker.py normally strips <page_number> markers when resolving each chunk's page
    # range; typesetting bypasses chunking entirely, so strip them here -- LaTeX paginates
    # on its own, and otherwise each marker leaks into the PDF as a stray "N".
    md = PAGE_MARKER_RE.sub("", md)
    md = _apply_heading_levels(md, heading_levels)
    md = _convert_footnotes_to_pandoc_notes(md)
    md = _FIGURE_CROP_ALT_RE.sub("![]", md)
    return _normalize_math_dollars(md)


def render_pdf(
    md_path: Path,
    out_path: Path,
    cfg: TypesetConfig | None = None,
    *,
    title: str | None = None,
    author: str | None = None,
    citation_cfg: CitationConfig | None = None,
) -> list[str]:
    """Typeset ``md_path`` (plus any images it links to, resolved relative to its own
    directory) into ``out_path``. Returns a summary line per distinct recoverable TeX
    error encountered (empty if the compile was clean).

    Two steps, not pandoc's own ``--pdf-engine`` path: ``pandoc -s`` markdown -> a
    standalone .tex file (deterministic, always succeeds), then xelatex .tex -> .pdf run
    directly, twice (the 2nd pass resolves TOC page numbers from the 1st pass's .toc).
    OCR'd math is occasionally malformed in ways ``_normalize_math_dollars`` can't fully
    repair; almost all of those land in TeX's *recoverable* error class, which nonstopmode
    auto-patches and continues past, still producing a complete PDF. pandoc's own PDF
    pipeline treats any nonzero engine exit code as total failure and discards that PDF, so
    success here is judged by whether a PDF actually came out, not by the exit code.

    ``cfg.clean_toc_with_llm`` (opt-in, off by default -- the whole point of this command
    is not needing an API key) runs extra Gemini calls (see ``_llm_heading_levels``) to
    catch headings the mechanical dedup can't, and to classify chapter-vs-section
    hierarchy so LaTeX page-breaks correctly and the TOC nests properly. A failure here
    (missing key, network) degrades to a loud warning, same as the main pipeline's
    citation stage -- the PDF is still the primary product and must still be written.
    """
    cfg = cfg or TypesetConfig()
    # resolve to absolute up front: every path below is built from md_path/work_dir and
    # passed as a subprocess argv *alongside* cwd=work_dir -- if md_path were relative, the
    # two would be resolved against different bases (argv against our original cwd, cwd=
    # itself against work_dir), doubling up work_dir in the effective lookup path.
    md_path = md_path.resolve()
    out_path = out_path.resolve()
    work_dir = md_path.parent

    lua_path = work_dir / f"_typeset_scale_images_{md_path.stem}.lua"
    lua_path.write_text(_LUA_FILTER_TEMPLATE.format(width=cfg.image_max_width))

    md_text = md_path.read_text()
    heading_levels = _llm_heading_levels(md_text, cfg, citation_cfg)

    pandoc_input = work_dir / f"_typeset_{md_path.name}"
    pandoc_input.write_text(_prepare_pandoc_input(md_text, heading_levels))

    tex_path = pandoc_input.with_suffix(".tex")
    pandoc_cmd = [
        "pandoc",
        str(pandoc_input),
        "-s",
        "-o",
        str(tex_path),
        "--toc",
        f"--toc-depth={cfg.toc_depth}",
        "--lua-filter",
        str(lua_path),
        "-V",
        f"geometry:margin={cfg.margin}",
        "-V",
        f"documentclass={cfg.documentclass}",
        "-V",
        f"fontsize={cfg.font_size}",
        "-V",
        f"linestretch={cfg.line_stretch}",
        "-V",
        "colorlinks=true",
    ]
    if cfg.main_font and _font_available(cfg.main_font):
        pandoc_cmd += ["-V", f"mainfont={cfg.main_font}"]
    elif cfg.main_font:
        logger.warning(
            "TypesetConfig.main_font %r not found by fontconfig -- falling back to "
            "xelatex's own default font",
            cfg.main_font,
        )
    if title:
        pandoc_cmd += ["--metadata", f"title={title}"]
    if author:
        pandoc_cmd += ["--metadata", f"author={author}"]
    subprocess.run(pandoc_cmd, check=True, cwd=work_dir)

    xelatex_cmd = [cfg.pdf_engine, "-interaction=nonstopmode", tex_path.name]
    for _pass in range(2):
        subprocess.run(xelatex_cmd, cwd=work_dir, capture_output=True)

    produced = tex_path.with_suffix(".pdf")
    log_path = tex_path.with_suffix(".log")
    errors: list[str] = []
    if log_path.exists():
        messages = re.findall(r"^! (.+)$", log_path.read_text(errors="replace"), re.M)
        errors = [f"{count}x {msg}" for msg, count in Counter(messages).most_common()]
    if not produced.exists():
        raise RuntimeError(f"{cfg.pdf_engine} produced no PDF -- see {log_path}")
    # not produced.rename(out_path): work_dir and out_path's directory are frequently
    # different filesystems (e.g. work_dir under the project tree, --out in /tmp), and
    # os.rename raises EXDEV across a filesystem boundary -- shutil.move falls back to
    # copy+delete automatically when a plain rename isn't possible.
    shutil.move(str(produced), str(out_path))

    # only the final PDF is worth keeping around; xelatex's working files are cleaned up
    # now that they've served their purpose (a failed compile above skips this, leaving
    # the .log/.tex in place to debug).
    for stray in work_dir.glob(f"{tex_path.stem}.*"):
        stray.unlink(missing_ok=True)
    lua_path.unlink(missing_ok=True)

    return errors
