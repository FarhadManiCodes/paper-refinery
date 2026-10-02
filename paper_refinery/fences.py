"""Code-fence handling for algorithm regions, shared by parse.py and cli.py.

An ``algorithm`` region is emitted as a fenced block. The OCR can already return such a
region fenced (a code listing's own python block), and older versions wrapped it a second
time. CommonMark pairs fences by parity, so one doubled block makes every later fence in
the document pair with the wrong partner: prose and math between code blocks become
"code" for anything fence-aware (confirmed live on a 1.35 MB book: 38 vs 5,153 math spans).

``fence_code`` is the exact fix, applied where the region boundaries are known.
``repair_doubled_fences`` repairs markdown already written with the old double wrapper
(parse checkpoints from earlier versions keep that markdown) without re-running OCR.
"""

from __future__ import annotations

import re

_FENCE_LINE_RE = re.compile(r"^\s*(`{3,}|~{3,})\s*([^`\s]*)\s*$")
_BACKTICK_RUN_RE = re.compile(r"`+")
_BARE_FENCE = "```"


def fence_code(content: str) -> str:
    """Wrap an algorithm/code region in exactly one balanced fence.

    An outer fence the content already carries is unwrapped first (info string kept, a
    missing closing fence tolerated), then the text is re-fenced with a backtick run
    longer than any inside it, so an embedded fence pair stays literal text.
    """
    lines = content.split("\n")
    info = ""
    opener = _FENCE_LINE_RE.match(lines[0]) if lines else None
    if opener:
        info = opener.group(2)
        body = lines[1:]
        closer = _FENCE_LINE_RE.match(body[-1]) if body else None
        if (
            closer
            and not closer.group(2)
            and closer.group(1)[0] == opener.group(1)[0]
            and len(closer.group(1)) >= len(opener.group(1))
        ):
            body = body[:-1]
        lines = body
    longest = max((len(m.group()) for m in _BACKTICK_RUN_RE.finditer("\n".join(lines))), default=0)
    fence = "`" * max(3, longest + 1)
    return "\n".join([f"{fence}{info}", *lines, fence])


def _is_fence_line(line: str) -> bool:
    return _FENCE_LINE_RE.match(line) is not None


def _closes(opener: re.Match[str], line: str) -> bool:
    """Whether ``line`` closes the fence ``opener`` matched: no info string, the same
    fence character, and at least as many of it (the CommonMark closing rule)."""
    closer = _FENCE_LINE_RE.match(line)
    return bool(
        closer
        and not closer.group(2)
        and closer.group(1)[0] == opener.group(1)[0]
        and len(closer.group(1)) >= len(opener.group(1))
    )


def _skip_clean_fence(lines: list[str], start: int) -> int:
    """Index just past the fenced block opened at ``lines[start]`` (to the end when the
    block never closes)."""
    opener = _FENCE_LINE_RE.match(lines[start])
    assert opener is not None
    for j in range(start + 1, len(lines)):
        if _closes(opener, lines[j]):
            return j + 1
    return len(lines)


def _old_wrapper_end(lines: list[str], start: int) -> int | None:
    """For a bare fence at ``lines[start]`` that the old wrapper may have emitted, the
    index of its outer closing fence; None when the block never closes.

    The old wrapper put exactly one bare fence before and after each region, so the outer
    closer is the first bare three-backtick line that is not inside an inner fence. An inner
    fence is a fence line directly after the outer opener, or any fence line met later; it
    ends at its own closer (same character, at least as long), so tilde and longer-backtick
    inner fences are tracked by their own opener rather than by the outer wrapper's shape.
    """
    inner: re.Match[str] | None = None
    j = start + 1
    if j < len(lines) and (inner := _FENCE_LINE_RE.match(lines[j])):
        j += 1
    while j < len(lines):
        line = lines[j]
        if inner is None:
            if line == _BARE_FENCE:
                return j
            inner = _FENCE_LINE_RE.match(line)
        elif _closes(inner, line):
            inner = None
        j += 1
    return None


def repair_doubled_fences(markdown: str) -> str:
    """Collapse algorithm regions the old wrapper fenced twice into one balanced fence.

    Idempotent, and a no-op on correctly fenced text: a block is only rewritten when the
    region content between the old wrapper's fences itself contains a fence. Fenced blocks
    with a language or a longer-than-three fence are skipped as already-clean output of
    ``fence_code``; a plain three-backtick block holding no inner fence is left as is.
    """
    lines = markdown.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line != _BARE_FENCE:
            if _is_fence_line(line):
                end = _skip_clean_fence(lines, i)
                out.extend(lines[i:end])
                i = end
            else:
                out.append(line)
                i += 1
            continue
        end = _old_wrapper_end(lines, i)
        if end is None:
            out.extend(lines[i:])
            break
        content = lines[i + 1 : end]
        if any(_is_fence_line(inner) for inner in content):
            out.extend(fence_code("\n".join(content).strip()).split("\n"))
        else:
            out.extend(lines[i : end + 1])
        i = end + 1
    return "\n".join(out)
