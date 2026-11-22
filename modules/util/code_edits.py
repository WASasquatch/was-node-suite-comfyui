"""Applying a language model's code reply: whole files and SEARCH/REPLACE blocks.

A bundle is each file's path alone on a line, then a fenced block holding the file; one
unnamed file is bare text. Newlines are ``\\n``.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

__all__ = [
    "MAX_CHARS",
    "MAX_FILES",
    "MAX_EDITS",
    "SourceFile",
    "Edit",
    "Outcome",
    "Result",
    "parse_bundle",
    "format_bundle",
    "parse_reply",
    "apply_reply",
    "report_text",
]

#: Characters the code and the reply may each hold.
MAX_CHARS = 2_000_000

#: Files a bundle may hold.
MAX_FILES = 200

#: Edits one reply may carry.
MAX_EDITS = 500

#: Lines of a file searched for the closest match to a block that did not apply.
MAX_HINT_LINES = 20_000

#: Candidate starting lines compared when looking for the closest match.
MAX_HINT_STARTS = 2_000

#: Lines of the closest match quoted in a report.
HINT_LINES = 6

_FENCE = re.compile(r"^[ \t]*(`{3,}|~{3,})[ \t]*([^\s`]*)[^`]*$")
_SEARCH = re.compile(r"^[ \t]*<{5,9}[ \t]*SEARCH\b.*$")
_DIVIDER = re.compile(r"^[ \t]*={5,9}[ \t]*$")
_REPLACE = re.compile(r"^[ \t]*>{5,9}[ \t]*REPLACE\b.*$")
_PATH_JUNK = " \t*`'\"#:"
_PATH_PREFIX = re.compile(r"^(?:file(?:name)?|path)\s*:\s*", re.IGNORECASE)
_PATH = re.compile(r"^[\w./\\@+-]+$")
_EXTENSION = re.compile(r"^[^.][^/\\]*\.[A-Za-z0-9]{1,10}$|^\.[\w.-]+$")
_BARE_NAMES = {"makefile", "dockerfile", "license", "readme", "procfile", "gemfile", "rakefile"}

#: Fence language by file extension, for a file the reply gave no language.
LANGUAGES = {
    ".py": "python", ".js": "javascript", ".mjs": "javascript", ".ts": "typescript",
    ".tsx": "tsx", ".jsx": "jsx", ".json": "json", ".html": "html", ".css": "css",
    ".md": "markdown", ".sh": "bash", ".ps1": "powershell", ".rs": "rust", ".go": "go",
    ".java": "java", ".c": "c", ".h": "c", ".cpp": "cpp", ".hpp": "cpp", ".cs": "csharp",
    ".rb": "ruby", ".php": "php", ".lua": "lua", ".sql": "sql", ".yaml": "yaml",
    ".yml": "yaml", ".toml": "toml", ".xml": "xml", ".kt": "kotlin", ".swift": "swift",
}


@dataclass
class SourceFile:
    """One file of a bundle.

    Attributes:
        path: The path as written, or ``""`` for the one unnamed file.
        text: The file's text.
        language: The fence's language tag, or ``""``.
    """

    path: str
    text: str
    language: str = ""


@dataclass
class Edit:
    """One change a reply asks for.

    Attributes:
        number: Its place in the reply, from 1.
        kind: ``"replace"`` for a SEARCH/REPLACE block, ``"whole"`` for a whole file,
            ``"broken"`` for a block missing its divider or closing marker.
        path: The file it names, or ``""`` where it names none.
        search: The SEARCH lines, empty for a whole file or an insertion.
        replace: The REPLACE lines, or the whole file's text.
        language: The fence's language tag, or ``""``.
    """

    number: int
    kind: str
    path: str
    search: str = ""
    replace: str = ""
    language: str = ""


@dataclass
class Outcome:
    """What became of one edit.

    Attributes:
        edit: The edit.
        applied: Whether it changed the code.
        how: How it matched, such as ``exact`` or ``indentation``, or why it did not.
        path: The file it landed in.
        line: The first line it changed, from 1, or 0.
        hint: The closest lines found for a block that did not apply, or ``""``.
    """

    edit: Edit
    applied: bool
    how: str
    path: str = ""
    line: int = 0
    hint: str = ""


@dataclass
class Result:
    """A reply applied to a bundle.

    Attributes:
        files: The bundle after the edits.
        before: The bundle as it came.
        outcomes: One per edit, in reply order.
        notes: Things read off the reply that were not edits, such as an unnamed code block left out.
        diff: A unified diff of every changed file.
    """

    files: list[SourceFile]
    before: list[SourceFile]
    outcomes: list[Outcome] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    diff: str = ""

    @property
    def applied(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.applied)

    @property
    def failed(self) -> int:
        return sum(1 for outcome in self.outcomes if not outcome.applied)


def _limit(text: str, name: str) -> str:
    text = "" if text is None else str(text)
    if len(text) > MAX_CHARS:
        raise ValueError(
            f"The {name} holds {len(text):,} characters; at most {MAX_CHARS:,} are read. "
            f"Split the work into smaller files."
        )
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _path_of(line: str) -> str:
    """The file path a line names, or ``""`` when it does not look like one."""
    text = _PATH_PREFIX.sub("", line.strip().strip(_PATH_JUNK).strip())
    text = text.strip(_PATH_JUNK)
    if not text or len(text) > 240 or not _PATH.match(text):
        return ""
    name = text.replace("\\", "/").rsplit("/", 1)[-1]
    if _EXTENSION.search(name) or name.lower() in _BARE_NAMES:
        return text
    return ""


def _language_for(path: str, given: str = "") -> str:
    if given:
        return given
    dot = path.rfind(".")
    return LANGUAGES.get(path[dot:].lower(), "") if dot >= 0 else ""


def _fences(lines: list[str]) -> list[tuple[int, int, str]]:
    """Every fenced block as ``(open line, close line, language)``; an unclosed one runs to the end."""
    found = []
    index = 0
    while index < len(lines):
        opened = _FENCE.match(lines[index])
        if not opened:
            index += 1
            continue
        mark = opened.group(1)
        close = index + 1
        while close < len(lines):
            stripped = lines[close].strip()
            if stripped and set(stripped) == {mark[0]} and len(stripped) >= len(mark):
                break
            close += 1
        found.append((index, close, opened.group(2)))
        index = close + 1
    return found


def _label_before(lines: list[str], index: int) -> str:
    """The path named on the nearest non-blank line above ``index``."""
    above = index - 1
    while above >= 0 and not lines[above].strip():
        above -= 1
    return _path_of(lines[above]) if above >= 0 else ""


def parse_bundle(text: str) -> list[SourceFile]:
    """Read a bundle into its files.

    Args:
        text: A bundle, or bare code for one unnamed file.

    Returns:
        The files in order. Empty text is an empty bundle.

    Raises:
        ValueError: The text is over :data:`MAX_CHARS` or names more than :data:`MAX_FILES` files.
    """
    text = _limit(text, "code")
    if not text.strip():
        return []
    lines = text.split("\n")
    fences = _fences(lines)
    named = [(start, end, lang, _label_before(lines, start)) for start, end, lang in fences]
    if not any(path for *_, path in named):
        if len(fences) == 1 and not "".join(lines[: fences[0][0]] + lines[fences[0][1] + 1:]).strip():
            start, end, lang = fences[0]
            return [SourceFile("", "\n".join(lines[start + 1:end]), lang)]
        return [SourceFile("", text.rstrip("\n"))]
    files: list[SourceFile] = []
    for start, end, lang, path in named:
        if not path:
            continue
        body = "\n".join(lines[start + 1:end])
        existing = next((item for item in files if item.path == path), None)
        if existing:
            existing.text, existing.language = body, lang or existing.language
        else:
            files.append(SourceFile(path, body, _language_for(path, lang)))
        if len(files) > MAX_FILES:
            raise ValueError(f"The code names more than {MAX_FILES} files; at most {MAX_FILES} are read.")
    return files


def format_bundle(files: list[SourceFile]) -> str:
    """Write files back as a bundle.

    Args:
        files: The files, in order.

    Returns:
        The bundle text: one unnamed file as its bare text, otherwise each path then its fence.
    """
    if not files:
        return ""
    if len(files) == 1 and not files[0].path:
        return files[0].text
    parts = []
    for item in files:
        longest = max((len(run) for run in re.findall(r"`{3,}", item.text)), default=2)
        mark = "`" * max(3, longest + 1)
        head = f"{item.path}\n" if item.path else ""
        parts.append(f"{head}{mark}{item.language}\n{item.text}\n{mark}")
    return "\n\n".join(parts)


def _path_above(lines: list[str], index: int, fallback: tuple[str, str]) -> tuple[str, str]:
    """The path and fence language above a SEARCH line, past blank and fence lines.

    Returns:
        ``(path, language)``, or ``fallback`` where no path stands above the block.
    """
    above = index - 1
    language = ""
    while above >= 0 and (not lines[above].strip() or _FENCE.match(lines[above])):
        opened = _FENCE.match(lines[above])
        if opened and opened.group(2) and not language:
            language = opened.group(2)
        above -= 1
    named = _path_of(lines[above]) if above >= 0 else ""
    if named:
        return named, language
    return fallback[0], language or fallback[1]


def _marker_blocks(lines: list[str]) -> list[tuple[int, int, Edit]]:
    """Every SEARCH/REPLACE block, found by its markers wherever code fences fall.

    Returns:
        ``(first line, last line, edit)`` for each block. A block missing its ``=======`` or
        ``>>>>>>> REPLACE`` line is an edit of kind ``"broken"``.
    """
    found = []
    index = 0
    last = ("", "")
    while index < len(lines):
        if not _SEARCH.match(lines[index]):
            index += 1
            continue
        last = _path_above(lines, index, last)
        divider = index + 1
        while divider < len(lines) and not (_DIVIDER.match(lines[divider]) or _SEARCH.match(lines[divider])):
            divider += 1
        closing = divider + 1
        if divider < len(lines) and _DIVIDER.match(lines[divider]):
            while closing < len(lines) and not (_REPLACE.match(lines[closing]) or _SEARCH.match(lines[closing])):
                closing += 1
        if divider >= len(lines) or closing >= len(lines) or not _REPLACE.match(lines[closing]):
            # An unfinished block runs to the next fence line, or the next block.
            stop = index + 1
            while stop < len(lines) and not (_FENCE.match(lines[stop]) or _SEARCH.match(lines[stop])):
                stop += 1
            stop = stop if stop < len(lines) and _FENCE.match(lines[stop]) else stop - 1
            found.append((index, stop, Edit(0, "broken", last[0], "\n".join(lines[index + 1:stop]))))
            index = stop + 1
            continue
        found.append((index, closing, Edit(
            0, "replace", last[0],
            search="\n".join(lines[index + 1:divider]),
            replace="\n".join(lines[divider + 1:closing]),
            language=last[1],
        )))
        if len(found) > MAX_EDITS:
            raise ValueError(f"The reply carries more than {MAX_EDITS} edits; at most {MAX_EDITS} are read.")
        index = closing + 1
    return found


def parse_reply(text: str) -> tuple[list[Edit], list[str]]:
    """Read the edits out of a model's reply.

    Args:
        text: The reply: SEARCH/REPLACE blocks, whole files under their paths, or both,
            with any prose around them.

    Returns:
        The edits in reply order, and notes on code blocks that were read as something else.

    Raises:
        ValueError: The reply is over :data:`MAX_CHARS` or carries more than :data:`MAX_EDITS` edits.
    """
    text = _limit(text, "reply")
    lines = text.split("\n")
    notes: list[str] = []
    placed: list[tuple[int, Edit]] = []
    masked = list(lines)
    for first, last, edit in _marker_blocks(lines):
        placed.append((first, edit))
        for index in range(first, min(last, len(lines) - 1) + 1):
            masked[index] = ""
    stray = 0
    unnamed: list[tuple[int, str, str]] = []
    for start, end, lang in _fences(masked):
        body_lines = masked[start + 1:end]
        if not "".join(body_lines).strip():
            continue
        if any(_DIVIDER.match(line) or _REPLACE.match(line) for line in body_lines):
            stray += 1
            continue
        body = "\n".join(body_lines)
        path = _label_before(masked, start)
        if path:
            placed.append((start, Edit(0, "whole", path, replace=body, language=lang)))
        else:
            unnamed.append((start + 1, body, lang))
    if stray:
        notes.append(
            f"{stray} code block{'s' if stray != 1 else ''} holding a stray ======= or >>>>>>> REPLACE "
            f"line {'were' if stray != 1 else 'was'} left out"
        )
    edits = [edit for _, edit in sorted(placed, key=lambda pair: pair[0])]
    if unnamed:
        if edits:
            notes.append(
                f"{len(unnamed)} code block{'s' if len(unnamed) != 1 else ''} with no file path "
                f"above {'them were' if len(unnamed) != 1 else 'it was'} left out"
            )
        else:
            line, body, lang = max(unnamed, key=lambda item: len(item[1]))
            edits.append(Edit(1, "whole", "", replace=body, language=lang))
            if len(unnamed) > 1:
                notes.append(
                    f"no code block named a file, so the longest of {len(unnamed)}, at reply "
                    f"line {line}, was read as the whole file"
                )
    for number, edit in enumerate(edits, 1):
        edit.number = number
    return edits, notes


def _indent(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _find(haystack: list[str], needle: list[str], key) -> int:
    """The first index where ``needle`` matches under ``key``, or -1."""
    wanted = [key(line) for line in needle]
    size = len(wanted)
    first = wanted[0]
    for index in range(len(haystack) - size + 1):
        if key(haystack[index]) != first:
            continue
        if all(key(haystack[index + offset]) == wanted[offset] for offset in range(1, size)):
            return index
    return -1


def _count(haystack: list[str], needle: list[str]) -> int:
    size = len(needle)
    return sum(
        1 for index in range(len(haystack) - size + 1) if haystack[index:index + size] == needle
    )


def _reindent(found: list[str], search: list[str], replace: list[str]) -> list[str]:
    """Give the replacement the file's indentation wherever the SEARCH lines used another."""
    mapping: dict[str, str] = {}
    for have, asked in zip(found, search):
        if have.strip():
            mapping.setdefault(_indent(asked), _indent(have))
    if all(asked == have for asked, have in mapping.items()):
        return replace
    keys = sorted(mapping, key=len, reverse=True)
    out = []
    for line in replace:
        if not line.strip():
            out.append(line)
            continue
        indent = _indent(line)
        key = indent if indent in mapping else next(
            (candidate for candidate in keys if candidate and indent.startswith(candidate)), None,
        )
        out.append(line if key is None else mapping[key] + indent[len(key):] + line.lstrip())
    return out


def _closest(lines: list[str], search: list[str]) -> tuple[int, float]:
    """The line, from 0, where the window most like ``search`` starts, and how alike it is."""
    if not lines or not search:
        return -1, 0.0
    lines = lines[:MAX_HINT_LINES]
    size = len(search)
    target = "\n".join(line.strip() for line in search)
    anchors = {line.strip() for line in search if line.strip()}
    starts = [
        index for index, line in enumerate(lines)
        if line.strip() and line.strip() in anchors
    ][:MAX_HINT_STARTS] or range(0, max(1, len(lines) - size + 1), max(1, len(lines) // 400))
    best, score = -1, 0.0
    matcher = difflib.SequenceMatcher(autojunk=False)
    matcher.set_seq2(target)
    for start in starts:
        for begin in (start, start - search.index(next(l for l in search if l.strip()))):
            if begin < 0 or begin >= len(lines):
                continue
            matcher.set_seq1("\n".join(line.strip() for line in lines[begin:begin + size]))
            ratio = matcher.ratio()
            if ratio > score:
                best, score = begin, ratio
    return best, score


def _trim_fences(lines: list[str]) -> list[str]:
    """The lines without code fence lines at either end."""
    start, end = 0, len(lines)
    while start < end and (_FENCE.match(lines[start]) or not lines[start].strip()):
        start += 1
    while end > start and (_FENCE.match(lines[end - 1]) or not lines[end - 1].strip()):
        end -= 1
    return lines[start:end]


def _apply_block(source: SourceFile, edit: Edit) -> Outcome:
    """Apply one SEARCH/REPLACE block to a file in place."""
    lines = source.text.split("\n") if source.text else []
    search = edit.search.split("\n") if edit.search else []
    replace = edit.replace.split("\n") if edit.replace else []
    while search and not search[-1].strip() and len(search) > 1:
        search.pop()
    if not search or not any(line.strip() for line in search):
        if not lines:
            source.text = "\n".join(replace)
            return Outcome(edit, True, "new file", source.path, 1)
        source.text = "\n".join(lines + replace)
        return Outcome(edit, True, "appended", source.path, len(lines) + 1)
    attempts = [(search, replace, "")]
    trimmed = _trim_fences(search)
    if trimmed != search and any(line.strip() for line in trimmed):
        attempts.append((trimmed, _trim_fences(replace), ", fence lines dropped"))
    for wanted, body, note in attempts:
        for how, key in (
            ("exact", lambda line: line),
            ("trailing whitespace", lambda line: line.rstrip()),
            ("indentation", lambda line: line.strip()),
        ):
            index = _find(lines, wanted, key)
            if index < 0:
                continue
            found = lines[index:index + len(wanted)]
            new = _reindent(found, wanted, body) if how == "indentation" else body
            source.text = "\n".join(lines[:index] + new + lines[index + len(wanted):])
            if how == "exact" and _count(lines, wanted) > 1:
                how = f"exact, first of {_count(lines, wanted)} matches"
            return Outcome(edit, True, how + note, source.path, index + 1)
    start, score = _closest(lines, search)
    hint = ""
    if start >= 0 and score >= 0.5:
        shown = lines[start:start + min(len(search), HINT_LINES)]
        hint = (
            f"closest lines {start + 1} to {start + len(shown)} ({score:.0%} alike):\n"
            + "\n".join(shown)
        )
    return Outcome(edit, False, "SEARCH lines not found", source.path, start + 1 if hint else 0, hint)


def _target(files: list[SourceFile], path: str) -> SourceFile | None:
    """The file an edit lands in: by path, or the only file where it names none or a new one."""
    if path:
        match = next((item for item in files if item.path == path), None)
        if match:
            return match
        tail = path.rsplit("/", 1)[-1]
        ends = [item for item in files if item.path and item.path.rsplit("/", 1)[-1] == tail]
        if len(ends) == 1:
            return ends[0]
    if len(files) == 1 and (not path or not files[0].path):
        return files[0]
    return None


def _diff(before: list[SourceFile], after: list[SourceFile]) -> str:
    old = {item.path: item.text for item in before}
    chunks = []
    for item in after:
        previous = old.get(item.path)
        if previous == item.text:
            continue
        name = item.path or "code"
        chunks.extend(difflib.unified_diff(
            (previous or "").split("\n") if previous else [],
            item.text.split("\n"),
            fromfile=f"a/{name}" if previous is not None else "/dev/null",
            tofile=f"b/{name}", lineterm="",
        ))
    return "\n".join(chunks)


def apply_reply(code: str, reply: str) -> Result:
    """Apply every edit in a reply to a bundle.

    Args:
        code: The bundle as it stands, or ``""`` for none yet.
        reply: The model's reply.

    Returns:
        The edited bundle, one outcome per edit, notes and the diff.

    Raises:
        ValueError: The code or reply is over a limit.
    """
    before = parse_bundle(code)
    files = [SourceFile(item.path, item.text, item.language) for item in before]
    edits, notes = parse_reply(reply)
    result = Result(files=files, before=before, notes=notes)
    for edit in edits:
        if edit.kind == "broken":
            result.outcomes.append(Outcome(
                edit, False, "unfinished block, with no ======= or no >>>>>>> REPLACE line", edit.path,
            ))
            continue
        target = _target(files, edit.path)
        if edit.kind == "whole":
            if target is None:
                if len(files) >= MAX_FILES:
                    result.outcomes.append(Outcome(edit, False, f"more than {MAX_FILES} files", edit.path))
                    continue
                files.append(SourceFile(edit.path, edit.replace, _language_for(edit.path, edit.language)))
                result.outcomes.append(Outcome(edit, True, "new file", edit.path, 1))
                continue
            if edit.path and not target.path:
                target.path = edit.path
            target.text = edit.replace
            target.language = _language_for(target.path, edit.language or target.language)
            result.outcomes.append(Outcome(edit, True, "whole file", target.path, 1))
            continue
        if target is None and edit.search.strip() and len(files) == 1:
            # A block naming the wrong file still lands in the only file there is.
            target = files[0]
        if target is None:
            if edit.path and not edit.search.strip():
                files.append(SourceFile(edit.path, edit.replace, _language_for(edit.path, edit.language)))
                result.outcomes.append(Outcome(edit, True, "new file", edit.path, 1))
                continue
            named = edit.path or "no file"
            result.outcomes.append(Outcome(
                edit, False, f"names {named}, which the code does not hold", edit.path,
            ))
            continue
        if edit.path and not target.path:
            target.path = edit.path
            target.language = _language_for(edit.path, edit.language)
        result.outcomes.append(_apply_block(target, edit))
    result.diff = _diff(before, files)
    return result


def report_text(result: Result) -> str:
    """The outcome as text a person or a model can act on.

    Args:
        result: What :func:`apply_reply` answered.

    Returns:
        One summary line, then a paragraph for each edit that did not apply, then the notes.
    """
    total = len(result.outcomes)
    touched = len({outcome.path for outcome in result.outcomes if outcome.applied})
    if not total:
        lines = ["No edits found in the reply: no SEARCH/REPLACE block and no code block."]
    else:
        lines = [
            f"Applied {result.applied} of {total} edit{'s' if total != 1 else ''}"
            f" to {touched} file{'s' if touched != 1 else ''}."
        ]
    for outcome in result.outcomes:
        if outcome.applied:
            continue
        edit = outcome.edit
        where = f" in {outcome.path}" if outcome.path else ""
        lines.append("")
        lines.append(f"Edit {edit.number}, SEARCH/REPLACE{where}: {outcome.how}.")
        first = next((line for line in edit.search.split("\n") if line.strip()), "")
        if first:
            lines.append(f"Its SEARCH began: {first.strip()}")
        if outcome.hint:
            lines.append(outcome.hint)
    for note in result.notes:
        lines.append("")
        lines.append(f"Note: {note}.")
    return "\n".join(lines)
