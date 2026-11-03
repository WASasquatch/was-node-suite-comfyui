"""Splitting a prompt into tags, and tidying the result.

:func:`strip_emphasis` reduces ``(tag:1.4)``, ``((tag))`` and ``[tag]`` to ``tag``, so two
spellings of one tag compare equal. :func:`clean` runs split, replace, remove, dedupe, sort
and limit in that order.
"""

from __future__ import annotations

import re

__all__ = [
    "SORTS",
    "WILDCARD",
    "clean",
    "glob_match",
    "parse_removals",
    "parse_replacements",
    "sort_tags",
    "split_tags",
    "strip_emphasis",
    "tag_key",
]

#: ``(tag:1.4)``, the A1111 weighted form, which is also what ComfyUI's own encoder reads.
_WEIGHTED = re.compile(r"^\(\s*(?P<body>.+?)\s*:\s*-?\d+(?:\.\d+)?\s*\)$", re.DOTALL)

#: ``(tag)`` and ``[tag]``, the repeat-to-emphasise forms, one nesting level at a time.
_BRACKETED = re.compile(r"^\(\s*(?P<body>.+?)\s*\)$|^\[\s*(?P<alt>.+?)\s*\]$", re.DOTALL)

#: Any run of whitespace, including the line breaks a multi-line prompt box produces.
_WHITESPACE = re.compile(r"\s+")

#: How many layers of emphasis are unwrapped before the tag is taken as it is.
_MAX_UNWRAP = 8

#: Stands for any run of characters, none included, in a tag to remove.
WILDCARD = "*"

#: What separates the two sides of a replace rule.
_ARROW = "->"

#: Sort name -> ``(key function, reverse)``. ``none`` is handled before this is consulted.
SORTS = {
    "a-z": (lambda tag: strip_emphasis(tag).casefold(), False),
    "z-a": (lambda tag: strip_emphasis(tag).casefold(), True),
    "shortest first": (len, False),
    "longest first": (len, True),
}


def _body_span(tag: str) -> tuple[int, int]:
    """Where a tag's words sit inside it, once whole-tag emphasis is unwrapped.

    Args:
        tag: One tag, already split out of the prompt.

    Returns:
        ``(start, end)`` indices into ``tag``; ``tag[start:end]`` is :func:`strip_emphasis`.
    """
    start = len(tag) - len(tag.lstrip())
    end = max(start, len(tag.rstrip()))
    for _ in range(_MAX_UNWRAP):
        body = tag[start:end]
        found = _WEIGHTED.match(body)
        name = "body"
        if found is None:
            found = _BRACKETED.match(body)
            if found is None:
                break
            name = "body" if found.group("body") is not None else "alt"
        inner = found.group(name)
        first = start + found.start(name)
        start = first + len(inner) - len(inner.lstrip())
        end = max(start, first + len(inner.rstrip()))
    return start, end


def strip_emphasis(tag: str) -> str:
    """Reduce a tag to its words, with emphasis markup removed.

    Args:
        tag: One tag, already split out of the prompt.

    Returns:
        The tag with whole-tag emphasis removed and surrounding space trimmed.
    """
    start, end = _body_span(tag)
    return tag[start:end]


def tag_key(tag: str, ignore_case: bool = True, ignore_emphasis: bool = True) -> str:
    """The value two tags are compared on by dedupe, remove and replace.

    Args:
        tag: One tag.
        ignore_case: Whether ``Neon Glow`` and ``neon glow`` count as the same tag.
        ignore_emphasis: Whether ``(neon glow:1.4)`` counts as the same tag as
            ``neon glow``.

    Returns:
        The comparison key, its internal whitespace collapsed to single spaces.
    """
    body = strip_emphasis(tag) if ignore_emphasis else tag.strip()
    body = _WHITESPACE.sub(" ", body)
    return body.casefold() if ignore_case else body


def split_tags(text: str, delimiter: str = ",") -> list[str]:
    """Split a prompt into tags.

    Args:
        text: The prompt.
        delimiter: What separates one tag from the next. An empty delimiter splits on
            whitespace, which turns the prompt into its words.

    Returns:
        The tags, in order, each trimmed of surrounding whitespace. Empty entries are
        kept; :func:`clean` decides whether an empty tag is dropped.
    """
    if not delimiter:
        return [part for part in _WHITESPACE.split(text.strip()) if part]
    return [part.strip() for part in text.split(delimiter)]


def sort_tags(tags: list[str], order: str) -> list[str]:
    """Order a list of tags.

    Args:
        tags: The tags.
        order: ``none`` to keep the order they arrived in, or a key of :data:`SORTS`. An
            unknown name keeps the order.

    Returns:
        The tags in the requested order, sorted by their words with emphasis removed.
    """
    entry = SORTS.get(order)
    if entry is None:
        return list(tags)
    key, reverse = entry
    return sorted(tags, key=key, reverse=reverse)


def glob_match(text: str, pattern: str) -> bool:
    """Whether a whole text matches a pattern in which ``*`` stands for any run of characters.

    Args:
        text: The text tested.
        pattern: The pattern, as ``*background`` or ``red * dress``.

    Returns:
        True when the pattern covers the whole text.
    """
    pieces = pattern.split(WILDCARD)
    if len(pieces) == 1:
        return text == pattern
    head, tail = pieces[0], pieces[-1]
    if len(text) < len(head) + len(tail):
        return False
    if not text.startswith(head) or not text.endswith(tail):
        return False
    position, stop = len(head), len(text) - len(tail)
    for piece in pieces[1:-1]:
        found = text.find(piece, position, stop)
        if found < 0:
            return False
        position = found + len(piece)
    return True


def parse_removals(remove: str, ignore_case: bool = True, ignore_emphasis: bool = True) -> list[str]:
    """Read the tags to remove into comparison keys.

    Args:
        remove: Tags separated by commas or line breaks, as ``watermark, *background``.
        ignore_case: Whether a case difference still matches.
        ignore_emphasis: Whether an emphasis difference still matches.

    Returns:
        One key per tag named, in order, each as :func:`tag_key` builds it.
    """
    keys = []
    for line in (remove or "").splitlines():
        for part in line.split(","):
            key = tag_key(part, ignore_case, ignore_emphasis)
            if key and key not in keys:
                keys.append(key)
    return keys


def parse_replacements(
    replace: str, ignore_case: bool = True, ignore_emphasis: bool = True
) -> dict[str, str]:
    """Read ``old tag -> new tag`` lines into a table keyed on the old tag.

    Args:
        replace: One rule per line. Blank lines are skipped.
        ignore_case: Whether a case difference still matches.
        ignore_emphasis: Whether an emphasis difference still matches.

    Returns:
        :func:`tag_key` of each old tag -> its new text, trimmed. The first rule for a tag
        wins.

    Raises:
        ValueError: A line has no ``->``, or nothing before it.
    """
    rules: dict[str, str] = {}
    for number, line in enumerate((replace or "").splitlines(), start=1):
        if not line.strip():
            continue
        old, arrow, new = line.partition(_ARROW)
        key = tag_key(old, ignore_case, ignore_emphasis)
        if not arrow or not key:
            missing = "no ->" if not arrow else "no tag before ->"
            raise ValueError(
                f"Prompt Tag Cleanup's replace line {number} ('{line.strip()[:80]}') has "
                f"{missing}. Write one rule per line as `old tag -> new tag`, such as "
                f"`woman -> [trigger]`, or `watermark ->` to drop a tag."
            )
        rules.setdefault(key, new.strip())
    return rules


def _replaced(
    tag: str, new: str, delimiter: str, ignore_emphasis: bool
) -> list[str]:
    """The tags one replace rule turns a tag into.

    Args:
        tag: The tag the rule matched.
        new: The rule's right side.
        delimiter: What separates tags, used to split a right side naming several.
        ignore_emphasis: Whether the tag's own emphasis is kept around each new tag.

    Returns:
        The new tags, none empty. An empty list drops the tag.
    """
    pieces = [piece for piece in split_tags(new, delimiter) if piece]
    if not ignore_emphasis:
        return pieces
    start, end = _body_span(tag)
    return [tag[:start] + piece + tag[end:] for piece in pieces]


def clean(
    text: str,
    delimiter: str = ",",
    dedupe: bool = True,
    ignore_case: bool = True,
    ignore_emphasis: bool = True,
    keep: str = "first",
    remove_empty: bool = True,
    collapse_whitespace: bool = True,
    order: str = "none",
    limit: int = 0,
    remove: str = "",
    replace: str = "",
    keep_first: int = 0,
) -> tuple[list[str], int]:
    """Split a prompt into tags and tidy them: replace, remove, dedupe, sort, limit.

    Args:
        text: The prompt.
        delimiter: What separates one tag from the next. Empty splits on whitespace.
        dedupe: Whether repeated tags are reduced to one.
        ignore_case: Whether a case difference still counts as the same tag.
        ignore_emphasis: Whether an emphasis difference still counts as the same tag.
        keep: ``first`` or ``last``, which of a set of duplicates survives. Position is
            the earliest occurrence either way.
        remove_empty: Whether tags holding nothing are dropped.
        collapse_whitespace: Whether runs of whitespace inside a tag become one space.
        order: ``none`` or a key of :data:`SORTS`.
        limit: Keep at most this many tags, counted last. 0 keeps all.
        remove: Tags to drop, comma separated, ``*`` standing for any run of characters.
        replace: ``old tag -> new tag`` rules, one per line. An empty right side drops the
            tag; a right side holding ``delimiter`` becomes several tags.
        keep_first: How many leading tags stay in place, unsorted and never cut by ``limit``.

    Returns:
        ``(tags, removed)``: the surviving tags in final order, and how many entries the
        tidy-up took out.

    Raises:
        ValueError: A replace line has no ``->`` or no tag before it.
    """
    tags = split_tags(text, delimiter)
    original = len(tags)
    added = 0

    if collapse_whitespace:
        tags = [_WHITESPACE.sub(" ", tag).strip() for tag in tags]

    rules = parse_replacements(replace, ignore_case, ignore_emphasis)
    if rules:
        rewritten = []
        for tag in tags:
            new = rules.get(tag_key(tag, ignore_case, ignore_emphasis))
            if new is None:
                rewritten.append(tag)
                continue
            pieces = _replaced(tag, new, delimiter, ignore_emphasis)
            added += max(0, len(pieces) - 1)
            rewritten.extend(pieces)
        tags = rewritten

    if remove_empty:
        tags = [tag for tag in tags if tag and strip_emphasis(tag)]

    removals = parse_removals(remove, ignore_case, ignore_emphasis)
    if removals:
        tags = [
            tag for tag in tags
            if not any(
                glob_match(tag_key(tag, ignore_case, ignore_emphasis), pattern)
                for pattern in removals
            )
        ]

    if dedupe:
        position: dict[str, int] = {}
        chosen: dict[str, str] = {}
        for tag in tags:
            key = tag_key(tag, ignore_case, ignore_emphasis)
            if key not in position:
                position[key] = len(position)
                chosen[key] = tag
            elif keep == "last":
                chosen[key] = tag
        tags = [chosen[key] for key in sorted(position, key=position.get)]

    pinned_count = max(0, int(keep_first))
    pinned, rest = tags[:pinned_count], sort_tags(tags[pinned_count:], order)
    if limit > 0:
        rest = rest[: max(0, limit - len(pinned))]
    tags = pinned + rest
    return tags, original + added - len(tags)
