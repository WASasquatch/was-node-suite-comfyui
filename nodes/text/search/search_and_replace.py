"""Regular-expression search and replace over a text input."""

from __future__ import annotations

import re

from comfy_api.latest import io

from ....modules import log
from ....modules.compat.sockets import LINK, WIDGET, input_source
from ....modules.compat.types import LIST, NUMBER
from ....modules.interface import run_result

logger = log.get_logger("nodes.text.search")

#: Matches the substitution records for the readout, one entry each. A run with more than this
#: reports how many were found and leaves out how many changed the text, since that second
#: count is only true of a walk that reached every match. The bodies and the sample rows are
#: unaffected: the marked spans sit in a window that opens on the first match, and the rows are
#: the first few matches past it.
MAX_SCANNED_MATCHES = 20000

#: The words the readout gives each source the text can arrive on.
SOURCE_WORDS = {LINK: "the wired link", WIDGET: "the text box"}

#: The three inputs a run reads, in the order the schema declares them. Each is reported as
#: what the run was handed on it, which is what the readout measures the node against.
INPUT_NAMES = ("text", "find", "replace")

#: What the readout calls the two bodies of text it draws: the text the run searched, and the
#: text the run produced.
BODY_SEARCHED = "before"
BODY_REPLACED = "after"

#: Characters of a pattern or a replacement the per pattern breakdown names. Each side is cut on
#: its own, so a long pattern leaves the replacement beside it readable rather than taking the
#: whole row to itself.
MAX_NAMED_CHARS = 24


#: Find and replace boxes beyond the first pair, and the highest box number, which is one more
#: than the count. They are appended after every input v2 declared, so a saved workflow's values
#: keep their places, and the interface draws only the filled ones and the next empty one.
#:
#: Each slot is written out rather than generated in a loop, so the schema can be read straight
#: from this source without running it. Eight pairs is what a person types by hand; anything
#: longer arrives on the LIST that `find` and `replace` also accept, which has no ceiling.
EXTRA_PAIRS = 7
LAST_PAIR = EXTRA_PAIRS + 1

#: What every appended box says, since they differ only by number. The first pair keeps its own
#: wording, which explains the node rather than the box.
MORE_FIND_HINT = "Another pattern to look for"
MORE_REPLACE_HINT = "What that pattern becomes"
MORE_FIND_TIP = (
    "Another pattern, searched in the same pass as the first. Every pattern reads the text "
    "as it arrived, so a replacement is never matched again and two patterns can swap; "
    "where two match at one place, the earlier box wins."
)
MORE_REPLACE_TIP = (
    "What the pattern in the box above becomes. Leave it empty to delete what that pattern "
    "matched, the same as the first pair."
)


#: The appended boxes, in the order they are drawn. A list at module scope, so the schema
#: names it and the source reader follows the name to these elements. Written out one call
#: at a time, each socket declared in full rather than built by a helper.
EXTRA_PAIR_INPUTS = [
    io.MultiType.Input(
        io.String.Input(
            "find_2",
            default="",
            multiline=True,
            optional=True,
            placeholder=MORE_FIND_HINT,
        ),
        [io.String, LIST],
        optional=True,
        tooltip=MORE_FIND_TIP,
    ),
    io.MultiType.Input(
        io.String.Input(
            "replace_2",
            default="",
            multiline=True,
            optional=True,
            placeholder=MORE_REPLACE_HINT,
        ),
        [io.String, LIST],
        optional=True,
        tooltip=MORE_REPLACE_TIP,
    ),
    io.MultiType.Input(
        io.String.Input(
            "find_3",
            default="",
            multiline=True,
            optional=True,
            placeholder=MORE_FIND_HINT,
        ),
        [io.String, LIST],
        optional=True,
        tooltip=MORE_FIND_TIP,
    ),
    io.MultiType.Input(
        io.String.Input(
            "replace_3",
            default="",
            multiline=True,
            optional=True,
            placeholder=MORE_REPLACE_HINT,
        ),
        [io.String, LIST],
        optional=True,
        tooltip=MORE_REPLACE_TIP,
    ),
    io.MultiType.Input(
        io.String.Input(
            "find_4",
            default="",
            multiline=True,
            optional=True,
            placeholder=MORE_FIND_HINT,
        ),
        [io.String, LIST],
        optional=True,
        tooltip=MORE_FIND_TIP,
    ),
    io.MultiType.Input(
        io.String.Input(
            "replace_4",
            default="",
            multiline=True,
            optional=True,
            placeholder=MORE_REPLACE_HINT,
        ),
        [io.String, LIST],
        optional=True,
        tooltip=MORE_REPLACE_TIP,
    ),
    io.MultiType.Input(
        io.String.Input(
            "find_5",
            default="",
            multiline=True,
            optional=True,
            placeholder=MORE_FIND_HINT,
        ),
        [io.String, LIST],
        optional=True,
        tooltip=MORE_FIND_TIP,
    ),
    io.MultiType.Input(
        io.String.Input(
            "replace_5",
            default="",
            multiline=True,
            optional=True,
            placeholder=MORE_REPLACE_HINT,
        ),
        [io.String, LIST],
        optional=True,
        tooltip=MORE_REPLACE_TIP,
    ),
    io.MultiType.Input(
        io.String.Input(
            "find_6",
            default="",
            multiline=True,
            optional=True,
            placeholder=MORE_FIND_HINT,
        ),
        [io.String, LIST],
        optional=True,
        tooltip=MORE_FIND_TIP,
    ),
    io.MultiType.Input(
        io.String.Input(
            "replace_6",
            default="",
            multiline=True,
            optional=True,
            placeholder=MORE_REPLACE_HINT,
        ),
        [io.String, LIST],
        optional=True,
        tooltip=MORE_REPLACE_TIP,
    ),
    io.MultiType.Input(
        io.String.Input(
            "find_7",
            default="",
            multiline=True,
            optional=True,
            placeholder=MORE_FIND_HINT,
        ),
        [io.String, LIST],
        optional=True,
        tooltip=MORE_FIND_TIP,
    ),
    io.MultiType.Input(
        io.String.Input(
            "replace_7",
            default="",
            multiline=True,
            optional=True,
            placeholder=MORE_REPLACE_HINT,
        ),
        [io.String, LIST],
        optional=True,
        tooltip=MORE_REPLACE_TIP,
    ),
    io.MultiType.Input(
        io.String.Input(
            "find_8",
            default="",
            multiline=True,
            optional=True,
            placeholder=MORE_FIND_HINT,
        ),
        [io.String, LIST],
        optional=True,
        tooltip=MORE_FIND_TIP,
    ),
    io.MultiType.Input(
        io.String.Input(
            "replace_8",
            default="",
            multiline=True,
            optional=True,
            placeholder=MORE_REPLACE_HINT,
        ),
        [io.String, LIST],
        optional=True,
        tooltip=MORE_REPLACE_TIP,
    ),
]


class TextFindAndReplace(io.ComfyNode):
    """Replace every match of ``find`` in ``text`` with ``replace``."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Text Find and Replace",
            display_name="Text Find and Replace",
            search_aliases=["Text Find and Replace", "search and replace", "regex", "substitute"],
            category="WAS Suite/Text/Search",
            description=(
                "Replace every regular-expression match in the text and report how many "
                "replacements were made."
            ),
            inputs=[
                io.String.Input(
                    "text",
                    default="",
                    multiline=True,
                    placeholder="Eg: a cat on a mat",
                    tooltip=(
                        "Text to search; STRING, as `a tabby cat`. Every filled find box is applied in "
                        "one pass over it."
                    ),
                ),
                io.MultiType.Input(
                    io.String.Input(
                        "find",
                        default="",
                        multiline=True,
                        placeholder="What to look for. A pattern, so cat|dog matches either",
                    ),
                    [io.String, LIST],
                    tooltip=(
                        "A regular expression, so 'cat|dog' matches either word and '\\s+' "
                        "matches whitespace. A backslash makes a special character literal. "
                        "Left empty, nothing is replaced."
                    ),
                ),
                io.MultiType.Input(
                    io.String.Input(
                        "replace",
                        default="",
                        multiline=True,
                        placeholder="What each match becomes. Leave empty to delete them",
                    ),
                    [io.String, LIST],
                    tooltip=(
                        "What each match becomes. Leave it empty to delete the matches. "
                        "\\1 and \\2 stand for the first and second parenthesised group of "
                        "the pattern, and a literal backslash has to be doubled."
                    ),
                ),
                *EXTRA_PAIR_INPUTS,
            ],
            outputs=[
                io.String.Output(
                    display_name="result_text",
                    tooltip="The text with every match replaced.",
                ),
                NUMBER.Output(
                    display_name="replacement_count_number",
                    tooltip=(
                        "How many replacements were made, for the NUMBER inputs of the "
                        "suite's own maths and logic nodes. 0 means the text was not "
                        "matched at all."
                    ),
                ),
                io.Float.Output(
                    display_name="replacement_count_float",
                    tooltip="The same count as a decimal, for example 3.0.",
                ),
                io.Int.Output(
                    display_name="replacement_count_int",
                    tooltip="The same count as a whole number, for a core INT input.",
                ),
            ],
        )

    @classmethod
    def execute(cls, text, find, replace, **extra) -> io.NodeOutput:
        pairs = _pairs(find, replace, extra)
        # The readout is measured off what the substitution itself did, so the walk is recorded
        # only while a browser is attached to read it and the run costs nothing otherwise.
        modified_text, count, walk = _apply(text, pairs, record=run_result.watching())
        _publish_run(text, pairs, modified_text, count, modified_text != text, walk)
        return io.NodeOutput(modified_text, count, float(count), int(count))



def _entries(value, keep_empty: bool = False) -> list[str]:
    """One box or one wired list, as the strings it holds.

    Args:
        value: What arrived on a find or replace input: a string typed into the box, a list
            from a LIST link, or None for a box the prompt did not carry.
        keep_empty: Whether an empty box is a string in its own right. False for a pattern,
            where empty would match at every position; True for a replacement, where empty is
            what deletes the matches.

    Returns:
        The strings, in order.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(entry) for entry in value]
    return [value] if value != "" or keep_empty else []


def _pairs(find, replace, extra: dict) -> list[tuple[str, str]]:
    """Every pattern paired with its replacement, in the order the boxes are read.

    An empty replacement box deletes what its pattern matched.

    Args:
        find: The first find box, or a list wired into it.
        replace: The first replace box, or a list wired into it.
        extra: The appended boxes, by input name.

    Returns:
        ``(pattern, replacement)`` pairs, the unpaired ones left out.
    """
    # A pattern with no box opposite it at all, which is what an appended pair the prompt never
    # carried looks like, is dropped rather than deleted, so an unfinished pair leaves the text
    # alone. An empty box is a box, and deletes.
    boxes = [(find, replace)]
    for index in range(2, LAST_PAIR + 1):
        boxes.append((extra.get(f"find_{index}"), extra.get(f"replace_{index}")))

    pairs = []
    for found, replaced in boxes:
        patterns = _entries(found)
        replacements = _entries(replaced, keep_empty=True)
        for position, pattern in enumerate(patterns):
            if position < len(replacements):
                pairs.append((pattern, replacements[position]))
    return pairs


def _apply(
    text: str, pairs: list[tuple[str, str]], record: bool = False
) -> tuple[str, int, dict | None]:
    """Replace every match in one pass over the text.

    Args:
        text: The text to search.
        pairs: ``(pattern, replacement)`` in the order the boxes were read.
        record: Whether to hand back what the walk did, for the readout to measure.

    Returns:
        ``(the text, how many matches were replaced, the walk)``. The walk is None unless
        ``record``, and otherwise ``{"spans": ..., "counts": ...}``. ``spans`` holds
        ``(start, end, result_start, result_end, same)`` for the first
        :data:`MAX_SCANNED_MATCHES` matches replaced, placing each match in the text it was
        found in and its replacement in the text that came out, ``same`` reading True where
        that replacement is the characters it stands in for. ``counts`` holds one total per
        pair, in the order of ``pairs``, with every match counted however many there were.

    Raises:
        re.error: A pattern is not a valid regular expression.
    """
    walk = {"spans": [], "counts": [0] * len(pairs)} if record else None
    if not pairs:
        return text, 0, walk

    # Where two patterns match at the same place the earlier pair wins, and where they
    # overlap the one starting earlier wins, which is what sorting on (start, pair) and
    # then skipping anything inside what has already been taken comes to.

    # Each pattern is scanned on its own rather than joined into one alternation, so its
    # groups keep the numbers the user wrote: a combined pattern would renumber them and
    # a replacement template of \1 would expand to whatever the join put first instead.
    found = []
    for index, (pattern, replacement) in enumerate(pairs):
        for match in re.finditer(pattern, text):
            found.append((match.start(), index, match, replacement))

    # Leftmost first, and the earlier box where two start together, which is the order the
    # boxes are read in. Sorting on the match itself is never reached and would not compare.
    found.sort(key=lambda entry: (entry[0], entry[1]))

    out = []
    cursor = 0
    count = 0
    written = 0
    for start, index, match, replacement in found:
        # A match inside one already taken is dropped rather than nested, so every character
        # of the result comes from exactly one pattern.
        if start < cursor:
            continue
        kept = text[cursor:start]
        expansion = match.expand(replacement)
        out.append(kept)
        out.append(expansion)
        count += 1
        if walk is not None:
            # Where the replacement lands is the length of the result so far, which is exact
            # for every pattern rather than worked out again from the offsets it moved.
            written += len(kept)
            walk["counts"][index] += 1
            if len(walk["spans"]) < MAX_SCANNED_MATCHES:
                walk["spans"].append((
                    start, match.end(), written, written + len(expansion),
                    expansion == match.group(0),
                ))
            written += len(expansion)
        # A zero width match consumes nothing, so the cursor may not go backwards and the
        # character it sat before is still written.
        cursor = max(cursor, match.end())
    out.append(text[cursor:])
    return "".join(out), count, walk


def _publish_run(
    text: str, pairs: list, result_text: str, found: int, moved: bool, walk: dict | None
) -> None:
    """Report what the run did to the node's own interface.

    Args:
        text: The text the run searched.
        pairs: The pattern and replacement pairs the run applied, in order.
        result_text: The text the substitution produced.
        found: Matches the substitution made, which is what the count outputs carry.
        moved: Whether the substitution changed the text.
        walk: What :func:`_apply` recorded, or None for a substitution that recorded nothing,
            which is reported no further.
    """
    try:
        # A browser that attached during the substitution has nothing recorded to be told
        # about, and reads the next run instead.
        if walk is None or not run_result.watching():
            return
        find, replace = pairs[0] if pairs else ("", "")
        rows, replaced, bodies, beyond = _measure(text, result_text, found, walk["spans"])
        counts = {"found": found}
        if replaced is not None:
            counts["replaced"] = replaced
        # One row per pair, so a pair that matched nothing reads as a zero rather than as an
        # absence. A wired list holding more pairs than a result carries is reported by the
        # total beside them instead.
        tallies = [
            {"name": _pair_name(pattern, replacement), "value": total}
            for (pattern, replacement), total in zip(
                pairs[: run_result.MAX_TALLIES], walk["counts"]
            )
        ]
        sources = {name: input_source(name) for name in INPUT_NAMES}
        facts = {}
        words = SOURCE_WORDS.get(sources["text"])
        if words is not None:
            # Named after the input rather than called a source, since all three inputs take a
            # link and only this one is measured.
            facts["text came from"] = words
        handed = {"text": text, "find": find, "replace": replace}
        inputs = [
            run_result.given(name, handed[name], _linked(sources[name]))
            for name in INPUT_NAMES
        ]
        run_result.publish(
            # A run that handed back the text it was given is the warning, however it got
            # there: no match at all, matches replaced with themselves, and the rarer case of
            # replacements that move two matches in opposite directions.
            status=run_result.OK if moved else run_result.WARNING,
            summary=_summary(found, replaced, moved),
            counts=counts,
            tallies=tallies,
            tallies_total=len(pairs),
            facts=facts,
            items=rows,
            items_total=beyond,
            bodies=bodies,
            inputs=inputs,
        )
    except Exception as error:
        logger.debug("Text Find and Replace did not report its run (%s)", error)


def _measure(
    text: str, result_text: str, found: int, spans: list[tuple]
) -> tuple[list[dict], int | None, list[dict], int]:
    """Read the recorded matches into the two bodies, the sample rows and the second count.

    Args:
        text: The text the run searched.
        result_text: The text the substitution produced.
        found: Matches the substitution made.
        spans: What :func:`_apply` recorded of the matches it replaced, in the order it
            replaced them.

    Returns:
        ``(rows, replaced, bodies, beyond)``. ``bodies`` is the text as searched and the
        result, each carrying a window of itself with the spans inside that window marked.
        ``rows`` holds the first :data:`modules.interface.run_result.MAX_ITEMS` matches the
        first body does not reach, each in context and noting the line it sits on and whether
        its replacement is the text it matched, none of them drawing a passage another row
        already drew. ``beyond`` counts every match outside that body, which is what the rows
        are a sample of. ``replaced`` counts the matches whose replacement differs from the
        text they matched, and is None when there are more than :data:`MAX_SCANNED_MATCHES`
        of them, where only ``found`` is measured. It is counted on every run, since a
        substitution can move two matches in opposite directions and hand back the text it
        was given.
    """
    rows = []
    searched_marks = []
    result_marks = []
    replaced = 0
    inside = 0
    line = 1
    cursor = 0
    reach = 0
    searched_window = None
    result_window = None
    for start, end, result_start, result_end, same in spans:
        if searched_window is None:
            # Both windows open on the first match, so the two bodies show the same passage of
            # the text on either side of the substitution.
            searched_window = run_result.window(len(text), start)
            result_window = run_result.window(len(result_text), result_start)
        # Counted forward from the last match rather than from the start of the text, so the
        # whole walk reads the text once however many rows it takes.
        line += text.count("\n", cursor, start)
        cursor = start
        replaced += not same
        if _within(start, end, searched_window):
            inside += 1
            if len(searched_marks) < run_result.MAX_MARKS:
                searched_marks.append((start, end))
        elif len(rows) < run_result.MAX_ITEMS and start >= reach:
            note = f"line {line}, replaced with itself" if same else f"line {line}"
            rows.append(run_result.excerpt(text, start, end, note=note))
            # The next row opens past the context this one keeps, so no two rows draw the same
            # passage of the text.
            reach = end + run_result.CONTEXT_CHARS
        if (len(result_marks) < run_result.MAX_MARKS
                and _within(result_start, result_end, result_window)):
            result_marks.append((result_start, result_end))
    if searched_window is None:
        searched_window = run_result.window(len(text))
        result_window = run_result.window(len(result_text))
    bodies = [
        run_result.body(BODY_SEARCHED, text, searched_marks, found, searched_window[0]),
        run_result.body(BODY_REPLACED, result_text, result_marks, found, result_window[0]),
    ]
    beyond = max(0, found - inside)
    return rows, replaced if len(spans) >= found else None, bodies, beyond


def _pair_name(pattern: str, replacement: str) -> str:
    """What the readout calls one find and replace pair in its per pattern breakdown.

    Args:
        pattern: The pattern the pair searched for.
        replacement: The substitution template it wrote in place of every match.

    Returns:
        A phrase naming both sides, quoted so a pattern of spaces reads as one, and saying
        that the matches are deleted where there is no replacement to name.
    """
    if replacement == "":
        return f"{_quoted(pattern)} is deleted"
    return f"{_quoted(pattern)} becomes {_quoted(replacement)}"


def _quoted(value: str) -> str:
    """One side of a pair in quotes, cut to :data:`MAX_NAMED_CHARS` characters.

    Args:
        value: A pattern or a replacement as the run read it.

    Returns:
        The value in single quotes, ending in an ellipsis where it was cut.
    """
    if len(value) > MAX_NAMED_CHARS:
        kept = value[: MAX_NAMED_CHARS - len(run_result.ELLIPSIS)]
        return f"'{kept}{run_result.ELLIPSIS}'"
    return f"'{value}'"


def _linked(source: str | None) -> bool | None:
    """Whether an input was filled by a link, as a run report states it.

    Args:
        source: What :func:`modules.compat.sockets.input_source` answered for that input.

    Returns:
        True for a link, False for the widget beside the input, and None where the running
        prompt could not be read, which publishes the value with no source named.
    """
    if source == LINK:
        return True
    if source == WIDGET:
        return False
    return None


def _within(start: int, end: int, window: tuple[int, int]) -> bool:
    """Whether a span falls in the piece of a text a body carries.

    Args:
        start: The span's first index.
        end: One past its last, equal to ``start`` for a span of no width.
        window: ``(start, stop)`` from :func:`modules.interface.run_result.window`.

    Returns:
        True when the span overlaps the window, and for a span of no width when its position
        is anywhere from the window's first character to one past its last.
    """
    if start == end:
        return window[0] <= start <= window[1]
    return start < window[1] and end > window[0]


def _summary(found: int, replaced: int | None, moved: bool) -> str:
    """The one line the readout leads with.

    Args:
        found: Matches the substitution made.
        replaced: Matches whose replacement differs from the text they matched, or None when
            there were too many to count.
        moved: Whether the substitution changed the text.

    Returns:
        A sentence naming both numbers, or naming the one that was measured.
    """
    if found == 0:
        return "Nothing matched, so the text came through unchanged."
    if not moved:
        if replaced is None:
            return f"{_matches(found)} replaced, and the text came out as it went in."
        if replaced:
            return (
                f"{_matches(found)}, {replaced:,} of them replaced with different text, and "
                "the text still came out as it went in."
            )
        return f"{_matches(found)}, each replaced with the same text, so nothing changed."
    if replaced is None:
        return f"{_matches(found)} replaced, too many to count how many changed the text."
    if replaced == found:
        return f"{_matches(found)} found and replaced."
    return (
        f"{_matches(found)}, {replaced:,} of them replaced and "
        f"{found - replaced:,} replaced with the same text."
    )


def _matches(count: int) -> str:
    """``1 match`` or ``12,000 matches``."""
    return f"{count:,} match" if count == 1 else f"{count:,} matches"
