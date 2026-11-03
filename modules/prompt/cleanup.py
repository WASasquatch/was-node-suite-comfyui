"""Text cleanup: letter case, spacing, line breaks, punctuation, the closing mark and quotes.

Each step is a function on a string. :func:`clean` runs the chosen steps in a fixed order,
and with every step off returns the text unchanged.
"""

from __future__ import annotations

import re

__all__ = [
    "CASES",
    "ENDINGS",
    "LINE_BREAKS",
    "change_case",
    "clean",
    "collapse_whitespace",
    "fix_ending",
    "fix_punctuation",
    "join_lines",
    "straighten_quotes",
]

#: The letter case settings, ``keep`` first.
CASES = ("keep", "lower", "upper", "sentence", "title")

#: The line break settings, ``keep`` first.
LINE_BREAKS = ("keep", "one line")

#: The closing mark settings, ``keep`` first.
ENDINGS = ("keep", "period", "none")

#: Curly quotes and apostrophes, mapped to their straight forms.
QUOTES = str.maketrans({
    "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"',
    "«": '"', "»": '"',
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
})

#: Marks that end a sentence.
TERMINALS = ".!?…。！？"

#: Marks that close a quotation or a bracket after the end of a sentence.
CLOSERS = "\"')]}”’»"

#: Any line break a text may hold.
_BREAKS = re.compile(r"\r\n|[\r\n\x0b\x0c\x1c-\x1e\x85  ]")

#: A run of spaces and tabs.
_BLANKS = re.compile(r"[ \t]+")

#: Spaces and tabs at the start of a line.
_LEADING_BLANKS = re.compile(r"^[ \t]+", re.MULTILINE)

#: Spaces and tabs at the end of a line, before any carriage return.
_TRAILING_BLANKS = re.compile(r"[ \t]+(?=\r?$)", re.MULTILINE)

#: A comma or semicolon followed by more of them, as ``, ,`` or ``;;``.
_REPEATED_SEPARATORS = re.compile(r"([,;])(?:[ \t]*[,;])+")

#: Full stops with spaces between them, as ``. .``. An ellipsis has none.
_SPACED_STOPS = re.compile(r"\.(?:[ \t]+\.)+")

#: A comma or semicolon standing just before the mark that ends a sentence.
_SEPARATOR_BEFORE_END = re.compile(r"[,;][ \t]*(?=[.!?])")

#: Spaces and tabs before a punctuation mark, a full stop opening a number such as ``.5`` aside.
_SPACE_BEFORE_MARK = re.compile(r"[ \t]+(?=[,;:!?]|\.(?!\d))")

#: A comma, semicolon or colon directly before a letter, with any spacing between.
_MARK_BEFORE_LETTER = re.compile(r"([,;:])[ \t]*(?=[^\W\d_])")

#: Separators and lone full stops at the start of a line.
_LEADING_SEPARATORS = re.compile(r"^[ \t]*(?:(?:[,;:]|\.(?![.\d]))[ \t]*)+", re.MULTILINE)

#: Commas and semicolons at the end of a line, before any carriage return.
_TRAILING_SEPARATORS = re.compile(r"(?:[ \t]*[,;])+[ \t]*(?=\r?$)", re.MULTILINE)

#: Full stops, commas and semicolons closing a text.
_TRAILING_STOPS = re.compile(r"(?:[ \t]*[.,;])+$")

#: Commas, semicolons and colons closing a text.
_TRAILING_JOINERS = re.compile(r"(?:[ \t]*[,;:])+$")

#: One token for case changes: a bracketed run, a word, or any other single character.
_TOKENS = re.compile(
    r"(?P<bracket>\[[^\[\]\n]*\])|(?P<word>\w+(?:['’]\w+)*)|(?P<other>.)",
    re.DOTALL,
)


def collapse_whitespace(text: str) -> str:
    """Collapse runs of spaces and tabs to one space, trim every line, and trim the text.

    Args:
        text: The text.

    Returns:
        The text with its spacing collapsed. Line breaks are kept.
    """
    text = _BLANKS.sub(" ", text)
    text = _LEADING_BLANKS.sub("", text)
    text = _TRAILING_BLANKS.sub("", text)
    return text.strip()


def join_lines(text: str) -> str:
    """Join every line into one, a space at each join and blank lines dropped.

    Args:
        text: The text.

    Returns:
        One line, or the text itself where it holds no line break.
    """
    if not _BREAKS.search(text):
        return text
    return " ".join(part.strip() for part in _BREAKS.split(text) if part.strip())


def fix_punctuation(text: str, space_after: bool = True) -> str:
    """Tidy the spacing around punctuation and drop stray separators.

    Args:
        text: The text.
        space_after: Put one space after ``,`` ``;`` and ``:`` where a letter follows.

    Returns:
        The text with no space before ``, . ; : ! ?``, repeated commas, semicolons and
        ``. .`` collapsed, a comma before a closing mark dropped, and separators at the start
        or end of a line removed.
    """
    text = _REPEATED_SEPARATORS.sub(r"\1", text)
    text = _SPACED_STOPS.sub(".", text)
    text = _SEPARATOR_BEFORE_END.sub("", text)
    text = _SPACE_BEFORE_MARK.sub("", text)
    if space_after:
        text = _MARK_BEFORE_LETTER.sub(r"\1 ", text)
    text = _LEADING_SEPARATORS.sub("", text)
    return _TRAILING_SEPARATORS.sub("", text)


def _capitalised(word: str) -> str:
    """A word with its first letter made a capital, unless it already holds one.

    Args:
        word: One word.

    Returns:
        The word, its first letter raised where the word is all small letters.
    """
    if not word[:1].islower() or any(char.isupper() for char in word):
        return word
    return word[0].upper() + word[1:]


def change_case(text: str, mode: str) -> str:
    """Change the letter case of a text, leaving bracketed runs such as ``[trigger]`` alone.

    Args:
        text: The text.
        mode: One of :data:`CASES`. ``sentence`` raises the first letter of each sentence and
            ``title`` of each word, both skipping a word that already holds a capital.

    Returns:
        The text in the chosen case. An unknown mode returns it unchanged.
    """
    if mode not in CASES[1:]:
        return text
    out = []
    starting = True
    ended = False
    for token in _TOKENS.finditer(text):
        kind = token.lastgroup
        piece = token.group()
        if kind == "bracket":
            starting = ended = False
        elif kind == "word":
            if mode == "lower":
                piece = piece.lower()
            elif mode == "upper":
                piece = piece.upper()
            elif mode == "title" or starting:
                piece = _capitalised(piece)
            starting = ended = False
        else:
            if mode == "lower":
                piece = piece.lower()
            elif mode == "upper":
                piece = piece.upper()
            if _BREAKS.match(piece):
                starting, ended = True, False
            elif piece in TERMINALS:
                ended = True
            elif piece.isspace():
                starting = starting or ended
            elif piece not in CLOSERS:
                ended = False
        out.append(piece)
    return "".join(out)


def _split_tail(text: str) -> tuple[str, str]:
    """A text and the whitespace closing it, as two strings.

    Args:
        text: The text.

    Returns:
        ``(body, tail)``, ``tail`` being the trailing whitespace.
    """
    body = text.rstrip()
    return body, text[len(body):]


def fix_ending(text: str, mode: str) -> str:
    """Settle how a text ends.

    Args:
        text: The text.
        mode: One of :data:`ENDINGS`. ``period`` adds a full stop unless the text already
            ends a sentence; ``none`` removes closing full stops, commas and semicolons.

    Returns:
        The text with its closing mark settled, any trailing whitespace kept.
    """
    body, tail = _split_tail(text)
    if not body or mode not in ENDINGS[1:]:
        return text
    if mode == "none":
        return _TRAILING_STOPS.sub("", body) + tail
    last = body.rstrip(CLOSERS)
    if last and last[-1] in TERMINALS:
        return text
    return _TRAILING_JOINERS.sub("", body) + "." + tail


def straighten_quotes(text: str) -> str:
    """Curly quotes and apostrophes replaced by straight ones.

    Args:
        text: The text.

    Returns:
        The text with each of :data:`QUOTES` replaced.
    """
    return text.translate(QUOTES)


def clean(
    text: str,
    case: str = "keep",
    whitespace: bool = True,
    line_breaks: str = "keep",
    punctuation: bool = True,
    ending: str = "keep",
    quotes: bool = False,
) -> str:
    """Run the chosen cleanup steps over a text: quotes, lines, spacing, punctuation, case, ending.

    Args:
        text: The text.
        case: One of :data:`CASES`.
        whitespace: Collapse runs of spaces and tabs and trim every line.
        line_breaks: One of :data:`LINE_BREAKS`.
        punctuation: Tidy the spacing around punctuation and drop stray separators.
        ending: One of :data:`ENDINGS`.
        quotes: Replace curly quotes and apostrophes with straight ones.

    Returns:
        The cleaned text. ``keep``, ``False``, ``keep``, ``False``, ``keep``, ``False``
        returns ``text`` itself.
    """
    if quotes:
        text = straighten_quotes(text)
    if line_breaks == "one line":
        text = join_lines(text)
    if whitespace:
        text = collapse_whitespace(text)
    if punctuation:
        text = fix_punctuation(text)
    text = change_case(text, case)
    return fix_ending(text, ending)
