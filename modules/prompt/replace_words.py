"""Word and phrase replacement across a batch of texts, and a phrase every text must hold.

Rules read ``finds -> replacement``, one per line. Finds are literal, matched in one pass
per text, longest first.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..util.patterns import MAX_GROWTH_CHARS
from . import cleanup

__all__ = [
    "ARROW",
    "MAX_FINDS",
    "PLACES",
    "Matcher",
    "Outcome",
    "Rule",
    "add_phrase",
    "apply",
    "compile_finds",
    "parse_rules",
    "tidy",
]

#: The node's name in messages.
NODE = "Text Replace Words"

#: What separates the finds of a rule from its replacement.
ARROW = "->"

#: What opens a line of rules that is a note.
COMMENT = "#"

#: Finds one set of rules may hold.
MAX_FINDS = 2000

#: Where the ensured phrase goes, ``start`` first.
PLACES = ("start", "end")

#: Characters of a rule line a message quotes.
SHOWN_CHARS = 80

#: Separators the ensured phrase is joined with that a sentence's closing mark moves past.
JOINERS = ",;"

#: One word character, the class a whole-word bound looks at.
_WORD_CHAR = re.compile(r"\w")

#: The marks closing a sentence at the end of a text.
_CLOSING = re.compile(r"[.!?…]+$")


@dataclass(frozen=True)
class Rule:
    """One line of rules: the phrases it finds and what each becomes.

    Attributes:
        finds: The phrases, as typed.
        replacement: What each becomes, empty to delete.
        line: The line it was read from, counting from 1.
    """

    finds: tuple[str, ...]
    replacement: str
    line: int

    @property
    def name(self) -> str:
        """The rule as written, as ``a woman, woman -> [trigger]``."""
        right = f" {self.replacement}" if self.replacement else ""
        return f"{', '.join(self.finds)} {ARROW}{right}"


@dataclass
class Outcome:
    """What the rules and the ensured phrase did to one text.

    Attributes:
        text: The text that came out.
        hits: Matches per rule, in rule order.
        ensured: Whether the ensured phrase was added.
        found: The first replaced span as it read before, or None.
        before: That span in the original text, as ``(start, end)``, or None.
        after: Its replacement in ``text``, or the added phrase where no rule matched, or None.
    """

    text: str
    hits: list[int]
    ensured: bool = False
    found: str | None = None
    before: tuple[int, int] | None = None
    after: tuple[int, int] | None = None

    @property
    def matched(self) -> bool:
        """Whether any rule matched the text."""
        return any(self.hits)


def _refusal(number: int, line: str, problem: str) -> str:
    """The message refusing one line of rules.

    Args:
        number: The line's number, counting from 1.
        line: The line as written.
        problem: What is wrong with it, completing ``it ...``.

    Returns:
        A message naming the line, the problem and the form a rule takes.
    """
    shown = line if len(line) <= SHOWN_CHARS else line[: SHOWN_CHARS - 3] + "..."
    return (
        f"{NODE} could not read line {number} of its rules, '{shown}': it {problem}. Write "
        f"each rule as the words to find, separated by commas, then {ARROW} and what they "
        f"become, as `a woman, woman {ARROW} [trigger]`. Leave the right side empty to delete "
        f"them, and start a line with {COMMENT} to keep it as a note."
    )


def parse_rules(text: str) -> list[Rule]:
    """Read rules written one per line, as ``a woman, woman -> [trigger]``.

    Args:
        text: The rules. Blank lines and lines starting with ``#`` are skipped.

    Returns:
        The rules, in the order written.

    Raises:
        ValueError: A line has no arrow or nothing before it, or the rules hold more than
            :data:`MAX_FINDS` finds.
    """
    rules = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith(COMMENT):
            continue
        left, arrow, right = line.partition(ARROW)
        if not arrow:
            raise ValueError(_refusal(number, line, f"has no {ARROW}"))
        finds = tuple(part.strip() for part in left.split(",") if part.strip())
        if not finds:
            raise ValueError(_refusal(number, line, f"has nothing before the {ARROW} to find"))
        rules.append(Rule(finds, right.strip(), number))
    total = sum(len(rule.finds) for rule in rules)
    if total > MAX_FINDS:
        raise ValueError(
            f"{NODE} was given {total:,} words and phrases to find, and takes at most "
            f"{MAX_FINDS:,}. Split the rules across two {NODE} nodes, one after the other."
        )
    return rules


def compile_finds(finds: list[str], whole_words: bool, ignore_case: bool) -> re.Pattern:
    """One pattern matching any of the finds, tried in the order given, each in its own group.

    Args:
        finds: Literal phrases, none empty. Spaces inside one match any run of spaces and tabs.
        whole_words: Refuse a match with a word character just outside a find's word edges.
        ignore_case: Match any letter case.

    Returns:
        The compiled pattern, group ``n`` holding a match of ``finds[n - 1]``.
    """
    branches = []
    for find in finds:
        words = find.split()
        body = r"[ \t]+".join(re.escape(word) for word in words)
        lead = r"(?<!\w)" if whole_words and _WORD_CHAR.match(words[0][0]) else ""
        tail = r"(?!\w)" if whole_words and _WORD_CHAR.match(words[-1][-1]) else ""
        branches.append(f"{lead}({body}){tail}")
    return re.compile("|".join(branches), re.IGNORECASE if ignore_case else 0)


class Matcher:
    """Every find of a set of rules compiled into one pattern, longest first."""

    def __init__(self, rules: list[Rule], whole_words: bool = True, ignore_case: bool = True):
        """Compile the finds.

        Args:
            rules: The rules. A find repeated in a later rule stays with the earlier one.
            whole_words: Match whole words only.
            ignore_case: Match any letter case.
        """
        self.rules = rules
        entries = []
        seen = set()
        for index, rule in enumerate(rules):
            for find in rule.finds:
                key = " ".join(find.split())
                key = key.casefold() if ignore_case else key
                if key in seen:
                    continue
                seen.add(key)
                entries.append((key, find, index))
        entries.sort(key=lambda entry: -len(entry[0]))
        self.owners = [index for _, _, index in entries]
        self.pattern = (
            compile_finds([find for _, find, _ in entries], whole_words, ignore_case)
            if entries else None
        )

    def phrase(self, found: str, index: int, keep_capitals: bool) -> str:
        """What one match becomes.

        Args:
            found: The matched text.
            index: The rule that matched it.
            keep_capitals: Carry a capitalised match onto a replacement starting with a letter.

        Returns:
            The replacement, in the match's capitals where ``keep_capitals`` applies.
        """
        replacement = self.rules[index].replacement
        if not keep_capitals or not replacement[:1].isalpha():
            return replacement
        cased = [char for char in found if char.isupper() or char.islower()]
        if len(cased) > 1 and all(char.isupper() for char in cased):
            return replacement.upper()
        if cased and cased[0].isupper():
            return replacement[0].upper() + replacement[1:]
        return replacement


def _replace(text: str, matcher: Matcher, keep_capitals: bool) -> tuple[str, list[int], list]:
    """Replace every match in one pass over a text.

    Args:
        text: The text.
        matcher: The compiled rules.
        keep_capitals: Carry a match's capitals onto its replacement.

    Returns:
        ``(text, hits, spans)``: the result, matches per rule, and one
        ``(start, end, new_start, new_end, found)`` per replacement.

    Raises:
        ValueError: The result would grow past :data:`MAX_GROWTH_CHARS` characters.
    """
    hits = [0] * len(matcher.rules)
    if matcher.pattern is None:
        return text, hits, []
    out = []
    spans = []
    cursor = 0
    written = 0
    growth = 0
    for match in matcher.pattern.finditer(text):
        index = matcher.owners[match.lastindex - 1]
        found = match.group()
        new = matcher.phrase(found, index, keep_capitals)
        growth += len(new) - len(found)
        if growth > MAX_GROWTH_CHARS:
            raise ValueError(
                f"{NODE} would make a text more than {MAX_GROWTH_CHARS:,} characters longer "
                f"than it was, and stopped. A long replacement for a short, common word "
                f"multiplies the text. Shorten the replacement, or split the text."
            )
        kept = text[cursor:match.start()]
        out.append(kept)
        out.append(new)
        written += len(kept)
        spans.append((match.start(), match.end(), written, written + len(new), found))
        written += len(new)
        hits[index] += 1
        cursor = match.end()
    out.append(text[cursor:])
    return "".join(out), hits, spans


def tidy(text: str) -> str:
    """Collapse doubled spaces and drop the stray commas and spacing a deleted word leaves.

    Args:
        text: The text after its replacements.

    Returns:
        The tidied text.
    """
    return cleanup.fix_punctuation(cleanup.collapse_whitespace(text), space_after=False)


def _alignment(before: str, after: str) -> list[int]:
    """Where each position of a text lands once characters have been taken out of it.

    Args:
        before: The text.
        after: The text with characters removed, a space and a tab counting alike.

    Returns:
        ``len(before) + 1`` positions into ``after``.
    """
    places = []
    cursor = 0
    for char in before:
        places.append(cursor)
        if cursor < len(after) and (
            after[cursor] == char or (char in " \t" and after[cursor] in " \t")
        ):
            cursor += 1
    places.append(cursor)
    return places


def add_phrase(text: str, phrase: str, at: str, separator: str) -> tuple[str, tuple[int, int]]:
    """Add a phrase at the start or end of a text.

    Args:
        text: The text.
        phrase: The phrase to add.
        at: ``end`` to append, anything else to prepend.
        separator: What joins the phrase to the text. A comma or semicolon also replaces one
            already at that end, and at the end goes before a closing ``.`` ``!`` or ``?``.

    Returns:
        ``(text, span)``, the span being where the phrase sits.
    """
    joiner = separator.strip()
    joining = bool(joiner) and all(char in JOINERS for char in joiner)
    if at == "end":
        body = text.rstrip()
        closing = ""
        if joining:
            mark = _CLOSING.search(body)
            if mark:
                closing = mark.group()
                body = body[: mark.start()]
            body = body.rstrip(" \t" + JOINERS)
        lead = body + separator if body else ""
        return lead + phrase + closing, (len(lead), len(lead) + len(phrase))
    body = text.lstrip(" \t" + JOINERS) if joining else text.lstrip()
    if not body:
        return phrase, (0, len(phrase))
    return phrase + separator + body, (0, len(phrase))


def apply(
    texts: list[str],
    rules: list[Rule],
    whole_words: bool = True,
    ignore_case: bool = True,
    keep_capitals: bool = True,
    tidied: bool = True,
    ensure: str = "",
    ensure_at: str = "start",
    separator: str = ", ",
) -> list[Outcome]:
    """Run the rules over every text, then add the ensured phrase where a non-blank text lacks it.

    Args:
        texts: The texts, in order.
        rules: From :func:`parse_rules`.
        whole_words: Match whole words only.
        ignore_case: Match any letter case, for the finds and the ensured phrase.
        keep_capitals: Carry a match's capitals onto its replacement.
        tidied: Run :func:`tidy` over each text a rule changed.
        ensure: A phrase every text must hold, or empty for none.
        ensure_at: One of :data:`PLACES`.
        separator: What joins the ensured phrase to a text.

    Returns:
        One :class:`Outcome` per text, in order.

    Raises:
        ValueError: A text would grow past :data:`MAX_GROWTH_CHARS` characters.
    """
    matcher = Matcher(rules, whole_words, ignore_case)
    phrase = ensure.strip()
    wanted = compile_finds([phrase], whole_words, ignore_case) if phrase else None
    outcomes = []
    for text in texts:
        result, hits, spans = _replace(text, matcher, keep_capitals)
        outcome = Outcome(result, hits)
        if spans:
            start, end, new_start, new_end, found = spans[0]
            outcome.found = found
            outcome.before = (start, end)
            outcome.after = (new_start, new_end)
        if tidied and result != text:
            outcome.text = tidy(result)
            if outcome.after is not None:
                places = _alignment(result, outcome.text)
                outcome.after = (places[outcome.after[0]], places[outcome.after[1]])
        if wanted is not None and outcome.text.strip() and not wanted.search(outcome.text):
            joined, span = add_phrase(outcome.text, phrase, ensure_at, separator)
            if outcome.after is not None and ensure_at != "end":
                shift = len(joined) - len(outcome.text)
                outcome.after = (outcome.after[0] + shift, outcome.after[1] + shift)
            elif outcome.after is not None:
                outcome.after = (min(outcome.after[0], span[0]), min(outcome.after[1], span[0]))
            if outcome.after is None:
                outcome.after = span
            outcome.text = joined
            outcome.ensured = True
        outcomes.append(outcome)
    return outcomes
