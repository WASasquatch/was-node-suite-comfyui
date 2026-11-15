"""Arabic joining and right-to-left ordering for a line drawn one character after another.

:func:`display_order` turns a typed line into the characters drawn left to right: Arabic
letters in joined presentation forms, lam-alef as one, right-to-left runs reversed.
"""

from __future__ import annotations

import unicodedata
from functools import lru_cache

__all__ = ["display_order", "holds_rtl"]

#: Bidirectional classes of a right-to-left letter.
RTL_CLASSES = frozenset({"R", "AL"})

#: Presentation form blocks, Forms-B first so a basic letter takes its forms from there.
FORMS_B = (0xFE70, 0xFEFF)
FORMS_A = (0xFB50, 0xFDFF)

#: Decomposition tags naming a positional form.
FORM_TAGS = {"<isolated>": "isolated", "<final>": "final", "<initial>": "initial", "<medial>": "medial"}

LAM = "ل"
TATWEEL = "ـ"
ZWJ = "‍"
ZWNJ = "‌"

#: Joining types: dual, right, join causing, transparent and non-joining.
DUAL = "D"
RIGHT = "R"
CAUSING = "C"
TRANSPARENT = "T"
NONE = "U"

#: Types a letter joins to on its left, and types that join to a letter on their right.
JOINS_LEFT = frozenset({DUAL, CAUSING})
JOINS_RIGHT = frozenset({DUAL, RIGHT, CAUSING})

#: Bidirectional classes of the explicit embedding and isolate controls, left out of the order.
CONTROLS = frozenset({"LRE", "RLE", "LRO", "RLO", "PDF", "BN", "LRI", "RLI", "FSI", "PDI"})

#: Neutral and weak separator classes.
NEUTRALS = frozenset({"B", "S", "WS", "ON"})

#: Each opening bracket and the closing one it pairs with.
BRACKETS = dict(zip(
    "([{⁅⁽₍〈〈《「『【〔〖〘〚（［｛",
    ")]}⁆⁾₎〉〉》」』】〕〗〙〛）］｝",
))

#: Brackets left open before pairing stops.
BRACKET_DEPTH = 63

#: Each mirrored character and the one drawn in its place inside a right-to-left run.
MIRRORS = dict(zip(
    "()[]{}<>«»‹›⁅⁆⁽⁾₍₎≤≥〈〉《》",
    ")(][}{><»«›‹⁆⁅⁾⁽₎₍≥≤〉〈》《",
))


@lru_cache(maxsize=1)
def _tables() -> tuple[dict, dict]:
    """The presentation forms Unicode gives each Arabic letter, and the lam-alef ligatures.

    Returns:
        ``(forms, ligatures)``: letter to ``{form: character}``, and alef to
        ``{form: ligature}`` for lam followed by that alef.
    """
    forms: dict[str, dict[str, str]] = {}
    ligatures: dict[str, dict[str, str]] = {}
    for low, high in (FORMS_B, FORMS_A):
        for code in range(low, high + 1):
            character = chr(code)
            parts = unicodedata.decomposition(character).split()
            if len(parts) < 2 or parts[0] not in FORM_TAGS:
                continue
            form = FORM_TAGS[parts[0]]
            letters = "".join(chr(int(part, 16)) for part in parts[1:])
            if len(letters) == 1:
                forms.setdefault(letters, {}).setdefault(form, character)
            elif (low, high) == FORMS_B and len(letters) == 2 and letters[0] == LAM:
                ligatures.setdefault(letters[1], {}).setdefault(form, character)
    return forms, ligatures


def _joining(character: str, forms: dict) -> str:
    """How a character joins its neighbours.

    Args:
        character: One character.
        forms: Letter to its presentation forms, from :func:`_tables`.

    Returns:
        One of :data:`DUAL`, :data:`RIGHT`, :data:`CAUSING`, :data:`TRANSPARENT` and :data:`NONE`.
    """
    if character in (TATWEEL, ZWJ):
        return CAUSING
    if character == ZWNJ:
        return NONE
    shapes = forms.get(character)
    if shapes:
        if "initial" in shapes or "medial" in shapes:
            return DUAL
        return RIGHT if "final" in shapes else NONE
    return TRANSPARENT if unicodedata.category(character) in ("Mn", "Me", "Cf") else NONE


def _shape(line: str) -> str:
    """Put each Arabic letter of a line in the form its neighbours call for.

    Args:
        line: The line in typed order.

    Returns:
        The line in typed order, letters in presentation forms and lam with alef as one.
    """
    forms, ligatures = _tables()
    kinds = [_joining(character, forms) for character in line]

    def neighbour(index: int, step: int) -> int:
        index += step
        while 0 <= index < len(line) and kinds[index] == TRANSPARENT:
            index += step
        return index if 0 <= index < len(line) else -1

    out = []
    dropped = set()
    for index, character in enumerate(line):
        if index in dropped:
            continue
        shapes = forms.get(character)
        if not shapes:
            out.append(character)
            continue
        before = neighbour(index, -1)
        after = neighbour(index, 1)
        joins_before = before >= 0 and kinds[before] in JOINS_LEFT and kinds[index] in JOINS_RIGHT
        joins_after = after >= 0 and kinds[index] in JOINS_LEFT and kinds[after] in JOINS_RIGHT
        if character == LAM and after >= 0 and line[after] in ligatures:
            tied = ligatures[line[after]].get("final" if joins_before else "isolated")
            if tied:
                out.append(tied)
                dropped.add(after)
                continue
        if joins_before and joins_after:
            form = "medial"
        elif joins_before:
            form = "final"
        elif joins_after:
            form = "initial"
        else:
            form = "isolated"
        out.append(shapes.get(form) or shapes.get("isolated") or character)
    return "".join(out)


def _weak(types: list[str], start: str) -> None:
    """Resolve the weak classes of a line in place, rules W1 to W7.

    Args:
        types: One class per character.
        start: ``L`` or ``R``, the class before the line.
    """
    count = len(types)
    previous = start
    for index in range(count):
        if types[index] == "NSM":
            types[index] = previous
        previous = types[index]
    strong = start
    for index in range(count):
        if types[index] in ("L", "R", "AL"):
            strong = types[index]
        elif types[index] == "EN" and strong == "AL":
            types[index] = "AN"
    for index in range(count):
        if types[index] == "AL":
            types[index] = "R"
    for index in range(1, count - 1):
        left, right = types[index - 1], types[index + 1]
        if types[index] == "ES" and left == right == "EN":
            types[index] = "EN"
        elif types[index] == "CS" and left == right and left in ("EN", "AN"):
            types[index] = left
    index = 0
    while index < count:
        if types[index] != "ET":
            index += 1
            continue
        end = index
        while end < count and types[end] == "ET":
            end += 1
        if (index > 0 and types[index - 1] == "EN") or (end < count and types[end] == "EN"):
            types[index:end] = ["EN"] * (end - index)
        index = end
    for index in range(count):
        if types[index] in ("ES", "ET", "CS"):
            types[index] = "ON"
    strong = start
    for index in range(count):
        if types[index] in ("L", "R"):
            strong = types[index]
        elif types[index] == "EN" and strong == "L":
            types[index] = "L"


def _strong(kind: str) -> str | None:
    """The direction a resolved class counts as beside a neutral: ``L``, ``R`` or None."""
    if kind == "L":
        return "L"
    if kind in ("R", "EN", "AN"):
        return "R"
    return None


def _brackets(characters: list[str], types: list[str], start: str) -> None:
    """Give each bracket pair one direction in place, rule N0.

    Args:
        characters: The line's characters.
        types: One class per character, weak classes resolved.
        start: ``L`` or ``R``, the class before the line and its embedding direction.
    """
    pairs = []
    open_brackets: list[tuple[str, int]] = []
    for index, character in enumerate(characters):
        if types[index] != "ON":
            continue
        if character in BRACKETS:
            if len(open_brackets) >= BRACKET_DEPTH:
                break
            open_brackets.append((BRACKETS[character], index))
            continue
        for depth in range(len(open_brackets) - 1, -1, -1):
            if open_brackets[depth][0] == character:
                pairs.append((open_brackets[depth][1], index))
                del open_brackets[depth:]
                break
    for opening, closing in sorted(pairs):
        inside = {_strong(kind) for kind in types[opening + 1:closing]} - {None}
        if not inside:
            continue
        if start in inside:
            direction = start
        else:
            direction = next(
                (_strong(kind) for kind in reversed(types[:opening]) if _strong(kind)), start
            )
        types[opening] = types[closing] = direction


def _neutral(types: list[str], start: str) -> None:
    """Resolve the neutral classes of a line in place, rules N1 and N2.

    Args:
        types: One class per character, weak classes resolved.
        start: ``L`` or ``R``, the class before and after the line and its embedding direction.
    """
    count = len(types)
    index = 0
    while index < count:
        if types[index] not in NEUTRALS:
            index += 1
            continue
        end = index
        while end < count and types[end] in NEUTRALS:
            end += 1
        left = start if index == 0 else types[index - 1]
        right = start if end == count else types[end]
        left = "L" if left == "L" else "R"
        right = "L" if right == "L" else "R"
        types[index:end] = [left if left == right else start] * (end - index)
        index = end


def display_order(line: str) -> str:
    """The characters of a line in the order they are drawn, left to right.

    Args:
        line: One line, in typed order.

    Returns:
        The line joined and ordered for drawing. A line with no right-to-left letter is
        returned unchanged.
    """
    if not holds_rtl(line):
        return line
    shaped = [character for character in _shape(line) if unicodedata.bidirectional(character) not in CONTROLS]
    if not shaped:
        return ""
    written = [unicodedata.bidirectional(character) or "L" for character in shaped]
    base = next((1 if kind in RTL_CLASSES else 0 for kind in written if kind in ("L", "R", "AL")), 0)
    start = "R" if base else "L"
    types = list(written)
    _weak(types, start)
    _brackets(shaped, types, start)
    _neutral(types, start)
    levels = []
    for kind in types:
        if base == 0:
            levels.append(1 if kind == "R" else 2 if kind in ("AN", "EN") else 0)
        else:
            levels.append(2 if kind in ("L", "AN", "EN") else 1)
    end = len(shaped)
    while end and written[end - 1] in ("WS", "S", "B"):
        end -= 1
        levels[end] = base
    for index, kind in enumerate(written):
        if kind in ("S", "B"):
            levels[index] = base

    # A mark in a right-to-left run lands before its letter, where a zero-width mark is drawn
    # over the glyph after it.
    order = list(zip(levels, shaped))
    highest = max(levels)
    odd = [level for level in levels if level % 2]
    lowest = min(odd) if odd else highest + 1
    for level in range(highest, lowest - 1, -1):
        index = 0
        while index < len(order):
            if order[index][0] < level:
                index += 1
                continue
            stop = index
            while stop < len(order) and order[stop][0] >= level:
                stop += 1
            order[index:stop] = order[index:stop][::-1]
            index = stop

    drawn = []
    for level, character in order:
        if level % 2 and unicodedata.mirrored(character):
            character = MIRRORS.get(character, character)
        if unicodedata.category(character) != "Cf":
            drawn.append(character)
    return "".join(drawn)


def holds_rtl(text: str) -> bool:
    """Whether a text holds a right-to-left letter, such as Arabic or Hebrew.

    Args:
        text: Any text.

    Returns:
        True when one of its characters is written right to left.
    """
    return any(unicodedata.bidirectional(character) in RTL_CLASSES for character in text)
